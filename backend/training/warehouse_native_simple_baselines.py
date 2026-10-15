"""Stronger public conventions for difficulty diagnostics, never neural control.

Detached from foundation training, checkpoint selection, and runtime deployment.
These programs are not humans without explanations. Their comparisons establish
only how these explicitly named conventions behave in the frozen calibration pool.
"""
from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
import random
import statistics
import time

from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.observations import task_order
from env.warehouse_native.partners import partner_action
from env.warehouse_native.scenarios import generate_manifest, reset_scenario

VERSION="warehouse-native-stronger-conventions-v2"
KINDS=("right_of_way","region_spillover","task_spillover","leader_follower")
DESCRIPTIONS={
    "right_of_way":"Robot 1 has fixed aisle priority. Tasks are nearest-first with distinct public assignments. A blocking Robot 2 steps to a reachable side cell outside Robot 1's path; it does not indefinitely wait in the path.",
    "region_spillover":"Prefer left/right pickup regions for Robot 1/2, then assign any remaining available job to an otherwise idle robot. Carrying always takes precedence; use the same explicit right-of-way convention.",
    "task_spillover":"Prefer odd/even public task IDs for Robot 1/2, then allocate remaining jobs to an otherwise idle robot. Carrying takes precedence; use the same explicit right-of-way convention.",
    "leader_follower":"Robot 1 actively chooses the nearest available pickup. Empty Robot 2 follows the leader's last physical position with a two-cell gap, but independently takes an unassigned pickup within four path steps. Any carried item is delivered first; blocking followers step aside.",
}
COMMON_RULES="All decisions use public current state and last executed actions only. Charging estimates the chosen public delivery route plus return and four reserve moves. An already docked robot has temporary exit/charging priority at the charger. Default aisle priority is Robot 1; when the previous joint step had no motion, charge, pickup or delivery, temporary priority alternates in two-step windows to recover from disagreement. Programs route around occupied cells, and their own simultaneous same-cell proposals respect this convention. No neural proposal or probability is read or rewritten."


@lru_cache(maxsize=32768)
def _path(layout_id,start,goal,blocked=()):
    layout=get_map_layout(layout_id)
    if start==goal:
        return (start,)
    if not layout.is_passable(goal) or goal in blocked:
        return ()
    previous={start:None}
    queue=deque([start])
    while queue:
        current=queue.popleft()
        for delta in MOVE_DELTAS.values():
            target=(current[0]+delta[0],current[1]+delta[1])
            if target in previous or target in blocked or not layout.is_passable(target):
                continue
            previous[target]=current
            if target==goal:
                route=[target]
                while route[-1]!=start:
                    route.append(previous[route[-1]])
                return tuple(reversed(route))
            queue.append(target)
    return ()


def _first_action(route):
    if len(route)<2:
        return "WAIT"
    delta=(route[1][0]-route[0][0],route[1][1]-route[0][1])
    return next(action for action,value in MOVE_DELTAS.items() if value==delta)


def _public_step(env,person,other,goal):
    route=_path(env.config.map_layout_id,person.position,goal,(other.position,))
    if route:
        return _first_action(route)
    # A stationary robot farther along a one-cell corridor must not cause
    # premature waiting. Advance along the free prefix, stopping at its cell.
    route=_path(env.config.map_layout_id,person.position,goal)
    return _first_action(route) if len(route)>1 and route[1]!=other.position else "WAIT"


def _distance(env,a,b):
    return shortest_path_distance(a,b,env.config.map_layout_id)


def _assign(env,kind):
    agents=env.state.agents
    tasks=[t for t in task_order(env.state) if t.status=="available"]
    empty=[a for a in agents if a.carrying_task_id is None]
    assigned={a.agent_id:env.state.task_by_id(a.carrying_task_id) if a.carrying_task_id else None for a in agents}
    if kind=="leader_follower":
        if not agents[0].carrying_task_id and tasks:
            assigned[agents[0].agent_id]=min(tasks,key=lambda t:(_distance(env,agents[0].position,t.pickup_position),t.task_id))
        leader_task=assigned[agents[0].agent_id]
        spare=[t for t in tasks if t is not leader_task and _distance(env,agents[1].position,t.pickup_position)<=4]
        if not agents[1].carrying_task_id and spare:
            assigned[agents[1].agent_id]=min(spare,key=lambda t:(_distance(env,agents[1].position,t.pickup_position),t.task_id))
        return assigned
    candidates=[]
    for people in permutations(empty,len(tasks)):
        violations=0
        for person,task in zip(people,tasks):
            role=int(person.agent_id[-1])-1
            if kind=="region_spillover":
                violations+=int(int(task.pickup_position[1]>=env.config.cols/2)!=role)
            if kind=="task_spillover":
                violations+=int((int(task.task_id.rsplit("_",1)[-1])-1)%2!=role)
        distance=sum(_distance(env,a.position,t.pickup_position) for a,t in zip(people,tasks))
        candidates.append((violations,distance,tuple(a.agent_id for a in people),people))
    if candidates:
        people=min(candidates,key=lambda item:item[:3])[3]
        for person,task in zip(people,tasks):
            assigned[person.agent_id]=task
    return assigned


def _goals(env,kind):
    tasks=_assign(env,kind)
    goals={}
    charger=env.layout.charger_position
    for role,person in enumerate(env.state.agents):
        task=tasks[person.agent_id]
        if task is not None:
            goal=task.delivery_position if person.carrying_task_id else task.pickup_position
            remaining=_distance(env,person.position,goal)+(0 if person.carrying_task_id else _distance(env,task.pickup_position,task.delivery_position))
            energy=2*(remaining+_distance(env,task.delivery_position,charger))+8
        elif kind=="leader_follower" and role==1:
            leader=env.state.agents[0]
            delta=MOVE_DELTAS.get(leader.last_executed_action,(0,0))
            last=(leader.position[0]-delta[0],leader.position[1]-delta[1])
            goal=person.position if _distance(env,person.position,leader.position)<=2 else last
            energy=2*(_distance(env,person.position,goal)+_distance(env,goal,charger))+8
        else:
            goal=env.layout.robot_start_positions[role]
            energy=2*_distance(env,person.position,charger)+8
        goals[person.agent_id]=charger if person.battery<min(100.,energy) else goal
    return goals


def _side_step(env,low,high,priority_route):
    """Nearest reachable bay off the priority route, with public energy check."""
    candidates=[]
    blocked=(high.position,)
    for pos in env.layout.passable_positions:
        if pos in priority_route or pos==env.layout.charger_position:
            continue
        route=_path(env.config.map_layout_id,low.position,pos,blocked)
        if not route:
            continue
        moves=len(route)-1
        if 2*(moves+_distance(env,pos,env.layout.charger_position))>low.battery:
            continue
        degree=sum(env.layout.is_passable((pos[0]+d[0],pos[1]+d[1])) for d in MOVE_DELTAS.values())
        candidates.append((moves,int(degree>1),pos,route))
    return _first_action(min(candidates,key=lambda row:row[:3])[3]) if candidates else "WAIT"


def baseline_actions(env,kind):
    if kind not in KINDS:
        raise ValueError("Unknown stronger convention")
    people=env.state.agents
    goals=_goals(env,kind)
    priority=0
    no_motion=all(person.last_executed_action=="WAIT" for person in people)
    no_charge=all(person.last_battery_delta<=0 for person in people)
    no_task_event=not any(task.claimed_frame==env.state.frame or task.delivered_frame==env.state.frame for task in (*env.state.tasks,*env.state.completed_tasks))
    if env.state.frame>0 and no_motion and no_charge and no_task_event:
        priority=1 if ((env.state.frame-1)//2)%2==0 else 0
    charger=env.layout.charger_position
    occupant=next((i for i,a in enumerate(people) if a.position==charger),None)
    if occupant is not None and any(goal==charger for goal in goals.values()):
        priority=occupant
    low_index=1-priority
    high,low=people[priority],people[low_index]
    if all(goal==charger for goal in goals.values()):
        if occupant is None:
            # Fixed robot priority, unless it cannot safely reach the charger
            # after the other's immediate docking; full initial battery is not
            # used as an invented queue reservation.
            low_index=1-priority;high,low=people[priority],people[low_index]
        bays=[p for p in env.layout.robot_start_positions if p!=high.position]
        goals[low.agent_id]=min(bays,key=lambda p:(_distance(env,low.position,p),p))
    high_route=_path(env.config.map_layout_id,high.position,goals[high.agent_id])
    actions={}
    for person,other in ((high,low),(low,high)):
        actions[person.agent_id]=_public_step(env,person,other,goals[person.agent_id])
    blocking=low.position in high_route[1:] and _distance(env,high.position,low.position)<=3
    if blocking:
        actions[low.agent_id]=_side_step(env,low,high,high_route)
    # Resolve only these programs' own conventions. This function is never
    # called with a neural policy action and never runs in native deployment.
    targets={}
    for person in people:
        delta=MOVE_DELTAS.get(actions[person.agent_id],(0,0))
        targets[person.agent_id]=(person.position[0]+delta[0],person.position[1]+delta[1])
    if targets[high.agent_id]==targets[low.agent_id]:
        actions[low.agent_id]="WAIT"
    return actions


def baseline_action(env,agent_id,kind):
    if agent_id not in env.agent_ids:
        raise ValueError("Unknown robot")
    return baseline_actions(env,kind)[agent_id]


def run_episode(entry,left,right):
    env=NativeWarehouseEnv();reset_scenario(env,entry)
    rngs=[random.Random(entry["seed"]+10101),random.Random(entry["seed"]+20202)]
    waits=[0,0];blocked=[0,0];pickup=[0,0];charge=[0.,0.]
    while not env.done:
        commands={}
        for i,kind in enumerate((left,right)):
            key=env.agent_ids[i]
            commands[key]=baseline_action(env,key,kind) if kind in KINDS else partner_action(env,key,kind,rngs[i])
        _,_,_,_,info=env.step(commands)
        for i,key in enumerate(env.agent_ids):
            waits[i]+=int(commands[key]=="WAIT")
            blocked[i]+=int(commands[key]!="WAIT" and info["executed_actions"][key]=="WAIT")
        for event in info["events"]:
            if event["event"]=="pickup":pickup[int(event["agent_id"][-1])-1]+=1
            if event["event"]=="charge":charge[int(event["agent_id"][-1])-1]+=event["amount"]
    state=env.state
    return {"scenario_id":entry["id"],"fingerprint":entry["fingerprint"],"left":left,"right":right,"deliveries":state.total_deliveries,"individual_deliveries":[a.deliveries_completed for a in state.agents],"both_active":all(a.active for a in state.agents),"shutdowns":state.shutdown_count,"collisions":state.robot_collision_events,"steps":state.frame,"native_score":state.user_score,"legacy_score":None,"submitted_waits":waits,"physically_blocked_moves":blocked,"pickups":pickup,"charge_gain":charge}


def summarize(rows):
    return {"left":rows[0]["left"],"right":rows[0]["right"],"episodes":len(rows),"mean_deliveries":statistics.mean(r["deliveries"] for r in rows),"both_active_rate":statistics.mean(r["both_active"] for r in rows),"mean_collisions":statistics.mean(r["collisions"] for r in rows),"mean_individual_deliveries":[statistics.mean(r["individual_deliveries"][i] for r in rows) for i in range(2)]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("output/warehouse_native/calibration/stronger_simple_baselines_v2.json"))
    parser.add_argument("--previous",type=Path,default=Path("output/warehouse_native/calibration/environment_reference_and_simple_baselines.json"))
    args=parser.parse_args(argv)
    if args.output.exists():parser.error("Refuse to overwrite a previous calibration report")
    root=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),*(root/"env/warehouse_native"/n for n in ("__init__.py","environment.py","observations.py","partners.py","scenarios.py")),*(root/"env/warehouse"/n for n in ("environment.py","domain.py","layouts.py","navigation.py","transition_outcome.py","rewards.py"))]
    hashes={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest=generate_manifest()
    manifest_hash=sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    previous=json.loads(args.previous.read_text())
    if previous["manifest_sha256"]!=manifest_hash:
        raise ValueError("Earlier weak baseline used different frozen scenarios")
    rows=[];summary=[];started=time.monotonic()
    for kind in KINDS:
        for left in (kind,"skilled"):
            group=[run_episode(entry,left,kind) for entry in manifest["splits"]["calibration"]]
            rows.extend(group);summary.append(summarize(group))
            print(json.dumps(summary[-1]),flush=True)
    if hashes!={str(p.relative_to(root)):sha256(p.read_bytes()).hexdigest() for p in paths}:
        raise RuntimeError("Calibration source changed during execution")
    report={"version":VERSION,"scope":"program_only_calibration_not_human_or_explanation_effect","excluded_from_foundation_selection":True,"manifest_sha256":manifest_hash,"sources":hashes,"runner_source":Path(__file__).read_text(),"descriptions":DESCRIPTIONS,"common_rules":COMMON_RULES,"summary":summary,"rows":rows,"previous_report":{"path":str(args.previous),"sha256":sha256(args.previous.read_bytes()).hexdigest(),"unchanged":True,"summary":previous["summary"],"limitations":["Original fixed_yield can repeatedly wait while blocking.","Original follow starts from WAIT and can remain stationary.","These weak programs are not estimates of competent humans without explanations."]},"limitations":["Fixed conventions are hand-written programs, not neural teammates or human participants.","Cross-convention pairs may disagree about priority or task responsibility.","Initial states are disjoint; task topologies are shared.","No threshold, reward, training source or checkpoint was changed based on these results."],"seconds":time.monotonic()-started}
    prior=args.output.with_name("stronger_simple_baselines.json")
    if prior.exists():
        report["prior_stronger_report"]={"path":str(prior),"sha256":sha256(prior.read_bytes()).hexdigest(),"summary":json.loads(prior.read_text())["summary"],"reason_for_revision":"public stall recovery for cross-convention mutual waiting; original report and source snapshot retained"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print("SAVED "+str(args.output),flush=True)


if __name__=="__main__":
    main()
