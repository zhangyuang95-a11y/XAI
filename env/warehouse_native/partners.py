"""Public-state program partners for training/reference only, never deployment.

No function accepts learner logits, sampled actions, or a current human command.
The joint reference is a geometry planner; its plans never enter observations or
rewards. Simple baselines intentionally have their named limited coordination.
"""
from __future__ import annotations

from functools import lru_cache
import heapq
from itertools import permutations, product
import random

from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .observations import task_order

PARTNER_KINDS = ("skilled","assertive","noisy","fixed_yield","fixed_region","fixed_task","follow")


def _choice(rng,values):
    return values[int(rng.integers(len(values)))] if hasattr(rng,"integers") else rng.choice(values)


def _assignments(env,kind):
    state=env.state
    tasks=[t for t in task_order(state) if t.status=="available"]
    empty=[a for a in state.agents if a.carrying_task_id is None]
    assigned={a.agent_id:state.task_by_id(a.carrying_task_id) if a.carrying_task_id else None for a in state.agents}
    distance=lambda a,b:shortest_path_distance(a,b,env.config.map_layout_id)
    if not tasks or not empty:
        return assigned
    if kind=="fixed_task":
        # Stable odd/even task ID responsibility, fallback only when its task is
        # actually carried. No runtime target chooses the learner's task.
        for a in empty:
            role=int(a.agent_id[-1])-1
            owned=[t for t in tasks if (int(t.task_id.rsplit("_",1)[-1])-1)%2==role]
            assigned[a.agent_id]=min(owned,key=lambda t:distance(a.position,t.pickup_position)) if owned else None
    elif kind=="fixed_region":
        for a in empty:
            role=int(a.agent_id[-1])-1
            owned=[t for t in tasks if int(t.pickup_position[1]>=env.config.cols/2)==role]
            assigned[a.agent_id]=min(owned,key=lambda t:distance(a.position,t.pickup_position)) if owned else None
    elif kind in ("assertive","follow"):
        for a in empty:
            assigned[a.agent_id]=min(tasks,key=lambda t:(distance(a.position,t.pickup_position),t.task_id))
    else:
        # Symmetric minimum total collection distance, no role has right-of-way.
        choices=[]
        for people in permutations(empty,len(tasks)):
            choices.append((sum(distance(a.position,t.pickup_position) for a,t in zip(people,tasks)),tuple(a.agent_id for a in people),people))
        if choices:
            people=min(choices,key=lambda x:(x[0],x[1]))[2]
            for a,t in zip(people,tasks):
                assigned[a.agent_id]=t
    return assigned


def _goals(env,kind):
    state=env.state
    assignments=_assignments(env,kind)
    distance=lambda a,b:shortest_path_distance(a,b,env.config.map_layout_id)
    goals={}
    for a in state.agents:
        task=assignments[a.agent_id]
        if task is None:
            goal=env.layout.robot_start_positions[int(a.agent_id[-1])-1]
            required=2*distance(a.position,env.layout.charger_position)+4
        else:
            goal=task.delivery_position if a.carrying_task_id else task.pickup_position
            work=distance(a.position,goal)+(0 if a.carrying_task_id else distance(task.pickup_position,task.delivery_position))
            required=2*(work+distance(task.delivery_position,env.layout.charger_position))+4
        if a.battery < min(100.,required):
            goal=env.layout.charger_position
        goals[a.agent_id]=goal
    charger=env.layout.charger_position
    if all(goals[a.agent_id]==charger for a in state.agents):
        occupied=next((a for a in state.agents if a.position==charger),None)
        priority=occupied or min(state.agents,key=lambda a:(a.battery-2*distance(a.position,charger),a.agent_id))
        other=next(a for a in state.agents if a.agent_id!=priority.agent_id)
        # Public waiting bay, not a charge reservation enforced on the learner.
        bays=[p for p in env.layout.robot_start_positions if p!=priority.position and p!=charger]
        goals[other.agent_id]=min(bays,key=lambda p:(distance(other.position,p),p))
    return goals


@lru_cache(maxsize=32768)
def _joint_route(layout_id,positions,goals):
    """First step of a shortest physical joint route, computed in isolation."""
    layout=get_map_layout(layout_id)
    if positions==goals:
        return ("WAIT","WAIT")
    neighbors={}
    for p in layout.passable_positions:
        neighbors[p]=[(a,(p[0]+d[0],p[1]+d[1])) for a,d in MOVE_DELTAS.items() if layout.is_passable((p[0]+d[0],p[1]+d[1]))]+[("WAIT",p)]
    # Minimize physical moves first, then elapsed joint steps. A shortest-time
    # route alone can waste a depleted waiting robot's battery in equal-time
    # loops. This public reference does not assign those costs to the learner.
    queue=[(0,0,positions,None)]
    seen={positions:(0,0)}
    while queue:
        cost,turns,current,first=heapq.heappop(queue)
        if (cost,turns)!=seen[current]:
            continue
        if current==goals:
            return first
        for left,right in product(neighbors[current[0]],neighbors[current[1]]):
            target=(left[1],right[1])
            if target[0]==target[1] or target==(current[1],current[0]):
                continue
            action=first or (left[0],right[0])
            candidate=(cost+int(left[0]!="WAIT")+int(right[0]!="WAIT"),turns+1)
            if target in seen and seen[target]<=candidate:
                continue
            seen[target]=candidate
            heapq.heappush(queue,(*candidate,target,action))
    return ("WAIT","WAIT")


def partner_action(env,agent_id,kind="skilled",rng=None):
    if kind not in PARTNER_KINDS:
        raise ValueError("Unknown native warehouse partner")
    rng=rng or random.Random(0)
    state=env.state
    agent=state.by_id(agent_id)
    other=next(a for a in state.agents if a.agent_id!=agent_id)
    if not agent.active:
        return "WAIT"
    legal=[a for a,d in MOVE_DELTAS.items() if env.layout.is_passable((agent.position[0]+d[0],agent.position[1]+d[1]))]+["WAIT"]
    if kind=="noisy" and float(rng.random())<.1:
        # The frozen protocol permits all five public commands, including an
        # invalid wall move; this is program-partner noise, never an NN mask.
        return _choice(rng,ACTIONS)
    goals=_goals(env,"skilled" if kind=="noisy" else kind)
    goal=goals[agent_id]
    if agent.position==goal and (goal==env.layout.charger_position or kind not in ("skilled","noisy","fixed_yield")):
        return "WAIT"
    if kind=="follow" and goal!=env.layout.charger_position:
        return other.last_executed_action if other.last_executed_action in legal else "WAIT"
    distance=lambda p,g:shortest_path_distance(p,g,env.config.map_layout_id)
    if kind=="fixed_yield" and agent_id=="robot_2" and distance(agent.position,other.position)<=2:
        return "WAIT"
    if kind in ("skilled","noisy","fixed_yield"):
        positions=tuple(a.position for a in state.agents)
        targets=tuple(goals[a.agent_id] for a in state.agents)
        return _joint_route(env.config.map_layout_id,positions,targets)[int(agent_id[-1])-1]
    candidates=[]
    for action in legal:
        delta=MOVE_DELTAS.get(action,(0,0))
        pos=(agent.position[0]+delta[0],agent.position[1]+delta[1])
        candidates.append((distance(pos,goal),ACTIONS.index(action),action))
    return min(candidates)[2]


def partner_actions(env,kind="skilled",rng=None):
    return {key:partner_action(env,key,kind,rng) for key in env.agent_ids}
