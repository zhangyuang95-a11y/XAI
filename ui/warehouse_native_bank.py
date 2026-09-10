"""Candidate-only, server-private frozen prediction bank; never participant-derived.

Generate after choosing an Actor:
 python -m ui.warehouse_native_bank --actor ACTOR.npz --foundation-scenarios SCENARIOS.json --output NEW_DIRECTORY
The foundation file is read only to exclude all of its actual initial-state
fingerprints. A new pool uses independent seeds and public 65/75/85/95% battery
initializations. It is never added to training, final_test or participant play.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random

from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.policy import ACTIONS
from env.warehouse_native.runtime import NativeRuntime
from env.warehouse_native.scenarios import scenario_fingerprint
from ui.warehouse_view import serialize_warehouse_state, warehouse_map_payload

VERSION = "warehouse-native-prediction-bank.v1"
KINDS = ("next_action", "wait_three")
ACTION_LABELS = {"UP": {"zh":"上","en":"Up"}, "DOWN":{"zh":"下","en":"Down"},
                 "LEFT":{"zh":"左","en":"Left"}, "RIGHT":{"zh":"右","en":"Right"}, "WAIT":{"zh":"等待","en":"Wait"}}


def _preview(snapshot):
    env = NativeWarehouseEnv(); env.restore(snapshot)
    state = serialize_warehouse_state(env.get_state(), selected_agent="robot_2", reveal_policy=False)
    for key in ("human_route_regret_units", "policy_hidden"):
        state.pop(key, None)
    for agent in state["agents"]:
        agent.pop("proposed_action", None); agent.pop("reward", None)
    return {"state":state,"map":warehouse_map_payload(env.layout)}


def _pool(foundation, count, seed_base):
    excluded = {entry["fingerprint"] for entries in foundation["splits"].values() for entry in entries}
    seen = set(excluded); scenes = []; env = NativeWarehouseEnv()
    for offset in range(max(1000, count*100)):
        seed = seed_base+offset
        env.reset(seed=seed)
        for index, agent in enumerate(env.state.agents):
            agent.battery = float((65,75,85,95)[(offset+index*2)%4])
        env.state.episode_id=1; env._episode_counter=1
        fingerprint = scenario_fingerprint(env)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        scenes.append({"id":f"question_bank_dev_{len(scenes):04d}","seed":seed,
                       "fingerprint":fingerprint,"snapshot":env.snapshot()})
        if len(scenes)==count:
            break
    return scenes, excluded


def _positions(env, origin):
    queue=deque([(tuple(origin),0)]); seen={tuple(origin)}
    while queue:
        position, distance=queue.popleft()
        if distance==3:
            continue
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            target=(position[0]+dr,position[1]+dc)
            if target not in seen and env.layout.is_passable(target):
                seen.add(target); queue.append((target,distance+1))
    return sorted(seen)


def _candidate(runtime, env, scene_id, kind):
    snapshot=env.snapshot(); before=digest(snapshot)
    item={"kind":kind,"scenario_id":scene_id,"frame":env.state.frame,"snapshot":snapshot,
          "snapshot_sha256":before,"preview":_preview(snapshot)}
    if kind=="next_action":
        actions,evidence=runtime.decision(env)
        item.update(answer=actions["robot_2"], diversity_key=actions["robot_2"], evidence=evidence,
                    prompt={"zh":"在下图状态，AI 2 下一步会选择哪个动作（碰撞处理前）？", "en":"In this state, which action will AI 2 select next, before collision resolution?"},
                    options=[{"value":a,"label":ACTION_LABELS[a]} for a in ACTIONS])
    else:
        branch=runtime.counterfactual(snapshot,["WAIT","WAIT","WAIT"],steps=3)
        if len(branch["transitions"])!=3:
            return None
        # Do not ask humans for a deterministic prediction that depends on a
        # not-yet-visible random replacement order sampled inside the branch.
        if any(t["after"]["state"]["next_task_index"] != t["before"]["state"]["next_task_index"] for t in branch["transitions"]):
            return None
        end=branch["transitions"][-1]["after"]["state"]["agents"][1]["position"]
        start=list(env.state.agents[1].position)
        alternatives=[p for p in _positions(env,start) if list(p)!=list(end)]
        if len(alternatives)<3:
            return None
        rng=random.Random(int(before[:16],16));rng.shuffle(alternatives)
        choices=[tuple(end),*alternatives[:3]];rng.shuffle(choices)
        options=[{"value":f"{r},{c}","label":{"zh":f"{chr(65+i)} · 第 {r+1} 行，第 {c+1} 列","en":f"{chr(65+i)} · Row {r+1}, column {c+1}"},"position":[r,c],"marker":chr(65+i)} for i,(r,c) in enumerate(choices)]
        item.update(answer=f"{end[0]},{end[1]}",diversity_key=f"{end[0]-start[0]},{end[1]-start[1]}",
                    evidence={"transitions_sha256":digest(branch["transitions"]),"assumed_player_actions":["WAIT"]*3},
                    options=options,prompt={"zh":"假设玩家连续等待三步，AI 2 在第三步结束时位于哪一格？字母标记仅表示选项位置。", "en":"If the player waits for three consecutive steps, where will AI 2 be after the third step? Letter markers identify the answer options only."})
        item["preview"]["question_markers"]=[{"position":o["position"],"label":o["marker"]} for o in options]
    if digest(env.snapshot())!=before:
        raise ValueError("Question generation changed its source state or RNG")
    return item


def _select(candidates):
    selected=[]
    for kind in KINDS:
        remaining=[c for c in candidates if c["kind"]==kind]
        chosen=[]; scenes=set(); used=Counter()
        while remaining and len(chosen)<4:
            remaining.sort(key=lambda c:(used[c["diversity_key"]],c["frame"],c["scenario_id"]))
            candidate=next((c for c in remaining if c["scenario_id"] not in scenes),None)
            if candidate is None:
                break
            chosen.append(candidate);scenes.add(candidate["scenario_id"]);used[candidate["diversity_key"]]+=1
            remaining=[c for c in remaining if c["scenario_id"] not in scenes]
        for index,item in enumerate(chosen):
            selected.append({"id":f"prediction_{kind}_{index+1}",**item})
    return selected


def _checks(items):
    result={}; passed=True
    for kind in KINDS:
        rows=[i for i in items if i["kind"]==kind]; distribution=Counter(i["diversity_key"] for i in rows)
        check={"count":len(rows),"independent_scenarios":len({i["scenario_id"] for i in rows}),
               "distinct_outcomes":len(distribution),"outcome_counts":dict(distribution)}
        check["passed"]=len(rows)==4 and check["independent_scenarios"]==4 and len(distribution)>=2 and max(distribution.values(),default=0)<=3
        result[kind]=check;passed=passed and check["passed"]
    return {"passed":passed,"categories":result,"required_per_kind":4,"formal_ready":False,
            "reason":"candidate_content_checks_passed_model_acceptance_separate" if passed else "insufficient_independent_items_or_single_dominant_behavior"}


def generate_bank(actor_path, foundation, output, *, pool_count=24, trajectory_steps=24, seed_base=831_710_000):
    if not 4<=pool_count<=100 or not 1<=trajectory_steps<=120 or seed_base<0:
        raise ValueError("Invalid bounded question-pool configuration")
    destination=Path(output);destination.mkdir(parents=True,exist_ok=False,mode=0o700)
    runtime=NativeRuntime(actor_path)
    scenes,excluded=_pool(foundation,pool_count,seed_base); candidates=[]
    for scene in scenes:
        env=runtime.environment(scene)
        for _ in range(trajectory_steps):
            if env.done:break
            for kind in KINDS:
                item=_candidate(runtime,env,scene["id"],kind)
                if item:candidates.append(item)
            # Same-Actor selfplay produces legal state histories; no participant or
            # teacher trace is read, and these trajectories are never training data.
            actions,_=runtime.decision(env);runtime.step(env,actions["robot_1"])
    items=_select(candidates)
    bank={"version":VERSION,"status":"candidate","formal_ready":False,
          "actor_sha256":runtime.actor.artifact_sha256,"runtime_signature":runtime.signature,
          "test_fixture":runtime.actor.metadata.get("test_fixture") is True,
          "foundation_manifest_sha256":digest(foundation),"generator_sha256":file_hash(Path(__file__)),
          "pool_namespace":"question_bank_development_v1","pool_seed_base":seed_base,
          "pool_scenes":scenes,"pool_disjoint_from_foundation":not bool({s["fingerprint"] for s in scenes}&excluded),
          "items":items,"checks":_checks(items),"candidate_frame_count":len(candidates),
          "model_capability_acceptance":"not_asserted_by_question_bank_generator",
          "counterfactual_filter":"three actual steps without newly sampled replacement tasks"}
    for name,value in (("bank.private.json",bank),("generation_report.private.json",{k:v for k,v in bank.items() if k not in ("pool_scenes","items")})):
        fd=os.open(destination/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w") as handle:handle.write(canonical(value))
    return bank


class NativeQuestionBank:
    """Recompute private answers at load; never trust a JSON ready flag."""
    def __init__(self,path,runtime,foundation):
        self.path=Path(path);self.bank=json.loads(self.path.read_text())
        b=self.bank
        if b.get("version")!=VERSION or b.get("actor_sha256")!=runtime.actor.artifact_sha256 or b.get("runtime_signature")!=runtime.signature:
            raise ValueError("Question bank Actor/runtime binding mismatch")
        if b.get("foundation_manifest_sha256")!=digest(foundation):
            raise ValueError("Question bank foundation split binding mismatch")
        excluded={s["fingerprint"] for values in foundation["splits"].values() for s in values}
        scenes={s["id"]:s for s in b["pool_scenes"]}
        fingerprints=set()
        for scene in scenes.values():
            env=runtime.environment(scene)
            fingerprint=scenario_fingerprint(env)
            if fingerprint in excluded or fingerprint in fingerprints or env.state.frame!=0:
                raise ValueError("Question-bank pool overlaps an existing split or itself")
            fingerprints.add(fingerprint)
        if len(scenes)!=len(b["pool_scenes"]):raise ValueError("Duplicate question-bank scenario IDs")
        items=b["items"]
        if len(items)>8 or len({i["id"] for i in items})!=len(items):raise ValueError("Invalid bank item count or IDs")
        for item in items:
            if item["kind"] not in KINDS or item["scenario_id"] not in scenes:raise ValueError("Unknown question source")
            if item["id"] not in {f"prediction_{item["kind"]}_{n}" for n in range(1,5)}:raise ValueError("Invalid prediction item ID")
            env=runtime.environment(scenes[item["scenario_id"]])
            if type(item["frame"]) is not int or not 0<=item["frame"]<120:raise ValueError("Invalid source frame")
            for _ in range(item["frame"]):
                if env.done:raise ValueError("Question source exceeds legal trajectory")
                actions,_=runtime.decision(env);runtime.step(env,actions["robot_1"])
            if digest(env.snapshot())!=item["snapshot_sha256"] or digest(item["snapshot"])!=item["snapshot_sha256"]:
                raise ValueError("Question snapshot is not its bound legal source trajectory")
            expected=_candidate(runtime,env,item["scenario_id"],item["kind"])
            if expected is None or digest({key:item.get(key) for key in expected})!=digest(expected):
                raise ValueError("Question answer, options or evidence failed recomputation")
        self.checks=_checks(items);self.eligible=self.checks["passed"]
        self.test_fixture=b.get("test_fixture") is True or runtime.actor.metadata.get("test_fixture") is True
        self.actor_sha256=runtime.actor.artifact_sha256
        self.runtime_signature=runtime.signature
        self.foundation_manifest_sha256=digest(foundation)
        self.signature=digest({"bank_sha256":file_hash(self.path),"loader_sha256":file_hash(Path(__file__)),"runtime_signature":runtime.signature})
        self.items=deepcopy(items)

    def public_items(self):
        if not self.eligible:return []
        return [{"id":i["id"],"type":"choice","prediction_kind":i["kind"],"required":True,
                 "prompt":deepcopy(i["prompt"]),"options":[{"value":o["value"],"label":deepcopy(o["label"])} for o in i["options"]],
                 "preview":deepcopy(i["preview"]),"source_frame":i["frame"],"source_scenario":i["scenario_id"]} for i in self.items]

    def summary(self):
        return {"status":"candidate_ready" if self.eligible else "candidate_failed","formal_ready":False,
                "available":self.eligible,"item_count":len(self.public_items()),"test_fixture":self.test_fixture}

    def grade(self,answers):
        if not self.eligible:raise ValueError("Question bank content checks have not passed")
        score={"bank_signature":self.signature,"candidate_only":True,"formal_ready":False}
        for kind in KINDS:
            items=[i for i in self.items if i["kind"]==kind]
            if any(answers.get(i["id"]) not in {o["value"] for o in i["options"]} for i in items):
                raise ValueError("Incomplete or invalid prediction answers")
            correct=sum(answers[i["id"]]==i["answer"] for i in items)
            score[kind]={"correct":correct,"total":len(items),"accuracy":correct/len(items)}
        return score


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor",type=Path,required=True);parser.add_argument("--foundation-scenarios",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--pool-count",type=int,default=24)
    parser.add_argument("--trajectory-steps",type=int,default=24);parser.add_argument("--seed-base",type=int,default=831_710_000)
    args=parser.parse_args(argv)
    bank=generate_bank(args.actor,json.loads(args.foundation_scenarios.read_text()),args.output,pool_count=args.pool_count,trajectory_steps=args.trajectory_steps,seed_base=args.seed_base)
    print(canonical({"status":"candidate","formal_ready":False,"content_checks_passed":bank["checks"]["passed"],"items":len(bank["items"]),"output":str(args.output)}))

if __name__=="__main__":main()
