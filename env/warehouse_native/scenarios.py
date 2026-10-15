"""Frozen, content-disjoint initial-state pools, independent of participant data."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import random

from env.warehouse.domain import collaborative_study_config
from .environment import NativeWarehouseEnv

SCENARIO_VERSION="warehouse-native-physical-splits-v1"
SPLIT_NAMES=("train","calibration","validation","extraction","explanation_test","final_test","play")
DEFAULT_COUNTS=dict(zip(SPLIT_NAMES,(512,100,50,100,100,100,12)))


def initial_content(env):
    """Exclude seed/RNG/episode, decorative heading and arbitrary task labels."""
    state=env.state
    return {"layout":env.layout.tiles,"charger":env.layout.charger_position,"horizon":env.config.horizon,
        "agents":[{"position":a.position,"battery":a.battery,"active":a.active,"carrying":a.carrying_task_id is not None} for a in state.agents],
        "tasks":sorted([{"pickup":t.pickup_position,"delivery":t.delivery_position,"status":t.status,"carrier":t.carrier_agent_id} for t in state.tasks],key=lambda t:(t["pickup"],t["delivery"]))}


def scenario_fingerprint(env):
    return sha256(json.dumps(initial_content(env),sort_keys=True,separators=(",",":")).encode()).hexdigest()


def generate_manifest(config=None,counts=None,seed=260908):
    config=config or collaborative_study_config()
    counts=dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts)-set(SPLIT_NAMES) or any(type(v) is not int or v<0 for v in counts.values()):
        raise ValueError("Invalid native scenario split counts")
    splits={name:[] for name in SPLIT_NAMES}
    seen=set()
    candidate_seed=int(seed)
    env=NativeWarehouseEnv(config)
    variation_rng=random.Random(int(seed)+918271)
    attempts=0
    limit=max(10000,100*sum(counts.values()))
    for name in ("play",*(name for name in SPLIT_NAMES if name!="play")):
        while len(splits[name])<counts.get(name,0):
            if attempts>=limit:
                raise RuntimeError("Insufficient distinct physical scenarios; no split manifest was produced")
            current_seed=candidate_seed
            candidate_seed+=1
            attempts+=1
            env.reset(seed=current_seed)
            # Original geometry admits only about 134 distinct initial task
            # arrangements. Public, substantive energy variation expands the
            # pool; a different seed or decorative heading never counts alone.
            for agent in env.state.agents:
                agent.battery=100. if name=="play" else float(variation_rng.choice((60,70,80,90,100)))
            # Each pool entry is its own episode, not a global generation index.
            env.state.episode_id=1
            env._episode_counter=1
            fingerprint=scenario_fingerprint(env)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            splits[name].append({"id":f"{name}_{len(splits[name]):04d}","seed":current_seed,"fingerprint":fingerprint,"snapshot":json.loads(json.dumps(env.snapshot()))})
    return {"version":SCENARIO_VERSION,"configuration":asdict(config),"seed_start":int(seed),"counts":{key:len(value) for key,value in splits.items()},"splits":splits,"split":splits,"initial_battery_values":[60,70,80,90,100],"generalization_scope":"initial_state_split_not_unseen_topology","scenario_protocol":{"play_reserved_first":True,"play_batteries":[100,100],"draw_limit":limit,"same_map_and_physics":True,"task_geometries_may_overlap":True},"content_fingerprint_excludes":["seed","rng","episode_id","heading","task_id"],"selection":"physical uniqueness only; no participant or performance data"}


def reset_scenario(env,entry):
    env.restore(entry["snapshot"])
    if scenario_fingerprint(env)!=entry["fingerprint"]:
        raise ValueError("Native scenario physical fingerprint mismatch")
    return env.observations(),env._info()
