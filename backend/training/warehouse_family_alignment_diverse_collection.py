"""Frozen unused-TRAIN explanation collection for the 3.95M alignment Actor.

The development and acceptance phases are separate durable runs.  Their scene
membership is fixed by original manifest index before any neural query:
``fit=train[192:384]``, ``dev=train[384:448]`` and
``accept=train[448:512]``.  Sampling uses the genuine alignment runtime and its
two-role physical branch sampler.  No program prediction, fitted tree, or
qualification result can influence which states are queried.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import gzip
import json
import os
import re
import time

import numpy as np

from .warehouse_native_common import ROOT, canonical, digest, file_hash, atomic_json
from . import warehouse_family_alignment_runtime_collection as alignment_collection
from . import warehouse_family_alignment_runtime_samples as samples
from . import warehouse_family_branch_teacher_data as data_io
from .warehouse_family_branch_teacher_data import (
    _put, _load, _empty, _row, PROFILES, ROLES, COUNTERS,
)
from .warehouse_native_evaluation import critical_groups
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS
from backend import warehouse_alignment_runtime as runtime_api
from backend.warehouse_family_explanation import actor_parameter_sha256

VERSION = "warehouse-family-alignment-diverse-explanation-collection.v1"
DATA_VERSION = data_io.VERSION
PRODUCTION_ACTOR_SHA256 = "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b"
PRODUCTION_ACTOR_TRAINING_CLOCK = 3_950_000
PHASE_POOLS = {
    "development": ("fit", "dev"),
    "acceptance": ("accept",),
}
POOL_RANGES = {
    "fit": (192, 384),
    "dev": (384, 448),
    "accept": (448, 512),
}
HORIZON = 120
ANCHOR_FRAMES = tuple(range(0, HORIZON, 10))
_HASH = re.compile(r"^[0-9a-f]{64}$")

# These aliases make the inherited, reviewed query and input admission explicit.
collect_branch_pairs = samples.collect_branch_pairs


def _phase(value):
    if value not in PHASE_POOLS:
        raise ValueError("Phase must be exactly 'development' or 'acceptance'")
    return value


def _pool_scenes(scenario_manifest):
    """Validate all three frozen pools, including pools unused by this phase."""
    if not isinstance(scenario_manifest, dict) or not isinstance(scenario_manifest.get("splits"), dict):
        raise ValueError("Original scenario manifest splits are required")
    train = scenario_manifest["splits"].get("train")
    if not isinstance(train, list) or len(train) < POOL_RANGES["accept"][1]:
        raise ValueError("At least 512 original neural TRAIN starts are required")

    pools = {name: train[start:stop] for name, (start, stop) in POOL_RANGES.items()}
    expected_counts = {name: stop - start for name, (start, stop) in POOL_RANGES.items()}
    if {name: len(rows) for name, rows in pools.items()} != expected_counts:
        raise ValueError("Frozen TRAIN ranges are incomplete")

    selected_ids, selected_fingerprints = set(), set()
    fingerprints_by_pool = {}
    for pool, rows in pools.items():
        fingerprints = set()
        for scene in rows:
            if (not isinstance(scene, dict) or not isinstance(scene.get("id"), str)
                    or not _HASH.fullmatch(str(scene.get("fingerprint", "")))
                    or scene.get("split", "train") != "train"
                    or scene.get("test_fixture", False) is not False):
                raise ValueError("Only genuine original neural TRAIN starts are permitted")
            if scene["id"] in selected_ids or scene["fingerprint"] in selected_fingerprints:
                raise ValueError("Frozen fit/dev/accept scene identity or fingerprint overlap")
            selected_ids.add(scene["id"])
            selected_fingerprints.add(scene["fingerprint"])
            fingerprints.add(scene["fingerprint"])
        fingerprints_by_pool[pool] = fingerprints

    outside = train[:POOL_RANGES["fit"][0]] + train[POOL_RANGES["accept"][1]:]
    outside += [scene for split, rows in scenario_manifest["splits"].items()
                if split != "train" for scene in rows]
    outside_fingerprints = {scene.get("fingerprint") for scene in outside
                            if isinstance(scene, dict)}
    if selected_fingerprints & outside_fingerprints:
        raise ValueError("Frozen explanation pools overlap another manifest scenario fingerprint")
    if any(fingerprints_by_pool[a] & fingerprints_by_pool[b]
           for index, a in enumerate(POOL_RANGES)
           for b in tuple(POOL_RANGES)[index + 1:]):
        raise ValueError("Frozen fit/dev/accept physical fingerprints are not independent")
    return pools


def context_matrix(scenario_manifest, phase):
    """Return the fixed four-profile contexts for one explicit phase."""
    phase = _phase(phase)
    pools = _pool_scenes(scenario_manifest)
    contexts = []
    for pool in PHASE_POOLS[phase]:
        start, _ = POOL_RANGES[pool]
        for offset, scene in enumerate(pools[pool]):
            scene_index = start + offset
            for profile_index, profile in enumerate(PROFILES):
                contexts.append({
                    "id": f"{phase}:{pool}:{scene_index:04d}:{profile}",
                    "phase": phase,
                    "pool": pool,
                    "scene_index": scene_index,
                    "scene_id": scene["id"],
                    "fingerprint": scene["fingerprint"],
                    "profile": profile,
                    "program_role": -1 if profile == "selfplay" else (scene_index + profile_index) % 2,
                    "rng_seed": 26091030 + scene_index * 10 + profile_index,
                })
    return contexts


def _context(actor, protocol, actor_sha, protocol_sha, scenarios, *, allow_test_fixture=False):
    """Reuse the alignment producer's exact Actor/protocol/manifest admission."""
    if type(allow_test_fixture) is not bool:
        raise ValueError("Explicit boolean fixture scope required")
    if not allow_test_fixture:
        if actor_sha != PRODUCTION_ACTOR_SHA256:
            raise ValueError("Diverse production collection is locked to the 3.95M Actor SHA256")
        return alignment_collection._context(actor, protocol, actor_sha, protocol_sha, scenarios)

    paths = {key: Path(value).expanduser().resolve() for key, value in
             (("actor", actor), ("protocol", protocol), ("scenarios", scenarios))}
    inputs = {key: {"path": str(path), "sha256": file_hash(path)}
              for key, path in paths.items()}
    if inputs["actor"]["sha256"] != actor_sha:
        raise ValueError("Fixture Actor bytes differ from their external SHA256")
    declared = _load(paths["protocol"])
    if digest(declared) != protocol_sha:
        raise ValueError("Fixture protocol semantic hash differs")
    scenes = _load(paths["scenarios"])
    runtime = runtime_api.AlignmentRuntime(
        paths["actor"], protocol=declared, expected_actor_sha256=actor_sha,
        expected_protocol_sha256=protocol_sha, allow_test_fixture=True,
    )
    if digest(scenes) != runtime.actor.metadata["scenario_manifest_sha256"]:
        raise ValueError("Fixture scenario manifest differs from Actor provenance")
    runtime_api.verify(runtime, allow_test_fixture=True)
    if any(file_hash(Path(item["path"])) != item["sha256"] for item in inputs.values()):
        raise ValueError("Fixture inputs changed while loading")
    return SimpleNamespace(runtime=runtime, scenarios=scenes), inputs


def _source_files():
    sources = runtime_api.runtime_sources()
    for module in (alignment_collection, samples, data_io):
        path = Path(module.__file__).resolve()
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    path = Path(__file__).resolve()
    sources[str(path.relative_to(ROOT))] = file_hash(path)
    return sources


def _plan(runtime, scenario_manifest, inputs, phase, *, allow_test_fixture=False):
    phase = _phase(phase)
    if type(allow_test_fixture) is not bool:
        raise ValueError("Explicit boolean fixture scope required")
    identity = runtime_api.verify(runtime, allow_test_fixture=allow_test_fixture)
    contexts = context_matrix(scenario_manifest, phase)
    pools = _pool_scenes(scenario_manifest)
    bindings = {
        "actor_sha256": runtime.actor_sha256,
        "actor_parameters_sha256": actor_parameter_sha256(runtime.actor),
        "protocol_sha256": runtime.protocol_sha256,
        "source_sha256": runtime.actor.metadata["source_sha256"],
        "runtime_signature": runtime.signature,
    }
    sources = _source_files()
    if sources != {**runtime.sources,
            str(Path(alignment_collection.__file__).resolve().relative_to(ROOT)): file_hash(alignment_collection.__file__),
            str(Path(samples.__file__).resolve().relative_to(ROOT)): file_hash(samples.__file__),
            str(Path(data_io.__file__).resolve().relative_to(ROOT)): file_hash(data_io.__file__),
            str(Path(__file__).resolve().relative_to(ROOT)): file_hash(__file__)}:
        raise ValueError("Alignment runtime and diverse collector source closure differ")
    pool_scene_counts = {name: len(pools[name]) for name in PHASE_POOLS[phase]}
    pool_fingerprints_sha256 = {
        name: digest([scene["fingerprint"] for scene in pools[name]])
        for name in PHASE_POOLS[phase]
    }
    context_count = len(contexts)
    actor_training_clock = (runtime.actor.metadata["source_counters"]["joint_steps"]
                            + runtime.actor.metadata["joint_steps"])
    if not allow_test_fixture and (runtime.actor_sha256 != PRODUCTION_ACTOR_SHA256
                                   or actor_training_clock != PRODUCTION_ACTOR_TRAINING_CLOCK):
        raise ValueError("Production collection requires the frozen 3.95M alignment Actor")
    return {
        "version": VERSION,
        "data_version": DATA_VERSION,
        "producer": VERSION,
        "phase": phase,
        "phase_pools": list(PHASE_POOLS[phase]),
        "pool_ranges": {name: list(POOL_RANGES[name]) for name in PHASE_POOLS[phase]},
        "pool_scene_counts": pool_scene_counts,
        "pool_fingerprints_sha256": pool_fingerprints_sha256,
        "context_count": context_count,
        "runtime_identity": identity,
        "input_files": deepcopy(inputs),
        "actor_bindings": bindings,
        "feature_names": list(runtime.actor.metadata["feature_names"]),
        "actor_training_clock": actor_training_clock,
        "scenarios_sha256": digest(scenario_manifest),
        "contexts": contexts,
        "horizon": HORIZON,
        "anchor_frames": list(ANCHOR_FRAMES),
        "source_split": "original_neural_train",
        "test_fixture": allow_test_fixture,
        "source_files": sources,
        "sampling_rule": "fixed_manifest_index_ranges_and_every_tenth_frame",
        "sampling_uses_tree_predictions": False,
        "all_nonterminal_pairs_retained": True,
        "groups_from_source_anchor": True,
        "both_actual_roles_physically_intervened": True,
        "labels_from_unmodified_neural_actor": True,
        "all_submitted_neural_roles_retained_including_inactive": True,
        "acceptance_independent_of_development_collection": phase == "acceptance",
        "development_pools_accessed": phase == "development",
        "acceptance_pool_accessed": phase == "acceptance",
        "cross_phase_data_accessed": False,
        "maximum_base_steps": context_count * HORIZON,
        "maximum_branch_steps": context_count * len(ANCHOR_FRAMES) * 10,
        "maximum_branch_queries": context_count * len(ANCHOR_FRAMES) * 11,
        "runtime_verification": "full_content_hashes_at_context_entry_and_exit_before_ACK; object/read-only-array checks within",
        "PPO_steps": 0,
        "optimizer_updates": 0,
        "tree_fits": 0,
        "qualification_evaluated": False,
    }


def _validate_output_not_input(output, inputs):
    for item in inputs.values():
        source_path = Path(item["path"])
        if output == source_path or output in source_path.parents:
            raise ValueError("New collection output cannot contain a frozen input")


def collect(output, *, phase, actor, protocol, actor_sha, protocol_sha, scenarios,
            stop_after_contexts=None, allow_test_fixture=False):
    """Collect or resume one phase only at committed context boundaries."""
    phase = _phase(phase)
    output = Path(output).expanduser().resolve()
    context, inputs = _context(actor, protocol, actor_sha, protocol_sha, scenarios,
                               allow_test_fixture=allow_test_fixture)
    _validate_output_not_input(output, inputs)
    if stop_after_contexts is not None and (type(stop_after_contexts) is not int
                                             or stop_after_contexts < 1):
        raise ValueError("Explicit positive context boundary required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = _plan(context.runtime, context.scenarios, inputs, phase,
                     allow_test_fixture=allow_test_fixture)
        plan_hash = digest(plan)
        if (output / "plan.json").exists():
            if _load(output / "plan.json") != plan:
                raise ValueError("Source, phase, or fixed collection matrix changed")
            state = _load(output / "state.json")
        else:
            _put(output / "plan.json", plan)
            state = {
                "version": VERSION,
                "phase": phase,
                "plan_sha256": plan_hash,
                "status": "prepared",
                "completed": [],
                "pending": None,
                "counts": dict.fromkeys(COUNTERS, 0),
                "elapsed_seconds": 0.0,
            }
            atomic_json(output / "state.json", state)
        if (state.get("version") != VERSION or state.get("phase") != phase
                or state.get("plan_sha256") != plan_hash or state.get("pending") is not None):
            raise ValueError("Pending/unacknowledged or differently bound collection cannot be resampled")
        if state.get("status") == "completed":
            read_data(output, phase=phase, allow_test_fixture=allow_test_fixture)
            return state

        runtime = context.runtime
        scenes = context.scenarios["splits"]["train"]
        count_at_start = len(state["completed"])
        started = time.monotonic()
        for index, ctx in enumerate(plan["contexts"]):
            if index < len(state["completed"]):
                entry = state["completed"][index]
                directory = (output / entry["path"]).parent
                if (entry.get("context") != ctx
                        or file_hash(output / entry["path"]) != entry.get("sha256")
                        or file_hash(directory / "events.jsonl.gz") != entry.get("events_sha256")
                        or _load(directory / "receipt.json") != entry):
                    raise ValueError("Previously confirmed context changed")
                continue
            if (stop_after_contexts is not None
                    and len(state["completed"]) - count_at_start >= stop_after_contexts):
                break

            directory = output / "episodes" / f"{index:04d}"
            directory.mkdir(parents=True, exist_ok=False)
            state.update(status="running", pending={"index": index, "context": ctx})
            atomic_json(output / "state.json", state)
            before_counts = deepcopy(state["counts"])
            data = _empty()
            env = runtime.environment(scenes[ctx["scene_index"]])
            rng = np.random.default_rng(ctx["rng_seed"])
            events = gzip.open(directory / "events.jsonl.gz", "xb", compresslevel=1)
            event_id = 0

            def record(event):
                nonlocal event_id
                key = {
                    "before_step": "branch_steps_attempted",
                    "after_step": "branch_steps",
                    "before_query": "branch_queries_attempted",
                    "after_query": "branch_queries",
                    "before_base_step": "base_steps_attempted",
                    "after_base_step": "base_steps",
                    "before_base_query": "base_queries_attempted",
                    "after_base_query": "base_queries",
                }.get(event["kind"])
                if key is not None:
                    state["counts"][key] += 1
                if (state["counts"]["base_steps_attempted"] > plan["maximum_base_steps"]
                        or state["counts"]["branch_steps_attempted"] > plan["maximum_branch_steps"]):
                    raise ValueError("Declared auxiliary environment budget exceeded")
                if state["counts"]["branch_queries_attempted"] > plan["maximum_branch_queries"]:
                    raise ValueError("Declared branch query budget exceeded")
                row = {
                    "context_id": ctx["id"],
                    "operation_id": f"{index:04d}:{event_id:06d}",
                    "event": event,
                }
                events.write((canonical(row) + "\n").encode())
                events.flush()
                os.fsync(events.fileno())
                event_id += 1
                return True

            try:
                with runtime.verified_context(env):
                    while not env.done and env.state.frame < plan["horizon"]:
                        frame = env.state.frame
                        before = env.snapshot()
                        before_hash = digest(before)
                        groups = {role: critical_groups(env, role) for role in ROLES}
                        program_role = ctx["program_role"]
                        program = (None if program_role < 0 else
                                   partner_action(env, ROLES[program_role], ctx["profile"], rng))
                        if digest(env.snapshot()) != before_hash:
                            raise ValueError("Profile policy changed the live environment")
                        if frame in plan["anchor_frames"]:
                            result = collect_branch_pairs(runtime, env, record_event=record)
                            decision = result["base_decision"]
                            observations = result["base_observations"]
                            proposals = decision["policy_actions"]
                            anchor_id = f"{ctx['id']}:{frame}"
                            indices = {}
                            for branch_index, branch in enumerate(result["branches"]):
                                if branch["done"]:
                                    continue
                                role = branch["target_role"]
                                indices[branch_index] = _row(
                                    data, branch["observations"][role],
                                    branch["next_decision"]["probabilities"][role],
                                    role, ctx, groups[role], frame + 1, "counterfactual",
                                    other_action=branch["other_action"], anchor_id=anchor_id,
                                )
                            for pair in result["pairs"]:
                                baseline, changed = pair["branch_indices"]
                                role = pair["target_role"]
                                data["pairs"].append({
                                    "baseline_index": indices[baseline],
                                    "changed_index": indices[changed],
                                    "target_role": role,
                                    "scene_fingerprint": ctx["fingerprint"],
                                    "groups": groups[role],
                                    "physical_effect": pair["physical_effect"],
                                    "nn_changed": pair["nn_changed"],
                                    "anchor_id": anchor_id,
                                    "other_action": pair["other_action"],
                                })
                        else:
                            record({"kind": "before_base_query", "snapshot": before})
                            proposals, decision = runtime.decision(env)
                            observations = {role: env.observations()[role].tolist() for role in ROLES}
                            record({"kind": "after_base_query", "decision": decision,
                                    "observations": observations})
                        if decision["post_policy_overrides"] != 0 or decision["masks"] is not False:
                            raise ValueError("NN actions altered")
                        submitted = dict(proposals)
                        if program_role >= 0:
                            submitted[ROLES[program_role]] = program
                        for role_index, role in enumerate(ROLES):
                            if role_index != program_role:
                                if submitted[role] != ACTIONS[int(np.argmax(decision["probabilities"][role]))]:
                                    raise ValueError("Executed command differs from frozen NN")
                                _row(data, observations[role], decision["probabilities"][role],
                                     role, ctx, groups[role], frame, "base")
                        record({"kind": "before_base_step", "before": before,
                                "decision": decision, "submitted": submitted})
                        _, reward, terminated, truncated, info = env.step(submitted)
                        if info["requested_actions"] != submitted:
                            raise ValueError("Submitted physical commands differ")
                        record({"kind": "after_base_step", "after": env.snapshot(),
                                "info": info, "rewards": reward,
                                "done": bool(terminated or truncated)})
                events.close()
                _put(directory / "data.json.gz", data)
                entry = {
                    "context": ctx,
                    "path": str((directory / "data.json.gz").relative_to(output)),
                    "sha256": file_hash(directory / "data.json.gz"),
                    "events_sha256": file_hash(directory / "events.jsonl.gz"),
                    "rows": len(data["observations"]),
                    "pairs": len(data["pairs"]),
                    "final_frame": env.state.frame,
                    "counts": {key: state["counts"][key] - before_counts[key] for key in COUNTERS},
                }
                _put(directory / "receipt.json", entry)
                state["completed"].append(entry)
                state.update(pending=None, status="ready",
                             elapsed_seconds=state["elapsed_seconds"] + time.monotonic() - started)
                started = time.monotonic()
                atomic_json(output / "state.json", state)
                print(json.dumps({
                    "phase": phase,
                    "completed_contexts": len(state["completed"]),
                    "total_contexts": len(plan["contexts"]),
                    "rows": entry["rows"],
                    "pairs": entry["pairs"],
                    "counts": state["counts"],
                }), flush=True)
            except BaseException as error:
                events.close()
                state.update(status="failed", failure=repr(error))
                atomic_json(output / "state.json", state)
                raise

        if len(state["completed"]) == len(plan["contexts"]):
            state.update(status="completed")
            atomic_json(output / "state.json", state)
            _put(output / "manifest.json", {
                **state,
                "plan_file_sha256": file_hash(output / "plan.json"),
                "actor_bindings": plan["actor_bindings"],
                "qualification_evaluated": False,
                "independent_acceptance_executed": phase == "acceptance",
            })
        return state


def _validate_plan(plan, phase, scenarios, *, allow_test_fixture=False):
    contexts = context_matrix(scenarios, phase)
    pools = _pool_scenes(scenarios)
    expected_sources = _source_files()
    expected_pool_names = list(PHASE_POOLS[phase])
    expected_counts = {pool: len(pools[pool]) for pool in PHASE_POOLS[phase]}
    expected_ranges = {pool: list(POOL_RANGES[pool]) for pool in PHASE_POOLS[phase]}
    expected_fingerprints = {
        pool: digest([scene["fingerprint"] for scene in pools[pool]])
        for pool in PHASE_POOLS[phase]
    }
    identity = plan.get("runtime_identity", {})
    fixed = (
        plan.get("version") == VERSION
        and plan.get("producer") == VERSION
        and plan.get("data_version") == DATA_VERSION
        and plan.get("phase") == phase
        and plan.get("phase_pools") == expected_pool_names
        and plan.get("pool_ranges") == expected_ranges
        and plan.get("pool_scene_counts") == expected_counts
        and plan.get("pool_fingerprints_sha256") == expected_fingerprints
        and plan.get("context_count") == len(contexts)
        and plan.get("contexts") == contexts
        and plan.get("scenarios_sha256") == digest(scenarios)
        and plan.get("horizon") == HORIZON
        and plan.get("anchor_frames") == list(ANCHOR_FRAMES)
        and plan.get("source_split") == "original_neural_train"
        and plan.get("test_fixture") is allow_test_fixture
        and plan.get("source_files") == expected_sources
        and identity.get("runtime_sources") == runtime_api.runtime_sources()
        and identity.get("runtime_sources_sha256") == digest(runtime_api.runtime_sources())
        and identity.get("runtime_version") == runtime_api.RUNTIME_VERSION
        and identity.get("family") == runtime_api.FAMILY
        and identity.get("test_fixture") is allow_test_fixture
        and plan.get("sampling_uses_tree_predictions") is False
        and plan.get("both_actual_roles_physically_intervened") is True
        and plan.get("labels_from_unmodified_neural_actor") is True
        and plan.get("PPO_steps") == 0
        and plan.get("optimizer_updates") == 0
        and plan.get("tree_fits") == 0
        and plan.get("qualification_evaluated") is False
    )
    if not fixed:
        raise ValueError("Diverse alignment collection plan, phase, range, or source identity differs")
    if not allow_test_fixture and (plan.get("actor_bindings", {}).get("actor_sha256")
                                   != PRODUCTION_ACTOR_SHA256
                                   or plan.get("actor_training_clock")
                                   != PRODUCTION_ACTOR_TRAINING_CLOCK):
        raise ValueError("Saved production data is not bound to the frozen 3.95M Actor")
    for item in plan.get("input_files", {}).values():
        if (not isinstance(item, dict) or set(item) != {"path", "sha256"}
                or not _HASH.fullmatch(str(item.get("sha256", "")))
                or file_hash(Path(item["path"])) != item["sha256"]):
            raise ValueError("Collection input bytes changed")
    if set(plan.get("input_files", {})) != {"actor", "protocol", "scenarios"}:
        raise ValueError("Exact Actor, protocol, and scenario inputs are required")
    scenario_item = plan["input_files"]["scenarios"]
    if _load(scenario_item["path"]) != scenarios:
        raise ValueError("Scenario input semantic content changed")
    return contexts


def _merge_completed(output, plan, manifest):
    """Merge the unchanged data codec into the phase-specific pool names."""
    pools = {pool: _empty() for pool in plan["phase_pools"]}
    for expected, entry in zip(plan["contexts"], manifest["completed"]):
        path = output / entry["path"]
        directory = path.parent
        if (entry.get("context") != expected or file_hash(path) != entry.get("sha256")
                or file_hash(directory / "events.jsonl.gz") != entry.get("events_sha256")
                or _load(directory / "receipt.json") != entry):
            raise ValueError("Acknowledged context data, events, or receipt changed")
        data = _load(path)
        if set(data) != set(_empty()):
            raise ValueError("Collected row codec fields changed")
        lengths = {len(data[key]) for key in data if key != "pairs"}
        if (lengths != {entry["rows"]} or len(data["pairs"]) != entry["pairs"]
                or any(value != expected["id"] for value in data["episode_ids"])
                or any(value != expected["fingerprint"] for value in data["scene_fingerprints"])):
            raise ValueError("Collected row counts or context bindings changed")
        for source in data["row_sources"]:
            if (not isinstance(source, dict) or source.get("context_id") != expected["id"]
                    or source.get("scene_id") != expected["scene_id"]):
                raise ValueError("Collected row provenance changed")
        target = pools[expected["pool"]]
        offset = len(target["observations"])
        for pair in data["pairs"]:
            baseline, changed = pair.get("baseline_index"), pair.get("changed_index")
            if (type(baseline) is not int or type(changed) is not int
                    or not 0 <= baseline < entry["rows"] or not 0 <= changed < entry["rows"]
                    or pair.get("scene_fingerprint") != expected["fingerprint"]):
                raise ValueError("Collected intervention pair binding changed")
            target["pairs"].append({
                **pair,
                "baseline_index": baseline + offset,
                "changed_index": changed + offset,
            })
        for key in data:
            if key != "pairs":
                target[key].extend(data[key])

    fingerprint_sets = {
        pool: set(data["scene_fingerprints"]) for pool, data in pools.items()
    }
    if any(fingerprint_sets[a] & fingerprint_sets[b]
           for index, a in enumerate(pools) for b in tuple(pools)[index + 1:]):
        raise ValueError("Physical explanation pools overlap")
    for data in pools.values():
        data["observations"] = np.asarray(data["observations"], dtype=np.float32)
        data["probabilities"] = np.asarray(data["probabilities"], dtype=np.float32)
    return pools


def read_data(output, phase=None, *, allow_test_fixture=False):
    """Read one complete phase and return its named pools without NN or physics."""
    output = Path(output).expanduser().resolve()
    plan = _load(output / "plan.json")
    saved_phase = plan.get("phase")
    if phase is None:
        phase = saved_phase
    phase = _phase(phase)
    if saved_phase != phase:
        raise ValueError("Requested phase differs from the independently saved collection")
    scenario_path = plan.get("input_files", {}).get("scenarios", {}).get("path")
    if not isinstance(scenario_path, str):
        raise ValueError("Frozen scenario input binding is missing")
    scenarios = _load(scenario_path)
    contexts = _validate_plan(plan, phase, scenarios,
                              allow_test_fixture=allow_test_fixture)
    manifest = _load(output / "manifest.json")
    if (manifest.get("version") != VERSION or manifest.get("phase") != phase
            or manifest.get("status") != "completed" or manifest.get("pending") is not None
            or manifest.get("plan_sha256") != digest(plan)
            or manifest.get("plan_file_sha256") != file_hash(output / "plan.json")
            or manifest.get("actor_bindings") != plan.get("actor_bindings")
            or manifest.get("qualification_evaluated") is not False
            or manifest.get("independent_acceptance_executed") is not (phase == "acceptance")
            or len(manifest.get("completed", [])) != len(contexts)):
        raise ValueError("Only the complete, phase-bound, acknowledged collection may be read")
    return _merge_completed(output, plan, manifest), plan, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=tuple(PHASE_POOLS), required=True)
    for name in ("output", "actor", "protocol", "actor-sha", "protocol-sha", "scenarios"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stop-after-contexts", type=int)
    parser.add_argument("--allow-test-fixture", action="store_true")
    args = parser.parse_args()
    collect(args.output, phase=args.phase, actor=args.actor, protocol=args.protocol,
            actor_sha=args.actor_sha, protocol_sha=args.protocol_sha,
            scenarios=args.scenarios, stop_after_contexts=args.stop_after_contexts,
            allow_test_fixture=args.allow_test_fixture)


if __name__ == "__main__":
    main()
