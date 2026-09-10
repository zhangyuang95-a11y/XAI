"""Independent first-tree extraction after a confirmed public-history warmup.

This workflow owns auxiliary sampling only. It neither changes a learner nor
publishes a model. A pending episode is never replayed after a lost response.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import gzip
import io
import json
from pathlib import Path
import time

import numpy as np

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import reserve_sampling
from .warehouse_native_partner_mix_run import write_bytes, write_json, decode, AUTHORIZATION, AUTHORIZATION_SHA
from .warehouse_native_public_feedback_initialization import initialization_sha256
from . import warehouse_native_continuation_rcpd as extraction
from . import warehouse_native_continuation_run as continuation
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.feedback import FeedbackConfig

VERSION = "warehouse-native-initial-observed197-rcpd-run.v1"


def sources():
    result = continuation.sources(); result.update(extraction.execution_sources())
    for path in (Path(__file__), ROOT / "env/warehouse_native/feedback.py", *sorted((ROOT / "core").glob("*.py"))):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def read_json(path): return json.loads(Path(path).read_bytes())


@contextmanager
def lease(output):
    with (Path(output) / "extraction.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def binding(root, path):
    return {"path": str(path.relative_to(root)), "sha256": file_hash(path), "size": path.stat().st_size}


def bound_file(root, value):
    path = Path(root) / value["path"]
    if (Path(value["path"]).is_absolute() or ".." in Path(value["path"]).parts
            or not path.resolve().is_relative_to(Path(root).resolve())
            or path.stat().st_size != value["size"] or file_hash(path) != value["sha256"]):
        raise ValueError("Bound extraction evidence changed or escapes its run")
    return path


def source_material(source_run, source_branch, source_step):
    descriptor, prepared, protocol, scenes = continuation._registered_source(
        source_run, source_branch, source_step)
    source_run = Path(source_run).resolve()
    folder = source_run / "branches" / source_branch
    report_path = folder / "validation" / f"step_{source_step:07d}" / "report.json"
    report = read_json(report_path)
    if not (report["warmup_capability"]["eligible"] or report["capability"]["eligible"]):
        raise ValueError("The confirmed source has not passed the frozen basic capability gate")
    actor = NumPyNativeActor(folder / "actors" / f"actor_{source_step:07d}.npz")
    # CPU tensor decoding only: compare every exact exported weight, with no
    # model construction, forward, optimizer call or active MPS operation.
    payload = decode(source_run / descriptor["checkpoint"]["path"], descriptor["checkpoint"]["sha256"])
    if payload["version"] != prepared["version"]: raise ValueError("Source checkpoint identity differs")
    weights = {key[len("actor."):]: value for key, value in payload["trainer"]["model"].items() if key.startswith("actor.")}
    if set(weights) != set(actor.weights) or any(not np.array_equal(value.numpy(), actor.weights[key]) for key, value in weights.items()):
        raise ValueError("The extracted Actor differs from its confirmed Torch checkpoint")
    expected = {key: actor.metadata[key] for key in extraction.REQUIRED_BINDINGS if key not in ("actor_sha256", "actor_parameters_sha256")}
    expected.update(actor_sha256=actor.artifact_sha256, actor_parameters_sha256=initialization_sha256(weights))
    for key in ("cycle_id", "feedback_branch"):
        if key in actor.metadata: expected[key] = actor.metadata[key]
    extraction.validate_actor(actor, expected_bindings=expected, protocol=protocol)
    return descriptor, protocol, scenes, actor, expected, report, file_hash(report_path)


def make_contexts(scene_pools, horizon, *, seed_base=4260908, sampling_mode="deterministic"):
    contexts = []
    for pool, scenes in scene_pools.items():
        for index, scene in enumerate(scenes):
            for profile_index, profile in enumerate(extraction.PROFILES):
                context = {"pool": pool, "scene_index": index, "scenario_id": scene["id"],
                    "scenario_fingerprint": scene["fingerprint"], "profile": profile,
                    "program_role": -1 if profile == "selfplay" else index % 2,
                    "seed": seed_base + len(contexts), "sampling_mode": sampling_mode,
                    "horizon": horizon, "index": len(contexts)}
                context["episode_id"] = digest(context)
                contexts.append(context)
    return contexts


def prepare(output, *, source_run, source_branch, source_step):
    output = Path(output).expanduser().resolve()
    if output.exists(): raise FileExistsError("First-tree extraction needs an independent new directory")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA: raise ValueError("Autonomous local authorization changed")
    descriptor, protocol, scenes, actor, expected, report, report_sha = source_material(source_run, source_branch, source_step)
    pools = {"train": deepcopy(scenes["splits"]["train"][:64]), "selection": deepcopy(scenes["splits"]["extraction"][:32])}
    if len(pools["train"]) != 64 or len(pools["selection"]) != 32: raise ValueError("Fixed source pool is incomplete")
    a, b = ({scene["fingerprint"] for scene in pool} for pool in pools.values())
    if len(a) != 64 or len(b) != 32 or a & b: raise ValueError("Actual initial-state partitions overlap")
    horizon = scenes["configuration"]["horizon"]
    if horizon != 120: raise ValueError("Original public task horizon changed")
    contexts = make_contexts(pools, horizon)
    plan = {"version": VERSION, "source_descriptor": descriptor, "source_report_sha256": report_sha,
        "source_capability": {key: report[key] for key in ("capability", "warmup_capability")},
        "actor_bindings": expected, "protocol_sha256": digest(protocol), "scene_pools_sha256": digest(pools),
        "contexts": contexts, "maximum_auxiliary_steps": sum(c["horizon"] for c in contexts), "ppo_steps": 0,
        "cumulative_fit_step": actor.metadata["source_counters"]["joint_steps"] + actor.metadata["joint_steps"],
        "extraction_config": extraction.extraction_config(), "runtime_sources": sources(),
        "authorization_sha256": AUTHORIZATION_SHA, "test_fixture": False, "formal_ready": False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan); write_json(output / "protocol.json", protocol)
    write_json(output / "scene_pools.json", pools); write_bytes(output / "actor.npz", actor.path.read_bytes())
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    for name in plan["runtime_sources"]: write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    with lease(output):
        reserve_sampling(output / "auxiliary_budget.json", 0, cap=plan["maximum_auxiliary_steps"])
        manifest = {"version": VERSION, "plan_sha256": digest(plan), "status": "prepared", "episodes": [],
            "actual_auxiliary_steps": 0, "reserved_auxiliary_steps": 0, "ppo_steps": 0}
        write_json(output / "manifest.json", manifest)
    return {"status": "prepared_first_tree_only", "output": str(output), "episodes": len(contexts),
        "maximum_auxiliary_steps": plan["maximum_auxiliary_steps"], "ppo_steps": 0}


def save_episode(root, context, data):
    folder = Path(root) / "episodes" / context["episode_id"]
    folder.mkdir(parents=True, exist_ok=False)
    raw = {key: value for key, value in data.items() if key not in ("observations", "probabilities")}
    arrays = io.BytesIO(); np.savez_compressed(arrays, observations=data["observations"], probabilities=data["probabilities"])
    write_bytes(folder / "arrays.npz", arrays.getvalue())
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    write_bytes(folder / "episode.json.gz", gzip.compress(encoded, compresslevel=1, mtime=0))
    return {"context": deepcopy(context), "status": "completed", "data_sha256": data["data_sha256"],
        "actual_joint_steps": data["joint_transitions"], "neural_rows": len(data["observations"]),
        "arrays": binding(root, folder / "arrays.npz"), "record": binding(root, folder / "episode.json.gz")}


def load_episode(root, entry, *, fixture=False):
    record = bound_file(root, entry["record"]); arrays = bound_file(root, entry["arrays"])
    data = json.loads(gzip.decompress(record.read_bytes()))
    with np.load(arrays, allow_pickle=False) as values:
        if set(values.files) != {"observations", "probabilities"}: raise ValueError("Unexpected extraction arrays")
        data.update({key: values[key].copy() for key in values.files})
    extraction._validate_data(data, fixture)
    context = entry["context"]; episode = data["episode"]
    if (data["data_sha256"] != entry["data_sha256"] or data["joint_transitions"] != entry["actual_joint_steps"]
            or len(data["observations"]) != entry["neural_rows"] or data["pool"] != context["pool"]
            or episode["id"] != context["episode_id"] or episode["fingerprint"] != context["scenario_fingerprint"]
            or any(episode[k] != context[k] for k in ("profile", "program_role", "seed", "sampling_mode", "horizon"))
            or digest(data["trace"]) != episode["trace_sha256"]):
        raise ValueError("Acknowledged episode binding differs")
    return data


def check_manifest(output, plan, manifest, *, fixture=False):
    if manifest.get("version") != VERSION or manifest.get("plan_sha256") != digest(plan): raise ValueError("Extraction manifest identity differs")
    entries = manifest["episodes"]
    if len(entries) > len(plan["contexts"]): raise ValueError("Too many extraction episodes")
    actual = reserved = 0
    for index, entry in enumerate(entries):
        if digest(entry["context"]) != digest(plan["contexts"][index]): raise ValueError("Extraction episode order changed")
        reserved += entry["context"]["horizon"]
        if entry["status"] != "completed": raise ValueError("Pending extraction episode requires diagnosis; it will not be replayed")
        data = load_episode(output, entry, fixture=fixture)
        if data["actor_bindings"] != plan["actor_bindings"]: raise ValueError("Episode belongs to a different frozen Actor")
        actual += data["joint_transitions"]
    budget = read_json(Path(output) / "auxiliary_budget.json")
    if (manifest["actual_auxiliary_steps"] != actual or manifest["reserved_auxiliary_steps"] != reserved
            or budget["reserved_joint_steps"] != reserved or budget["cap"] != plan["maximum_auxiliary_steps"]):
        raise ValueError("Extraction reservations and acknowledged records differ")
    return manifest


def merge_saved(output, entries, pool, *, fixture=False):
    """Validate each stored trace, then release it before reading the next one."""
    merged = None; arrays = {"observations": [], "probabilities": []}; seen = set()
    for entry in entries:
        if entry["context"]["pool"] != pool: continue
        data = load_episode(output, entry, fixture=fixture)
        if merged is None:
            merged = {key: deepcopy(data[key]) for key in ("version", "pool", "actor_bindings", "feature_names",
                "teacher_rows_in_fit", "neural_submitted_overrides", "test_fixture", "extraction_config_sha256", "actor_training_clock")}
            merged.update(episode_ids=[], groups=[], row_sources=[], scene_fingerprints=set(), joint_transitions=0, episodes=[])
        elif any(digest(merged[key]) != digest(data[key]) for key in ("pool", "actor_bindings", "feature_names", "actor_training_clock")):
            raise ValueError("Stored extraction episodes have different Actor or pool bindings")
        if data["episode"]["id"] in seen: raise ValueError("Duplicate stored extraction episode")
        seen.add(data["episode"]["id"])
        for key in arrays: arrays[key].append(data[key])
        for key in ("episode_ids", "groups", "row_sources"): merged[key].extend(data[key])
        merged["scene_fingerprints"].update(data["scene_fingerprints"])
        merged["joint_transitions"] += data["joint_transitions"]
        merged["episodes"].append({**deepcopy(data["episode"]), "data_sha256": data["data_sha256"]})
        del data
    if merged is None: raise ValueError("No acknowledged data for extraction pool")
    for key, chunks in arrays.items(): merged[key] = np.concatenate(chunks, axis=0)
    merged["scene_fingerprints"] = sorted(merged["scene_fingerprints"])
    merged["data_sha256"] = extraction._data_digest(merged)
    extraction._validate_data(merged, fixture)
    return merged


def collect_dataset(output, plan, pools, actor, protocol, *, until_episodes=None, fixture=False, config=None):
    """Caller owns the exclusive run lease. Completed episodes are read, never sampled again."""
    output = Path(output); manifest = check_manifest(output, plan, read_json(output / "manifest.json"), fixture=fixture)
    limit = len(plan["contexts"]) if until_episodes is None else until_episodes
    if type(limit) is not int or not len(manifest["episodes"]) <= limit <= len(plan["contexts"]):
        raise ValueError("Invalid fixed episode limit")
    for context in plan["contexts"][len(manifest["episodes"]):limit]:
        reserved = reserve_sampling(output / "auxiliary_budget.json", context["horizon"], cap=plan["maximum_auxiliary_steps"])
        if reserved != manifest["reserved_auxiliary_steps"] + context["horizon"]: raise ValueError("Unexpected reservation sequence")
        manifest["episodes"].append({"status": "pending", "context": deepcopy(context)})
        manifest.update(status="sampling", reserved_auxiliary_steps=reserved)
        write_json(output / "manifest.json", manifest, replace=True)
        data = None
        try:
            data = extraction.collect_episode(actor, pools[context["pool"]][context["scene_index"]],
                **{key: context[key] for key in ("pool", "profile", "program_role", "seed", "episode_id", "sampling_mode")},
                expected_bindings=plan["actor_bindings"], protocol=protocol, allow_test_fixture=fixture, config=config)
            if not 0 < data["joint_transitions"] <= context["horizon"]: raise ValueError("Actual episode exceeds its reservation")
            saved = save_episode(output, context, data)
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "message": str(error), "context": context,
                "known_completed_joint_steps": getattr(error, "actual_joint_steps", data["joint_transitions"] if data is not None else None),
                "known_environment_step_calls": getattr(error, "environment_step_calls", data["joint_transitions"] if data is not None else None),
                "reserved_horizon": context["horizon"], "sampling_replay_allowed": False}
            manifest["episodes"][-1]["failure"] = failure
            write_json(output / "manifest.json", manifest, replace=True)
            raise
        manifest["episodes"][-1] = saved
        manifest["actual_auxiliary_steps"] += data["joint_transitions"]
        write_json(output / "manifest.json", manifest, replace=True)
        print(json.dumps({"event": "extraction_episode_ack", "episode": len(manifest["episodes"]),
            "episodes": len(plan["contexts"]), "actual_auxiliary_steps": manifest["actual_auxiliary_steps"]}), flush=True)
    if limit == len(plan["contexts"]):
        if manifest.get("status") != "completed":
            manifest["status"] = "sampling_complete"; write_json(output / "manifest.json", manifest, replace=True)
    return manifest


def fit_collected(output, manifest, plan, actor, *, prior_manager_state=None, feedback_config=None, fixture=False):
    """Fit acknowledged real data; no environment or neural sampling here."""
    output = Path(output)
    if len(manifest["episodes"]) != len(plan["contexts"]): raise ValueError("Cannot fit an unfinished fixed collection")
    request = {"plan_sha256": digest(plan), "acknowledged_episodes_sha256": digest(manifest["episodes"]),
        "prior_manager_state_sha256": digest(prior_manager_state),
        "feedback_config_sha256": digest(asdict(feedback_config or FeedbackConfig())),
        "actor_sha256": actor.artifact_sha256, "cumulative_fit_step": plan["cumulative_fit_step"]}
    request_path = output / "fit_request.json"
    if request_path.exists():
        if read_json(request_path) != request: raise ValueError("Fit source, prior schedule, or configuration changed")
        if manifest.get("fit_request") is None: raise ValueError("Unacknowledged fit request requires diagnosis")
        bound_file(output, manifest["fit_request"])
    else:
        write_json(request_path, request)
        manifest.update(status="fitting", fit_request=binding(output, request_path))
        write_json(output / "manifest.json", manifest, replace=True)
    if (output / "fit_result.json").exists():
        if manifest.get("fit_result") is None: raise ValueError("Unacknowledged fit requires diagnosis")
        return read_json(bound_file(output, manifest["fit_result"]))
    merged = {}
    for pool in ("train", "selection"):
        merged[pool] = merge_saved(output, manifest["episodes"], pool, fixture=fixture)
    try:
        result = extraction.fit_feedback(merged["train"], merged["selection"], step=plan["cumulative_fit_step"],
            feature_names=actor.metadata["feature_names"], prior_manager_state=prior_manager_state,
            feedback_config=feedback_config, allow_test_fixture=fixture)
    except ValueError as error:
        # Overlap/coverage input rejection remains a real failed result. The
        # recorded episodes stay intact, and no dataset row is filtered away.
        result = {"version": extraction.VERSION, "reliable": False, "fit_report": {"reason": str(error)},
            "manager_state": None, "program": None, "actual_joint_steps": 0, "neural_training_updates": 0,
            "test_fixture": fixture, "explanation_qualified": False,
            "evidence_sha256": digest({"plan_sha256": digest(plan), "error": str(error),
                "datasets": {key: value["data_sha256"] for key, value in merged.items()}})}
    write_json(output / "fit_result.json", result)
    manifest.update(status="completed", fit_result=binding(output, output / "fit_result.json"),
        reliable=result["reliable"], explanation_qualified=False)
    write_json(output / "manifest.json", manifest, replace=True)
    return result


def read_bundle(output, *, expected_manifest_sha256=None, require_reliable=True, fixture=False):
    """Bound first-tree input for a future pair; never grants release permission."""
    if fixture: raise ValueError("Production bundle reader does not fabricate foundation fixture receipts")
    output = Path(output).expanduser().resolve()
    with lease(output):
        plan, protocol, pools = (read_json(output / name) for name in ("plan.json", "protocol.json", "scene_pools.json"))
        if (plan["version"] != VERSION or plan["test_fixture"] is not False or plan["runtime_sources"] != sources()
                or plan["protocol_sha256"] != digest(protocol) or plan["scene_pools_sha256"] != digest(pools)
                or plan["authorization_sha256"] != file_hash(output / "authorization.json")
                or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA): raise ValueError("Extraction bundle preparation changed")
        for name, sha in plan["runtime_sources"].items():
            if file_hash(output / "source_snapshot" / name) != sha: raise ValueError("Extraction bundle source archive changed")
        if expected_manifest_sha256 is not None and file_hash(output / "manifest.json") != expected_manifest_sha256:
            raise ValueError("Selected first-tree manifest differs from its caller binding")
        descriptor = plan["source_descriptor"]
        actual, _, _, _, expected, _, report_sha = source_material(descriptor["run"], descriptor["branch"], descriptor["step"])
        if actual != descriptor or expected != plan["actor_bindings"] or report_sha != plan["source_report_sha256"]:
            raise ValueError("First-tree foundation ancestry changed")
        manifest = check_manifest(output, plan, read_json(output / "manifest.json"))
        if manifest["status"] != "completed" or len(manifest["episodes"]) != len(plan["contexts"]):
            raise ValueError("First-tree extraction has not completed its registered data and fit")
        actor = NumPyNativeActor(output / "actor.npz")
        extraction.validate_actor(actor, expected_bindings=plan["actor_bindings"], protocol=protocol)
        result = read_json(bound_file(output, manifest["fit_result"]))
        request = read_json(bound_file(output, manifest["fit_request"]))
        expected_request = {"plan_sha256": digest(plan), "acknowledged_episodes_sha256": digest(manifest["episodes"]),
            "prior_manager_state_sha256": digest(None), "feedback_config_sha256": digest(asdict(FeedbackConfig())),
            "actor_sha256": actor.artifact_sha256, "cumulative_fit_step": plan["cumulative_fit_step"]}
        if request != expected_request: raise ValueError("Initial fit inputs do not match the acknowledged collection")
        if result.get("test_fixture") is not False or result.get("explanation_qualified") is not False:
            raise ValueError("First-tree provenance or qualification differs")
        if require_reliable:
            from env.warehouse_native.feedback import FeedbackManager
            if result.get("reliable") is not True or not result.get("manager_state"): raise ValueError("Initial extracted tree is not reliable")
            manager = FeedbackManager(actor.metadata["feature_names"])
            manager.load_state_dict(result["manager_state"])
            fit = result["fit_report"]
            data_hashes = {pool: merge_saved(output, manifest["episodes"], pool)["data_sha256"] for pool in pools}
            if (not manager.reliable or manager.current_lambda != 0 or digest(manager.last_fit_report) != digest(fit)
                    or manager.last_fit_step != plan["cumulative_fit_step"]
                    or fit["source_actor_sha256"] != plan["actor_bindings"]["actor_sha256"]
                    or fit["observed197_bindings"]["actor_parameters_sha256"] != plan["actor_bindings"]["actor_parameters_sha256"]
                    or fit["observed197_bindings"]["training_data_sha256"] != data_hashes["train"]
                    or fit["observed197_bindings"]["selection_data_sha256"] != data_hashes["selection"]
                    or result["evidence_sha256"] != digest({"binding": fit["observed197_bindings"], "fit_report": fit})):
                raise ValueError("First-tree fit does not match its actual bound Actor/data/clock")
        return {"plan": plan, "manifest": manifest, "fit_result": result, "actor": actor, "protocol": protocol,
            "pools": pools, "manifest_sha256": file_hash(output / "manifest.json")}


def run(output, *, until_episodes=None):
    output = Path(output).expanduser().resolve()
    with lease(output):
        plan, protocol, pools = (read_json(output / name) for name in ("plan.json", "protocol.json", "scene_pools.json"))
        if (plan["version"] != VERSION or plan["test_fixture"] is not False or plan["runtime_sources"] != sources()
                or plan["protocol_sha256"] != digest(protocol) or plan["scene_pools_sha256"] != digest(pools)
                or plan["authorization_sha256"] != file_hash(output / "authorization.json")
                or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA): raise ValueError("Frozen extraction preparation changed")
        for name, sha in plan["runtime_sources"].items():
            if file_hash(output / "source_snapshot" / name) != sha: raise ValueError("Archived extraction source changed")
        descriptor = plan["source_descriptor"]
        actual, _, _, _, expected, _, report_sha = source_material(descriptor["run"], descriptor["branch"], descriptor["step"])
        if actual != descriptor or expected != plan["actor_bindings"] or report_sha != plan["source_report_sha256"]:
            raise ValueError("The acknowledged foundation source changed")
        actor = NumPyNativeActor(output / "actor.npz")
        extraction.validate_actor(actor, expected_bindings=plan["actor_bindings"], protocol=protocol)
        manifest = collect_dataset(output, plan, pools, actor, protocol, until_episodes=until_episodes)
        if len(manifest["episodes"]) != len(plan["contexts"]): return {"status": "episode_boundary_completed", "episodes": len(manifest["episodes"])}
        result = fit_collected(output, manifest, plan, actor)
        return {"status": "first_tree_completed", "reliable": result["reliable"],
            "actual_auxiliary_steps": manifest["actual_auxiliary_steps"], "reserved_auxiliary_steps": manifest["reserved_auxiliary_steps"],
            "ppo_steps": 0, "explanation_qualified": False, "website_model_changed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true"); action.add_argument("--run", action="store_true")
    parser.add_argument("--output", required=True); parser.add_argument("--source-run")
    parser.add_argument("--source-branch"); parser.add_argument("--source-step", type=int)
    parser.add_argument("--until-episodes", type=int)
    args = parser.parse_args()
    result = prepare(args.output, source_run=args.source_run, source_branch=args.source_branch, source_step=args.source_step) if args.prepare else run(args.output, until_episodes=args.until_episodes)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
