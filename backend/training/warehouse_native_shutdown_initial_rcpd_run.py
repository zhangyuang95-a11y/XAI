"""Finite first-tree extraction for the selected fixed own-shutdown endpoint.

The public CLI admits only the actual, acknowledged 250k comparison winner.
This workflow owns at most 46,080 auxiliary steps and zero PPO steps. Neither
an initial tree nor this transport reader grants explanation/release permission.
Unconfirmed episode or fit attempts require diagnosis, never automatic replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
import fcntl
from dataclasses import asdict
from hashlib import sha256
import gzip
import io
import json
from pathlib import Path

import numpy as np

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import reserve_sampling
from .warehouse_native_partner_mix_run import write_bytes, write_json, AUTHORIZATION, AUTHORIZATION_SHA
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_initial_rcpd_run import lease, binding, save_episode
from . import warehouse_native_shutdown_result as result_reader
from . import warehouse_native_shutdown_rcpd as extraction
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager

VERSION = "warehouse-native-own-shutdown-initial-rcpd-run.v2"
SOURCE_STEP = 250000
MAXIMUM_AUXILIARY_STEPS = 46080


def sources():
    result = extraction.execution_sources()
    result.update(result_reader.runner.sources())
    for path in (Path(__file__), Path(result_reader.__file__),
                 ROOT / "backend/training/warehouse_native_initial_rcpd_run.py"):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def read_json(path): return json.loads(Path(path).read_bytes())


def bound_bytes(root, value):
    if (type(value) is not dict or set(value) != {"path", "sha256", "size"}
            or type(value["path"]) is not str or type(value["size"]) is not int or value["size"] < 0):
        raise ValueError("Invalid evidence binding")
    root = Path(root).resolve(); relative = Path(value["path"])
    if relative.is_absolute() or not relative.parts or any(p in (".", "..") for p in relative.parts):
        raise ValueError("Evidence path escapes its run")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink(): raise ValueError("Evidence aliases are forbidden")
    raw = path.read_bytes()
    if len(raw) != value["size"] or sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("Bound extraction evidence changed")
    return raw


def _without_counts(descriptor):
    return {k: v for k, v in descriptor.items() if k != "inspection_counts"}


def _checkpoint_weights(payload, descriptor, prepared, protocol, actor):
    """Verify the decoded CPU envelope and all six arrays, without constructing a NN."""
    if (type(payload) is not dict or payload.get("version") != prepared["version"]
            or payload.get("cycle_id") != prepared["cycle_id"] or payload.get("branch") != "own_credit"
            or payload.get("shutdown_arm") != descriptor["shutdown_arm"]
            or payload.get("operation_id") != descriptor["operation_id"]):
        raise ValueError("Confirmed checkpoint envelope identity differs")
    state = payload.get("trainer", {})
    metadata = actor.metadata
    for key in ("cycle_id", "branch", "shutdown_arm", "own_shutdown_beta", "own_shutdown_reward_version",
                "delivery_credit_alpha", "public_feedback_version", "public_feedback_mode", "protocol_sha256",
                "scenario_manifest_sha256", "source_sha256", "initialization_sha256", "source_lineage",
                "source_counters", "joint_steps", "obs_dim", "state_dim", "feature_names", "test_fixture"):
        if digest(state.get(key)) != digest(metadata.get(key)):
            raise ValueError("Confirmed checkpoint and NPZ identity differ: " + key)
    if (state.get("version") != extraction.admission.TRAINER_VERSION
            or state.get("protocol") != protocol or state.get("joint_steps") != descriptor["step"]
            or state.get("test_fixture") is not False):
        raise ValueError("Source trainer version, finite step or original protocol differs")
    model = state.get("model", {})
    weights = {key[len("actor."):]: value for key, value in model.items() if key.startswith("actor.")}
    if set(weights) != set(actor.weights): raise ValueError("Source does not contain exactly the six exported Actor arrays")
    for name, value in weights.items():
        if (not hasattr(value, "device") or value.device.type != "cpu"
                or str(value.dtype) != "torch.float32" or value.requires_grad
                or not np.array_equal(value.numpy(), actor.weights[name])):
            raise ValueError("The exported Actor differs from the actual CPU checkpoint weights")
    return weights


def source_material(source_run, shutdown_arm, source_step=SOURCE_STEP):
    """Capability/ranking before NPZ construction or checkpoint decoding."""
    if source_step != SOURCE_STEP or type(source_step) is not int:
        raise ValueError("Only the predeclared fixed 250000 endpoint may seed the first tree")
    if shutdown_arm not in ("beta0", "beta1"): raise ValueError("Invalid shutdown arm")
    proof = result_reader.compare_fixed(source_run)
    choice = proof.get("selection") or {}
    if (proof.get("status") != "fixed_endpoint_complete" or proof.get("test_fixture") is not False
            or proof.get("primary_endpoint") != SOURCE_STEP or choice.get("selected_arm") != shutdown_arm
            or choice.get("next_stage_capability_eligible") is not True):
        raise ValueError("The completed fixed comparison has no qualified selected source for this arm")
    descriptor, prepared, protocol, scenes = result_reader.registered_source(source_run, shutdown_arm, source_step)
    expected_source = proof["source_descriptors"][source_step][shutdown_arm]
    if (_without_counts(descriptor) != _without_counts(expected_source)
            or descriptor.get("next_stage_capability_eligible") is not True):
        raise ValueError("Selected comparison/source receipts differ")
    source = Path(source_run).resolve(); folder = source / "branches" / shutdown_arm
    report_raw = (folder / "validation" / f"step_{source_step:07d}" / "report.json").read_bytes()
    if sha256(report_raw).hexdigest() != descriptor["validation_report_sha256"]:
        raise ValueError("Selected capability report changed")
    report = json.loads(report_raw)
    actor_path = folder / "actors" / f"actor_{source_step:07d}.npz"
    if file_hash(actor_path) != descriptor["actor_sha256"]: raise ValueError("Selected NPZ changed")
    actor = NumPyNativeActor(actor_path)
    checkpoint = descriptor["checkpoint"]
    checkpoint_raw = bound_bytes(source, {"path": checkpoint["path"], "sha256": checkpoint["sha256"],
        "size": (source / checkpoint["path"]).stat().st_size})
    import torch
    payload = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=False)
    weights = _checkpoint_weights(payload, descriptor, prepared, protocol, actor)
    expected = {k: actor.metadata[k] for k in extraction.REQUIRED_BINDINGS
                if k not in ("actor_sha256", "actor_parameters_sha256")}
    expected.update(actor_sha256=descriptor["actor_sha256"], actor_parameters_sha256=initialization_sha256(weights))
    extraction.validate_actor(actor, expected_bindings=expected, protocol=protocol)
    return descriptor, protocol, scenes, actor, expected, report, proof


def make_contexts(scene_pools, horizon, *, seed_base=4260908, sampling_mode="deterministic"):
    """Pool order is contractual, never JSON/dict insertion order."""
    if (type(scene_pools) is not dict or set(scene_pools) != {"train", "selection"}
            or any(type(scene_pools[k]) is not list for k in ("train", "selection"))
            or type(horizon) is not int or not 1 <= horizon <= 120
            or type(seed_base) is not int or seed_base < 0
            or sampling_mode not in ("deterministic", "stochastic")):
        raise ValueError("Exact train/selection pools and finite episode context parameters required")
    contexts = []
    for pool in ("train", "selection"):
        for index, scene in enumerate(scene_pools[pool]):
            for profile in extraction.PROFILES:
                context = {"pool": pool, "scene_index": index, "scenario_id": scene["id"],
                    "scenario_fingerprint": scene["fingerprint"], "profile": profile,
                    "program_role": -1 if profile == "selfplay" else index % 2,
                    "seed": seed_base + len(contexts), "sampling_mode": sampling_mode,
                    "horizon": horizon, "index": len(contexts)}
                context["episode_id"] = digest(context)
                contexts.append(context)
    return contexts


def _pools(scenes):
    pools = {"train": deepcopy(scenes["splits"]["train"][:64]), "selection": deepcopy(scenes["splits"]["extraction"][:32])}
    sets = [{s["fingerprint"] for s in pools[key]} for key in ("train", "selection")]
    if [len(pools[key]) for key in ("train", "selection")] != [64, 32] or list(map(len, sets)) != [64, 32] or sets[0] & sets[1]:
        raise ValueError("The fixed independent 64/32 development pools are incomplete")
    used = sets[0] | sets[1]
    for split, entries in scenes["splits"].items():
        if split not in ("train", "extraction") and used & {x["fingerprint"] for x in entries}:
            raise ValueError("Extraction initial-state fingerprints overlap a held-out partition")
    if scenes["configuration"]["horizon"] != 120: raise ValueError("Original horizon differs")
    return pools


def prepare(output, *, source_run, shutdown_arm, source_step=SOURCE_STEP):
    output, source = Path(output).expanduser().resolve(), Path(source_run).expanduser().resolve()
    if output.exists(): raise FileExistsError("Keep earlier extraction attempts; choose a new directory")
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Extraction output must be separate from its original source")
    if file_hash(AUTHORIZATION) != AUTHORIZATION_SHA: raise ValueError("Original autonomous authorization changed")
    descriptor, protocol, scenes, actor, expected, report, proof = source_material(source, shutdown_arm, source_step)
    pools = _pools(scenes); contexts = make_contexts(pools, 120)
    plan = {"version": VERSION, "source_descriptor": descriptor, "fixed_comparison_sha256": digest(proof),
        "source_report_sha256": descriptor["validation_report_sha256"],
        "source_capability": {key: report[key] for key in ("capability", "warmup_capability")},
        "actor_bindings": expected, "protocol_sha256": digest(protocol), "scene_pools_sha256": digest(pools),
        "contexts": contexts, "maximum_auxiliary_steps": MAXIMUM_AUXILIARY_STEPS, "ppo_steps": 0,
        "cumulative_fit_step": actor.metadata["source_counters"]["joint_steps"] + actor.metadata["joint_steps"],
        "extraction_config": extraction.extraction_config(), "runtime_sources": sources(),
        "authorization_sha256": AUTHORIZATION_SHA, "test_fixture": False, "formal_ready": False,
        "release_ready": False, "sampling_replay_allowed": False, "fit_replay_allowed": False}
    if sum(c["horizon"] for c in contexts) != MAXIMUM_AUXILIARY_STEPS: raise ValueError("Fixed auxiliary cap differs")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan.json", plan), ("protocol.json", protocol), ("scene_pools.json", pools),
                        ("fixed_comparison.json", proof), ("source_report.json", report)):
        write_json(output / name, value)
    write_bytes(output / "actor.npz", actor.path.read_bytes())
    # The report's source hash is its original bytes, not this canonical copy.
    original_report = source / "branches" / shutdown_arm / "validation" / f"step_{source_step:07d}" / "report.json"
    write_bytes(output / "original_source_report.json", original_report.read_bytes())
    write_bytes(output / "authorization.json", AUTHORIZATION.read_bytes())
    for name in plan["runtime_sources"]: write_bytes(output / "source_snapshot" / name, (ROOT / name).read_bytes())
    with lease(output):
        reserve_sampling(output / "auxiliary_budget.json", 0, cap=MAXIMUM_AUXILIARY_STEPS)
        write_json(output / "manifest.json", {"version": VERSION, "plan_sha256": digest(plan), "status": "prepared",
            "episodes": [], "actual_auxiliary_steps": 0, "reserved_auxiliary_steps": 0, "ppo_steps": 0})
    return {"status": "prepared_first_tree_only", "output": str(output), "episodes": 384,
        "maximum_auxiliary_steps": MAXIMUM_AUXILIARY_STEPS, "ppo_steps": 0, "release_ready": False}


def load_episode(root, entry, *, fixture=False):
    data = json.loads(gzip.decompress(bound_bytes(root, entry["record"])))
    with np.load(io.BytesIO(bound_bytes(root, entry["arrays"])), allow_pickle=False) as values:
        if set(values.files) != {"observations", "probabilities"}: raise ValueError("Unexpected extraction arrays")
        data.update({key: values[key].copy() for key in values.files})
    extraction._validate_data(data, fixture)
    context, episode = entry["context"], data["episode"]
    if (data["data_sha256"] != entry["data_sha256"] or data["joint_transitions"] != entry["actual_joint_steps"]
            or len(data["observations"]) != entry["neural_rows"] or data["pool"] != context["pool"]
            or episode["id"] != context["episode_id"] or episode["fingerprint"] != context["scenario_fingerprint"]
            or any(episode[k] != context[k] for k in ("profile", "program_role", "seed", "sampling_mode", "horizon"))
            or digest(data["trace"]) != episode["trace_sha256"]):
        raise ValueError("Acknowledged extraction episode differs")
    return data


def check_manifest(output, plan, manifest, *, fixture=False):
    if (plan.get("test_fixture") is not fixture or manifest.get("version") != VERSION
            or manifest.get("plan_sha256") != digest(plan) or manifest.get("ppo_steps") != 0
            or plan.get("version") != VERSION): raise ValueError("Extraction manifest identity differs")
    entries = manifest["episodes"]
    if len(entries) > len(plan["contexts"]): raise ValueError("Too many extraction episodes")
    actual = reserved = 0
    for index, entry in enumerate(entries):
        if entry["context"] != plan["contexts"][index]: raise ValueError("Extraction order changed")
        reserved += entry["context"]["horizon"]
        if entry["status"] != "completed": raise ValueError("Pending extraction episode cannot be replayed automatically")
        data = load_episode(output, entry, fixture=fixture)
        if data["actor_bindings"] != plan["actor_bindings"]: raise ValueError("Episode Actor differs")
        if not 0 < data["joint_transitions"] <= entry["context"]["horizon"]: raise ValueError("Episode exceeded reservation")
        actual += data["joint_transitions"]
    budget = read_json(Path(output) / "auxiliary_budget.json")
    if (manifest["actual_auxiliary_steps"] != actual or manifest["reserved_auxiliary_steps"] != reserved
            or budget["reserved_joint_steps"] != reserved or budget["cap"] != plan["maximum_auxiliary_steps"]
            or reserved > plan["maximum_auxiliary_steps"]): raise ValueError("Extraction reservations and confirmations differ")
    return manifest


def merge_saved(output, entries, pool, *, fixture=False):
    """Stream traces one at a time; retain only the 197-row arrays and receipts."""
    merged = None; arrays = {"observations": [], "probabilities": []}; seen = set(); receipts = []; first = None
    for entry in entries:
        if entry["context"]["pool"] != pool: continue
        data = load_episode(output, entry, fixture=fixture); receipt = data["collector_receipt"]
        if merged is None:
            merged = {k: deepcopy(data[k]) for k in ("version", "pool", "actor_bindings", "feature_names", "teacher_rows_in_fit",
                "neural_submitted_overrides", "test_fixture", "extraction_config_sha256", "actor_training_clock")}
            merged.update(episode_ids=[], groups=[], row_sources=[], scene_fingerprints=set(), joint_transitions=0, episodes=[])
            first = deepcopy(receipt)
        elif (any(digest(merged[k]) != digest(data[k]) for k in ("pool", "actor_bindings", "feature_names", "actor_training_clock"))
              or any(digest(first[k]) != digest(receipt[k]) for k in ("actor_metadata", "weights_sha256", "configuration"))):
            raise ValueError("Stored episodes have different Actor, physics or pool bindings")
        if data["episode"]["id"] in seen: raise ValueError("Duplicate stored episode")
        seen.add(data["episode"]["id"])
        for key in arrays: arrays[key].append(data[key])
        for key in ("episode_ids", "groups", "row_sources"): merged[key].extend(data[key])
        merged["scene_fingerprints"].update(data["scene_fingerprints"])
        merged["joint_transitions"] += data["joint_transitions"]
        merged["episodes"].append({**deepcopy(data["episode"]), "data_sha256": data["data_sha256"]})
        receipts.append({"episode": deepcopy(data["episode"]), "collector_receipt_sha256": data["collector_receipt_sha256"]})
        del data
    if merged is None: raise ValueError("No acknowledged data for extraction pool")
    for key, chunks in arrays.items(): merged[key] = np.concatenate(chunks)
    merged["scene_fingerprints"] = sorted(merged["scene_fingerprints"])
    merged["data_sha256"] = extraction.data_api._data_digest(merged)
    contract = {k: first[k] for k in ("weights_sha256", "configuration", "actor_training_clock")}
    contract["metadata"] = first["actor_metadata"]
    extraction._receipt(merged, contract=contract, sources=extraction.execution_sources(), episodes=receipts)
    extraction._validate_data(merged, fixture)
    return merged


def collect_dataset(output, plan, pools, actor, protocol, *, until_episodes=None, fixture=False, config=None):
    """Private fixture-capable primitive. The public runner authenticates a real source first."""
    output = Path(output); manifest = check_manifest(output, plan, read_json(output / "manifest.json"), fixture=fixture)
    limit = len(plan["contexts"]) if until_episodes is None else until_episodes
    if type(limit) is not int or not len(manifest["episodes"]) <= limit <= len(plan["contexts"]): raise ValueError("Invalid episode limit")
    for context in plan["contexts"][len(manifest["episodes"]):limit]:
        reserved = reserve_sampling(output / "auxiliary_budget.json", context["horizon"], cap=plan["maximum_auxiliary_steps"])
        if reserved != manifest["reserved_auxiliary_steps"] + context["horizon"]: raise ValueError("Reservation sequence differs")
        pending = deepcopy(manifest)
        pending["episodes"].append({"status": "pending", "context": deepcopy(context)})
        pending.update(status="sampling", reserved_auxiliary_steps=reserved)
        write_json(output / "manifest.json", pending, replace=True); manifest = pending
        data = None
        try:
            data = extraction.collect_episode(actor, pools[context["pool"]][context["scene_index"]],
                **{k: context[k] for k in ("pool", "profile", "program_role", "seed", "episode_id", "sampling_mode")},
                expected_bindings=plan["actor_bindings"], protocol=protocol, allow_test_fixture=fixture, config=config)
            if not 0 < data["joint_transitions"] <= context["horizon"]: raise ValueError("Episode exceeds reservation")
            saved = save_episode(output, context, data)
            confirmed = deepcopy(manifest); confirmed["episodes"][-1] = saved
            confirmed["actual_auxiliary_steps"] += data["joint_transitions"]
            write_json(output / "manifest.json", confirmed, replace=True)
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "message": str(error), "context": context,
                "known_completed_joint_steps": getattr(error, "actual_joint_steps", data["joint_transitions"] if data is not None else None),
                "known_environment_step_calls": getattr(error, "environment_step_calls", data["joint_transitions"] if data is not None else None),
                "reserved_horizon": context["horizon"], "sampling_replay_allowed": False,
                "confirmation_not_claimed": True}
            # Failure journaling must not replace the acknowledged manifest with
            # an in-memory confirmation. Preserve the original exception too.
            try: write_json(output / "sampling_failure.json", failure)
            except BaseException as journal_error: error.audit_journal_write_error = str(journal_error)
            error.extraction_failure = failure
            raise
        manifest = confirmed
        print(json.dumps({"event": "shutdown_extraction_episode_ack", "episode": len(manifest["episodes"]),
            "episodes": len(plan["contexts"]), "actual_auxiliary_steps": manifest["actual_auxiliary_steps"]}), flush=True)
    if limit == len(plan["contexts"]) and manifest.get("status") not in ("completed", "fitting"):
        manifest["status"] = "sampling_complete"; write_json(output / "manifest.json", manifest, replace=True)
    return manifest


def _fit_request(plan, manifest, actor):
    return {"version": VERSION, "plan_sha256": digest(plan), "acknowledged_episodes_sha256": digest(manifest["episodes"]),
        "prior_manager_state_sha256": digest(None), "feedback_config_sha256": digest(asdict(FeedbackConfig())),
        "actor_sha256": actor.artifact_sha256, "cumulative_fit_step": plan["cumulative_fit_step"],
        "maximum_candidate_fits": 9, "retry_allowed": False}


def fit_collected(output, manifest, plan, actor, *, fixture=False):
    output = Path(output)
    check_manifest(output, plan, manifest, fixture=fixture)
    if len(manifest["episodes"]) != len(plan["contexts"]): raise ValueError("Cannot fit unfinished collection")
    datasets = {pool: merge_saved(output, manifest["episodes"], pool, fixture=fixture) for pool in ("train", "selection")}
    request = _fit_request(plan, manifest, actor)
    request["dataset_receipts"] = {k: v["collector_receipt_sha256"] for k, v in datasets.items()}
    if (output / "fit_request.json").exists():
        if manifest.get("fit_request") is None or json.loads(bound_bytes(output, manifest["fit_request"])) != request:
            raise ValueError("Fit request is changed or unacknowledged")
        if manifest.get("status") != "completed" or manifest.get("fit_result") is None:
            raise ValueError("Pending fit requires diagnosis; automatic re-fitting is forbidden")
        return json.loads(bound_bytes(output, manifest["fit_result"]))
    # Dataset receipt hashes are added to the immutable attempt, and the reader
    # independently rebuilds these same hashes from the acknowledged episodes.
    write_json(output / "fit_request.json", request)
    manifest.update(status="fitting", fit_request=binding(output, output / "fit_request.json"))
    write_json(output / "manifest.json", manifest, replace=True)
    try:
        result = extraction.fit_feedback(datasets["train"], datasets["selection"], step=plan["cumulative_fit_step"],
            feature_names=actor.metadata["feature_names"], allow_test_fixture=fixture)
    except ValueError as error:
        result = {"version": extraction.VERSION, "reliable": False, "fit_report": {"reason": str(error)},
            "manager_state": None, "program": None, "actual_joint_steps": 0, "neural_training_updates": 0,
            "test_fixture": fixture, "explanation_qualified": False, "release_eligible": False,
            "input_rejection": {"error_type": type(error).__name__, "request_sha256": digest(request)},
            "evidence_sha256": digest({"request": request, "reason": str(error)})}
    except BaseException as error:
        failure = {"request_sha256": digest(request), "error_type": type(error).__name__, "message": str(error),
            "fit_replay_allowed": False, "maximum_candidate_fits": 9, "actual_candidate_fits_unknown": True}
        try: write_json(output / "fit_failure.json", failure)
        except BaseException as journal_error: error.audit_journal_write_error = str(journal_error)
        raise
    write_json(output / "fit_result.json", result)
    confirmed = deepcopy(manifest)
    confirmed.update(status="completed", fit_result=binding(output, output / "fit_result.json"),
        reliable=result["reliable"], explanation_qualified=False)
    write_json(output / "manifest.json", confirmed, replace=True)
    manifest.clear(); manifest.update(confirmed)
    return result


def _verify_fit(output, manifest, plan, actor, *, require_reliable=True, fixture=False):
    """Read-boundary validation only: no NN forward, physics step or tree fitting."""
    datasets = {p: merge_saved(output, manifest["episodes"], p, fixture=fixture) for p in ("train", "selection")}
    request = _fit_request(plan, manifest, actor)
    request["dataset_receipts"] = {k: v["collector_receipt_sha256"] for k, v in datasets.items()}
    if json.loads(bound_bytes(output, manifest["fit_request"])) != request: raise ValueError("Fit inputs differ from confirmed data")
    result = json.loads(bound_bytes(output, manifest["fit_result"]))
    if (result.get("version") != extraction.VERSION or result.get("test_fixture") is not fixture
            or result.get("explanation_qualified") is not False or result.get("release_eligible") is not False
            or manifest.get("reliable") is not result.get("reliable")):
        raise ValueError("Fit provenance or qualification differs")
    if result.get("input_rejection"):
        if (result.get("reliable") is not False or result.get("program") is not None or result.get("manager_state") is not None
                or result["input_rejection"].get("request_sha256") != digest(request)
                or result["evidence_sha256"] != digest({"request": request, "reason": result["fit_report"]["reason"]})):
            raise ValueError("Recorded failed fit differs")
    else:
        algorithm = {k: deepcopy(v) for k, v in result.items() if k not in
            ("data_version", "fit_algorithm_version", "algorithm_result_sha256", "shutdown_extraction", "shutdown_evidence_sha256", "release_eligible")}
        algorithm["version"] = extraction.DATA_VERSION
        provenance = {"version": extraction.VERSION, "data_version": extraction.DATA_VERSION,
            "actor_bindings": deepcopy(plan["actor_bindings"]), "actor_metadata": deepcopy(actor.metadata),
            "train_receipt_sha256": datasets["train"]["collector_receipt_sha256"],
            "selection_receipt_sha256": datasets["selection"]["collector_receipt_sha256"],
            "execution_sources": extraction.execution_sources(), "extraction_config_sha256": digest(extraction.extraction_config())}
        if (result.get("data_version") != extraction.DATA_VERSION or result.get("fit_algorithm_version") != extraction.DATA_VERSION
                or result.get("shutdown_extraction") != provenance or result.get("algorithm_result_sha256") != digest(algorithm)
                or result.get("shutdown_evidence_sha256") != digest({"provenance": provenance, "algorithm_result": algorithm})):
            raise ValueError("New producer/old algorithm evidence binding differs")
        manager = FeedbackManager(actor.metadata["feature_names"]); manager.load_state_dict(result["manager_state"])
        fit = result["fit_report"]
        binding_value = {"config_sha256": digest(extraction.data_api.extraction_config()),
            "training_data_sha256": datasets["train"]["data_sha256"], "selection_data_sha256": datasets["selection"]["data_sha256"],
            "actor_sha256": plan["actor_bindings"]["actor_sha256"], "actor_parameters_sha256": plan["actor_bindings"]["actor_parameters_sha256"],
            "cumulative_fit_step": plan["cumulative_fit_step"]}
        if (manager.current_lambda != 0 or manager.reliable is not result["reliable"]
                or manager.last_fit_report != fit or fit.get("observed197_bindings") != binding_value
                or result["program"] != (manager.program.to_dict() if manager.program else None)
                or result["evidence_sha256"] != digest({"binding": binding_value, "fit_report": fit})):
            raise ValueError("Saved program/manager/source/data differ")
        if result["reliable"]:
            if (manager.last_fit_step != plan["cumulative_fit_step"] or manager.program is None
                    or tuple(manager.program.feature_names) != tuple(actor.metadata["feature_names"])
                    or manager.program.metadata.get("native_source_actor_sha256") != plan["actor_bindings"]["actor_sha256"]):
                raise ValueError("Reliable program Actor or fit clock differs")
            metrics = extraction.data_api._fidelity(manager, datasets["selection"])
            if metrics != fit.get("selection_metrics"): raise ValueError("Stored tree fidelity differs from actual saved rows")
            p = manager._predict(manager.program, datasets["selection"]["observations"])
            y = datasets["selection"]["probabilities"]
            mean_kl = float(np.mean(np.sum(y * (np.log(y.clip(1e-8)) - np.log(p.clip(1e-8))), axis=-1)))
            if (metrics["overall"]["fidelity"] < .9 or mean_kl > manager.config.maximum_mean_kl
                    or any(g["rows"] < manager.config.minimum_critical_rows or g["scenarios"] < 10
                           or g["fidelity"] is None or g["fidelity"] < .85 or g["non_wait"]["rows"] == 0
                           or g["non_wait"]["fidelity"] < .85 for g in metrics["critical"].values())):
                raise ValueError("Saved tree does not meet the unchanged reliability gates")
    if require_reliable and result.get("reliable") is not True: raise ValueError("Initial extracted tree is not reliable")
    return result


def _preflight_files(output, *, allow_test_fixture=False):
    """Pure copied-file verification; a successful result is not a source receipt."""
    if type(allow_test_fixture) is not bool: raise ValueError("Explicit fixture scope required")
    plan, protocol, pools = (read_json(output / name) for name in ("plan.json", "protocol.json", "scene_pools.json"))
    if (plan.get("version") != VERSION or plan.get("test_fixture") is not allow_test_fixture or plan.get("ppo_steps") != 0
            or plan.get("runtime_sources") != sources() or plan.get("protocol_sha256") != digest(protocol)
            or plan.get("scene_pools_sha256") != digest(pools) or plan.get("maximum_auxiliary_steps") != MAXIMUM_AUXILIARY_STEPS
            or plan.get("contexts") != make_contexts(pools, 120) or plan.get("extraction_config") != extraction.extraction_config()
            or plan.get("authorization_sha256") != AUTHORIZATION_SHA or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA
            or file_hash(output / "authorization.json") != AUTHORIZATION_SHA):
        raise ValueError("Frozen production extraction preparation differs")
    for name, sha in plan["runtime_sources"].items():
        raw = bound_bytes(output / "source_snapshot", {"path": name, "sha256": sha, "size": (ROOT / name).stat().st_size})
    return plan, protocol, pools


def _read_preparation(output):
    plan, protocol, pools = _preflight_files(output)
    descriptor = plan["source_descriptor"]
    actual, actual_protocol, scenes, actor, expected, report, proof = source_material(descriptor["run"], descriptor["shutdown_arm"], descriptor["step"])
    if (actual != descriptor or actual_protocol != protocol or _pools(scenes) != pools or expected != plan["actor_bindings"]
            or digest(proof) != plan["fixed_comparison_sha256"] or read_json(output / "fixed_comparison.json") != json.loads(json.dumps(proof))
            or file_hash(output / "original_source_report.json") != plan["source_report_sha256"]
            or read_json(output / "source_report.json") != report
            or file_hash(output / "actor.npz") != actor.artifact_sha256
            or plan["cumulative_fit_step"] != actor.metadata["source_counters"]["joint_steps"] + actor.metadata["joint_steps"]):
        raise ValueError("First-tree selected fixed source or copied material changed")
    return plan, protocol, pools, actor


@contextmanager
def _read_lease(output):
    path = output / "extraction.lock"
    if path.is_symlink(): raise ValueError("Extraction lock alias is forbidden")
    with path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_bundle(output, *, expected_manifest_sha256, require_reliable=True):
    output = Path(output).expanduser().resolve()
    if file_hash(output / "manifest.json") != expected_manifest_sha256: raise ValueError("External first-tree manifest binding differs")
    # No fixture switch: synthetic transportation cannot acquire a production source gate.
    with _read_lease(output):
        plan, protocol, pools, actor = _read_preparation(output)
        manifest = check_manifest(output, plan, read_json(output / "manifest.json"))
        if manifest["status"] != "completed" or len(manifest["episodes"]) != len(plan["contexts"]):
            raise ValueError("Fixed extraction is incomplete")
        result = _verify_fit(output, manifest, plan, actor, require_reliable=require_reliable)
        if file_hash(output / "manifest.json") != expected_manifest_sha256: raise ValueError("Manifest changed while reading")
        return {"plan": plan, "manifest": manifest, "fit_result": result, "actor": actor, "protocol": protocol,
            "pools": pools, "manifest_sha256": expected_manifest_sha256, "release_ready": False,
            "explanation_qualified": False, "sampling_steps_in_reader": 0, "neural_forwards_in_reader": 0}


def run(output, *, until_episodes=None):
    output = Path(output).expanduser().resolve()
    with lease(output):
        plan, protocol, pools, actor = _read_preparation(output)
        manifest = collect_dataset(output, plan, pools, actor, protocol, until_episodes=until_episodes)
        if len(manifest["episodes"]) != len(plan["contexts"]): return {"status": "episode_boundary_completed", "episodes": len(manifest["episodes"])}
        result = fit_collected(output, manifest, plan, actor)
        return {"status": "first_tree_completed", "reliable": result["reliable"], "ppo_steps": 0,
            "actual_auxiliary_steps": manifest["actual_auxiliary_steps"], "reserved_auxiliary_steps": manifest["reserved_auxiliary_steps"],
            "release_ready": False, "explanation_qualified": False, "website_model_changed": False,
            "manifest_sha256": file_hash(output / "manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true"); action.add_argument("--run", action="store_true"); action.add_argument("--verify", action="store_true")
    parser.add_argument("--output", required=True); parser.add_argument("--source-run")
    parser.add_argument("--shutdown-arm", choices=("beta0", "beta1")); parser.add_argument("--source-step", type=int, default=SOURCE_STEP)
    parser.add_argument("--until-episodes", type=int); parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--allow-unreliable", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        if not args.source_run or not args.shutdown_arm: parser.error("--prepare requires --source-run and --shutdown-arm")
        result = prepare(args.output, source_run=args.source_run, shutdown_arm=args.shutdown_arm, source_step=args.source_step)
    elif args.run: result = run(args.output, until_episodes=args.until_episodes)
    else:
        if not args.expected_manifest_sha256: parser.error("--verify requires an external --expected-manifest-sha256")
        bundle = read_bundle(args.output, expected_manifest_sha256=args.expected_manifest_sha256, require_reliable=not args.allow_unreliable)
        result = {"status": "bound_first_tree_read", "reliable": bundle["fit_result"]["reliable"],
            "manifest_sha256": bundle["manifest_sha256"], "release_ready": False, "explanation_qualified": False}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
