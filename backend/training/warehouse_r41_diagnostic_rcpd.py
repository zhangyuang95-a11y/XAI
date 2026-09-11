"""Intervention-aware RCPD for the frozen r4.1 diagnostic Actor.

This is additive evidence.  It neither edits the terminal Actor nor revises the
failed r4.1 behavior admission.  Exact observation hashes form indivisible
train/validation groups; intervention endpoints from the strict diagnostic
environment receive higher fitting weight.  A separate final-test audit is
still required before the program can be used for participant explanations.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import warnings

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_evaluation import critical_groups
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_model_tree import (
    R41DiagnosticModelTreeProgram,
    VERSION as MODEL_TREE_VERSION,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-rcpd.v3"
SCENE_MANIFEST_VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
WORKLOAD_SCREEN_VERSION = "warehouse-r41-diagnostic-workload-screen.v2"
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
ROUTER_DEPTH = 8
ROUTER_LEAVES = 32
ROUTER_MIN_SAMPLES_LEAF = 64
RANKING_C = 1.0
# The v3 diagnostic rows contain one finite sparse46/C=30 leaf that reaches
# convergence at iteration 2,090.  A 3,000-iteration ceiling keeps that
# pre-registered candidate in the failure ledger instead of aborting evidence
# generation before the hard fidelity gates can be evaluated.
LOGISTIC_MAX_ITER = 3000
MODEL_TREE_PROFILES = tuple(
    {"name": f"sparse{feature_cap}_c{str(logistic_c).replace('.', '_')}",
     "feature_cap": feature_cap, "logistic_c": logistic_c}
    for feature_cap in (40, 42, 44, 46, 48, 50, 52)
    for logistic_c in (.3, 1.0, 3.0, 10.0, 30.0)
) + ({"name": "dense_c1", "feature_cap": None, "logistic_c": 1.0},)
MIN_OVERALL = .90
MIN_NONWAIT = .90
MIN_CRITICAL = .85
MIN_DIRECTION = .85
MAX_JSON_BYTES = 256 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset((
    "observations", "probabilities", "action_indices", "weights",
    "observation_hashes", "scene_fingerprints", "episode_ids", "frames",
    "group_bits", "kinds", "anchor_ids", "branch_actions",
    "physical_hashes", "submitted_equal", "split_validation",
))
_CANDIDATE_FIELDS = frozenset((
    "profile", "metrics", "gate", "complexity", "fit_diagnostics",
    "program_content_sha256", "program",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "scene_manifest_version": SCENE_MANIFEST_VERSION,
        "source_splits": ["train", "conflict_validation"],
        "holdout_split": "final_test",
        "partners": list(PARTNERS),
        "evaluated_role": "robot_2",
        "partition": (
            "connected components of exact float32 observation hashes; "
            "intervention anchors kept intact"
        ),
        "validation_component_modulus": 5,
        "intervention_anchor": "every fifth critical pre-action frame",
        "intervention_actions": list(ACTIONS),
        "program_family": "axis routing tree with sparse public-input linear-softmax leaves",
        "fit_access": {
            "fit_rows": "train only",
            "prediction_input": "public observation vector only",
            "actor_parameter_arrays": False,
            "actor_logits": False,
            "actor_hidden_states": False,
            "intervention_branch_metadata": False,
            "physical_hash": False,
        },
        "weights": {
            "ordinary": 1.0,
            "nonwait_multiplier": 1.5,
            "critical_multiplier": 2.0,
            "intervention_multiplier": 4.0,
            "actor_changed_intervention_multiplier": 4.0,
            "action_balance": (
                "inverse square-root frequency from training labels only, "
                "normalized to training-frequency-weighted mean one"
            ),
        },
        "router": {"kind": "axis CART classifier", "depth_cap": ROUTER_DEPTH,
                   "leaf_cap": ROUTER_LEAVES,
                   "minimum_samples_leaf": ROUTER_MIN_SAMPLES_LEAF,
                   "random_state": 41},
        "leaf": {"kind": "multinomial public-input linear-softmax",
                 "ranking_fit_c": RANKING_C,
                 "profiles": [dict(row) for row in MODEL_TREE_PROFILES],
                 "solver": "lbfgs", "maximum_iterations": LOGISTIC_MAX_ITER,
                 "standardization": "train-only scaler folded into raw coefficients"},
        "candidate_selection": "minimum nonzero coefficients, then router size and mean KL",
        "maximum_depth": ROUTER_DEPTH,
        "maximum_leaves": ROUTER_LEAVES,
        "thresholds": {
            "overall": MIN_OVERALL,
            "nonwait": MIN_NONWAIT,
            "critical": MIN_CRITICAL,
            "effective_intervention_direction": MIN_DIRECTION,
        },
        "ppo_joint_steps": 0,
        "optimizer_updates": 0,
        "actor_changed": False,
        "runtime_action_override": False,
        "independent_final_test_audit_required": True,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return local_source_hashes((Path(__file__),))


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read(path: Path, label: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(label + " is oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Diagnostic RCPD output path is unsafe")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write(path, (canonical(value) + "\n").encode())


def _write_npz(path: Path, value: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise ValueError("Diagnostic RCPD evidence already exists")
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **value)
        stream.flush(); os.fsync(stream.fileno())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if (not path.is_file() or path.is_symlink()
            or path.resolve() != path.absolute()
            or path.stat().st_size > 256 * 1024 * 1024):
        raise ValueError("Diagnostic RCPD evidence archive is unsafe or oversized")
    with zipfile.ZipFile(path) as archive:
        expected = {name + ".npy" for name in _FIELDS}
        infos = archive.infolist()
        if ({info.filename for info in infos} != expected
                or len(infos) != len(expected)
                or any(info.is_dir() or info.file_size < 0 or info.compress_size < 0
                       for info in infos)
                or sum(info.file_size for info in infos) > 512 * 1024 * 1024):
            raise ValueError("Diagnostic RCPD evidence archive contents differ")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _FIELDS:
            raise ValueError("Diagnostic RCPD evidence array set differs")
        return {name: archive[name].copy() for name in archive.files}


def _obs_hash(obs: np.ndarray) -> str:
    return sha256(np.asarray(obs, dtype="<f4").tobytes()).hexdigest()


def _group_bits(groups: Sequence[str]) -> int:
    return sum(1 << GROUPS.index(group) for group in groups)


def _physical_hash(snapshot: Mapping[str, Any]) -> str:
    state = snapshot["state"]
    value = {
        "frame": state["frame"],
        "agents": state["agents"],
        "tasks": state["tasks"],
        "completed_tasks": state["completed_tasks"],
        "total_deliveries": state["total_deliveries"],
        "terminated": state["terminated"],
        "truncated": state["truncated"],
        "terminal_reason": state["terminal_reason"],
    }
    return digest(value)


def _runtime(actor_path: Path, protocol_path: Path, manifest_path: Path,
             manifest: Mapping[str, Any]) -> R41DiagnosticOnlineAlignmentRuntime:
    content = deepcopy(dict(manifest)); claimed = content.pop("content_sha256", None)
    return R41DiagnosticOnlineAlignmentRuntime(
        actor_path,
        training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor_path),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(_read(protocol_path, "protocol")),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )


def _scene_splits(manifest: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    splits = manifest.get("splits")
    if (manifest.get("version") != SCENE_MANIFEST_VERSION
            or not isinstance(splits, dict)
            or set(splits) != {"train", "conflict_validation", "final_test",
                               "question_bank", "tutorial"}):
        raise ValueError("Diagnostic manifest has no split registry")
    train = splits.get("train")
    validation = splits.get("conflict_validation")
    final = splits.get("final_test")
    if (not isinstance(train, list) or len(train) != 128
            or not isinstance(validation, list) or len(validation) != 64
            or not isinstance(final, list) or len(final) != 64):
        raise ValueError("Exact diagnostic 128/64/64 split registry required")
    fit = deepcopy(train + validation)
    fingerprints = [row.get("fingerprint") for row in fit]
    holdout = {row.get("fingerprint") for row in final}
    if (len(set(fingerprints)) != len(fingerprints)
            or set(fingerprints) & holdout or len(holdout) != 64):
        raise ValueError("Diagnostic RCPD source and final holdout overlap")
    return fit, deepcopy(final)


def _validate_manifest_actor(manifest: Mapping[str, Any],
                             actor: NumPyNativeActor) -> None:
    frozen = manifest.get("frozen_actor")
    workload = manifest.get("workload_screen")
    sources = manifest.get("producer_sources")
    if (frozen != {"sha256": actor.artifact_sha256,
                   "actor_parameters_sha256": actor.metadata.get(
                       "actor_parameters_sha256")}
            or not isinstance(workload, dict)
            or workload.get("version")
                != WORKLOAD_SCREEN_VERSION
            or not isinstance(workload.get("contract"), dict)
            or workload.get("contract_sha256") != digest(workload["contract"])
            or _HEX.fullmatch(str(workload.get("source_sha256", ""))) is None
            or not isinstance(sources, dict)
            or manifest.get("producer_sources_sha256") != digest(sources)
            or not isinstance(manifest.get("workload_generation_reports"), list)):
        raise ValueError("Diagnostic workload-screen/Actor manifest binding differs")


def _validate_designation(*, designation_path: Path,
                          expected_designation_sha256: str,
                          actor_path: Path, protocol_path: Path,
                          actor: NumPyNativeActor,
                          protocol: Mapping[str, Any]) -> dict[str, Any]:
    if (_HEX.fullmatch(str(expected_designation_sha256)) is None
            or file_hash(designation_path) != expected_designation_sha256):
        raise ValueError("Diagnostic designation hash differs")
    designation = _read(designation_path, "diagnostic designation")
    designated = designation.get("bindings", {})
    if (designation.get("version") != "warehouse-r41-diagnostic-actor-designation.v1"
            or designation.get("designated") is not True
            or designation.get("behavior_performance_gate_passed") is not False
            or designation.get("behavior_performance_gate_waived") is not True
            or designation.get("formal_ready") is not False
            or designation.get("formal_sample_eligible") is not False
            or designation.get("runtime_action_override") is not False
            or designated.get("actor_sha256") != file_hash(actor_path)
            or designated.get("actor_parameters_sha256")
                != actor.metadata.get("actor_parameters_sha256")
            or designated.get("protocol_file_sha256") != file_hash(protocol_path)
            or designated.get("protocol_content_sha256") != digest(protocol)):
        raise ValueError("External diagnostic Actor designation differs")
    return designation


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
                     fit_scenes: Sequence[Mapping[str, Any]]) -> None:
    if set(arrays) != _FIELDS:
        raise ValueError("Diagnostic RCPD evidence array set differs")
    count = len(arrays["observations"])
    vector_dtypes = {
        "action_indices": np.dtype(np.uint8), "weights": np.dtype(np.float32),
        "observation_hashes": np.dtype("U64"), "scene_fingerprints": np.dtype("U64"),
        "episode_ids": np.dtype("U180"), "frames": np.dtype(np.int16),
        "group_bits": np.dtype(np.uint8), "kinds": np.dtype("U16"),
        "anchor_ids": np.dtype("U240"), "branch_actions": np.dtype("U8"),
        "physical_hashes": np.dtype("U64"), "submitted_equal": np.dtype(np.bool_),
        "split_validation": np.dtype(np.bool_),
    }
    if (count <= 0 or arrays["observations"].shape != (count, actor.obs_dim)
            or arrays["observations"].dtype != np.dtype(np.float32)
            or arrays["probabilities"].shape != (count, len(ACTIONS))
            or arrays["probabilities"].dtype != np.dtype(np.float32)
            or any(arrays[name].shape != (count,) or arrays[name].dtype != dtype
                   for name, dtype in vector_dtypes.items())):
        raise ValueError("Diagnostic RCPD evidence array shapes or dtypes differ")
    observations = arrays["observations"]
    probabilities = arrays["probabilities"]
    labels = arrays["action_indices"]
    if (not np.isfinite(observations).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6)
            or np.any(labels >= len(ACTIONS))
            or not np.all(arrays["submitted_equal"])
            or np.any(arrays["frames"] < 0)
            or np.any(arrays["group_bits"] >= (1 << len(GROUPS)))):
        raise ValueError("Diagnostic RCPD numeric or action-authority evidence differs")
    expected_hashes = np.asarray([_obs_hash(row) for row in observations], dtype="U64")
    if not np.array_equal(expected_hashes, arrays["observation_hashes"]):
        raise ValueError("Diagnostic RCPD observation hashes differ")
    logits = actor.logits(observations)
    actor_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    actor_probabilities /= actor_probabilities.sum(axis=1, keepdims=True)
    # Runtime decisions are evaluated one observation at a time, whereas this
    # authentication pass uses one large matrix multiplication.  Accelerate's
    # two valid float32 reduction orders differ by at most a few ulps while
    # producing exactly the same deterministic action.  The stored logits are
    # not available in this compact archive, so keep a tight 4e-6 numeric bound
    # and require every argmax independently below.
    if (not np.allclose(probabilities, actor_probabilities, rtol=8e-6, atol=4e-6)
            or not np.array_equal(labels, np.argmax(actor_probabilities, axis=1).astype(np.uint8))):
        raise ValueError("Diagnostic RCPD rows differ from the frozen Actor")
    expected_scenes = {str(scene["fingerprint"]) for scene in fit_scenes}
    expected_episodes = {
        f"{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in fit_scenes for partner in PARTNERS
    }
    if (set(map(str, arrays["scene_fingerprints"])) != expected_scenes
            or set(map(str, arrays["episode_ids"])) != expected_episodes):
        raise ValueError("Diagnostic RCPD scenes or episodes differ")
    kinds = arrays["kinds"]
    ordinary = kinds == "ordinary"
    intervention = kinds == "intervention"
    if (not np.all(ordinary | intervention)
            or not np.all(arrays["branch_actions"][ordinary] == "")
            or not np.all(arrays["physical_hashes"][ordinary] == "")
            or np.any(arrays["anchor_ids"][intervention] == "")
            or not set(map(str, arrays["branch_actions"][intervention])).issubset(ACTIONS)
            or any(_HEX.fullmatch(str(value)) is None
                   for value in arrays["physical_hashes"][intervention])):
        raise ValueError("Diagnostic RCPD ordinary/intervention schema differs")
    ordinary_indices = np.flatnonzero(ordinary)
    expected_ordinary_anchors = np.asarray([
        (f"{arrays['episode_ids'][index]}:{int(arrays['frames'][index])}"
         if int(arrays["frames"][index]) % 5 == 0
            and int(arrays["group_bits"][index]) != 0 else "")
        for index in ordinary_indices
    ], dtype="U240")
    if not np.array_equal(arrays["anchor_ids"][ordinary_indices],
                          expected_ordinary_anchors):
        raise ValueError("Diagnostic RCPD source-anchor identity differs")
    minimal_rows = []
    for index, (observation, anchor) in enumerate(
            zip(observations, arrays["anchor_ids"])):
        minimal_rows.append({
            "observation": observation,
            "anchor": str(anchor),
            "action": ACTIONS[int(labels[index])],
            "groups": tuple(group for bit, group in enumerate(GROUPS)
                            if int(arrays["group_bits"][index]) & (1 << bit)),
            "kind": str(arrays["kinds"][index]),
            "branch_action": str(arrays["branch_actions"][index]),
            "physical_hash": str(arrays["physical_hashes"][index]),
        })
    expected_split = _partition(minimal_rows)
    if not np.array_equal(expected_split, arrays["split_validation"]):
        raise ValueError("Diagnostic RCPD observation-group partition differs")
    train_hashes = set(map(str, arrays["observation_hashes"][~expected_split]))
    validation_hashes = set(map(str, arrays["observation_hashes"][expected_split]))
    if train_hashes & validation_hashes:
        raise ValueError("Diagnostic RCPD saved observations overlap")

    # Class balancing is part of the fit.  Derive it exclusively from the
    # training partition so validation labels cannot influence any parameter,
    # feature ranking, or sample weight used by the surrogate.
    anchors: dict[str, dict[str, int]] = {}
    for index in np.flatnonzero(intervention):
        anchors.setdefault(str(arrays["anchor_ids"][index]), {})[
            str(arrays["branch_actions"][index])] = int(index)
    changed = np.zeros(count, dtype=np.bool_)
    for branches in anchors.values():
        wait = branches.get("WAIT")
        if wait is None:
            continue
        for index in branches.values():
            changed[index] = bool(
                arrays["physical_hashes"][index] != arrays["physical_hashes"][wait]
                and labels[index] != labels[wait])
    expected_weights = _sample_weights(
        labels, expected_split, arrays["group_bits"], intervention, changed)
    if (not np.isfinite(arrays["weights"]).all()
            or not np.array_equal(expected_weights, arrays["weights"])):
        raise ValueError("Diagnostic RCPD intervention weights differ")


def _collect(runtime: R41DiagnosticOnlineAlignmentRuntime,
             scenes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    environment_steps = 0
    for scene_index, scene in enumerate(scenes):
        for partner_index, partner in enumerate(PARTNERS):
            env = runtime.environment(scene)
            rng = np.random.default_rng(41_900_000 + scene_index * 101 + partner_index)
            episode = f"{scene['id']}:{scene['fingerprint']}:{partner}"
            while not env.done:
                source = env.snapshot(); source_sha = digest(source)
                observations = env.observations()
                actions, decision = runtime.decision(env)
                groups = tuple(critical_groups(env, "robot_2"))
                obs = observations["robot_2"].astype(np.float32, copy=True)
                probs = np.asarray(decision["probabilities"]["robot_2"], dtype=np.float32)
                anchor = (f"{episode}:{env.state.frame}"
                          if env.state.frame % 5 == 0 and groups else "")
                rows.append({
                    "observation": obs, "probabilities": probs,
                    "action": actions["robot_2"], "scene": scene["fingerprint"],
                    "episode": episode, "frame": int(env.state.frame),
                    "groups": groups, "kind": "ordinary", "anchor": anchor,
                    "branch_action": "", "physical_hash": "",
                    "submitted_equal": True, "source_state_sha256": source_sha,
                    "actor_changed_pair": False,
                })
                if anchor:
                    endpoints: dict[str, dict[str, Any] | None] = {}
                    for player_action in ACTIONS:
                        branch = runtime.from_snapshot(source)
                        transition = runtime.step(branch, player_action)
                        environment_steps += 1
                        if transition["submitted_actions"]["robot_2"] != \
                                transition["policy_actions"]["robot_2"]:
                            raise RuntimeError("Diagnostic branch overrode the Actor")
                        if branch.done:
                            endpoints[player_action] = None
                            continue
                        branch_obs = branch.observations()["robot_2"].astype(np.float32, copy=True)
                        branch_actions, branch_decision = runtime.decision(branch)
                        endpoints[player_action] = {
                            "observation": branch_obs,
                            "probabilities": np.asarray(
                                branch_decision["probabilities"]["robot_2"], dtype=np.float32),
                            "action": branch_actions["robot_2"],
                            "groups": tuple(critical_groups(branch, "robot_2")),
                            "physical_hash": _physical_hash(transition["after"]),
                            "source_state_sha256": digest(transition["after"]),
                        }
                    wait = endpoints["WAIT"]
                    for player_action, endpoint in endpoints.items():
                        if endpoint is None:
                            continue
                        changed = bool(wait is not None
                            and endpoint["physical_hash"] != wait["physical_hash"]
                            and endpoint["action"] != wait["action"])
                        rows.append({
                            **endpoint, "scene": scene["fingerprint"],
                            "episode": episode, "frame": int(env.state.frame) + 1,
                            "kind": "intervention", "anchor": anchor,
                            "branch_action": player_action, "submitted_equal": True,
                            "actor_changed_pair": changed,
                        })
                player = partner_action(env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_sha:
                    raise RuntimeError("Diagnostic collection changed source state")
                transition = runtime.step(env, player)
                environment_steps += 1
                if transition["submitted_actions"]["robot_2"] != actions["robot_2"]:
                    raise RuntimeError("Diagnostic trajectory overrode the Actor")
    return rows, environment_steps


def _partition(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Apply the original exact-hash/anchor connected-component split."""
    hashes = [_obs_hash(row["observation"]) for row in rows]
    parent = {value: value for value in hashes}
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    anchors: dict[str, list[str]] = {}
    for value, row in zip(hashes, rows):
        if row.get("anchor"):
            anchors.setdefault(str(row["anchor"]), []).append(value)
    for values in anchors.values():
        for value in values[1:]: union(values[0], value)
    validation_roots = {
        root for root in {find(value) for value in hashes}
        if int(sha256(("r41-diagnostic-split:" + root).encode()).hexdigest()[:8], 16) % 5 == 0
    }
    result = np.asarray([find(value) in validation_roots for value in hashes], dtype=np.bool_)
    if result.sum() < 256 or (~result).sum() < 512:
        raise ValueError("Observation-group partition produced an undersized split")
    train_hashes = {value for value, val in zip(hashes, result) if not val}
    val_hashes = {value for value, val in zip(hashes, result) if val}
    if train_hashes & val_hashes:
        raise AssertionError("Exact observation hash crossed the split")
    for values in anchors.values():
        anchor_assignments = {value in val_hashes for value in values}
        if len(anchor_assignments) != 1:
            raise AssertionError("Intervention anchor crossed the split")
    return result


def _arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    split = _partition(rows)
    labels = np.asarray([ACTIONS.index(row["action"]) for row in rows], dtype=np.uint8)
    group_bits = np.asarray([_group_bits(row["groups"]) for row in rows], dtype=np.uint8)
    kinds = np.asarray([row["kind"] for row in rows], dtype="U16")
    changed = np.asarray([row["actor_changed_pair"] for row in rows], dtype=np.bool_)
    weights = _sample_weights(labels, split, group_bits, kinds == "intervention", changed)
    return {
        "observations": np.stack([row["observation"] for row in rows]).astype(np.float32),
        "probabilities": np.stack([row["probabilities"] for row in rows]).astype(np.float32),
        "action_indices": labels,
        "weights": weights,
        "observation_hashes": np.asarray([_obs_hash(row["observation"]) for row in rows], dtype="U64"),
        "scene_fingerprints": np.asarray([row["scene"] for row in rows], dtype="U64"),
        "episode_ids": np.asarray([row["episode"] for row in rows], dtype="U180"),
        "frames": np.asarray([row["frame"] for row in rows], dtype=np.int16),
        "group_bits": group_bits,
        "kinds": kinds,
        "anchor_ids": np.asarray([row["anchor"] for row in rows], dtype="U240"),
        "branch_actions": np.asarray([row["branch_action"] for row in rows], dtype="U8"),
        "physical_hashes": np.asarray([row["physical_hash"] for row in rows], dtype="U64"),
        "submitted_equal": np.asarray([row["submitted_equal"] for row in rows], dtype=np.bool_),
        "split_validation": split,
    }


def _sample_weights(labels: np.ndarray, split_validation: np.ndarray,
                    group_bits: np.ndarray, intervention: np.ndarray,
                    actor_changed: np.ndarray) -> np.ndarray:
    """Compute fit weights without observing a validation label aggregate."""
    labels = np.asarray(labels, dtype=np.uint8)
    split_validation = np.asarray(split_validation, dtype=np.bool_)
    group_bits = np.asarray(group_bits, dtype=np.uint8)
    intervention = np.asarray(intervention, dtype=np.bool_)
    actor_changed = np.asarray(actor_changed, dtype=np.bool_)
    if any(value.shape != labels.shape for value in (
            split_validation, group_bits, intervention, actor_changed)):
        raise ValueError("Diagnostic sample-weight inputs differ")
    counts = np.bincount(labels[~split_validation], minlength=len(ACTIONS)).astype(float)
    balancing = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    balancing /= np.average(balancing, weights=np.maximum(counts, 1.0))
    weights = balancing[labels].astype(np.float32)
    weights[labels != ACTIONS.index("WAIT")] *= np.float32(1.5)
    weights[group_bits != 0] *= np.float32(2.0)
    weights[intervention] *= np.float32(4.0)
    weights[actor_changed] *= np.float32(4.0)
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Diagnostic sample weights are invalid")
    return weights


def _predict(program: R41DiagnosticModelTreeProgram,
             observations: np.ndarray) -> np.ndarray:
    return program.predict_batch(observations)


def _metrics(program: R41DiagnosticModelTreeProgram, arrays: Mapping[str, np.ndarray],
             mask: np.ndarray) -> dict[str, Any]:
    pred = _predict(program, arrays["observations"])
    labels = arrays["action_indices"]
    correct = pred == labels
    nonwait = labels != ACTIONS.index("WAIT")
    kinds = arrays["kinds"]
    bits = arrays["group_bits"]
    scenes = arrays["scene_fingerprints"]
    def stat(selected):
        idx = np.flatnonzero(mask & selected)
        return {"rows": int(len(idx)), "scenes": len(set(map(str, scenes[idx]))),
                "fidelity": float(correct[idx].mean()) if len(idx) else 0.0}
    critical = {group: stat((bits & (1 << i)) != 0) for i, group in enumerate(GROUPS)}
    anchors: dict[str, dict[str, int]] = {}
    for i in np.flatnonzero(mask & (kinds == "intervention")):
        anchors.setdefault(str(arrays["anchor_ids"][i]), {})[
            str(arrays["branch_actions"][i])] = int(i)
    effective, effective_correct, effective_scenes = 0, 0, set()
    for branches in anchors.values():
        wait_i = branches.get("WAIT")
        if wait_i is None: continue
        for action in ACTIONS[:-1]:
            changed_i = branches.get(action)
            if changed_i is None: continue
            if (arrays["physical_hashes"][wait_i] != arrays["physical_hashes"][changed_i]
                    and labels[wait_i] != labels[changed_i]):
                effective += 1
                effective_correct += int(correct[wait_i] and correct[changed_i])
                effective_scenes.add(str(scenes[wait_i]))
    target = arrays["probabilities"][mask]
    approx = program.predict_proba_batch(arrays["observations"][mask])
    mean_kl = float(np.mean(np.sum(target * (
        np.log(target.clip(1e-8)) - np.log(approx.clip(1e-8))), axis=-1)))
    return {
        "overall": stat(np.ones(len(mask), dtype=np.bool_)),
        "nonwait": stat(nonwait),
        "critical": critical,
        "effective_intervention_direction": {
            "pairs": effective, "scenes": len(effective_scenes),
            "fidelity": effective_correct / effective if effective else 0.0,
        },
        "mean_kl": mean_kl,
    }


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall": metrics["overall"]["fidelity"] >= MIN_OVERALL,
        "nonwait": metrics["nonwait"]["fidelity"] >= MIN_NONWAIT,
        "effective_intervention_direction": (
            metrics["effective_intervention_direction"]["scenes"] >= 10
            and metrics["effective_intervention_direction"]["fidelity"] >= MIN_DIRECTION),
    }
    for group in GROUPS:
        item = metrics["critical"][group]
        checks[group] = item["scenes"] >= 10 and item["fidelity"] >= MIN_CRITICAL
    return {"checks": checks, "passed": all(checks.values())}


def _router_payload(router: DecisionTreeClassifier,
                    model_indices: Mapping[int, int]) -> dict[str, Any]:
    tree = router.tree_
    nodes = []
    for index in range(tree.node_count):
        left = int(tree.children_left[index]); right = int(tree.children_right[index])
        if left == right:
            nodes.append({"kind": "leaf", "model_index": model_indices[index]})
        else:
            nodes.append({"kind": "split", "feature_index": int(tree.feature[index]),
                          "threshold": float(tree.threshold[index]),
                          "left": left, "right": right})
    return {"depth": int(router.get_depth()), "leaf_count": int(router.get_n_leaves()),
            "nodes": nodes}


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                  *, c: float) -> tuple[LogisticRegression, int]:
    model = LogisticRegression(C=c, max_iter=LOGISTIC_MAX_ITER, solver="lbfgs",
                               tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y, sample_weight=weights)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Diagnostic leaf classifier did not converge")
    if (not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all()
            or np.any(model.n_iter_ >= LOGISTIC_MAX_ITER)):
        raise RuntimeError("Diagnostic leaf classifier fit is non-finite or unfinished")
    return model, int(np.max(model.n_iter_))


def _fit_foundation(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train = ~arrays["split_validation"]
    observations = arrays["observations"].astype(np.float64)
    labels = arrays["action_indices"]
    weights = arrays["weights"].astype(np.float64)
    variable = observations[train].std(axis=0) > 1e-8
    variable_indices = np.flatnonzero(variable)
    if len(variable_indices) == 0:
        raise RuntimeError("Diagnostic model-tree has no variable public inputs")
    scaler = StandardScaler().fit(observations[train][:, variable])
    scaled = scaler.transform(observations[:, variable])
    if (not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
            or not np.isfinite(scaled).all() or np.any(scaler.scale_ <= 0.0)):
        raise RuntimeError("Diagnostic train-only public-input scaler is invalid")
    router = DecisionTreeClassifier(
        max_depth=ROUTER_DEPTH, max_leaf_nodes=ROUTER_LEAVES,
        min_samples_leaf=ROUTER_MIN_SAMPLES_LEAF, random_state=41,
    )
    router.fit(observations[train], labels[train], sample_weight=weights[train])
    leaf_ids = router.apply(observations)
    leaf_nodes = sorted(map(int, np.unique(router.apply(observations[train]))))
    if (router.get_depth() > ROUTER_DEPTH or router.get_n_leaves() > ROUTER_LEAVES
            or len(leaf_nodes) != router.get_n_leaves()):
        raise RuntimeError("Diagnostic routing-tree bounds or training leaves differ")
    model_indices = {node: index for index, node in enumerate(leaf_nodes)}
    rankings = {}
    for node in leaf_nodes:
        train_mask = train & (leaf_ids == node)
        classes = np.unique(labels[train_mask])
        if len(classes) == 1:
            rankings[node] = {"classes": classes, "ranking": None,
                              "ranking_iterations": 0}
            continue
        dense, ranking_iterations = _fit_logistic(
            scaled[train_mask], labels[train_mask], weights[train_mask], c=RANKING_C)
        norms = np.linalg.norm(dense.coef_, axis=0)
        rankings[node] = {"classes": classes,
            "ranking": np.lexsort((variable_indices, -norms)),
            "ranking_iterations": ranking_iterations}
    return {"train": train, "observations": observations, "labels": labels,
        "weights": weights, "variable_indices": variable_indices,
        "scaler": scaler, "scaled": scaled, "router": router,
        "leaf_ids": leaf_ids, "leaf_nodes": leaf_nodes,
        "model_indices": model_indices, "rankings": rankings}


def _fit_one(actor: NumPyNativeActor, arrays: Mapping[str, np.ndarray],
             profile: Mapping[str, Any], binding: str,
             source_full_manifest_bindings: Mapping[str, str], *,
             foundation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    foundation = _fit_foundation(arrays) if foundation is None else foundation
    train = foundation["train"]
    observations = foundation["observations"]
    labels = foundation["labels"]
    weights = foundation["weights"]
    variable_indices = foundation["variable_indices"]
    scaler = foundation["scaler"]
    scaled = foundation["scaled"]
    router = foundation["router"]
    leaf_ids = foundation["leaf_ids"]
    leaf_nodes = foundation["leaf_nodes"]
    model_indices = foundation["model_indices"]
    models = []
    fitted_predictions = np.empty(len(labels), dtype=np.uint8)
    fitted_probabilities = np.zeros((len(labels), len(ACTIONS)), dtype=np.float64)
    diagnostics = []
    feature_cap = profile["feature_cap"]
    logistic_c = float(profile["logistic_c"])
    for node in leaf_nodes:
        train_mask = train & (leaf_ids == node)
        predict_mask = leaf_ids == node
        classes = foundation["rankings"][node]["classes"]
        if len(classes) == 1:
            class_index = int(classes[0])
            fitted_predictions[predict_mask] = class_index
            fitted_probabilities[predict_mask, class_index] = 1.0
            models.append({"router_node": node, "classes": [class_index],
                           "feature_indices": [], "coefficients": [[]],
                           "intercepts": [0.0]})
            diagnostics.append({"router_node": node,
                "train_rows": int(np.sum(train_mask)),
                "classes": [ACTIONS[class_index]], "constant": True,
                "feature_count": 0, "nonzero_coefficients": 0,
                "ranking_iterations": 0, "refit_iterations": 0,
                "converged": True})
            continue
        ranking_iterations = foundation["rankings"][node]["ranking_iterations"]
        ranking = foundation["rankings"][node]["ranking"]
        selected_scaled = ranking if feature_cap is None else ranking[:int(feature_cap)]
        refit, refit_iterations = _fit_logistic(
            scaled[train_mask][:, selected_scaled], labels[train_mask],
            weights[train_mask], c=logistic_c)
        raw_indices = variable_indices[selected_scaled]
        order = np.argsort(raw_indices, kind="stable")
        raw_indices = raw_indices[order]
        scaled_coefficients = refit.coef_[:, order]
        selected_means = scaler.mean_[selected_scaled][order]
        selected_scales = scaler.scale_[selected_scaled][order]
        raw_coefficients = scaled_coefficients / selected_scales[None, :]
        raw_intercepts = refit.intercept_ - np.sum(
            scaled_coefficients * selected_means[None, :] / selected_scales[None, :],
            axis=1,
        )
        if len(refit.classes_) == 2:
            coefficients = [np.zeros(len(raw_indices), dtype=np.float64).tolist(),
                            raw_coefficients[0].astype(float).tolist()]
            intercepts = [0.0, float(raw_intercepts[0])]
        else:
            coefficients = raw_coefficients.astype(float).tolist()
            intercepts = raw_intercepts.astype(float).tolist()
        models.append({"router_node": node,
            "classes": [int(value) for value in refit.classes_],
            "feature_indices": [int(value) for value in raw_indices],
            "coefficients": coefficients, "intercepts": intercepts})
        with warnings.catch_warnings():
            # Some Accelerate builds emit spurious small-matrix matmul runtime
            # warnings after a converged finite fit.  Authenticate the actual
            # returned probabilities below instead of letting backend noise
            # contaminate the evidence log.
            warnings.simplefilter("ignore", RuntimeWarning)
            fitted_predictions[predict_mask] = refit.predict(
                scaled[predict_mask][:, selected_scaled]).astype(np.uint8)
            local_probabilities = refit.predict_proba(
                scaled[predict_mask][:, selected_scaled])
        if (not np.isfinite(local_probabilities).all()
                or not np.allclose(local_probabilities.sum(axis=1), 1.0,
                                   rtol=0.0, atol=2e-12)):
            raise RuntimeError("Diagnostic fitted leaf probabilities are invalid")
        row_indices = np.flatnonzero(predict_mask)
        fitted_probabilities[np.ix_(row_indices, refit.classes_.astype(int))] = \
            local_probabilities
        diagnostics.append({"router_node": node,
            "train_rows": int(np.sum(train_mask)),
            "classes": [ACTIONS[int(value)] for value in refit.classes_],
            "constant": False, "feature_count": len(raw_indices),
            "nonzero_coefficients": int(np.count_nonzero(raw_coefficients)),
            "ranking_iterations": ranking_iterations,
            "refit_iterations": refit_iterations, "converged": True})

    metadata = {
        "diagnostic_rcpd_version": VERSION,
        "diagnostic_rcpd_binding_sha256": binding,
        "source_full_manifest_bindings": dict(source_full_manifest_bindings),
        "native_source_actor_sha256": actor.artifact_sha256,
        "source_actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "program_family": "axis_router_sparse_public_linear_softmax_leaves",
        "candidate_profile": dict(profile),
        "router_depth_cap": ROUTER_DEPTH,
        "router_leaf_cap": ROUTER_LEAVES,
        "router_min_samples_leaf": ROUTER_MIN_SAMPLES_LEAF,
        "action_legality_features": {},
        "action_constraint_reason_features": {},
        "runtime_controller": "native_neural_actor_only",
        "runtime_action_override": False,
        "program_feedback_into_actor": False,
        "ppo_joint_steps": 0,
        "optimizer_updates": 0,
        "formal_ready": False,
    }
    payload = {"version": MODEL_TREE_VERSION, "action_names": list(ACTIONS),
        "feature_names": list(actor.metadata["feature_names"]),
        "router": _router_payload(router, model_indices),
        "leaf_models": models, "metadata": metadata}
    program = R41DiagnosticModelTreeProgram.from_dict(payload)
    restored = R41DiagnosticModelTreeProgram.from_dict(program.to_dict())
    restored_predictions = restored.predict_batch(observations)
    if not np.array_equal(restored_predictions, fitted_predictions):
        raise RuntimeError("Serialized diagnostic model-tree changed an action")
    restored_probabilities = restored.predict_proba_batch(observations)
    probability_error = float(np.max(np.abs(
        restored_probabilities - fitted_probabilities)))
    if probability_error > 1e-10:
        raise RuntimeError("Serialized diagnostic model-tree changed probabilities")
    metrics = _metrics(program, arrays, arrays["split_validation"])
    gate = _gate(metrics)
    complexity = program.complexity()
    fit_diagnostics = {
        "train_rows": int(np.sum(train)),
        "validation_rows_excluded_from_fit": int(np.sum(arrays["split_validation"])),
        "public_features_available": observations.shape[1],
        "public_features_nonconstant_train": len(variable_indices),
        "router_depth": int(router.get_depth()),
        "router_leaves": int(router.get_n_leaves()),
        "leaf_class_scores": int(sum(len(model["classes"]) for model in models)),
        "leaf_models": diagnostics,
        "all_row_actions_identical_after_serialization": True,
        "maximum_probability_absolute_error_after_serialization": probability_error,
        "sklearn_version": sklearn.__version__,
        "fit_inputs": ["public_observation", "frozen_actor_action", "registered_sample_weight"],
        "prediction_inputs": ["public_observation"],
        "actor_parameter_arrays_accessed": False,
        "actor_logits_used": False,
        "actor_hidden_states_used": False,
        "intervention_branch_metadata_used_at_prediction": False,
        "physical_hash_used_at_prediction": False,
    }
    return {"profile": dict(profile), "metrics": metrics,
            "gate": gate, "complexity": complexity,
            "fit_diagnostics": fit_diagnostics,
            "program_content_sha256": digest(program.to_dict()),
            "program": program.to_dict()}


def _choose(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [row for row in candidates if row["gate"]["passed"]]
    return min(eligible, key=lambda row: (
        row["complexity"]["nonzero_coefficients"],
        row["complexity"]["maximum_features_per_leaf"],
        row["complexity"]["router_leaves"], row["complexity"]["router_depth"],
        row["metrics"]["mean_kl"], row["profile"]["name"])) if eligible else None


def extract(*, actor_path: str | Path, protocol_path: str | Path,
            manifest_path: str | Path, designation_path: str | Path,
            expected_designation_sha256: str, output: str | Path) -> dict[str, Any]:
    actor_path = _regular(actor_path, "Actor")
    protocol_path = _regular(protocol_path, "training protocol")
    manifest_path = _regular(manifest_path, "diagnostic manifest")
    designation_path = _regular(designation_path, "diagnostic designation")
    manifest = _read(manifest_path, "diagnostic manifest")
    protocol = _read(protocol_path, "training protocol")
    actor = NumPyNativeActor(actor_path)
    _validate_manifest_actor(manifest, actor)
    _validate_designation(designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        actor=actor, protocol=protocol)
    fit_scenes, holdout = _scene_splits(manifest)
    runtime = _runtime(actor_path, protocol_path, manifest_path, manifest)
    sources = producer_sources()
    content = deepcopy(manifest); content_sha = content.pop("content_sha256", None)
    bindings = {
        "designation_sha256": expected_designation_sha256,
        "actor_file_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "protocol_file_sha256": file_hash(protocol_path),
        "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": file_hash(manifest_path),
        "manifest_content_sha256": content_sha,
        "manifest_semantic_sha256": digest(manifest),
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
        "holdout_fingerprints_sha256": digest(sorted(row["fingerprint"] for row in holdout)),
    }
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    _write_json(output / "inputs.json", {"version": VERSION, "bindings": bindings,
        "contract": contract(), "sources": sources, "formal_ready": False})
    rows, environment_steps = _collect(runtime, fit_scenes)
    arrays = _arrays(rows)
    if (not np.all(arrays["submitted_equal"])
            or set(arrays) != _FIELDS):
        raise RuntimeError("Diagnostic RCPD action authority or schema differs")
    _write_npz(output / "rows.npz", arrays)
    binding = digest({"bindings": bindings, "rows_sha256": file_hash(output / "rows.npz")})
    foundation = _fit_foundation(arrays)
    candidates = [
        _fit_one(actor, arrays, profile, binding,
                 runtime.source_full_manifest_bindings, foundation=foundation)
        for profile in MODEL_TREE_PROFILES
    ]
    selected = _choose(candidates)
    _write_json(output / "candidates.json", {"version": VERSION,
        "binding_sha256": binding, "candidate_profiles": [
            dict(row) for row in MODEL_TREE_PROFILES],
        "candidates": candidates})
    if selected is not None:
        _write_json(output / "program.json", selected["program"])
        raw = (output / "program.json").read_bytes()
        compressed_base64_bytes = len(base64.b64encode(gzip.compress(raw, mtime=0)))
        if len(raw) > 512 * 1024 or compressed_base64_bytes > 384 * 1024:
            raise RuntimeError("Selected diagnostic program exceeds its package budget")
    else:
        compressed_base64_bytes = None
    train_hash = set(map(str, arrays["observation_hashes"][~arrays["split_validation"]]))
    val_hash = set(map(str, arrays["observation_hashes"][arrays["split_validation"]]))
    artifacts = {name: file_hash(output / name) for name in
        ("inputs.json", "rows.npz", "candidates.json")}
    report = {
        "version": VERSION,
        "status": "passed" if selected is not None else "failed",
        "explanation_eligible": selected is not None,
        "formal_ready": False,
        "bindings": bindings,
        "diagnostic_rcpd_binding_sha256": binding,
        "rows": len(rows),
        "ordinary_rows": int(np.sum(arrays["kinds"] == "ordinary")),
        "intervention_rows": int(np.sum(arrays["kinds"] == "intervention")),
        "train_rows": int(np.sum(~arrays["split_validation"])),
        "validation_rows": int(np.sum(arrays["split_validation"])),
        "exact_observation_overlap": len(train_hash & val_hash),
        "candidate_count": len(candidates),
        "candidate_profiles": [dict(row) for row in MODEL_TREE_PROFILES],
        "selected": None if selected is None else {key: deepcopy(selected[key]) for key in
            ("profile", "metrics", "gate", "complexity", "fit_diagnostics",
             "program_content_sha256")},
        "program_file_sha256": None if selected is None else file_hash(output / "program.json"),
        "program_package_size": None if selected is None else {
            "json_bytes": (output / "program.json").stat().st_size,
            "gzip_base64_bytes": compressed_base64_bytes,
            "json_limit_bytes": 512 * 1024,
            "gzip_base64_limit_bytes": 384 * 1024,
            "passed": True,
        },
        "evidence_artifacts": artifacts,
        "execution": {"environment_steps": environment_steps,
            "actor_queries": len(rows), "router_fits": len(candidates),
            "leaf_profile_fits": len(candidates),
            "ppo_joint_steps": 0, "optimizer_updates": 0,
            "actor_changed": False, "runtime_action_overrides": 0},
    }
    if (producer_sources() != sources
            or file_hash(actor_path) != bindings["actor_file_sha256"]
            or file_hash(protocol_path) != bindings["protocol_file_sha256"]
            or file_hash(manifest_path) != bindings["manifest_file_sha256"]
            or file_hash(designation_path) != bindings["designation_sha256"]):
        raise RuntimeError("Frozen diagnostic RCPD inputs changed during extraction")
    _write_json(output / "report.json", report)
    read_saved_report(output,
        expected_report_sha256=file_hash(output / "report.json"),
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        program_path=(output / "program.json") if selected is not None else None,
        require_passed=False)
    return report


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      actor_path: str | Path, protocol_path: str | Path,
                      manifest_path: str | Path, designation_path: str | Path,
                      expected_designation_sha256: str,
                      program_path: str | Path | None,
                      require_passed: bool = True) -> dict[str, Any]:
    """Recompute diagnostic RCPD bindings and metrics from saved evidence."""
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Diagnostic RCPD evidence directory is missing or unsafe")
    report_path = output / "report.json"
    if (_HEX.fullmatch(str(expected_report_sha256)) is None
            or not report_path.is_file() or report_path.is_symlink()
            or report_path.resolve() != report_path
            or file_hash(report_path) != expected_report_sha256):
        raise ValueError("Diagnostic RCPD report hash differs")
    actor_path = _regular(actor_path, "Actor")
    protocol_path = _regular(protocol_path, "training protocol")
    manifest_path = _regular(manifest_path, "diagnostic manifest")
    designation_path = _regular(designation_path, "diagnostic designation")
    actor = NumPyNativeActor(actor_path)
    protocol = _read(protocol_path, "training protocol")
    manifest = _read(manifest_path, "diagnostic manifest")
    _validate_manifest_actor(manifest, actor)
    _validate_designation(designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        actor=actor, protocol=protocol)
    fit_scenes, holdout = _scene_splits(manifest)
    runtime = _runtime(actor_path, protocol_path, manifest_path, manifest)
    sources = producer_sources()
    content = deepcopy(manifest); content_sha = content.pop("content_sha256", None)
    if content_sha != digest(content):
        raise ValueError("Diagnostic manifest content digest differs")
    bindings = {
        "designation_sha256": expected_designation_sha256,
        "actor_file_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor.metadata.get("actor_parameters_sha256"),
        "protocol_file_sha256": file_hash(protocol_path),
        "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": file_hash(manifest_path),
        "manifest_content_sha256": content_sha,
        "manifest_semantic_sha256": digest(manifest),
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
        "holdout_fingerprints_sha256": digest(
            sorted(row["fingerprint"] for row in holdout)),
    }
    inputs_path = output / "inputs.json"
    inputs = _read(_regular(inputs_path, "Diagnostic RCPD inputs"),
                   "Diagnostic RCPD inputs")
    if inputs != {"version": VERSION, "bindings": bindings,
                  "contract": contract(), "sources": sources,
                  "formal_ready": False}:
        raise ValueError("Diagnostic RCPD input provenance differs")
    report = _read(report_path, "Diagnostic RCPD report")
    artifacts = report.get("evidence_artifacts")
    expected_artifacts = {"inputs.json", "rows.npz", "candidates.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Diagnostic RCPD evidence artifact set differs")
    for name, expected in artifacts.items():
        path = output / name
        if (path.parent != output or not path.is_file() or path.is_symlink()
                or _HEX.fullmatch(str(expected)) is None or file_hash(path) != expected):
            raise ValueError("Diagnostic RCPD evidence artifact changed")
    arrays = _load_npz(output / "rows.npz")
    _validate_arrays(arrays, actor=actor, fit_scenes=fit_scenes)
    binding = digest({"bindings": bindings,
                      "rows_sha256": artifacts["rows.npz"]})
    stored = _read(output / "candidates.json", "Diagnostic RCPD candidates")
    if (stored.get("version") != VERSION
            or stored.get("binding_sha256") != binding
            or stored.get("candidate_profiles") != [
                dict(row) for row in MODEL_TREE_PROFILES]
            or not isinstance(stored.get("candidates"), list)):
        raise ValueError("Diagnostic RCPD candidate registry differs")
    candidates = stored["candidates"]
    if (len(candidates) != len(MODEL_TREE_PROFILES)
            or [item.get("profile") for item in candidates if isinstance(item, dict)]
                != [dict(row) for row in MODEL_TREE_PROFILES]):
        raise ValueError("Diagnostic RCPD candidate profile registry differs")
    checked: list[dict[str, Any]] = []
    foundation = _fit_foundation(arrays)
    for profile, item in zip(MODEL_TREE_PROFILES, candidates):
        if not isinstance(item, dict) or set(item) != _CANDIDATE_FIELDS:
            raise ValueError("Diagnostic RCPD candidate schema differs")
        # Refit from authenticated train rows.  Matching the complete program
        # proves the router, scaler, feature ranking, and every leaf model
        # excluded validation labels rather than trusting a claimed ledger.
        expected = _fit_one(actor, arrays, profile, binding,
                            runtime.source_full_manifest_bindings,
                            foundation=foundation)
        if digest(expected) != digest(item):
            raise ValueError("Diagnostic RCPD candidate differs from train-only refit")
        program = R41DiagnosticModelTreeProgram.from_dict(item["program"])
        if (tuple(program.action_names) != tuple(ACTIONS)
                or tuple(program.feature_names) != tuple(actor.metadata["feature_names"])
                or program.metadata.get("diagnostic_rcpd_version") != VERSION
                or program.metadata.get("diagnostic_rcpd_binding_sha256") != binding
                or program.metadata.get("source_full_manifest_bindings")
                    != runtime.source_full_manifest_bindings
                or program.metadata.get("native_source_actor_sha256") != actor.artifact_sha256
                or program.metadata.get("source_actor_parameters_sha256")
                    != actor.metadata.get("actor_parameters_sha256")
                or program.metadata.get("runtime_controller")
                    != "native_neural_actor_only"
                or program.metadata.get("runtime_action_override") is not False
                or program.metadata.get("program_feedback_into_actor") is not False
                or program.metadata.get("ppo_joint_steps") != 0
                or program.metadata.get("optimizer_updates") != 0
                or program.metadata.get("candidate_profile") != item["profile"]
                or program.metadata.get("router_depth_cap") != ROUTER_DEPTH
                or program.metadata.get("router_leaf_cap") != ROUTER_LEAVES
                or program.metadata.get("router_min_samples_leaf")
                    != ROUTER_MIN_SAMPLES_LEAF
                or program.router_depth > ROUTER_DEPTH
                or program.router_leaf_count > ROUTER_LEAVES):
            raise ValueError("Diagnostic RCPD candidate program binding differs")
        checked.append(expected)
    selected = _choose(checked)
    selected_summary = None if selected is None else {
        key: deepcopy(selected[key]) for key in
        ("profile", "metrics", "gate", "complexity", "fit_diagnostics",
         "program_content_sha256")}
    status = "passed" if selected is not None else "failed"
    program_sha256 = None
    if selected is not None:
        if program_path is None:
            raise ValueError("Passing diagnostic RCPD requires its selected program")
        supplied = Path(program_path).expanduser().absolute()
        if (supplied != output / "program.json" or supplied.is_symlink()
                or not supplied.is_file()
                or _read(supplied, "Diagnostic selected program") != selected["program"]):
            raise ValueError("Diagnostic RCPD selected program differs")
        program_sha256 = file_hash(supplied)
        raw = supplied.read_bytes()
        package_size = {"json_bytes": len(raw),
            "gzip_base64_bytes": len(base64.b64encode(gzip.compress(raw, mtime=0))),
            "json_limit_bytes": 512 * 1024,
            "gzip_base64_limit_bytes": 384 * 1024,
            "passed": len(raw) <= 512 * 1024 and len(base64.b64encode(
                gzip.compress(raw, mtime=0))) <= 384 * 1024}
        if not package_size["passed"]:
            raise ValueError("Passing diagnostic program exceeds its package budget")
    elif program_path is not None or (output / "program.json").exists():
        raise ValueError("Failed diagnostic RCPD cannot expose a selected program")
    else:
        package_size = None
    expected_report = {
        "version": VERSION, "status": status,
        "explanation_eligible": selected is not None, "formal_ready": False,
        "bindings": bindings, "diagnostic_rcpd_binding_sha256": binding,
        "rows": len(arrays["observations"]),
        "ordinary_rows": int(np.sum(arrays["kinds"] == "ordinary")),
        "intervention_rows": int(np.sum(arrays["kinds"] == "intervention")),
        "train_rows": int(np.sum(~arrays["split_validation"])),
        "validation_rows": int(np.sum(arrays["split_validation"])),
        "exact_observation_overlap": 0,
        "candidate_count": len(checked),
        "candidate_profiles": [dict(row) for row in MODEL_TREE_PROFILES],
        "selected": selected_summary,
        "program_file_sha256": program_sha256,
        "program_package_size": package_size,
        "evidence_artifacts": artifacts,
        "execution": {"environment_steps": report.get("execution", {}).get(
            "environment_steps"), "actor_queries": len(arrays["observations"]),
            "router_fits": len(checked), "leaf_profile_fits": len(checked),
            "ppo_joint_steps": 0,
            "optimizer_updates": 0, "actor_changed": False,
            "runtime_action_overrides": 0},
    }
    environment_steps = expected_report["execution"]["environment_steps"]
    if type(environment_steps) is not int or environment_steps < 0:
        raise ValueError("Diagnostic RCPD execution accounting differs")
    if digest(report) != digest(expected_report):
        raise ValueError("Diagnostic RCPD report differs from immutable evidence")
    if require_passed and status != "passed":
        raise ValueError("Diagnostic RCPD did not meet the registered explanation gates")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--expected-designation-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = extract(actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, designation_path=args.designation,
        expected_designation_sha256=args.expected_designation_sha256,
        output=args.output)
    print(canonical({"status": report["status"], "selected": report["selected"],
                     "output": str(Path(args.output).absolute())}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "SCENE_MANIFEST_VERSION", "WORKLOAD_SCREEN_VERSION",
           "PARTNERS", "GROUPS", "MODEL_TREE_PROFILES",
           "contract", "producer_sources", "extract", "read_saved_report", "main"]
