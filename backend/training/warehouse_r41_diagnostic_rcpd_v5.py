"""Strict scene-separated RCPD for the frozen r4.1 diagnostic Actor.

The development fit uses the 128 registered training scenes and dense
counterfactual endpoints at every public critical frame.  The 64 registered
conflict-validation scenes are excluded from all fitting and use the frozen
every-fifth-frame intervention schedule.  A four-member ensemble of explicit
axis-routing model trees is serialized with deterministic int8 leaf
quantization.  Every runtime input remains one of the 197 public features.

This is development evidence.  It never reads the registered final-test
trajectories; the candidate must be frozen before the separate final audit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import warnings
import zipfile

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training import warehouse_r41_diagnostic_rcpd as legacy
from backend.warehouse_r41_diagnostic_model_tree import (
    R41DiagnosticModelTreeProgram,
    VERSION as MEMBER_VERSION,
)
from backend.warehouse_r41_diagnostic_model_tree_ensemble import (
    ENVELOPE_VERSION,
    R41DiagnosticModelTreeEnsemble,
    VERSION as PROGRAM_VERSION,
    load_program_envelope,
    make_program_envelope,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-rcpd.v5"
SCENE_MANIFEST_VERSION = "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
WORKLOAD_SCREEN_VERSION = "warehouse-r41-diagnostic-workload-screen.v2"
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
MEMBER_SEED_CANDIDATES = (11, 23, 37, 53, 71, 89, 107, 131, 157, 181, 211)
MEMBER_COUNT = 4
ROUTER_DEPTH = 12
ROUTER_LEAVES = 256
ROUTER_MIN_SAMPLES_LEAF = 24
ROUTER_MAX_FEATURES = .9
FEATURE_CAP = 80
RANKING_C = 1.0
REFIT_C = 10.0
LOGISTIC_MAX_ITER = 5000
PAIR_ENDPOINT_MULTIPLIER = 2.0
MIN_OVERALL = .90
MIN_NONWAIT = .90
MIN_CRITICAL = .85
MIN_DIRECTION = .85
MAX_JSON_BYTES = 256 * 1024 * 1024
MAX_PROGRAM_BYTES = 512 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset((
    "observations", "probabilities", "action_indices", "weights",
    "observation_hashes", "scene_fingerprints", "episode_ids", "frames",
    "group_bits", "kinds", "anchor_ids", "branch_actions",
    "physical_hashes", "source_state_hashes", "submitted_equal",
    "trajectory_done", "split_validation",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "scene_manifest_version": SCENE_MANIFEST_VERSION,
        "source_splits": {"fit": "train", "development_validation": "conflict_validation"},
        "holdout_split": "final_test",
        "partners": list(PARTNERS),
        "evaluated_role": "robot_2",
        "partition": (
            "registered scene split; validation wins on any exact float32 "
            "public-observation overlap and the duplicate training row is excluded"
        ),
        "intervention_schedule": {
            "train": "every critical pre-action frame",
            "conflict_validation": "critical pre-action frame divisible by five",
            "actions": list(ACTIONS),
        },
        "program_family": (
            "mean-probability ensemble of four explicit axis-routing trees "
            "with sparse public-input linear-softmax leaves"
        ),
        "member_seed_candidates": list(MEMBER_SEED_CANDIDATES),
        "member_count": MEMBER_COUNT,
        "candidate_selection_disclosure": (
            "all four-member subsets of eleven pre-registered router seeds are "
            "compared on conflict_validation only; selection maximizes the minimum "
            "per-group intervention-direction fidelity, then overall direction "
            "fidelity, then the minimum ordinary gate fidelity; final-test states "
            "and labels are not accessed"
        ),
        "serialization_selection_disclosure": (
            "symmetric int8 qmax 127 was frozen after a development-only "
            "compression comparison; smaller tested ranges missed at least one "
            "fidelity gate and final-test states and labels were not accessed"
        ),
        "fit_access": {
            "fit_rows": "registered train scenes only",
            "prediction_input": "public observation vector only",
            "frozen_actor_action_label": True,
            "actor_parameter_arrays": False,
            "actor_logits_as_program_input": False,
            "actor_hidden_states": False,
            "intervention_branch_metadata_as_program_input": False,
            "physical_hash_as_program_input": False,
            "validation_labels": "candidate selection and gates only; never member fitting",
            "final_test": False,
        },
        "weights": {
            "base": legacy.contract()["weights"],
            "effective_train_pair_endpoint_multiplier": PAIR_ENDPOINT_MULTIPLIER,
            "validation_weights_used_for_fit": False,
        },
        "member": {
            "router": {"kind": "axis CART classifier", "depth_cap": ROUTER_DEPTH,
                       "leaf_cap": ROUTER_LEAVES,
                       "minimum_samples_leaf": ROUTER_MIN_SAMPLES_LEAF,
                       "maximum_features": ROUTER_MAX_FEATURES,
                       "bootstrap": "member-seed train-only rows with replacement"},
            "leaf": {"kind": "multinomial public-input linear-softmax",
                     "ranking_fit_c": RANKING_C, "feature_cap": FEATURE_CAP,
                     "refit_c": REFIT_C, "solver": "lbfgs",
                     "maximum_iterations": LOGISTIC_MAX_ITER,
                     "standardization": "train-only scaler folded into raw coefficients"},
        },
        "serialization": {
            "program_version": PROGRAM_VERSION,
            "envelope_version": ENVELOPE_VERSION,
            "leaf_coefficients": "symmetric per-class int8, qmax 127",
            "router_thresholds": "float32",
            "leaf_intercepts": "float32",
        },
        "thresholds": {"overall": MIN_OVERALL, "nonwait": MIN_NONWAIT,
                       "critical": MIN_CRITICAL,
                       "effective_intervention_direction": MIN_DIRECTION,
                       "effective_intervention_direction_by_group": MIN_DIRECTION},
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
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError(label + " is missing, linked, noncanonical, or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Diagnostic v5 output path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write(path, (canonical(value) + "\n").encode("utf-8"))


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise ValueError("Diagnostic v5 row archive already exists")
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > 256 * 1024 * 1024):
        raise ValueError("Diagnostic v5 row archive is unsafe or oversized")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        expected = {name + ".npy" for name in _FIELDS}
        if ({item.filename for item in infos} != expected or len(infos) != len(expected)
                or any(item.is_dir() or item.file_size < 0 or item.compress_size < 0
                       for item in infos)
                or sum(item.file_size for item in infos) > 512 * 1024 * 1024):
            raise ValueError("Diagnostic v5 row archive contents differ")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _FIELDS:
            raise ValueError("Diagnostic v5 row archive schema differs")
        return {name: archive[name].copy() for name in archive.files}


def _decode(array: np.ndarray) -> np.ndarray:
    return np.char.decode(array, "ascii")


def _scene_splits(manifest: Mapping[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    legacy._scene_splits(manifest)
    splits = manifest["splits"]
    train = deepcopy(splits["train"])
    validation = deepcopy(splits["conflict_validation"])
    final = deepcopy(splits["final_test"])
    if len(train) != 128 or len(validation) != 64 or len(final) != 64:
        raise ValueError("Exact diagnostic v5 scene registry required")
    sets = [{row["fingerprint"] for row in part}
            for part in (train, validation, final)]
    if any(len(value) != expected for value, expected in zip(sets, (128, 64, 64))) \
            or any(sets[left] & sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Diagnostic v5 scene split overlaps")
    return train, validation, final


def _collect(runtime, scenes: Sequence[Mapping[str, Any]], *, scene_offset: int,
             dense_critical: bool, progress_label: str | None = None
             ) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    environment_steps = 0
    for local_index, scene in enumerate(scenes):
        scene_index = scene_offset + local_index
        for partner_index, partner in enumerate(PARTNERS):
            env = runtime.environment(scene)
            rng = np.random.default_rng(41_900_000 + scene_index * 101 + partner_index)
            episode = f"{scene['id']}:{scene['fingerprint']}:{partner}"
            while not env.done:
                source = env.snapshot(); source_sha = digest(source)
                observations = env.observations()
                actions, decision = runtime.decision(env)
                groups = tuple(critical_groups(env, "robot_2"))
                anchor_enabled = bool(groups) and (dense_critical or env.state.frame % 5 == 0)
                anchor = f"{episode}:{env.state.frame}" if anchor_enabled else ""
                ordinary_index = len(rows)
                rows.append({
                    "observation": observations["robot_2"].astype(np.float32, copy=True),
                    "probabilities": np.asarray(
                        decision["probabilities"]["robot_2"], dtype=np.float32),
                    "action": actions["robot_2"], "scene": scene["fingerprint"],
                    "episode": episode, "frame": int(env.state.frame),
                    "groups": groups, "kind": "ordinary", "anchor": anchor,
                    "branch_action": "", "physical_hash": "",
                    "source_state_hash": source_sha, "submitted_equal": True,
                    "actor_changed_pair": False, "trajectory_done": False,
                })
                if anchor:
                    endpoints: dict[str, dict[str, Any] | None] = {}
                    for player_action in ACTIONS:
                        branch = runtime.from_snapshot(source)
                        transition = runtime.step(branch, player_action)
                        environment_steps += 1
                        if transition["submitted_actions"]["robot_2"] != \
                                transition["policy_actions"]["robot_2"]:
                            raise RuntimeError("Diagnostic v5 branch overrode the Actor")
                        if branch.done:
                            endpoints[player_action] = None
                            continue
                        branch_observation = branch.observations()["robot_2"].astype(
                            np.float32, copy=True)
                        branch_actions, branch_decision = runtime.decision(branch)
                        endpoints[player_action] = {
                            "observation": branch_observation,
                            "probabilities": np.asarray(
                                branch_decision["probabilities"]["robot_2"],
                                dtype=np.float32),
                            "action": branch_actions["robot_2"],
                            "groups": tuple(critical_groups(branch, "robot_2")),
                            "physical_hash": legacy._physical_hash(transition["after"]),
                            "source_state_hash": digest(transition["after"]),
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
                            "actor_changed_pair": changed, "trajectory_done": False,
                        })
                player = partner_action(env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_sha:
                    raise RuntimeError("Diagnostic v5 collection changed source state")
                transition = runtime.step(env, player)
                environment_steps += 1
                if (transition["submitted_actions"]["robot_2"] != actions["robot_2"]
                        or transition["submitted_actions"]["robot_2"]
                           != transition["policy_actions"]["robot_2"]):
                    raise RuntimeError("Diagnostic v5 trajectory overrode the Actor")
                rows[ordinary_index]["trajectory_done"] = bool(transition["done"])
        if progress_label and ((local_index + 1) % 16 == 0
                               or local_index + 1 == len(scenes)):
            print(canonical({"stage": "collect", "split": progress_label,
                "scenes_complete": local_index + 1, "scene_total": len(scenes),
                "rows": len(rows), "environment_steps": environment_steps}),
                flush=True)
    return rows, environment_steps


def _rows_to_arrays(train_rows: Sequence[Mapping[str, Any]],
                    validation_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    raw_train_ordinary = sum(row["kind"] == "ordinary" for row in train_rows)
    raw_validation_ordinary = sum(
        row["kind"] == "ordinary" for row in validation_rows)
    raw_train_anchors = len({str(row["anchor"]) for row in train_rows
                             if row.get("anchor")})
    raw_validation_anchors = len({str(row["anchor"]) for row in validation_rows
                                  if row.get("anchor")})
    validation_hashes = {legacy._obs_hash(row["observation"])
                         for row in validation_rows}
    kept_train = [row for row in train_rows
                  if legacy._obs_hash(row["observation"]) not in validation_hashes]
    affected_anchors = {str(row["anchor"]) for row in train_rows
                        if row.get("anchor")
                        and legacy._obs_hash(row["observation"]) in validation_hashes}
    rows = [*kept_train, *validation_rows]
    split = np.zeros(len(rows), dtype=np.bool_)
    split[len(kept_train):] = True
    labels = np.asarray([ACTIONS.index(row["action"]) for row in rows], dtype=np.uint8)
    bits = np.asarray([legacy._group_bits(row["groups"]) for row in rows], dtype=np.uint8)
    kinds = np.asarray([row["kind"] for row in rows], dtype="S16")
    anchors: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows):
        if row["kind"] == "intervention":
            anchors.setdefault(str(row["anchor"]), {})[str(row["branch_action"])] = index
    changed = np.zeros(len(rows), dtype=np.bool_)
    effective_train_endpoints: set[int] = set()
    for branches in anchors.values():
        wait = branches.get("WAIT")
        if wait is None:
            continue
        for index in branches.values():
            is_changed = bool(rows[index]["physical_hash"] != rows[wait]["physical_hash"]
                              and labels[index] != labels[wait])
            changed[index] = is_changed
            if is_changed and not split[index] and not split[wait]:
                effective_train_endpoints.update((wait, index))
    weights = legacy._sample_weights(
        labels, split, bits, kinds == b"intervention", changed)
    if effective_train_endpoints:
        weights[np.asarray(sorted(effective_train_endpoints), dtype=np.int64)] *= \
            np.float32(PAIR_ENDPOINT_MULTIPLIER)
    arrays = {
        "observations": np.stack([row["observation"] for row in rows]).astype(np.float32),
        "probabilities": np.stack([row["probabilities"] for row in rows]).astype(np.float32),
        "action_indices": labels,
        "weights": weights.astype(np.float32),
        "observation_hashes": np.asarray(
            [legacy._obs_hash(row["observation"]) for row in rows], dtype="S64"),
        "scene_fingerprints": np.asarray([row["scene"] for row in rows], dtype="S64"),
        "episode_ids": np.asarray([row["episode"] for row in rows], dtype="S180"),
        "frames": np.asarray([row["frame"] for row in rows], dtype=np.int16),
        "group_bits": bits,
        "kinds": kinds,
        "anchor_ids": np.asarray([row["anchor"] for row in rows], dtype="S240"),
        "branch_actions": np.asarray([row["branch_action"] for row in rows], dtype="S8"),
        "physical_hashes": np.asarray([row["physical_hash"] for row in rows], dtype="S64"),
        "source_state_hashes": np.asarray(
            [row["source_state_hash"] for row in rows], dtype="S64"),
        "submitted_equal": np.asarray([row["submitted_equal"] for row in rows], dtype=np.bool_),
        "trajectory_done": np.asarray([row["trajectory_done"] for row in rows], dtype=np.bool_),
        "split_validation": split,
    }
    accounting = {
        "raw_train_rows": len(train_rows),
        "raw_validation_rows": len(validation_rows),
        "raw_train_ordinary_rows": raw_train_ordinary,
        "raw_validation_ordinary_rows": raw_validation_ordinary,
        "raw_train_anchor_count": raw_train_anchors,
        "raw_validation_anchor_count": raw_validation_anchors,
        "raw_train_environment_steps": (
            raw_train_ordinary + raw_train_anchors * len(ACTIONS)),
        "raw_validation_environment_steps": (
            raw_validation_ordinary + raw_validation_anchors * len(ACTIONS)),
        "raw_collection_environment_steps": (
            raw_train_ordinary + raw_validation_ordinary
            + (raw_train_anchors + raw_validation_anchors) * len(ACTIONS)),
        "train_rows_removed_for_exact_validation_overlap": (
            len(train_rows) - len(kept_train)),
        "train_anchors_affected_by_duplicate_row_removal": len(affected_anchors),
        "effective_train_pair_endpoint_rows": len(effective_train_endpoints),
    }
    return arrays, accounting


def _effective_pairs(arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    kinds = _decode(arrays["kinds"])
    anchors = _decode(arrays["anchor_ids"])
    branches = _decode(arrays["branch_actions"])
    physical = _decode(arrays["physical_hashes"])
    labels = arrays["action_indices"]
    grouped: dict[str, dict[str, int]] = {}
    for index in np.flatnonzero(mask & (kinds == "intervention")):
        grouped.setdefault(str(anchors[index]), {})[str(branches[index])] = int(index)
    pairs = []
    for values in grouped.values():
        wait = values.get("WAIT")
        if wait is None:
            continue
        for action in ACTIONS[:-1]:
            changed = values.get(action)
            if (changed is not None and physical[wait] != physical[changed]
                    and labels[wait] != labels[changed]):
                pairs.append((wait, changed))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _expected_weights(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    split = arrays["split_validation"]
    kinds = _decode(arrays["kinds"])
    labels = arrays["action_indices"]
    physical = _decode(arrays["physical_hashes"])
    anchors = _decode(arrays["anchor_ids"])
    branches = _decode(arrays["branch_actions"])
    changed = np.zeros(len(labels), dtype=np.bool_)
    grouped: dict[str, dict[str, int]] = {}
    for index in np.flatnonzero(kinds == "intervention"):
        grouped.setdefault(str(anchors[index]), {})[str(branches[index])] = int(index)
    endpoints: set[int] = set()
    for values in grouped.values():
        wait = values.get("WAIT")
        if wait is None:
            continue
        for index in values.values():
            active = physical[index] != physical[wait] and labels[index] != labels[wait]
            changed[index] = active
            if active and not split[index] and not split[wait]:
                endpoints.update((wait, index))
    result = legacy._sample_weights(
        labels, split, arrays["group_bits"], kinds == "intervention", changed)
    if endpoints:
        result[np.asarray(sorted(endpoints), dtype=np.int64)] *= \
            np.float32(PAIR_ENDPOINT_MULTIPLIER)
    return result


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
                     train_scenes: Sequence[Mapping[str, Any]],
                     validation_scenes: Sequence[Mapping[str, Any]]) -> None:
    if set(arrays) != _FIELDS:
        raise ValueError("Diagnostic v5 evidence array set differs")
    count = len(arrays["observations"])
    expected_dtypes = {
        "action_indices": np.dtype(np.uint8), "weights": np.dtype(np.float32),
        "observation_hashes": np.dtype("S64"), "scene_fingerprints": np.dtype("S64"),
        "episode_ids": np.dtype("S180"), "frames": np.dtype(np.int16),
        "group_bits": np.dtype(np.uint8), "kinds": np.dtype("S16"),
        "anchor_ids": np.dtype("S240"), "branch_actions": np.dtype("S8"),
        "physical_hashes": np.dtype("S64"), "source_state_hashes": np.dtype("S64"),
        "submitted_equal": np.dtype(np.bool_), "trajectory_done": np.dtype(np.bool_),
        "split_validation": np.dtype(np.bool_),
    }
    if (count <= 0 or arrays["observations"].shape != (count, actor.obs_dim)
            or arrays["observations"].dtype != np.dtype(np.float32)
            or arrays["probabilities"].shape != (count, len(ACTIONS))
            or arrays["probabilities"].dtype != np.dtype(np.float32)
            or any(arrays[name].shape != (count,) or arrays[name].dtype != dtype
                   for name, dtype in expected_dtypes.items())):
        raise ValueError("Diagnostic v5 evidence shapes or dtypes differ")
    observations = arrays["observations"]
    probabilities = arrays["probabilities"]
    labels = arrays["action_indices"]
    if (not np.isfinite(observations).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or not np.allclose(probabilities.sum(1), 1.0, rtol=0.0, atol=2e-6)
            or np.any(labels >= len(ACTIONS)) or not np.all(arrays["submitted_equal"])
            or np.any(arrays["frames"] < 0)
            or np.any(arrays["group_bits"] >= (1 << len(GROUPS)))):
        raise ValueError("Diagnostic v5 numeric or action-authority evidence differs")
    expected_hashes = np.asarray(
        [legacy._obs_hash(row) for row in observations], dtype="S64")
    if not np.array_equal(expected_hashes, arrays["observation_hashes"]):
        raise ValueError("Diagnostic v5 public-observation hashes differ")
    logits = actor.logits(observations)
    actor_probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    actor_probabilities /= actor_probabilities.sum(axis=1, keepdims=True)
    if (not np.allclose(probabilities, actor_probabilities, rtol=8e-6, atol=4e-6)
            or not np.array_equal(labels, actor_probabilities.argmax(1).astype(np.uint8))):
        raise ValueError("Diagnostic v5 rows differ from the frozen Actor")
    split = arrays["split_validation"]
    scenes = _decode(arrays["scene_fingerprints"])
    episodes = _decode(arrays["episode_ids"])
    train_fingerprints = {row["fingerprint"] for row in train_scenes}
    validation_fingerprints = {row["fingerprint"] for row in validation_scenes}
    if (set(map(str, scenes[~split])) != train_fingerprints
            or set(map(str, scenes[split])) != validation_fingerprints
            or np.any(np.isin(scenes[~split], list(validation_fingerprints)))
            or np.any(np.isin(scenes[split], list(train_fingerprints)))):
        raise ValueError("Diagnostic v5 registered scene partition differs")
    expected_train_episodes = {
        f"{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in train_scenes for partner in PARTNERS
    }
    expected_validation_episodes = {
        f"{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in validation_scenes for partner in PARTNERS
    }
    expected_episodes = {
        f"{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in (*train_scenes, *validation_scenes) for partner in PARTNERS
    }
    if set(map(str, episodes)) != expected_episodes:
        raise ValueError("Diagnostic v5 episode matrix differs")
    train_hashes = set(map(bytes, arrays["observation_hashes"][~split]))
    validation_hashes = set(map(bytes, arrays["observation_hashes"][split]))
    if train_hashes & validation_hashes:
        raise ValueError("Diagnostic v5 exact observation crossed the scene split")
    kinds = _decode(arrays["kinds"])
    anchors = _decode(arrays["anchor_ids"])
    branches = _decode(arrays["branch_actions"])
    physical = _decode(arrays["physical_hashes"])
    ordinary = kinds == "ordinary"
    intervention = kinds == "intervention"
    if (not np.all(ordinary | intervention)
            or np.any(arrays["trajectory_done"][intervention])
            or not np.all(branches[ordinary] == "") or not np.all(physical[ordinary] == "")
            or np.any(anchors[intervention] == "")
            or not set(map(str, branches[intervention])).issubset(ACTIONS)
            or any(_HEX.fullmatch(str(value)) is None for value in physical[intervention])
            or any(_HEX.fullmatch(str(value)) is None
                   for value in _decode(arrays["source_state_hashes"]))):
        raise ValueError("Diagnostic v5 row-kind schema differs")
    frames = arrays["frames"]
    bits = arrays["group_bits"]
    expected_anchor = np.asarray([
        (f"{episodes[index]}:{int(frames[index])}"
         if bits[index] != 0 and (not split[index] or int(frames[index]) % 5 == 0)
         else "")
        for index in np.flatnonzero(ordinary)
    ], dtype="U240")
    if not np.array_equal(anchors[ordinary], expected_anchor):
        raise ValueError("Diagnostic v5 intervention schedule differs")
    # Validation trajectories remain complete.  Train trajectories may have
    # frame gaps only where an exactly duplicated public observation was
    # removed so no observation can occur on both sides of development.
    for episode in expected_validation_episodes:
        indices = np.flatnonzero(ordinary & (episodes == episode))
        ordered = indices[np.argsort(frames[indices], kind="stable")]
        if (not len(ordered)
                or not np.array_equal(frames[ordered], np.arange(len(ordered)))
                or np.any(arrays["trajectory_done"][ordered[:-1]])
                or not bool(arrays["trajectory_done"][ordered[-1]])):
            raise ValueError("Diagnostic v5 trajectory is incomplete")
    for episode in expected_train_episodes:
        indices = np.flatnonzero(ordinary & (episodes == episode))
        # A trajectory may retain_hold only intervention endpoints when all of
        # its ordinary public observations also occur on validation.  The
        # episode identity is still present and no duplicate row is fitted.
        if (len(indices) and (len(np.unique(frames[indices])) != len(indices)
                or np.any(np.diff(np.sort(frames[indices])) <= 0))):
            raise ValueError("Diagnostic v5 filtered training trajectory differs")
    anchor_splits: dict[str, set[bool]] = {}
    for anchor, validation in zip(anchors, split):
        if anchor:
            anchor_splits.setdefault(str(anchor), set()).add(bool(validation))
    if any(len(values) != 1 for values in anchor_splits.values()):
        raise ValueError("Diagnostic v5 intervention anchor crossed the split")
    if not np.array_equal(arrays["weights"], _expected_weights(arrays)):
        raise ValueError("Diagnostic v5 train-only weights differ")


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, c: float):
    model = LogisticRegression(C=c, max_iter=LOGISTIC_MAX_ITER, solver="lbfgs", tol=1e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y, sample_weight=weights)
    if (any(issubclass(item.category, ConvergenceWarning) for item in caught)
            or not np.isfinite(model.coef_).all()
            or not np.isfinite(model.intercept_).all()
            or np.any(model.n_iter_ >= LOGISTIC_MAX_ITER)):
        raise RuntimeError("Diagnostic v5 leaf classifier did not converge")
    return model, int(np.max(model.n_iter_))


def _fit_member(arrays: Mapping[str, np.ndarray], *, actor: NumPyNativeActor,
                seed: int, metadata: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    train = ~arrays["split_validation"]
    observations = arrays["observations"].astype(np.float64)
    labels = arrays["action_indices"]
    weights = arrays["weights"].astype(np.float64)
    variable = observations[train].std(0) > 1e-8
    variable_indices = np.flatnonzero(variable)
    scaler = StandardScaler().fit(observations[train][:, variable])
    scaled = scaler.transform(observations[:, variable])
    train_indices = np.flatnonzero(train)
    bootstrap = np.random.default_rng(seed).choice(
        train_indices, len(train_indices), replace=True)
    router = DecisionTreeClassifier(
        max_depth=ROUTER_DEPTH, max_leaf_nodes=ROUTER_LEAVES,
        min_samples_leaf=ROUTER_MIN_SAMPLES_LEAF,
        max_features=ROUTER_MAX_FEATURES, random_state=seed,
    ).fit(observations[bootstrap], labels[bootstrap], sample_weight=weights[bootstrap])
    leaf_ids = router.apply(observations)
    leaf_nodes = sorted(map(int, np.unique(router.apply(observations[train]))))
    if (router.get_depth() > ROUTER_DEPTH or router.get_n_leaves() > ROUTER_LEAVES
            or len(leaf_nodes) != router.get_n_leaves()):
        raise RuntimeError("Diagnostic v5 router bounds differ")
    models = []
    max_ranking_iterations = max_refit_iterations = 0
    for node in leaf_nodes:
        train_mask = train & (leaf_ids == node)
        classes = np.unique(labels[train_mask])
        if len(classes) == 1:
            models.append({"router_node": node, "classes": [int(classes[0])],
                           "feature_indices": [], "coefficients": [[]],
                           "intercepts": [0.0]})
            continue
        dense, ranking_iterations = _fit_logistic(
            scaled[train_mask], labels[train_mask], weights[train_mask], c=RANKING_C)
        max_ranking_iterations = max(max_ranking_iterations, ranking_iterations)
        norms = np.linalg.norm(dense.coef_, axis=0)
        ranking = np.lexsort((variable_indices, -norms))
        selected_scaled = ranking[:FEATURE_CAP]
        refit, refit_iterations = _fit_logistic(
            scaled[train_mask][:, selected_scaled], labels[train_mask],
            weights[train_mask], c=REFIT_C)
        max_refit_iterations = max(max_refit_iterations, refit_iterations)
        raw_indices = variable_indices[selected_scaled]
        order = np.argsort(raw_indices, kind="stable")
        raw_indices = raw_indices[order]
        scaled_coefficients = refit.coef_[:, order]
        selected_means = scaler.mean_[selected_scaled][order]
        selected_scales = scaler.scale_[selected_scaled][order]
        raw_coefficients = scaled_coefficients / selected_scales[None, :]
        raw_intercepts = refit.intercept_ - np.sum(
            scaled_coefficients * selected_means[None, :] / selected_scales[None, :],
            axis=1)
        if len(refit.classes_) == 2:
            coefficients = [np.zeros(len(raw_indices)).tolist(),
                            raw_coefficients[0].astype(float).tolist()]
            intercepts = [0.0, float(raw_intercepts[0])]
        else:
            coefficients = raw_coefficients.astype(float).tolist()
            intercepts = raw_intercepts.astype(float).tolist()
        models.append({"router_node": node,
            "classes": [int(value) for value in refit.classes_],
            "feature_indices": [int(value) for value in raw_indices],
            "coefficients": coefficients, "intercepts": intercepts})
    model_indices = {node: index for index, node in enumerate(leaf_nodes)}
    payload = {
        "version": MEMBER_VERSION,
        "action_names": list(ACTIONS),
        "feature_names": list(actor.metadata["feature_names"]),
        "router": legacy._router_payload(router, model_indices),
        "leaf_models": models,
        "metadata": {**deepcopy(dict(metadata)), "ensemble_member_seed": seed},
    }
    program = R41DiagnosticModelTreeProgram.from_dict(payload)
    diagnostics = {
        "seed": seed, "router_depth": int(router.get_depth()),
        "router_leaves": int(router.get_n_leaves()),
        "constant_leaves": sum(len(model["classes"]) == 1 for model in models),
        "maximum_ranking_iterations": max_ranking_iterations,
        "maximum_refit_iterations": max_refit_iterations,
        "unquantized_complexity": program.complexity(),
    }
    return program.to_dict(), diagnostics


def _metrics_from_probabilities(probabilities: np.ndarray,
             arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if (probabilities.shape != (len(arrays["observations"]), len(ACTIONS))
            or not np.isfinite(probabilities).all()
            or not np.allclose(probabilities.sum(1), 1.0, rtol=0.0, atol=2e-12)):
        raise ValueError("Diagnostic v5 candidate probabilities differ")
    predictions = np.argmax(probabilities, axis=1).astype(np.uint8)
    labels = arrays["action_indices"]
    correct = predictions == labels
    scenes = _decode(arrays["scene_fingerprints"])
    bits = arrays["group_bits"]
    def stat(selected):
        indices = np.flatnonzero(mask & selected)
        return {"rows": int(len(indices)),
                "scenes": len(set(map(str, scenes[indices]))),
                "fidelity": float(correct[indices].mean()) if len(indices) else 0.0}
    critical = {name: stat((bits & (1 << index)) != 0)
                for index, name in enumerate(GROUPS)}
    pairs = _effective_pairs(arrays, mask)
    pair_correct = (correct[pairs[:, 0]] & correct[pairs[:, 1]]) if len(pairs) \
        else np.empty(0, dtype=np.bool_)
    direction_by_group = {}
    for index, name in enumerate(GROUPS):
        selected = (bits[pairs[:, 0]] & (1 << index)) != 0 if len(pairs) \
            else np.empty(0, dtype=np.bool_)
        direction_by_group[name] = {
            "pairs": int(np.sum(selected)),
            "scenes": len(set(map(str, scenes[pairs[selected, 0]])))
                if np.any(selected) else 0,
            "fidelity": float(pair_correct[selected].mean())
                if np.any(selected) else 0.0,
        }
    target = arrays["probabilities"][mask]
    approximate = probabilities[mask]
    mean_kl = float(np.mean(np.sum(target * (
        np.log(target.clip(1e-8)) - np.log(approximate.clip(1e-8))), axis=-1)))
    return {
        "overall": stat(np.ones(len(mask), dtype=np.bool_)),
        "nonwait": stat(labels != ACTIONS.index("WAIT")),
        "critical": critical,
        "effective_intervention_direction": {
            "pairs": int(len(pairs)),
            "scenes": len(set(map(str, scenes[pairs[:, 0]]))) if len(pairs) else 0,
            "fidelity": float(pair_correct.mean()) if len(pair_correct) else 0.0,
            "by_group": direction_by_group,
        },
        "mean_kl": mean_kl,
    }


def _metrics(program: R41DiagnosticModelTreeEnsemble,
             arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    return _metrics_from_probabilities(
        program.predict_proba_batch(arrays["observations"]), arrays, mask)


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall": metrics["overall"]["fidelity"] >= MIN_OVERALL,
        "nonwait": metrics["nonwait"]["fidelity"] >= MIN_NONWAIT,
        "effective_intervention_direction": (
            metrics["effective_intervention_direction"]["scenes"] >= 10
            and metrics["effective_intervention_direction"]["fidelity"] >= MIN_DIRECTION),
    }
    for group in GROUPS:
        row = metrics["critical"][group]
        checks[group] = row["scenes"] >= 10 and row["fidelity"] >= MIN_CRITICAL
        direction = metrics["effective_intervention_direction"]["by_group"][group]
        checks[f"effective_intervention_direction_{group}"] = (
            direction["scenes"] >= 10 and direction["fidelity"] >= MIN_DIRECTION)
    return {"checks": checks, "passed": all(checks.values())}


def _fit_candidate(actor: NumPyNativeActor, arrays: Mapping[str, np.ndarray], *,
                   binding: str, source_full_manifest_bindings: Mapping[str, Any]):
    base_metadata = {
        "diagnostic_rcpd_version": VERSION,
        "diagnostic_rcpd_binding_sha256": binding,
        "source_full_manifest_bindings": deepcopy(dict(source_full_manifest_bindings)),
        "native_source_actor_sha256": actor.artifact_sha256,
        "source_actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "program_family": "four_axis_router_sparse_public_linear_softmax_ensemble",
        "member_seed_candidates": list(MEMBER_SEED_CANDIDATES),
        "router_depth_cap": ROUTER_DEPTH, "router_leaf_cap": ROUTER_LEAVES,
        "router_min_samples_leaf": ROUTER_MIN_SAMPLES_LEAF,
        "maximum_features_per_leaf": FEATURE_CAP,
        "coefficient_quantization_qmax": 127,
        "action_legality_features": {}, "action_constraint_reason_features": {},
        "runtime_controller": "native_neural_actor_only",
        "runtime_action_override": False, "program_feedback_into_actor": False,
        "ppo_joint_steps": 0, "optimizer_updates": 0, "formal_ready": False,
    }
    unquantized: dict[int, dict[str, Any]] = {}
    diagnostics_by_seed: dict[int, dict[str, Any]] = {}
    probabilities_by_seed: dict[int, np.ndarray] = {}
    for seed in MEMBER_SEED_CANDIDATES:
        member, diagnostics = _fit_member(
            arrays, actor=actor, seed=seed, metadata=base_metadata)
        unquantized[seed] = member
        diagnostics_by_seed[seed] = diagnostics
        # Quantize each member by the exact runtime serializer before comparing
        # candidates.  Four copies have the same mean distribution and avoid a
        # second, selection-only model representation.
        singleton = R41DiagnosticModelTreeEnsemble.from_unquantized_members(
            [(seed * 10 + index, member) for index in range(MEMBER_COUNT)],
            metadata=base_metadata)
        probabilities_by_seed[seed] = singleton.predict_proba_batch(
            arrays["observations"])

    compared = []
    validation = arrays["split_validation"]
    for seeds in combinations(MEMBER_SEED_CANDIDATES, MEMBER_COUNT):
        probabilities = np.mean(
            [probabilities_by_seed[seed] for seed in seeds], axis=0)
        metrics = _metrics_from_probabilities(probabilities, arrays, validation)
        direction = metrics["effective_intervention_direction"]
        ordinary_minimum = min(
            metrics["overall"]["fidelity"], metrics["nonwait"]["fidelity"],
            *(metrics["critical"][group]["fidelity"] for group in GROUPS))
        score = [
            min(direction["by_group"][group]["fidelity"] for group in GROUPS),
            direction["fidelity"], ordinary_minimum,
        ]
        compared.append({
            "member_seeds": list(seeds),
            "score": score,
            "gate": _gate(metrics),
            "metrics": metrics,
        })
    compared.sort(key=lambda row: (
        -row["score"][0], -row["score"][1], -row["score"][2],
        tuple(row["member_seeds"])))
    selected = compared[0]
    selected_seeds = tuple(selected["member_seeds"])
    metadata = {**base_metadata, "member_seeds": list(selected_seeds),
        "development_candidate_count": len(compared),
        "development_selection_score": list(selected["score"])}
    program = R41DiagnosticModelTreeEnsemble.from_unquantized_members(
        [(seed, unquantized[seed]) for seed in selected_seeds], metadata=metadata)
    restored = R41DiagnosticModelTreeEnsemble.from_dict(program.to_dict())
    predictions = program.predict_batch(arrays["observations"])
    if not np.array_equal(predictions, restored.predict_batch(arrays["observations"])):
        raise RuntimeError("Diagnostic v5 compact serialization changed an action")
    metrics = _metrics(program, arrays, validation)
    gate = _gate(metrics)
    if metrics != selected["metrics"] or gate != selected["gate"]:
        raise RuntimeError("Diagnostic v5 selected candidate replay differs")
    return program, {
        "profile": {"member_seeds": list(selected_seeds),
            "member_seed_candidates": list(MEMBER_SEED_CANDIDATES),
            "router_depth": ROUTER_DEPTH, "router_leaves": ROUTER_LEAVES,
            "router_min_samples_leaf": ROUTER_MIN_SAMPLES_LEAF,
            "router_max_features": ROUTER_MAX_FEATURES,
            "feature_cap": FEATURE_CAP, "ranking_c": RANKING_C,
            "refit_c": REFIT_C, "pair_endpoint_multiplier": PAIR_ENDPOINT_MULTIPLIER,
            "coefficient_qmax": 127},
        "selection": {
            "candidate_count": len(compared),
            "passing_count": sum(row["gate"]["passed"] for row in compared),
            "rule": contract()["candidate_selection_disclosure"],
            "selected_score": list(selected["score"]),
            "ranked_candidates": compared,
            "final_test_accessed": False,
        },
        "metrics": metrics, "gate": gate, "complexity": program.complexity(),
        "fit_diagnostics": {
            "train_rows": int(np.sum(~arrays["split_validation"])),
            "validation_rows_excluded_from_fit": int(np.sum(arrays["split_validation"])),
            "public_features_available": actor.obs_dim,
            "members": [diagnostics_by_seed[seed]
                        for seed in MEMBER_SEED_CANDIDATES],
            "sklearn_version": sklearn.__version__,
            "fit_inputs": ["public_observation", "frozen_actor_action",
                           "registered_train_only_sample_weight"],
            "prediction_inputs": ["public_observation"],
            "validation_labels_used_for_member_fit": False,
            "validation_labels_used_for_candidate_selection": True,
            "final_test_accessed": False,
            "actor_parameter_arrays_accessed": False,
            "actor_logits_used_as_program_input": False,
            "actor_hidden_states_used": False,
            "runtime_action_override": False,
        },
        "program_content_sha256": digest(program.to_dict()),
    }


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
    legacy._validate_manifest_actor(manifest, actor)
    legacy._validate_designation(designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        actor=actor, protocol=protocol)
    train_scenes, validation_scenes, holdout = _scene_splits(manifest)
    runtime = legacy._runtime(actor_path, protocol_path, manifest_path, manifest)
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
        "holdout_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in holdout)),
    }
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o700)
    _write_json(output / "inputs.json", {"version": VERSION, "bindings": bindings,
        "contract": contract(), "sources": sources, "formal_ready": False})
    train_rows, train_steps = _collect(
        runtime, train_scenes, scene_offset=0, dense_critical=True,
        progress_label="train")
    validation_rows, validation_steps = _collect(
        runtime, validation_scenes, scene_offset=len(train_scenes),
        dense_critical=False, progress_label="conflict_validation")
    arrays, row_accounting = _rows_to_arrays(train_rows, validation_rows)
    _validate_arrays(arrays, actor=actor, train_scenes=train_scenes,
                     validation_scenes=validation_scenes)
    _write_npz(output / "rows.npz", arrays)
    binding = digest({"bindings": bindings, "rows_sha256": file_hash(output / "rows.npz")})
    program, candidate = _fit_candidate(
        actor, arrays, binding=binding,
        source_full_manifest_bindings=runtime.source_full_manifest_bindings)
    _write_json(output / "candidates.json", {
        "version": VERSION, "binding_sha256": binding,
        "selection_status": "frozen_development_candidate",
        "selection_disclosure": contract()["candidate_selection_disclosure"],
        "candidate": candidate,
    })
    envelope = make_program_envelope(program)
    _write_json(output / "program.json", envelope)
    if ((output / "program.json").stat().st_size > MAX_PROGRAM_BYTES
            or load_program_envelope(_read(output / "program.json", "program")).to_dict()
               != program.to_dict()):
        raise RuntimeError("Diagnostic v5 release program budget or round trip differs")
    train_hashes = set(map(bytes, arrays["observation_hashes"][~arrays["split_validation"]]))
    validation_hashes = set(map(bytes, arrays["observation_hashes"][arrays["split_validation"]]))
    artifacts = {name: file_hash(output / name) for name in
                 ("inputs.json", "rows.npz", "candidates.json")}
    expected_steps = row_accounting["raw_collection_environment_steps"]
    report = {
        "version": VERSION,
        "status": "passed" if candidate["gate"]["passed"] else "failed",
        "explanation_eligible": bool(candidate["gate"]["passed"]),
        "formal_ready": False, "bindings": bindings,
        "diagnostic_rcpd_binding_sha256": binding,
        "rows": len(arrays["observations"]),
        "ordinary_rows": int(np.sum(_decode(arrays["kinds"]) == "ordinary")),
        "intervention_rows": int(np.sum(_decode(arrays["kinds"]) == "intervention")),
        "train_rows": int(np.sum(~arrays["split_validation"])),
        "validation_rows": int(np.sum(arrays["split_validation"])),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "exact_observation_overlap": len(train_hashes & validation_hashes),
        "row_accounting": row_accounting,
        "candidate": candidate,
        "program_file_sha256": file_hash(output / "program.json"),
        "program_content_sha256": digest(program.to_dict()),
        "program_transport_payload_sha256": envelope["payload_sha256"],
        "program_json_bytes": (output / "program.json").stat().st_size,
        "evidence_artifacts": artifacts,
        "execution": {"environment_steps": train_steps + validation_steps,
            "derived_environment_steps": expected_steps,
            "actor_queries": len(arrays["observations"]),
            "router_fits": len(MEMBER_SEED_CANDIDATES), "leaf_ensemble_fits": 1,
            "ppo_joint_steps": 0, "optimizer_updates": 0,
            "actor_changed": False, "runtime_action_overrides": 0,
            "final_test_accessed": False},
    }
    if train_steps + validation_steps != expected_steps:
        raise RuntimeError("Diagnostic v5 environment-step accounting differs")
    if (producer_sources() != sources or file_hash(actor_path) != bindings["actor_file_sha256"]
            or file_hash(protocol_path) != bindings["protocol_file_sha256"]
            or file_hash(manifest_path) != bindings["manifest_file_sha256"]
            or file_hash(designation_path) != bindings["designation_sha256"]):
        raise RuntimeError("Frozen diagnostic v5 inputs changed during extraction")
    _write_json(output / "report.json", report)
    read_saved_report(output,
        expected_report_sha256=file_hash(output / "report.json"),
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        program_path=output / "program.json", require_passed=False)
    return report


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      actor_path: str | Path, protocol_path: str | Path,
                      manifest_path: str | Path, designation_path: str | Path,
                      expected_designation_sha256: str,
                      program_path: str | Path | None,
                      require_passed: bool = True) -> dict[str, Any]:
    """Reauthenticate rows, refit all members, and recompute every gate."""
    output = Path(output).expanduser().absolute()
    if not output.is_dir() or output.is_symlink() or output.resolve() != output:
        raise ValueError("Diagnostic v5 evidence directory is missing or unsafe")
    report_path = output / "report.json"
    if (_HEX.fullmatch(str(expected_report_sha256)) is None
            or not report_path.is_file() or report_path.is_symlink()
            or file_hash(report_path) != expected_report_sha256):
        raise ValueError("Diagnostic v5 report hash differs")
    actor_path = _regular(actor_path, "Actor")
    protocol_path = _regular(protocol_path, "training protocol")
    manifest_path = _regular(manifest_path, "diagnostic manifest")
    designation_path = _regular(designation_path, "diagnostic designation")
    actor = NumPyNativeActor(actor_path)
    protocol = _read(protocol_path, "training protocol")
    manifest = _read(manifest_path, "diagnostic manifest")
    legacy._validate_manifest_actor(manifest, actor)
    legacy._validate_designation(designation_path=designation_path,
        expected_designation_sha256=expected_designation_sha256,
        actor_path=actor_path, protocol_path=protocol_path,
        actor=actor, protocol=protocol)
    train_scenes, validation_scenes, holdout = _scene_splits(manifest)
    runtime = legacy._runtime(actor_path, protocol_path, manifest_path, manifest)
    sources = producer_sources()
    content = deepcopy(manifest); content_sha = content.pop("content_sha256", None)
    if content_sha != digest(content):
        raise ValueError("Diagnostic v5 manifest content digest differs")
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
        "holdout_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in holdout)),
    }
    inputs = _read(output / "inputs.json", "Diagnostic v5 inputs")
    if inputs != {"version": VERSION, "bindings": bindings,
                  "contract": contract(), "sources": sources,
                  "formal_ready": False}:
        raise ValueError("Diagnostic v5 input provenance differs")
    report = _read(report_path, "Diagnostic v5 report")
    artifacts = report.get("evidence_artifacts")
    expected_artifacts = {"inputs.json", "rows.npz", "candidates.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Diagnostic v5 evidence artifact set differs")
    for name, expected in artifacts.items():
        path = output / name
        if (not path.is_file() or path.is_symlink() or path.parent != output
                or _HEX.fullmatch(str(expected)) is None or file_hash(path) != expected):
            raise ValueError("Diagnostic v5 evidence artifact changed")
    arrays = _load_npz(output / "rows.npz")
    _validate_arrays(arrays, actor=actor, train_scenes=train_scenes,
                     validation_scenes=validation_scenes)
    binding = digest({"bindings": bindings, "rows_sha256": artifacts["rows.npz"]})
    program, candidate = _fit_candidate(
        actor, arrays, binding=binding,
        source_full_manifest_bindings=runtime.source_full_manifest_bindings)
    stored = _read(output / "candidates.json", "Diagnostic v5 candidates")
    if stored != {"version": VERSION, "binding_sha256": binding,
                  "selection_status": "frozen_development_candidate",
                  "selection_disclosure": contract()["candidate_selection_disclosure"],
                  "candidate": candidate}:
        raise ValueError("Diagnostic v5 candidate differs from train-only refit")
    if program_path is None:
        raise ValueError("Diagnostic v5 candidate requires its program")
    supplied = _regular(program_path, "Diagnostic v5 program")
    if supplied != output / "program.json" or supplied.stat().st_size > MAX_PROGRAM_BYTES:
        raise ValueError("Diagnostic v5 program path or size differs")
    saved_program = load_program_envelope(_read(supplied, "Diagnostic v5 program"))
    if saved_program.to_dict() != program.to_dict():
        raise ValueError("Diagnostic v5 program differs from authenticated refit")
    train_hashes = set(map(bytes, arrays["observation_hashes"][~arrays["split_validation"]]))
    validation_hashes = set(map(bytes, arrays["observation_hashes"][arrays["split_validation"]]))
    row_accounting = report.get("row_accounting")
    expected_accounting_fields = {
        "raw_train_rows", "raw_validation_rows",
        "raw_train_ordinary_rows", "raw_validation_ordinary_rows",
        "raw_train_anchor_count", "raw_validation_anchor_count",
        "raw_train_environment_steps", "raw_validation_environment_steps",
        "raw_collection_environment_steps",
        "train_rows_removed_for_exact_validation_overlap",
        "train_anchors_affected_by_duplicate_row_removal",
        "effective_train_pair_endpoint_rows",
    }
    kinds = _decode(arrays["kinds"])
    split = arrays["split_validation"]
    anchors = _decode(arrays["anchor_ids"])
    retained_train_anchors = len(set(map(str, anchors[(anchors != "") & ~split])))
    retained_validation_anchors = len(set(map(str, anchors[(anchors != "") & split])))
    train_pairs = _effective_pairs(arrays, ~split)
    effective_train_endpoints = len(set(map(int, train_pairs.reshape(-1))))
    if (not isinstance(row_accounting, dict)
            or set(row_accounting) != expected_accounting_fields
            or any(type(value) is not int or value < 0
                   for value in row_accounting.values())
            or row_accounting["raw_validation_rows"]
               != int(np.sum(arrays["split_validation"]))
            or row_accounting["raw_train_rows"]
               - row_accounting["train_rows_removed_for_exact_validation_overlap"]
               != int(np.sum(~arrays["split_validation"]))
            or row_accounting["raw_validation_ordinary_rows"]
               != int(np.sum(split & (kinds == "ordinary")))
            or row_accounting["raw_train_ordinary_rows"]
               < int(np.sum(~split & (kinds == "ordinary")))
            or row_accounting["raw_train_anchor_count"] < retained_train_anchors
            or row_accounting["raw_validation_anchor_count"]
               != retained_validation_anchors
            or row_accounting["effective_train_pair_endpoint_rows"]
               != effective_train_endpoints
            or row_accounting["raw_train_environment_steps"]
               != row_accounting["raw_train_ordinary_rows"]
                  + row_accounting["raw_train_anchor_count"] * len(ACTIONS)
            or row_accounting["raw_validation_environment_steps"]
               != row_accounting["raw_validation_ordinary_rows"]
                  + row_accounting["raw_validation_anchor_count"] * len(ACTIONS)
            or row_accounting["raw_collection_environment_steps"]
               != row_accounting["raw_train_environment_steps"]
                  + row_accounting["raw_validation_environment_steps"]):
        raise ValueError("Diagnostic v5 row accounting differs")
    expected_steps = row_accounting["raw_collection_environment_steps"]
    execution = report.get("execution", {})
    if (type(execution.get("environment_steps")) is not int
            or execution["environment_steps"] != expected_steps):
        raise ValueError("Diagnostic v5 execution accounting differs")
    expected_report = {
        "version": VERSION,
        "status": "passed" if candidate["gate"]["passed"] else "failed",
        "explanation_eligible": bool(candidate["gate"]["passed"]),
        "formal_ready": False, "bindings": bindings,
        "diagnostic_rcpd_binding_sha256": binding,
        "rows": len(arrays["observations"]),
        "ordinary_rows": int(np.sum(_decode(arrays["kinds"]) == "ordinary")),
        "intervention_rows": int(np.sum(_decode(arrays["kinds"]) == "intervention")),
        "train_rows": int(np.sum(~arrays["split_validation"])),
        "validation_rows": int(np.sum(arrays["split_validation"])),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(validation_scenes),
        "exact_observation_overlap": len(train_hashes & validation_hashes),
        "row_accounting": report.get("row_accounting"),
        "candidate": candidate,
        "program_file_sha256": file_hash(supplied),
        "program_content_sha256": digest(program.to_dict()),
        "program_transport_payload_sha256": _read(
            supplied, "Diagnostic v5 program")["payload_sha256"],
        "program_json_bytes": supplied.stat().st_size,
        "evidence_artifacts": artifacts,
        "execution": {"environment_steps": expected_steps,
            "derived_environment_steps": expected_steps,
            "actor_queries": len(arrays["observations"]),
            "router_fits": len(MEMBER_SEED_CANDIDATES), "leaf_ensemble_fits": 1,
            "ppo_joint_steps": 0, "optimizer_updates": 0,
            "actor_changed": False, "runtime_action_overrides": 0,
            "final_test_accessed": False},
    }
    if digest(report) != digest(expected_report):
        raise ValueError("Diagnostic v5 report differs from immutable evidence")
    if require_passed and expected_report["status"] != "passed":
        raise ValueError("Diagnostic v5 did not meet its registered explanation gates")
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
    print(canonical({"status": report["status"], "candidate": report["candidate"],
                     "report": str((Path(args.output) / "report.json").resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "SCENE_MANIFEST_VERSION", "PARTNERS", "GROUPS", "MEMBER_SEED_CANDIDATES", "MEMBER_COUNT",
    "MIN_OVERALL", "MIN_NONWAIT", "MIN_CRITICAL", "MIN_DIRECTION", "_FIELDS",
    "contract", "producer_sources", "_scene_splits", "_rows_to_arrays",
    "_validate_arrays", "_effective_pairs", "_metrics", "_gate",
    "_fit_candidate", "extract", "read_saved_report", "main",
]
