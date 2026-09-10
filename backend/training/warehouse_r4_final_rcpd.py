"""Extract the final warehouse r4 RCPD from one frozen Actor.

This command is deliberately separate from PPO training.  It collects complete
robot_2 trajectories from the registered ``extraction`` split, labels every
observation with the current frozen NumPy Actor, fits all registered tree
capacities on one scene-disjoint pool, and selects on another.  It never loads
an optimizer or feeds the selected program back into the Actor.

The saved report is useful evidence only together with its sibling NPZ pools
and candidate-program file.  :func:`read_saved_report` replays all numerical
claims from those immutable files and the supplied Actor before a runtime
bundle may consume the selected program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from backend.training.warehouse_native_common import (
    ROOT, canonical, digest, file_hash,
)
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from backend.training.warehouse_r4_production_admission import local_source_hashes
from core.policy_program_regularizer import program_complexity
from core.program import ExecutableProgram
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario


VERSION = "warehouse-r4-final-rcpd.v1"
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
TRAIN_SCENES = 70
VALIDATION_SCENES = 30
TRAIN_SEED = 260_911_210
VALIDATION_SEED = 260_911_310
DEPTHS = (4, 6, 8, 10, 12)
LEAVES = (16, 32, 64, 128, 256)
MINIMUM_OVERALL_FIDELITY = .90
MINIMUM_CRITICAL_FIDELITY = .85
EVIDENCE_FILES = ("inputs.json", "train_rows.npz", "validation_rows.npz",
                  "candidates.json")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ARRAY_FIELDS = frozenset((
    "observations", "probabilities", "episode_ids", "scenario_fingerprints",
    "partners", "frames", "after_frames", "done", "group_bits",
    "policy_action_indices", "player_action_indices", "submitted_equal",
    "source_state_sha256",
))
_CANDIDATE_FIELDS = frozenset((
    "depth_cap", "leaf_cap", "metrics", "gate", "complexity",
    "program_content_sha256", "program",
))
MAX_JSON_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> Any:
    path = Path(path).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError("Final-RCPD JSON is missing, linked, noncanonical, or oversized")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field in final-RCPD evidence")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Non-finite JSON value in final-RCPD evidence: " + value)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=nonfinite)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_split": "extraction",
        "source_scenes": TRAIN_SCENES + VALIDATION_SCENES,
        "train_scenes": TRAIN_SCENES,
        "validation_scenes": VALIDATION_SCENES,
        "partition": "registered extraction order: first 70 train, final 30 validation",
        "partners": list(PARTNERS),
        "collection_seeds": {"train": TRAIN_SEED, "validation": VALIDATION_SEED},
        "evaluated_role": "robot_2",
        "deterministic_actor": True,
        "complete_episodes": True,
        "candidate_depths": list(DEPTHS),
        "candidate_leaves": list(LEAVES),
        "min_samples_leaf": 4,
        "minimum_overall_fidelity": MINIMUM_OVERALL_FIDELITY,
        "minimum_critical_fidelity": MINIMUM_CRITICAL_FIDELITY,
        "critical_groups": list(GROUPS),
        "selection": "minimum structural loss, then KL, fidelity, depth cap, leaf cap",
        "ppo_joint_steps": 0,
        "optimizer_updates": 0,
        "program_feedback_into_actor": False,
        "runtime_action_override": False,
        "independent_intervention_audit_required": True,
    }


def producer_sources() -> dict[str, str]:
    return local_source_hashes((Path(__file__),))


def _regular_input(value: str | Path, label: str) -> Path:
    supplied = Path(value).expanduser().absolute()
    if (supplied.is_symlink() or supplied.resolve() != supplied
            or not supplied.is_file()):
        raise ValueError(label + " must be a canonical regular file")
    return supplied


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Final-RCPD output path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    _write_new(path, (canonical(value) + "\n").encode("utf-8"))


def _write_npz(path: Path, value: Mapping[str, np.ndarray]) -> None:
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Final-RCPD output path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            np.savez_compressed(stream, **value)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float32)
    result = np.exp(value - value.max(axis=-1, keepdims=True))
    result /= result.sum(axis=-1, keepdims=True)
    return result.astype(np.float32)


def _group_bits(groups: Sequence[str]) -> int:
    if len(groups) != len(set(groups)) or not set(groups).issubset(GROUPS):
        raise ValueError("Unknown or duplicate critical group")
    return sum(1 << GROUPS.index(group) for group in groups)


def _groups(bits: np.ndarray) -> list[tuple[str, ...]]:
    return [tuple(group for index, group in enumerate(GROUPS)
                  if int(value) & (1 << index)) for value in bits]


def _collect_pool(actor: NumPyNativeActor, scenes: Sequence[Mapping[str, Any]],
                  *, pool: str, seed: int) -> dict[str, np.ndarray]:
    values: dict[str, list[Any]] = {name: [] for name in _ARRAY_FIELDS}
    config = collaborative_study_config()
    for scene_index, scene in enumerate(scenes):
        for partner_index, partner in enumerate(PARTNERS):
            episode = f"{pool}:{scene['id']}:{scene['fingerprint']}:{partner}"
            environment = PublicFeedbackEnvironment(
                config, REWARD, collision_cost=.05, mode="observed")
            reset_scenario(environment, scene)
            rng = np.random.default_rng(seed + scene_index * 101 + partner_index)
            while not environment.done:
                snapshot = environment.snapshot()
                snapshot_sha256 = digest(snapshot)
                frame = int(environment.state.frame)
                observations = environment.observations()
                if (tuple(environment.feature_names) != tuple(actor.metadata["feature_names"])
                        or observations["robot_2"].shape != (actor.obs_dim,)):
                    raise ValueError("Actor and final-extraction observation schemas differ")
                actions, probabilities = actor.act(observations, deterministic=True)
                if digest(environment.snapshot()) != snapshot_sha256:
                    raise RuntimeError("Frozen Actor inference changed the source state")
                player = partner_action(environment, "robot_1", partner, rng)
                if digest(environment.snapshot()) != snapshot_sha256:
                    raise RuntimeError("Extraction partner changed the source state")
                groups = critical_groups(environment, "robot_2")
                action_index = ACTIONS.index(actions["robot_2"])
                _, _, terminated, truncated, info = environment.step({
                    "robot_1": player, "robot_2": actions["robot_2"],
                })
                equal = (info["requested_actions"]["robot_2"] == actions["robot_2"]
                         and actions["robot_2"] == ACTIONS[int(np.argmax(
                             probabilities["robot_2"]))])
                if not equal:
                    raise RuntimeError("Environment command differs from the frozen Actor")
                row = {
                    "observations": observations["robot_2"].astype(np.float32, copy=True),
                    "probabilities": probabilities["robot_2"].astype(np.float32, copy=True),
                    "episode_ids": episode,
                    "scenario_fingerprints": scene["fingerprint"],
                    "partners": partner,
                    "frames": frame,
                    "after_frames": int(environment.state.frame),
                    "done": bool(terminated or truncated),
                    "group_bits": _group_bits(groups),
                    "policy_action_indices": action_index,
                    "player_action_indices": ACTIONS.index(player),
                    "submitted_equal": equal,
                    "source_state_sha256": snapshot_sha256,
                }
                for name, value in row.items():
                    values[name].append(value)
    if not values["observations"]:
        raise ValueError("Final RCPD extraction collected no neural rows")
    return {
        "observations": np.stack(values["observations"]).astype(np.float32),
        "probabilities": np.stack(values["probabilities"]).astype(np.float32),
        "episode_ids": np.asarray(values["episode_ids"], dtype="U180"),
        "scenario_fingerprints": np.asarray(values["scenario_fingerprints"], dtype="U64"),
        "partners": np.asarray(values["partners"], dtype="U16"),
        "frames": np.asarray(values["frames"], dtype=np.int16),
        "after_frames": np.asarray(values["after_frames"], dtype=np.int16),
        "done": np.asarray(values["done"], dtype=np.bool_),
        "group_bits": np.asarray(values["group_bits"], dtype=np.uint8),
        "policy_action_indices": np.asarray(values["policy_action_indices"], dtype=np.uint8),
        "player_action_indices": np.asarray(values["player_action_indices"], dtype=np.uint8),
        "submitted_equal": np.asarray(values["submitted_equal"], dtype=np.bool_),
        "source_state_sha256": np.asarray(values["source_state_sha256"], dtype="U64"),
    }


def _validate_pool(value: Mapping[str, np.ndarray], *, name: str,
                   actor: NumPyNativeActor, expected_scenes: Sequence[Mapping[str, Any]]) -> None:
    if set(value) != _ARRAY_FIELDS:
        raise ValueError(f"{name} final-RCPD evidence fields differ")
    rows = len(value["observations"])
    maximum_rows = len(expected_scenes) * len(PARTNERS) * collaborative_study_config().horizon
    if (rows < 128 or rows > maximum_rows
            or any(array.ndim < 1 or len(array) != rows for array in value.values())):
        raise ValueError(f"{name} final-RCPD evidence is incomplete")
    if (value["observations"].shape != (rows, actor.obs_dim)
            or value["probabilities"].shape != (rows, len(ACTIONS))
            or not np.isfinite(value["observations"]).all()
            or not np.isfinite(value["probabilities"]).all()
            or np.any(value["probabilities"] < 0)
            or not np.allclose(value["probabilities"].sum(-1), 1., atol=1e-6)
            or not np.all(value["after_frames"] == value["frames"] + 1)
            or not np.all(value["submitted_equal"])
            or np.any(value["group_bits"] >= (1 << len(GROUPS)))
            or np.any(value["policy_action_indices"] >= len(ACTIONS))
            or np.any(value["player_action_indices"] >= len(ACTIONS))):
        raise ValueError(f"{name} final-RCPD evidence values differ")
    logits = actor.logits(value["observations"])
    probabilities = _softmax(logits)
    # Per-step two-row inference and batched replay can take different BLAS
    # reduction paths.  Preserve exact argmax labels and the repository's
    # established 1e-4 NumPy parity bound while using a tighter probability
    # tolerance here.
    if (not np.allclose(probabilities, value["probabilities"], atol=1e-5, rtol=1e-5)
            or not np.array_equal(probabilities.argmax(-1), value["policy_action_indices"])):
        raise ValueError(f"{name} labels do not come from the supplied frozen Actor")
    expected_fingerprints = {scene["fingerprint"] for scene in expected_scenes}
    if set(map(str, value["scenario_fingerprints"])) != expected_fingerprints:
        raise ValueError(f"{name} scene membership differs")
    episode_rows: dict[str, list[int]] = {}
    for index, episode in enumerate(map(str, value["episode_ids"])):
        episode_rows.setdefault(episode, []).append(index)
    expected_episodes = {
        f"{name}:{scene['id']}:{scene['fingerprint']}:{partner}"
        for scene in expected_scenes for partner in PARTNERS
    }
    if set(episode_rows) != expected_episodes:
        raise ValueError(f"{name} episode matrix differs")
    for indices in episode_rows.values():
        ordered = sorted(indices, key=lambda index: int(value["frames"][index]))
        if ([int(value["frames"][index]) for index in ordered] != list(range(len(ordered)))
                or any(bool(value["done"][index]) for index in ordered[:-1])
                or not bool(value["done"][ordered[-1]])):
            raise ValueError(f"{name} contains a partial or discontinuous episode")
        first = ordered[0]
        fingerprint = str(value["scenario_fingerprints"][first])
        partner = str(value["partners"][first])
        if (any(str(value["scenario_fingerprints"][index]) != fingerprint
                or str(value["partners"][index]) != partner for index in ordered)
                or not str(value["episode_ids"][first]).endswith(
                    f":{fingerprint}:{partner}")):
            raise ValueError(f"{name} episode provenance columns disagree")
    if set(map(str, value["partners"])) != set(PARTNERS):
        raise ValueError(f"{name} partner coverage differs")
    if any(_HEX.fullmatch(str(value)) is None for value in value["source_state_sha256"]):
        raise ValueError(f"{name} source-state provenance differs")


def _replay_pool(value: Mapping[str, np.ndarray], *, name: str,
                 actor: NumPyNativeActor,
                 expected_scenes: Sequence[Mapping[str, Any]], seed: int) -> None:
    """Rebuild every saved row from its registered physical trajectory."""
    config = collaborative_study_config()
    by_episode: dict[str, list[int]] = {}
    for index, episode in enumerate(map(str, value["episode_ids"])):
        by_episode.setdefault(episode, []).append(index)
    for scene_index, scene in enumerate(expected_scenes):
        for partner_index, partner in enumerate(PARTNERS):
            episode = f"{name}:{scene['id']}:{scene['fingerprint']}:{partner}"
            indices = sorted(by_episode[episode],
                             key=lambda row: int(value["frames"][row]))
            environment = PublicFeedbackEnvironment(
                config, REWARD, collision_cost=.05, mode="observed")
            reset_scenario(environment, scene)
            rng = np.random.default_rng(seed + scene_index * 101 + partner_index)
            for index in indices:
                snapshot_sha256 = digest(environment.snapshot())
                frame = int(environment.state.frame)
                observations = environment.observations()
                actions, probabilities = actor.act(observations, deterministic=True)
                if digest(environment.snapshot()) != snapshot_sha256:
                    raise RuntimeError("Frozen Actor changed a replayed source state")
                player = partner_action(environment, "robot_1", partner, rng)
                groups = critical_groups(environment, "robot_2")
                if (digest(environment.snapshot()) != snapshot_sha256
                        or frame != int(value["frames"][index])
                        or snapshot_sha256 != str(value["source_state_sha256"][index])
                        or not np.array_equal(
                            observations["robot_2"].astype(np.float32, copy=False),
                            value["observations"][index])
                        or not np.allclose(probabilities["robot_2"],
                                           value["probabilities"][index],
                                           atol=1e-6, rtol=1e-6)
                        or ACTIONS.index(actions["robot_2"])
                            != int(value["policy_action_indices"][index])
                        or ACTIONS.index(player)
                            != int(value["player_action_indices"][index])
                        or _group_bits(groups) != int(value["group_bits"][index])):
                    raise ValueError(
                        f"{name} row is not from its registered physical trajectory")
                submitted = {"robot_1": player, "robot_2": actions["robot_2"]}
                _, _, terminated, truncated, info = environment.step(submitted)
                if (info["requested_actions"] != submitted
                        or int(environment.state.frame)
                            != int(value["after_frames"][index])
                        or bool(terminated or truncated) != bool(value["done"][index])):
                    raise ValueError(f"{name} replay transition differs")
            if not environment.done:
                raise ValueError(f"{name} replay contains a partial episode")


def _metrics(program: ExecutableProgram, value: Mapping[str, np.ndarray]) -> dict[str, Any]:
    manager = FeedbackManager(program.feature_names)
    predicted = manager._predict(program, value["observations"])
    target = value["probabilities"]
    correct = predicted.argmax(-1) == target.argmax(-1)
    nonwait = value["policy_action_indices"] != ACTIONS.index("WAIT")
    labels = _groups(value["group_bits"])

    def stat(mask: np.ndarray) -> dict[str, Any]:
        indices = np.flatnonzero(mask)
        return {
            "rows": int(len(indices)),
            "scenes": len(set(map(str, value["scenario_fingerprints"][indices]))),
            "fidelity": float(correct[indices].mean()) if len(indices) else None,
        }

    critical, critical_nonwait = {}, {}
    for group in GROUPS:
        mask = np.asarray([group in row for row in labels], dtype=np.bool_)
        critical[group] = stat(mask)
        critical_nonwait[group] = stat(mask & nonwait)
    mean_kl = float(np.mean(np.sum(target * (
        np.log(target.clip(1e-8)) - np.log(predicted.clip(1e-8))), axis=-1)))
    return {
        "overall": stat(np.ones(len(correct), dtype=np.bool_)),
        "nonwait": stat(nonwait),
        "critical": critical,
        "critical_nonwait": critical_nonwait,
        "mean_kl": mean_kl,
    }


def _gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "overall": metrics["overall"]["fidelity"] is not None
                   and metrics["overall"]["fidelity"] >= MINIMUM_OVERALL_FIDELITY,
    }
    for group in GROUPS:
        item = metrics["critical"][group]
        checks[group] = (item["rows"] > 0 and item["fidelity"] is not None
                         and item["fidelity"] >= MINIMUM_CRITICAL_FIDELITY)
    return {"checks": checks, "passed": all(checks.values())}


def _candidate_config(depth: int, leaves: int) -> FeedbackConfig:
    return FeedbackConfig(
        warmup_steps=0, ramp_steps=0, lambda_max=.01,
        depths=(depth,), leaves=(leaves,), min_samples_leaf=4,
        minimum_fidelity=MINIMUM_OVERALL_FIDELITY,
        minimum_critical_fidelity=MINIMUM_CRITICAL_FIDELITY,
        minimum_training_rows=256, minimum_validation_rows=128,
    )


def _fit_candidates(actor: NumPyNativeActor, train: Mapping[str, np.ndarray],
                    validation: Mapping[str, np.ndarray], binding_sha256: str):
    candidates = []
    train_groups, validation_groups = _groups(train["group_bits"]), _groups(validation["group_bits"])
    actor_sha256 = actor.artifact_sha256
    for depth in DEPTHS:
        for leaves in LEAVES:
            manager = FeedbackManager(tuple(actor.metadata["feature_names"]),
                                      _candidate_config(depth, leaves))
            fit = manager.fit(
                train["observations"], train["probabilities"],
                validation["observations"], validation["probabilities"],
                step=int(actor.metadata.get("cumulative_joint_steps",
                         actor.metadata.get("joint_steps", 0))),
                source_actor_sha256=actor_sha256,
                train_episode_ids=list(map(str, train["episode_ids"])),
                val_episode_ids=list(map(str, validation["episode_ids"])),
                train_groups=train_groups, val_groups=validation_groups,
            )
            source = manager.program
            if source is None:
                raise RuntimeError("RCPD candidate fit did not produce a program")
            metadata = deepcopy(dict(source.metadata))
            metrics_metadata = deepcopy(metadata.get("metrics", {}))
            metrics_metadata.update({
                "feedback_eligible": False,
                "feedback_weight": 0.0,
                "explanation_eligible": False,
                "feedback_ineligibility_reasons": [
                    "final_actor_frozen_no_further_training"],
                "explanation_ineligibility_reasons": [
                    "independent_intervention_audit_not_run"],
            })
            metadata.update({
                "metrics": metrics_metadata,
                "program_roles": ["final_explanation_evidence_pending_audit"],
                "final_rcpd_version": VERSION,
                "final_rcpd_binding_sha256": binding_sha256,
                "native_source_actor_sha256": actor_sha256,
                "source_actor_parameters_sha256": actor.metadata.get("actor_parameters_sha256"),
                "candidate_depth_cap": depth,
                "candidate_leaf_cap": leaves,
                "ppo_joint_steps": 0,
                "optimizer_updates": 0,
                "program_feedback_into_actor": False,
                "runtime_controller": "native_neural_actor_only",
                "runtime_action_override": False,
                "explanation_qualified": False,
                "release_ready": False,
            })
            program = ExecutableProgram(tuple(ACTIONS), tuple(actor.metadata["feature_names"]),
                                        source.root, metadata)
            metrics = _metrics(program, validation)
            gate = _gate(metrics)
            complexity = program_complexity(
                program, max_depth=max(DEPTHS), max_leaf_count=max(LEAVES),
                max_predicate_count=max(LEAVES) - 1).to_dict()
            candidates.append({
                "depth_cap": depth,
                "leaf_cap": leaves,
                "metrics": metrics,
                "gate": gate,
                "complexity": complexity,
                "program_content_sha256": digest(program.to_dict()),
                "program": program.to_dict(),
            })
    return candidates


def _selection_key(candidate: Mapping[str, Any]):
    metrics = candidate["metrics"]
    return (candidate["complexity"]["loss"], metrics["mean_kl"],
            -metrics["overall"]["fidelity"], candidate["depth_cap"],
            candidate["leaf_cap"])


def _select(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate["gate"]["passed"]]
    return min(eligible, key=_selection_key) if eligible else None


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if (not path.is_file() or path.is_symlink()
            or path.resolve() != path.absolute()):
        raise ValueError("Final-RCPD evidence path is not a regular file")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Final-RCPD evidence archive exceeds its size bound")
    with zipfile.ZipFile(path) as archive:
        expected = {name + ".npy" for name in _ARRAY_FIELDS}
        infos = archive.infolist()
        if ({info.filename for info in infos} != expected
                or len(infos) != len(expected)
                or any(info.is_dir() or info.file_size < 0 or info.compress_size < 0
                       for info in infos)
                or sum(info.file_size for info in infos) > 96 * 1024 * 1024):
            raise ValueError("Final-RCPD evidence archive contents differ")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _ARRAY_FIELDS:
            raise ValueError("Final-RCPD evidence array set differs")
        return {name: archive[name].copy() for name in archive.files}


def _scenario_contract(scenarios: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    required = {"train": 512, "calibration": 100, "validation": 50,
                "extraction": 100, "explanation_test": 100,
                "final_test": 100, "play": 12}
    splits = scenarios.get("splits")
    split = splits.get("extraction") if isinstance(splits, dict) else None
    if (scenarios.get("version") != "warehouse-native-physical-splits-v1"
            or not isinstance(splits, dict) or set(splits) != set(required)
            or scenarios.get("counts") != required
            or any(not isinstance(rows, list) or len(rows) != required[name]
                   for name, rows in splits.items())
            or not isinstance(split, list) or len(split) != TRAIN_SCENES + VALIDATION_SCENES):
        raise ValueError("Exact registered 100-scene extraction split required")
    all_fingerprints = {scene.get("fingerprint") for rows in splits.values()
                        for scene in rows}
    total_rows = sum(len(rows) for rows in splits.values())
    if len(all_fingerprints) != total_rows or any(
            not isinstance(scene, dict)
            or type(scene.get("id")) is not str
            or _HEX.fullmatch(str(scene.get("fingerprint", ""))) is None
            or not isinstance(scene.get("snapshot"), dict)
            for rows in splits.values() for scene in rows):
        raise ValueError("Registered scenario fingerprints overlap or are malformed")
    return split[:TRAIN_SCENES], split[TRAIN_SCENES:]


def extract(*, actor_path: str | Path, scenarios_path: str | Path,
            output: str | Path) -> dict[str, Any]:
    actor_path = _regular_input(actor_path, "Final-RCPD Actor")
    scenarios_path = _regular_input(scenarios_path, "Final-RCPD scenarios")
    supplied_output = Path(output).expanduser().absolute()
    if (supplied_output.exists() or supplied_output.is_symlink()
            or supplied_output.resolve() != supplied_output
            or supplied_output.parent.is_symlink()
            or (supplied_output.parent.exists()
                and supplied_output.parent.resolve() != supplied_output.parent.absolute())):
        raise FileExistsError("Final RCPD extraction requires a new output directory")
    output = supplied_output
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    actor = NumPyNativeActor(actor_path)
    if (tuple(actor.metadata.get("actions", ())) != tuple(ACTIONS)
            or actor.metadata.get("action_masks") is not False
            or actor.metadata.get("runtime_action_override") is not False
            or tuple(actor.metadata.get("feature_names", ())) == ()
            or len(actor.metadata["feature_names"]) != actor.obs_dim
            or _HEX.fullmatch(str(actor.metadata.get("actor_parameters_sha256", ""))) is None):
        raise ValueError("Frozen Actor is incompatible with final RCPD extraction")
    scenarios = _read_json(scenarios_path)
    train_scenes, validation_scenes = _scenario_contract(scenarios)
    sources = producer_sources()
    initial = {
        "actor_file_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor.metadata.get("actor_parameters_sha256"),
        "scenario_file_sha256": file_hash(scenarios_path),
        "scenario_content_sha256": digest(scenarios),
        "train_fingerprints_sha256": digest([scene["fingerprint"] for scene in train_scenes]),
        "validation_fingerprints_sha256": digest([scene["fingerprint"] for scene in validation_scenes]),
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
    }
    _write_json(output / "inputs.json", {
        "version": VERSION, "test_fixture": False, "formal_ready": False,
        "contract": contract(), "bindings": initial, "sources": sources,
    })
    train = _collect_pool(actor, train_scenes, pool="train", seed=TRAIN_SEED)
    validation = _collect_pool(actor, validation_scenes, pool="validation",
                               seed=VALIDATION_SEED)
    _validate_pool(train, name="train", actor=actor, expected_scenes=train_scenes)
    _replay_pool(train, name="train", actor=actor, expected_scenes=train_scenes,
                 seed=TRAIN_SEED)
    _validate_pool(validation, name="validation", actor=actor,
                   expected_scenes=validation_scenes)
    _replay_pool(validation, name="validation", actor=actor,
                 expected_scenes=validation_scenes, seed=VALIDATION_SEED)
    train_observations = {sha256(row.astype("<f4", copy=False).tobytes()).hexdigest()
                          for row in train["observations"]}
    validation_observations = {sha256(row.astype("<f4", copy=False).tobytes()).hexdigest()
                               for row in validation["observations"]}
    if (set(map(str, train["episode_ids"])) & set(map(str, validation["episode_ids"]))
            or train_observations & validation_observations):
        raise ValueError("Final RCPD train/validation evidence overlaps")
    _write_npz(output / "train_rows.npz", train)
    _write_npz(output / "validation_rows.npz", validation)
    evidence_before_fit = {
        name: file_hash(output / name)
        for name in ("inputs.json", "train_rows.npz", "validation_rows.npz")
    }
    binding_sha256 = digest({"inputs": initial, "evidence": evidence_before_fit})
    candidates = _fit_candidates(actor, train, validation, binding_sha256)
    _write_json(output / "candidates.json", {
        "version": VERSION, "binding_sha256": binding_sha256,
        "candidates": candidates,
    })
    selected = _select(candidates)
    status = "passed" if selected is not None else "failed"
    if selected is not None:
        _write_json(output / "program.json", selected["program"])
    evidence = {name: file_hash(output / name) for name in EVIDENCE_FILES}
    report = {
        "version": VERSION, "status": status, "test_fixture": False,
        "formal_ready": False, "bindings": initial,
        "final_rcpd_binding_sha256": binding_sha256,
        "candidate_grid": {"depths": list(DEPTHS), "leaves": list(LEAVES)},
        "candidate_count": len(candidates),
        "train_rows": len(train["observations"]),
        "validation_rows": len(validation["observations"]),
        "train_episodes": len(set(map(str, train["episode_ids"]))),
        "validation_episodes": len(set(map(str, validation["episode_ids"]))),
        "episode_overlap": 0, "exact_observation_overlap": 0,
        "selected": None if selected is None else {
            key: deepcopy(selected[key]) for key in (
                "depth_cap", "leaf_cap", "metrics", "gate", "complexity",
                "program_content_sha256")
        },
        "program_file_sha256": None if selected is None else file_hash(output / "program.json"),
        "evidence_artifacts": evidence,
        "execution": {"evidence_environment_steps": len(train["observations"])
                      + len(validation["observations"]),
                      "complete_episodes": len(set(map(str, train["episode_ids"])))
                      + len(set(map(str, validation["episode_ids"]))),
                      "tree_fits": len(candidates), "ppo_joint_steps": 0,
                      "optimizer_updates": 0, "actor_changed": False,
                      "program_feedback_into_actor": False,
                      "runtime_action_overrides": 0},
    }
    if (file_hash(actor_path) != initial["actor_file_sha256"]
            or file_hash(scenarios_path) != initial["scenario_file_sha256"]
            or producer_sources() != sources):
        raise RuntimeError("Frozen final-RCPD inputs changed during extraction")
    _write_json(output / "report.json", report)
    # Re-read everything before claiming success.  A failed fit remains a
    # complete diagnostic output but cannot supply a runtime program.
    read_saved_report(output, expected_report_sha256=file_hash(output / "report.json"),
                      actor_path=actor_path, scenarios_path=scenarios_path,
                      program_path=(output / "program.json") if selected else None,
                      require_passed=False)
    return report


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      actor_path: str | Path, scenarios_path: str | Path,
                      program_path: str | Path | None,
                      require_passed: bool = True) -> dict[str, Any]:
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Final-RCPD evidence directory is missing, linked, or noncanonical")
    report_path = output / "report.json"
    actor_path = _regular_input(actor_path, "Final-RCPD Actor")
    scenarios_path = _regular_input(scenarios_path, "Final-RCPD scenarios")
    if (_HEX.fullmatch(str(expected_report_sha256)) is None
            or not report_path.is_file() or report_path.is_symlink()
            or file_hash(report_path) != expected_report_sha256):
        raise ValueError("Final-RCPD report hash differs")
    report = _read_json(report_path)
    scenarios = _read_json(scenarios_path)
    train_scenes, validation_scenes = _scenario_contract(scenarios)
    actor = NumPyNativeActor(actor_path)
    bindings = {
        "actor_file_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor.metadata.get("actor_parameters_sha256"),
        "scenario_file_sha256": file_hash(scenarios_path),
        "scenario_content_sha256": digest(scenarios),
        "train_fingerprints_sha256": digest([scene["fingerprint"] for scene in train_scenes]),
        "validation_fingerprints_sha256": digest([scene["fingerprint"] for scene in validation_scenes]),
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(producer_sources()),
    }
    inputs = _read_json(output / "inputs.json")
    if inputs != {"version": VERSION, "test_fixture": False, "formal_ready": False,
                  "contract": contract(), "bindings": bindings,
                  "sources": producer_sources()}:
        raise ValueError("Final-RCPD input provenance differs")
    artifacts = report.get("evidence_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(EVIDENCE_FILES):
        raise ValueError("Final-RCPD evidence artifact set differs")
    for name, expected in artifacts.items():
        path = output / name
        if (path.parent != output or not path.is_file() or path.is_symlink()
                or _HEX.fullmatch(str(expected)) is None or file_hash(path) != expected):
            raise ValueError("Final-RCPD evidence artifact changed")
    train, validation = _load_npz(output / "train_rows.npz"), _load_npz(
        output / "validation_rows.npz")
    _validate_pool(train, name="train", actor=actor, expected_scenes=train_scenes)
    _replay_pool(train, name="train", actor=actor, expected_scenes=train_scenes,
                 seed=TRAIN_SEED)
    _validate_pool(validation, name="validation", actor=actor,
                   expected_scenes=validation_scenes)
    _replay_pool(validation, name="validation", actor=actor,
                 expected_scenes=validation_scenes, seed=VALIDATION_SEED)
    train_obs = {sha256(row.astype("<f4", copy=False).tobytes()).digest()
                 for row in train["observations"]}
    validation_obs = {sha256(row.astype("<f4", copy=False).tobytes()).digest()
                      for row in validation["observations"]}
    if (set(map(str, train["episode_ids"])) & set(map(str, validation["episode_ids"]))
            or train_obs & validation_obs):
        raise ValueError("Final-RCPD saved evidence overlaps")
    evidence_before_fit = {name: artifacts[name] for name in
                           ("inputs.json", "train_rows.npz", "validation_rows.npz")}
    binding_sha256 = digest({"inputs": bindings, "evidence": evidence_before_fit})
    stored_candidates = _read_json(output / "candidates.json")
    if (not isinstance(stored_candidates, dict)
            or stored_candidates.get("version") != VERSION
            or stored_candidates.get("binding_sha256") != binding_sha256
            or not isinstance(stored_candidates.get("candidates"), list)):
        raise ValueError("Final-RCPD candidate evidence differs")
    candidates = stored_candidates["candidates"]
    expected_grid = [(depth, leaves) for depth in DEPTHS for leaves in LEAVES]
    if (len(candidates) != len(expected_grid)
            or [(item.get("depth_cap"), item.get("leaf_cap")) for item in candidates]
               != expected_grid):
        raise ValueError("Final-RCPD candidate grid differs")
    checked = []
    for item in candidates:
        if not isinstance(item, dict) or set(item) != _CANDIDATE_FIELDS:
            raise ValueError("Final-RCPD candidate schema differs")
        program = ExecutableProgram.from_dict(item.get("program", {}))
        if (tuple(program.action_names) != tuple(ACTIONS)
                or tuple(program.feature_names) != tuple(actor.metadata["feature_names"])
                or program.metadata.get("final_rcpd_version") != VERSION
                or program.metadata.get("final_rcpd_binding_sha256") != binding_sha256
                or program.metadata.get("native_source_actor_sha256") != actor.artifact_sha256
                or program.metadata.get("program_feedback_into_actor") is not False
                or program.metadata.get("ppo_joint_steps") != 0
                or program.metadata.get("optimizer_updates") != 0
                or program.metadata.get("metrics", {}).get("feedback_eligible") is not False
                or program.metadata.get("metrics", {}).get("feedback_weight") != 0.0
                or program.metadata.get("runtime_action_override") is not False
                or program.root.depth() > item["depth_cap"]
                or program.root.leaf_count() > item["leaf_cap"]):
            raise ValueError("Final-RCPD candidate program binding differs")
        metrics = _metrics(program, validation)
        gate = _gate(metrics)
        complexity = program_complexity(
            program, max_depth=max(DEPTHS), max_leaf_count=max(LEAVES),
            max_predicate_count=max(LEAVES) - 1).to_dict()
        expected = {
            "depth_cap": item["depth_cap"], "leaf_cap": item["leaf_cap"],
            "metrics": metrics, "gate": gate, "complexity": complexity,
            "program_content_sha256": digest(program.to_dict()),
            "program": item["program"],
        }
        if digest(expected) != digest(item):
            raise ValueError("Final-RCPD candidate metrics differ from saved evidence")
        checked.append(expected)
    selected = _select(checked)
    selected_summary = None if selected is None else {
        key: deepcopy(selected[key]) for key in (
            "depth_cap", "leaf_cap", "metrics", "gate", "complexity",
            "program_content_sha256")
    }
    status = "passed" if selected is not None else "failed"
    program_sha256 = None
    if selected is not None:
        if program_path is None:
            raise ValueError("Passing final-RCPD report requires its selected program")
        supplied_program_path = Path(program_path)
        if supplied_program_path.is_symlink():
            raise ValueError("Final-RCPD selected program cannot be a symlink")
        program_path = supplied_program_path.resolve()
        if (program_path != output / "program.json" or not program_path.is_file()
                or _read_json(program_path) != selected["program"]):
            raise ValueError("Final-RCPD selected program differs")
        program_sha256 = file_hash(program_path)
    expected_execution = {
        "evidence_environment_steps": len(train["observations"]) + len(validation["observations"]),
        "complete_episodes": len(set(map(str, train["episode_ids"])))
                           + len(set(map(str, validation["episode_ids"]))),
        "tree_fits": len(checked), "ppo_joint_steps": 0, "optimizer_updates": 0,
        "actor_changed": False, "program_feedback_into_actor": False,
        "runtime_action_overrides": 0,
    }
    expected_report = {
        "version": VERSION, "status": status, "test_fixture": False,
        "formal_ready": False, "bindings": bindings,
        "final_rcpd_binding_sha256": binding_sha256,
        "candidate_grid": {"depths": list(DEPTHS), "leaves": list(LEAVES)},
        "candidate_count": len(checked), "train_rows": len(train["observations"]),
        "validation_rows": len(validation["observations"]),
        "train_episodes": len(set(map(str, train["episode_ids"]))),
        "validation_episodes": len(set(map(str, validation["episode_ids"]))),
        "episode_overlap": 0, "exact_observation_overlap": 0,
        "selected": selected_summary, "program_file_sha256": program_sha256,
        "evidence_artifacts": artifacts, "execution": expected_execution,
    }
    if digest(report) != digest(expected_report):
        raise ValueError("Final-RCPD report differs from immutable evidence")
    if require_passed and status != "passed":
        raise ValueError("Final-RCPD extraction did not meet the registered fidelity gates")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = extract(actor_path=args.actor, scenarios_path=args.scenarios,
                     output=args.output)
    print(canonical({
        "status": report["status"], "selected": report["selected"],
        "candidate_count": report["candidate_count"],
        "ppo_joint_steps": 0, "program_feedback_into_actor": False,
        "report_sha256": file_hash(Path(args.output) / "report.json"),
    }))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEPTHS", "LEAVES", "GROUPS", "MINIMUM_CRITICAL_FIDELITY",
    "MINIMUM_OVERALL_FIDELITY", "PARTNERS", "TRAIN_SEED", "VALIDATION_SEED",
    "VERSION", "contract", "extract", "producer_sources", "read_saved_report",
]
