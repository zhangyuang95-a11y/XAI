"""One frozen explanation audit for the r4.1 diagnostic v8 public program.

The audit records the *raw 197-value public observations* supplied to the
explicit v8 tree program.  It stores them in a compressed, pickle-free NPZ,
along with Actor actions, program actions, structured-trace hashes, physical
branch hashes, and action-authority evidence.  The strict reader reloads the
exact program bytes and independently recomputes every observation hash,
prediction, trace hash, metric, and gate from those raw observations.

``replay_saved_audit`` is a separate physical replay: it reconstructs every
episode and intervention from the frozen scene registry and Actor, then
compares the complete evidence arrays.  Neither the reader nor replay fits,
selects, or changes a program or an Actor action.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout_api
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    R41DiagnosticPublicTreeProgramV8,
    VERSION as PROGRAM_VERSION,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
    diagnostic_runtime_sources,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action


VERSION = "warehouse-r41-diagnostic-explanation-audit.v8"
FINAL_ONCE_VERSION = "warehouse-r41-diagnostic-final-once.v8"
RCPD_VERSION = "warehouse-r41-diagnostic-rcpd.v8"
FRESH_HOLDOUT_VERSION = holdout_api.VERSION
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
EXPLANATION_SCENES = 64
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
_GROUP_INDEX = {name: index for index, name in enumerate(GROUPS)}
LEDGER_ENV = "WAREHOUSE_R41_FINAL_LEDGER_DIR"
DEFAULT_LEDGER_ROOT = (
    Path.home() / ".local/state/policylens/warehouse_r41_diagnostic_v8_final_once"
)

AGENT_PHYSICS = (
    "agent_id", "position", "battery", "active", "carrying_task_id",
    "deliveries_completed", "last_battery_delta", "steps_since_charging",
    "charger_wait_streak",
)
TASK_PHYSICS = (
    "task_id", "pickup_position", "delivery_position", "status",
    "carrier_agent_id", "created_frame", "claimed_frame", "delivered_frame",
)
JOINT_PHYSICS = (
    "next_task_index", "total_deliveries", "terminated", "truncated",
    "terminal_reason",
)

ARRAY_KEYS = frozenset((
    "ordinary_observations", "ordinary_observation_hashes",
    "ordinary_trace_hashes", "ordinary_scene_fingerprints",
    "ordinary_scene_indexes", "ordinary_partner_indexes", "ordinary_frames",
    "ordinary_group_bits", "ordinary_player_actions", "ordinary_actor_actions",
    "ordinary_submitted_actor_actions", "ordinary_executed_actor_actions",
    "ordinary_program_actions", "ordinary_after_physical_hashes",
    "ordinary_done", "ordinary_source_unchanged", "ordinary_zero_overrides",
    "pair_scene_fingerprints", "pair_scene_indexes", "pair_partner_indexes",
    "pair_frames", "pair_group_bits", "pair_player_actions", "pair_active",
    "pair_physical_effect", "pair_actor_changed", "pair_correct",
    "pair_wait_valid", "pair_changed_valid", "pair_wait_observations",
    "pair_changed_observations", "pair_wait_observation_hashes",
    "pair_changed_observation_hashes", "pair_wait_trace_hashes",
    "pair_changed_trace_hashes", "pair_wait_actor_actions",
    "pair_changed_actor_actions", "pair_wait_program_actions",
    "pair_changed_program_actions", "pair_wait_physical_hashes",
    "pair_changed_physical_hashes", "pair_wait_submitted_equal",
    "pair_changed_submitted_equal", "pair_source_unchanged",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "rcpd_version": RCPD_VERSION,
        "program_version": PROGRAM_VERSION,
        "fresh_holdout_version": FRESH_HOLDOUT_VERSION,
        "split": "fresh_final_test",
        "scenes": EXPLANATION_SCENES,
        "partners": list(PARTNERS),
        "horizon": 120,
        "evaluated_role": "robot_2",
        "raw_public_observations_persisted": True,
        "pickle_allowed": False,
        "reader_recomputes_predictions": True,
        "reader_recomputes_structured_trace_hashes": True,
        "independent_physical_replay_supported": True,
        "ordinary_fidelity_min": 0.90,
        "ordinary_nonwait_min": 0.90,
        "critical_fidelity_min": 0.85,
        "critical_minimum_scenes": 10,
        "effective_direction_min": 0.85,
        "effective_direction_minimum_scenes": 10,
        "intervention_anchor": "pre-action frame divisible by 10 with a public critical group",
        "intervention_population": "different physical projection and different next Actor action versus WAIT",
        "intervention_correct": "program agrees with Actor at both nonterminal endpoints",
        "tree_controls_runtime": False,
        "runtime_action_override": False,
        "test_fixture_allowed": False,
        "exact_development_observation_overlap_allowed": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(holdout_api.__file__).resolve(),
        ROOT / "backend/warehouse_r41_diagnostic_public_tree_program_v8.py",
        ROOT / "backend/warehouse_r41_diagnostic_public_features_v8.py",
        ROOT / "backend/warehouse_r41_diagnostic_boosted_tree.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/training/warehouse_native_evaluation.py",
        ROOT / "env/warehouse_native/partners.py",
    )
    result = {}
    for path in paths:
        path = path.resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError("Explanation-audit source closure is unsafe")
        result[path.relative_to(ROOT.resolve()).as_posix()] = file_hash(path)
    for name, value in diagnostic_runtime_sources().items():
        result["runtime/" + name] = value
    return dict(sorted(result.items()))


def _regular(value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > maximum):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read_json(value: str | Path, label: str) -> dict[str, Any]:
    path = _regular(value, label)

    def pairs(rows):
        result = {}
        for key, item in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = item
        return result

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)
        ),
    )
    if not isinstance(result, dict):
        raise ValueError(label + " must be one JSON object")
    return result


def _write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Explanation-audit output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _ledger_root() -> Path:
    raw = os.environ.get(LEDGER_ENV)
    return Path(raw).expanduser().absolute() if raw else DEFAULT_LEDGER_ROOT.absolute()


def _claim_receipt(
    claim_receipt_path: str | Path, *, expected_claim_sha256: str,
    expected_campaign_key: str, expected_candidate_identity_sha256: str,
    expected_holdout_completion_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    if any(_HEX.fullmatch(str(value)) is None for value in (
            expected_claim_sha256, expected_campaign_key,
            expected_candidate_identity_sha256,
            expected_holdout_completion_sha256)):
        raise ValueError("Final claim identity is malformed")
    claim_dir, receipt = holdout_api._claim_receipt(
        claim_receipt_path, expected_claim_sha256=expected_claim_sha256,
        expected_campaign_key=expected_campaign_key,
        expected_candidate_identity_sha256=expected_candidate_identity_sha256,
    )
    holdout_completed = _regular(
        claim_dir / "holdout_completed.json", "holdout completion phase")
    marker = _read_json(holdout_completed, "holdout completion phase")
    if (file_hash(holdout_completed) != expected_holdout_completion_sha256
            or marker.get("version") != FRESH_HOLDOUT_VERSION
            or marker.get("status") != "completed_program_blind"
            or marker.get("campaign_key") != expected_campaign_key
            or marker.get("candidate_identity_sha256")
                != expected_candidate_identity_sha256
            or marker.get("attempt_started_sha256") != expected_claim_sha256):
        raise ValueError("Final claim/holdout phase receipt differs")
    return claim_dir, receipt


def _claim_marker(directory: Path, name: str, value: Mapping[str, Any]) -> Path:
    path = directory / name
    _write_json(path, dict(value))
    return path


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or set(arrays) != ARRAY_KEYS):
        raise ValueError("Explanation-audit NPZ destination/schema differs")
    temporary = path.parent / ("." + path.name + ".partial")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() or path.is_symlink():
            raise ValueError("Explanation-audit NPZ destination appeared")
        os.rename(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = _regular(path, "raw explanation evidence", maximum=MAX_NPZ_BYTES)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != ARRAY_KEYS:
            raise ValueError("Raw explanation evidence array set differs")
        arrays = {key: np.array(saved[key], copy=True) for key in saved.files}
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise ValueError("Object arrays are forbidden in final evidence")
    return arrays


def _load_program(path: Path) -> tuple[R41DiagnosticPublicTreeProgramV8, dict[str, Any]]:
    payload = _read_json(path, "v8 public-tree program")
    if payload.get("version") != PROGRAM_VERSION:
        raise ValueError("Exact v8 public-tree program required")
    program = R41DiagnosticPublicTreeProgramV8.from_dict(payload)
    if (program.to_dict() != payload or tuple(program.action_names) != tuple(ACTIONS)
            or tuple(program.base_program.classes) != tuple(range(len(ACTIONS)))
            or program.metadata.get("diagnostic_rcpd_version") != RCPD_VERSION
            or program.metadata.get("actions") != list(ACTIONS)
            or program.metadata.get("classes") != list(range(len(ACTIONS)))
            or program.metadata.get("runtime_action_override") is not False):
        raise ValueError("V8 public-tree program round trip/action registry differs")
    return program, payload


def _observation_hash(observation: Any) -> str:
    value = np.asarray(observation, dtype="<f4")
    if value.shape != (197,) or not np.isfinite(value).all():
        raise ValueError("Final-audit public observation differs")
    return sha256(value.tobytes(order="C")).hexdigest()


def _trace_hash(program: R41DiagnosticPublicTreeProgramV8,
                feature_names: Sequence[str], observation: np.ndarray) -> str:
    features = dict(zip(feature_names, map(float, observation)))
    trace = program.trace(features)
    if (trace.get("prediction") not in ACTIONS
            or trace.get("prediction_index") != _ACTION_INDEX[trace["prediction"]]):
        raise ValueError("Structured public-tree trace differs")
    return digest(trace)


def _group_bits(groups: Sequence[str]) -> int:
    if len(groups) != len(set(groups)) or not set(groups).issubset(GROUPS):
        raise ValueError("Critical group registry differs")
    return sum(1 << _GROUP_INDEX[group] for group in groups)


def _physical(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    state = snapshot["state"]
    return {
        "agents": [{key: agent[key] for key in AGENT_PHYSICS}
                   for agent in state["agents"]],
        "tasks": [{key: task[key] for key in TASK_PHYSICS}
                  for task in state["tasks"]],
        "completed_tasks": [{key: task[key] for key in TASK_PHYSICS}
                            for task in state["completed_tasks"]],
        **{key: state[key] for key in JOINT_PHYSICS},
    }


def _branch(runtime: R41DiagnosticOnlineAlignmentRuntime,
            program: R41DiagnosticPublicTreeProgramV8,
            feature_names: Sequence[str], snapshot: Mapping[str, Any],
            player_action: str) -> dict[str, Any]:
    env = runtime.from_snapshot(snapshot)
    transition = runtime.step(env, player_action)
    valid = not transition["done"]
    observation = np.zeros(197, dtype=np.float32)
    observation_hash = trace_hash = ""
    actor_action = program_action = -1
    if valid:
        observation = np.asarray(env.observations()["robot_2"], dtype=np.float32)
        observation_hash = _observation_hash(observation)
        actor_name = runtime.decision(env)[0]["robot_2"]
        actor_action = _ACTION_INDEX[actor_name]
        program_action = int(program.predict_batch(observation[None, :])[0])
        trace_hash = _trace_hash(program, feature_names, observation)
    return {
        "valid": valid,
        "observation": observation,
        "observation_hash": observation_hash,
        "trace_hash": trace_hash,
        "actor_action": actor_action,
        "program_action": program_action,
        "physical_hash": digest(_physical(transition["after"])),
        "submitted_equal": (
            transition["submitted_actions"]["robot_2"]
            == transition["policy_actions"]["robot_2"]
            and transition["decision"]["post_policy_overrides"] == 0
        ),
    }


def _as_arrays(ordinary: Sequence[Mapping[str, Any]],
               pairs: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    def a(prefix: str, key: str, dtype: Any) -> np.ndarray:
        source = ordinary if prefix == "ordinary" else pairs
        return np.asarray([row[key] for row in source], dtype=dtype)

    return {
        "ordinary_observations": np.asarray(
            [row["observation"] for row in ordinary], dtype=np.float32),
        "ordinary_observation_hashes": a("ordinary", "observation_hash", "S64"),
        "ordinary_trace_hashes": a("ordinary", "trace_hash", "S64"),
        "ordinary_scene_fingerprints": a("ordinary", "fingerprint", "S64"),
        "ordinary_scene_indexes": a("ordinary", "scene_index", np.uint16),
        "ordinary_partner_indexes": a("ordinary", "partner_index", np.uint8),
        "ordinary_frames": a("ordinary", "frame", np.uint16),
        "ordinary_group_bits": a("ordinary", "group_bits", np.uint8),
        "ordinary_player_actions": a("ordinary", "player_action", np.uint8),
        "ordinary_actor_actions": a("ordinary", "actor_action", np.uint8),
        "ordinary_submitted_actor_actions": a(
            "ordinary", "submitted_actor_action", np.uint8),
        "ordinary_executed_actor_actions": a(
            "ordinary", "executed_actor_action", np.uint8),
        "ordinary_program_actions": a("ordinary", "program_action", np.uint8),
        "ordinary_after_physical_hashes": a(
            "ordinary", "after_physical_hash", "S64"),
        "ordinary_done": a("ordinary", "done", np.bool_),
        "ordinary_source_unchanged": a(
            "ordinary", "source_unchanged", np.bool_),
        "ordinary_zero_overrides": a(
            "ordinary", "zero_overrides", np.bool_),
        "pair_scene_fingerprints": a("pair", "fingerprint", "S64"),
        "pair_scene_indexes": a("pair", "scene_index", np.uint16),
        "pair_partner_indexes": a("pair", "partner_index", np.uint8),
        "pair_frames": a("pair", "frame", np.uint16),
        "pair_group_bits": a("pair", "group_bits", np.uint8),
        "pair_player_actions": a("pair", "player_action", np.uint8),
        "pair_active": a("pair", "active", np.bool_),
        "pair_physical_effect": a("pair", "physical_effect", np.bool_),
        "pair_actor_changed": a("pair", "actor_changed", np.bool_),
        "pair_correct": a("pair", "correct", np.bool_),
        "pair_wait_valid": a("pair", "wait_valid", np.bool_),
        "pair_changed_valid": a("pair", "changed_valid", np.bool_),
        "pair_wait_observations": np.asarray(
            [row["wait_observation"] for row in pairs], dtype=np.float32
        ).reshape((-1, 197)),
        "pair_changed_observations": np.asarray(
            [row["changed_observation"] for row in pairs], dtype=np.float32
        ).reshape((-1, 197)),
        "pair_wait_observation_hashes": a(
            "pair", "wait_observation_hash", "S64"),
        "pair_changed_observation_hashes": a(
            "pair", "changed_observation_hash", "S64"),
        "pair_wait_trace_hashes": a("pair", "wait_trace_hash", "S64"),
        "pair_changed_trace_hashes": a("pair", "changed_trace_hash", "S64"),
        "pair_wait_actor_actions": a("pair", "wait_actor_action", np.int8),
        "pair_changed_actor_actions": a(
            "pair", "changed_actor_action", np.int8),
        "pair_wait_program_actions": a("pair", "wait_program_action", np.int8),
        "pair_changed_program_actions": a(
            "pair", "changed_program_action", np.int8),
        "pair_wait_physical_hashes": a("pair", "wait_physical_hash", "S64"),
        "pair_changed_physical_hashes": a(
            "pair", "changed_physical_hash", "S64"),
        "pair_wait_submitted_equal": a(
            "pair", "wait_submitted_equal", np.bool_),
        "pair_changed_submitted_equal": a(
            "pair", "changed_submitted_equal", np.bool_),
        "pair_source_unchanged": a("pair", "source_unchanged", np.bool_),
    }


def _collect_evidence(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    program: R41DiagnosticPublicTreeProgramV8,
    scenes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    feature_names = tuple(runtime.environment(scenes[0]).feature_names)
    if tuple(program.base_feature_names) != feature_names:
        raise ValueError("Program raw public feature registry differs from runtime")
    ordinary: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    branch_steps = 0
    for partner_index, partner in enumerate(PARTNERS):
        for scene_index, scene in enumerate(scenes):
            env = runtime.environment(scene)
            rng = np.random.default_rng(17000 + partner_index * 1000 + scene_index)
            while not env.done:
                source = env.snapshot(); source_hash = digest(source)
                observation = np.asarray(
                    env.observations()["robot_2"], dtype=np.float32)
                observation_hash = _observation_hash(observation)
                actions, decision = runtime.decision(env)
                decision_source_unchanged = digest(env.snapshot()) == source_hash
                actor_action = _ACTION_INDEX[actions["robot_2"]]
                program_action = int(program.predict_batch(observation[None, :])[0])
                trace_hash = _trace_hash(program, feature_names, observation)
                groups = tuple(critical_groups(env, "robot_2"))
                group_bits = _group_bits(groups)
                if env.state.frame % 10 == 0 and groups:
                    wait = _branch(runtime, program, feature_names, source, "WAIT")
                    branch_steps += 1
                    for player_name in ACTIONS[:-1]:
                        changed = _branch(
                            runtime, program, feature_names, source, player_name)
                        branch_steps += 1
                        active = wait["valid"] and changed["valid"]
                        physical_effect = bool(
                            active and wait["physical_hash"] != changed["physical_hash"])
                        actor_changed = bool(
                            active and wait["actor_action"] != changed["actor_action"])
                        correct = bool(
                            active
                            and wait["actor_action"] == wait["program_action"]
                            and changed["actor_action"] == changed["program_action"]
                        )
                        pairs.append({
                            "fingerprint": scene["fingerprint"],
                            "scene_index": scene_index,
                            "partner_index": partner_index,
                            "frame": env.state.frame,
                            "group_bits": group_bits,
                            "player_action": _ACTION_INDEX[player_name],
                            "active": active,
                            "physical_effect": physical_effect,
                            "actor_changed": actor_changed,
                            "correct": correct,
                            **{"wait_" + key: value for key, value in wait.items()},
                            **{"changed_" + key: value for key, value in changed.items()},
                            "source_unchanged": digest(env.snapshot()) == source_hash,
                        })
                player_name = partner_action(env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_hash:
                    raise RuntimeError("Final-audit partner or branch mutated source state")
                transition = runtime.step(env, player_name)
                ordinary.append({
                    "observation": observation,
                    "observation_hash": observation_hash,
                    "trace_hash": trace_hash,
                    "fingerprint": scene["fingerprint"],
                    "scene_index": scene_index,
                    "partner_index": partner_index,
                    "frame": source["state"]["frame"],
                    "group_bits": group_bits,
                    "player_action": _ACTION_INDEX[player_name],
                    "actor_action": actor_action,
                    "submitted_actor_action": _ACTION_INDEX[
                        transition["submitted_actions"]["robot_2"]],
                    "executed_actor_action": _ACTION_INDEX[
                        transition["executed_actions"]["robot_2"]],
                    "program_action": program_action,
                    "after_physical_hash": digest(_physical(transition["after"])),
                    "done": transition["done"],
                    "source_unchanged": decision_source_unchanged,
                    "zero_overrides": (
                        decision["post_policy_overrides"] == 0
                        and transition["decision"]["post_policy_overrides"] == 0
                    ),
                })
    return _as_arrays(ordinary, pairs), {
        "base_steps": len(ordinary),
        "counterfactual_steps": branch_steps,
        "episodes": len(scenes) * len(PARTNERS),
        "neural_updates": 0,
        "tree_fits": 0,
    }


def _decode(values: np.ndarray, label: str, *, allow_empty: bool = False) -> np.ndarray:
    if values.dtype.kind != "S":
        raise ValueError(label + " must be fixed ASCII bytes")
    decoded = np.char.decode(values, "ascii")
    for value in decoded:
        text = str(value)
        if not (allow_empty and text == "") and _HEX.fullmatch(text) is None:
            raise ValueError(label + " contains a malformed SHA-256")
    return decoded


def _stat(mask: np.ndarray, correct: np.ndarray,
          fingerprints: np.ndarray) -> dict[str, Any]:
    return {
        "rows": int(mask.sum()),
        "scenes": len(set(map(str, fingerprints[mask]))),
        "fidelity": float(correct[mask].mean()) if np.any(mask) else 0.0,
    }


def _validate_shapes(arrays: Mapping[str, np.ndarray]) -> tuple[int, int]:
    if set(arrays) != ARRAY_KEYS:
        raise ValueError("Raw explanation evidence array set differs")
    ordinary_n = len(arrays["ordinary_observations"])
    pair_n = len(arrays["pair_wait_observations"])
    for key, array in arrays.items():
        expected = ordinary_n if key.startswith("ordinary_") else pair_n
        if len(array) != expected:
            raise ValueError("Raw explanation evidence row counts differ")
        if array.dtype.kind == "O":
            raise ValueError("Object arrays are forbidden in final evidence")
    if (arrays["ordinary_observations"].shape != (ordinary_n, 197)
            or arrays["pair_wait_observations"].shape != (pair_n, 197)
            or arrays["pair_changed_observations"].shape != (pair_n, 197)
            or arrays["ordinary_observations"].dtype != np.dtype("float32")
            or arrays["pair_wait_observations"].dtype != np.dtype("float32")
            or arrays["pair_changed_observations"].dtype != np.dtype("float32")
            or not np.isfinite(arrays["ordinary_observations"]).all()
            or not np.isfinite(arrays["pair_wait_observations"]).all()
            or not np.isfinite(arrays["pair_changed_observations"]).all()):
        raise ValueError("Raw public observation matrix differs")
    return ordinary_n, pair_n


def _recompute_program_evidence(
    arrays: Mapping[str, np.ndarray],
    program: R41DiagnosticPublicTreeProgramV8,
    feature_names: Sequence[str],
) -> None:
    observations = arrays["ordinary_observations"]
    claimed_hashes = _decode(
        arrays["ordinary_observation_hashes"], "ordinary observation hashes")
    claimed_traces = _decode(
        arrays["ordinary_trace_hashes"], "ordinary trace hashes")
    predicted = program.predict_batch(observations)
    if not np.array_equal(predicted, arrays["ordinary_program_actions"]):
        raise ValueError("Stored ordinary program predictions differ")
    for index, observation in enumerate(observations):
        if (_observation_hash(observation) != str(claimed_hashes[index])
                or _trace_hash(program, feature_names, observation)
                   != str(claimed_traces[index])):
            raise ValueError("Stored ordinary raw observation/trace differs")

    for prefix in ("wait", "changed"):
        valid = arrays[f"pair_{prefix}_valid"]
        endpoint = arrays[f"pair_{prefix}_observations"]
        hashes = _decode(arrays[f"pair_{prefix}_observation_hashes"],
                         prefix + " observation hashes", allow_empty=True)
        traces = _decode(arrays[f"pair_{prefix}_trace_hashes"],
                         prefix + " trace hashes", allow_empty=True)
        stored_predictions = arrays[f"pair_{prefix}_program_actions"]
        if (np.any(stored_predictions[~valid] != -1)
                or np.any(arrays[f"pair_{prefix}_actor_actions"][~valid] != -1)
                or np.any(hashes[~valid] != "") or np.any(traces[~valid] != "")
                or np.any(endpoint[~valid] != 0)):
            raise ValueError("Terminal intervention endpoint encoding differs")
        if np.any(valid):
            predicted = program.predict_batch(endpoint[valid])
            if not np.array_equal(predicted, stored_predictions[valid]):
                raise ValueError("Stored intervention program predictions differ")
        for index in np.flatnonzero(valid):
            observation = endpoint[index]
            if (_observation_hash(observation) != str(hashes[index])
                    or _trace_hash(program, feature_names, observation)
                       != str(traces[index])):
                raise ValueError("Stored intervention raw observation/trace differs")


def _metrics(arrays: Mapping[str, np.ndarray], *, development_hashes: set[str]) -> dict[str, Any]:
    ordinary_n, pair_n = _validate_shapes(arrays)
    fingerprints = _decode(
        arrays["ordinary_scene_fingerprints"], "ordinary fingerprints")
    pair_fingerprints = _decode(
        arrays["pair_scene_fingerprints"], "pair fingerprints")
    if (set(fingerprints) != set(pair_fingerprints)
            or len(set(fingerprints)) != EXPLANATION_SCENES):
        raise ValueError("Final-audit scene coverage differs")
    scene_identity: dict[int, str] = {}
    for index, fingerprint in zip(
            arrays["ordinary_scene_indexes"], fingerprints):
        previous = scene_identity.setdefault(int(index), str(fingerprint))
        if previous != str(fingerprint):
            raise ValueError("Final-audit scene index/fingerprint mapping differs")
    if set(scene_identity) != set(range(EXPLANATION_SCENES)):
        raise ValueError("Final-audit scene indexes differ")
    for index, fingerprint in zip(
            arrays["pair_scene_indexes"], pair_fingerprints):
        if scene_identity.get(int(index)) != str(fingerprint):
            raise ValueError("Intervention scene index/fingerprint mapping differs")
    actions = arrays["ordinary_actor_actions"]
    program_actions = arrays["ordinary_program_actions"]
    if (np.any(actions >= len(ACTIONS)) or np.any(program_actions >= len(ACTIONS))
            or np.any(arrays["ordinary_player_actions"] >= len(ACTIONS))
            or np.any(arrays["ordinary_submitted_actor_actions"] >= len(ACTIONS))
            or np.any(arrays["ordinary_executed_actor_actions"] >= len(ACTIONS))
            or np.any(arrays["ordinary_partner_indexes"] >= len(PARTNERS))
            or np.any(arrays["pair_partner_indexes"] >= len(PARTNERS))
            or np.any(arrays["pair_player_actions"] >= len(ACTIONS) - 1)):
        raise ValueError("Final-audit action/partner encoding differs")
    for prefix in ("wait", "changed"):
        endpoint_actions = arrays[f"pair_{prefix}_actor_actions"]
        endpoint_program = arrays[f"pair_{prefix}_program_actions"]
        valid = arrays[f"pair_{prefix}_valid"]
        if (np.any(endpoint_actions[valid] < 0)
                or np.any(endpoint_actions[valid] >= len(ACTIONS))
                or np.any(endpoint_program[valid] < 0)
                or np.any(endpoint_program[valid] >= len(ACTIONS))):
            raise ValueError("Final-audit intervention action encoding differs")
    if np.any(arrays["ordinary_group_bits"] >= (1 << len(GROUPS))):
        raise ValueError("Final-audit ordinary critical-group bits differ")
    if np.any(arrays["pair_group_bits"] >= (1 << len(GROUPS))):
        raise ValueError("Final-audit intervention critical-group bits differ")
    _decode(arrays["ordinary_after_physical_hashes"], "ordinary physical hashes")
    _decode(arrays["pair_wait_physical_hashes"], "wait physical hashes")
    _decode(arrays["pair_changed_physical_hashes"], "changed physical hashes")
    correct = actions == program_actions
    all_rows = np.ones(ordinary_n, dtype=bool)
    nonwait = actions != _ACTION_INDEX["WAIT"]
    fidelity = {
        "overall": _stat(all_rows, correct, fingerprints),
        "nonwait": _stat(nonwait, correct, fingerprints),
        "by_group": {},
    }
    for group, index in _GROUP_INDEX.items():
        mask = (arrays["ordinary_group_bits"] & (1 << index)) != 0
        fidelity["by_group"][group] = _stat(mask, correct, fingerprints)

    wait_valid = arrays["pair_wait_valid"]
    changed_valid = arrays["pair_changed_valid"]
    active = wait_valid & changed_valid
    physical = arrays["pair_wait_physical_hashes"] != arrays["pair_changed_physical_hashes"]
    actor_changed = arrays["pair_wait_actor_actions"] != arrays["pair_changed_actor_actions"]
    pair_correct = (
        active
        & (arrays["pair_wait_actor_actions"] == arrays["pair_wait_program_actions"])
        & (arrays["pair_changed_actor_actions"] == arrays["pair_changed_program_actions"])
    )
    if (not np.array_equal(arrays["pair_active"], active)
            or not np.array_equal(arrays["pair_physical_effect"], active & physical)
            or not np.array_equal(arrays["pair_actor_changed"], active & actor_changed)
            or not np.array_equal(arrays["pair_correct"], pair_correct)):
        raise ValueError("Stored intervention derivations differ")
    effective = active & physical & actor_changed
    direction = {
        "all_pairs": pair_n,
        "eligible_pairs": int(effective.sum()),
        "overall": _stat(effective, pair_correct, pair_fingerprints),
        "by_group": {},
    }
    for group, index in _GROUP_INDEX.items():
        mask = effective & ((arrays["pair_group_bits"] & (1 << index)) != 0)
        direction["by_group"][group] = _stat(mask, pair_correct, pair_fingerprints)

    audit_hashes = set(map(str, _decode(
        arrays["ordinary_observation_hashes"], "ordinary observation hashes")))
    for prefix in ("wait", "changed"):
        values = _decode(arrays[f"pair_{prefix}_observation_hashes"],
                         prefix + " observation hashes", allow_empty=True)
        audit_hashes.update(str(value) for value in values if str(value))
    overlap = audit_hashes & development_hashes
    checks = {
        "complete_64_scene_three_partner_matrix": False,
        "ordinary_fidelity": bool(ordinary_n)
            and fidelity["overall"]["fidelity"] >= 0.90,
        "ordinary_nonwait_fidelity": bool(nonwait.any())
            and fidelity["nonwait"]["fidelity"] >= 0.90,
        "effective_direction_fidelity": direction["overall"]["scenes"] >= 10
            and direction["overall"]["fidelity"] >= 0.85,
        "action_authority": bool(ordinary_n)
            and bool(np.all(actions == arrays["ordinary_submitted_actor_actions"]))
            and bool(np.all(arrays["ordinary_zero_overrides"]))
            and bool(np.all(arrays["pair_wait_submitted_equal"]))
            and bool(np.all(arrays["pair_changed_submitted_equal"])),
        "source_state_unchanged": bool(ordinary_n)
            and bool(np.all(arrays["ordinary_source_unchanged"]))
            and bool(np.all(arrays["pair_source_unchanged"])),
        "exact_development_observation_separation": not overlap,
        "raw_public_observations_present": bool(ordinary_n),
        "program_never_controls_action": True,
    }
    # Every intervention anchor has exactly the four non-WAIT player actions.
    anchor_rows: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(zip(
            arrays["pair_scene_indexes"].tolist(),
            arrays["pair_partner_indexes"].tolist(),
            arrays["pair_frames"].tolist())):
        anchor_rows.setdefault(tuple(map(int, key)), []).append(index)
    if any(len(indexes) != len(ACTIONS) - 1
           or set(map(int, arrays["pair_player_actions"][indexes]))
              != set(range(len(ACTIONS) - 1))
           for indexes in anchor_rows.values()):
        raise ValueError("Intervention anchor does not contain four directions")
    wait_fields = (
        "pair_wait_valid", "pair_wait_observations",
        "pair_wait_observation_hashes", "pair_wait_trace_hashes",
        "pair_wait_actor_actions", "pair_wait_program_actions",
        "pair_wait_physical_hashes", "pair_wait_submitted_equal",
    )
    for indexes in anchor_rows.values():
        for key in wait_fields:
            values = arrays[key][indexes]
            if not np.all(values == values[0]):
                raise ValueError("Intervention anchor WAIT endpoint differs")
    # Derive complete episode coverage and frame continuity from the raw rows.
    episodes = 0
    for partner_index in range(len(PARTNERS)):
        for scene_index in range(EXPLANATION_SCENES):
            mask = ((arrays["ordinary_partner_indexes"] == partner_index)
                    & (arrays["ordinary_scene_indexes"] == scene_index))
            frames = arrays["ordinary_frames"][mask]
            done = arrays["ordinary_done"][mask]
            if (len(frames) and np.array_equal(frames, np.arange(len(frames)))
                    and not np.any(done[:-1]) and bool(done[-1])):
                episodes += 1
    checks["complete_64_scene_three_partner_matrix"] = (
        episodes == EXPLANATION_SCENES * len(PARTNERS))
    for group in GROUPS:
        ordinary_group = fidelity["by_group"][group]
        direction_group = direction["by_group"][group]
        checks["ordinary_" + group] = (
            ordinary_group["scenes"] >= 10 and ordinary_group["fidelity"] >= 0.85)
        checks["direction_" + group] = (
            direction_group["scenes"] >= 10 and direction_group["fidelity"] >= 0.85)
    return {
        "fidelity": fidelity,
        "intervention_direction": direction,
        "exact_development_observation_overlap": len(overlap),
        "episodes": episodes,
        "checks": checks,
        "passed": all(checks.values()),
        "evaluated_role": "robot_2",
        "no_effect_pairs_counted_as_success": False,
    }


def _development_hashes(paths: Sequence[Path]) -> set[str]:
    hashes, _, _ = holdout_api._npz_development_observations(
        paths, required_fingerprints=set())
    return hashes


def _runtime(actor_path: Path, protocol_path: Path, manifest_path: Path,
             protocol: Mapping[str, Any], manifest: Mapping[str, Any]):
    content = deepcopy(manifest); manifest_content = content.pop("content_sha256", None)
    return R41DiagnosticOnlineAlignmentRuntime(
        actor_path,
        training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor_path),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=manifest_content,
        expected_manifest_semantic_sha256=digest(manifest),
    )


def _validate_inputs(
    *, actor_path: Path, protocol_path: Path, program_path: Path,
    rcpd_report_path: Path, manifest_path: Path, designation_path: Path,
    development_expansion_path: Path, fresh_holdout_path: Path,
    fresh_holdout_report_path: Path, development_rows_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any], list[dict], dict[str, Any], dict[str, Any]]:
    protocol = _read_json(protocol_path, "training protocol")
    manifest = _read_json(manifest_path, "conflict manifest")
    designation = _read_json(designation_path, "Actor designation")
    expansion = _read_json(development_expansion_path, "development expansion")
    holdout = _read_json(fresh_holdout_path, "fresh final holdout")
    holdout_report = _read_json(fresh_holdout_report_path, "fresh final report")
    rcpd_report = _read_json(rcpd_report_path, "v8 RCPD report")
    actor_sha256 = file_hash(actor_path)
    rows_bindings = {path.name: file_hash(path) for path in development_rows_paths}
    if len(rows_bindings) != len(development_rows_paths):
        raise ValueError("Development row artifact names must be unique")
    if (rcpd_report.get("version") != RCPD_VERSION
            or rcpd_report.get("status") != "passed_development_gates"
            or rcpd_report.get("explanation_eligible") is not True
            or rcpd_report.get("formal_ready") is not False
            or rcpd_report.get("program_file_sha256") != file_hash(program_path)
            or rcpd_report.get("bindings", {}).get("actor_file_sha256") != actor_sha256
            or rcpd_report.get("bindings", {}).get("protocol_file_sha256")
                != file_hash(protocol_path)
            or rcpd_report.get("bindings", {}).get("manifest_file_sha256")
                != file_hash(manifest_path)
            or rcpd_report.get("bindings", {}).get("designation_file_sha256")
                != file_hash(designation_path)
            or rcpd_report.get("bindings", {}).get("expansion_registry_file_sha256")
                != file_hash(development_expansion_path)
            or rcpd_report.get("evidence_artifacts", {}).get("rows.npz")
                not in set(rows_bindings.values())):
        raise ValueError("Exact frozen passing RCPD v8 candidate required")
    scenes = holdout.get("scenes")
    if (fresh_holdout_path.parent != fresh_holdout_report_path.parent):
        raise ValueError("Fresh-final registry/report must share an evidence directory")
    authenticated_holdout = holdout_api.read_saved_holdout(
        fresh_holdout_path.parent,
        expected_holdout_sha256=file_hash(fresh_holdout_path),
        expected_report_sha256=file_hash(fresh_holdout_report_path),
    )
    if authenticated_holdout != holdout:
        raise ValueError("Fresh-final authenticated content differs")
    if (holdout.get("version") != FRESH_HOLDOUT_VERSION
            or holdout.get("status") != "passed_program_blind_registry"
            or holdout.get("program_access") is not False
            or holdout.get("program_predictions_access") is not False
            or holdout.get("statistics", {}).get("accepted") != EXPLANATION_SCENES
            or holdout.get("statistics", {}).get("public_observation_overlap") != 0
            or not isinstance(scenes, list) or len(scenes) != EXPLANATION_SCENES
            or holdout_report.get("holdout_file_sha256") != file_hash(fresh_holdout_path)
            or holdout_report.get("holdout_content_sha256") != holdout.get("content_sha256")
            or holdout.get("bindings", {}).get("actor_sha256") != actor_sha256
            or holdout.get("bindings", {}).get("protocol_file_sha256")
                != file_hash(protocol_path)
            or holdout.get("bindings", {}).get("manifest_file_sha256")
                != file_hash(manifest_path)
            or holdout.get("bindings", {}).get("designation_file_sha256")
                != file_hash(designation_path)
            or holdout.get("bindings", {}).get("development_registries", {}).get(
                holdout_api.DEVELOPMENT_EXPANSION_VERSION, {}).get("file_sha256")
                != file_hash(development_expansion_path)):
        raise ValueError("Exact frozen v4 final holdout required")
    other = {row["fingerprint"] for rows in manifest["splits"].values() for row in rows}
    heldout = {row.get("fingerprint") for row in scenes}
    if len(heldout) != EXPLANATION_SCENES or heldout & other:
        raise ValueError("Fresh-final scenes overlap a registered manifest split")
    if (designation.get("bindings", {}).get("actor_sha256") != actor_sha256
            or designation.get("runtime_action_override") is not False
            or expansion.get("version") != holdout_api.DEVELOPMENT_EXPANSION_VERSION
            or expansion.get("program_access") is not False):
        raise ValueError("Diagnostic designation/development boundary differs")
    return protocol, manifest, deepcopy(scenes), rcpd_report, holdout


def audit(
    *, actor_path: str | Path, protocol_path: str | Path,
    program_path: str | Path, rcpd_report_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    development_expansion_path: str | Path,
    fresh_holdout_path: str | Path, fresh_holdout_report_path: str | Path,
    development_rows_paths: Sequence[str | Path], output: str | Path,
    claim_receipt_path: str | Path, expected_claim_sha256: str,
    expected_campaign_key: str, expected_candidate_identity_sha256: str,
    expected_holdout_completion_sha256: str,
) -> dict[str, Any]:
    claim_dir, _ = _claim_receipt(
        claim_receipt_path, expected_claim_sha256=expected_claim_sha256,
        expected_campaign_key=expected_campaign_key,
        expected_candidate_identity_sha256=expected_candidate_identity_sha256,
        expected_holdout_completion_sha256=expected_holdout_completion_sha256,
    )
    _claim_marker(claim_dir, "audit_started.json", {
        "version": VERSION, "status": "started_no_retry",
        "campaign_key": expected_campaign_key,
        "candidate_identity_sha256": expected_candidate_identity_sha256,
        "attempt_started_sha256": expected_claim_sha256,
        "holdout_completed_sha256": expected_holdout_completion_sha256,
    })
    paths = {
        "actor": _regular(actor_path, "frozen Actor"),
        "protocol": _regular(protocol_path, "training protocol"),
        "program": _regular(program_path, "v8 public-tree program"),
        "rcpd_report": _regular(rcpd_report_path, "v8 RCPD report"),
        "manifest": _regular(manifest_path, "conflict manifest"),
        "designation": _regular(designation_path, "Actor designation"),
        "development_expansion": _regular(
            development_expansion_path, "development expansion"),
        "fresh_holdout": _regular(fresh_holdout_path, "fresh final holdout"),
        "fresh_holdout_report": _regular(
            fresh_holdout_report_path, "fresh final report"),
    }
    holdout_phase = _read_json(
        claim_dir / "holdout_completed.json", "holdout completion phase")
    v3_exclusion_path = _regular(
        paths["fresh_holdout"].parent / "v3_exclusion.json",
        "claim-bound v3 exclusion")
    if (paths["fresh_holdout"].parent != paths["fresh_holdout_report"].parent
            or holdout_phase.get("holdout_file_sha256")
                != file_hash(paths["fresh_holdout"])
            or holdout_phase.get("report_file_sha256")
                != file_hash(paths["fresh_holdout_report"])
            or holdout_phase.get("v3_exclusion_file_sha256")
                != file_hash(v3_exclusion_path)):
        raise ValueError("Audit inputs differ from the claimed holdout phase")
    row_paths = [_regular(path, "development rows", maximum=MAX_NPZ_BYTES)
                 for path in development_rows_paths]
    protocol, manifest, scenes, rcpd_report, holdout = _validate_inputs(
        actor_path=paths["actor"], protocol_path=paths["protocol"],
        program_path=paths["program"], rcpd_report_path=paths["rcpd_report"],
        manifest_path=paths["manifest"], designation_path=paths["designation"],
        development_expansion_path=paths["development_expansion"],
        fresh_holdout_path=paths["fresh_holdout"],
        fresh_holdout_report_path=paths["fresh_holdout_report"],
        development_rows_paths=row_paths,
    )
    program, program_payload = _load_program(paths["program"])
    if rcpd_report.get("program_content_sha256") != digest(program_payload):
        raise ValueError("RCPD report program content binding differs")
    runtime = _runtime(paths["actor"], paths["protocol"], paths["manifest"],
                       protocol, manifest)
    if runtime.config.horizon != 120:
        raise ValueError("Final-audit horizon differs")
    initial_hashes = {name: file_hash(path) for name, path in paths.items()}
    initial_hashes["development_rows"] = digest([
        {"path": str(path), "sha256": file_hash(path)} for path in row_paths])
    sources = producer_sources()
    bindings = {
        "actor_sha256": file_hash(paths["actor"]),
        "actor_parameters_sha256": runtime.actor.metadata["actor_parameters_sha256"],
        "protocol_file_sha256": file_hash(paths["protocol"]),
        "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": file_hash(paths["manifest"]),
        "manifest_semantic_sha256": digest(manifest),
        "designation_file_sha256": file_hash(paths["designation"]),
        "expansion_registry_file_sha256": file_hash(paths["development_expansion"]),
        "program_file_sha256": file_hash(paths["program"]),
        "program_content_sha256": digest(program_payload),
        "rcpd_report_file_sha256": file_hash(paths["rcpd_report"]),
        "fresh_holdout_file_sha256": file_hash(paths["fresh_holdout"]),
        "fresh_holdout_content_sha256": holdout["content_sha256"],
        "fresh_holdout_report_file_sha256": file_hash(paths["fresh_holdout_report"]),
        "development_rows": {path.name: file_hash(path) for path in row_paths},
        "runtime_signature": runtime.signature,
        "runtime_manifest_signature": runtime.runtime_manifest_signature,
        "source_full_manifest_bindings": runtime.source_full_manifest_bindings,
        "source_full_manifest_bindings_sha256": digest(
            runtime.source_full_manifest_bindings),
        "runtime_sources": diagnostic_runtime_sources(),
        "runtime_sources_sha256": digest(diagnostic_runtime_sources()),
        "holdout_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in scenes)),
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(sources),
    }
    output_path = Path(output).expanduser().absolute()
    if (output_path.exists() or output_path.is_symlink()
            or not output_path.parent.is_dir() or output_path.parent.is_symlink()):
        raise ValueError("Explanation-audit output must be a new directory")
    output_path.mkdir(mode=0o700)
    inputs = {
        "version": VERSION,
        "contract": contract(),
        "bindings": bindings,
        "producer_sources": sources,
        "input_file_sha256": initial_hashes,
        "holdout_fingerprints": sorted(row["fingerprint"] for row in scenes),
        "formal_ready": False,
    }
    _write_json(output_path / "inputs.json", inputs)
    arrays, execution = _collect_evidence(runtime, program, scenes)
    _write_npz(output_path / "evidence.npz", arrays)
    statistics = _metrics(arrays, development_hashes=_development_hashes(row_paths))
    evidence_artifacts = {
        "inputs.json": file_hash(output_path / "inputs.json"),
        "evidence.npz": file_hash(output_path / "evidence.npz"),
    }
    report = {
        "version": VERSION,
        "status": "passed" if statistics["passed"] else "failed",
        "test_fixture": False,
        "formal_ready": False,
        "bindings": bindings,
        "statistics": statistics,
        "zero_nn_overrides": statistics["checks"]["action_authority"],
        "source_state_unchanged": statistics["checks"]["source_state_unchanged"],
        "counterfactual_isolated": bool(np.all(arrays["pair_source_unchanged"])),
        "program_never_controls_action": True,
        "raw_public_observations_persisted": True,
        "execution": {"accounting_complete": True, "pending_operation": None,
                      "counts": execution},
        "evidence_artifacts": evidence_artifacts,
    }
    report["content_sha256"] = digest(report)
    if ({name: file_hash(path) for name, path in paths.items()}
            != {name: value for name, value in initial_hashes.items()
                if name != "development_rows"}
            or digest([{"path": str(path), "sha256": file_hash(path)}
                       for path in row_paths]) != initial_hashes["development_rows"]
            or producer_sources() != sources):
        raise RuntimeError("Frozen final-audit input changed during execution")
    _write_json(output_path / "report.json", report)
    read_saved_report(
        output_path,
        expected_report_sha256=file_hash(output_path / "report.json"),
        program_path=paths["program"],
        development_rows_paths=row_paths,
        expected_bindings=bindings,
        require_passed=False,
    )
    _claim_marker(claim_dir, "audit_completed.json", {
        "version": VERSION, "status": report["status"],
        "campaign_key": expected_campaign_key,
        "candidate_identity_sha256": expected_candidate_identity_sha256,
        "attempt_started_sha256": expected_claim_sha256,
        "holdout_completed_sha256": expected_holdout_completion_sha256,
        "report_file_sha256": file_hash(output_path / "report.json"),
        "evidence_file_sha256": file_hash(output_path / "evidence.npz"),
    })
    return deepcopy(report)


def read_saved_report(
    output: str | Path, *, expected_report_sha256: str,
    program_path: str | Path, development_rows_paths: Sequence[str | Path],
    expected_bindings: Mapping[str, Any], require_passed: bool = True,
) -> dict[str, Any]:
    """Recompute predictions, structured traces, metrics, and all hard gates."""
    output_path = Path(output).expanduser().absolute()
    if (not output_path.is_dir() or output_path.is_symlink()
            or output_path.resolve() != output_path):
        raise ValueError("Explanation-audit evidence directory is unsafe")
    report_path = _regular(output_path / "report.json", "explanation report")
    if file_hash(report_path) != expected_report_sha256:
        raise ValueError("Explanation report hash differs")
    report = _read_json(report_path, "explanation report")
    content = deepcopy(report); claimed_content = content.pop("content_sha256", None)
    if (claimed_content != digest(content) or report.get("version") != VERSION
            or report.get("bindings") != dict(expected_bindings)):
        raise ValueError("Explanation report identity differs")
    artifacts = report.get("evidence_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"inputs.json", "evidence.npz"}:
        raise ValueError("Explanation evidence artifact registry differs")
    for name, expected in artifacts.items():
        path = _regular(output_path / name, "explanation artifact")
        if path.parent != output_path or file_hash(path) != expected:
            raise ValueError("Explanation evidence artifact changed")
    inputs = _read_json(output_path / "inputs.json", "explanation inputs")
    if (inputs.get("version") != VERSION or inputs.get("contract") != contract()
            or inputs.get("bindings") != dict(expected_bindings)
            or inputs.get("producer_sources") != producer_sources()
            or digest(inputs["contract"]) != expected_bindings.get("contract_sha256")
            or digest(inputs["producer_sources"])
                != expected_bindings.get("producer_sources_sha256")):
        raise ValueError("Explanation inputs/source closure differs")
    input_files = inputs.get("input_file_sha256")
    expected_input_bindings = {
        "actor": expected_bindings.get("actor_sha256"),
        "protocol": expected_bindings.get("protocol_file_sha256"),
        "program": expected_bindings.get("program_file_sha256"),
        "rcpd_report": expected_bindings.get("rcpd_report_file_sha256"),
        "manifest": expected_bindings.get("manifest_file_sha256"),
        "designation": expected_bindings.get("designation_file_sha256"),
        "development_expansion": expected_bindings.get(
            "expansion_registry_file_sha256"),
        "fresh_holdout": expected_bindings.get("fresh_holdout_file_sha256"),
        "fresh_holdout_report": expected_bindings.get(
            "fresh_holdout_report_file_sha256"),
    }
    if (not isinstance(input_files, Mapping)
            or any(input_files.get(key) != value
                   for key, value in expected_input_bindings.items())
            or any(_HEX.fullmatch(str(value)) is None
                   for value in input_files.values())):
        raise ValueError("Explanation input file bindings differ")
    program_path = _regular(program_path, "v8 public-tree program")
    if file_hash(program_path) != expected_bindings.get("program_file_sha256"):
        raise ValueError("Explanation program binding differs")
    program, payload = _load_program(program_path)
    if digest(payload) != expected_bindings.get("program_content_sha256"):
        raise ValueError("Explanation program content binding differs")
    row_paths = [_regular(path, "development rows", maximum=MAX_NPZ_BYTES)
                 for path in development_rows_paths]
    if ({path.name: file_hash(path) for path in row_paths}
            != expected_bindings.get("development_rows")):
        raise ValueError("Development rows binding differs")
    expected_rows_input = digest([
        {"path": str(path), "sha256": file_hash(path)} for path in row_paths
    ])
    if (set(input_files) != {*expected_input_bindings, "development_rows"}
            or input_files.get("development_rows") != expected_rows_input):
        raise ValueError("Explanation development-row input digest differs")
    arrays = _load_npz(output_path / "evidence.npz")
    _recompute_program_evidence(arrays, program, program.base_feature_names)
    statistics = _metrics(arrays, development_hashes=_development_hashes(row_paths))
    anchor_count = len(set(zip(
        arrays["pair_scene_indexes"].tolist(),
        arrays["pair_partner_indexes"].tolist(),
        arrays["pair_frames"].tolist(),
    )))
    expected_execution = {
        "base_steps": len(arrays["ordinary_observations"]),
        "counterfactual_steps": anchor_count * len(ACTIONS),
        "episodes": EXPLANATION_SCENES * len(PARTNERS),
        "neural_updates": 0,
        "tree_fits": 0,
    }
    expected = deepcopy(report)
    expected.update({
        "status": "passed" if statistics["passed"] else "failed",
        "statistics": statistics,
        "zero_nn_overrides": statistics["checks"]["action_authority"],
        "source_state_unchanged": statistics["checks"]["source_state_unchanged"],
        "counterfactual_isolated": bool(np.all(arrays["pair_source_unchanged"])),
        "program_never_controls_action": True,
        "raw_public_observations_persisted": True,
        "execution": {"accounting_complete": True, "pending_operation": None,
                      "counts": expected_execution},
    })
    expected["content_sha256"] = digest({
        key: value for key, value in expected.items() if key != "content_sha256"
    })
    if expected != report:
        raise ValueError("Explanation report differs from raw public evidence")
    if require_passed and report["status"] != "passed":
        raise ValueError("Diagnostic explanation audit did not meet its hard gates")
    return deepcopy(report)


def replay_saved_audit(
    output: str | Path, *, actor_path: str | Path, protocol_path: str | Path,
    program_path: str | Path, manifest_path: str | Path,
    fresh_holdout_path: str | Path, expected_evidence_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay physics and compare every saved evidence array."""
    output_path = Path(output).expanduser().absolute()
    evidence_path = _regular(output_path / "evidence.npz", "saved evidence",
                             maximum=MAX_NPZ_BYTES)
    if file_hash(evidence_path) != expected_evidence_sha256:
        raise ValueError("Physical replay evidence anchor differs")
    actor_path = _regular(actor_path, "frozen Actor")
    protocol_path = _regular(protocol_path, "training protocol")
    program_path = _regular(program_path, "v8 public-tree program")
    manifest_path = _regular(manifest_path, "conflict manifest")
    fresh_holdout_path = _regular(fresh_holdout_path, "fresh final holdout")
    for actual, key in (
        (file_hash(actor_path), "actor_sha256"),
        (file_hash(protocol_path), "protocol_file_sha256"),
        (file_hash(program_path), "program_file_sha256"),
        (file_hash(manifest_path), "manifest_file_sha256"),
        (file_hash(fresh_holdout_path), "fresh_holdout_file_sha256"),
    ):
        if actual != expected_bindings.get(key):
            raise ValueError("Physical replay input binding differs: " + key)
    holdout = _read_json(fresh_holdout_path, "fresh final holdout")
    scenes = holdout.get("scenes")
    if (holdout.get("version") != FRESH_HOLDOUT_VERSION
            or not isinstance(scenes, list) or len(scenes) != EXPLANATION_SCENES):
        raise ValueError("Physical replay holdout differs")
    protocol = _read_json(protocol_path, "training protocol")
    manifest = _read_json(manifest_path, "conflict manifest")
    runtime = _runtime(actor_path, protocol_path, manifest_path, protocol, manifest)
    runtime_sources = diagnostic_runtime_sources()
    if (runtime.signature != expected_bindings.get("runtime_signature")
            or runtime.runtime_manifest_signature
                != expected_bindings.get("runtime_manifest_signature")
            or runtime.actor.metadata.get("actor_parameters_sha256")
                != expected_bindings.get("actor_parameters_sha256")
            or runtime.source_full_manifest_bindings
                != expected_bindings.get("source_full_manifest_bindings")
            or digest(runtime.source_full_manifest_bindings)
                != expected_bindings.get("source_full_manifest_bindings_sha256")
            or runtime_sources != expected_bindings.get("runtime_sources")
            or digest(runtime_sources)
                != expected_bindings.get("runtime_sources_sha256")
            or digest(producer_sources())
                != expected_bindings.get("producer_sources_sha256")):
        raise ValueError("Physical replay runtime/source binding differs")
    program, _ = _load_program(program_path)
    replayed, execution = _collect_evidence(runtime, program, scenes)
    saved = _load_npz(evidence_path)
    if set(replayed) != set(saved):
        raise ValueError("Physical replay array registry differs")
    for key in sorted(saved):
        if saved[key].dtype != replayed[key].dtype or not np.array_equal(saved[key], replayed[key]):
            raise ValueError("Physical replay differs: " + key)
    return {
        "version": VERSION + ".physical-replay.v1",
        "status": "passed",
        "evidence_file_sha256": expected_evidence_sha256,
        "arrays_sha256": digest({key: sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                  for key, value in sorted(saved.items())}),
        "execution": execution,
        "program_fits": 0,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
    }


__all__ = [
    "VERSION", "RCPD_VERSION", "FRESH_HOLDOUT_VERSION", "PARTNERS", "GROUPS",
    "EXPLANATION_SCENES", "ARRAY_KEYS", "contract", "producer_sources",
    "read_saved_report", "replay_saved_audit",
]
