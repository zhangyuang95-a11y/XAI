"""Strict post-claim explanation audit for the v9 warehouse candidate.

This module does not select a final split and has no salt reader. Its caller
must first make an irrevocable final-test claim, then collect the final rows
twice with the frozen neural Actor. The program is used only to explain and
score those rows; it is never passed to the environment runtime.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_outer_collection_v9 as collection_api
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v9 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import digest, file_hash
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    R41DiagnosticPublicTreeProgramV9,
)
from env.warehouse_native.policy import NumPyNativeActor


VERSION = "warehouse-r41-diagnostic-explanation-audit.v9"
STATUS_PASSED = "passed_final_nine_gates_and_physical_replay"
STATUS_FAILED = "failed_final_explanation_audit"
FINAL_SCENE_COUNT = 64
PREDICTION_BATCH_SIZE = 16_384
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_REPORT_FIELDS = frozenset((
    "version", "status", "bindings", "metrics", "nine_gate", "isolation",
    "coverage", "physical_counterfactual_audit", "action_authority",
    "execution", "producer_sources", "producer_sources_sha256",
    "explanation_eligible", "formal_ready", "content_sha256",
))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "final_scene_count": FINAL_SCENE_COUNT,
        "whole_scene_isolation": True,
        "zero_cross_split_public_observation_overlap": True,
        "independent_physical_replay": True,
        "hard_gate_count": 9,
        "program_controls_runtime_actions": False,
        "actor_action_override": False,
        "protected_final_access": "only after an authenticated permanent claim",
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _decode(array: np.ndarray, label: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be a one-dimensional byte-string array")
    try:
        return list(map(str, np.char.decode(array, "ascii")))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be ASCII") from error


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must contain one JSON object")
    return value


def _semantic_array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    value: dict[str, Any] = {}
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        value[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "bytes_sha256": sha256(array.tobytes(order="C")).hexdigest(),
        }
    return digest(value)


def _assert_exact_replay(
    rows: Mapping[str, np.ndarray], replay: Mapping[str, np.ndarray],
) -> None:
    if set(rows) != set(replay):
        raise ValueError("Physical replay row schema differs")
    for name in sorted(rows):
        left, right = rows[name], replay[name]
        if (left.dtype != right.dtype or left.shape != right.shape
                or not np.array_equal(left, right, equal_nan=False)):
            raise ValueError("Physical replay differs at row field " + name)


def _episode_coverage(arrays: Mapping[str, np.ndarray], scenes: set[str]) -> int:
    kinds = np.asarray(_decode(arrays["kinds"], "final row kinds"))
    ordinary = kinds == "ordinary"
    fingerprints = np.asarray(_decode(
        arrays["scene_fingerprints"], "final scene fingerprints"))
    episodes = np.asarray(_decode(arrays["episode_ids"], "final episode ids"))
    count = 0
    for fingerprint in scenes:
        selected_episodes = set(map(str, episodes[ordinary & (fingerprints == fingerprint)]))
        if len(selected_episodes) != len(rows_api.PARTNERS):
            raise ValueError("Final scene does not cover every frozen partner")
        for episode in selected_episodes:
            mask = ordinary & (episodes == episode)
            frames = arrays["frames"][mask]
            done = arrays["trajectory_done"][mask]
            if (len(frames) == 0
                    or not np.array_equal(frames, np.arange(len(frames)))
                    or np.any(done[:-1]) or not bool(done[-1])):
                raise ValueError("Final episode frame coverage differs")
            count += 1
    return count


def _predict(
    program: R41DiagnosticPublicTreeProgramV9, observations: np.ndarray,
) -> np.ndarray:
    chunks = [program.predict_proba_batch(
        observations[start:start + PREDICTION_BATCH_SIZE])
        for start in range(0, len(observations), PREDICTION_BATCH_SIZE)]
    return np.concatenate(chunks, axis=0) if chunks else np.empty(
        (0, len(metrics_api.ACTIONS)), dtype=np.float64)


def audit_rows(
    *, actor_path: str | Path, program_path: str | Path,
    program_payload: Mapping[str, Any],
    program_sha256: str, arrays: Mapping[str, np.ndarray],
    replay_arrays: Mapping[str, np.ndarray], scenes: Sequence[Mapping[str, Any]],
    development_observation_hashes: set[str], outer_observation_hashes: set[str],
    development_scene_fingerprints: set[str], outer_scene_fingerprints: set[str],
    bindings: Mapping[str, str], environment_steps: int,
    replay_environment_steps: int,
) -> dict[str, Any]:
    """Audit already collected final rows. Call only after permanent claim."""
    actor_file = Path(actor_path).expanduser().absolute()
    if (not actor_file.is_file() or actor_file.is_symlink()
            or actor_file.resolve() != actor_file
            or file_hash(actor_file) != bindings.get("actor_sha256")):
        raise ValueError("Exact frozen Actor required for final audit")
    if (type(program_sha256) is not str or _HEX.fullmatch(program_sha256) is None
            or bindings.get("program_sha256") != program_sha256):
        raise ValueError("Exact locked program identity required for final audit")
    program_file = Path(program_path).expanduser().absolute()
    if (not program_file.is_file() or program_file.is_symlink()
            or program_file.resolve() != program_file
            or file_hash(program_file) != program_sha256
            or _strict_json_object(program_file.read_bytes(), "locked v9 program")
                != dict(program_payload)):
        raise ValueError("Locked program bytes/payload differ for final audit")
    if (type(environment_steps) is not int or environment_steps <= 0
            or replay_environment_steps != environment_steps):
        raise ValueError("Physical replay environment-step accounting differs")

    actor = NumPyNativeActor(actor_file)
    program = R41DiagnosticPublicTreeProgramV9.from_dict(program_payload)
    if (tuple(program.action_names) != tuple(metrics_api.ACTIONS)
            or tuple(program.base_feature_names)
                != tuple(actor.metadata.get("feature_names", ()))):
        raise ValueError("Program and Actor public feature registries differ")
    scene_rows = list(scenes)
    expected_scenes = {str(row.get("fingerprint")) for row in scene_rows}
    if (len(scene_rows) != FINAL_SCENE_COUNT
            or len(expected_scenes) != FINAL_SCENE_COUNT
            or any(_HEX.fullmatch(value) is None for value in expected_scenes)):
        raise ValueError("Exact 64-scene final registry required")

    collection_api._validate_static_rows(arrays)
    collection_api._validate_static_rows(replay_arrays)
    projection_api._validate_projection_replay_arrays(
        arrays, actor=actor, scenes=scene_rows)
    projection_api._validate_projection_replay_arrays(
        replay_arrays, actor=actor, scenes=scene_rows)
    _assert_exact_replay(arrays, replay_arrays)
    if not np.all(arrays["split_validation"]):
        raise ValueError("Every final row must belong to the final split")

    row_scenes = set(_decode(
        arrays["scene_fingerprints"], "final row scene fingerprints"))
    final_hashes = set(_decode(
        arrays["observation_hashes"], "final observation hashes"))
    scene_overlap_development = row_scenes & set(development_scene_fingerprints)
    scene_overlap_outer = row_scenes & set(outer_scene_fingerprints)
    observation_overlap_development = final_hashes & set(
        development_observation_hashes)
    observation_overlap_outer = final_hashes & set(outer_observation_hashes)
    if (row_scenes != expected_scenes or scene_overlap_development
            or scene_overlap_outer or observation_overlap_development
            or observation_overlap_outer):
        raise ValueError("Final scenes or public observations overlap prior splits")

    episodes = _episode_coverage(arrays, expected_scenes)
    pairs = rows_api._effective_pairs(
        arrays, np.ones(len(arrays["observations"]), dtype=np.bool_))
    pair_bits = metrics_api._pair_group_bits(arrays, pairs)
    probabilities = _predict(program, arrays["observations"])
    metrics = metrics_api._metrics_from_probabilities(
        probabilities, arrays,
        np.ones(len(arrays["observations"]), dtype=np.bool_),
        pairs=pairs, pair_group_bits=pair_bits)
    gate = metrics_api._gate(metrics)
    expected_gates = {
        "overall", "nonwait", "effective_intervention_direction",
        *metrics_api.GROUPS,
        *("effective_intervention_direction_" + group
          for group in metrics_api.GROUPS),
    }
    if set(gate.get("checks", {})) != expected_gates or len(gate["checks"]) != 9:
        raise ValueError("Final explanation gate registry is not the frozen nine gates")

    kinds = np.asarray(_decode(arrays["kinds"], "final row kinds"))
    intervention = kinds == "intervention"
    physical = _decode(arrays["physical_hashes"], "final physical hashes")
    source_states = _decode(arrays["source_state_hashes"], "final source hashes")
    physical_valid = all(
        (not intervention[index] and value == "")
        or (intervention[index] and _HEX.fullmatch(value) is not None)
        for index, value in enumerate(physical))
    source_valid = all(_HEX.fullmatch(value) is not None for value in source_states)
    physical_passed = bool(len(pairs)) and physical_valid and source_valid
    authority = {
        "all_submitted_actions_equal_policy_actions": bool(
            np.all(arrays["submitted_equal"])),
        "all_actor_probabilities_and_actions_exact": True,
        "runtime_action_overrides": 0,
        "program_controls_runtime_actions": False,
    }
    isolation = {
        "whole_scene_registry_matches_rows": row_scenes == expected_scenes,
        "development_scene_overlap": len(scene_overlap_development),
        "outer_scene_overlap": len(scene_overlap_outer),
        "development_observation_overlap": len(observation_overlap_development),
        "outer_observation_overlap": len(observation_overlap_outer),
        "zero_cross_split_observation_overlap": not (
            observation_overlap_development or observation_overlap_outer),
    }
    passed = bool(gate["passed"] and physical_passed
                  and authority["all_submitted_actions_equal_policy_actions"]
                  and authority["all_actor_probabilities_and_actions_exact"])
    sources = producer_sources()
    report: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS_PASSED if passed else STATUS_FAILED,
        "bindings": dict(bindings),
        "metrics": metrics,
        "nine_gate": gate,
        "isolation": isolation,
        "coverage": {
            "scenes": len(expected_scenes), "episodes": episodes,
            "partners_per_scene": len(rows_api.PARTNERS),
            "rows": len(arrays["observations"]),
            "effective_intervention_pairs": len(pairs),
        },
        "physical_counterfactual_audit": {
            "independent_replay_exact": True,
            "rows_semantic_sha256": _semantic_array_sha256(arrays),
            "replay_rows_semantic_sha256": _semantic_array_sha256(replay_arrays),
            "environment_steps": environment_steps,
            "replay_environment_steps": replay_environment_steps,
            "intervention_physical_hashes_valid": physical_valid,
            "source_state_hashes_valid": source_valid,
            "passed": physical_passed,
        },
        "action_authority": authority,
        "execution": {
            "program_fits": 0, "actor_updates": 0,
            "candidate_or_program_selected_on_final": False,
            "final_rows_read_after_permanent_claim": True,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "explanation_eligible": passed,
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    return report


def validate_report(
    value: Mapping[str, Any], *, expected_bindings: Mapping[str, str],
    require_passed: bool,
) -> dict[str, Any]:
    if (set(value) != _REPORT_FIELDS or value.get("version") != VERSION
            or value.get("formal_ready") is not False
            or value.get("bindings") != dict(expected_bindings)
            or value.get("content_sha256") != digest({
                key: child for key, child in value.items()
                if key != "content_sha256"})
            or value.get("producer_sources") != producer_sources()
            or value.get("producer_sources_sha256") != digest(producer_sources())):
        raise ValueError("Saved v9 final explanation audit differs")
    try:
        recomputed = metrics_api._gate(value["metrics"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Saved v9 final metrics cannot be gated") from error
    if recomputed != value.get("nine_gate") or len(recomputed["checks"]) != 9:
        raise ValueError("Saved v9 final nine-gate result differs")
    passed = (
        value.get("status") == STATUS_PASSED
        and value.get("explanation_eligible") is True
        and recomputed.get("passed") is True
        and value.get("physical_counterfactual_audit", {}).get("passed") is True
        and value.get("isolation", {}).get(
            "zero_cross_split_observation_overlap") is True
        and value.get("action_authority", {}).get("runtime_action_overrides") == 0
        and value.get("action_authority", {}).get(
            "program_controls_runtime_actions") is False
    )
    if require_passed and not passed:
        raise ValueError("Saved v9 final explanation audit did not pass")
    return deepcopy(dict(value))


__all__ = [
    "VERSION", "STATUS_PASSED", "STATUS_FAILED", "FINAL_SCENE_COUNT",
    "contract", "producer_sources", "audit_rows", "validate_report",
]
