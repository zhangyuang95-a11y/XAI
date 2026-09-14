"""Freeze a v13 outer after the one-shot v12 final attempt was burned.

Selection starts from the same fixed identity population, authenticates the
permanent v12 final closeout, excludes the consumed v12 outer and every burned
v12 final identity, and screens a fixed public ordering against both one-way
observation-hash projections.  Screening uses Actor replay only to advance the
environment and compare hashes; actions, probabilities, program outputs,
labels, protected-final material, and private final salt are unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as burned_v12_api
from backend.training import warehouse_r41_diagnostic_rcpd_v12_outer_split as v12
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v13-fresh-outer-registry.v1"
REPORT_VERSION = VERSION
STATUS = "frozen_hash_screened_pending_outer_collection"
SELECTION_SALT = "warehouse-r41-v13-fresh-development-outer-20260914-v1"
FAMILY_IDS = tuple(v12.FAMILY_IDS)
FAMILY_QUOTAS = dict(v12.FAMILY_QUOTAS)
FRESH_OUTER_SCENE_COUNT = 64
SOURCE_ROW_SCENE_COUNT = v12.SOURCE_ROW_SCENE_COUNT
FIXED_CANDIDATE_SCENE_COUNT = v12.FIXED_CANDIDATE_SCENE_COUNT
EXPECTED_REMAINING_SCENE_COUNT = 1375
EXPECTED_REMAINING_FAMILY_COUNTS = {
    "conflict_family_01": 224,
    "conflict_family_02": 225,
    "conflict_family_03": 227,
    "conflict_family_04": 224,
    "conflict_family_05": 237,
    "conflict_family_06": 238,
}
SCENE_OFFSET = 704
MAX_JSON_BYTES = v12.MAX_JSON_BYTES
_HEX = re.compile(r"[0-9a-f]{64}\Z")

RECOVERY_BOUNDARY = {
    **v12.RECOVERY_BOUNDARY,
    "burned_v12_final_closed_before_selection": True,
    "consumed_v12_outer_identities_excluded": True,
    "burned_v12_final_identities_excluded": True,
    "consumed_v12_outer_observation_hashes_excluded_by_full_replay": True,
    "burned_v12_final_observation_hashes_excluded_by_full_replay": True,
    "consumed_v12_outer_rows_opened": False,
    "burned_v12_final_recollected_rows_opened": False,
    "promoted_rows_used_during_identity_selection": False,
    "prior_validation_projection_includes_consumed_v12_outer": True,
    "prior_validation_projection_includes_burned_v12_final": True,
}
RECOVERY_BINDINGS = frozenset(set(v12.RECOVERY_BINDINGS) | {
    "burned_v12_final_closeout_file_sha256",
    "burned_v12_final_closeout_content_sha256",
    "consumed_v12_outer_rows_sha256",
    "consumed_v12_outer_rows_semantic_sha256",
    "consumed_v12_outer_selected_identity_sha256",
    "consumed_v12_outer_observation_hashes_sha256",
    "burned_v12_final_selected_identity_sha256",
    "burned_v12_final_observation_hashes_sha256",
    "promotion_closeout_file_sha256",
    "promotion_closeout_content_sha256",
    "combined_promoted_rows_sha256",
    "combined_promoted_rows_semantic_sha256",
    "combined_promoted_observation_hashes_sha256",
})
_RECOVERY_BOUNDARY = RECOVERY_BOUNDARY
_RECOVERY_BINDINGS = RECOVERY_BINDINGS

EXPECTED_MANIFEST_SHA256 = v12.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = v12.EXPECTED_MANIFEST_VALIDATION_SHA256
EXPECTED_FAILED_ROWS_SHA256 = v12.EXPECTED_FAILED_ROWS_SHA256
EXPECTED_ORIGINAL_EXPANSION_SHA256 = v12.EXPECTED_ORIGINAL_EXPANSION_SHA256
EXPECTED_CURRENT_OUTER_REGISTRY_SHA256 = v12.EXPECTED_CURRENT_OUTER_REGISTRY_SHA256
EXPECTED_CURRENT_OUTER_REPORT_SHA256 = v12.EXPECTED_CURRENT_OUTER_REPORT_SHA256
EXPECTED_FORMAL_SELECTION_SHA256 = v12.EXPECTED_FORMAL_SELECTION_SHA256
EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 = v12.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256
EXPECTED_RETIRED_PROJECTION_SHA256 = v12.EXPECTED_RETIRED_PROJECTION_SHA256
EXPECTED_RETIRED_REPORT_SHA256 = v12.EXPECTED_RETIRED_REPORT_SHA256


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _rank(row: Mapping[str, Any]) -> str:
    return digest({
        "salt": SELECTION_SALT,
        "family_id": row["family_id"],
        "fingerprint": row["fingerprint"],
    })


def _public_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in (
        "batch_index", "family_id", "seed", "fingerprint")}


def _remaining_and_selected(
    candidates: Sequence[Mapping[str, Any]], *, excluded_seeds: set[int],
    excluded_fingerprints: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if len(candidates) != FIXED_CANDIDATE_SCENE_COUNT:
        raise ValueError("Fixed candidate population count differs")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in candidates:
        seed, fingerprint = v12.v11.v9._identity(raw, "fixed v13 candidate")
        family = raw.get("family_id")
        batch_index, family_offset = raw.get("batch_index"), raw.get("family_offset")
        if (family not in FAMILY_IDS or type(batch_index) is not int
                or type(family_offset) is not int):
            raise ValueError("Fixed v13 candidate identity differs")
        if seed not in excluded_seeds and fingerprint not in excluded_fingerprints:
            by_family[str(family)].append(deepcopy(dict(raw)))
    remaining = [row for family in FAMILY_IDS for row in by_family[family]]
    counts = dict(sorted(Counter(row["family_id"] for row in remaining).items()))
    if (len(remaining) != EXPECTED_REMAINING_SCENE_COUNT
            or counts != EXPECTED_REMAINING_FAMILY_COUNTS):
        raise ValueError("Fixed v13 remaining population differs")
    selected: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (
            _rank(row), row["fingerprint"]))
        quota = FAMILY_QUOTAS[family]
        if len(ordered) < quota:
            raise ValueError("Insufficient v13 identities for " + family)
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (FAMILY_IDS.index(row["family_id"]), _rank(row)))
    if (len(selected) != FRESH_OUTER_SCENE_COUNT
            or dict(sorted(Counter(row["family_id"] for row in selected).items()))
                != FAMILY_QUOTAS):
        raise RuntimeError("V13 outer family quota differs")
    return remaining, selected, counts


def _materialize_scene(identity: Mapping[str, Any], index: int) -> dict[str, Any]:
    seed, expected_fingerprint = v12.v11.v9._identity(identity, "selected v13 outer")
    batch_index, family = identity.get("batch_index"), identity.get("family_id")
    if type(batch_index) is not int or family not in FAMILY_IDS:
        raise ValueError("Selected v13 outer identity differs")
    env = R41DiagnosticConflictWarehouseEnv()
    v12.v11.v9.retired_api._identity_only_reset(env, seed=seed)
    reset_family = conflict_family_id(env.state.tasks)
    if reset_family != family:
        raise ValueError("Selected v13 outer family differs at reset")
    v12.v11.v9.scenes_api._candidate_start_state(
        env, seed=seed, batch_index=batch_index, family_id=reset_family)
    if diagnostic_scene_fingerprint(env) != expected_fingerprint:
        raise ValueError("Selected v13 outer fingerprint differs")
    scene = v12.v11.v9.scenes_api._scene_from_environment(
        env, scene_id=f"diagnostic_v13_fresh_outer_{index:04d}",
        split="development_outer", seed=seed, batch_index=batch_index)
    if (scene.get("fingerprint") != expected_fingerprint
            or scene.get("family_id") != family):
        raise ValueError("Materialised v13 outer identity differs")
    return scene


def _component_projection(component: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    projection = component.get("observation_hash_projection")
    if not isinstance(projection, Mapping):
        raise ValueError(label + " observation-hash projection missing")
    values = projection.get("outer_observation_hashes")
    if values is None:
        values = projection.get("unique_observation_hashes")
    if (not isinstance(values, list) or values != sorted(set(values))
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in values)):
        raise ValueError(label + " observation-hash projection differs")
    return {**projection, "outer_observation_hashes": list(values)}


def _closeout_component(
    closeout: Mapping[str, Any], name: str, label: str,
) -> Mapping[str, Any]:
    component = closeout.get(name)
    if not isinstance(component, Mapping):
        raise ValueError(label + " component missing from v12 closeout")
    return component


def _component_identities(
    component: Mapping[str, Any], label: str,
) -> list[dict[str, Any]]:
    raw = component.get("identities")
    if not isinstance(raw, list):
        raw = component.get("selected_identities")
    if not isinstance(raw, list) or len(raw) != FRESH_OUTER_SCENE_COUNT:
        raise ValueError(label + " identities differ")
    identities = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {
                "batch_index", "family_id", "seed", "fingerprint"}:
            raise ValueError(label + " identity fields differ")
        identities.append(_public_identity(item))
    if (len({row["seed"] for row in identities}) != len(identities)
            or len({row["fingerprint"] for row in identities})
                != len(identities)
            or component.get("identities_sha256") != digest(identities)):
        raise ValueError(label + " identity digest differs")
    return identities


def replay_exclusion_closure(
    *, actor_path: str | Path, protocol_path: str | Path,
    designation_path: str | Path, manifest_path: str | Path,
    failed_rows_path: str | Path, original_expansion_path: str | Path,
    current_outer_registry_path: str | Path,
    current_outer_report_path: str | Path,
    failure_closeout_path: str | Path,
    expected_failure_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
    formal_selection_path: str | Path,
    previous_development_path: str | Path,
    retired_projection_path: str | Path,
    consumed_v9_attempt_closeout_path: str | Path,
    expected_consumed_v9_attempt_closeout_sha256: str,
    permanent_v9_attempt_registry: str | Path,
    burned_v10_final_closeout_path: str | Path,
    expected_burned_v10_final_closeout_sha256: str,
    permanent_v10_final_closeout_registry: str | Path,
    consumed_v11_outer_closeout_path: str | Path,
    expected_consumed_v11_outer_closeout_sha256: str,
    permanent_v11_outer_attempt_registry: str | Path,
    burned_v12_final_closeout_path: str | Path,
    expected_burned_v12_final_closeout_sha256: str,
    permanent_v12_final_closeout_registry: str | Path,
) -> dict[str, Any]:
    base = v12.replay_exclusion_closure(
        actor_path=actor_path, protocol_path=protocol_path,
        designation_path=designation_path, manifest_path=manifest_path,
        failed_rows_path=failed_rows_path,
        original_expansion_path=original_expansion_path,
        current_outer_registry_path=current_outer_registry_path,
        current_outer_report_path=current_outer_report_path,
        failure_closeout_path=failure_closeout_path,
        expected_failure_closeout_sha256=expected_failure_closeout_sha256,
        permanent_closeout_registry=permanent_closeout_registry,
        formal_selection_path=formal_selection_path,
        previous_development_path=previous_development_path,
        retired_projection_path=retired_projection_path,
        consumed_v9_attempt_closeout_path=consumed_v9_attempt_closeout_path,
        expected_consumed_v9_attempt_closeout_sha256=(
            expected_consumed_v9_attempt_closeout_sha256),
        permanent_v9_attempt_registry=permanent_v9_attempt_registry,
        burned_v10_final_closeout_path=burned_v10_final_closeout_path,
        expected_burned_v10_final_closeout_sha256=(
            expected_burned_v10_final_closeout_sha256),
        permanent_v10_final_closeout_registry=(
            permanent_v10_final_closeout_registry),
        consumed_v11_outer_closeout_path=consumed_v11_outer_closeout_path,
        expected_consumed_v11_outer_closeout_sha256=(
            expected_consumed_v11_outer_closeout_sha256),
        permanent_v11_outer_attempt_registry=(
            permanent_v11_outer_attempt_registry),
    )
    closeout_path = v12.v11.v9._regular(
        burned_v12_final_closeout_path, "burned v12 final closeout",
        maximum=MAX_JSON_BYTES)
    closeout = burned_v12_api.read_saved_closeout_public(
        closeout_path,
        expected_closeout_sha256=expected_burned_v12_final_closeout_sha256,
        permanent_closeout_registry=permanent_v12_final_closeout_registry)
    consumed_outer = _closeout_component(
        closeout, "consumed_v12_outer", "consumed v12 outer")
    burned_final = _closeout_component(
        closeout, "burned_final", "burned v12 final")
    outer_identities = _component_identities(
        consumed_outer, "consumed v12 outer")
    final_identities = _component_identities(
        burned_final, "burned v12 final")
    candidate_public = {
        (row["seed"], row["fingerprint"]): _public_identity(row)
        for row in base["candidates"]}
    for label, identities in (("consumed v12 outer", outer_identities),
                              ("burned v12 final", final_identities)):
        if any(candidate_public.get((row["seed"], row["fingerprint"])) != row
               for row in identities):
            raise ValueError(label + " identities are not exact fixed candidates")
        if any(row["seed"] in base["excluded_seeds"]
               or row["fingerprint"] in base["excluded_fingerprints"]
               for row in identities):
            raise ValueError(label + " identity overlaps an older exclusion")
    outer_keys = {(row["seed"], row["fingerprint"])
                  for row in outer_identities}
    final_keys = {(row["seed"], row["fingerprint"])
                  for row in final_identities}
    if (outer_keys & final_keys
            or {row["seed"] for row in outer_identities}
                & {row["seed"] for row in final_identities}
            or {row["fingerprint"] for row in outer_identities}
                & {row["fingerprint"] for row in final_identities}
            or dict(sorted(Counter(
                row["family_id"] for row in outer_identities).items()))
                != FAMILY_QUOTAS
            or dict(sorted(Counter(
                row["family_id"] for row in final_identities).items()))
                != FAMILY_QUOTAS):
        raise ValueError("Consumed v12 outer and burned final identities overlap")
    if (consumed_outer.get("eligible_for_outer_or_final_reuse") is not False
            or burned_final.get(
                "post_closeout_deterministic_development_recollection")
                is not True
            or burned_final.get("retry_allowed") is not False
            or burned_final.get("protected_salt_reused") is not False):
        raise ValueError("V12 closeout disposition differs")
    combined = outer_identities + final_identities
    return {
        **base,
        "excluded_seeds": set(base["excluded_seeds"]) | {
            int(row["seed"]) for row in combined},
        "excluded_fingerprints": set(base["excluded_fingerprints"]) | {
            str(row["fingerprint"]) for row in combined},
        "v12_consumed_outer_identities": outer_identities,
        "v12_burned_final_identities": final_identities,
        "burned_v12_closeout": closeout,
        "consumed_v12_outer": consumed_outer,
        "burned_v12_final": burned_final,
        "paths": {**base["paths"], "burned_v12_closeout": closeout_path},
    }


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh replacement outer after burned one-shot v12 final",
        "selection_salt": SELECTION_SALT,
        "fixed_candidate_population": FIXED_CANDIDATE_SCENE_COUNT,
        "required_remaining_population": EXPECTED_REMAINING_SCENE_COUNT,
        "required_remaining_family_counts": deepcopy(
            EXPECTED_REMAINING_FAMILY_COUNTS),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "selection_inputs": [
            "seed", "scene fingerprint", "family", "exclusion membership",
            "fixed replay public observation hashes",
            "v8-v12 outer plus burned-v12-final observation-hash union",
        ],
        "consumed_v9_outer_permanently_excluded": True,
        "consumed_v10_outer_permanently_excluded": True,
        "consumed_v11_outer_permanently_excluded": True,
        "consumed_v12_outer_permanently_excluded": True,
        "burned_v12_final_permanently_excluded": True,
        "selection_uses_promoted_rows": False,
        "prior_validation_wins_projection": (
            "deduplicated v8, consumed-v9, consumed-v10, consumed-v11, "
            "consumed-v12-outer, and burned-v12-final one-way hashes"),
        "selection_uses_actor": "only to advance a fixed label-blind replay",
        "selection_uses_observations": "hash equality only",
        "selection_uses_actions_probabilities_program_or_metrics": False,
        "protected_final_access": False,
        "private_salt_access": False,
        "formal_ready": False,
    }


def _source_projection_content_sha256(projection: Mapping[str, Any]) -> str:
    value = projection.get("source_projection_content_sha256")
    if value is None:
        value = projection.get("content_sha256")
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Source projection content SHA-256 differs")
    return value


def _prior_validation_wins_projection(
    *, old_closeout: Mapping[str, Any],
    attempt_closeout: Mapping[str, Any],
    final_closeout: Mapping[str, Any],
    consumed_v11_closeout: Mapping[str, Any],
    consumed_v12_outer: Mapping[str, Any],
    burned_v12_final: Mapping[str, Any],
    burned_v12_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    old = _component_projection(old_closeout["consumed_outer"], "v8 outer")
    consumed_v9 = _component_projection(
        attempt_closeout["consumed_outer"], "v9 outer")
    consumed_v10 = _component_projection(
        final_closeout["consumed_outer"], "v10 outer")
    consumed_v11 = _component_projection(
        consumed_v11_closeout["consumed_outer"], "v11 outer")
    consumed_v12 = _component_projection(consumed_v12_outer, "v12 outer")
    burned_final = _component_projection(burned_v12_final, "burned v12 final")
    components = {
        "consumed_v8": old["outer_observation_hashes"],
        "consumed_v9": consumed_v9["outer_observation_hashes"],
        "consumed_v10": consumed_v10["outer_observation_hashes"],
        "consumed_v11": consumed_v11["outer_observation_hashes"],
        "consumed_v12_outer": consumed_v12["outer_observation_hashes"],
        "burned_v12_final": burned_final["outer_observation_hashes"],
    }
    combined = sorted(set().union(*(set(values) for values in components.values())))
    projection: dict[str, Any] = {
        "version": VERSION + ".validation-wins-exclusion.v1",
        "source_v8_closeout_content_sha256": old_closeout["content_sha256"],
        "source_v8_projection_content_sha256": _source_projection_content_sha256(old),
        "source_v9_closeout_content_sha256": attempt_closeout["content_sha256"],
        "source_v9_projection_content_sha256": _source_projection_content_sha256(consumed_v9),
        "source_v10_final_closeout_content_sha256": final_closeout["content_sha256"],
        "source_v10_projection_content_sha256": _source_projection_content_sha256(consumed_v10),
        "source_v11_closeout_content_sha256": consumed_v11_closeout["content_sha256"],
        "source_v11_projection_content_sha256": _source_projection_content_sha256(consumed_v11),
        "source_v12_final_closeout_content_sha256": burned_v12_closeout["content_sha256"],
        "source_v12_outer_projection_content_sha256": _source_projection_content_sha256(consumed_v12),
        "source_v12_final_projection_content_sha256": _source_projection_content_sha256(burned_final),
        "outer_observation_hashes": combined,
        "unique_outer_observation_hash_count": len(combined),
        "outer_observation_hashes_sha256": digest(combined),
        "component_unique_counts": {
            name: len(values) for name, values in components.items()},
        "component_hashes_sha256": {
            name: digest(values) for name, values in components.items()},
        "selector_rule": (
            "remove fit rows matching any prior outer observation hash before "
            "reading actions, probabilities, or raw observations"),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
        "selection_used_this_projection": True,
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    return projection


def _hash_screened_selection(
    *, remaining: Sequence[Mapping[str, Any]],
    prior_projection: Mapping[str, Any], closure: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    forbidden_values = prior_projection.get("outer_observation_hashes")
    if (not isinstance(forbidden_values, list)
            or forbidden_values != sorted(set(forbidden_values))):
        raise ValueError("Historical observation-hash projection differs")
    forbidden = set(forbidden_values)
    runtime = v12.v11.v9.manifest_binding.build_runtime(
        actor_path=closure["paths"]["actor"],
        protocol_path=closure["paths"]["protocol"],
        manifest_path=closure["screening_manifest"])
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in remaining:
        by_family[str(row["family_id"])].append(row)
    selected: list[dict[str, Any]] = []
    scenes: list[dict[str, Any]] = []
    selected_audit: list[dict[str, Any]] = []
    rejected_by_family: dict[str, int] = {}
    screened_by_family: dict[str, int] = {}
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (
            _rank(row), str(row["fingerprint"])))
        accepted = rejected = screened = 0
        for raw in ordered:
            if accepted >= FAMILY_QUOTAS[family]:
                break
            slot = len(selected)
            scene = _materialize_scene(raw, slot)
            rows, _environment_steps = v12.v11.projection_v10.rows_v7._collect(
                runtime, [scene], scene_offset=SCENE_OFFSET + slot,
                dense_critical=False, progress_label=None)
            hashes = [v12.v11.projection_v10.rows_v7.legacy._obs_hash(
                row["observation"]) for row in rows]
            overlap = set(hashes) & forbidden
            screened += 1
            del rows
            if overlap:
                rejected += 1
                continue
            public = deepcopy(dict(raw))
            selected.append(public)
            scenes.append(scene)
            selected_audit.append({
                "family_id": family,
                "fingerprint": public["fingerprint"],
                "seed": public["seed"],
                "row_count": len(hashes),
                "unique_observation_count": len(set(hashes)),
                "observation_hashes_sha256": digest(hashes),
                "historical_observation_overlap": 0,
            })
            accepted += 1
        if accepted != FAMILY_QUOTAS[family]:
            raise ValueError("Insufficient hash-isolated v13 identities for " + family)
        rejected_by_family[family] = rejected
        screened_by_family[family] = screened
    audit = {
        "screened_scene_count": sum(screened_by_family.values()),
        "rejected_historical_observation_overlap_scene_count": sum(
            rejected_by_family.values()),
        "screened_by_family": screened_by_family,
        "rejected_by_family": rejected_by_family,
        "selected_scene_count": len(selected),
        "selected_historical_observation_overlap": 0,
        "historical_unique_observation_count": len(forbidden),
        "historical_observation_hashes_sha256": digest(forbidden_values),
        "selected": selected_audit,
    }
    audit["content_sha256"] = digest(audit)
    return selected, scenes, audit


def _required_sha(component: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = component.get(name)
        if type(value) is str and _HEX.fullmatch(value) is not None:
            return value
    raise ValueError("Required SHA-256 field missing: " + "/".join(names))


def create_registry(**kwargs: Any) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = producer_sources()
    closure = replay_exclusion_closure(**kwargs)
    remaining, _identity_selection, remaining_counts = _remaining_and_selected(
        closure["candidates"], excluded_seeds=closure["excluded_seeds"],
        excluded_fingerprints=closure["excluded_fingerprints"])
    closeout = closure["burned_v12_closeout"]
    consumed_v12 = closure["consumed_v12_outer"]
    burned_final = closure["burned_v12_final"]
    prior_projection = _prior_validation_wins_projection(
        old_closeout=closure["old_closeout"],
        attempt_closeout=closure["attempt_closeout"],
        final_closeout=closure["burned_final_closeout"],
        consumed_v11_closeout=closure["consumed_v11_closeout"],
        consumed_v12_outer=consumed_v12,
        burned_v12_final=burned_final,
        burned_v12_closeout=closeout)
    selected, scenes, hash_screen = _hash_screened_selection(
        remaining=remaining, prior_projection=prior_projection,
        closure=closure)
    selected_public = [_public_identity(row) for row in selected]
    if (any(row["seed"] in closure["excluded_seeds"]
            or row["fingerprint"] in closure["excluded_fingerprints"]
            for row in selected_public)
            or len({row["seed"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT):
        raise RuntimeError("V13 outer overlaps an exposed identity")
    if [(row["seed"], row["fingerprint"]) for row in scenes] != [
            (row["seed"], row["fingerprint"]) for row in selected]:
        raise RuntimeError("V13 outer materialisation changed frozen set")
    prior_projection_sha = sha256(
        (canonical(prior_projection) + "\n").encode("utf-8")).hexdigest()
    paths = closure["paths"]
    attempted = closure["attempt_closeout"]
    burned_v10 = closure["burned_final_closeout"]
    consumed_v11 = closure["consumed_v11_closeout"]
    v10_projection = burned_v10["consumed_outer"]["observation_hash_projection"]
    v11_projection = consumed_v11["consumed_outer"]["observation_hash_projection"]
    v12_projection = _component_projection(consumed_v12, "v12 outer")
    final_projection = _component_projection(burned_final, "burned v12 final")
    combined = _closeout_component(
        closeout, "combined_promoted_development", "combined promoted development")
    combined_projection = _component_projection(
        combined, "combined promoted development")
    closeout_file_sha = kwargs["expected_burned_v12_final_closeout_sha256"]
    closeout_content_sha = closeout["content_sha256"]
    bindings = {
        "actor_sha256": v12.v11.v9.manifest_binding.FROZEN_ACTOR_SHA256,
        "actor_parameters_sha256": v12.v11.v9.manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256,
        "protocol_sha256": file_hash(paths["protocol"]),
        "designation_sha256": file_hash(paths["designation"]),
        "manifest_file_sha256": file_hash(paths["manifest"]),
        "manifest_validation_file_sha256": file_hash(paths["manifest_validation"]),
        "original_expansion_file_sha256": file_hash(paths["original"]),
        "failed_rows_file_sha256": file_hash(paths["rows"]),
        "consumed_outer_registry_file_sha256": file_hash(paths["current"]),
        "consumed_outer_report_file_sha256": file_hash(paths["current_report"]),
        "failure_closeout_file_sha256": kwargs["expected_failure_closeout_sha256"],
        "failure_closeout_content_sha256": closure["old_closeout"]["content_sha256"],
        "consumed_v9_attempt_closeout_file_sha256": kwargs[
            "expected_consumed_v9_attempt_closeout_sha256"],
        "consumed_v9_attempt_closeout_content_sha256": attempted["content_sha256"],
        "consumed_v9_selected_identity_sha256": attempted["consumed_outer"][
            "identities_sha256"],
        "burned_v10_final_closeout_file_sha256": kwargs[
            "expected_burned_v10_final_closeout_sha256"],
        "burned_v10_final_closeout_content_sha256": burned_v10["content_sha256"],
        "consumed_v10_selected_identity_sha256": burned_v10["consumed_outer"][
            "identities_sha256"],
        "consumed_v10_outer_observation_hashes_sha256": v10_projection[
            "outer_observation_hashes_sha256"],
        "consumed_v11_outer_closeout_file_sha256": kwargs[
            "expected_consumed_v11_outer_closeout_sha256"],
        "consumed_v11_outer_closeout_content_sha256": consumed_v11[
            "content_sha256"],
        "consumed_v11_outer_rows_sha256": consumed_v11["consumed_outer"][
            "rows_sha256"],
        "consumed_v11_outer_rows_semantic_sha256": consumed_v11[
            "consumed_outer"]["rows_semantic_sha256"],
        "consumed_v11_selected_identity_sha256": consumed_v11[
            "consumed_outer"]["identities_sha256"],
        "consumed_v11_outer_observation_hashes_sha256": v11_projection[
            "outer_observation_hashes_sha256"],
        "burned_v12_final_closeout_file_sha256": closeout_file_sha,
        "burned_v12_final_closeout_content_sha256": closeout_content_sha,
        "promotion_closeout_file_sha256": closeout_file_sha,
        "promotion_closeout_content_sha256": closeout_content_sha,
        "consumed_v12_outer_rows_sha256": _required_sha(
            consumed_v12, "rows_sha256"),
        "consumed_v12_outer_rows_semantic_sha256": _required_sha(
            consumed_v12, "rows_semantic_sha256"),
        "consumed_v12_outer_selected_identity_sha256": _required_sha(
            consumed_v12, "identities_sha256"),
        "consumed_v12_outer_observation_hashes_sha256": _required_sha(
            v12_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
        "burned_v12_final_selected_identity_sha256": _required_sha(
            burned_final, "identities_sha256"),
        "burned_v12_final_observation_hashes_sha256": _required_sha(
            final_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
        "combined_promoted_rows_sha256": _required_sha(
            combined, "rows_sha256"),
        "combined_promoted_rows_semantic_sha256": _required_sha(
            combined, "rows_semantic_sha256"),
        "combined_promoted_observation_hashes_sha256": _required_sha(
            combined_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
        "formal_selection_file_sha256": file_hash(paths["formal"]),
        "previous_development_file_sha256": file_hash(paths["previous"]),
        "retired_projection_file_sha256": file_hash(paths["retired"]),
        "retired_projection_report_file_sha256": file_hash(paths["retired_report"]),
        "outer_observation_hash_projection_file_sha256": prior_projection_sha,
        "outer_observation_hash_projection_content_sha256": prior_projection[
            "content_sha256"],
        "contract_sha256": digest(contract()),
    }
    if set(bindings) != RECOVERY_BINDINGS:
        raise RuntimeError("V13 recovery binding closure differs")
    exclusion_counts = {
        "source_row_scene_fingerprints": len(closure["row_fingerprints"]),
        "source_rows_in_fixed_candidate_population": len(
            closure["row_candidate_fingerprints"]),
        "original_expansion_trace_identities": len(closure["trace_fingerprints"]),
        "consumed_outer_identities": len(closure["current_identities"]),
        "consumed_v9_outer_identities": len(closure["v9_consumed_identities"]),
        "consumed_v10_outer_identities": len(closure["v10_consumed_identities"]),
        "consumed_v11_outer_identities": len(closure["v11_consumed_identities"]),
        "consumed_v12_outer_identities": len(
            closure["v12_consumed_outer_identities"]),
        "burned_v12_final_identities": len(
            closure["v12_burned_final_identities"]),
        "formal_xy_identities": len(closure["formal_fingerprints"]),
        "previous_development_identities": len(closure["previous_fingerprints"]),
        "retired_exposed_identities": len(closure["retired_fingerprints"]),
        "union_candidate_identities_excluded": (
            len(closure["candidates"]) - len(remaining)),
    }
    exclusion_digests = {
        "excluded_seeds_sha256": digest(sorted(closure["excluded_seeds"])),
        "excluded_scene_fingerprints_sha256": digest(sorted(
            closure["excluded_fingerprints"])),
        "source_row_scene_fingerprints_sha256": digest(sorted(
            closure["row_fingerprints"])),
        "original_expansion_trace_fingerprints_sha256": digest(sorted(
            closure["trace_fingerprints"])),
        "consumed_outer_identities_sha256": digest(closure["current_identities"]),
        "consumed_v9_outer_identities_sha256": digest(
            closure["v9_consumed_identities"]),
        "consumed_v10_outer_identities_sha256": digest(
            closure["v10_consumed_identities"]),
        "consumed_v10_outer_observation_hashes_sha256": v10_projection[
            "outer_observation_hashes_sha256"],
        "consumed_v11_outer_identities_sha256": digest(
            closure["v11_consumed_identities"]),
        "consumed_v11_outer_observation_hashes_sha256": v11_projection[
            "outer_observation_hashes_sha256"],
        "consumed_v12_outer_identities_sha256": digest(
            closure["v12_consumed_outer_identities"]),
        "consumed_v12_outer_observation_hashes_sha256": bindings[
            "consumed_v12_outer_observation_hashes_sha256"],
        "burned_v12_final_identities_sha256": digest(
            closure["v12_burned_final_identities"]),
        "burned_v12_final_observation_hashes_sha256": bindings[
            "burned_v12_final_observation_hashes_sha256"],
        "retired_exposed_fingerprints_sha256": digest(sorted(
            closure["retired_fingerprints"])),
    }
    statistics = {
        "fixed_candidate_scene_count": len(closure["candidates"]),
        "remaining_candidate_scene_count": len(remaining),
        "remaining_family_counts": remaining_counts,
        "selected_outer_scene_count": len(scenes),
        "selected_outer_family_counts": dict(sorted(Counter(
            row["family_id"] for row in scenes).items())),
        "selected_exposed_seed_overlap": 0,
        "selected_exposed_fingerprint_overlap": 0,
        "hash_screen": deepcopy(hash_screen),
        "combined_promoted_unique_observation_count": len(
            combined_projection["outer_observation_hashes"]),
        **exclusion_counts,
    }
    boundary = deepcopy(RECOVERY_BOUNDARY)
    registry: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": bindings,
        "development_outer": scenes,
        "selected_outer_identities": selected_public,
        "exclusion_counts": exclusion_counts,
        "exclusion_digests": exclusion_digests,
        "statistics": statistics,
        "information_boundary": boundary,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "formal_ready": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
    }
    registry["content_sha256"] = digest(registry)
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "status": STATUS,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": deepcopy(bindings),
        "selection": {
            "salt": SELECTION_SALT,
            "family_quotas": deepcopy(FAMILY_QUOTAS),
            "selected_identity_sha256": digest(selected_public),
            "fixed_remaining_identity_sha256": digest([
                _public_identity(row) for row in sorted(
                    remaining, key=lambda item: (
                        item["family_id"], item["batch_index"],
                        item["family_offset"]))]),
            "exclusion_digests": deepcopy(exclusion_digests),
        },
        "statistics": deepcopy(statistics),
        "information_boundary": deepcopy(boundary),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    if producer_sources() != sources:
        raise RuntimeError("v13 registry source closure changed during selection")
    return registry, report, prior_projection


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    registry, report, projection = create_registry(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v13 registry output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v13 registry output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / "outer_observation_hashes.json", projection)
        if file_hash(temporary / "outer_observation_hashes.json") != report[
                "bindings"]["outer_observation_hash_projection_file_sha256"]:
            raise RuntimeError("v13 prior projection bytes differ")
        _write_exclusive(temporary / "development_outer.json", registry)
        report["registry_file_sha256"] = file_hash(
            temporary / "development_outer.json")
        report["content_sha256"] = digest({
            key: child for key, child in report.items()
            if key != "content_sha256"})
        _write_exclusive(temporary / "report.json", report)
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(report)


def read_saved_registry(
    registry_path: str | Path, report_path: str | Path, *,
    expected_registry_sha256: str, expected_report_sha256: str,
    burned_v12_final_closeout_path: str | Path,
    expected_burned_v12_final_closeout_sha256: str,
    permanent_v12_final_closeout_registry: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry_file, registry = v12.v11.v9._strict_json(
        registry_path, "saved v13 outer registry",
        expected_sha256=expected_registry_sha256)
    report_file, report = v12.v11.v9._strict_json(
        report_path, "saved v13 outer registry report",
        expected_sha256=expected_report_sha256)
    closeout = burned_v12_api.read_saved_closeout_public(
        burned_v12_final_closeout_path,
        expected_closeout_sha256=expected_burned_v12_final_closeout_sha256,
        permanent_closeout_registry=permanent_v12_final_closeout_registry)
    consumed_v12 = _closeout_component(
        closeout, "consumed_v12_outer", "consumed v12 outer")
    burned_final = _closeout_component(
        closeout, "burned_final", "burned v12 final")
    combined = _closeout_component(
        closeout, "combined_promoted_development", "combined promoted development")
    outer_identities = _component_identities(consumed_v12, "consumed v12 outer")
    final_identities = _component_identities(burned_final, "burned v12 final")
    v12_projection = _component_projection(consumed_v12, "consumed v12 outer")
    final_projection = _component_projection(burned_final, "burned v12 final")
    combined_projection = _component_projection(
        combined, "combined promoted development")
    scenes, identities = (registry.get("development_outer"),
                          registry.get("selected_outer_identities"))
    if (registry.get("version") != VERSION or registry.get("status") != STATUS
            or not v12.v11.v9._content_valid(registry)
            or registry.get("contract") != contract()
            or registry.get("formal_ready") is not False
            or any(registry.get(name) is not False for name in (
                "program_access", "program_predictions_access",
                "action_labels_access", "probabilities_access",
                "final_audit_rows_access"))
            or not isinstance(scenes, list)
            or len(scenes) != FRESH_OUTER_SCENE_COUNT
            or not isinstance(identities, list)
            or len(identities) != FRESH_OUTER_SCENE_COUNT
            or registry.get("producer_sources_sha256")
                != digest(dict(sorted(
                    registry.get("producer_sources", {}).items())))):
        raise ValueError("Saved v13 outer registry semantics differ")
    public = []
    for index, (scene, raw) in enumerate(zip(scenes, identities)):
        if not isinstance(raw, Mapping) or set(raw) != {
                "batch_index", "family_id", "seed", "fingerprint"}:
            raise ValueError("Saved v13 outer identity fields differ")
        identity = _public_identity(raw)
        v12.v11.v9._identity(identity, "saved v13 outer identity")
        if (not isinstance(scene, Mapping)
                or scene.get("id")
                    != f"diagnostic_v13_fresh_outer_{index:04d}"
                or any(scene.get(name) != identity[name]
                       for name in ("family_id", "seed", "fingerprint"))):
            raise ValueError("Saved v13 outer materialisation differs")
        public.append(identity)
    bindings = registry.get("bindings")
    stats = registry.get("statistics")
    boundary = registry.get("information_boundary")
    expected_bindings = {
        "burned_v12_final_closeout_file_sha256": (
            expected_burned_v12_final_closeout_sha256),
        "burned_v12_final_closeout_content_sha256": closeout["content_sha256"],
        "promotion_closeout_file_sha256": (
            expected_burned_v12_final_closeout_sha256),
        "promotion_closeout_content_sha256": closeout["content_sha256"],
        "consumed_v12_outer_rows_sha256": _required_sha(
            consumed_v12, "rows_sha256"),
        "consumed_v12_outer_rows_semantic_sha256": _required_sha(
            consumed_v12, "rows_semantic_sha256"),
        "consumed_v12_outer_selected_identity_sha256": _required_sha(
            consumed_v12, "identities_sha256"),
        "consumed_v12_outer_observation_hashes_sha256": _required_sha(
            v12_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
        "burned_v12_final_selected_identity_sha256": _required_sha(
            burned_final, "identities_sha256"),
        "burned_v12_final_observation_hashes_sha256": _required_sha(
            final_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
        "combined_promoted_rows_sha256": _required_sha(
            combined, "rows_sha256"),
        "combined_promoted_rows_semantic_sha256": _required_sha(
            combined, "rows_semantic_sha256"),
        "combined_promoted_observation_hashes_sha256": _required_sha(
            combined_projection, "outer_observation_hashes_sha256",
            "unique_observation_hashes_sha256"),
    }
    if (not isinstance(bindings, Mapping) or set(bindings) != RECOVERY_BINDINGS
            or any(bindings.get(name) != value
                   for name, value in expected_bindings.items())
            or bindings.get("contract_sha256") != digest(contract())
            or not isinstance(stats, Mapping)
            or stats.get("remaining_candidate_scene_count")
                != EXPECTED_REMAINING_SCENE_COUNT
            or stats.get("remaining_family_counts")
                != EXPECTED_REMAINING_FAMILY_COUNTS
            or stats.get("consumed_v12_outer_identities")
                != FRESH_OUTER_SCENE_COUNT
            or stats.get("burned_v12_final_identities")
                != FRESH_OUTER_SCENE_COUNT
            or stats.get("union_candidate_identities_excluded") != 785
            or stats.get("combined_promoted_unique_observation_count")
                != len(combined_projection["outer_observation_hashes"])
            or stats.get("selected_outer_scene_count")
                != FRESH_OUTER_SCENE_COUNT
            or stats.get("selected_outer_family_counts") != FAMILY_QUOTAS
            or stats.get("selected_exposed_seed_overlap") != 0
            or stats.get("selected_exposed_fingerprint_overlap") != 0
            or boundary != RECOVERY_BOUNDARY
            or stats.get("hash_screen", {}).get(
                "selected_historical_observation_overlap") != 0):
        raise ValueError("Saved v13 exclusion closure differs")
    selection = report.get("selection")
    if (report.get("version") != REPORT_VERSION
            or report.get("status") != STATUS
            or not v12.v11.v9._content_valid(report)
            or report.get("formal_ready") is not False
            or report.get("registry_file_sha256") != file_hash(registry_file)
            or report.get("registry_content_sha256")
                != registry["content_sha256"]
            or report.get("bindings") != bindings
            or report.get("statistics") != stats
            or report.get("information_boundary") != boundary
            or report.get("producer_sources")
                != registry.get("producer_sources")
            or not isinstance(selection, Mapping)
            or selection.get("salt") != SELECTION_SALT
            or selection.get("family_quotas") != FAMILY_QUOTAS
            or selection.get("selected_identity_sha256") != digest(public)
            or selection.get("exclusion_digests")
                != registry.get("exclusion_digests")
            or file_hash(report_file) != expected_report_sha256):
        raise ValueError("Saved v13 registry report differs")
    consumed_keys = {(row["seed"], row["fingerprint"])
                     for row in outer_identities + final_identities}
    if consumed_keys & {(row["seed"], row["fingerprint"])
                        for row in public}:
        raise ValueError("Saved v13 registry reuses a v12 exposed identity")
    return deepcopy(registry), deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "actor", "protocol", "designation", "manifest", "failed-rows",
        "original-expansion", "current-outer-registry",
        "current-outer-report", "failure-closeout",
        "expected-failure-closeout-sha256", "permanent-closeout-registry",
        "formal-selection", "previous-development", "retired-projection",
        "consumed-v9-attempt-closeout",
        "expected-consumed-v9-attempt-closeout-sha256",
        "permanent-v9-attempt-registry", "burned-v10-final-closeout",
        "expected-burned-v10-final-closeout-sha256",
        "permanent-v10-final-closeout-registry",
        "consumed-v11-outer-closeout",
        "expected-consumed-v11-outer-closeout-sha256",
        "permanent-v11-outer-attempt-registry",
        "burned-v12-final-closeout",
        "expected-burned-v12-final-closeout-sha256",
        "permanent-v12-final-closeout-registry", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    report = build(
        actor_path=args.actor, protocol_path=args.protocol,
        designation_path=args.designation, manifest_path=args.manifest,
        failed_rows_path=args.failed_rows,
        original_expansion_path=args.original_expansion,
        current_outer_registry_path=args.current_outer_registry,
        current_outer_report_path=args.current_outer_report,
        failure_closeout_path=args.failure_closeout,
        expected_failure_closeout_sha256=args.expected_failure_closeout_sha256,
        permanent_closeout_registry=args.permanent_closeout_registry,
        formal_selection_path=args.formal_selection,
        previous_development_path=args.previous_development,
        retired_projection_path=args.retired_projection,
        consumed_v9_attempt_closeout_path=args.consumed_v9_attempt_closeout,
        expected_consumed_v9_attempt_closeout_sha256=(
            args.expected_consumed_v9_attempt_closeout_sha256),
        permanent_v9_attempt_registry=args.permanent_v9_attempt_registry,
        burned_v10_final_closeout_path=args.burned_v10_final_closeout,
        expected_burned_v10_final_closeout_sha256=(
            args.expected_burned_v10_final_closeout_sha256),
        permanent_v10_final_closeout_registry=(
            args.permanent_v10_final_closeout_registry),
        consumed_v11_outer_closeout_path=args.consumed_v11_outer_closeout,
        expected_consumed_v11_outer_closeout_sha256=(
            args.expected_consumed_v11_outer_closeout_sha256),
        permanent_v11_outer_attempt_registry=(
            args.permanent_v11_outer_attempt_registry),
        burned_v12_final_closeout_path=args.burned_v12_final_closeout,
        expected_burned_v12_final_closeout_sha256=(
            args.expected_burned_v12_final_closeout_sha256),
        permanent_v12_final_closeout_registry=(
            args.permanent_v12_final_closeout_registry),
        output=args.output,
    )
    print(canonical({
        "status": report["status"],
        "registry": str(Path(args.output).resolve() / "development_outer.json"),
        "selected_identity_sha256": report["selection"][
            "selected_identity_sha256"],
        "remaining_candidate_scene_count": report["statistics"][
            "remaining_candidate_scene_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "REPORT_VERSION", "STATUS", "SELECTION_SALT", "FAMILY_IDS",
    "FAMILY_QUOTAS", "FRESH_OUTER_SCENE_COUNT", "SOURCE_ROW_SCENE_COUNT",
    "FIXED_CANDIDATE_SCENE_COUNT", "SCENE_OFFSET",
    "EXPECTED_REMAINING_SCENE_COUNT", "EXPECTED_REMAINING_FAMILY_COUNTS",
    "RECOVERY_BOUNDARY", "RECOVERY_BINDINGS", "_RECOVERY_BOUNDARY",
    "_RECOVERY_BINDINGS", "contract", "producer_sources",
    "replay_exclusion_closure", "create_registry", "build",
    "read_saved_registry", "main", "_remaining_and_selected",
    "_materialize_scene", "_prior_validation_wins_projection",
    "_hash_screened_selection",
]
