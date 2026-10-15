"""Freeze the v11 outer set after the v10 final attempt was burned.

The replacement is selected from the same fixed 2,160-scene identity
population.  It excludes every previously exposed identity, including the 64
v10 outer scenes, then screens the fixed, salted candidate order by replaying
the frozen Actor.  Screening compares only one-way observation hashes against
the consumed v10 outer projection.  The selection code never inspects or
publishes replay actions, probabilities, program predictions, or scores.
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

from backend.training import warehouse_r41_diagnostic_outer_attempt_closeout_v10 as attempt_closeout_api
from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_v11 as final_closeout_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_split as v9
from backend.training import warehouse_r41_diagnostic_rcpd_v10_outer_split as v10
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v10 as projection_v10
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticConflictWarehouseEnv
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
)


VERSION = "warehouse-r41-diagnostic-rcpd-v11-fresh-outer-registry.v1"
REPORT_VERSION = VERSION
STATUS = "frozen_hash_screened_pending_outer_collection"
ROOT = Path(__file__).resolve().parents[2]
SELECTION_SALT = "warehouse-r41-v11-fresh-development-outer-20260914-v1"
FAMILY_IDS = tuple(v9.FAMILY_IDS)
FAMILY_QUOTAS = dict(v9.FAMILY_QUOTAS)
FRESH_OUTER_SCENE_COUNT = 64
SOURCE_ROW_SCENE_COUNT = v9.SOURCE_ROW_SCENE_COUNT
FIXED_CANDIDATE_SCENE_COUNT = v9.FIXED_CANDIDATE_SCENE_COUNT
EXPECTED_REMAINING_SCENE_COUNT = 1567
EXPECTED_REMAINING_FAMILY_COUNTS = {
    "conflict_family_01": 257,
    "conflict_family_02": 258,
    "conflict_family_03": 260,
    "conflict_family_04": 257,
    "conflict_family_05": 267,
    "conflict_family_06": 268,
}
SCENE_OFFSET = 576
MAX_JSON_BYTES = v9.MAX_JSON_BYTES
_HEX = re.compile(r"[0-9a-f]{64}\Z")

RECOVERY_BOUNDARY = {
    "consumed_v9_attempt_closed_before_selection": True,
    "burned_v10_final_closed_before_selection": True,
    "consumed_v9_identities_excluded": True,
    "consumed_v10_identities_excluded": True,
    "consumed_v10_observation_hashes_excluded_by_full_replay": True,
    "all_previously_exposed_identities_excluded": True,
    "candidate_identity_rank_frozen_before_hash_replay": True,
    "selected_identities_frozen_after_hash_screen": True,
    "failed_rows_fields_read": ["scene_fingerprints"],
    "consumed_v9_rows_opened": False,
    "consumed_v10_rows_opened": False,
    "candidate_replay_observation_hashes_read": True,
    "candidate_replay_raw_observations_retained_or_published": False,
    "candidate_replay_actions_or_probabilities_inspected": False,
    "candidate_replay_actions_or_probabilities_published": False,
    "observation_hash_projection_used_for_identity_selection": True,
    "prior_v10_observation_hashes_used_as_exclusion_set": True,
    "actor_loaded_only_for_fixed_hash_replay": True,
    "program_predictions_or_metrics_read_for_selection": False,
    "protected_final_access": False,
    "private_salt_access": False,
    "formal_ready": False,
}
_RECOVERY_BOUNDARY = RECOVERY_BOUNDARY
RECOVERY_BINDINGS = frozenset((
    "actor_sha256", "actor_parameters_sha256", "protocol_sha256",
    "designation_sha256", "manifest_file_sha256",
    "manifest_validation_file_sha256", "original_expansion_file_sha256",
    "failed_rows_file_sha256", "consumed_outer_registry_file_sha256",
    "consumed_outer_report_file_sha256", "failure_closeout_file_sha256",
    "failure_closeout_content_sha256",
    "consumed_v9_attempt_closeout_file_sha256",
    "consumed_v9_attempt_closeout_content_sha256",
    "consumed_v9_selected_identity_sha256",
    "burned_v10_final_closeout_file_sha256",
    "burned_v10_final_closeout_content_sha256",
    "consumed_v10_selected_identity_sha256",
    "consumed_v10_outer_observation_hashes_sha256",
    "formal_selection_file_sha256", "previous_development_file_sha256",
    "retired_projection_file_sha256", "retired_projection_report_file_sha256",
    "outer_observation_hash_projection_file_sha256",
    "outer_observation_hash_projection_content_sha256", "contract_sha256",
))
_RECOVERY_BINDINGS = RECOVERY_BINDINGS

# The still-valid development and earlier closeout remain unchanged.
EXPECTED_MANIFEST_SHA256 = v9.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = v9.EXPECTED_MANIFEST_VALIDATION_SHA256
EXPECTED_FAILED_ROWS_SHA256 = v9.EXPECTED_FAILED_ROWS_SHA256
EXPECTED_ORIGINAL_EXPANSION_SHA256 = v9.EXPECTED_ORIGINAL_EXPANSION_SHA256
EXPECTED_CURRENT_OUTER_REGISTRY_SHA256 = v9.EXPECTED_CURRENT_OUTER_REGISTRY_SHA256
EXPECTED_CURRENT_OUTER_REPORT_SHA256 = v9.EXPECTED_CURRENT_OUTER_REPORT_SHA256
EXPECTED_FORMAL_SELECTION_SHA256 = v9.EXPECTED_FORMAL_SELECTION_SHA256
EXPECTED_PREVIOUS_DEVELOPMENT_SHA256 = v9.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256
EXPECTED_RETIRED_PROJECTION_SHA256 = v9.EXPECTED_RETIRED_PROJECTION_SHA256
EXPECTED_RETIRED_REPORT_SHA256 = v9.EXPECTED_RETIRED_REPORT_SHA256


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
        seed, fingerprint = v9._identity(raw, "fixed v11 candidate")
        family = raw.get("family_id")
        batch_index = raw.get("batch_index")
        family_offset = raw.get("family_offset")
        if (family not in FAMILY_IDS or type(batch_index) is not int
                or type(family_offset) is not int):
            raise ValueError("Fixed v11 candidate identity differs")
        if seed not in excluded_seeds and fingerprint not in excluded_fingerprints:
            by_family[str(family)].append(deepcopy(dict(raw)))
    remaining = [row for family in FAMILY_IDS for row in by_family[family]]
    counts = dict(sorted(Counter(row["family_id"] for row in remaining).items()))
    if (len(remaining) != EXPECTED_REMAINING_SCENE_COUNT
            or counts != EXPECTED_REMAINING_FAMILY_COUNTS):
        raise ValueError("Fixed replacement candidate remaining population differs")
    selected: list[dict[str, Any]] = []
    for family in FAMILY_IDS:
        ordered = sorted(by_family[family], key=lambda row: (
            _rank(row), row["fingerprint"]))
        quota = FAMILY_QUOTAS[family]
        if len(ordered) < quota:
            raise ValueError("Insufficient replacement identities for " + family)
        selected.extend(ordered[:quota])
    selected.sort(key=lambda row: (FAMILY_IDS.index(row["family_id"]), _rank(row)))
    if (len(selected) != FRESH_OUTER_SCENE_COUNT
            or dict(sorted(Counter(row["family_id"] for row in selected).items()))
                != FAMILY_QUOTAS):
        raise RuntimeError("Replacement outer family quota differs")
    return remaining, selected, counts


def _materialize_scene(identity: Mapping[str, Any], index: int) -> dict[str, Any]:
    seed, expected_fingerprint = v9._identity(identity, "selected v11 outer")
    batch_index, family = identity.get("batch_index"), identity.get("family_id")
    if type(batch_index) is not int or family not in FAMILY_IDS:
        raise ValueError("Selected v11 outer batch identity differs")
    env = R41DiagnosticConflictWarehouseEnv()
    v9.retired_api._identity_only_reset(env, seed=seed)
    reset_family = conflict_family_id(env.state.tasks)
    if reset_family != family:
        raise ValueError("Selected v11 outer family differs at reset")
    v9.scenes_api._candidate_start_state(
        env, seed=seed, batch_index=batch_index, family_id=reset_family)
    if diagnostic_scene_fingerprint(env) != expected_fingerprint:
        raise ValueError("Selected v11 outer fingerprint differs")
    scene = v9.scenes_api._scene_from_environment(
        env, scene_id=f"diagnostic_v11_fresh_outer_{index:04d}",
        split="development_outer", seed=seed, batch_index=batch_index)
    if (scene.get("fingerprint") != expected_fingerprint
            or scene.get("family_id") != family):
        raise ValueError("Materialised v11 outer identity differs")
    return scene


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
) -> dict[str, Any]:
    """Replay prior exclusions and add the passed/burned v10 campaign."""
    base = v10.replay_exclusion_closure(
        manifest_path=manifest_path, failed_rows_path=failed_rows_path,
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
        permanent_v9_attempt_registry=permanent_v9_attempt_registry)
    closeout_file = v9._regular(
        burned_v10_final_closeout_path, "burned v10 final closeout",
        maximum=MAX_JSON_BYTES)
    closeout = final_closeout_api.read_saved_closeout(
        closeout_file,
        expected_closeout_sha256=expected_burned_v10_final_closeout_sha256,
        permanent_closeout_registry=permanent_v10_final_closeout_registry)
    v10_identities = [
        _public_identity(row) for row in closeout["consumed_outer"]["identities"]]
    candidates = base["candidates"]
    candidate_public = {
        (row["seed"], row["fingerprint"]): _public_identity(row)
        for row in candidates}
    if (len(v10_identities) != FRESH_OUTER_SCENE_COUNT
            or any(candidate_public.get(
                (row["seed"], row["fingerprint"])) != row
                for row in v10_identities)):
        raise ValueError("Consumed v10 identities are not exact fixed candidates")
    excluded_seeds = set(base["excluded_seeds"]) | {
        int(row["seed"]) for row in v10_identities}
    excluded_fingerprints = set(base["excluded_fingerprints"]) | {
        str(row["fingerprint"]) for row in v10_identities}
    actor = v9._authenticate_opaque(
        actor_path, "frozen v11 screening Actor",
        expected_sha256=v9.manifest_binding.FROZEN_ACTOR_SHA256)
    protocol = v9._authenticate_opaque(
        protocol_path, "frozen v11 screening protocol",
        expected_sha256=
            "374ae398115672a243bd2917937ff056fdc762499b470bf11ad44f5fdecc13c8")
    designation = v9._authenticate_opaque(
        designation_path, "frozen v11 Actor designation",
        expected_sha256=
            "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815")
    manifest = base["paths"]["manifest"]
    return {
        **base, "excluded_seeds": excluded_seeds,
        "excluded_fingerprints": excluded_fingerprints,
        "v10_consumed_identities": v10_identities,
        "burned_final_closeout": closeout,
        "paths": {**base["paths"], "actor": actor, "protocol": protocol,
                  "designation": designation,
                  "burned_final_closeout": closeout_file},
        "screening_manifest": manifest,
    }

def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "purpose": "fresh replacement outer after burned v10 final attempt",
        "selection_salt": SELECTION_SALT,
        "fixed_candidate_population": FIXED_CANDIDATE_SCENE_COUNT,
        "required_remaining_population": EXPECTED_REMAINING_SCENE_COUNT,
        "required_remaining_family_counts": deepcopy(
            EXPECTED_REMAINING_FAMILY_COUNTS),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "selection_inputs": [
            "seed", "scene fingerprint", "family", "exclusion membership",
            "fixed replay public observation hashes",
            "consumed v10 outer observation-hash exclusion"],
        "consumed_v9_outer_permanently_excluded": True,
        "consumed_v10_outer_permanently_excluded": True,
        "burned_v10_final_proven_pre_secret": True,
        "selection_uses_consumed_v9_rows": False,
        "selection_uses_consumed_v10_rows": False,
        "prior_validation_wins_projection": (
            "deduplicated v8, consumed-v9, and consumed-v10 one-way hashes"),
        "selection_uses_prior_validation_wins_projection": False,
        "selection_uses_actor": "only to advance a fixed label-blind replay",
        "selection_uses_observations": "hash equality only",
        "selection_uses_actions_probabilities_program_or_metrics": False,
        "protected_final_access": False,
        "formal_ready": False,
    }


def _prior_validation_wins_projection(
    *, old_closeout: Mapping[str, Any],
    attempt_closeout: Mapping[str, Any],
    final_closeout: Mapping[str, Any],
) -> dict[str, Any]:
    old = old_closeout.get("consumed_outer", {}).get(
        "observation_hash_projection", {})
    consumed_v9 = attempt_closeout.get("consumed_outer", {}).get(
        "observation_hash_projection", {})
    consumed_v10 = final_closeout.get("consumed_outer", {}).get(
        "observation_hash_projection", {})
    components = {
        "consumed_v8": old.get("outer_observation_hashes"),
        "consumed_v9": consumed_v9.get("outer_observation_hashes"),
        "consumed_v10": consumed_v10.get("outer_observation_hashes"),
    }
    if any(not isinstance(values, list) or values != sorted(set(values))
           or any(type(item) is not str or _HEX.fullmatch(item) is None
                  for item in values) for values in components.values()):
        raise ValueError("Prior outer observation-hash projections differ")
    combined = sorted(set().union(*(set(values) for values in components.values())))
    projection: dict[str, Any] = {
        "version": VERSION + ".validation-wins-exclusion.v1",
        "source_v8_closeout_content_sha256": old_closeout["content_sha256"],
        "source_v8_projection_content_sha256": old["content_sha256"],
        "source_v9_closeout_content_sha256": attempt_closeout["content_sha256"],
        "source_v9_projection_content_sha256": consumed_v9[
            "source_projection_content_sha256"],
        "source_v10_final_closeout_content_sha256": final_closeout[
            "content_sha256"],
        "source_v10_projection_content_sha256": consumed_v10[
            "source_projection_content_sha256"],
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
        "actions_included": False, "probabilities_included": False,
        "labels_included": False, "selection_used_this_projection": False,
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    return projection

def _hash_screened_selection(
    *, remaining: Sequence[Mapping[str, Any]], closure: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select ranked scenes whose fixed replay is disjoint from v10 outer."""
    forbidden_values = closure["burned_final_closeout"]["consumed_outer"][
        "observation_hash_projection"]["outer_observation_hashes"]
    if (not isinstance(forbidden_values, list)
            or forbidden_values != sorted(set(forbidden_values))):
        raise ValueError("Consumed v10 observation-hash projection differs")
    forbidden = set(forbidden_values)
    runtime = v9.manifest_binding.build_runtime(
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
        accepted = 0
        rejected = 0
        screened = 0
        for raw in ordered:
            if accepted >= FAMILY_QUOTAS[family]:
                break
            slot = len(selected)
            scene = _materialize_scene(raw, slot)
            rows, _environment_steps = projection_v10.rows_v7._collect(
                runtime, [scene], scene_offset=SCENE_OFFSET + slot,
                dense_critical=False, progress_label=None)
            hashes = [projection_v10.rows_v7.legacy._obs_hash(
                row["observation"]) for row in rows]
            overlap = set(hashes) & forbidden
            screened += 1
            # No action/probability array is inspected; release references are
            # dropped before the next candidate is considered.
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
                "consumed_v10_observation_overlap": 0,
            })
            accepted += 1
        if accepted != FAMILY_QUOTAS[family]:
            raise ValueError("Insufficient hash-isolated identities for " + family)
        rejected_by_family[family] = rejected
        screened_by_family[family] = screened
    audit = {
        "screened_scene_count": sum(screened_by_family.values()),
        "rejected_v10_observation_overlap_scene_count": sum(
            rejected_by_family.values()),
        "screened_by_family": screened_by_family,
        "rejected_by_family": rejected_by_family,
        "selected_scene_count": len(selected),
        "selected_v10_observation_overlap": 0,
        "consumed_v10_unique_observation_count": len(forbidden),
        "consumed_v10_observation_hashes_sha256": digest(forbidden_values),
        "selected": selected_audit,
    }
    audit["content_sha256"] = digest(audit)
    return selected, scenes, audit


def create_registry(**kwargs: Any) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = producer_sources()
    closure = replay_exclusion_closure(**kwargs)
    remaining, _identity_only_selection, remaining_counts = _remaining_and_selected(
        closure["candidates"], excluded_seeds=closure["excluded_seeds"],
        excluded_fingerprints=closure["excluded_fingerprints"])
    selected, scenes, hash_screen = _hash_screened_selection(
        remaining=remaining, closure=closure)
    selected_public = [_public_identity(row) for row in selected]
    if (any(row["seed"] in closure["excluded_seeds"]
            or row["fingerprint"] in closure["excluded_fingerprints"]
            for row in selected_public)
            or len({row["seed"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT
            or len({row["fingerprint"] for row in selected_public})
                != FRESH_OUTER_SCENE_COUNT):
        raise RuntimeError("Replacement outer overlaps an exposed identity")
    if [(row["seed"], row["fingerprint"]) for row in scenes] != [
            (row["seed"], row["fingerprint"]) for row in selected]:
        raise RuntimeError("Replacement outer materialisation changed frozen set")
    projection = _prior_validation_wins_projection(
        old_closeout=closure["old_closeout"],
        attempt_closeout=closure["attempt_closeout"],
        final_closeout=closure["burned_final_closeout"])
    projection_file_sha256 = sha256(
        (canonical(projection) + "\n").encode("utf-8")).hexdigest()
    paths = closure["paths"]
    attempt_closeout = closure["attempt_closeout"]
    final_closeout = closure["burned_final_closeout"]
    consumed_v10_projection = final_closeout["consumed_outer"][
        "observation_hash_projection"]
    bindings = {
        "actor_sha256": v9.manifest_binding.FROZEN_ACTOR_SHA256,
        "actor_parameters_sha256": (
            v9.manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256),
        "protocol_sha256": file_hash(paths["protocol"]),
        "designation_sha256": file_hash(paths["designation"]),
        "manifest_file_sha256": file_hash(paths["manifest"]),
        "manifest_validation_file_sha256": file_hash(
            paths["manifest_validation"]),
        "original_expansion_file_sha256": file_hash(paths["original"]),
        "failed_rows_file_sha256": file_hash(paths["rows"]),
        "consumed_outer_registry_file_sha256": file_hash(paths["current"]),
        "consumed_outer_report_file_sha256": file_hash(paths["current_report"]),
        "failure_closeout_file_sha256": kwargs[
            "expected_failure_closeout_sha256"],
        "failure_closeout_content_sha256": closure["old_closeout"][
            "content_sha256"],
        "consumed_v9_attempt_closeout_file_sha256": kwargs[
            "expected_consumed_v9_attempt_closeout_sha256"],
        "consumed_v9_attempt_closeout_content_sha256": attempt_closeout[
            "content_sha256"],
        "consumed_v9_selected_identity_sha256": attempt_closeout[
            "consumed_outer"]["identities_sha256"],
        "burned_v10_final_closeout_file_sha256": kwargs[
            "expected_burned_v10_final_closeout_sha256"],
        "burned_v10_final_closeout_content_sha256": final_closeout[
            "content_sha256"],
        "consumed_v10_selected_identity_sha256": final_closeout[
            "consumed_outer"]["identities_sha256"],
        "consumed_v10_outer_observation_hashes_sha256": (
            consumed_v10_projection["outer_observation_hashes_sha256"]),
        "formal_selection_file_sha256": file_hash(paths["formal"]),
        "previous_development_file_sha256": file_hash(paths["previous"]),
        "retired_projection_file_sha256": file_hash(paths["retired"]),
        "retired_projection_report_file_sha256": file_hash(
            paths["retired_report"]),
        "outer_observation_hash_projection_file_sha256": (
            projection_file_sha256),
        "outer_observation_hash_projection_content_sha256": projection[
            "content_sha256"],
        "contract_sha256": digest(contract()),
    }
    exclusion_counts = {
        "source_row_scene_fingerprints": len(closure["row_fingerprints"]),
        "source_rows_in_fixed_candidate_population": len(
            closure["row_candidate_fingerprints"]),
        "original_expansion_trace_identities": len(
            closure["trace_fingerprints"]),
        "consumed_outer_identities": len(closure["current_identities"]),
        "consumed_v9_outer_identities": len(
            closure["v9_consumed_identities"]),
        "consumed_v10_outer_identities": len(
            closure["v10_consumed_identities"]),
        "formal_xy_identities": len(closure["formal_fingerprints"]),
        "previous_development_identities": len(
            closure["previous_fingerprints"]),
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
        "consumed_outer_identities_sha256": digest(
            closure["current_identities"]),
        "consumed_v9_outer_identities_sha256": digest(
            closure["v9_consumed_identities"]),
        "consumed_v10_outer_identities_sha256": digest(
            closure["v10_consumed_identities"]),
        "consumed_v10_outer_observation_hashes_sha256": (
            consumed_v10_projection["outer_observation_hashes_sha256"]),
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
        raise RuntimeError("v11 registry source closure changed during selection")
    return registry, report, projection


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
    parent = _directory(destination.parent, "v11 registry output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v11 registry output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / "outer_observation_hashes.json", projection)
        if file_hash(temporary / "outer_observation_hashes.json") != report[
                "bindings"]["outer_observation_hash_projection_file_sha256"]:
            raise RuntimeError("v11 historical projection bytes differ")
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
    consumed_v9_attempt_closeout_path: str | Path,
    expected_consumed_v9_attempt_closeout_sha256: str,
    permanent_v9_attempt_registry: str | Path,
    burned_v10_final_closeout_path: str | Path,
    expected_burned_v10_final_closeout_sha256: str,
    permanent_v10_final_closeout_registry: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate a saved v11 registry and both consumed-attempt roots."""
    registry_file, registry = v9._strict_json(
        registry_path, "saved v11 outer registry",
        expected_sha256=expected_registry_sha256)
    report_file, report = v9._strict_json(
        report_path, "saved v11 outer registry report",
        expected_sha256=expected_report_sha256)
    closeout = attempt_closeout_api.read_saved_closeout(
        consumed_v9_attempt_closeout_path,
        expected_closeout_sha256=expected_consumed_v9_attempt_closeout_sha256,
        permanent_attempt_registry=permanent_v9_attempt_registry)
    final_closeout = final_closeout_api.read_saved_closeout(
        burned_v10_final_closeout_path,
        expected_closeout_sha256=expected_burned_v10_final_closeout_sha256,
        permanent_closeout_registry=permanent_v10_final_closeout_registry)
    scenes = registry.get("development_outer")
    identities = registry.get("selected_outer_identities")
    if (registry.get("version") != VERSION or registry.get("status") != STATUS
            or not v9._content_valid(registry)
            or registry.get("contract") != contract()
            or registry.get("formal_ready") is not False
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("action_labels_access") is not False
            or registry.get("probabilities_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or not isinstance(scenes, list) or len(scenes) != 64
            or not isinstance(identities, list) or len(identities) != 64
            or registry.get("producer_sources_sha256")
                != digest(dict(sorted(
                    registry.get("producer_sources", {}).items())))):
        raise ValueError("Saved v11 outer registry semantics differ")
    public: list[dict[str, Any]] = []
    for index, (scene, raw) in enumerate(zip(scenes, identities)):
        if not isinstance(raw, Mapping) or set(raw) != {
                "batch_index", "family_id", "seed", "fingerprint"}:
            raise ValueError("Saved v11 outer identity fields differ")
        identity = _public_identity(raw)
        v9._identity(identity, "saved v11 outer identity")
        if (not isinstance(scene, Mapping)
                or scene.get("id")
                    != f"diagnostic_v11_fresh_outer_{index:04d}"
                or any(scene.get(name) != identity[name]
                       for name in ("family_id", "seed", "fingerprint"))):
            raise ValueError("Saved v11 outer materialisation differs")
        public.append(identity)
    bindings = registry.get("bindings")
    statistics = registry.get("statistics")
    boundary = registry.get("information_boundary")
    if (not isinstance(bindings, Mapping) or set(bindings) != RECOVERY_BINDINGS
            or bindings.get("failed_rows_file_sha256")
                != EXPECTED_FAILED_ROWS_SHA256
            or bindings.get("failure_closeout_file_sha256")
                != "90fb466abaec7fb49a2b6cfff1d79fa74d0f3365fc8102785bb5c1a9dfd0e4c0"
            or bindings.get("consumed_v9_attempt_closeout_file_sha256")
                != expected_consumed_v9_attempt_closeout_sha256
            or bindings.get("consumed_v9_attempt_closeout_content_sha256")
                != closeout["content_sha256"]
            or bindings.get("consumed_v9_selected_identity_sha256")
                != closeout["consumed_outer"]["identities_sha256"]
            or bindings.get("burned_v10_final_closeout_file_sha256")
                != expected_burned_v10_final_closeout_sha256
            or bindings.get("burned_v10_final_closeout_content_sha256")
                != final_closeout["content_sha256"]
            or bindings.get("consumed_v10_selected_identity_sha256")
                != final_closeout["consumed_outer"]["identities_sha256"]
            or bindings.get("consumed_v10_outer_observation_hashes_sha256")
                != final_closeout["consumed_outer"][
                    "observation_hash_projection"][
                        "outer_observation_hashes_sha256"]
            or bindings.get("contract_sha256") != digest(contract())
            or not isinstance(statistics, Mapping)
            or statistics.get("remaining_candidate_scene_count")
                != EXPECTED_REMAINING_SCENE_COUNT
            or statistics.get("remaining_family_counts")
                != EXPECTED_REMAINING_FAMILY_COUNTS
            or statistics.get("consumed_v9_outer_identities") != 64
            or statistics.get("consumed_v10_outer_identities") != 64
            or statistics.get("union_candidate_identities_excluded") != 593
            or statistics.get("selected_outer_scene_count") != 64
            or statistics.get("selected_outer_family_counts") != FAMILY_QUOTAS
            or statistics.get("selected_exposed_seed_overlap") != 0
            or statistics.get("selected_exposed_fingerprint_overlap") != 0
            or not isinstance(boundary, Mapping)
            or boundary.get("consumed_v9_attempt_closed_before_selection")
                is not True
            or boundary.get("consumed_v9_identities_excluded") is not True
            or boundary.get("consumed_v10_identities_excluded") is not True
            or boundary.get(
                "consumed_v10_observation_hashes_excluded_by_full_replay")
                is not True
            or boundary.get("consumed_v9_rows_opened") is not False
            or boundary != RECOVERY_BOUNDARY
            or statistics.get("hash_screen", {}).get(
                "selected_v10_observation_overlap") != 0):
        raise ValueError("Saved v11 exclusion closure differs")
    selection = report.get("selection")
    if (report.get("version") != REPORT_VERSION
            or report.get("status") != STATUS
            or not v9._content_valid(report)
            or report.get("formal_ready") is not False
            or report.get("registry_file_sha256") != file_hash(registry_file)
            or report.get("registry_content_sha256")
                != registry["content_sha256"]
            or report.get("bindings") != bindings
            or report.get("statistics") != statistics
            or report.get("information_boundary") != boundary
            or report.get("producer_sources")
                != registry.get("producer_sources")
            or report.get("producer_sources_sha256")
                != registry.get("producer_sources_sha256")
            or not isinstance(selection, Mapping)
            or selection.get("salt") != SELECTION_SALT
            or selection.get("family_quotas") != FAMILY_QUOTAS
            or selection.get("selected_identity_sha256") != digest(public)
            or selection.get("exclusion_digests")
                != registry.get("exclusion_digests")
            or file_hash(report_file) != expected_report_sha256):
        raise ValueError("Saved v11 registry report differs")
    consumed_keys = {
        (row["seed"], row["fingerprint"])
        for row in closeout["consumed_outer"]["identities"]
    } | {(row["seed"], row["fingerprint"])
         for row in final_closeout["consumed_outer"]["identities"]}
    if consumed_keys & {(row["seed"], row["fingerprint"]) for row in public}:
        raise ValueError("Saved v11 registry reuses a consumed identity")
    return deepcopy(registry), deepcopy(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--failed-rows", required=True)
    parser.add_argument("--original-expansion", required=True)
    parser.add_argument("--current-outer-registry", required=True)
    parser.add_argument("--current-outer-report", required=True)
    parser.add_argument("--failure-closeout", required=True)
    parser.add_argument("--expected-failure-closeout-sha256", required=True)
    parser.add_argument("--permanent-closeout-registry", required=True)
    parser.add_argument("--formal-selection", required=True)
    parser.add_argument("--previous-development", required=True)
    parser.add_argument("--retired-projection", required=True)
    parser.add_argument("--consumed-v9-attempt-closeout", required=True)
    parser.add_argument(
        "--expected-consumed-v9-attempt-closeout-sha256", required=True)
    parser.add_argument("--permanent-v9-attempt-registry", required=True)
    parser.add_argument("--burned-v10-final-closeout", required=True)
    parser.add_argument(
        "--expected-burned-v10-final-closeout-sha256", required=True)
    parser.add_argument("--permanent-v10-final-closeout-registry", required=True)
    parser.add_argument("--output", required=True)
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
    "_RECOVERY_BINDINGS",
    "contract", "producer_sources", "replay_exclusion_closure",
    "create_registry", "build", "read_saved_registry", "main",
    "_remaining_and_selected", "_materialize_scene",
    "_prior_validation_wins_projection", "_hash_screened_selection",
]
