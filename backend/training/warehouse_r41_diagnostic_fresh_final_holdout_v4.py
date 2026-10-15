"""Freeze the one fresh r4.1 diagnostic final holdout for RCPD v8.

This producer is deliberately explanation-program blind.  It selects only
from the pre-existing conflict-manifest candidate population, using a frozen
salt, scene identity, the public workload screen, and exact public-observation
separation.  It never imports, opens, or accepts an explanation program or an
RCPD report.

The exclusion closure contains every public development split, formal X/Y,
the original development supplement, both v8 development-expansion splits,
and a claim-bound v3 tombstone.  Retired v1/v2 contribute only their frozen
identity projection: their Actor outputs, trajectories, observations, labels,
and metrics are never reconstructed.  The protected historical final is
opened only after the irreversible marker; its saved workload receipts are
never inspected, while fixed public scene inputs drive an independent
observation-only replay.  Any identity or public-observation overlap fails
closed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import pwd
import random
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    VERSION as MANIFEST_VERSION,
)
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_designation_v2 as designation_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as input_snapshot_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as outer_split_api
from backend.training import warehouse_r41_diagnostic_retired_identity_projection_v8 as retired_identity_api
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    VERSION as WORKLOAD_VERSION,
    screen_scene,
)
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v3 as legacy_v3
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action


VERSION = "warehouse-r41-diagnostic-fresh-final-holdout.v4"
FINAL_ONCE_VERSION = "warehouse-r41-diagnostic-final-once.v8"
V3_TOMBSTONE_VERSION = "warehouse-r41-diagnostic-retired-v3-exclusion.v1"
DEVELOPMENT_EXPANSION_VERSION = outer_split_api.VERSION
DEVELOPMENT_SUPPLEMENT_VERSION = "warehouse-r41-diagnostic-development-supplement.v1"
PARTNERS = ("skilled", "assertive", "noisy")
TOTAL_SCENES = 64
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
HOLDOUT_SALT_DOMAIN = b"warehouse-r41-v8-final-holdout-salt\0"
# Provision only after the source and tests are frozen.  The final-once
# controller refuses this explicit placeholder.
HOLDOUT_SALT_COMMITMENT = (
    "6dec9810eacc530aef592db4b024ee77d01d419852ea2f6965ba87fc65814d80"
)
LEDGER_ENV = "WAREHOUSE_R41_FINAL_LEDGER_DIR"
PERMANENT_ANCHOR_ENV = "WAREHOUSE_R41_FINAL_PERMANENT_ANCHOR"
_ACCOUNT_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
DEFAULT_LEDGER_ROOT = (
    _ACCOUNT_HOME / ".local/state/policylens/warehouse_r41_diagnostic_v8_final_once"
)
DEFAULT_PERMANENT_ANCHOR = (
    _ACCOUNT_HOME / ".local/state/policylens/warehouse_r41_diagnostic_v8_final_once.anchor"
)
EXPECTED_ACTOR_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
EXPECTED_PROTOCOL_SHA256 = (
    "374ae398115672a243bd2917937ff056fdc762499b470bf11ad44f5fdecc13c8"
)
EXPECTED_MANIFEST_SHA256 = (
    "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
)
EXPECTED_DESIGNATION_SHA256 = (
    "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
)
EXPECTED_EXPANSION_REGISTRY_SHA256 = (
    "a598f78b0b8054bfe9e3cb0befa9c1007f0bcf6e0e599ac2565523d1f0e62332"
)
EXPECTED_EXPANSION_REPORT_SHA256 = (
    "bf21daa21e3799c701c641a34b72130d95bdb64294a908d78d901938623e3115"
)
EXPECTED_SELECTED_SCENES_SHA256 = (
    "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
)
EXPECTED_SOURCE_EXPANSION_FILE_SHA256 = (
    "a687fd3fd4b145ed432af77f3ce26726d4e328df4875e4d69fc3ed351ad98748"
)
EXPECTED_FORMAL_SELECTION_FILE_SHA256 = EXPECTED_SELECTED_SCENES_SHA256
EXPECTED_OLD_OUTER_ORDERED_FINGERPRINTS_SHA256 = (
    "91b6e38343142ba4c478cde51c79e749432bb72899e241c5c7ba26975ee9a6f5"
)
EXPECTED_OLD_OUTER_ORDERED_SEEDS_SHA256 = (
    "1daeae625a859b4b000989c3364bbadbe9f0b88c2653d69e97a6bbac736860bc"
)
OLD_OUTER_COLLECTION_SCENE_OFFSET = 320
EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256 = (
    "cdd17b1a8d46b54dfeec70f74fb3acb43dc1b575c54f58cf9be982ad89d5f111"
)
EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256 = (
    "4188b293f5c0e02cc80ece7fa1c0841134369c740fffea2aecef90653e077db5"
)
CANDIDATE_ARTIFACT_NAMES = frozenset({
    "inputs.json", "prior_rows_reauthentication_report.json",
    "prior_v7_rows.npz", "source_v7_report.json",
    "expansion_rows_reauthentication_report.json",
    "source_expansion_collection_report.json", "expansion_rows.npz",
    "development_expansion.json", "development_expansion_report.json",
    "fit_config.json", "fit_selector_report.json",
    "fit_selector_source_v8_report.json", "fit_selector_source_v8_rows.npz",
    "fit_selector_fit_only_rows.npz", "fit_selector_scope.json",
    "fit_selector_config_registry.json", "fit_selector_inner_split_audit.json",
    "fit_selector_inner_selection.json", "fit_selector_selected_config.json",
    "fit_selector_inner_fit_program.json", "rows.npz", "pairs.npz",
    "weights_audit.json",
    "program.json", "candidate.json", "report.json",
})
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_HISTORICAL_PUBLIC_SCENE_FIELDS = frozenset({
    "id", "split", "seed", "fingerprint", "diagnostic_contract_sha256",
    "diagnostic_conflict_graph_sha256", "conflict_families_sha256",
    "family_id", "initial_edge_id", "task_geometry_signature",
    "initial_conflict", "initial_public_joint_work_steps",
    "initial_robot_positions", "successor_stream_seed", "snapshot",
})


class _HistoricalFinalPrivatePhaseError(RuntimeError):
    """Fixed-detail failure after historical-final access has been burned."""


class _HistoricalFinalPrivatePhaseInterrupt(BaseException):
    """Fixed-detail interruption after historical-final access has been burned."""


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_manifest_version": MANIFEST_VERSION,
        "workload_screen_version": WORKLOAD_VERSION,
        "workload_screen_contract_sha256": WORKLOAD_CONTRACT_SHA256,
        "purpose": "single program-blind final explanation audit for RCPD v8",
        "selection_population": "source manifest play_candidates only",
        "selection_order": (
            "ascending sha256(claim-revealed committed salt, family, scene fingerprint)"
        ),
        "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
        "selection_rule": (
            "first exact-final-workload-safe scene per family with a new seed, "
            "new fingerprint, and no exact public-observation overlap"
        ),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "scene_count": TOTAL_SCENES,
        "partners": list(PARTNERS),
        "horizon": 120,
        "development_registries": [
            DEVELOPMENT_SUPPLEMENT_VERSION,
            DEVELOPMENT_EXPANSION_VERSION,
        ],
        "retired_identity_projection_version": retired_identity_api.VERSION,
        "retired_identity_projection_file_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
        "retired_identity_projection_report_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        "retired_identity_count": (
            retired_identity_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES),
        "full_retired_holdout_access": False,
        "internal_v3_exclusion_registry": V3_TOMBSTONE_VERSION,
        "program_access": False,
        "program_predictions_access": False,
        "actor_logits_access": False,
        "actor_hidden_state_access": False,
        "participant_data_access": False,
        "final_labels_used_for_selection": False,
        "retired_final_reuse": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    # Use the real CLI import closure, including package initializers and their
    # import-time effects.
    sources = local_source_hashes((Path(__file__).resolve(),))
    for name, source_sha256 in designation_api.source_closure().items():
        if name in sources and sources[name] != source_sha256:
            raise ValueError("Fresh-final designation source closure differs")
        sources[name] = source_sha256
    return dict(sorted(sources.items()))


def _ledger_root() -> Path:
    # Environment variables are intentionally ignored.  A caller-selectable
    # ledger would let the same campaign claim a fresh directory and retry.
    path = DEFAULT_LEDGER_ROOT.expanduser().absolute()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Final-once ledger must be at its fixed repository-external path")
    return path


def _anchor_path() -> Path:
    path = DEFAULT_PERMANENT_ANCHOR.expanduser().absolute()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Final-once anchor must be at its fixed repository-external path")
    return path


def _campaign_identity() -> dict[str, Any]:
    if (_HEX.fullmatch(HOLDOUT_SALT_COMMITMENT) is None
            or HOLDOUT_SALT_COMMITMENT == "0" * 64):
        raise ValueError("Private holdout salt commitment is not provisioned")
    return {
        "version": FINAL_ONCE_VERSION,
        "actor_file_sha256": EXPECTED_ACTOR_SHA256,
        "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "designation_file_sha256": EXPECTED_DESIGNATION_SHA256,
        "development_expansion_registry_sha256": (
            EXPECTED_EXPANSION_REGISTRY_SHA256
        ),
        "development_expansion_report_sha256": (
            EXPECTED_EXPANSION_REPORT_SHA256),
        "retired_identity_projection_file_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
        "retired_identity_projection_report_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        "holdout_version": VERSION,
        "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
    }


def _claim_receipt(
    claim_receipt_path: str | Path, *, expected_claim_sha256: str,
    expected_campaign_key: str, expected_candidate_identity_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    """Authenticate the irrevocable campaign claim before final-scene access."""
    if any(_HEX.fullmatch(str(value)) is None for value in (
            expected_claim_sha256, expected_campaign_key,
            expected_candidate_identity_sha256)):
        raise ValueError("Final claim identity is malformed")
    path = Path(claim_receipt_path).expanduser().absolute()
    root = _ledger_root()
    expected_identity = _campaign_identity()
    fixed_key = digest(expected_identity)
    if (path.name != "attempt_started.json" or path.parent.name != expected_campaign_key
            or expected_campaign_key != fixed_key
            or expected_candidate_identity_sha256 != digest(expected_identity)
            or path.parent.parent != root or not path.is_file() or path.is_symlink()
            or path.resolve() != path):
        raise ValueError("A valid fixed-campaign claim is required")
    anchor_path = _regular(_anchor_path(), "permanent final-once anchor")
    candidate_marker_path = _regular(
        path.parent / "candidate_authenticated.json",
        "strict candidate authentication phase")
    originals = {
        "claim": path,
        "anchor": anchor_path,
        "candidate": candidate_marker_path,
    }
    expected = {
        "claim": expected_claim_sha256,
        "anchor": file_hash(anchor_path),
        "candidate": file_hash(candidate_marker_path),
    }
    with input_snapshot_api.ImmutableInputSnapshot(
            originals, expected_sha256=expected,
            relative_names={
                "claim": "claim/attempt_started.json",
                "anchor": "anchor/permanent_anchor.json",
                "candidate": "claim/candidate_authenticated.json",
            }, prefix="warehouse-r41-claim-auth-") as frozen:
        receipt = _read(frozen.paths["claim"], "final-once claim")
        anchor = _read(frozen.paths["anchor"], "permanent final-once anchor")
        candidate_marker = _read(
            frozen.paths["candidate"],
            "strict candidate authentication phase")
        candidate_artifacts = candidate_marker.get("candidate_artifacts")
        if (receipt.get("version") != FINAL_ONCE_VERSION
            or receipt.get("status") != "started_irrevocable_no_retry"
            or receipt.get("key") != fixed_key
            or receipt.get("campaign_key") != expected_campaign_key
            or receipt.get("candidate_identity_sha256")
                != expected_candidate_identity_sha256
            or receipt.get("identity") != expected_identity
            or receipt.get("program_evaluation_started") is not False
            or anchor.get("version") != FINAL_ONCE_VERSION
            or anchor.get("status") != "claimed_irrevocable_no_retry"
            or anchor.get("campaign_key") != fixed_key
            or anchor.get("candidate_identity_sha256")
                != expected_candidate_identity_sha256
            or anchor.get("identity") != expected_identity
            or receipt.get("permanent_anchor_path") != str(anchor_path)
            or receipt.get("permanent_anchor_sha256")
                != file_hash(frozen.paths["anchor"])
            or candidate_marker.get("status")
                != "passed_strict_reader_and_refit"
            or candidate_marker.get("campaign_key") != fixed_key
            or candidate_marker.get("candidate_identity_sha256")
                != expected_candidate_identity_sha256
            or candidate_marker.get("attempt_started_sha256")
                != expected_claim_sha256
            or candidate_marker.get("require_passed") is not True
            or candidate_marker.get("refit") is not True
            or not isinstance(candidate_artifacts, Mapping)
            or set(candidate_artifacts) != CANDIDATE_ARTIFACT_NAMES
            or any(_HEX.fullmatch(str(value)) is None
                   for value in candidate_artifacts.values())
            or candidate_marker.get("candidate_artifacts_sha256")
                != digest(dict(candidate_artifacts))
            or candidate_marker.get("program_file_sha256")
                != candidate_artifacts.get("program.json")
            or candidate_marker.get("rcpd_report_file_sha256")
                != candidate_artifacts.get("report.json")
            or candidate_marker.get("rows_file_sha256")
                != candidate_artifacts.get("rows.npz")
            or candidate_marker.get(
                "prior_rows_reauthentication_report_file_sha256")
                != candidate_artifacts.get(
                    "prior_rows_reauthentication_report.json")
            or candidate_marker.get("prior_v7_source_report_file_sha256")
                != candidate_artifacts.get("source_v7_report.json")
            or candidate_marker.get("prior_v7_rows_file_sha256")
                != candidate_artifacts.get("prior_v7_rows.npz")
            or candidate_marker.get(
                "expansion_rows_reauthentication_report_file_sha256")
                != candidate_artifacts.get(
                    "expansion_rows_reauthentication_report.json")
            or candidate_marker.get(
                "expansion_source_collection_report_file_sha256")
                != candidate_artifacts.get(
                    "source_expansion_collection_report.json")
            or candidate_marker.get("expansion_rows_file_sha256")
                != candidate_artifacts.get("expansion_rows.npz")
            or candidate_marker.get(
                "development_expansion_registry_file_sha256")
                != candidate_artifacts.get("development_expansion.json")
            or candidate_marker.get(
                "development_expansion_report_file_sha256")
                != candidate_artifacts.get("development_expansion_report.json")
            or candidate_marker.get("fit_config_file_sha256")
                != candidate_artifacts.get("fit_config.json")
            or candidate_marker.get("fit_selector_report_file_sha256")
                != candidate_artifacts.get("fit_selector_report.json")
            or candidate_marker.get("fit_selector_scope_file_sha256")
                != candidate_artifacts.get("fit_selector_scope.json")
            or candidate_marker.get("fit_selector_selected_config_file_sha256")
                != candidate_artifacts.get("fit_selector_selected_config.json")
            or candidate_marker.get(
                "fit_selector_source_v8_report_file_sha256")
                != candidate_artifacts.get("fit_selector_source_v8_report.json")
            or candidate_marker.get("fit_selector_source_v8_rows_file_sha256")
                != candidate_artifacts.get("fit_selector_source_v8_rows.npz")
            or candidate_marker.get("actor_file_sha256") != EXPECTED_ACTOR_SHA256
            or candidate_marker.get("protocol_file_sha256")
                != EXPECTED_PROTOCOL_SHA256
            or candidate_marker.get("manifest_file_sha256")
                != EXPECTED_MANIFEST_SHA256
            or candidate_marker.get("designation_file_sha256")
                != EXPECTED_DESIGNATION_SHA256
            or candidate_marker.get("selected_scenes_file_sha256")
                != EXPECTED_SELECTED_SCENES_SHA256
            or candidate_marker.get("development_registries", {}).get(
                DEVELOPMENT_EXPANSION_VERSION)
                != EXPECTED_EXPANSION_REGISTRY_SHA256
            or (path.parent / "attempt_completed.json").exists()):
            raise ValueError("Final claim receipt differs")
        frozen.verify()
        return path.parent, receipt


def _build_claimed_historical_final_observation_exclusion(
        *args: Any, **kwargs: Any,
) -> tuple[list[dict], list[dict], dict[str, Any], set[str], dict[str, Any]]:
    """Cross the private historical boundary without exporting its frames.

    The sensitive worker receives the raw committed salt and may temporarily
    hold protected historical identities and observation hashes.  A caller
    must never be able to recover those values by walking an exception
    traceback.  This outer frame therefore erases its call containers and
    raises a fresh fixed-detail exception only after the worker exception and
    all of its traceback frames have gone out of scope.
    """

    def sensitive(
        *, manifest_path: str | Path,
        runtime: R41DiagnosticOnlineAlignmentRuntime,
        selection_manifest: Mapping[str, Any],
        selection_salt: bytes,
        selected_scenes_path: str | Path,
        development_registry_paths: Sequence[str | Path],
        retired_identity_projection_path: str | Path,
        v3_tombstone: Mapping[str, Any],
        claim_receipt_path: str | Path,
        expected_claim_sha256: str,
        expected_campaign_key: str,
        expected_candidate_identity_sha256: str,
        excluded_fingerprints: set[str],
        excluded_seeds: set[int],
        preexisting_observation_hashes: set[str],
        replayed_excluded_observation_hashes: set[str],
        row_artifact_evidence: Mapping[str, tuple[Path, set[str], set[str]]],
        input_guard: Callable[[str], None] | None = None,
        register_phase_input: Callable[[str, Path], None] | None = None,
    ) -> tuple[list[dict], list[dict], dict[str, Any], set[str], dict[str, Any]]:
        """Build the old-final exclusion inside one irrevocable private phase.

        Full historical scenes and their raw identities never leave this function.
        Fresh selection is performed here while those private identity sets are in
        scope, and only the newly selected scenes plus aggregate commitments are
        returned.  The frozen Actor is executed only to advance registered public
        workloads; its actions and probabilities are neither returned nor
        persisted.  Publishing the ``started`` marker before opening the manifest
        makes a crash consume the sole final attempt rather than enable a retry.
        """
        claim_dir, _ = _claim_receipt(
            claim_receipt_path,
            expected_claim_sha256=expected_claim_sha256,
            expected_campaign_key=expected_campaign_key,
            expected_candidate_identity_sha256=expected_candidate_identity_sha256,
        )
        if input_guard is not None:
            input_guard("historical final claim authentication")
        holdout_started_path = claim_dir / "holdout_started.json"
        holdout_started, holdout_started_sha256 = _read_exact_json(
            holdout_started_path, "holdout phase marker")
        candidate_path = claim_dir / "candidate_authenticated.json"
        candidate_marker, candidate_marker_sha256 = _read_exact_json(
            candidate_path, "candidate authentication phase")
        if (
            holdout_started.get("version") != VERSION
            or holdout_started.get("status") != "started_no_retry"
            or holdout_started.get("campaign_key") != expected_campaign_key
            or holdout_started.get("candidate_identity_sha256")
                != expected_candidate_identity_sha256
            or holdout_started.get("attempt_started_sha256") != expected_claim_sha256
            or holdout_started.get("candidate_authenticated_sha256")
                != candidate_marker_sha256
            or holdout_started.get("selection_salt_commitment")
                != HOLDOUT_SALT_COMMITMENT
            or any((claim_dir / name).exists() for name in (
                "historical_exclusion_started.json",
                "historical_exclusion_completed.json", "holdout_completed.json",
                "audit_started.json", "audit_completed.json",
                "attempt_completed.json",
            ))
        ):
            raise ValueError("Historical final access requires the active holdout phase")
        if (
            not isinstance(runtime, R41DiagnosticOnlineAlignmentRuntime)
            or runtime.actor_sha256 != EXPECTED_ACTOR_SHA256
            or runtime.verify_binding() != runtime.signature
            or not excluded_fingerprints
            or not excluded_seeds
            or not preexisting_observation_hashes
            or not replayed_excluded_observation_hashes
            or not isinstance(selection_manifest, Mapping)
            or not isinstance(selection_salt, bytes)
            or len(selection_salt) != 32
            or sha256(HOLDOUT_SALT_DOMAIN + selection_salt).hexdigest()
                != HOLDOUT_SALT_COMMITMENT
            or not isinstance(v3_tombstone, Mapping)
            or not isinstance(row_artifact_evidence, Mapping)
            or set(row_artifact_evidence) != {
                "merged_rows.npz", "prior_rows.npz", "expansion_rows.npz"
            }
        ):
            raise ValueError("Historical final exclusion inputs differ")
        if input_guard is not None:
            input_guard("historical final private phase entry")
        row_sources: dict[str, dict[str, Any]] = {}
        for name, evidence in sorted(row_artifact_evidence.items()):
            if (not isinstance(evidence, tuple) or len(evidence) != 3
                    or not isinstance(evidence[1], set)
                    or not isinstance(evidence[2], set)
                    or not evidence[1] or not evidence[2]):
                raise ValueError("Historical final row evidence differs")
            path = _regular(evidence[0], name, maximum=MAX_NPZ_BYTES)
            (actual_observations, actual_fingerprints, actual_file_sha256,
             row_count) = _npz_development_artifact(path, label=name)
            if (actual_fingerprints != evidence[1]
                    or actual_observations != evidence[2]
                    or _HEX.fullmatch(actual_file_sha256) is None):
                raise ValueError("Historical final row evidence does not match its file")
            row_sources[name] = {
                "file_sha256": actual_file_sha256,
                "row_count": row_count,
                "unique_scene_fingerprint_count": len(evidence[1]),
                "scene_fingerprints_sha256": digest(sorted(evidence[1])),
                "unique_public_observation_count": len(evidence[2]),
                "public_observations_sha256": digest(sorted(evidence[2])),
            }
        if input_guard is not None:
            input_guard("historical final row authentication")

        # Reconstruct every selection input from fixed artifacts before publishing
        # the historical-access marker.  Callers cannot choose a salt, manifest,
        # retired identity set, or exclusion set to probe protected membership.
        authenticated_manifest = manifest_binding.read_saved_manifest(
            manifest_path, actor_path=runtime._actor_path,
            replay_scope="development")
        if canonical(authenticated_manifest) != canonical(selection_manifest):
            raise ValueError("Historical selection manifest differs")
        selected_path = _regular(selected_scenes_path, "formal X/Y selection")
        selected, _ = _read_exact_json(
            selected_path, "formal X/Y selection",
            expected_sha256=EXPECTED_SELECTED_SCENES_SHA256)
        xy = [
            deepcopy(row) for key in ("X", "Y")
            for row in selected.get(key, [])
        ]
        if len(xy) != 6:
            raise ValueError("Historical selection requires six X/Y scenes")
        development_paths = [
            _regular(path, "development registry")
            for path in development_registry_paths
        ]
        (development_scenes, development_bindings,
         development_values) = _development_registry_evidence(development_paths)
        if candidate_marker.get("development_registries") != {
            version: binding["file_sha256"]
            for version, binding in sorted(development_bindings.items())
        } or {
            "merged_rows.npz": candidate_marker.get("rows_file_sha256"),
            "prior_rows.npz": candidate_marker.get("prior_v7_rows_file_sha256"),
            "expansion_rows.npz": candidate_marker.get("expansion_rows_file_sha256"),
        } != {
            name: row_sources[name]["file_sha256"] for name in sorted(row_sources)
        }:
            raise ValueError("Historical development registry binding differs")
        projected_identities, projected_binding = _retired_identity_projection(
            _regular(
                retired_identity_projection_path,
                "retired identity projection"))
        expansion = development_values[DEVELOPMENT_EXPANSION_VERSION]
        _validate_retired_expansion_binding(projected_binding, expansion)
        old_outer_scenes, fresh_outer_scenes = _outer_split_exclusion_scenes(
            authenticated_manifest, expansion)
        projected_fingerprints, projected_seeds, _ = (
            _projected_identity_exclusions(
                manifest=authenticated_manifest, identities=projected_identities))
        claim_binding = {
            "campaign_key": expected_campaign_key,
            "attempt_started_sha256": expected_claim_sha256,
            "candidate_identity_sha256": expected_candidate_identity_sha256,
            "candidate_authenticated_sha256": candidate_marker_sha256,
        }
        expected_v3 = _build_v3_tombstone(
            runtime=runtime, actor=runtime.actor,
            manifest=authenticated_manifest, selected=selected,
            legacy_development_hashes=set(
                row_artifact_evidence["prior_rows.npz"][2]),
            projected_retired_fingerprints=projected_fingerprints,
            projected_retired_seeds=projected_seeds,
            claim_binding=claim_binding,
        )
        if canonical(v3_tombstone) != canonical(expected_v3):
            raise ValueError("Historical v3 exclusion differs")
        v3_registry = [{
            "version": V3_TOMBSTONE_VERSION,
            "scenes": expected_v3["scenes"],
            "selection_trace": expected_v3["selection_trace"],
        }]
        v3_fingerprints, v3_seeds, v3_observations, _ = (
            _selection_exposure_closure(
                runtime=runtime, actor=runtime.actor,
                manifest=authenticated_manifest, registries=v3_registry))
        registered = [
            deepcopy(row)
            for rows in authenticated_manifest["splits"].values()
            for row in rows
        ]
        expected_fingerprints = {
            row["fingerprint"]
            for row in [
                *registered, *xy, *development_scenes,
                *old_outer_scenes, *fresh_outer_scenes,
                *expected_v3["scenes"],
            ]
        } | projected_fingerprints | v3_fingerprints
        expected_seeds = {
            row["seed"]
            for row in [
                *registered, *xy, *development_scenes,
                *old_outer_scenes, *fresh_outer_scenes,
                *expected_v3["scenes"],
            ]
        } | projected_seeds | v3_seeds
        xy_observations: set[str] = set()
        for index, scene in enumerate(xy):
            xy_observations.update(
                _exact_final_workload_observations(runtime, scene, index))
        auxiliary_observations: set[str] = set()
        for split_name in ("tutorial", "question_bank"):
            for index, scene in enumerate(
                    authenticated_manifest["splits"][split_name]):
                auxiliary_observations.update(_auxiliary_workload_observations(
                    runtime, scene, split=split_name, scene_index=index))
        old_outer_observations = _old_outer_collection_observations(
            runtime, old_outer_scenes)
        expected_replayed_observations = (
            set(v3_observations) | xy_observations | auxiliary_observations
            | old_outer_observations)
        expected_row_observations = set().union(*(
            evidence[2] for evidence in row_artifact_evidence.values()))
        if (set(excluded_fingerprints) != expected_fingerprints
                or set(excluded_seeds) != expected_seeds
                or set(replayed_excluded_observation_hashes)
                    != expected_replayed_observations
                or set(preexisting_observation_hashes)
                    != expected_row_observations | expected_replayed_observations):
            raise ValueError("Historical public exclusion commitment differs")
        public_exclusion_commitment = {
            "scene_fingerprint_count": len(expected_fingerprints),
            "scene_fingerprints_sha256": digest(sorted(expected_fingerprints)),
            "seed_count": len(expected_seeds),
            "seeds_sha256": digest(sorted(expected_seeds)),
            "replayed_observation_count": len(expected_replayed_observations),
            "replayed_observations_sha256": digest(
                sorted(expected_replayed_observations)),
            "forbidden_observation_count": len(preexisting_observation_hashes),
            "forbidden_observations_sha256": digest(
                sorted(preexisting_observation_hashes)),
            "retired_identity_projection_file_sha256": (
                projected_binding["file_sha256"]),
            "retired_identity_projection_report_sha256": (
                projected_binding["audit_file_sha256"]),
            "retired_identity_count": len(projected_fingerprints),
            "retired_actor_executed": False,
            "retired_observations_derived": False,
            "v3_exclusion_content_sha256": expected_v3["content_sha256"],
            "old_exposed_outer_scene_count": len(old_outer_scenes),
            "old_exposed_outer_fingerprints_sha256": digest(sorted(
                row["fingerprint"] for row in old_outer_scenes)),
            "old_exposed_outer_seeds_sha256": digest(sorted(
                row["seed"] for row in old_outer_scenes)),
            "old_exposed_outer_public_observation_count": len(
                old_outer_observations),
            "old_exposed_outer_public_observations_sha256": digest(
                sorted(old_outer_observations)),
            "fresh_outer_scene_count": len(fresh_outer_scenes),
            "fresh_outer_fingerprints_sha256": digest(sorted(
                row["fingerprint"] for row in fresh_outer_scenes)),
            "fresh_outer_seeds_sha256": digest(sorted(
                row["seed"] for row in fresh_outer_scenes)),
        }
        if input_guard is not None:
            input_guard("historical selection input reconstruction")

        started_value = {
            "version": VERSION + ".historical-exclusion.v1",
            "status": "started_irrevocable_no_retry",
            "campaign_key": expected_campaign_key,
            "candidate_identity_sha256": expected_candidate_identity_sha256,
            "attempt_started_sha256": expected_claim_sha256,
            "candidate_authenticated_sha256": candidate_marker_sha256,
            "holdout_started_sha256": holdout_started_sha256,
            "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
            "row_artifact_evidence": deepcopy(row_sources),
            "row_artifact_evidence_sha256": digest(row_sources),
            "public_exclusion_commitment": public_exclusion_commitment,
            "public_exclusion_commitment_sha256": digest(
                public_exclusion_commitment),
            "historical_final_access_refunds_attempt": False,
            "formal_ready": False,
        }
        historical_started_sha256 = _json_file_sha256(started_value)
        historical_started_path = _claim_marker(
            claim_dir, "historical_exclusion_started.json", started_value,
            before_publish=(
                None if input_guard is None
                else lambda: input_guard("historical final phase start")
            ),
        )
        if register_phase_input is not None:
            register_phase_input(
                "historical_exclusion_started", historical_started_path)
        if input_guard is not None:
            input_guard("historical final phase start")

        private_phase_interrupted: bool | None = None
        try:
            # This is the sole pre-completion full-manifest parse.  It occurs only
            # after the durable claim, candidate marker, holdout marker, and the
            # single-use historical-access marker above.  Exact file/content hashes
            # authenticate the bytes.  Saved final workload receipts, actions,
            # probabilities, labels, and metrics are deliberately never inspected.
            manifest_file = _regular(manifest_path, "frozen historical manifest")
            full_manifest, manifest_file_sha256 = _read_exact_json(
                manifest_file, "frozen historical manifest",
                expected_sha256=EXPECTED_MANIFEST_SHA256)
            content = deepcopy(full_manifest)
            claimed_content = content.pop("content_sha256", None)
            if (
                claimed_content != manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256
                or claimed_content != digest(content)
                or digest(full_manifest)
                    != manifest_binding.EXPECTED_MANIFEST_SEMANTIC_SHA256
            ):
                raise ValueError("Historical final manifest content differs")
            manifest_binding._validate_contract(full_manifest)
            manifest_binding._validate_source_bindings(full_manifest)
            splits = full_manifest.get("splits")
            protected_rows = (
                splits.get("final_test") if isinstance(splits, Mapping) else None)
            if not isinstance(protected_rows, list):
                raise ValueError("Historical final scene registry differs")
            scenes = [
                _historical_public_scene(scene, index)
                for index, scene in enumerate(protected_rows)
            ]
            identities = [
                {"id": row.get("id"), "seed": row.get("seed"),
                 "fingerprint": row.get("fingerprint")}
                for row in scenes
            ]
            if (
                len(identities) != TOTAL_SCENES
                or len(scenes) != TOTAL_SCENES
                or digest(identities)
                    != manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256
                or digest(sorted(row["fingerprint"] for row in identities))
                    != manifest_binding.EXPECTED_FINAL_FINGERPRINTS_SHA256
                or digest(sorted(row["seed"] for row in identities))
                    != manifest_binding.EXPECTED_FINAL_SEEDS_SHA256
            ):
                raise ValueError("Historical final scene registry differs")

            historical_fingerprints = {str(row["fingerprint"]) for row in identities}
            historical_seeds = {int(row["seed"]) for row in identities}
            public_excluded_fingerprints = set(excluded_fingerprints)
            public_excluded_seeds = set(excluded_seeds)
            if (len(historical_fingerprints) != TOTAL_SCENES
                    or len(historical_seeds) != TOTAL_SCENES
                    or historical_fingerprints & public_excluded_fingerprints
                    or historical_seeds & public_excluded_seeds):
                raise RuntimeError("Historical final scene identity overlaps development")

            observation_hashes: set[str] = set()
            for scene_index, scene in enumerate(scenes):
                observation_hashes.update(
                    _exact_final_workload_observations(runtime, scene, scene_index))
            if not observation_hashes or observation_hashes & preexisting_observation_hashes:
                raise RuntimeError("Historical final public observation overlaps development")
            row_overlap_receipts: dict[str, dict[str, Any]] = {}
            for name, (_, row_fingerprints, row_observations) in sorted(
                    row_artifact_evidence.items()):
                scene_overlap = len(historical_fingerprints & row_fingerprints)
                observation_overlap = len(observation_hashes & row_observations)
                if scene_overlap or observation_overlap:
                    raise RuntimeError("Historical final overlaps row artifact: " + name)
                row_overlap_receipts[name] = {
                    **deepcopy(row_sources[name]),
                    "historical_scene_fingerprint_overlap": scene_overlap,
                    "historical_public_observation_overlap": observation_overlap,
                    "zero_historical_overlap": True,
                }
            if any(
                _read_regular_bytes(
                    _regular(evidence[0], name, maximum=MAX_NPZ_BYTES), name,
                    maximum=MAX_NPZ_BYTES)[1]
                    != row_sources[name]["file_sha256"]
                for name, evidence in row_artifact_evidence.items()
            ):
                raise RuntimeError("Development row artifact changed during exclusion")

            combined_excluded_fingerprints = (
                set(public_excluded_fingerprints) | set(historical_fingerprints))
            combined_excluded_seeds = (
                set(public_excluded_seeds) | set(historical_seeds))
            forbidden_observations = (
                set(preexisting_observation_hashes) | set(observation_hashes))
            replayed_observations = (
                set(replayed_excluded_observation_hashes) | set(observation_hashes))
            receipt = {
                "version": VERSION + ".historical-exclusion.v1",
                "status": "completed_observation_hash_exclusion",
                "campaign_key": expected_campaign_key,
                "candidate_identity_sha256": expected_candidate_identity_sha256,
                "attempt_started_sha256": expected_claim_sha256,
                "candidate_authenticated_sha256": candidate_marker_sha256,
                "holdout_started_sha256": holdout_started_sha256,
                "historical_exclusion_started_sha256": (
                    historical_started_sha256),
                "manifest_file_sha256": manifest_file_sha256,
                "historical_final_identity_sha256": (
                    manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256),
                "historical_final_scene_count": len(identities),
                "historical_final_public_observation_count": len(observation_hashes),
                "historical_final_public_observations_sha256": digest(
                    sorted(observation_hashes)),
                "combined_excluded_scene_fingerprint_count": len(
                    combined_excluded_fingerprints),
                "combined_excluded_scene_fingerprints_sha256": digest(
                    sorted(combined_excluded_fingerprints)),
                "combined_excluded_seed_count": len(combined_excluded_seeds),
                "combined_excluded_seeds_sha256": digest(
                    sorted(combined_excluded_seeds)),
                "combined_forbidden_public_observation_count": len(
                    forbidden_observations),
                "combined_forbidden_public_observations_sha256": digest(
                    sorted(forbidden_observations)),
                "combined_replayed_excluded_public_observation_count": len(
                    replayed_observations),
                "combined_replayed_excluded_public_observations_sha256": digest(
                    sorted(replayed_observations)),
                "development_row_scene_overlap": 0,
                "preexisting_scene_identity_overlap": 0,
                "preexisting_public_observation_overlap": 0,
                "row_artifact_evidence": row_overlap_receipts,
                "row_artifact_evidence_sha256": digest(row_overlap_receipts),
                "historical_final_replayed_for_exclusion_only": True,
                "historical_final_actor_executed_for_observation_exclusion_only": True,
                "historical_final_actor_outputs_exposed": False,
                "historical_final_actor_outputs_persisted": False,
                "historical_final_labels_used": False,
                "historical_final_saved_metrics_replayed_for_authentication": False,
                "historical_final_metrics_exposed": False,
                "historical_final_metrics_persisted": False,
                "historical_final_metrics_used_for_fit_or_program_selection": False,
                "historical_final_used_for_fit_or_program_selection": False,
                "historical_final_scenes_returned": False,
                "fresh_selection_conditioned_on_historical_final": False,
                "fresh_selection_private_overlap_fallback": False,
                "retry_allowed": False,
                "formal_ready": False,
            }
            # Select from public development exclusions only.  Historical identities
            # and observations may veto this one predetermined result, but they never
            # alter ranking, rejection reasons, fallback, or any returned trace.  This
            # prevents the public trace from becoming a membership oracle for the
            # private historical final split.
            scenes_selected, trace, statistics, accepted_hashes = _select(
                runtime=runtime,
                actor=runtime.actor,
                manifest=selection_manifest,
                selection_salt=selection_salt,
                excluded_fingerprints=set(public_excluded_fingerprints),
                excluded_seeds=set(public_excluded_seeds),
                forbidden_observation_hashes=set(preexisting_observation_hashes),
            )
            historical_selected_scene_overlap = len(
                {row["fingerprint"] for row in scenes_selected}
                & historical_fingerprints)
            historical_selected_seed_overlap = len(
                {row["seed"] for row in scenes_selected} & historical_seeds)
            historical_selected_observation_overlap = len(
                set(accepted_hashes) & observation_hashes)
            if (historical_selected_scene_overlap
                    or historical_selected_seed_overlap
                    or historical_selected_observation_overlap
                    or accepted_hashes & preexisting_observation_hashes):
                raise RuntimeError("Fresh-final isolation invariant failed")
            receipt.update({
                "fresh_selection_historical_scene_fingerprint_overlap": 0,
                "fresh_selection_historical_seed_overlap": 0,
                "fresh_selection_historical_public_observation_overlap": 0,
            })
            receipt["receipt_sha256"] = digest(receipt)
            historical_completed_path = _claim_marker(
                claim_dir, "historical_exclusion_completed.json", receipt,
                before_publish=(
                    None if input_guard is None
                    else lambda: input_guard("historical final completion")
                ),
            )
            if register_phase_input is not None:
                register_phase_input(
                    "historical_exclusion_completed", historical_completed_path)
            # Drop the only references to full historical scene objects before handing
            # control back to fresh-final selection.
            del scenes, identities, protected_rows, full_manifest
            return (
                scenes_selected, trace, statistics, set(accepted_hashes),
                deepcopy(receipt),
            )
        except BaseException as error:
            # Nothing from the private historical-final phase, including an
            # exception payload or chained traceback, may cross this boundary.
            # Preserve the interrupted/ordinary distinction using fixed-detail
            # exception classes so the controller can still burn the attempt
            # with the correct terminal status.
            private_phase_interrupted = not isinstance(error, Exception)
        # Raise only after leaving the handling context so even an introspected
        # ``__context__`` cannot recover the private originating exception.
        if private_phase_interrupted is False:
            raise _HistoricalFinalPrivatePhaseError(
                "historical_final_private_phase_failed") from None
        if private_phase_interrupted is True:
            raise _HistoricalFinalPrivatePhaseInterrupt(
                "historical_final_private_phase_interrupted") from None
        raise AssertionError("Historical final private phase did not terminate")

    outcome = None
    interrupted: bool | None = None
    try:
        outcome = sensitive(
            *args, **kwargs)
    except BaseException as error:
        interrupted = not isinstance(error, Exception)
    finally:
        args = ()
        kwargs.clear()
        sensitive = None
    if interrupted is False:
        raise _HistoricalFinalPrivatePhaseError(
            "historical_final_private_phase_failed") from None
    if interrupted is True:
        raise _HistoricalFinalPrivatePhaseInterrupt(
            "historical_final_private_phase_interrupted") from None
    if outcome is None:
        raise AssertionError("Historical final private phase returned no result")
    return outcome




def _claim_marker(
    directory: Path, name: str, value: Mapping[str, Any], *,
    before_publish: Callable[[], None] | None = None,
) -> Path:
    path = directory / name
    _write_phase_new(path, dict(value), before_publish=before_publish)
    return path


def _read_committed_salt(path: str | Path) -> bytes:
    if HOLDOUT_SALT_COMMITMENT == "0" * 64:
        raise ValueError("Private holdout salt commitment is not provisioned")
    value, _ = _read_regular_bytes(
        path, "private holdout salt", maximum=4096)
    if (len(value) < 32
            or sha256(HOLDOUT_SALT_DOMAIN + value).hexdigest()
               != HOLDOUT_SALT_COMMITMENT):
        raise ValueError("Private holdout salt does not match its frozen commitment")
    return value


def _regular(value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > maximum):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read_regular_bytes(
    value: str | Path, label: str, *, maximum: int = MAX_JSON_BYTES,
) -> tuple[bytes, str]:
    """Read and hash one canonical file through the same immutable fd.

    Callers that bind parsed content to a digest must never hash a pathname and
    then reopen it for parsing: an atomic A->B->A swap could otherwise make the
    parser consume bytes that are absent from the frozen-input guards.  The
    descriptor is opened with ``O_NOFOLLOW`` and both the hash and parser input
    come from this single byte snapshot.
    """
    path = Path(value).expanduser().absolute()
    if path.resolve() != path or path.parent.resolve() != path.parent.absolute():
        raise ValueError(label + " must be a canonical regular file")
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError(label + " must be a canonical regular file") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError(label + " must be a canonical regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ValueError(label + " exceeds its size limit")
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    return raw, sha256(raw).hexdigest()


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be UTF-8 JSON") from error
    value = json.loads(
        source,
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _read_exact_json(
    path: str | Path, label: str, *, expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    raw, actual_sha256 = _read_regular_bytes(path, label)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(label + " bytes differ")
    return _decode_json(raw, label), actual_sha256


def _read(path: str | Path, label: str) -> dict[str, Any]:
    return _read_exact_json(path, label)[0]


def _private_runtime_input_snapshots(
    *, actor_raw: bytes, protocol_raw: bytes,
) -> tuple[tempfile.TemporaryDirectory, Path, Path]:
    """Materialize immutable private copies used by the runtime constructor."""
    temporary = tempfile.TemporaryDirectory(
        prefix="warehouse-r41-final-runtime-inputs-")
    directory = Path(temporary.name).resolve()
    paths = (directory / "actor.npz", directory / "protocol.json")
    try:
        for path, raw in zip(paths, (actor_raw, protocol_raw)):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        if (file_hash(paths[0]) != EXPECTED_ACTOR_SHA256
                or file_hash(paths[1]) != EXPECTED_PROTOCOL_SHA256):
            raise RuntimeError("Private runtime input snapshot differs")
        return temporary, paths[0], paths[1]
    except BaseException:
        temporary.cleanup()
        raise


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: item for key, item in value.items()
                                   if key != "content_sha256"}))


def _json_file_sha256(value: Any) -> str:
    """Return the exact hash produced by ``_write_new`` for one JSON value."""
    return sha256((canonical(value) + "\n").encode("utf-8")).hexdigest()


def _implicit_input_paths(
    *, manifest_path: Path, designation_path: Path, claim_dir: Path,
    claim_receipt_path: Path,
) -> dict[str, Path]:
    """Resolve every fixed file consumed transitively by the holdout build."""
    validation_path = _regular(
        manifest_path.parent / "validation.json", "manifest validation")
    input_snapshot_api.read_authenticated_bytes(
        validation_path, label="manifest validation",
        expected_sha256=manifest_binding.EXPECTED_VALIDATION_SHA256)
    designation_raw = input_snapshot_api.read_authenticated_bytes(
        designation_path, label="Actor designation",
        expected_sha256=EXPECTED_DESIGNATION_SHA256)
    components = designation_binding.resolve_bound_components_from_bytes(
        designation_raw, original_path=designation_path,
        expected_sha256=EXPECTED_DESIGNATION_SHA256)
    implicit = {
        "manifest_validation": validation_path,
        "attempt_started": _regular(claim_receipt_path, "final-once claim"),
        "permanent_anchor": _regular(
            _anchor_path(), "permanent final-once anchor"),
        "candidate_authenticated": _regular(
            claim_dir / "candidate_authenticated.json",
            "strict candidate authentication phase"),
    }
    implicit.update({
        "designation_component:" + name: _regular(
            path, "designation component " + name, maximum=MAX_NPZ_BYTES)
        for name, path in sorted(components.items())
    })
    return implicit


def _input_hash_snapshot(paths: Mapping[str, Path]) -> dict[str, str]:
    if not paths or any(type(name) is not str for name in paths):
        raise ValueError("Fresh-final input registry differs")
    return {
        name: file_hash(_regular(
            path, "frozen input " + name, maximum=MAX_NPZ_BYTES))
        for name, path in sorted(paths.items())
    }


def _guard_frozen_holdout_inputs(
    paths: Mapping[str, Path], *, expected_sources: Mapping[str, str],
    expected_hashes: Mapping[str, str], phase: str,
) -> None:
    """Detect source or direct/transitive input drift at a phase boundary."""
    if producer_sources() != dict(expected_sources):
        raise RuntimeError(
            "Fresh-final producer source changed during " + phase)
    actual_hashes = _input_hash_snapshot(paths)
    if actual_hashes != dict(expected_hashes):
        raise RuntimeError(
            "Fresh-final frozen input changed during " + phase)
    # A producer file could change while large NPZ inputs are being hashed.
    if producer_sources() != dict(expected_sources):
        raise RuntimeError(
            "Fresh-final producer source changed during " + phase)


def _publish_staged_holdout(
    output_path: Path, *, artifacts: Mapping[str, Any],
    expected_hashes: Mapping[str, str], before_publish: Callable[[], None],
) -> dict[str, str]:
    """Write the complete public result privately, then publish it atomically."""
    if set(artifacts) != {"v3_exclusion.json", "holdout.json", "report.json"}:
        raise ValueError("Fresh-final staged artifact registry differs")
    if set(expected_hashes) != set(artifacts):
        raise ValueError("Fresh-final staged artifact hash registry differs")
    if (output_path.exists() or output_path.is_symlink()
            or not output_path.parent.is_dir() or output_path.parent.is_symlink()
            or output_path.parent.resolve() != output_path.parent.absolute()):
        raise ValueError(
            "Fresh-final output must be a new directory under an existing parent")
    staging = output_path.parent / ("." + output_path.name + ".partial")
    if staging.exists() or staging.is_symlink():
        raise ValueError("Fresh-final staging destination already exists")
    published = False
    try:
        staging.mkdir(mode=0o700)
        for name, value in sorted(artifacts.items()):
            _write_new(staging / name, value)
        actual_hashes = {
            name: file_hash(staging / name) for name in sorted(artifacts)
        }
        if actual_hashes != dict(expected_hashes):
            raise RuntimeError("Fresh-final staged artifact hash differs")
        before_publish()
        if output_path.exists() or output_path.is_symlink():
            raise ValueError("Fresh-final output appeared during publication")
        os.rename(staging, output_path)
        published = True
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return actual_hashes
    finally:
        if not published and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def _write_new(path: Path, value: Any) -> None:
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Fresh-final output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _write_phase_payload(stream: Any, raw: bytes) -> None:
    """Finish the private phase payload before it can acquire its final name."""
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def _link_no_replace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` and fail if ``target`` already exists."""
    os.link(source, target, follow_symlinks=False)


def _write_phase_new(
    path: Path, value: Any, *,
    before_publish: Callable[[], None] | None = None,
) -> None:
    """Publish a complete claim phase without exposing partial final bytes."""
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Fresh-final phase destination is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    temporary = path.parent / ("." + path.name + ".partial")
    descriptor = None
    temporary_created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            _write_phase_payload(stream, raw)
        if before_publish is not None:
            before_publish()
        _link_no_replace(temporary, path)
        temporary.unlink()
        temporary_created = False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            temporary.unlink(missing_ok=True)


def _observation_hash(observation: Any) -> str:
    value = np.asarray(observation, dtype="<f4")
    if value.shape != (197,) or not np.isfinite(value).all():
        raise ValueError("Fresh-final public observation differs")
    return sha256(value.tobytes(order="C")).hexdigest()


def _npz_development_artifact(
    supplied: str | Path, *, label: str = "development rows",
) -> tuple[set[str], set[str], str, int]:
    path = _regular(supplied, label, maximum=MAX_NPZ_BYTES)
    raw, file_sha256 = _read_regular_bytes(
        path, label, maximum=MAX_NPZ_BYTES)
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as rows:
            required = {"observations", "observation_hashes", "scene_fingerprints"}
            if not required.issubset(rows.files):
                raise ValueError("Development rows lack public separation fields")
            observations = np.asarray(rows["observations"])
            hashes = np.asarray(rows["observation_hashes"])
            fingerprints = np.asarray(rows["scene_fingerprints"])
            if (observations.ndim != 2 or observations.shape[1] != 197
                    or observations.dtype != np.dtype("float32")
                    or hashes.shape != (len(observations),)
                    or fingerprints.shape != (len(observations),)
                    or hashes.dtype.kind != "S" or fingerprints.dtype.kind != "S"
                    or hashes.dtype.itemsize != 64 or fingerprints.dtype.itemsize != 64
                    or not np.isfinite(observations).all()):
                raise ValueError("Development public row schema differs")
            decoded_hashes = np.char.decode(hashes, "ascii")
            decoded_fingerprints = np.char.decode(fingerprints, "ascii")
            for index, (observation, claimed) in enumerate(
                    zip(observations, decoded_hashes)):
                if _HEX.fullmatch(str(claimed)) is None:
                    raise ValueError("Development observation hash is malformed")
                if _observation_hash(observation) != str(claimed):
                    raise ValueError(
                        f"Development observation hash differs at row {index}"
                    )
            if any(_HEX.fullmatch(str(value)) is None
                   for value in decoded_fingerprints):
                raise ValueError("Development scene fingerprint is malformed")
            return (
                set(map(str, decoded_hashes)),
                set(map(str, decoded_fingerprints)),
                file_sha256,
                len(observations),
            )
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Development"):
            raise
        raise ValueError("Development row artifact is not a valid NPZ") from error


def _npz_development_observations(
    paths: Sequence[Path], *, required_fingerprints: set[str],
) -> tuple[set[str], set[str], dict[str, str]]:
    if not paths:
        raise ValueError("At least one frozen development row artifact is required")
    observation_hashes: set[str] = set()
    scene_fingerprints: set[str] = set()
    bindings: dict[str, str] = {}
    for supplied in paths:
        path = _regular(supplied, "development rows", maximum=MAX_NPZ_BYTES)
        if path.name in bindings:
            raise ValueError("Development row artifact names must be unique")
        observations, fingerprints, file_sha256, _ = (
            _npz_development_artifact(path))
        bindings[path.name] = file_sha256
        observation_hashes.update(observations)
        scene_fingerprints.update(fingerprints)
    missing = required_fingerprints - scene_fingerprints
    if missing:
        raise ValueError("Development rows do not cover every frozen development scene")
    return observation_hashes, scene_fingerprints, dict(sorted(bindings.items()))


def _scene_rows(value: Any, *, label: str, expected_count: int | None = None) -> list[dict]:
    if (not isinstance(value, list)
            or (expected_count is not None and len(value) != expected_count)):
        raise ValueError(label + " scene count differs")
    result = []
    fingerprints: set[str] = set()
    seeds: set[int] = set()
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError(label + " scene schema differs")
        fingerprint, seed = row.get("fingerprint"), row.get("seed")
        if (_HEX.fullmatch(str(fingerprint)) is None or type(seed) is not int
                or seed < 0 or fingerprint in fingerprints or seed in seeds):
            raise ValueError(label + " scene identity differs")
        fingerprints.add(str(fingerprint)); seeds.add(seed)
        result.append(deepcopy(dict(row)))
    return result


def _expected_fresh_outer_bindings() -> dict[str, str]:
    """Return the complete, flat binding schema for the frozen outer registry."""
    return {
        "actor_parameters_sha256": (
            manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256),
        "actor_sha256": EXPECTED_ACTOR_SHA256,
        "candidate_batch_reports_sha256": (
            manifest_binding.EXPECTED_CANDIDATE_REPORTS_SHA256),
        "candidate_batches_identity_sha256": (
            manifest_binding.EXPECTED_CANDIDATE_BATCHES_SHA256),
        "contract_sha256": digest(outer_split_api.contract()),
        "designation_file_sha256": EXPECTED_DESIGNATION_SHA256,
        "formal_selection_file_sha256": (
            EXPECTED_FORMAL_SELECTION_FILE_SHA256),
        "manifest_development_projection_sha256": (
            manifest_binding.EXPECTED_DEVELOPMENT_PROJECTION_SHA256),
        "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_validation_file_sha256": (
            manifest_binding.EXPECTED_VALIDATION_SHA256),
        "previous_development_file_sha256": (
            outer_split_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256),
        "retired_projection_file_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
        "retired_projection_report_file_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        "source_expansion_file_sha256": (
            EXPECTED_SOURCE_EXPANSION_FILE_SHA256),
        "source_manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "source_rows_file_sha256": outer_split_api.EXPECTED_SOURCE_ROWS_SHA256,
    }


def _expected_fresh_outer_information_boundary() -> dict[str, Any]:
    """Freeze every negative-information claim made before outer collection."""
    return {
        "actor_loaded_or_inferred": False,
        "candidate_identity_selection_frozen_before_materialisation": True,
        "candidate_population_disjoint_from_all_manifest_base_splits_authenticated": True,
        "candidate_program_or_metrics_read": False,
        "formal_ready": False,
        "full_manifest_json_parsed": False,
        "observations_generated": False,
        "old_outer_previously_exposed": True,
        "outer_actor_rows_collected": False,
        "outer_candidate_scored": False,
        "protected_final_access": False,
        "source_fit_payload_copied": True,
        "source_fit_payload_used_for_selection": False,
        "source_rows_action_labels_read": False,
        "source_rows_fields_read": ["scene_fingerprints"],
        "source_rows_observations_read": False,
        "source_rows_probabilities_read": False,
        "workload_screen_or_replay_run": False,
    }


def _validate_fresh_outer_registry(value: Mapping[str, Any]) -> None:
    """Authenticate the identity-only outer split before any final access."""
    if (not isinstance(value, Mapping)
            or value.get("version") != outer_split_api.VERSION
            or value.get("status") != outer_split_api.STATUS
            or not _content_valid(value)
            or value.get("contract") != outer_split_api.contract()
            or value.get("information_boundary")
                != _expected_fresh_outer_information_boundary()
            or value.get("bindings") != _expected_fresh_outer_bindings()
            or value.get("program_access") is not False
            or value.get("program_predictions_access") is not False
            or value.get("final_audit_rows_access") is not False
            or value.get("final_labels_used_for_selection") is not False
            or value.get("runtime_action_override") is not False
            or value.get("formal_ready") is not False):
        raise ValueError("Fresh outer development registry contract differs")

    fit = _scene_rows(
        value.get("fit_supplement"), label="development fit supplement",
        expected_count=outer_split_api.FIT_SUPPLEMENT_SCENES)
    outer = _scene_rows(
        value.get("development_validation"), label="fresh outer validation",
        expected_count=outer_split_api.VALIDATION_SCENES)
    identities = _scene_rows(
        value.get("selected_outer_identities"),
        label="selected fresh outer identity",
        expected_count=outer_split_api.VALIDATION_SCENES)
    identity_keys = {"batch_index", "family_id", "seed", "fingerprint"}
    projected = []
    for row in outer:
        if (row.get("split") != "development_validation"
                or row.get("family_id") not in outer_split_api.FAMILY_IDS
                or type(row.get("batch_index")) is not int
                or row["batch_index"] < 0):
            raise ValueError("Fresh outer validation identity differs")
        projected.append({key: row[key] for key in identity_keys})
    if (any(row.get("split") != "fit_supplement" for row in fit)
            or any(set(row) != identity_keys for row in identities)
            or identities != projected
            or ({row["seed"] for row in fit} & {row["seed"] for row in outer})
            or ({row["fingerprint"] for row in fit}
                & {row["fingerprint"] for row in outer})):
        raise ValueError("Fresh outer development split identity differs")

    family_counts = dict(sorted(Counter(
        row["family_id"] for row in outer).items()))
    if (family_counts != dict(sorted(outer_split_api.FAMILY_QUOTAS.items()))
            or value.get("statistics", {}).get("fresh_outer_scenes")
                != outer_split_api.VALIDATION_SCENES
            or value.get("statistics", {}).get("fresh_outer_family_counts")
                != family_counts):
        raise ValueError("Fresh outer family quota differs")

    exposed = value.get("previously_exposed_validation_scene_fingerprints")
    if (not isinstance(exposed, list)
            or len(exposed) != outer_split_api.OLD_OUTER_SCENE_COUNT
            or len(set(exposed)) != outer_split_api.OLD_OUTER_SCENE_COUNT
            or any(_HEX.fullmatch(str(item)) is None for item in exposed)
            or set(exposed) & {row["fingerprint"] for row in outer}):
        raise ValueError("Previously exposed outer identity differs")


def _validate_fresh_outer_report(
    report: Mapping[str, Any], registry: Mapping[str, Any],
) -> None:
    """Authenticate the registry's frozen sibling report as a separate input."""
    expected_keys = {
        "bindings", "content_sha256", "formal_ready", "information_boundary",
        "producer_sources", "producer_sources_sha256", "registry_content_sha256",
        "registry_file_sha256", "selection", "statistics", "status", "version",
    }
    selection = report.get("selection") if isinstance(report, Mapping) else None
    selected = registry.get("selected_outer_identities")
    exposed = registry.get("previously_exposed_validation_scene_fingerprints")
    if (not isinstance(report, Mapping)
            or set(report) != expected_keys
            or report.get("version") != outer_split_api.REPORT_VERSION
            or report.get("status") != outer_split_api.STATUS
            or not _content_valid(report)
            or report.get("registry_file_sha256")
                != EXPECTED_EXPANSION_REGISTRY_SHA256
            or report.get("registry_content_sha256")
                != registry.get("content_sha256")
            or report.get("bindings") != registry.get("bindings")
            or report.get("statistics") != registry.get("statistics")
            or report.get("information_boundary")
                != registry.get("information_boundary")
            or report.get("producer_sources") != registry.get("producer_sources")
            or report.get("producer_sources_sha256")
                != registry.get("producer_sources_sha256")
            or report.get("formal_ready") is not False
            or not isinstance(selection, Mapping)
            or set(selection) != {
                "salt", "family_quotas", "selected_identity_sha256",
                "source_rows_scene_fingerprints_sha256",
                "previously_exposed_outer_fingerprints_sha256",
                "prior_expansion_trace_fingerprints_sha256",
            }
            or selection.get("salt") != outer_split_api.SELECTION_SALT
            or selection.get("family_quotas")
                != dict(sorted(outer_split_api.FAMILY_QUOTAS.items()))
            or selection.get("selected_identity_sha256") != digest(selected)
            or selection.get("previously_exposed_outer_fingerprints_sha256")
                != digest(exposed)
            or _HEX.fullmatch(str(selection.get(
                "source_rows_scene_fingerprints_sha256"))) is None
            or _HEX.fullmatch(str(selection.get(
                "prior_expansion_trace_fingerprints_sha256"))) is None):
        raise ValueError("Fresh outer development registry report differs")


def _outer_split_exclusion_scenes(
    manifest: Mapping[str, Any], expansion: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover old-outer seeds from the authenticated manifest and bind fresh64."""
    _validate_fresh_outer_registry(expansion)
    authentication = manifest.get("authentication")
    batches = manifest.get("candidate_batches")
    if (manifest.get("content_sha256")
            != manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256
            or not isinstance(authentication, Mapping)
            or authentication.get(
                "candidate_scenes_disjoint_from_all_base_splits_authenticated")
                is not True
            or not isinstance(batches, list)):
        raise ValueError("Authenticated candidate manifest is required")

    exposed = set(expansion["previously_exposed_validation_scene_fingerprints"])
    by_fingerprint: dict[str, list[dict[str, Any]]] = {
        fingerprint: [] for fingerprint in exposed
    }
    for batch in batches:
        if not isinstance(batch, list):
            raise ValueError("Authenticated candidate manifest batches differ")
        for scene in batch:
            if (isinstance(scene, Mapping)
                    and scene.get("fingerprint") in by_fingerprint):
                by_fingerprint[str(scene["fingerprint"])].append(deepcopy(dict(scene)))
    if any(len(rows) != 1 for rows in by_fingerprint.values()):
        raise ValueError("Exposed outer fingerprint does not map uniquely to a seed")
    # The old expansion collector seeded each partner trajectory from the
    # accepted scene's position (320..383), so recover the exact historical
    # selection order rather than sorting the exposed identities themselves.
    old_outer = []
    for family in outer_split_api.FAMILY_IDS:
        candidates = [
            deepcopy(dict(scene))
            for batch in batches
            for scene in batch
            if isinstance(scene, Mapping) and scene.get("family_id") == family
        ]
        candidates.sort(key=lambda scene: digest({
            "salt": outer_split_api.expansion_api.VALIDATION_ORDER_SALT,
            "family_id": family,
            "fingerprint": scene["fingerprint"],
        }))
        old_outer.extend(
            scene for scene in candidates if scene["fingerprint"] in exposed)
    fresh_outer = [deepcopy(row) for row in expansion["development_validation"]]
    old_seeds = {row.get("seed") for row in old_outer}
    fresh_seeds = {row.get("seed") for row in fresh_outer}
    old_fingerprints = {row.get("fingerprint") for row in old_outer}
    fresh_fingerprints = {row.get("fingerprint") for row in fresh_outer}
    expected_families = Counter(outer_split_api.FAMILY_QUOTAS)
    if (len(old_outer) != outer_split_api.OLD_OUTER_SCENE_COUNT
            or len(old_seeds) != outer_split_api.OLD_OUTER_SCENE_COUNT
            or len(fresh_outer) != outer_split_api.VALIDATION_SCENES
            or len(fresh_seeds) != outer_split_api.VALIDATION_SCENES
            or old_seeds & fresh_seeds
            or old_fingerprints & fresh_fingerprints
            or Counter(row.get("family_id") for row in old_outer)
                != expected_families
            or Counter(row.get("family_id") for row in fresh_outer)
                != expected_families
            or digest([row["fingerprint"] for row in old_outer])
                != EXPECTED_OLD_OUTER_ORDERED_FINGERPRINTS_SHA256
            or digest([row["seed"] for row in old_outer])
                != EXPECTED_OLD_OUTER_ORDERED_SEEDS_SHA256):
        raise ValueError("Old and fresh outer exclusion closure differs")
    return old_outer, fresh_outer


def _development_registry_evidence(
    paths: Sequence[Path],
) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    values: dict[str, dict] = {}
    file_hashes: dict[str, str] = {}
    report_bindings: dict[str, dict[str, str]] = {}
    scenes: list[dict] = []
    for path in paths:
        value, file_sha256 = _read_exact_json(path, "development registry")
        version = value.get("version")
        if version not in {DEVELOPMENT_SUPPLEMENT_VERSION, DEVELOPMENT_EXPANSION_VERSION}:
            raise ValueError("Unexpected development registry version")
        if version in values or not _content_valid(value):
            raise ValueError("Development registry identity differs")
        if (value.get("program_access") is not False
                or value.get("final_audit_rows_access") is not False):
            raise ValueError("Development registry crossed the final/program boundary")
        if version == DEVELOPMENT_SUPPLEMENT_VERSION:
            scenes.extend(_scene_rows(value.get("scenes"), label="development supplement",
                                     expected_count=64))
        else:
            if file_sha256 != EXPECTED_EXPANSION_REGISTRY_SHA256:
                raise ValueError("Exact fresh outer development registry required")
            _validate_fresh_outer_registry(value)
            report_path = _regular(
                path.parent / "report.json", "fresh outer registry report")
            report, report_file_sha256 = _read_exact_json(
                report_path, "fresh outer registry report",
                expected_sha256=EXPECTED_EXPANSION_REPORT_SHA256)
            _validate_fresh_outer_report(report, value)
            report_bindings[str(version)] = {
                "report_file_sha256": report_file_sha256,
                "report_content_sha256": report["content_sha256"],
            }
            scenes.extend(_scene_rows(value.get("fit_supplement"),
                                     label="development fit supplement", expected_count=128))
            scenes.extend(_scene_rows(value.get("development_validation"),
                                     label="development validation", expected_count=64))
        values[str(version)] = value
        file_hashes[str(version)] = file_sha256
    if set(values) != {DEVELOPMENT_SUPPLEMENT_VERSION, DEVELOPMENT_EXPANSION_VERSION}:
        raise ValueError("Exactly the v1 supplement and v8 expansion are required")
    if len({row["fingerprint"] for row in scenes}) != len(scenes):
        raise ValueError("Development registries overlap one another")
    bindings = {
        version: {"file_sha256": file_hashes[version],
                  "content_sha256": values[version]["content_sha256"],
                  **report_bindings.get(version, {})}
        for version in sorted(values)
    }
    return scenes, bindings, values


def _development_registries(
    paths: Sequence[Path],
) -> tuple[list[dict], dict[str, dict]]:
    scenes, bindings, _ = _development_registry_evidence(paths)
    return scenes, bindings


def _retired_identity_projection(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate and return only the declassified seed/fingerprint union."""
    projection_path = _regular(path, "retired identity projection")
    report_path = _regular(
        projection_path.parent / "report.json",
        "retired identity projection report")
    projection = retired_identity_api.read_saved_projection(
        projection_path,
        expected_projection_sha256=(
            EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
        expected_report_sha256=(
            EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
    )
    identities = projection.get("exposed_identities")
    if (not isinstance(identities, list)
            or len(identities)
                != retired_identity_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES):
        raise ValueError("Exact retired identity-only projection required")
    rows = _scene_rows(
        identities, label="retired exposed identity projection",
        expected_count=retired_identity_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES)
    binding = {
        "version": retired_identity_api.VERSION,
        "file_sha256": file_hash(projection_path),
        "content_sha256": projection["content_sha256"],
        "audit_file_sha256": file_hash(report_path),
        "source_file_sha256": {
            source["version"]: source["source_file_sha256"]
            for source in projection["sources"]
        },
        "identity_count": len(rows),
    }
    return rows, binding


def _validate_retired_expansion_binding(
    projection_binding: Mapping[str, Any], expansion: Mapping[str, Any],
) -> None:
    expansion_bindings = expansion.get("bindings")
    if (not isinstance(expansion_bindings, Mapping)
            or "retired_identity_projection" in expansion_bindings
            or expansion_bindings.get("retired_projection_file_sha256")
                != projection_binding.get("file_sha256")
            or expansion_bindings.get("retired_projection_report_file_sha256")
                != projection_binding.get("audit_file_sha256")
            or projection_binding.get("file_sha256")
                != EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256
            or projection_binding.get("audit_file_sha256")
                != EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256
            or projection_binding.get("identity_count")
                != retired_identity_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES):
        raise ValueError(
            "Development expansion retired identity projection differs")


def _historical_public_scene(
    scene: Mapping[str, Any], scene_index: int,
) -> dict[str, Any]:
    """Project one protected row onto fixed public environment inputs only.

    The omitted ``workload_screen`` object contains saved actions, metrics and
    labels.  The historical-final boundary permits none of those values to be
    read, compared, replayed or persisted.  Exact manifest-byte and content
    commitments authenticate the container; this projection supplies only the
    environment state required for independent observation exclusion.
    """
    if (not isinstance(scene, Mapping)
            or set(scene) != _HISTORICAL_PUBLIC_SCENE_FIELDS | {"workload_screen"}):
        raise ValueError("Historical final public scene schema differs")
    result = {
        key: deepcopy(scene[key]) for key in _HISTORICAL_PUBLIC_SCENE_FIELDS
    }
    if (result["id"] != f"diagnostic_final_test_{scene_index:04d}"
            or result["split"] != "final_test"
            or type(result["seed"]) is not int
            or result["seed"] < 0
            or type(result["fingerprint"]) is not str
            or _HEX.fullmatch(result["fingerprint"]) is None):
        raise ValueError("Historical final public scene identity differs")
    return result


def _exact_final_workload_observations(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scene: Mapping[str, Any], scene_index: int,
) -> set[str]:
    """Return every public observation the v8 audit can pass to its program."""
    result: set[str] = set()
    for partner_index, partner in enumerate(PARTNERS):
        env = runtime.environment(scene)
        rng = np.random.default_rng(17000 + partner_index * 1000 + scene_index)
        first_after = None
        while not env.done:
            source = env.snapshot()
            result.add(_observation_hash(env.observations()["robot_2"]))
            if env.state.frame % 10 == 0 and critical_groups(env, "robot_2"):
                for action in ACTIONS:
                    branch = runtime.from_snapshot(source)
                    transition = runtime.step(branch, action)
                    if not transition["done"]:
                        result.add(_observation_hash(
                            branch.observations()["robot_2"]))
            player = partner_action(env, "robot_1", partner, rng)
            transition = runtime.step(env, player)
            if first_after is None:
                first_after = transition["after"]
        # Retired v1/v2/v3 registries included the language-facing WAIT-three
        # branch after the first transition.  Preserve that exact exposure
        # closure for replay and conservatively exclude it from the new split.
        if first_after is not None:
            branch = runtime.from_snapshot(first_after)
            for _ in range(3):
                if branch.done:
                    break
                result.add(_observation_hash(
                    branch.observations()["robot_2"]))
                runtime.step(branch, "WAIT")
    return result


def _old_outer_collection_observations(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scenes: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Replay the exact public inputs used by the retired outer collector."""
    if len(scenes) != outer_split_api.OLD_OUTER_SCENE_COUNT:
        raise ValueError("Old outer observation replay scene count differs")
    result: set[str] = set()
    for local_index, scene in enumerate(scenes):
        scene_index = OLD_OUTER_COLLECTION_SCENE_OFFSET + local_index
        for partner_index, partner in enumerate(PARTNERS):
            env = runtime.environment(scene)
            rng = np.random.default_rng(
                41_900_000 + scene_index * 101 + partner_index)
            while not env.done:
                source = env.snapshot()
                source_sha256 = digest(source)
                result.add(_observation_hash(env.observations()["robot_2"]))
                groups = tuple(critical_groups(env, "robot_2"))
                if groups and env.state.frame % 5 == 0:
                    for player_action in ACTIONS:
                        branch = runtime.from_snapshot(source)
                        transition = runtime.step(branch, player_action)
                        if (transition["submitted_actions"]["robot_2"]
                                != transition["policy_actions"]["robot_2"]):
                            raise RuntimeError(
                                "Old outer replay overrode the frozen Actor")
                        if not branch.done:
                            result.add(_observation_hash(
                                branch.observations()["robot_2"]))
                player = partner_action(env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_sha256:
                    raise RuntimeError(
                        "Old outer partner changed the observation source")
                transition = runtime.step(env, player)
                if (transition["submitted_actions"]["robot_2"]
                        != transition["policy_actions"]["robot_2"]):
                    raise RuntimeError(
                        "Old outer replay overrode the frozen Actor")
    if not result:
        raise RuntimeError("Old outer observation replay is empty")
    return result


def _question_bank_workload_observations(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scene: Mapping[str, Any], scene_index: int,
) -> set[str]:
    """Collect exactly the Actor inputs exposed by the frozen qbank workload."""
    env = runtime.environment(scene)
    rng = random.Random(260_910_800 + scene_index)
    result: set[str] = set()
    for _ in range(min(80, int(env.config.horizon))):
        if int(env.state.frame) >= 1 and not env.done:
            # The question-bank candidate reads the current Actor decision.
            result.add(_observation_hash(env.observations()["robot_2"]))
            branch = runtime.from_snapshot(env.snapshot())
            for _ in range(3):
                if branch.done:
                    break
                result.add(_observation_hash(
                    branch.observations()["robot_2"]))
                runtime.step(branch, "WAIT")
        if env.done:
            break
        # The ordinary qbank trajectory also reads the Actor at this frame.
        result.add(_observation_hash(env.observations()["robot_2"]))
        before = digest(env.snapshot())
        player_action = partner_action(
            env, "robot_1", "assertive", rng)
        if digest(env.snapshot()) != before:
            raise RuntimeError(
                "Question-bank partner changed the observation source")
        runtime.step(env, player_action)
    return result


def _tutorial_workload_observations(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scene: Mapping[str, Any], scene_index: int,
) -> set[str]:
    """Collect states shown by the exact frozen neutral choreography."""
    if scene_index != 0:
        raise ValueError("Neutral tutorial has exactly one registered scene")
    from ui import warehouse_alignment_r41_tutorial as base

    env = runtime.environment(scene)
    result = {_observation_hash(env.observations()["robot_2"])}

    def advance(joint: Mapping[str, str]) -> None:
        if env.done:
            raise RuntimeError("Neutral tutorial terminated during replay")
        env.step(dict(joint))
        result.add(_observation_hash(env.observations()["robot_2"]))

    prelude = (
        {"robot_1": "RIGHT", "robot_2": "LEFT"},
        {"robot_1": "WAIT", "robot_2": "WAIT"},
        {"robot_1": "RIGHT", "robot_2": "WAIT"},
        {"robot_1": "WAIT", "robot_2": "UP"},
        {"robot_1": "UP", "robot_2": "DOWN"},
    )
    for joint in prelude:
        advance(joint)
    worker = env.state.by_id("robot_1")
    partner = env.state.by_id("robot_2")
    candidates = []
    for task in env.state.tasks:
        to_pickup = base._path_actions(
            env, worker.position, task.pickup_position,
            blocked=(partner.position,))
        to_delivery = base._path_actions(
            env, task.pickup_position, task.delivery_position,
            blocked=(partner.position,))
        candidates.append((
            len(to_pickup) + len(to_delivery), task.task_id, task,
            to_pickup, to_delivery))
    _, task_id, task, to_pickup, _ = min(
        candidates, key=lambda row: row[:2])
    for action in to_pickup:
        advance({"robot_1": action, "robot_2": "WAIT"})
    if env.state.by_id("robot_1").carrying_task_id != task_id:
        raise RuntimeError("Neutral tutorial did not pick up its task")
    to_delivery = base._path_actions(
        env, env.state.by_id("robot_1").position, task.delivery_position,
        blocked=(env.state.by_id("robot_2").position,))
    for action in to_delivery:
        advance({"robot_1": action, "robot_2": "WAIT"})
    return result


def _auxiliary_workload_observations(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scene: Mapping[str, Any], *, split: str, scene_index: int,
) -> set[str]:
    if split == "tutorial":
        return _tutorial_workload_observations(runtime, scene, scene_index)
    if split == "question_bank":
        return _question_bank_workload_observations(
            runtime, scene, scene_index)
    raise ValueError("Unsupported auxiliary workload split")


def _ordered_candidates(
    manifest: Mapping[str, Any], family: str, selection_salt: bytes,
) -> list[dict]:
    rows = [deepcopy(row) for batch in manifest["candidate_batches"]
            for row in batch if row["family_id"] == family]
    rows.sort(key=lambda row: digest({
        "salt": selection_salt.hex(),
        "family_id": family,
        "fingerprint": row["fingerprint"],
    }))
    return rows


def _candidate_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch in manifest.get("candidate_batches", []):
        if not isinstance(batch, list):
            raise ValueError("Manifest candidate batch schema differs")
        for row in batch:
            if not isinstance(row, Mapping):
                raise ValueError("Manifest candidate schema differs")
            fingerprint = str(row.get("fingerprint"))
            if _HEX.fullmatch(fingerprint) is None or fingerprint in result:
                raise ValueError("Manifest candidate fingerprint registry differs")
            result[fingerprint] = deepcopy(dict(row))
    if not result:
        raise ValueError("Manifest candidate population is empty")
    return result


def _projected_identity_exclusions(
    *, manifest: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[int], dict[str, Any]]:
    """Authenticate retired identities without reconstructing their workloads.

    The v1/v2 declassification boundary permits only the fixed seed and
    fingerprint union to be used.  Candidate geometry is consulted solely to
    prove that each identity has one unique public-manifest mapping.  No Actor,
    trajectory, public observation, outcome, or metric is derived here.
    """
    candidates = _candidate_index(manifest)
    fingerprints: set[str] = set()
    seeds: set[int] = set()
    for identity in identities:
        if not isinstance(identity, Mapping):
            raise ValueError("Retired projected identity differs")
        fingerprint = str(identity.get("fingerprint"))
        seed = identity.get("seed")
        candidate = candidates.get(fingerprint)
        if (candidate is None or type(seed) is not int
                or candidate.get("seed") != seed):
            raise ValueError(
                "Retired projected identity has no unique public candidate")
        fingerprints.add(fingerprint)
        seeds.add(seed)
    if (len(fingerprints)
            != retired_identity_api.EXPECTED_GLOBAL_EXPOSED_IDENTITIES
            or len(seeds) != len(fingerprints)):
        raise ValueError("Retired projected identity exclusion differs")
    statistics = {
        "projection_version": retired_identity_api.VERSION,
        "projected_identity_count": len(fingerprints),
        "projected_fingerprints_sha256": digest(sorted(fingerprints)),
        "projected_seeds_sha256": digest(sorted(seeds)),
        "retired_actor_executed": False,
        "retired_observations_derived": False,
        "retired_outcomes_accessed": False,
        "retired_trajectories_accessed": False,
        "retired_metrics_accessed": False,
    }
    return fingerprints, seeds, statistics


def _selection_exposure_closure(
    *, runtime: R41DiagnosticOnlineAlignmentRuntime, actor: Any,
    manifest: Mapping[str, Any], registries: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[int], set[str], dict[str, Any]]:
    """Reconstruct a newly generated claim-bound exclusion trace."""
    candidates = _candidate_index(manifest)
    fingerprints: set[str] = set()
    seeds: set[int] = set()
    observations: set[str] = set()
    per_registry: dict[str, Any] = {}
    for registry in registries:
        version = str(registry.get("version"))
        scenes = registry.get("scenes")
        trace = registry.get("selection_trace")
        if not isinstance(scenes, list) or not isinstance(trace, list) or not trace:
            raise ValueError("Retired exposure registry schema differs")
        accepted = {str(row.get("fingerprint")) for row in scenes}
        trace_accepted: set[str] = set()
        replayed = failed_screens = 0
        registry_observations: set[str] = set()
        for row in trace:
            if not isinstance(row, Mapping):
                raise ValueError("Retired selection trace row differs")
            fingerprint = str(row.get("fingerprint"))
            candidate = candidates.get(fingerprint)
            scene_index = row.get("scene_index")
            if (candidate is None or type(scene_index) is not int or scene_index < 0
                    or row.get("family_id") != candidate.get("family_id")
                    or type(row.get("accepted")) is not bool):
                raise ValueError("Retired selection trace identity differs")
            fingerprints.add(fingerprint)
            seed = candidate.get("seed")
            if type(seed) is not int or seed < 0:
                raise ValueError("Retired trace candidate seed differs")
            seeds.add(seed)
            if row["accepted"]:
                trace_accepted.add(fingerprint)
            receipt = row.get("workload_receipt")
            if receipt is None:
                continue
            if not isinstance(receipt, Mapping):
                raise ValueError("Retired workload receipt differs")
            actual_receipt = screen_scene(
                candidate, split="final_test", scene_index=scene_index, actor=actor)
            if dict(receipt) != actual_receipt:
                raise ValueError("Retired workload receipt does not replay")
            if receipt.get("passed") is not True:
                failed_screens += 1
                continue
            hashes = _exact_final_workload_observations(
                runtime, candidate, scene_index)
            claimed_count = row.get(
                "public_observation_hash_count",
                row.get("public_observation_count"),
            )
            claimed_digest = row.get(
                "public_observation_hashes_sha256",
                row.get("public_observations_sha256"),
            )
            if claimed_count != len(hashes) or claimed_digest != digest(sorted(hashes)):
                raise ValueError("Retired public-observation exposure does not replay")
            observations.update(hashes)
            registry_observations.update(hashes)
            replayed += 1
        if trace_accepted != accepted:
            raise ValueError("Retired accepted scenes differ from its selection trace")
        per_registry[version] = {
            "accepted_scenes": len(accepted),
            "touched_trace_scenes": len({str(row["fingerprint"]) for row in trace}),
            "replayed_trace_workloads": replayed,
            "failed_workload_screens": failed_screens,
            "public_observation_count": len(registry_observations),
            "public_observations_sha256": digest(sorted(registry_observations)),
        }
    return fingerprints, seeds, observations, dict(sorted(per_registry.items()))


def _build_v3_tombstone(
    *, runtime: R41DiagnosticOnlineAlignmentRuntime, actor: Any,
    manifest: Mapping[str, Any], selected: Mapping[str, Any],
    legacy_development_hashes: set[str],
    projected_retired_fingerprints: set[str],
    projected_retired_seeds: set[int],
    claim_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Burn the never-materialized v3 selection inside the claimed campaign."""
    base_fingerprints = {
        row["fingerprint"]
        for rows in manifest["splits"].values()
        for row in rows
    }
    participant_fingerprints = {
        row["fingerprint"] for key in ("X", "Y") for row in selected[key]
    }
    participant_seeds = {
        row["seed"] for key in ("X", "Y") for row in selected[key]
    }
    excluded_fingerprints = (
        set(base_fingerprints) | set(participant_fingerprints)
        | set(projected_retired_fingerprints)
    )
    excluded_seeds = set(participant_seeds) | set(projected_retired_seeds)
    forbidden = set(legacy_development_hashes)
    scenes: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    accepted_hashes: set[str] = set()
    for family in FAMILY_IDS:
        family_count = 0
        for candidate in legacy_v3._ordered_candidates(manifest, family):
            if family_count >= FAMILY_QUOTAS[family]:
                break
            reason = None
            receipt = None
            hashes: set[str] = set()
            if (candidate["fingerprint"] in excluded_fingerprints
                    or candidate["seed"] in excluded_seeds):
                reason = "excluded_scene_or_seed"
            else:
                scene_index = len(scenes)
                receipt = screen_scene(
                    candidate, split="final_test", scene_index=scene_index,
                    actor=actor)
                if receipt.get("passed") is not True:
                    reason = "exact_workload_failed"
                else:
                    hashes = _exact_final_workload_observations(
                        runtime, candidate, scene_index)
                    if hashes & forbidden:
                        reason = "development_observation_overlap"
                    elif hashes & accepted_hashes:
                        reason = "fresh_holdout_observation_overlap"
            admitted = reason is None
            trace.append({
                "family_id": family,
                "fingerprint": candidate["fingerprint"],
                "scene_index": len(scenes),
                "accepted": admitted,
                "rejection_reason": reason,
                "workload_receipt": receipt,
                "public_observation_hash_count": len(hashes),
                "public_observation_hashes_sha256": digest(sorted(hashes)),
            })
            if admitted:
                frozen = deepcopy(candidate)
                frozen["id"] = f"diagnostic_fresh_final_{len(scenes):04d}"
                frozen["split"] = "fresh_final_test"
                frozen["workload_screen"] = receipt
                scenes.append(frozen)
                accepted_hashes.update(hashes)
                forbidden.update(hashes)
                excluded_fingerprints.add(frozen["fingerprint"])
                excluded_seeds.add(frozen["seed"])
                family_count += 1
        if family_count != FAMILY_QUOTAS[family]:
            raise RuntimeError("Fresh-final v3 family quota unavailable: " + family)
    statistics = {
        "accepted": len(scenes),
        "evaluated": len(trace),
        "rejected": dict(sorted(Counter(
            row["rejection_reason"] for row in trace
            if row["rejection_reason"]
        ).items())),
        "families": dict(sorted(Counter(
            row["family_id"] for row in scenes
        ).items())),
        "development_public_observation_count": len(legacy_development_hashes),
        "retired_identity_exclusion_count": len(projected_retired_fingerprints),
        "retired_public_observation_count": 0,
        "retired_actor_executed": False,
        "fresh_final_public_observation_count": len(accepted_hashes),
        "public_observation_overlap": 0,
    }
    value = {
        "version": V3_TOMBSTONE_VERSION,
        "status": "burned_program_blind_exclusion",
        "legacy_selection_version": legacy_v3.VERSION,
        "legacy_selection_source_sha256": file_hash(Path(legacy_v3.__file__).resolve()),
        "legacy_selection_contract_sha256": digest(legacy_v3.contract()),
        "claim_binding": deepcopy(dict(claim_binding)),
        "scenes": scenes,
        "selection_trace": trace,
        "statistics": statistics,
        "program_access": False,
        "program_predictions_access": False,
        "final_labels_used_for_program_selection": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _select(
    *, runtime: R41DiagnosticOnlineAlignmentRuntime, actor: Any,
    manifest: Mapping[str, Any], selection_salt: bytes,
    excluded_fingerprints: set[str],
    excluded_seeds: set[int], forbidden_observation_hashes: set[str],
) -> tuple[list[dict], list[dict], dict[str, Any], set[str]]:
    accepted: list[dict] = []
    trace: list[dict] = []
    accepted_hashes: set[str] = set()
    for family in FAMILY_IDS:
        count = 0
        for candidate in _ordered_candidates(manifest, family, selection_salt):
            if count >= FAMILY_QUOTAS[family]:
                break
            reason = None
            receipt = None
            hashes: set[str] = set()
            if (candidate["fingerprint"] in excluded_fingerprints
                    or candidate["seed"] in excluded_seeds):
                reason = "excluded_scene_or_seed"
            else:
                index = len(accepted)
                receipt = screen_scene(candidate, split="final_test",
                                       scene_index=index, actor=actor)
                if receipt.get("passed") is not True:
                    reason = "exact_workload_failed"
                else:
                    hashes = _exact_final_workload_observations(
                        runtime, candidate, index)
                    if hashes & forbidden_observation_hashes:
                        reason = "excluded_public_observation_overlap"
                    elif hashes & accepted_hashes:
                        reason = "within_holdout_public_observation_overlap"
            admitted = reason is None
            trace.append({
                "family_id": family,
                "fingerprint": candidate["fingerprint"],
                "seed": candidate["seed"],
                "scene_index": len(accepted),
                "accepted": admitted,
                "rejection_reason": reason,
                "workload_receipt": receipt,
                "public_observation_count": len(hashes),
                "public_observations_sha256": digest(sorted(hashes)),
            })
            if admitted:
                scene = deepcopy(candidate)
                scene["id"] = f"diagnostic_v8_fresh_final_{len(accepted):04d}"
                scene["split"] = "fresh_final_test"
                scene["workload_screen"] = receipt
                accepted.append(scene)
                accepted_hashes.update(hashes)
                excluded_fingerprints.add(scene["fingerprint"])
                excluded_seeds.add(scene["seed"])
                forbidden_observation_hashes.update(hashes)
                count += 1
        if count != FAMILY_QUOTAS[family]:
            raise RuntimeError("Fresh-final family quota unavailable: " + family)
    statistics = {
        "accepted": len(accepted),
        "evaluated": len(trace),
        "families": dict(sorted(Counter(row["family_id"] for row in accepted).items())),
        "rejected": dict(sorted(Counter(
            row["rejection_reason"] for row in trace if row["rejection_reason"]
        ).items())),
        "fresh_final_public_observation_count": len(accepted_hashes),
        "public_observation_overlap": 0,
        "unique_fingerprints": len({row["fingerprint"] for row in accepted}),
        "unique_seeds": len({row["seed"] for row in accepted}),
    }
    return accepted, trace, statistics, accepted_hashes


def build(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    selected_scenes_path: str | Path,
    development_registry_paths: Sequence[str | Path],
    development_rows_paths: Sequence[str | Path],
    legacy_v3_rows_path: str | Path,
    expansion_rows_path: str | Path,
    retired_identity_projection_path: str | Path, output: str | Path,
    claim_receipt_path: str | Path, expected_claim_sha256: str,
    expected_campaign_key: str, expected_candidate_identity_sha256: str,
    selection_salt_path: str | Path,
) -> dict[str, Any]:
    """Run the holdout without exporting salt or private worker tracebacks."""

    def sensitive(
        *, actor_path: str | Path, protocol_path: str | Path,
        manifest_path: str | Path, designation_path: str | Path,
        selected_scenes_path: str | Path,
        development_registry_paths: Sequence[str | Path],
        development_rows_paths: Sequence[str | Path],
        legacy_v3_rows_path: str | Path,
        expansion_rows_path: str | Path,
        retired_identity_projection_path: str | Path, output: str | Path,
        claim_receipt_path: str | Path, expected_claim_sha256: str,
        expected_campaign_key: str, expected_candidate_identity_sha256: str,
        selection_salt_path: str | Path,
        _sensitive_phase_state: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        # Freeze the complete producer closure before claim authentication.  The
        # committed private salt is intentionally excluded and remains unread
        # until the irreversible holdout marker has been published.
        initial_sources = producer_sources()
        claim_dir, _ = _claim_receipt(
            claim_receipt_path, expected_claim_sha256=expected_claim_sha256,
            expected_campaign_key=expected_campaign_key,
            expected_candidate_identity_sha256=expected_candidate_identity_sha256,
        )
        if producer_sources() != initial_sources:
            raise RuntimeError(
                "Fresh-final producer source changed during claim authentication")
        if _sensitive_phase_state is not None:
            # The irreversible campaign claim is authenticated.  Sanitize all
            # later failures before any manifest/program bytes are hashed,
            # copied, parsed, or retained by descriptor tracebacks.
            _sensitive_phase_state["active"] = True
        output_path = Path(output).expanduser().absolute()
        if (output_path.exists() or output_path.is_symlink()
                or not output_path.parent.is_dir() or output_path.parent.is_symlink()):
            raise ValueError("Fresh-final output must be a new directory under an existing parent")
        actor_path = _regular(actor_path, "frozen Actor")
        protocol_path = _regular(protocol_path, "training protocol")
        manifest_path = _regular(manifest_path, "conflict manifest")
        designation_path = _regular(designation_path, "Actor designation")
        selected_path = _regular(selected_scenes_path, "formal X/Y selection")
        development_paths = [_regular(path, "development registry")
                             for path in development_registry_paths]
        fresh_outer_paths = [
            path for path in development_paths
            if file_hash(path) == EXPECTED_EXPANSION_REGISTRY_SHA256
        ]
        if len(fresh_outer_paths) != 1:
            raise ValueError("Exact fresh outer development registry required")
        fresh_outer_path = fresh_outer_paths[0]
        fresh_outer_report_path = _regular(
            fresh_outer_path.parent / "report.json", "fresh outer registry report")
        if file_hash(fresh_outer_report_path) != EXPECTED_EXPANSION_REPORT_SHA256:
            raise ValueError("Exact fresh outer development registry report required")
        row_paths = [_regular(path, "development rows", maximum=MAX_NPZ_BYTES)
                     for path in development_rows_paths]
        legacy_rows_path = _regular(
            legacy_v3_rows_path, "legacy v3 development rows", maximum=MAX_NPZ_BYTES)
        expansion_rows_path = _regular(
            expansion_rows_path, "expansion development rows", maximum=MAX_NPZ_BYTES)
        retired_projection_path = _regular(
            retired_identity_projection_path, "retired identity projection")
        retired_projection_report_path = _regular(
            retired_projection_path.parent / "report.json",
            "retired identity projection report")
        claim_receipt_file = _regular(claim_receipt_path, "final-once claim")
        input_paths: dict[str, Path] = {
            "actor": actor_path,
            "protocol": protocol_path,
            "manifest": manifest_path,
            "designation": designation_path,
            "selected_scenes": selected_path,
            "legacy_v3_rows": legacy_rows_path,
            "expansion_rows": expansion_rows_path,
        }
        input_paths.update({
            f"development_registry:{index}": path
            for index, path in enumerate(development_paths)
        })
        input_paths.update({
            f"development_rows:{index}": path
            for index, path in enumerate(row_paths)
        })
        input_paths.update({
            "retired_identity_projection": retired_projection_path,
            "retired_identity_projection_report": retired_projection_report_path,
            "fresh_outer_registry_report": fresh_outer_report_path,
        })
        input_paths.update(_implicit_input_paths(
            manifest_path=manifest_path,
            designation_path=designation_path,
            claim_dir=claim_dir,
            claim_receipt_path=claim_receipt_file,
        ))
        expected_input_hashes = _input_hash_snapshot(input_paths)
        semantic_originals = {
            "actor": actor_path,
            "protocol": protocol_path,
            "designation": designation_path,
            "fresh_outer_registry": fresh_outer_path,
            "fresh_outer_registry_report": fresh_outer_report_path,
            **{
                "designation_" + name.split(":", 1)[1]: path
                for name, path in input_paths.items()
                if name.startswith("designation_component:")
            },
        }
        semantic_expected = {
            "actor": EXPECTED_ACTOR_SHA256,
            "protocol": EXPECTED_PROTOCOL_SHA256,
            "designation": EXPECTED_DESIGNATION_SHA256,
            "fresh_outer_registry": EXPECTED_EXPANSION_REGISTRY_SHA256,
            "fresh_outer_registry_report": EXPECTED_EXPANSION_REPORT_SHA256,
            "designation_actor": designation_api.EXPECTED_ACTOR_SHA256,
            "designation_protocol": designation_api.EXPECTED_PROTOCOL_FILE_SHA256,
            "designation_training_ledger": designation_api.EXPECTED_LEDGER_SHA256,
            "designation_dual_evaluation": (
                designation_api.EXPECTED_DUAL_EVALUATION_SHA256),
            "designation_failure_closeout": designation_api.EXPECTED_CLOSEOUT_SHA256,
        }
        if set(semantic_originals) != set(semantic_expected):
            raise ValueError("Fresh-final semantic input registry differs")
        semantic_relative = {
            "actor": "runtime/actor.npz",
            "protocol": "runtime/protocol.json",
            "designation": "designation/designation.json",
            "fresh_outer_registry": "fresh_outer/development_expansion.json",
            "fresh_outer_registry_report": "fresh_outer/report.json",
            **{
                name: "designation/components/" + name.removeprefix(
                    "designation_") + path.suffix
                for name, path in semantic_originals.items()
                if name.startswith("designation_") and name != "designation"
            },
        }
        semantic_maximum = {
            name: MAX_NPZ_BYTES if name in {"actor", "designation_actor"}
            else MAX_JSON_BYTES
            for name in semantic_originals
        }
        semantic_snapshot = None

        def guard(phase: str) -> None:
            if semantic_snapshot is not None:
                try:
                    semantic_snapshot.verify()
                except RuntimeError as error:
                    raise RuntimeError(
                        "Fresh-final immutable input changed during " + phase
                    ) from error
            _guard_frozen_holdout_inputs(
                input_paths, expected_sources=initial_sources,
                expected_hashes=expected_input_hashes, phase=phase)

        def register_phase_input(name: str, path: Path) -> None:
            if name in input_paths or name in expected_input_hashes:
                raise ValueError("Fresh-final phase input was registered twice")
            guard("pre-" + name + " registration")
            input_paths[name] = _regular(path, name + " phase marker")
            expected_input_hashes[name] = file_hash(input_paths[name])
            guard("post-" + name + " registration")

        with input_snapshot_api.ImmutableInputSnapshot(
                semantic_originals, expected_sha256=semantic_expected,
                relative_names=semantic_relative, maximum_bytes=semantic_maximum,
                prefix="warehouse-r41-fresh-final-") as semantic_snapshot:
            frozen_development_paths = [
                semantic_snapshot.paths["fresh_outer_registry"]
                if path == fresh_outer_path else path
                for path in development_paths
            ]
            guard("pre-holdout authentication")
            # Re-run the semantic claim/anchor/candidate validation against the bytes
            # now covered by the transaction snapshot.  This closes the interval
            # between the initial claim gate and transitive-input discovery.
            _claim_receipt(
                claim_receipt_file, expected_claim_sha256=expected_claim_sha256,
                expected_campaign_key=expected_campaign_key,
                expected_candidate_identity_sha256=expected_candidate_identity_sha256,
            )
            guard("claim-chain reauthentication")
            holdout_started_value = {
                "version": VERSION, "status": "started_no_retry",
                "campaign_key": expected_campaign_key,
                "candidate_identity_sha256": expected_candidate_identity_sha256,
                "attempt_started_sha256": expected_claim_sha256,
                "candidate_authenticated_sha256": file_hash(
                    claim_dir / "candidate_authenticated.json"),
                "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
            }
            holdout_started_sha256 = _json_file_sha256(holdout_started_value)
            holdout_started_path = _claim_marker(
                claim_dir, "holdout_started.json", holdout_started_value,
                before_publish=lambda: guard("holdout phase start"),
            )
            guard("post-holdout phase start")
            input_paths["holdout_started"] = _regular(
                holdout_started_path, "holdout phase marker")
            expected_input_hashes["holdout_started"] = holdout_started_sha256
            guard("holdout transaction freeze")
            selection_salt = _read_committed_salt(selection_salt_path)
            guard("private salt authentication")
            candidate_marker, candidate_marker_sha256 = _read_exact_json(
                claim_dir / "candidate_authenticated.json",
                "strict candidate authentication phase")
            if (len(row_paths) != 1
                    or file_hash(actor_path) != EXPECTED_ACTOR_SHA256
                    or file_hash(protocol_path) != EXPECTED_PROTOCOL_SHA256
                    or file_hash(manifest_path) != EXPECTED_MANIFEST_SHA256
                    or file_hash(designation_path) != EXPECTED_DESIGNATION_SHA256
                    or file_hash(selected_path) != EXPECTED_SELECTED_SCENES_SHA256
                    or candidate_marker.get("rows_file_sha256") != file_hash(row_paths[0])
                    or candidate_marker.get("prior_v7_rows_file_sha256")
                        != file_hash(legacy_rows_path)
                    or candidate_marker.get("expansion_rows_file_sha256")
                        != file_hash(expansion_rows_path)):
                raise ValueError("Claim-authenticated final campaign input differs")
            guard("candidate input authentication")

            manifest = manifest_binding.read_saved_manifest(
                manifest_path,
                actor_path=semantic_snapshot.paths["actor"],
                replay_scope="development")
            if (manifest.get("version") != MANIFEST_VERSION
                    or manifest.get("content_sha256")
                        != manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256):
                raise ValueError("Frozen conflict manifest identity differs")
            selected, selected_file_sha256 = _read_exact_json(
                selected_path, "formal X/Y selection",
                expected_sha256=EXPECTED_SELECTED_SCENES_SHA256)
            designation_component_snapshots = {
                name: semantic_snapshot.paths["designation_" + name]
                for name in designation_api.ARTIFACT_NAMES
            }
            designation_component_originals = {
                name: semantic_snapshot.original_paths["designation_" + name]
                for name in designation_api.ARTIFACT_NAMES
            }
            designation = designation_binding.read_bound_designation_snapshot(
                semantic_snapshot.paths["designation"],
                original_path=designation_path,
                components=designation_component_snapshots,
                original_components=designation_component_originals,
                expected_sha256=EXPECTED_DESIGNATION_SHA256)
            guard("manifest and designation authentication")
            actor_raw, actor_sha256 = _read_regular_bytes(
                semantic_snapshot.paths["actor"], "frozen Actor",
                maximum=MAX_NPZ_BYTES)
            if actor_sha256 != EXPECTED_ACTOR_SHA256:
                raise ValueError("Frozen diagnostic Actor bytes differ")
            if (selected.get("release_eligible") is not True
                    or selected.get("actor_sha256") != actor_sha256
                    or manifest.get("frozen_actor", {}).get("sha256") != actor_sha256
                    or designation.get("bindings", {}).get("actor_sha256") != actor_sha256
                    or designation.get("behavior_performance_gate_waived") is not True
                    or designation.get("runtime_action_override") is not False):
                raise ValueError("Frozen diagnostic Actor/scene designation differs")

            (development_scenes, development_bindings,
             development_values) = _development_registry_evidence(
                 frozen_development_paths)
            retired_identities, retired_projection_binding = (
                _retired_identity_projection(retired_projection_path))
            supplement = development_values[DEVELOPMENT_SUPPLEMENT_VERSION]
            expansion = development_values[DEVELOPMENT_EXPANSION_VERSION]
            _validate_retired_expansion_binding(
                retired_projection_binding, expansion)
            old_outer_scenes, fresh_outer_scenes = (
                _outer_split_exclusion_scenes(manifest, expansion))
            if (candidate_marker.get("development_registries") != {
                    version: binding["file_sha256"]
                    for version, binding in sorted(development_bindings.items())
                }
                    or development_bindings[DEVELOPMENT_EXPANSION_VERSION][
                        "file_sha256"] != EXPECTED_EXPANSION_REGISTRY_SHA256
                    or supplement.get("bindings", {}).get("actor_sha256") != actor_sha256
                    or supplement.get("bindings", {}).get("source_manifest_sha256")
                        != file_hash(manifest_path)
                    or supplement.get("bindings", {}).get("selected_scenes_sha256")
                        != selected_file_sha256
                    or expansion.get("bindings", {}).get("actor_sha256") != actor_sha256
                    or expansion.get("bindings", {}).get("source_manifest_file_sha256")
                        != file_hash(manifest_path)
                    or expansion.get("bindings", {}).get("formal_selection_file_sha256")
                        != selected_file_sha256
                    or expansion.get("bindings", {}).get("designation_file_sha256")
                        != file_hash(designation_path)
                    or expansion.get("bindings", {}).get("previous_development_file_sha256")
                        != next(binding["file_sha256"] for version, binding
                                in development_bindings.items()
                                if version == DEVELOPMENT_SUPPLEMENT_VERSION)):
                raise ValueError("Development registry frozen-input binding differs")
            guard("development and retired identity projection authentication")
            registered = [deepcopy(row) for rows in manifest["splits"].values() for row in rows]
            xy = [deepcopy(row) for key in ("X", "Y") for row in selected.get(key, [])]
            if len(xy) != 6:
                raise ValueError("Exactly six frozen X/Y participant scenes are required")
            # The fitted row registry contains the train/conflict splits plus the two
            # explicitly registered supplements.  Tutorial and frozen question-bank
            # scenes were never fit rows; requiring them here would misstate row
            # coverage.  Their exact public-observation closure is replayed separately
            # below and is still excluded from the fresh final holdout.
            fitted_manifest_splits = ("train", "conflict_validation")
            required_development_fingerprints = {
                row["fingerprint"] for row in development_scenes
            } | {
                row["fingerprint"]
                for split_name in fitted_manifest_splits
                for row in manifest["splits"][split_name]
            }
            development_hashes, development_row_scenes, row_bindings = (
                _npz_development_observations(
                    row_paths,
                    required_fingerprints=required_development_fingerprints,
                )
            )
            legacy_development_hashes, legacy_row_scenes, legacy_rows_binding = (
                _npz_development_observations(
                    [legacy_rows_path], required_fingerprints=set())
            )
            expansion_development_hashes, expansion_row_scenes, expansion_rows_binding = (
                _npz_development_observations(
                    [expansion_rows_path], required_fingerprints=set())
            )
            guard("development row authentication")
            protocol_raw, protocol_file_sha256 = _read_regular_bytes(
                semantic_snapshot.paths["protocol"], "training protocol")
            if protocol_file_sha256 != EXPECTED_PROTOCOL_SHA256:
                raise ValueError("Training protocol bytes differ")
            protocol = _decode_json(protocol_raw, "training protocol")
            manifest_content = manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256
            runtime = manifest_binding.build_runtime(
                actor_path=semantic_snapshot.paths["actor"],
                protocol_path=semantic_snapshot.paths["protocol"],
                manifest_path=manifest_path)
            # Keep the immutable fd-derived snapshots alive for the entire holdout
            # transaction.  The runtime must never reopen caller-controlled Actor or
            # protocol paths after authentication.
            runtime._holdout_runtime_input_snapshot = semantic_snapshot
            guard("runtime construction")

            claim_binding = {
                "campaign_key": expected_campaign_key,
                "attempt_started_sha256": expected_claim_sha256,
                "candidate_identity_sha256": expected_candidate_identity_sha256,
                "candidate_authenticated_sha256": candidate_marker_sha256,
            }
            projected_fingerprints, projected_seeds, projection_stats = (
                _projected_identity_exclusions(
                    manifest=manifest, identities=retired_identities))
            guard("retired identity projection mapping")
            v3_tombstone = _build_v3_tombstone(
                runtime=runtime, actor=runtime.actor, manifest=manifest, selected=selected,
                legacy_development_hashes=legacy_development_hashes,
                projected_retired_fingerprints=projected_fingerprints,
                projected_retired_seeds=projected_seeds,
                claim_binding=claim_binding,
            )
            guard("v3 exclusion construction")
            v3_exposure_registries = [{
                "version": V3_TOMBSTONE_VERSION,
                "scenes": v3_tombstone["scenes"],
                "selection_trace": v3_tombstone["selection_trace"],
            }]
            v3_fingerprints, v3_seeds, v3_observations, v3_exposure_stats = (
                _selection_exposure_closure(
                    runtime=runtime, actor=runtime.actor, manifest=manifest,
                    registries=v3_exposure_registries,
                )
            )
            exposed_fingerprints = projected_fingerprints | v3_fingerprints
            exposed_seeds = projected_seeds | v3_seeds
            exposed_observations = set(v3_observations)
            exposure_stats = {
                "retired_identity_projection": projection_stats,
                **v3_exposure_stats,
            }
            all_excluded = [*registered, *xy, *development_scenes,
                            *old_outer_scenes, *fresh_outer_scenes,
                            *(row for registry in v3_exposure_registries
                              for row in registry["scenes"])]
            excluded_fingerprints = (
                {row["fingerprint"] for row in all_excluded} | exposed_fingerprints)
            excluded_seeds = {row["seed"] for row in all_excluded} | exposed_seeds

            # Formal X/Y, public tutorial/question-bank workloads, and every burned
            # holdout are public-observation exclusions.
            # The historical final split is added by the single-use claimed phase
            # below; the development manifest intentionally contains no final scenes.
            xy_observations: set[str] = set()
            for index, scene in enumerate(xy):
                xy_observations.update(
                    _exact_final_workload_observations(runtime, scene, index))
            auxiliary_development_observations: set[str] = set()
            auxiliary_development_scenes = []
            for split_name in ("tutorial", "question_bank"):
                split_scenes = manifest["splits"][split_name]
                auxiliary_development_scenes.extend(split_scenes)
                for index, scene in enumerate(split_scenes):
                    auxiliary_development_observations.update(
                        _auxiliary_workload_observations(
                            runtime, scene, split=split_name, scene_index=index))
            old_outer_observations = _old_outer_collection_observations(
                runtime, old_outer_scenes)
            all_development_hashes = (
                set(development_hashes)
                | set(legacy_development_hashes)
                | set(expansion_development_hashes)
            )
            all_development_fingerprints = (
                set(development_row_scenes)
                | set(legacy_row_scenes)
                | set(expansion_row_scenes)
            )
            replayed_preexisting_observations = (
                set(exposed_observations)
                | set(xy_observations)
                | set(auxiliary_development_observations)
                | set(old_outer_observations)
            )
            preexisting_observations = (
                set(all_development_hashes) | set(replayed_preexisting_observations)
            )
            guard("pre-historical final access")
            scenes, trace, statistics, accepted_hashes, historical_exclusion_receipt = (
                _build_claimed_historical_final_observation_exclusion(
                    manifest_path=manifest_path,
                    runtime=runtime,
                    selection_manifest=manifest,
                    selection_salt=selection_salt,
                    selected_scenes_path=selected_path,
                    development_registry_paths=frozen_development_paths,
                    retired_identity_projection_path=retired_projection_path,
                    v3_tombstone=v3_tombstone,
                    claim_receipt_path=claim_receipt_path,
                    expected_claim_sha256=expected_claim_sha256,
                    expected_campaign_key=expected_campaign_key,
                    expected_candidate_identity_sha256=(
                        expected_candidate_identity_sha256),
                    excluded_fingerprints=excluded_fingerprints,
                    excluded_seeds=excluded_seeds,
                    preexisting_observation_hashes=preexisting_observations,
                    replayed_excluded_observation_hashes=(
                        replayed_preexisting_observations),
                    row_artifact_evidence={
                        "merged_rows.npz": (
                            row_paths[0], set(development_row_scenes),
                            set(development_hashes)),
                        "prior_rows.npz": (
                            legacy_rows_path, set(legacy_row_scenes),
                            set(legacy_development_hashes)),
                        "expansion_rows.npz": (
                            expansion_rows_path, set(expansion_row_scenes),
                            set(expansion_development_hashes)),
                    },
                    input_guard=guard,
                    register_phase_input=register_phase_input,
                )
            )
            selection_salt = b""
            guard("post-historical final access")
            statistics.update({
                "excluded_scene_fingerprints": historical_exclusion_receipt[
                    "combined_excluded_scene_fingerprint_count"],
                "excluded_scene_seeds": historical_exclusion_receipt[
                    "combined_excluded_seed_count"],
                "development_public_observation_count": len(all_development_hashes),
                "replayed_excluded_public_observation_count": (
                    historical_exclusion_receipt[
                        "combined_replayed_excluded_public_observation_count"]),
                "development_row_scene_count": len(all_development_fingerprints),
                "auxiliary_development_scene_count": len(auxiliary_development_scenes),
                "auxiliary_development_public_observation_count": len(
                    auxiliary_development_observations),
                "historical_final_scene_count": historical_exclusion_receipt[
                    "historical_final_scene_count"],
                "historical_final_public_observation_count": (
                    historical_exclusion_receipt[
                        "historical_final_public_observation_count"]),
                "historical_final_development_scene_overlap": 0,
                "historical_final_development_observation_overlap": 0,
                "retired_selection_trace_exposure": exposure_stats,
                "retired_touched_scene_count": len(exposed_fingerprints),
                "retired_touched_observation_count": len(exposed_observations),
                "retired_v1_v2_actor_executed": False,
                "retired_v1_v2_observations_derived": False,
                "old_exposed_outer_scene_count": len(old_outer_scenes),
                "old_exposed_outer_public_observation_count": len(
                    old_outer_observations),
                "old_exposed_outer_public_observations_sha256": digest(
                    sorted(old_outer_observations)),
                "fresh_outer_scene_count": len(fresh_outer_scenes),
                "old_and_fresh_outer_identity_overlap": 0,
            })
            guard("final evidence construction")
            sources = dict(initial_sources)
            bindings = {
                "actor_sha256": actor_sha256,
                "protocol_file_sha256": protocol_file_sha256,
                "protocol_content_sha256": digest(protocol),
                "manifest_file_sha256": file_hash(manifest_path),
                "manifest_content_sha256": manifest_content,
                "manifest_semantic_sha256": (
                    manifest_binding.EXPECTED_MANIFEST_SEMANTIC_SHA256),
                "designation_file_sha256": file_hash(designation_path),
                "designation_semantic_sha256": digest(designation),
                "selected_scenes_file_sha256": selected_file_sha256,
                "selected_scenes_semantic_sha256": digest(selected),
                "development_registries": development_bindings,
                "fresh_outer_registry_report_file_sha256": (
                    EXPECTED_EXPANSION_REPORT_SHA256),
                "fresh_outer_registry_report_content_sha256": (
                    development_bindings[DEVELOPMENT_EXPANSION_VERSION][
                        "report_content_sha256"]),
                "development_rows": row_bindings,
                "legacy_v3_rows": legacy_rows_binding,
                "expansion_rows": expansion_rows_binding,
                "retired_identity_projection": retired_projection_binding,
                "v3_exclusion_file_sha256": _json_file_sha256(v3_tombstone),
                "v3_exclusion_content_sha256": v3_tombstone["content_sha256"],
                "claim": claim_binding,
                "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
                "excluded_fingerprints_sha256": historical_exclusion_receipt[
                    "combined_excluded_scene_fingerprints_sha256"],
                "excluded_seeds_sha256": historical_exclusion_receipt[
                    "combined_excluded_seeds_sha256"],
                "development_observations_sha256": digest(
                    sorted(all_development_hashes)),
                "auxiliary_development_observations_sha256": digest(
                    sorted(auxiliary_development_observations)),
                "replayed_excluded_observations_sha256": historical_exclusion_receipt[
                    "combined_replayed_excluded_public_observations_sha256"],
                "forbidden_observations_sha256": historical_exclusion_receipt[
                    "combined_forbidden_public_observations_sha256"],
                "historical_exclusion_started_sha256": (
                    historical_exclusion_receipt[
                        "historical_exclusion_started_sha256"]),
                "historical_exclusion_completed_sha256": (
                    _json_file_sha256(historical_exclusion_receipt)),
                "historical_exclusion_receipt_sha256": (
                    historical_exclusion_receipt["receipt_sha256"]),
                "fresh_final_observations_sha256": digest(sorted(accepted_hashes)),
                "contract_sha256": digest(contract()),
                "producer_sources_sha256": digest(sources),
            }
            holdout = {
                "version": VERSION,
                "status": "passed_program_blind_registry",
                "contract": contract(),
                "bindings": bindings,
                "scenes": scenes,
                "selection_trace": trace,
                "statistics": statistics,
                "program_access": False,
                "program_predictions_access": False,
                "final_labels_used_for_selection": False,
                "runtime_action_override": False,
                "formal_ready": False,
            }
            holdout["content_sha256"] = digest(holdout)
            holdout_file_sha256 = _json_file_sha256(holdout)
            report = {
                "version": VERSION,
                "status": "passed_program_blind_registry",
                "bindings": bindings,
                "holdout_file_sha256": holdout_file_sha256,
                "holdout_content_sha256": holdout["content_sha256"],
                "v3_exclusion_file_sha256": _json_file_sha256(v3_tombstone),
                "statistics": statistics,
                "producer_sources": sources,
                "program_access": False,
                "program_predictions_access": False,
                "final_labels_used_for_selection": False,
                "formal_ready": False,
            }
            report["content_sha256"] = digest(report)
            expected_artifact_hashes = {
                "v3_exclusion.json": _json_file_sha256(v3_tombstone),
                "holdout.json": holdout_file_sha256,
                "report.json": _json_file_sha256(report),
            }
            guard("pre-publication")
            try:
                published_hashes = _publish_staged_holdout(
                    output_path,
                    artifacts={
                        "v3_exclusion.json": v3_tombstone,
                        "holdout.json": holdout,
                        "report.json": report,
                    },
                    expected_hashes=expected_artifact_hashes,
                    before_publish=lambda: guard("staged publication"),
                )
                if published_hashes != expected_artifact_hashes:
                    raise RuntimeError("Fresh-final published bytes differ")
                guard("post-publication")
                holdout_completed_value = {
                    "version": VERSION, "status": "completed_program_blind",
                    **claim_binding,
                    "historical_exclusion_completed_sha256": (
                        _json_file_sha256(historical_exclusion_receipt)),
                    "holdout_file_sha256": expected_artifact_hashes[
                        "holdout.json"],
                    "report_file_sha256": expected_artifact_hashes["report.json"],
                    "v3_exclusion_file_sha256": expected_artifact_hashes[
                        "v3_exclusion.json"],
                }
                holdout_completed_sha256 = _json_file_sha256(
                    holdout_completed_value)
                _claim_marker(
                    claim_dir, "holdout_completed.json", holdout_completed_value,
                    before_publish=lambda: guard("holdout completion"),
                )
            except BaseException:
                # Public output is valid only together with the completion marker.  A
                # late drift burns the claimed attempt but must not leave a result that
                # could be mistaken for a completed final holdout.
                if not (claim_dir / "holdout_completed.json").exists():
                    shutil.rmtree(output_path, ignore_errors=True)
                raise
            result = deepcopy(report)
            result["_publication_attestation"] = {
                "artifacts": dict(sorted(expected_artifact_hashes.items())),
                "holdout_completed_sha256": holdout_completed_sha256,
                "historical_exclusion_started_sha256": (
                    historical_exclusion_receipt[
                        "historical_exclusion_started_sha256"]),
                "historical_exclusion_completed_sha256": (
                    _json_file_sha256(historical_exclusion_receipt)),
            }
            return result

    phase_state = {"active": False}
    outcome = None
    failure_kind: str | None = None
    try:
        outcome = sensitive(
            actor_path=actor_path, protocol_path=protocol_path,
            manifest_path=manifest_path, designation_path=designation_path,
            selected_scenes_path=selected_scenes_path,
            development_registry_paths=development_registry_paths,
            development_rows_paths=development_rows_paths,
            legacy_v3_rows_path=legacy_v3_rows_path,
            expansion_rows_path=expansion_rows_path,
            retired_identity_projection_path=retired_identity_projection_path,
            output=output, claim_receipt_path=claim_receipt_path,
            expected_claim_sha256=expected_claim_sha256,
            expected_campaign_key=expected_campaign_key,
            expected_candidate_identity_sha256=(
                expected_candidate_identity_sha256),
            selection_salt_path=selection_salt_path,
            _sensitive_phase_state=phase_state,
        )
    except BaseException as error:
        if not isinstance(error, Exception):
            failure_kind = "private_interrupt"
        else:
            failure_kind = "private_failure"
    finally:
        phase_state.clear()
        sensitive = None
    if failure_kind == "private_interrupt":
        raise _HistoricalFinalPrivatePhaseInterrupt(
            "historical_final_private_phase_interrupted") from None
    if failure_kind == "private_failure":
        raise _HistoricalFinalPrivatePhaseError(
            "historical_final_private_phase_failed") from None
    if outcome is None:
        raise AssertionError("Fresh-final holdout returned no result")
    return outcome



def read_saved_holdout(
    output: str | Path, *, expected_holdout_sha256: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    """Authenticate the frozen registry without re-running final workloads."""
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Fresh-final evidence directory is unsafe")
    holdout_path, report_path = output / "holdout.json", output / "report.json"
    holdout, holdout_file_sha256 = _read_exact_json(
        holdout_path, "saved fresh-final holdout",
        expected_sha256=expected_holdout_sha256)
    report, report_file_sha256 = _read_exact_json(
        report_path, "saved fresh-final report",
        expected_sha256=expected_report_sha256)
    v3_expected_sha256 = report.get("v3_exclusion_file_sha256")
    if _HEX.fullmatch(str(v3_expected_sha256)) is None:
        raise ValueError("Saved fresh-final v3 exclusion hash differs")
    v3_path = _regular(output / "v3_exclusion.json", "saved v3 exclusion")
    v3_tombstone, v3_file_sha256 = _read_exact_json(
        v3_path, "saved v3 exclusion",
        expected_sha256=str(v3_expected_sha256))
    if (not _content_valid(holdout) or not _content_valid(report)
            or not _content_valid(v3_tombstone)
            or v3_tombstone.get("version") != V3_TOMBSTONE_VERSION
            or v3_tombstone.get("status") != "burned_program_blind_exclusion"
            or v3_tombstone.get("program_access") is not False
            or v3_tombstone.get("program_predictions_access") is not False
            or v3_tombstone.get("final_labels_used_for_program_selection") is not False
            or holdout.get("version") != VERSION or report.get("version") != VERSION
            or holdout.get("status") != "passed_program_blind_registry"
            or report.get("status") != "passed_program_blind_registry"
            or holdout.get("contract") != contract()
            or holdout.get("program_access") is not False
            or holdout.get("program_predictions_access") is not False
            or holdout.get("final_labels_used_for_selection") is not False
            or holdout.get("statistics", {}).get("accepted") != TOTAL_SCENES
            or holdout.get("statistics", {}).get("public_observation_overlap") != 0
            or report.get("holdout_file_sha256") != expected_holdout_sha256
            or report.get("holdout_content_sha256") != holdout["content_sha256"]
            or report.get("v3_exclusion_file_sha256") != v3_file_sha256
            or holdout.get("bindings", {}).get("v3_exclusion_file_sha256")
                != v3_file_sha256
            or holdout.get("bindings", {}).get("v3_exclusion_content_sha256")
                != v3_tombstone.get("content_sha256")
            or v3_tombstone.get("claim_binding")
                != holdout.get("bindings", {}).get("claim")
            or report.get("bindings") != holdout.get("bindings")
            or report.get("statistics") != holdout.get("statistics")
            or report.get("producer_sources") != producer_sources()
            or holdout.get("bindings", {}).get("contract_sha256")
                != digest(contract())
            or holdout.get("bindings", {}).get("producer_sources_sha256")
                != digest(producer_sources())
            or holdout.get("statistics", {}).get("families")
                != dict(sorted(FAMILY_QUOTAS.items()))
            or holdout.get("statistics", {}).get("unique_fingerprints")
                != TOTAL_SCENES
            or holdout.get("statistics", {}).get("unique_seeds")
                != TOTAL_SCENES):
        raise ValueError("Saved fresh-final evidence differs")
    _scene_rows(holdout.get("scenes"), label="saved fresh final",
                expected_count=TOTAL_SCENES)
    # Reject persistent drift after parsing while preserving the same-fd
    # identity used for every semantic decision above.
    if (_read_regular_bytes(
            holdout_path, "saved fresh-final holdout")[1]
                != holdout_file_sha256
            or _read_regular_bytes(
                report_path, "saved fresh-final report")[1]
                != report_file_sha256
            or _read_regular_bytes(
                v3_path, "saved v3 exclusion")[1]
                != v3_file_sha256):
        raise RuntimeError("Saved fresh-final evidence changed during read")
    return deepcopy(holdout)


__all__ = [
    "VERSION", "TOTAL_SCENES", "FAMILY_QUOTAS", "contract",
    "producer_sources", "read_saved_holdout",
]
