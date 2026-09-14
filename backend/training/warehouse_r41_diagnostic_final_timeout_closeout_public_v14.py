"""Public-only reader for the v14 closeout of the timed-out v13 final.

The reader authenticates four JSON files and their permanent mirrors.  It has
no salt path, runtime, model, NumPy, or row-loader dependency.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

from backend.training.warehouse_native_common import digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-timeout-closeout.v14"
STATUS = "burned_v13_timeout_irrevocably_closed_and_excluded"
CLAIM_NAME = "closeout_claim.json"
RECEIPT_NAME = "closeout_receipt.json"
IDENTITY_NAME = "burned_candidate_universe.json"
PROJECTION_NAME = "burned_observation_hashes.json"
EXPECTED_V13_ATTEMPT_KEY = (
    "6bdc48bccbfafb209b8e48e770c43e33dbdb513deab46b4bbc9c9da417998edb"
)
EXPECTED_V13_ANCHOR_SHA256 = (
    "a5094bdbbe1d35ce6c9b4bb751e6c38d2138634414d251cc61f079fd1f8be1ad"
)
EXPECTED_V13_ANCHOR_CONTENT_SHA256 = (
    "0f16eed3fee6d3ab3053dead744b9bd884e8908b227b837fe18b8920a93e2b18"
)
EXPECTED_V13_COMPLETION_SHA256 = (
    "f7cebd15bf22f25b9e3f352cb74522c6d44b6b36b322631af372de8a71899040"
)
EXPECTED_V13_COMPLETION_CONTENT_SHA256 = (
    "7d690828b8dec50a85c21c87bd607c8dd49627b06737e7f94b85f1d731704d3f"
)
EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256 = (
    "63eeeeeae0c8dfaa7d122e929fd1847f3ff62a065196307597b35fefc69c69cf"
)
EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256 = (
    "4bd200f524c18a41b8edb5e37e83dc69c8427b188681659c5e9efcc18eeea650"
)
EXPECTED_V13_CONFIG_SHA256 = (
    "8a2058e813ae0badb5d4e0723960a4054ef15b1c2dc045ace4be7ac26b9f111e"
)
EXPECTED_V13_CONFIG_CONTENT_SHA256 = (
    "c330f5c9aed197055604db19ccbcccedd1f93041577a942d2a1076713456e931"
)
EXPECTED_PROJECTOR_VERSION = (
    "warehouse-r41-diagnostic-final-observation-projection.v13"
)
EXPECTED_PROJECTOR_CONTRACT_SHA256 = (
    "1e1fd49de5a3e315afd097c17151cae5e6dd0de4870b1c3376eb6b50c1af3072"
)
EXPECTED_PROJECTOR_SOURCES_SHA256 = (
    "c29caf35a1a76ea4c1db779a7ef3ce02f6924430d0e78c0edd8ceb9a02b0f070"
)
EXPECTED_SELECTION_VERSION = "warehouse-r41-diagnostic-final-materializer.v13"
EXPECTED_FINAL_SCENE_OFFSET = 900_000
RANKING_INPUT_COUNT = 2160
ACCEPTED_SCENE_COUNT = 64
MAX_JSON_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY_FIELDS = {
    "batch_index", "family_offset", "family_id", "seed", "fingerprint",
}


def contract() -> dict[str, Any]:
    # Frozen copy: importing the salt-capable producer is deliberately avoided.
    return {
        "version": VERSION,
        "v13_failure_campaign_required_entries": [
            "attempt_anchor.json", "attempt_completed.json",
        ],
        "v13_public_material_or_rows_published": False,
        "v13_materializer_may_have_read_salt_or_identity": True,
        "permanent_o_excl_closeout_claim_before_config_identity_or_salt": True,
        "burned_v13_salt_read_only_after_closeout_claim": True,
        "burned_v13_salt_published": False,
        "frozen_v13_selection_executed_to_completion": True,
        "completed_evaluated_prefix_conservatively_retired": True,
        "all_v13_ranking_input_identities_retired": True,
        "hash_projection_is_exact_v13_collector_workload": True,
        "raw_observations_published": False,
        "actions_or_probabilities_read_from_projector": False,
        "actions_or_probabilities_published": False,
        "program_file_read": False,
        "labels_or_rows_read_or_generated": False,
        "v13_outer_promoted_to_development": False,
        "candidate_program_or_outer_refit": False,
        "same_v13_final_attempt_retry_permitted": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({name: child for name, child in value.items()
                                   if name != "content_sha256"}))


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, bytes, dict[str, Any]]:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat(follow_symlinks=False).st_size <= MAX_JSON_BYTES
            or file_hash(path) != _sha(expected_sha256, label + " SHA-256")):
        raise ValueError("Exact " + label + " bytes required")
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if (not isinstance(parsed, dict) or file_hash(path) != expected_sha256
            or not _content_valid(parsed)):
        raise ValueError(label + " content differs")
    return path, raw, parsed


def _identity(value: Any, label: str) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS
            or type(value.get("batch_index")) is not int
            or value["batch_index"] < 0
            or type(value.get("family_offset")) is not int
            or value["family_offset"] < 0
            or type(value.get("family_id")) is not str
            or not value["family_id"]
            or type(value.get("seed")) is not int
            or isinstance(value["seed"], bool) or value["seed"] < 0
            or type(value.get("fingerprint")) is not str
            or _HEX.fullmatch(value["fingerprint"]) is None):
        raise ValueError(label + " identity differs")
    return deepcopy(dict(value))


def _identity_list(value: Any, label: str, *, expected_count: int | None = None
                   ) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(label + " identities must be a list")
    rows = [_identity(row, label) for row in value]
    if ((expected_count is not None and len(rows) != expected_count)
            or len({row["seed"] for row in rows}) != len(rows)
            or len({row["fingerprint"] for row in rows}) != len(rows)):
        raise ValueError(label + " identity population differs")
    return rows


def _validate_claim(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "version", "status", "closeout_claim_key", "inputs",
        "v13_failure_campaign_authenticated_before_claim",
        "v13_failure_campaign_entry_count",
        "v13_salt_or_identity_access_may_have_occurred",
        "config_identity_or_salt_accessed_by_closeout_before_claim",
        "retry_allowed", "formal_ready", "content_sha256",
    }
    inputs = value.get("inputs")
    input_fields = {
        "scheme", "v13_attempt_key", "v13_anchor_sha256",
        "v13_anchor_content_sha256", "v13_completion_sha256",
        "v13_completion_content_sha256",
        "v13_controller_source_closure_sha256",
        "v13_materializer_source_closure_sha256",
        "timeout_closeout_source_closure_sha256", "contract_sha256",
        "candidate_lock_sha256", "actor_sha256", "protocol_sha256",
        "runtime_manifest_sha256", "program_sha256", "outer_result_sha256",
    }
    if (set(value) != fields
            or value.get("version") != VERSION + ".claim.v1"
            or value.get("status")
                != "v13_timeout_closeout_irrevocably_claimed"
            or not isinstance(inputs, Mapping)
            or set(inputs) != input_fields
            or inputs.get("v13_attempt_key") != EXPECTED_V13_ATTEMPT_KEY
            or inputs.get("v13_anchor_sha256") != EXPECTED_V13_ANCHOR_SHA256
            or inputs.get("v13_anchor_content_sha256")
                != EXPECTED_V13_ANCHOR_CONTENT_SHA256
            or inputs.get("v13_completion_sha256")
                != EXPECTED_V13_COMPLETION_SHA256
            or inputs.get("v13_completion_content_sha256")
                != EXPECTED_V13_COMPLETION_CONTENT_SHA256
            or inputs.get("v13_controller_source_closure_sha256")
                != EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256
            or inputs.get("v13_materializer_source_closure_sha256")
                != EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or inputs.get("contract_sha256") != digest(contract())
            or value.get("closeout_claim_key") != digest(dict(inputs))
            or value.get("v13_failure_campaign_authenticated_before_claim") is not True
            or value.get("v13_failure_campaign_entry_count") != 2
            or value.get("v13_salt_or_identity_access_may_have_occurred") is not True
            or value.get(
                "config_identity_or_salt_accessed_by_closeout_before_claim") is not False
            or value.get("retry_allowed") is not False
            or value.get("formal_ready") is not False
            or not _content_valid(value)):
        raise ValueError("Permanent v14 timeout-closeout claim differs")
    for name, child in inputs.items():
        if name == "scheme":
            if child != VERSION + ".v13-timeout-failure.v1":
                raise ValueError("Timeout-closeout claim scheme differs")
        else:
            _sha(child, "timeout-closeout claim " + name)
    return deepcopy(dict(value))


def _validate_universe(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "version", "status", "ranking_input_identities",
        "ranking_input_count", "ranking_input_identities_sha256",
        "historically_excluded_seeds", "historically_excluded_seed_count",
        "historically_excluded_seeds_sha256",
        "historically_excluded_fingerprints",
        "historically_excluded_fingerprint_count",
        "historically_excluded_fingerprints_sha256",
        "projector_eligible_identities", "projector_eligible_identity_count",
        "projector_eligible_identities_sha256", "v13_outer_identities",
        "v13_outer_identity_count", "v13_outer_identities_sha256",
        "completed_evaluated_prefix", "completed_evaluated_prefix_count",
        "completed_evaluated_prefix_sha256",
        "reconstructed_accepted_identities",
        "reconstructed_accepted_identity_count",
        "reconstructed_accepted_identities_sha256",
        "all_ranking_input_identities_retired",
        "all_completed_prefix_identities_retired",
        "same_v13_candidate_universe_reuse_permitted",
        "scene_snapshots_included", "rng_state_or_salt_included",
        "formal_ready", "content_sha256",
    }
    if (set(value) != fields
            or value.get("version") != VERSION + ".burned-candidate-universe.v1"
            or value.get("status")
                != "all_v13_ranking_inputs_and_completed_prefix_retired"
            or not _content_valid(value)
            or value.get("all_ranking_input_identities_retired") is not True
            or value.get("all_completed_prefix_identities_retired") is not True
            or value.get("same_v13_candidate_universe_reuse_permitted") is not False
            or value.get("scene_snapshots_included") is not False
            or value.get("rng_state_or_salt_included") is not False
            or value.get("formal_ready") is not False):
        raise ValueError("Burned v13 candidate universe semantics differ")
    ranking = _identity_list(
        value.get("ranking_input_identities"), "v13 ranking input",
        expected_count=RANKING_INPUT_COUNT)
    eligible = _identity_list(
        value.get("projector_eligible_identities"), "v13 projector eligible")
    outer = _identity_list(
        value.get("v13_outer_identities"), "v13 outer",
        expected_count=ACCEPTED_SCENE_COUNT)
    accepted = _identity_list(
        value.get("reconstructed_accepted_identities"),
        "reconstructed v13 accepted", expected_count=ACCEPTED_SCENE_COUNT)
    excluded_seeds = value.get("historically_excluded_seeds")
    excluded_fingerprints = value.get("historically_excluded_fingerprints")
    evaluated = value.get("completed_evaluated_prefix")
    if (not isinstance(excluded_seeds, list)
            or excluded_seeds != sorted(set(excluded_seeds))
            or any(type(child) is not int or isinstance(child, bool) or child < 0
                   for child in excluded_seeds)
            or not isinstance(excluded_fingerprints, list)
            or excluded_fingerprints != sorted(set(excluded_fingerprints))
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for child in excluded_fingerprints)
            or not isinstance(evaluated, list)
            or not evaluated):
        raise ValueError("Burned v13 exclusion lists differ")
    ranking_keys = {(row["seed"], row["fingerprint"]) for row in ranking}
    eligible_keys = {(row["seed"], row["fingerprint"]) for row in eligible}
    outer_keys = {(row["seed"], row["fingerprint"]) for row in outer}
    accepted_keys = {(row["seed"], row["fingerprint"]) for row in accepted}
    if (not eligible_keys <= ranking_keys or not outer_keys <= ranking_keys
            or not accepted_keys <= eligible_keys
            or any(row["seed"] not in set(excluded_seeds)
                   or row["fingerprint"] not in set(excluded_fingerprints)
                   for row in outer)):
        raise ValueError("Burned v13 identity subset relation differs")
    evaluated_identities: list[dict[str, Any]] = []
    for index, row in enumerate(evaluated):
        extra = {
            "evaluation_index", "projector_scene_index", "accepted",
            "accepted_scene_index", "row_count",
            "ordered_observation_hashes_sha256", "unique_observation_count",
            "unique_observation_hashes_sha256", "environment_steps",
        }
        if (not isinstance(row, Mapping)
                or set(row) != _IDENTITY_FIELDS | extra
                or row.get("evaluation_index") != index
                or type(row.get("projector_scene_index")) is not int
                or row["projector_scene_index"] < 0
                or type(row.get("accepted")) is not bool
                or (row["accepted"] and type(row.get("accepted_scene_index")) is not int)
                or (not row["accepted"] and row.get("accepted_scene_index") is not None)
                or type(row.get("row_count")) is not int or row["row_count"] <= 0
                or type(row.get("unique_observation_count")) is not int
                or row["unique_observation_count"] <= 0
                or type(row.get("environment_steps")) is not int
                or row["environment_steps"] <= 0):
            raise ValueError("Completed v13 evaluated-prefix record differs")
        _sha(row["ordered_observation_hashes_sha256"],
             "evaluated ordered observations")
        _sha(row["unique_observation_hashes_sha256"],
             "evaluated unique observations")
        evaluated_identities.append(_identity(
            {name: row[name] for name in _IDENTITY_FIELDS}, "evaluated v13"))
    evaluated_keys = {
        (row["seed"], row["fingerprint"]) for row in evaluated_identities}
    accepted_evaluated = sorted(
        (row for row in evaluated if row["accepted"]),
        key=lambda row: row["accepted_scene_index"])
    accepted_from_evaluated = [
        {name: row[name] for name in _IDENTITY_FIELDS}
        for row in accepted_evaluated]
    if (len(evaluated_keys) != len(evaluated)
            or not evaluated_keys <= eligible_keys
            or not accepted_keys <= evaluated_keys
            or sum(bool(row["accepted"]) for row in evaluated)
                != ACCEPTED_SCENE_COUNT
            or [row["accepted_scene_index"] for row in accepted_evaluated]
                != list(range(ACCEPTED_SCENE_COUNT))
            or accepted_from_evaluated != accepted):
        raise ValueError("Completed v13 evaluated-prefix population differs")
    checks = {
        "ranking_input_count": len(ranking),
        "ranking_input_identities_sha256": digest(ranking),
        "historically_excluded_seeds_sha256": digest(excluded_seeds),
        "historically_excluded_seed_count": len(excluded_seeds),
        "historically_excluded_fingerprints_sha256": digest(
            excluded_fingerprints),
        "historically_excluded_fingerprint_count": len(
            excluded_fingerprints),
        "projector_eligible_identity_count": len(eligible),
        "projector_eligible_identities_sha256": digest(eligible),
        "v13_outer_identity_count": len(outer),
        "v13_outer_identities_sha256": digest(outer),
        "completed_evaluated_prefix_count": len(evaluated),
        "completed_evaluated_prefix_sha256": digest(evaluated),
        "reconstructed_accepted_identity_count": len(accepted),
        "reconstructed_accepted_identities_sha256": digest(accepted),
    }
    if any(value.get(name) != child for name, child in checks.items()):
        raise ValueError("Burned v13 universe digest or count differs")
    return deepcopy(dict(value))


def _validate_projection(value: Mapping[str, Any], universe: Mapping[str, Any]
                         ) -> dict[str, Any]:
    ordered = value.get("ordered_observation_hashes")
    unique = value.get("unique_observation_hashes")
    timing = value.get("timing")
    boundary = value.get("information_boundary")
    accepted_projection = value.get("frozen_accepted_projection")
    accepted_fields = {
        "version", "contract_sha256", "producer_sources_sha256",
        "scene_offset", "scene_count", "partners",
        "critical_anchor_period", "dense_critical", "row_count",
        "ordered_observation_hashes_sha256", "unique_observation_count",
        "unique_observation_hashes_sha256", "scene_summaries",
        "scene_summaries_sha256", "environment_steps",
        "prior_observation_overlap", "within_final_observation_overlap",
        "actions_read", "probabilities_read", "program_access",
        "labels_read", "content_sha256",
    }
    accepted_summaries = (accepted_projection.get("scene_summaries")
                          if isinstance(accepted_projection, Mapping) else None)
    accepted_summary_fields = {
        "local_scene_index", "scene_index", "fingerprint", "row_count",
        "ordered_observation_hashes_sha256", "unique_observation_count",
        "unique_observation_hashes_sha256", "accepted_scene_index",
    }
    accepted_projection_valid = (
        isinstance(accepted_projection, Mapping)
        and set(accepted_projection) == accepted_fields
        and accepted_projection.get("version") == EXPECTED_PROJECTOR_VERSION
        and accepted_projection.get("contract_sha256")
            == EXPECTED_PROJECTOR_CONTRACT_SHA256
        and accepted_projection.get("producer_sources_sha256")
            == EXPECTED_PROJECTOR_SOURCES_SHA256
        and accepted_projection.get("scene_offset")
            == EXPECTED_FINAL_SCENE_OFFSET
        and accepted_projection.get("scene_count") == ACCEPTED_SCENE_COUNT
        and accepted_projection.get("partners") == [
            "skilled", "assertive", "noisy"]
        and accepted_projection.get("critical_anchor_period") == 5
        and accepted_projection.get("dense_critical") is False
        and accepted_projection.get("row_count") > 0
        and accepted_projection.get("unique_observation_count") > 0
        and accepted_projection.get("environment_steps") > 0
        and accepted_projection.get("prior_observation_overlap") == 0
        and accepted_projection.get("within_final_observation_overlap") == 0
        and all(accepted_projection.get(name) is False for name in (
            "actions_read", "probabilities_read", "program_access",
            "labels_read"))
        and isinstance(accepted_summaries, list)
        and len(accepted_summaries) == ACCEPTED_SCENE_COUNT
        and accepted_projection.get("scene_summaries_sha256")
            == digest(accepted_summaries)
        and _content_valid(accepted_projection)
    )
    accepted_ordered: list[str] = []
    trace_offset = 0
    trace_rows_fit = isinstance(ordered, list)
    if trace_rows_fit:
        for record in universe["completed_evaluated_prefix"]:
            next_offset = trace_offset + record["row_count"]
            segment = ordered[trace_offset:next_offset]
            if len(segment) != record["row_count"]:
                trace_rows_fit = False
                break
            if (record["ordered_observation_hashes_sha256"] != digest(segment)
                    or record["unique_observation_count"] != len(set(segment))
                    or record["unique_observation_hashes_sha256"]
                        != digest(sorted(set(segment)))):
                trace_rows_fit = False
                break
            if record["accepted"]:
                accepted_ordered.extend(segment)
            trace_offset = next_offset
        trace_rows_fit = trace_rows_fit and trace_offset == len(ordered)
    accepted_unique = sorted(set(accepted_ordered))
    if accepted_projection_valid:
        for index, summary in enumerate(accepted_summaries):
            accepted_record = next(
                row for row in universe["completed_evaluated_prefix"]
                if row["accepted_scene_index"] == index)
            start = sum(row["row_count"] for row in accepted_summaries[:index])
            summary_segment = accepted_ordered[
                start:start + accepted_record["row_count"]]
            if (not isinstance(summary, Mapping)
                    or set(summary) != accepted_summary_fields
                    or summary.get("local_scene_index") != 0
                    or summary.get("scene_index")
                        != EXPECTED_FINAL_SCENE_OFFSET + index
                    or summary.get("accepted_scene_index") != index
                    or summary.get("fingerprint")
                        != accepted_record["fingerprint"]
                    or summary.get("row_count")
                        != accepted_record["row_count"]
                    or summary.get("ordered_observation_hashes_sha256")
                        != digest(summary_segment)
                    or summary.get("unique_observation_count")
                        != len(set(summary_segment))
                    or summary.get("unique_observation_hashes_sha256")
                        != digest(sorted(set(summary_segment)))
                    or type(summary.get("fingerprint")) is not str
                    or _HEX.fullmatch(summary["fingerprint"]) is None
                    or type(summary.get("row_count")) is not int
                    or summary["row_count"] <= 0
                    or type(summary.get("unique_observation_count")) is not int
                    or summary["unique_observation_count"] <= 0
                    or _HEX.fullmatch(str(summary.get(
                        "ordered_observation_hashes_sha256"))) is None
                    or _HEX.fullmatch(str(summary.get(
                        "unique_observation_hashes_sha256"))) is None):
                accepted_projection_valid = False
                break
    fields = {
        "version", "status", "selection_algorithm_version",
        "selection_algorithm_source_closure_sha256", "projector_version",
        "projector_contract", "projector_contract_sha256",
        "projector_sources", "projector_sources_sha256",
        "completed_evaluated_prefix_count", "accepted_scene_count",
        "ordered_observation_hashes", "ordered_observation_hashes_sha256",
        "unique_observation_hashes", "unique_observation_hashes_sha256",
        "row_count", "unique_observation_count", "environment_steps",
        "frozen_accepted_projection", "timing", "information_boundary",
        "formal_ready", "content_sha256",
    }
    if (set(value) != fields
            or value.get("version")
            != VERSION + ".burned-observation-projection.v1"
            or value.get("status")
                != "completed_v13_evaluated_prefix_hashes_permanently_excluded"
            or value.get("selection_algorithm_version")
                != EXPECTED_SELECTION_VERSION
            or value.get("selection_algorithm_source_closure_sha256")
                != EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or value.get("projector_version") != EXPECTED_PROJECTOR_VERSION
            or value.get("projector_contract_sha256")
                != EXPECTED_PROJECTOR_CONTRACT_SHA256
            or value.get("projector_sources_sha256")
                != EXPECTED_PROJECTOR_SOURCES_SHA256
            or not _content_valid(value)
            or not isinstance(ordered, list) or not ordered
            or any(type(child) is not str or _HEX.fullmatch(child) is None
                   for child in ordered)
            or not isinstance(unique, list) or unique != sorted(set(ordered))
            or value.get("ordered_observation_hashes_sha256") != digest(ordered)
            or value.get("unique_observation_hashes_sha256") != digest(unique)
            or value.get("row_count") != len(ordered)
            or value.get("unique_observation_count") != len(unique)
            or value.get("completed_evaluated_prefix_count")
                != universe["completed_evaluated_prefix_count"]
            or value.get("accepted_scene_count") != ACCEPTED_SCENE_COUNT
            or type(value.get("environment_steps")) is not int
            or value["environment_steps"] <= 0
            or value["environment_steps"] != sum(
                row["environment_steps"]
                for row in universe["completed_evaluated_prefix"])
            or not trace_rows_fit
            or not isinstance(timing, Mapping)
            or set(timing) != {
                "elapsed_seconds", "clock", "process_model",
                "parallel_workers", "python_implementation",
                "python_version", "platform", "logical_cpu_count",
            }
            or type(timing.get("elapsed_seconds")) not in (int, float)
            or isinstance(timing["elapsed_seconds"], bool)
            or not math.isfinite(timing["elapsed_seconds"])
            or timing["elapsed_seconds"] <= 0
            or timing.get("clock") != "time.perf_counter"
            or timing.get("process_model")
                != "single_process_in_process_frozen_v13_selector"
            or timing.get("parallel_workers") != 1
            or type(timing.get("python_implementation")) is not str
            or not timing["python_implementation"]
            or type(timing.get("python_version")) is not str
            or not timing["python_version"]
            or type(timing.get("platform")) is not str
            or not timing["platform"]
            or (timing.get("logical_cpu_count") is not None
                and (type(timing["logical_cpu_count"]) is not int
                     or timing["logical_cpu_count"] <= 0))
            or not isinstance(boundary, Mapping)
            or set(boundary) != {
                "closeout_claim_preceded_config_identity_and_salt_access",
                "burned_salt_used_only_for_exact_ranking_reconstruction",
                "salt_or_rng_state_included", "raw_observations_included",
                "actor_actions_included", "actor_probabilities_included",
                "program_file_or_predictions_accessed",
                "labels_or_rows_accessed_or_generated",
                "runtime_action_override", "formal_ready",
            }
            or boundary.get(
                "closeout_claim_preceded_config_identity_and_salt_access") is not True
            or boundary.get(
                "burned_salt_used_only_for_exact_ranking_reconstruction") is not True
            or any(boundary.get(name) is not False for name in (
                "salt_or_rng_state_included", "raw_observations_included",
                "actor_actions_included", "actor_probabilities_included",
                "program_file_or_predictions_accessed",
                "labels_or_rows_accessed_or_generated", "runtime_action_override",
                "formal_ready"))
            or not accepted_projection_valid
            or accepted_projection.get("ordered_observation_hashes_sha256")
                != digest(accepted_ordered)
            or accepted_projection.get("row_count") != len(accepted_ordered)
            or accepted_projection.get("unique_observation_count")
                != len(accepted_unique)
            or accepted_projection.get("unique_observation_hashes_sha256")
                != digest(accepted_unique)
            or accepted_projection.get("environment_steps") != sum(
                row["environment_steps"]
                for row in universe["completed_evaluated_prefix"]
                if row["accepted"])
            or value.get("projector_sources_sha256")
                != digest(dict(value.get("projector_sources", {})))
            or value.get("projector_contract_sha256")
                != digest(value.get("projector_contract"))
            or value.get("formal_ready") is not False):
        raise ValueError("Burned v13 observation projection differs")
    _sha(value.get("selection_algorithm_source_closure_sha256"),
         "v13 selection source closure")
    _sha(value.get("projector_contract_sha256"), "v13 projector contract")
    return deepcopy(dict(value))


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            one, two = first.read(1024 * 1024), second.read(1024 * 1024)
            if one != two:
                return False
            if not one:
                return True


def read_saved_closeout_public(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate public JSON companions without any protected dependency."""
    receipt_path, receipt_raw, receipt = _strict_json(
        path, "v14 v13-timeout closeout",
        expected_sha256=expected_closeout_sha256)
    top_fields = {
        "version", "status", "closeout_key", "closeout_claim_key",
        "burned_final_attempt_key", "contract", "v13_timeout_failure",
        "burned_candidate_universe", "burned_observation_projection",
        "disposition", "information_boundary", "bindings",
        "producer_sources", "producer_sources_sha256", "formal_ready",
        "content_sha256",
    }
    sources = receipt.get("producer_sources")
    if (set(receipt) != top_fields or receipt.get("version") != VERSION
            or receipt.get("status") != STATUS
            or receipt.get("contract") != contract()
            or receipt.get("burned_final_attempt_key")
                != EXPECTED_V13_ATTEMPT_KEY
            or receipt.get("formal_ready") is not False
            or not isinstance(sources, Mapping) or not sources
            or any(type(name) is not str or not name
                   or type(child) is not str or _HEX.fullmatch(child) is None
                   for name, child in sources.items())
            or receipt.get("producer_sources_sha256")
                != digest(dict(sorted(sources.items())))
            or not _content_valid(receipt)):
        raise ValueError("Exact public v14 timeout closeout required")
    bindings = receipt.get("bindings")
    binding_fields = {
        "candidate_lock_sha256", "actor_sha256", "protocol_sha256",
        "runtime_manifest_sha256", "program_sha256", "outer_result_sha256",
        "v13_anchor_sha256", "v13_anchor_content_sha256",
        "v13_completion_sha256", "v13_completion_content_sha256",
        "v13_controller_source_closure_sha256",
        "v13_materializer_source_closure_sha256",
        "v13_materializer_config_sha256",
        "v13_materializer_config_content_sha256",
        "closeout_claim_file_sha256", "closeout_claim_content_sha256",
        "burned_candidate_universe_file_sha256",
        "burned_candidate_universe_content_sha256",
        "burned_observation_hashes_file_sha256",
        "burned_observation_hashes_content_sha256",
        "producer_sources_sha256",
    }
    if (not isinstance(bindings, Mapping) or set(bindings) != binding_fields
            or any(_HEX.fullmatch(str(child)) is None
                   for child in bindings.values())
            or bindings.get("v13_anchor_sha256")
                != EXPECTED_V13_ANCHOR_SHA256
            or bindings.get("v13_anchor_content_sha256")
                != EXPECTED_V13_ANCHOR_CONTENT_SHA256
            or bindings.get("v13_completion_sha256")
                != EXPECTED_V13_COMPLETION_SHA256
            or bindings.get("v13_completion_content_sha256")
                != EXPECTED_V13_COMPLETION_CONTENT_SHA256
            or bindings.get("v13_controller_source_closure_sha256")
                != EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256
            or bindings.get("v13_materializer_source_closure_sha256")
                != EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or bindings.get("v13_materializer_config_sha256")
                != EXPECTED_V13_CONFIG_SHA256
            or bindings.get("v13_materializer_config_content_sha256")
                != EXPECTED_V13_CONFIG_CONTENT_SHA256
            or bindings.get("producer_sources_sha256")
                != receipt.get("producer_sources_sha256")):
        raise ValueError("V14 timeout closeout bindings differ")
    directory = _directory(receipt_path.parent,
                           "v14 timeout-closeout artifact directory")
    names = {CLAIM_NAME, RECEIPT_NAME, IDENTITY_NAME, PROJECTION_NAME}
    if {entry.name for entry in directory.iterdir()} != names:
        raise ValueError("V14 timeout-closeout artifact set differs")
    _, claim_raw, claim = _strict_json(
        directory / CLAIM_NAME, "v14 timeout-closeout claim",
        expected_sha256=bindings["closeout_claim_file_sha256"])
    claim = _validate_claim(claim)
    inherited = {
        "candidate_lock_sha256": "candidate_lock_sha256",
        "actor_sha256": "actor_sha256",
        "protocol_sha256": "protocol_sha256",
        "runtime_manifest_sha256": "runtime_manifest_sha256",
        "program_sha256": "program_sha256",
        "outer_result_sha256": "outer_result_sha256",
    }
    if any(bindings[name] != claim["inputs"][input_name]
           for name, input_name in inherited.items()):
        raise ValueError("V14 timeout closeout inherited candidate differs")
    if (claim["inputs"]["timeout_closeout_source_closure_sha256"]
            != receipt["producer_sources_sha256"]):
        raise ValueError("V14 timeout closeout producer closure differs")
    _, universe_raw, universe = _strict_json(
        directory / IDENTITY_NAME, "burned v13 candidate universe",
        expected_sha256=bindings["burned_candidate_universe_file_sha256"])
    universe = _validate_universe(universe)
    _, projection_raw, projection = _strict_json(
        directory / PROJECTION_NAME, "burned v13 observation hashes",
        expected_sha256=bindings["burned_observation_hashes_file_sha256"])
    projection = _validate_projection(projection, universe)
    expected_failure = {
        "controller_status": "burned_failed",
        "controller_reason": "protected_final_phase_failed",
        "controller_materializer_timeout_seconds": 600,
        "timeout_attribution": (
            "operator-observed; v13 completion records the generic "
            "protected-final failure"),
        "permanent_campaign_entries": [
            "attempt_anchor.json", "attempt_completed.json"],
        "permanent_campaign_entry_count": 2,
        "public_output_published": False,
        "materializer_output_published": False,
        "final_material_rows_parity_or_audit_published": False,
        "salt_or_identity_may_have_been_accessed_before_timeout": True,
        "retry_allowed": False,
    }
    expected_universe_summary = {
        "ranking_input_count": universe["ranking_input_count"],
        "ranking_input_identities_sha256": universe[
            "ranking_input_identities_sha256"],
        "projector_eligible_identity_count": universe[
            "projector_eligible_identity_count"],
        "projector_eligible_identities_sha256": universe[
            "projector_eligible_identities_sha256"],
        "v13_outer_identity_count": universe["v13_outer_identity_count"],
        "v13_outer_identities_sha256": universe[
            "v13_outer_identities_sha256"],
        "completed_evaluated_prefix_count": universe[
            "completed_evaluated_prefix_count"],
        "completed_evaluated_prefix_sha256": universe[
            "completed_evaluated_prefix_sha256"],
        "accepted_scene_count": universe[
            "reconstructed_accepted_identity_count"],
        "accepted_identities_sha256": universe[
            "reconstructed_accepted_identities_sha256"],
        "all_ranking_inputs_retired": True,
    }
    expected_projection_summary = {
        "row_count": projection["row_count"],
        "unique_observation_count": projection["unique_observation_count"],
        "ordered_observation_hashes_sha256": projection[
            "ordered_observation_hashes_sha256"],
        "unique_observation_hashes_sha256": projection[
            "unique_observation_hashes_sha256"],
        "environment_steps": projection["environment_steps"],
        "elapsed_seconds": projection["timing"]["elapsed_seconds"],
        "accepted_scene_count": projection["accepted_scene_count"],
    }
    expected_disposition = {
        "v13_candidate_and_program_reusable_without_refit": True,
        "v13_passed_outer_reusable_without_rerun": True,
        "v14_final_must_use_disjoint_expanded_candidate_universe": True,
        "v14_final_must_exclude_all_published_identity_and_hash_values": True,
        "same_v13_final_attempt_retry_permitted": False,
        "formal_ready": False,
    }
    expected_boundary = {
        "permanent_closeout_claim_preceded_closeout_salt_access": True,
        "burned_v13_salt_reopened_after_closeout_claim": True,
        "salt_value_published": False,
        "actions_probabilities_labels_or_program_accessed": False,
        "labeled_rows_generated": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    if (claim["content_sha256"]
            != bindings["closeout_claim_content_sha256"]
            or universe["content_sha256"]
                != bindings["burned_candidate_universe_content_sha256"]
            or projection["content_sha256"]
                != bindings["burned_observation_hashes_content_sha256"]
            or receipt.get("closeout_key") != claim["closeout_claim_key"]
            or receipt.get("closeout_claim_key") != claim["closeout_claim_key"]
            or receipt.get("v13_timeout_failure") != expected_failure
            or receipt.get("burned_candidate_universe")
                != expected_universe_summary
            or receipt.get("burned_observation_projection")
                != expected_projection_summary
            or receipt.get("disposition") != expected_disposition
            or receipt.get("information_boundary") != expected_boundary):
        raise ValueError("V14 timeout-closeout companion binding differs")

    permanent = _directory(permanent_closeout_registry,
                           "permanent v14 timeout-closeout registry")
    campaign = _directory(permanent / claim["closeout_claim_key"],
                          "permanent v14 timeout-closeout campaign")
    if {entry.name for entry in campaign.iterdir()} != names:
        raise ValueError("Permanent v14 timeout-closeout artifact set differs")
    for name in names:
        if not _same_bytes(directory / name, campaign / name):
            raise ValueError("Permanent v14 timeout-closeout bytes differ: " + name)
    if (receipt_raw != (campaign / RECEIPT_NAME).read_bytes()
            or claim_raw != (campaign / CLAIM_NAME).read_bytes()
            or universe_raw != (campaign / IDENTITY_NAME).read_bytes()
            or projection_raw != (campaign / PROJECTION_NAME).read_bytes()):
        raise RuntimeError("V14 timeout-closeout mirror changed during read")
    return deepcopy(receipt)


__all__ = [
    "VERSION", "STATUS", "CLAIM_NAME", "RECEIPT_NAME", "IDENTITY_NAME",
    "PROJECTION_NAME", "contract", "read_saved_closeout_public",
]
