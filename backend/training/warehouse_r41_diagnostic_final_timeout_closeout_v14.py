"""Irrevocably close the timed-out v13 protected-final attempt.

The v13 controller permanently consumed its candidate/outer pair and then
published only its anchor and burned-failure completion.  The materializer
process may have opened the committed salt and may have evaluated a prefix of
the protected candidate ranking before its 600 second timeout.  The exact
interruption point is unknowable.

This append-only closeout first authenticates that two-file failure campaign
and creates a permanent ``O_EXCL`` closeout claim.  Only after that claim does
it open the already-burned v13 salt.  It runs the frozen v13 selector and
observation-only projector to normal completion, observes every hash-only
projector return, and retires the completed evaluated prefix.  That prefix is
a conservative superset of anything the interrupted deterministic run could
have exposed.  It never opens the program, collects labeled rows, or returns
actions, probabilities, logits, or raw observations.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_final_materializer_v13 as materializer
from backend.training import warehouse_r41_diagnostic_final_once_v13 as final_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


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
RANKING_INPUT_COUNT = 2160
ACCEPTED_SCENE_COUNT = 64
CONTROLLER_TIMEOUT_SECONDS = 600
MAX_JSON_BYTES = 512 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "v13_failure_campaign_required_entries": [
            final_api.ANCHOR_NAME, final_api.COMPLETION_NAME,
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


def _regular(value: str | Path, label: str, *, expected_sha256: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat(follow_symlinks=False).st_size <= MAX_JSON_BYTES
            or file_hash(path) != _sha(expected_sha256, label + " SHA-256")):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, dict[str, Any]]:
    path = _regular(value, label, expected_sha256=expected_sha256)
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
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if (not isinstance(parsed, dict) or file_hash(path) != expected_sha256
            or not _content_valid(parsed)):
        raise ValueError(label + " content differs")
    return path, parsed


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {name: value.get(name) for name in (
        "batch_index", "family_offset", "family_id", "seed", "fingerprint")}
    if (type(result["batch_index"]) is not int
            or result["batch_index"] < 0
            or type(result["family_offset"]) is not int
            or result["family_offset"] < 0
            or type(result["family_id"]) is not str
            or not result["family_id"]
            or type(result["seed"]) is not int
            or isinstance(result["seed"], bool) or result["seed"] < 0
            or type(result["fingerprint"]) is not str
            or _HEX.fullmatch(result["fingerprint"]) is None):
        raise ValueError("V13 candidate identity differs")
    return result


def _authenticate_timeout_failure(
    *, final_anchor_path: str | Path, final_completion_path: str | Path,
    permanent_final_registry: str | Path, failed_public_output: str | Path,
) -> dict[str, Any]:
    """Authenticate only the burned public failure; access no selection input."""
    anchor_path, anchor = _strict_json(
        final_anchor_path, "v13 final anchor",
        expected_sha256=EXPECTED_V13_ANCHOR_SHA256)
    completion_path, completion = _strict_json(
        final_completion_path, "v13 final completion",
        expected_sha256=EXPECTED_V13_COMPLETION_SHA256)
    attempt_key = anchor.get("attempt_key")
    if (attempt_key != EXPECTED_V13_ATTEMPT_KEY
            or anchor.get("version") != final_api.VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "final_attempt_irrevocably_claimed"
            or anchor.get("candidate_and_outer_authenticated_before_claim") is not True
            or anchor.get("final_identity_or_rows_accessed_before_claim") is not False
            or anchor.get("retry_allowed") is not False
            or anchor.get("formal_ready") is not False):
        raise ValueError("Exact burned v13 final anchor required")
    if anchor.get("content_sha256") != EXPECTED_V13_ANCHOR_CONTENT_SHA256:
        raise ValueError("Exact burned v13 final anchor content required")
    if (set(completion) != final_api._FAILURE_COMPLETION_FIELDS
            or completion.get("version") != final_api.VERSION
            or completion.get("status") != final_api.STATUS_FAILED
            or completion.get("attempt_key") != attempt_key
            or completion.get("attempt_anchor_content_sha256")
                != anchor["content_sha256"]
            or completion.get("reason") != "protected_final_phase_failed"
            or completion.get("final_consumed") is not True
            or completion.get("retry_allowed") is not False
            or completion.get("program_fits") != 0
            or completion.get("actor_updates") != 0
            or completion.get("runtime_action_override") is not False
            or completion.get("formal_ready") is not False
            or completion.get("content_sha256")
                != EXPECTED_V13_COMPLETION_CONTENT_SHA256):
        raise ValueError("Exact burned v13 failure completion required")

    controller_sources = final_api.producer_sources()
    materializer_source, materializer_sources = (
        final_api._official_materializer_binding())
    if (digest(controller_sources)
            != EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256
            or digest(materializer_sources)
                != EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or anchor.get("bindings", {}).get(
                "final_controller_source_closure_sha256")
                != EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256
            or anchor.get("bindings", {}).get(
                "final_materializer_source_closure_sha256")
                != EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or completion.get("producer_sources_sha256")
                != EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256):
        raise ValueError("V13 controller or materializer source closure differs")

    permanent = _directory(permanent_final_registry,
                           "permanent v13 final registry")
    campaign = _directory(permanent / attempt_key,
                          "burned v13 final campaign")
    expected_entries = {final_api.ANCHOR_NAME, final_api.COMPLETION_NAME}
    if ({entry.name for entry in campaign.iterdir()} != expected_entries
            or anchor_path != (campaign / final_api.ANCHOR_NAME).absolute()
            or completion_path != (campaign / final_api.COMPLETION_NAME).absolute()
            or (campaign / final_api.MATERIAL_NAME).exists()
            or (campaign / final_api.ROWS_NAME).exists()
            or (campaign / final_api.PARITY_NAME).exists()
            or (campaign / final_api.AUDIT_NAME).exists()
            or (campaign / "materializer_output.json").exists()):
        raise ValueError("Burned v13 campaign must contain only anchor and completion")
    absent = Path(failed_public_output).expanduser().absolute()
    if (absent.exists() or absent.is_symlink()
            or not absent.parent.is_dir() or absent.parent.is_symlink()
            or absent.parent.resolve() != absent.parent):
        raise ValueError("V13 public final output must be absent")
    # Reuse the frozen materializer's full anchor validator.  It reads only the
    # already-read anchor and never the config, candidate population, or salt.
    checked_path, checked_anchor = materializer._authenticate_claim(anchor_path)
    if checked_path != anchor_path or checked_anchor != anchor:
        raise RuntimeError("V13 materializer anchor authentication differs")
    return {
        "anchor_path": anchor_path, "completion_path": completion_path,
        "anchor": anchor, "completion": completion,
        "campaign": campaign, "permanent_registry": permanent,
        "materializer_source": materializer_source,
        "controller_sources": controller_sources,
        "materializer_sources": materializer_sources,
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_permanent_claim(
    *, evidence: Mapping[str, Any], permanent_closeout_registry: str | Path,
) -> tuple[Path, dict[str, Any], bytes]:
    sources = producer_sources()
    inputs = {
        "scheme": VERSION + ".v13-timeout-failure.v1",
        "v13_attempt_key": evidence["anchor"]["attempt_key"],
        "v13_anchor_sha256": EXPECTED_V13_ANCHOR_SHA256,
        "v13_anchor_content_sha256": evidence["anchor"]["content_sha256"],
        "v13_completion_sha256": EXPECTED_V13_COMPLETION_SHA256,
        "v13_completion_content_sha256": evidence["completion"][
            "content_sha256"],
        "v13_controller_source_closure_sha256": (
            EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256),
        "v13_materializer_source_closure_sha256": (
            EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256),
        "timeout_closeout_source_closure_sha256": digest(sources),
        "contract_sha256": digest(contract()),
        "candidate_lock_sha256": evidence["anchor"]["attempt_key_inputs"][
            "candidate_lock_sha256"],
        "actor_sha256": evidence["anchor"]["attempt_key_inputs"][
            "actor_sha256"],
        "protocol_sha256": evidence["anchor"]["attempt_key_inputs"][
            "protocol_sha256"],
        "runtime_manifest_sha256": evidence["anchor"]["attempt_key_inputs"][
            "runtime_manifest_sha256"],
        "program_sha256": evidence["anchor"]["attempt_key_inputs"][
            "program_sha256"],
        "outer_result_sha256": evidence["anchor"]["attempt_key_inputs"][
            "outer_result_sha256"],
    }
    claim_key = digest(inputs)
    claim: dict[str, Any] = {
        "version": VERSION + ".claim.v1",
        "status": "v13_timeout_closeout_irrevocably_claimed",
        "closeout_claim_key": claim_key,
        "inputs": inputs,
        "v13_failure_campaign_authenticated_before_claim": True,
        "v13_failure_campaign_entry_count": 2,
        "v13_salt_or_identity_access_may_have_occurred": True,
        "config_identity_or_salt_accessed_by_closeout_before_claim": False,
        "retry_allowed": False,
        "formal_ready": False,
    }
    claim["content_sha256"] = digest(claim)
    raw = _json_bytes(claim)
    registry = _directory(permanent_closeout_registry,
                          "permanent v14 timeout-closeout registry")
    campaign = registry / claim_key
    try:
        os.mkdir(campaign, 0o700)
    except FileExistsError:
        raise FileExistsError("This v13 timeout closeout is already claimed") from None
    try:
        _write_exclusive(campaign / CLAIM_NAME, raw)
        _fsync_directory(campaign)
        _fsync_directory(registry)
    except BaseException:
        # The directory itself is the irrevocable claim.  Never remove it.
        raise
    return campaign, claim, raw


def _read_config_after_claim(
    *, claim_path: Path, materializer_config_path: str | Path,
) -> tuple[Path, dict[str, Path]]:
    if not claim_path.is_file() or claim_path.name != CLAIM_NAME:
        raise RuntimeError("Permanent timeout-closeout claim must exist first")
    config_path, value = _strict_json(
        materializer_config_path, "frozen v13 materializer config",
        expected_sha256=EXPECTED_V13_CONFIG_SHA256)
    paths = value.get("paths")
    if (value.get("version") != materializer.CONFIG_VERSION
            or value.get("content_sha256")
                != EXPECTED_V13_CONFIG_CONTENT_SHA256
            or not isinstance(paths, Mapping)
            or set(paths) != materializer._CONFIG_PATH_FIELDS
            or any(type(child) is not str or not child for child in paths.values())):
        raise ValueError("Exact frozen v13 materializer config required")
    return config_path, {
        name: Path(child).expanduser().absolute() for name, child in paths.items()
    }


def _universe(prepared: Mapping[str, Any],
              authenticated: Mapping[str, Any]) -> dict[str, Any]:
    inputs = [_identity(row) for row in prepared["identities"]]
    stable = sorted(inputs, key=lambda row: (
        row["family_id"], row["batch_index"], row["seed"], row["fingerprint"]))
    if (len(stable) != RANKING_INPUT_COUNT
            or len({row["seed"] for row in stable}) != RANKING_INPUT_COUNT
            or len({row["fingerprint"] for row in stable})
                != RANKING_INPUT_COUNT):
        raise ValueError("Exact 2160-scene v13 ranking input universe required")
    excluded_seeds = sorted(map(int, prepared["excluded_seeds"]))
    excluded_fingerprints = sorted(map(str, prepared["excluded_fingerprints"]))
    eligible = [row for row in stable
                if row["seed"] not in set(excluded_seeds)
                and row["fingerprint"] not in set(excluded_fingerprints)]
    if (not eligible
            or any(_HEX.fullmatch(value) is None for value in excluded_fingerprints)
            or len(excluded_seeds) != len(set(excluded_seeds))
            or len(excluded_fingerprints) != len(set(excluded_fingerprints))):
        raise ValueError("V13 pre-selection exclusion universe differs")
    by_fingerprint = {row["fingerprint"]: row for row in stable}
    outer_raw = authenticated.get("registry", {}).get(
        "selected_outer_identities")
    if not isinstance(outer_raw, list):
        raise ValueError("V13 outer identities are missing")
    outer: list[dict[str, Any]] = []
    public_outer_fields = {"batch_index", "family_id", "seed", "fingerprint"}
    for raw in outer_raw:
        if not isinstance(raw, Mapping) or set(raw) != public_outer_fields:
            raise ValueError("V13 outer public identity schema differs")
        full = by_fingerprint.get(str(raw.get("fingerprint")))
        if (full is None or any(raw.get(name) != full.get(name)
                                for name in public_outer_fields)):
            raise ValueError("V13 outer identity differs from ranking universe")
        outer.append(deepcopy(full))
    if (len(outer) != materializer.registry_api.FRESH_OUTER_SCENE_COUNT
            or any(not isinstance(row, Mapping) for row in outer)
            or len({row["seed"] for row in outer}) != len(outer)
            or len({row["fingerprint"] for row in outer}) != len(outer)
            or any(row["seed"] not in set(excluded_seeds)
                   or row["fingerprint"] not in set(excluded_fingerprints)
                   for row in outer)):
        raise ValueError("Exact v13 outer identity exclusion required")
    return {
        "ranking_input_identities": stable,
        "ranking_input_count": len(stable),
        "ranking_input_identities_sha256": digest(stable),
        "historically_excluded_seeds": excluded_seeds,
        "historically_excluded_seed_count": len(excluded_seeds),
        "historically_excluded_seeds_sha256": digest(excluded_seeds),
        "historically_excluded_fingerprints": excluded_fingerprints,
        "historically_excluded_fingerprint_count": len(
            excluded_fingerprints),
        "historically_excluded_fingerprints_sha256": digest(
            excluded_fingerprints),
        "projector_eligible_identities": eligible,
        "projector_eligible_identity_count": len(eligible),
        "projector_eligible_identities_sha256": digest(eligible),
        "v13_outer_identities": outer,
        "v13_outer_identity_count": len(outer),
        "v13_outer_identities_sha256": digest(outer),
    }


def _traced_selection(
    *, prepared: Mapping[str, Any], salt: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], float]:
    """Run the unmodified frozen selector while observing hash-only returns."""
    projection_api = materializer.final_projection_api
    original = projection_api.project_observation_hashes
    source_before = projection_api.producer_sources()
    trace: list[dict[str, Any]] = []
    identity_by_fingerprint = {
        str(row["fingerprint"]): _identity(row)
        for row in prepared["identities"]
    }

    def observed(runtime: Any, scenes: Sequence[Mapping[str, Any]], *,
                 scene_offset: int, dense_critical: bool = False):
        result = projection_api.validate_projection(original(
            runtime, scenes, scene_offset=scene_offset,
            dense_critical=dense_critical))
        if len(scenes) != 1 or len(result["scenes"]) != 1:
            raise RuntimeError("V13 selector projector call shape differs")
        fingerprint = str(scenes[0].get("fingerprint"))
        identity = identity_by_fingerprint.get(fingerprint)
        if identity is None:
            raise RuntimeError("Projected v13 identity is outside ranking input")
        trace.append({
            "identity": deepcopy(identity),
            "scene_index": scene_offset,
            "ordered_observation_hashes": list(
                result["ordered_observation_hashes"]),
            "row_count": result["row_count"],
            "ordered_observation_hashes_sha256": result[
                "ordered_observation_hashes_sha256"],
            "unique_observation_count": result[
                "unique_observation_count"],
            "unique_observation_hashes_sha256": result[
                "unique_observation_hashes_sha256"],
            "environment_steps": result["environment_steps"],
        })
        return result

    started = time.perf_counter()
    projection_api.project_observation_hashes = observed
    try:
        accepted, statistics = materializer._select_and_replay(
            runtime=prepared["runtime"], identities=prepared["identities"],
            scene_by_fingerprint=prepared["scene_by_fingerprint"], salt=salt,
            excluded_seeds=set(prepared["excluded_seeds"]),
            excluded_fingerprints=set(prepared["excluded_fingerprints"]),
            forbidden_observation_hashes=set(
                prepared["forbidden_observation_hashes"]))
    finally:
        projection_api.project_observation_hashes = original
    elapsed = time.perf_counter() - started
    if (projection_api.project_observation_hashes is not original
            or projection_api.producer_sources() != source_before):
        raise RuntimeError("Frozen v13 projector source or call binding changed")
    accepted_by_fingerprint = {
        row["fingerprint"]: index for index, row in enumerate(accepted)}
    evaluated: list[dict[str, Any]] = []
    accepted_ordered: list[str] = []
    for index, item in enumerate(trace):
        identity = item["identity"]
        accepted_index = accepted_by_fingerprint.get(identity["fingerprint"])
        evaluated.append({
            **identity,
            "evaluation_index": index,
            "projector_scene_index": item["scene_index"],
            "accepted": accepted_index is not None,
            "accepted_scene_index": accepted_index,
        })
        if accepted_index is not None:
            accepted_ordered.extend(item["ordered_observation_hashes"])
    projection = statistics["projection"]
    if (len(accepted) != ACCEPTED_SCENE_COUNT
            or statistics.get("evaluated") != len(trace)
            or len(evaluated) < ACCEPTED_SCENE_COUNT
            or digest(accepted_ordered)
                != projection["ordered_observation_hashes_sha256"]
            or len(accepted_ordered) != projection["row_count"]
            or digest(sorted(set(accepted_ordered)))
                != projection["unique_observation_hashes_sha256"]):
        raise RuntimeError("Traced v13 selector differs from frozen selector output")
    return accepted, statistics, trace, elapsed


def _postclaim_reconstruct(
    *, claim_path: Path, evidence: Mapping[str, Any],
    materializer_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path, paths = _read_config_after_claim(
        claim_path=claim_path,
        materializer_config_path=materializer_config_path)
    if paths["permanent_final_registry"] != evidence["permanent_registry"]:
        raise ValueError("V13 config permanent registry differs from failure")
    authenticated = materializer._authenticate_public_inputs(
        claim_path=evidence["anchor_path"], anchor=evidence["anchor"],
        paths=paths)
    prepared = materializer._prepare_selection(
        paths=paths, authenticated=authenticated)
    universe_core = _universe(prepared, authenticated)
    # This is the first salt stat/read by the closeout.  The claim is already
    # durable in its permanent registry and the check above is repeated here.
    if not claim_path.is_file() or claim_path.parent.parent.is_symlink():
        raise RuntimeError("Permanent timeout-closeout claim disappeared")
    salt = materializer._read_committed_salt(paths["private_salt"])
    try:
        accepted, statistics, trace, elapsed = _traced_selection(
            prepared=prepared, salt=salt)
    finally:
        salt = b""

    evaluated = []
    all_ordered: list[str] = []
    environment_steps = 0
    accepted_fingerprints = {
        str(row["fingerprint"]) for row in accepted}
    for index, item in enumerate(trace):
        identity = item["identity"]
        accepted_index = next((position for position, row in enumerate(accepted)
                               if row["fingerprint"] == identity["fingerprint"]),
                              None)
        evaluated.append({
            **identity,
            "evaluation_index": index,
            "projector_scene_index": item["scene_index"],
            "accepted": identity["fingerprint"] in accepted_fingerprints,
            "accepted_scene_index": accepted_index,
            "row_count": item["row_count"],
            "ordered_observation_hashes_sha256": item[
                "ordered_observation_hashes_sha256"],
            "unique_observation_count": item["unique_observation_count"],
            "unique_observation_hashes_sha256": item[
                "unique_observation_hashes_sha256"],
            "environment_steps": item["environment_steps"],
        })
        all_ordered.extend(item["ordered_observation_hashes"])
        environment_steps += int(item["environment_steps"])
    identity_by_fingerprint = {
        row["fingerprint"]: row
        for row in universe_core["ranking_input_identities"]
    }
    accepted_identities = [deepcopy(identity_by_fingerprint[
        str(row["fingerprint"])]) for row in accepted]
    universe: dict[str, Any] = {
        "version": VERSION + ".burned-candidate-universe.v1",
        "status": "all_v13_ranking_inputs_and_completed_prefix_retired",
        **universe_core,
        "completed_evaluated_prefix": evaluated,
        "completed_evaluated_prefix_count": len(evaluated),
        "completed_evaluated_prefix_sha256": digest(evaluated),
        "reconstructed_accepted_identities": accepted_identities,
        "reconstructed_accepted_identity_count": len(accepted_identities),
        "reconstructed_accepted_identities_sha256": digest(
            accepted_identities),
        "all_ranking_input_identities_retired": True,
        "all_completed_prefix_identities_retired": True,
        "same_v13_candidate_universe_reuse_permitted": False,
        "scene_snapshots_included": False,
        "rng_state_or_salt_included": False,
        "formal_ready": False,
    }
    universe["content_sha256"] = digest(universe)
    projector_sources = materializer.final_projection_api.producer_sources()
    projection: dict[str, Any] = {
        "version": VERSION + ".burned-observation-projection.v1",
        "status": "completed_v13_evaluated_prefix_hashes_permanently_excluded",
        "selection_algorithm_version": materializer.VERSION,
        "selection_algorithm_source_closure_sha256": digest(
            evidence["materializer_sources"]),
        "projector_version": materializer.final_projection_api.VERSION,
        "projector_contract": materializer.final_projection_api.contract(),
        "projector_contract_sha256": digest(
            materializer.final_projection_api.contract()),
        "projector_sources": projector_sources,
        "projector_sources_sha256": digest(projector_sources),
        "completed_evaluated_prefix_count": len(evaluated),
        "accepted_scene_count": len(accepted_identities),
        "ordered_observation_hashes": all_ordered,
        "ordered_observation_hashes_sha256": digest(all_ordered),
        "unique_observation_hashes": sorted(set(all_ordered)),
        "unique_observation_hashes_sha256": digest(sorted(set(all_ordered))),
        "row_count": len(all_ordered),
        "unique_observation_count": len(set(all_ordered)),
        "environment_steps": environment_steps,
        "frozen_accepted_projection": deepcopy(statistics["projection"]),
        "timing": {
            "elapsed_seconds": elapsed,
            "clock": "time.perf_counter",
            "process_model": "single_process_in_process_frozen_v13_selector",
            "parallel_workers": 1,
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "information_boundary": {
            "closeout_claim_preceded_config_identity_and_salt_access": True,
            "burned_salt_used_only_for_exact_ranking_reconstruction": True,
            "salt_or_rng_state_included": False,
            "raw_observations_included": False,
            "actor_actions_included": False,
            "actor_probabilities_included": False,
            "program_file_or_predictions_accessed": False,
            "labels_or_rows_accessed_or_generated": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "formal_ready": False,
    }
    projection["content_sha256"] = digest(projection)
    metadata = {
        "config_path": config_path,
        "authenticated": authenticated,
        "statistics": statistics,
    }
    return universe, projection, metadata


def _publish(
    *, destination: Path, campaign: Path, claim: Mapping[str, Any],
    claim_raw: bytes, evidence: Mapping[str, Any], universe: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    universe_raw = _json_bytes(universe)
    projection_raw = _json_bytes(projection)
    sources = producer_sources()
    bindings = {
        "candidate_lock_sha256": evidence["anchor"]["attempt_key_inputs"][
            "candidate_lock_sha256"],
        "actor_sha256": evidence["anchor"]["attempt_key_inputs"][
            "actor_sha256"],
        "protocol_sha256": evidence["anchor"]["attempt_key_inputs"][
            "protocol_sha256"],
        "runtime_manifest_sha256": evidence["anchor"]["attempt_key_inputs"][
            "runtime_manifest_sha256"],
        "program_sha256": evidence["anchor"]["attempt_key_inputs"][
            "program_sha256"],
        "outer_result_sha256": evidence["anchor"]["attempt_key_inputs"][
            "outer_result_sha256"],
        "v13_anchor_sha256": EXPECTED_V13_ANCHOR_SHA256,
        "v13_anchor_content_sha256": evidence["anchor"]["content_sha256"],
        "v13_completion_sha256": EXPECTED_V13_COMPLETION_SHA256,
        "v13_completion_content_sha256": evidence["completion"][
            "content_sha256"],
        "v13_controller_source_closure_sha256": (
            EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256),
        "v13_materializer_source_closure_sha256": (
            EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256),
        "v13_materializer_config_sha256": EXPECTED_V13_CONFIG_SHA256,
        "v13_materializer_config_content_sha256": (
            EXPECTED_V13_CONFIG_CONTENT_SHA256),
        "closeout_claim_file_sha256": sha256(claim_raw).hexdigest(),
        "closeout_claim_content_sha256": claim["content_sha256"],
        "burned_candidate_universe_file_sha256": sha256(
            universe_raw).hexdigest(),
        "burned_candidate_universe_content_sha256": universe[
            "content_sha256"],
        "burned_observation_hashes_file_sha256": sha256(
            projection_raw).hexdigest(),
        "burned_observation_hashes_content_sha256": projection[
            "content_sha256"],
        "producer_sources_sha256": digest(sources),
    }
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "closeout_key": claim["closeout_claim_key"],
        "closeout_claim_key": claim["closeout_claim_key"],
        "burned_final_attempt_key": EXPECTED_V13_ATTEMPT_KEY,
        "contract": contract(),
        "v13_timeout_failure": {
            "controller_status": final_api.STATUS_FAILED,
            "controller_reason": "protected_final_phase_failed",
            "controller_materializer_timeout_seconds": (
                CONTROLLER_TIMEOUT_SECONDS),
            "timeout_attribution": (
                "operator-observed; v13 completion records the generic "
                "protected-final failure"),
            "permanent_campaign_entries": [
                final_api.ANCHOR_NAME, final_api.COMPLETION_NAME],
            "permanent_campaign_entry_count": 2,
            "public_output_published": False,
            "materializer_output_published": False,
            "final_material_rows_parity_or_audit_published": False,
            "salt_or_identity_may_have_been_accessed_before_timeout": True,
            "retry_allowed": False,
        },
        "burned_candidate_universe": {
            "ranking_input_count": universe["ranking_input_count"],
            "ranking_input_identities_sha256": universe[
                "ranking_input_identities_sha256"],
            "projector_eligible_identity_count": universe[
                "projector_eligible_identity_count"],
            "projector_eligible_identities_sha256": universe[
                "projector_eligible_identities_sha256"],
            "v13_outer_identity_count": universe[
                "v13_outer_identity_count"],
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
        },
        "burned_observation_projection": {
            "row_count": projection["row_count"],
            "unique_observation_count": projection[
                "unique_observation_count"],
            "ordered_observation_hashes_sha256": projection[
                "ordered_observation_hashes_sha256"],
            "unique_observation_hashes_sha256": projection[
                "unique_observation_hashes_sha256"],
            "environment_steps": projection["environment_steps"],
            "elapsed_seconds": projection["timing"]["elapsed_seconds"],
            "accepted_scene_count": projection["accepted_scene_count"],
        },
        "disposition": {
            "v13_candidate_and_program_reusable_without_refit": True,
            "v13_passed_outer_reusable_without_rerun": True,
            "v14_final_must_use_disjoint_expanded_candidate_universe": True,
            "v14_final_must_exclude_all_published_identity_and_hash_values": True,
            "same_v13_final_attempt_retry_permitted": False,
            "formal_ready": False,
        },
        "information_boundary": {
            "permanent_closeout_claim_preceded_closeout_salt_access": True,
            "burned_v13_salt_reopened_after_closeout_claim": True,
            "salt_value_published": False,
            "actions_probabilities_labels_or_program_accessed": False,
            "labeled_rows_generated": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "bindings": bindings,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    receipt_raw = _json_bytes(receipt)
    for name, raw in (
        (IDENTITY_NAME, universe_raw),
        (PROJECTION_NAME, projection_raw),
        (RECEIPT_NAME, receipt_raw),
    ):
        _write_exclusive(campaign / name, raw)
    _fsync_directory(campaign)

    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=destination.parent))
    try:
        for name, raw in (
            (CLAIM_NAME, claim_raw), (IDENTITY_NAME, universe_raw),
            (PROJECTION_NAME, projection_raw), (RECEIPT_NAME, receipt_raw),
        ):
            _write_exclusive(temporary / name, raw)
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return receipt


def build(
    *, final_anchor_path: str | Path, final_completion_path: str | Path,
    permanent_final_registry: str | Path, failed_public_output: str | Path,
    materializer_config_path: str | Path,
    permanent_closeout_registry: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Claim first, then reconstruct and publish the conservative exposure."""
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v14 timeout-closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    evidence = _authenticate_timeout_failure(
        final_anchor_path=final_anchor_path,
        final_completion_path=final_completion_path,
        permanent_final_registry=permanent_final_registry,
        failed_public_output=failed_public_output)
    campaign, claim, claim_raw = _create_permanent_claim(
        evidence=evidence,
        permanent_closeout_registry=permanent_closeout_registry)
    universe, projection, _metadata = _postclaim_reconstruct(
        claim_path=campaign / CLAIM_NAME, evidence=evidence,
        materializer_config_path=materializer_config_path)
    receipt = _publish(
        destination=parent / destination.name, campaign=campaign,
        claim=claim, claim_raw=claim_raw, evidence=evidence,
        universe=universe, projection=projection)
    from backend.training import warehouse_r41_diagnostic_final_timeout_closeout_public_v14 as public_api
    saved = public_api.read_saved_closeout_public(
        destination / RECEIPT_NAME,
        expected_closeout_sha256=file_hash(destination / RECEIPT_NAME),
        permanent_closeout_registry=permanent_closeout_registry)
    if saved != receipt:
        raise RuntimeError("Published v14 timeout closeout reread differs")
    return deepcopy(receipt)


__all__ = [
    "VERSION", "STATUS", "CLAIM_NAME", "RECEIPT_NAME", "IDENTITY_NAME",
    "PROJECTION_NAME", "EXPECTED_V13_ATTEMPT_KEY",
    "EXPECTED_V13_ANCHOR_SHA256", "EXPECTED_V13_ANCHOR_CONTENT_SHA256",
    "EXPECTED_V13_COMPLETION_SHA256",
    "EXPECTED_V13_CONFIG_SHA256", "RANKING_INPUT_COUNT",
    "ACCEPTED_SCENE_COUNT", "contract", "producer_sources", "build",
]
