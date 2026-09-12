"""Freeze the one fresh r4.1 diagnostic final holdout for RCPD v8.

This producer is deliberately explanation-program blind.  It selects only
from the pre-existing conflict-manifest candidate population, using a frozen
salt, scene identity, the public workload screen, and exact public-observation
separation.  It never imports, opens, or accepts an explanation program or an
RCPD report.

The exclusion closure contains every registered manifest split (including the
original final split), the formal X/Y scenes, the original development
supplement, both splits in the v8 development expansion, and all three burned
fresh-final registries.  Development NPZ files supply the exact observations
seen during fitting/selection; original/frozen-final and participant-scene
workloads are replayed with the same public final-audit protocol.  The caller
supplies retired v1/v2 only; the never-materialized v3 selection is generated
and burned inside the already claimed campaign.  Every touched selection-trace
scene is excluded.  Any scene, seed, or observation overlap fails closed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import pwd
import re
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    VERSION as MANIFEST_VERSION,
    validate_diagnostic_manifest,
)
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
DEVELOPMENT_EXPANSION_VERSION = "warehouse-r41-diagnostic-development-expansion.v8"
DEVELOPMENT_SUPPLEMENT_VERSION = "warehouse-r41-diagnostic-development-supplement.v1"
EXTERNAL_RETIRED_VERSIONS = (
    "warehouse-r41-diagnostic-fresh-final-holdout.v1",
    "warehouse-r41-diagnostic-fresh-final-holdout.v2",
)
RETIRED_VERSIONS = EXTERNAL_RETIRED_VERSIONS
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
    "d80f2736c6d5e359764f6ae4c09ccfa277f26e98a8910c1f18d86d84cb99851f"
)
EXPECTED_EXPANSION_REGISTRY_SHA256 = (
    "abbced9b99eb51857ad36c2d7b9351864c4472f5f9de922228d5744ae4d49d0a"
)
EXPECTED_SELECTED_SCENES_SHA256 = (
    "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
)
EXPECTED_RETIRED_HOLDOUT_SHA256 = {
    "warehouse-r41-diagnostic-fresh-final-holdout.v1": (
        "6c3ff25f9917451815979173d98881fe5e5e98258b514d199d50ee362d2746c2"
    ),
    "warehouse-r41-diagnostic-fresh-final-holdout.v2": (
        "7d72424744eea7547516cffe414c15578802c7204b90bdf1b129a1e9acfbf17b"
    ),
}
CANDIDATE_ARTIFACT_NAMES = frozenset({
    "program.json", "report.json", "rows.npz", "prior_v7_report.json",
    "prior_v7_rows.npz", "expansion_rows.npz", "fit_config.json",
})
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
_HEX = re.compile(r"[0-9a-f]{64}\Z")


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
        "external_retired_final_registries": list(EXTERNAL_RETIRED_VERSIONS),
        "external_retired_final_sha256": dict(
            sorted(EXPECTED_RETIRED_HOLDOUT_SHA256.items())),
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
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


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
        "retired_holdout_sha256": dict(
            sorted(EXPECTED_RETIRED_HOLDOUT_SHA256.items())),
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
            or path.resolve() != path or file_hash(path) != expected_claim_sha256):
        raise ValueError("A valid fixed-campaign claim is required")
    receipt = _read(path, "final-once claim")
    anchor_path = _anchor_path()
    anchor = _read(anchor_path, "permanent final-once anchor")
    candidate_marker_path = path.parent / "candidate_authenticated.json"
    candidate_marker = _read(
        candidate_marker_path, "strict candidate authentication phase")
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
            or receipt.get("permanent_anchor_sha256") != file_hash(anchor_path)
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
            or candidate_marker.get("prior_v7_report_file_sha256")
                != candidate_artifacts.get("prior_v7_report.json")
            or candidate_marker.get("prior_v7_rows_file_sha256")
                != candidate_artifacts.get("prior_v7_rows.npz")
            or candidate_marker.get("expansion_rows_file_sha256")
                != candidate_artifacts.get("expansion_rows.npz")
            or candidate_marker.get("fit_config_file_sha256")
                != candidate_artifacts.get("fit_config.json")
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
    return path.parent, receipt


def _claim_marker(directory: Path, name: str, value: Mapping[str, Any]) -> Path:
    path = directory / name
    _write_phase_new(path, dict(value))
    return path


def _read_committed_salt(path: str | Path) -> bytes:
    if HOLDOUT_SALT_COMMITMENT == "0" * 64:
        raise ValueError("Private holdout salt commitment is not provisioned")
    value = _regular(path, "private holdout salt", maximum=4096).read_bytes()
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


def _read(path: str | Path, label: str) -> dict[str, Any]:
    path = _regular(path, label)

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (type(claimed) is str and _HEX.fullmatch(claimed) is not None
            and claimed == digest({key: item for key, item in value.items()
                                   if key != "content_sha256"}))


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


def _write_phase_new(path: Path, value: Any) -> None:
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
        bindings[path.name] = file_hash(path)
        with np.load(path, allow_pickle=False) as rows:
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
            observation_hashes.update(map(str, decoded_hashes))
            scene_fingerprints.update(map(str, decoded_fingerprints))
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


def _development_registries(paths: Sequence[Path]) -> tuple[list[dict], dict[str, dict]]:
    values: dict[str, dict] = {}
    scenes: list[dict] = []
    for path in paths:
        value = _read(path, "development registry")
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
            if (value.get("status") != "passed_program_blind_registry"
                    or value.get("program_predictions_access") is not False
                    or value.get("final_labels_used_for_selection") is not False):
                raise ValueError("Development expansion boundary differs")
            scenes.extend(_scene_rows(value.get("fit_supplement"),
                                     label="development fit supplement", expected_count=128))
            scenes.extend(_scene_rows(value.get("development_validation"),
                                     label="development validation", expected_count=64))
        values[str(version)] = value
    if set(values) != {DEVELOPMENT_SUPPLEMENT_VERSION, DEVELOPMENT_EXPANSION_VERSION}:
        raise ValueError("Exactly the v1 supplement and v8 expansion are required")
    if len({row["fingerprint"] for row in scenes}) != len(scenes):
        raise ValueError("Development registries overlap one another")
    bindings = {
        version: {"file_sha256": file_hash(path),
                  "content_sha256": values[version]["content_sha256"]}
        for version in sorted(values)
        for path in paths if _read(path, "development registry").get("version") == version
    }
    return scenes, bindings


def _retired_registries(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    values: dict[str, tuple[dict, Path]] = {}
    for path in paths:
        value = _read(path, "retired fresh-final registry")
        version = value.get("version")
        if version not in EXTERNAL_RETIRED_VERSIONS or version in values:
            raise ValueError("Exactly one retired v1 and v2 registry is required")
        if (file_hash(path) != EXPECTED_RETIRED_HOLDOUT_SHA256[str(version)]
                or not _content_valid(value)
                or value.get("program_access") is not False
                or value.get("contract", {}).get("program_predictions_access") is not False
                or value.get("statistics", {}).get("accepted") != TOTAL_SCENES
                or value.get("statistics", {}).get("public_observation_overlap") != 0):
            raise ValueError("Retired fresh-final registry identity differs")
        values[str(version)] = (value, path)
    if set(values) != set(EXTERNAL_RETIRED_VERSIONS):
        raise ValueError("Exactly one retired v1 and v2 registry is required")
    registries: list[dict[str, Any]] = []
    for version in EXTERNAL_RETIRED_VERSIONS:
        value = values[version][0]
        scenes = _scene_rows(
            value.get("scenes"), label="retired " + version,
            expected_count=TOTAL_SCENES)
        trace = value.get("selection_trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError("Retired fresh-final selection trace is required")
        registries.append({"version": version, "scenes": scenes,
                           "selection_trace": deepcopy(trace), "raw": value})
    bindings = {version: {
        "file_sha256": file_hash(values[version][1]),
        "content_sha256": values[version][0]["content_sha256"],
    } for version in EXTERNAL_RETIRED_VERSIONS}
    return registries, bindings


def _validate_retired_expansion_bindings(
    retired_bindings: Mapping[str, Mapping[str, str]],
    expansion: Mapping[str, Any],
) -> None:
    expansion_bindings = expansion.get("bindings")
    frozen = (expansion_bindings.get("retired_holdouts")
              if isinstance(expansion_bindings, Mapping) else None)
    if (dict(retired_bindings) != frozen
            or not isinstance(frozen, Mapping)
            or set(frozen) != set(EXTERNAL_RETIRED_VERSIONS)
            or any(frozen[version].get("file_sha256")
                   != EXPECTED_RETIRED_HOLDOUT_SHA256[version]
                   for version in EXTERNAL_RETIRED_VERSIONS)):
        raise ValueError(
            "Retired v1/v2 bytes must match the fixed development expansion")


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


def _retired_exposure_closure(
    *, runtime: R41DiagnosticOnlineAlignmentRuntime, actor: Any,
    manifest: Mapping[str, Any], registries: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[int], set[str], dict[str, Any]]:
    """Reconstruct every retired selection-trace exposure, not only admissions."""
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
    retired_registries: Sequence[Mapping[str, Any]],
    claim_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Burn the never-materialized v3 selection inside the claimed campaign."""
    scenes, trace, statistics = legacy_v3._select(
        runtime=runtime, actor=actor, manifest=manifest, selected=selected,
        development_hashes=set(legacy_development_hashes),
        previous_fresh_scene_sets=[deepcopy(row["scenes"])
                                   for row in retired_registries],
    )
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
    retired_holdout_paths: Sequence[str | Path], output: str | Path,
    claim_receipt_path: str | Path, expected_claim_sha256: str,
    expected_campaign_key: str, expected_candidate_identity_sha256: str,
    selection_salt_path: str | Path,
) -> dict[str, Any]:
    claim_dir, _ = _claim_receipt(
        claim_receipt_path, expected_claim_sha256=expected_claim_sha256,
        expected_campaign_key=expected_campaign_key,
        expected_candidate_identity_sha256=expected_candidate_identity_sha256,
    )
    output_path = Path(output).expanduser().absolute()
    if (output_path.exists() or output_path.is_symlink()
            or not output_path.parent.is_dir() or output_path.parent.is_symlink()):
        raise ValueError("Fresh-final output must be a new directory under an existing parent")
    _claim_marker(claim_dir, "holdout_started.json", {
        "version": VERSION, "status": "started_no_retry",
        "campaign_key": expected_campaign_key,
        "candidate_identity_sha256": expected_candidate_identity_sha256,
        "attempt_started_sha256": expected_claim_sha256,
        "candidate_authenticated_sha256": file_hash(
            claim_dir / "candidate_authenticated.json"),
        "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
    })
    selection_salt = _read_committed_salt(selection_salt_path)
    output_path.mkdir(mode=0o700)
    actor_path = _regular(actor_path, "frozen Actor")
    protocol_path = _regular(protocol_path, "training protocol")
    manifest_path = _regular(manifest_path, "conflict manifest")
    designation_path = _regular(designation_path, "Actor designation")
    selected_path = _regular(selected_scenes_path, "formal X/Y selection")
    development_paths = [_regular(path, "development registry")
                         for path in development_registry_paths]
    row_paths = [_regular(path, "development rows", maximum=MAX_NPZ_BYTES)
                 for path in development_rows_paths]
    legacy_rows_path = _regular(
        legacy_v3_rows_path, "legacy v3 development rows", maximum=MAX_NPZ_BYTES)
    retired_paths = [_regular(path, "retired fresh-final registry")
                     for path in retired_holdout_paths]
    all_input_paths = [actor_path, protocol_path, manifest_path,
                       designation_path, selected_path, *development_paths,
                       *row_paths, legacy_rows_path, *retired_paths]
    initial_file_hashes = {str(path): file_hash(path) for path in all_input_paths}
    initial_sources = producer_sources()
    candidate_marker = _read(
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
                != file_hash(legacy_rows_path)):
        raise ValueError("Claim-authenticated final campaign input differs")

    manifest = _read(manifest_path, "conflict manifest")
    validate_diagnostic_manifest(manifest, replay=False)
    if manifest.get("version") != MANIFEST_VERSION or not _content_valid(manifest):
        raise ValueError("Frozen conflict manifest identity differs")
    selected = _read(selected_path, "formal X/Y selection")
    designation = _read(designation_path, "Actor designation")
    actor_sha256 = file_hash(actor_path)
    if (selected.get("release_eligible") is not True
            or selected.get("actor_sha256") != actor_sha256
            or manifest.get("frozen_actor", {}).get("sha256") != actor_sha256
            or designation.get("bindings", {}).get("actor_sha256") != actor_sha256
            or designation.get("behavior_performance_gate_waived") is not True
            or designation.get("runtime_action_override") is not False):
        raise ValueError("Frozen diagnostic Actor/scene designation differs")

    development_scenes, development_bindings = _development_registries(
        development_paths)
    retired_registries, retired_bindings = _retired_registries(retired_paths)
    development_values = {
        _read(path, "development registry")["version"]:
            _read(path, "development registry")
        for path in development_paths
    }
    supplement = development_values[DEVELOPMENT_SUPPLEMENT_VERSION]
    expansion = development_values[DEVELOPMENT_EXPANSION_VERSION]
    _validate_retired_expansion_bindings(retired_bindings, expansion)
    if (candidate_marker.get("development_registries") != {
            version: binding["file_sha256"]
            for version, binding in sorted(development_bindings.items())
        }
            or file_hash(next(
                path for path in development_paths
                if _read(path, "development registry").get("version")
                   == DEVELOPMENT_EXPANSION_VERSION
            )) != EXPECTED_EXPANSION_REGISTRY_SHA256
            or supplement.get("bindings", {}).get("actor_sha256") != actor_sha256
            or supplement.get("bindings", {}).get("source_manifest_sha256")
                != file_hash(manifest_path)
            or supplement.get("bindings", {}).get("selected_scenes_sha256")
                != file_hash(selected_path)
            or expansion.get("bindings", {}).get("actor_sha256") != actor_sha256
            or expansion.get("bindings", {}).get("source_manifest_file_sha256")
                != file_hash(manifest_path)
            or expansion.get("bindings", {}).get("selected_scenes_file_sha256")
                != file_hash(selected_path)
            or expansion.get("bindings", {}).get("designation_file_sha256")
                != file_hash(designation_path)
            or expansion.get("bindings", {}).get("previous_development_file_sha256")
                != next(binding["file_sha256"] for version, binding
                        in development_bindings.items()
                        if version == DEVELOPMENT_SUPPLEMENT_VERSION)):
        raise ValueError("Development registry frozen-input binding differs")
    retired_values = {
        _read(path, "retired fresh-final registry")["version"]:
            _read(path, "retired fresh-final registry")
        for path in retired_paths
    }
    for version, value in retired_values.items():
        bindings = value.get("bindings", {})
        if (bindings.get("actor_sha256") != actor_sha256
                or bindings.get("protocol_sha256") != file_hash(protocol_path)
                or bindings.get("source_manifest_sha256") != file_hash(manifest_path)
                or bindings.get("selected_scenes_sha256") != file_hash(selected_path)):
            raise ValueError("Retired fresh-final frozen-input binding differs: " + version)
    registered = [deepcopy(row) for rows in manifest["splits"].values() for row in rows]
    xy = [deepcopy(row) for key in ("X", "Y") for row in selected.get(key, [])]
    if len(xy) != 6:
        raise ValueError("Exactly six frozen X/Y participant scenes are required")
    required_development_fingerprints = {
        row["fingerprint"] for row in development_scenes
    } | {
        row["fingerprint"]
        for split_name, rows in manifest["splits"].items()
        if split_name != "final_test"
        for row in rows
    }
    development_hashes, development_row_scenes, row_bindings = (
        _npz_development_observations(
            row_paths,
            required_fingerprints=required_development_fingerprints,
        )
    )
    legacy_development_hashes, _, legacy_rows_binding = (
        _npz_development_observations(
            [legacy_rows_path], required_fingerprints=set())
    )
    protocol = _read(protocol_path, "training protocol")
    content = deepcopy(manifest); manifest_content = content.pop("content_sha256")
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor_path,
        training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=actor_sha256,
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=manifest_content,
        expected_manifest_semantic_sha256=digest(manifest),
    )

    claim_binding = {
        "campaign_key": expected_campaign_key,
        "attempt_started_sha256": expected_claim_sha256,
        "candidate_identity_sha256": expected_candidate_identity_sha256,
        "candidate_authenticated_sha256": file_hash(
            claim_dir / "candidate_authenticated.json"),
    }
    v3_tombstone = _build_v3_tombstone(
        runtime=runtime, actor=runtime.actor, manifest=manifest, selected=selected,
        legacy_development_hashes=legacy_development_hashes,
        retired_registries=retired_registries, claim_binding=claim_binding,
    )
    _write_new(output_path / "v3_exclusion.json", v3_tombstone)
    exposure_registries = [*retired_registries, {
        "version": V3_TOMBSTONE_VERSION,
        "scenes": v3_tombstone["scenes"],
        "selection_trace": v3_tombstone["selection_trace"],
    }]
    exposed_fingerprints, exposed_seeds, exposed_observations, exposure_stats = (
        _retired_exposure_closure(
            runtime=runtime, actor=runtime.actor, manifest=manifest,
            registries=exposure_registries,
        )
    )
    all_excluded = [*registered, *xy, *development_scenes,
                    *(row for registry in exposure_registries
                      for row in registry["scenes"])]
    excluded_fingerprints = (
        {row["fingerprint"] for row in all_excluded} | exposed_fingerprints)
    excluded_seeds = {row["seed"] for row in all_excluded} | exposed_seeds

    # These workloads may not be present in the RCPD rows and must be excluded
    # explicitly: the original final split, formal X/Y scenes, and all burned
    # final holdouts.  Their original per-registry indexes are preserved.
    forbidden = set(development_hashes)
    replay_sets = [manifest["splits"]["final_test"], xy]
    replay_observations: set[str] = set(exposed_observations)
    for scenes in replay_sets:
        for index, scene in enumerate(scenes):
            replay_observations.update(
                _exact_final_workload_observations(runtime, scene, index))
    forbidden.update(replay_observations)

    frozen_forbidden = set(forbidden)
    scenes, trace, statistics, accepted_hashes = _select(
        runtime=runtime,
        actor=runtime.actor,
        manifest=manifest,
        selection_salt=selection_salt,
        excluded_fingerprints=set(excluded_fingerprints),
        excluded_seeds=set(excluded_seeds),
        forbidden_observation_hashes=set(frozen_forbidden),
    )
    if ({row["fingerprint"] for row in scenes} & excluded_fingerprints
            or {row["seed"] for row in scenes} & excluded_seeds
            or accepted_hashes & frozen_forbidden):
        raise RuntimeError("Fresh-final isolation invariant failed")
    statistics.update({
        "excluded_scene_fingerprints": len(excluded_fingerprints),
        "excluded_scene_seeds": len(excluded_seeds),
        "development_public_observation_count": len(development_hashes),
        "replayed_excluded_public_observation_count": len(replay_observations),
        "development_row_scene_count": len(development_row_scenes),
        "retired_selection_trace_exposure": exposure_stats,
        "retired_touched_scene_count": len(exposed_fingerprints),
        "retired_touched_observation_count": len(exposed_observations),
    })
    sources = producer_sources()
    if (sources != initial_sources
            or {str(path): file_hash(path) for path in all_input_paths}
               != initial_file_hashes):
        raise RuntimeError("Fresh-final frozen input/source changed during selection")
    bindings = {
        "actor_sha256": actor_sha256,
        "protocol_file_sha256": file_hash(protocol_path),
        "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": file_hash(manifest_path),
        "manifest_content_sha256": manifest_content,
        "manifest_semantic_sha256": digest(manifest),
        "designation_file_sha256": file_hash(designation_path),
        "designation_semantic_sha256": digest(designation),
        "selected_scenes_file_sha256": file_hash(selected_path),
        "selected_scenes_semantic_sha256": digest(selected),
        "development_registries": development_bindings,
        "development_rows": row_bindings,
        "legacy_v3_rows": legacy_rows_binding,
        "retired_holdouts": retired_bindings,
        "v3_exclusion_file_sha256": file_hash(output_path / "v3_exclusion.json"),
        "v3_exclusion_content_sha256": v3_tombstone["content_sha256"],
        "claim": claim_binding,
        "selection_salt_commitment": HOLDOUT_SALT_COMMITMENT,
        "excluded_fingerprints_sha256": digest(sorted(excluded_fingerprints)),
        "excluded_seeds_sha256": digest(sorted(excluded_seeds)),
        "development_observations_sha256": digest(sorted(development_hashes)),
        "replayed_excluded_observations_sha256": digest(sorted(replay_observations)),
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
    _write_new(output_path / "holdout.json", holdout)
    report = {
        "version": VERSION,
        "status": "passed_program_blind_registry",
        "bindings": bindings,
        "holdout_file_sha256": file_hash(output_path / "holdout.json"),
        "holdout_content_sha256": holdout["content_sha256"],
        "v3_exclusion_file_sha256": file_hash(output_path / "v3_exclusion.json"),
        "statistics": statistics,
        "producer_sources": sources,
        "program_access": False,
        "program_predictions_access": False,
        "final_labels_used_for_selection": False,
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    _write_new(output_path / "report.json", report)
    _claim_marker(claim_dir, "holdout_completed.json", {
        "version": VERSION, "status": "completed_program_blind",
        **claim_binding,
        "holdout_file_sha256": file_hash(output_path / "holdout.json"),
        "report_file_sha256": file_hash(output_path / "report.json"),
        "v3_exclusion_file_sha256": file_hash(output_path / "v3_exclusion.json"),
    })
    return deepcopy(report)


def read_saved_holdout(
    output: str | Path, *, expected_holdout_sha256: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    """Authenticate the frozen registry without re-running final workloads."""
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Fresh-final evidence directory is unsafe")
    holdout_path, report_path = output / "holdout.json", output / "report.json"
    if (file_hash(holdout_path) != expected_holdout_sha256
            or file_hash(report_path) != expected_report_sha256):
        raise ValueError("Fresh-final evidence hash differs")
    holdout = _read(holdout_path, "saved fresh-final holdout")
    report = _read(report_path, "saved fresh-final report")
    v3_path = _regular(output / "v3_exclusion.json", "saved v3 exclusion")
    v3_tombstone = _read(v3_path, "saved v3 exclusion")
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
            or report.get("v3_exclusion_file_sha256") != file_hash(v3_path)
            or holdout.get("bindings", {}).get("v3_exclusion_file_sha256")
                != file_hash(v3_path)
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
    return deepcopy(holdout)


__all__ = [
    "VERSION", "TOTAL_SCENES", "FAMILY_QUOTAS", "contract",
    "producer_sources", "read_saved_holdout",
]
