"""Public-only reader for a frozen v13 burned-final closeout.

This reader intentionally has no NumPy/ZIP dependency and never opens an NPZ
member.  It authenticates the receipt, the three public JSON companions, the
opaque row-file bytes, and their append-only permanent mirrors.  A selector
must freeze its public observation-hash masks before it uses a separate strict
row loader to open either row archive.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from backend.training.warehouse_native_common import digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-attempt-closeout.v13"
STATUS = "burned_v12_final_irrevocably_closed_and_promoted"
RECEIPT_NAME = "closeout_receipt.json"
IDENTITY_NAME = "burned_identity_registry.json"
PROMOTED_PROJECTION_NAME = "ordered_promoted_observation_hashes.json"
COMBINED_PROJECTION_NAME = "combined_promoted_observation_hashes.json"
PROMOTED_ROWS_NAME = "promoted_rows.npz"
COMBINED_ROWS_NAME = "combined_promoted_rows.npz"
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def contract() -> dict[str, Any]:
    # Frozen copy of the producer contract.  Do not import the producer here:
    # its historic source closure must remain independently authenticatable.
    return {
        "version": VERSION,
        "authenticate_burned_completion_before_scene_access": True,
        "same_final_attempt_retry_permitted": False,
        "protected_salt_reused": False,
        "protected_materializer_invoked": False,
        "post_closeout_deterministic_development_recollection": True,
        "independent_physical_replay": True,
        "runtime_action_override": False,
        "program_controls_runtime_actions": False,
        "combined_promoted_source_order": [
            "consumed_v11_outer", "consumed_v12_outer", "burned_v12_final",
        ],
        "all_promoted_rows_marked_development": True,
        "burned_final_identity_and_observation_reuse_permitted": False,
        "consumed_v12_outer_identity_and_observation_reuse_permitted": False,
        "formal_ready": False,
    }


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    claimed = value.get("content_sha256")
    return (
        type(claimed) is str
        and _HEX.fullmatch(claimed) is not None
        and claimed == digest({key: child for key, child in value.items()
                               if key != "content_sha256"})
    )


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, maximum: int,
             expected_sha256: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum
            or file_hash(path) != _sha(expected_sha256, label + " SHA-256")):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, bytes, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES,
                    expected_sha256=expected_sha256)
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
    if not isinstance(parsed, dict) or file_hash(path) != expected_sha256:
        raise RuntimeError(label + " changed during public-only read")
    return path, raw, parsed


def _validate_projection(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _content_valid(value):
        raise ValueError(label + " projection content differs")
    ordered = value.get("ordered_observation_hashes")
    unique = value.get("outer_observation_hashes")
    if (not isinstance(ordered, list) or not isinstance(unique, list)
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in ordered)
            or unique != sorted(set(unique))
            or sorted(set(ordered)) != unique
            or value.get("row_count") != len(ordered)
            or value.get("ordered_observation_hashes_sha256") != digest(ordered)
            or value.get("unique_outer_observation_hash_count") != len(unique)
            or value.get("outer_observation_hashes_sha256") != digest(unique)
            or any(value.get(name) is not False for name in (
                "raw_observation_values_included", "action_values_included",
                "probability_values_included", "label_values_included"))):
        raise ValueError(label + " one-way projection differs")
    return deepcopy(dict(value))


def _validate_identity_registry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not _content_valid(value):
        raise ValueError("Burned identity registry content differs")
    identities = value.get("identities")
    if (value.get("version") != VERSION + ".burned-identity-registry.v1"
            or value.get("status")
                != "burned_final_identities_permanently_excluded"
            or not isinstance(identities, list)
            or value.get("scene_count") != len(identities)
            or value.get("identities_sha256") != digest(identities)
            or value.get("scene_snapshots_included") is not False
            or value.get("rng_state_included") is not False
            or value.get("hidden_selection_material_included") is not False):
        raise ValueError("Burned identity registry semantics differ")
    seen_seeds: set[int] = set()
    seen_fingerprints: set[str] = set()
    for row in identities:
        if (not isinstance(row, Mapping)
                or set(row) != {"batch_index", "family_id", "seed", "fingerprint"}
                or type(row["batch_index"]) is not int
                or type(row["family_id"]) is not str or not row["family_id"]
                or type(row["seed"]) is not int or isinstance(row["seed"], bool)
                or type(row["fingerprint"]) is not str
                or _HEX.fullmatch(row["fingerprint"]) is None
                or row["seed"] in seen_seeds
                or row["fingerprint"] in seen_fingerprints):
            raise ValueError("Burned public identity differs")
        seen_seeds.add(row["seed"])
        seen_fingerprints.add(row["fingerprint"])
    return deepcopy(dict(value))


def _same_bytes(left: Path, right: Path) -> bool:
    if (left.stat(follow_symlinks=False).st_size
            != right.stat(follow_symlinks=False).st_size):
        return False
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            one = first.read(1024 * 1024)
            two = second.read(1024 * 1024)
            if one != two:
                return False
            if not one:
                return True


def read_saved_closeout_public(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate only public companions; never open an NPZ member."""
    receipt_path, receipt_raw, value = _strict_json(
        path, "v13 burned-final closeout",
        expected_sha256=expected_closeout_sha256)
    sources = value.get("producer_sources")
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or value.get("contract") != contract() or not _content_valid(value)
            or value.get("formal_ready") is not False
            or type(value.get("closeout_key")) is not str
            or _HEX.fullmatch(value["closeout_key"]) is None
            or not isinstance(sources, Mapping) or not sources
            or any(type(name) is not str or not name
                   or type(checksum) is not str or _HEX.fullmatch(checksum) is None
                   for name, checksum in sources.items())
            or value.get("producer_sources_sha256")
                != digest(dict(sorted(sources.items())))
            or value.get("burned_final", {}).get("retry_allowed") is not False
            or value.get("burned_final", {}).get("protected_salt_reused") is not False
            or value.get("disposition", {}).get(
                "same_protected_final_attempt_retry_permitted") is not False
            or value.get("information_boundary", {}).get(
                "protected_salt_statted_or_read") is not False):
        raise ValueError("Saved v13 public closeout semantics differ")

    directory = _directory(receipt_path.parent, "v13 closeout artifact directory")
    expected_names = {
        RECEIPT_NAME, IDENTITY_NAME, PROMOTED_PROJECTION_NAME,
        COMBINED_PROJECTION_NAME, PROMOTED_ROWS_NAME, COMBINED_ROWS_NAME,
    }
    if {entry.name for entry in directory.iterdir()} != expected_names:
        raise ValueError("Saved v13 public closeout artifact set differs")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Saved v13 public closeout bindings differ")

    _, _, identity = _strict_json(
        directory / IDENTITY_NAME, "burned identity registry",
        expected_sha256=bindings["burned_identity_registry_sha256"])
    identity = _validate_identity_registry(identity)
    _, _, promoted_projection = _strict_json(
        directory / PROMOTED_PROJECTION_NAME, "burned final projection",
        expected_sha256=bindings["burned_projection_sha256"])
    promoted_projection = _validate_projection(
        promoted_projection, "burned final")
    _, _, combined_projection = _strict_json(
        directory / COMBINED_PROJECTION_NAME, "combined promoted projection",
        expected_sha256=bindings["combined_projection_sha256"])
    combined_projection = _validate_projection(
        combined_projection, "combined promoted")
    burned = value.get("burned_final")
    combined = value.get("combined_promoted_development")
    if (not isinstance(burned, Mapping) or not isinstance(combined, Mapping)
            or burned.get("identities") != identity["identities"]
            or burned.get("identities_sha256") != identity["identities_sha256"]
            or burned.get("observation_hash_projection") != promoted_projection
            or combined.get("source_order")
                != contract()["combined_promoted_source_order"]
            or combined.get("observation_hash_projection") != combined_projection
            or combined.get("all_rows_split_validation_true") is not True
            or burned.get("rows_sha256") != bindings["promoted_rows_sha256"]
            or burned.get("rows_semantic_sha256")
                != bindings["promoted_rows_semantic_sha256"]
            or combined.get("rows_sha256")
                != bindings["combined_promoted_rows_sha256"]
            or combined.get("rows_semantic_sha256")
                != bindings["combined_promoted_rows_semantic_sha256"]
            or identity["content_sha256"]
                != bindings["burned_identity_registry_content_sha256"]
            or promoted_projection["content_sha256"]
                != bindings["burned_projection_content_sha256"]
            or combined_projection["content_sha256"]
                != bindings["combined_projection_content_sha256"]):
        raise ValueError("Saved v13 public companion binding differs")

    opaque_rows = {
        PROMOTED_ROWS_NAME: bindings["promoted_rows_sha256"],
        COMBINED_ROWS_NAME: bindings["combined_promoted_rows_sha256"],
    }
    local_rows = {
        name: _regular(directory / name, name, maximum=MAX_NPZ_BYTES,
                       expected_sha256=checksum)
        for name, checksum in opaque_rows.items()
    }
    # Semantic row SHA values remain opaque promises at this stage.  They are
    # checked only after the selector freezes both public hash masks and uses
    # its strict NPZ loader.
    for name in (
        "promoted_rows_semantic_sha256",
        "combined_promoted_rows_semantic_sha256",
    ):
        _sha(bindings.get(name), name)

    permanent = _directory(permanent_closeout_registry,
                           "permanent v13 closeout registry")
    campaign = _directory(permanent / value["closeout_key"],
                          "permanent v13 closeout campaign")
    if {entry.name for entry in campaign.iterdir()} != expected_names:
        raise ValueError("Permanent v13 public closeout artifact set differs")
    for name in expected_names:
        maximum = MAX_NPZ_BYTES if name.endswith(".npz") else MAX_JSON_BYTES
        checksum = (expected_closeout_sha256 if name == RECEIPT_NAME
                    else file_hash(directory / name))
        permanent_file = _regular(
            campaign / name, "permanent " + name, maximum=maximum,
            expected_sha256=checksum)
        if not _same_bytes(directory / name, permanent_file):
            raise ValueError("Permanent v13 closeout artifact differs: " + name)
    if (receipt_raw != (campaign / RECEIPT_NAME).read_bytes()
            or not all(path.stat(follow_symlinks=False).st_size > 0
                       for path in local_rows.values())):
        raise ValueError("Permanent v13 public receipt/row binding differs")
    return deepcopy(value)


__all__ = [
    "VERSION", "STATUS", "RECEIPT_NAME", "IDENTITY_NAME",
    "PROMOTED_PROJECTION_NAME", "COMBINED_PROJECTION_NAME",
    "PROMOTED_ROWS_NAME", "COMBINED_ROWS_NAME", "contract",
    "read_saved_closeout_public",
]
