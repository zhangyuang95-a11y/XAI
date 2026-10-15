from __future__ import annotations

from hashlib import sha256
import inspect
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as subject
from backend.training import warehouse_r41_diagnostic_outer_collection_v12 as collection_api
from backend.training.warehouse_native_common import canonical, digest


A, B, C = "a" * 64, "b" * 64, "c" * 64


def _raw(value: dict) -> bytes:
    return (canonical(value) + "\n").encode()


def _with_content(value: dict) -> dict:
    value = dict(value)
    value["content_sha256"] = digest(value)
    return value


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    output = tmp_path / "published"
    permanent = tmp_path / "permanent"
    campaign = permanent / C
    output.mkdir(); campaign.mkdir(parents=True)
    identity = _with_content({
        "version": subject.VERSION + ".burned-identity-registry.v1",
        "status": "burned_final_identities_permanently_excluded",
        "scene_count": 1,
        "identities": [{
            "batch_index": 0, "family_id": "conflict_family_01",
            "seed": 1, "fingerprint": A,
        }],
        "identities_sha256": digest([{
            "batch_index": 0, "family_id": "conflict_family_01",
            "seed": 1, "fingerprint": A,
        }]),
        "family_counts": {"conflict_family_01": 1},
        "scene_snapshots_included": False, "rng_state_included": False,
        "hidden_selection_material_included": False, "formal_ready": False,
    })

    def projection(version: str, source: str, ordered: list[str]) -> dict:
        unique = sorted(set(ordered))
        return _with_content({
            "version": version,
            "status": "development_exposed_one_way_observation_hash_projection",
            "source": source, "row_count": len(ordered),
            "ordered_observation_hashes": ordered,
            "ordered_observation_hashes_sha256": digest(ordered),
            "unique_outer_observation_hash_count": len(unique),
            "outer_observation_hashes": unique,
            "outer_observation_hashes_sha256": digest(unique),
            "raw_observation_values_included": False,
            "action_values_included": False,
            "probability_values_included": False,
            "label_values_included": False, "formal_ready": False,
        })

    promoted = projection("promoted.v1", "burned", [A])
    combined = projection("combined.v1", "combined", [A, B])
    companions = {
        subject.IDENTITY_NAME: _raw(identity),
        subject.PROMOTED_PROJECTION_NAME: _raw(promoted),
        subject.COMBINED_PROJECTION_NAME: _raw(combined),
        subject.PROMOTED_ROWS_NAME: b"opaque-promoted-row-archive",
        subject.COMBINED_ROWS_NAME: b"opaque-combined-row-archive",
    }
    hashes = {name: sha256(raw).hexdigest() for name, raw in companions.items()}
    sources = {
        "backend/training/warehouse_r41_diagnostic_final_attempt_closeout_v13.py": B,
    }
    receipt = {
        "version": subject.VERSION, "status": subject.STATUS,
        "closeout_key": C, "attempt_key": B, "contract": subject.contract(),
        "burned_final": {
            "retry_allowed": False, "protected_salt_reused": False,
            "identities": identity["identities"],
            "identities_sha256": identity["identities_sha256"],
            "rows_sha256": hashes[subject.PROMOTED_ROWS_NAME],
            "rows_semantic_sha256": A,
            "observation_hash_projection": promoted,
        },
        "combined_promoted_development": {
            "source_order": subject.contract()["combined_promoted_source_order"],
            "rows_sha256": hashes[subject.COMBINED_ROWS_NAME],
            "rows_semantic_sha256": B,
            "all_rows_split_validation_true": True,
            "observation_hash_projection": combined,
        },
        "bindings": {
            "burned_identity_registry_sha256": hashes[subject.IDENTITY_NAME],
            "burned_identity_registry_content_sha256": identity["content_sha256"],
            "burned_projection_sha256": hashes[subject.PROMOTED_PROJECTION_NAME],
            "burned_projection_content_sha256": promoted["content_sha256"],
            "combined_projection_sha256": hashes[subject.COMBINED_PROJECTION_NAME],
            "combined_projection_content_sha256": combined["content_sha256"],
            "promoted_rows_sha256": hashes[subject.PROMOTED_ROWS_NAME],
            "promoted_rows_semantic_sha256": A,
            "combined_promoted_rows_sha256": hashes[subject.COMBINED_ROWS_NAME],
            "combined_promoted_rows_semantic_sha256": B,
        },
        "disposition": {"same_protected_final_attempt_retry_permitted": False},
        "information_boundary": {"protected_salt_statted_or_read": False},
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources), "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    companions[subject.RECEIPT_NAME] = _raw(receipt)
    for name, raw in companions.items():
        (output / name).write_bytes(raw)
        (campaign / name).write_bytes(raw)
    return output / subject.RECEIPT_NAME, permanent, sha256(
        companions[subject.RECEIPT_NAME]).hexdigest()


def test_public_reader_never_loads_npz_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, permanent, checksum = _fixture(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("row members opened before public masks froze")

    monkeypatch.setattr(collection_api, "load_authenticated_rows", forbidden)
    value = subject.read_saved_closeout_public(
        receipt, expected_closeout_sha256=checksum,
        permanent_closeout_registry=permanent)
    assert value["closeout_key"] == C


def test_public_reader_rejects_changed_opaque_row_bytes(tmp_path: Path) -> None:
    receipt, permanent, checksum = _fixture(tmp_path)
    (receipt.parent / subject.COMBINED_ROWS_NAME).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Exact combined_promoted_rows"):
        subject.read_saved_closeout_public(
            receipt, expected_closeout_sha256=checksum,
            permanent_closeout_registry=permanent)


def test_public_reader_has_no_private_row_decoder() -> None:
    source = inspect.getsource(subject)
    assert "import numpy" not in source
    assert "np.load" not in source
    assert "zipfile" not in source
    assert "load_authenticated_rows" not in source
    assert "final_once" not in source

