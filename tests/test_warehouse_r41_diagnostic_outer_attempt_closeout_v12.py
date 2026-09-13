from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_outer_attempt_closeout_v12 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _content(value: dict) -> dict:
    value["content_sha256"] = digest(value)
    return value


def _identities() -> list[dict]:
    rows = []
    index = 0
    for family, count in subject.FAMILY_COUNTS.items():
        for _ in range(count):
            rows.append({
                "batch_index": index % 3,
                "family_id": family,
                "seed": 48_100_000 + index,
                "fingerprint": _fp(f"v11-scene-{index}"),
            })
            index += 1
    return rows


def test_contract_closes_scored_failure_without_opening_private_inputs():
    value = subject.contract()
    assert value["consumed_outer_is_development_exposed"] is True
    assert value["promoted_rows_require_exact_file_and_semantic_sha256"] is True
    assert value["outer_identity_reuse_permitted"] is False
    assert value["row_archive_members_opened"] is False
    assert value["protected_final_access"] is False
    assert value["private_salt_access"] is False
    source = inspect.getsource(subject)
    assert "np.load" not in source
    assert "zipfile" not in source


def test_registry_identity_bundle_is_exact_and_family_balanced():
    identities = _identities()
    scenes = [{
        "id": f"diagnostic_v11_fresh_outer_{index:04d}",
        **identity,
    } for index, identity in enumerate(identities)]
    sources = {"producer.py": "a" * 64}
    registry = _content({
        "version": subject.REGISTRY_VERSION,
        "status": subject.REGISTRY_STATUS,
        "selected_outer_identities": deepcopy(identities),
        "development_outer": scenes,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    })
    report = _content({
        "version": subject.REGISTRY_VERSION,
        "status": subject.REGISTRY_STATUS,
        "registry_file_sha256": "b" * 64,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": None,
        "statistics": None,
        "information_boundary": None,
        "selection": {"selected_identity_sha256": digest(identities)},
        "producer_sources": sources,
        "formal_ready": False,
    })
    assert subject._registry_identities(
        registry, report, registry_sha256="b" * 64) == identities
    registry["development_outer"][0]["seed"] += 1
    registry["content_sha256"] = digest({
        key: value for key, value in registry.items()
        if key != "content_sha256"})
    with pytest.raises(ValueError, match="materialised identity"):
        subject._registry_identities(
            registry, report, registry_sha256="b" * 64)


def test_saved_closeout_requires_promotable_semantic_hash_and_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    identities = _identities()
    hashes = sorted(_fp(f"obs-{index}") for index in range(5))
    attempt_key = _fp("attempt")
    permanent = tmp_path / "permanent"
    campaign = permanent / attempt_key
    campaign.mkdir(parents=True)
    anchor = campaign / "attempt_anchor.json"
    result = campaign / "outer_result.json"
    anchor.write_text("anchor\n", encoding="ascii")
    result.write_text("result\n", encoding="ascii")
    monkeypatch.setattr(subject, "ATTEMPT_KEY", attempt_key)
    monkeypatch.setattr(subject, "EXPECTED_ATTEMPT_ANCHOR_SHA256", file_hash(anchor))
    monkeypatch.setattr(subject, "EXPECTED_ATTEMPT_RESULT_SHA256", file_hash(result))
    monkeypatch.setattr(subject, "UNIQUE_OBSERVATION_COUNT", len(hashes))
    sources = {"closeout.py": "c" * 64}
    value = _content({
        "version": subject.VERSION,
        "status": subject.STATUS,
        "attempt_key": attempt_key,
        "contract": subject.contract(),
        "bindings": {
            "v11_attempt_anchor_sha256": file_hash(anchor),
            "v11_attempt_result_sha256": file_hash(result),
            "v11_projection_content_sha256": "d" * 64,
        },
        "consumed_outer": {
            "scene_count": subject.SCENE_COUNT,
            "identities": identities,
            "identities_sha256": digest(identities),
            "row_count": subject.ROW_COUNT,
            "rows_sha256": subject.EXPECTED_COLLECTION_ROWS_SHA256,
            "rows_semantic_sha256": (
                subject.EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256),
            "observation_hash_projection": {
                "source_projection_sha256": subject.EXPECTED_PROJECTION_SHA256,
                "source_projection_content_sha256": "d" * 64,
                "outer_observation_hashes": hashes,
                "unique_outer_observation_hash_count": len(hashes),
                "outer_observation_hashes_sha256": digest(hashes),
                "raw_observations_included": False,
                "actions_included": False,
                "probabilities_included": False,
                "labels_included": False,
            },
        },
        "failure": {
            "outer_consumed": True,
            "retry_permitted": False,
            "metrics_produced": True,
            "gate_passed": False,
            "failed_checks": ["effective_intervention_direction_shared_charger"],
        },
        "disposition": {
            "eligible_for_development_use": True,
            "eligible_for_outer_claim": False,
            "outer_identity_reuse_permitted": False,
        },
        "information_boundary": {
            "row_archive_members_opened": False,
            "protected_final_access": False,
            "private_salt_access": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    })
    closeout = tmp_path / "closeout.json"
    closeout.write_text(canonical(value) + "\n", encoding="utf-8")
    saved = subject.read_saved_closeout(
        closeout, expected_closeout_sha256=file_hash(closeout),
        permanent_attempt_registry=permanent)
    assert saved["consumed_outer"]["rows_semantic_sha256"] == (
        subject.EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256)

    forged = deepcopy(value)
    forged["failure"]["retry_permitted"] = True
    forged["content_sha256"] = digest({
        key: child for key, child in forged.items()
        if key != "content_sha256"})
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(canonical(forged) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantics differ"):
        subject.read_saved_closeout(
            forged_path, expected_closeout_sha256=file_hash(forged_path),
            permanent_attempt_registry=permanent)

