from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_outer_attempt_closeout_v10 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _content(value: dict) -> dict:
    value["content_sha256"] = digest(value)
    return value


def _identities() -> list[dict]:
    result = []
    index = 0
    for family, count in subject.FAMILY_COUNTS.items():
        for _ in range(count):
            result.append({
                "batch_index": index % 3,
                "family_id": family,
                "seed": 900_000 + index,
                "fingerprint": _fp(f"scene-{index}"),
            })
            index += 1
    return result


def test_contract_closes_attempt_without_opening_row_members():
    contract = subject.contract()
    assert contract["consumed_scene_count"] == 64
    assert contract["outer_identity_reuse_permitted"] is False
    assert contract["rows_opened"] is False
    assert contract["observations_actions_probabilities_or_labels_read"] is False
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "np.load" not in source
    assert "zipfile" not in source
    assert "fresh_final_holdout" not in source


def test_registry_identity_bundle_is_exact_and_family_balanced():
    identities = _identities()
    scenes = [{
        "id": f"diagnostic_v9_fresh_outer_{index:04d}",
        **row,
    } for index, row in enumerate(identities)]
    registry = _content({
        "version": subject.V9_REGISTRY_VERSION,
        "status": subject.V9_REGISTRY_STATUS,
        "selected_outer_identities": deepcopy(identities),
        "development_outer": scenes,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "formal_ready": False,
        "producer_sources": {"f.py": "a" * 64},
        "producer_sources_sha256": digest({"f.py": "a" * 64}),
    })
    report = _content({
        "version": subject.V9_REGISTRY_VERSION,
        "status": subject.V9_REGISTRY_STATUS,
        "registry_file_sha256": "b" * 64,
        "registry_content_sha256": registry["content_sha256"],
        "bindings": None,
        "statistics": None,
        "information_boundary": None,
        "selection": {"selected_identity_sha256": digest(identities)},
        "producer_sources": deepcopy(registry["producer_sources"]),
        "producer_sources_sha256": registry["producer_sources_sha256"],
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


def test_permanent_failure_pair_and_retry_false_are_required(
        tmp_path, monkeypatch):
    identities = _identities()
    projected_hashes = sorted(_fp(f"obs-{index}") for index in range(30776))
    permanent = tmp_path / "permanent"
    campaign_key = _fp("campaign")
    campaign = permanent / campaign_key
    campaign.mkdir(parents=True)
    anchor_path = campaign / "attempt_anchor.json"
    result_path = campaign / "outer_result.json"
    anchor_path.write_text("anchor\n", encoding="ascii")
    result_path.write_text("result\n", encoding="ascii")
    monkeypatch.setattr(subject, "V9_ATTEMPT_KEY", campaign_key)
    monkeypatch.setattr(subject, "EXPECTED_V9_ATTEMPT_ANCHOR_SHA256",
                        file_hash(anchor_path))
    monkeypatch.setattr(subject, "EXPECTED_V9_ATTEMPT_RESULT_SHA256",
                        file_hash(result_path))
    sources = {"closeout.py": "c" * 64}
    receipt = _content({
        "version": subject.VERSION,
        "status": subject.STATUS,
        "attempt_key": campaign_key,
        "contract": subject.contract(),
        "bindings": {
            "v9_attempt_anchor_sha256": file_hash(anchor_path),
            "v9_attempt_result_sha256": file_hash(result_path),
        },
        "consumed_outer": {
            "scene_count": 64,
            "identities": identities,
            "identities_sha256": digest(identities),
            "observation_hash_projection": {
                "source_projection_sha256": (
                    subject.EXPECTED_V9_PROJECTION_SHA256),
                "outer_observation_hashes": projected_hashes,
                "unique_outer_observation_hash_count": len(projected_hashes),
                "outer_observation_hashes_sha256": digest(projected_hashes),
                "raw_observations_included": False,
                "actions_included": False,
                "probabilities_included": False,
                "labels_included": False,
            },
        },
        "failure": {
            "outer_consumed": True,
            "retry_permitted": False,
            "metrics_produced": False,
        },
        "disposition": {
            "eligible_for_outer_claim": False,
            "outer_identity_reuse_permitted": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    })
    closeout_path = tmp_path / "closeout_receipt.json"
    closeout_path.write_text(canonical(receipt) + "\n", encoding="utf-8")
    saved = subject.read_saved_closeout(
        closeout_path, expected_closeout_sha256=file_hash(closeout_path),
        permanent_attempt_registry=permanent)
    assert saved["consumed_outer"]["identities_sha256"] == digest(identities)

    forged = deepcopy(receipt)
    forged["failure"]["retry_permitted"] = True
    forged["content_sha256"] = digest({
        key: value for key, value in forged.items()
        if key != "content_sha256"})
    forged_path = tmp_path / "forged.json"
    forged_path.write_text(canonical(forged) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="semantics differ"):
        subject.read_saved_closeout(
            forged_path,
            expected_closeout_sha256=file_hash(forged_path),
            permanent_attempt_registry=permanent)
