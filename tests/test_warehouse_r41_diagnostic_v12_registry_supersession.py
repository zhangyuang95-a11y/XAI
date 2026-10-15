from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_v12_registry_supersession as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _save(path: Path, value: dict) -> str:
    value["content_sha256"] = digest(value)
    path.write_text(canonical(value) + "\n", encoding="utf-8")
    return file_hash(path)


def _fixture(tmp_path: Path) -> dict:
    identities = [{"seed": index} for index in range(64)]
    selected = digest(identities)
    common_digests = {"consumed_v11_outer_identities_sha256": "a" * 64}
    historical = "b" * 64
    old_registry = {
        "selected_outer_identities": identities,
        "exclusion_digests": deepcopy(common_digests),
        "bindings": {
            "consumed_v10_outer_observation_hashes_sha256": historical},
    }
    new_registry = deepcopy(old_registry)
    new_registry["exclusion_digests"][
        "consumed_v10_outer_observation_hashes_sha256"] = historical
    old_registry_path = tmp_path / "old_registry.json"
    new_registry_path = tmp_path / "new_registry.json"
    old_registry_sha = _save(old_registry_path, old_registry)
    new_registry_sha = _save(new_registry_path, new_registry)
    old_report_path = tmp_path / "old_report.json"
    new_report_path = tmp_path / "new_report.json"
    old_report_sha = _save(old_report_path, {
        "registry_file_sha256": old_registry_sha,
        "registry_content_sha256": old_registry["content_sha256"],
        "selection": {"selected_identity_sha256": selected},
    })
    new_report_sha = _save(new_report_path, {
        "registry_file_sha256": new_registry_sha,
        "registry_content_sha256": new_registry["content_sha256"],
        "selection": {"selected_identity_sha256": selected},
    })
    old_projection_path = tmp_path / "old_projection.json"
    new_projection_path = tmp_path / "new_projection.json"
    projection = {
        "outer_observation_hashes": ["c" * 64],
        "outer_observation_hashes_sha256": digest(["c" * 64]),
    }
    old_projection_sha = _save(old_projection_path, deepcopy(projection))
    new_projection_sha = _save(new_projection_path, deepcopy(projection))
    return {
        "old_registry_path": old_registry_path,
        "old_registry_sha256": old_registry_sha,
        "old_report_path": old_report_path,
        "old_report_sha256": old_report_sha,
        "old_prior_projection_path": old_projection_path,
        "old_prior_projection_sha256": old_projection_sha,
        "new_registry_path": new_registry_path,
        "new_registry_sha256": new_registry_sha,
        "new_report_path": new_report_path,
        "new_report_sha256": new_report_sha,
        "new_prior_projection_path": new_projection_path,
        "new_prior_projection_sha256": new_projection_sha,
        "failed_projection_output": tmp_path / "never-published",
    }


def test_supersession_requires_same_identities_and_only_missing_digest(tmp_path):
    kwargs = _fixture(tmp_path)
    value = subject.create_receipt(**kwargs)
    assert value["identity_audit"]["selected_identities_exactly_equal"] is True
    assert value["failure_boundary"]["fresh_projection_replay_started"] is False
    assert value["failure_boundary"]["labels_or_probabilities_accessed"] is False

    changed = Path(kwargs["new_registry_path"])
    payload = __import__("json").loads(changed.read_text())
    payload["selected_outer_identities"][0]["seed"] = 99
    kwargs["new_registry_sha256"] = _save(changed, {
        key: value for key, value in payload.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="source-closure-only"):
        subject.create_receipt(**kwargs)


def test_contract_forbids_consumption_or_secret_access():
    value = subject.contract()
    assert value["selected_identity_change_permitted"] is False
    assert value["labels_or_probabilities_accessed"] is False
    assert value["outer_attempt_claimed"] is False
    assert value["protected_final_access"] is False
    assert value["private_salt_access"] is False


def test_saved_supersession_is_sha_and_source_authenticated(tmp_path):
    kwargs = _fixture(tmp_path)
    output = tmp_path / "receipt"
    built = subject.build(output=output, **kwargs)
    saved = subject.read_saved_receipt(
        output / subject.RECEIPT_NAME,
        expected_receipt_sha256=file_hash(output / subject.RECEIPT_NAME))
    assert saved == built
    assert saved["failure_boundary"][
        "registry_validation_precedes_fresh_replay"] is True
    with pytest.raises(ValueError, match="Exact v12 registry supersession"):
        subject.read_saved_receipt(
            output / subject.RECEIPT_NAME,
            expected_receipt_sha256="0" * 64)
