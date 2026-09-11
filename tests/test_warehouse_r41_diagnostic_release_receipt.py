from __future__ import annotations

from hashlib import sha256
import json

import pytest

from backend.training import warehouse_r41_diagnostic_release_receipt as receipt
from backend.training.warehouse_native_common import canonical, file_hash
from ui import warehouse_alignment_r41_diagnostic_release as release


def _rollback(root):
    folder = root / "rollback"; folder.mkdir()
    files = {
        "render.r3.yaml": b"services: []\n",
        "warehouse_alignment_online.zip": b"old zip",
        "warehouse_alignment_release.b64": b"b2xkIHppcA==\n",
    }
    registry = {}
    for name, raw in files.items():
        path = folder / name; path.write_bytes(raw); path.chmod(0o600)
        registry[name] = {"sha256": sha256(raw).hexdigest(), "size": len(raw)}
    value = {
        "version": receipt.EXPECTED_ROLLBACK_VERSION,
        "status": receipt.EXPECTED_ROLLBACK_STATUS,
        "public_url": "https://policylens-warehouse-study.onrender.com",
        "files": registry,
    }
    path = folder / "rollback_receipt.json"
    path.write_text(canonical(value) + "\n"); path.chmod(0o600)
    return path


def test_diagnostic_receipt_is_nonformal_ephemeral_and_rollback_bound(
        monkeypatch, tmp_path):
    monkeypatch.setattr(receipt, "ROOT", tmp_path.resolve())
    rollback = _rollback(tmp_path)
    monkeypatch.setattr(
        receipt, "EXPECTED_ROLLBACK_RECEIPT_SHA256", file_hash(rollback))
    value = {name: "0" * 64 for name in receipt.FIELDS}
    value.update({
        "version": receipt.VERSION, "status": receipt.STATUS,
        "release_version": release.PUBLIC_RELEASE_VERSION,
        "pilot_class": release.PILOT_CLASS,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "selected_scene_fingerprints": [f"{index + 1:064x}" for index in range(7)],
        "rollback_receipt_path": "rollback/rollback_receipt.json",
        "rollback_receipt_sha256": file_hash(rollback),
        "self_path": "release_receipt.json",
    })
    path = tmp_path / "release_receipt.json"
    path.write_text(canonical(value) + "\n"); path.chmod(0o600)
    assert receipt.read_saved_receipt(
        path, expected_sha256=file_hash(path)) == value
    changed = dict(value); changed["data_persistent"] = True
    path.write_text(canonical(changed) + "\n")
    with pytest.raises(ValueError, match="Exact non-formal"):
        receipt.read_saved_receipt(path, expected_sha256=file_hash(path))


def test_real_saved_r3_rollback_is_accepted_when_present():
    path = receipt.ROOT / (
        "output/warehouse_native/r41_diagnostic_r3_rollback_20260911_130056/"
        "rollback_receipt.json")
    if not path.is_file():
        pytest.skip("local r3 rollback was not retained")
    receipt._validate_rollback(
        path,
        receipt.EXPECTED_ROLLBACK_RECEIPT_SHA256,
    )


def test_rollback_validator_rejects_an_alternate_receipt_even_if_self_bound(
        monkeypatch, tmp_path):
    monkeypatch.setattr(receipt, "ROOT", tmp_path.resolve())
    alternate = _rollback(tmp_path)
    alternate_sha = file_hash(alternate)
    assert alternate_sha != receipt.EXPECTED_ROLLBACK_RECEIPT_SHA256
    with pytest.raises(ValueError, match="Exact saved r3 rollback receipt SHA-256"):
        receipt._validate_rollback(alternate, alternate_sha)
