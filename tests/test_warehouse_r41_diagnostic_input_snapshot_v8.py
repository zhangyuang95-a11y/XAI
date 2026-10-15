from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_designation_v2 as designation
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as binding
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as subject


ROOT = Path(__file__).resolve().parents[1]


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def test_snapshot_semantics_are_immune_to_original_a_b_a_replacement(tmp_path):
    original = tmp_path / "input.json"
    original.write_bytes(b'{"value":"A"}\n')
    expected = _sha(original.read_bytes())
    with subject.ImmutableInputSnapshot(
        {"input": original}, expected_sha256={"input": expected},
    ) as frozen:
        original.write_bytes(b'{"value":"B"}\n')
        assert frozen.paths["input"].read_bytes() == b'{"value":"A"}\n'
        original.write_bytes(b'{"value":"A"}\n')
        frozen.verify()


def test_snapshot_rejects_changed_original_and_never_changes_frozen_copy(tmp_path):
    original = tmp_path / "rows.npz"
    original.write_bytes(b"A-row-archive")
    expected = _sha(original.read_bytes())
    with subject.ImmutableInputSnapshot(
        {"rows": original}, expected_sha256={"rows": expected},
    ) as frozen:
        original.write_bytes(b"B-row-archive")
        assert frozen.paths["rows"].read_bytes() == b"A-row-archive"
        with pytest.raises(RuntimeError, match="fixed inputs changed"):
            frozen.verify()


def test_snapshot_requires_exact_bytes_before_semantic_use(tmp_path):
    original = tmp_path / "actor.npz"
    original.write_bytes(b"attacker")
    with pytest.raises(ValueError, match="Exact actor bytes"):
        subject.ImmutableInputSnapshot(
            {"actor": original},
            expected_sha256={"actor": _sha(b"frozen")},
        )


def test_snapshot_preserves_required_manifest_sibling_without_aliasing(tmp_path):
    manifest = tmp_path / "manifest.json"
    validation = tmp_path / "validation.json"
    manifest.write_bytes(b"manifest")
    validation.write_bytes(b"validation")
    with subject.ImmutableInputSnapshot(
        {"manifest": manifest, "validation": validation},
        expected_sha256={
            "manifest": _sha(b"manifest"),
            "validation": _sha(b"validation"),
        },
        relative_names={
            "manifest": "frozen/manifest.json",
            "validation": "frozen/validation.json",
        },
    ) as frozen:
        assert frozen.paths["manifest"].parent == frozen.paths["validation"].parent
        assert frozen.paths["manifest"].parent != tmp_path
        assert frozen.paths["manifest"].read_bytes() == b"manifest"
        assert frozen.paths["validation"].read_bytes() == b"validation"


def test_cycle_free_designation_authenticates_from_one_component_snapshot():
    designation_path = ROOT / (
            "output/warehouse_native/r41_diagnostic_designation_v2_sourceclosure2_20260913/"
        "diagnostic_actor_designation.json")
    raw = subject.read_authenticated_bytes(
        designation_path, label="designation",
        expected_sha256=binding.EXPECTED_DESIGNATION_SHA256)
    components = binding.resolve_bound_components_from_bytes(
        raw, original_path=designation_path)
    paths = {"designation": designation_path}
    paths.update({"designation_" + name: path
                  for name, path in components.items()})
    expected = {
        "designation": binding.EXPECTED_DESIGNATION_SHA256,
        "designation_actor": designation.EXPECTED_ACTOR_SHA256,
        "designation_protocol": designation.EXPECTED_PROTOCOL_FILE_SHA256,
        "designation_training_ledger": designation.EXPECTED_LEDGER_SHA256,
        "designation_dual_evaluation": designation.EXPECTED_DUAL_EVALUATION_SHA256,
        "designation_failure_closeout": designation.EXPECTED_CLOSEOUT_SHA256,
    }
    with subject.ImmutableInputSnapshot(
        paths, expected_sha256=expected,
    ) as frozen:
        saved = binding.read_bound_designation_snapshot(
            frozen.paths["designation"], original_path=designation_path,
            components={name: frozen.paths["designation_" + name]
                        for name in components},
            original_components=components,
        )
        assert saved["bindings"]["actor_sha256"] \
            == designation.EXPECTED_ACTOR_SHA256
        frozen.verify()


def test_public_designation_reader_uses_the_immutable_snapshot_path():
    designation_path = ROOT / (
            "output/warehouse_native/r41_diagnostic_designation_v2_sourceclosure2_20260913/"
        "diagnostic_actor_designation.json")

    saved = binding.read_bound_designation(designation_path)

    assert saved["bindings"]["actor_sha256"] \
        == designation.EXPECTED_ACTOR_SHA256


def test_designation_hash_cannot_be_rebound_by_a_caller(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="cannot be overridden"):
        binding.resolve_bound_components(
            missing, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="cannot be overridden"):
        binding.read_bound_designation(
            missing, expected_sha256="0" * 64)
