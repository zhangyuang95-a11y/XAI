from __future__ import annotations

from copy import deepcopy

import pytest

from backend.training.warehouse_r41_diagnostic_conflict_scenarios import make_scene
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial


def _payload(monkeypatch):
    scene = make_scene("diagnostic_tutorial", "tutorial", 4_410_000)
    manifest = {
        "version": "warehouse-r41-diagnostic-conflict-scene-manifest.v3",
        "diagnostic_contract_version": "warehouse-r41-diagnostic-conflict.v2",
        "content_sha256": "a" * 64,
        "splits": {"tutorial": [scene]},
    }
    monkeypatch.setattr(
        "backend.training.warehouse_r41_diagnostic_conflict_scenarios."
        "validate_diagnostic_manifest",
        lambda *args, **kwargs: {"passed": True},
    )
    payload = tutorial.build_neutral_tutorial(
        manifest, manifest_file_sha256="b" * 64,
    )
    return scene, payload


def test_diagnostic_tutorial_physically_covers_rules_without_actor(monkeypatch):
    scene, payload = _payload(monkeypatch)
    result = tutorial.validate_neutral_tutorial(payload, tutorial_scene=scene)
    assert result["passed"] is True
    assert payload["version"] == "warehouse-alignment-diagnostic-neutral-tutorial.v3"
    assert payload["uses_final_actor"] is False
    assert payload["duration_ms"] == 380
    assert len(payload["frames"]) == 14
    assert all(payload["coverage"][name] for name in (
        "pickup_frames", "delivery_frames", "simultaneous_movement_frames",
        "collision_frames", "wait_frames", "charge_frames",
    ))
    assert set(payload["bindings"]) == tutorial._BINDING_FIELDS
    assert payload["bindings"]["scene_manifest_version"] == (
        "warehouse-r41-diagnostic-conflict-scene-manifest.v3"
    )


def test_diagnostic_tutorial_rejects_fabricated_frame(monkeypatch):
    scene, payload = _payload(monkeypatch)
    changed = deepcopy(payload)
    changed["frames"][5]["state"]["frame"] = 99
    with pytest.raises(ValueError, match="fabricated"):
        tutorial.validate_neutral_tutorial(changed, tutorial_scene=scene)


def test_diagnostic_tutorial_rejects_old_manifest_identity(monkeypatch):
    scene, payload = _payload(monkeypatch)
    changed = deepcopy(payload)
    changed["bindings"]["scene_manifest_version"] = (
        "warehouse-r41-diagnostic-conflict-scene-manifest.v2"
    )
    with pytest.raises(ValueError, match="manifest version"):
        tutorial.validate_neutral_tutorial(changed, tutorial_scene=scene)


@pytest.mark.parametrize(("field", "value", "message"), (
    ("diagnostic_contract_version", "warehouse-r41-diagnostic-conflict.v1",
     "contract version"),
    ("diagnostic_conflict_graph_sha256", "0" * 64,
     "scene or source binding"),
))
def test_diagnostic_tutorial_rejects_changed_v3_contract(
        monkeypatch, field, value, message):
    scene, payload = _payload(monkeypatch)
    changed = deepcopy(payload)
    changed["bindings"][field] = value
    with pytest.raises(ValueError, match=message):
        tutorial.validate_neutral_tutorial(changed, tutorial_scene=scene)


def test_diagnostic_tutorial_sources_bind_diagnostic_runtime_and_environment():
    sources = tutorial.producer_sources()
    for name in (
        "ui/warehouse_alignment_r41_diagnostic_tutorial.py",
        "backend/warehouse_r41_diagnostic_online_runtime.py",
        "env/warehouse_native/r41_diagnostic_conflict.py",
    ):
        assert name in sources
