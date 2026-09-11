from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training.warehouse_r41_diagnostic_conflict_scenarios import make_scene
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256,
    FROZEN_ACTOR_SHA256,
    load_frozen_actor,
    screen_scene,
    validate_receipt,
)
from backend.training.warehouse_native_common import digest


ROOT = Path(__file__).resolve().parents[1]
ACTOR = (
    ROOT
    / "output/warehouse_native/r41_active_2m_20260911/boundaries/step_2000000/actor.npz"
)
UNSCREENED = (
    ROOT
    / "output/warehouse_native/r41_diagnostic_conflict_scenes_v1_20260911/manifest.json"
)
PRIVATE_ACTOR_REQUIRED = pytest.mark.skipif(
    not ACTOR.is_file(),
    reason="frozen private Actor is supplied outside the public checkout",
)
PRIVATE_V1_MANIFEST_REQUIRED = pytest.mark.skipif(
    not UNSCREENED.is_file(),
    reason="archived private v1 manifest is supplied outside the public checkout",
)


@PRIVATE_ACTOR_REQUIRED
@PRIVATE_V1_MANIFEST_REQUIRED
def test_obsolete_v1_scene_is_rejected_before_workload_replay():
    actor = load_frozen_actor(ACTOR)
    assert actor.artifact_sha256 == FROZEN_ACTOR_SHA256
    manifest = json.loads(UNSCREENED.read_text(encoding="utf-8"))
    scene = manifest["splits"]["train"][4]
    with pytest.raises(ValueError, match="Diagnostic scene contract binding differs"):
        screen_scene(scene, split="train", scene_index=4, actor=actor)


def test_obsolete_v1_workload_receipt_is_rejected():
    scene = {"fingerprint": "current-scene"}
    receipt = {
        "version": "warehouse-r41-diagnostic-workload-screen.v1",
        "contract_sha256": CONTRACT_SHA256,
        "frozen_actor_sha256": None,
        "scene_fingerprint": scene["fingerprint"],
        "split": "tutorial",
        "scene_index": 0,
        "passed": True,
        "failure": None,
        "metrics": {
            "actor_action_override_frames": 0,
            "new_pickup_on_agent": 0,
            "new_delivery_on_agent": 0,
            "new_endpoint_on_agent": 0,
            "immediate_task_recreation": 0,
        },
    }
    receipt["receipt_sha256"] = digest(receipt)
    with pytest.raises(ValueError, match="workload receipt differs"):
        validate_receipt(receipt, scene=scene, split="tutorial", scene_index=0)


@PRIVATE_ACTOR_REQUIRED
def test_robust_v2_scene_completes_exact_question_bank_workload():
    actor = load_frozen_actor(ACTOR)
    scene = make_scene("workload_v2_question", "question_bank", 49_500_000)
    receipt = screen_scene(
        scene, split="question_bank", scene_index=0, actor=actor
    )
    assert receipt["passed"] is True
    assert receipt["failure"] is None
    assert receipt["metrics"]["actor_action_override_frames"] == 0
    assert receipt["metrics"]["new_pickup_on_agent"] == 0
    assert receipt["metrics"]["new_delivery_on_agent"] == 0
    assert receipt["metrics"]["new_endpoint_on_agent"] == 0
    assert receipt["metrics"]["immediate_task_recreation"] == 0
