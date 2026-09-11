from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from backend.training import warehouse_r41_final_rcpd as final
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor


ROOT = Path(__file__).parents[1]
ACTOR = (ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/"
         "actors/actor_0050000.npz")
MANIFEST = ROOT / "output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json"


def test_r41_extraction_partition_uses_disjoint_frozen_splits(monkeypatch):
    if not MANIFEST.is_file():
        pytest.skip("requires the private frozen r4.1 conflict manifest")
    manifest = json.loads(MANIFEST.read_text())
    # The manifest was already independently replayed at freeze time; this
    # unit test focuses on the extraction partition and avoids repeating the
    # 6,156-case closure audit.
    monkeypatch.setattr(
        "backend.training.warehouse_r41_conflict_scenarios.validate_conflict_manifest",
        lambda value, replay=True: {"passed": True},
    )
    train, validation = final._scenario_contract(manifest)
    assert len(train) == 70 and len(validation) == 30
    assert {row["fingerprint"] for row in train}.isdisjoint(
        {row["fingerprint"] for row in validation})


def test_r41_saved_rows_are_current_actor_labels_and_physically_replayable():
    if not ACTOR.is_file() or not MANIFEST.is_file():
        pytest.skip("requires private frozen Actor and conflict manifest")
    actor = NumPyNativeActor(ACTOR)
    scene = json.loads(MANIFEST.read_text())["splits"]["train"][0]
    rows = final._collect_pool(actor, [scene], pool="train", seed=final.TRAIN_SEED)
    final._validate_pool(rows, name="train", actor=actor, expected_scenes=[scene])
    final._replay_pool(rows, name="train", actor=actor,
                       expected_scenes=[scene], seed=final.TRAIN_SEED)
    changed = deepcopy(rows)
    changed["player_action_indices"][0] = (
        int(changed["player_action_indices"][0]) + 1
    ) % len(ACTIONS)
    with pytest.raises(ValueError, match="r4.1 trajectory"):
        final._replay_pool(changed, name="train", actor=actor,
                           expected_scenes=[scene], seed=final.TRAIN_SEED)


def test_base_specialization_is_restored_after_failure(monkeypatch):
    from backend.training import warehouse_r4_final_rcpd as base

    old_version = base.VERSION
    old_contract = base.contract
    monkeypatch.setattr(base, "extract", lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("stop")))
    with pytest.raises(RuntimeError, match="stop"):
        final.extract(actor_path="actor", scenarios_path="scenes", output="out")
    assert base.VERSION == old_version
    assert base.contract is old_contract
