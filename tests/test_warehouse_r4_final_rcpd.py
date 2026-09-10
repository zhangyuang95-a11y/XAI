from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r4_final_rcpd as final
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.policy import NumPyNativeActor


ROOT = Path(__file__).parents[1]
ACTOR = (ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/"
         "actors/actor_0050000.npz")
SCENARIOS = ROOT / "output/warehouse_native/v1-foundation-seed260908/scenarios.json"


def _evidence(actor, scenes, *, pool, frames=22):
    observations, episode_ids, fingerprints, partners = [], [], [], []
    frame_values, after, done = [], [], []
    for scene_index, scene in enumerate(scenes):
        for partner in final.PARTNERS:
            episode = f"{pool}:{scene['id']}:{scene['fingerprint']}:{partner}"
            for frame in range(frames):
                row = np.zeros(actor.obs_dim, dtype=np.float32)
                row[0] = scene_index / 100
                row[1] = final.PARTNERS.index(partner) / 10
                row[2] = frame / 120
                observations.append(row)
                episode_ids.append(episode)
                fingerprints.append(scene["fingerprint"])
                partners.append(partner)
                frame_values.append(frame); after.append(frame + 1); done.append(frame == frames - 1)
    observations = np.stack(observations)
    probabilities = final._softmax(actor.logits(observations))
    return {
        "observations": observations,
        "probabilities": probabilities,
        "episode_ids": np.asarray(episode_ids, dtype="U180"),
        "scenario_fingerprints": np.asarray(fingerprints, dtype="U64"),
        "partners": np.asarray(partners, dtype="U16"),
        "frames": np.asarray(frame_values, dtype=np.int16),
        "after_frames": np.asarray(after, dtype=np.int16),
        "done": np.asarray(done, dtype=np.bool_),
        "group_bits": np.full(len(observations), 7, dtype=np.uint8),
        "policy_action_indices": probabilities.argmax(-1).astype(np.uint8),
        "player_action_indices": np.zeros(len(observations), dtype=np.uint8),
        "submitted_equal": np.ones(len(observations), dtype=np.bool_),
        "source_state_sha256": np.asarray([
            sha256(f"state-{index}".encode()).hexdigest() for index in range(len(observations))
        ], dtype="U64"),
    }


def test_final_extraction_contract_has_zero_training_and_complete_grid():
    value = final.contract()
    assert value["ppo_joint_steps"] == value["optimizer_updates"] == 0
    assert value["program_feedback_into_actor"] is False
    assert value["candidate_depths"][:3] == [4, 6, 8]
    assert value["candidate_leaves"][:3] == [16, 32, 64]
    assert len(final.DEPTHS) * len(final.LEAVES) == 25


def test_registered_extraction_partition_is_scene_disjoint():
    scenarios = json.loads(SCENARIOS.read_text())
    train, validation = final._scenario_contract(scenarios)
    assert len(train) == 70 and len(validation) == 30
    assert {row["fingerprint"] for row in train}.isdisjoint(
        {row["fingerprint"] for row in validation})


def test_saved_pool_labels_must_come_from_supplied_actor():
    actor = NumPyNativeActor(ACTOR)
    scenarios = json.loads(SCENARIOS.read_text())["splits"]["extraction"][:2]
    value = _evidence(actor, scenarios, pool="train")
    final._validate_pool(value, name="train", actor=actor, expected_scenes=scenarios)
    changed = deepcopy(value)
    changed["probabilities"][0] = np.roll(changed["probabilities"][0], 1)
    with pytest.raises(ValueError, match="supplied frozen Actor"):
        final._validate_pool(changed, name="train", actor=actor, expected_scenes=scenarios)


def test_saved_pool_replays_registered_physical_trajectory():
    actor = NumPyNativeActor(ACTOR)
    scene = json.loads(SCENARIOS.read_text())["splits"]["extraction"][0]
    value = final._collect_pool(actor, [scene], pool="train", seed=final.TRAIN_SEED)
    final._validate_pool(value, name="train", actor=actor, expected_scenes=[scene])
    final._replay_pool(value, name="train", actor=actor,
                       expected_scenes=[scene], seed=final.TRAIN_SEED)
    changed = deepcopy(value)
    changed["player_action_indices"][0] = (
        int(changed["player_action_indices"][0]) + 1) % len(ACTIONS)
    with pytest.raises(ValueError, match="registered physical trajectory"):
        final._replay_pool(changed, name="train", actor=actor,
                           expected_scenes=[scene], seed=final.TRAIN_SEED)


def test_selection_uses_simplest_eligible_candidate_only():
    def candidate(loss, fidelity, passed, depth, leaves):
        return {"complexity": {"loss": loss}, "metrics": {
            "mean_kl": .1, "overall": {"fidelity": fidelity}},
            "gate": {"passed": passed}, "depth_cap": depth, "leaf_cap": leaves}
    values = [candidate(.4, .99, True, 8, 64), candidate(.2, .91, True, 6, 32),
              candidate(.1, .89, False, 4, 16)]
    assert final._select(values) is values[1]
    assert final._select([values[2]]) is None


def test_json_reader_rejects_duplicate_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        final._read_json(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        final._read_json(nonfinite)
