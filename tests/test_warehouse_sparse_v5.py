"""Contract tests for the bounded Warehouse sparse-oblique program."""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from backend.training.warehouse_sparse_v5 import (
    MAX_DEPTH, MAX_FEATURES_PER_NODE, MAX_LEAVES,
    SelectiveLeafProgram, audit_frozen_program, fit_selective_program,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks, observation_feature_names, observations


ACTOR_HASH = "a" * 64
OBS_NAMES = observation_feature_names()


@pytest.fixture(scope="module")
def episode_rows():
    rows = {role: {} for role in AGENTS}
    for seed in range(100, 114):
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=seed)
        state = env.get_state()
        state.participant_controlled_agent_id = "robot_1"
        env.set_state(state)
        rng = np.random.default_rng(seed)
        for _ in range(24):
            if env.state.terminated or env.state.truncated:
                break
            obs, masks = observations(env.state), action_masks(env.state)
            for index, role in enumerate(AGENTS):
                rows[role].setdefault(str(seed), []).append({
                    "obs": obs[index].copy(), "mask": masks[index].copy(),
                    "episode": str(seed),
                })
            actions = {role: ACTIONS[int(rng.choice(np.flatnonzero(masks[index] > 0.5)))]
                       for index, role in enumerate(AGENTS)}
            env.step(actions)
    return rows


def _parts(rows):
    seeds = [str(seed) for seed in range(100, 112)]
    return [[row for seed in group for row in rows[seed]]
            for group in (seeds[:3], seeds[3:6], seeds[6:9], seeds[9:12])]


def _masked_oracle(row):
    legal = np.asarray(row["mask"], dtype=np.float64) > 0.5
    assert legal[ACTIONS.index("WAIT")]
    distribution = np.zeros(len(ACTIONS), dtype=np.float64)
    distribution[ACTIONS.index("WAIT")] = 1.0
    return distribution


def _variable_oracle(row):
    features = dict(zip(OBS_NAMES, row["obs"], strict=True))
    row_value = float(features["self_row"])
    column_value = float(features["self_column"])
    logits = np.array([2.0 - 3.0 * row_value, 3.0 * row_value,
                       2.0 - 3.0 * column_value, 3.0 * column_value, 0.5])
    logits[np.asarray(row["mask"]) <= 0.5] = -1e9
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


@pytest.fixture(scope="module")
def constant_program(episode_rows):
    return fit_selective_program(
        "robot_2", *_parts(episode_rows["robot_2"]), _masked_oracle,
        actor_role_sha256=ACTOR_HASH, random_seed=7)


@pytest.fixture(scope="module")
def variable_program(episode_rows):
    return fit_selective_program(
        "robot_1", *_parts(episode_rows["robot_1"]), _variable_oracle,
        actor_role_sha256=ACTOR_HASH, random_seed=7)


def test_sparse_nodes_are_bounded_and_observable(variable_program, episode_rows):
    program = variable_program
    assert program.report["max_depth"] <= MAX_DEPTH
    assert program.report["leaf_count"] <= MAX_LEAVES
    assert program.report["max_features_per_node"] == MAX_FEATURES_PER_NODE
    assert program.projections
    assert all(1 <= len(item["features"]) <= MAX_FEATURES_PER_NODE
               for item in program.projections)
    sample = episode_rows["robot_1"]["100"][0]
    with_hidden = {**sample, "hidden_state": {"goal": "different"},
                   "post_action_reward": 9999, "teacher_action": "WAIT"}
    assert program.lookup(sample) == program.lookup(with_hidden)
    assert all("terms" in node and len(node["terms"]) <= MAX_FEATURES_PER_NODE
               for node in program.lookup(sample)["path"])


def test_episode_splits_reject_overlap(episode_rows):
    fit, calibration, eligibility, validation = _parts(episode_rows["robot_1"])
    with pytest.raises(ValueError, match="overlap"):
        fit_selective_program(
            "robot_1", fit, [*calibration, fit[0]], eligibility, validation,
            _variable_oracle, actor_role_sha256=ACTOR_HASH)


def test_serialization_hash_and_sparse_bounds(variable_program, tmp_path):
    path = tmp_path / "sparse.json"
    digest = variable_program.save(path)
    restored = SelectiveLeafProgram.load(
        path, expected_sha256=digest, actor_role_sha256=ACTOR_HASH)
    assert restored.sha256() == digest
    assert restored.to_dict() == variable_program.to_dict()
    invalid = deepcopy(variable_program.to_dict())
    invalid["projections"][0]["features"] = [0, 1, 2, 3]
    invalid["projections"][0]["coefficients"] = [1.0] * 4
    with pytest.raises(RuntimeError, match="feature or coefficient bounds"):
        SelectiveLeafProgram.from_dict(invalid, actor_role_sha256=ACTOR_HASH)
    content = json.loads(path.read_text())
    content["role"] = "robot_2"
    path.write_text(json.dumps(content))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        SelectiveLeafProgram.load(
            path, expected_sha256=digest, actor_role_sha256=ACTOR_HASH)


def test_covered_requires_actor_agreement_and_physical_legality(
        constant_program, episode_rows):
    sample = episode_rows["robot_2"]["100"][0]
    assert constant_program.report["validation"]["explanation_online_passed"]
    evidence = constant_program.covered(
        sample, actor_action="WAIT", purpose="explanation")
    assert evidence is not None and evidence["action"] == "WAIT"
    assert constant_program.covered(
        sample, actor_action="UP", purpose="explanation") is None
    invalid_wait = {**sample, "mask": sample["mask"].copy()}
    invalid_wait["mask"][ACTIONS.index("WAIT")] = 0.0
    assert constant_program.covered(
        invalid_wait, actor_action="WAIT", purpose="explanation") is None
    with pytest.raises(ValueError, match="purpose"):
        constant_program.covered(sample, actor_action="WAIT", purpose="other")


def test_independent_audit_rejects_reuse(constant_program, episode_rows):
    audit_rows = [row for seed in ("112", "113")
                  for row in episode_rows["robot_2"][seed]]
    audit = audit_frozen_program(constant_program, audit_rows, _masked_oracle)
    assert audit["actor_role_sha256"] == ACTOR_HASH
    assert audit["program_sha256"] == constant_program.sha256()
    assert audit["development_seeds"] == [112, 113]
    assert audit["feedback"]["coverage"] >= 0.2
    assert audit["explanation"]["agreement"] >= 0.9
    with pytest.raises(ValueError, match="reused"):
        audit_frozen_program(
            constant_program, episode_rows["robot_2"]["100"], _masked_oracle)
