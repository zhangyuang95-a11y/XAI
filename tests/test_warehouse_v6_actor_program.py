"""Warehouse v6 parity, program fidelity gates, and paired runner smoke."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor,
)
from backend.training.warehouse_v6_program import (
    _metrics, fit_program, audit_frozen_program,
)
from backend.training.warehouse_v6_runner import (
    _program_feedback, source_closure_sha256, train_pair,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import action_masks
from env.warehouse.public_observation_v6 import (
    encode_policy_features, encode_public_observation,
)

SMOKE_PAIR_SEED = 986_000
SMOKE_EVALUATION_SEEDS = (987_000, 987_001)


def _state(seed: int):
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=seed)
    marked = env.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    env.set_state(marked)
    return env


def _split_rows(actor):
    groups = []
    for start in (100, 102, 104, 106):
        rows = []
        for seed in (start, start + 1):
            env = _state(seed)
            for _ in range(4):
                obs, masks = encode_public_observation(env.state), action_masks(env.state)
                rows.append({"obs": obs[0].copy(), "mask": masks[0].copy(),
                             "episode": str(seed)})
                commands, _ = actor.act(env.state, base_seed=seed)
                env.step(commands)
        groups.append(rows)
    return groups


def test_actor_torch_numpy_export_parity_on_public_states(tmp_path):
    torch.manual_seed(41)
    model = WarehouseV6Actor()
    path = tmp_path / "v6.npz"
    artifact = export_actor(model, path, pair_seed=SMOKE_PAIR_SEED,
                            variant="regularized", candidate_id="test")
    frozen = FrozenWarehouseV6Actor.load(path, expected_sha256=artifact["sha256"],
                                          expected_role_sha256=artifact["role_sha256"])
    for seed in (101, 102, 103):
        env = _state(seed)
        for _ in range(3):
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            with torch.no_grad():
                expected = model.outputs(torch.from_numpy(obs)[None],
                                         torch.from_numpy(masks)[None])[0].numpy()
            actual = np.where(masks > 0.5, frozen.logits(obs, masks), -1e9)
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
            commands, distributions = frozen.act(env.state, base_seed=seed)
            for role, row in (("robot_1", 0), ("robot_2", 1)):
                assert masks[row, ACTIONS.index(commands[role])] == 1
                assert distributions[role].proposed_action == commands[role]
            env.step(commands)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        FrozenWarehouseV6Actor.load(path, expected_sha256="0" * 64)


def test_program_uses_full_masked_nn_and_rejects_raw_illegal_actions():
    torch.manual_seed(42)
    actor = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    fit, calibration, eligibility, validation = _split_rows(actor)
    program = fit_program("robot_1", actor, fit, calibration, eligibility, validation)
    assert program.report["split_episodes"] == {
        "fit": 2, "calibration": 2, "eligibility": 2, "validation": 2}
    sample = validation[0]
    encoded = encode_policy_features(sample["obs"], sample["mask"])
    leaf, _ = program._leaf(sample)
    assert leaf == int(actor.roles["robot_1"].leaf(encoded))
    assert len(program.lookup(sample)["path"]) == 4
    with pytest.raises(ValueError, match="reused fitting episodes"):
        audit_frozen_program(program, actor, validation)
    illegal = int(np.flatnonzero(sample["mask"] <= 0.5)[0])
    changed = program.leaf_logits.copy()
    changed[leaf] = -20
    changed[leaf, illegal] = 20
    invalid = replace(program, leaf_logits=changed)
    metric = _metrics(invalid, actor.roles["robot_1"], [sample], {leaf})
    assert metric["illegal_count"] == 1
    assert metric["masked_illegal_count"] == 0
    assert invalid.lookup(sample)["raw_action_legal"] is False


def test_program_feedback_has_a_real_actor_gradient():
    torch.manual_seed(43)
    model = WarehouseV6Actor()
    actor = FrozenWarehouseV6Actor.from_model(model)
    program = fit_program("robot_1", actor, *_split_rows(actor))
    env = _state(100)
    obs, masks = encode_public_observation(env.state), action_masks(env.state)
    role_obs = torch.from_numpy(obs[:1])
    role_mask = torch.from_numpy(masks[:1])
    logits = model.role_logits("robot_1", role_obs, role_mask)
    action = int(logits.argmax(-1)[0])
    leaf, _ = program._leaf({"obs": obs[0], "mask": masks[0]})
    changed = program.leaf_logits.copy()
    changed[leaf, action] += 0.2
    report = dict(program.report)
    report["validation"] = {**report["validation"], "feedback_online_passed": True}
    program = replace(program, leaf_logits=changed,
                      feedback_leaves=frozenset({leaf}), report=report)
    penalty, count, _ = _program_feedback(
        logits, role_obs, role_mask, torch.tensor([action]),
        program, coefficient=0.01)
    assert count == 1
    gradients = torch.autograd.grad(penalty, (parameter for parameter in
                                    model.actors["robot_1"].parameters()
                                    if parameter.requires_grad),
                                    allow_unused=True)
    assert sum(float(grad.square().sum()) for grad in gradients if grad is not None) > 0


def test_paired_smoke_keeps_bc_start_and_zero_raw_feedback(tmp_path):
    spec = {
        "candidate_id": "smoke", "pair_seed": SMOKE_PAIR_SEED, "joint_steps": 120,
        "output_dir": str(tmp_path), "scenario_namespace": "smoke-v6",
        "checkpoint_seeds": list(SMOKE_EVALUATION_SEEDS), "bc_episodes": 2,
        "bc_epochs": 1, "dagger_rounds": 0, "smoke": True,
    }
    report = train_pair(spec)
    raw, regularized = (report["variants"][variant]
                        for variant in ("raw", "regularized"))
    assert raw["initial_role_sha256"] == regularized["initial_role_sha256"]
    assert raw["scenario_schedule_sha256"] == regularized["scenario_schedule_sha256"]
    assert raw["actor_decisions"] == regularized["actor_decisions"] == 240
    assert raw["feedback_rows"] == {"robot_1": 0, "robot_2": 0}
    assert raw["feedback_gradient_norm"] == {"robot_1": 0.0, "robot_2": 0.0}
    assert all(value > 0 for value in raw["role_weight_change_norm"].values())
    closure = source_closure_sha256()
    assert report["source_closure_sha256"] == closure
    assert raw["source_closure_sha256"] == regularized["source_closure_sha256"] == closure
    checkpoint = (tmp_path / "smoke" /
                  f"pair_{SMOKE_PAIR_SEED}_steps_120" /
                  "raw.checkpoint.pt")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["source_closure_sha256"] = "0" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint identity"):
        train_pair(spec)
