"""v6.16 program certificates and current-Actor core qualification."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training.warehouse_v6_actor import FEATURE_DIM, FEATURE_NAMES
from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, WarehouseV6Actor, V617_MODEL_VERSION,
)
from backend.training.warehouse_v6_program import (
    JOINT_CERTIFIED_TOPOLOGY_METHOD, JOINT_PROGRAM_VERSION,
    WarehouseV6Program, actor_sparse_topology_matches,
    certified_program_actions, fit_program,
)
from backend.training.warehouse_v614_certificate_tree import (
    CertificateSearchConfig, FeatureSchema, _pure_mask_gate_actions,
    fit_certificate_tree,
)
from backend.training.warehouse_v614_masked_refit import (
    EpisodeRows, RefitConfig, qualify_actor_core,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.pure_nn import action_masks
from env.warehouse.public_action_features_v617 import encode_actor_tree_features


def _manual_program(*, oblique: bool = False) -> WarehouseV6Program:
    weights = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    weights[0, FEATURE_NAMES.index("legal_up")] = 1.0
    if oblique:
        weights[0, FEATURE_NAMES.index("self_column")] = 1.0
    bias = np.array([-1.4 if oblique else -0.5], dtype=np.float32)
    logits = np.zeros((2, len(ACTIONS)), dtype=np.float32)
    logits[0, ACTIONS.index("WAIT")] = 2.0
    logits[1, ACTIONS.index("UP")] = 2.0
    report = {
        "fit_method": JOINT_CERTIFIED_TOPOLOGY_METHOD,
        "max_depth": 1, "leaf_count": 2, "max_features_per_node": 3,
        "validation": {"feedback_passed": True, "explanation_passed": True},
    }
    return WarehouseV6Program(
        "robot_1", "a" * 64, 0, weights, bias,
        np.array([-1], dtype=np.int64), np.array([-2], dtype=np.int64),
        logits, frozenset({1}), frozenset({1}), report)


def test_joint_program_requires_pure_mask_gate_even_when_oblique_path_proves_move():
    pure = _manual_program()
    assert certified_program_actions(pure)[1] == frozenset({
        ACTIONS.index("WAIT"), ACTIONS.index("UP")})
    data = pure.to_dict()
    assert data["version"] == JOINT_PROGRAM_VERSION
    assert WarehouseV6Program.from_dict(
        data, actor_role_sha256="a" * 64).feedback_leaves == frozenset({1})

    # The shared interval checker can prove this oblique right branch implies
    # legal UP, but v2 requires an explicit, dedicated legal bit decision.
    oblique = _manual_program(oblique=True)
    assert ACTIONS.index("UP") not in certified_program_actions(oblique)[1]
    with pytest.raises(RuntimeError, match="path legality proof"):
        WarehouseV6Program.from_dict(
            oblique.to_dict(), actor_role_sha256="a" * 64)


def test_recalibrated_leaf_action_is_recertified_before_eligibility(monkeypatch):
    from backend.training import warehouse_v6_program as module

    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=120)
    states = []
    for position in ((0, 3), (1, 3)):
        state = env.get_state()
        state.agents[0].position = position
        env.set_state(state)
        obs, masks = encode_public_observation(env.state), action_masks(env.state)
        states.append((obs[0].copy(), masks[0].copy()))
    assert states[0][1][ACTIONS.index("UP")] == 0
    assert states[1][1][ACTIONS.index("UP")] == 1
    assert all(mask[ACTIONS.index("DOWN")] == 1 for _, mask in states)

    weights = {
        "tree.node_count": np.array([1], dtype=np.int64),
        "tree.leaf_count": np.array([2], dtype=np.int64),
        "tree.split.weight": np.zeros((1, FEATURE_DIM), dtype=np.float32),
        "tree.split.bias": np.array([-0.5], dtype=np.float32),
        "tree.left_refs": np.array([-1], dtype=np.int64),
        "tree.right_refs": np.array([-2], dtype=np.int64),
        "tree.leaf_logits": np.zeros((2, len(ACTIONS)), dtype=np.float32),
    }
    weights["tree.split.weight"][0, FEATURE_NAMES.index("legal_up")] = 1.0
    mask_index = FEATURE_NAMES.index("legal_up")

    class Role:
        role_sha256 = "b" * 64
        feature_dim = FEATURE_DIM
        feature_names = FEATURE_NAMES
        def __init__(self):
            self.weights = {**weights, "protected_leaf_mask": np.ones(2, np.float32)}
        def leaf(self, encoded):
            return (encoded[:, mask_index] > 0.5).astype(np.int64)
        def logits_encoded(self, encoded):
            values = np.zeros((len(encoded), len(ACTIONS)), dtype=np.float32)
            values[:, ACTIONS.index("WAIT")] = 2.0
            values[:, ACTIONS.index("UP")] = 1.0
            return values

    role = Role()
    actor = SimpleNamespace(roles={"robot_1": role})
    groups = []
    for purpose, seed in zip(
            ("fit", "calibration", "eligibility", "validation"),
            (101, 102, 103, 104), strict=True):
        groups.append([
            {"obs": obs, "mask": mask, "episode": f"{purpose}-{seed}-{index}",
             "base_seed": seed * 10 + index, "sampled_action": "WAIT"}
            for index, (obs, mask) in enumerate(states)
        ])

    def flip_after_fit(logits, _actor, _x, _masks, _leaves, **_kwargs):
        changed = logits.copy()
        changed[1] = -2.0
        changed[1, ACTIONS.index("DOWN")] = 3.0
        return changed

    monkeypatch.setattr(module, "_calibrate_leaf_logits", flip_after_fit)
    monkeypatch.setattr(module, "_eligible", lambda *_args: True)
    program = fit_program(
        "robot_1", actor, *groups,
        method=JOINT_CERTIFIED_TOPOLOGY_METHOD)
    assert actor_sparse_topology_matches(program, role)
    assert program.report["fit_method"] == JOINT_CERTIFIED_TOPOLOGY_METHOD
    assert program.report["leaf_path_certificates"]["1"]["raw_top_action"] == "DOWN"
    assert not program.report["leaf_path_certificates"]["1"]["raw_top_path_certified"]
    assert program.report["leaf_eligibility"]["1"]["feedback"]["illegal_count"] == 0
    assert program.report["leaf_eligibility"]["1"]["feedback"][
        "path_uncertified_count"] == 1
    assert 1 not in program.feedback_leaves | program.explanation_leaves
    assert program.to_dict()["version"] == JOINT_PROGRAM_VERSION

    # A damaged asset cannot designate the unproven leaf as evidence later.
    tampered = replace(program, feedback_leaves=frozenset({1}))
    with pytest.raises(RuntimeError, match="path legality proof"):
        WarehouseV6Program.from_dict(
            tampered.to_dict(), actor_role_sha256=role.role_sha256)


def test_pure_gate_search_never_certifies_movement_via_oblique_condition():
    names = ("self_column", "same_target_collision_streak_capped",
             "robots_shortest_distance", "self_charger_return_margin",
             *(f"legal_{action.lower()}" for action in ACTIONS))
    schema = FeatureSchema(names, ACTIONS)
    x = np.zeros((300, len(names)), dtype=np.float32)
    x[:, 0] = (np.arange(300) % 10) / 10.0
    x[:, 2] = 0.5
    masks = np.zeros((300, len(ACTIONS)), dtype=np.float32)
    masks[:, ACTIONS.index("UP")] = (np.arange(300) % 10 >= 5).astype(np.float32)
    masks[:, ACTIONS.index("WAIT")] = 1.0
    x[:, schema.mask_feature_indices] = masks
    target = np.zeros_like(masks)
    target[:, ACTIONS.index("UP")] = 0.9 * masks[:, ACTIONS.index("UP")]
    target[:, ACTIONS.index("WAIT")] = 1.0 - target[:, ACTIONS.index("UP")]
    episodes = [f"synthetic-fit:{i // 10}" for i in range(len(x))]
    tree = fit_certificate_tree(
        x, masks, target, episodes, schema=schema,
        frozen_actor_sha256="c" * 64,
        config=CertificateSearchConfig(
            beam_width=1, max_search_leaves=3, max_axis_features=3,
            max_exact_candidates_per_round=8,
            max_exact_candidates_per_leaf=2,
            pure_movement_gates_only=True))
    pure = _pure_mask_gate_actions(tree)
    assert tree.report["pure_movement_gates_only"] is True
    assert tree.report["core_pure_gate_leaf_count"] > 0
    assert all(int(np.argmax(tree.leaf_logits[leaf])) in pure[leaf]
               for leaf in range(tree.leaf_count) if tree.core_mask[leaf])
    assert any(int(np.argmax(tree.leaf_logits[leaf])) == ACTIONS.index("UP")
               for leaf in range(tree.leaf_count) if tree.core_mask[leaf])


def test_actor_core_qualification_uses_current_logits_and_disjoint_purposes():
    from backend.training.warehouse_v6_program import (
        _masked_probabilities, certified_actor_actions,
    )

    frozen = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor(model_version=V617_MODEL_VERSION))
    role = frozen.roles["robot_1"]
    env = WarehouseMultiAgentEnv(participant_study_config())
    groups = {}
    for purpose, base in zip(
            ("fit", "calibration", "eligibility", "validation"),
            (300, 400, 500, 600), strict=True):
        features, masks, ids, seeds = [], [], [], []
        for offset, position in enumerate(((0, 3), (1, 3))):
            seed = base + offset
            env.reset(seed=seed)
            state = env.get_state()
            state.agents[0].position = position
            env.set_state(state)
            obs, physical = encode_public_observation(env.state), action_masks(env.state)
            features.append(encode_actor_tree_features(obs[0], physical[0]))
            masks.append(physical[0])
            ids.append(f"nn-visit:{seed}")
            seeds.append(seed)
        x = np.stack(features)
        m = np.stack(masks)
        target = _masked_probabilities(role.logits_encoded(x), m)
        groups[purpose] = EpisodeRows(
            x, m, target.argmax(1).astype(np.int64), target,
            np.zeros(len(x), dtype=np.float32), tuple(ids), tuple(seeds),
            role.role_sha256)
    config = RefitConfig(
        min_leaf_rows=1, min_leaf_episodes=1, max_episode_fraction=1.0,
        min_global_agreement=0.01, max_global_masked_kl=10.0,
        min_stratum_rows=1, min_stratum_episodes=1,
        min_stratum_agreement=0.01, max_stratum_masked_kl=10.0,
        min_masked_margin=1e-8, min_internal_core_mass=0.0,
        min_internal_expected_actionable_mass=0.0)
    mask, report = qualify_actor_core(groups, frozen_role=role, config=config)
    assert mask.shape == (16,)
    assert report["method"] == "actor_risk_protected_core_qualification_v3"
    assert report["target_parity_max_abs_error"]["fit"] <= 1e-7
    assert report["actor_tree_logits_sha256"]
    proofs = certified_actor_actions(role)
    assert all(int(np.argmax(role.weights["tree.leaf_logits"][leaf])) in proofs[leaf]
               for leaf in np.flatnonzero(mask))
    with pytest.raises(ValueError, match="overlap"):
        qualify_actor_core(
            {**groups, "validation": groups["fit"]},
            frozen_role=role, config=config)
