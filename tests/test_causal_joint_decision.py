from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

from env.warehouse.credit_assignment import FrozenMission, mission_goal_distance
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.decision_protocol import (
    DECISION_AUDIT_SCHEMA,
    canonical_sha256,
    distribution_decision_metadata,
)
from env.warehouse.domain import WarehouseConfig
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from env.warehouse.navigation import ACTIONS
from env.warehouse.partner_policies import (
    PARTNER_PROFILES,
    participant_surrogate_distribution,
    robust_partner_robot_two_action,
)
from env.warehouse.transition_audit import (
    necessary_participant_standoff_clearance,
)


def _policy_and_frozen_state(seed: int = 2_601):
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=seed)
    state = environment.get_state()
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=seed),
        device="cpu",
    )
    return environment, observations, state, policy


def test_two_actors_use_one_batched_forward_from_one_frozen_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment, observations, state, policy = _policy_and_frozen_state()
    calls: list[tuple[int, ...]] = []
    original = policy.network.actor_logits

    def recording_forward(local):
        calls.append(tuple(int(value) for value in local.shape))
        return original(local)

    monkeypatch.setattr(policy.network, "actor_logits", recording_forward)
    actions, distributions = policy.act(
        OrderedDict(reversed(tuple(observations.items()))),
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )

    assert calls == [(2, len(observations["robot_1"]))]
    assert set(actions) == {"robot_1", "robot_2"}
    assert set(distributions) == {"robot_1", "robot_2"}


def test_actor_order_and_sampling_are_independent_under_fixed_key() -> None:
    environment, observations, state, policy = _policy_and_frozen_state(2_602)
    forward_actions, forward_distributions = policy.act(
        observations,
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )
    reverse_actions, reverse_distributions = policy.act(
        OrderedDict(reversed(tuple(observations.items()))),
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )

    assert reverse_actions == forward_actions
    for agent_id in environment.agent_ids:
        assert reverse_distributions[agent_id].probabilities == pytest.approx(
            forward_distributions[agent_id].probabilities,
            abs=0.0,
        )
        assert reverse_distributions[agent_id].logits == pytest.approx(
            forward_distributions[agent_id].logits,
            abs=0.0,
        )


def test_robot_two_logits_do_not_depend_on_robot_one_current_action() -> None:
    environment, observations, state, policy = _policy_and_frozen_state(2_603)
    _, baseline = policy.act(
        observations,
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )

    # Participant/robot-1 input is applied only after both distributions have
    # been locked.  All five downstream joint commands therefore retain the
    # exact same robot-2 decision evidence.
    for robot_one_action in ACTIONS:
        actions, current = policy.act(
            observations,
            environment.global_state(),
            deterministic=False,
            decision_key=(state.episode_id, state.frame),
        )
        actions["robot_1"] = robot_one_action
        assert current["robot_2"].logits == pytest.approx(
            baseline["robot_2"].logits,
            abs=0.0,
        )
        assert current["robot_2"].probabilities == pytest.approx(
            baseline["robot_2"].probabilities,
            abs=0.0,
        )


def test_future_rng_or_next_state_cannot_change_current_action() -> None:
    environment, observations, state, policy = _policy_and_frozen_state(2_604)
    first_actions, first_distributions = policy.act(
        observations,
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )

    # Advance an independent environment with a different future action and
    # task RNG.  The frozen S_t observations and keyed decision remain equal.
    future = WarehouseMultiAgentEnv(environment.config)
    future.reset(seed=9_999_999)
    future.set_state(state)
    future.step({"robot_1": "WAIT", "robot_2": "WAIT"})

    second_actions, second_distributions = policy.act(
        observations,
        environment.global_state(),
        deterministic=False,
        decision_key=(state.episode_id, state.frame),
    )
    assert second_actions == first_actions
    for agent_id in environment.agent_ids:
        np.testing.assert_array_equal(
            second_distributions[agent_id].probabilities,
            first_distributions[agent_id].probabilities,
        )


def test_environment_emits_tamper_resistant_joint_decision_audit() -> None:
    environment, observations, before, policy = _policy_and_frozen_state(2_605)
    actions, distributions = policy.act(
        observations,
        environment.global_state(),
        deterministic=False,
        decision_key=(before.episode_id, before.frame),
    )
    metadata = distribution_decision_metadata(
        distributions,
        decision_source="test_actor",
    ) | {
        # Required invariants must win over untrusted caller metadata.
        "same_pre_move_state": False,
        "environment_step_calls": 99,
        "pre_move_state_sha256": "forged",
    }
    _, _, _, _, info = environment.step(actions, decision_metadata=metadata)
    after = environment.get_state()
    audit = info["decision_audit"]

    assert audit["schema_version"] == DECISION_AUDIT_SCHEMA
    assert audit["same_pre_move_state"] is True
    assert audit["environment_step_calls"] == 1
    assert audit["pre_move_state_sha256"] == canonical_sha256(before)
    assert audit["post_move_state_sha256"] == canonical_sha256(after)
    assert audit["pre_move_observation_sha256"] == {
        agent_id: canonical_sha256(observation)
        for agent_id, observation in sorted(observations.items())
    }
    assert audit["joint_action"] == actions
    assert audit["decision_source"] == "test_actor"
    assert set(audit["action_distributions"]) == {"robot_1", "robot_2"}


def test_noncoordinated_partner_profiles_do_not_assume_teammate_vacates() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=2_606)
    state = environment.get_state()
    state.by_id("robot_1").position = (7, 3)
    state.by_id("robot_1").battery = 2.0
    state.by_id("robot_2").position = environment.layout.charger_position
    environment.set_state(state)

    for profile in PARTNER_PROFILES:
        probabilities = participant_surrogate_distribution(
            environment,
            profile=profile,
        )
        assert probabilities.shape == (len(ACTIONS),)
        assert float(probabilities.sum()) == pytest.approx(1.0)
        assert np.all(probabilities >= 0.0)
        if profile != "coordinated":
            assert probabilities[ACTIONS.index("RIGHT")] == pytest.approx(0.0)


def test_participant_control_mode_is_episode_known_not_current_action() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    ai_observations, _ = environment.reset(seed=2_607)
    participant_state = environment.get_state()
    participant_state.participant_controlled_agent_id = "robot_1"
    environment.set_state(participant_state)
    participant_observations = environment.observations()

    for agent_id in environment.agent_ids:
        changed = np.flatnonzero(
            ai_observations[agent_id] != participant_observations[agent_id]
        )
        participant_index = 22 if agent_id == "robot_1" else 23
        assert participant_index in changed
        assert participant_observations[agent_id][participant_index] == pytest.approx(1.0)
        # Episode provenance may also change the frozen safety mask and public
        # coordination features. It must never depend on the participant's
        # not-yet-submitted current action.
        assert np.isfinite(participant_observations[agent_id]).all()


def test_participant_collision_term_dominates_a_saturated_mission_preference() -> None:
    environment, observations, _, policy = _policy_and_frozen_state(2_608)
    row = np.asarray(observations["robot_2"], dtype=np.float32).copy()
    network = policy.network
    risky = ACTIONS.index("DOWN")
    wait = ACTIONS.index("WAIT")
    start = network.joint_collision_matrix_start
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.float32)
    matrix[risky, :] = 1.0
    row[start : start + len(ACTIONS) ** 2] = matrix.reshape(-1)
    row[
        network.teammate_legal_action_mask_start :
        network.teammate_legal_action_mask_start + len(ACTIONS)
    ] = 1.0
    row[-len(ACTIONS) :] = 1.0

    ai_row = row.copy()
    ai_row[23] = 0.0
    participant_row = row.copy()
    participant_row[23] = 1.0
    with torch.no_grad():
        ai_logits = policy.network.actor_logits(torch.as_tensor(ai_row[None]))[0]
        participant_logits = policy.network.actor_logits(
            torch.as_tensor(participant_row[None])
        )[0]
    ai_relative = float(ai_logits[risky] - ai_logits[wait])
    participant_relative = float(
        participant_logits[risky] - participant_logits[wait]
    )
    assert participant_relative < ai_relative - 50.0


def test_energy_exhaustion_term_keeps_terminal_moves_below_wait() -> None:
    _, observations, _, policy = _policy_and_frozen_state(2_609)
    row = np.asarray(observations["robot_2"], dtype=np.float32).copy()
    row[2] = policy.environment_config.move_battery_cost / 100.0
    row[-len(ACTIONS) :] = 1.0
    with torch.no_grad():
        logits = policy.network.actor_logits(torch.as_tensor(row[None]))[0]
    wait_logit = float(logits[ACTIONS.index("WAIT")])
    assert max(float(logits[index]) for index in range(len(ACTIONS) - 1)) < (
        wait_logit - 50.0
    )


def test_charge_first_credit_tracks_charger_not_attached_task() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=20))
    environment.reset(seed=2_613)
    state = environment.get_state()
    agent = state.by_id("robot_2")
    agent.position = (5, 4)
    task = state.tasks[0]
    mission = FrozenMission(
        "charge",
        environment.layout.charger_position,
        task,
    )

    toward_charger = mission_goal_distance(
        environment,
        state,
        agent,
        mission,
        (6, 4),
    )
    away_from_charger = mission_goal_distance(
        environment,
        state,
        agent,
        mission,
        (4, 4),
    )

    assert toward_charger < away_from_charger


def test_dual_charger_approach_clears_exit_before_priority_entry() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=11_811_008)
    state = environment.get_state()
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    state.participant_controlled_agent_id = "robot_1"
    human.position = (7, 3)
    human.battery = 20.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.charge_mode_active = True
    ai.position = (6, 4)
    ai.battery = 12.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.charge_mode_active = True
    environment.set_state(state)

    first = stable_coordination_actions(environment)
    assert first == {"robot_1": "UP", "robot_2": "WAIT"}

    state.by_id("robot_1").position = (6, 3)
    environment.set_state(state)
    second = stable_coordination_actions(environment)
    assert second == {"robot_1": "WAIT", "robot_2": "DOWN"}


def test_dual_charger_clearance_priority_does_not_flip_after_yield_cost() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=11_811_012)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (3, 3)
    human.battery = 40.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.charge_mode_active = True
    human.last_executed_action = "LEFT"
    ai.position = (2, 4)
    ai.battery = 40.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.charge_mode_active = True
    ai.last_executed_action = "WAIT"
    environment.set_state(state)

    # Robot 1 has already yielded once for Robot 2. It continues clearing
    # instead of becoming the new priority merely because both slacks tied.
    clearance = stable_coordination_actions(environment)
    assert clearance == {"robot_1": "LEFT", "robot_2": "WAIT"}

    state.by_id("robot_1").position = (3, 2)
    state.by_id("robot_1").battery = 38.0
    state.by_id("robot_1").last_executed_action = "LEFT"
    environment.set_state(state)
    passage = stable_coordination_actions(environment)
    assert passage == {"robot_1": "WAIT", "robot_2": "DOWN"}


def test_robust_partner_label_honors_safe_lower_energy_charger_handoff() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=11_811_009)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (6, 4)
    human.battery = 12.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.charge_mode_active = True
    ai.position = environment.layout.charger_position
    ai.battery = 46.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.charge_mode_active = True
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "RIGHT"}
    assert robust_partner_robot_two_action(
        environment,
        preferred_action=coordinated["robot_2"],
    ) == "RIGHT"


def test_robust_partner_label_completes_second_charger_clearance_phase() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=11_811_010)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (6, 4)
    human.battery = 12.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.charge_mode_active = True
    ai_task = state.tasks[1]
    ai_task.status = "carried"
    ai_task.carrier_agent_id = ai.agent_id
    ai_task.delivery_position = (1, 1)
    ai.position = (7, 5)
    ai.battery = 42.0
    ai.carrying_task_id = ai_task.task_id
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.charge_mode_active = True
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "UP"}
    assert robust_partner_robot_two_action(
        environment,
        preferred_action=coordinated["robot_2"],
    ) == "UP"


def test_recent_charger_departure_cannot_immediately_reverse_the_handoff() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=11_811_011)
    state = environment.get_state()
    state.frame = 50
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (5, 4)
    human.battery = 18.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.charge_mode_active = True
    ai.position = (6, 4)
    ai.battery = 64.0
    ai.navigation_goal_kind = "wait"
    ai.navigation_goal_position = ai.position
    ai.charge_mode_active = False
    ai.last_charger_departure_frame = 49
    ai.last_executed_action = "UP"
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated["robot_2"] != "DOWN"
    assert robust_partner_robot_two_action(
        environment,
        preferred_action=coordinated["robot_2"],
    ) == coordinated["robot_2"]


def test_loaded_delivery_clearance_persists_until_entry_is_robust() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_614)
    state = environment.get_state()
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    first_task, second_task = state.tasks
    first_task.status = "carried"
    first_task.carrier_agent_id = human.agent_id
    first_task.delivery_position = (2, 7)
    human.carrying_task_id = first_task.task_id
    human.position = (5, 3)
    human.navigation_goal_kind = "delivery"
    human.navigation_goal_position = first_task.delivery_position
    human.last_executed_action = "LEFT"
    second_task.status = "carried"
    second_task.carrier_agent_id = ai.agent_id
    second_task.delivery_position = (5, 4)
    ai.carrying_task_id = second_task.task_id
    ai.position = (4, 4)
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = second_task.delivery_position
    environment.set_state(state)

    plan = environment.get_state().active_coordination_plan
    assert plan is not None
    assert plan["phase"] == "SINGLE_STEP"
    assert plan["priority_agent_id"] == "robot_2"
    observations = environment.observations()
    assert observations["robot_1"][-len(ACTIONS) :].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert observations["robot_2"][-len(ACTIONS) :].tolist() == [
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    state.by_id("robot_1").position = (5, 2)
    state.by_id("robot_1").last_executed_action = "LEFT"
    environment.set_state(state)
    entry = stable_coordination_actions(environment)
    assert entry == {"robot_1": "WAIT", "robot_2": "DOWN"}


def test_participant_delivery_priority_advances_while_actor_yields() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_615)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    first_task, second_task = state.tasks
    first_task.status = "carried"
    first_task.carrier_agent_id = human.agent_id
    first_task.delivery_position = (1, 1)
    human.carrying_task_id = first_task.task_id
    human.position = (1, 2)
    human.navigation_goal_kind = "delivery"
    human.navigation_goal_position = first_task.delivery_position
    second_task.status = "carried"
    second_task.carrier_agent_id = ai.agent_id
    second_task.delivery_position = (2, 4)
    ai.carrying_task_id = second_task.task_id
    # Keep the peer on the inner side of the loaded participant. Placing it
    # at the shelf dead end would intentionally invoke the stronger public
    # single-lane egress rule instead of this delivery-priority case.
    ai.position = (1, 3)
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = second_task.delivery_position
    environment.set_state(state)

    actions = stable_coordination_actions(environment)
    assert actions == {"robot_1": "LEFT", "robot_2": "WAIT"}


def test_idle_participant_can_claim_pickup_to_clear_delivery_endpoint() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_616)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (1, 1)
    human.carrying_task_id = None
    human.navigation_goal_kind = "wait"
    human.navigation_goal_position = human.position
    available, carried = state.tasks
    available.status = "available"
    available.pickup_position = (1, 0)
    available.delivery_position = (4, 8)
    carried.status = "carried"
    carried.carrier_agent_id = ai.agent_id
    carried.pickup_position = (5, 0)
    carried.delivery_position = (1, 2)
    ai.position = (1, 3)
    ai.carrying_task_id = carried.task_id
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = carried.delivery_position
    environment.set_state(state)

    clearance = stable_coordination_actions(environment)
    assert clearance == {"robot_1": "LEFT", "robot_2": "WAIT"}


def test_single_lane_outer_robot_keeps_egress_priority_until_inner_robot_clears() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_617)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    outer = state.by_id("robot_1")
    inner = state.by_id("robot_2")
    outer.position = (5, 0)
    outer.navigation_goal_kind = "pickup"
    outer.navigation_goal_position = (4, 8)
    inner.position = (5, 3)
    inner.navigation_goal_kind = "pickup"
    inner.navigation_goal_position = (5, 0)
    environment.set_state(state)
    goals = {"robot_1": (4, 8), "robot_2": (5, 0)}

    # The robot at the shelf end advances for two frozen decisions. The
    # inner robot cannot alternate back toward it merely because their gap
    # changed by one cell.
    first = stable_coordination_actions(environment, goal_overrides=goals)
    assert first == {"robot_1": "RIGHT", "robot_2": "WAIT"}

    state.by_id("robot_1").position = (5, 1)
    environment.set_state(state)
    second = stable_coordination_actions(environment, goal_overrides=goals)
    assert second == {"robot_1": "RIGHT", "robot_2": "WAIT"}

    # Once adjacent, the inner robot clears toward the spine before the outer
    # robot proceeds. This is a public S_t handshake, not a reaction to the
    # participant's unobserved current-frame action.
    state.by_id("robot_1").position = (5, 2)
    environment.set_state(state)
    clearance = stable_coordination_actions(environment, goal_overrides=goals)
    assert clearance == {"robot_1": "WAIT", "robot_2": "RIGHT"}

    state.by_id("robot_2").position = (5, 4)
    environment.set_state(state)
    exit_step = stable_coordination_actions(environment, goal_overrides=goals)
    assert exit_step == {"robot_1": "RIGHT", "robot_2": "WAIT"}


def test_single_lane_egress_priority_persists_after_peer_leaves_the_row() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_619)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (2, 4)
    human.last_executed_action = "UP"
    human.navigation_goal_kind = "delivery"
    human.navigation_goal_position = (4, 5)
    ai.position = (3, 2)
    ai.last_executed_action = "LEFT"
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = (5, 2)
    environment.set_state(state)
    goals = {"robot_1": (4, 5), "robot_2": (5, 2)}

    # Robot 1 has just performed the public vertical clearance. Although the
    # robots are no longer on the same row, Robot 2 retains the egress role.
    first = stable_coordination_actions(environment, goal_overrides=goals)
    assert first == {"robot_1": "WAIT", "robot_2": "RIGHT"}

    state.by_id("robot_1").last_executed_action = "WAIT"
    state.by_id("robot_2").position = (3, 3)
    state.by_id("robot_2").last_executed_action = "RIGHT"
    environment.set_state(state)
    second_clearance = stable_coordination_actions(
        environment,
        goal_overrides=goals,
    )
    assert second_clearance == {"robot_1": "UP", "robot_2": "WAIT"}

    state.by_id("robot_1").position = (1, 4)
    state.by_id("robot_1").last_executed_action = "UP"
    state.by_id("robot_2").last_executed_action = "WAIT"
    environment.set_state(state)
    final_entry = stable_coordination_actions(environment, goal_overrides=goals)
    assert final_entry == {"robot_1": "WAIT", "robot_2": "RIGHT"}


@pytest.mark.parametrize("profile", PARTNER_PROFILES)
def test_every_partner_profile_respects_public_single_lane_egress(profile: str) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_618)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (5, 1)
    human.navigation_goal_kind = "pickup"
    human.navigation_goal_position = (4, 8)
    ai.position = (5, 3)
    ai.navigation_goal_kind = "pickup"
    ai.navigation_goal_position = (5, 0)
    environment.set_state(state)

    probabilities = participant_surrogate_distribution(
        environment,
        profile=profile,
    )
    if profile == "hesitant":
        assert probabilities[ACTIONS.index("RIGHT")] == pytest.approx(0.75)
        assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(0.25)
    elif profile == "cautious":
        # Cautious timing may spend one observable frame confirming that the
        # Actor yields, but it must take the same public egress on the next
        # frozen state rather than reversing its route.
        assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(1.0)
        state.ineffective_joint_wait_streak = 1
        environment.set_state(state)
        retry = participant_surrogate_distribution(
            environment,
            profile=profile,
        )
        assert retry[ACTIONS.index("RIGHT")] == pytest.approx(1.0)
    else:
        assert probabilities[ACTIONS.index("RIGHT")] == pytest.approx(1.0)


@pytest.mark.parametrize("profile", PARTNER_PROFILES)
def test_every_partner_profile_respects_public_dual_charger_wait(
    profile: str,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_620)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (5, 4)
    human.battery = 8.0
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.last_executed_action = "WAIT"
    ai.position = (3, 4)
    ai.battery = 6.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.last_executed_action = "WAIT"
    environment.set_state(state)

    # Neither depleted robot can afford an additional side clearance. The
    # common S_t priority therefore commits Robot 1 to WAIT while Robot 2
    # advances; no profile may independently walk into the target cell.
    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "DOWN"}
    probabilities = participant_surrogate_distribution(
        environment,
        profile=profile,
    )
    assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(1.0)


def test_parallel_charger_progress_preserves_the_leading_robot_priority() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_621)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    leader = state.by_id("robot_1")
    follower = state.by_id("robot_2")
    leader.position = (4, 4)
    leader.battery = 28.0
    leader.charge_mode_active = True
    leader.navigation_goal_kind = "charge"
    leader.navigation_goal_position = environment.layout.charger_position
    leader.last_action = "LEFT"
    leader.last_executed_action = "LEFT"
    follower.position = (3, 4)
    follower.battery = 26.0
    follower.charge_mode_active = True
    follower.navigation_goal_kind = "charge"
    follower.navigation_goal_position = environment.layout.charger_position
    follower.last_action = "DOWN"
    follower.last_executed_action = "DOWN"
    environment.set_state(state)

    assert stable_coordination_actions(environment) == {
        "robot_1": "DOWN",
        "robot_2": "WAIT",
    }


@pytest.mark.parametrize("profile", PARTNER_PROFILES)
def test_every_partner_profile_yields_to_critical_ai_charger_route(
    profile: str,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_622)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (6, 3)
    human.battery = 52.0
    carried_task = state.tasks[0]
    carried_task.status = "carried"
    carried_task.carrier_agent_id = human.agent_id
    carried_task.delivery_position = (1, 2)
    human.carrying_task_id = carried_task.task_id
    human.navigation_goal_kind = "delivery"
    human.navigation_goal_position = (1, 2)
    human.last_executed_action = "WAIT"
    ai.position = (5, 4)
    ai.battery = 6.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.last_executed_action = "WAIT"
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "DOWN"}
    probabilities = participant_surrogate_distribution(
        environment,
        profile=profile,
    )
    assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(1.0)


def test_ai_follows_public_dual_charger_clearance_away_from_station() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_623)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (6, 4)
    human.battery = 22.0
    carried_task = state.tasks[0]
    carried_task.status = "carried"
    carried_task.carrier_agent_id = human.agent_id
    human.carrying_task_id = carried_task.task_id
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.last_executed_action = "WAIT"
    ai.position = (7, 5)
    ai.battery = 58.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.last_executed_action = "WAIT"
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "UP"}
    for profile in PARTNER_PROFILES:
        participant = participant_surrogate_distribution(
            environment,
            profile=profile,
            coordinated_actions=coordinated,
        )
        assert participant[ACTIONS.index("WAIT")] == pytest.approx(1.0)
    assert robust_partner_robot_two_action(
        environment,
        preferred_action="WAIT",
    ) == "UP"
    # The deterministic public coordination label remains UP. A newly
    # initialized neural head is intentionally not an acceptance oracle;
    # deployment parity is tested with the trained checkpoint instead.
    assert coordinated["robot_2"] == "UP"


def test_public_charger_handoff_completes_after_lateral_departure() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_625)
    state = environment.get_state()
    state.frame = 20
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (7, 3)
    human.battery = 48.0
    human.charge_mode_active = True
    human.last_action = "LEFT"
    human.last_executed_action = "LEFT"
    human.last_charger_departure_frame = state.frame
    ai.position = (6, 4)
    ai.battery = 36.0
    ai.charge_mode_active = True
    ai.last_action = "WAIT"
    ai.last_executed_action = "WAIT"
    state.tasks[0].pickup_position = (1, 0)
    state.tasks[0].delivery_position = (6, 8)
    state.tasks[1].pickup_position = (3, 0)
    state.tasks[1].delivery_position = (4, 8)
    environment.set_state(state)

    coordinated = stable_coordination_actions(environment)
    assert coordinated == {"robot_1": "WAIT", "robot_2": "DOWN"}
    for profile in PARTNER_PROFILES:
        participant = participant_surrogate_distribution(
            environment,
            profile=profile,
            coordinated_actions=coordinated,
        )
        for action_index, probability in enumerate(participant):
            if probability <= 0.0:
                continue
            _, _, invalid, collision, _, _ = environment._resolve_motion(
                state,
                {
                    "robot_1": ACTIONS[action_index],
                    "robot_2": "DOWN",
                },
            )
            assert not invalid
            assert not collision

    policy = MAPPOPolicy(
        environment.config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_625),
        device="cpu",
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
    assert ACTIONS[int(torch.argmax(logits))] == "DOWN"


def test_charger_priority_actor_never_enters_participant_current_cell() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_626)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human_task, ai_task = state.tasks
    human.position = (5, 4)
    human.battery = 16.0
    human.carrying_task_id = human_task.task_id
    human_task.status = "carried"
    human_task.carrier_agent_id = human.agent_id
    human_task.delivery_position = (4, 8)
    ai.position = (4, 4)
    ai.battery = 14.0
    ai.carrying_task_id = ai_task.task_id
    ai_task.status = "carried"
    ai_task.carrier_agent_id = ai.agent_id
    ai_task.delivery_position = (2, 8)
    environment.set_state(state)
    before = environment.get_state()

    assert all(
        agent.navigation_goal_kind == "charge"
        for agent in before.agents
    )
    assert stable_coordination_actions(environment) == {
        "robot_1": "LEFT",
        "robot_2": "WAIT",
    }
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_626),
        device="cpu",
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
        probabilities = torch.softmax(logits, dim=-1)

    assert ACTIONS[int(torch.argmax(logits))] == "WAIT"
    assert float(probabilities[:-1].sum()) < 1e-6


def test_critical_charger_actor_advances_under_public_participant_wait() -> None:
    config = WarehouseConfig(horizon=120, battery_safety_margin=4.0)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_627)
    state = environment.get_state()
    state.frame = 107
    state.participant_controlled_agent_id = "robot_1"
    state.last_coordination_events = (
        {
            "event": "coordination_yield",
            "yielding_agent_id": "robot_2",
            "passing_agent_id": "robot_1",
        },
    )
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human_task = state.tasks[0]
    human.position = (2, 4)
    human.battery = 52.0
    human.carrying_task_id = human_task.task_id
    human_task.status = "carried"
    human_task.carrier_agent_id = human.agent_id
    human_task.pickup_position = (2, 8)
    human_task.delivery_position = (1, 2)
    human_task.created_frame = 53
    human_task.claimed_frame = 79
    human_task.claimed_battery = 30.0
    human_task.shortest_safe_delivery_steps = 22.0
    human_task.safe_path_charge_planned = True
    human.heading = "UP"
    human.last_action = "UP"
    human.last_executed_action = "UP"
    human.last_charger_departure_frame = 99
    human.carrying_task_at_last_charger_departure = human_task.task_id
    ai.position = (0, 4)
    ai.battery = 20.0
    ai.route_commitment_task_id = state.tasks[1].task_id
    ai.charge_mode_active = True
    ai.heading = "UP"
    ai.last_action = "UP"
    ai.last_executed_action = "UP"
    ai.last_charger_departure_frame = 92
    state.tasks[1].pickup_position = (4, 8)
    state.tasks[1].delivery_position = (3, 2)
    state.tasks[1].created_frame = 99
    environment.set_state(state)
    before = environment.get_state()

    assert before.by_id(ai.agent_id).navigation_goal_kind == "charge"
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_627),
        device="cpu",
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]

    assert ACTIONS[int(torch.argmax(logits))] == "DOWN"


def test_critical_dual_charger_actor_uses_public_reservation_after_clearance() -> None:
    """A critical AI must not freeze after the participant vacates its route."""

    config = WarehouseConfig(horizon=120, battery_safety_margin=4.0)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_629)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (0, 4)
    human.battery = 22.0
    human.charge_mode_active = True
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.last_action = "UP"
    human.last_executed_action = "UP"
    ai.position = (1, 3)
    ai.battery = 16.0
    ai.charge_mode_active = True
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    ai.last_action = "WAIT"
    ai.last_executed_action = "WAIT"
    environment.set_state(state)

    assert stable_coordination_actions(environment) == {
        "robot_1": "WAIT",
        "robot_2": "RIGHT",
    }
    for profile in PARTNER_PROFILES:
        probabilities = participant_surrogate_distribution(
            environment,
            profile=profile,
        )
        assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(1.0)

    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_629),
        device="cpu",
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]

    assert ACTIONS[int(torch.argmax(logits))] == "RIGHT"


def test_critical_dual_charger_actor_uses_safe_single_lane_clearance() -> None:
    """A lower-priority peer clears away, never toward the critical Actor."""

    config = WarehouseConfig(horizon=120, battery_safety_margin=4.0)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_630)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (6, 6)
    human.battery = 48.0
    human.charge_mode_active = True
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    ai.position = (6, 8)
    ai.battery = 16.0
    ai.charge_mode_active = True
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    environment.set_state(state)

    assert stable_coordination_actions(environment) == {
        "robot_1": "LEFT",
        "robot_2": "WAIT",
    }
    for profile in PARTNER_PROFILES:
        probabilities = participant_surrogate_distribution(
            environment,
            profile=profile,
        )
        assert probabilities[ACTIONS.index("RIGHT")] == pytest.approx(0.0)
        assert (
            probabilities[ACTIONS.index("LEFT")]
            + probabilities[ACTIONS.index("WAIT")]
        ) == pytest.approx(1.0)
        for action, probability in zip(ACTIONS, probabilities):
            if probability <= 0.0:
                continue
            assert not environment._resolve_motion(
                environment.get_state(),
                {"robot_1": action, "robot_2": "LEFT"},
            )[3]

    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_630),
        device="cpu",
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]

    actor_action = ACTIONS[int(torch.argmax(logits))]
    assert actor_action in {"LEFT", "WAIT"}
    for profile in PARTNER_PROFILES:
        probabilities = participant_surrogate_distribution(
            environment,
            profile=profile,
        )
        for participant_action, probability in zip(ACTIONS, probabilities):
            if probability <= 0.0:
                continue
            assert not environment._resolve_motion(
                environment.get_state(),
                {"robot_1": participant_action, "robot_2": actor_action},
            )[3]


def test_critical_charger_actor_waits_for_loaded_participant_to_clear_station() -> None:
    config = WarehouseConfig(horizon=120, battery_safety_margin=4.0)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_628)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    carried = state.tasks[0]
    carried.status = "carried"
    carried.carrier_agent_id = human.agent_id
    carried.pickup_position = (2, 8)
    carried.delivery_position = (1, 1)
    carried.claimed_frame = 28
    carried.claimed_battery = 70.0
    carried.shortest_safe_delivery_steps = 20.0
    human.position = environment.layout.charger_position
    human.battery = 56.0
    human.carrying_task_id = carried.task_id
    human.route_commitment_task_id = carried.task_id
    ai.position = (6, 3)
    ai.battery = 10.0
    ai.charge_mode_active = True
    state.tasks[1].pickup_position = (6, 8)
    state.tasks[1].delivery_position = (4, 7)
    environment.set_state(state)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_628),
        device="cpu",
    )

    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
        probabilities = torch.softmax(logits, dim=-1)

    # LEFT from the station targets (7, 3), while UP targets (6, 4).
    # Robot 2 cannot know which legal exit the participant will take, so both
    # simultaneous target cells remain conservatively excluded this frame.
    assert float(probabilities[ACTIONS.index("DOWN")]) < 1e-6
    assert float(probabilities[ACTIONS.index("RIGHT")]) < 1e-6


def test_charging_actor_cannot_stochastically_leave_uncontested_station() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_624)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (1, 0)
    human.navigation_goal_kind = "pickup"
    human.navigation_goal_position = (1, 2)
    ai.position = environment.layout.charger_position
    ai.battery = 40.0
    ai.navigation_goal_kind = "charge"
    ai.navigation_goal_position = environment.layout.charger_position
    environment.set_state(state)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_624),
        device="cpu",
    )

    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
        probabilities = torch.softmax(logits, dim=-1)

    assert ACTIONS[int(torch.argmax(logits))] == "WAIT"
    assert float(probabilities[:-1].sum()) < 1e-6


def test_recent_return_does_not_penalize_productive_wait_on_station() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_629)
    state = environment.get_state()
    state.frame = 40
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (1, 0)
    human.last_action = "UP"
    human.last_executed_action = "UP"
    ai.position = environment.layout.charger_position
    ai.battery = 50.0
    ai.charge_mode_active = True
    ai.last_charger_departure_frame = 39
    ai.last_action = "DOWN"
    ai.last_executed_action = "DOWN"
    environment.set_state(state)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_629),
        device="cpu",
    )

    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
        probabilities = torch.softmax(logits, dim=-1)

    assert ACTIONS[int(torch.argmax(logits))] == "WAIT"
    assert float(probabilities[:-1].sum()) < 1e-6


def test_loaded_actor_cannot_reverse_into_charger_without_a_charge_goal() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_625)
    state = environment.get_state()
    state.frame = 40
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    ai = state.by_id("robot_2")
    human.position = (5, 4)
    human.last_executed_action = "WAIT"
    carried_task = state.tasks[0]
    carried_task.status = "carried"
    carried_task.carrier_agent_id = ai.agent_id
    carried_task.delivery_position = (1, 2)
    ai.position = (6, 4)
    ai.battery = 80.0
    ai.carrying_task_id = carried_task.task_id
    ai.navigation_goal_kind = "delivery"
    ai.navigation_goal_position = carried_task.delivery_position
    ai.last_charger_departure_frame = state.frame - 4
    ai.deliveries_at_last_charger_departure = ai.deliveries_completed
    ai.team_deliveries_at_last_charger_departure = state.total_deliveries
    ai.carrying_task_at_last_charger_departure = ai.carrying_task_id
    environment.set_state(state)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_625),
        device="cpu",
    )

    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(
                environment.observations()["robot_2"][None],
                dtype=torch.float32,
            )
        )[0]
        probabilities = torch.softmax(logits, dim=-1)

    assert float(probabilities[ACTIONS.index("DOWN")]) < 1e-6


def test_one_observed_participant_stall_allows_only_a_robust_retreat() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=2_610)
    state = environment.get_state()
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    state.participant_controlled_agent_id = "robot_1"
    state.ineffective_joint_wait_streak = 1
    robot_one.position = (4, 4)
    robot_one.last_executed_action = "WAIT"
    robot_one.navigation_goal_kind = "delivery"
    robot_one.navigation_goal_position = (3, 2)
    robot_two.position = (4, 5)
    robot_two.battery = 74.0
    robot_two.carrying_task_id = "task_2"
    robot_two.navigation_goal_kind = "delivery"
    robot_two.navigation_goal_position = (2, 7)
    environment.set_state(state)

    assert necessary_participant_standoff_clearance(
        environment,
        state,
        robot_two,
        candidate_action="RIGHT",
    )
    assert robust_partner_robot_two_action(
        environment,
        preferred_action="WAIT",
    ) == "RIGHT"
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_610),
        device="cpu",
    )
    robot_two_observation = environment.observations()["robot_2"]
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(robot_two_observation[None], dtype=torch.float32)
        )[0]
    assert ACTIONS[int(torch.argmax(logits))] == "RIGHT"


def test_cautious_participant_retries_after_one_observed_joint_stall() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_611)
    state = environment.get_state()
    robot_one = state.by_id("robot_1")
    robot_two = state.by_id("robot_2")
    state.participant_controlled_agent_id = "robot_1"
    task, teammate_task = state.tasks
    task.status = "carried"
    task.carrier_agent_id = "robot_1"
    task.delivery_position = (4, 7)
    robot_one.position = (4, 6)
    robot_one.carrying_task_id = task.task_id
    robot_one.navigation_goal_kind = "delivery"
    robot_one.navigation_goal_position = (4, 7)
    robot_two.position = (4, 8)
    teammate_task.status = "carried"
    teammate_task.carrier_agent_id = "robot_2"
    teammate_task.delivery_position = robot_two.position
    robot_two.carrying_task_id = teammate_task.task_id
    robot_two.navigation_goal_kind = "delivery"
    robot_two.navigation_goal_position = robot_two.position
    environment.set_state(state)

    initial = participant_surrogate_distribution(
        environment,
        profile="cautious",
    )
    assert initial[ACTIONS.index("WAIT")] == pytest.approx(1.0)

    state.ineffective_joint_wait_streak = 1
    robot_one.last_executed_action = "WAIT"
    environment.set_state(state)
    retry = participant_surrogate_distribution(
        environment,
        profile="cautious",
    )
    assert retry[ACTIONS.index("WAIT")] == pytest.approx(0.0)
    assert float(retry[:-1].sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("profile", PARTNER_PROFILES)
def test_participant_does_not_leave_while_public_charge_mode_is_active(
    profile: str,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=2_617)
    state = environment.get_state()
    state.participant_controlled_agent_id = "robot_1"
    human = state.by_id("robot_1")
    teammate = state.by_id("robot_2")
    human.position = environment.layout.charger_position
    human.battery = 38.0
    human.charge_mode_active = True
    human.navigation_goal_kind = "charge"
    human.navigation_goal_position = environment.layout.charger_position
    human.last_executed_action = "WAIT"
    teammate.position = (6, 6)
    teammate.battery = 66.0
    teammate.charge_mode_active = False
    teammate.navigation_goal_kind = "wait"
    teammate.navigation_goal_position = teammate.position
    environment.set_state(state)

    probabilities = participant_surrogate_distribution(
        environment,
        profile=profile,
    )
    assert probabilities[ACTIONS.index("WAIT")] == pytest.approx(1.0)
