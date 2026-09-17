"""Deployable Web study backed by the real warehouse environment and Actor.

Unlike ``tests/browser_fixture_server.py``, this service never fabricates a
two-step round.  The demonstration and both interactive rounds use
``WarehouseMultiAgentEnv`` for task sampling, movement, collisions, charging,
scoring, task replacement, and the 120-step terminal boundary.

The training checkpoint is exported to a dependency-light NumPy artifact for
Render.  It evaluates the same decentralized neural Actor as the PyTorch
runtime, then applies the explicitly versioned frozen-state coordination
controller to Robot 2's submitted action.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Mapping
from uuid import uuid4

from backend.adapters.warehouse import WarehouseAdapter
from backend.adapters.warehouse_context import _transition_events
from core.policy_contracts import ActionDistribution
from env.warehouse.domain import (
    WarehouseConfig,
    WarehouseState,
    collaborative_study_config,
)
from env.warehouse.contracts import RUNTIME_CONTROLLER
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.numpy_policy import NumpyWarehousePolicy
from env.warehouse.runtime_coordination import (
    guard_participant_action,
    select_ai_ai_joint_actions,
    select_human_ai_action,
)
from ui.warehouse_view import (
    _study_question_focus,
    serialize_warehouse_state,
    warehouse_map_payload,
)


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "ui" / "web"
# Public NumPy tutorial seed, calibrated independently from scored rounds.
# With fixed stochastic Actor sampling it produces a complete productive
# mission with claiming, delivery, charging, and coordination-yield events.
TUTORIAL_SEED = 40_786
TASK1_SEED = 51_000
TASK2_SEED = 51_500
DEPLOYED_ACTOR_PATH = (
    ROOT / "output" / "deployment" / "warehouse_mappo_v68_6x7_actor.npz"
)
DEPLOYED_ACTOR = (
    NumpyWarehousePolicy.load(DEPLOYED_ACTOR_PATH)
    if DEPLOYED_ACTOR_PATH.exists()
    else None
)


def _require_deployed_actor() -> NumpyWarehousePolicy:
    if DEPLOYED_ACTOR is None:
        raise RuntimeError(
            "The causal-coordination v68 6x7 NumPy Actor has not been exported. "
            "Train and pass the release gates before starting the public study."
        )
    return DEPLOYED_ACTOR


@dataclass(frozen=True)
class PreviewFrame:
    state: WarehouseState
    actions: Mapping[str, str]
    goal_overrides: Mapping[str, tuple[int, int]]
    rewards: Mapping[str, float]
    info: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    transition: Mapping[str, Any] | None
    action_distributions: Mapping[str, ActionDistribution]


def _neural_actions(
    environment: WarehouseMultiAgentEnv,
    *,
    base_seed: int,
    deterministic: bool,
    participant_controlled: bool = False,
) -> tuple[
    dict[str, str],
    dict[str, ActionDistribution],
    dict[str, Any],
]:
    """Propose neural actions, then apply the frozen-state runtime protocol."""

    state = environment.get_state()
    policy_actions, distributions = _require_deployed_actor().act(
        environment.observations(),
        deterministic=deterministic,
        base_seed=base_seed,
        decision_key=(state.episode_id, state.frame),
    )
    if participant_controlled:
        robot_two, runtime = select_human_ai_action(
            environment,
            policy_actions["robot_2"],
        )
        return {**dict(policy_actions), "robot_2": robot_two}, distributions, runtime
    selected, runtime = select_ai_ai_joint_actions(environment, policy_actions)
    return selected, distributions, runtime


def _transition_payload(
    before: WarehouseState,
    after: WarehouseState,
    actions: Mapping[str, str],
    info: Mapping[str, Any],
    *,
    reveal_ai_action: bool,
    loop: bool,
) -> dict[str, Any]:
    executed = dict(info.get("executed_actions", {}))
    invalid = set(info.get("invalid_move_agents", ()))
    collision = bool(info.get("robot_collision_event", False))
    events = _transition_events(info)
    trace = info.get("decision_trace", {})
    runtime = (
        trace.get("runtime_decision", {})
        if isinstance(trace, Mapping)
        else {}
    )
    runtime = runtime if isinstance(runtime, Mapping) else {}
    policy_actions = dict(runtime.get("policy_actions", {}))
    selected_actions = dict(runtime.get("selected_actions", {}))
    agents: list[dict[str, Any]] = []
    for after_agent in after.agents:
        before_agent = before.by_id(after_agent.agent_id)
        proposed = str(actions.get(after_agent.agent_id, "WAIT"))
        actual = str(executed.get(after_agent.agent_id, after_agent.last_executed_action))
        nn_proposed = str(policy_actions.get(after_agent.agent_id, proposed))
        controller_selected = str(
            selected_actions.get(after_agent.agent_id, proposed)
        )
        changed = controller_selected != nn_proposed
        agents.append(
            {
                "id": after_agent.agent_id,
                "from_position": list(before_agent.position),
                "to_position": list(after_agent.position),
                "proposed_action": (
                    proposed
                    if reveal_ai_action or after_agent.agent_id == "robot_1"
                    else None
                ),
                "nn_proposed_action": (
                    nn_proposed
                    if reveal_ai_action or after_agent.agent_id == "robot_1"
                    else None
                ),
                "controller_selected_action": (
                    controller_selected
                    if reveal_ai_action or after_agent.agent_id == "robot_1"
                    else None
                ),
                "environment_executed_action": actual,
                "controller_changed_action": bool(changed),
                "controller_reason": (
                    str(runtime.get("controller_reason", ""))
                    if after_agent.agent_id == "robot_2"
                    else None
                ),
                "executed_action": actual,
                "battery_before": float(before_agent.battery),
                "battery_after": float(after_agent.battery),
                "battery_delta": float(after_agent.battery - before_agent.battery),
                "blocked": bool(
                    proposed in MOVE_DELTAS
                    and before_agent.position == after_agent.position
                    and actual == "WAIT"
                ),
                "invalid": after_agent.agent_id in invalid,
                "collision": collision,
                "charging": bool(
                    actual == "WAIT" and after_agent.battery > before_agent.battery
                ),
            }
        )
    return {
        "from_frame": int(before.frame),
        "to_frame": int(after.frame),
        "loop": bool(loop),
        "conflict_kind": info.get("robot_collision_kind"),
        "events": [dict(item) for item in events],
        "agents": agents,
        "before_state": serialize_warehouse_state(
            before,
            selected_agent=("robot_2" if reveal_ai_action else "robot_1"),
            actions={
                agent.agent_id: agent.last_executed_action
                for agent in before.agents
            },
            rewards=before.last_rewards,
            events=before.last_coordination_events,
            reveal_policy=reveal_ai_action,
        ),
    }


def _initial_frame(environment: WarehouseMultiAgentEnv) -> PreviewFrame:
    return PreviewFrame(
        state=environment.get_state(),
        actions={},
        goal_overrides={},
        rewards={},
        info={},
        events=(),
        transition=None,
        action_distributions={},
    )


def build_development_tutorial() -> tuple[PreviewFrame, ...]:
    """Generate one verified 120-step trajectory from the real simulator."""

    environment = WarehouseMultiAgentEnv(
        replace(collaborative_study_config(), participant_detour_scoring=False)
    )
    environment.reset(seed=TUTORIAL_SEED)
    state = environment.get_state()
    state.by_id("robot_2").battery = 35.0
    environment.set_state(state)
    frames = [_initial_frame(environment)]
    for _ in range(environment.config.horizon):
        before = environment.get_state()
        actions, distributions, runtime_decision = _neural_actions(
            environment,
            base_seed=TUTORIAL_SEED,
            deterministic=False,
        )
        _, rewards, terminated, truncated, info = environment.step(
            actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source="numpy_actor_plus_joint_optimizer_ai_ai",
                policy_actions=runtime_decision.get("policy_actions", {}),
                selected_actions=actions,
                runtime_decision=runtime_decision,
            ),
        )
        after = environment.get_state()
        events = _transition_events(info)
        frames.append(
            PreviewFrame(
                state=after,
                actions=dict(actions),
                goal_overrides={},
                rewards=dict(rewards),
                info=deepcopy(dict(info)),
                events=tuple(deepcopy(events)),
                transition=_transition_payload(
                    before,
                    after,
                    actions,
                    info,
                    reveal_ai_action=True,
                    loop=False,
                )
                | {"before_stage": "instructions"},
                action_distributions=dict(distributions),
            )
        )
        if terminated or truncated:
            break
    final = frames[-1].state
    if (
        len(frames) != environment.config.horizon + 1
        or final.frame != environment.config.horizon
        or final.terminal_reason != "horizon"
        or final.total_deliveries < 1
    ):
        raise RuntimeError(
            "Development tutorial is not one complete, productive 120-step round."
        )
    required_events = {"claimed", "delivered", "charging", "coordination_yield"}
    observed_events = {
        str(event.get("event", ""))
        for frame in frames
        for event in frame.events
    }
    missing = sorted(required_events - observed_events)
    if missing:
        raise RuntimeError(
            "Development tutorial is missing required events: " + ", ".join(missing)
        )
    return tuple(frames)


def _event_tags(events: tuple[Mapping[str, Any], ...]) -> list[str]:
    labels = {
        "claimed": "pickup",
        "delivered": "delivery",
        "charging": "charging",
        "charger_queue": "charger_queue",
        "coordination_yield": "yield",
        "collision_risk": "conflict",
        "head_on_conflict_risk": "conflict",
        "robot_collision": "collision",
        "shared_charger_occupancy": "charger_penalty",
    }
    result: list[str] = []
    for event in events:
        label = labels.get(str(event.get("event", "")))
        if label and label not in result:
            result.append(label)
    return result


class DevelopmentPreviewState:
    """Single-user development state for the production browser assets."""

    def __init__(
        self,
        *,
        tutorial_frames: tuple[PreviewFrame, ...] | None = None,
        reference_payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.lock = threading.RLock()
        self.tutorial_frames = (
            tutorial_frames
            if tutorial_frames is not None
            else build_development_tutorial()
        )
        self.environment = WarehouseMultiAgentEnv(collaborative_study_config())
        self.environment.reset(seed=TASK1_SEED)
        self.policy_seed = TASK1_SEED
        self.round_frame = _initial_frame(self.environment)
        self.stage = "idle"
        self.version = 0
        self.run_id: str | None = None
        self.locale = "en"
        self.condition = "explanation"
        self.participant_id = ""
        self.tutorial_index = 0
        self.tutorial_max_index = 0
        self.task1: dict[str, Any] | None = None
        self.task2: dict[str, Any] | None = None
        self.last_explanation: dict[str, Any] | None = None
        self.action_bubble: dict[str, Any] | None = None
        # The latest penalty is a durable, frame-bound UI affordance.  It is
        # kept independently from the five-step explanation context so a
        # participant can ask about a penalty several steps after it fired.
        self.latest_charger_penalty: dict[str, Any] | None = None
        self.explanation_count = 0
        self.round_frames: list[PreviewFrame] = [self.round_frame]
        self.question_log: list[dict[str, Any]] = []
        self.pending_question_sequences: list[int] = []
        self.operations: dict[str, dict[str, Any]] = {}
        self.command_requests: list[str] = []
        self.reference_requests = 0
        self.reference_payload = (
            reference_payload
            if reference_payload is not None
            else self._build_reference_payload()
        )

    def _build_reference_payload(self) -> dict[str, Any]:
        actor = _require_deployed_actor()
        frames = [
            {
                "index": index,
                "state": serialize_warehouse_state(
                    frame.state,
                    selected_agent="robot_2",
                    actions=frame.actions,
                    rewards=frame.rewards,
                    events=frame.events,
                    reveal_policy=True,
                ),
                "transition": (
                    {**frame.transition, "loop": index > 0}
                    if frame.transition is not None
                    else None
                ),
                "event_tags": _event_tags(frame.events),
            }
            for index, frame in enumerate(self.tutorial_frames)
        ]
        identity = {
            "schema_version": "warehouse-reference-timeline.v2",
            "trajectory_kind": "ai_ai_demonstration",
            "trajectory_seed": TUTORIAL_SEED,
            "agent_control": {"robot_1": "ai", "robot_2": "ai"},
            "policy_model_version": actor.metadata.model_version,
            "policy_artifact_sha256": actor.artifact_sha256,
            "map_layout_id": self.environment.layout.layout_id,
            "map": warehouse_map_payload(self.environment.layout),
            "frames": frames,
        }
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {**identity, "trajectory_hash": sha256(encoded).hexdigest()}

    def commands(self) -> tuple[str, ...]:
        task1_commands = (
            ("human_action", "ask_explanation")
            if self.condition == "explanation"
            else ("human_action",)
        )
        values = {
            "idle": (),
            "instructions": (
                "tutorial_advance",
                "tutorial_restart",
                "tutorial_select",
                "begin_task1",
            ),
            "task1": task1_commands,
            "task1_complete": ("begin_task2",),
            "task2": ("human_action",),
            "survey": ("submit_survey",),
            "completed": (),
        }[self.stage]
        return ("set_language", "restart", *values) if self.stage != "idle" else values

    def _start_round(self, stage: str, seed: int) -> None:
        self.environment = WarehouseMultiAgentEnv(collaborative_study_config())
        self.environment.reset(seed=seed)
        participant_state = self.environment.get_state()
        participant_state.participant_controlled_agent_id = (
            self.environment.config.human_agent_id
        )
        self.environment.set_state(participant_state)
        self.policy_seed = int(seed)
        self.round_frame = _initial_frame(self.environment)
        self.round_frames = [self.round_frame]
        self.stage = stage
        self.last_explanation = None
        self.action_bubble = None
        self.latest_charger_penalty = None
        self.pending_question_sequences = []

    @staticmethod
    def _summary(round_name: str, seed: int, state: WarehouseState) -> dict[str, Any]:
        latencies = [
            float(task.delivered_frame - task.claimed_frame)
            for task in state.completed_tasks
            if task.delivered_frame is not None and task.claimed_frame is not None
        ]
        return {
            "round_name": round_name,
            "seed": int(seed),
            "score": float(state.user_score),
            "steps": int(state.frame),
            "deliveries": int(state.total_deliveries),
            "robot_collisions": int(state.robot_collision_events),
            "shutdowns": int(state.shutdown_count),
            "shared_charger_penalties": float(
                state.score_breakdown.get("shared_charger_occupancy", 0.0)
            ),
            "human_route_regret_units": float(state.human_route_regret_units),
            "mean_delivery_latency": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "terminal_reason": state.terminal_reason,
        }

    def _advance_round(self, action: str) -> None:
        if action not in ACTIONS:
            raise ValueError(f"Unknown participant action: {action}")
        round_name = self.stage
        before = self.environment.get_state()
        actions, distributions, runtime_decision = _neural_actions(
            self.environment,
            base_seed=self.policy_seed,
            deterministic=False,
            participant_controlled=True,
        )
        # The preview AI chooses from the frozen pre-move state without the
        # participant's current command. Replace robot 1 only after robot 2's
        # action has been selected, then resolve the pair simultaneously.
        participant_action, participant_guard = guard_participant_action(
            self.environment,
            action,
        )
        actions["robot_1"] = participant_action
        runtime_decision = {
            **runtime_decision,
            "participant_action_guard": participant_guard,
            "selected_actions": dict(actions),
        }
        _, rewards, terminated, truncated, info = self.environment.step(
            actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source="participant_plus_robust_numpy_actor",
                participant_overrides={"robot_1": action},
                policy_actions=runtime_decision.get("policy_actions", {}),
                selected_actions=actions,
                runtime_decision=runtime_decision,
            ),
        )
        after = self.environment.get_state()
        events = _transition_events(info)
        for event in events:
            if str(event.get("event", "")) == "shared_charger_occupancy":
                self.latest_charger_penalty = {
                    **deepcopy(dict(event)),
                    "event_id": str(event.get("event_id", "")),
                    "frame": int(after.frame),
                }
        transition_payload = _transition_payload(
            before,
            after,
            actions,
            info,
            reveal_ai_action=False,
            loop=False,
        ) | {"before_stage": round_name}
        self.round_frame = PreviewFrame(
            state=after,
            actions=dict(actions),
            goal_overrides={},
            rewards=dict(rewards),
            info=deepcopy(dict(info)),
            events=tuple(deepcopy(events)),
            transition=transition_payload,
            action_distributions=dict(distributions),
        )
        self.round_frames.append(self.round_frame)
        self._refresh_action_bubble()
        for sequence in self.pending_question_sequences:
            for record in reversed(self.question_log):
                if int(record.get("question_sequence", -1)) == sequence:
                    record["post_question_action"] = action
                    record["post_question_frame"] = int(after.frame)
                    break
        self.pending_question_sequences = []
        if not (terminated or truncated):
            return
        if round_name == "task1":
            self.task1 = self._summary("task1", TASK1_SEED, after)
            self.stage = "task1_complete"
        else:
            self.task2 = self._summary("task2", TASK2_SEED, after)
            self.stage = "survey"

    def _display_frame(self) -> tuple[PreviewFrame, bool, str]:
        if self.stage == "instructions":
            return self.tutorial_frames[self.tutorial_index], True, "ai_ai_demonstration"
        trajectory_kind = (
            "human_ai_task1"
            if self.stage in {"task1", "task1_complete"}
            else "human_ai_task2"
        )
        return self.round_frame, self.stage not in {"task1", "task2"}, trajectory_kind

    def view(self) -> dict[str, Any]:
        actor = _require_deployed_actor()
        frame, reveal_policy, trajectory_kind = self._display_frame()
        state_payload = serialize_warehouse_state(
            frame.state,
            selected_agent=("robot_2" if reveal_policy else "robot_1"),
            actions=frame.actions,
            rewards=frame.rewards,
            events=frame.events,
            reveal_policy=reveal_policy,
        )
        summaries = {
            key: value
            for key, value in (("task1", self.task1), ("task2", self.task2))
            if value is not None
        }
        tutorial_count = len(self.tutorial_frames)
        timeline_index = self.tutorial_index if self.stage == "instructions" else frame.state.frame
        timeline_count = tutorial_count if self.stage == "instructions" else len(self.round_frames)
        return {
            "map": warehouse_map_payload(self.environment.layout),
            "state": state_payload,
            "transition": frame.transition,
            "selected_agent": state_payload["selected_agent"],
            "agent_ids": list(self.environment.agent_ids),
            "timeline": {
                "index": int(timeline_index),
                "max_index": timeline_count - 1,
                "count": timeline_count,
                "simulator_frame": int(frame.state.frame),
                "trajectory_kind": trajectory_kind,
                "trajectory_seed": (
                    TUTORIAL_SEED
                    if self.stage == "instructions"
                    else TASK1_SEED if self.stage in {"task1", "task1_complete"} else TASK2_SEED
                ),
                "agent_control": (
                    {"robot_1": "ai", "robot_2": "ai"}
                    if self.stage == "instructions"
                    else {"robot_1": "human", "robot_2": "ai"}
                ),
            },
            "study": {
                "release_version": "sep2-rule-assisted-20260917-hotfix1",
                "pilot_class": "internal_pre_experiment",
                "formal_ready": False,
                "run_id": self.run_id,
                "stage": self.stage,
                "state_version": self.version,
                "locale": self.locale,
                "participant_id": self.participant_id,
                "condition": self.condition if self.stage != "idle" else None,
                "group_code": (
                    "A" if self.condition == "explanation" else "B"
                ) if self.stage != "idle" else None,
                "group_explanation_available": (
                    self.condition == "explanation"
                    if self.stage != "idle" else None
                ),
                "test_condition_selector": True,
                "development_controller": (
                    RUNTIME_CONTROLLER
                ),
                "runtime_action_mode": "nn_proposal_plus_rule_assisted_robot_2",
                "rule_version": "shared-charger-occupancy.v1",
                "action_override_authorized": True,
                "action_override_count": sum(
                    int(
                        bool(
                            frame.info.get("decision_trace", {})
                            .get("runtime_decision", {})
                            .get("selection_changed_policy", False)
                        )
                    )
                    for frame in self.round_frames
                    if isinstance(frame.info.get("decision_trace", {}), Mapping)
                ),
                "formal_policy_loaded": True,
                "policy_model_version": actor.metadata.model_version,
                "policy_artifact_sha256": actor.artifact_sha256,
                "progress": (
                    int(self.environment.state.frame)
                    if self.stage in {"task1", "task2"}
                    and self.environment.state is not None
                    else 0
                ),
                "total": int(self.environment.config.horizon),
                "round_summaries": summaries,
                "score_delta": (
                    self.task2["score"] - self.task1["score"]
                    if self.task1 is not None and self.task2 is not None
                    and self.stage == "completed"
                    else None
                ),
                "explanation_presented": self.explanation_count > 0,
                "explanation_count": self.explanation_count,
                "live_explanation_available": bool(
                    self.stage == "task1" and self.condition == "explanation"
                ),
                "action_bubble": (
                    self._localized_action_bubble()
                    if self.stage == "task1" and self.condition == "explanation"
                    else None
                ),
                "latest_charger_penalty": (
                    deepcopy(self.latest_charger_penalty)
                    if self.stage == "task1" and self.condition == "explanation"
                    else None
                ),
                "controlled_agent": "robot_1",
                "explanation_target_agent": "robot_2",
                "explanation_target_agents": ["robot_2"],
                "tutorial": {
                    "frame_index": self.tutorial_index,
                    "index": self.tutorial_index,
                    "max_played_index": self.tutorial_max_index,
                    "total_frames": tutorial_count,
                    "seed": TUTORIAL_SEED,
                    "continuous_mission": True,
                    "mission_start_frame": 0,
                    "mission_end_frame": self.tutorial_frames[-1].state.frame,
                    "complete": self.tutorial_index >= tutorial_count - 1,
                },
                "survey_submitted": self.stage == "completed",
                "allowed_commands": list(self.commands()),
            },
            "trial": None,
            "last_explanation": self._localized_explanation(),
        }

    def _refresh_action_bubble(self) -> None:
        """Create the concise per-step Group A label from the executed decision."""

        self.action_bubble = None
        if self.stage != "task1" or self.condition != "explanation":
            return
        index = len(self.round_frames) - 1
        if index < 1:
            return
        try:
            adapter, snapshot = self._round_explanation_snapshot(index)
            variants = {
                language: adapter.concise_study_explanation(
                    snapshot,
                    target_agent="robot_2",
                    policy=None,
                    focus="action",
                    language=language,
                )
                for language in ("en", "zh-CN")
            }
            outcome = self.round_frames[index]
            trace = outcome.info.get("decision_trace", {})
            runtime = trace.get("runtime_decision", {}) if isinstance(trace, Mapping) else {}
            runtime = runtime if isinstance(runtime, Mapping) else {}
            if runtime.get("mode") == "human_ai_robust_selection":
                variants = self._controller_action_bubble(
                    self.round_frames[index - 1].state,
                    outcome.state,
                    outcome,
                    runtime,
                )
        except Exception:
            # A bubble must never stop a confirmed movement.  This fallback
            # reports only the action that the environment recorded.
            outcome = self.round_frames[index]
            action = str(outcome.info.get("executed_actions", {}).get(
                "robot_2", outcome.actions.get("robot_2", "WAIT")
            ))
            en, zh = self._action_words(action)
            variants = {
                "en": f"Robot 2 moved {en} in this step.",
                "zh-CN": f"机器人2本步向{zh}移动。",
            }
        self.action_bubble = {
            "target_agent": "robot_2",
            "frame": int(self.round_frames[index].state.frame),
            "_language_variants": variants,
        }

    def _localized_action_bubble(self) -> dict[str, Any] | None:
        if self.action_bubble is None:
            return None
        variants = self.action_bubble.get("_language_variants", {})
        locale = "zh-CN" if self.locale != "en" else "en"
        text = variants.get(locale) if isinstance(variants, Mapping) else None
        return {
            "target_agent": "robot_2",
            "frame": self.action_bubble.get("frame"),
            "text": str(text or ""),
        }

    def _localized_explanation(self) -> dict[str, Any] | None:
        if self.last_explanation is None:
            return None
        variants = self.last_explanation.get("_language_variants", {})
        locale = "zh-CN" if self.locale != "en" else "en"
        selected = variants.get(locale) if isinstance(variants, Mapping) else None
        return dict(selected) if isinstance(selected, Mapping) else dict(self.last_explanation)

    def reference_trajectory(self) -> dict[str, Any]:
        if self.stage != "instructions":
            raise RuntimeError(
                "The AI-AI trajectory is available only as the instructions demonstration."
            )
        self.reference_requests += 1
        return self.reference_payload

    def _round_explanation_snapshot(
        self,
        index: int,
    ) -> tuple[WarehouseAdapter, Any]:
        """Rebuild evidence for one executed Human-AI Task 1 transition."""

        if index < 1 or index >= len(self.round_frames):
            raise ValueError("The requested Task 1 action has not been executed.")
        before = self.round_frames[index - 1]
        outcome = self.round_frames[index]
        environment = WarehouseMultiAgentEnv(collaborative_study_config())
        environment.reset(seed=TASK1_SEED)
        environment.set_state(before.state)
        actor = _require_deployed_actor()
        adapter = WarehouseAdapter(environment)
        snapshot = adapter.snapshot(None)
        return adapter, replace(
            snapshot,
            action_distributions=dict(outcome.action_distributions),
            proposed_actions=dict(outcome.actions),
            executed_actions=dict(outcome.info.get("executed_actions", {})),
            rewards=dict(outcome.rewards),
            metadata={
                **dict(snapshot.metadata),
                "decision_evidence_aligned": True,
                "decision_outcome_frame": int(outcome.state.frame),
                "decision_goal_overrides": {
                    agent_id: tuple(position)
                    for agent_id, position in outcome.goal_overrides.items()
                },
                "action_resolution": dict(
                    outcome.info.get("action_resolution", {})
                ),
                "decision_trace": deepcopy(
                    outcome.info.get("decision_trace", {})
                ),
                "environment_events": tuple(outcome.events),
                "development_controller": (
                    RUNTIME_CONTROLLER
                ),
                "policy_model_version": actor.metadata.model_version,
                "policy_artifact_sha256": actor.artifact_sha256,
            },
        )

    @staticmethod
    def _action_words(action: str) -> tuple[str, str]:
        en = {
            "UP": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
            "WAIT": "wait",
        }
        zh = {
            "UP": "上",
            "DOWN": "下",
            "LEFT": "左",
            "RIGHT": "右",
            "WAIT": "等待",
        }
        return en.get(action, action.lower()), zh.get(action, action)

    def _controller_action_bubble(
        self,
        before: WarehouseState,
        after: WarehouseState,
        outcome: PreviewFrame,
        runtime: Mapping[str, Any],
    ) -> dict[str, str]:
        """Render runtime facts as public prose, never as reason codes."""

        before_agent = before.by_id("robot_2")
        after_agent = after.by_id("robot_2")
        selected = str(
            runtime.get("selected_actions", {}).get(
                "robot_2", outcome.actions.get("robot_2", "WAIT")
            )
        )
        actual = str(
            outcome.info.get("executed_actions", {}).get(
                "robot_2", selected
            )
        )
        selected_record = runtime.get("selected_ai_action", {})
        selected_record = (
            selected_record if isinstance(selected_record, Mapping) else {}
        )
        task_id = str(
            before_agent.carrying_task_id
            or before_agent.route_commitment_task_id
            or before_agent.goal_id
            or ""
        )
        task_number = task_id.removeprefix("task_") if task_id else ""
        en_task = f"task {task_number}" if task_number else "the current task"
        zh_task = f"任务{task_number}" if task_number else "当前任务"
        en_action, zh_action = self._action_words(actual)
        en_selected, zh_selected = self._action_words(selected)

        collision = bool(
            outcome.info.get("robot_collision_event")
            and actual == "WAIT"
            and selected in MOVE_DELTAS
        )
        if collision:
            return {
                "en": (
                    f"Robot 2 tried to move {en_selected}, but the joint move "
                    "collided and the environment cancelled it."
                ),
                "zh-CN": (
                    f"机器人2本步尝试向{zh_selected}，但联合移动发生碰撞，"
                    "环境取消了这一步。"
                ),
            }

        if actual == "WAIT":
            battery_before = float(before_agent.battery)
            battery_after = float(after_agent.battery)
            if battery_after > battery_before:
                return {
                    "en": (
                        f"Robot 2 is charging; its battery rose from "
                        f"{battery_before:g}% to {battery_after:g}%."
                    ),
                    "zh-CN": (
                        f"机器人2正在充电，电量从{battery_before:g}%恢复到"
                        f"{battery_after:g}%。"
                    ),
                }
            if bool(runtime.get("participant_occupies_charger")):
                return {
                    "en": (
                        "You are occupying the charger, so Robot 2 waits "
                        "beside it instead of entering your cell."
                    ),
                    "zh-CN": (
                        "充电桩当前被你占用，机器人2先在旁边等待，"
                        "避免进入同一格发生碰撞。"
                    ),
                }
            if bool(runtime.get("pass_through_plan_current")):
                return {
                    "en": (
                        "Robot 2 waits for the already-started pass-through "
                        "step, then will recheck the route."
                    ),
                    "zh-CN": (
                        "机器人2暂时等待已开始的通行步骤完成，下一步重新检查路线。"
                    ),
                }
            if bool(selected_record.get("energy_violation")):
                return {
                    "en": (
                        "Robot 2 keeps its battery because the available move "
                        "would not preserve a feasible return to charging."
                    ),
                    "zh-CN": (
                        "当前移动会破坏返桩电量余量，机器人2先保留电量。"
                    ),
                }
            if bool(runtime.get("physical_clearance_required")):
                return {
                    "en": (
                        "Robot 2 waits for the confirmed aisle clearance "
                        "shown in the current state."
                    ),
                    "zh-CN": "当前状态仍有已确认的通道阻挡，机器人2先等待让路。",
                }
            return {
                "en": (
                    "Robot 2 waits for this step; no movement was executed, "
                    "and the route will be checked again next step."
                ),
                "zh-CN": "机器人2本步暂时停留，没有执行移动，下一步会重新检查路线。",
            }

        goal_kind = str(
            before_agent.navigation_goal_kind
            or before_agent.goal_type
            or ""
        ).lower()
        if before_agent.carrying_task_id is not None or goal_kind == "delivery":
            en_reason = f"Robot 2 moved {en_action} to deliver {en_task}."
            zh_reason = f"机器人2向{zh_action}移动，把{zh_task}的货物送往交付点。"
        elif goal_kind in {"charge", "go_to_charger"}:
            en_reason = f"Robot 2 moved {en_action} toward the charger."
            zh_reason = f"机器人2向{zh_action}移动，前往充电桩补电。"
        else:
            en_reason = f"Robot 2 moved {en_action} toward {en_task}'s pickup point."
            zh_reason = f"机器人2向{zh_action}移动，前往{zh_task}的取货点。"

        if bool(runtime.get("selection_changed_policy")):
            en_reason += " The current occupied cells and route were checked before this move."
            zh_reason += " 这一步先检查了当前占用位置和可通行路线。"
        return {"en": en_reason, "zh-CN": zh_reason}

    def _ask_live_task1(
        self,
        *,
        question: str,
        focus: str,
        penalty_event_id: str | None = None,
        penalty_frame: int | None = None,
    ) -> None:
        """Answer from completed Task 1 transitions, never from the future.

        Ordinary questions intentionally use a compact recent context.  A
        charger-penalty question is different: it is anchored to the exact
        durable penalty event selected by the UI, so it remains answerable
        after more than five subsequent steps.
        """

        if self.stage != "task1" or self.condition != "explanation":
            raise RuntimeError("Live questions are available only to Group A during Task 1.")

        completed = list(enumerate(self.round_frames))[1:]
        recent = completed[-5:]

        def proposed(frame: PreviewFrame, agent_id: str) -> str:
            return str(frame.actions.get(agent_id, "WAIT"))

        def executed(frame: PreviewFrame, agent_id: str) -> str:
            values = frame.info.get("executed_actions", {})
            values = values if isinstance(values, Mapping) else {}
            return str(values.get(agent_id, proposed(frame, agent_id)))

        anchor: tuple[int, PreviewFrame] | None = recent[-1] if recent else None
        if focus == "wait":
            anchor = next(
                (
                    item
                    for item in reversed(recent)
                    if proposed(item[1], "robot_2") == "WAIT"
                    or executed(item[1], "robot_2") == "WAIT"
                ),
                None,
            )
        elif focus == "collision":
            anchor = next(
                (
                    item
                    for item in reversed(recent)
                    if bool(item[1].info.get("robot_collision_event", False))
                    or bool(item[1].info.get("robot_collision_kind"))
                ),
                None,
            )
        elif focus == "charger_penalty":
            penalty_matches: list[tuple[int, PreviewFrame, Mapping[str, Any]]] = []
            for item in completed:
                for event in item[1].info.get("rule_events", ()):
                    if not isinstance(event, Mapping):
                        continue
                    if str(event.get("event", "")) != "shared_charger_occupancy":
                        continue
                    if penalty_event_id and str(event.get("event_id", "")) != str(penalty_event_id):
                        continue
                    if penalty_frame is not None and int(item[1].state.frame) != int(penalty_frame):
                        continue
                    penalty_matches.append((item[0], item[1], event))
            if penalty_matches:
                anchor = (penalty_matches[-1][0], penalty_matches[-1][1])

        current_frame = int(self.round_frame.state.frame)
        if anchor is None:
            anchor_index = current_frame
            anchor_frame = current_frame
            context_frames = [int(frame.state.frame) for _, frame in recent]
            if focus == "collision":
                answer_en = "No collision occurred in the last five steps."
                answer_zh = "最近五步内没有发生碰撞。"
            elif focus == "charger_penalty":
                if penalty_event_id or penalty_frame is not None:
                    answer_en = "That shared-charger penalty event is not available in this Task 1 run."
                    answer_zh = "这次任务1中没有找到所选的共享充电桩处罚事件。"
                else:
                    answer_en = "No shared-charger penalty has been recorded in this Task 1 run."
                    answer_zh = "本次任务1还没有记录共享充电桩处罚。"
            elif focus == "wait":
                answer_en = "Robot 2 did not wait in the last five steps."
                answer_zh = "机器人2在最近五步内没有等待。"
            else:
                answer_en = "Robot 2 has not completed an action in Task 1 yet."
                answer_zh = "机器人2在任务1中还没有完成任何动作。"
            evidence: dict[str, Any] = {
                "event_type": focus,
                "anchor_frame": anchor_frame,
                "context_frames": context_frames,
                "reason_code": (
                    "NO_MATCHING_PENALTY_EVENT"
                    if focus == "charger_penalty"
                    else "NO_MATCHING_RECENT_EVENT"
                ),
                "fact_valid": True,
            }
            recent_collision = False
        else:
            anchor_index, outcome = anchor
            anchor_frame = int(outcome.state.frame)
            context = [
                item
                for item in completed
                if int(item[1].state.frame) <= anchor_frame
            ][-5:]
            context_frames = [int(frame.state.frame) for _, frame in context]
            adapter, snapshot = self._round_explanation_snapshot(anchor_index)
            trace = snapshot.metadata.get("decision_trace", {})
            trace = trace if isinstance(trace, Mapping) else {}
            agents = trace.get("agents", {})
            agents = agents if isinstance(agents, Mapping) else {}
            agent_trace = agents.get("robot_2", {})
            agent_trace = agent_trace if isinstance(agent_trace, Mapping) else {}
            frozen_goal = agent_trace.get("frozen_goal", {})
            frozen_goal = frozen_goal if isinstance(frozen_goal, Mapping) else {}
            charging = agent_trace.get("charging_state", {})
            charging = charging if isinstance(charging, Mapping) else {}
            feasibility = [
                item
                for item in agent_trace.get("battery_feasibility", ())
                if isinstance(item, Mapping)
            ]
            goal_id = str(
                frozen_goal.get("goal_id")
                or agent_trace.get("committed_task")
                or ""
            )
            energy_item = next(
                (
                    item
                    for item in feasibility
                    if str(item.get("task_id", "")) == goal_id
                ),
                min(
                    feasibility,
                    key=lambda item: float(item.get("required_energy", 0.0)),
                )
                if feasibility
                else {},
            )
            plan = agent_trace.get("joint_coordination_plan")
            plan = plan if isinstance(plan, Mapping) else {}
            recent_collision = bool(
                outcome.info.get("robot_collision_event", False)
                or outcome.info.get("robot_collision_kind")
            )
            penalty_events = tuple(
                event
                for event in outcome.info.get("rule_events", ())
                if isinstance(event, Mapping)
                and str(event.get("event", "")) == "shared_charger_occupancy"
            )
            evidence = {
                "event_type": focus,
                "anchor_frame": anchor_frame,
                "context_frames": context_frames,
                "human_action": proposed(outcome, "robot_1"),
                "ai_action": proposed(outcome, "robot_2"),
                "executed_actions": dict(snapshot.executed_actions),
                "collision_type": outcome.info.get("robot_collision_kind"),
                "current_goal": (
                    frozen_goal.get("navigation_kind")
                    or frozen_goal.get("goal_type")
                ),
                "current_battery": charging.get("battery"),
                "required_energy": energy_item.get("required_energy"),
                "priority_basis": plan.get("priority_basis"),
                "task_id": goal_id or energy_item.get("task_id"),
                "reason_code": agent_trace.get("primary_reason_code"),
                "pre_state_hash": trace.get("pre_state_hash"),
                "outcome_frame": trace.get("outcome_frame"),
                "fact_valid": bool(trace.get("fact_valid", False)),
                "shared_charger_penalty_events": [
                    dict(event) for event in penalty_events
                ],
            }
            if focus == "charger_penalty":
                event = penalty_events[-1] if penalty_events else {}
                evidence["penalty_event_id"] = str(event.get("event_id", ""))
                responsible = str(event.get("responsible_agent_id", "robot_2"))
                occupant_before = float(event.get("occupant_battery_before", 0.0))
                occupant_after = float(event.get("occupant_battery_after", occupant_before))
                teammate_battery = float(event.get("teammate_battery_before", 0.0))
                distance = int(event.get("teammate_distance_to_charger", 0))
                if responsible == "robot_2":
                    answer_en = (
                        f"Robot 2 remained on the charger after charging from {occupant_before:g}% "
                        f"to {occupant_after:g}%. You had {teammate_battery:g}% battery and were "
                        f"{distance} cells away, so Robot 2's continued occupancy triggered the −50 penalty."
                    )
                    answer_zh = (
                        f"机器人2在充电后仍占用充电桩，电量从{occupant_before:g}%升至{occupant_after:g}%。"
                        f"你当时电量为{teammate_battery:g}%，距充电桩{distance}格，因此本步触发−50处罚。"
                    )
                else:
                    answer_en = (
                        f"Robot 1 remained on the charger after charging from {occupant_before:g}% "
                        f"to {occupant_after:g}%. Robot 2 had {teammate_battery:g}% battery and was "
                        f"{distance} cells away, so the shared-charger rule triggered −50."
                    )
                    answer_zh = (
                        f"机器人1在充电后仍占用充电桩，电量从{occupant_before:g}%升至{occupant_after:g}%。"
                        f"机器人2当时电量为{teammate_battery:g}%，距充电桩{distance}格，因此触发共享充电桩−50处罚。"
                    )
            elif focus == "collision" and recent_collision:
                human_action = proposed(outcome, "robot_1")
                ai_action = proposed(outcome, "robot_2")
                human_en, human_zh = self._action_words(human_action)
                ai_en, ai_zh = self._action_words(ai_action)
                kind = str(
                    outcome.info.get("robot_collision_kind") or "joint_conflict"
                )
                kind_en = {
                    "same_target": "a same-target-cell conflict",
                    "swap": "a position-swap conflict",
                    "occupied_stationary": "an occupied-cell conflict",
                }.get(kind, "a joint movement conflict")
                kind_zh = {
                    "same_target": "同一目标格冲突",
                    "swap": "位置交换冲突",
                    "occupied_stationary": "进入未释放格子的冲突",
                }.get(kind, "联合移动冲突")
                answer_en = (
                    f"You moved {human_en} while Robot 2 simultaneously moved {ai_en}; "
                    f"the joint resolver recorded {kind_en}. Robot 2 decided from the "
                    "shared pre-move state and did not observe your current action first."
                )
                answer_zh = (
                    f"你向{human_zh}移动，机器人2同时向{ai_zh}移动；联合解析器记录为{kind_zh}。"
                    "机器人2依据共同的移动前状态决策，没有提前看到你的本帧动作。"
                )
            else:
                routed_focus = {
                    "wait": "action",
                    "human_influence": "collaboration",
                    "goal": "task",
                }.get(focus, focus)
                answer_en = adapter.concise_study_explanation(
                    snapshot,
                    target_agent="robot_2",
                    policy=None,
                    focus=routed_focus,
                    language="en",
                )
                answer_zh = adapter.concise_study_explanation(
                    snapshot,
                    target_agent="robot_2",
                    policy=None,
                    focus=routed_focus,
                    language="zh-CN",
                )

        self.explanation_count += 1
        sequence = self.explanation_count
        common = {
            "target_agent": "robot_2",
            "question": question,
            "question_sequence": sequence,
            "question_focus": focus,
            "selected_timeline_frame": anchor_frame,
            "decision_evidence_frame": anchor_frame,
            "anchor_frame": anchor_frame,
            "context_frames": context_frames,
            "current_frame": current_frame,
            "trajectory_kind": "human_ai_task1",
            "trajectory_seed": TASK1_SEED,
            "agent_control": {"robot_1": "human", "robot_2": "ai"},
            "structured_evidence": evidence,
            "fact_validation": {
                "passed": bool(evidence.get("fact_valid", False)),
                "future_frames_used": False,
                "max_context_frame": max(context_frames, default=anchor_frame),
            },
            "recent_collision": recent_collision,
            "answer_en": answer_en,
            "answer_zh": answer_zh,
        }
        variants = {
            "en": {
                **common,
                "explanation": answer_en,
                "explanation_document": {
                    "schema_version": "live-human-ai-explanation.v1",
                    "text": answer_en,
                },
            },
            "zh-CN": {
                **common,
                "explanation": answer_zh,
                "explanation_document": {
                    "schema_version": "live-human-ai-explanation.v1",
                    "text": answer_zh,
                },
            },
        }
        selected_locale = "zh-CN" if self.locale != "en" else "en"
        report = dict(variants[selected_locale])
        report["_language_variants"] = variants
        report["language_documents"] = {
            language: variant["explanation_document"]
            for language, variant in variants.items()
        }
        self.last_explanation = report
        self.question_log.append(
            {
                "participant_id": self.participant_id,
                "condition": self.condition,
                "round": "task1",
                "question_sequence": sequence,
                "question": question,
                "question_focus": focus,
                "penalty_event_id": (
                    evidence.get("penalty_event_id")
                    if focus == "charger_penalty"
                    else None
                ),
                "target_agent": "robot_2",
                "current_frame": current_frame,
                "anchor_frame": anchor_frame,
                "context_frames": context_frames,
                "trajectory_kind": "human_ai_task1",
                "agent_control": {"robot_1": "human", "robot_2": "ai"},
                "answer_en": answer_en,
                "answer_zh": answer_zh,
                "structured_evidence": deepcopy(evidence),
                "fact_validation": deepcopy(common["fact_validation"]),
                "post_question_action": None,
                "post_question_frame": None,
            }
        )
        self.pending_question_sequences.append(sequence)

    def command(self, envelope: Mapping[str, Any]) -> dict[str, Any]:
        operation = str(envelope["operation_id"])
        if operation in self.operations:
            return self.operations[operation]
        command = str(envelope["command"])
        payload = dict(envelope.get("payload", {}))
        self.command_requests.append(command)
        if command in {"start", "restart"}:
            self.participant_id = str(payload.get("participant_id", "development-preview"))
            override = str(payload.get("condition_override", "auto"))
            self.condition = (
                override
                if override in {"control", "explanation"}
                else "explanation"
            )
            self.locale = str(payload.get("locale", "en"))
            self.run_id = uuid4().hex
            self.stage = "instructions"
            self.tutorial_index = 0
            self.tutorial_max_index = 0
            self.task1 = None
            self.task2 = None
            self.last_explanation = None
            self.action_bubble = None
            self.latest_charger_penalty = None
            self.explanation_count = 0
            self.question_log = []
            self.pending_question_sequences = []
            self._start_round("instructions", TASK1_SEED)
        elif command == "set_language":
            self.locale = str(payload.get("locale", self.locale))
        elif command == "tutorial_advance" and self.stage == "instructions":
            self.tutorial_index = min(
                len(self.tutorial_frames) - 1,
                self.tutorial_index + 1,
            )
            self.tutorial_max_index = max(
                self.tutorial_max_index,
                self.tutorial_index,
            )
        elif command == "tutorial_restart" and self.stage == "instructions":
            self.tutorial_index = 0
        elif command == "tutorial_select" and self.stage == "instructions":
            requested = int(payload.get("index", 0))
            self.tutorial_index = max(
                0,
                min(self.tutorial_max_index, requested),
            )
        elif command == "begin_task1" and self.stage == "instructions":
            self._start_round("task1", TASK1_SEED)
        elif command == "human_action" and self.stage in {"task1", "task2"}:
            self._advance_round(str(payload.get("action", "")))
        elif command == "begin_task2" and self.stage == "task1_complete":
            self._start_round("task2", TASK2_SEED)
        elif (
            command == "ask_explanation"
            and self.stage == "task1"
            and self.condition == "explanation"
        ):
            if any(
                key in payload
                for key in ("selected_frame", "trajectory_hash", "reference_index")
            ):
                raise ValueError(
                    "Live Task 1 questions cannot select an AI-AI replay frame."
                )
            target_agent = str(payload.get("target_agent", "robot_2"))
            if target_agent != "robot_2":
                raise ValueError("Live study questions must target Robot 2.")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("Question cannot be empty.")
            requested_focus = str(payload.get("question_kind", "")).strip().lower()
            allowed_focuses = {
                "action",
                "energy",
                "charge_threshold",
                "task",
                "collaboration",
                "allocation",
                "collision",
                "wait",
                "charger_penalty",
                "human_influence",
                "goal",
            }
            if requested_focus and requested_focus not in allowed_focuses:
                raise ValueError("Unknown explanation question kind.")
            focus = requested_focus or _study_question_focus(question)
            penalty_event_id = payload.get("penalty_event_id", payload.get("event_id"))
            penalty_frame_value = payload.get("penalty_frame", payload.get("anchor_frame"))
            try:
                penalty_frame = int(penalty_frame_value) if penalty_frame_value is not None else None
            except (TypeError, ValueError):
                raise ValueError("Invalid penalty frame.") from None
            self._ask_live_task1(
                question=question,
                focus=focus,
                penalty_event_id=(str(penalty_event_id) if penalty_event_id else None),
                penalty_frame=penalty_frame,
            )
        elif command == "submit_survey" and self.stage == "survey":
            self.stage = "completed"
        else:
            raise RuntimeError(f"Command {command!r} is not valid during {self.stage!r}.")
        self.version += 1
        result = {
            "run_id": self.run_id,
            "state_version": self.version,
            "view": self.view(),
        }
        self.operations[operation] = result
        return result


class DevelopmentPreviewSessions:
    """Keep independent preview state for each browser page."""

    def __init__(self, *, max_sessions: int = 256) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive.")
        self.lock = threading.RLock()
        self.max_sessions = int(max_sessions)
        template = DevelopmentPreviewState()
        self.tutorial_frames = template.tutorial_frames
        self.reference_payload = template.reference_payload
        self.states: OrderedDict[str, DevelopmentPreviewState] = OrderedDict()

    @staticmethod
    def _key(page_id: str | None) -> str:
        value = str(page_id or "").strip()
        if not value:
            return "anonymous"
        if len(value) <= 128:
            return value
        return sha256(value.encode("utf-8")).hexdigest()

    def state_for(self, page_id: str | None) -> DevelopmentPreviewState:
        key = self._key(page_id)
        with self.lock:
            state = self.states.get(key)
            if state is not None:
                self.states.move_to_end(key)
                return state
            state = DevelopmentPreviewState(
                tutorial_frames=self.tutorial_frames,
                reference_payload=self.reference_payload,
            )
            self.states[key] = state
            while len(self.states) > self.max_sessions:
                self.states.popitem(last=False)
            return state


SESSIONS = DevelopmentPreviewSessions() if DEPLOYED_ACTOR is not None else None


def _require_preview_sessions() -> DevelopmentPreviewSessions:
    if SESSIONS is None:
        _require_deployed_actor()
        raise AssertionError("Unreachable: an Actor exists without preview sessions.")
    return SESSIONS


class Handler(BaseHTTPRequestHandler):
    def preview_state(self) -> DevelopmentPreviewState:
        return _require_preview_sessions().state_for(
            self.headers.get("X-Warehouse-Page")
        )

    def do_GET(self) -> None:
        state = self.preview_state()
        with state.lock:
            if self.path == "/api/view":
                return self.send_json(state.view())
            if self.path == "/api/study/reference-trajectory":
                return self.send_json(state.reference_trajectory())
            if self.path == "/api/fixture-metrics":
                return self.send_json(
                    {
                        "command_requests": state.command_requests,
                        "reference_requests": state.reference_requests,
                    }
                )
        target = {
            "/": WEB / "index.html",
            "/index.html": WEB / "index.html",
            "/assets/styles.css": WEB / "styles.css",
            "/assets/app.js": WEB / "app.js",
            "/assets/favicon.svg": WEB / "favicon.svg",
        }.get(self.path)
        if target is None:
            self.send_error(404)
            return
        content = target.read_bytes()
        content_type = (
            "text/html; charset=utf-8"
            if target.suffix == ".html"
            else "text/css; charset=utf-8"
            if target.suffix == ".css"
            else "text/javascript; charset=utf-8"
            if target.suffix == ".js"
            else "image/svg+xml"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = self.preview_state()
            with state.lock:
                if self.path == "/api/study/command":
                    return self.send_json(state.command(payload))
            self.send_error(404)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, payload: Any, *, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, *_args: Any) -> None:
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the production UI with the real warehouse development backend."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _require_preview_sessions()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
