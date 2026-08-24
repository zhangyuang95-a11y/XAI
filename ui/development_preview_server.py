"""Development-only Web preview backed by the real warehouse environment.

Unlike ``tests/browser_fixture_server.py``, this service never fabricates a
two-step round.  The demonstration and both interactive rounds use
``WarehouseMultiAgentEnv`` for task sampling, movement, collisions, charging,
scoring, task replacement, and the 120-step terminal boundary.

The repository currently has no accepted neural-policy artifact bundle, so
this preview uses the warehouse's deterministic coordination controller.  It
is intentionally a UI/environment preview, not a formal study deployment.
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
from env.warehouse.coordination import (
    stable_coordination_actions,
    stable_coordination_goal_overrides,
)
from env.warehouse.domain import WarehouseConfig, WarehouseState
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from ui.warehouse_view import (
    _study_question_focus,
    serialize_warehouse_state,
    warehouse_map_payload,
)


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "ui" / "web"
TUTORIAL_SEED = 42_026
TASK1_SEED = 51_000
TASK2_SEED = 51_500


@dataclass(frozen=True)
class PreviewFrame:
    state: WarehouseState
    actions: Mapping[str, str]
    goal_overrides: Mapping[str, tuple[int, int]]
    rewards: Mapping[str, float]
    info: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    transition: Mapping[str, Any] | None


def _controller_actions(
    environment: WarehouseMultiAgentEnv,
    *,
    fixed_actions: Mapping[str, str] | None = None,
    goal_overrides: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, str]:
    """Return deterministic development actions without changing dynamics."""

    actions = stable_coordination_actions(
        environment,
        fixed_actions=fixed_actions,
        goal_overrides=goal_overrides,
    )
    for agent_id in environment.agent_ids:
        actions.setdefault(agent_id, "WAIT")
    return actions


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
    agents: list[dict[str, Any]] = []
    for after_agent in after.agents:
        before_agent = before.by_id(after_agent.agent_id)
        proposed = str(actions.get(after_agent.agent_id, "WAIT"))
        actual = str(executed.get(after_agent.agent_id, after_agent.last_executed_action))
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
    )


def build_development_tutorial() -> tuple[PreviewFrame, ...]:
    """Generate one verified 120-step trajectory from the real simulator."""

    environment = WarehouseMultiAgentEnv(
        replace(WarehouseConfig(), participant_detour_scoring=False)
    )
    environment.reset(seed=TUTORIAL_SEED)
    state = environment.get_state()
    state.by_id("robot_2").battery = 35.0
    environment.set_state(state)
    frames = [_initial_frame(environment)]
    for _ in range(environment.config.horizon):
        before = environment.get_state()
        goal_overrides = stable_coordination_goal_overrides(environment)
        actions = _controller_actions(
            environment,
            goal_overrides=goal_overrides,
        )
        _, rewards, terminated, truncated, info = environment.step(actions)
        after = environment.get_state()
        events = _transition_events(info)
        frames.append(
            PreviewFrame(
                state=after,
                actions=dict(actions),
                goal_overrides=dict(goal_overrides),
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
                ),
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
        self.environment = WarehouseMultiAgentEnv(WarehouseConfig())
        self.environment.reset(seed=TASK1_SEED)
        self.round_frame = _initial_frame(self.environment)
        self.stage = "idle"
        self.version = 0
        self.run_id: str | None = None
        self.locale = "en"
        self.condition = "explanation"
        self.participant_id = ""
        self.tutorial_index = 0
        self.tutorial_max_index = 0
        self.reference_index = 1
        self.task1: dict[str, Any] | None = None
        self.task2: dict[str, Any] | None = None
        self.last_explanation: dict[str, Any] | None = None
        self.explanation_target_agent = "robot_2"
        self.explanation_count = 0
        self.operations: dict[str, dict[str, Any]] = {}
        self.command_requests: list[str] = []
        self.timeline_uploads = 0
        self.reference_requests = 0
        self.reference_payload = (
            reference_payload
            if reference_payload is not None
            else self._build_reference_payload()
        )

    def _build_reference_payload(self) -> dict[str, Any]:
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
            "trajectory_kind": "ai_ai_reference",
            "trajectory_seed": TUTORIAL_SEED,
            "agent_control": {"robot_1": "ai", "robot_2": "ai"},
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
        values = {
            "idle": (),
            "instructions": (
                "tutorial_advance",
                "tutorial_restart",
                "tutorial_select",
                "begin_task1",
            ),
            "task1": ("human_action",),
            "task1_complete": ("begin_task2",),
            "explanation": (
                "timeline_select",
                "timeline_back",
                "timeline_forward",
                "ask_explanation",
                "finish_explanation",
            ),
            "task2": ("human_action",),
            "survey": ("submit_survey",),
            "completed": (),
        }[self.stage]
        return ("set_language", "restart", *values) if self.stage != "idle" else values

    def _start_round(self, stage: str, seed: int) -> None:
        self.environment = WarehouseMultiAgentEnv(WarehouseConfig())
        self.environment.reset(seed=seed)
        self.round_frame = _initial_frame(self.environment)
        self.stage = stage
        self.last_explanation = None

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
        goal_overrides = stable_coordination_goal_overrides(self.environment)
        actions = _controller_actions(
            self.environment,
            fixed_actions={"robot_1": action},
            goal_overrides=goal_overrides,
        )
        if actions["robot_1"] != action:
            raise RuntimeError("Development controller replaced the participant action.")
        _, rewards, terminated, truncated, info = self.environment.step(actions)
        after = self.environment.get_state()
        events = _transition_events(info)
        self.round_frame = PreviewFrame(
            state=after,
            actions=dict(actions),
            goal_overrides=dict(goal_overrides),
            rewards=dict(rewards),
            info=deepcopy(dict(info)),
            events=tuple(deepcopy(events)),
            transition=_transition_payload(
                before,
                after,
                actions,
                info,
                reveal_ai_action=False,
                loop=False,
            ),
        )
        if not (terminated or truncated):
            return
        if round_name == "task1":
            self.task1 = self._summary("task1", TASK1_SEED, after)
            self.stage = "explanation" if self.condition == "explanation" else "task1_complete"
            self.reference_index = 1
        else:
            self.task2 = self._summary("task2", TASK2_SEED, after)
            self.stage = "survey"

    def _display_frame(self) -> tuple[PreviewFrame, bool, str]:
        if self.stage == "instructions":
            return self.tutorial_frames[self.tutorial_index], True, "ai_ai_demonstration"
        if self.stage == "explanation":
            return self.tutorial_frames[self.reference_index], True, "ai_ai_reference"
        return self.round_frame, self.stage not in {"task1", "task2"}, f"human_ai_{self.stage}"

    def view(self) -> dict[str, Any]:
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
        timeline_index = (
            self.tutorial_index
            if self.stage == "instructions"
            else self.reference_index
            if self.stage == "explanation"
            else frame.state.frame
        )
        return {
            "map": warehouse_map_payload(self.environment.layout),
            "state": state_payload,
            "transition": frame.transition,
            "selected_agent": state_payload["selected_agent"],
            "agent_ids": list(self.environment.agent_ids),
            "timeline": {
                "index": int(timeline_index),
                "max_index": tutorial_count - 1,
                "count": tutorial_count,
                "simulator_frame": int(frame.state.frame),
                "trajectory_kind": trajectory_kind,
                "trajectory_seed": (
                    TUTORIAL_SEED
                    if self.stage in {"instructions", "explanation"}
                    else TASK1_SEED if self.stage == "task1" else TASK2_SEED
                ),
                "agent_control": (
                    {"robot_1": "ai", "robot_2": "ai"}
                    if self.stage in {"instructions", "explanation"}
                    else {"robot_1": "human", "robot_2": "ai"}
                ),
            },
            "study": {
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
                "development_controller": "warehouse_stable_coordination_teacher",
                "formal_policy_loaded": False,
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
                "explanation_duration_seconds": 600,
                "explanation_seconds_remaining": (
                    600 if self.stage == "explanation" else None
                ),
                "controlled_agent": "robot_1",
                "explanation_target_agent": self.explanation_target_agent,
                "explanation_target_agents": ["robot_1", "robot_2"],
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
            "last_explanation": self.last_explanation,
        }

    def reference_trajectory(self) -> dict[str, Any]:
        self.reference_requests += 1
        return self.reference_payload

    def _explanation_snapshot(
        self,
        index: int,
    ) -> tuple[WarehouseAdapter, Any]:
        """Rebuild decision-aligned adapter evidence for one tutorial step."""

        if index < 1 or index >= len(self.tutorial_frames):
            raise ValueError("Select an executed reference action frame.")
        before = self.tutorial_frames[index - 1]
        outcome = self.tutorial_frames[index]
        environment = WarehouseMultiAgentEnv(
            replace(WarehouseConfig(), participant_detour_scoring=False)
        )
        environment.reset(seed=TUTORIAL_SEED)
        environment.set_state(before.state)
        adapter = WarehouseAdapter(environment)
        snapshot = adapter.snapshot(None)
        masks = environment.action_masks()
        distributions = {}
        for agent_id in environment.agent_ids:
            proposed = str(outcome.actions.get(agent_id, "WAIT"))
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id,
                actions=ACTIONS,
                probabilities=tuple(
                    1.0 if action == proposed else 0.0 for action in ACTIONS
                ),
                action_mask=tuple(float(item) for item in masks[agent_id]),
                proposed_action=proposed,
            )
        return adapter, replace(
            snapshot,
            action_distributions=distributions,
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
                "environment_events": tuple(outcome.events),
                "development_controller": (
                    "warehouse_stable_coordination_teacher"
                ),
            },
        )

    def _grounded_development_explanation(
        self,
        *,
        index: int,
        target_agent: str,
        focus: str,
        language: str,
    ) -> str:
        """Render only facts derivable from the selected real transition."""

        adapter, snapshot = self._explanation_snapshot(index)
        facts = tuple(adapter.evidence_facts(snapshot, target_agent, None))
        by_predicate = {fact.predicate: fact for fact in facts}

        def render(predicate: str) -> str:
            fact = by_predicate.get(predicate)
            if fact is None:
                return ""
            value = fact.value
            if isinstance(value, Mapping):
                value = {**dict(value), "study_focus": focus}
            return adapter.explanation_verbalize_unit(
                {
                    "predicate": fact.predicate,
                    "arguments": fact.arguments,
                    "value": value,
                },
                language,
            ).strip()

        preferred = {
            "energy": ("energy_decision_context", "charging_outcome"),
            "charge_threshold": ("energy_decision_context",),
            "collaboration": (
                "charger_queue_context",
                "collaboration_context",
            ),
            "allocation": ("collaboration_context",),
            "collision": (
                "action_resolution_reason",
                "collaboration_context",
            ),
            "task": ("movement_outcome", "collaboration_context"),
        }
        if focus != "action":
            for predicate in preferred.get(focus, ("movement_outcome",)):
                rendered = render(predicate)
                if rendered:
                    return rendered

        objective_text = render("shared_objective_selection_reason")
        action_text = render("executed_action")
        resolution = by_predicate.get("action_resolution_reason")
        resolution_changed = bool(
            resolution is not None
            and isinstance(resolution.value, Mapping)
            and resolution.value.get("environment_changed_action", False)
        )
        collaboration = by_predicate.get("collaboration_context")
        collaboration_limited = bool(
            collaboration is not None
            and isinstance(collaboration.value, Mapping)
            and collaboration.value.get("teammate_directly_limited_action", False)
        )
        collaboration_enabled = bool(
            collaboration is not None
            and isinstance(collaboration.value, Mapping)
            and collaboration.value.get("enabled_teammate_action", False)
        )
        reason_order = ["charger_queue_context", "charging_outcome"]
        if resolution_changed:
            reason_order.append("action_resolution_reason")
        if (
            (collaboration_limited or collaboration_enabled)
            and not resolution_changed
        ):
            reason_order.append("collaboration_context")
        if not collaboration_enabled:
            reason_order.append("movement_outcome")
        reasons = [
            rendered
            for predicate in reason_order
            if (rendered := render(predicate))
        ]
        parts = [item for item in (objective_text, action_text, *reasons) if item]
        if parts:
            if language == "zh-CN":
                return "。".join(item.rstrip("。") for item in parts) + "。"
            return ". ".join(item.rstrip(".") for item in parts) + "."
        if action_text:
            suffix = (
                "所选帧没有显示更具体的环境约束。"
                if language == "zh-CN"
                else "The selected frame does not show a more specific environment constraint."
            )
            return f"{action_text}。{suffix}" if language == "zh-CN" else f"{action_text}. {suffix}"
        return (
            "所选帧没有足够的可验证证据。"
            if language == "zh-CN"
            else "The selected frame does not contain enough verifiable evidence."
        )

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
            self.reference_index = 1
            self.task1 = None
            self.task2 = None
            self.last_explanation = None
            self.explanation_count = 0
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
        elif command == "finish_explanation" and self.stage == "explanation":
            self._start_round("task2", TASK2_SEED)
        elif command == "ask_explanation" and self.stage == "explanation":
            self.explanation_target_agent = str(payload.get("target_agent", "robot_2"))
            if self.explanation_target_agent not in {"robot_1", "robot_2"}:
                raise ValueError("Select robot 1 or robot 2 for the explanation.")
            selected = int(payload.get("selected_frame", self.reference_index))
            self.reference_index = max(1, min(len(self.tutorial_frames) - 1, selected))
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
            }
            if requested_focus and requested_focus not in allowed_focuses:
                raise ValueError("Unknown explanation question kind.")
            focus = requested_focus or _study_question_focus(question)
            language = "zh-CN" if self.locale != "en" else "en"
            text = self._grounded_development_explanation(
                index=self.reference_index,
                target_agent=self.explanation_target_agent,
                focus=focus,
                language=language,
            )
            self.explanation_count += 1
            self.last_explanation = {
                "explanation": text,
                "explanation_document": {
                    "schema_version": "development-evidence-explanation.v1",
                    "text": text,
                },
                "explanation_mode": "deterministic_environment_evidence",
                "explanation_method_label": "Development evidence explanation",
                "target_agent": self.explanation_target_agent,
                "question": question,
                "question_focus": focus,
                "selected_timeline_frame": self.reference_index,
                "decision_evidence_frame": self.reference_index - 1,
                "question_seed": 120_000 + self.explanation_count,
            }
        elif command == "timeline_select" and self.stage == "explanation":
            self.reference_index = max(
                1,
                min(len(self.tutorial_frames) - 1, int(payload.get("index", 1))),
            )
        elif command == "timeline_back" and self.stage == "explanation":
            self.reference_index = max(1, self.reference_index - 1)
        elif command == "timeline_forward" and self.stage == "explanation":
            self.reference_index = min(
                len(self.tutorial_frames) - 1,
                self.reference_index + 1,
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


SESSIONS = DevelopmentPreviewSessions()


class Handler(BaseHTTPRequestHandler):
    def preview_state(self) -> DevelopmentPreviewState:
        return SESSIONS.state_for(self.headers.get("X-Warehouse-Page"))

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
                        "timeline_uploads": state.timeline_uploads,
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
                if self.path == "/api/study/timeline-events":
                    state.timeline_uploads += 1
                    return self.send_json(
                        {"ok": True, "recorded": len(payload.get("events", []))}
                    )
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
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
