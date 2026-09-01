"""Server-facing application state for the Warehouse XAI web interface.

This module deliberately contains no HTTP or browser code.  It adapts the
existing policy, simulator, explanation engine, and human-study state machine
to JSON-compatible commands so the same backend can be hosted by any web
server later.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import threading
import time
from typing import Any, Callable, Mapping

from backend.adapters.warehouse import WarehouseAdapter
from .collaborative_study import (
    CollaborativeStudyConfig,
    CollaborativeDeliveryStudy,
    CollaborativeStudyAssignment as StudyAssignment,
    RoundSummary,
)
from .tutorial import (
    TutorialTrajectory,
)
from backend.simulation.query_engine import (
    EXPLANATION_MODE_RCPD_TRACE,
    EXPLANATION_MODES,
    WarehouseQueryEngine,
)
from env.warehouse.environment import (
    WarehouseMultiAgentEnv,
    WarehouseState,
)
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    RUNTIME_ACTION_SOURCE,
    RUNTIME_CONTROLLER,
)
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.layouts import DEFAULT_MAP_LAYOUT, MapLayout
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.runtime_coordination import (
    causal_participant_actions,
    guard_participant_action,
    select_human_ai_action,
)

from .timeline import Timeline, TimelineFrame


from .warehouse_view import (
    AI_AI_AGENT_CONTROL,
    HUMAN_AI_AGENT_CONTROL,
    _point,
    _study_question_focus,
    serialize_warehouse_state,
    warehouse_map_payload,
)


class WarehouseWebSession:
    """One browser's isolated environment, timeline, and study state."""

    def __init__(
        self,
        *,
        policy: MAPPOPolicy,
        engine_factory: Callable[
            [WarehouseAdapter, MAPPOPolicy], WarehouseQueryEngine
        ],
        seed: int,
        tutorial: TutorialTrajectory | None = None,
        test_condition_selector: bool = False,
    ) -> None:
        self.lock = threading.RLock()
        self.seed = int(seed)
        # Network weights are shared read-only, but every browser owns its RNG.
        # This prevents an explanation rollout from mutating another session's
        # policy state and removes the need for an application-wide lock.
        self.policy = policy.fork_for_inference(seed=self.seed)
        self.environment = WarehouseMultiAgentEnv(policy.environment_config)
        self.environment.reset(seed=self.seed)
        self.adapter = WarehouseAdapter(self.environment)
        self.engine = engine_factory(self.adapter, self.policy)
        self.selected_agent = self.environment.agent_ids[0]
        self.timeline = Timeline()
        initial = self.adapter.snapshot(self.policy)
        self.timeline.reset(
            TimelineFrame(
                snapshot=initial,
                distributions=initial.action_distributions,
            )
        )
        self.engine.precompile_frame_irs(initial)
        self.trajectory_kind = "unassigned"
        self.trajectory_seed: int | None = self.seed
        self.trajectory_agent_control = dict(HUMAN_AI_AGENT_CONTROL)
        self.human_study = CollaborativeDeliveryStudy(
            CollaborativeStudyConfig(
                seed=self.seed + 41000,
                require_instructions=True,
                require_survey=True,
                event_sink=self._capture_event,
            ),
        )
        self.run_id: str | None = None
        self.owner_page_id: str | None = None
        self.state_version = 0
        self.locale = "en"
        self._events: list[Mapping[str, Any]] = []
        self.tutorial = tutorial
        self.test_condition_selector = bool(test_condition_selector)
        self.tutorial_index = 0
        self.tutorial_max_index = 0
        self.tutorial_complete = False
        self.last_explanation_report: dict[str, Any] | None = None
        self.explanation_question_sequence = 0
        self._pending_live_question_sequences: list[int] = []
        self._engine_factory = engine_factory
        self._task1_timeline_checkpoint: tuple[Any, int] | None = None

    def reset_study_machine(self) -> None:
        """Replace participant state while retaining this browser's model session."""

        config = self.human_study.config
        self.human_study = CollaborativeDeliveryStudy(config)
        self.run_id = None
        self.state_version = 0
        self._events.clear()
        self.last_explanation_report = None
        self.explanation_question_sequence = 0
        self._pending_live_question_sequences = []
        self.tutorial_index = 0
        self.tutorial_max_index = 0
        self.tutorial_complete = False
        self._task1_timeline_checkpoint = None
        self.trajectory_kind = "unassigned"
        self.trajectory_seed = self.seed
        self.trajectory_agent_control = dict(HUMAN_AI_AGENT_CONTROL)

    def _capture_event(self, event: Mapping[str, Any]) -> None:
        self._events.append(dict(event))

    def drain_events(self) -> list[Mapping[str, Any]]:
        events = list(self._events)
        self._events.clear()
        return events

    @property
    def current(self) -> TimelineFrame:
        return self.timeline.current

    def _collaborative_round_active(self) -> bool:
        return self.human_study.stage in {"task1", "task2"}

    def _timeline_state_payload(self) -> dict[str, Any]:
        frame = self.current
        return serialize_warehouse_state(
            frame.snapshot.state,
            selected_agent=self.selected_agent,
            actions=frame.actions,
            distributions=frame.distributions,
            rewards=frame.rewards,
            events=frame.info.get("environment_events", frame.info),
            reveal_policy=self.human_study.stage not in {
                "task1", "task1_complete", "task2",
            },
        )

    def _transition_payload_for(
        self,
        frame: TimelineFrame,
        *,
        reveal_joint_actions: bool,
        loop: bool,
    ) -> dict[str, Any] | None:
        """Describe the exact joint transition ending at the selected frame.

        Motion geometry is safe to expose after a step because both endpoint
        positions are already visible.  Proposed AI actions remain hidden
        during live rounds and are revealed only for the tutorial/replay.
        """

        before = frame.decision_snapshot
        if before is None or int(frame.snapshot.frame) <= int(before.frame):
            return None
        after_state = frame.snapshot.state
        before_state = before.state
        executed_actions = dict(before.executed_actions)
        invalid_agents = set(frame.info.get("invalid_move_agents", ()))
        collision = bool(frame.info.get("robot_collision_event", False))
        agents: list[dict[str, Any]] = []
        for after_agent in after_state.agents:
            before_agent = before_state.by_id(after_agent.agent_id)
            proposed = frame.actions.get(after_agent.agent_id)
            executed = executed_actions.get(
                after_agent.agent_id,
                after_agent.last_executed_action,
            )
            agents.append(
                {
                    "id": after_agent.agent_id,
                    "from_position": _point(before_agent.position),
                    "to_position": _point(after_agent.position),
                    "proposed_action": (
                        proposed
                        if reveal_joint_actions or after_agent.agent_id == "robot_1"
                        else None
                    ),
                    "executed_action": str(executed),
                    "battery_before": float(before_agent.battery),
                    "battery_after": float(after_agent.battery),
                    "battery_delta": float(
                        after_agent.battery - before_agent.battery
                    ),
                    "blocked": bool(
                        proposed in {"UP", "DOWN", "LEFT", "RIGHT"}
                        and before_agent.position == after_agent.position
                        and str(executed) == "WAIT"
                    ),
                    "invalid": after_agent.agent_id in invalid_agents,
                    "collision": collision,
                    "charging": bool(
                        str(executed) == "WAIT"
                        and after_agent.battery > before_agent.battery
                    ),
                }
            )
        return {
            "from_frame": int(before.frame),
            "to_frame": int(frame.snapshot.frame),
            "loop": bool(loop),
            "conflict_kind": frame.info.get("robot_collision_kind"),
            "events": [
                dict(item)
                for item in frame.info.get("environment_events", ())
                if isinstance(item, Mapping)
            ],
            "agents": agents,
        }

    def _transition_payload(self) -> dict[str, Any] | None:
        return self._transition_payload_for(
            self.current,
            reveal_joint_actions=self.human_study.stage == "instructions",
            loop=False,
        )

    @staticmethod
    def _event_tags(frame: TimelineFrame) -> list[str]:
        tags: list[str] = []
        mapping = {
            "claimed": "pickup",
            "delivered": "delivery",
            "charging": "charging",
            "charger_queue": "charger_queue",
            "coordination_yield": "yield",
            "collision_risk": "conflict",
            "head_on_conflict_risk": "conflict",
            "robot_collision": "collision",
        }
        for item in frame.info.get("environment_events", ()):
            if not isinstance(item, Mapping):
                continue
            tag = mapping.get(str(item.get("event", "")))
            if tag and tag not in tags:
                tags.append(tag)
        return tags

    def _reference_trajectory_identity(
        self,
        *,
        selected_agent_marker: str,
    ) -> dict[str, Any]:
        """Build reference evidence independently of interactive UI focus.

        ``selected_agent`` is presentation state, not trajectory evidence.  A
        stable marker is therefore used for the public payload.  The optional
        legacy markers exist only so pages opened before this correction can
        finish an in-progress explanation without a false stale-hash error.
        """

        if self.human_study.stage != "instructions":
            raise RuntimeError("The demonstration trajectory is available only during instructions.")
        frames: list[dict[str, Any]] = []
        for index, frame in enumerate(self.timeline.frames):
            frames.append(
                {
                    "index": index,
                    "state": serialize_warehouse_state(
                        frame.snapshot.state,
                        selected_agent=selected_agent_marker,
                        actions=frame.actions,
                        rewards=frame.rewards,
                        events=frame.info.get("environment_events", ()),
                        reveal_policy=True,
                    ),
                    "transition": self._transition_payload_for(
                        frame,
                        reveal_joint_actions=True,
                        loop=True,
                    ),
                    "event_tags": self._event_tags(frame),
                }
            )
        runtime_overrides = sum(
            str(
                (
                    frame.decision_snapshot.proposed_actions
                    if frame.decision_snapshot is not None
                    else {}
                ).get(agent_id, "WAIT")
            )
            != str(frame.actions.get(agent_id, "WAIT"))
            for frame in self.timeline.frames
            if frame.decision_snapshot is not None
            for agent_id in frame.actions
        )
        return {
            "schema_version": "warehouse-reference-timeline.v2",
            "trajectory_kind": self.trajectory_kind,
            "trajectory_seed": self.trajectory_seed,
            "action_execution_version": ACTION_EXECUTION_VERSION,
            "runtime_controller": RUNTIME_CONTROLLER,
            "rollout_action_source": RUNTIME_ACTION_SOURCE,
            "post_policy_action_interventions": int(runtime_overrides),
            "agent_control": dict(self.trajectory_agent_control),
            "map_layout_id": self.environment.layout.layout_id,
            "frames": frames,
        }

    @staticmethod
    def _reference_identity_hash(identity: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def reference_trajectory_payload(self) -> dict[str, Any]:
        identity = self._reference_trajectory_identity(
            selected_agent_marker="",
        )
        return {
            **identity,
            "trajectory_hash": self._reference_identity_hash(identity),
            "map": warehouse_map_payload(self.environment.layout),
        }

    def _study_payload(self) -> dict[str, Any]:
        current = (
            int(self.environment.state.frame)
            if self.human_study.stage in {"task1", "task1_complete", "task2"}
            and self.environment.state is not None
            else 0
        )
        total = int(self.human_study.config.horizon)
        completed = self.human_study.stage == "completed"
        allowed_human_actions = (
            list(causal_participant_actions(self.environment))
            if self.human_study.stage in {"task1", "task2"}
            else []
        )
        return {
            "run_id": getattr(self, "run_id", None),
            "stage": self.human_study.stage,
            "state_version": int(getattr(self, "state_version", 0)),
            "locale": getattr(self, "locale", getattr(self.human_study, "language", "en")),
            "participant_id": self.human_study.participant_id,
            "condition": self.human_study.condition,
            "group_code": self.human_study.group_code,
            "group_explanation_available": (
                self.human_study.condition == "explanation"
                if self.human_study.assignment is not None
                else None
            ),
            "test_condition_selector": bool(
                getattr(self, "test_condition_selector", False)
            ),
            "progress": current,
            "total": total,
            "round_summaries": {
                name: summary.to_dict()
                for name, summary in self.human_study.round_summaries.items()
            },
            "score_delta": self.human_study.score_delta if completed else None,
            "explanation_presented": self.human_study.explanation_count > 0,
            "explanation_count": self.human_study.explanation_count,
            "live_explanation_available": bool(
                self.human_study.stage == "task1"
                and self.human_study.condition == "explanation"
            ),
            "controlled_agent": "robot_1",
            "allowed_human_actions": allowed_human_actions,
            "explanation_target_agent": "robot_2",
            "explanation_target_agents": ["robot_2"],
            "tutorial": self._tutorial_payload(),
            "survey_submitted": getattr(self.human_study, "survey", None) is not None,
            "allowed_commands": list(self.allowed_commands()),
        }

    def allowed_commands(self) -> tuple[str, ...]:
        stage = self.human_study.stage
        common = ("set_language", "restart")
        if stage == "instructions":
            values = (
                "tutorial_advance", "tutorial_restart", "tutorial_select",
                "begin_task1",
            )
        elif stage == "task1":
            values = (
                ("human_action", "ask_explanation")
                if self.human_study.condition == "explanation"
                else ("human_action",)
            )
        elif stage == "task2":
            values = ("human_action",)
        elif stage == "task1_complete":
            values = ("begin_task2",)
        elif stage == "survey":
            values = ("submit_survey",)
        elif stage == "completed":
            values = ()
        elif stage == "abandoned":
            values = ("restart",)
            common = ()
        else:
            values = ()
            common = ()
        return tuple(dict.fromkeys((*common, *values)))

    def _tutorial_payload(self) -> dict[str, Any]:
        total = self.timeline.count if self.human_study.stage == "instructions" else 0
        tutorial = getattr(self, "tutorial", None)
        return {
            "frame_index": int(getattr(self, "tutorial_index", 0)),
            "max_played_index": int(getattr(self, "tutorial_max_index", 0)),
            "total_frames": total,
            "seed": int(tutorial.seed) if tutorial is not None else None,
            "continuous_mission": bool(tutorial is not None),
            "mission_start_frame": 0 if tutorial is not None else None,
            "mission_end_frame": (
                int(tutorial.frames[-1].next_snapshot.frame)
                if tutorial is not None and tutorial.frames
                else None
            ),
            "milestones": (
                [
                    {"event": event, "frame": int(frame), "agent_id": agent_id}
                    for event, frame, agent_id in tutorial.milestones
                ]
                if tutorial is not None
                else []
            ),
            "complete": bool(getattr(self, "tutorial_complete", False)),
        }

    def _public_explanation_report(self) -> dict[str, Any] | None:
        report = self.last_explanation_report
        if (
            report is None
            or self.human_study.assignment is None
        ):
            return report
        variants = report.get("_language_variants", {})
        locale = getattr(self, "locale", "en")
        if isinstance(variants, Mapping) and locale in variants:
            report = dict(variants[locale])
        document = report.get("explanation_document", {})
        public_text = (
            str(document.get("text", "")).strip()
            if isinstance(document, Mapping)
            else ""
        ) or str(report.get("explanation", "")).strip()
        return {
            "explanation": public_text,
            "explanation_document": {
                "schema_version": "explanation-document.public.v3",
                "text": public_text,
            },
            "explanation_mode": "blinded",
            "explanation_method_label": "Explanation",
            "target_agent": report.get("target_agent"),
            "question_seed": report.get("question_seed"),
            "question_sequence": report.get("question_sequence"),
            "selected_timeline_frame": report.get("selected_timeline_frame"),
            "decision_evidence_frame": report.get("decision_evidence_frame"),
            "trajectory_kind": report.get("trajectory_kind"),
            "trajectory_seed": report.get("trajectory_seed"),
            "agent_control": report.get("agent_control", {}),
            "anchor_frame": report.get("anchor_frame"),
            "context_frames": report.get("context_frames", ()),
            "question_focus": report.get("question_focus"),
            # Claims, verdict provenance, fallback diagnostics, and grounding
            # may encode program-specific source IDs.  They remain in the
            # server log but are withheld from the blinded browser payload.
            "claims": (),
            "verdicts": (),
            "posthoc_warnings": (),
            "scene_edit": report.get("scene_edit"),
            "edited_state": report.get("edited_state"),
        }

    def view(self) -> dict[str, Any]:
        state = self._timeline_state_payload()
        return {
            "state": state,
            "map": warehouse_map_payload(self.environment.layout),
            "timeline": {
                "index": int(self.timeline.index),
                "max_index": int(self.timeline.max_index),
                "count": int(self.timeline.count),
                "simulator_frame": int(self.current.snapshot.frame),
                "trajectory_kind": self.trajectory_kind,
                "trajectory_seed": self.trajectory_seed,
                "agent_control": dict(self.trajectory_agent_control),
            },
            "transition": self._transition_payload(),
            "selected_agent": self.selected_agent,
            "agent_ids": list(self.environment.agent_ids),
            "study": self._study_payload(),
            "trial": None,
            "last_explanation": self._public_explanation_report(),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "study": self.human_study.checkpoint(),
            "timeline": self.timeline.checkpoint(),
            "selected_agent": self.selected_agent,
            "last_explanation_report": self.last_explanation_report,
            "explanation_question_sequence": self.explanation_question_sequence,
            "run_id": self.run_id,
            "owner_page_id": self.owner_page_id,
            "state_version": self.state_version,
            "locale": self.locale,
            "tutorial_index": self.tutorial_index,
            "tutorial_max_index": self.tutorial_max_index,
            "tutorial_complete": self.tutorial_complete,
            "trajectory_kind": self.trajectory_kind,
            "trajectory_seed": self.trajectory_seed,
            "trajectory_agent_control": dict(self.trajectory_agent_control),
            "events": list(self._events),
            "environment_object": self.environment,
            "adapter_object": self.adapter,
            "engine_object": self.engine,
            "environment_state": self.environment.get_state(),
            "environment_rng_state": self.environment.get_rng_state(),
            "task1_timeline_checkpoint": self._task1_timeline_checkpoint,
            "pending_live_question_sequences": list(self._pending_live_question_sequences),
        }

    def restore_checkpoint(self, value: Mapping[str, Any]) -> None:
        self.human_study.restore_checkpoint(value["study"])
        self.timeline.restore_checkpoint(value["timeline"])
        self.environment = value["environment_object"]
        self.adapter = value["adapter_object"]
        self.engine = value["engine_object"]
        self.environment.set_state(value["environment_state"])
        self.environment.set_rng_state(value["environment_rng_state"])
        self._task1_timeline_checkpoint = value["task1_timeline_checkpoint"]
        self._pending_live_question_sequences = list(
            value.get("pending_live_question_sequences", ())
        )
        self.trajectory_kind = str(value["trajectory_kind"])
        self.trajectory_seed = value["trajectory_seed"]
        self.trajectory_agent_control = dict(value["trajectory_agent_control"])
        for name in (
            "selected_agent", "last_explanation_report",
            "explanation_question_sequence",
            "run_id", "state_version", "locale",
            "owner_page_id",
            "tutorial_index", "tutorial_max_index", "tutorial_complete",
        ):
            setattr(self, name, value[name])
        self._events = list(value["events"])

    def _timeline_frame_from_rollout(
        self,
        rollout_frame: Any,
    ) -> TimelineFrame | None:
        if rollout_frame.next_snapshot is None:
            return None
        decision_snapshot = replace(
            rollout_frame.snapshot,
            executed_actions=dict(rollout_frame.executed_actions),
            rewards=dict(rollout_frame.reward),
            metadata={
                **dict(rollout_frame.snapshot.metadata),
                "action_resolution": dict(
                    rollout_frame.info.get("action_resolution", {})
                ),
                "decision_trace": deepcopy(
                    rollout_frame.info.get("decision_trace", {})
                ),
                "decision_outcome_frame": rollout_frame.next_snapshot.frame,
                "decision_evidence_aligned": True,
                "decision_deterministic": True,
                "environment_events": tuple(
                    rollout_frame.environment_events
                ),
            },
        )
        return TimelineFrame(
            snapshot=rollout_frame.next_snapshot,
            decision_snapshot=decision_snapshot,
            actions=rollout_frame.proposed_actions,
            distributions=rollout_frame.distributions,
            rewards=rollout_frame.reward,
            info={
                **rollout_frame.info,
                "environment_events": rollout_frame.environment_events,
                "done": rollout_frame.done,
            },
        )

    def select_frame(self, index: int) -> dict[str, Any]:
        if self._collaborative_round_active():
            raise RuntimeError("The timeline is locked during a collaborative round.")
        if self.human_study.stage != "instructions":
            raise RuntimeError("The timeline is unavailable during this study stage.")
        if self.human_study.stage == "instructions" and int(index) > self.tutorial_max_index:
            raise RuntimeError("Unplayed tutorial frames cannot be skipped.")
        selected_index = int(index)
        self.timeline.select(selected_index)
        if self.human_study.stage == "instructions":
            self.tutorial_index = self.timeline.index
        self.adapter.restore(self.current.snapshot, self.policy)
        return self.view()

    def back(self) -> dict[str, Any]:
        return self.select_frame(max(0, self.timeline.index - 1))

    def forward(self) -> dict[str, Any]:
        if self.human_study.stage == "instructions":
            return self.tutorial_advance()
        if self.timeline.index < self.timeline.max_index:
            return self.select_frame(self.timeline.index + 1)
        return self.view()

    def _load_tutorial_timeline(self) -> None:
        if self.tutorial is None or not self.tutorial.frames:
            raise RuntimeError("Verified tutorial material is unavailable.")
        # The verified AI-AI rollout is presentation material for instructions
        # only.  Live explanations never load or read this timeline.
        self.environment = WarehouseMultiAgentEnv(self.policy.environment_config)
        self.environment.reset(seed=int(self.tutorial.seed))
        self.adapter = WarehouseAdapter(self.environment)
        self.engine = self._engine_factory(self.adapter, self.policy)
        first = self.tutorial.frames[0]
        timeline_frames = [
            TimelineFrame(
                snapshot=first.snapshot,
                distributions=first.distributions,
                info={"tutorial_continuous_mission": True},
            )
        ]
        for rollout_frame in self.tutorial.frames:
            converted = self._timeline_frame_from_rollout(rollout_frame)
            if converted is None:
                raise RuntimeError("Verified tutorial contains a missing next state.")
            previous = timeline_frames[-1].snapshot
            if (
                converted.snapshot.state.episode_id != previous.state.episode_id
                or converted.snapshot.frame != previous.frame + 1
            ):
                raise RuntimeError("Verified tutorial is not one continuous task.")
            timeline_frames.append(converted)
        self.timeline.reset(timeline_frames[0])
        for frame in timeline_frames[1:]:
            self.timeline.append(frame)
        self.timeline.select(0)
        self.trajectory_kind = "ai_ai_demonstration"
        self.trajectory_seed = int(self.tutorial.seed)
        self.trajectory_agent_control = dict(AI_AI_AGENT_CONTROL)
        self.tutorial_index = 0
        self.tutorial_max_index = 0
        self.tutorial_complete = len(timeline_frames) == 1
        self.selected_agent = self.tutorial.focus_agent
        self.adapter.restore(self.current.snapshot, self.policy)

    def tutorial_advance(self) -> dict[str, Any]:
        if self.human_study.stage != "instructions":
            raise RuntimeError("The tutorial is not active.")
        was_complete = self.tutorial_complete
        if self.tutorial_index < self.timeline.max_index:
            self.tutorial_index += 1
            self.tutorial_max_index = max(self.tutorial_max_index, self.tutorial_index)
            self.timeline.select(self.tutorial_index)
            self.adapter.restore(self.current.snapshot, self.policy)
        self.tutorial_complete = self.tutorial_max_index >= self.timeline.max_index
        if self.tutorial_complete and not was_complete:
            self._capture_event(
                {
                    "event": "tutorial_completed",
                    "tutorial_seed": self.tutorial.seed,
                    "trajectory_kind": "continuous_complete_mission",
                    "displayed_frames": self.timeline.count,
                }
            )
        return self.view()

    def tutorial_restart(self) -> dict[str, Any]:
        if self.human_study.stage != "instructions":
            raise RuntimeError("The tutorial is not active.")
        return self.select_frame(0)

    def _start_collaborative_round(self, round_name: str, seed: int) -> None:
        self.environment = WarehouseMultiAgentEnv(self.policy.environment_config)
        self.environment.reset(seed=int(seed))
        participant_state = self.environment.get_state()
        participant_state.participant_controlled_agent_id = (
            self.environment.config.human_agent_id
        )
        self.environment.set_state(participant_state)
        self.adapter = WarehouseAdapter(self.environment)
        self.engine = self._engine_factory(self.adapter, self.policy)
        initial = self.adapter.snapshot(self.policy)
        self.timeline.reset(
            TimelineFrame(
                snapshot=initial,
                distributions=initial.action_distributions,
                info={"collaborative_round": round_name},
            )
        )
        self.trajectory_kind = f"human_ai_{round_name}"
        self.trajectory_seed = int(seed)
        self.trajectory_agent_control = dict(HUMAN_AI_AGENT_CONTROL)
        self.selected_agent = "robot_1"
        self.last_explanation_report = None

    def begin_task1(self) -> dict[str, Any]:
        tutorial_completed = bool(self.tutorial_complete)
        tutorial_last_frame = int(self.tutorial_index)
        tutorial_total_frames = int(self.timeline.count)
        tutorial_displayed_frames = min(
            tutorial_total_frames,
            int(self.tutorial_max_index) + 1,
        )
        tutorial_remaining_frames = max(
            0,
            tutorial_total_frames - tutorial_displayed_frames,
        )
        tutorial_completion_fraction = (
            float(tutorial_displayed_frames) / float(tutorial_total_frames)
            if tutorial_total_frames > 0
            else 0.0
        )
        if not tutorial_completed:
            self._capture_event(
                {
                    "event": "tutorial_ended_early",
                    "tutorial_seed": int(self.tutorial.seed),
                    "trajectory_kind": "continuous_partial_mission",
                    "last_displayed_frame": tutorial_last_frame,
                    "displayed_frames": tutorial_displayed_frames,
                    "total_frames": tutorial_total_frames,
                    "remaining_frames": tutorial_remaining_frames,
                    "completion_fraction": tutorial_completion_fraction,
                }
            )
        self.human_study.begin_task1()
        assert self.human_study.assignment is not None
        self._start_collaborative_round(
            "task1",
            self.human_study.assignment.task1_seed,
        )
        self._capture_event(
            {
                "event": "tutorial_acknowledged",
                "tutorial_completed": tutorial_completed,
                "ended_early": not tutorial_completed,
                "last_displayed_frame": tutorial_last_frame,
                "displayed_frames": tutorial_displayed_frames,
                "total_frames": tutorial_total_frames,
                "remaining_frames": tutorial_remaining_frames,
                "completion_fraction": tutorial_completion_fraction,
            }
        )
        return {"trial": None, "view": self.view()}

    def _ask_live_task1(
        self,
        question: str,
        *,
        question_focus: str,
        target_agent: str,
    ) -> dict[str, Any]:
        """Answer from an event-anchored prefix of the real Task 1 timeline.

        This method is intentionally read-only: it neither selects a replay
        frame nor restores the adapter.  Consequently a question cannot move
        the environment, consume battery, or expose any frame after the
        selected event.
        """

        if self.human_study.stage != "task1" or self.human_study.condition != "explanation":
            raise RuntimeError("Live questions are available only to Group A during Task 1.")
        if target_agent != "robot_2":
            raise ValueError("Live study questions must target Robot 2.")
        if self.trajectory_kind != "human_ai_task1" or self.trajectory_agent_control != HUMAN_AI_AGENT_CONTROL:
            raise RuntimeError("Live explanations require the real Human-AI Task 1 timeline.")

        started_at = time.perf_counter()
        completed = [
            frame for frame in self.timeline.frames[: self.timeline.index + 1]
            if frame.decision_snapshot is not None
        ]
        recent = completed[-5:]

        def proposed(frame: TimelineFrame) -> str:
            return str(frame.actions.get("robot_2", "WAIT"))

        def executed(frame: TimelineFrame) -> str:
            decision = frame.decision_snapshot
            return str(
                (decision.executed_actions if decision is not None else {}).get(
                    "robot_2", proposed(frame)
                )
            )

        anchor: TimelineFrame | None = recent[-1] if recent else None
        if question_focus == "wait":
            anchor = next(
                (frame for frame in reversed(recent) if proposed(frame) == "WAIT" or executed(frame) == "WAIT"),
                None,
            )
        elif question_focus == "collision":
            anchor = next(
                (
                    frame for frame in reversed(recent)
                    if bool(frame.info.get("robot_collision_event", False))
                    or bool(frame.info.get("robot_collision_kind"))
                ),
                None,
            )

        current_frame = int(self.current.snapshot.frame)
        if anchor is None:
            anchor_frame = current_frame
            context_frames = [int(frame.snapshot.frame) for frame in recent]
            if question_focus == "collision":
                answer_en = "No collision occurred in the last five steps."
                answer_zh = "最近五步内没有发生碰撞。"
            elif question_focus == "wait":
                answer_en = "Robot 2 did not wait in the last five steps."
                answer_zh = "机器人2在最近五步内没有等待。"
            else:
                answer_en = "Robot 2 has not completed an action in Task 1 yet."
                answer_zh = "机器人2在任务1中还没有完成任何动作。"
            evidence: dict[str, Any] = {
                "event_type": question_focus,
                "anchor_frame": anchor_frame,
                "context_frames": context_frames,
                "reason_code": "NO_MATCHING_RECENT_EVENT",
                "fact_valid": True,
            }
            recent_collision = False
        else:
            anchor_frame = int(anchor.snapshot.frame)
            prefix = [
                frame for frame in completed
                if int(frame.snapshot.frame) <= anchor_frame
            ]
            context = prefix[-5:]
            context_frames = [int(frame.snapshot.frame) for frame in context]
            decision_snapshot = anchor.decision_snapshot
            assert decision_snapshot is not None
            trace = decision_snapshot.metadata.get("decision_trace", {})
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
                item for item in agent_trace.get("battery_feasibility", ())
                if isinstance(item, Mapping)
            ]
            goal_id = str(
                frozen_goal.get("goal_id")
                or agent_trace.get("committed_task")
                or ""
            )
            energy_item = next(
                (item for item in feasibility if str(item.get("task_id", "")) == goal_id),
                min(feasibility, key=lambda item: float(item.get("required_energy", 0.0)))
                if feasibility else {},
            )
            plan = agent_trace.get("joint_coordination_plan")
            plan = plan if isinstance(plan, Mapping) else {}
            recent_collision = bool(anchor.info.get("robot_collision_event", False))
            evidence = {
                "event_type": question_focus,
                "anchor_frame": anchor_frame,
                "context_frames": context_frames,
                "human_action": str(anchor.actions.get("robot_1", "WAIT")),
                "ai_action": proposed(anchor),
                "executed_actions": dict(decision_snapshot.executed_actions),
                "collision_type": anchor.info.get("robot_collision_kind"),
                "current_goal": frozen_goal.get("navigation_kind") or frozen_goal.get("goal_type"),
                "current_battery": charging.get("battery"),
                "required_energy": energy_item.get("required_energy"),
                "priority_basis": plan.get("priority_basis"),
                "task_id": goal_id or energy_item.get("task_id"),
                "reason_code": agent_trace.get("primary_reason_code"),
                "pre_state_hash": trace.get("pre_state_hash"),
                "outcome_frame": trace.get("outcome_frame"),
                "fact_valid": bool(trace.get("fact_valid", False)),
            }
            if question_focus == "collision" and recent_collision:
                human_action = str(anchor.actions.get("robot_1", "WAIT"))
                ai_action = proposed(anchor)
                action_en = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right", "WAIT": "wait"}
                action_zh = {"UP": "上", "DOWN": "下", "LEFT": "左", "RIGHT": "右", "WAIT": "等待"}
                kind = str(anchor.info.get("robot_collision_kind") or "joint_conflict")
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
                    f"You moved {action_en.get(human_action, human_action.lower())} while Robot 2 simultaneously moved {action_en.get(ai_action, ai_action.lower())}; the joint resolver recorded {kind_en}. "
                    "Robot 2 decided from the shared pre-move state and did not observe your current action first."
                )
                answer_zh = (
                    f"你向{action_zh.get(human_action, human_action)}移动，机器人2同时向{action_zh.get(ai_action, ai_action)}移动；联合解析器记录为{kind_zh}。"
                    "机器人2依据共同的移动前状态决策，没有提前看到你的本帧动作。"
                )
            else:
                routed_focus = {
                    "wait": "action",
                    "human_influence": "collaboration",
                    "goal": "task",
                }.get(question_focus, question_focus)
                answer_en = self.adapter.concise_study_explanation(
                    decision_snapshot,
                    target_agent="robot_2",
                    policy=self.policy,
                    focus=routed_focus,
                    language="en",
                )
                answer_zh = self.adapter.concise_study_explanation(
                    decision_snapshot,
                    target_agent="robot_2",
                    policy=self.policy,
                    focus=routed_focus,
                    language="zh-CN",
                )

        sequence = self.explanation_question_sequence + 1
        self.explanation_question_sequence = sequence
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        common = {
            "target_agent": "robot_2",
            "question_sequence": sequence,
            "question_focus": question_focus,
            "selected_timeline_frame": anchor_frame,
            "decision_evidence_frame": anchor_frame,
            "anchor_frame": anchor_frame,
            "context_frames": context_frames,
            "current_frame": current_frame,
            "trajectory_kind": self.trajectory_kind,
            "trajectory_seed": self.trajectory_seed,
            "agent_control": dict(self.trajectory_agent_control),
            "structured_evidence": evidence,
            "fact_validation": {
                "passed": bool(evidence.get("fact_valid", False)),
                "future_frames_used": False,
                "max_context_frame": max(context_frames, default=anchor_frame),
            },
            "recent_collision": recent_collision,
            "latency_ms": latency_ms,
            "answer_en": answer_en,
            "answer_zh": answer_zh,
        }
        variants = {
            "en": {**common, "explanation": answer_en, "explanation_document": {"text": answer_en}},
            "zh-CN": {**common, "explanation": answer_zh, "explanation_document": {"text": answer_zh}},
        }
        report = dict(variants[self.locale])
        report["_language_variants"] = variants
        report["language_documents"] = {
            locale: value["explanation_document"] for locale, value in variants.items()
        }
        self.last_explanation_report = report
        self.human_study.record_explanation(
            question=question,
            report=common,
            response_seconds=latency_ms / 1000.0,
        )
        self._pending_live_question_sequences.append(sequence)
        return {"report": self._public_explanation_report(), "view": self.view()}

    def ask(
        self,
        question: str,
        explanation_mode: str,
        *,
        study_request: bool = False,
        study_target_agent: str | None = None,
        study_question_kind: str | None = None,
    ) -> dict[str, Any]:
        prompt = question.strip()
        if not prompt:
            raise ValueError("Question cannot be empty.")
        if explanation_mode not in EXPLANATION_MODES:
            raise ValueError(f"Unknown explanation mode: {explanation_mode}")
        allowed_focuses = {
            "action", "energy", "charge_threshold", "task",
            "collaboration", "allocation", "collision", "wait",
            "human_influence", "goal",
        }
        requested_focus = str(study_question_kind or "").strip().lower()
        if requested_focus and requested_focus not in allowed_focuses:
            raise ValueError("Unknown explanation question kind.")
        question_focus = (
            requested_focus or _study_question_focus(prompt)
            if study_request
            else "action"
        )
        if study_request:
            return self._ask_live_task1(
                prompt,
                question_focus=question_focus,
                target_agent=str(study_target_agent or "robot_2"),
            )
        raise RuntimeError("Questions are available only to Group A during Task 1.")

    def start_study(
        self,
        *,
        assignment: StudyAssignment,
        language: str = "en",
    ) -> dict[str, Any]:
        if self.tutorial is None:
            raise RuntimeError("Verified tutorial material is unavailable.")
        if int(assignment.demo_seed) != int(self.tutorial.seed):
            raise RuntimeError(
                "The assigned demonstration seed does not match the verified tutorial."
            )
        if assignment.demo_seed in {assignment.task1_seed, assignment.task2_seed}:
            raise RuntimeError("The demonstration seed overlaps a scored round.")
        self.human_study.start(
            assignment,
            language=language,
        )
        self.selected_agent = assignment.target_agent
        self.last_explanation_report = None
        self.locale = language
        self._load_tutorial_timeline()
        return {
            "trial": None,
            "view": self.view(),
        }

    def _round_summary(self, round_name: str, seed: int) -> RoundSummary:
        state = self.environment.get_state()
        latencies = [
            float(task.delivered_frame - task.claimed_frame)
            for task in state.completed_tasks
            if task.delivered_frame is not None and task.claimed_frame is not None
        ]
        return RoundSummary(
            round_name=round_name,
            seed=int(seed),
            score=float(state.user_score),
            steps=int(state.frame),
            deliveries=int(state.total_deliveries),
            robot_collisions=int(state.robot_collision_events),
            shutdowns=int(state.shutdown_count),
            human_route_regret_units=float(state.human_route_regret_units),
            mean_delivery_latency=(
                sum(latencies) / len(latencies) if latencies else None
            ),
            terminal_reason=state.terminal_reason,
        )

    def submit_human_action(self, action: str) -> dict[str, Any]:
        round_name = self.human_study.stage
        if round_name not in {"task1", "task2"}:
            raise RuntimeError("No collaborative delivery round is active.")
        if action not in self.policy.action_names:
            raise ValueError("Unknown participant action.")
        before = self.adapter.snapshot(self.policy)
        decision_state = self.environment.get_state()
        proposed, distributions = self.policy.act(
            before.observations,
            before.global_state,
            deterministic=False,
            decision_key=(decision_state.episode_id, decision_state.frame),
        )
        ai_action, runtime_decision = select_human_ai_action(
            self.environment,
            proposed["robot_2"],
        )
        participant_action, participant_guard = guard_participant_action(
            self.environment,
            action,
        )
        requested_joint_actions = {
            **dict(proposed),
            "robot_1": participant_action,
            "robot_2": ai_action,
        }
        # Robot 2 is selected from S_t before the participant command is
        # consulted.  The runtime evidence above proves it against every
        # causally submit-able human action; both selected actions are still
        # resolved atomically by one environment step.
        joint_actions = dict(requested_joint_actions)
        _, rewards, terminated, truncated, info = self.environment.step(
            joint_actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source="participant_plus_robust_pytorch_actor",
                participant_overrides={self.environment.config.human_agent_id: action},
                policy_actions=proposed,
                selected_actions=joint_actions,
                runtime_decision={
                    **runtime_decision,
                    "participant_action_guard": participant_guard,
                    "selected_actions": dict(joint_actions),
                },
            ),
        )
        after = self.adapter.snapshot(self.policy)
        decision = replace(
            before,
            proposed_actions=dict(requested_joint_actions),
            executed_actions=dict(info.get("executed_actions", {})),
            rewards=dict(rewards),
            metadata={
                **dict(before.metadata),
                "action_resolution": dict(info.get("action_resolution", {})),
                "decision_trace": deepcopy(info.get("decision_trace", {})),
                "decision_outcome_frame": after.frame,
                "decision_evidence_aligned": True,
                "decision_deterministic": False,
                "collaborative_round": round_name,
                "human_requested_action": action,
                "ai_network_action": proposed.get("robot_2"),
                "ai_submitted_action": joint_actions.get("robot_2"),
                "action_execution": "causal_robust_simultaneous_human_ai",
                "shared_decision_state_hash": (
                    info.get("decision_trace", {}).get("pre_state_hash")
                    if isinstance(info.get("decision_trace"), Mapping)
                    else None
                ),
            },
        )
        self.timeline.append(
            TimelineFrame(
                snapshot=after,
                decision_snapshot=decision,
                actions=dict(requested_joint_actions),
                distributions=distributions,
                rewards=rewards,
                info={**dict(info), "collaborative_round": round_name},
            )
        )
        self.human_study.record_step(
            {
                "frame": int(after.frame),
                "human_requested_action": action,
                "human_submitted_action": participant_action,
                "ai_network_action": proposed.get("robot_2"),
                "ai_submitted_action": joint_actions.get("robot_2"),
                "action_execution": "causal_robust_simultaneous_human_ai",
                "executed_actions": dict(info.get("executed_actions", {})),
                "task_changes": list(info.get("task_changes", ())),
                "robot_collision_event": bool(info.get("robot_collision_event", False)),
                "invalid_move_agents": list(info.get("invalid_move_agents", ())),
                "delivered_task_ids": list(info.get("delivered_task_ids", ())),
                "route_regret": dict(info.get("route_regret", {})),
                "score_delta": float(sum(info.get("reward_breakdown", {}).values())),
                "score_components_delta": dict(info.get("reward_breakdown", {})),
                "state_before": serialize_warehouse_state(
                    before.state,
                    selected_agent="robot_1",
                    reveal_policy=False,
                ),
                "state_after": serialize_warehouse_state(
                    after.state,
                    selected_agent="robot_1",
                    reveal_policy=False,
                ),
            }
        )
        for question_sequence in self._pending_live_question_sequences:
            self._capture_event(
                {
                    "event": "live_explanation_followup_action",
                    "question_sequence": int(question_sequence),
                    "round": round_name,
                    "frame": int(after.frame),
                    "post_question_action": action,
                }
            )
        self._pending_live_question_sequences = []
        round_complete = bool(terminated or truncated)
        if round_complete:
            assert self.human_study.assignment is not None
            seed = (
                self.human_study.assignment.task1_seed
                if round_name == "task1"
                else self.human_study.assignment.task2_seed
            )
            if round_name == "task1":
                self._task1_timeline_checkpoint = self.timeline.checkpoint()
            self.human_study.finish_round(
                self._round_summary(round_name, seed)
            )
        return {
            "round_complete": round_complete,
            "view": self.view(),
        }

    def begin_task2(self) -> dict[str, Any]:
        self.human_study.begin_task2()
        assert self.human_study.assignment is not None
        self._start_collaborative_round(
            "task2",
            self.human_study.assignment.task2_seed,
        )
        return {"view": self.view()}

    def set_language(self, locale: str) -> dict[str, Any]:
        self.human_study.set_language(locale)
        self.locale = locale
        report = self.last_explanation_report
        if isinstance(report, Mapping):
            variants = report.get("_language_variants", {})
            if isinstance(variants, Mapping) and locale in variants:
                selected = dict(variants[locale])
                selected["_language_variants"] = variants
                selected["language_documents"] = report.get("language_documents", {})
                self.last_explanation_report = selected
        return {"view": self.view()}

    def submit_survey(self, survey: Mapping[str, Any]) -> dict[str, Any]:
        self.human_study.submit_survey(survey)
        return {"view": self.view()}

    def explain_study(
        self,
        question: str,
        *,
        target_agent: str = "robot_2",
        question_kind: str | None = None,
    ) -> dict[str, Any]:
        if target_agent != "robot_2":
            raise ValueError("Live study questions must target Robot 2.")
        result = self.ask(
            question,
            EXPLANATION_MODE_RCPD_TRACE,
            study_request=True,
            study_target_agent="robot_2",
            study_question_kind=question_kind,
        )
        result["report"] = self._public_explanation_report()
        result["view"] = self.view()
        return result
