"""Server-facing application state for the Warehouse XAI web interface.

This module deliberately contains no HTTP or browser code.  It adapts the
existing policy, simulator, explanation engine, and human-study state machine
to JSON-compatible commands so the same backend can be hosted by any web
server later.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import re
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
from env.warehouse.contracts import ACTION_EXECUTION_VERSION, RUNTIME_CONTROLLER
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.layouts import DEFAULT_MAP_LAYOUT, MapLayout
from env.warehouse.policy import MAPPOPolicy

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
            reveal_joint_actions=self.human_study.stage
            in {"instructions", "explanation"},
            loop=self.human_study.stage == "explanation",
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

        if self.human_study.stage != "explanation":
            raise RuntimeError("The reference trajectory is available only during explanation.")
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
        return {
            "schema_version": "warehouse-reference-timeline.v2",
            "trajectory_kind": self.trajectory_kind,
            "trajectory_seed": self.trajectory_seed,
            "action_execution_version": ACTION_EXECUTION_VERSION,
            "runtime_controller": RUNTIME_CONTROLLER,
            "rollout_action_source": "mappo_actor",
            "post_policy_action_interventions": 0,
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

    def reference_trajectory_hash_is_compatible(self, value: str) -> bool:
        """Accept only canonical or selection-only legacy hashes."""

        submitted = str(value).strip()
        if not submitted:
            return True
        compatible = {
            self._reference_identity_hash(
                self._reference_trajectory_identity(
                    selected_agent_marker=marker,
                )
            )
            for marker in ("", "robot_1", "robot_2")
        }
        return submitted in compatible

    def _study_payload(self) -> dict[str, Any]:
        current = (
            int(self.environment.state.frame)
            if self.human_study.stage in {"task1", "task1_complete", "task2"}
            and self.environment.state is not None
            else 0
        )
        total = int(self.human_study.config.horizon)
        completed = self.human_study.stage == "completed"
        remaining = self.human_study.explanation_seconds_remaining
        config = getattr(self.human_study, "config", None)
        deadline = (
            (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat()
            if remaining is not None
            else None
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
            "explanation_duration_seconds": int(getattr(config, "explanation_time_limit_seconds", 600)),
            "explanation_deadline": deadline,
            "explanation_seconds_remaining": remaining,
            "controlled_agent": "robot_1",
            "explanation_target_agent": (
                self.selected_agent
                if self.selected_agent in {"robot_1", "robot_2"}
                else "robot_2"
            ),
            "explanation_target_agents": ["robot_1", "robot_2"],
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
        elif stage in {"task1", "task2"}:
            values = ("human_action",)
        elif stage == "task1_complete":
            values = ("begin_task2",)
        elif stage == "explanation":
            values = (
                "timeline_select", "timeline_back", "timeline_forward",
                "ask_explanation", "finish_explanation",
            )
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
        if self.human_study.stage not in {"instructions", "explanation"}:
            raise RuntimeError("The timeline is unavailable during this study stage.")
        if self.human_study.stage == "instructions" and int(index) > self.tutorial_max_index:
            raise RuntimeError("Unplayed tutorial frames cannot be skipped.")
        selected_index = int(index)
        if self.human_study.stage == "explanation" and self.timeline.count > 1:
            selected_index = max(1, selected_index)
        self.timeline.select(selected_index)
        if self.human_study.stage == "instructions":
            self.tutorial_index = self.timeline.index
        elif self.human_study.stage == "explanation":
            if hasattr(self, "_events"):
                self._capture_event(
                    {
                        "event": "trajectory_frame_browsed",
                        "timeline_index": int(
                            getattr(self.timeline, "index", index)
                        ),
                        "environment_frame": int(
                            getattr(self.current.snapshot, "frame", index)
                        ),
                        "study_stage": "explanation",
                        "trajectory_kind": self.trajectory_kind,
                        "trajectory_seed": self.trajectory_seed,
                        "agent_control": dict(self.trajectory_agent_control),
                        "immutable": True,
                    }
                )
        self.adapter.restore(self.current.snapshot, self.policy)
        return self.view()

    def back(self) -> dict[str, Any]:
        minimum = 1 if self.human_study.stage == "explanation" else 0
        return self.select_frame(max(minimum, self.timeline.index - 1))

    def forward(self) -> dict[str, Any]:
        if self.human_study.stage == "instructions":
            return self.tutorial_advance()
        if self.timeline.index < self.timeline.max_index:
            return self.select_frame(self.timeline.index + 1)
        return self.view()

    def _load_tutorial_timeline(self, *, for_explanation: bool = False) -> None:
        if self.tutorial is None or not self.tutorial.frames:
            raise RuntimeError("Verified tutorial material is unavailable.")
        # Explanation uses a fresh runtime around the immutable verified
        # AI-AI rollout.  The Task 1 adapter and engine are never reused as an
        # explanation evidence source.
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
        selected_index = min(1, self.timeline.max_index) if for_explanation else 0
        self.timeline.select(selected_index)
        self.trajectory_kind = (
            "ai_ai_reference" if for_explanation else "ai_ai_demonstration"
        )
        self.trajectory_seed = int(self.tutorial.seed)
        self.trajectory_agent_control = dict(AI_AI_AGENT_CONTROL)
        if not for_explanation:
            self.tutorial_index = 0
            self.tutorial_max_index = 0
            self.tutorial_complete = len(timeline_frames) == 1
        self.selected_agent = "robot_2" if for_explanation else self.tutorial.focus_agent
        self.adapter.restore(self.current.snapshot, self.policy)

        if for_explanation:
            self._capture_event(
                {
                    "event": "explanation_reference_loaded",
                    "trajectory_kind": self.trajectory_kind,
                    "trajectory_seed": self.trajectory_seed,
                    "agent_control": dict(self.trajectory_agent_control),
                    "displayed_frames": self.timeline.count,
                    "initial_timeline_index": self.timeline.index,
                    "immutable": True,
                    "reuses_demonstration": True,
                }
            )

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

    def _resolve_query_frame(self, plan: Any) -> tuple[TimelineFrame, Any, Any, Any]:
        try:
            requested = (
                self.timeline.simulator_frame(plan.frame_reference)
                if plan.frame_reference is not None
                else self.current
            )
        except KeyError:
            requested = self.current
        action_focused = (
            plan.intent.value in {"explanatory", "why_not"}
            and (
                plan.requires_program_trace
                or any(
                    "action" in str(item).lower()
                    for item in plan.target_variables
                )
                or "program_trace" in plan.evidence_requirements
            )
        )
        execution_snapshot = requested.snapshot
        execution_plan = plan
        def resolve_frame(frame_id: int) -> Any:
            return self.timeline.simulator_frame(frame_id).snapshot

        resolver: Callable[[int], Any] | None = resolve_frame
        if action_focused and requested.decision_snapshot is not None:
            execution_snapshot = requested.decision_snapshot
            execution_plan = replace(
                plan,
                frame_reference=execution_snapshot.frame,
            )
            resolver = None
        return requested, execution_snapshot, execution_plan, resolver

    def ask(
        self,
        question: str,
        explanation_mode: str,
        *,
        study_request: bool = False,
        accepted_before_deadline: bool = False,
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
            "collaboration", "allocation", "collision",
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
            if self.human_study.stage != "explanation":
                raise RuntimeError(
                    "The study is not waiting for a free-form question."
                )
            if (
                self.human_study.explanation_time_expired
                and not accepted_before_deadline
            ):
                raise RuntimeError(
                    "The ten-minute explanation period has ended."
                )
            explanation_mode = EXPLANATION_MODE_RCPD_TRACE
            target_agent = str(study_target_agent or "robot_2")
            if target_agent not in {"robot_1", "robot_2"}:
                raise ValueError("Select robot 1 or robot 2 for the explanation.")
            self.selected_agent = target_agent
        else:
            accepted_before_deadline = False
            target_agent = self.selected_agent
        if not study_request:
            raise RuntimeError("Questions are available only in the Task 1 explanation stage.")
        snapshot = self.current.snapshot
        started_at = time.perf_counter()
        plan = self.engine.planner.parse(
            prompt,
            selected_frame=snapshot.frame,
            environment_schema={
                "observations": dict(self.adapter.observation_schema()),
                "actions": list(self.adapter.action_schema()),
                "entities": dict(self.adapter.entity_schema()),
                **dict(
                    self.adapter.question_vocabulary()
                    if hasattr(self.adapter, "question_vocabulary")
                    else {}
                ),
                # The selected entity is part of the question context.  It is
                # used only when the participant writes an implicit subject
                # such as "why wait?"; an explicit entity mention wins.
                "focus_entity": getattr(self, "selected_agent", None),
            },
            cache_context=(
                self.engine.question_cache_context(snapshot)
                if hasattr(self.engine, "question_cache_context")
                else {}
            ),
        )
        if study_request:
            targets = set(plan.prediction_targets)
            if targets and targets != {target_agent}:
                raise ValueError(
                    "The robot named in the question does not match the selected robot."
                )
            if plan.requires_scene_edit:
                raise ValueError(
                    "Counterfactual edits are disabled during the study; ask why the "
                    "selected robot executed its recorded action."
                )
        if plan.clarification_required:
            raise ValueError(
                plan.clarification_reason
                or "Please specify the target robot or requested action."
            )
        if study_request:
            # The replay slider is authoritative.  Parser intent or a number in
            # the free-form text must not detach the question from the selected
            # immutable AI-AI reference transition.
            requested = self.current
            if requested.decision_snapshot is None:
                raise ValueError(
                    "Select an executed AI-AI reference action frame before asking a question."
                )
            execution_snapshot = requested.decision_snapshot
            execution_plan = replace(
                plan,
                frame_reference=execution_snapshot.frame,
                # Study questions are always about the action visible in the
                # selected immutable transition.  The free-form parser may
                # otherwise interpret Chinese phrases such as ``要等待`` as an
                # objective query, which can yield an empty document when
                # private navigation goals are intentionally withheld.
                requires_policy_query=True,
                requires_program_trace=True,
                target_variables=(f"{target_agent}.observed_action",),
                evidence_requirements=tuple(
                    dict.fromkeys(
                        (
                            *(
                                item
                                for item in plan.evidence_requirements
                                if not str(item).startswith("study_focus:")
                            ),
                            "state",
                            "policy",
                            "program_trace",
                            f"study_focus:{question_focus}",
                        )
                    )
                ),
            )
            resolver = None
        else:
            requested, execution_snapshot, execution_plan, resolver = (
                self._resolve_query_frame(plan)
            )
        if study_request and requested.decision_snapshot is None:
            raise ValueError(
                "Select an executed AI-AI reference action frame before asking a question."
            )
        question_sequence = self.explanation_question_sequence + 1
        seed_material = (
            f"{self.seed}:{self.run_id or 'session'}:{question_sequence}"
        ).encode("utf-8")
        question_seed = int.from_bytes(
            sha256(seed_material).digest()[:4], "big"
        ) & 0x7FFFFFFF
        self.explanation_question_sequence = question_sequence
        bilingual = study_request
        languages = ("en", "zh-CN") if bilingual else (execution_plan.response_language,)
        reports: dict[str, dict[str, Any]] = {}
        answer = None
        for response_language in languages:
            localized_plan = replace(
                execution_plan,
                response_language=response_language,
            )
            localized_answer = self.engine.execute_plan(
                localized_plan,
                execution_snapshot,
                language=response_language,
                seed=question_seed,
                snapshot_resolver=resolver,
                explanation_mode=explanation_mode,
                _parse_diagnostics=dict(self.engine.planner.last_diagnostics),
            )
            if not str(localized_answer.user_visible_explanation).strip():
                raise RuntimeError(
                    "The explanation system could not produce a grounded "
                    "answer for the selected action frame."
                )
            answer = localized_answer
            localized_report = localized_answer.to_dict()
            localized_text = str(
                localized_report.get("explanation_document", {}).get("text", "")
            ).strip()
            semantic_valid = self._validate_study_explanation_text(
                localized_text,
                focus=question_focus,
                language=response_language,
            )
            if study_request:
                localized_text = self._deterministic_study_explanation(
                    execution_snapshot,
                    target_agent=target_agent,
                    focus=question_focus,
                    language=response_language,
                )
                semantic_valid = self._validate_study_explanation_text(
                    localized_text,
                    focus=question_focus,
                    language=response_language,
                )
                localized_report["explanation"] = localized_text
                document = dict(localized_report.get("explanation_document", {}))
                document["text"] = localized_text
                localized_report["explanation_document"] = document
            elif not semantic_valid:
                localized_text = self._deterministic_study_explanation(
                    execution_snapshot,
                    target_agent=target_agent,
                    focus=question_focus,
                    language=response_language,
                )
                localized_report["explanation"] = localized_text
                document = dict(localized_report.get("explanation_document", {}))
                document["text"] = localized_text
                localized_report["explanation_document"] = document
            localized_report["semantic_validation"] = {
                "passed": bool(semantic_valid),
                "fallback_used": bool(study_request or not semantic_valid),
                "question_focus": question_focus,
            }
            reports[response_language] = localized_report
        assert answer is not None
        report = dict(reports.get(self.locale, next(iter(reports.values()))))
        request_total_ms = (time.perf_counter() - started_at) * 1000.0
        diagnostics = dict(report.get("generation_diagnostics", {}))
        diagnostics["request_total_ms"] = request_total_ms
        report["generation_diagnostics"] = diagnostics
        report["latency_ms"] = request_total_ms
        report.update(
            {
                "selected_timeline_frame": int(requested.snapshot.frame),
                "decision_evidence_frame": int(execution_snapshot.frame),
                "target_agent": target_agent,
                "question_seed": question_seed,
                "question_sequence": question_sequence,
                "question_focus": question_focus,
                "trajectory_kind": self.trajectory_kind,
                "trajectory_seed": self.trajectory_seed,
                "agent_control": dict(self.trajectory_agent_control),
                "explanation_method_label": (
                    "Explanation"
                    if study_request
                    else "RCPD execution trace"
                    if explanation_mode == EXPLANATION_MODE_RCPD_TRACE
                    else "Neural evidence without program trace"
                ),
                "edited_state": (
                    serialize_warehouse_state(
                        answer.scene_edit.edited_snapshot.state,
                        selected_agent=self.selected_agent,
                    )
                    if answer.scene_edit is not None
                    else None
                ),
            }
        )
        if bilingual:
            for localized in reports.values():
                localized.update(
                    {
                        "selected_timeline_frame": int(requested.snapshot.frame),
                        "decision_evidence_frame": int(execution_snapshot.frame),
                        "target_agent": target_agent,
                        "question_seed": question_seed,
                        "question_sequence": question_sequence,
                        "question_focus": question_focus,
                        "trajectory_kind": self.trajectory_kind,
                        "trajectory_seed": self.trajectory_seed,
                        "agent_control": dict(self.trajectory_agent_control),
                        "explanation_method_label": "Explanation",
                    }
                )
            report["_language_variants"] = reports
            report["language_documents"] = {
                locale: value.get("explanation_document", {})
                for locale, value in reports.items()
            }
        self.last_explanation_report = report
        if (
            study_request
            and self.human_study.stage == "explanation"
            and answer.user_visible_explanation
        ):
            self.human_study.record_explanation(
                question=execution_plan.raw_text,
                report=report,
                response_seconds=time.perf_counter() - started_at,
                accepted_before_deadline=accepted_before_deadline,
            )
        return {
            "report": report,
            "view": self.view(),
        }

    @staticmethod
    def _validate_study_explanation_text(
        text: str,
        *,
        focus: str,
        language: str,
    ) -> bool:
        normalized = " ".join(str(text).split()).casefold()
        if not normalized or any(
            token in normalized
            for token in (
                "trace_type", "path_index", "candidate.", "action_constraint", "{",
                "selection probability", "highest probability", "概率最高",
            )
        ):
            return False
        if language == "en":
            if len(normalized.split()) > 80:
                return False
            if sum(normalized.count(mark) for mark in (".", "?", "!")) > 3:
                return False
        elif len(normalized) > 240 or normalized.count("。") > 3:
            return False
        has_number = bool(re.search(r"\d+(?:\.\d+)?", normalized))
        requirements = {
            "action": (
                any(token in normalized for token in ("executed", "waited", "moved", "执行", "等待", "移动"))
                and len(normalized) >= 35
            ),
            "energy": has_number and any(
                token in normalized for token in ("battery", "charge", "电量", "充电")
            ),
            "charge_threshold": has_number and all(
                any(token in normalized for token in group)
                for group in (
                    ("minimum", "at least", "最低", "至少"),
                    ("wait", "等待"),
                    ("battery", "电量"),
                )
            ),
            "collaboration": any(
                token in normalized for token in ("direct", "teammate", "直接", "队友")
            ),
            "allocation": any(
                token in normalized for token in ("assignment", "task", "分工", "任务")
            ),
            "collision": any(
                token in normalized for token in ("conflict", "collision", "target", "冲突", "碰撞", "目标格")
            ),
            "task": any(
                token in normalized for token in ("task", "pickup", "deliver", "任务", "取货", "交付")
            ),
        }
        return bool(requirements.get(focus, requirements["action"]))

    def _deterministic_study_explanation(
        self,
        snapshot: Any,
        *,
        target_agent: str,
        focus: str,
        language: str,
    ) -> str:
        return self.adapter.concise_study_explanation(
            snapshot,
            target_agent=target_agent,
            policy=self.policy,
            focus=focus,
            language=language,
        )

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
        requested_joint_actions = {
            **dict(proposed),
            "robot_1": action,
        }
        # Robot 1 is the participant command. Robot 2 is the unmodified,
        # sampled MAPPO Actor command. Environment dynamics alone may
        # subsequently block a move or resolve a robot conflict.
        joint_actions = dict(requested_joint_actions)
        _, rewards, terminated, truncated, info = self.environment.step(
            joint_actions,
            decision_metadata=distribution_decision_metadata(
                distributions,
                decision_source="participant_plus_pytorch_actor",
                participant_overrides={self.environment.config.human_agent_id: action},
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
                "action_execution": "independent_simultaneous_mappo_actor",
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
                "ai_network_action": proposed.get("robot_2"),
                "ai_submitted_action": joint_actions.get("robot_2"),
                "action_execution": "independent_simultaneous_mappo_actor",
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
            if round_name == "task1" and self.human_study.stage == "explanation":
                self._load_tutorial_timeline(for_explanation=True)
        return {
            "round_complete": round_complete,
            "view": self.view(),
        }

    def finish_explanation(self) -> dict[str, Any]:
        self.human_study.finish_explanation()
        assert self.human_study.assignment is not None
        self._start_collaborative_round(
            "task2",
            self.human_study.assignment.task2_seed,
        )
        return {"view": self.view()}

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
        accepted_before_deadline: bool = False,
        selected_frame: int | None = None,
        trajectory_hash: str | None = None,
        question_kind: str | None = None,
    ) -> dict[str, Any]:
        if selected_frame is not None:
            reference = self.reference_trajectory_payload()
            if trajectory_hash and not self.reference_trajectory_hash_is_compatible(
                trajectory_hash
            ):
                raise ValueError("The reference trajectory changed; reload it before asking.")
            requested_index = int(selected_frame)
            if requested_index < 1 or requested_index > self.timeline.max_index:
                raise ValueError("Select an executed reference action frame.")
            self.timeline.select(requested_index)
            self.adapter.restore(self.current.snapshot, self.policy)
        result = self.ask(
            question,
            EXPLANATION_MODE_RCPD_TRACE,
            study_request=True,
            accepted_before_deadline=accepted_before_deadline,
            study_target_agent=target_agent,
            study_question_kind=question_kind,
        )
        result["report"] = self._public_explanation_report()
        result["view"] = self.view()
        return result
