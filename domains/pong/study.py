from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any, Callable, Mapping
from uuid import uuid4

from .config import PongConfig
from .environment.engine import PongEnvironment, VALID_ACTIONS
from .explanation.evidence import ActionEvidence, ExplanationEngine, QuestionAnswerer
from .policies.rule_demo import RuleDemoController
from .policies.frozen_nn import FrozenNNController
from study_three_tasks import VERSION as STUDY_VERSION, permits as study_permits, task as study_task


@dataclass
class PongStudySession:
    """Continuous three-task session with A-group explanations only in Task 2."""

    group: str = "A"
    participant_id: str = "local"
    seed: int = 260920
    condition_source: str = "manual_self_select"
    study_protocol: str | None = None
    config: PongConfig = field(default_factory=PongConfig)
    nn_policy: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None
    task: int = 1
    session_id: str = field(default_factory=lambda: uuid4().hex)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    environment: PongEnvironment = field(init=False)
    controller: Any = field(init=False)
    explanations: ExplanationEngine = field(default_factory=ExplanationEngine, init=False)
    questions: QuestionAnswerer = field(default_factory=QuestionAnswerer, init=False)
    evidence_history: dict[int, ActionEvidence] = field(default_factory=dict, init=False)
    frame_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    snapshot_history: dict[int, dict[str, Any]] = field(default_factory=dict, init=False)
    question_history: list[dict[str, Any]] = field(default_factory=list, init=False)
    review_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    completed_runs: list[dict[str, Any]] = field(default_factory=list, init=False)
    input_action: str = field(default="stay", init=False)
    paused: bool = field(default=False, init=False)
    current_decision: Any = field(default=None, init=False)
    intent_bubble: dict[str, Any] | None = field(default=None, init=False)
    _bubble_signature: tuple[Any, ...] | None = field(default=None, init=False)
    _bubble_last_update_time: float = field(default=-1.0, init=False)
    _last_target_ball_id: str | None = field(default=None, init=False)
    _last_conflict_signature: tuple[str, ...] = field(default=(), init=False)
    _review_event_ids: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.group = str(self.group).upper()
        if self.group not in {"A", "B"}:
            raise ValueError("group must be A or B")
        if self.study_protocol is None:
            self.study_protocol = self.config.study_protocol
        if self.study_protocol not in {"coordination_explanation", "rule_discovery"}:
            raise ValueError("study_protocol must be coordination_explanation or rule_discovery")
        self.config.validate()
        self.environment = PongEnvironment(self.config, seed=self.seed)
        if self.config.control_mode == "frozen_nn":
            if self.nn_policy is None:
                raise ValueError("frozen_nn control requires an explicit frozen nn_policy")
            self.controller = FrozenNNController(self.environment, self.nn_policy)
        else:
            self.controller = RuleDemoController(self.environment.config)
        self.current_decision = self.controller.choose(self.environment.frame())
        self._update_intent_bubble(self.current_decision, self.environment.frame())
        self.frame_history = [self._public_frame(self.environment.frame())]
        self.snapshot_history = {0: deepcopy(self.environment.snapshot())}

    @property
    def explanations_enabled(self) -> bool:
        return study_permits("pong", self.task, self.group)

    @property
    def review_available(self) -> bool:
        return study_permits("pong", self.task, self.group, review=True) and (self.paused or self.environment.terminal)

    @property
    def live_intent_enabled(self) -> bool:
        return self.explanations_enabled and not self.environment.terminal

    @property
    def task2_available(self) -> bool:
        return self.task == 1 and self.environment.terminal

    @property
    def next_task_available(self) -> bool:
        return self.task < 3 and self.environment.terminal

    def set_input(self, action: str) -> dict[str, Any]:
        action = str(action)
        if action not in VALID_ACTIONS:
            raise ValueError(f"Actions must be one of {VALID_ACTIONS}")
        self.input_action = action
        return self.summary()

    def set_paused(self, paused: bool) -> dict[str, Any]:
        self.paused = bool(paused)
        if self.paused:
            self.input_action = "stay"
        return self.summary()

    def tick(self, action: str | None = None) -> dict[str, Any]:
        """Advance one fixed physics frame; the web runner calls this continuously."""
        if action is not None:
            self.set_input(action)
        if self.paused or self.environment.terminal:
            return self.summary()
        before_snapshot = deepcopy(self.environment.snapshot())
        before = self.environment.frame()
        decision = self.controller.choose(before)
        self.current_decision = decision
        self._update_intent_bubble(decision, before)
        transition = self.environment.step(self.input_action, decision.action)
        evidence = ActionEvidence.from_transition(
            transition,
            decision,
            study_protocol=self.study_protocol or self.config.study_protocol,
            controller_source="frozen_nn" if self.config.control_mode == "frozen_nn" else "rule_demo",
            controller_version=self.controller.version,
        )
        self.evidence_history[transition.before.frame] = evidence
        self.frame_history.append(self._public_frame(transition.after))
        self.snapshot_history[transition.before.frame] = before_snapshot
        self.snapshot_history[transition.after.frame] = deepcopy(self.environment.snapshot())
        self._record_review_events(before, transition, decision)
        return self.summary()

    # Kept as a small compatibility alias for deterministic tests and adapters.
    def step(self, player_action: str = "stay") -> dict[str, Any]:
        return self.tick(player_action)

    def review(self, frame_index: int | None = None) -> dict[str, Any]:
        if not self.review_available:
            return {"allowed": False, "reason": "review_not_available", "answer": None}
        index = len(self.frame_history) - 1 if frame_index is None else int(frame_index)
        if index < 0 or index >= len(self.frame_history):
            return {"allowed": False, "reason": "frame_not_found", "answer": None}
        # A selected state has two distinct, named transitions: the decision
        # leaving it, and the result arriving at it.  Never silently substitute
        # one for the other when a caller asks for a particular frame.
        decision_evidence = self.evidence_history.get(index)
        result_evidence = self.evidence_history.get(index - 1) if index > 0 else None
        return {
            "allowed": True,
            "protocol_version": STUDY_VERSION,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task": self.task,
            "frame": index,
            "state": self.frame_history[index],
            "evidence": decision_evidence.to_dict() if decision_evidence else None,
            "decision_evidence": decision_evidence.to_dict() if decision_evidence else None,
            "result_evidence": result_evidence.to_dict() if result_evidence else None,
            "ball_ids": list(self.config.ball_ids),
            "events": deepcopy(self.review_events),
            "event": self._event_for_frame(index),
        }

    def ask(self, question: str, frame_index: int | None = None,
            ball_id: str | None = None) -> dict[str, Any]:
        if not self.review_available:
            return {"allowed": False, "reason": "review_not_available", "answer": None}
        if frame_index is None:
            frame_index = len(self.frame_history) - 1
        frame_index = int(frame_index)
        asks_for_result = any(token in str(question) for token in ("刚才", "结果", "接住", "漏", "发生"))
        evidence = (
            self.evidence_history.get(frame_index - 1)
            if asks_for_result and frame_index > 0
            else self.evidence_history.get(frame_index)
        )
        if evidence is None:
            return {"allowed": False, "reason": "frame_has_no_action_evidence", "answer": None}
        if ball_id is not None and ball_id not in self.config.ball_ids:
            return {"allowed": False, "reason": "unknown_ball_id", "answer": None}
        answer = self.questions.answer(str(question), evidence, ball_id)
        counterfactual = self._counterfactual_answer(str(question), frame_index)
        if counterfactual:
            answer = counterfactual
        record = {
            "protocol_version": STUDY_VERSION,
            "domain_id": "pong",
            "session_id": self.session_id,
            "task_run_id": self.run_id,
            "question": str(question),
            "frame": frame_index,
            "time_seconds": frame_index * self.config.fixed_dt,
            "ball_id": ball_id,
            "answer": answer,
            "counterfactual": bool(counterfactual),
            "evidence_binding": "result" if asks_for_result else "decision",
            "run_id": self.run_id,
            "task": self.task,
        }
        self.question_history.append(record)
        return {"allowed": True, **record}

    def advance_task(self) -> dict[str, Any]:
        if self.task >= 3:
            raise RuntimeError("Pong has already completed Task 3")
        if not self.environment.terminal:
            raise RuntimeError(f"finish Task {self.task} before starting the next task")
        self.completed_runs.append({
            "task_id": self.task, "seed": self.seed,
            "run_id": self.run_id, "missed_balls": self.environment.frame().missed_balls,
            "questions": sum(row["task"] == self.task for row in self.question_history),
        })
        self.task += 1
        self.seed = int(study_task("pong", self.task)["seed"])
        self.run_id = uuid4().hex
        self.environment.reset(seed=self.seed)
        self.controller.reset()
        self.input_action = "stay"
        self.paused = False
        self.evidence_history.clear()
        self.review_events.clear()
        self._review_event_ids.clear()
        self._last_target_ball_id = None
        self._last_conflict_signature = ()
        self.frame_history = [self._public_frame(self.environment.frame())]
        self.snapshot_history = {0: deepcopy(self.environment.snapshot())}
        self.current_decision = self.controller.choose(self.environment.frame())
        self.intent_bubble = None
        self._bubble_signature = None
        self._bubble_last_update_time = -1.0
        self._update_intent_bubble(self.current_decision, self.environment.frame())
        return self.summary()

    def _counterfactual_answer(self, question: str, frame_index: int) -> str | None:
        """Answer a short counterfactual without touching the live session."""
        lowered = question.casefold()
        if not any(token in question for token in ("如果", "假如", "要是")):
            return None
        if not any(token in lowered for token in ("不移动", "不动", "停", "左", "右", "stay")):
            return None
        snapshot = self.snapshot_history.get(frame_index)
        if snapshot is None or snapshot.get("phase") == "terminal":
            return "选定帧已经结束，无法从这一帧继续模拟。"
        counter = PongEnvironment(self.config, seed=self.seed)
        counter.restore(deepcopy(snapshot))
        alternative = "stay"
        if "向左" in question or "左移" in question:
            alternative = "left"
        elif "向右" in question or "右移" in question:
            alternative = "right"
        decision = self.controller.choose(counter.frame())
        transition = counter.step(alternative, decision.action)
        encounters = [event for event in transition.events if event.get("event") == "encounter"]
        if encounters:
            result = "；".join(
                f"{event['ball_id']}这一步{'接住' if event.get('outcome') == 'caught' else '漏接'}"
                for event in encounters
            )
        else:
            result = "这一秒内没有新的接球结算"
        return (
            f"这是从第{frame_index}帧快照开始的隔离模拟：你改为{_action_text(alternative)}，"
            f"机器人2仍按当时控制器选择{_action_text(decision.action)}；{result}。"
        )

    def summary(self) -> dict[str, Any]:
        frame = self.environment.frame()
        total = frame.total_opportunities
        payload = {
            "domain_id": self.config.domain_id,
            "study_version": STUDY_VERSION,
            "session_id": self.session_id,
            "task_run_id": self.run_id,
            "task_seed": self.seed,
            "task_purpose": study_task("pong", self.task)["purpose"],
            "explanation_allowed": self.explanations_enabled,
            "version": self.config.version,
            "paddle_width": self.config.paddle_width,
            "paddle_width_percent": self.config.paddle_width_percent,
            "grid_columns": self.config.grid_columns,
            "grid_rows": self.config.grid_rows,
            "paddle_y": self.config.paddle_y,
            "paddle_width_cells": self.config.paddle_width,
            "paddle_height_cells": self.config.paddle_height_cells,
            "fixed_dt": self.config.fixed_dt,
            "continuous_motion": True,
            "run_id": self.run_id,
            "participant_id": self.participant_id,
            "group": self.group,
            "assignment_source": self.condition_source,
            "study_protocol": self.study_protocol,
            "task": self.task,
            "phase": "review" if self.review_available else frame.phase,
            "paused": self.paused,
            "input_action": self.input_action,
            "review_available": self.review_available,
            "task2_available": self.task2_available,
            "task3_available": self.task == 2 and self.environment.terminal,
            "next_task_available": self.next_task_available,
            "completed_runs": deepcopy(self.completed_runs),
            "frame": self._public_frame(frame),
            "replay_frame_count": len(self.frame_history) if self.review_available else 0,
            "total_opportunities": total,
            "successful_opportunities": frame.successful_opportunities,
            "missed_balls": frame.missed_balls,
            "miss_rate": frame.missed_balls / total if total else 0.0,
            "missed_by_type": dict(frame.missed_by_type),
            "questions": len(self.question_history),
        }
        if self.live_intent_enabled and self.intent_bubble is not None:
            payload["intent_bubble"] = deepcopy(self.intent_bubble)
        if self.review_available:
            payload["replay_events"] = deepcopy(self.review_events)
        # The live game exposes only common gameplay facts. Controller details
        # become available to the A-group review response during Task 2 only.
        if self.review_available:
            payload.update({
                "control_mode": self.config.control_mode,
                "controller_version": self.controller.version,
            })
        return payload

    def _update_intent_bubble(self, decision: Any, frame: Any) -> None:
        """Keep a stable, human-readable intent summary for the live A view.

        The bubble is an assignment cue, not a live stopwatch. It changes when
        the target, responsibility, readiness or feasibility changes; small
        position changes do not replace the sentence every physics frame.
        """
        target_id = decision.target_ball_id
        distance = decision.distance_cells
        bucket = _distance_bucket(distance)
        action_kind = "move" if decision.action in {"left", "right"} else "wait"
        signature = (
            decision.intent_type,
            target_id,
            decision.target_kind,
            decision.contact_side,
            bool(decision.requires_partner),
            decision.commitment_state,
            action_kind,
            bucket,
        )
        now = float(frame.time_seconds)
        old = self._bubble_signature
        core_changed = old is None or signature[:-1] != old[:-1]
        bucket_changed = old is not None and signature[-1] != old[-1]
        if old is not None and not core_changed and (
            not bucket_changed or now - self._bubble_last_update_time < 0.75
        ):
            assert self.intent_bubble is not None
            return
        self._bubble_signature = signature
        self._bubble_last_update_time = now
        self.intent_bubble = {
            "frame": int(frame.frame),
            "intent_type": decision.intent_type,
            "target_ball_id": target_id,
            "target_kind": decision.target_kind,
            "contact_side": decision.contact_side,
            "distance_cells": distance,
            "player_distance_cells": decision.player_distance_cells,
            "requires_partner": bool(decision.requires_partner),
            "action": decision.action,
            "opportunity_id": decision.opportunity_id,
            "commitment_state": decision.commitment_state,
            "player_closest_ball_id": decision.player_closest_ball_id,
            "player_closest_distance_cells": decision.player_closest_distance_cells,
            "small_first_ball_id": decision.small_first_ball_id,
            "small_first_feasible": decision.small_first_feasible,
            "distance_label": _distance_label(distance),
            "text": _intent_text(decision, self.config),
            "detail_text": _intent_details(decision, self.config),
        }

    def _record_review_events(self, before: Any, transition: Any, decision: Any) -> None:
        if self._last_target_ball_id != decision.target_ball_id:
            if decision.target_ball_id is not None:
                self._append_review_event({
                    "event_id": self._event_id(before.frame, decision.target_ball_id, "target_change"),
                    "type": "target_change",
                    "frame": before.frame,
                    "time_seconds": before.time_seconds,
                    "ball_id": decision.target_ball_id,
                    "label": _target_event_label(decision),
                })
            self._last_target_ball_id = decision.target_ball_id
        conflict_signature = tuple(sorted(decision.conflict_ball_ids))
        if conflict_signature and conflict_signature != self._last_conflict_signature:
            self._append_review_event({
                "event_id": (
                    f"{self.run_id}:task{self.task}:"
                    f"{decision.opportunity_id or 'no-opportunity'}:"
                    f"{'+'.join(conflict_signature)}:competing_opportunities"
                ),
                "type": "competing_opportunities", "frame": before.frame,
                "time_seconds": before.time_seconds, "ball_id": decision.target_ball_id,
                "label": f"{', '.join(conflict_signature)} 的接球时间接近；机器人2选择了当前更紧急且可覆盖的目标。",
            })
        self._last_conflict_signature = conflict_signature
        for candidate in decision.candidates:
            if not candidate.get("viable") and not candidate.get("handoff"):
                opportunity = str(candidate.get("opportunity_id") or candidate.get("ball_id"))
                self._append_review_event({
                    "event_id": f"{self.run_id}:task{self.task}:{opportunity}:unreachable",
                    "type": "unreachable_opportunity", "frame": before.frame,
                    "time_seconds": before.time_seconds, "ball_id": candidate.get("ball_id"),
                    "label": f"{candidate.get('ball_id')} 在这次预计接球前双方距离不足，不能承诺接住。",
                })
        for raw in transition.events:
            kind = raw.get("event")
            if kind == "encounter":
                outcome = raw.get("outcome")
                event_type = "catch" if outcome == "caught" else "miss"
                if raw.get("ball_kind") == "large" and outcome == "caught":
                    event_type = "cooperative_catch"
                label = _encounter_event_label(raw)
                self._append_review_event({
                    "event_id": self._event_id(int(raw.get("frame", transition.after.frame)), str(raw.get("encounter_id") or raw.get("ball_id")), event_type),
                    "type": event_type,
                    "frame": int(raw.get("frame", transition.after.frame)),
                    "time_seconds": float(raw.get("time_seconds", transition.after.time_seconds)),
                    "ball_id": raw.get("ball_id"),
                    "label": label,
                })
            elif kind == "miss_scored":
                penalty = int(raw.get("miss_penalty", 1))
                self._append_review_event({
                    "event_id": self._event_id(int(raw.get("frame", transition.after.frame)), str(raw.get("encounter_id") or raw.get("ball_id")), "miss_scored"),
                    "type": "miss_scored",
                    "frame": int(raw.get("frame", transition.after.frame)),
                    "time_seconds": float(raw.get("time_seconds", transition.after.time_seconds)),
                    "ball_id": raw.get("ball_id"),
                    "label": f"{raw.get('ball_id', '该球')}漏接已计为{penalty}次漏接。",
                })

    def _event_id(self, frame: int, subject: str, event_type: str) -> str:
        return f"{self.run_id}:task{self.task}:frame{frame}:{subject}:{event_type}"

    def _append_review_event(self, event: dict[str, Any]) -> None:
        event_id = str(event["event_id"])
        if event_id in self._review_event_ids:
            return
        self._review_event_ids.add(event_id)
        self.review_events.append(event)

    def _event_for_frame(self, frame_index: int) -> dict[str, Any] | None:
        candidates = [event for event in self.review_events if int(event.get("frame", -1)) == int(frame_index)]
        return deepcopy(candidates[0]) if candidates else None

    @staticmethod
    def _public_frame(frame: Any) -> dict[str, Any]:
        data = frame.to_dict()
        # Outcome and ball identity are shared gameplay feedback. Coverage,
        # target and controller facts remain in the post-game evidence only.
        data["encounter_events"] = [
            {
                key: event[key]
                for key in ("event", "encounter_id", "ball_id", "ball_kind", "outcome", "time_seconds", "miss_penalty")
                if key in event
            }
            for event in data.get("encounter_events", [])
            if event.get("event") in {"encounter", "miss_scored"}
        ]
        return data


def _action_text(action: str) -> str:
    return {"left": "向左", "right": "向右", "stay": "停止移动"}.get(action, action)


def _distance_bucket(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    if distance <= 0:
        return "0"
    if distance <= 2:
        return "1-2"
    if distance <= 5:
        return "3-5"
    return "6+"


def _distance_label(distance: float | None) -> str:
    if distance is None:
        return ""
    return f"距离约 {_cells(distance)} 格"


def _intent_text(decision: Any, config: PongConfig) -> str:
    if decision.intent_type == "handoff" and decision.target_ball_id:
        return f"你已守住小球{decision.target_ball_id}，我继续分担其余来球。"
    if decision.intent_type == "no_urgent":
        return "当前没有我能及时承担的来球，我保持位置等待新的分工。"
    if decision.target_ball_id is None:
        return "当前没有需要接的球，我先停留观察。"
    kind = "合作大球" if decision.target_kind == "large" else "小球"
    target = f"{kind}{decision.target_ball_id}"
    if decision.requires_partner:
        side = "左" if decision.contact_side == "left" else "右"
        if decision.commitment_state == "waiting_for_large":
            return f"我已守住{target}的{side}侧，会等这次接球结算；你需要覆盖另一侧。"
        return f"我负责{target}的{side}侧；你需要同时覆盖另一侧。"
    player_ball = decision.player_closest_ball_id
    if player_ball and player_ball != decision.target_ball_id:
        return f"我去接{target}；小球{player_ball}离你更近，建议你优先守它。"
    return f"我去接{target}，因为它的接球位置离我更近。"


def _intent_details(decision: Any, config: PongConfig) -> str:
    """Short action guidance; intentionally omits an always-changing clock."""
    if decision.target_ball_id is None:
        return ""
    distance = "距离正在重新计算" if decision.distance_cells is None else f"我离接球位置约{_cells(decision.distance_cells)}格"
    detail = distance
    if decision.requires_partner:
        if decision.player_distance_cells is not None:
            detail += f"；你离另一侧约{_cells(decision.player_distance_cells)}格"
        if decision.small_first_ball_id and decision.small_first_feasible is not None:
            if decision.small_first_feasible:
                detail += f"。先接小球{decision.small_first_ball_id}后仍赶得上这颗大球"
            else:
                detail += f"。先接小球{decision.small_first_ball_id}会赶不上这颗大球"
        return detail + "。"
    if decision.player_distance_cells is not None and decision.distance_cells is not None:
        if decision.distance_cells < decision.player_distance_cells:
            detail += "；按当前接球格计算，我离它更近"
        elif decision.distance_cells > decision.player_distance_cells:
            detail += "；按当前接球格计算，机器人1更近"
        else:
            detail += "；按当前接球格计算，我们距离相同"
    return detail + "。"


def _cells(distance: float) -> str:
    return str(max(0, int(round(distance))))


def _target_event_label(decision: Any) -> str:
    kind = "合作大球" if decision.target_kind == "large" else "小球"
    side = ""
    if decision.contact_side:
        side = f"的{'左侧' if decision.contact_side == 'left' else '右侧'}接触点"
    return f"目标切换为{kind}{decision.target_ball_id}{side}。"


def _encounter_event_label(event: dict[str, Any]) -> str:
    ball_id = event.get("ball_id", "该球")
    if event.get("outcome") == "caught":
        if event.get("ball_kind") == "large":
            return f"合作大球{ball_id}由两块球拍完成共同接球。"
        return f"小球{ball_id}被接住。"
    return f"{('合作大球' if event.get('ball_kind') == 'large' else '小球')}{ball_id}这次没有被接住。"
