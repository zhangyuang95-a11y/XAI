from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..environment.model import PongTransition
from ..policies.rule_demo import ControllerDecision


@dataclass(frozen=True)
class ActionEvidence:
    """Immutable facts for exactly one ``before -> after`` Pong transition."""

    domain_id: str
    version: str
    study_protocol: str
    frame: int
    after_frame: int
    time_seconds: float
    player_action: str
    ai_action: str
    controller_source: str
    controller_version: str
    intent_type: str
    ai_target_x: int | None
    ai_target_ball_id: str | None
    ai_target_kind: str | None
    ai_contact_side: str | None
    distance_cells: int | None
    player_distance_cells: int | None
    eta_updates: int | None
    step_seconds: float
    requires_partner: bool
    opportunity_id: str | None
    small_first_ball_id: str | None
    small_first_feasible: bool | None
    reason: str
    candidates: tuple[dict[str, Any], ...]
    conflict_ball_ids: tuple[str, ...]
    before_player_x: int
    before_ai_x: int
    after_player_x: int
    after_ai_x: int
    balls: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    @classmethod
    def from_transition(
        cls, transition: PongTransition, decision: ControllerDecision, *,
        study_protocol: str = "coordination_explanation",
        controller_source: str = "rule_demo",
        controller_version: str = "rule-demo-grid-three-small-two-large-v6-smooth",
    ) -> "ActionEvidence":
        before, after = transition.before, transition.after
        return cls(
            domain_id=after.domain_id, version=after.version, study_protocol=study_protocol,
            frame=before.frame, after_frame=after.frame, time_seconds=before.time_seconds,
            player_action=transition.player_action, ai_action=transition.ai_action,
            controller_source=controller_source, controller_version=controller_version,
            intent_type=decision.intent_type, ai_target_x=decision.target_x,
            ai_target_ball_id=decision.target_ball_id, ai_target_kind=decision.target_kind,
            ai_contact_side=decision.contact_side, distance_cells=decision.distance_cells,
            player_distance_cells=decision.player_distance_cells,
            eta_updates=decision.eta_updates,
            step_seconds=max(0.0, float(after.time_seconds - before.time_seconds)),
            requires_partner=decision.requires_partner,
            opportunity_id=decision.opportunity_id,
            small_first_ball_id=decision.small_first_ball_id,
            small_first_feasible=decision.small_first_feasible, reason=decision.reason,
            candidates=tuple(dict(item) for item in decision.candidates),
            conflict_ball_ids=tuple(decision.conflict_ball_ids),
            before_player_x=before.player_x, before_ai_x=before.ai_x,
            after_player_x=after.player_x, after_ai_x=after.ai_x,
            balls=tuple(ball.to_dict() for ball in before.balls),
            events=tuple(dict(event) for event in transition.events),
        )

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() | {
            "balls": [dict(ball) for ball in self.balls],
            "events": [dict(event) for event in self.events],
            "candidates": [dict(item) for item in self.candidates],
            "conflict_ball_ids": list(self.conflict_ball_ids),
        }


class ExplanationEngine:
    """Natural-language answers grounded in one saved transition only."""

    def review_text(self, evidence: ActionEvidence, ball_id: str | None = None,
                    language: str = "zh") -> str:
        selected = ball_id or evidence.ai_target_ball_id
        encounter = next((event for event in evidence.events
                          if event.get("event") == "encounter" and event.get("ball_id") == selected), None)
        if language.startswith("en"):
            return self._english(evidence, selected, encounter)
        return self._chinese(evidence, selected, encounter)

    def _chinese(self, evidence: ActionEvidence, ball_id: str | None,
                 encounter: dict[str, Any] | None) -> str:
        target = ball_id or evidence.ai_target_ball_id
        if encounter is not None:
            label = _ball_label(encounter.get("ball_id"), encounter.get("ball_kind"))
            if encounter.get("outcome") == "caught":
                return f"第{evidence.frame}帧，{label}的实际接球格被覆盖，因此这次接住了。"
            penalty = 3 if encounter.get("ball_kind") == "large" else 1
            return f"第{evidence.frame}帧，{label}的实际接球格没有被完整覆盖，因此这次漏接，计为{penalty}次漏接。"
        if evidence.intent_type == "handoff" and target:
            return f"第{evidence.frame}帧，机器人1已经覆盖{_ball_label(target, evidence.ai_target_kind)}的预计接球格；机器人2没有把它当成唯一目标，继续检查其余来球。"
        if evidence.intent_type == "no_urgent":
            return f"第{evidence.frame}帧，当前没有机器人2能在预计时间内承担的来球，因此它保持位置。"
        if target:
            label = _ball_label(target, evidence.ai_target_kind)
            point = "左侧接触格" if evidence.ai_contact_side == "left" else "右侧接触格" if evidence.ai_contact_side == "right" else "预计接球格"
            distance = f"机器人2距{point}{evidence.distance_cells}格" if evidence.distance_cells is not None else "距离正在重新计算"
            eta = _eta_text(evidence.eta_updates, evidence.step_seconds)
            if evidence.requires_partner:
                teammate = (
                    f"机器人1距另一侧{evidence.player_distance_cells}格"
                    if evidence.player_distance_cells is not None else "另一侧由机器人1覆盖"
                )
                detour = _small_first_text(evidence)
                return f"第{evidence.frame}帧，机器人2提交{_action_text(evidence.ai_action)}，前往{label}的{point}（{distance}，{eta}到接球线）；{teammate}。{detour}"
            closer = _nearest_text(evidence)
            return f"第{evidence.frame}帧，机器人2提交{_action_text(evidence.ai_action)}，前往{label}的{point}（{distance}，{eta}到接球线）。{closer}"
        return f"第{evidence.frame}帧，机器人2提交{_action_text(evidence.ai_action)}。"

    def _english(self, evidence: ActionEvidence, ball_id: str | None,
                 encounter: dict[str, Any] | None) -> str:
        target = ball_id or evidence.ai_target_ball_id or "the ball"
        if encounter is not None:
            outcome = "caught" if encounter.get("outcome") == "caught" else "missed"
            return f"At frame {evidence.frame}, the actual contact cells for {target} were {outcome}."
        if evidence.intent_type == "handoff":
            return f"At frame {evidence.frame}, Robot 1 already covered {target}'s predicted contact cell, so Robot 2 kept evaluating the other balls."
        if evidence.intent_type == "no_urgent":
            return f"At frame {evidence.frame}, no ball was reachable by Robot 2 in time, so it held position."
        return f"At frame {evidence.frame}, Robot 2 submitted {evidence.ai_action} toward {target}."


class QuestionAnswerer:
    def answer(self, question: str, evidence: ActionEvidence,
               ball_id: str | None = None, language: str = "zh") -> str:
        return ExplanationEngine().review_text(evidence, ball_id, language)


def _action_text(action: str) -> str:
    return {"left": "向左移动", "right": "向右移动", "stay": "停留"}.get(action, action)


def _ball_label(ball_id: str | None, kind: str | None) -> str:
    if not ball_id:
        return "当前来球"
    return f"{'合作大球' if kind == 'large' else '小球'}{ball_id}"


def _eta_text(updates: int | None, step_seconds: float) -> str:
    if updates is None:
        return "到达时间正在重新计算"
    return f"约{max(0.0, updates * step_seconds):.1f}秒"


def _nearest_text(evidence: ActionEvidence) -> str:
    if evidence.distance_cells is None or evidence.player_distance_cells is None:
        return ""
    if evidence.distance_cells < evidence.player_distance_cells:
        return "按当前接球格计算，机器人2离它更近。"
    if evidence.distance_cells > evidence.player_distance_cells:
        return "按当前接球格计算，机器人1更近；机器人2仍保留这次可覆盖的分工。"
    return "按当前接球格计算，双方距离相同。"


def _small_first_text(evidence: ActionEvidence) -> str:
    ball_id = evidence.small_first_ball_id
    if not ball_id or evidence.small_first_feasible is None:
        return ""
    if evidence.small_first_feasible:
        return f"按当前距离估计，先接小球{ball_id}后仍赶得上这颗大球。"
    return f"按当前距离估计，若先接小球{ball_id}就赶不上这颗大球，所以先与机器人1合拢接大球。"
