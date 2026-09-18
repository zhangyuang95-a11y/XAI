from __future__ import annotations

from dataclasses import dataclass
import re
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
    ai_target_x: float | None
    ai_target_ball_id: str | None
    ai_target_kind: str | None
    ai_contact_side: str | None
    distance_cells: float | None
    player_distance_cells: float | None
    eta_updates: int | None
    time_until_contact: float | None
    requires_partner: bool
    opportunity_id: str | None
    commitment_state: str
    player_closest_ball_id: str | None
    player_closest_distance_cells: float | None
    small_first_ball_id: str | None
    small_first_feasible: bool | None
    reason: str
    candidates: tuple[dict[str, Any], ...]
    conflict_ball_ids: tuple[str, ...]
    before_player_x: float
    before_ai_x: float
    after_player_x: float
    after_ai_x: float
    balls: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]

    @classmethod
    def from_transition(
        cls, transition: PongTransition, decision: ControllerDecision, *,
        study_protocol: str = "coordination_explanation",
        controller_source: str = "rule_demo",
        controller_version: str = "rule-demo-continuous-24x14-v7",
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
            time_until_contact=decision.time_until_contact,
            requires_partner=decision.requires_partner,
            opportunity_id=decision.opportunity_id,
            commitment_state=decision.commitment_state,
            player_closest_ball_id=decision.player_closest_ball_id,
            player_closest_distance_cells=decision.player_closest_distance_cells,
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
                return f"{label}的实际接球格被覆盖，因此这次接住了。"
            penalty = 3 if encounter.get("ball_kind") == "large" else 1
            return f"{label}的实际接球格没有被完整覆盖，因此这次漏接，计为{penalty}次漏接。"
        if evidence.intent_type == "handoff" and target:
            return f"机器人1已经覆盖{_ball_label(target, evidence.ai_target_kind)}的预计接球格；机器人2没有把它当成唯一目标，继续检查其余来球。"
        if evidence.intent_type == "no_urgent":
            return "当前没有机器人2能在预计时间内承担的来球，因此它保持位置。"
        if target:
            label = _ball_label(target, evidence.ai_target_kind)
            point = "左侧接触格" if evidence.ai_contact_side == "left" else "右侧接触格" if evidence.ai_contact_side == "right" else "预计接球格"
            distance = f"机器人2距{point}约{_cells(evidence.distance_cells)}格" if evidence.distance_cells is not None else "距离正在重新计算"
            if evidence.requires_partner:
                teammate = (
                    f"机器人1距另一侧约{_cells(evidence.player_distance_cells)}格"
                    if evidence.player_distance_cells is not None else "另一侧由机器人1覆盖"
                )
                detour = _small_first_text(evidence)
                state = "已就位等待这次接球结算" if evidence.commitment_state == "waiting_for_large" else "正在靠近分配侧"
                return f"机器人2{_action_text(evidence.ai_action)}，前往{label}的{point}（{distance}）；{teammate}，{state}。{detour}"
            closer = _nearest_text(evidence)
            return f"机器人2{_action_text(evidence.ai_action)}，前往{label}的{point}（{distance}）。{closer}"
        return f"机器人2{_action_text(evidence.ai_action)}。"

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
        question = question.strip()
        if language.startswith("en"):
            return ExplanationEngine().review_text(evidence, ball_id, language)
        if not question:
            return "请问一个具体问题，例如‘为什么接B1’或‘我该去哪里？’。"
        mentioned = re.search(r"[AB][123]", question.upper())
        target = mentioned.group(0) if mentioned else ball_id or evidence.ai_target_ball_id
        if any(token in question for token in ("漏", "接住", "结果", "发生")):
            encounter = next((item for item in evidence.events if item.get("event") == "encounter"
                              and (target is None or item.get("ball_id") == target)), None)
            if encounter is None:
                return f"这次动作记录里没有{target or '目标球'}的接球结算，请选择发生接球的帧。"
            if encounter.get("outcome") == "caught":
                return (f"{_ball_label(target, encounter.get('ball_kind'))}实际接住了；"
                        + ("双方分别覆盖了两侧。" if encounter.get("ball_kind") == "large" else
                           "至少一块球拍覆盖了接触点。"))
            coverage = encounter.get("coverage", {})
            if encounter.get("ball_kind") == "large":
                player = coverage.get("player", ())
                ai = coverage.get("ai", ())
                return (f"{_ball_label(target, 'large')}漏接，计3次。实际接球时你覆盖了"
                        f"{sum(bool(value) for value in player)}侧，机器人2覆盖了"
                        f"{sum(bool(value) for value in ai)}侧，双方未分别覆盖左右接触点。")
            return f"{_ball_label(target, 'small')}漏接，计1次；两块球拍都没有覆盖接触点。"
        if any(token in question for token in ("为什么不", "为何不", "怎么不", "没去接")):
            if target is None:
                return "请指出你问的是A1、A2、A3、B1还是B2。"
            if target == evidence.ai_target_ball_id:
                return f"这一帧实际选择了{_ball_label(target, evidence.ai_target_kind)}。"
            candidate = next((item for item in evidence.candidates if item.get("ball_id") == target), None)
            if candidate is None:
                return f"这帧没有{target}的可核验接球预测，不能断定它一定不可达。"
            if not candidate.get("viable"):
                return f"{target}在当前预测中不满足接球可达条件，因此未列入可执行分工。"
            return f"{target}当时也可能接到；控制器选择了{evidence.ai_target_ball_id or '另一方案'}，需要结合其他同时来球比较。"
        if any(token in question for token in ("我该", "我应该", "我去", "怎么配合", "哪边")):
            if evidence.requires_partner and evidence.ai_target_ball_id:
                other = "右侧" if evidence.ai_contact_side == "left" else "左侧"
                return f"请你覆盖合作大球{evidence.ai_target_ball_id}的{other}；机器人2负责另一侧。你必须实际到位。"
            return (f"机器人2当前负责{evidence.ai_target_ball_id}，你可关注另一颗能及时接到的球。"
                    if evidence.ai_target_ball_id else "当前没有可核验的玩家分工，请查看下一个接球机会。")
        if any(token in question for token in ("为什么停", "为什么不动", "为什么等待")):
            if evidence.commitment_state == "waiting_for_large":
                return f"机器人2已覆盖{evidence.ai_target_ball_id}的分配侧，等待该次接球结算。"
            return f"机器人2本步提交{_action_text(evidence.ai_action)}；这帧没有已证实的等待原因。"
        if any(token in question for token in ("NN", "神经", "规则", "谁控制")):
            return f"这帧提交动作由{evidence.controller_source}控制器产生；记录的动作是{_action_text(evidence.ai_action)}。"
        if any(token in question for token in ("为什么", "为何", "哪个球", "什么球", "准备接", "接什么")):
            return ExplanationEngine().review_text(evidence, target, language)
        return "我能回答这一帧的分工、动作和接球结果；请指出具体球或想问的选择。"


def _action_text(action: str) -> str:
    return {"left": "向左移动", "right": "向右移动", "stay": "停留"}.get(action, action)


def _ball_label(ball_id: str | None, kind: str | None) -> str:
    if not ball_id:
        return "当前来球"
    return f"{'合作大球' if kind == 'large' else '小球'}{ball_id}"


def _cells(distance: float) -> str:
    return str(max(0, int(round(distance))))


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
