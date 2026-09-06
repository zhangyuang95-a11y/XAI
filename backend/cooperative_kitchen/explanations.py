"""Frame-bound neural evidence, post-hoc programs and isolated counterfactuals.

The language model parses intent and chooses facts. It cannot write unchecked
kitchen facts or claim that a tree branch caused a neural action.
"""
from __future__ import annotations
from dataclasses import asdict
import copy
import hashlib
import json
import re
from pathlib import Path

from backend.adapters.base import EvidenceFact
from core.program import ExecutableProgram
from env.cooperative_kitchen import CooperativeKitchen, OBSERVATION_FEATURES
from .policy import ACTIONS
from .llm import KitchenLLMClient, PARSER_PROMPT, ANSWER_PROMPT, redact_question

ACTION_TEXT = {"UP": ("向上", "move up"), "DOWN": ("向下", "move down"),
               "LEFT": ("向左", "move left"), "RIGHT": ("向右", "move right"),
               "INTERACT": ("交互", "interact"), "WAIT": ("等待", "wait")}
ITEM_TEXT = {None: ("空", "empty"), "onion": ("洋葱", "onion"), "plate": ("盘子", "plate"), "soup": ("汤", "soup")}


def readable_trace(steps, language):
    """State observations along a matched branch; raw predicates stay in evidence."""
    lang = int(language == "en"); descriptions = []
    stations = {"ingredient": ("食材处", "the ingredient dispenser"), "plate": ("盘子处", "the plate dispenser"),
                "pot": ("锅", "the pot"), "serve": ("出餐口", "the serving station"),
                "left_trash": ("左侧垃圾桶", "the left bin"), "right_trash": ("右侧垃圾桶", "the right bin"),
                "upper_counter": ("上方工作台", "the upper counter"), "lower_counter": ("下方工作台", "the lower counter")}
    for step in steps:
        key, value = step.feature, step.observed_value
        if key.startswith("self_holding_") and value > .5:
            item = key.removeprefix("self_holding_"); item = None if item == "empty" else item
            descriptions.append((f"队友手中为{ITEM_TEXT[item][0]}", f"the teammate’s hand is {ITEM_TEXT[item][1]}")[lang])
        elif key.endswith("_interaction_distance"):
            station = key.removesuffix("_interaction_distance")
            if station in stations:
                n = round(value * 20)
                descriptions.append((f"队友无法直接到达{stations[station][0]}" if n >= 99 else f"队友到{stations[station][0]}完成一次交互最短需要 {n} 个动作",
                                     f"the teammate cannot directly reach {stations[station][1]}" if n >= 99 else f"interacting with {stations[station][1]} takes at least {n} actions")[lang])
        elif key == "pot_ready":
            descriptions.append(("锅已煮熟" if value > .5 else "锅尚未煮熟", "the pot is ready" if value > .5 else "the pot is not ready")[lang])
        elif key == "role_right":
            descriptions.append(("队友位于右侧" if value > .5 else "队友位于左侧", "the teammate is on the right" if value > .5 else "the teammate is on the left")[lang])
    return list(dict.fromkeys(descriptions))[:3]


def snapshot_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def isolated_branch(policy, snapshot, human_actions, ai_override=None):
    if not 1 <= len(human_actions) <= 3 or any(a not in ACTIONS for a in human_actions):
        raise ValueError("Counterfactuals require 1–3 primitive player actions")
    if ai_override is not None and ai_override not in ACTIONS:
        raise ValueError("Unknown teammate action")
    before = snapshot_hash(snapshot)
    env = CooperativeKitchen(); env.restore(copy.deepcopy(snapshot))
    frames = []
    for i, human in enumerate(human_actions):
        if env.public_view()["done"]: break
        actions, distributions = policy.act({"ai": env.observations()["ai"]})
        proposed = actions["ai"]
        actual = ai_override if i == 0 and ai_override else proposed
        result = env.step({"human": human, "ai": actual}, include_state=True)
        frames.append({"state": result["state"], "events": result["events"], "human_action": human,
                       "ai_proposal": proposed, "ai_action": actual, "distribution": distributions["ai"]})
    if snapshot_hash(snapshot) != before:
        raise AssertionError("Counterfactual mutated its source snapshot")
    return {"source_sha256": before, "assumptions": {"human_actions": list(human_actions),
            "ai_override_first": ai_override, "future_ai": "frozen_actor_argmax"},
            "frames": frames, "final_state": env.public_view()}


def _validate_intent(value):
    if not isinstance(value, dict): return None
    if set(value) - {"intent", "actor", "action", "steps", "anchor", "rule", "repeat"}: return None
    for key in ("intent", "actor", "action", "anchor", "rule"):
        if value.get(key) is not None and not isinstance(value[key], str): return None
    if value.get("intent") not in {"why", "waiting", "alternative", "counterfactual", "rules", "failure", "clarify"}: return None
    if value.get("actor", "ai") not in {"human", "ai"}: return None
    if value.get("action") not in (*ACTIONS, None): return None
    if type(value.get("steps", 1)) is not int or not 1 <= value.get("steps", 1) <= 3: return None
    if value.get("anchor", "next") not in {"next", "executed"}: return None
    if value.get("rule") not in {None, "recipe", "controls", "score", "handoff"}: return None
    if value["intent"] == "rules" and value.get("rule") is None: return None
    if type(value.get("repeat", False)) is not bool: return None
    if value["intent"] in {"alternative", "counterfactual"} and value.get("action") is None: return None
    return {"intent": "why", "actor": "ai", "action": None, "steps": 1, "anchor": "next", "rule": None, "repeat": False, **value}


_EXPLICIT_ACTION_PATTERNS = {
    "UP": r"(?:向上|往上|朝上|上移)|\b(?:move|go|walk|step)(?:\s+one\s+step)?\s+up(?:wards?)?\b|^up\b",
    "DOWN": r"(?:向下|往下|朝下|下移)|\b(?:move|go|walk|step)(?:\s+one\s+step)?\s+down(?:wards?)?\b|^down\b",
    "LEFT": r"(?:向左|往左|朝左|左移)|\b(?:move|go|walk|step)(?:\s+one\s+step)?\s+left\b|^left\b",
    "RIGHT": r"(?:向右|往右|朝右|右移)|\b(?:move|go|walk|step)(?:\s+one\s+step)?\s+right\b|^right\b",
    "INTERACT": r"交互|按(?:下)?\s*e(?:键)?|\binteract(?:ing)?\b|\bpress(?:ing)?\s+(?:the\s+)?e(?:\s+key)?\b",
    "WAIT": r"等待|原地等|\bwait(?:ing)?\b",
}


def _explicit_intent_constraint(question):
    """Recognize only clear surface contrasts, never infer state or motives.

    This constrains a cloud parse; it does not generate an intent or answer.
    Ambiguous descriptions, multiple named actions and negative player inputs
    have no inferred action constraint. In particular, 'pick it up' is not UP.
    """
    text = question.strip().lower().replace("’", "'")
    text = re.sub(r"^(?:请问|想问一下)[，,：:\s]*", "", text)
    alternative = None
    chinese = re.match(r"^(?:为什么|为何|为啥)(?P<subject>[^，,。？！?!]{0,20}?)(?:没有|不|没)(?P<action>.+)$", text)
    if chinese and not re.search(r"我|玩家|人类", chinese.group("subject")):
        alternative = chinese.group("action")
    english = re.match(r"^why\s+(?P<subject>[^?!.]{0,60}?)(?:\bnot\b|\b(?:doesn't|don't|didn't|won't|wouldn't|isn't)\b)\s*(?P<action>.+)$", text)
    if english and not re.search(r"\b(?:i|we|player|human)\b", english.group("subject") + " " + english.group("action").split(",")[0]):
        alternative = english.group("action")
    player = re.match(r"^(?:(?:如果|假如|假设|要是|倘若)\s*我|我\s*(?:如果|假如|要是))(?P<action>.+)$", text)
    if not player:
        player = re.match(r"^(?:if|what\s+if|suppose|assuming|what\s+(?:happens|would\s+happen)\s+if)\s+i\s+(?P<action>.+)$", text)
    segment = player.group("action") if player else alternative
    if segment is None:
        return None
    if player and re.match(r"\s*(?:不|没|别|don't\b|do\s+not\b|never\b)", segment):
        return None
    actions = [action for action, pattern in _EXPLICIT_ACTION_PATTERNS.items() if re.search(pattern, segment, re.I)]
    if len(actions) != 1:
        return None
    return {"intent": "counterfactual" if player else "alternative", "actor": "human" if player else "ai", "action": actions[0]}


def _semantic_intent_error(parsed, constraint):
    # Clarification is a safe unresolved outcome, never a successful answer.
    if constraint and parsed["intent"] != "clarify" and any(parsed[key] != value for key, value in constraint.items()):
        return "explicit_question_semantics_mismatch"
    return None


class ExplanationEngine:
    def __init__(self, policy, program_path=None, *, client=None, extraction_report=None):
        self.policy = policy
        self.client = client if client is not None else KitchenLLMClient()
        self.program = None
        self.program_error = None
        self.program_verified = False
        if program_path:
            try:
                with open(program_path) as handle: value = json.load(handle)
                program = ExecutableProgram.from_dict(value)
                if tuple(program.action_names) != ACTIONS or set(program.feature_names) != set(OBSERVATION_FEATURES):
                    raise ValueError("Program observation/action mismatch")
                if program.metadata.get("actor_sha256") != policy.artifact_sha256:
                    raise ValueError("Program belongs to another Actor")
                if program.metadata.get("action_legality_features"):
                    raise ValueError("Kitchen directions must not be masked")
                self.program = program
                # The deployed service supplies the report already verified by
                # the release manifest. Local extraction tools may still use a
                # sibling report. An explicitly supplied empty/failed report
                # must never fall back to a different on-disk audit.
                report = copy.deepcopy(extraction_report)
                if report is None:
                    report_path = Path(program_path).parent / "extraction_report.json"
                    if report_path.exists():
                        report = json.loads(report_path.read_text())
                if report is not None:
                    if not isinstance(report, dict):
                        raise ValueError("Extraction report must be an object")
                    self.program_verified = (report.get("extraction_gate") is True
                        and report.get("actor_sha256") == policy.artifact_sha256
                        and report.get("program_sha256") == hashlib.sha256(Path(program_path).read_bytes()).hexdigest())
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.program_error = type(exc).__name__

    def _parse(self, question, kind, anchor, diagnostics):
        base = {"intent": kind, "actor": "human" if kind == "counterfactual" else "ai", "action": "WAIT" if kind == "counterfactual" else None,
                "steps": 3 if kind == "counterfactual" else 1, "anchor": anchor, "rule": None, "repeat": kind == "counterfactual"}
        if kind in {"why", "waiting", "counterfactual"}:
            return base
        beyond_horizon = re.search(r"(?:[4-9]|\d{2,}|four|five|six|seven|eight|nine|ten|四|五|六|七|八|九|十)\s*(?:步|steps?|turns?)", question, re.I)
        constraint = _explicit_intent_constraint(question)
        if self.client.configured:
            feedback = None
            for attempt in range(2):
                payload = {"question": redact_question(question), "selected_anchor": anchor}
                if feedback is not None:
                    payload["validation_feedback"] = feedback
                value, log = self.client.request(PARSER_PROMPT, payload)
                record = {"stage": "parse", "attempt": attempt + 1, **log}
                diagnostics["calls"].append(record)
                parsed = _validate_intent(value)
                if parsed is not None:
                    record["parsed_intent"] = dict(parsed)
                    semantic_error = _semantic_intent_error(parsed, constraint)
                    if semantic_error:
                        record["validation_error"] = semantic_error
                        feedback = {"issue": semantic_error, "explicit_question_constraint": constraint}
                        continue
                    diagnostics["parser_verified"] = True
                    diagnostics["parsed_intent"] = dict(parsed)
                    if beyond_horizon and parsed["intent"] in {"counterfactual", "alternative"}:
                        diagnostics["parser_rejection"] = "requested_horizon_exceeds_limit"
                        return {**parsed, "intent": "clarify"}
                    return parsed
                record["validation_error"] = "invalid_intent_schema"
                feedback = {"issue": "invalid_intent_schema"}
            if constraint:
                diagnostics["parser_rejection"] = "explicit_question_not_verified"
                return {**base, "intent": "clarify"}
        # Conservative offline behavior: recognize a small explicit vocabulary,
        # otherwise ask for clarification instead of inventing an interpretation.
        text = question.strip().lower()
        if re.search(r"(几|多少|多久|how many|how long).*(洋葱|煮|onion|cook)", text) or any(x in text for x in ("菜谱", "recipe")):
            return {**base, "intent": "rules", "rule": "recipe"}
        if any(x in text for x in ("计分", "得分", "score")):
            return {**base, "intent": "rules", "rule": "score"}
        if any(x in text for x in ("失败", "结束", "fail", "ended")):
            return {**base, "intent": "failure"}
        if beyond_horizon:
            return {**base, "intent": "clarify"}
        if any(x in text for x in ("如果我等待", "if i wait")) and not re.search(r"[4-9]|\d{2}", text):
            return {**base, "intent": "counterfactual", "actor": "human", "action": "WAIT", "steps": 3, "repeat": True}
        if text in {"为什么选择这个动作", "why this action", "why this action?"}: return {**base, "intent": "why"}
        if text in {"你在等什么", "what are you waiting for", "what are you waiting for?"}: return {**base, "intent": "waiting"}
        return {**base, "intent": "clarify"}

    def generate(self, snapshot, question="", *, kind="free", language="zh", anchor="next"):
        language = "en" if language == "en" else "zh"; lang = int(language == "en")
        source = snapshot_hash(snapshot)
        env = CooperativeKitchen(); env.restore(copy.deepcopy(snapshot)); state = env.public_view()
        diagnostics = {"calls": [], "parser_verified": False, "llm_success": False,
                       "source_sha256": source, "actor_sha256": self.policy.artifact_sha256,
                       "configuration": self.client.config, "program_error": self.program_error}
        intent = self._parse(question, kind, anchor, diagnostics)
        # This interface receives the selected frame's state, not a previous
        # distribution. Do not mislabel a recomputed next action as executed.
        if intent["anchor"] == "executed" or intent["actor"] == "human" and intent["intent"] in {"why", "waiting", "alternative"}:
            intent["intent"] = "clarify"
        facts = []
        def add(id, predicate, value, zh, en, group="state"):
            fact = EvidenceFact(id, predicate, ("ai",), value, (group,), (zh, en))
            facts.append({**asdict(fact), "text": (zh, en)[lang]})
        turn = state["turn"]
        if state["done"]:
            add("frame", "frame_binding", turn, f"所选画面为第 {turn} 步，回合已结束，不会再执行下一动作。", f"The selected frame is step {turn}. This round has ended and will execute no next action.", "binding")
            if intent["intent"] not in {"rules", "failure", "clarify"}: intent["intent"] = "failure"
        else:
            add("frame", "frame_binding", turn, f"以下回答基于第 {turn} 步结束后的所选画面，讨论队友下一次决策。", f"This answer uses the selected state after step {turn} and concerns the teammate’s next decision.", "binding")
        mandatory = ["frame"]
        actions, distributions = self.policy.act({"ai": env.observations()["ai"]})
        selected = actions["ai"]; distribution = distributions["ai"]
        probability = distribution["probabilities"][ACTIONS.index(selected)]
        name = ACTION_TEXT[selected]
        add("actor", "neural_argmax", {"action": selected, "probability": probability},
            f"冻结神经策略选择“{name[0]}”，输出概率为 {probability:.1%}。正式动作由该分布的最大值确定。",
            f"The frozen neural policy selects “{name[1]}”, with output probability {probability:.1%}; execution uses the distribution’s maximum.", "neural")
        if not state["done"] and intent["intent"] in {"why", "waiting", "alternative"}: mandatory.append("actor")
        holding = state["actors"][1]["holding"]
        add("holding", "held_item", holding, f"队友当前手持：{ITEM_TEXT[holding][0]}。", f"The teammate’s held item is {ITEM_TEXT[holding][1]}.")
        values = list(state["counters"].values())
        add("counters", "counter_items", values, f"上、下共享工作台分别为：{ITEM_TEXT[values[0]][0]}、{ITEM_TEXT[values[1]][0]}。每张台只能放一件物品。",
            f"The upper and lower shared counters contain {ITEM_TEXT[values[0]][1]} and {ITEM_TEXT[values[1]][1]}. Each holds one item.")
        pot = state["pot"]
        add("pot", "pot_state", pot, f"锅中有 {pot['ingredients']} 份洋葱，剩余烹饪步数为 {pot['remaining']}，{'已煮熟' if pot['ready'] else '尚未煮熟'}。",
            f"The pot has {pot['ingredients']} onions, {pot['remaining']} cooking steps remaining, and is {'ready' if pot['ready'] else 'not ready'}.")
        if self.program and not state["done"]:
            features = dict(zip(OBSERVATION_FEATURES, map(float, env.observations()["ai"])))
            execution = self.program.execute(features)
            trace = asdict(execution.trace)
            matches = execution.action == selected
            diagnostics["program_action_matches"] = matches
            add("program", "program_agreement", {"action": execution.action, "matches": matches, "trace": trace},
                ("抽取程序在这帧与神经策略动作一致。它是事后近似，不能据此确定神经网络的内部原因。" if matches else "抽取程序在这帧与神经策略不一致，不能用该程序分支解释为确定原因。"),
                ("The extracted program agrees with the neural action here. It is a post-hoc approximation, not proof of the network’s internal cause." if matches else "The extracted program disagrees with the neural action here; its branch cannot be asserted as the reason."), "program")
            if intent["intent"] in {"why", "waiting", "alternative"}: mandatory.append("program")
            if not self.program_verified:
                add("program_candidate", "program_acceptance", False, "该抽取程序尚未通过整体保真度验收，当前仅用于诊断。", "This extracted program has not passed its overall fidelity gate and is diagnostic only.", "program")
                if intent["intent"] in {"why", "waiting", "alternative"}: mandatory.append("program_candidate")
            if matches and execution.trace.tree_steps:
                zh_terms = readable_trace(execution.trace.tree_steps, "zh")
                en_terms = readable_trace(execution.trace.tree_steps, "en")
                if zh_terms:
                    add("branch", "program_trace", trace, "该近似程序经过的分支使用了这些状态信息：" + "；".join(zh_terms) + "。", "The approximate program’s executed branch uses these state observations: " + "; ".join(en_terms) + ".", "program")
        else:
            add("no_program", "program_unavailable", True, "当前没有与该神经策略匹配的已核验抽取程序；可确认动作输出，不能确定其内部原因。", "No verified extracted program matches this neural policy; the action output is known, but its internal cause is not.", "program")
            if intent["intent"] in {"why", "waiting", "alternative"}: mandatory.append("no_program")
        add("recipe", "recipe_rule", {"onions": 3, "subsequent_steps": 4}, "每锅需要三份洋葱，第三份入锅后再经过四个联合步煮熟；手持盘子面向锅交互可装汤，汤需交回左侧出餐。", "Each soup needs three onions and four subsequent joint steps after the third onion is added. Interact with the pot while holding a plate to collect soup, then hand it left for serving.", "rule")
        add("controls", "control_rule", True, "方向键移动并调整朝向；即使不能移动也会转向。E 交互，空格等待；每个操作消耗一步，阅读、提问和回放不推进时间。", "Directions move and turn; blocked movement still turns. E interacts and Space waits. Each action costs a step; reading, questions and replay do not advance time.", "rule")
        add("score", "score_rule", {"orders": state["orders"], "turn": turn}, f"得分为 100 × 出餐数 − 步数，当前为 {100*state['orders']-turn}；目标是在 {state['maxSteps']} 步内完成 {state['targetOrders']} 份汤。", f"Score is 100 × soups served − steps, currently {100*state['orders']-turn}; the target is {state['targetOrders']} soups within {state['maxSteps']} steps.", "rule")
        add("handoff", "handoff_rule", True, "每人只能手持一件物品，每张共享台只能放一件物品。共享台已有物品时不能再放入另一件，需先取走原物品腾出空间。共享台用于左右区域交接；双方交互读取同一步行动前的物品状态，因此不能在同一步放下后立即被另一方取走；同一物品争用按步数奇偶决定优先方。", "Each actor can carry one item and each shared counter holds one item. An occupied counter cannot accept another item; its existing item must first be removed to make space. Shared counters transfer items between regions. Both interactions use the pre-step item state, so an item placed this step cannot be picked up this step; contested items use alternating step-parity priority.", "rule")
        branch = None
        if intent["intent"] == "waiting":
            add("wait_status", "wait_selected", selected == "WAIT", "队友此时选择等待。等待本身不代表错误；当前可观察状态如下。" if selected == "WAIT" else "队友在这帧下一次选择的动作不是等待。", "The teammate selects WAIT here. Waiting alone is not an error; the observable state is listed below." if selected == "WAIT" else "The teammate’s next selected action at this frame is not WAIT.", "neural")
            mandatory += ["wait_status", "holding", "counters", "pot"]
        elif intent["intent"] == "rules":
            mandatory.append(intent["rule"] or "controls")
            if intent["rule"] == "handoff": mandatory += ["counters", "holding"]
        elif intent["intent"] == "failure":
            text = ("该回合已完成目标。", "The target was completed.") if state["done"] and state["orders"] >= state["targetOrders"] else (("步数已用尽，出餐数尚未达到目标。", "The step budget was exhausted before reaching the soup target.") if state["done"] else ("所选画面中回合尚未结束，不能称为失败。", "The selected state is not terminal; it cannot be described as a failure."))
            add("terminal", "terminal_status", {"done": state["done"], "reason": state["reason"]}, *text, group="event"); mandatory += ["terminal", "score"]
        elif intent["intent"] in {"counterfactual", "alternative"}:
            action = intent["action"]
            if intent["intent"] == "alternative":
                alt_probability = distribution["probabilities"][ACTIONS.index(action)]
                add("alternative", "alternative_probability", {"action": action, "probability": alt_probability}, f"替代动作“{ACTION_TEXT[action][0]}”的神经输出概率为 {alt_probability:.1%}。", f"The alternative action “{ACTION_TEXT[action][1]}” has neural output probability {alt_probability:.1%}.", "neural")
                mandatory.append("alternative")
            seq = ([action] * intent["steps"] if intent["repeat"] else [action] + ["WAIT"] * (intent["steps"] - 1)) if intent["actor"] == "human" else ["WAIT"] * intent["steps"]
            override = action if intent["actor"] == "ai" else None
            branch = isolated_branch(self.policy, snapshot, seq, override)
            sequence_text = " → ".join(ACTION_TEXT[a][lang] for a in seq)
            abstract = (f"并仅在第一步强制队友{ACTION_TEXT[action][0]}（这是策略干预，不是玩家可执行操作）" if override else "，队友每步按冻结神经策略决策")
            abstract_en = (f"; force the teammate to {ACTION_TEXT[action][1]} on the first step (a policy intervention, not a player control)" if override else "; the teammate follows its frozen neural policy")
            add("assumption", "counterfactual_assumption", branch["assumptions"], f"隔离模拟假设：玩家接下来依次{sequence_text}{abstract}。最多模拟 {len(seq)} 步，终局会提前停止。", f"Isolated simulation assumes player actions {sequence_text}{abstract_en}. Simulate at most {len(seq)} steps, stopping at terminal state.", "counterfactual")
            end = branch["final_state"]
            ai_sequence = " → ".join(ACTION_TEXT[f["ai_action"]][lang] for f in branch["frames"]) or ("无" if lang == 0 else "none")
            add("outcome", "counterfactual_outcome", {"turn": end["turn"], "orders": end["orders"], "pot": end["pot"], "actors": end["actors"]}, f"在这个假设下，队友依次{ai_sequence}；模拟结束于第 {end['turn']} 步，出餐数为 {end['orders']}，锅剩余 {end['pot']['remaining']} 个烹饪步。这不是对玩家实际后续操作的预测。", f"Under this assumption the teammate actions are {ai_sequence}. The branch ends at step {end['turn']}, with {end['orders']} soups served and {end['pot']['remaining']} cooking steps remaining. This does not predict the player’s actual future inputs.", "counterfactual")
            mandatory += ["assumption", "outcome"]
        if intent["intent"] == "clarify":
            add("clarify", "clarification_required", True, "请明确要询问队友下一动作、交接规则，还是你执行某个动作后的三步以内结果。若问已执行动作，请选择该动作之前的一帧。", "Please specify the teammate’s next action, a handoff rule, or the outcome of a player action within three steps. For an executed action, select the frame immediately before it.", "binding")
            mandatory.append("clarify")
        if state["done"] and "actor" in mandatory: mandatory.remove("actor")
        if state["done"] and intent["intent"] not in {"rules", "failure"}:
            add("ended", "no_next_action", True, "这个回合已经结束，不会再执行下一动作。", "This round has ended and will execute no next action.", "binding")
            mandatory.append("ended")
        chosen = list(dict.fromkeys(mandatory))
        if intent["intent"] == "why": chosen += ["holding", "counters", "pot"]
        lookup = {f["fact_id"]: f for f in facts}
        # The model sees only the question and the minimum rendered evidence,
        # never participant/run IDs, raw snapshots or private item provenance.
        if self.client.configured and intent["intent"] != "clarify":
            payload = {"question": redact_question(question), "intent": intent, "language": language,
                       "mandatory": mandatory, "facts": [{"id": f["fact_id"], "text": f["text"]} for f in facts if f["fact_id"] in set(chosen + ["branch"])]}
            allowed = {f["id"] for f in payload["facts"]}
            for attempt in range(2):
                value, log = self.client.request(ANSWER_PROMPT, payload)
                diagnostics["calls"].append({"stage": "answer", "attempt": attempt + 1, **log})
                ids = value.get("fact_ids") if isinstance(value, dict) else None
                if (isinstance(value, dict) and not set(value) - {"fact_ids", "clarification"} and isinstance(ids, list)
                        and 1 <= len(ids) <= 10 and all(isinstance(x, str) and x in allowed for x in ids)
                        and set(mandatory) <= set(ids) and value.get("clarification") is False):
                    chosen = list(dict.fromkeys(ids)); diagnostics["llm_success"] = True; break
            if not diagnostics["llm_success"]: diagnostics["fallback"] = "verification_failed"
        else: diagnostics["fallback"] = "missing_cloud_configuration" if not self.client.configured else "clarification_required"
        if snapshot_hash(snapshot) != source: raise AssertionError("Question mutated live snapshot")
        text = "\n\n".join(lookup[id]["text"] for id in chosen)
        return {"title": "行为解释" if lang == 0 else "Behavior explanation", "text": text, "frame": turn,
                "kind": intent["intent"], "anchor": "next", "verified": True,
                "source_summary": "来源：所选状态、冻结神经策略与已核验模拟。" if lang == 0 else "Sources: selected state, frozen neural policy and verified simulation.",
                "evidence": {"facts": facts, "selected_fact_ids": chosen, "neural_distribution": distribution, "counterfactual": branch},
                "diagnostics": diagnostics}
