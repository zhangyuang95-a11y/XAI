"""Bounded frame-bound evidence, with no invented intention or action advice.

Counterfactual control remains the runtime's player intervention under the same
neural Actor. A test fixture can exercise answers only through an explicit test
flag; it can never set production explanation eligibility.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from types import SimpleNamespace
import numpy as np

from core.program import ExecutableProgram
from .environment import NativeWarehouseEnv
from .policy import ACTIONS

LABELS = {"zh": dict(zip(ACTIONS,("向上","向下","向左","向右","等待"))),
          "en": dict(zip(ACTIONS,("up","down","left","right","wait")))}
EXPLANATION_VERSION = "warehouse-native-evidence-answers.v3"
ACCEPTANCE_VERSION = "warehouse-native-explanation-acceptance.v2"
ANSWER_AUDIT_VERSION = "warehouse-native-behavioral-answers.v3"
REQUIRED_CATEGORIES = ("narrow_passage","shared_pickup","shared_charger")
SUPPORTED_INTENTS = ("reason","alternative","counterfactual","failure","rules","clarify")
ALIASES = {"UP":r"向上|往上|上移|\b(?:move\s+|go\s+)?up\b",
           "DOWN":r"向下|往下|下移|\b(?:move\s+|go\s+)?down\b",
           "LEFT":r"向左|往左|左移|\b(?:move\s+|go\s+)?left\b",
           "RIGHT":r"向右|往右|右移|\b(?:move\s+|go\s+)?right\b",
           "WAIT":r"等待|不动|停留|\bwait(?:ing|ed)?\b|\bstay\s+still\b"}
NUMBERS = dict(zip(("零","一","二","两","三","四","五","六","七","八","九","十"),(0,1,2,2,3,4,5,6,7,8,9,10)))
NUMBERS.update(dict(zip(("zero","one","two","three","four","five","six","seven","eight","nine","ten"),range(11))))
NUMBER = r"(?:-?\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten|[零一二两三四五六七八九十]+)"
UNIT = r"\s*(?:步|次|steps?\b|turns?\b|times?\b)"


def _number(value):
    if value in NUMBERS:
        return NUMBERS[value]
    try:
        return float(value)
    except ValueError:
        return 4  # Compound Chinese quantities exceed this bounded grammar.


def _clarify(reason):
    return {"intent":"clarify","reason":reason}


def parse_question(question, focus=None):
    if not isinstance(question,str) or not question.strip() or len(question)>1000:
        return _clarify("invalid_question")
    if focus not in (None,"executed","next"):
        return _clarify("invalid_focus")
    text = question.strip().casefold().replace("’", "'").replace("不动","等待")
    # "Just wait" commonly means "only wait", not a past action. Explicit
    # tense/last-step markers determine history; "just" alone never does.
    past = bool(re.search(r"刚才|上一步|此前那步|\bdid\b|\bhad\b|\blast (?:step|action|turn)\b",text))
    future = bool(re.search(r"下一|接下来|未来|\bnext\b|\bwill\b",text))
    moves = [action for action,pattern in ALIASES.items() if re.search(pattern,text)]
    hypothetical = bool(re.search(r"如果|假如|假设|\bwhat if\b|\bif i\b",text))
    if hypothetical:
        if not re.search(r"我|\bi\b",text):
            return _clarify("counterfactual_player_required")
        if len(moves)!=1 or re.search(r"还是|或者|\bor\b|\bdon't\b|\bdo not\b|如果我不|假如我不",text):
            return _clarify("one_player_action_required")
        if re.search(r"(?:队友|机器人|teammate|robot).{0,15}(?:向[上下左右]|wait|move)",text):
            return _clarify("teammate_action_intervention_not_supported")
        counts = [_number(m.group(1)) for m in re.finditer("(?<![第\\d])("+NUMBER+")"+UNIT,text)]
        if any(not np.isfinite(value) or value!=int(value) or not 1<=value<=3 for value in counts) or re.search(r"半步|\bhalf\b",text):
            return _clarify("at_most_three_steps")
        forecasts = [_number(m.group(1)) for m in re.finditer(r"(?:接下来|未来|后续|预测|之后|\bnext|\bfollowing|\bover the next)\s*(?:的\s*)?("+NUMBER+")"+UNIT,text)]
        if len(set(forecasts))>1:
            return _clarify("ambiguous_horizon")
        duration_match = re.search("(?:"+ALIASES[moves[0]]+r")\s*(?:移动\s*|for\s*)?("+NUMBER+")"+UNIT,text)
        duration = int(_number(duration_match.group(1))) if duration_match else None
        once_twice = re.search("(?:"+ALIASES[moves[0]]+r")\s+(once|twice)\b",text)
        if once_twice:
            duration = 1 if once_twice.group(1)=="once" else 2
        horizon = int(forecasts[0]) if forecasts else duration if duration is not None else max(map(int,counts),default=3)
        if duration is not None and duration>horizon:
            return _clarify("action_duration_exceeds_horizon")
        repeat = bool(re.search(r"连续|重复|每一步|每步|\brepeat\b|\beach (?:step|turn)\b|\bevery (?:step|turn)\b",text))
        if repeat and re.search(r"只.{0,3}第一步|\bonly (?:on )?the first\b",text):
            return _clarify("conflicting_repetition")
        repeated = duration if duration is not None else horizon if repeat or moves[0]=="WAIT" else 1
        return {"intent":"counterfactual","focus":"executed" if past else "next","steps":horizon,
                "player_actions":[moves[0]]*repeated,"unspecified_followups":"WAIT"}
    if re.search(r"如果|假如|\bif\b",text):
        return _clarify("unsupported_counterfactual")
    if re.search(r"失败|断电|回合结束|为什么结束|\bfail(?:ed|ure)?\b|shutdown|ended|\b(?:round|game) end\b|game over",text):
        return {"intent":"failure","focus":"next"}
    if re.search(r"规则|耗电|充多少|充电速度|rules|energy cost|charge rate",text):
        return {"intent":"rules","focus":"next"}
    if not re.search(r"为什么|为何|在等什么|\bwhy\b|waiting for",text):
        return _clarify("unsupported_intent")
    if not moves and not re.search(r"这样|动作|选择|在等什么|do that|did that|that action|this action|next action|choose|chosen|select|choice",text):
        return _clarify("unsupported_behavior_question")
    if re.search(r"最优|最好|攻略|路线|计划|想法|打算|\bbest\b|\boptimal\b|\bplan\b|\bintention\b",text):
        return _clarify("intentions_or_strategy_not_supported")
    if re.search(r"为什么我|为何我|\bwhy (?:did|do|am|was|will|can't|cannot) i\b",text):
        return _clarify("player_decision_reason_not_supported")
    if past and future:
        return _clarify("ambiguous_time_reference")
    when = "executed" if past else "next" if future else focus or "executed"
    alternative = bool(re.search(r"为什么.{0,8}(?:不|没)|为何.{0,8}(?:不|没)|怎么不|\bnot\b|\bdidn't\b|\bdon't\b|\bdoesn't\b|\bwasn't\b|\binstead\b",text))
    if len(moves)>1 or alternative and len(moves)!=1:
        return _clarify("one_alternative_action_required")
    return {"intent":"alternative" if alternative else "reason","focus":when,
            "alternative_action":moves[0] if alternative else None,
            "mentioned_action":moves[0] if moves else "WAIT" if "在等什么" in text else None}


def explanation_source_hashes():
    root = Path(__file__).resolve().parents[2]
    native = Path(__file__).resolve().parent
    paths = [native/name for name in ("explanation.py","runtime.py","environment.py","observations.py","policy.py")]
    paths += [root/"env/warehouse"/name for name in ("environment.py","domain.py","navigation.py","layouts.py","transition_outcome.py")]
    return {str(path.relative_to(root)):sha256(path.read_bytes()).hexdigest() for path in paths}


def _rate(value,minimum):
    return type(value) in (int,float) and np.isfinite(value) and minimum<=value<=1


def _count(value,minimum):
    return type(value) is int and type(minimum) is int and value>=minimum


def verify_historical_transition(record, runtime, actor_sha256):
    """Recompute one recorded physical transition in an isolated environment.

    This is evidence checking, never live play. A valid record adds exactly one
    audit environment step, including when the question concerns its AFTER
    frame. Bad identity/decision fields are rejected before that step; a bad
    physical result is rejected after the one attempted replay. Callers count
    actual runtime.step/env.step calls, not predicted successful answer counts.
    """
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

    try:
        env = NativeWarehouseEnv()
        env.restore(record["before"])
        if env.done or env.state.frame+1 != record["after"]["state"]["frame"]:
            raise ValueError("Not a complete nonterminal-to-next-frame transition")
        submitted = record["submitted_actions"]
        if (set(submitted) != set(env.agent_ids) or any(a not in ACTIONS for a in submitted.values())
                or record["participant_action"] != submitted["robot_1"]):
            raise ValueError("Recorded player action differs from its joint submission")
        _, expected = runtime.decision(env)
        observed = record["decision"]
        if expected["actor_sha256"] != actor_sha256 or set(observed) != set(expected):
            raise ValueError("Decision belongs to a different Actor or schema")
        for key in expected:
            if key == "probabilities":
                if set(observed[key]) != set(expected[key]):
                    raise ValueError("Missing neural role distribution")
                for role in expected[key]:
                    probabilities = np.asarray(observed[key][role], dtype=float)
                    if (probabilities.shape != (5,) or not np.isfinite(probabilities).all()
                            or not np.allclose(probabilities, expected[key][role], atol=1e-6, rtol=1e-5)):
                        raise ValueError("Recorded neural probabilities differ")
            elif canonical(observed[key]) != canonical(expected[key]):
                raise ValueError("Recorded neural decision differs")
        if submitted["robot_2"] != expected["policy_actions"]["robot_2"]:
            raise ValueError("Submitted teammate action differs from the Actor")
        replay = runtime.step(env, submitted["robot_1"])
        for key in ("before", "after", "submitted_actions", "executed_actions", "rewards", "done", "participant_action"):
            if canonical(record[key]) != canonical(replay[key]):
                raise ValueError("Recorded physical outcome differs from replay: "+key)
        return replay
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError("historical_neural_decision_cannot_be_verified") from error


class NativeExplainer:
    def __init__(self,program_path,acceptance_path,actor_sha256,*,allow_test_fixture=False):
        from .runtime import runtime_signature
        self.program_path,self.acceptance_path = Path(program_path),Path(acceptance_path)
        payload = json.loads(self.program_path.read_text())
        source = payload["program"] if payload.get("version")=="warehouse_native_rcpd_feedback_v1" else payload
        self.program = ExecutableProgram.from_dict(source)
        self.program_content_sha256 = sha256(json.dumps(self.program.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.acceptance = json.loads(self.acceptance_path.read_text())
        self.actor_sha256 = actor_sha256
        self.program_sha256 = sha256(self.program_path.read_bytes()).hexdigest()
        self.acceptance_sha256 = sha256(self.acceptance_path.read_bytes()).hexdigest()
        self.test_fixture = self.acceptance.get("test_fixture") is True
        self.sources = explanation_source_hashes()
        self.runtime_signature = runtime_signature(SimpleNamespace(artifact_sha256=actor_sha256))
        audit,checks = self.acceptance,{}
        self.bound_artifacts = {}
        def artifact(name):
            contract = audit.get(name,{})
            relative = Path(str(contract.get("path","")))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                checks[name+"_binding"] = False
                return {}
            path = self.acceptance_path.parent/relative
            valid = (path.resolve().is_relative_to(self.acceptance_path.parent.resolve())
                     and path.is_file() and sha256(path.read_bytes()).hexdigest()==contract.get("sha256"))
            checks[name+"_binding"] = valid
            if not valid:
                return {}
            self.bound_artifacts[path] = contract["sha256"]
            return json.loads(path.read_text())
        checks["acceptance_schema"] = audit.get("version")==ACCEPTANCE_VERSION
        checks["actor_binding"] = audit.get("actor_sha256")==actor_sha256 and self.program.metadata.get("native_source_actor_sha256")==actor_sha256
        checks["program_binding"] = audit.get("program_sha256")==self.program_sha256
        checks["runtime_binding"] = audit.get("runtime_signature")==self.runtime_signature
        checks["source_binding"] = audit.get("sources")==self.sources
        env = NativeWarehouseEnv()
        checks["feature_action_contract"] = set(self.program.feature_names)==set(env.feature_names) and tuple(self.program.action_names)==ACTIONS
        checks["no_action_masks"] = not self.program.metadata.get("action_legality_features") and not self.program.metadata.get("action_constraint_reason_features")
        stack,valid_tree = [self.program.root],self.program.root.depth()<=8 and self.program.root.leaf_count()<=64
        while stack:
            node = stack.pop()
            if node.is_leaf:
                p = np.asarray(node.probabilities)
                valid_tree &= p.shape==(5,) and np.isfinite(p).all() and np.all(p>=0) and np.isclose(p.sum(),1)
            else:
                valid_tree &= node.feature in env.feature_names and node.threshold is not None and np.isfinite(node.threshold)
                stack += [node.left,node.right]
        checks["bounded_valid_tree"] = bool(valid_tree)
        checks["independent_holdout"] = audit.get("independent_holdout") is True
        checks["explicit_acceptance"] = audit.get("eligible") is True
        checks["capability_passed"] = audit.get("foundation_capability_passed") is True
        holdout = audit.get("holdout",{})
        checks["holdout_provenance"] = (holdout.get("split")=="explanation_test" and holdout.get("exact_observation_overlap")==0 and holdout.get("scenario_overlap")==0
            and all(re.fullmatch(r"[0-9a-f]{64}",str(holdout.get(k,""))) for k in ("scenario_manifest_sha256","fit_observation_sha256","holdout_observation_sha256"))
            and holdout.get("fit_observation_sha256")!=holdout.get("holdout_observation_sha256"))
        metrics = audit.get("metrics",{})
        checks["overall_fidelity"] = _rate(metrics.get("action_fidelity"),.9) and _count(metrics.get("rows"),10)
        for category in REQUIRED_CATEGORIES:
            item = metrics.get("critical",{}).get(category,{})
            checks[category+"_fidelity_coverage"] = _rate(item.get("fidelity"),.85) and _count(item.get("scenarios"),10) and _count(item.get("rows"),item.get("scenarios",10))
        intervention = artifact("intervention_artifact")
        rows = intervention.get("rows",[])
        valid_rows = all(r.get("baseline_nn") in ACTIONS and r.get("intervention_nn") in ACTIONS and r.get("baseline_tree") in ACTIONS and r.get("intervention_tree") in ACTIONS
            and r.get("neural_action_changed") is (r.get("baseline_nn")!=r.get("intervention_nn"))
            and str(r.get("scenario_id","")).startswith("explanation_test_") for r in rows)
        effective = [r for r in rows if r.get("physical_effect") is True and r.get("baseline_nn")!=r.get("intervention_nn")]
        agrees = lambda r:r["baseline_nn"]==r["baseline_tree"] and r["intervention_nn"]==r["intervention_tree"]
        rate = float(np.mean([agrees(r) for r in effective])) if effective and valid_rows else 0.
        checks["effective_intervention_records"] = (valid_rows and intervention.get("no_effect_pairs_never_counted_as_success") is True
            and intervention.get("effective_changed_pairs")==len(effective) and len({r.get("scenario_id") for r in effective})>=10)
        checks["effective_intervention_fidelity"] = rate>=.85 and _rate(metrics.get("intervention_direction_fidelity"),.85) and np.isclose(metrics.get("intervention_direction_fidelity",0),rate)
        for category in REQUIRED_CATEGORIES:
            subset = [r for r in effective if category in r.get("groups",[])]
            checks[category+"_effective_interventions"] = bool(valid_rows and len({r.get("scenario_id") for r in subset})>=10 and subset and np.mean([agrees(r) for r in subset])>=.85)
        answers = artifact("behavioral_answer_artifact")
        cases = answers.get("cases",[])
        coverage = {(c.get("language"),c.get("intent")) for c in cases if c.get("passed") is True and c.get("frame_binding_passed") is True and c.get("evidence_verified") is True
                    and isinstance(c.get("question"),str) and c["question"].strip() and isinstance(c.get("answer"),str) and c["answer"].strip()}
        checks["behavioral_answer_audit"] = (audit.get("behavioral_answer_audit_passed") is True and answers.get("version")==ANSWER_AUDIT_VERSION and answers.get("passed") is True
            and answers.get("actor_sha256")==actor_sha256 and answers.get("program_sha256")==self.program_sha256 and answers.get("runtime_signature")==self.runtime_signature
            and answers.get("renderer_version")==EXPLANATION_VERSION and {(lang,intent) for lang in ("zh","en") for intent in SUPPORTED_INTENTS}<=coverage
            and all(c.get("passed") is True for c in cases))
        checks["not_test_fixture"] = not self.test_fixture and answers.get("test_fixture") is not True
        self.checks = {key:bool(value) for key,value in checks.items()}
        self.ineligibility_reasons = [key for key,value in self.checks.items() if not value]
        self.eligible = all(self.checks.values())
        fixture_exemptions = {"not_test_fixture","explicit_acceptance","capability_passed"}
        self.fixture_enabled = bool(allow_test_fixture and self.test_fixture and all(v for k,v in self.checks.items() if k not in fixture_exemptions))
        self.signature = sha256(json.dumps({"actor":actor_sha256,"program":self.program_sha256,"acceptance":self.acceptance_sha256,"sources":self.sources,"version":EXPLANATION_VERSION},sort_keys=True).encode()).hexdigest()

    def _assert_current(self,runtime):
        if not self.eligible and not self.fixture_enabled:
            raise ValueError("explanation_not_validated")
        if runtime.actor.artifact_sha256!=self.actor_sha256:
            raise ValueError("explanation_actor_version_mismatch")
        if runtime.signature!=self.runtime_signature or explanation_source_hashes()!=self.sources:
            raise ValueError("explanation_runtime_version_mismatch")
        if self.fixture_enabled!=(runtime.actor.metadata.get("test_fixture") is True):
            raise ValueError("explanation_fixture_scope_mismatch")
        if sha256(self.program_path.read_bytes()).hexdigest()!=self.program_sha256 or sha256(self.acceptance_path.read_bytes()).hexdigest()!=self.acceptance_sha256 or any(sha256(path.read_bytes()).hexdigest()!=expected for path,expected in self.bound_artifacts.items()):
            raise ValueError("explanation_artifact_changed")
        if sha256(json.dumps(self.program.to_dict(),sort_keys=True,allow_nan=False).encode()).hexdigest()!=self.program_content_sha256:
            raise ValueError("explanation_program_memory_changed")

    def answer(self, request, frame_record, runtime):
        self._assert_current(runtime)
        language = "en" if request.get("language") == "en" else "zh"
        parsed = parse_question(request.get("question"), request.get("focus"))
        current = frame_record["after"]
        selected_frame = current["state"]["frame"]
        if "frame" in request and (type(request["frame"]) is not int or request["frame"]!=selected_frame):
            raise ValueError("explanation_frame_mismatch")
        mentions = re.findall(r"第\s*(\d+)\s*(?:帧|步)|\bframe\s+(\d+)\b|\bat step\s+(\d+)\b",str(request.get("question","")).casefold())
        if any(int(next(value for value in match if value))!=selected_frame for match in mentions):
            parsed = _clarify("select_requested_frame")
        if parsed["intent"] == "clarify":
            return clarify_answer(parsed["reason"],selected_frame,language)
        focus = parsed.get("focus","executed")
        use_before = focus=="executed"
        if use_before and "before" not in frame_record:
            return clarify_answer("no_executed_action_at_initial_frame",selected_frame,language)
        if "before" in frame_record:
            if frame_record["before"]["state"]["frame"]+1!=selected_frame:
                raise ValueError("historical_frame_sequence_mismatch")
            frame_record = verify_historical_transition(frame_record,runtime,self.actor_sha256)
            current = frame_record["after"]
        bound = frame_record["before"] if use_before else current
        if parsed["intent"] == "counterfactual":
            branch = runtime.counterfactual(bound, parsed["player_actions"], steps=parsed["steps"])
            start_frame = bound["state"]["frame"]
            assumptions = "、".join(LABELS[language][a] for a in branch["assumed_player_actions"])
            outcomes = []
            for transition in branch["transitions"]:
                before, after = transition["before"]["state"], transition["after"]["state"]
                action = transition["submitted_actions"]["robot_2"]
                actual = transition["executed_actions"]["robot_2"]
                if language == "zh":
                    outcomes.append(f"第 {after['frame']} 步队友选择{LABELS[language][action]}，实际执行{LABELS[language][actual]}，团队新增配送 {after['total_deliveries']-before['total_deliveries']} 件")
                else:
                    outcomes.append(f"step {after['frame']}: teammate chose {LABELS[language][action]}, physically executed {LABELS[language][actual]}, and team deliveries increased by {after['total_deliveries']-before['total_deliveries']}")
                if after.get("terminated") or after.get("truncated"):
                    outcomes[-1] += "，回合已结束" if language=="zh" else ", and the round ended"
            if not outcomes:
                return "该帧的回合已经结束，不能继续推进。" if language == "zh" else "The round has already ended at this frame."
            return (f"从第 {start_frame} 帧开始，假设你的动作依次为：{assumptions}；未指定的后续动作按等待处理。" + "；".join(outcomes) + "。这是同一个 NN 在隔离副本中的结果；真实回合未改变。来源：冻结 NN 与环境模拟。"
                    if language == "zh" else f"Starting at frame {start_frame}, assume your actions are: {assumptions}; unspecified following actions are WAIT. " + "; ".join(outcomes) + ". These results use the same NN in an isolated copy; the live round is unchanged. Sources: frozen NN and environment simulation.")
        env = NativeWarehouseEnv()
        env.restore(bound)
        if parsed["intent"] == "rules":
            return (f"成功移动一格时，电量按 {env.config.move_battery_cost:g} 个百分点扣减，最低为 0%；剩余电量不足该数值也可以尝试移动。在充电格以外耗尽电量会断电并结束回合；恰好以 0% 到达充电格不会断电，之后实际停留一回合最多补充 {env.config.charge_per_wait:g} 个百分点，上限为 100%。撞墙或机器人冲突会阻止移动，仍消耗一个回合；这些结果不会被改选成另一个方向。来源：公开环境规则。"
                    if language == "zh" else f"A successful one-cell move reduces battery by {env.config.move_battery_cost:g} percentage points, with a floor of 0%; movement can still be attempted with less charge. Exhausting battery away from the charger causes shutdown and ends the round. Arriving at the charger with exactly 0% does not cause shutdown; subsequently staying there restores up to {env.config.charge_per_wait:g} percentage points per turn, capped at 100%. Walls and robot conflicts prevent movement but still consume a turn; no alternative direction is substituted. Source: public environment rules.")
        if parsed["intent"] == "failure":
            state = current["state"]
            reason = state.get("terminal_reason")
            text = {"battery_shutdown": ("至少一台机器人在充电格以外耗尽了电量", "at least one robot exhausted its battery away from the charger"),
                    "horizon": ("已达到回合步数上限", "the turn limit was reached")}
            if reason not in text:
                return "这一帧没有记录环境终局失败。" if language == "zh" else "No environmental terminal failure is recorded at this frame."
            return (f"第 {selected_frame} 帧结束的直接原因是{text[reason][0]}。这说明终局触发条件，不等于证明此前某个动作是唯一原因。来源：该帧终局记录。"
                    if language == "zh" else f"At frame {selected_frame}, the round ended because {text[reason][1]}. This identifies the terminal trigger, not a proven sole cause among earlier actions. Source: the frame's terminal record.")
        if not use_before and env.done:
            return f"第 {selected_frame} 帧回合已结束，没有下一次神经决策。" if language=="zh" else f"The round ended at frame {selected_frame}; no next neural decision exists."
        obs = env.observations()["robot_2"]
        logits = runtime.actor.logits(obs[None, :])[0]
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        neural_action = ACTIONS[int(logits.argmax())]
        features = dict(zip(env.feature_names, map(float, obs)))
        tree_action = self.program.predict(features)
        if use_before:
            recorded = frame_record.get("decision",{})
            recorded_probs = np.asarray(recorded.get("probabilities",{}).get("robot_2",[]))
            if (recorded.get("actor_sha256")!=self.actor_sha256 or recorded.get("frame")!=bound["state"]["frame"]
                    or recorded.get("policy_actions",{}).get("robot_2")!=neural_action
                    or recorded.get("observation_hashes",{}).get("robot_2")!=sha256(obs.tobytes()).hexdigest()
                    or recorded_probs.shape!=(5,) or not np.isfinite(recorded_probs).all()
                    or not np.allclose(recorded_probs,probabilities,atol=1e-6,rtol=1e-5)
                    or frame_record.get("submitted_actions",{}).get("robot_2")!=neural_action):
                raise ValueError("historical_neural_decision_cannot_be_verified")
        when = f"第 {selected_frame} 步执行前" if use_before else f"第 {selected_frame} 帧的下一次决策"
        when_en = f"before executing step {selected_frame}" if use_before else f"for the next decision at frame {selected_frame}"
        alternative = parsed.get("alternative_action")
        mentioned = parsed.get("mentioned_action")
        if alternative==neural_action or (not alternative and mentioned and mentioned!=neural_action):
            if use_before and frame_record.get("executed_actions",{}).get("robot_2")=="WAIT" and neural_action!="WAIT":
                return physical_wait_answer(when,when_en,neural_action,frame_record,env,language)
            return (f"{when}，神经策略实际选择的是{LABELS[language][neural_action]}。请确认所选帧和问题中的动作。"
                    if language=="zh" else f"{when_en.capitalize()}, the neural policy actually chose {LABELS[language][neural_action]}. Please confirm the selected frame and action in your question.")
        comparison = alternative_answer(alternative,neural_action,probabilities,logits,env,language) if alternative else ""
        if tree_action != neural_action:
            return (f"{when}，NN 将{LABELS[language][neural_action]}排在首位。"+comparison+"近似决策树选择了不同动作，因此不能用这个树分支解释本次选择。来源：实际 NN 输出与树的一致性核验。"
                    if language == "zh" else f"{when_en.capitalize()}, the NN ranked {LABELS[language][neural_action]} first. "+comparison+"The approximate tree selected a different action, so its branch cannot explain this choice. Sources: actual NN output and tree-agreement check.")
        trace = self.program.trace(features)
        conditions = [describe_predicate(t, env, language) for t in trace]
        conditions = [c for c in conditions if c]
        condition_text = "；".join(conditions[:3]) if language == "zh" else "; ".join(conditions[:3])
        if language == "zh":
            prefix = f"{when}，NN 将{LABELS[language][neural_action]}排在首位，近似树的动作与它一致。"
            details = f"该树分支检查的部分条件为：{condition_text}。" if condition_text else "该分支的条件暂未完成文字化呈现。"
            return prefix + comparison + details + "这些条件是策略近似证据，不能证明 NN 的内部动机或唯一原因，也不代表队友承诺了后续路线。来源：实际 NN 输出与抽取的近似树。"
        prefix = f"{when_en.capitalize()}, the NN ranked {LABELS[language][neural_action]} first, matching the approximate tree. "
        details = f"Some conditions checked on this branch were: {condition_text}. " if condition_text else "This branch's conditions do not yet have a text rendering. "
        return prefix + comparison + details + "These are approximate policy evidence, not proof of the NN's internal motive or sole cause, or a commitment to a future route. Sources: actual NN output and extracted approximate tree."


def clarify_answer(reason,frame,language):
    if reason=="select_requested_frame":
        return f"当前选中第 {frame} 帧，请先选中问题提到的历史帧。" if language=="zh" else f"Frame {frame} is selected. Please first select the historical frame named in your question."
    if reason=="no_executed_action_at_initial_frame":
        return f"第 {frame} 帧尚无已执行动作，请改问下一次决策或选中后续历史帧。" if language=="zh" else f"Frame {frame} has no executed action. Ask about the next decision or select a later historical frame."
    if reason=="at_most_three_steps":
        return "这里只支持一至三步预测，请缩短预测范围。" if language=="zh" else "Predictions support one to three steps only. Please shorten the horizon."
    return ("请明确所选帧队友的已执行或下一动作；问‘为什么不’时请指定一个替代动作。反事实只支持你的一种动作，最多三步，未指定的后续动作按等待处理。不推测队友动机或提供攻略。"
            if language=="zh" else "Please specify the teammate's executed or next action. A why-not question needs one alternative. Counterfactuals support one player action over at most three steps; unspecified following actions are WAIT. Intentions and strategy advice are outside scope.")


def physical_wait_answer(when,when_en,action,record,env,language):
    from env.warehouse.navigation import MOVE_DELTAS
    r,c = env.state.by_id("robot_2").position
    dr,dc = MOVE_DELTAS[action]
    wall = not env.layout.is_passable((r+dr,c+dc))
    collision = record["after"]["state"]["robot_collision_events"]>record["before"]["state"]["robot_collision_events"]
    reason = ("目标格为墙或边界" if wall else "发生机器人冲突" if collision else "移动被环境阻止") if language=="zh" else ("the target was a wall or boundary" if wall else "a robot conflict occurred" if collision else "the environment prevented movement")
    return (f"{when}，NN 选择{LABELS[language][action]}，但实际执行等待：{reason}。不能把物理停留解释成 NN 主动等待。来源：神经记录与物理结果。"
            if language=="zh" else f"{when_en.capitalize()}, the NN chose {LABELS[language][action]}, but physically waited because {reason}. Physical waiting is not evidence that the NN chose WAIT. Sources: neural record and physical outcome.")


def alternative_answer(alternative,chosen,probabilities,logits,env,language):
    from env.warehouse.navigation import MOVE_DELTAS
    alt,index = ACTIONS.index(alternative),ACTIONS.index(chosen)
    tie = logits[alt]==logits[index]
    if language=="zh":
        text = f"{LABELS[language][alternative]}的 NN 概率为 {100*probabilities[alt]:.2f}%，所选{LABELS[language][chosen]}为 {100*probabilities[index]:.2f}%。"
        text += "两者并列最高，确定性 argmax 按固定动作顺序取首项。" if tie else "确定性策略选择概率最高的动作。"
    else:
        text = f"NN probability for {LABELS[language][alternative]} was {100*probabilities[alt]:.2f}%, versus {100*probabilities[index]:.2f}% for chosen {LABELS[language][chosen]}. "
        text += "They tied for the maximum; deterministic argmax uses the fixed action order. " if tie else "The deterministic policy chose the highest-probability action. "
    if alternative=="WAIT":
        return text+("等待没有被动作掩码排除。" if language=="zh" else "WAIT was not removed by an action mask. ")
    r,c = env.state.by_id("robot_2").position
    dr,dc = MOVE_DELTAS[alternative]
    passable = env.layout.is_passable((r+dr,c+dc))
    if language=="zh":
        return text+("该方向邻格可通行，但是否移动成功还取决于双方冲突。" if passable else "该方向邻格是墙或边界，但动作没有被掩码排除。")+"这是环境事实，不证明 NN 正是因此没有选择它。"
    return text+("The adjacent cell was traversable; physical conflicts can still prevent movement. " if passable else "The adjacent cell was a wall or boundary, but the action was not masked. ")+"This environmental fact does not establish why the NN ranked the action lower. "


def describe_predicate(step, env, language):
    """Translate only known physical scalar predicates; unknowns stay omitted."""
    feature = step.feature
    scale = 1.
    names = {"self.battery": ("队友电量", "teammate battery", 100),
             "other.battery": ("玩家电量", "player battery", 100),
             "other.path_distance": ("双方通路距离", "path distance between robots", env.config.rows*env.config.cols-1),
             "charger.self.path_distance": ("队友到充电站的通路距离", "teammate path distance to charger", env.config.rows*env.config.cols-1),
             "charger.other.path_distance": ("玩家到充电站的通路距离", "player path distance to charger", env.config.rows*env.config.cols-1),
             "time.remaining": ("剩余回合数", "remaining turns", env.config.horizon)}
    if feature not in names:
        return None
    zh, en, scale = names[feature]
    relation = "≤" if step.result else ">"
    return f"{zh if language=='zh' else en} {relation} {step.threshold*scale:.2f}（{step.observed_value*scale:.2f}）"
