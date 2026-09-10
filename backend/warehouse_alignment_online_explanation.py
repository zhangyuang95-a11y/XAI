"""Dependency-light, frame-bound explanations for the online alignment runtime.

Only frozen NumPy policy evidence, an executable JSON program, and isolated
physics are used.  This module imports no training or model-fitting package.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import numpy as np

from backend.warehouse_alignment_online_runtime import (
    ACTIONS, OnlineAlignmentRuntime, digest, runtime_sources,
)
from core.program import ExecutableProgram

VERSION = "warehouse-alignment-online-readable-answers.v2"

def explanation_sources():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root/"backend/warehouse_alignment_online_runtime.py", root/"core/program.py"]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in paths}

LABELS = {"zh": dict(zip(ACTIONS,("向上","向下","向左","向右","等待"))),
          "en": dict(zip(ACTIONS,("up","down","left","right","wait")))}
EXPLANATION_VERSION = "warehouse-native-readable-evidence-answers.v4"
ACCEPTANCE_VERSION = "warehouse-native-explanation-acceptance.v2"
ANSWER_AUDIT_VERSION = "warehouse-native-behavioral-answers.v3"
REQUIRED_CATEGORIES = ("narrow_passage","shared_pickup","shared_charger")
SUPPORTED_INTENTS = (
    "reason", "alternative", "counterfactual", "failure", "rules",
    "collision", "influence", "goal", "energy", "clarify",
)
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
    if re.search(r"碰撞|相撞|撞到|冲突|collid|collision|conflict", text):
        return {"intent":"collision","focus":"executed"}
    if re.search(r"(?:我的|玩家).{0,12}(?:影响|改变)|(?:影响|改变).{0,12}(?:队友|机器人)|\b(?:my|player) action.{0,16}(?:affect|change|influence)|\b(?:affect|change|influence).{0,16}(?:teammate|robot)", text):
        return {"intent":"influence","focus":"executed" if not future else "next"}
    if re.search(r"(?:当前|现在).{0,8}(?:目标|任务|方向)|(?:目标|任务).{0,8}(?:是什么|哪个|哪里)|trying to do|current (?:goal|task|objective)|heading (?:for|toward)", text):
        return {"intent":"goal","focus":"next"}
    if re.search(r"需要.{0,4}充电|该.{0,4}充电|电量够|充电吗|need(?:s|ed)? to charge|need charging|enough (?:battery|charge)", text):
        return {"intent":"energy","focus":"next"}
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



def clarify_answer(reason,frame,language):
    if reason=="select_requested_frame":
        return f"当前选中第 {frame} 帧，请先选中问题提到的历史帧。" if language=="zh" else f"Frame {frame} is selected. Please first select the historical frame named in your question."
    if reason=="no_executed_action_at_initial_frame":
        return f"第 {frame} 帧尚无已执行动作，请改问下一次决策或选中后续历史帧。" if language=="zh" else f"Frame {frame} has no executed action. Ask about the next decision or select a later historical frame."
    if reason=="at_most_three_steps":
        return "这里只支持一至三步预测，请缩短预测范围。" if language=="zh" else "Predictions support one to three steps only. Please shorten the horizon."
    return ("请明确所选帧队友的已执行或下一动作；问‘为什么不’时请指定一个替代动作。反事实只支持你的一种动作，最多三步，未指定的后续动作按等待处理；不能据此推测队友动机或提供攻略。"
            if language=="zh" else "Please specify the teammate's executed or next action; a why-not question needs one alternative. Counterfactuals support one player action over at most three steps, with unspecified following actions treated as waits; they do not establish hidden intentions or provide strategy advice.")


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


def describe_physical_predicate(step, env, language):
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


def verify_historical_transition(record, runtime, actor_sha256):
    """Verify pre-step identity then replay exactly one step in an isolated197 env."""
    try:
        if record.get('runtime_signature') != runtime.signature:
            raise ValueError('Historical runtime signature differs')
        env = runtime.from_snapshot(record['before'])
        runtime.from_snapshot(record['after'])
        if env.done or env.state.frame+1 != record['after']['state']['frame']:
            raise ValueError('Historical frame sequence differs')
        submitted = record['submitted_actions']
        if (set(submitted) != set(env.agent_ids) or any(a not in ACTIONS for a in submitted.values())
                or record['participant_action'] != submitted['robot_1']):
            raise ValueError('Historical player command differs')
        _, expected = runtime.decision(env)
        observed = record['decision']
        if expected['actor_sha256'] != actor_sha256 or set(observed) != set(expected):
            raise ValueError('Historical neural source differs')
        for key, value in expected.items():
            if key == 'probabilities':
                if set(observed[key]) != set(value): raise ValueError('Incomplete neural probabilities')
                for role in value:
                    p = np.asarray(observed[key][role], dtype=float)
                    if p.shape != (5,) or not np.isfinite(p).all() or not np.allclose(p, value[role], atol=1e-6, rtol=1e-5):
                        raise ValueError('Historical neural distribution differs')
            elif digest(observed[key]) != digest(value): raise ValueError('Historical decision differs: '+key)
        if submitted['robot_2'] != expected['policy_actions']['robot_2']:
            raise ValueError('Historical NN command was overwritten')
        replay = runtime.step(env, submitted['robot_1'])
        # Actual producer fields, including confirmed history, RNG and events.
        if set(record) != set(replay): raise ValueError('Historical transition schema differs')
        for key in replay:
            if key != 'decision' and digest(record[key]) != digest(replay[key]):
                raise ValueError('Historical physical outcome differs: '+key)
        return replay
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError('historical_neural_decision_cannot_be_verified') from error


def count_boundary(threshold, horizon):
    """Exact integer cutoff for float32 log1p history codes and float64 tree threshold."""
    if type(horizon) is not int or horizon < 1 or not np.isfinite(threshold):
        raise ValueError('Invalid count encoding boundary')
    grid=np.asarray([math.log1p(n)/math.log1p(horizon) for n in range(horizon+1)],np.float32)
    allowed=np.flatnonzero(grid.astype(np.float64) <= float(threshold))
    return int(allowed[-1]) if len(allowed) else -1


def describe_predicate(step, env, language):
    """Named history predicates refer only to the immediately confirmed past."""
    feature = step.feature
    if not feature.startswith('history.'):
        return describe_physical_predicate(step, env, language)
    names = env.feature_names
    if feature not in names or step.operator != '<=' or not np.isfinite(step.threshold):
        raise ValueError('Unknown or invalid history predicate')
    actual = float(env.observations()['robot_2'][names.index(feature)])
    if not np.isclose(actual, step.observed_value, atol=1e-7, rtol=0) or bool(actual <= step.threshold) != step.result:
        raise ValueError('History predicate differs from the actual197 observation')
    history = env.public_history(); zh = language == 'zh'
    relation = '≤' if step.result else '>'
    if feature == 'history.valid':
        status = ('已知' if history['valid'] else '未知') if zh else ('known' if history['valid'] else 'unknown')
        return (f'上一已确认回合历史为{status}，已知指示值 {relation} {step.threshold:.6g}（实际 {actual:g}）' if zh
                else f'the preceding confirmed transition is {status}; history-valid indicator {relation} {step.threshold:.6g} (actual {actual:g})')
    if not history['valid']:
        return ('上一回合历史未知；此历史特征的零编码不能证明此前没有该动作或冲突' if zh
                else 'previous-transition history is unknown; its zero encoding does not prove the absence of an action or conflict')
    counts = {'history.self.consecutive_move_canceled': ('队友连续移动被取消次数','teammate consecutive canceled moves',history['consecutive_move_canceled']['robot_2']),
        'history.other.consecutive_move_canceled': ('玩家连续移动被取消次数','player consecutive canceled moves',history['consecutive_move_canceled']['robot_1']),
        'history.joint.consecutive_collision': ('双方连续冲突次数','consecutive joint conflicts',history['consecutive_collision'])}
    if feature in counts:
        cn,en,count = counts[feature]; horizon=env.config.horizon
        encoded = float(np.float32(math.log1p(count)/math.log1p(horizon)))
        decoded = math.expm1(actual*math.log1p(horizon))
        if actual != encoded or not math.isclose(decoded,count,abs_tol=3e-5):
            raise ValueError('Consecutive count and real log1p encoding differ')
        # Convert the threshold with the actual float32 log1p grid, avoiding a
        # rounded 1.999999 threshold being falsely rendered as count<=2.
        boundary=count_boundary(step.threshold,horizon)
        return (f'{cn} {relation} {boundary}（实际 {count} 次；按 log1p 计数编码核验）' if zh
                else f'{en} {relation} {boundary} (actual {count}; checked against log1p count encoding)')
    parts=feature.split('.')
    if len(parts)==4 and parts[1] in ('self','other') and parts[2]=='submitted' and parts[3] in ACTIONS:
        who=('队友' if parts[1]=='self' else '玩家') if zh else ('teammate' if parts[1]=='self' else 'player')
        name=(f'{who}上一回合提交{LABELS[language][parts[3]]}的指示值' if zh
              else f'indicator that the {who} submitted {LABELS[language][parts[3]]} on the preceding turn')
    elif feature in ('history.self.move_canceled','history.other.move_canceled'):
        who=('队友' if parts[1]=='self' else '玩家') if zh else ('teammate' if parts[1]=='self' else 'player')
        name=f'{who}上一回合移动被取消的指示值' if zh else f'indicator of the {who} preceding move being canceled'
    elif len(parts)==3 and parts[1]=='collision':
        kinds={'none':('无机器人冲突','no robot conflict'),'same_target':('争用同一目标格','same-target conflict'),
            'swap':('交换位置冲突','swap conflict'),'occupied_stationary':('目标格被停留者占用','stationary-occupant conflict')}
        if parts[2] not in kinds: raise ValueError('Unknown history collision type')
        label=kinds[parts[2]][0 if zh else 1]
        name=f'上一回合{label}的指示值' if zh else f'indicator of {label} on the preceding turn'
    else: raise ValueError('History predicate has no explicit translation')
    return f'{name} {relation} {step.threshold:.6g}（实际 {actual:g}）' if zh else f'{name} {relation} {step.threshold:.6g} (actual {actual:g})'


_MAIN_ANSWER_FORBIDDEN = re.compile(
    r"\b(?:NN|argmax|SHA(?:-?256)?)\b|神经(?:网络|策略|输出)|动作概率|决策树|"
    r"树分支|阈值|指示值|特征编码|\b(?:action |policy )?probabilit(?:y|ies)\b|"
    r"\bdecision tree\b|\btree branch\b|\bthreshold\b|\bfeature encod\w*\b|"
    r"\bhash(?:es)?\b",
    re.IGNORECASE,
)


def _response(answer, evidence_detail):
    """Keep participant prose separate from audit-oriented evidence."""
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Explanation answer is empty")
    if _MAIN_ANSWER_FORBIDDEN.search(answer):
        raise ValueError("Participant answer contains audit-only terminology")
    if not isinstance(evidence_detail, str) or not evidence_detail.strip():
        raise ValueError("Explanation evidence detail is empty")
    return {"answer": answer.strip(), "evidence_detail": evidence_detail.strip()}


def _agent(env, agent_id="robot_2"):
    return env.state.by_id(agent_id)


def _ordered_tasks(env):
    return sorted(env.state.tasks, key=lambda item: int(item.task_id.rsplit("_", 1)[-1]))


def _task_slot(env, task_id):
    return next((index for index, task in enumerate(_ordered_tasks(env), 1)
                 if task.task_id == task_id), None)


def _objective_candidates(env):
    """Return only objectives established by the public task state.

    Carrying establishes a unique delivery objective.  An empty robot has a
    set of available pickup candidates; it does not have a hidden commitment.
    """
    teammate = _agent(env)
    tasks = _ordered_tasks(env)
    if teammate.carrying_task_id:
        task = next((item for item in tasks if item.task_id == teammate.carrying_task_id), None)
        return [] if task is None else [{"kind": "delivery", "task": task,
            "slot": _task_slot(env, task.task_id), "target": tuple(task.delivery_position)}]
    return [{"kind": "pickup", "task": task, "slot": index,
             "target": tuple(task.pickup_position)}
            for index, task in enumerate(tasks, 1) if task.status == "available"]


def _objective_label(objective, language):
    slot = objective.get("slot")
    point = "B" if objective.get("kind") == "delivery" else "A"
    if language == "zh":
        return f"任务{slot}的{point}点" if slot is not None else f"当前{point}点"
    return f"task {slot}'s {point} point" if slot is not None else f"the current {point} point"


def _distance(env, start, target):
    from env.warehouse.navigation import shortest_path_distance
    return int(shortest_path_distance(tuple(start), tuple(target), env.config.map_layout_id))


def _position_after_action(env, action):
    from env.warehouse.navigation import MOVE_DELTAS
    position = tuple(_agent(env).position)
    if action == "WAIT":
        return position
    dr, dc = MOVE_DELTAS[action]
    target = (position[0] + dr, position[1] + dc)
    return target if env.layout.is_passable(target) else position


def _best_progress(env, destination):
    origin = tuple(_agent(env).position)
    candidates = []
    for objective in _objective_candidates(env):
        before = _distance(env, origin, objective["target"])
        after = _distance(env, destination, objective["target"])
        candidates.append({**objective, "before": before, "after": after,
                           "improvement": before - after})
    improving = [item for item in candidates if item["improvement"] > 0]
    if not improving:
        return None
    return min(improving, key=lambda item: (-item["improvement"], item["before"],
                                             item.get("slot") or 999))


def _trace_supports_objective(trace, objective):
    if not objective:
        return False
    slot = objective.get("slot")
    if slot is None:
        return False
    index = slot - 1
    kind = objective.get("kind")
    prefix = f"task.{index}.{kind}.self."
    return any(getattr(step, "feature", "").startswith(prefix) for step in trace)


def _objective_feature_intervention(env, runtime, obs, objective, neural_action,
                                    probabilities):
    """Probe sensitivity to one target relation without touching the live state.

    This is deliberately an abstract observation intervention, not a claim
    about a physically reachable player action.  It can support a concise
    directional description only when removing the target relation materially
    lowers the chosen action's support.
    """
    if not objective or objective.get("slot") is None:
        return {"supported": False}
    prefix = f"task.{objective['slot'] - 1}.{objective['kind']}.self."
    modified = np.asarray(obs, dtype=np.float32).copy()
    changed = []
    for index, feature in enumerate(env.feature_names):
        if not feature.startswith(prefix):
            continue
        changed.append(feature)
        modified[index] = 0.0 if feature.endswith(("relative_row", "relative_column")) else 1.0
    if not changed:
        return {"supported": False}
    before = sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()
    logits = runtime.actor.logits(modified[None])[0].astype(float)
    altered = np.exp(logits - logits.max())
    altered /= altered.sum()
    chosen_index = ACTIONS.index(neural_action)
    altered_action = ACTIONS[int(np.argmax(altered))]
    drop = float(probabilities[chosen_index] - altered[chosen_index])
    return {
        "supported": bool(altered_action != neural_action or drop >= 0.05),
        "kind": "abstract_target_relation_intervention",
        "changed_feature_count": len(changed),
        "original_observation_sha256": before,
        "altered_action": altered_action,
        "chosen_probability_drop": drop,
    }


def _probability_line(probabilities, language):
    pairs = [f"{LABELS[language][action]} {100 * float(probabilities[index]):.2f}%"
             for index, action in enumerate(ACTIONS)]
    return "、".join(pairs) if language == "zh" else ", ".join(pairs)


def _technical_detail(*, language, frame, focus, neural_action=None,
                      probabilities=None, tree_action=None, tree_matches=None,
                      conditions=(), actor_sha256=None, replay_verified=False,
                      counterfactual=False, extra=()):
    zh = language == "zh"
    lines = [f"绑定帧：{frame}（{'已执行动作' if focus == 'executed' else '下一次决策'}）"
             if zh else f"Bound frame: {frame} ({'executed action' if focus == 'executed' else 'next decision'})"]
    if neural_action in ACTIONS:
        lines.append((f"策略提交动作：{LABELS[language][neural_action]}"
                      if zh else f"Policy-submitted action: {LABELS[language][neural_action]}"))
    if probabilities is not None:
        lines.append(("动作概率：" if zh else "Action probabilities: ")
                     + _probability_line(probabilities, language))
    if tree_action in ACTIONS:
        agreement = "一致" if tree_matches else "不一致"
        agreement_en = "agrees" if tree_matches else "disagrees"
        lines.append((f"策略近似程序：{LABELS[language][tree_action]}（与提交动作{agreement}）"
                      if zh else f"Policy approximation: {LABELS[language][tree_action]} ({agreement_en} with the submitted action)"))
    if conditions and tree_matches:
        separator = "；" if zh else "; "
        lines.append(("近似分支条件：" if zh else "Approximate branch conditions: ")
                     + separator.join(tuple(conditions)[:3]))
    elif tree_action in ACTIONS and not tree_matches:
        lines.append("近似分支因动作不一致而未用于主回答。" if zh
                     else "The approximate branch was excluded from the main answer because its action disagreed.")
    if replay_verified:
        lines.append("历史动作、策略提交和物理结果已通过隔离重放核验。" if zh
                     else "Historical action, policy submission, and physical outcome passed isolated replay verification.")
    if counterfactual:
        lines.append("反事实使用同一冻结策略的隔离副本，未改变真实回合。" if zh
                     else "The counterfactual used an isolated copy of the same frozen policy and did not change the live run.")
    lines.extend(str(item) for item in extra if item)
    if actor_sha256:
        lines.append((f"Actor SHA-256：{actor_sha256}" if zh else f"Actor SHA-256: {actor_sha256}"))
    return "\n".join(lines)


class OnlineAlignmentExplainer:
    """Evidence renderer only. A release verifier must separately authorize use."""
    def __init__(self, program_path, *, expected_program_sha256, runtime, allow_test_fixture=False):
        if type(runtime) is not OnlineAlignmentRuntime or type(allow_test_fixture) is not bool or runtime.test_fixture is not allow_test_fixture:
            raise ValueError('Explicit matching observed197 runtime scope is required')
        runtime.verify_binding()
        self.program_path=Path(program_path).expanduser().resolve()
        raw=self.program_path.read_bytes()
        if not isinstance(expected_program_sha256,str) or not re.fullmatch(r'[0-9a-f]{64}',expected_program_sha256) or sha256(raw).hexdigest()!=expected_program_sha256:
            raise ValueError('Program differs from its external hash')
        payload=json.loads(raw)
        source=payload['program'] if payload.get('version')=='warehouse_native_rcpd_feedback_v1' else payload
        self.program=ExecutableProgram.from_dict(source)
        if (tuple(self.program.feature_names)!=tuple(runtime.actor.metadata['feature_names'])
                or tuple(self.program.action_names)!=ACTIONS
                or self.program.metadata.get('native_source_actor_sha256')!=runtime.actor_sha256
                or self.program.metadata.get('action_legality_features') or self.program.metadata.get('action_constraint_reason_features')
                or self.program.root.depth()>12 or self.program.root.leaf_count()>256):
            raise ValueError('Program source, observed197 schema or unmasked action contract differs')
        stack=[self.program.root]
        while stack:
            node=stack.pop()
            if node.is_leaf:
                p=np.asarray(node.probabilities)
                if p.shape!=(5,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('Invalid program leaf')
            else:
                if node.feature not in self.program.feature_names or node.threshold is None or not np.isfinite(node.threshold) or node.left is None or node.right is None:raise ValueError('Invalid program predicate')
                stack.extend((node.left,node.right))
        self.actor_sha256=runtime.actor_sha256;self.runtime_signature=runtime.signature
        self.program_sha256=expected_program_sha256;self.program_content_sha256=digest(self.program.to_dict())
        self.sources=explanation_sources();self.test_fixture=allow_test_fixture
        self.eligible=False;self.participant_enabled=False;self.study_ready=False;self.explanation_qualified=True;self.release_ready=False;self.fixture_enabled=allow_test_fixture
        self.signature=digest({'version':VERSION,'runtime':self.runtime_signature,'program':self.program_sha256,'sources':self.sources})
        self.contract_report={'version':VERSION,'signature':self.signature,'test_fixture':allow_test_fixture,
            'eligibility_evaluated':True,'explanation_eligible':True,'explanation_qualified':True,'study_ready':False,'participant_enabled':False,'release_ready':False,
            'scope':'component_evidence_rendering_only_requires_separate_qualified_release'}

    def _assert_current(self,runtime):
        if (type(runtime) is not OnlineAlignmentRuntime or runtime.verify_binding()!=self.runtime_signature
                or runtime.actor_sha256!=self.actor_sha256 or runtime.test_fixture is not self.test_fixture
                or explanation_sources()!=self.sources):raise ValueError('explanation_runtime_version_mismatch')
        if sha256(self.program_path.read_bytes()).hexdigest()!=self.program_sha256 or digest(self.program.to_dict())!=self.program_content_sha256:
            raise ValueError('explanation_program_changed')
        if (self.eligible or self.participant_enabled or self.study_ready or not self.explanation_qualified or self.release_ready):raise ValueError('This component cannot grant participant qualification')

    def answer(self, request, frame_record, runtime):
        self._assert_current(runtime)
        language = "en" if request.get("language") == "en" else "zh"
        parsed = parse_question(request.get("question"), request.get("focus"))
        if frame_record.get("runtime_signature") != runtime.signature:
            raise ValueError("explanation_runtime_version_mismatch")
        runtime.from_snapshot(frame_record["after"])
        current = frame_record["after"]
        selected_frame = current["state"]["frame"]
        if "frame" in request and (type(request["frame"]) is not int
                                    or request["frame"] != selected_frame):
            raise ValueError("explanation_frame_mismatch")
        mentions = re.findall(
            r"第\s*(\d+)\s*(?:帧|步)|\bframe\s+(\d+)\b|\bat step\s+(\d+)\b",
            str(request.get("question", "")).casefold(),
        )
        if any(int(next(value for value in match if value)) != selected_frame
               for match in mentions):
            parsed = _clarify("select_requested_frame")
        if parsed["intent"] == "clarify":
            return _response(
                clarify_answer(parsed["reason"], selected_frame, language),
                _technical_detail(language=language, frame=selected_frame,
                                  focus=request.get("focus", "executed"),
                                  actor_sha256=self.actor_sha256,
                                  extra=(("问题未能绑定到单一、受支持的证据查询。"
                                          if language == "zh" else
                                          "The question could not be bound to one supported evidence query."),)),
            )

        focus = parsed.get("focus", "executed")
        use_before = focus == "executed"
        if use_before and "before" not in frame_record:
            return _response(
                clarify_answer("no_executed_action_at_initial_frame", selected_frame, language),
                _technical_detail(language=language, frame=selected_frame, focus=focus,
                                  actor_sha256=self.actor_sha256),
            )
        replay_verified = False
        if "before" in frame_record:
            if frame_record["before"]["state"]["frame"] + 1 != selected_frame:
                raise ValueError("historical_frame_sequence_mismatch")
            frame_record = verify_historical_transition(
                frame_record, runtime, self.actor_sha256)
            current = frame_record["after"]
            replay_verified = True
        bound = frame_record["before"] if use_before else current

        if parsed["intent"] == "counterfactual":
            branch = runtime.counterfactual(
                bound, parsed["player_actions"], steps=parsed["steps"])
            start_frame = bound["state"]["frame"]
            assumed = branch["assumed_player_actions"]
            teammate_actions = [transition["submitted_actions"]["robot_2"]
                                for transition in branch["transitions"]]
            deliveries = sum(
                transition["after"]["state"]["total_deliveries"]
                - transition["before"]["state"]["total_deliveries"]
                for transition in branch["transitions"]
            )
            collisions = sum(
                transition["after"]["state"]["robot_collision_events"]
                - transition["before"]["state"]["robot_collision_events"]
                for transition in branch["transitions"]
            )
            if not teammate_actions:
                answer = ("该帧的回合已经结束，不能继续推进。" if language == "zh"
                          else "The round has already ended at this frame.")
            elif language == "zh":
                choices = "、".join(LABELS[language][action] for action in teammate_actions)
                answer = (f"如果你按设定动作推进 {len(teammate_actions)} 步，机器人2会依次选择{choices}；"
                          f"团队新增配送 {deliveries} 件，发生 {collisions} 次碰撞。"
                          "这是隔离模拟，真实回合没有改变。")
            else:
                choices = ", ".join(LABELS[language][action] for action in teammate_actions)
                answer = (f"If you take the specified actions for {len(teammate_actions)} steps, "
                          f"Robot 2 chooses {choices}; the team adds {deliveries} deliveries "
                          f"and has {collisions} collisions. The live round is unchanged.")
            detail_rows = []
            for transition in branch["transitions"]:
                after = transition["after"]["state"]
                submitted = transition["submitted_actions"]["robot_2"]
                executed = transition["executed_actions"]["robot_2"]
                detail_rows.append(
                    (f"第 {after['frame']} 步：你{LABELS[language][transition['participant_action']]}；"
                     f"策略提交{LABELS[language][submitted]}，实际执行{LABELS[language][executed]}。")
                    if language == "zh" else
                    (f"Step {after['frame']}: you {LABELS[language][transition['participant_action']]}; "
                     f"policy submitted {LABELS[language][submitted]}, physically executed {LABELS[language][executed]}.")
                )
            return _response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256, replay_verified=replay_verified,
                counterfactual=True,
                extra=(("玩家动作假设：" if language == "zh" else "Player-action assumptions: ")
                       + ("、" if language == "zh" else ", ").join(
                           LABELS[language][action] for action in assumed), *detail_rows),
            ))

        env = runtime.from_snapshot(bound)
        if parsed["intent"] == "rules":
            answer = (
                f"成功移动一格消耗 {env.config.move_battery_cost:g}% 电量；在充电格实际停留一步最多恢复 "
                f"{env.config.charge_per_wait:g}%。撞墙或机器人冲突会取消移动，但仍消耗一步。"
                if language == "zh" else
                f"A successful one-cell move costs {env.config.move_battery_cost:g}% battery; "
                f"staying on the charger restores up to {env.config.charge_per_wait:g}% per turn. "
                "Walls and robot conflicts cancel movement but still consume a turn."
            )
            return _response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256,
                extra=(("依据：公开环境配置与物理规则。" if language == "zh"
                        else "Basis: public environment configuration and physics."),),
            ))
        if parsed["intent"] == "failure":
            reason = current["state"].get("terminal_reason")
            labels = {
                "battery_shutdown": ("至少一台机器人在充电格外耗尽电量",
                                     "at least one robot exhausted its battery away from the charger"),
                "horizon": ("达到本局步数上限", "the round reached its turn limit"),
            }
            if reason not in labels:
                answer = ("这一帧没有记录回合失败。" if language == "zh"
                          else "No round failure is recorded at this frame.")
            else:
                answer = (f"本局在第 {selected_frame} 帧结束，因为{labels[reason][0]}。"
                          if language == "zh" else
                          f"The round ended at frame {selected_frame} because {labels[reason][1]}.")
            return _response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256, replay_verified=replay_verified,
                extra=((f"终局记录：{reason or 'none'}" if language == "zh"
                        else f"Terminal record: {reason or 'none'}"),),
            ))
        if not use_before and env.done:
            answer = (f"第 {selected_frame} 帧回合已结束，没有下一次决策。"
                      if language == "zh" else
                      f"The round ended at frame {selected_frame}; there is no next decision.")
            return _response(answer, _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                actor_sha256=self.actor_sha256))

        obs = env.observations()["robot_2"]
        _, next_decision = runtime.decision(env)
        probabilities = np.asarray(next_decision["probabilities"]["robot_2"], dtype=float)
        neural_action = next_decision["policy_actions"]["robot_2"]
        features = dict(zip(env.feature_names, map(float, obs)))
        tree_action = self.program.predict(features)
        trace = self.program.trace(features)
        conditions = [describe_predicate(step, env, language) for step in trace]
        conditions = [condition for condition in conditions if condition]
        tree_matches = tree_action == neural_action
        if use_before:
            recorded = frame_record.get("decision", {})
            recorded_probs = np.asarray(
                recorded.get("probabilities", {}).get("robot_2", []))
            if (recorded.get("actor_sha256") != self.actor_sha256
                    or recorded.get("frame") != bound["state"]["frame"]
                    or recorded.get("policy_actions", {}).get("robot_2") != neural_action
                    or recorded.get("observation_hashes", {}).get("robot_2")
                    != sha256(obs.tobytes()).hexdigest()
                    or recorded_probs.shape != (5,)
                    or not np.isfinite(recorded_probs).all()
                    or not np.allclose(recorded_probs, probabilities,
                                       atol=1e-6, rtol=1e-5)
                    or frame_record.get("submitted_actions", {}).get("robot_2")
                    != neural_action):
                raise ValueError("historical_neural_decision_cannot_be_verified")

        def evidence(*extra):
            return _technical_detail(
                language=language, frame=selected_frame, focus=focus,
                neural_action=neural_action, probabilities=probabilities,
                tree_action=tree_action, tree_matches=tree_matches,
                conditions=conditions, actor_sha256=self.actor_sha256,
                replay_verified=replay_verified, extra=extra,
            )

        teammate = _agent(env)
        after_env = runtime.from_snapshot(current)
        after_teammate = _agent(after_env)
        public_history = after_env.public_history() if use_before else env.public_history()
        collision_kind = public_history.get("collision_kind") if public_history.get("valid") else "none"

        if parsed["intent"] == "collision":
            if not use_before:
                answer = ("是否碰撞还取决于你下一步的动作，目前不能提前确定。"
                          if language == "zh" else
                          "A collision also depends on your next action, so it cannot be determined yet.")
            elif collision_kind == "same_target":
                player_action = frame_record["submitted_actions"]["robot_1"]
                answer = (f"机器人2选择了{LABELS[language][neural_action]}，你选择了{LABELS[language][player_action]}；"
                          "双方要进入同一格，因此环境取消了两人的移动。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][neural_action]} and you chose {LABELS[language][player_action]}. "
                          "Both moves targeted the same cell, so the environment canceled them.")
            elif collision_kind == "swap":
                player_action = frame_record["submitted_actions"]["robot_1"]
                answer = (f"机器人2选择了{LABELS[language][neural_action]}，你选择了{LABELS[language][player_action]}；"
                          "双方会互换位置，因此环境取消了两人的移动。"
                          if language == "zh" else
                          f"Robot 2 chose {LABELS[language][neural_action]} and you chose {LABELS[language][player_action]}. "
                          "The moves would swap positions, so the environment canceled them.")
            elif collision_kind == "occupied_stationary":
                canceled = public_history.get("move_canceled", {})
                player_action = frame_record["submitted_actions"]["robot_1"]
                if canceled.get("robot_2") is True and canceled.get("robot_1") is False:
                    answer = (f"机器人2选择了{LABELS[language][neural_action]}，但它要进入的格子仍被你占用；"
                              "环境因此取消了机器人2的移动，它实际留在原地。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][neural_action]}, but you remained in its target cell. "
                              "The environment canceled Robot 2's move, so it stayed in place.")
                elif canceled.get("robot_1") is True and canceled.get("robot_2") is False:
                    answer = (f"机器人2选择了{LABELS[language][neural_action]}并留在原格；"
                              f"你选择了{LABELS[language][player_action]}进入它占用的格子，因此环境取消了你的移动。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][neural_action]} and remained in its cell. "
                              f"You chose {LABELS[language][player_action]} into that occupied cell, so the environment canceled your move.")
                else:
                    answer = (f"机器人2选择了{LABELS[language][neural_action]}，你选择了{LABELS[language][player_action]}；"
                              "其中一方要进入另一方没有离开的格子，因此环境取消了冲突移动。"
                              if language == "zh" else
                              f"Robot 2 chose {LABELS[language][neural_action]} and you chose {LABELS[language][player_action]}. "
                              "One robot targeted the cell the other did not leave, so the environment canceled the conflicting move.")
            else:
                answer = ("所选步骤没有记录机器人碰撞。" if language == "zh"
                          else "No robot collision is recorded for the selected step.")
            return _response(answer, evidence(
                (f"物理冲突类型：{collision_kind or 'none'}" if language == "zh"
                 else f"Physical conflict type: {collision_kind or 'none'}")))

        if parsed["intent"] == "influence":
            if use_before:
                answer = (
                    "你的本步动作没有影响机器人2对同一步的选择：双方只看到行动前状态，再同时提交动作。"
                    "这一步确认后会进入下一次决策可见的历史记录。"
                    if language == "zh" else
                    "Your action did not affect Robot 2's choice on the same step: both chose from the pre-action state and submitted simultaneously. "
                    "Once confirmed, this step becomes visible in the next decision's history."
                )
            else:
                answer = (
                    "你的上一已确认动作属于机器人2当前可见的历史信息，但现有证据不能单独证明它改变了这次选择。"
                    if language == "zh" else
                    "Your preceding confirmed action is part of Robot 2's currently visible history, but the evidence does not isolate whether it changed this choice."
                )
            return _response(answer, evidence(
                ("因果边界：玩家本步命令不进入同一步策略输入。" if language == "zh"
                 else "Causal boundary: the player's current command is absent from the same-step policy input.")))

        if parsed["intent"] == "energy":
            charger = tuple(env.layout.charger_position)
            distance = _distance(env, teammate.position, charger)
            required = distance * env.config.move_battery_cost
            if tuple(teammate.position) == charger:
                recoverable = min(env.config.charge_per_wait,
                                  max(0.0, 100.0 - teammate.battery))
                answer = (
                    f"机器人2当前电量为 {teammate.battery:g}%，并在充电格上；实际停留一步可恢复 {recoverable:g}%。"
                    if language == "zh" else
                    f"Robot 2 has {teammate.battery:g}% battery and is on the charger; staying for a turn restores {recoverable:g}%."
                )
            elif teammate.battery < required:
                answer = (
                    f"机器人2当前电量为 {teammate.battery:g}%，到充电格还有 {distance} 格；按每格 {env.config.move_battery_cost:g}% 计算，仅抵达就需要约 {required:g}%，电量不足。"
                    if language == "zh" else
                    f"Robot 2 has {teammate.battery:g}% battery and is {distance} cells from the charger. At {env.config.move_battery_cost:g}% per move, reaching it alone needs about {required:g}%, so the battery is insufficient."
                )
            else:
                answer = (
                    f"机器人2当前电量为 {teammate.battery:g}%，到充电格还有 {distance} 格；按每格 {env.config.move_battery_cost:g}% 计算，抵达所需电量约为 {required:g}%。"
                    if language == "zh" else
                    f"Robot 2 has {teammate.battery:g}% battery and is {distance} cells from the charger; at {env.config.move_battery_cost:g}% per move, reaching it needs about {required:g}%."
                )
            return _response(answer, evidence(
                ("这里只核验当前电量与抵达成本，不推断未记录的长期充电计划。"
                 if language == "zh" else
                 "This checks current battery and travel cost only; it does not infer an unrecorded charging plan.")))

        if parsed["intent"] == "goal":
            objectives = _objective_candidates(env)
            if teammate.carrying_task_id and objectives:
                target = _objective_label(objectives[0], language)
                answer = (f"机器人2正携带任务{objectives[0]['slot']}的货物；当前可确认的任务方向是送到{target}。"
                          if language == "zh" else
                          f"Robot 2 is carrying task {objectives[0]['slot']}; its confirmed task direction is delivery to {target}.")
            else:
                projected = _position_after_action(env, neural_action)
                progress = _best_progress(env, projected)
                if progress:
                    target = _objective_label(progress, language)
                    answer = (
                        f"机器人2目前空载；它下一步会选择{LABELS[language][neural_action]}。"
                        f"如果移动没有被碰撞取消，到{target}的距离会从 {progress['before']} 格缩短到 {progress['after']} 格，但这不能确认长期分工。"
                        if language == "zh" else
                        f"Robot 2 is empty-handed and will next choose {LABELS[language][neural_action]}. "
                        f"If no collision cancels the move, its distance to {target} falls from {progress['before']} to {progress['after']} cells; this does not establish a long-term assignment."
                    )
                else:
                    labels = [_objective_label(item, language) for item in objectives]
                    available = ("、".join(labels) if language == "zh" else ", ".join(labels)) or ("无" if language == "zh" else "none")
                    answer = (
                        f"机器人2目前空载，可领取的目标有{available}；现有证据无法确认它已承诺其中哪一个。"
                        if language == "zh" else
                        f"Robot 2 is empty-handed; available pickup targets are {available}. The evidence does not show a commitment to one of them."
                    )
            return _response(answer, evidence(
                ("任务方向只依据公开持货状态与本步可验证的几何变化。"
                 if language == "zh" else
                 "Task direction is based only on public carrying state and verifiable geometric change.")))

        alternative = parsed.get("alternative_action")
        mentioned = parsed.get("mentioned_action")
        executed_action = (frame_record.get("executed_actions", {}).get("robot_2")
                           if use_before else None)
        if alternative == neural_action or (not alternative and mentioned
                                             and mentioned != neural_action):
            if use_before and executed_action == "WAIT" and neural_action != "WAIT":
                reason = ("机器人冲突" if collision_kind and collision_kind != "none"
                          else "墙或边界")
                reason_en = ("a robot conflict" if collision_kind and collision_kind != "none"
                             else "a wall or boundary")
                answer = (
                    f"机器人2选择了{LABELS[language][neural_action]}，但{reason}阻止了移动，所以实际留在原地。"
                    if language == "zh" else
                    f"Robot 2 chose {LABELS[language][neural_action]}, but {reason_en} blocked the move, so it stayed in place."
                )
            else:
                answer = (
                    f"机器人2在所选步骤实际选择了{LABELS[language][neural_action]}，请确认问题中的动作。"
                    if language == "zh" else
                    f"Robot 2 actually chose {LABELS[language][neural_action]} on the selected step; check the action named in the question."
                )
            return _response(answer, evidence())

        destination = (tuple(after_teammate.position) if use_before
                       else _position_after_action(env, neural_action))
        progress = _best_progress(env, destination)

        if parsed["intent"] == "alternative":
            if alternative is None:
                answer = ("请指定一个替代动作，例如“为什么没有向上”。"
                          if language == "zh" else
                          "Name one alternative action, such as “Why not move up?”")
                return _response(answer, evidence())
            if not tree_matches:
                answer = (
                    f"机器人2选择了{LABELS[language][neural_action]}，没有选择{LABELS[language][alternative]}；现有证据无法可靠说明更具体的原因。"
                    if language == "zh" else
                    f"Robot 2 chose {LABELS[language][neural_action]} rather than {LABELS[language][alternative]}; the available evidence cannot reliably identify a more specific reason."
                )
                return _response(answer, evidence(
                    ("近似程序与实际提交动作不一致，因此其分支未用于主回答。"
                     if language == "zh" else
                     "The approximation disagreed with the submitted action, so its branch was excluded from the main answer.")))
            if progress:
                alternative_position = _position_after_action(env, alternative)
                alternative_distance = _distance(env, alternative_position, progress["target"])
                target = _objective_label(progress, language)
                if language == "zh":
                    answer = (f"机器人2选择了{LABELS[language][neural_action]}，没有选择{LABELS[language][alternative]}；"
                              f"前者把它到{target}的距离从 {progress['before']} 格缩短到 {progress['after']} 格，"
                              f"后者会变为 {alternative_distance} 格。")
                else:
                    answer = (f"Robot 2 chose {LABELS[language][neural_action]} rather than {LABELS[language][alternative]}. "
                              f"The chosen move changes its distance to {target} from {progress['before']} to {progress['after']} cells; "
                              f"the alternative would make it {alternative_distance} cells.")
            else:
                answer = (
                    f"机器人2选择了{LABELS[language][neural_action]}，没有选择{LABELS[language][alternative]}；现有证据无法可靠说明更具体的物理原因。"
                    if language == "zh" else
                    f"Robot 2 chose {LABELS[language][neural_action]} rather than {LABELS[language][alternative]}; the evidence does not reliably establish a more specific physical reason."
                )
            return _response(answer, evidence())

        if use_before and executed_action == "WAIT" and neural_action != "WAIT":
            reason = ("双方发生机器人冲突" if collision_kind and collision_kind != "none"
                      else "目标格是墙或边界")
            reason_en = ("the robots conflicted" if collision_kind and collision_kind != "none"
                         else "the target cell was a wall or boundary")
            answer = (
                f"机器人2选择了{LABELS[language][neural_action]}，但{reason}，所以实际留在原地。"
                if language == "zh" else
                f"Robot 2 chose {LABELS[language][neural_action]}, but {reason_en}, so it physically stayed in place."
            )
            return _response(answer, evidence(
                ("策略选择与环境执行结果已分开核验。" if language == "zh"
                 else "The policy choice and physical outcome were verified separately.")))

        before_battery = float(teammate.battery)
        after_battery = float(after_teammate.battery)
        if neural_action == "WAIT":
            if use_before and after_battery > before_battery:
                answer = (
                    f"机器人2刚才等待是在充电；电量从 {before_battery:g}% 恢复到 {after_battery:g}%。"
                    if language == "zh" else
                    f"Robot 2 waited to charge; its battery rose from {before_battery:g}% to {after_battery:g}%."
                )
            else:
                answer = ("机器人2选择了等待；现有证据无法可靠说明更具体的原因。"
                          if language == "zh" else
                          "Robot 2 chose to wait; the available evidence cannot reliably identify a more specific reason.")
            return _response(answer, evidence())

        event = next((item for item in frame_record.get("events", [])
                      if item.get("agent_id") == "robot_2"
                      and item.get("event") in ("pickup", "delivery")), None) if use_before else None
        if event:
            slot = _task_slot(env, event.get("task_id"))
            if event["event"] == "pickup":
                answer = (f"机器人2刚才{LABELS[language][neural_action]}到达任务{slot}的A点，并领取了货物。"
                          if language == "zh" else
                          f"Robot 2 moved {LABELS[language][neural_action]} to task {slot}'s A point and collected the item.")
            else:
                answer = (f"机器人2刚才{LABELS[language][neural_action]}到达任务{slot}的B点，并完成了交付。"
                          if language == "zh" else
                          f"Robot 2 moved {LABELS[language][neural_action]} to task {slot}'s B point and completed the delivery.")
            return _response(answer, evidence())

        if progress:
            target = _objective_label(progress, language)
            intervention = _objective_feature_intervention(
                env, runtime, obs, progress, neural_action, probabilities)
            supported = (tree_matches and _trace_supports_objective(trace, progress)
                         and intervention.get("supported") is True)
            if language == "zh":
                if use_before and supported:
                    answer = (f"机器人2刚才{LABELS[language][neural_action]}，是在靠近{target}；"
                              f"这一步把距离从 {progress['before']} 格缩短到 {progress['after']} 格。")
                elif use_before:
                    answer = (f"机器人2刚才选择了{LABELS[language][neural_action]}；这一步把它到{target}的距离"
                              f"从 {progress['before']} 格缩短到 {progress['after']} 格，但现有证据无法可靠说明更具体的原因。")
                else:
                    answer = (f"机器人2下一步会选择{LABELS[language][neural_action]}；"
                              f"如果移动没有被碰撞取消，到{target}的距离会从 {progress['before']} 格缩短到 {progress['after']} 格。")
            else:
                if use_before and supported:
                    answer = (f"Robot 2 moved {LABELS[language][neural_action]} to approach {target}; "
                              f"the step reduced the distance from {progress['before']} to {progress['after']} cells.")
                elif use_before:
                    answer = (f"Robot 2 chose {LABELS[language][neural_action]}; the step reduced its distance "
                              f"to {target} from {progress['before']} to {progress['after']} cells, but the available "
                              "evidence cannot reliably identify a more specific reason.")
                else:
                    answer = (f"Robot 2 will next choose {LABELS[language][neural_action]}. If no collision cancels "
                              f"the move, its distance to {target} falls from {progress['before']} to {progress['after']} cells.")
            intervention_detail = (
                f"抽象目标关系干预：中和{target}的相对位置特征后，提交动作变为"
                f"{LABELS[language].get(intervention.get('altered_action'), '—')}，原动作支持度下降 "
                f"{100 * intervention.get('chosen_probability_drop', 0.0):.2f} 个百分点。"
                if language == "zh" else
                f"Abstract target-relation intervention: after neutralizing relation features for {target}, "
                f"the submitted action became {LABELS[language].get(intervention.get('altered_action'), '—')} "
                f"and support for the original action fell by {100 * intervention.get('chosen_probability_drop', 0.0):.2f} percentage points."
            )
            return _response(answer, evidence(
                "主回答只陈述可验证的任务进展；近似分支与抽象干预仅在方向一致时提供有限支持。"
                if language == "zh" else
                "The main answer states verifiable task progress; the approximation and abstract intervention provide limited support only when directionally consistent.",
                intervention_detail))

        answer = (
            f"机器人2选择了{LABELS[language][neural_action]}；这一步没有缩短它到当前取货点、交付点或充电格的路线，现有证据无法可靠说明更具体的原因。"
            if language == "zh" else
            f"Robot 2 chose {LABELS[language][neural_action]}; the move did not shorten its route to a current pickup, delivery, or the charger, so the evidence cannot reliably identify a more specific reason."
        )
        return _response(answer, evidence())



# Natural aliases for deployment adapters.
AlignmentOnlineExplainer = OnlineAlignmentExplainer
DiverseAlignmentExplainer = OnlineAlignmentExplainer
