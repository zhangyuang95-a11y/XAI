"""Independent raw behavioral-answer verification, never a release bypass.

Expected text is constructed from a frozen case specification and independently
replayed physical/neural facts. It does not call the production parser, text
helpers, or counterfactual renderer to construct its oracle. The production
renderer is invoked only afterwards as a second, exact-source binding check.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from copy import deepcopy
import json
import re

import numpy as np

from backend.training.warehouse_native_common import ROOT,digest,file_hash
from core.program import ExecutableProgram
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.policy import ACTIONS,NumPyNativeActor
from env.warehouse_native.runtime import NativeRuntime
from env.warehouse_native.scenarios import scenario_fingerprint
from env.warehouse_native import explanation as renderer

VERSION="warehouse-native-behavioral-answers.v3"
CASE_SPEC_VERSION="warehouse-native-behavioral-case-spec.v1"
SCENARIO_COUNT=12
TRAJECTORY_SPEC_VERSION="warehouse-native-answer-trajectories.v1"
LABELS={"zh":dict(zip(ACTIONS,("向上","向下","向左","向右","等待"))),
        "en":dict(zip(ACTIONS,("up","down","left","right","wait")))}


def case_specs():
    """Fixed full matrix; do not select cases according to observed answers.

    ``expected`` describes semantics, not the answer content or a pass flag.
    ``request_frame_offset`` creates one explicitly expected rejection case.
    All forty specifications run in all first twelve explanation-test scenes.
    """
    rows=[
      ("reason_past","first","为什么刚才这样做？","Why did you do that?",{"intent":"reason","focus":"executed"}),
      ("reason_next","third","为什么下一步选择这个动作？","Why choose this next action?",{"intent":"reason","focus":"next"}),
      ("reason_initial","initial","为什么刚才这样做？","Why did you do that?",{"intent":"reason","focus":"executed","clarification_reason":"no_executed_action_at_initial_frame"}),
      ("reason_terminal","terminal","为什么下一步选择这个动作？","Why choose this next action?",{"intent":"reason","focus":"next"}),
      ("alternative_wait","first","为什么刚才不等待？","Why didn't you wait?",{"intent":"alternative","focus":"executed","alternative_action":"WAIT"}),
      ("alternative_left","first","为什么刚才不向左？","Why didn't you move left?",{"intent":"alternative","focus":"executed","alternative_action":"LEFT"}),
      ("physical_wait","first","你刚才为什么等待？","Why did you wait?",{"intent":"reason","focus":"executed","mentioned_action":"WAIT"}),
      ("wait_three","third","如果我等待三步？","What if I wait three steps?",{"intent":"counterfactual","focus":"next","player_actions":["WAIT"]*3,"steps":3}),
      ("left_first","first","如果我向左一步，接下来三步怎样？","What if I move left once over the next three steps?",{"intent":"counterfactual","focus":"next","player_actions":["LEFT"],"steps":3}),
      ("left_repeat","third","如果我连续向左三步？","What if I move left for three steps?",{"intent":"counterfactual","focus":"next","player_actions":["LEFT"]*3,"steps":3}),
      ("left_past","third","如果我刚才向左一步，接下来三步怎样？","What if I had moved left once over the next three steps?",{"intent":"counterfactual","focus":"executed","player_actions":["LEFT"],"steps":3}),
      ("just_wait","third","如果我只等待三步？","What if I just wait three steps?",{"intent":"counterfactual","focus":"next","player_actions":["WAIT"]*3,"steps":3}),
      ("terminal_counterfactual","terminal","如果我等待三步？","What if I wait three steps?",{"intent":"counterfactual","focus":"next","player_actions":["WAIT"]*3,"steps":3}),
      ("rules","initial","规则中的移动耗电和充电速度是多少？","What are the movement energy cost and charge rate rules?",{"intent":"rules","focus":"next"}),
      ("failure_terminal","terminal","为什么回合结束了？","Why did the round end?",{"intent":"failure","focus":"next"}),
      ("failure_running","first","为什么回合结束了？","Why did the round end?",{"intent":"failure","focus":"next"}),
      ("ambiguous","initial","如果我向左还是向右？","What if I move left or right?",{"intent":"clarify","clarification_reason":"one_player_action_required"}),
      ("too_long","initial","如果我等待四步？","What if I wait four steps?",{"intent":"clarify","clarification_reason":"at_most_three_steps"}),
      ("strategy","initial","为什么选择最优路线？","Why choose the optimal route?",{"intent":"clarify","clarification_reason":"intentions_or_strategy_not_supported"}),
      ("wrong_requested_frame","first","为什么刚才这样做？","Why did you do that?",{"intent":"reason","focus":"executed","error":"explanation_frame_mismatch"}),
    ]
    result=[]
    for identity,anchor,zh,en,expected in rows:
        for language,question in (("zh",zh),("en",en)):
            spec={"id":identity+"_"+language,"language":language,"question":question,"anchor":anchor,"expected":deepcopy(expected)}
            if identity=="wrong_requested_frame":spec["request_frame_offset"]=1
            result.append(spec)
    return result


def answer_verification_sources():
    paths={Path(__file__),ROOT/"backend/training/warehouse_native_answer_audit.py",
        ROOT/"backend/training/warehouse_native_extract_replay.py"}
    for directory in ("env/warehouse_native","env/warehouse","core"):
        paths.update((ROOT/directory).glob("*.py"))
    return {str(path.relative_to(ROOT)):file_hash(path) for path in sorted(paths)}


def request_for_spec(spec,frame):
    request={"question":spec["question"],"language":spec["language"],"frame":frame+spec.get("request_frame_offset",0)}
    if "focus" in spec:request["focus"]=spec["focus"]
    return request


def trajectory_spec():
    return {"version":TRAJECTORY_SPEC_VERSION,"scene_pool":"first_12_explanation_test",
        "profiles":["skilled","assertive","noisy"],"profile_assignment":"scene_index_modulo_three",
        "player_role":"robot_1","seed_sequence":[98131,"scene_index",1],"neural_execution":"deterministic"}


def _strict_json(value):
    # Round trip rejects NaN/Infinity as well as non-JSON evidence values.
    return json.loads(json.dumps(value,allow_nan=False))


def _equal(a,b,message):
    if digest(a)!=digest(b):raise ValueError(message)


def _text_hash(value):return sha256(value.encode("utf-8")).hexdigest()


def _clarification(reason,frame,language):
    if reason=="no_executed_action_at_initial_frame":
        return f"第 {frame} 帧尚无已执行动作，请改问下一次决策或选中后续历史帧。" if language=="zh" else f"Frame {frame} has no executed action. Ask about the next decision or select a later historical frame."
    if reason=="select_requested_frame":
        return f"当前选中第 {frame} 帧，请先选中问题提到的历史帧。" if language=="zh" else f"Frame {frame} is selected. Please first select the historical frame named in your question."
    if reason=="at_most_three_steps":
        return "这里只支持一至三步预测，请缩短预测范围。" if language=="zh" else "Predictions support one to three steps only. Please shorten the horizon."
    return ("请明确所选帧队友的已执行或下一动作；问‘为什么不’时请指定一个替代动作。反事实只支持你的一种动作，最多三步，未指定的后续动作按等待处理。不推测队友动机或提供攻略。"
        if language=="zh" else "Please specify the teammate's executed or next action. A why-not question needs one alternative. Counterfactuals support one player action over at most three steps; unspecified following actions are WAIT. Intentions and strategy advice are outside scope.")


def _trace_facts(program,env):
    """Walk actual nodes independently of program.trace or text predicates."""
    obs=env.observations()["robot_2"];features=dict(zip(env.feature_names,map(float,obs)))
    node=program.root;trace=[]
    while not node.is_leaf:
        observed=features[node.feature];left=observed<=node.threshold
        trace.append({"feature":node.feature,"value":observed,"threshold":node.threshold,"left":bool(left)})
        node=node.left if left else node.right
    distribution=np.asarray(node.probabilities,dtype=float)
    return ACTIONS[int(distribution.argmax())],trace


def _conditions(trace,env,language):
    names={"self.battery":("队友电量","teammate battery",100),"other.battery":("玩家电量","player battery",100),
        "other.path_distance":("双方通路距离","path distance between robots",env.config.rows*env.config.cols-1),
        "charger.self.path_distance":("队友到充电站的通路距离","teammate path distance to charger",env.config.rows*env.config.cols-1),
        "charger.other.path_distance":("玩家到充电站的通路距离","player path distance to charger",env.config.rows*env.config.cols-1),
        "time.remaining":("剩余回合数","remaining turns",env.config.horizon)}
    output=[]
    for item in trace:
        if item["feature"] not in names:continue
        zh,en,scale=names[item["feature"]];relation="≤" if item["left"] else ">"
        output.append(f"{zh if language=='zh' else en} {relation} {item['threshold']*scale:.2f}（{item['value']*scale:.2f}）")
    return output[:3]


def _independent_step(env,runtime,player):
    """Direct physics plus raw logits, without runtime.step/decision helpers."""
    if env.done:raise ValueError("audit_prefix_after_terminal")
    before=env.snapshot();observations=env.observations();actions={};probabilities={};hashes={}
    # Runtime's real forward is batch-two. Preserve its float32 operation shape.
    matrix=np.stack([observations[name] for name in sorted(observations)])
    logits=runtime.actor.logits(matrix)
    probs=np.exp(logits-logits.max(axis=1,keepdims=True));probs/=probs.sum(axis=1,keepdims=True)
    for i,name in enumerate(sorted(observations)):
        actions[name]=ACTIONS[int(logits[i].argmax())];probabilities[name]=probs[i].tolist()
        hashes[name]=sha256(np.asarray(observations[name],dtype=np.float32).tobytes()).hexdigest()
    decision={"version":"warehouse-native-direct-runtime.v1","actor_sha256":runtime.actor.sha256,"frame":env.state.frame,
        "policy_actions":actions,"probabilities":probabilities,"observation_hashes":hashes,"masks":False,"post_policy_overrides":0}
    submitted={"robot_1":player,"robot_2":actions["robot_2"]}
    _,rewards,terminated,truncated,info=env.step(submitted)
    return {"before":before,"after":env.snapshot(),"decision":decision,"participant_action":player,
        "submitted_actions":submitted,"executed_actions":info["executed_actions"],"rewards":rewards,"done":bool(terminated or truncated)}


def _alternative_text(action,chosen,probabilities,logits,env,language):
    from env.warehouse.navigation import MOVE_DELTAS
    alt=ACTIONS.index(action);best=ACTIONS.index(chosen);label=LABELS[language]
    tie=bool(logits[alt]==logits[best])
    if language=="zh":
        text=f"{label[action]}的 NN 概率为 {100*probabilities[alt]:.2f}%，所选{label[chosen]}为 {100*probabilities[best]:.2f}%。"
        text+="两者并列最高，确定性 argmax 按固定动作顺序取首项。" if tie else "确定性策略选择概率最高的动作。"
    else:
        text=f"NN probability for {label[action]} was {100*probabilities[alt]:.2f}%, versus {100*probabilities[best]:.2f}% for chosen {label[chosen]}. "
        text+="They tied for the maximum; deterministic argmax uses the fixed action order. " if tie else "The deterministic policy chose the highest-probability action. "
    if action=="WAIT":return text+("等待没有被动作掩码排除。" if language=="zh" else "WAIT was not removed by an action mask. ")
    r,c=env.state.by_id("robot_2").position;dr,dc=MOVE_DELTAS[action];open_cell=env.layout.is_passable((r+dr,c+dc))
    if language=="zh":return text+("该方向邻格可通行，但是否移动成功还取决于双方冲突。" if open_cell else "该方向邻格是墙或边界，但动作没有被掩码排除。")+"这是环境事实，不证明 NN 正是因此没有选择它。"
    return text+("The adjacent cell was traversable; physical conflicts can still prevent movement. " if open_cell else "The adjacent cell was a wall or boundary, but the action was not masked. ")+"This environmental fact does not establish why the NN ranked the action lower. "


def expected_answer(spec,record,runtime,program):
    """Return complete permitted text and private facts; no renderer is called."""
    expected=spec["expected"];language=spec["language"];frame=record["after"]["state"]["frame"]
    if "error" in expected:return "",{"expected_error":expected["error"]}
    if "clarification_reason" in expected:
        return _clarification(expected["clarification_reason"],frame,language),{"clarification_reason":expected["clarification_reason"]}
    intent=expected["intent"];past=expected.get("focus")=="executed"
    if past and "before" not in record:raise ValueError("missing_historical_record")
    snapshot=record["before"] if past else record["after"]
    env=NativeWarehouseEnv();env.restore(snapshot)
    facts={"intent":intent,"selected_frame":frame,"bound_frame":env.state.frame,"focus":"executed" if past else "next"}
    if intent=="counterfactual":
        steps=expected["steps"];assumed=expected["player_actions"]+["WAIT"]*(steps-len(expected["player_actions"]))
        if not 1<=steps<=3 or any(a not in ACTIONS for a in assumed):raise ValueError("invalid_frozen_counterfactual_spec")
        transitions=[]
        for action in assumed:
            if env.done:break
            transitions.append(_independent_step(env,runtime,action))
        facts.update(assumed_player_actions=assumed,transitions=transitions)
        if not transitions:
            return ("该帧的回合已经结束，不能继续推进。" if language=="zh" else "The round has already ended at this frame."),facts
        outcomes=[];label=LABELS[language]
        for transition in transitions:
            before=transition["before"]["state"];after=transition["after"]["state"]
            chosen=label[transition["submitted_actions"]["robot_2"]];actual=label[transition["executed_actions"]["robot_2"]]
            delivered=after["total_deliveries"]-before["total_deliveries"]
            sentence=(f"第 {after['frame']} 步队友选择{chosen}，实际执行{actual}，团队新增配送 {delivered} 件" if language=="zh" else
                f"step {after['frame']}: teammate chose {chosen}, physically executed {actual}, and team deliveries increased by {delivered}")
            if after.get("terminated") or after.get("truncated"):sentence+="，回合已结束" if language=="zh" else ", and the round ended"
            outcomes.append(sentence)
        assumptions="、".join(label[a] for a in assumed);origin=snapshot["state"]["frame"]
        text=(f"从第 {origin} 帧开始，假设你的动作依次为：{assumptions}；未指定的后续动作按等待处理。"+"；".join(outcomes)+"。这是同一个 NN 在隔离副本中的结果；真实回合未改变。来源：冻结 NN 与环境模拟。"
            if language=="zh" else f"Starting at frame {origin}, assume your actions are: {assumptions}; unspecified following actions are WAIT. "+"; ".join(outcomes)+". These results use the same NN in an isolated copy; the live round is unchanged. Sources: frozen NN and environment simulation.")
        return text,facts
    if intent=="rules":
        facts.update(move_cost=env.config.move_battery_cost,charge_per_wait=env.config.charge_per_wait)
        # These closed templates are versioned together with the renderer. Any
        # renderer wording change must be independently reviewed here as well.
        return _rule_text(env,language),facts
    if intent=="failure":
        reason=record["after"]["state"].get("terminal_reason");facts["terminal_reason"]=reason
        phrases={"battery_shutdown":("至少一台机器人在充电格以外耗尽了电量","at least one robot exhausted its battery away from the charger"),
            "horizon":("已达到回合步数上限","the turn limit was reached")}
        if reason not in phrases:
            return ("这一帧没有记录环境终局失败。" if language=="zh" else "No environmental terminal failure is recorded at this frame."),facts
        text=(f"第 {frame} 帧结束的直接原因是{phrases[reason][0]}。这说明终局触发条件，不等于证明此前某个动作是唯一原因。来源：该帧终局记录。"
            if language=="zh" else f"At frame {frame}, the round ended because {phrases[reason][1]}. This identifies the terminal trigger, not a proven sole cause among earlier actions. Source: the frame's terminal record.")
        return text,facts
    if not past and env.done:
        return (f"第 {frame} 帧回合已结束，没有下一次神经决策。" if language=="zh" else f"The round ended at frame {frame}; no next neural decision exists."),facts
    obs=env.observations()["robot_2"];logits=runtime.actor.logits(obs[None,:])[0]
    probabilities=np.exp(logits-logits.max());probabilities/=probabilities.sum();chosen=ACTIONS[int(logits.argmax())]
    tree_action,trace=_trace_facts(program,env)
    facts.update(neural_action=chosen,probabilities=probabilities.tolist(),tree_action=tree_action,tree_trace=trace)
    label=LABELS[language];when=f"第 {frame} 步执行前" if past else f"第 {frame} 帧的下一次决策"
    when_en=f"before executing step {frame}" if past else f"for the next decision at frame {frame}"
    alternative=expected.get("alternative_action");mentioned=expected.get("mentioned_action")
    if alternative==chosen or (not alternative and mentioned and mentioned!=chosen):
        if past and record["executed_actions"]["robot_2"]=="WAIT" and chosen!="WAIT":
            return _physical_wait_text(when,when_en,chosen,record,env,language),facts
        return ((f"{when}，神经策略实际选择的是{label[chosen]}。请确认所选帧和问题中的动作。" if language=="zh" else
            f"{when_en.capitalize()}, the neural policy actually chose {label[chosen]}. Please confirm the selected frame and action in your question.")),facts
    comparison=_alternative_text(alternative,chosen,probabilities,logits,env,language) if alternative else ""
    if tree_action!=chosen:
        text=(f"{when}，NN 将{label[chosen]}排在首位。"+comparison+"近似决策树选择了不同动作，因此不能用这个树分支解释本次选择。来源：实际 NN 输出与树的一致性核验。"
            if language=="zh" else f"{when_en.capitalize()}, the NN ranked {label[chosen]} first. "+comparison+"The approximate tree selected a different action, so its branch cannot explain this choice. Sources: actual NN output and tree-agreement check.")
        return text,facts
    conditions=_conditions(trace,env,language);condition_text=("；" if language=="zh" else "; ").join(conditions)
    if language=="zh":
        prefix=f"{when}，NN 将{label[chosen]}排在首位，近似树的动作与它一致。"
        details=f"该树分支检查的部分条件为：{condition_text}。" if conditions else "该分支的条件暂未完成文字化呈现。"
        text=prefix+comparison+details+"这些条件是策略近似证据，不能证明 NN 的内部动机或唯一原因，也不代表队友承诺了后续路线。来源：实际 NN 输出与抽取的近似树。"
    else:
        prefix=f"{when_en.capitalize()}, the NN ranked {label[chosen]} first, matching the approximate tree. "
        details=f"Some conditions checked on this branch were: {condition_text}. " if conditions else "This branch's conditions do not yet have a text rendering. "
        text=prefix+comparison+details+"These are approximate policy evidence, not proof of the NN's internal motive or sole cause, or a commitment to a future route. Sources: actual NN output and extracted approximate tree."
    return text,facts


def _rule_text(env,language):
    return (f"成功移动一格时，电量按 {env.config.move_battery_cost:g} 个百分点扣减，最低为 0%；剩余电量不足该数值也可以尝试移动。在充电格以外耗尽电量会断电并结束回合；恰好以 0% 到达充电格不会断电，之后实际停留一回合最多补充 {env.config.charge_per_wait:g} 个百分点，上限为 100%。撞墙或机器人冲突会阻止移动，仍消耗一个回合；这些结果不会被改选成另一个方向。来源：公开环境规则。"
        if language=="zh" else f"A successful one-cell move reduces battery by {env.config.move_battery_cost:g} percentage points, with a floor of 0%; movement can still be attempted with less charge. Exhausting battery away from the charger causes shutdown and ends the round. Arriving at the charger with exactly 0% does not cause shutdown; subsequently staying there restores up to {env.config.charge_per_wait:g} percentage points per turn, capped at 100%. Walls and robot conflicts prevent movement but still consume a turn; no alternative direction is substituted. Source: public environment rules.")


def _physical_wait_text(when,when_en,action,record,env,language):
    from env.warehouse.navigation import MOVE_DELTAS
    r,c=env.state.by_id("robot_2").position;dr,dc=MOVE_DELTAS[action]
    wall=not env.layout.is_passable((r+dr,c+dc))
    collision=record["after"]["state"]["robot_collision_events"]>record["before"]["state"]["robot_collision_events"]
    reason=("目标格为墙或边界" if wall else "发生机器人冲突" if collision else "移动被环境阻止") if language=="zh" else (
        "the target was a wall or boundary" if wall else "a robot conflict occurred" if collision else "the environment prevented movement")
    return (f"{when}，NN 选择{LABELS[language][action]}，但实际执行等待：{reason}。不能把物理停留解释成 NN 主动等待。来源：神经记录与物理结果。" if language=="zh" else
        f"{when_en.capitalize()}, the NN chose {LABELS[language][action]}, but physically waited because {reason}. Physical waiting is not evidence that the NN chose WAIT. Sources: neural record and physical outcome.")


def _program(path,runtime):
    payload=json.loads(Path(path).read_text())
    source=payload['program'] if payload.get('version')=='warehouse_native_rcpd_feedback_v1' else payload
    program=ExecutableProgram.from_dict(source)
    if program.metadata.get('native_source_actor_sha256')!=runtime.actor.sha256:raise ValueError('behavior_program_actor_mismatch')
    env=NativeWarehouseEnv()
    if set(program.feature_names)!=set(env.feature_names) or tuple(program.action_names)!=ACTIONS or program.root.depth()>8 or program.root.leaf_count()>64:
        raise ValueError('behavior_program_contract_mismatch')
    stack=[program.root]
    while stack:
        node=stack.pop()
        if node.is_leaf:
            p=np.asarray(node.probabilities,dtype=float)
            if p.shape!=(5,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('behavior_invalid_leaf')
        else:
            if node.feature not in env.feature_names or not np.isfinite(node.threshold):raise ValueError('behavior_invalid_predicate')
            stack.extend([node.left,node.right])
    return program


def _parsed_binding(spec,request):
    """Parser output is checked against the predeclared semantic expectation.

    It is NOT used to construct expected text; a parser error cannot choose its
    own expected intent, frame or counterfactual actions here.
    """
    parsed=renderer.parse_question(request['question'],request.get('focus'))
    expected=spec['expected']
    for key in ('intent','focus','alternative_action','mentioned_action','player_actions','steps'):
        if key in expected and parsed.get(key)!=expected[key]:raise ValueError('behavior_parser_semantics_mismatch:'+key)
    if expected['intent']=='clarify' and parsed.get('reason')!=expected['clarification_reason']:
        raise ValueError('behavior_parser_clarification_mismatch')
    return parsed


def _initial_scene(scene,configuration):
    env=NativeWarehouseEnv();env.restore(scene['snapshot'])
    from dataclasses import asdict
    if env.state.frame!=0 or scenario_fingerprint(env)!=scene['fingerprint'] or asdict(env.config)!=configuration:
        raise ValueError('behavior_source_initial_state_mismatch')
    return env


def _frozen_weights(runtime):
    fresh=NumPyNativeActor(runtime.actor.path)
    if fresh.sha256!=runtime.actor.sha256 or fresh.metadata!=runtime.actor.metadata or any(not np.array_equal(fresh.weights[name],runtime.actor.weights[name]) for name in fresh.weights):
        raise ValueError('behavior_actor_changed_or_memory_modified')
    if NativeRuntime(runtime.actor.path).signature!=runtime.signature:raise ValueError('behavior_runtime_changed')


def verify_answer_report(report,program_path,runtime,scenarios,*,allow_test_fixture=False):
    """Recompute complete fixed-case evidence; never trust per-case pass flags.

    Raises on malformed global provenance. Case failures are returned explicitly.
    Test actors require an explicit Python-only argument and never qualify a
    production release. The default rejects every test-fixture input.
    """
    _strict_json(report);sources=answer_verification_sources();specs=case_specs()
    fixture=runtime.actor.metadata.get('test_fixture') is True
    if report.get('test_fixture',False) is not fixture or fixture and not allow_test_fixture:
        raise ValueError('behavior_fixture_cannot_qualify_production')
    expected_header={'version':VERSION,'renderer_version':renderer.EXPLANATION_VERSION,
        'actor_sha256':runtime.actor.sha256,'program_sha256':file_hash(program_path),'runtime_signature':runtime.signature,
        'scenario_manifest_sha256':digest(scenarios),'case_spec_version':CASE_SPEC_VERSION,'case_spec_sha256':digest(specs),
        'trajectory_spec_sha256':digest(trajectory_spec()),'sources':sources}
    for key,value in expected_header.items():_equal(report[key],value,'behavior_header_mismatch:'+key)
    _frozen_weights(runtime);program=_program(program_path,runtime)
    program_hash=file_hash(program_path);memory_hash=digest(program.to_dict());scene_list=scenarios['splits']['explanation_test'][:SCENARIO_COUNT]
    if len(scene_list)!=SCENARIO_COUNT or len({s['id'] for s in scene_list})!=SCENARIO_COUNT or any(not s['id'].startswith('explanation_test_') for s in scene_list):
        raise ValueError('behavior_exact_scene_pool_missing')
    initial_ids={s['id']:s for s in scene_list}
    if len({s['fingerprint'] for s in scene_list})!=SCENARIO_COUNT:raise ValueError('behavior_duplicate_initial_scenes')
    expected_ids={s['id']+'/'+spec['id'] for s in scene_list for spec in specs}
    cases=report['cases']
    if not isinstance(cases,list) or len(cases)!=len(expected_ids) or {c['case_id'] for c in cases}!=expected_ids:
        raise ValueError('behavior_frozen_case_matrix_incomplete_or_duplicated')
    mapping={c['case_id']:c for c in cases};by_spec={s['id']:s for s in specs}
    # Each scene has one fixed rollout, shared by all its anchors. Different
    # prefixes cannot be picked per question to improve the audit.
    records={};prefixes={};replay_steps=0
    from env.warehouse_native.partners import partner_action
    for scene_index,scene in enumerate(scene_list):
        terminal=mapping[scene['id']+'/failure_terminal_zh']['player_actions']
        if not isinstance(terminal,list) or not 1<=len(terminal)<=scenarios['configuration']['horizon'] or any(a not in ACTIONS for a in terminal):
            raise ValueError('behavior_invalid_terminal_prefix')
        env=_initial_scene(scene,scenarios['configuration']);frames={0:{'after':env.snapshot()}}
        profile=('skilled','assertive','noisy')[scene_index%3]
        partner_rng=np.random.default_rng(np.random.SeedSequence([98131,scene_index,1]))
        for player in terminal:
            if player!=partner_action(env,'robot_1',profile,partner_rng):raise ValueError('behavior_fixed_player_trajectory_mismatch')
            record=_independent_step(env,runtime,player);replay_steps+=1;frames[env.state.frame]=record
        if not env.done:raise ValueError('behavior_terminal_anchor_is_not_terminal')
        records[scene['id']]=frames;prefixes[scene['id']]=terminal
    from backend.training.warehouse_native_answer_audit import offline_context
    context=offline_context(program_path,runtime,allow_test_fixture=allow_test_fixture)
    outcomes=[];coverage=set();independent_counterfactual_steps=0
    for case in cases:
        try:
            if case.get('missing_case'):raise ValueError('behavior_required_anchor_missing')
            sid=case['scenario_id'];specid=case['case_id'].removeprefix(sid+'/')
            if sid not in initial_ids or specid not in by_spec or case['case_id']!=sid+'/'+specid:raise ValueError('behavior_case_identity_mismatch')
            scene=initial_ids[sid];spec=by_spec[specid];full=prefixes[sid]
            target={'initial':0,'first':1,'third':3,'terminal':len(full)}[spec['anchor']]
            if target not in records[sid]:raise ValueError('behavior_required_anchor_missing')
            _equal(case['player_actions'],full[:target],'behavior_question_prefix_changed')
            _equal(case['initial_fingerprint'],scene['fingerprint'],'behavior_case_initial_binding_mismatch')
            request=request_for_spec(spec,target)
            _equal(case['request'],request,'behavior_question_or_request_changed')
            _equal(case['expected'],spec['expected'],'behavior_self_reported_expected_semantics_changed')
            if (case.get('language')!=spec['language'] or case.get('intent')!=spec['expected']['intent']
                    or case.get('question')!=spec['question']):
                raise ValueError('behavior_self_reported_coverage_changed')
            record=records[sid][target]
            if case['record_sha256']!=digest(record):raise ValueError('behavior_record_does_not_match_real_prefix')
            if not isinstance(case['answer'],str) or case['answer_sha256']!=_text_hash(case['answer']):raise ValueError('behavior_answer_hash_mismatch')
            _parsed_binding(spec,request)
            before_record=digest(record);before_sources=digest(sources)
            expected,facts=expected_answer(spec,record,runtime,program)
            if case['answer']!=expected:raise ValueError('behavior_text_has_wrong_or_unsupported_factual_claim')
            actual_error=None;actual_text=''
            try:actual_text=renderer.NativeExplainer.answer(context,request,deepcopy(record),runtime)
            except ValueError as error:actual_error=str(error)
            intended_error=spec['expected'].get('error')
            if actual_error!=intended_error or case.get('error')!=intended_error:
                raise ValueError('behavior_actual_rejection_differs_from_spec')
            if actual_text!=case['answer']:raise ValueError('behavior_current_renderer_text_changed')
            if digest(record)!=before_record or digest(program.to_dict())!=memory_hash or digest(sources)!=before_sources:
                raise ValueError('behavior_verification_mutated_evidence')
            independent_counterfactual_steps+=len(facts.get('transitions',[]))
            if intended_error is None:coverage.add((spec['language'],spec['expected']['intent']))
            outcomes.append({'case_id':case['case_id'],'passed':True,'frame_binding_passed':True,'evidence_verified':True,
                'language':spec['language'],'intent':spec['expected']['intent'],'selected_frame':target,
                'independent_facts_sha256':digest(facts),'expected_error':intended_error,
                'tree_agreement':facts.get('tree_action')==facts.get('neural_action') if 'tree_action' in facts else None,
                'counterfactual_steps':len(facts.get('transitions',[]))})
        except (ValueError,KeyError,TypeError,IndexError) as error:
            outcomes.append({'case_id':case.get('case_id'),'passed':False,'reason':str(error)})
    required={(lang,intent) for lang in ('zh','en') for intent in ('reason','alternative','counterfactual','failure','rules','clarify')}
    _frozen_weights(runtime)
    if sources!=answer_verification_sources() or file_hash(program_path)!=program_hash:raise ValueError('behavior_sources_or_program_changed_during_verification')
    passed=all(c['passed'] for c in outcomes) and required<=coverage
    return {'version':VERSION,'passed':passed,'test_fixture':fixture,'release_eligible':False,'formal_ready':False,
        'rows':len(cases),'cases':outcomes,'coverage':[{'language':a,'intent':b} for a,b in sorted(coverage)],
        'required_coverage_complete':required<=coverage,'actual_tree_agreement_cases':sum(c.get('tree_agreement') is True for c in outcomes),
        'actual_tree_disagreement_cases':sum(c.get('tree_agreement') is False for c in outcomes),
        'prefix_replay_steps':replay_steps,'independent_counterfactual_steps':independent_counterfactual_steps,
        'renderer_audit_steps':'Historical physical revalidation and counterfactual calls are additional; caller must count actual env.step calls.',
        'neural_updates':0,'scope':'Fixed raw bilingual cases and independent closed text grammar, plus exact renderer replay; not capability or explanation efficacy.'}
