"""Independent observed197 bilingual answer audit, never a release grant.

A private, exact PublicHistoryRuntime retains its real type/source/Actor checks.
Instance-local instrumentation calls the original real env.step/logits methods;
it only meters those operations and is not a participant runtime or global patch.
The oracle uses fixed semantics, raw logits, full197 snapshots and direct physics.
No production renderer/parser/trace helper constructs expected facts or text.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from contextlib import contextmanager
import json, math, re
import numpy as np

from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment, HISTORY_FEATURE_NAMES
from backend.warehouse_public_history_runtime import PublicHistoryRuntime, RUNTIME_VERSION
from backend.warehouse_public_history_explanation import PublicHistoryExplainer, explanation_sources, VERSION as RENDERER_VERSION
from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor
from env.warehouse_native.partners import partner_action
from ui.warehouse_native_answer_verification import (case_specs, request_for_spec, LABELS,
    _clarification, _rule_text, _alternative_text, _physical_wait_text)

VERSION='warehouse-native-observed197-answer-verification.v1'
CASE_SPEC_VERSION='warehouse-native-behavioral-case-spec.v1'
SCENARIO_COUNT=12
TRAJECTORY_SPEC={'version':'warehouse-native-observed197-answer-trajectories.v1',
    'scene_pool':'first_12_explanation_test','profiles':['skilled','assertive','noisy'],
    'profile_assignment':'scene_index_modulo_three','player_role':'robot_1',
    'seed_sequence':[98131,'scene_index',1],'neural_execution':'deterministic',
    'history':'real_confirmed_observed197_only'}


def answer_verification_sources():
    sources=explanation_sources()
    for path in (Path(__file__),ROOT/'ui/warehouse_native_answer_verification.py'):
        sources[str(path.relative_to(ROOT))]=file_hash(path)
    return sources


def input_bindings(runtime,program_path,scenarios):
    """Read-only anchor description; caller must independently approve/bind it."""
    return {'actor_sha256':runtime.actor_sha256,'actor_metadata_sha256':digest(runtime.actor.metadata),
        'program_sha256':file_hash(program_path),'runtime_signature':runtime.signature,
        'protocol_sha256':runtime.protocol_sha256,'scenario_manifest_sha256':digest(scenarios),
        'configuration_sha256':digest(asdict(runtime.config)),
        'case_spec_sha256':digest(case_specs()),'trajectory_spec_sha256':digest(TRAJECTORY_SPEC),
        'sources_sha256':digest(answer_verification_sources())}


class AuditFailure(ValueError):
    """Failure retains actual counts and any unacknowledged operation; never retry."""
    def __init__(self,reason,report):
        super().__init__(reason);self.report=report


class _ExecutionStop(Exception):
    """Accounting stops must not be swallowed by renderer ValueError wrappers."""


class _Meter:
    def __init__(self,budget,before_step,after_step,execution_id):
        self.budget=budget;self.before=before_step;self.after=after_step;self.execution_id=execution_id
        self.phase='setup';self.context={};self.pending=None
        self.counts={'environment_steps':0,'environment_step_attempts':0,'acknowledged_steps':0,
            'numpy_forward_calls':0,'numpy_forward_rows':0,'numpy_actor_load_attempts':0,'numpy_actor_loads':0,
            'environment_constructions':0,'neural_updates':0,'torch_loads':0,
            'steps_by_phase':{},'forwards_by_phase':{}}
    @contextmanager
    def during(self,phase,**context):
        previous=(self.phase,self.context);self.phase,self.context=phase,context
        try:yield
        finally:self.phase,self.context=previous
    def instrument(self,runtime):
        # Only this newly constructed audit runtime receives wrappers. No
        # runtime/env class, validation path, source hash, policy or state edits.
        old_new=runtime._new_environment;old_logits=runtime.actor.logits
        self.counts['environment_constructions']=1
        def logits(observations):
            self.counts['numpy_forward_calls']+=1
            self.counts['numpy_forward_rows']+=len(np.asarray(observations))
            self.counts['forwards_by_phase'][self.phase]=self.counts['forwards_by_phase'].get(self.phase,0)+1
            return old_logits(observations)
        def new_environment():
            env=old_new();self.counts['environment_constructions']+=1
            if type(env) is not PublicFeedbackEnvironment:raise ValueError('audit_requires_real197_environment')
            old_step=env.step
            def step(actions):
                if self.pending is not None:raise _ExecutionStop('audit_unacknowledged_operation_no_retry')
                if self.counts['environment_step_attempts']>=self.budget:raise _ExecutionStop('audit_auxiliary_step_budget_exhausted')
                before=env.snapshot()
                context={'operation_id':f'{self.execution_id}:{self.counts["environment_step_attempts"]}',
                    'phase':self.phase,**self.context,'frame':env.state.frame,
                    'before_sha256':digest(before),'submitted_actions':deepcopy(actions)}
                self.pending={**context,'status':'awaiting_permission'}
                if self.before is not None and self.before(deepcopy(context)) is not True:
                    raise _ExecutionStop('audit_before_step_not_permitted')
                self.pending['status']='reserved';self.counts['environment_step_attempts']+=1
                result=old_step(actions)
                self.counts['environment_steps']+=1
                self.counts['steps_by_phase'][self.phase]=self.counts['steps_by_phase'].get(self.phase,0)+1
                _,rewards,terminated,truncated,info=result
                completed={**context,'status':'executed_unacknowledged','actual_steps':1,
                    'before':before,'after':env.snapshot(),'rewards':deepcopy(rewards),
                    'terminated':bool(terminated),'truncated':bool(truncated),'info':deepcopy(info)}
                self.pending=completed
                if self.after is not None and self.after(deepcopy(completed)) is not True:
                    raise _ExecutionStop('audit_after_step_not_acknowledged')
                self.counts['acknowledged_steps']+=1;self.pending=None
                return result
            env.step=step
            return env
        runtime.actor.logits=logits;runtime._new_environment=new_environment
        runtime.audit_instrumentation_only=True
    def report(self):
        return {'actual_execution':deepcopy(self.counts),'auxiliary_step_budget':self.budget,
            'remaining_step_attempt_budget':self.budget-self.counts['environment_step_attempts'],
            'pending_operation':deepcopy(self.pending),'automatic_retry':False,
            'accounting_complete':self.pending is None}


def _equal(a,b,reason):
    if digest(a)!=digest(b):raise ValueError(reason)


def _preflight(runtime,program_path,scenarios,expected_bindings,allow_test_fixture,
               execution_permitted,step_budget,before_step,after_step,execution_id,meter):
    if execution_permitted is not True:raise ValueError('audit_execution_not_explicitly_permitted')
    if type(step_budget) is not int or not 1<=step_budget<=12000:raise ValueError('audit_requires_finite_auxiliary_budget')
    if type(runtime) is not PublicHistoryRuntime or type(runtime.actor) is not NumPyNativeActor:
        raise ValueError('audit_requires_actual197_runtime_and_numpy_actor')
    if type(allow_test_fixture) is not bool or runtime.test_fixture is not allow_test_fixture:
        raise ValueError('audit_fixture_scope_mismatch')
    if type(execution_id) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{3,80}',execution_id):
        raise ValueError('audit_execution_id_required')
    if (before_step is None)!=(after_step is None) or (before_step is not None and (not callable(before_step) or not callable(after_step))):
        raise ValueError('audit_requires_both_step_callbacks')
    if not allow_test_fixture and (before_step is None or after_step is None):
        raise ValueError('production_audit_requires_durable_external_step_accounting')
    runtime.verify_binding()
    if not isinstance(expected_bindings,dict) or expected_bindings!=input_bindings(runtime,program_path,scenarios):
        raise ValueError('audit_external_input_binding_mismatch')
    if scenarios.get('test_fixture',False) is not allow_test_fixture:
        raise ValueError('audit_scenario_fixture_scope_mismatch')
    _equal(scenarios['configuration'],asdict(runtime.config),'audit_scene_configuration_mismatch')
    scenes=scenarios['splits']['explanation_test'][:SCENARIO_COUNT]
    if not scenes or (not allow_test_fixture and len(scenes)!=SCENARIO_COUNT):
        raise ValueError('audit_exact_explanation_scene_pool_required')
    if len({s['id'] for s in scenes})!=len(scenes) or len({s['fingerprint'] for s in scenes})!=len(scenes):
        raise ValueError('audit_duplicate_initial_scenes')
    if any(not s['id'].startswith('explanation_test_') for s in scenes):
        raise ValueError('audit_wrong_scene_pool')
    if any(s['snapshot']['state']['frame']!=0 for s in scenes):raise ValueError('audit_requires_initial_scenes')
    sources=answer_verification_sources()
    meter.counts['numpy_actor_load_attempts']+=1
    fresh=PublicHistoryRuntime(runtime.actor.path,protocol=runtime.protocol,
        expected_actor_sha256=runtime.actor_sha256,expected_protocol_sha256=runtime.protocol_sha256,
        expected_bindings=deepcopy(runtime.actor.metadata),allow_test_fixture=allow_test_fixture,config=runtime.config)
    meter.counts['numpy_actor_loads']+=1
    if fresh.signature!=runtime.signature:raise ValueError('audit_fresh_runtime_differs')
    meter.instrument(fresh)
    explainer=PublicHistoryExplainer(program_path,expected_program_sha256=expected_bindings['program_sha256'],
        runtime=fresh,allow_test_fixture=allow_test_fixture)
    # Oracle program is decoded independently from the renderer object.
    raw=json.loads(Path(program_path).read_text())
    program=ExecutableProgram.from_dict(raw['program'] if raw.get('version')=='warehouse_native_rcpd_feedback_v1' else raw)
    _equal(program.to_dict(),explainer.program.to_dict(),'audit_program_decode_mismatch')
    return fresh,explainer,program,deepcopy(scenes),sources,meter


def _independent_step(env,runtime,player):
    """Raw batch-two logits and actual public-history physics, not runtime.step."""
    if env.done:raise ValueError('audit_prefix_after_terminal')
    before=env.snapshot();observations=env.observations();roles=sorted(env.agent_ids)
    matrix=np.stack([observations[role] for role in roles]);logits=runtime.actor.logits(matrix)
    probabilities=np.exp(logits-logits.max(axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
    actions={role:ACTIONS[int(logits[i].argmax())] for i,role in enumerate(roles)}
    decision={'version':RUNTIME_VERSION,'runtime_signature':runtime.signature,'actor_sha256':runtime.actor_sha256,
        'protocol_sha256':runtime.protocol_sha256,'frame':env.state.frame,'policy_actions':deepcopy(actions),
        'proposed_actions':deepcopy(actions),'probabilities':{role:probabilities[i].tolist() for i,role in enumerate(roles)},
        'observation_hashes':{role:sha256(np.asarray(observations[role],np.float32).tobytes()).hexdigest() for role in roles},
        'masks':False,'post_policy_overrides':0,'robot_1_policy_is_not_participant_input':True}
    submitted={'robot_1':player,'robot_2':actions['robot_2']}
    _,rewards,terminated,truncated,info=env.step(submitted)
    if info['requested_actions']!=submitted:raise ValueError('audit_physical_input_changed')
    return {'before':before,'after':env.snapshot(),'decision':decision,'policy_actions':deepcopy(actions),
        'proposed_actions':deepcopy(actions),'participant_action':player,'submitted_actions':deepcopy(submitted),
        'executed_actions':deepcopy(info['executed_actions']),'physical_actions':deepcopy(info['executed_actions']),
        'events':deepcopy(info['events']),'rewards':rewards,'info':info,'done':bool(terminated or truncated),
        'runtime_signature':runtime.signature}


def _trace_facts(program,env):
    features=dict(zip(env.feature_names,map(float,env.observations()['robot_2'])))
    node=program.root;trace=[]
    while not node.is_leaf:
        value=features[node.feature];left=value<=node.threshold
        trace.append({'feature':node.feature,'value':value,'threshold':node.threshold,'left':bool(left)})
        node=node.left if left else node.right
    return ACTIONS[int(np.asarray(node.probabilities).argmax())],trace


def count_cutoff(threshold,horizon):
    """Independent discrete float32 codebook, not inverse/floor or renderer helper."""
    if type(horizon) is not int or not 1<=horizon<=120 or not np.isfinite(threshold):
        raise ValueError('audit_invalid_count_boundary')
    return max((n for n in range(horizon+1) if float(np.float32(np.log1p(float(n))/np.log1p(float(horizon))))<=threshold),default=-1)


def _history_condition(item,env,language):
    feature,value,threshold,left=item['feature'],item['value'],item['threshold'],item['left']
    history=env.public_history();zh=language=='zh';rel='≤' if left else '>'
    if feature not in HISTORY_FEATURE_NAMES:raise ValueError('audit_unknown_history_predicate')
    if feature=='history.valid':
        status=('已知' if history['valid'] else '未知') if zh else ('known' if history['valid'] else 'unknown')
        return (f'上一已确认回合历史为{status}，已知指示值 {rel} {threshold:.6g}（实际 {value:g}）' if zh else
            f'the preceding confirmed transition is {status}; history-valid indicator {rel} {threshold:.6g} (actual {value:g})')
    if not history['valid']:
        return ('上一回合历史未知；此历史特征的零编码不能证明此前没有该动作或冲突' if zh else
            'previous-transition history is unknown; its zero encoding does not prove the absence of an action or conflict')
    counts={'history.self.consecutive_move_canceled':('队友连续移动被取消次数','teammate consecutive canceled moves',history['consecutive_move_canceled']['robot_2']),
        'history.other.consecutive_move_canceled':('玩家连续移动被取消次数','player consecutive canceled moves',history['consecutive_move_canceled']['robot_1']),
        'history.joint.consecutive_collision':('双方连续冲突次数','consecutive joint conflicts',history['consecutive_collision'])}
    if feature in counts:
        cn,en,count=counts[feature]
        if value!=float(np.float32(np.log1p(float(count))/np.log1p(float(env.config.horizon)))):
            raise ValueError('audit_history_count_encoding_mismatch')
        cutoff=count_cutoff(threshold,env.config.horizon)
        return (f'{cn} {rel} {cutoff}（实际 {count} 次；按 log1p 计数编码核验）' if zh else
            f'{en} {rel} {cutoff} (actual {count}; checked against log1p count encoding)')
    parts=feature.split('.')
    if len(parts)==4 and parts[2]=='submitted':
        who=('队友' if parts[1]=='self' else '玩家') if zh else ('teammate' if parts[1]=='self' else 'player')
        name=f'{who}上一回合提交{LABELS[language][parts[3]]}的指示值' if zh else f'indicator that the {who} submitted {LABELS[language][parts[3]]} on the preceding turn'
    elif feature in ('history.self.move_canceled','history.other.move_canceled'):
        who=('队友' if parts[1]=='self' else '玩家') if zh else ('teammate' if parts[1]=='self' else 'player')
        name=f'{who}上一回合移动被取消的指示值' if zh else f'indicator of the {who} preceding move being canceled'
    else:
        kinds={'none':('无机器人冲突','no robot conflict'),'same_target':('争用同一目标格','same-target conflict'),
            'swap':('交换位置冲突','swap conflict'),'occupied_stationary':('目标格被停留者占用','stationary-occupant conflict')}
        label=kinds[parts[2]][0 if zh else 1]
        name=f'上一回合{label}的指示值' if zh else f'indicator of {label} on the preceding turn'
    return f'{name} {rel} {threshold:.6g}（实际 {value:g}）' if zh else f'{name} {rel} {threshold:.6g} (actual {value:g})'


def _conditions(trace,env,language):
    from ui.warehouse_native_answer_verification import _conditions as physical_conditions
    output=[]
    for item in trace:
        if item['feature'].startswith('history.'):
            output.append(_history_condition(item,env,language))
        else:output.extend(physical_conditions([item],env,language))
    return output[:3]


def expected_answer(spec,record,runtime,program):
    """Return complete permitted text and private facts; no renderer is called."""
    expected=spec["expected"];language=spec["language"];frame=record["after"]["state"]["frame"]
    if "error" in expected:return "",{"expected_error":expected["error"]}
    if "clarification_reason" in expected:
        return _clarification(expected["clarification_reason"],frame,language),{"clarification_reason":expected["clarification_reason"]}
    intent=expected["intent"];past=expected.get("focus")=="executed"
    if past and "before" not in record:raise ValueError("missing_historical_record")
    snapshot=record["before"] if past else record["after"]
    env=runtime.from_snapshot(snapshot)
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
    obs=env.observations()["robot_2"]; matrix=np.stack([env.observations()[role] for role in sorted(env.agent_ids)])
    logits=runtime.actor.logits(matrix)[sorted(env.agent_ids).index("robot_2")]
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



def _header(runtime,program_path,scenarios,sources,scene_count):
    return {'version':VERSION,'renderer_version':RENDERER_VERSION,'test_fixture':runtime.test_fixture,
        'bindings':input_bindings(runtime,program_path,scenarios),'sources':sources,
        'case_spec_version':CASE_SPEC_VERSION,'case_specs':case_specs(),'trajectory_spec':TRAJECTORY_SPEC,
        'scene_count':scene_count,'required_production_scene_count':SCENARIO_COUNT,
        'full_production_matrix_present':scene_count==SCENARIO_COUNT and not runtime.test_fixture,
        'configuration':asdict(runtime.config)}


def _rollouts(runtime,scenes,meter):
    records={};prefixes={};traces={}
    for i,scene in enumerate(scenes):
        sid=scene['id'];profile=('skilled','assertive','noisy')[i%3]
        with meter.during('fixed_prefix',scenario_id=sid,profile=profile):
            env=runtime.environment(scene)
            frames={0:{'after':env.snapshot(),'runtime_signature':runtime.signature}}
            rng=np.random.default_rng(np.random.SeedSequence([98131,i,1]));actions=[];trajectory=[]
            while not env.done:
                # Program sees public pre-state, before the NN output is computed.
                player=partner_action(env,'robot_1',profile,rng)
                record=_independent_step(env,runtime,player)
                actions.append(player);frames[env.state.frame]=record;trajectory.append(record)
            records[sid]=frames;prefixes[sid]=actions
            traces[sid]={'initial_fingerprint':scene['fingerprint'],'profile':profile,
                'seed_sequence':[98131,i,1],'player_actions':actions,'transitions':trajectory}
    return records,prefixes,traces


def _render(explainer,runtime,request,record,meter,identity):
    with meter.during('renderer',case_id=identity):
        try:return explainer.answer(deepcopy(request),deepcopy(record),runtime),None
        except ValueError as error:
            if meter.pending is not None or 'budget_exhausted' in str(error):raise
            return '',str(error)


def _oracle(spec,record,runtime,program,meter,identity):
    # The fixed semantics, not the production parser's output, choose the oracle.
    with meter.during('independent_oracle',case_id=identity):
        return expected_answer(spec,record,runtime,program)


def _finish(runtime,original_runtime,program_path,program,program_digest,sources,bindings,scenarios,meter):
    runtime.verify_binding();original_runtime.verify_binding()
    _equal(program.to_dict(),program_digest,'audit_oracle_program_mutated')
    _equal(answer_verification_sources(),sources,'audit_source_changed_during_execution')
    _equal(input_bindings(original_runtime,program_path,scenarios),bindings,'audit_input_changed_during_execution')
    if meter.pending is not None:raise ValueError('audit_pending_operation_no_qualification')


def _execute(operation,report,runtime,program_path,scenarios,*,expected_bindings,step_budget,
             execution_permitted=False,execution_id=None,allow_test_fixture=False,before_step=None,after_step=None):
    meter=_Meter(step_budget if type(step_budget) is int else 0,before_step,after_step,execution_id)
    try:
        rt,explainer,program,scenes,sources,meter=_preflight(runtime,program_path,scenarios,expected_bindings,
            allow_test_fixture,execution_permitted,step_budget,before_step,after_step,execution_id,meter)
        header=_header(rt,program_path,scenarios,sources,len(scenes));program_digest=deepcopy(program.to_dict())
        specs=case_specs();expected_ids={scene['id']+'/'+spec['id'] for scene in scenes for spec in specs}
        if operation=='verify':
            json.dumps(report,allow_nan=False)
            for key,value in header.items():_equal(report[key],value,'audit_header_mismatch:'+key)
            _equal(report['scenario_inputs'],scenes,'audit_scene_input_mismatch')
            cases=report['cases']
            if not isinstance(cases,list) or len(cases)!=len(expected_ids) or {c['case_id'] for c in cases}!=expected_ids:
                raise ValueError('audit_frozen_case_matrix_incomplete_or_duplicated')
            mapping={c['case_id']:c for c in cases}
        records,prefixes,traces=_rollouts(rt,scenes,meter)
        if operation=='verify':_equal(report['trajectories'],traces,'audit_raw_trajectory_differs_from_true197_replay')
        generated=[];outcomes=[];coverage=set();predicate_features=set()
        for scene in scenes:
            sid=scene['id'];full=prefixes[sid]
            for spec in specs:
                identity=sid+'/'+spec['id']
                target={'initial':0,'first':1,'third':3,'terminal':len(full)}[spec['anchor']]
                if target not in records[sid]:
                    missing={'case_id':identity,'scenario_id':sid,'missing_case':True,'anchor':spec['anchor']}
                    generated.append(missing);outcomes.append({**missing,'passed':False,'reason':'audit_required_anchor_missing'})
                    continue
                record=records[sid][target];record_before=digest(record)
                request=request_for_spec(spec,target)
                base={'case_id':identity,'scenario_id':sid,'initial_fingerprint':scene['fingerprint'],
                    'player_actions':full[:target],'request':request,'expected':spec['expected'],
                    'language':spec['language'],'intent':spec['expected']['intent'],'question':spec['question'],
                    'record_sha256':record_before}
                if operation=='generate':
                    text,error=_render(explainer,rt,request,record,meter,identity)
                    generated.append({**base,'answer':text,'answer_sha256':sha256(text.encode()).hexdigest(),'error':error})
                    continue
                try:
                    supplied=mapping[identity]
                    for key,value in base.items():_equal(supplied[key],value,'audit_case_binding_mismatch:'+key)
                    if type(supplied['answer']) is not str or sha256(supplied['answer'].encode()).hexdigest()!=supplied['answer_sha256']:
                        raise ValueError('audit_answer_hash_mismatch')
                    expected,facts=_oracle(spec,record,rt,program,meter,identity)
                    if supplied['answer']!=expected:raise ValueError('audit_wrong_or_unsupported_answer_fact')
                    text,error=_render(explainer,rt,request,record,meter,identity)
                    if supplied.get('error')!=spec['expected'].get('error') or error!=spec['expected'].get('error'):
                        raise ValueError('audit_actual_rejection_differs_from_fixed_spec')
                    if supplied['answer']!=text:raise ValueError('audit_current_renderer_differs_from_saved_answer')
                    # Check parser semantics against fixed expectations without
                    # allowing them to select the oracle or its output.
                    from ui.warehouse_native_answer_verification import _parsed_binding
                    _parsed_binding(spec,request)
                    if digest(record)!=record_before:raise ValueError('audit_record_mutated')
                    predicate_features.update(t['feature'] for t in facts.get('tree_trace',[]))
                    if 'error' not in spec['expected']:coverage.add((spec['language'],spec['expected']['intent']))
                    outcomes.append({'case_id':identity,'passed':True,'selected_frame':target,
                        'independent_facts':facts,'independent_facts_sha256':digest(facts),
                        'tree_agreement':facts['tree_action']==facts['neural_action'] if 'tree_action' in facts else None,
                        'expected_error':error,'counterfactual_steps':len(facts.get('transitions',[]))})
                except (ValueError,KeyError,TypeError,IndexError) as error:
                    if meter.pending is not None or 'budget_exhausted' in str(error):raise
                    outcomes.append({'case_id':identity,'passed':False,'reason':str(error)})
        _finish(rt,runtime,program_path,program,program_digest,sources,expected_bindings,scenarios,meter)
        common={'release_eligible':False,'formal_ready':False,'research_qualification_evaluated':False,
            'audit_runtime_only':True,'execution':meter.report(),
            'scope':'Observed197 raw bilingual evidence audit only; no training, policy qualification, intervention-fidelity or explanation-effect claim.'}
        if operation=='generate':
            return {**header,**common,'scenario_inputs':scenes,'trajectories':traces,'cases':generated,
                'independently_verified':False,'passed':False,'missing_anchors':sum(bool(c.get('missing_case')) for c in generated)}
        required={(lang,intent) for lang in ('zh','en') for intent in ('reason','alternative','counterfactual','failure','rules','clarify')}
        passed=all(c['passed'] for c in outcomes) and required<=coverage
        return {**header,**common,'passed':passed,'input_report_sha256':digest(report),
            'rows':len(outcomes),'cases':outcomes,'coverage':[{'language':a,'intent':b} for a,b in sorted(coverage)],
            'required_coverage_complete':required<=coverage,'actual_tree_agreement_cases':sum(c.get('tree_agreement') is True for c in outcomes),
            'actual_tree_disagreement_cases':sum(c.get('tree_agreement') is False for c in outcomes),
            'actual_predicate_features':sorted(predicate_features),
            'uncovered_history_predicates':sorted(set(HISTORY_FEATURE_NAMES)-predicate_features),
            'history_predicate_full_coverage':set(HISTORY_FEATURE_NAMES)<=predicate_features,
            'unverified':['independent_RCPD_action_fidelity','intervention_direction_fidelity','human_explanation_effect','release_qualification'],
            'source_hashes_verified_after_execution':True}
    except Exception as error:
        if isinstance(error,AuditFailure):raise
        accounting=meter.report() if meter else {'actual_execution':{'environment_steps':0,'numpy_forward_calls':0},'pending_operation':None}
        raise AuditFailure(str(error),{'version':VERSION,'passed':False,'release_eligible':False,
            'reason':str(error),'execution':accounting,'automatic_retry':False}) from error
    except BaseException as interruption:
        # Preserve Ctrl-C/SystemExit, but do not discard reservations or a step
        # already completed before its durable acknowledgement callback failed.
        interruption.audit_report={'version':VERSION,'passed':False,'release_eligible':False,
            'reason':type(interruption).__name__,'execution':meter.report(),'automatic_retry':False}
        raise


def generate_answer_report(runtime,program_path,scenarios,*,expected_bindings,step_budget,
                           execution_permitted=False,execution_id=None,allow_test_fixture=False,before_step=None,after_step=None):
    """Produce all fixed raw cases; this does not declare them independently valid."""
    return _execute('generate',None,runtime,program_path,scenarios,expected_bindings=expected_bindings,
        step_budget=step_budget,execution_permitted=execution_permitted,execution_id=execution_id,
        allow_test_fixture=allow_test_fixture,before_step=before_step,after_step=after_step)


def verify_answer_report(report,program_path,runtime,scenarios,*,expected_bindings,step_budget,
                         execution_permitted=False,execution_id=None,allow_test_fixture=False,before_step=None,after_step=None):
    """Independent fixed-prefix, NN,197history and <=3-step physical fact replay.

    Callers must durably reserve before each step and acknowledge afterwards in
    production. Any exception leaves its explicit pending/count report; no
    automatic retry or resume is provided. Old177 snapshots are never restored.
    """
    return _execute('verify',report,runtime,program_path,scenarios,expected_bindings=expected_bindings,
        step_budget=step_budget,execution_permitted=execution_permitted,execution_id=execution_id,
        allow_test_fixture=allow_test_fixture,before_step=before_step,after_step=after_step)
