"""Genuine diverse Alignment answers: frozen 40x12 cases, independent oracle and ACKs.

No new question selection, LLM, policy training or tree fitting. The immutable
legacy question specifications, pure text-oracle rules, Meter and linear
reservation/ACK implementation are reused as their actual versions. Runtime and
renderer admission are explicit new types; no metadata or registry is recast.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse
import json
import numpy as np

from backend import warehouse_alignment_runtime as runtime_api
from backend import warehouse_alignment_diverse_explanation as renderer
from backend.training import warehouse_family_explanation_system_run as system
from backend.training import warehouse_family_answer_run as journal
from backend.training.warehouse_native_common import ROOT,digest,file_hash
from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS
from env.warehouse_native.partners import partner_action
from ui import warehouse_family_answer_verification as inherited
from ui.warehouse_native_answer_verification import (case_specs,request_for_spec,LABELS,
    _clarification,_rule_text,_alternative_text,_physical_wait_text,_parsed_binding)

VERSION='warehouse-alignment-diverse-free-question-audit.v1'
CASE_SPEC_VERSION=inherited.CASE_SPEC_VERSION
SCENARIO_COUNT=12
PHASE_CAP=12000
TRAJECTORY_SPEC=deepcopy(inherited.TRAJECTORY_SPEC)
_Meter=inherited._Meter
_trace_facts,_conditions=inherited._trace_facts,inherited._conditions
_read,_write=journal._read,journal._write


def sources():
    result=renderer.explanation_sources()
    result.update(inherited.answer_verification_sources())
    for path in (Path(__file__),Path(journal.__file__)):
        result[str(path.relative_to(ROOT))]=file_hash(path)
    return result


def bindings(runtime,program_path,scenarios):
    return {'actor_sha256':runtime.actor_sha256,'runtime_signature':runtime.signature,
        'program_sha256':file_hash(program_path),'protocol_sha256':runtime.protocol_sha256,
        'scenario_manifest_sha256':digest(scenarios),'case_spec_sha256':digest(case_specs()),
        'trajectory_spec_sha256':digest(TRAJECTORY_SPEC),'sources_sha256':digest(sources())}



def _runtime_choices(logits):
    """Independent arithmetic with exactly the deployed float32 tie semantics."""
    probabilities=np.exp(logits-logits.max(axis=-1,keepdims=True))
    probabilities/=probabilities.sum(axis=-1,keepdims=True)
    return probabilities,np.argmax(probabilities,axis=-1)

def _independent_step(env,runtime,player):
    """Raw batch-two logits and actual public-history physics, not runtime.step."""
    if env.done:raise ValueError('audit_prefix_after_terminal')
    before=env.snapshot();observations=env.observations();roles=sorted(env.agent_ids)
    matrix=np.stack([observations[role] for role in roles]);logits=runtime.actor.logits(matrix)
    probabilities,indices=_runtime_choices(logits)
    actions={role:ACTIONS[int(indices[i])] for i,role in enumerate(roles)}
    decision={'version':runtime.contract_report['version'],'runtime_signature':runtime.signature,'actor_sha256':runtime.actor_sha256,
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
    probabilities,index=_runtime_choices(logits);chosen=ACTIONS[int(index)]
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


def _phase(operation,rt,explainer,scenes,meter,generated=None):
    """Explicitly injectable execution kernel; actual public run admits Alignment."""
    # The original oracle uses this module's independent raw inference function.
    program=ExecutableProgram.from_dict(deepcopy(explainer.program.to_dict()))
    records,prefixes,traces=_rollouts(rt,scenes,meter)
    if generated is not None and digest(generated['trajectories'])!=digest(traces): raise ValueError('Independent full trajectories differ')
    generated_map={} if generated is None else {c['case_id']:c for c in generated['cases']}
    rows=[];coverage=set();specs=case_specs()
    for scene in scenes:
        sid=scene['id'];full=prefixes[sid]
        live_record=records[sid][max(records[sid])]
        live=rt.from_snapshot(live_record['after']);live_hash=digest(live.snapshot());rng_hash=digest(live.get_rng_state())
        context=rt.verified_context(live) if hasattr(rt,'verified_context') else nullcontext()
        with context:
            for spec in specs:
                identity=sid+'/'+spec['id'];target={'initial':0,'first':1,'third':3,'terminal':len(full)}[spec['anchor']]
                if target not in records[sid]:
                    rows.append({'case_id':identity,'passed':False,'missing_anchor':spec['anchor']});continue
                record=records[sid][target];before=digest(record);request=request_for_spec(spec,target)
                base={'case_id':identity,'scenario_id':sid,'initial_fingerprint':scene['fingerprint'],
                    'selected_frame':target,'language':spec['language'],'request':request,'expected':spec['expected'],
                    'record_sha256':before,'question':spec['question']}
                text,error=inherited._render(explainer,rt,request,record,meter,identity)
                if operation=='generate':
                    rows.append({**base,'answer':text,'answer_sha256':sha256(text.encode()).hexdigest(),'error':error});continue
                saved=generated_map[identity]
                with meter.during('independent_oracle',case_id=identity):
                    expected,facts=expected_answer(spec,record,rt,program)
                _parsed_binding(spec,request)
                binding_ok=all(digest(saved.get(k))==digest(v) for k,v in base.items())
                disagreement=facts.get('tree_action') is not None and facts.get('tree_action')!=facts.get('neural_action')
                # Some mistaken-action questions correctly clarify first. Only
                # answers that actually assert a tree reason require disclosure.
                phrase='不能用这个树分支解释本次选择' if spec['language']=='zh' else 'its branch cannot explain this choice'
                required=phrase in expected
                passed=(binding_ok and saved.get('answer')==text==expected
                    and saved.get('answer_sha256')==sha256(text.encode()).hexdigest()
                    and saved.get('error')==error==spec['expected'].get('error')
                    and digest(record)==before and (not required or phrase in text))
                if 'error' not in spec['expected']:coverage.add((spec['language'],spec['expected']['intent']))
                rows.append({**base,'passed':passed,'answer':text,'answer_sha256':sha256(text.encode()).hexdigest(),
                    'independent_answer':expected,'independent_facts':facts,'independent_facts_sha256':digest(facts),
                    'error':error,'tree_disagreement':disagreement,'mismatch_disclosure_required':required,
                    'mismatch_disclosure_passed':not required or phrase in text})
            if digest(live.snapshot())!=live_hash or digest(live.get_rng_state())!=rng_hash: raise ValueError('Answer evaluation mutated live state/RNG')
    required={(l,i) for l in ('zh','en') for i in ('reason','alternative','counterfactual','failure','rules','clarify')}
    return {'version':VERSION,'operation':operation,'cases':rows,'trajectories':traces,
        'passed':operation=='verify' and all(r['passed'] for r in rows) and required<=coverage,
        'coverage':[list(v) for v in sorted(coverage)],'required_coverage_complete':required<=coverage,
        'actual_tree_disagreement_cases':sum(bool(r.get('tree_disagreement')) for r in rows),
        'mismatch_disclosure_cases':sum(bool(r.get('mismatch_disclosure_required')) for r in rows),
        'execution':meter.report(),'independently_verified':operation=='verify',
        'research_qualification_evaluated':False,'release_ready':False}


def _bound_inputs(system_output,system_report_sha,system_plan_sha,scenarios_path,scenarios_sha):
    system_output=Path(system_output).resolve()
    verified=system.read_result(system_output,expected_report_sha256=system_report_sha,expected_plan_sha256=system_plan_sha)
    if verified['system_qualified'] is not True: raise ValueError('Qualified system evidence required before answer execution')
    p=system._json(system_output/'plan.json',system_plan_sha)
    scene_raw=system._read(scenarios_path,scenarios_sha);scenarios=json.loads(scene_raw)
    source=p['collection_plan']['input_files']
    if (digest(scenarios)!=p['collection_plan']['scenario_manifest_sha256']
            if 'scenario_manifest_sha256' in p['collection_plan'] else
            file_hash(scenarios_path)!=source['scenarios']['sha256']):
        raise ValueError('Original frozen scene manifest differs')
    return p,verified,scenarios


def run(system_output,scenarios_path,output,*,expected_system_report_sha256,expected_system_plan_sha256,
        expected_scenarios_sha256):
    """Generate and independently replay the actual480 question matrix, once."""
    root=Path(output).expanduser().resolve();system_output=Path(system_output).resolve()
    if root.exists() or root.is_relative_to(system_output) or system_output.is_relative_to(root): raise ValueError('New independent answer output required')
    p,qualified,scenarios=_bound_inputs(system_output,expected_system_report_sha256,expected_system_plan_sha256,
        scenarios_path,expected_scenarios_sha256)
    source=p['collection_plan']['input_files'];protocol=system._json(source['protocol']['path'],source['protocol']['sha256'])
    program_path=system_output/'system_qualified_program.json';program_sha=qualified['receipt']['qualified_program_file_sha256']
    runtime=runtime_api.AlignmentRuntime(source['actor']['path'],protocol=protocol,
        expected_actor_sha256=source['actor']['sha256'],expected_protocol_sha256=digest(protocol))
    runtime_api.verify(runtime)
    if (digest(scenarios)!=runtime.actor.metadata['scenario_manifest_sha256']
            or scenarios.get('test_fixture',False) is not False or scenarios['configuration']!=asdict(runtime.config)):
        raise ValueError('Original actual runtime scenario manifest/scope differs')
    scenes=scenarios['splits']['explanation_test'][:SCENARIO_COUNT]
    if (len(scenes)!=SCENARIO_COUNT or len({s['fingerprint'] for s in scenes})!=SCENARIO_COUNT
            or len({s['id'] for s in scenes})!=SCENARIO_COUNT
            or any(not s['id'].startswith('explanation_test_') or s['snapshot']['state']['frame']!=0 for s in scenes)):
        raise ValueError('Fixed twelve original explanation initial states required')
    excluded={s['fingerprint'] for key,pool in scenarios['splits'].items() if key!='explanation_test' for s in pool}
    if excluded & {s['fingerprint'] for s in scenes}: raise ValueError('Explanation pool overlaps another split')
    # Construct the real qualified explainer before any trajectory or output.
    renderer.DiverseAlignmentExplainer(program_path,expected_program_sha256=program_sha,runtime=runtime)
    code=sources();anchor=bindings(runtime,program_path,scenarios)
    plan={'version':VERSION,'test_fixture':False,'bindings':anchor,'sources':code,
        'execution_id':'diverse-answers-'+digest(anchor)[:20],'case_specs':case_specs(),'trajectory_spec':TRAJECTORY_SPEC,
        'system_output':str(system_output),'system_report_sha256':expected_system_report_sha256,
        'system_plan_sha256':expected_system_plan_sha256,'scenarios_path':str(Path(scenarios_path).resolve()),
        'scenarios_file_sha256':expected_scenarios_sha256,'scene_inputs':scenes,
        'phase_caps':dict.fromkeys(journal.PHASES,PHASE_CAP),'maximum_environment_steps':2*PHASE_CAP,
        'accounting_version':journal.VERSION,'automatic_retry':False,'refunds':False}
    root.mkdir(parents=True,exist_ok=False);_write(root/'plan.json',plan)
    for phase in journal.PHASES:
        folder=root/phase;folder.mkdir();(folder/'operations').mkdir()
        _write(folder/'state.json',journal._phase_state(phase))
    reports={}
    try:
        with journal._locked(root):
            for phase in journal.PHASES:
                j=journal._Journal(root,phase,plan)
                state=deepcopy(j.state);state['status']='running';j._commit(state)
                meter=_Meter(PHASE_CAP,j.before,j.after,j.execution_id)
                fresh=runtime_api.fresh_instance(runtime)
                meter.counts['numpy_actor_load_attempts']+=1;meter.counts['numpy_actor_loads']+=1
                meter.instrument(fresh)
                explainer=renderer.DiverseAlignmentExplainer(program_path,expected_program_sha256=program_sha,runtime=fresh)
                report=_phase(phase,fresh,explainer,scenes,meter,reports.get('generate'))
                report.update(bindings=anchor,sources=code,test_fixture=False,
                    input_report_sha256=digest(reports['generate']) if phase=='verify' else None)
                runtime_api.verify(fresh);runtime_api.verify(runtime);explainer._assert_current(fresh)
                if sources()!=code or bindings(runtime,program_path,scenarios)!=anchor: raise ValueError('Answer input/source changed')
                _write(root/phase/'report.json',report)
                state=deepcopy(j.state);state.update(status='completed',report=journal._binding(root,root/phase/'report.json'))
                j._commit(state);reports[phase]=report
        _bound_inputs(system_output,expected_system_report_sha256,expected_system_plan_sha256,
            scenarios_path,expected_scenarios_sha256)
        receipt={'version':VERSION,'status':'answers_verified' if reports['verify']['passed'] else 'candidate_blocked',
            'passed':reports['verify']['passed'],'plan_file_sha256':file_hash(root/'plan.json'),
            'reports':{phase:journal._binding(root,root/phase/'report.json') for phase in journal.PHASES},
            'bindings':anchor,'sources_sha256':digest(code),'full_matrix':40*SCENARIO_COUNT,
            'phase_caps':plan['phase_caps'],'research_qualification_evaluated':False,'release_ready':False}
        _write(root/'answer_receipt.json',receipt)
        return {**receipt,'receipt_file_sha256':file_hash(root/'answer_receipt.json')}
    except BaseException as error:
        failure={'version':VERSION,'passed':False,'error_type':type(error).__name__,'error':str(error),
            'automatic_retry':False,'release_ready':False,'phase_states':{phase:_read(root/phase/'state.json') for phase in journal.PHASES}}
        if 'meter' in locals():failure['execution']=meter.report()
        _write(root/'failure.json',failure);raise


def _read_phase(root,phase,plan):
    """Read original journal operations with their actual unchanged schema."""
    state=_read(root/phase/'state.json')
    if (state['version']!=journal.VERSION or state['phase']!=phase or state['status']!='completed'
            or state['cap']!=PHASE_CAP or state['pending'] is not None
            or state['automatic_retry'] is not False or state['refund_allowed'] is not False
            or type(state['reserved_steps']) is not int or state['reserved_steps']!=state['acknowledged_steps']
            or not 0<=state['reserved_steps']<=PHASE_CAP): raise ValueError('Incomplete original answer journal')
    phases={};expected_files=set();physical_records=[]
    for index in range(state['acknowledged_steps']):
        prefix=f'{phase}/operations/{index:05d}'; names=[prefix+s for s in ('.reservation.json','.record.json.gz','.ack.json')]
        expected_files.update(Path(n).name for n in names)
        ack=_read(root/names[2]);reservation=journal._bound(root,ack['reservation'],names[0]);record=journal._bound(root,ack['record'],names[1])
        op=f"{plan['execution_id']}-{phase}:{index}"
        if (ack['version']!=journal.VERSION or ack['driver_phase']!=phase or ack['operation_id']!=op
                or ack['actual_steps']!=1 or record['operation_id']!=op or record['actual_steps']!=1
                or reservation['version']!=journal.VERSION or reservation['driver_phase']!=phase
                or reservation['reserved_steps']!=1 or reservation['status']!='permanently_reserved'
                or record['status']!='executed_unacknowledged'
                or any(digest(record.get(k))!=digest(v) for k,v in reservation['context'].items())):
            raise ValueError('Original ACK identity/record differs')
        journal._validate_record(record);physical_records.append(record);phases[record['phase']]=phases.get(record['phase'],0)+1
    if {p.name for p in (root/phase/'operations').iterdir()}!=expected_files: raise ValueError('Unregistered journal file')
    report=journal._bound(root,state['report'],f'{phase}/report.json')
    execution=report['execution'];counts=execution['actual_execution']
    if (report['version']!=VERSION or report['operation']!=phase or report['bindings']!=plan['bindings'] or report['sources']!=plan['sources']
            or report['test_fixture'] is not False or execution['pending_operation'] is not None
            or execution['auxiliary_step_budget']!=PHASE_CAP or execution['accounting_complete'] is not True
            or any(counts[k]!=state['acknowledged_steps'] for k in ('environment_steps','environment_step_attempts','acknowledged_steps'))
            or counts['steps_by_phase']!=phases or counts['neural_updates']!=0 or counts['torch_loads']!=0):
        raise ValueError('Report physical accounting differs from original ACKs')
    _check_trace_acks(report,physical_records,plan)
    return report


def _check_trace_acks(report,operations,plan):
    """Link stored trajectory/fact transitions to actual durable physical calls."""
    def matches(transition,operation):
        return (all(digest(transition[k])==digest(operation[k]) for k in ('before','after','info','rewards'))
            and transition['submitted_actions']==operation['submitted_actions']
            and transition['executed_actions']==operation['info']['executed_actions']
            and transition['done']==bool(operation['terminated'] or operation['truncated']))
    expected_scenes=[s['id'] for s in plan['scene_inputs']]
    if set(report['trajectories'])!=set(expected_scenes):raise ValueError('Trajectory scene matrix differs')
    for scene in plan['scene_inputs']:
        trace=report['trajectories'][scene['id']]
        actual=[o for o in operations if o['phase']=='fixed_prefix' and o.get('scenario_id')==scene['id']]
        rows=trace['transitions']
        if (not rows or len(rows)!=len(actual) or trace['initial_fingerprint']!=scene['fingerprint']
                or trace['player_actions']!=[r['submitted_actions']['robot_1'] for r in rows]
                or any(not matches(t,a) for t,a in zip(rows,actual))
                or any(digest(a['after'])!=digest(b['before']) for a,b in zip(rows,rows[1:]))
                or not rows[-1]['done']):raise ValueError('Saved trajectory differs from acknowledged original physics')
    if report['operation']=='verify':
        for case in report['cases']:
            if 'independent_facts' not in case:continue
            transitions=case['independent_facts'].get('transitions',[])
            actual=[o for o in operations if o['phase']=='independent_oracle' and o.get('case_id')==case['case_id']]
            if len(transitions)!=len(actual) or any(not matches(t,a) for t,a in zip(transitions,actual)):
                raise ValueError('Independent counterfactual facts differ from acknowledged physical calls')


def read_receipt(output,*,expected_receipt_sha256,expected_bindings):
    """Strict byte/record reader, with no new forward/physics/text execution."""
    root=Path(output).resolve()
    if (root/'failure.json').exists(): raise ValueError('Failed answer run')
    receipt=system._json(root/'answer_receipt.json',expected_receipt_sha256)
    plan=system._json(root/'plan.json',receipt['plan_file_sha256'])
    if (receipt['version']!=VERSION or plan['version']!=VERSION or plan['test_fixture'] is not False
            or receipt['bindings']!=expected_bindings or plan['bindings']!=expected_bindings
            or plan['sources']!=sources() or receipt['sources_sha256']!=digest(sources())
            or plan['case_specs']!=case_specs() or plan['trajectory_spec']!=TRAJECTORY_SPEC
            or plan['phase_caps']!=dict.fromkeys(journal.PHASES,PHASE_CAP)
            or plan['maximum_environment_steps']!=PHASE_CAP*2 or receipt['phase_caps']!=plan['phase_caps']):
        raise ValueError('Answer receipt source/matrix/input differs')
    _,qualified,scenarios=_bound_inputs(plan['system_output'],plan['system_report_sha256'],plan['system_plan_sha256'],
        plan['scenarios_path'],plan['scenarios_file_sha256'])
    if (plan['scene_inputs']!=scenarios['splits']['explanation_test'][:SCENARIO_COUNT]
            or qualified['receipt']['qualified_program_file_sha256']!=expected_bindings['program_sha256']
            or digest(scenarios)!=expected_bindings['scenario_manifest_sha256']): raise ValueError('Qualified program/scenes differ')
    reports={phase:_read_phase(root,phase,plan) for phase in journal.PHASES}
    for phase in journal.PHASES:
        if receipt['reports'][phase]!=journal._binding(root,root/phase/'report.json'): raise ValueError('Receipt report binding differs')
    generated,verified=reports['generate'],reports['verify']
    ids={s['id']+'/'+spec['id'] for s in plan['scene_inputs'] for spec in case_specs()}
    if (len(plan['scene_inputs'])!=SCENARIO_COUNT or len(ids)!=480 or receipt['full_matrix']!=480
            or generated['passed'] is not False or generated['independently_verified'] is not False
            or verified['input_report_sha256']!=digest(generated)
            or digest(generated['trajectories'])!=digest(verified['trajectories'])): raise ValueError('Independent matrix/trajectory binding differs')
    for report in reports.values():
        if len(report['cases'])!=480 or {c['case_id'] for c in report['cases']}!=ids: raise ValueError('Missing/duplicate answer rows')
    raw={r['case_id']:r for r in generated['cases']};coverage=set();passed=True
    for row in verified['cases']:
        original=raw[row['case_id']]
        if 'missing_anchor' in row: passed=False;continue
        spec=next(s for s in case_specs() if s['id']==row['case_id'].rsplit('/',1)[-1])
        trajectory=generated['trajectories'][row['scenario_id']]['transitions']
        target={'initial':0,'first':1,'third':3,'terminal':len(trajectory)}[spec['anchor']]
        expected_record=({'after':trajectory[0]['before'],'runtime_signature':expected_bindings['runtime_signature']}
            if target==0 else trajectory[target-1])
        if (row['selected_frame']!=target or row['record_sha256']!=digest(expected_record)
                or row['request']!=request_for_spec(spec,target) or row['expected']!=spec['expected']
                or row['question']!=spec['question'] or row['language']!=spec['language']):
            raise ValueError('Question semantics or selected frame differs from frozen specification')
        text=row['answer'];phrase='不能用这个树分支解释本次选择' if row['language']=='zh' else 'its branch cannot explain this choice'
        computed=(original['answer']==text==row['independent_answer']
            and sha256(text.encode()).hexdigest()==row['answer_sha256']==original['answer_sha256']
            and row['independent_facts_sha256']==digest(row['independent_facts'])
            and row['error']==original['error']==spec['expected'].get('error')
            and row['mismatch_disclosure_required'] is (phrase in row['independent_answer'])
            and (not row['mismatch_disclosure_required'] or phrase in text)
            and all(digest(original.get(k))==digest(row.get(k)) for k in ('scenario_id','initial_fingerprint','selected_frame','language','request','expected','record_sha256','question')))
        if computed is not row['passed']: raise ValueError('Raw answer comparison contradicts saved outcome')
        passed &= computed
        if 'error' not in spec['expected']:coverage.add((spec['language'],spec['expected']['intent']))
    required={(l,i) for l in ('zh','en') for i in ('reason','alternative','counterfactual','failure','rules','clarify')}
    passed &= required<=coverage
    if (passed is not verified['passed'] or passed is not receipt['passed'] or verified['required_coverage_complete'] is not (required<=coverage)
            or receipt['status']!=('answers_verified' if passed else 'candidate_blocked')): raise ValueError('Recomputed question coverage/outcomes differ')
    return {'receipt':deepcopy(receipt),'generated_report':generated,'verification_report':verified,
        'passed':passed,'reader_environment_steps':0,'reader_NN_queries':0,'release_ready':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('system-output','system-report-sha','system-plan-sha','scenarios','scenarios-sha','output'):
        parser.add_argument('--'+name,required=True)
    a=parser.parse_args(argv)
    print(json.dumps(run(a.system_output,a.scenarios,a.output,expected_system_report_sha256=a.system_report_sha,
        expected_system_plan_sha256=a.system_plan_sha,expected_scenarios_sha256=a.scenarios_sha),ensure_ascii=False,sort_keys=True))


if __name__=='__main__':main()
