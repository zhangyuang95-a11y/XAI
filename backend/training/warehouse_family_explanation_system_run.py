"""Fixed acceptance-log cases, genuine Alignment NN branches, and independent replay.

Prepare reads acknowledged acceptance bytes, never executes a policy/physics. Run
uses two independently loaded genuine Alignment runtimes. The second inference
path explicitly evaluates logits/float32 softmax and restores the authoritative
public environment directly. No trees choose cases, actions, or interventions.

This produces candidate *component* evidence, not free-language/UI/release
qualification. Answer checks cover the two bilingual factual templates here;
they do not certify the older explainer or a new language model. A pending run
is never resumed, retried, or refunded automatically.
"""
from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import argparse
import gzip
import json
import os

import numpy as np

from backend import warehouse_alignment_runtime as runtime_api
from backend.training import warehouse_family_explanation_acceptance as acceptance
from backend.training import warehouse_family_alignment_diverse_collection as collector
from backend.training import warehouse_family_alignment_diverse_acceptance as program_api
from backend.training.warehouse_family_explanation_audit import physical_projection
from backend.training.warehouse_native_public_feedback import PublicFeedbackEnvironment
from backend.training.warehouse_native_public_feedback_evaluation import REWARD
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash

VERSION = 'warehouse-family-explanation-system-replay.v3'
ROLES, ACTIONS = acceptance.ROLES, acceptance.ACTIONS


def execution_sources():
    result = runtime_api.runtime_sources()
    result.update(program_api.execution_sources())
    result.update(acceptance.execution_sources())
    for module in (collector,):
        result[str(Path(module.__file__).relative_to(ROOT))] = file_hash(module.__file__)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    result['backend/training/warehouse_family_explanation_audit.py'] = file_hash(ROOT/'backend/training/warehouse_family_explanation_audit.py')
    return dict(sorted(result.items()))


def _read(path, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular anchored input required')
    raw = path.read_bytes()
    if expected is not None and sha256(raw).hexdigest() != expected: raise ValueError('External input SHA differs: '+str(path))
    return raw


def _json(path, expected=None):
    raw = _read(path, expected)
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def _put(path, value, *, replace=False):
    path = Path(path); raw = canonical(value).encode()
    if path.suffix == '.gz': raw = gzip.compress(raw, compresslevel=1, mtime=0)
    temporary = path.with_name(path.name+'.pending') if replace else path
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    if replace: os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _events(raw, context, receipt):
    if sha256(raw).hexdigest() != receipt['events_sha256']: raise ValueError('Acknowledged events changed')
    before, previous, frames = None, None, []
    for index, line in enumerate(gzip.decompress(raw).splitlines()):
        row = json.loads(line)
        if (row.get('context_id') != context['id'] or type(row.get('operation_id')) is not str
                or row['operation_id'].split(':')[-1] != f'{index:06d}'):
            raise ValueError('Event identity/order differs')
        event = row['event']
        if event['kind'] == 'before_base_step':
            if before is not None: raise ValueError('Unconfirmed base step')
            before = event
            if previous is not None and digest(previous) != digest(before['before']): raise ValueError('Live log snapshot chain differs')
        elif event['kind'] == 'after_base_step':
            if before is None: raise ValueError('Base completion has no reservation')
            if (event['after']['state']['frame'] != before['before']['state']['frame']+1
                    or before['submitted'] != event['info']['requested_actions']):
                raise ValueError('Base frame/actual submitted action differs')
            frames.append(before); previous = event['after']; before = None
    if before is not None or not frames or previous['state']['frame'] != receipt['final_frame']:
        raise ValueError('Incomplete acknowledged live trajectory')
    return frames


def _frame_matrix(context, frames, data):
    """No model/tree call; retain all base NN rows and fixed anchor interventions."""
    base = {}; groups = {}
    for i, source in enumerate(data['row_sources']):
        role, frame = data['roles'][i], source['frame']
        if source['context_id'] != context['id'] or role not in ROLES: raise ValueError('Row role/context differs')
        if data['kind'][i] == 'base':
            key = (frame, role)
            if key in base or ROLES.index(role) == context['program_role']: raise ValueError('Program actions are not NN base labels')
            base[key] = i; groups[key] = data['groups'][i]
        elif data['kind'][i] == 'counterfactual':
            key = (frame-1, role)
            if key in groups and groups[key] != data['groups'][i]: raise ValueError('Public group attribution differs')
            groups[key] = data['groups'][i]
    result=[]; seen=set()
    for frame in frames:
        snap=frame['before']; n=snap['state']['frame']; rows=[]; counter=[]
        for role in ROLES:
            common={'fingerprint':context['fingerprint'],'role':role,'groups':deepcopy(groups.get((n,role),[]))}
            prefix=f"{context['id']}:{n}:{role}"
            if (n,role) in base:
                i=base[n,role];seen.add((n,role))
                p=data['probabilities'][i]
                if (digest(p)!=digest(frame['decision']['probabilities'][role])
                        or frame['submitted'][role] != ACTIONS[int(np.argmax(p))]):
                    raise ValueError('Saved ordinary labels differ from actually submitted NN')
                rows.append({'case':dict(common,id=prefix+':base'),'saved_probabilities':p,
                    'observation':data['observations'][i]})
            if n % 20 == 0:
                # One direction per role/anchor, fixed solely by source
                # identities before any tree or branch outcome is observed.
                # Four partner profiles across all 64 scenes retain broad
                # coverage without repeating all four directions at every
                # historical frame.
                profile_index=collector.PROFILES.index(context['profile'])
                direction=ACTIONS[(context['scene_index']+profile_index+n//20+ROLES.index(role))%4]
                counter.append(dict(common,id=prefix+':cf1:'+direction,horizon=1,intervention_action=direction))
                # Fixed initial-frame three-step check; never selected by outcome.
                if n == 0:
                    counter.append(dict(common,id=prefix+':cf3:LEFT',horizon=3,intervention_action='LEFT'))
        result.append({'snapshot':snap,'base':rows,'counterfactual':counter})
    if seen != set(base): raise ValueError('Ordinary labels have no acknowledged pre-action snapshot')
    return result


def prepare(collection_root, program_path, output, *, expected_collection_plan_sha256,
            expected_collection_manifest_sha256, expected_program_sha256):
    """Production preparation only: zero NN/physics/PT; no fixture CLI bypass."""
    source=Path(collection_root).resolve(); output=Path(output).absolute(); program_path=Path(program_path).resolve()
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('New separate output required')
    source_plan_raw=_read(source/'plan.json',expected_collection_plan_sha256)
    source_manifest_raw=_read(source/'manifest.json',expected_collection_manifest_sha256)
    pools, plan, manifest=collector.read_data(source,phase='acceptance')
    del pools  # strict existing reader checked all ACKs; never use fit/dev outcomes
    if (digest(plan)!=digest(json.loads(source_plan_raw)) or digest(manifest)!=digest(json.loads(source_manifest_raw))
            or set(plan['phase_pools'])!={'accept'}): raise ValueError('Acceptance source changed while reading')
    raw_program=_read(program_path,expected_program_sha256)
    program_api._validate_program(json.loads(raw_program),tuple(plan['feature_names']),plan['actor_bindings'],allow_test_fixture=False)
    input_files={str(source/'plan.json'):expected_collection_plan_sha256,
                 str(source/'manifest.json'):expected_collection_manifest_sha256,str(program_path):expected_program_sha256}
    for path, expected in program_api._collection_file_bindings(plan): input_files[str(path)]=expected
    scenarios=_json(plan['input_files']['scenarios']['path'],plan['input_files']['scenarios']['sha256'])
    holdout=sorted({c['fingerprint'] for c in plan['contexts']})
    excluded=sorted({r['fingerprint'] for rows in scenarios['splits'].values() for r in rows}-set(holdout))
    batches=[]; ordinary=[]; counter=[]
    for index, entry in enumerate(manifest['completed']):
        directory=source/Path(entry['path']).parent
        if not directory.resolve().is_relative_to(source): raise ValueError('Collection path escapes root')
        data_raw=_read(source/entry['path'],entry['sha256']); events_raw=_read(directory/'events.jsonl.gz',entry['events_sha256'])
        receipt_raw=_read(directory/'receipt.json')
        if json.loads(receipt_raw)!=entry: raise ValueError('ACK receipt differs')
        for path,raw in ((source/entry['path'],data_raw),(directory/'events.jsonl.gz',events_raw),(directory/'receipt.json',receipt_raw)):
            input_files[str(path)]=sha256(raw).hexdigest()
        data=json.loads(gzip.decompress(data_raw)); ctx=plan['contexts'][index]
        matrix=_frame_matrix(ctx,_events(events_raw,ctx,entry),data)
        batches.append({'context':ctx,'frames':matrix})
        ordinary.extend(row['case'] for frame in matrix for row in frame['base'])
        counter.extend(case for frame in matrix for case in frame['counterfactual'])
    case_plan={'version':acceptance.PLAN_VERSION,'contract_sha256':digest(acceptance.contract()),
        'holdout_fingerprints':holdout,'excluded_fingerprints':excluded,'base_cases':ordinary,'counterfactual_cases':counter}
    if len(holdout)!=64 or len(batches)!=256: raise ValueError('Fixed 64-scene/256-context matrix required')
    cap=sum(4*c['horizon'] for c in counter)
    sources=execution_sources()
    prepared={'version':VERSION,'test_fixture':False,'collection_root':str(source),'input_files':input_files,
        'collection_plan':plan,'sources':sources,'program_path':str(program_path),'case_plan':case_plan,
        'case_batches_sha256':digest(batches),'maximum_environment_steps':cap,
        'sampling_rule':'all ordinary NN rows; all four profiles, both roles every twentieth frame, one identity-cycled direction h1; frame0 LEFT h3',
        'tree_outcomes_used_to_select_cases':False,'automatic_retry':False,'refunds':False,
        'PPO_steps':0,'PT_loads':0,'qualification_evaluated':False,'release_ready':False}
    for path, expected in input_files.items(): _read(path,expected)
    if execution_sources()!=sources: raise ValueError('Sources changed during preparation')
    output.mkdir(parents=True,exist_ok=False)
    _put(output/'cases.json.gz',batches); prepared['case_file_sha256']=file_hash(output/'cases.json.gz')
    _put(output/'plan.json',prepared)
    return {'status':'prepared','plan_sha256':file_hash(output/'plan.json'),'maximum_environment_steps':cap,
        'base_cases':len(ordinary),'counterfactual_cases':len(counter),'environment_steps':0,'NN_queries':0,'release_ready':False}


def _raw_environment(runtime,snapshot):
    env=PublicFeedbackEnvironment(runtime.config,deepcopy(REWARD),collision_cost=.05,mode='observed')
    env.restore(deepcopy(snapshot),require_feedback=True)
    return env


def _query(runtime,env,counts,independent=False):
    before=digest(env.snapshot()); counts['nn_queries_attempted']+=1
    if independent:
        obs=env.observations(); logits=runtime.actor.logits(np.stack([obs[r] for r in ROLES]))
        p=np.exp(logits-logits.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        probs={r:p[i].tolist() for i,r in enumerate(ROLES)}
        commands={r:ACTIONS[int(p[i].argmax())] for i,r in enumerate(ROLES)}
    else:
        commands,decision=runtime.decision(env);probs=decision['probabilities']
        if decision['masks'] is not False or decision['post_policy_overrides']!=0: raise ValueError('NN action override')
    counts['nn_queries']+=1;counts['nn_rows']+=2
    if digest(env.snapshot())!=before: raise ValueError('Inference changed live state/RNG')
    return commands,probs


def _tree(program,env,role,counts):
    counts['tree_routes']+=1
    values=program.predict_proba(dict(zip(env.feature_names,map(float,env.observations()[role]))))
    return [float(values[a]) for a in ACTIONS]


def _branch(runtime,live,case,action,counts,*,independent):
    snapshot=live.snapshot();env=_raw_environment(runtime,snapshot) if independent else runtime.from_snapshot(snapshot)
    assumed=[action]+['WAIT']*(case['horizon']-1); trace=[];other=next(r for r in ROLES if r!=case['role'])
    for command in assumed:
        before=env.snapshot(); proposals,probs=_query(runtime,env,counts,independent)
        submitted={case['role']:proposals[case['role']],other:command}
        counts['environment_steps_attempted']+=1
        _,_,terminated,truncated,info=env.step(submitted)
        counts['environment_steps']+=1
        if info['requested_actions']!=submitted: raise ValueError('Physical submission overridden')
        after=env.snapshot()
        trace.append({'before_sha256':digest(before),'after_sha256':digest(after),
            'physical_after_sha256':digest(physical_projection(after['state'])),'after_rng_sha256':digest(env.get_rng_state()),
            'frame':before['state']['frame'],'nn_probabilities':probs[case['role']],
            'submitted_actions':submitted,'executed_actions':info['executed_actions'],'terminal':bool(terminated or truncated)})
        if terminated or truncated: break
    next_probs=None if env.done else _query(runtime,env,counts,independent)[1][case['role']]
    return {'assumed_actions':assumed,'steps':trace,'next_probabilities':next_probs},env


def _run_pair(runtime,live,case,counts,bindings,*,independent):
    before=digest(live.snapshot());rng=digest(live.get_rng_state()); branches=[];endpoints=[]
    for action in ('WAIT',case['intervention_action']):
        branch,endpoint=_branch(runtime,live,case,action,counts,independent=independent)
        branches.append(branch);endpoints.append(endpoint)
    result={'implementation':'independent_raw_nn_physics' if independent else 'runtime_isolated_counterfactual',
        'execution_id':case['id']+(':raw' if independent else ':engine'),
        'actor_sha256':runtime.actor_sha256,'runtime_signature':runtime.signature,
        'sources_sha256':bindings['independent_replay_sources_sha256' if independent else 'runtime_sources_sha256'],
        'live_before_sha256':before,'live_after_sha256':digest(live.snapshot()),
        'rng_before_sha256':rng,'rng_after_sha256':digest(live.get_rng_state()),'branches':branches}
    return result,endpoints


def _action(probabilities):
    return None if probabilities is None else ACTIONS[int(np.argmax(probabilities))]


def _text(case,language,actions,tree=None):
    """Actual simple renderer; evidence statements only, no coordination advice."""
    prefix=(f"帧 {case['frame']}，{case['role']}：" if language=='zh' else f"Frame {case['frame']}, {case['role']}: ")
    if tree is not None:
        content=(f'神经动作 {actions[0]}；近似树动作 {tree}。' if language=='zh' else f'NN action {actions[0]}; approximate tree action {tree}.')
        if tree!=actions[0]: content+=('不能用这个树分支解释本次选择。' if language=='zh' else ' its branch cannot explain this choice.')
    else:
        labels=['terminal' if a is None else a for a in actions]
        content=(f"等待 / {case['intervention_action']} 后（共 {case['horizon']} 步，其余等待）的下一神经动作：{labels[0]} / {labels[1]}。"
            if language=='zh' else f"Next NN actions after WAIT / {case['intervention_action']} ({case['horizon']} steps, remaining actions WAIT): {labels[0]} / {labels[1]}.")
    return prefix+content


def _oracle_text(case,language,probabilities,tree_probabilities=None):
    """Separate textual oracle consumes independent raw-NN evidence, not renderer output."""
    labels=[None if p is None else ACTIONS[max(range(5),key=p.__getitem__)] for p in probabilities]
    lead={'zh':f"帧 {case['frame']}，{case['role']}：",'en':f"Frame {case['frame']}, {case['role']}: "}[language]
    if tree_probabilities is not None:
        approx=ACTIONS[max(range(5),key=tree_probabilities.__getitem__)]
        if language=='zh':
            return lead+f'神经动作 {labels[0]}；近似树动作 {approx}。'+('不能用这个树分支解释本次选择。' if approx!=labels[0] else '')
        return lead+f'NN action {labels[0]}; approximate tree action {approx}.'+(' its branch cannot explain this choice.' if approx!=labels[0] else '')
    first,second=['terminal' if value is None else value for value in labels]
    if language=='zh': return lead+f"等待 / {case['intervention_action']} 后（共 {case['horizon']} 步，其余等待）的下一神经动作：{first} / {second}。"
    return lead+f"Next NN actions after WAIT / {case['intervention_action']} ({case['horizon']} steps, remaining actions WAIT): {first} / {second}."


def _execute_frame(engine,replayer,program,item,bindings,counts,*,context_already_verified=False):
    """Dependency-injectable kernel; public run admits only exact AlignmentRuntime."""
    live=engine.from_snapshot(item['snapshot']);oracle_live=_raw_environment(replayer,item['snapshot'])
    before=digest(live.snapshot()); rng=digest(live.get_rng_state())
    result={'base_rows':[],'counterfactual_rows':[],'answer_rows':[]}
    contexts=[nullcontext() if context_already_verified else
        (r.verified_context(e) if hasattr(r,'verified_context') else nullcontext())
        for r,e in ((engine,live),(replayer,oracle_live))]
    with contexts[0],contexts[1]:
        if item['base']:
            _,p=_query(engine,live,counts);_,q=_query(replayer,oracle_live,counts,True)
            for row in item['base']:
                case=row['case'];role=case['role'];features=live.observations()[role]
                if (not np.array_equal(features,np.asarray(row['observation'],np.float32))
                        or not np.array_equal(np.asarray(p[role],np.float32),np.asarray(row['saved_probabilities'],np.float32))
                        or p[role]!=q[role]): raise ValueError('Fresh NN/observation differs from acknowledged source labels')
                tree=_tree(program,live,role,counts)
                result['base_rows'].append({'id':case['id'],'nn_probabilities':p[role],'tree_probabilities':tree})
                c={**case,'frame':live.state.frame};mismatch=_action(tree)!=_action(p[role])
                for language in ('zh','en'):
                    result['answer_rows'].append({'case_id':case['id'],'language':language,'evidence_method':'nn_and_approximate_tree',
                        'text':_text(c,language,[_action(p[role])],_action(tree)),
                        'independent_expected_text':_oracle_text(c,language,[q[role]],tree),
                        'mismatch_disclosed':mismatch,'claims_tree_as_internal_cause':False})
        for case in item['counterfactual']:
            actual,endpoints=_run_pair(engine,live,case,counts,bindings,independent=False)
            replay,_=_run_pair(replayer,oracle_live,case,counts,bindings,independent=True)
            tree=[None if e.done else _tree(program,e,case['role'],counts) for e in endpoints]
            result['counterfactual_rows'].append({'id':case['id'],'engine':actual,'independent_replay':replay,'tree_endpoint_probabilities':tree})
            c={**case,'frame':live.state.frame}
            for language in ('zh','en'):
                result['answer_rows'].append({'case_id':case['id'],'language':language,'evidence_method':'isolated_nn_branch',
                    'text':_text(c,language,[_action(b['next_probabilities']) for b in actual['branches']]),
                    'independent_expected_text':_oracle_text(c,language,[b['next_probabilities'] for b in replay['branches']]),
                    'mismatch_disclosed':False,'claims_tree_as_internal_cause':False})
    if digest(live.snapshot())!=before or digest(live.get_rng_state())!=rng: raise ValueError('Live state/RNG changed')
    return result


def new_counts():
    return dict(environment_steps_attempted=0,environment_steps=0,nn_queries_attempted=0,nn_queries=0,
        nn_rows=0,tree_routes=0,PPO_steps=0,optimizer_updates=0,PT_loads=0,tree_fits=0)


def run(output,*,expected_plan_sha256):
    output=Path(output).resolve(); plan=_json(output/'plan.json',expected_plan_sha256)
    if (plan['version']!=VERSION or plan['test_fixture'] is not False or plan['sources']!=execution_sources()
            or (output/'ledger.json').exists()): raise ValueError('Wrong sources or prior execution; no automatic retry/resume')
    for path,expected in plan['input_files'].items(): _read(path,expected)
    batches=_json(output/'cases.json.gz',plan['case_file_sha256'])
    if digest(batches)!=plan['case_batches_sha256']: raise ValueError('Frozen case matrix changed')
    inputs=plan['collection_plan']['input_files']; protocol=_json(inputs['protocol']['path'],inputs['protocol']['sha256'])
    actor_path=inputs['actor']['path']; actor_sha=inputs['actor']['sha256']
    engine=runtime_api.AlignmentRuntime(actor_path,protocol=protocol,expected_actor_sha256=actor_sha,expected_protocol_sha256=digest(protocol))
    replay=runtime_api.fresh_instance(engine)
    program=program_api._validate_program(_json(plan['program_path'],plan['input_files'][plan['program_path']]),
        tuple(engine.actor.metadata['feature_names']),plan['collection_plan']['actor_bindings'],allow_test_fixture=False)
    bindings={'actor_sha256':actor_sha,'program_sha256':plan['input_files'][plan['program_path']],
        'protocol_sha256':digest(protocol),'runtime_signature':engine.signature,'runtime_sources_sha256':digest(engine.sources),
        'independent_replay_sources_sha256':digest(plan['sources']),'answer_verifier_sources_sha256':digest(plan['sources']),
        'plan_sha256':digest(plan['case_plan']),'contract_sha256':digest(acceptance.contract()),
        'acceptance_sources_sha256':digest(acceptance.execution_sources())}
    if engine.signature!=plan['collection_plan']['actor_bindings']['runtime_signature']: raise ValueError('Frozen runtime identity differs')
    counts=new_counts();ledger={'version':VERSION,'plan_sha256':expected_plan_sha256,'cap':plan['maximum_environment_steps'],
        'reserved':0,'confirmed':0,'pending':None,'completed':[],'counts':counts,'automatic_retry':False,'refunds':False}
    _put(output/'ledger.json',ledger);(output/'records').mkdir(exist_ok=False)
    evidence={'version':acceptance.EVIDENCE_VERSION,'test_fixture':False,'bindings':bindings,'plan':plan['case_plan'],
        'base_rows':[],'counterfactual_rows':[],'answer_rows':[]}
    try:
        # Full Actor/protocol/source hashes are checked once at both run
        # boundaries.  All per-frame decisions inside use the runtime's
        # object/array guard, while every durable context remains separately
        # reserved and acknowledged below.  Rehashing the complete source
        # closure for every historical frame changes no evidence and made the
        # fixed matrix unnecessarily hours long.
        first_snapshot=batches[0]['frames'][0]['snapshot']
        engine_anchor=engine.from_snapshot(first_snapshot)
        replay_anchor=_raw_environment(replay,first_snapshot)
        with engine.verified_context(engine_anchor), replay.verified_context(replay_anchor):
            for index,batch in enumerate(batches):
                cap=sum(4*c['horizon'] for frame in batch['frames'] for c in frame['counterfactual'])
                pending={'index':index,'context_id':batch['context']['id'],'reserved':cap}
                candidate=deepcopy(ledger);candidate['reserved']+=cap;candidate['pending']=pending
                if candidate['reserved']>ledger['cap']: raise ValueError('Registered physical budget exceeded')
                _put(output/'ledger.json',candidate,replace=True);ledger=candidate
                record={'context':batch['context'],'base_rows':[],'counterfactual_rows':[],'answer_rows':[]}
                start=counts['environment_steps']
                for item in batch['frames']:
                    rows=_execute_frame(engine,replay,program,item,bindings,counts,
                        context_already_verified=True)
                    for key in rows: record[key].extend(rows[key])
                record['counts_after']=deepcopy(counts)
                path=output/'records'/f'{index:04d}.json.gz';_put(path,record)
                candidate=deepcopy(ledger);candidate['confirmed']+=counts['environment_steps']-start
                candidate['completed'].append({'index':index,'path':str(path.relative_to(output)),'sha256':file_hash(path),
                    'actual_steps':counts['environment_steps']-start});candidate['pending']=None;candidate['counts']=deepcopy(counts)
                _put(output/'ledger.json',candidate,replace=True);ledger=candidate
                for key in ('base_rows','counterfactual_rows','answer_rows'): evidence[key].extend(record[key])
        for path,expected in plan['input_files'].items(): _read(path,expected)
        if execution_sources()!=plan['sources']: raise ValueError('Source changed while executing')
        engine.verify_binding();replay.verify_binding()
        result=acceptance.evaluate_records(evidence,expected_evidence_sha256=acceptance.digest(evidence),expected_bindings=bindings)
        _put(output/'evidence.json.gz',evidence)
        report={'version':VERSION,'status':'component_records_passed' if result['records_passed'] else 'candidate_blocked',
            'plan_sha256':expected_plan_sha256,'evidence_file_sha256':file_hash(output/'evidence.json.gz'),
            'counts':counts,'record_acceptance':result,'sources':plan['sources'],
            'text_scope':'two bilingual evidence templates, not the free-question/UI explainer',
            'qualification_evaluated':False,'explanation_eligible':False,'release_ready':False}
        _put(output/'report.json',report)
        if result['records_passed']:
            for path,expected in plan['input_files'].items(): _read(path,expected)
            if execution_sources()!=plan['sources']: raise ValueError('Source changed before qualification')
            qualified=_qualified_program(plan,report,file_hash(output/'report.json'))
            _put(output/'system_qualified_program.json',qualified)
            receipt={'version':VERSION,'report_file_sha256':file_hash(output/'report.json'),
                'plan_file_sha256':expected_plan_sha256,
                'ordinary_tree_source_program_sha256':bindings['program_sha256'],
                'qualified_program_file_sha256':file_hash(output/'system_qualified_program.json'),
                'qualification_scope':'ordinary_tree_and_isolated_nn_engine',
                'free_question_answer_qualified':False,'release_ready':False}
            _put(output/'qualification_receipt.json',receipt)
        return report
    except BaseException as error:
        failure={'version':VERSION,'status':'blocked_execution','error_type':type(error).__name__,'error':str(error),
            'counts_observed_in_process':counts,'durable_ledger':_json(output/'ledger.json'),
            'automatic_retry':False,'refunds':False,'release_ready':False}
        # A publication error is never a qualification: remove only this run's
        # known new marker. Preserve the report and all failed raw records.
        for name in ('qualification_receipt.json','system_qualified_program.json'):
            try: (output/name).unlink(missing_ok=True)
            except OSError as cleanup_error: failure.setdefault('revocation_errors',[]).append(str(cleanup_error))
        _put(output/'failure.json',failure)
        raise


def _qualified_program(plan,report,report_sha):
    raw=_json(plan['program_path'],plan['input_files'][plan['program_path']])
    if report['record_acceptance']['records_passed'] is not True: raise ValueError('Failed records cannot qualify a program')
    raw=deepcopy(raw); metadata=raw['metadata']
    metadata['explanation_system_qualified']=True
    metadata['explanation_system_acceptance']={
        'contract_version':acceptance.VERSION,'producer_version':VERSION,
        'actor_sha256':report['record_acceptance']['bindings']['actor_sha256'],
        'input_program_sha256':plan['input_files'][plan['program_path']],
        'ordinary_tree_source_program_sha256':plan['input_files'][plan['program_path']],
        'case_plan_sha256':digest(plan['case_plan']),
        'evidence_file_sha256':report['evidence_file_sha256'],
        'report_file_sha256':report_sha,'sources_sha256':digest(plan['sources']),
        'scope':'ordinary_tree_and_isolated_nn_engine',
        'free_question_answer_qualified':False,'release_ready':False}
    return raw


def read_result(output,*,expected_report_sha256,expected_plan_sha256):
    """Pure complete-record/byte verification; no new NN, physics or tree query.

    External SHA anchors must be supplied from the producer receipt, never
    computed from an untrusted candidate as a way to assert authenticity.
    This reader rechecks original input locations; it is not a portable bundle.
    """
    output=Path(output).resolve()
    if (output/'failure.json').exists(): raise ValueError('Failed execution cannot be admitted')
    plan=_json(output/'plan.json',expected_plan_sha256)
    report=_json(output/'report.json',expected_report_sha256)
    if (plan['version']!=VERSION or plan['test_fixture'] is not False
            or plan['sources']!=execution_sources() or report['sources']!=plan['sources']
            or report['version']!=VERSION or report['plan_sha256']!=expected_plan_sha256):
        raise ValueError('Producer version, source, scope or report binding differs')
    for path,expected in plan['input_files'].items(): _read(path,expected)
    cases=_json(output/'cases.json.gz',plan['case_file_sha256'])
    if digest(cases)!=plan['case_batches_sha256']: raise ValueError('Case matrix differs')
    evidence=_json(output/'evidence.json.gz',report['evidence_file_sha256'])
    if evidence['plan']!=plan['case_plan'] or evidence['test_fixture'] is not False: raise ValueError('Case plan/scope differs')
    recomputed=acceptance.evaluate_records(evidence,expected_evidence_sha256=acceptance.digest(evidence),
        expected_bindings=evidence['bindings'])
    if recomputed!=report['record_acceptance']: raise ValueError('Raw acceptance record recomputation differs')
    actual=report['record_acceptance']['bindings'];inputs=plan['collection_plan']['input_files']
    expected={'actor_sha256':inputs['actor']['sha256'],'program_sha256':plan['input_files'][plan['program_path']],
        'protocol_sha256':plan['collection_plan']['actor_bindings']['protocol_sha256'],
        'runtime_signature':plan['collection_plan']['actor_bindings']['runtime_signature'],
        'runtime_sources_sha256':digest(runtime_api.runtime_sources()),
        'independent_replay_sources_sha256':digest(plan['sources']),
        'answer_verifier_sources_sha256':digest(plan['sources']),
        'plan_sha256':digest(plan['case_plan']),'contract_sha256':digest(acceptance.contract()),
        'acceptance_sources_sha256':digest(acceptance.execution_sources())}
    if actual!=expected: raise ValueError('Canonical bindings differ from frozen actual producer inputs')
    ledger=_json(output/'ledger.json')
    if (ledger['version']!=VERSION or ledger['plan_sha256']!=expected_plan_sha256 or ledger['pending'] is not None
            or ledger['cap']!=plan['maximum_environment_steps'] or ledger['counts']!=report['counts']
            or len(ledger['completed'])!=len(cases) or ledger['automatic_retry'] is not False or ledger['refunds'] is not False):
        raise ValueError('Incomplete or mismatched execution ledger')
    merged={key:[] for key in ('base_rows','counterfactual_rows','answer_rows')};steps=0;reserved=0;prior=new_counts()
    for index,(batch,entry) in enumerate(zip(cases,ledger['completed'])):
        expected_path=f'records/{index:04d}.json.gz'
        if entry['index']!=index or entry['path']!=expected_path: raise ValueError('Confirmed context identity differs')
        raw=_json(output/expected_path,entry['sha256'])
        if raw['context']!=batch['context']: raise ValueError('Context source differs')
        for key in merged: merged[key].extend(raw[key])
        actual_steps=sum(len(b['steps']) for row in raw['counterfactual_rows']
            for name in ('engine','independent_replay') for b in row[name]['branches'])
        if actual_steps!=entry['actual_steps'] or raw['counts_after']['environment_steps']-prior['environment_steps']!=actual_steps:
            raise ValueError('Confirmed physical count differs from actual saved branches')
        steps+=actual_steps;prior=raw['counts_after']
        reserved+=sum(4*c['horizon'] for f in batch['frames'] for c in f['counterfactual'])
    if (any(merged[k]!=evidence[k] for k in merged) or steps!=ledger['confirmed'] or reserved!=ledger['reserved']
            or reserved!=ledger['cap'] or prior!=report['counts']
            or any(report['counts'][k]!=0 for k in ('PPO_steps','optimizer_updates','PT_loads','tree_fits'))):
        raise ValueError('ACK records, aggregate evidence or execution accounting differ')
    passed=recomputed['records_passed']
    if report['status']!=('component_records_passed' if passed else 'candidate_blocked'): raise ValueError('Report status disagrees with evidence')
    qualified=None;receipt=None
    if passed:
        qualified=_qualified_program(plan,report,expected_report_sha256)
        saved=_json(output/'system_qualified_program.json')
        receipt=_json(output/'qualification_receipt.json')
        expected_receipt={'version':VERSION,'report_file_sha256':expected_report_sha256,
            'plan_file_sha256':expected_plan_sha256,'ordinary_tree_source_program_sha256':actual['program_sha256'],
            'qualified_program_file_sha256':file_hash(output/'system_qualified_program.json'),
            'qualification_scope':'ordinary_tree_and_isolated_nn_engine',
            'free_question_answer_qualified':False,'release_ready':False}
        if saved!=qualified or receipt!=expected_receipt: raise ValueError('Qualified program or receipt differs')
    elif (output/'system_qualified_program.json').exists() or (output/'qualification_receipt.json').exists():
        raise ValueError('Failed gate has an unexpected qualification marker')
    return {'report':deepcopy(report),'qualified_program':qualified,'receipt':receipt,
        'system_qualified':passed,'free_question_answer_qualified':False,'release_ready':False,
        'reader_environment_steps':0,'reader_NN_queries':0,'reader_PT_loads':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--output',required=True)
    for key in ('collection','program','collection-plan-sha','collection-manifest-sha','program-sha','plan-sha'): parser.add_argument('--'+key)
    args=parser.parse_args(argv)
    if args.prepare:
        required=('collection','program','collection_plan_sha','collection_manifest_sha','program_sha')
        if any(not getattr(args,k) for k in required): parser.error('Prepare requires all source paths and external SHAs')
        result=prepare(args.collection,args.program,args.output,expected_collection_plan_sha256=args.collection_plan_sha,
            expected_collection_manifest_sha256=args.collection_manifest_sha,expected_program_sha256=args.program_sha)
    else:
        if not args.plan_sha: parser.error('Run requires externally saved --plan-sha')
        result=run(args.output,expected_plan_sha256=args.plan_sha)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))


if __name__=='__main__': main()
