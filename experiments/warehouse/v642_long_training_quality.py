"""Fresh matched long-development quality and independent replay, never release.

This is a separate finite namespace and single-use stream/budget claim. It
changes neither spent calibration collection nor its admission/source meaning.
Only audited complete long training and all four fixed final programs permit
planning fresh scenes; ordinary JSON declarations are never capabilities.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from copy import deepcopy
import gzip,json,os
import numpy as np
from . import v642_calibration_quality as quality,v642_long_training_seed_plan as seeds
from . import seed_metadata_audit_indexed_v642 as metadata,seed_triage_indexed_v642 as triage,exact_seed_audit as exact
from . import seed_interval_index_v642 as interval_source
from .v626_protocol import require,pointer,bound,digest,file_sha
from .v642_fresh_program_metrics import evaluate as score_programs
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features,FEATURE_DIM
from backend.training.warehouse_v622_program import masked_probabilities
from backend.training.warehouse_v6_runner import _set_dual_nn_mode
from backend.training.warehouse_v629_environment_replay import capture_environment,restore_environment
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.public_observation_v6 import encode_public_observation,public_pre_action_snapshot
from env.warehouse.pure_nn import action_masks
from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG
from scripts.run_warehouse_v625 import write_new
from scripts.audit_warehouse_v642_confirmation import verify_frame
from scripts.measure_warehouse_v628_collection import episode as replay_public_episode

ROOT=seeds.ROOT
OUTPUT=ROOT/'output/warehouse_v642/long-training-quality'
VERSION='warehouse-v642-long-development-matched-quality.v1'
AUDIT_VERSION='warehouse-v642-independent-long-development-pair-gate.v1'
REGISTRY_REF=deepcopy(seeds.REGISTRY_REF)
FILES=('experiments/warehouse/v642_long_training_quality.py',
       'tests/test_warehouse_v642_long_training_quality.py','docs/warehouse_v642_long_training_quality.md',
       'scripts/run_warehouse_v642_long_quality.py','scripts/audit_warehouse_v642_long_quality.py',
       'experiments/warehouse/v642_fresh_program_metrics.py','experiments/warehouse/v642_calibration_quality.py',
       'scripts/audit_warehouse_v642_confirmation.py','scripts/measure_warehouse_v628_collection.py')


def source_files():
    from . import v642_long_training_owner as owner,v642_long_training_assets as training,v642_long_training_final_programs as final
    sources=owner.source_files();sources.update(training.source_files());sources.update(final.source_files())
    sources.update({name:file_sha(ROOT/name) for name in FILES})
    return dict(sorted(sources.items()))


def assets(report_ref):
    from .v642_long_training_assets import read_assets
    return read_assets(report_ref)


def identity(fixed):
    value=bound(fixed['report'])
    require(value['phase']=='development' and value['stream_id'] in seeds.protocol.DEVELOPMENT_STREAMS
        and value['joint_step_budget'] in seeds.protocol.BUDGETS,'formal or unknown long quality stream/budget')
    return {key:deepcopy(value[key]) for key in ('phase','stream_id','joint_step_budget','seed_plan')}


def identity_name(meta):
    require(meta['phase']=='development' and meta['stream_id'] in seeds.protocol.DEVELOPMENT_STREAMS
        and meta['joint_step_budget'] in seeds.protocol.BUDGETS,'formal quality is not admitted')
    return f"{meta['phase']}-{meta['stream_id']}-{meta['joint_step_budget']}"


def task_metrics(outcomes,cases,actors):
    report=quality.evaluate(outcomes,cases,actors)
    report.update(version=VERSION,recipe=collection_recipe(),
        scope='matched_long_development_quality_each_variant_overall_gate_and_task_strata')
    return report


def live(plan_ref,p,folder):
    # Large immutable training/observed pools are independently replayed on
    # entry and finish. Between episodes bind the loaded plan, frozen assets,
    # four programs and current execution source; do not rehash GB histories
    # 225 times or rerun program qualification on quality outcomes.
    require(pointer(plan_ref['path'])==plan_ref and p['source_files']==source_files(),'long quality plan/source changed')
    managed_tree(p)
    for ref in [*p['fixed']['actors'].values(),*(r for pair in p['final_programs']['programs'].values() for r in pair.values())]:
        require(pointer(ref['path'])==ref,'frozen long quality Actor/program changed')


def verify_physics(case,events,initial_ref,final_ref):
    before=bound(initial_ref);env=restore_environment(before)
    require(env.state.frame==0 and env.state.episode_id==1
        and env.config==participant_study_config(seed=case['scenario_seed'])
        and env.state.participant_controlled_agent_id=='robot_1','long quality initial physical scene differs')
    for event in events:
        require(public_pre_action_snapshot(env.state)==event['pre_state_public'],'long quality physical history disconnected')
        _,rewards,terminated,truncated,info=env.step(event['proposed_actions'])
        require(rewards==event['rewards'] and terminated is event['terminated'] and truncated is event['truncated']
            and sorted(info['shutdowns'])==sorted(event['shutdowns']) and not info['action_override_agents']
            and public_pre_action_snapshot(env.state)==event['post_state_public'],'long quality physical transition or scoring differs')
    require((env.state.terminated or env.state.truncated) and capture_environment(env)==bound(final_ref),
        'long quality private final state, RNG or complete terminal event differs')
    return dict(physical_frames=len(events),environment_constructors=0,new_environment_resets=0)


def declaration(rows,tag):
    quality.validate_cases(rows);uses=[]
    for stream in seeds.protocol.DEVELOPMENT_STREAMS:
        for variant in quality.VARIANTS:
            for case in rows:
                for kind,arg,key in (('scenario','task_expansion_base','base'),('env','random.Random(config.seed)','scenario_seed'),
                    ('env','reset(seed)','scenario_seed'),('action_rng','FrozenActor.act(base_seed)','action_seed')):
                    item=dict(kind=kind,purpose='v642_shared_long_development_quality',argument=arg,value=case[key],
                        group=stream+':'+variant+':'+case['case_id'],reuse_tag=tag+':paired:'+key+':'+str(case[key]))
                    if kind=='action_rng':item['consumed_as_random_draw']=True
                    uses.append(item)
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id='warehouse-v642-long-quality:'+tag,
        uses=uses,cases_sha256=digest(rows),planned_resets=675,independent_scenes=75,authorization_granted=False)


def final_programs(ref,completed_assets_ref):
    from .v642_long_training_final_programs import read_completed
    return read_completed(ref,completed_assets_ref)


def collection_recipe():
    return dict(quality.recipe(),version=VERSION,quality_evaluation_only=True,
        scope='long_development_pair_not_calibration_or_model_acceptance',
        frozen_final_programs_required_before_allocation=True,
        fresh_program_gate='each_raw_regularized_role_F_E_45_expected_F_40_agree_95_97_KL_020_illegal_unproved_zero',
        fresh_program_qualification_changes=0,task_and_all_four_program_gates_must_pass=True)


def prepare_cohort(*,streams,run_id,base_start,action_start):
    # All three independent long completion readers and twelve program gates
    # must pass before caller seed arithmetic or any output is touched.
    require(type(streams) is list and len(streams)==3,'three explicit fixed development streams required')
    fixed_rows={};budget=None
    for row in streams:
        require(set(row)=={'completed_assets','final_programs'},'unknown cohort prerequisite')
        fixed=assets(row['completed_assets']);programs=final_programs(row['final_programs'],row['completed_assets'])
        meta=identity(fixed);name=meta['stream_id']
        require(name not in fixed_rows and (budget is None or meta['joint_step_budget']==budget),
            'cohort requires exactly three different streams of the same budget')
        budget=meta['joint_step_budget'];fixed_rows[name]=dict(fixed=fixed,final_programs=programs,identity=meta)
    require(set(fixed_rows)==set(seeds.protocol.DEVELOPMENT_STREAMS),'missing development stream')
    seeds.legacy.confirmation.run_name(run_id)
    require(not cohort_claim_path(budget).exists(),'budget already consumed its shared quality cohort')
    rows=quality.cases(base_start,action_start)
    for i,row in enumerate(rows):row['case_id']=f'long-quality-{budget}-{i:03d}'
    sources=source_files();folder=OUTPUT/run_id;folder.mkdir(parents=True,exist_ok=False)
    value=dict(version=VERSION,run_id=run_id,joint_step_budget=budget,cases=rows,streams=fixed_rows,
        recipe=collection_recipe(),source_files=sources,independent_scenes=75,finite_reset_slots=675,
        reference_repeats_are_paired_not_independent_scenes=True,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'cohort.json',value);(folder/'cohort.json').chmod(0o444);cohort_ref=pointer(folder/'cohort.json')
    exact_value=declaration(rows,cohort_ref['sha256'])
    write_new(folder/'exact-plan.json',exact_value);(folder/'exact-plan.json').chmod(0o444)
    exact.read_exact_plan(folder/'exact-plan.json')
    for name,sha in sources.items():
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==sha,'quality source changed preparing cohort')
        target.write_bytes(raw);target.chmod(0o444)
    plans={}
    for stream,row in sorted(fixed_rows.items()):
        output=folder/stream;output.mkdir()
        plan=dict(version=VERSION,cohort=cohort_ref,run_id=run_id,**row['identity'],fixed=row['fixed'],
            final_programs=row['final_programs'],cases=rows,recipe=collection_recipe(),source_files=sources,
            exact_plan=pointer(folder/'exact-plan.json'),training_authority=False,sealed_authority=False,pilot_opened=False)
        write_new(output/'plan.json',plan);(output/'plan.json').chmod(0o444);plans[stream]=pointer(output/'plan.json')
    write_new(folder/'cohort-completed.json',dict(cohort=cohort_ref,plans=plans));(folder/'cohort-completed.json').chmod(0o444)
    return dict(cohort=cohort_ref,plans=plans)


def prepare(*args,**kwargs):
    raise RuntimeError('single-stream quality allocation is forbidden; prepare the shared three-stream cohort')


def cohort_claim_path(budget):return OUTPUT/'claims'/f'development-{budget}-cohort.json'


def cohort_data(p):
    ref=p['cohort'];c=bound(ref);folder=OUTPUT/seeds.legacy.confirmation.run_name(c['run_id'])
    require(ref==pointer(folder/'cohort.json') and c['version']==VERSION and c['source_files']==source_files()
        and c['recipe']==collection_recipe() and c['independent_scenes']==75 and c['finite_reset_slots']==675
        and c['reference_repeats_are_paired_not_independent_scenes'] is True
        and set(c['streams'])==set(seeds.protocol.DEVELOPMENT_STREAMS),'shared cohort source or slots changed')
    return c,folder


def checked(plan_ref):
    p=bound(plan_ref);c,root=cohort_data(p);folder=root/p['stream_id'];meta=identity(p['fixed'])
    require(plan_ref==pointer(folder/'plan.json') and p['version']==VERSION and p['cases']==c['cases']
        and p['recipe']==c['recipe'] and p['source_files']==c['source_files']
        and p['training_authority'] is p['sealed_authority'] is p['pilot_opened'] is False
        and all(p[k]==v for k,v in meta.items()) and p['joint_step_budget']==c['joint_step_budget'],
        'long quality plan source, cohort, phase or stream differs')
    quality.validate_cases(c['cases'])
    seal=bound(pointer(root/'cohort-completed.json'))
    require(seal['cohort']==p['cohort'] and seal['plans']=={v:pointer(root/v/'plan.json') for v in seeds.protocol.DEVELOPMENT_STREAMS},
        'shared cohort stream plan seal changed')
    for stream,row in c['streams'].items():
        require(row['fixed']==assets(row['fixed']['report']) and row['identity']==identity(row['fixed'])
            and row['identity']['stream_id']==stream and row['identity']['joint_step_budget']==c['joint_step_budget']
            and row['final_programs']==final_programs(row['final_programs']['report'],row['fixed']['report']),
            'one shared cohort training/program prerequisite changed')
    require(c['streams'][p['stream_id']]==dict(fixed=p['fixed'],final_programs=p['final_programs'],identity=meta)
        and p['exact_plan']==pointer(root/'exact-plan.json')
        and bound(p['exact_plan'])==declaration(c['cases'],p['cohort']['sha256']), 'cohort original seed arguments changed')
    for name,sha in c['source_files'].items():
        q=root/'source'/name;require(file_sha(q)==sha and not q.stat().st_mode&0o222,'quality source snapshot changed')
    return p,folder


def exclusions(p):
    c,folder=cohort_data(p)
    return [folder,cohort_claim_path(c['joint_step_budget']),
        *(claim_path(dict(phase='development',stream_id=v,joint_step_budget=c['joint_step_budget'])) for v in seeds.protocol.DEVELOPMENT_STREAMS)]


def managed_tree(p):
    """An exclusion contains only the exact source-bound cohort outputs."""
    c,folder=cohort_data(p)
    allowed={'cohort.json','cohort-completed.json','exact-plan.json'}
    allowed.update('source/'+name for name in c['source_files'])
    for stream in seeds.protocol.DEVELOPMENT_STREAMS:
        allowed.add(stream+'/plan.json')
        allowed.update(stream+'/collection/'+name for name in ('actual-resets.jsonl','reset-preauthorization.jsonl',
            'result.json','completed.json','failed.json','supervisor-exit.json'))
        allowed.update(stream+'/collection.supervision/'+name for name in ('started.json','exit.json','run.log','supervisor-failed.json'))
        for case in c['cases']:
            for variant in quality.VARIANTS:
                stem=stream+'/collection/'+variant+'-'+case['case_id']
                allowed.update(stem+suffix for suffix in ('.npz','.jsonl.gz','-initial.json','-final.json'))
    ledger=folder/'seed-ledger';clear=ledger/'clearance.json'
    if clear.exists():
        baseline=bound(pointer(clear)).get('seed_audit_outputs',[])
        allowed.add('seed-ledger/clearance.json')
        allowed.update(str(Path(ref['path']).relative_to(folder)) for ref in baseline)
    # Before clearance the audit itself owns this single, not-yet-executable
    # ledger directory. Once sealed its exact output inventory is committed.
    for path in folder.rglob('*'):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()),'cohort output contains a symlink or non-file')
        if path.is_file():
            require(str(path.relative_to(folder)) in allowed or (not clear.exists() and path.is_relative_to(ledger)),
                'unregistered file inside shared cohort exclusion')


def history_roots():
    return [ROOT.parent/name for name in seeds.HISTORY_NAMES]


def template_bindings(rows,*,execution):
    recorded=seeds.legacy.template_bindings(rows,check_current=False)
    if execution:
        current={}
        for root in history_roots():
            path=root/'backend/training/seed_ledger.py'
            if path.is_file():
                raw=path.read_bytes()
                if b'500_004' in raw and b'10_500_204' in raw:current[str(path.resolve())]=file_sha(path)
        require(current==recorded,'historical source reservations changed')
    return recorded


def history(p,folder):
    registry=REGISTRY_REF;traces=bound(registry)['registered_trace_paths']
    stats={};selected=[]
    for path in metadata.candidates(history_roots(),
        exclusions(p),registered_trace_paths=traces,selection_stats=stats):
        stat=path.stat();selected.append(dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only'],registry,traces


def audit_seeds(plan_ref):
    p,folder=checked(plan_ref);cohort,root=cohort_data(p)
    require(not cohort_claim_path(p['joint_step_budget']).exists(),'shared quality cohort already issued a single-use clearance')
    out=root/'seed-ledger';out.mkdir(exist_ok=False)
    families=exact.compile_families(bound(p['exact_plan']));_,_,registry,traces=history(p,folder)
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=bound(p['exact_plan'])['candidate_id'],
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=list(map(str,exclusions(p))),registered_trace_paths=traces)
    write_new(out/'proposal.json',proposal)
    metadata.audit(out/'proposal.json',history_roots(),out/'bounded')
    triage.triage(out/'bounded/inventory.json',out/'proposal.json',out/'triaged')
    result=exact.assess(p['exact_plan']['path'],out/'triaged/result.json',out/'exact',context_source_root=ROOT)
    seeds.legacy.conflict_free(result)
    selected,weights,_,_=history(p,folder)
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selected},'history changed during quality seed audit')
    templates=template_bindings(result['inventory'],execution=True)
    checked(plan_ref);now=datetime.now(timezone.utc)
    audit_outputs=[]
    for path in sorted(out.rglob('*')):
        require(not path.is_symlink(),'seed audit output cannot be symlinked')
        if path.is_file():path.chmod(0o444);audit_outputs.append(pointer(path))
    value=dict(version=VERSION,cohort=p['cohort'],exact=pointer(out/'exact/result.json'),registry=registry,traces=traces,
        selection=selected,weights=weights,templates=templates,issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),
        seed_audit_outputs=audit_outputs,environment_reset_ceiling=675,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(out/'clearance.json',value);(out/'clearance.json').chmod(0o444)
    claim=cohort_claim_path(p['joint_step_budget']);claim.parent.mkdir(parents=True,exist_ok=True)
    write_new(claim,dict(version=VERSION,cohort=p['cohort'],clearance=pointer(out/'clearance.json'),
        single_use_even_if_interrupted=True,finite_reset_slots=675,pilot_opened=False));claim.chmod(0o444)
    return pointer(out/'clearance.json')


def clearance(plan_ref,*,execution):
    p,folder=checked(plan_ref);cohort,root=cohort_data(p);ref=pointer(root/'seed-ledger/clearance.json');c=bound(ref);e=bound(c['exact'])
    issued=datetime.fromisoformat(c['issued_utc']);expires=datetime.fromisoformat(c['expires_utc'])
    managed_tree(p)
    for item in c['seed_audit_outputs']:_asset(item,[])
    require(c['version']==VERSION and c['cohort']==p['cohort'] and c['environment_reset_ceiling']==675
        and c['training_authority'] is c['sealed_authority'] is c['pilot_opened'] is False
        and issued.tzinfo and expires.tzinfo and expires-issued==timedelta(hours=24)
        and (not execution or issued<=datetime.now(timezone.utc)<expires),'quality clearance scope/binding/lifetime changed')
    require(e['exact_plan']==p['exact_plan'] and e['exact_auditor_sha256']==file_sha(exact.__file__), 'quality exact input/source changed')
    inv=bound(e['bounded_or_triaged_inventory']);projection=exact.project_inventory(inv,exact.compile_families(bound(p['exact_plan'])))
    if e.get('context_annotations'):projection=exact.apply_context(projection,bound(e['context_annotations']))
    require(e['summary']==projection['summary'] and digest(e['inventory'])==digest(projection['inventory']), 'quality exact audit failed replay')
    seeds.legacy.conflict_free(e)
    roots=sorted(str(root.resolve()) for root in history_roots())
    require(len(roots)==4 and all(Path(root).is_dir() for root in roots)
        and sorted(e['scan_roots'])==sorted(inv['scan_roots'])==roots and not inv['missing_roots']
        and e['selection_scope']==inv['selection_scope']==metadata.SELECTION_SCOPE
        and sorted(inv['scope_exclusions'])==sorted(map(str,[root/'seed-ledger/proposal.json',root/'seed-ledger/bounded',*exclusions(p)]))
        and inv['read_model_weight_files'] is e['read_model_weight_files'] is False
        and e['scope_exclusions']==inv['scope_exclusions']
        and inv['auditor_sha256']==file_sha(metadata.__file__) and inv['triage_driver_sha256']==file_sha(triage.__file__)
        and inv['interval_index_source_sha256']==file_sha(interval_source.__file__), 'quality audit scope/source changed')
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in e['inventory'] if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in c['selection']},'quality history selection differs from exact audit')
    require(bound(pointer(cohort_claim_path(p['joint_step_budget'])))==dict(version=VERSION,cohort=p['cohort'],
        clearance=ref,single_use_even_if_interrupted=True,finite_reset_slots=675,pilot_opened=False),'cohort clearance claim changed')
    registry=REGISTRY_REF
    require(c['registry']==registry and pointer(registry['path'])==registry
        and c['traces']==e['registered_trace_paths']==inv['registered_trace_paths']==bound(registry)['registered_trace_paths'], 'registered quality history traces changed')
    if execution:
        selected,weights,_,_=history(p,folder)
        require(selected==c['selection'] and weights==c['weights'],'historical metadata changed before quality collection')
        require(template_bindings(e['inventory'],execution=True)==c['templates'],'source reservations changed')
    return ref,c


def claim_path(p):return OUTPUT/'claims'/(identity_name(p)+'.json')


def collect(plan_ref):
    p,folder=checked(plan_ref);clearance_ref,c=clearance(plan_ref,execution=True)
    require(not claim_path(p).exists(),'calibration quality collection is once-only even after interruption')
    claim_path(p).parent.mkdir(parents=True,exist_ok=True)
    write_new(claim_path(p),dict(version=VERSION,plan=plan_ref,cohort=p['cohort'],clearance=clearance_ref,pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),single_use_even_if_interrupted=True,pilot_opened=False))
    claim_path(p).chmod(0o444);out=folder/'collection';out.mkdir(exist_ok=False)
    try:
        models={v:FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],
            expected_role_sha256=p['fixed']['actor_role_sha256'][v]) for v,ref in p['fixed']['actors'].items()}
        results=[];entries=[]
        with (out/'reset-preauthorization.jsonl').open('x') as journal,capture_stage1_resets(out/'actual-resets.jsonl') as resets:
            for case in p['cases']:
                for variant in quality.VARIANTS:
                    live(plan_ref,p,folder);require(pointer(clearance_ref['path'])==clearance_ref
                        and datetime.now(timezone.utc)<datetime.fromisoformat(c['expires_utc']), 'quality sources/clearance changed or expired')
                    auth=dict(case=case,variant=variant,plan=plan_ref,actor=p['fixed']['actors'][variant],authorized_utc=datetime.now(timezone.utc).isoformat())
                    journal.write(json.dumps(auth)+'\n');journal.flush();os.fsync(journal.fileno())
                    with stage1_reset_phase('quality'):
                        env=WarehouseMultiAgentEnv(participant_study_config(seed=case['scenario_seed']));env.reset(seed=case['scenario_seed'])
                    _set_dual_nn_mode(env);actor=models[variant];ref=p['fixed']['actors'][variant]
                    initial=out/(variant+'-'+case['case_id']+'-initial.json');write_new(initial,capture_environment(env));initial.chmod(0o444)
                    rows=[];arrays={key:[] for key in ('x','m','p','sampled')}
                    while not(env.state.terminated or env.state.truncated):
                        before=public_pre_action_snapshot(env.state);obs=encode_public_observation(env.state);m=action_masks(env.state)
                        actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
                        probabilities=masked_probabilities(actor.logits(obs,m),m)
                        from env.warehouse.navigation import ACTIONS
                        indices=[ACTIONS.index(actions[r]) for r in ('robot_1','robot_2')]
                        _,rewards,terminated,truncated,info=env.step(actions)
                        require(info['proposed_actions']==actions and not info['action_override_agents'],'quality execution overrode a complete-NN action')
                        rows.append(dict(case=case,frame=len(rows),pre_state_public=before,post_state_public=public_pre_action_snapshot(env.state),
                            proposed_actions=actions,actor_sha256=ref['sha256'],action_override_agents=[],
                            rewards=rewards,terminated=terminated,truncated=truncated,shutdowns=list(info['shutdowns'])))
                        for key,value in (('x',encode_actor_tree_features(obs,m)),('m',m),('p',probabilities),('sampled',indices)):arrays[key].append(value)
                    a={key:np.asarray(values,dtype=np.int64 if key=='sampled' else np.float32) for key,values in arrays.items()}
                    checked_actions=sum(verify_frame(actor,case,event,a,i) for i,event in enumerate(rows))
                    outcome=replay_public_episode(case,rows,len(rows),ref['sha256'])
                    require(outcome['deliveries']==env.state.total_deliveries and outcome['robot_collisions']==env.state.robot_collision_events
                        and outcome['shutdown']==bool(env.state.shutdown_count),'independent public quality counters differ from environment')
                    outcome.update(variant=variant,actor=ref,complete=True,NN_action_checks=checked_actions,action_overrides=0)
                    stem=out/(variant+'-'+case['case_id']);trace=stem.with_suffix('.jsonl.gz');array=stem.with_suffix('.npz')
                    with gzip.open(trace,'xt') as h:
                        for row in rows:h.write(json.dumps(row,separators=(',',':'))+'\n')
                    with array.open('xb') as h:np.savez_compressed(h,**a)
                    trace.chmod(0o444);array.chmod(0o444);results.append(outcome)
                    final=out/(variant+'-'+case['case_id']+'-final.json');write_new(final,capture_environment(env));final.chmod(0o444)
                    verify_physics(case,rows,pointer(initial),pointer(final))
                    entries.append(dict(case=case,variant=variant,trace=pointer(trace),arrays=pointer(array),
                        initial_environment=pointer(initial),final_environment=pointer(final)))
                if len(results)%45==0:print(json.dumps(dict(quality_episodes_completed=len(results),total=225)),flush=True)
        for name in ('actual-resets.jsonl','reset-preauthorization.jsonl'):
            (out/name).chmod(0o444)
        require(len(resets)==225 and [r['seed'] for r in resets]==[case['scenario_seed'] for case in p['cases'] for _ in quality.VARIANTS], 'quality reset inventory/order differs')
        checked(plan_ref);clearance(plan_ref,execution=True);metrics=task_metrics(results,p['cases'],p['fixed']['actors'])
        program_metrics=score_programs(p['final_programs'],p['fixed']['actor_role_sha256'],entries,p['cases'])
        result=dict(version=VERSION,plan=plan_ref,cohort=p['cohort'],phase=p['phase'],stream_id=p['stream_id'],
            joint_step_budget=p['joint_step_budget'],seed_plan=p['seed_plan'],claim=pointer(claim_path(p)),clearance=clearance_ref,metrics=metrics,
            program_metrics=program_metrics,development_pair_gate_passed=metrics['task_retention_passed'] and program_metrics['passed'],
            outcomes=results,entries=entries,actual_resets=pointer(out/'actual-resets.jsonl'),reset_preauthorization=pointer(out/'reset-preauthorization.jsonl'),
            actual_reset_count=225,task_retention_passed=metrics['task_retention_passed'],
            budget_lock_issued=False,formal_training_authorized=False,subsequent_formal_program_evaluation_required=True,model_acceptance_passed=False,pilot_opened=False)
        write_new(out/'result.json',result);(out/'result.json').chmod(0o444)
        write_new(out/'completed.json',dict(result=pointer(out/'result.json')));(out/'completed.json').chmod(0o444)
        return pointer(out/'result.json')
    except BaseException as error:
        write_new(out/'failed.json',dict(version=VERSION,plan=plan_ref,error_type=type(error).__name__,error=str(error),
            single_use_claim_retained=True,automatic_retry=False,pilot_opened=False));raise


def _asset(ref, inputs):
    path = Path(ref['path'])
    require(pointer(path) == ref and not path.is_symlink()
            and not path.stat().st_mode & 0o222,
            'long development input changed, missing, symlinked or writable')
    inputs.append(deepcopy(ref))
    return path


def audit(result_ref, supervisor_exit_ref):
    """Read one completed, actually exited collection and replay its evidence.

    Original recorded action seeds are numerically replayed, not allocated as
    new random streams. This grants neither development resets nor release.
    """
    inputs = []
    result_path = _asset(result_ref, inputs)
    execution = bound(supervisor_exit_ref)
    _asset(supervisor_exit_ref, inputs)
    from .v642_long_stage_supervisor import verify_execution
    subprocess_refs=verify_execution(supervisor_exit_ref,result_ref,stage='long-quality')
    for ref in subprocess_refs:_asset(ref,inputs)
    result = bound(result_ref)
    require(result['version'] == VERSION
            and result['model_acceptance_passed'] is result['pilot_opened'] is False,
            'unknown long development result or release scope')
    plan, folder = checked(result['plan'])
    output = folder / 'collection'
    require(result_path.resolve() == output / 'result.json'
            and Path(supervisor_exit_ref['path']).resolve() == output / 'supervisor-exit.json'
            and not (output / 'failed.json').exists(), 'failed or relocated long development collection')
    seal = pointer(output / 'completed.json')
    _asset(seal, inputs)
    require(bound(seal) == dict(result=result_ref), 'long development completion seal changed')
    _asset(result['plan'],inputs)
    _asset(plan['cohort'],inputs)
    for ref in [*plan['fixed']['inputs'],*plan['final_programs']['inputs']]:_asset(ref,inputs)
    clearance_ref, seed_clearance = clearance(result['plan'], execution=False)
    claim_ref = pointer(claim_path(plan))
    _asset(clearance_ref,inputs)
    cohort,cohort_folder=cohort_data(plan)
    for ref in [*seed_clearance['seed_audit_outputs'],pointer(cohort_claim_path(plan['joint_step_budget'])),
        pointer(cohort_folder/'cohort-completed.json'),*[pointer(cohort_folder/v/'plan.json') for v in seeds.protocol.DEVELOPMENT_STREAMS]]:
        _asset(ref,inputs)
    _asset(claim_ref, inputs)
    claim = bound(claim_ref)
    issued = datetime.fromisoformat(seed_clearance['issued_utc'])
    expires = datetime.fromisoformat(seed_clearance['expires_utc'])
    started = datetime.fromisoformat(claim['started_utc'])
    require(issued.tzinfo and expires.tzinfo and started.tzinfo
            and issued <= started < expires and claim['version'] == VERSION
            and claim['plan']==result['plan'] and claim['cohort']==plan['cohort']==result['cohort']
            and claim['clearance']==clearance_ref
            and claim['single_use_even_if_interrupted'] is True
            and claim['pilot_opened'] is False
            and result['claim']==claim_ref and result['clearance']==clearance_ref
            and claim['pid']==execution['worker_pid']
            and all(result[key]==plan[key] for key in ('phase','stream_id','joint_step_budget','seed_plan')),
            'long development claim or start-time seed clearance differs')
    resets_path = _asset(result['actual_resets'], inputs)
    auth_path = _asset(result['reset_preauthorization'], inputs)
    require(resets_path == output / 'actual-resets.jsonl'
            and auth_path == output / 'reset-preauthorization.jsonl', 'relocated long development reset journals')
    resets = [json.loads(line) for line in resets_path.read_text().splitlines()]
    auth = [json.loads(line) for line in auth_path.read_text().splitlines()]
    expected = [(variant, case) for case in plan['cases'] for variant in quality.VARIANTS]
    require(len(resets) == len(auth) == len(result['entries']) == len(expected) == 225
            and type(result['actual_reset_count']) is int and result['actual_reset_count'] == 225,
            'missing or additional long development episodes or resets')
    models = {variant: FrozenWarehouseV6Actor.load(ref['path'], expected_sha256=ref['sha256'],
              expected_role_sha256=plan['fixed']['actor_role_sha256'][variant])
              for variant, ref in plan['fixed']['actors'].items()}
    outcomes = []
    action_checks = 0
    previous_time = started
    for index, (variant, case) in enumerate(expected):
        entry = result['entries'][index]
        actor_ref = plan['fixed']['actors'][variant]
        authorized = datetime.fromisoformat(auth[index]['authorized_utc'])
        require(entry['variant'] == variant and entry['case'] == case
                and auth[index] == dict(case=case, variant=variant, plan=result['plan'],
                    actor=actor_ref, authorized_utc=auth[index]['authorized_utc'])
                and previous_time <= authorized < expires
                and resets[index]['seed'] == case['scenario_seed']
                and resets[index]['episode_id'] == 1 and resets[index]['phase'] == 'quality'
                and all(resets[index][key] == ACTION_PHYSICS_CONFIG[key]
                    for key in ('map_layout_id', 'move_battery_cost', 'charge_per_wait')),
                'long development actor, reset physics/order or authorization time differs')
        previous_time=authorized
        require(authorized<=datetime.fromisoformat(execution['finished_utc']),'reset authorization follows actual process exit')
        stem = output / (variant + '-' + case['case_id'])
        trace_path = _asset(entry['trace'], inputs)
        array_path = _asset(entry['arrays'], inputs)
        require(trace_path == stem.with_suffix('.jsonl.gz') and array_path == stem.with_suffix('.npz'),
                'long development trajectory assets are outside their declared scene')
        with gzip.open(trace_path, 'rt') as handle:
            events = [json.loads(line) for line in handle]
        with np.load(array_path, allow_pickle=False) as archive:
            require(set(archive.files) == {'x', 'm', 'p', 'sampled'}, 'unknown long development arrays')
            arrays = {key: archive[key].copy() for key in archive.files}
        n=len(events)
        require(0<n<=120 and arrays['x'].shape==(n,2,FEATURE_DIM)
            and arrays['m'].shape==arrays['p'].shape==(n,2,5) and arrays['sampled'].shape==(n,2)
            and arrays['sampled'].dtype==np.int64
            and all(arrays[k].dtype==np.float32 and np.isfinite(arrays[k]).all() for k in ('x','m','p')),
            'long quality array dimensions or arithmetic differ')
        require(all(len(value) == len(events) for value in arrays.values())
                and all(event['action_override_agents'] == [] for event in events),
                'long development arrays, events or NN overrides differ')
        initial=_asset(entry['initial_environment'],inputs);final=_asset(entry['final_environment'],inputs)
        require(initial==Path(str(stem)+'-initial.json') and final==Path(str(stem)+'-final.json'),'quality private physics assets relocated')
        verify_physics(case,events,entry['initial_environment'],entry['final_environment'])
        checks = sum(verify_frame(models[variant], case, event, arrays, i)
                     for i, event in enumerate(events))
        outcome = replay_public_episode(case, events, len(events), actor_ref['sha256'])
        outcome.update(variant=variant, actor=actor_ref, complete=True, NN_action_checks=checks, action_overrides=0)
        outcomes.append(outcome)
        action_checks += checks
    require(outcomes == result['outcomes'], 'long development public counters differ from producer outcomes')
    metrics = task_metrics(outcomes,plan['cases'],plan['fixed']['actors'])
    programs = score_programs(plan['final_programs'], plan['fixed']['actor_role_sha256'], result['entries'], plan['cases'])
    require(metrics == result['metrics'] and programs == result['program_metrics']
            and result['task_retention_passed'] is metrics['task_retention_passed']
            and result['development_pair_gate_passed'] is
                (metrics['task_retention_passed'] and programs['passed']),
            'long development task/program metrics or combined verdict do not reproduce')
    require(checked(result['plan'])[0] == plan, 'long development prerequisites changed during replay')
    for ref in inputs:
        _asset(ref, [])
    return dict(version=AUDIT_VERSION, quality_result=result_ref,supervisor_exit=supervisor_exit_ref,
        completed_assets=plan['fixed']['report'],cohort=plan['cohort'],seed_plan=plan['seed_plan'],
        phase=plan['phase'],stream_id=plan['stream_id'],joint_step_budget=plan['joint_step_budget'],
        quality_cases_sha256=digest(plan['cases']),
        plan=result['plan'], input_asset_refs=inputs,development_pair_gate_passed=metrics['task_retention_passed'] and programs['passed'],
        metrics=metrics,task_metrics=metrics,program_metrics=programs, independent_scenes=75, complete_episodes=225,
        replayed_role_actions=action_checks, new_environment_resets=0, NN_updates=0,
        program_fit_calls=0, qualification_changes=0, training_authority=False,
        sealed_authority=False, model_acceptance_passed=False, pilot_opened=False,
        source_files=source_files(),source_sha256=file_sha(__file__))
