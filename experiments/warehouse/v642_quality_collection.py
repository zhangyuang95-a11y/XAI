"""Once-only, fresh matched-scene quality collection after real 20k completion.

Programs do not choose actions. This gate is narrower than final acceptance:
current-NN program fidelity and all subsequent development/formal gates remain.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from copy import deepcopy
import gzip,json,os
import numpy as np
from . import v642_calibration_quality as quality,v642_calibration_pipeline as training
from . import seed_metadata_audit_indexed_v642 as metadata,seed_triage_indexed_v642 as triage,exact_seed_audit as exact
from . import seed_interval_index_v642 as interval_source
from .v626_protocol import require,pointer,bound,digest,file_sha
from .v642_update_receipt_audit import audit_updates
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor,encode_actor_tree_features
from backend.training.warehouse_v622_program import masked_probabilities
from backend.training.warehouse_v6_runner import _set_dual_nn_mode
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.public_observation_v6 import encode_public_observation,public_pre_action_snapshot
from env.warehouse.pure_nn import action_masks
from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
from scripts.run_warehouse_v625 import write_new
from scripts.audit_warehouse_v642_calibration_results import audit_source_files
from scripts.audit_warehouse_v642_confirmation import verify_frame
from scripts.measure_warehouse_v628_collection import episode as replay_public_episode

ROOT=training.owner.ROOT
OUTPUT=ROOT/'output/warehouse_v642/calibration-quality'
VERSION='warehouse-v642-once-only-matched-quality-collection.v1'
REGISTRY_REF=deepcopy(training.owner.seeds.confirmation.INVENTORY_REF)
FILES=('experiments/warehouse/v642_quality_collection.py','experiments/warehouse/v642_calibration_quality.py',
    'experiments/warehouse/v642_update_receipt_audit.py','backend/training/warehouse_v642_checkpoint_export.py',
    'scripts/audit_warehouse_v642_calibration_results.py','experiments/warehouse/seed_interval_index_v642.py',
    'experiments/warehouse/seed_metadata_audit_indexed_v642.py','experiments/warehouse/seed_triage_indexed_v642.py',
    'tests/test_warehouse_v642_calibration_quality.py','tests/test_warehouse_v642_quality_collection.py',
    'scripts/run_warehouse_v642_calibration_quality.py','docs/warehouse_v642_calibration_quality_plan.md')
_ASSET_CACHE={}


def source_files():
    result=training.source_files()
    for name in FILES:result[name]=file_sha(ROOT/name)
    return dict(sorted(result.items()))


def assets(report_ref):
    """Replay source-bound completed-training counters before any new seeds."""
    key=digest(dict(report=report_ref,training_sources=training.source_files(),audit_sources=audit_source_files()))
    if key in _ASSET_CACHE:
        saved=_ASSET_CACHE[key]
        for ref in saved['inputs']:require(pointer(ref['path'])==ref,'training/export evidence asset changed')
        return deepcopy(saved)
    r=bound(report_ref)
    require(r['version']=='warehouse-v642-completed-calibration-assets.v1' and r['numerical_training_completed'] is True
        and r['model_acceptance_passed'] is r['publishable'] is r['pilot_opened'] is False
        and r['source_sha256']==file_sha(ROOT/'scripts/audit_warehouse_v642_calibration_results.py')
        and r.get('audit_source_files')==audit_source_files(),
        'quality collection needs actual audited 20k completion, never a placeholder')
    completed=bound(r['controller_completed']);contract=bound(completed['controller_contract'])
    require(completed['version']==training.VERSION and completed['calibration_training_finished'] is True
        and completed['model_acceptance_passed'] is False and completed['pilot_opened'] is completed['publishable'] is False
        and contract['source_files']==training.source_files(), 'completed controller or source changed')
    folder=Path(r['controller_completed']['path']).parent
    require(not (folder/'failed.json').exists() and r['controller_completed']==pointer(folder/'controller-completed.json'),
        'failed training cannot be quality evaluated')
    observed=bound(r['supervisor_exit'])
    require(observed['exit_code']==0 and observed['controller_completed']==r['controller_completed']
        and observed['observed_by']=='exec_tool_terminal_result',
        'training terminal exit was not actually observed')
    done=bound(completed['owner_completed']);claim_ref=pointer(training.owner.claim_path());claim=bound(claim_ref)
    require(done['claim']==claim_ref and claim['plan']==contract['seed_plan'] and claim['descriptor']==done['descriptor'],
        'quality assets do not belong to the claimed training pair')
    require(folder==Path(done['descriptor']['path']).parent
        and completed['owner_completed']==pointer(folder/'completed.json')
        and completed['controller_contract']==pointer(folder.parent/'controller-source-contract.json'),
        'completed controller/owner assets are outside the actual claimed training directory')
    actors={'reference':bound(done['descriptor'])['reference_actor']};roles={};refs=[report_ref,r['controller_completed'],
        completed['controller_contract'],r['supervisor_exit'],completed['owner_completed'],claim_ref,done['descriptor']]
    for variant in ('raw','regularized'):
        final_ref=completed['checkpoints'][variant][-1];final=bound(final_ref)
        require(final['core']['processed_joint_steps']==20000 and final['updates']==done['updates'][variant],
            'missing final 20k checkpoint or mismatched update list')
        recount=audit_updates(final['updates'],variant=variant,expected_joint_steps=20000,core=final['core'])
        require(recount==r['receipt_audit'][variant], 'accepted program gradient report does not reproduce')
        export=r['exports'][variant]['final'];actors[variant]=export['actor'];roles[variant]=export['role_sha256']
        require(export['checkpoint']==final['training_state'] and export['role_sha256']==final['core']['actor_role_sha256'],
            'frozen final Actor differs from its full checkpoint/core')
        refs.extend([final_ref,final['training_state'],*final['updates'],actors[variant]])
        FrozenWarehouseV6Actor.load(actors[variant]['path'],expected_sha256=actors[variant]['sha256'],expected_role_sha256=roles[variant])
    reference=FrozenWarehouseV6Actor.load(actors['reference']['path'],expected_sha256=actors['reference']['sha256'])
    roles['reference']=deepcopy(reference.role_sha256)
    require(len({ref['sha256'] for ref in actors.values()})==3, 'quality comparison needs three distinct bound Actor assets')
    require(all(len({roles[v][role] for v in quality.VARIANTS})==3 for role in ('robot_1','robot_2')),
        'reference/raw/regularized need distinct actual role weights, not just different NPZ metadata')
    refs.append(actors['reference'])
    require(type(r.get('input_asset_refs')) is list
        and pointer(folder/'owner-journal.jsonl') in r['input_asset_refs']
        and pointer(folder/'actual-resets.jsonl') in r['input_asset_refs'],
        'completed training report must bind actual optimizer/refit and reset journal bytes')
    refs.extend(deepcopy(r['input_asset_refs']))
    for ref in refs:require(pointer(ref['path'])==ref,'training/export evidence asset changed')
    value=dict(report=report_ref,actors=actors,actor_role_sha256=roles,inputs=refs)
    _ASSET_CACHE[key]=deepcopy(value)
    return value


def declaration(rows):
    quality.validate_cases(rows)
    declaration=[]
    for variant in quality.VARIANTS:
        for case in rows:
            for kind,arg,key in (('scenario','task_expansion_base','base'),('env','random.Random(config.seed)','scenario_seed'),
                ('env','reset(seed)','scenario_seed'),('action_rng','FrozenActor.act(base_seed)','action_seed')):
                item=dict(kind=kind,purpose='v642_20k_task_retention',argument=arg,value=case[key],
                    group=variant+':'+case['case_id'],reuse_tag=key+':'+str(case[key]))
                if kind=='action_rng':item['consumed_as_random_draw']=True
                declaration.append(item)
    return dict(schema='warehouse-exact-seed-plan.v1',candidate_id=training.owner.seeds.partitions.admission.v.CANDIDATE+'-20k-quality',
        uses=declaration,cases_sha256=digest(rows),authorization_granted=False)


def prepare(report_ref,*,run_id,base_start,action_start):
    fixed=assets(report_ref)  # Before seed allocation and output creation.
    training.owner.seeds.confirmation.run_name(run_id)
    require(not claim_path().exists(),'this candidate already consumed its calibration quality attempt')
    rows=quality.cases(base_start,action_start);sources=source_files();folder=OUTPUT/run_id
    folder.mkdir(parents=True,exist_ok=False);exact_value=declaration(rows)
    write_new(folder/'exact-plan.json',exact_value);(folder/'exact-plan.json').chmod(0o444)
    exact.read_exact_plan(folder/'exact-plan.json')
    for name,sha in sources.items():
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        raw=(ROOT/name).read_bytes();require(file_sha(ROOT/name)==sha,'quality source changed while preparing')
        target.write_bytes(raw);target.chmod(0o444)
    value=dict(version=VERSION,run_id=run_id,fixed=fixed,cases=rows,recipe=quality.recipe(),source_files=sources,
        exact_plan=pointer(folder/'exact-plan.json'),training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(folder/'plan.json',value);(folder/'plan.json').chmod(0o444)
    return pointer(folder/'plan.json')


def checked(plan_ref):
    p=bound(plan_ref);folder=OUTPUT/p['run_id']
    require(Path(plan_ref['path']).resolve()==folder/'plan.json' and p['version']==VERSION
        and p['recipe']==quality.recipe() and p['source_files']==source_files()
        and p['training_authority'] is p['sealed_authority'] is p['pilot_opened'] is False, 'quality plan/source/scope changed')
    quality.validate_cases(p['cases'])
    require(p['fixed']==assets(p['fixed']['report']) and bound(p['exact_plan'])==declaration(p['cases']),
        'quality training prerequisites or exact original arguments changed')
    for ref in p['fixed']['inputs']:require(pointer(ref['path'])==ref,'quality prerequisite asset changed')
    for name,sha in p['source_files'].items():
        q=folder/'source'/name;require(file_sha(q)==sha and not q.stat().st_mode&0o222,'quality source snapshot changed')
    return p,folder


def history_roots():
    return [ROOT.parent/name for name in training.owner.seeds.confirmation.HISTORY_NAMES]


def template_bindings(rows,*,execution):
    recorded=training.owner.seeds.template_bindings(rows,check_current=False)
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
        [folder],registered_trace_paths=traces,selection_stats=stats):
        stat=path.stat();selected.append(dict(path=str(path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
    return selected,stats['excluded_weights_name_stat_only'],registry,traces


def audit_seeds(plan_ref):
    p,folder=checked(plan_ref);require(not claim_path().exists(),'spent quality attempt cannot re-audit')
    out=folder/'seed-ledger';out.mkdir(exist_ok=False)
    families=exact.compile_families(bound(p['exact_plan']));_,_,registry,traces=history(p,folder)
    proposal=dict(schema='warehouse-seed-proposal.v1',candidate_id=bound(p['exact_plan'])['candidate_id'],
        families=[{k:r[k] for k in ('name','kind','values','query_only')} for r in families],
        exclude_paths=[str(folder)],registered_trace_paths=traces)
    write_new(out/'proposal.json',proposal)
    metadata.audit(out/'proposal.json',history_roots(),out/'bounded')
    triage.triage(out/'bounded/inventory.json',out/'proposal.json',out/'triaged')
    result=exact.assess(p['exact_plan']['path'],out/'triaged/result.json',out/'exact',context_source_root=ROOT)
    training.owner.seeds.conflict_free(result)
    selected,weights,_,_=history(p,folder)
    audited={r['path']:(r['bytes'],r.get('mtime_ns')) for r in result['inventory'] if r['classification']!='source_template_environment_reservation'}
    require(audited=={r['path']:(r['bytes'],r['mtime_ns']) for r in selected},'history changed during quality seed audit')
    templates=template_bindings(result['inventory'],execution=True)
    checked(plan_ref);now=datetime.now(timezone.utc)
    value=dict(version=VERSION,plan=plan_ref,exact=pointer(out/'exact/result.json'),registry=registry,traces=traces,
        selection=selected,weights=weights,templates=templates,issued_utc=now.isoformat(),expires_utc=(now+timedelta(hours=24)).isoformat(),
        environment_reset_ceiling=225,training_authority=False,sealed_authority=False,pilot_opened=False)
    write_new(out/'clearance.json',value);(out/'clearance.json').chmod(0o444)
    return pointer(out/'clearance.json')


def clearance(plan_ref,*,execution):
    p,folder=checked(plan_ref);ref=pointer(folder/'seed-ledger/clearance.json');c=bound(ref);e=bound(c['exact'])
    issued=datetime.fromisoformat(c['issued_utc']);expires=datetime.fromisoformat(c['expires_utc'])
    require(c['version']==VERSION and c['plan']==plan_ref and c['environment_reset_ceiling']==225
        and c['training_authority'] is c['sealed_authority'] is c['pilot_opened'] is False
        and issued.tzinfo and expires.tzinfo and expires-issued==timedelta(hours=24)
        and (not execution or issued<=datetime.now(timezone.utc)<expires),'quality clearance scope/binding/lifetime changed')
    require(e['exact_plan']==p['exact_plan'] and e['exact_auditor_sha256']==file_sha(exact.__file__), 'quality exact input/source changed')
    inv=bound(e['bounded_or_triaged_inventory']);projection=exact.project_inventory(inv,exact.compile_families(bound(p['exact_plan'])))
    if e.get('context_annotations'):projection=exact.apply_context(projection,bound(e['context_annotations']))
    require(e['summary']==projection['summary'] and digest(e['inventory'])==digest(projection['inventory']), 'quality exact audit failed replay')
    training.owner.seeds.conflict_free(e)
    roots=sorted(str(root.resolve()) for root in history_roots())
    require(sorted(e['scan_roots'])==sorted(inv['scan_roots'])==roots and not inv['missing_roots']
        and e['selection_scope']==inv['selection_scope']==metadata.SELECTION_SCOPE
        and sorted(inv['scope_exclusions'])==sorted(map(str,[folder/'seed-ledger/proposal.json',folder/'seed-ledger/bounded',folder]))
        and inv['read_model_weight_files'] is e['read_model_weight_files'] is False
        and e['scope_exclusions']==inv['scope_exclusions']
        and inv['auditor_sha256']==file_sha(metadata.__file__) and inv['triage_driver_sha256']==file_sha(triage.__file__)
        and inv['interval_index_source_sha256']==file_sha(interval_source.__file__), 'quality audit scope/source changed')
    registry=REGISTRY_REF
    require(c['registry']==registry and pointer(registry['path'])==registry
        and c['traces']==e['registered_trace_paths']==inv['registered_trace_paths']==bound(registry)['registered_trace_paths'], 'registered quality history traces changed')
    if execution:
        selected,weights,_,_=history(p,folder)
        require(selected==c['selection'] and weights==c['weights'],'historical metadata changed before quality collection')
        require(template_bindings(e['inventory'],execution=True)==c['templates'],'source reservations changed')
    return ref,c


def claim_path():return training.owner.seeds.partitions.admission.v.OUTPUT/'calibration-quality.claim.json'


def collect(plan_ref):
    p,folder=checked(plan_ref);clearance_ref,c=clearance(plan_ref,execution=True)
    require(not claim_path().exists(),'calibration quality collection is once-only even after interruption')
    write_new(claim_path(),dict(version=VERSION,plan=plan_ref,clearance=clearance_ref,pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(),single_use_even_if_interrupted=True,pilot_opened=False))
    claim_path().chmod(0o444);out=folder/'collection';out.mkdir(exist_ok=False)
    try:
        models={v:FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'],
            expected_role_sha256=p['fixed']['actor_role_sha256'][v]) for v,ref in p['fixed']['actors'].items()}
        results=[];entries=[]
        with (out/'reset-preauthorization.jsonl').open('x') as journal,capture_stage1_resets(out/'actual-resets.jsonl') as resets:
            for case in p['cases']:
                for variant in quality.VARIANTS:
                    checked(plan_ref);require(pointer(clearance_ref['path'])==clearance_ref
                        and datetime.now(timezone.utc)<datetime.fromisoformat(c['expires_utc']), 'quality sources/clearance changed or expired')
                    auth=dict(case=case,variant=variant,plan=plan_ref,actor=p['fixed']['actors'][variant],authorized_utc=datetime.now(timezone.utc).isoformat())
                    journal.write(json.dumps(auth)+'\n');journal.flush();os.fsync(journal.fileno())
                    with stage1_reset_phase('quality'):
                        env=WarehouseMultiAgentEnv(participant_study_config(seed=case['scenario_seed']));env.reset(seed=case['scenario_seed'])
                    _set_dual_nn_mode(env);actor=models[variant];ref=p['fixed']['actors'][variant]
                    rows=[];arrays={key:[] for key in ('x','m','p','sampled')}
                    while not(env.state.terminated or env.state.truncated):
                        before=public_pre_action_snapshot(env.state);obs=encode_public_observation(env.state);m=action_masks(env.state)
                        actions,_=actor.act(env.state,base_seed=case['action_seed'],deterministic=False)
                        probabilities=masked_probabilities(actor.logits(obs,m),m)
                        from env.warehouse.navigation import ACTIONS
                        indices=[ACTIONS.index(actions[r]) for r in ('robot_1','robot_2')]
                        _,_,_,_,info=env.step(actions)
                        require(info['proposed_actions']==actions and not info['action_override_agents'],'quality execution overrode a complete-NN action')
                        rows.append(dict(case=case,frame=len(rows),pre_state_public=before,post_state_public=public_pre_action_snapshot(env.state),
                            proposed_actions=actions,actor_sha256=ref['sha256'],action_override_agents=[]))
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
                    entries.append(dict(case=case,variant=variant,trace=pointer(trace),arrays=pointer(array)))
                if len(results)%45==0:print(json.dumps(dict(quality_episodes_completed=len(results),total=225)),flush=True)
        require(len(resets)==225 and [r['seed'] for r in resets]==[case['scenario_seed'] for case in p['cases'] for _ in quality.VARIANTS], 'quality reset inventory/order differs')
        checked(plan_ref);metrics=quality.evaluate(results,p['cases'],p['fixed']['actors'])
        result=dict(version=VERSION,plan=plan_ref,claim=pointer(claim_path()),clearance=clearance_ref,metrics=metrics,
            outcomes=results,entries=entries,actual_resets=pointer(out/'actual-resets.jsonl'),reset_preauthorization=pointer(out/'reset-preauthorization.jsonl'),
            actual_reset_count=225,task_retention_passed=metrics['task_retention_passed'],
            program_fidelity_evaluation_required=True,model_acceptance_passed=False,pilot_opened=False)
        write_new(out/'result.json',result);(out/'result.json').chmod(0o444)
        write_new(out/'completed.json',dict(result=pointer(out/'result.json')));(out/'completed.json').chmod(0o444)
        return pointer(out/'result.json')
    except BaseException as error:
        write_new(out/'failed.json',dict(version=VERSION,plan=plan_ref,error_type=type(error).__name__,error=str(error),
            single_use_claim_retained=True,automatic_retry=False,pilot_opened=False));raise
