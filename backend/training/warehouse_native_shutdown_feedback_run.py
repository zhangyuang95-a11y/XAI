"""Finite same-source observed197 PPO versus genuine RCPD feedback development.

Both arms retain the exact source learner. Validation and extraction have
separate finite accounts; extraction never controls a runtime action. A lost
sampling acknowledgement stops this run. No final-test, UI or release is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import contextlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save, reserve_sampling
from .warehouse_native_partner_mix_run import write_bytes, write_json, decode, save_batch, export_actor, AUTHORIZATION, AUTHORIZATION_SHA
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_native_cycle_budget import CycleBudget
from . import warehouse_native_continuation_run as continuation
from . import warehouse_native_shutdown_continuation_run as source_loader
from . import warehouse_native_shutdown_initial_rcpd_run as initial_tree_reader
from . import warehouse_native_shutdown_stage_rcpd as extraction
from . import warehouse_native_expanded_rcpd as expanded
from . import warehouse_native_expanded_tree_read as expanded_reader
from env.warehouse.layouts import get_map_layout
from hashlib import sha256
from .warehouse_native_initial_rcpd_run import save_episode, binding, make_contexts
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager
from dataclasses import asdict
import gzip
import io

VERSION = 'warehouse-native-own-shutdown-feedback-run.v2'
BRANCHES = ('control', 'feedback')


def sources():
    from .warehouse_native_shutdown_feedback_trainer import execution_sources
    from .warehouse_native_shutdown_stage_evaluation import execution_sources as eval_sources
    result = source_loader.sources(); result.update(execution_sources()); result.update(eval_sources())
    result.update(initial_tree_reader.sources()); result.update(extraction.execution_sources())
    result.update(expanded_reader.runner.sources()); result.update(expanded.execution_sources(extraction))
    reader_path=Path(expanded_reader.__file__);result[str(reader_path.relative_to(ROOT))]=file_hash(reader_path)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _json(path): return json.loads(Path(path).read_bytes())
def _same(a,b,why):
    if digest(a) != digest(b): raise ValueError(why)
def _bound(root,path): return {'path':str(Path(path).relative_to(root)), 'sha256':file_hash(path), 'size':Path(path).stat().st_size}
def _check(root,binding): return continuation._check_file(root,binding)
def _gate_eligible(report):
    return report.get('capability',{}).get('eligible') is True or report.get('warmup_capability',{}).get('eligible') is True

def _gate_values(report, reference):
    partners = ('skilled','assertive','noisy')
    return {'validation_score': float(np.mean([report['summary'][p]['mean_team_deliveries'] for p in partners])),
        'reference_score': float(np.mean([reference['summary'][p]['mean_team_deliveries'] for p in partners])),
        'capability_eligible': _gate_eligible(report)}


def _contexts(pools,horizon):
    if set(pools)!={'train','selection'}:raise ValueError('Only registered train and selection pools are accepted')
    # Explicit order survives canonical JSON sorting and cannot change seeds/episode IDs.
    return make_contexts({name:pools[name] for name in ('train','selection')},horizon)


def _first_bundle(path, expected_sha, *, fixture=False):
    if fixture:
        raise ValueError('Synthetic initial-tree transport requires an explicit test injection; it is never a production reader')
    return expanded_reader.read_bundle(path, expected_manifest_sha256=expected_sha, require_reliable=True)


def _source(source_run, shutdown_arm, step, *, device, fixture):
    if fixture:
        raise ValueError('Synthetic qualified source transport requires explicit test injection')
    descriptor,protocol,scenes,actor,bindings,report,proof = initial_tree_reader.source_material(source_run,shutdown_arm,step)
    learner,actual,saved_scenes = source_loader.load_source(source_run,shutdown_arm,step,device=device,
        expected=descriptor,allow_test_fixture=False)
    _same(initial_tree_reader._without_counts(descriptor),actual,'Source descriptor changed during genuine restoration')
    _same(scenes,saved_scenes,'Actual source scenarios differ')
    return learner,actual,scenes,report,descriptor['validation_report_sha256']


def _install(trainer, fit, actor_bindings, report, report_sha, reference):
    if fit.get('reliable') is not True: raise ValueError('A reliable same-Actor initial tree is required')
    trainer.attach_feedback_state(fit['manager_state'], source_actor_sha256=actor_bindings['actor_sha256'],
        source_actor_parameters_sha256=actor_bindings['actor_parameters_sha256'], evidence_sha256=fit['evidence_sha256'])
    if not trainer.feedback.reliable: raise ValueError('The accepted tree manager is not reliable')
    return trainer.update_feedback_gate(**_gate_values(report,reference),capability_report_sha256=report_sha)


def _validate_first(bundle, descriptor, state, report_sha, fixture):
    plan,fit=bundle['extraction_plan'],bundle['fit_result']
    _same(initial_tree_reader._without_counts(bundle['source_descriptor']),initial_tree_reader._without_counts(descriptor),
        'Expanded first tree belongs to another selected source')
    if (bundle['source_report_sha256']!=report_sha or plan.get('test_fixture') is not fixture
            or plan.get('version')!=expanded_reader.runner.VERSION
            or plan.get('maximum_rcpd_calls')!=13 or plan.get('maximum_sklearn_fits')!=104):
        raise ValueError('Expanded extraction source, finite search or fixture differs')
    _same(plan.get('extraction_contract'),expanded.contract(),'Expanded extraction contract differs')
    _same(plan.get('actor_bindings'),bundle['actor_bindings'],'Expanded plan and returned Actor bindings differ')
    if (fit.get('version')!=expanded.VERSION or fit.get('reliable') is not True or fit.get('test_fixture') is not fixture
            or fit.get('prediction_semantics')!=expanded.program_batch.VERSION
            or bundle['manifest'].get('status')!='completed'
            or bundle['audit'].get('reliable') is not True or bundle['audit'].get('verified_candidates')!=13):
        raise ValueError('Expanded initial RCPD evidence is incomplete or unreliable')
    clock=state['source_counters']['joint_steps']+state['joint_steps']
    if plan.get('cumulative_fit_step')!=clock:raise ValueError('Expanded fit belongs to another source clock')
    actor_weights={k[len('actor.'):]:v for k,v in state['model'].items() if k.startswith('actor.')}
    if semantic(actor_weights)!=bundle['actor_bindings']['actor_parameters_sha256']:
        raise ValueError('Expanded tree was not extracted from these exact source Actor weights')
    if descriptor.get('actor_sha256')!=bundle['actor_bindings']['actor_sha256']:
        raise ValueError('Expanded tree used another original Actor export')
    return fit


def _physical_fingerprint(entry,configuration):
    """Recompute the frozen initial-content fingerprint directly from saved public state; no reset/step."""
    snapshot=entry['snapshot'];state=snapshot['state']
    _same(snapshot.get('configuration'),configuration,'Pool initial-state configuration differs')
    if type(state.get('frame')) is not int or state['frame']!=0 or state.get('terminated') is not False or state.get('truncated') is not False:
        raise ValueError('Only original live frame-zero pool entries may be used')
    layout=get_map_layout(configuration['map_layout_id'])
    content={'layout':layout.tiles,'charger':layout.charger_position,'horizon':configuration['horizon'],
        'agents':[{'position':a['position'],'battery':a['battery'],'active':a['active'],'carrying':a['carrying_task_id'] is not None}
                  for a in state['agents']],
        'tasks':sorted([{'pickup':t['pickup_position'],'delivery':t['delivery_position'],'status':t['status'],'carrier':t['carrier_agent_id']}
            for t in state['tasks']],key=lambda t:(t['pickup'],t['delivery']))}
    value=sha256(json.dumps(content,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if entry.get('fingerprint')!=value:raise ValueError('Pool physical initial-state fingerprint differs')
    return value


def _refresh_pools(bundle,scenes,fixture):
    original,extra=bundle['original_scene_pools'],bundle['expanded_train_pools']
    if set(original)!={'train','selection'} or set(extra)!={'train','selection'} or extra['selection']!=[]:
        raise ValueError('Expanded evidence may only append training scenes')
    first_count,extra_count,selection_count=len(original['train']),len(extra['train']),len(original['selection'])
    if (first_count<1 or extra_count<1 or (not fixture and (first_count,extra_count,selection_count)!=(64,128,32))):
        raise ValueError('The fixed 64 plus 128 train / 32 original selection pools are required')
    pools={'train':deepcopy(original['train'])+deepcopy(extra['train']),'selection':deepcopy(original['selection'])}
    _same(original['train'],scenes['splits']['train'][:first_count],'Original training prefix changed')
    _same(extra['train'],scenes['splits']['train'][first_count:first_count+extra_count],'Expanded training append order changed')
    _same(original['selection'],scenes['splits']['extraction'][:selection_count],'Original selection pool changed')
    groups={};all_ids=set()
    for pool,split in (('train','train'),('selection','extraction')):
        ids=[entry['id'] for entry in pools[pool]]
        if (any(not x.startswith(split+'_') for x in ids) or len(set(ids))!=len(ids) or all_ids&set(ids)
                or any(entry.get('split',split)!=split or entry.get('test_fixture',False) is not fixture for entry in pools[pool])):
            raise ValueError('Refresh pool identity or role differs')
        all_ids.update(ids)
        values=[_physical_fingerprint(entry,scenes['configuration']) for entry in pools[pool]]
        if len(set(values))!=len(values):raise ValueError('Duplicate physical state inside refresh pool')
        groups[pool]=set(values)
    if groups['train']&groups['selection']:raise ValueError('Training and selection physical states overlap')
    # Other registered split fingerprints are exclusion metadata only; no held-out trajectory is read or evaluated.
    for split,entries in scenes['splits'].items():
        if split=='train':continue
        if groups['train']&{row['fingerprint'] for row in entries}:raise ValueError('Expanded training overlaps a held-out pool')
        if split!='extraction' and groups['selection']&{row['fingerprint'] for row in entries}:
            raise ValueError('Original selection overlaps another held-out pool')
    return pools


def _separate(output, *inputs):
    if output.exists(): raise FileExistsError('Use an independent new paired directory')
    for item in inputs:
        item=Path(item).resolve()
        if output==item or output in item.parents or item in output.parents:
            raise ValueError('Paired output must be separate from its source evidence')


def prepare(output, *, source_run, shutdown_arm, source_step=250000, initial_tree, initial_tree_manifest_sha256,
            ppo_cap=250000, validation_interval=50000, cycle_id=None, device='mps', allow_test_fixture=False,
            feedback_config=None):
    from .warehouse_native_shutdown_feedback_trainer import ShutdownFeedbackTrainer, make_protocol
    output=Path(output).expanduser().resolve();source_run=Path(source_run).expanduser().resolve();initial_tree=Path(initial_tree).expanduser().resolve()
    _separate(output,source_run,initial_tree)
    if file_hash(AUTHORIZATION)!=AUTHORIZATION_SHA: raise ValueError('Autonomous authorization changed')
    if (type(ppo_cap) is not int or ppo_cap<16 or type(validation_interval) is not int or validation_interval<1
            or type(allow_test_fixture) is not bool or device not in ('cpu','mps')): raise ValueError('Invalid finite paired preparation')
    if not allow_test_fixture and (ppo_cap!=250000 or validation_interval!=50000 or source_step!=250000):
        raise ValueError('Production feedback fixes the 250000 source, per-arm cap, and 50000 refresh interval')
    first=_first_bundle(initial_tree,initial_tree_manifest_sha256,fixture=allow_test_fixture)
    if first['fit_result'].get('reliable') is not True:raise ValueError('Reliable initial tree is required before source loading')
    source,descriptor,scenes,report,report_sha=_source(source_run,shutdown_arm,source_step,device=device,fixture=allow_test_fixture)
    source_branch=source.branch
    if source.shutdown_arm!=shutdown_arm:raise ValueError('The retained source beta differs')
    state=source.state_dict();state_sha=semantic(state);fit=_validate_first(first,descriptor,state,report_sha,allow_test_fixture)
    n=source.cfg['environments']
    if ppo_cap%n or validation_interval%n: raise ValueError('PPO boundaries must align with environment batch size')
    endpoints=list(range(validation_interval,ppo_cap,validation_interval))+[ppo_cap]
    probe=min(4096,ppo_cap);probe-=probe%n
    if probe<n: raise ValueError('A complete probe rollout is required')
    protocol=make_protocol(source,source_checkpoint_sha256=descriptor['checkpoint']['sha256'],source_state_sha256=state_sha,
        ppo_cap=ppo_cap,cycle_id=cycle_id or output.name,evaluation_checkpoints=endpoints,
        feedback_config=feedback_config,test_fixture=allow_test_fixture)
    pools=_refresh_pools(first,scenes,allow_test_fixture);horizon=scenes['configuration']['horizon']
    contexts=_contexts(pools,horizon)
    one_aux=sum(c['horizon'] for c in contexts)
    if not allow_test_fixture and (len(pools['train'])!=192 or len(pools['selection'])!=32 or one_aux!=107520):
        raise ValueError('Production refresh must use the fixed 192/32 pools and four profiles')
    evalcap=len(endpoints)*3*len(scenes['splits']['validation'])*horizon
    runtime=sources();baseline=_json(source_run/'baselines.json')
    initial_binding={'root':str(initial_tree),'manifest_sha256':initial_tree_manifest_sha256,
        'fit_sha256':digest(fit),'actor_bindings':deepcopy(first['actor_bindings']),
        'extraction_plan_sha256':digest(first['extraction_plan']),
        'original_scene_pools_sha256':digest(first['original_scene_pools']),
        'expanded_train_pools_sha256':digest(first['expanded_train_pools'])}
    prepared={'version':VERSION,'cycle_id':protocol['cycle_id'],'source_branch':source_branch,'shutdown_arm':shutdown_arm,'own_shutdown_beta':source.beta,'device':device,
        'source_descriptor':descriptor,'source_checkpoint_sha256':descriptor['checkpoint']['sha256'],
        'source_state_sha256':state_sha,'source_report_sha256':report_sha,'protocol_sha256':digest(protocol),
        'scenario_manifest_sha256':digest(scenes),'runtime_sources':runtime,'initial_tree':initial_binding,
        'baselines_sha256':file_hash(source_run/'baselines.json'),'authorization_record_sha256':AUTHORIZATION_SHA,
        'budget_caps':{b:{'ppo':ppo_cap,'evaluation':evalcap} for b in BRANCHES},
        'auxiliary_caps':{f'step_{s:07d}':one_aux for s in endpoints},'maximum_auxiliary_steps':one_aux*len(endpoints),
        'refresh_pools_sha256':digest(pools),'refresh_contexts_sha256':digest(contexts),
        'extraction_contract':expanded.contract(),'maximum_rcpd_calls_per_refresh':13,'maximum_sklearn_fits_per_refresh':104,
        'refresh_pool_counts':{'original_train':len(first['original_scene_pools']['train']),'expanded_train':len(first['expanded_train_pools']['train']),
            'combined_train':len(pools['train']),'selection':len(pools['selection'])},
        'primary_endpoint':ppo_cap,'validation_endpoints':endpoints,'probe_endpoint':probe,'test_fixture':allow_test_fixture,
        'performance_guard':'mean_team_deliveries_across_fixed_three_partners_relative_to_initial_reference_ratio',
        'admission':'verified_selected_own_shutdown_and_reliable_expanded_same_actor_tree',
        'created_unix':time.time(),'formal_ready':False,'explanation_qualified':False}
    prepared['identity']={k:deepcopy(v) for k,v in prepared.items() if k not in ('created_unix','formal_ready','explanation_qualified')}
    trainers={b:ShutdownFeedbackTrainer(protocol,source,feedback_branch=b,
        expected_source_state_sha256=state_sha,source_checkpoint_sha256=prepared['source_checkpoint_sha256'],
        device=device,test_fixture=allow_test_fixture) for b in BRANCHES}
    _install(trainers['feedback'],fit,initial_binding['actor_bindings'],report,report_sha,baseline['reference'])
    _same(semantic(trainers['control'].native.state_dict()),semantic(trainers['feedback'].native.state_dict()),'Initial arms differ in native learning state')
    matched={}
    native=trainers['control'].native.state_dict()
    for key in ('model','optimizers','envs','rng','python_rng','numpy_rng','torch_rng','partner_kinds','program_roles',
                'scenario_ids','episode_context','episode_returns','episode_reward_components'):
        matched[key]=semantic(native[key]);_same(matched[key],semantic(state[key]),'Source learning state changed: '+key)
    if device=='mps':_same(semantic(native['mps_rng']),semantic(state['mps_rng']),'Source MPS RNG changed')
    if sources()!=runtime:raise ValueError('Sources changed during preparation')
    output.mkdir(parents=True,exist_ok=False)
    for name,value in [('protocol',protocol),('prepared',prepared),('scenarios',scenes),('refresh_pools',pools),
            ('refresh_contexts',contexts),('initial_fit',fit)]:write_json(output/(name+'.json'),value)
    write_bytes(output/'source_validation.json',(source_run/'branches'/shutdown_arm/'validation'/f'step_{source_step:07d}'/'report.json').read_bytes())
    write_bytes(output/'baselines.json',(source_run/'baselines.json').read_bytes());write_bytes(output/'authorization.json',AUTHORIZATION.read_bytes())
    for name in runtime:write_bytes(output/'source_snapshot'/name,(ROOT/name).read_bytes())
    ledger=CycleBudget.create(output,prepared['identity'])
    with ledger.lease():
        for branch,trainer in trainers.items():
            checkpoint=output/'branches'/branch/'checkpoints/initial.pt'
            atomic_torch_save(checkpoint,{'version':VERSION,'cycle_id':protocol['cycle_id'],'branch':branch,
                'source_branch':source_branch,'shutdown_arm':shutdown_arm,'operation_id':None,'trainer':trainer.state_dict(),'new_environment_steps':0})
            ledger.initialize_head('ppo',branch,str(checkpoint.relative_to(output)),file_hash(checkpoint))
            ep=output/'branches'/branch/'initial_evaluation_reference.json';write_json(ep,{'source_descriptor':descriptor,'new_environment_steps':0})
            ledger.initialize_head('evaluation',branch,str(ep.relative_to(output)),file_hash(ep))
    write_json(output/'initialization_check.json',{'identical_native_learning_state':True,'matched':matched,
        'source_state_sha256':state_sha,'new_environment_steps':0,'program_runtime_controller':False})
    return {'status':'prepared','output':str(output),'budget_caps':prepared['budget_caps'],
        'maximum_auxiliary_steps':prepared['maximum_auxiliary_steps'],'initial_extraction_budget_is_separate':True}


def read_prepared(output, *, allow_test_fixture=False):
    output=Path(output).expanduser().resolve();p=_json(output/'prepared.json')
    if p.get('version')!=VERSION or p.get('test_fixture') is not allow_test_fixture or p['runtime_sources']!=sources():
        raise ValueError('Paired version, sources or fixture boundary changed')
    for key,value in p['identity'].items():_same(p.get(key),value,'Paired identity changed: '+key)
    if file_hash(AUTHORIZATION)!=AUTHORIZATION_SHA or p['authorization_record_sha256']!=AUTHORIZATION_SHA:
        raise ValueError('Paired autonomous authorization changed')
    for name,key in [('authorization.json','authorization_record_sha256'),('baselines.json','baselines_sha256'),('source_validation.json','source_report_sha256')]:
        if file_hash(output/name)!=p[key]:raise ValueError('Bound paired input changed: '+name)
    for name,sha in p['runtime_sources'].items():
        if file_hash(output/'source_snapshot'/name)!=sha:raise ValueError('Archived paired execution source changed')
    protocol,scenes,pools,contexts=(_json(output/(n+'.json')) for n in ('protocol','scenarios','refresh_pools','refresh_contexts'))
    for value,key in [(protocol,'protocol_sha256'),(scenes,'scenario_manifest_sha256'),(pools,'refresh_pools_sha256'),(contexts,'refresh_contexts_sha256')]:
        if digest(value)!=p[key]:raise ValueError('Paired bound configuration changed: '+key)
    _same(contexts,_contexts(pools,scenes['configuration']['horizon']),'Refresh context order or local seeds differ')
    descriptor=p['source_descriptor']; source,actual,saved_scenes,report,report_sha=_source(descriptor['run'],descriptor['shutdown_arm'],descriptor['step'],device=p['device'],fixture=allow_test_fixture)
    _same(actual,descriptor,'Paired source descriptor changed');_same(scenes,saved_scenes,'Paired scenarios changed')
    if (source.shutdown_arm!=p['shutdown_arm'] or source.beta!=p['own_shutdown_beta']
            or semantic(source.state_dict())!=p['source_state_sha256'] or report_sha!=p['source_report_sha256']):raise ValueError('Exact paired source state changed')
    first=_first_bundle(p['initial_tree']['root'],p['initial_tree']['manifest_sha256'],fixture=allow_test_fixture)
    fit=_validate_first(first,descriptor,source.state_dict(),report_sha,allow_test_fixture)
    _same(pools,_refresh_pools(first,scenes,allow_test_fixture),'Registered expanded refresh pools changed')
    for value,key in ((first['extraction_plan'],'extraction_plan_sha256'),(first['original_scene_pools'],'original_scene_pools_sha256'),
                      (first['expanded_train_pools'],'expanded_train_pools_sha256')):
        _same(digest(value),p['initial_tree'][key],'Original expanded tree input changed: '+key)
    _same(p['initial_tree']['actor_bindings'],first['actor_bindings'],'Initial expanded Actor binding changed')
    _same(p['extraction_contract'],expanded.contract(),'Prepared RCPD contract changed')
    _same(p['refresh_pool_counts'],{'original_train':len(first['original_scene_pools']['train']),
        'expanded_train':len(first['expanded_train_pools']['train']),'combined_train':len(pools['train']),
        'selection':len(pools['selection'])},'Expanded pool counts changed')
    if p['maximum_rcpd_calls_per_refresh']!=13 or p['maximum_sklearn_fits_per_refresh']!=104:
        raise ValueError('Prepared finite fit counts changed')
    if digest(fit)!=p['initial_tree']['fit_sha256'] or digest(_json(output/'initial_fit.json'))!=digest(fit):raise ValueError('Initial feedback evidence changed')
    cap=protocol['budget']['maximum_ppo_joint_steps_per_arm'];endpoints=protocol['evaluation']['checkpoints_ppo_steps']
    _same(p['validation_endpoints'],endpoints,'Fixed paired endpoints changed')
    one_eval=3*len(scenes['splits']['validation'])*scenes['configuration']['horizon'];one_aux=sum(c['horizon'] for c in contexts)
    _same(p['budget_caps'],{b:{'ppo':cap,'evaluation':len(endpoints)*one_eval} for b in BRANCHES},'Paired sampling caps differ')
    _same(p['auxiliary_caps'],{f'step_{s:07d}':one_aux for s in endpoints},'Refresh caps differ')
    if p['maximum_auxiliary_steps']!=one_aux*len(endpoints):raise ValueError('Auxiliary total cap differs')
    initial=_json(output/'initialization_check.json')
    if initial.get('identical_native_learning_state') is not True or initial['source_state_sha256']!=p['source_state_sha256']:
        raise ValueError('Paired initialization was not completed')
    return p,protocol,scenes,source


def auxiliary_accounting(output,prepared):
    folder=output/'branches/feedback/refresh';reserved=actual=0;reports={}
    if folder.exists():
        for child in folder.iterdir():
            if not child.is_dir() or child.name not in prepared['auxiliary_caps']:raise ValueError('Unregistered auxiliary sampling directory')
            plan=_json(child/'plan.json');budget=_json(child/'auxiliary_budget.json');manifest=_json(child/'manifest.json')
            if (plan['maximum_auxiliary_steps']!=prepared['auxiliary_caps'][child.name] or budget['cap']!=plan['maximum_auxiliary_steps']
                    or manifest['plan_sha256']!=digest(plan) or budget['reserved_joint_steps']!=manifest['reserved_auxiliary_steps']
                    or any(e['status']!='completed' for e in manifest['episodes'])):
                raise ValueError('Unconfirmed auxiliary sampling needs diagnosis; never resample')
            if plan.get('pair_identity_sha256')!=digest(prepared['identity']):raise ValueError('Refresh belongs to another pair')
            check_manifest(child,plan,manifest,fixture=prepared['test_fixture'])
            reserved+=budget['reserved_joint_steps'];actual+=manifest['actual_auxiliary_steps']
            reports[child.name]={'reserved':budget['reserved_joint_steps'],'acknowledged':manifest['actual_auxiliary_steps']}
    if reserved>prepared['maximum_auxiliary_steps']:raise ValueError('Paired auxiliary cap exceeded')
    return {'cap':prepared['maximum_auxiliary_steps'],'reserved':reserved,'acknowledged':actual,'boundaries':reports}


def effective_checkpoint(output,ledger,branch):
    head=ledger.head('ppo',branch);path=_check(output,head);payload=decode(path,head['sha256'])
    if payload.get('version')!=VERSION or payload.get('branch')!=branch:raise ValueError('Wrong acknowledged feedback checkpoint')
    if head['step']:
        operation=ledger.read()['operations'].get(payload.get('operation_id'))
        if (not operation or operation['status']!='acknowledged' or operation['completion']['checkpoint']!=head
                or payload.get('audit',{}).get('neural_overrides')!=0):raise ValueError('Missing original PPO acknowledgment')
    for binding in payload.get('evidence',{}).values():_check(output,binding)
    marker=output/'branches'/branch/'boundaries'/f"step_{head['step']:07d}.json"
    if not marker.exists():
        marker=marker.with_name(marker.stem+'_refresh_begin.json')
    if marker.exists():
        rec=_json(marker);_same(rec['predecessor_head'],head,'Boundary refers to another acknowledged checkpoint')
        if (rec['branch']!=branch or rec['version']!=VERSION
                or rec.get('new_training_steps')!=0):raise ValueError('Boundary condition differs')
        for binding in rec['evidence'].values():_check(output,binding)
        bpath=_check(output,rec['checkpoint']);boundary=decode(bpath,rec['checkpoint']['sha256'])
        if (boundary.get('version')!=VERSION or boundary.get('branch')!=branch
                or boundary.get('predecessor_head')!=head or boundary.get('evidence')!=rec['evidence']):raise ValueError('Boundary checkpoint binding differs')
        _same(semantic(payload['trainer']['native_state']),semantic(boundary['trainer']['native_state']),'Boundary changed native learner state')
        payload=boundary
    return head,payload


def train_to(output,trainer,ledger,branch,target):
    if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
        raise ValueError('Unconfirmed paired operation requires diagnosis; never resample')
    n=trainer.cfg['environments']
    while trainer.joint_steps<target:
        if trainer.joint_steps in trainer.protocol['evaluation']['checkpoints_ppo_steps']:
            marker=output/'branches'/branch/'boundaries'/f'step_{trainer.joint_steps:07d}.json'
            if not marker.exists():raise ValueError('Fixed boundary validation and refresh must finish before further PPO')
        if any(ledger.read()['branches'][branch][kind]['pending'] for kind in ('ppo','evaluation')):
            raise ValueError('Unconfirmed paired operation requires diagnosis; never resample')
        amount=min(n*trainer.cfg['rollout_steps'],target-trainer.joint_steps,ledger.remaining('ppo',branch));amount-=amount%n
        if amount<=0:raise ValueError('Reserved budget cannot reach this fixed paired endpoint')
        head,old=effective_checkpoint(output,ledger,branch);before=semantic(trainer.state_dict())
        if head['step']!=trainer.joint_steps or semantic(old['trainer'])!=before:raise ValueError('Learner differs from confirmed paired predecessor')
        opid='ppo_'+uuid.uuid4().hex;reservation=ledger.reserve('ppo',branch,amount,opid,expected_step=trainer.joint_steps)
        if not reservation['execution_permitted']:raise ValueError('Repeated reservation is not execution permission')
        # The latest completed validation remains the ramp/performance evidence
        # between fixed refreshes. Every current-state ramp change is checkpointed.
        capability=trainer.capability_evidence
        if trainer.feedback_enabled and capability is not None and trainer.feedback.reliable:
            gate=capability['gate'];trainer.update_feedback_gate(gate['validation_score'],
                capability_eligible=capability['capability_eligible'],reference_score=gate['reference_score'],
                capability_report_sha256=capability['report_sha256'])
        started=time.monotonic();batch,metrics=trainer.train_chunk(amount//n)
        if (trainer.joint_steps!=head['step']+amount or len(batch['transition_records'])!=amount
                or batch['audit']['neural_overrides']!=0 or any(not np.isfinite(v) for v in metrics.values())
                or any(r.get('feedback_branch')!=branch or r.get('shutdown_arm')!=trainer.shutdown_arm
                       or r.get('own_shutdown_beta')!=trainer.beta for r in batch['transition_records'])
                or (branch=='control' and metrics.get('feedback_loss',0)!=0)):
            raise ValueError('Actual paired training differs from reserved NN-only contract')
        evidence=save_batch(output,branch,opid,batch);episodes=deepcopy(trainer.completed_episodes);trainer.completed_episodes.clear()
        path=output/'branches'/branch/'checkpoints'/(f'step_{trainer.joint_steps:07d}_'+opid+'.pt')
        payload={'version':VERSION,'cycle_id':trainer.protocol['cycle_id'],'branch':branch,'source_branch':trainer.branch,'shutdown_arm':trainer.shutdown_arm,
            'operation_id':opid,'trainer':trainer.state_dict(),'audit':batch['audit'],'metrics':metrics,'evidence':evidence,
            'actual_steps':amount,'completed_episodes':episodes,'before_state_sha256':before,'elapsed_seconds':time.monotonic()-started}
        atomic_torch_save(path,payload);ledger.ack(opid,str(path.relative_to(output)),file_hash(path),amount)
        write_json(output/'progress.json',{'status':'training','branch':branch,'steps':trainer.joint_steps,
            'metrics':metrics,'ledger':ledger.read()},replace=True)
        print(json.dumps({'event':'paired_ppo_ack','branch':branch,'steps':trainer.joint_steps,
            'lambda':metrics.get('feedback_lambda',0),'neural_overrides':batch['audit']['neural_overrides']}),flush=True)


def evaluate_boundary(output,trainer,ledger,branch,scenes):
    from .warehouse_native_shutdown_stage_evaluation import evaluate, required_bindings
    actor=export_actor(output,trainer,branch,ledger.head('ppo',branch))
    bindings={key:actor.metadata[key] for key in required_bindings(trainer.protocol) if key!='actor_sha256'};bindings['actor_sha256']=actor.artifact_sha256
    baseline=_json(output/'baselines.json')
    confirmed={opid for opid,op in ledger.read()['operations'].items() if op['status']=='acknowledged' and op['request']['kind']=='evaluation' and op['request']['branch']==branch}
    def reserve(context):
        op=ledger.reserve('evaluation',branch,context['horizon'],context['operation_id'])
        if not op['execution_permitted']:raise ValueError('Already reserved validation cannot run again')
    def ack(result):ledger.ack(result['operation_id'],str(Path(result['row_path']).relative_to(output)),result['row_sha256'],result['actual_steps'])
    cap=3*len(scenes['splits']['validation'])*scenes['configuration']['horizon']
    report=evaluate(actor,scenes['splits']['validation'],trainer.protocol,
        output/'branches'/branch/'validation'/f'step_{trainer.joint_steps:07d}',cap,expected_bindings=bindings,
        reference_report=baseline['reference'],random_report=baseline['random'],before_episode=reserve,on_episode=ack,
        confirmed_operation_ids=confirmed,allow_test_fixture=trainer.test_fixture,
        config=collaborative_study_config(horizon=scenes["configuration"]["horizon"]))
    return report,actor


def _report_record(output,condition,step,prepared,protocol,scenes,account,fixture):
    """Original saved matrix + actual external acknowledgments, never fresh evaluation."""
    from . import warehouse_native_shutdown_stage_evaluation as evaluation
    folder=output/"branches"/condition/"validation"/f"step_{step:07d}"
    report=_json(folder/"report.json");manifest=_json(folder/"manifest.json")
    binding=report["actor_bindings"]
    if (binding["joint_steps"]!=step or binding["shutdown_arm"]!=prepared["shutdown_arm"] or binding["feedback_branch"]!=condition
            or binding["protocol_sha256"]!=prepared["protocol_sha256"]
            or binding["source_checkpoint_sha256"]!=prepared["source_checkpoint_sha256"]
            or binding["initialization_sha256"]!=prepared["source_state_sha256"]):
        raise ValueError("Stored validation belongs to another learner boundary")
    actor=output/"branches"/condition/"actors"/f"actor_{step:07d}.npz"
    parity=_json(actor.with_suffix(".json"))
    matches=[op for op in account["operations"].values() if op["status"]=="acknowledged"
        and op["request"]["kind"]=="ppo" and op["request"]["branch"]==condition
        and op["completion"]["checkpoint"]["step"]==step]
    if (len(matches)!=1 or parity["checkpoint_sha256"]!=matches[0]["completion"]["checkpoint"]["sha256"]
            or parity["actor"]["sha256"]!=binding["actor_sha256"] or parity.get("argmax_equal") is not True
            or not 0<=parity["maximum_absolute_error"]<=1e-4):raise ValueError("Bound endpoint parity is incomplete")
    continuation._check_file(output,matches[0]["completion"]["checkpoint"])
    confirmed=set()
    for entry in manifest["episodes"]:
        context=entry["context"];op=account["operations"].get(context["operation_id"])
        if (not op or op["status"]!="acknowledged" or op["request"]["branch"]!=condition
                or op["request"]["kind"]!="evaluation" or op["request"]["steps"]!=context["horizon"]):
            raise ValueError("Validation episode lacks its external reservation")
        rowpath=continuation._check_file(folder,entry["row"])
        row=_json(rowpath);completion=op["completion"]
        if (completion["checkpoint"]["path"]!=str(rowpath.relative_to(output))
                or completion["checkpoint"]["sha256"]!=entry["row"]["sha256"]
                or completion["actual_steps"]!=row["steps"]):raise ValueError("Validation row was not acknowledged")
        confirmed.add(context["operation_id"])
    baseline=_json(output/"baselines.json")
    maximum=3*len(scenes["splits"]["validation"])*scenes["configuration"]["horizon"]
    return evaluation.read_completed(actor,scenes["splits"]["validation"],protocol,folder,maximum,
        expected_bindings=binding,reference_report=baseline["reference"],random_report=baseline["random"],
        confirmed_operation_ids=confirmed,expected_report_sha256=file_hash(folder/"report.json"),allow_test_fixture=fixture,
        config=collaborative_study_config(horizon=scenes["configuration"]["horizon"]))


def load_episode(root, entry, *, fixture=False):
    data = json.loads(gzip.decompress(initial_tree_reader.bound_bytes(root, entry["record"])))
    with np.load(io.BytesIO(initial_tree_reader.bound_bytes(root, entry["arrays"])), allow_pickle=False) as values:
        if set(values.files) != {"observations", "probabilities"}: raise ValueError("Unexpected extraction arrays")
        data.update({key: values[key].copy() for key in values.files})
    extraction._validate_data(data, fixture)
    context, episode = entry["context"], data["episode"]
    if (data["data_sha256"] != entry["data_sha256"] or data["joint_transitions"] != entry["actual_joint_steps"]
            or len(data["observations"]) != entry["neural_rows"] or data["pool"] != context["pool"]
            or episode["id"] != context["episode_id"] or episode["fingerprint"] != context["scenario_fingerprint"]
            or any(episode[k] != context[k] for k in ("profile", "program_role", "seed", "sampling_mode", "horizon"))
            or digest(data["trace"]) != episode["trace_sha256"]):
        raise ValueError("Acknowledged extraction episode differs")
    return data


def check_manifest(output, plan, manifest, *, fixture=False):
    if (plan.get("test_fixture") is not fixture or manifest.get("version") != VERSION
            or manifest.get("plan_sha256") != digest(plan) or manifest.get("ppo_steps") != 0
            or plan.get("version") != VERSION): raise ValueError("Extraction manifest identity differs")
    entries = manifest["episodes"]
    if len(entries) > len(plan["contexts"]): raise ValueError("Too many extraction episodes")
    actual = reserved = 0
    for index, entry in enumerate(entries):
        if entry["context"] != plan["contexts"][index]: raise ValueError("Extraction order changed")
        reserved += entry["context"]["horizon"]
        if entry["status"] != "completed": raise ValueError("Pending extraction episode cannot be replayed automatically")
        data = load_episode(output, entry, fixture=fixture)
        if data["actor_bindings"] != plan["actor_bindings"]: raise ValueError("Episode Actor differs")
        if not 0 < data["joint_transitions"] <= entry["context"]["horizon"]: raise ValueError("Episode exceeded reservation")
        actual += data["joint_transitions"]
    budget = _json(Path(output) / "auxiliary_budget.json")
    if (manifest["actual_auxiliary_steps"] != actual or manifest["reserved_auxiliary_steps"] != reserved
            or budget["reserved_joint_steps"] != reserved or budget["cap"] != plan["maximum_auxiliary_steps"]
            or reserved > plan["maximum_auxiliary_steps"]): raise ValueError("Extraction reservations and confirmations differ")
    return manifest


def merge_saved(output, entries, pool, *, fixture=False):
    """Stream traces one at a time; retain only the 197-row arrays and receipts."""
    merged = None; arrays = {"observations": [], "probabilities": []}; seen = set(); receipts = []; first = None
    for entry in entries:
        if entry["context"]["pool"] != pool: continue
        data = load_episode(output, entry, fixture=fixture); receipt = data["collector_receipt"]
        if merged is None:
            merged = {k: deepcopy(data[k]) for k in ("version", "pool", "actor_bindings", "feature_names", "teacher_rows_in_fit",
                "neural_submitted_overrides", "test_fixture", "extraction_config_sha256", "actor_training_clock")}
            merged.update(episode_ids=[], groups=[], row_sources=[], scene_fingerprints=set(), joint_transitions=0, episodes=[])
            first = deepcopy(receipt)
        elif (any(digest(merged[k]) != digest(data[k]) for k in ("pool", "actor_bindings", "feature_names", "actor_training_clock"))
              or any(digest(first[k]) != digest(receipt[k]) for k in ("actor_metadata", "weights_sha256", "configuration", "training_stage"))):
            raise ValueError("Stored episodes have different Actor, physics or pool bindings")
        if data["episode"]["id"] in seen: raise ValueError("Duplicate stored episode")
        seen.add(data["episode"]["id"])
        for key in arrays: arrays[key].append(data[key])
        for key in ("episode_ids", "groups", "row_sources"): merged[key].extend(data[key])
        merged["scene_fingerprints"].update(data["scene_fingerprints"])
        merged["joint_transitions"] += data["joint_transitions"]
        merged["episodes"].append({**deepcopy(data["episode"]), "data_sha256": data["data_sha256"]})
        receipts.append({"episode": deepcopy(data["episode"]), "collector_receipt_sha256": data["collector_receipt_sha256"]})
        del data
    if merged is None: raise ValueError("No acknowledged data for extraction pool")
    for key, chunks in arrays.items(): merged[key] = np.concatenate(chunks)
    merged["scene_fingerprints"] = sorted(merged["scene_fingerprints"])
    merged["data_sha256"] = extraction.data_api._data_digest(merged)
    contract = {k: first[k] for k in ("weights_sha256", "configuration", "actor_training_clock", "training_stage")}
    contract["metadata"] = first["actor_metadata"]
    extraction._receipt(merged, contract=contract, sources=extraction.execution_sources(), episodes=receipts)
    extraction._validate_data(merged, fixture)
    return merged


def collect_dataset(output, plan, pools, actor, protocol, *, until_episodes=None, fixture=False, config=None):
    """Private fixture-capable primitive. The public runner authenticates a real source first."""
    output = Path(output); manifest = check_manifest(output, plan, _json(output / "manifest.json"), fixture=fixture)
    limit = len(plan["contexts"]) if until_episodes is None else until_episodes
    if type(limit) is not int or not len(manifest["episodes"]) <= limit <= len(plan["contexts"]): raise ValueError("Invalid episode limit")
    for context in plan["contexts"][len(manifest["episodes"]):limit]:
        reserved = reserve_sampling(output / "auxiliary_budget.json", context["horizon"], cap=plan["maximum_auxiliary_steps"])
        if reserved != manifest["reserved_auxiliary_steps"] + context["horizon"]: raise ValueError("Reservation sequence differs")
        pending = deepcopy(manifest)
        pending["episodes"].append({"status": "pending", "context": deepcopy(context)})
        pending.update(status="sampling", reserved_auxiliary_steps=reserved)
        write_json(output / "manifest.json", pending, replace=True); manifest = pending
        data = None
        try:
            data = extraction.collect_episode(actor, pools[context["pool"]][context["scene_index"]],
                **{k: context[k] for k in ("pool", "profile", "program_role", "seed", "episode_id", "sampling_mode")},
                expected_bindings=plan["actor_bindings"], protocol=protocol, allow_test_fixture=fixture, config=config)
            if not 0 < data["joint_transitions"] <= context["horizon"]: raise ValueError("Episode exceeds reservation")
            saved = save_episode(output, context, data)
            confirmed = deepcopy(manifest); confirmed["episodes"][-1] = saved
            confirmed["actual_auxiliary_steps"] += data["joint_transitions"]
            write_json(output / "manifest.json", confirmed, replace=True)
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "message": str(error), "context": context,
                "known_completed_joint_steps": getattr(error, "actual_joint_steps", data["joint_transitions"] if data is not None else None),
                "known_environment_step_calls": getattr(error, "environment_step_calls", data["joint_transitions"] if data is not None else None),
                "reserved_horizon": context["horizon"], "sampling_replay_allowed": False,
                "confirmation_not_claimed": True}
            # Failure journaling must not replace the acknowledged manifest with
            # an in-memory confirmation. Preserve the original exception too.
            try: write_json(output / "sampling_failure.json", failure)
            except BaseException as journal_error: error.audit_journal_write_error = str(journal_error)
            error.extraction_failure = failure
            raise
        manifest = confirmed
        print(json.dumps({"event": "shutdown_extraction_episode_ack", "episode": len(manifest["episodes"]),
            "episodes": len(plan["contexts"]), "actual_auxiliary_steps": manifest["actual_auxiliary_steps"]}), flush=True)
    if limit == len(plan["contexts"]) and manifest.get("status") not in ("completed", "fitting"):
        manifest["status"] = "sampling_complete"; write_json(output / "manifest.json", manifest, replace=True)
    return manifest


def _fit_request(plan,manifest,actor,prior,config):
    return {'version':VERSION,'plan_sha256':digest(plan),'acknowledged_episodes_sha256':digest(manifest['episodes']),
        'prior_manager_state_sha256':digest(prior),'feedback_config_sha256':digest(asdict(config)),
        'actor_sha256':actor.artifact_sha256,'cumulative_fit_step':plan['cumulative_fit_step'],
        'maximum_candidate_fits':len(expanded.CANDIDATES),'maximum_sklearn_fits':sum(d for d,_ in expanded.CANDIDATES),
        'extraction_contract_sha256':digest(expanded.contract()),'retry_allowed':False}


def fit_collected(output,manifest,plan,actor,*,prior_manager_state,feedback_config,fixture=False):
    _same(asdict(feedback_config),asdict(expanded.feedback_config()),'Expanded refresh FeedbackConfig differs')
    output=Path(output);check_manifest(output,plan,manifest,fixture=fixture)
    if len(manifest['episodes'])!=len(plan['contexts']):raise ValueError('Cannot fit incomplete current-Actor collection')
    datasets={pool:merge_saved(output,manifest['episodes'],pool,fixture=fixture) for pool in ('train','selection')}
    request=_fit_request(plan,manifest,actor,prior_manager_state,feedback_config)
    request['dataset_receipts']={k:v['collector_receipt_sha256'] for k,v in datasets.items()}
    if (output/'fit_request.json').exists():
        if (manifest.get('fit_request') is None or json.loads(initial_tree_reader.bound_bytes(output,manifest['fit_request']))!=request):
            raise ValueError('Cached fit request differs or was not acknowledged')
        if manifest.get('status')!='completed' or manifest.get('fit_result') is None:
            raise ValueError('Pending fit requires diagnosis; never repeat fitting')
        result=json.loads(initial_tree_reader.bound_bytes(output,manifest['fit_result']))
        if result.get('test_fixture') is not fixture or result.get('version')!=expanded.VERSION:
            raise ValueError('Saved stage fit provenance differs')
        return result
    write_json(output/'fit_request.json',request)
    manifest.update(status='fitting',fit_request=binding(output,output/'fit_request.json'))
    write_json(output/'manifest.json',manifest,replace=True)
    try:
        result=extraction.fit_feedback(datasets['train'],datasets['selection'],step=plan['cumulative_fit_step'],
            feature_names=actor.metadata['feature_names'],feedback_config=feedback_config,
            prior_manager_state=prior_manager_state,allow_test_fixture=fixture)
    except ValueError as error:
        result={'version':expanded.VERSION,'prediction_semantics':expanded.program_batch.VERSION,'reliable':False,'fit_report':{'reason':str(error)},'manager_state':None,'program':None,
            'actual_joint_steps':0,'neural_training_updates':0,'test_fixture':fixture,'explanation_qualified':False,'release_eligible':False,
            'input_rejection':{'error_type':type(error).__name__,'request_sha256':digest(request)},
            'evidence_sha256':digest({'request':request,'reason':str(error)})}
    except BaseException as error:
        failure={'request_sha256':digest(request),'error_type':type(error).__name__,'message':str(error),
            'fit_replay_allowed':False,'maximum_candidate_fits':request['maximum_candidate_fits'],
            'maximum_sklearn_fits':request['maximum_sklearn_fits'],'actual_candidate_fits_unknown':True}
        try:write_json(output/'fit_failure.json',failure)
        except BaseException as journal_error:error.audit_journal_write_error=str(journal_error)
        raise
    write_json(output/'fit_result.json',result)
    confirmed=deepcopy(manifest);confirmed.update(status='completed',fit_result=binding(output,output/'fit_result.json'),
        reliable=result['reliable'],explanation_qualified=False)
    write_json(output/'manifest.json',confirmed,replace=True);manifest.clear();manifest.update(confirmed)
    return result


def refresh(output,prepared,trainer,actor,prior):
    folder=output/'branches/feedback/refresh'/f'step_{trainer.joint_steps:07d}'
    if folder.name not in prepared['auxiliary_caps']:raise ValueError('Unregistered refresh boundary')
    bindings={key:actor.metadata[key] for key in extraction.required_bindings(trainer.protocol) if key!='actor_sha256'}
    bindings.update(actor_sha256=actor.artifact_sha256,cycle_id=trainer.protocol['cycle_id'],feedback_branch='feedback')
    pools=_json(output/'refresh_pools.json');contexts=_json(output/'refresh_contexts.json')
    plan={'version':VERSION,'pair_version':VERSION,'pair_identity_sha256':digest(prepared['identity']),
        'actor_bindings':bindings,'protocol_sha256':digest(trainer.protocol),'scene_pools_sha256':digest(pools),
        'contexts':contexts,'maximum_auxiliary_steps':prepared['auxiliary_caps'][folder.name],
        'cumulative_fit_step':trainer.feedback_clock,'ppo_steps':0,'test_fixture':trainer.test_fixture,
        'prior_manager_sha256':digest(prior),'extraction_config':extraction.extraction_config(),
        'expanded_fit_contract':expanded.contract(),'maximum_rcpd_calls':13,'maximum_sklearn_fits':104,
        'runtime_sources':prepared['runtime_sources']}
    if folder.exists():_same(_json(folder/'plan.json'),plan,'Refresh plan or prior manager changed')
    else:
        folder.mkdir(parents=True,exist_ok=False);write_json(folder/'plan.json',plan)
        reserve_sampling(folder/'auxiliary_budget.json',0,cap=plan['maximum_auxiliary_steps'])
        write_json(folder/'manifest.json',{'version':VERSION,'plan_sha256':digest(plan),'status':'prepared',
            'episodes':[],'actual_auxiliary_steps':0,'reserved_auxiliary_steps':0,'ppo_steps':0})
    manifest=collect_dataset(folder,plan,pools,actor,trainer.protocol,fixture=trainer.test_fixture,
        config=collaborative_study_config(horizon=_json(output/'scenarios.json')['configuration']['horizon']))
    _same(trainer.protocol['feedback_config'],asdict(expanded.feedback_config()),'Learner and expanded fit configurations differ')
    config=expanded.feedback_config()
    fit=fit_collected(folder,manifest,plan,actor,prior_manager_state=prior,
        feedback_config=config,fixture=trainer.test_fixture)
    return fit,bindings,folder


def finish_boundary(output,prepared,trainer,ledger,branch,report,actor):
    head=ledger.head('ppo',branch);folder=output/'branches'/branch/'boundaries';marker=folder/f'step_{trainer.joint_steps:07d}.json'
    if marker.exists():
        _,saved=effective_checkpoint(output,ledger,branch);trainer.load_state_dict(saved['trainer']);return _json(marker)
    original=semantic(trainer.native.state_dict());evidence={}
    validation=output/'branches'/branch/'validation'/f'step_{trainer.joint_steps:07d}'
    evidence['validation_report']=_bound(output,validation/'report.json');evidence['validation_manifest']=_bound(output,validation/'manifest.json')
    gate={'active':False,'lambda':0.,'reason':'control'}
    if branch=='feedback':
        begin=folder/f'step_{trainer.joint_steps:07d}_refresh_begin.json'
        if begin.exists():
            record=_json(begin);_same(record['predecessor_head'],head,'Refresh start belongs to another predecessor')
            saved=decode(_check(output,record['checkpoint']),record['checkpoint']['sha256'])
            _same(semantic(saved['trainer']['native_state']),original,'Refresh start changed native learner')
            trainer.load_state_dict(saved['trainer']);prior=record['prior_manager_state']
            _same(prior,trainer.feedback.state_dict(),'Saved refresh prior differs from closed manager')
        else:
            prior=trainer.begin_feedback_refresh()
            start_path=folder/f'step_{trainer.joint_steps:07d}_refresh_begin.pt'
            if start_path.exists():raise ValueError('Unacknowledged refresh-start checkpoint needs diagnosis')
            start_payload={'version':VERSION,'branch':branch,'cycle_id':trainer.protocol['cycle_id'],
                'predecessor_head':head,'trainer':trainer.state_dict(),'evidence':deepcopy(evidence),'new_training_steps':0}
            atomic_torch_save(start_path,start_payload)
            write_json(begin,{'version':VERSION,'branch':branch,'predecessor_head':head,'checkpoint':_bound(output,start_path),
                'evidence':deepcopy(evidence),'prior_manager_state':prior,'new_training_steps':0})
        fit,bindings,refreshed=refresh(output,prepared,trainer,actor,prior)
        for name in ('fit_result.json','fit_request.json','manifest.json','plan.json','auxiliary_budget.json'):
            evidence['refresh_'+name.replace('.','_')]=_bound(output,refreshed/name)
        if fit.get('reliable') is True:
            gate=_install(trainer,fit,bindings,report,evidence['validation_report']['sha256'],_json(output/'baselines.json')['reference'])
        else:
            trainer.fail_feedback_refresh(fit['fit_report'])
            gate={'active':False,'lambda':0.,'reason':'unreliable_refreshed_program'}
    if semantic(trainer.native.state_dict())!=original:raise ValueError('Validation/extraction mutated native learner')
    path=folder/f'step_{trainer.joint_steps:07d}.pt'
    if path.exists():raise ValueError('Unacknowledged boundary checkpoint needs diagnosis')
    payload={'version':VERSION,'branch':branch,'cycle_id':trainer.protocol['cycle_id'],'predecessor_head':head,
        'trainer':trainer.state_dict(),'evidence':evidence,'gate':gate,'new_training_steps':0}
    atomic_torch_save(path,payload)
    receipt={'version':VERSION,'branch':branch,'predecessor_head':head,'checkpoint':_bound(output,path),
        'evidence':evidence,'gate':gate,'new_training_steps':0}
    write_json(marker,receipt)
    return receipt


def advance(output, *, until=None, allow_test_fixture=False):
    from .warehouse_native_shutdown_feedback_trainer import ShutdownFeedbackTrainer
    output=Path(output).expanduser().resolve();raw=_json(output/'prepared.json')
    until=raw['primary_endpoint'] if until is None else until
    if type(until) is not int or until not in (raw['probe_endpoint'],*raw['validation_endpoints']):
        raise ValueError('Use only the registered paired probe or validation boundaries')
    ledger=CycleBudget(output,raw['identity'])
    with ledger.lease():
        if any(op['status'] in ('pending','abandoned') for op in ledger.read()['operations'].values()):
            raise ValueError('Pending paired sampling requires diagnosis; no automatic replay')
        auxiliary_accounting(output,raw)
        p,protocol,scenes,source=read_prepared(output,allow_test_fixture=allow_test_fixture)
        trainers={};reports={}
        for branch in BRANCHES:
            trainer=ShutdownFeedbackTrainer(protocol,source,feedback_branch=branch,expected_source_state_sha256=p['source_state_sha256'],
                source_checkpoint_sha256=p['source_checkpoint_sha256'],device=p['device'],test_fixture=allow_test_fixture)
            head,payload=effective_checkpoint(output,ledger,branch)
            for old_step in p['validation_endpoints']:
                if old_step>=head['step']:break
                marker=output/'branches'/branch/'boundaries'/f'step_{old_step:07d}.json'
                if not marker.exists():raise ValueError('Missing completed fixed boundary before acknowledged later PPO')
                receipt=_json(marker)
                if receipt['version']!=VERSION or receipt['branch']!=branch:raise ValueError('Earlier boundary identity differs')
                _check(output,receipt['checkpoint'])
                for binding in receipt['evidence'].values():_check(output,binding)
            trainer.load_state_dict(payload['trainer'])
            if trainer.joint_steps!=head['step'] or trainer.joint_steps>until:raise ValueError('Paired counter differs from confirmed head or requested endpoint')
            trainers[branch]=trainer
        for target in sorted(set((p['probe_endpoint'],*p['validation_endpoints']))):
            if target>until:break
            for branch,trainer in trainers.items():
                if trainer.joint_steps>target:continue
                train_to(output,trainer,ledger,branch,target)
                if target in p['validation_endpoints']:
                    report,actor=evaluate_boundary(output,trainer,ledger,branch,scenes)
                    verified=_report_record(output,branch,target,p,protocol,scenes,ledger.read(),allow_test_fixture)
                    _same(report,verified,'Live and acknowledged validation reports differ')
                    finish_boundary(output,p,trainer,ledger,branch,report,actor);reports[branch+':'+str(target)]=report
                    print(json.dumps({'event':'paired_validation_complete','branch':branch,'steps':target,
                        'primary_value':report['primary_value'],'warmup':report['warmup_capability']['eligible']}),flush=True)
        counts={branch:{'ppo_steps':t.joint_steps,'optimizer_updates':t.optimizer_updates,'actor_optimizer_steps':t.actor_optimizer_steps,
            'critic_optimizer_steps':t.critic_optimizer_steps} for branch,t in trainers.items()}
        if counts['control']!=counts['feedback']:raise ValueError('Paired arms did not receive equal PPO steps and update counts')
        result={'status':'fixed_endpoint_completed' if until==p['primary_endpoint'] else 'boundary_completed','until':until,
            'counts':counts,'ledger':ledger.read(),'auxiliary':auxiliary_accounting(output,p),'formal_ready':False,
            'explanation_qualified':False,'total_interaction_equal':False,'control_terminal_tree_extracted':False,
            'initial_tree_budget_is_separate':True,
            'reports':{k:{x:y for x,y in v.items() if x not in ('rows','artifacts')} for k,v in reports.items()}}
        write_json(output/f'completion_{until:07d}.json',result,replace=True);write_json(output/'progress.json',result,replace=True)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    action=parser.add_mutually_exclusive_group(required=True);action.add_argument('--prepare',action='store_true');action.add_argument('--run',action='store_true')
    for name in ('source-run','shutdown-arm','initial-tree','initial-tree-manifest-sha256','cycle-id'):parser.add_argument('--'+name)
    parser.add_argument('--source-step',type=int);parser.add_argument('--ppo-cap',type=int,default=250000)
    parser.add_argument('--validation-interval',type=int,default=50000);parser.add_argument('--until',type=int)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps');args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.prepare:
        if any(getattr(args,k) is None for k in ('source_run','shutdown_arm','source_step','initial_tree','initial_tree_manifest_sha256')):
            parser.error('Preparation requires a registered source and externally bound reliable initial tree')
        result=prepare(args.output,source_run=args.source_run,shutdown_arm=args.shutdown_arm,source_step=args.source_step,
            initial_tree=args.initial_tree,initial_tree_manifest_sha256=args.initial_tree_manifest_sha256,
            ppo_cap=args.ppo_cap,validation_interval=args.validation_interval,cycle_id=args.cycle_id,device=args.device)
    else:
        class Tee:
            def __init__(self,*streams):self.streams=streams
            def write(self,value):
                for stream in self.streams:stream.write(value);stream.flush()
                return len(value)
            def flush(self):
                for stream in self.streams:stream.flush()
        with (Path(args.output)/'stdout.log').open('a') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)):
            result=advance(args.output,until=args.until)
    print(json.dumps({k:v for k,v in result.items() if k not in ('ledger','reports')}),flush=True)


if __name__=='__main__':main()
