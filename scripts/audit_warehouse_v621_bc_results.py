#!/usr/bin/env python3
"""Independent trace arithmetic for an entirely completed finite BC batch.

No reset, environment construction, fitter, optimizer, model reconstruction or
selection is called. Native receipt checking is reused after the reporter has
established all6/conditional9 completion; program arithmetic is separate NumPy.
The native autograd receipt is bound and interpreted, never called a PPO update.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.warehouse import v621_bc_protocol as protocol
from experiments.warehouse.v621_bc_audit_math import audit_program
ACTIONS=('UP','DOWN','LEFT','RIGHT','WAIT')
ROLES=('robot_1','robot_2')


def digest(value):return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_identity(arrays):
    h=sha256()
    for name,item in sorted(arrays.items()):
        a=np.ascontiguousarray(item);h.update(name.encode());h.update(a.dtype.str.encode());h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes())
    return h.hexdigest()


class Evidence:
    def __init__(self,root):self.root=Path(root).resolve(strict=True);self.bound={}
    def raw(self,path,expected=None):
        p=Path(path);p=p if p.is_absolute() else self.root/p;p=p.resolve(strict=True)
        if not p.is_relative_to(self.root) or not p.is_file():raise RuntimeError('audit input outside checkout')
        data=p.read_bytes();actual=sha256(data).hexdigest()
        if expected is not None and expected!=actual:raise RuntimeError('audit input SHA mismatch: '+str(p))
        if str(p) in self.bound and self.bound[str(p)]!=actual:raise RuntimeError('audit input changed')
        self.bound[str(p)]=actual;return data
    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def pointer(self,ref):return self.json(ref['path'],ref['sha256'])
    def verify(self):
        for path,value in self.bound.items():
            if sha256(Path(path).read_bytes()).hexdigest()!=value:raise RuntimeError('audit input changed during review')


def compare_common(actual,expected,prefix='metric'):
    """Compare actual native fields; retain richer independent diagnostics."""
    compared=[]
    for key,a in actual.items():
        if key not in expected:continue
        b=expected[key];label=prefix+'.'+key
        if isinstance(a,dict) and isinstance(b,dict):compared.extend(compare_common(a,b,label));continue
        if type(a) is int or type(a) is bool or a is None or isinstance(a,str):
            if type(a) is not type(b) or a!=b:raise RuntimeError(label+' differs')
        elif type(a) is float:
            if b is None or not math.isfinite(a) or not np.isclose(a,b,atol=2e-6,rtol=2e-5):raise RuntimeError(label+' differs')
        else:
            if a!=b:raise RuntimeError(label+' differs')
        compared.append(label)
    return compared


def masked(logits,masks,control=False):
    z=np.asarray(logits,np.float32 if control else np.float64);m=np.asarray(masks)>0.5
    if z.shape!=m.shape or z.shape[-1]!=5 or not np.isfinite(z).all() or np.any(~m.any(-1)):raise RuntimeError('invalid fullNN mask/distribution')
    z=np.where(m,z,-1e9) if control else np.where(m,z,-np.inf)
    exp=np.exp(z-z.max(-1,keepdims=True));p=exp/exp.sum(-1,keepdims=True)
    if np.any(p[~m]!=0) or not np.isfinite(p).all():raise RuntimeError('NN mask normalization differs')
    return p


def journal_audit(e,folder,initialization,recipe,spec,spec_sha256):
    # Independent expansion; no reset_schedule/capability execution is called.
    seeds=protocol.seed_plan()['initializations'][str(initialization)];expected=[]
    for phase,_ in protocol.STAGES:
        if ('public_teacher' in phase)!=(recipe=='shared_cache'):continue
        expected.extend(dict(phase=phase,**case) for case in seeds['stages'][phase])
    if recipe=='shared_cache':expected.extend({'phase':'critic_setup','scenario_seed':s,'action_seed':None} for s in seeds['critic_environment_reset_seeds'])
    else:expected.extend(dict(phase='spent_development_evaluation',scenario_seed=c['scenario_seed'],action_seed=c['action_seed']) for c in protocol.diagnostics.source_cases())
    pre=[json.loads(line) for line in e.raw(folder/'reset-preauthorization.jsonl').splitlines()]
    native=[json.loads(line) for line in e.raw(folder/'native-successful-resets.jsonl').splitlines()]
    start=e.json(folder/'capability.started.json');cap=e.json(folder/'capability.completed.json');entry=e.json(folder/'completed.json')
    for value in (start,cap):
        if value['initialization']!=initialization or value['recipe']!=recipe:raise RuntimeError('capability identity differs')
    if (start['source_closure_sha256']!=spec['source_closure_sha256'] or start['specification_sha256']!=spec_sha256
        or start['scheduled_reset_count']!=len(expected) or cap['preauthorized_resets']!=len(expected)
        or entry['actual_reset_count']!=len(expected) or entry['source_closure_sha256']!=spec['source_closure_sha256']):raise RuntimeError('cap reset counts/source/spec differ')
    if e.pointer(entry['capability_completed'])!=cap:raise RuntimeError('entry/cap completion pointer differs')
    if len(pre)!=len(native) or len(pre)!=len(expected):raise RuntimeError('incomplete reset journal')
    for i,(a,b,c) in enumerate(zip(pre,native,expected,strict=True)):
        if a['index']!=i or {k:a[k] for k in ('phase','scenario_seed','action_seed')}!=c or b['seed']!=c['scenario_seed'] or b['phase']!='bc':raise RuntimeError('ordered pre/native reset differs')
    if cap['preauthorization_journal_sha256']!=e.bound[str((folder/'reset-preauthorization.jsonl').resolve())]:raise RuntimeError('cap journal SHA differs')
    return {'actual_resets':len(native),'preauthorized_resets':len(pre),'phase_counts':dict(Counter(r['phase'] for r in expected)),'native_action_seed_recorded':False,'action_seed_verified_in_prejournal_and_development_trace':True}


def cache_audit(e,folder,initialization,spec,spec_sha256):
    cache=e.json(folder/'cache.manifest.json');entry=e.json(folder/'completed.json')
    if e.pointer(entry['result'])!=cache or cache['status']!='completed' or cache['initialization']!=initialization:raise RuntimeError('shared cache incomplete')
    if cache['source_closure_sha256']!=spec['source_closure_sha256'] or cache['seed_plan_sha256']!=spec['seed_plan']['sha256'] or cache['model_implementation_sha256']!=spec['model_implementation_sha256']:raise RuntimeError('shared cache binding differs')
    if set(cache['teacher_assets'])!=set(protocol.TEACHER_STAGES):raise RuntimeError('teacher stages differ')
    assets={}
    for name,ref in cache['teacher_assets'].items():
        if Path(ref['path']).resolve().parent!=folder.resolve():raise RuntimeError('teacher asset is outside its initialization cache')
        raw=e.raw(ref['path'],ref['sha256'])
        from io import BytesIO
        with np.load(BytesIO(raw),allow_pickle=False) as z:identity=array_identity({key:z[key] for key in z.files})
        if identity!=ref['data_sha256'] or Path(ref['path']).stat().st_mode&0o222:raise RuntimeError('teacher semantic hash or readonly differs')
        assets[name]={'sha256':ref['sha256'],'data_sha256':identity}
    e.raw(cache['critic_asset']['path'],cache['critic_asset']['sha256']);critic=e.pointer(cache['critic_report'])
    if critic['model_sha256']!=cache['critic_asset']['model_sha256'] or Path(cache['critic_asset']['path']).stat().st_mode&0o222:raise RuntimeError('critic cache identity/readonly differs')
    return {'manifest':{'path':str(folder/'cache.manifest.json'),'sha256':e.bound[str((folder/'cache.manifest.json').resolve())]},'teacher_assets':assets,'critic_model_sha256':critic['model_sha256'],'journals':journal_audit(e,folder,initialization,'shared_cache',spec,spec_sha256)}


def trace_rows(e,collection,actor,artifact):
    from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state,encode_public_observation
    from env.warehouse.pure_nn import action_masks
    from env.warehouse.public_action_features_v617 import encode_actor_tree_features
    from env.warehouse.decision_protocol import independent_agent_seed
    cases=protocol.diagnostics.source_cases()
    if collection['cases']!=cases or [r['case'] for r in collection['outcomes']]!=cases:raise RuntimeError('full275 cohort differs')
    parts={s:{r:[] for r in ROLES} for s in protocol.diagnostics.SPLITS};quality=[];checks=0
    for case,outcome in zip(cases,collection['outcomes'],strict=True):
        ref=collection['traces'][str(case['base'])];events=[json.loads(line) for line in e.raw(ref['path'],ref['sha256']).splitlines()]
        if not 1<=len(events)<=120:raise RuntimeError('episode frames missing')
        last=None;deliveries=collisions=0;shutdown=False
        for frame,event in enumerate(events):
            pre,post=event['pre_state_public'],event['post_state_public']
            if (event['frame']!=frame or pre['frame']!=frame or post['frame']!=frame+1 or event['episode_id']!=pre['episode_id']
                or post['episode_id']!=pre['episode_id'] or event['case']!=case or event['actor_sha256']!=artifact['sha256']
                or event['post_policy_interventions']!=0 or case['sampling_mode']=='argmax' and case['original_split']!='fit'):raise RuntimeError('trace metadata/frame binding differs')
            if digest(pre)!=event['pre_state_public_sha256'] or digest(post)!=event['post_state_public_sha256'] or last is not None and last!=digest(pre):raise RuntimeError('public state chain broken')
            last=digest(post);state=rebuild_public_pre_action_state(pre);obs=encode_public_observation(state);masks=action_masks(state)
            before_ids={t['task_id'] for t in pre['tasks']};after_ids={t['task_id'] for t in post['tasks']};removed=before_ids-after_ids
            if len(removed)!=len(after_ids-before_ids):raise RuntimeError('public delivery replacement count differs')
            for task in pre['tasks']:
                if task['task_id'] in removed and not any(a['carrying_task_id']==task['task_id'] and a['position']==task['delivery_position'] for a in post['agents']):
                    # Delivery clears carrying ID. Precarrier and its resulting
                    # delivery position establish the physical completion.
                    carrier=next((a['agent_id'] for a in pre['agents'] if a['carrying_task_id']==task['task_id']),None)
                    if carrier is None or next(a for a in post['agents'] if a['agent_id']==carrier)['position']!=task['delivery_position']:raise RuntimeError('removed task lacks physical delivery')
            deliveries+=len(removed);collisions+=bool(post['last_robot_collision_event'])
            terminal_shutdown=any(a['active'] and not b['active'] for a,b in zip(pre['agents'],post['agents'],strict=True))
            if terminal_shutdown and frame!=len(events)-1:raise RuntimeError('trace continues after real shutdown termination')
            shutdown|=terminal_shutdown
            for index,role in enumerate(ROLES):
                row=event['roles'][role]
                if (row['frame']!=frame or row['base_seed']!=case['scenario_seed'] or row['base']!=case['base'] or row['task']!=case['task']
                    or row['episode_id']!=event['episode_id'] or row['episode']!=f'development-spent-base-{case["base"]}' or row['actor_role_sha256']!=artifact['role_sha256'][role]
                    or not np.array_equal(np.asarray(row['obs'],np.float32),obs[index]) or not np.array_equal(np.asarray(row['mask'],np.float32),masks[index])):raise RuntimeError('NN input is not public state/physical mask')
                encoded=encode_actor_tree_features(obs[index][None,:],masks[index][None,:]);p=masked(actor.roles[role].logits_encoded(encoded),masks[index][None,:],control=getattr(actor,'model_version','')=='warehouse-dual-actor.v6.18-quality-selective')[0]
                if not np.allclose(p,row['nn_probabilities'],atol=2e-6,rtol=2e-5):raise RuntimeError('recorded target is not complete boundNN distribution')
                rng=independent_agent_seed(base_seed=case['action_seed'],episode_id=event['episode_id'],frame=frame,agent_id=role)
                selected=int(p.argmax()) if case['sampling_mode']=='argmax' else int(np.random.default_rng(rng).choice(5,p=p))
                if row['rng_seed']!=rng or row['sampled_action']!=ACTIONS[selected] or event['executed_actions'][role]!=ACTIONS[selected] or post['agents'][index]['last_action']!=ACTIONS[selected] or not masks[index,selected]:raise RuntimeError('submitted action differs from boundNN+physical mask/RNG')
                parts[case['original_split']][role].append({**row,'encoded':encoded[0],'nn_probabilities':p});checks+=1
        if len(events)<120 and not shutdown:raise RuntimeError('short trace is not a complete shutdown episode')
        computed={'deliveries':deliveries,'robot_collisions':collisions,'shutdown':shutdown,'physical_frames':len(events)}
        if any(type(outcome[k]) is not type(v) or outcome[k]!=v for k,v in computed.items()):raise RuntimeError('outcome counters differ from public trace')
        if case['original_split']=='validation':quality.append({'case':case,**computed})
    if len(quality)!=75 or any(r['case']['sampling_mode']!='sampled' for r in quality) or collection['NN_action_checks']!=checks or collection['omitted_frames']!=0 or collection['post_policy_interventions']!=0:raise RuntimeError('natural quality/NN frame denominator differs')
    for role in ROLES:
        ids=[{r['episode'] for r in parts[s][role]} for s in protocol.diagnostics.SPLITS];scenes=[{r['base_seed'] for r in parts[s][role]} for s in protocol.diagnostics.SPLITS]
        if any(ids[i]&ids[j] or scenes[i]&scenes[j] for i in range(4) for j in range(i)):raise RuntimeError('whole episode/scenario split overlap')
    if collection['physical_frames_per_role']!={s:len(parts[s][ROLES[0]]) for s in protocol.diagnostics.SPLITS}:raise RuntimeError('split frame denominator differs')
    return parts,{'episodes':75,'mean_deliveries':sum(r['deliveries'] for r in quality)/75,'mean_collisions':sum(r['robot_collisions'] for r in quality)/75,'shutdown_episode_rate':sum(r['shutdown'] for r in quality)/75,'NN_action_checks':checks,'joint_physical_frames':checks//2}


def arrays(rows):
    return (np.stack([r['encoded'] for r in rows]),np.stack([r['mask'] for r in rows]),np.stack([r['nn_probabilities'] for r in rows]),np.asarray([ACTIONS.index(r['sampled_action']) for r in rows]),np.asarray([r['episode'] for r in rows]),np.asarray([r['base_seed'] for r in rows]))

def metric_audit(raw,rows,names,bounds,shared):return audit_program(raw,*arrays(rows),names,bounds,shared=shared)

def development_gate(metrics):
    f,e=metrics['feedback'],metrics['explanation']
    checks={'feedback_coverage':f['coverage']>=.45,'explanation_coverage':e['coverage']>=.45,
        'expected_feedback_coverage':f['expected_actionable_coverage']>=.40,
        'feedback_agreement':f['agreement']>=.95,'explanation_agreement':e['agreement']>=.97,
        'feedback_kl':f['kl'] is not None and f['kl']<=.20,'explanation_kl':e['kl'] is not None and e['kl']<=.20}
    for name,section in (('feedback',f),('explanation',e)):
        for metric in ('whole_tree_raw_illegal_count','whole_tree_path_uncertified_count'):checks[name+'_'+metric]=type(section[metric]) is int and section[metric]==0
    return {'passed':all(checks.values()),'checks':checks}

def qualification_audit(raw,parts,role,names,bounds,shared,fit_context):
    qualified={'feedback':[],'explanation':[]};boundary=[]
    for leaf in range(len(raw['leaf_logits'])):
        singleton=deepcopy(raw);singleton['feedback_leaves']=singleton['explanation_leaves']=[leaf]
        lm=metric_audit(singleton,parts['eligibility'][role],names,bounds,shared)
        has_path=int(np.asarray(raw['leaf_logits'][leaf],np.float32).argmax()) in lm['context']['path_certificates'][leaf]
        for purpose,threshold in (('feedback',.93),('explanation',.95)):
            m=lm[purpose]
            eligible=(m['rows']>=20 and m['distinct_episode_count']>=3 and m['distinct_scenario_seed_count']>=3
                and m['max_episode_fraction']<=.5 and m['agreement']>=threshold and m['agreement_wilson_lower_90']>=threshold-.08
                and m['kl'] is not None and m['kl']<=.25 and m['illegal_count']==m['path_uncertified_count']==0)
            if m['kl'] is not None and abs(m['kl']-.25)<=2e-6:boundary.append({'leaf':leaf,'purpose':purpose,'KL':m['kl']})
            if np.any(fit_context['leaf_indices']==leaf) and has_path and eligible:qualified[purpose].append(leaf)
    if qualified['feedback']!=raw['feedback_leaves'] or qualified['explanation']!=raw['explanation_leaves']:
        raise RuntimeError('independent eligibility qualification differs; inspect scalar numeric boundary evidence '+json.dumps(boundary))
    for split in protocol.diagnostics.SPLITS:
        if (raw['report']['split_episode_ids'][split]!=sorted({str(r['episode']) for r in parts[split][role]})
            or raw['report']['split_scenario_seed_ids'][split]!=sorted({str(r['base_seed']) for r in parts[split][role]})):
            raise RuntimeError('program episode/scenario partition provenance differs')
    return {'sets':qualified,'KL_boundary_rows_reported_not_tolerance_reclassified':boundary}

def probe_audit(record,metric,context,rows,raw,program_hash):
    # Independent scalar q||p KL value on exactly the native512 physical
    # batches. Native norm is bound evidence; this does not call autograd.
    if record['current_fidelity']['program_sha256']!=program_hash or record['current_fidelity']['current_actor_role_sha256']!=raw['actor_role_sha256']:raise RuntimeError('probe identity differs')
    f=metric['feedback'];compare_common(f,record['current_fidelity'],'probe.current_fidelity');formal=(f['coverage']>=.20 and f['actionable_coverage']>=.20 and f['agreement']>=.85 and f['kl'] is not None and f['kl']<=.35)
    dev=(f['coverage']>=.30 and f['actionable_coverage']>=.30 and f['agreement']>=.93 and f['kl'] is not None and f['kl']<=.25)
    zero_keys=('illegal_count','path_uncertified_count','whole_tree_raw_illegal_count','whole_tree_path_uncertified_count','core_KL_bound_violations')
    gate=formal and dev and all(f[k]==0 for k in zero_keys)
    if (record['current_fidelity']['passed'] is not gate or record['program_coefficient']!=.01 or record['clone_virtual_frozen_at_step']!=150000
        or record['post_75_percent_structure_and_core_logits_frozen_clone'] is not True or record['real_PPO_feedback_or_human_benefit_claimed'] is not False
        or record['optimizer_steps']!=0 or record['actual_PPO_joint_steps']!=0 or record['actor_parameters_unchanged'] is not True):raise RuntimeError('gradient probe gate/scope differs')
    nn=context['nn_top'];top=context['raw_top'];sampled=arrays(rows)[3];selected=np.isin(context['leaf_indices'],raw['feedback_leaves'])&context['path_certified']&(top==nn)&(sampled==nn)
    q=context['program_probabilities'];p=arrays(rows)[2];row_loss=np.sum(q*(np.log(np.maximum(q,1e-300))-np.log(np.maximum(p,1e-300))),axis=1)
    expected_loss=.01*float(np.sum(row_loss*selected)/len(rows)) if gate else 0.
    variants=record['variants'];r,reg=variants['raw'],variants['regularized']
    if r['feedback_rows']!=0 or r['weighted_loss']!=0 or r['maximum_gradient_norm']!=0 or r['nonzero_gradient_batches']!=0:raise RuntimeError('raw external loss/gradient not exactly zero')
    if reg['feedback_rows']!=(int(selected.sum()) if gate else 0) or not np.isclose(reg['weighted_loss'],expected_loss,atol=2e-6,rtol=2e-5):raise RuntimeError('weighted probe samples/KL differ')
    norm=reg['maximum_gradient_norm']
    if type(norm) not in (int,float) or not math.isfinite(norm) or norm<0 or not gate and norm!=0:raise RuntimeError('native gradient norm/gate differs')
    batches=reg['nonzero_gradient_batches']
    if (type(batches) is not int or not 0<=batches<=math.ceil(len(rows)/512) or bool(batches)!=(norm>0)
        or not gate and batches!=0):raise RuntimeError('native gradient batch counters are inconsistent')
    reason=('independent_validation_gate_closed' if not gate else 'no_sampled_program_NN_top_match' if not selected.any() else 'native_autograd_reports_zero_remaining_parameter_derivative' if norm==0 else 'positive_native_autograd_probe_bound_not_PPO')
    return {'gate_open':gate,'potential_matching_rows':int(selected.sum()),'actual_probe_rows':reg['feedback_rows'],'independent_weighted_q_to_p_KL':expected_loss,'native_maximum_remaining_parameter_gradient_norm':norm,'native_autograd_reexecuted':False,'raw_zero_loss_and_gradient_bound':True,'interpretation':reason,'actual_PPO_joint_steps':0,'optimizer_steps':0}


def audit(*,root,specification_path):
    # This existing preflight opens no Actor until all6/conditional9 metadata
    # and summaries are complete. It performs native receipt checks, not the
    # independent routing/metric arithmetic implemented below.
    from scripts.report_warehouse_v621_bc_metrics import prepare_report
    reporting=prepare_report(root=root,specification_path=specification_path)
    e=Evidence(root);spec=e.json(specification_path);directory=Path(spec['output_directory']);e.bound.update(reporting['bound_evidence_sha256']);e.bound.update(reporting['reporting_sources_sha256'])
    from experiments.warehouse import v621_bc_audit_math as independent_math
    for path in (Path(__file__),Path(independent_math.__file__)):e.raw(path)
    spec_sha=reporting['specification']['sha256']
    from backend.training.warehouse_v6_tree_routing import certified_feature_bounds
    caches={str(i):cache_audit(e,directory/f'init-{i}'/'shared_cache',i,spec,spec_sha) for i in protocol.INITIALIZATIONS}
    models=[];starts={};total_resets=3*542
    for measured in reporting['models']:
        i,recipe=measured['initialization'],measured['recipe'];folder=directory/f'init-{i}'/recipe;result=e.pointer(measured['evidence']['result']);collection=e.pointer(result['development_collection']);training=e.pointer(result['training_report'])
        if training['shared_teacher_cache']!=caches[str(i)]['manifest'] or training['external_program_feedback']!=0 or training['external_program_actor_updates']!=0:raise RuntimeError('BC cache or external-feedback provenance differs')
        primary=training['shared_initial_primary_nn_sha256']
        if set(primary)!=set(ROLES) or any(not isinstance(h,str) or len(h)!=64 for h in primary.values()):raise RuntimeError('primary start identity missing')
        starts.setdefault(str(i),{})[recipe]=primary
        if recipe=='additive_control':from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor as Actor
        else:from backend.training.warehouse_v621_actor import FrozenWarehouseV621Actor as Actor
        e.raw(result['actor']['path'],result['actor']['sha256']);actor=Actor.load(result['actor']['path'],expected_sha256=result['actor']['sha256'],expected_role_sha256=result['actor']['role_sha256'])
        parts,quality=trace_rows(e,collection,actor,result['actor']);compare_common({k:v for k,v in quality.items() if k in result['quality']},result['quality'],'quality')
        quality_passed=quality['mean_deliveries']>=5.5 and quality['mean_collisions']<=6 and quality['shutdown_episode_rate']<=.05
        if result['quality_passed'] is not quality_passed:raise RuntimeError('quality gate differs from independent complete public trajectories')
        roles={}
        for role in ROLES:
            raw=e.pointer(result['roles'][role]['program']);names=tuple(actor.roles[role].feature_names);bounds=certified_feature_bounds(names);shared=recipe!='additive_control';metrics={};compared=[]
            for split in protocol.diagnostics.SPLITS:
                independent=metric_audit(raw,parts[split][role],names,bounds,shared);context=independent.pop('context');metrics[split]=independent
                compared.extend(compare_common(independent,result['roles'][role]['metrics'][split],role+'.'+split))
                if split=='validation':validation_context=context
                if split=='fit':fit_context=context
            if shared:
                w=actor.roles[role].weights;n=int(w['tree.node_count'][0]);leaf_count=int(w['tree.leaf_count'][0]);tree=raw['structure']
                for k,v in {'split_weights':w['tree.split.weight'][:n],'split_bias':w['tree.split.bias'][:n],'left_refs':w['tree.left_refs'][:n],'right_refs':w['tree.right_refs'][:n]}.items():
                    if not np.array_equal(v,np.asarray(tree[k])):raise RuntimeError('shared program topology differs from completeNN')
                if not np.array_equal(raw['core_mask'],w['core_leaf_mask'][:leaf_count]) or not np.array_equal(np.asarray(raw['anchors'],np.float32),w['tree.leaf_logits'][:leaf_count]):raise RuntimeError('shared core/anchor binding differs')
                probe=probe_audit(result['roles'][role]['gradient_probe'],metrics['validation'],validation_context,parts['validation'][role],raw,digest(raw))
            else:probe={'applicable':False,'reason':'architecture_control_not_a_raw_PPO_training_variant'}
            qualification=qualification_audit(raw,parts,role,names,bounds,shared,fit_context)
            gate=development_gate(metrics['validation'])
            if gate!=result['roles'][role]['development_gate']:raise RuntimeError('independent program development gate differs')
            roles[role]={'metrics':metrics,'independent_development_gate':gate,'native_fields_compared':compared,'qualification':qualification,'probe':probe}
        job_spec_sha=reporting['native_summaries']['backup']['value']['specification']['sha256'] if recipe=='shared_b050' else spec_sha
        journals=journal_audit(e,folder,i,recipe,spec,job_spec_sha);total_resets+=journals['actual_resets']
        models.append({'initialization':i,'recipe':recipe,'quality':quality,'quality_passed':quality_passed,'roles':roles,'journals':journals,'passed_is_not_admission':result['passed']})
    for i,recipes in starts.items():
        if any(v!=next(iter(recipes.values())) for v in recipes.values()):raise RuntimeError('same-initialization primary residual tensors differ across BC architectures')
    if len({tuple(sorted(v[next(iter(v))].items())) for v in starts.values()})!=3:raise RuntimeError('three distinct initializations unexpectedly share primary tensors')
    if total_resets!=3*542+1005*len(models):raise RuntimeError('batch reset budget differs')
    main_quality_failure=any(not row['quality_passed'] for row in models if row['recipe']=='shared_b035')
    if (len(models)==9)!=main_quality_failure:raise RuntimeError('backup cohort differs from independently reconstructed main quality branch')
    e.verify()
    if protocol.source_sha256(root)!=spec['source_closure_sha256']:raise RuntimeError('scientific194 source changed')
    return {'schema':'warehouse-v621-completed-BC-independent-audit.v1','passed':True,'source_closure_sha256':protocol.source_sha256(root),'specification':{'path':str(Path(specification_path).resolve()),'sha256':e.bound[str(Path(specification_path).resolve())]},'models':models,'shared_caches':caches,'same_init_primary_start_sha256':starts,'total_successful_resets':total_resets,
        'native_receipt_checker_used':True,'production_program_metric_evaluator_or_program_leaf_route_calls':0,'frozen_NN_forward_and_public_physics_encoder_reused':True,
        'native_autograd_probe_reexecuted':False,'full_actor_start_equal_across_architectures_claimed':False,'formal_raw_regularized_pair_training_occurred':False,
        'measurement_scope':{'physical_trace_NN_recomputation_per_model':'all275_spent_development_episodes_all_physical_frames_both_roles',
            'quality_episodes_per_model':75,'fresh_quality_or_human_collaboration_claimed':False,
            'BC730_intermediate_visits_independent_frame_replay_performed':False,
            'BC730_scope':'exact_reset_journal_and_immutable_native_producer_provenance_only_intermediate_Actor_states_not_saved',
            'gradient_scope':'independent_selected_samples_and_lambda01_KL_value_plus_bound_native_autograd_receipt_no_new_autograd_or_optimizer',
            'cache_scope':'teacher_NPZ_semantic_and_file_hashes_critic_file_hash_plus_report_binding_no_new_critic_evaluation'},
        'bound_inputs_sha256':dict(sorted(e.bound.items())),'environment_resets':0,'fit_calls':0,'optimizer_steps':0,'actor_updates':0,'PPO_joint_steps':0,'fresh_precheck_authorized':False,'pilot_opened':False,'publishable':False,
        'auditor_sources':{str(Path(__file__).resolve()):sha256(Path(__file__).read_bytes()).hexdigest(),str(Path(independent_math.__file__).resolve()):sha256(Path(independent_math.__file__).read_bytes()).hexdigest()}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise RuntimeError('audit output must be new')
    result=audit(root=ROOT,specification_path=a.spec);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as h:json.dump(result,h,sort_keys=True,separators=(',',':'),allow_nan=False);h.write('\n')
    print(json.dumps({'passed':result['passed'],'models':len(result['models']),'environment_resets':0,'PPO_joint_steps':0,'publishable':False}))
if __name__=='__main__':raise SystemExit(main())
