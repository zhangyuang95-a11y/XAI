#!/usr/bin/env python3
"""Posthoc cached-data strata and scenario CIs; never used for selection.

No new NN inference, fitter, rollout, optimizer, or admission authority.  The
2000 bootstrap draws use statistical RNG seed zero, not an environment seed.
Intervals condition on these already-fitted, overlapping CV programs; they
do not quantify training-seed uncertainty or constitute independent evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

BOOTSTRAP_DRAWS=2000
STATISTICAL_RNG_SEED=0
COLS=('frames','F_qualified','E_qualified','F_actual','E_actual','F_expected','E_expected',
      'F_agree','E_agree','F_KL_sum','E_KL_sum','raw_illegal','path_unproven')
METRICS={'F_coverage':('F_qualified','frames'),'E_coverage':('E_qualified','frames'),
    'F_actual_coverage':('F_actual','frames'),'E_actual_coverage':('E_actual','frames'),
    'F_expected_coverage':('F_expected','frames'),'E_expected_coverage':('E_expected','frames'),
    'F_agreement_qualified_only':('F_agree','F_qualified'),
    'E_agreement_qualified_only':('E_agree','E_qualified'),
    'F_KL_qualified_only':('F_KL_sum','F_qualified'),
    'E_KL_qualified_only':('E_KL_sum','E_qualified')}
PAIRED_METRICS=tuple(k for k in METRICS if k.endswith('_coverage'))


def _sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def bootstrap_weights(cases, *, draws=BOOTSTRAP_DRAWS, seed=STATISTICAL_RNG_SEED):
    """Resample whole bases within fold x task, never independent frames."""
    if type(draws) is not int or draws<2 or type(seed) is not int or seed<0:
        raise ValueError('invalid reporting-only bootstrap configuration')
    bases=[c['base'] for c in cases]
    if len(set(bases))!=len(bases) or not bases:
        raise ValueError('each natural scenario/base must appear once')
    strata=defaultdict(list)
    for i,c in enumerate(cases):
        if c.get('sampling_mode')!='sampled' or c.get('task') not in (1,2,3):
            raise ValueError('posthoc validation must contain natural NN episodes only')
        strata[(c['fold'],c['task'])].append(i)
    rng=np.random.Generator(np.random.PCG64(seed))
    result=np.zeros((draws,len(cases)),np.int32)
    for key,indices in sorted(strata.items()):
        indices=np.asarray(indices)
        selected=rng.integers(0,len(indices),size=(draws,len(indices)))
        for b,row in enumerate(selected):result[b,indices]=np.bincount(row,minlength=len(indices))
    return result


def ratio_estimate(numerator,denominator,weights):
    numerator,denominator=np.asarray(numerator,np.float64),np.asarray(denominator,np.float64)
    if (numerator.shape!=denominator.shape or numerator.ndim!=1 or weights.shape[1]!=len(numerator)
            or not np.isfinite(numerator).all() or not np.isfinite(denominator).all()
            or np.any(denominator<0)):
        raise ValueError('invalid whole-scenario sufficient statistics')
    denominator_sum=float(denominator.sum());support=int((denominator>0).sum())
    # Explicit reduction avoids platform BLAS floating-status warnings and
    # never expands an episode into independent resampled physical rows.
    totals=np.einsum('bn,n->b',weights,denominator,optimize=False,dtype=np.float64)
    samples=np.full(len(weights),np.nan)
    valid=totals>0
    numerators=np.einsum('bn,n->b',weights,numerator,optimize=False,dtype=np.float64)
    samples[valid]=numerators[valid]/totals[valid]
    point=float(numerator.sum()/denominator_sum) if denominator_sum else None
    ci=[float(x) for x in np.quantile(samples[valid],[.025,.975])] if support>=2 and valid.sum()>=2 else None
    return {'point':point,'ci95':ci,'scenario_support':support,'denominator':denominator_sum,
        'valid_bootstrap_replicates':int(valid.sum()),'total_bootstrap_replicates':len(weights),
        'zero_denominator_replicates':int((~valid).sum()),'_samples':samples}


def paired_difference(left,right):
    """Same cluster draws across arms; intended for unconditional coverage."""
    a,b=left['_samples'],right['_samples'];valid=np.isfinite(a)&np.isfinite(b)
    support=min(left['scenario_support'],right['scenario_support'])
    return {'point':left['point']-right['point'] if left['point'] is not None and right['point'] is not None else None,
        'ci95':[float(x) for x in np.quantile((a-b)[valid],[.025,.975])] if support>=2 and valid.sum()>=2 else None,
        'valid_bootstrap_replicates':int(valid.sum()),'paired_same_scenario_draws':True}


def _strata(x,obs,p,sampled,task,names,obs_names):
    battery=obs[:,obs_names.index('self_battery_fraction')]
    low=x[:,names.index('self_charger_return_margin')]*50<=4
    conflict=x[:,names.index('same_target_collision_streak_capped')]>0
    near=x[:,names.index('robots_shortest_distance')]*42<=2
    result={'all:all':np.ones(len(x),bool),f'task:{task}':np.ones(len(x),bool)}
    for i,label in enumerate(('UP','DOWN','LEFT','RIGHT','WAIT')):
        result['NN_top_action:'+label]=p.argmax(-1)==i
        result['sampled_NN_proposal:'+label]=sampled==i
        result[f'task_x_NN_top:{task}/{label}']=p.argmax(-1)==i
    result.update({'battery_fraction:at_most_10_percent':battery<=.10,
        'battery_fraction:10_to_20_percent':(battery>.10)&(battery<=.20),
        'battery_fraction:above_20_percent':battery>.20,
        'charger_return_margin:at_most_4_moves':low,'charger_return_margin:above_4_moves':~low,
        'conflict_streak:positive':conflict,'conflict_streak:zero':~conflict,
        'near_robot:at_most_2_moves':near,'near_robot:above_2_moves':~near})
    return result


def frame_statistics(raw,rows,task,*,old16=False):
    """Re-evaluate only cached program outputs; NN probabilities stay cached."""
    from scripts import audit_warehouse_v621_diagnostics as math
    from backend.training.warehouse_v6_actor import FEATURE_NAMES
    from env.warehouse.public_observation_v6 import OBSERVATION_FEATURE_NAMES
    from env.warehouse.public_action_features_v617 import encode_actor_tree_features
    obs=np.asarray([r['obs'] for r in rows],np.float32);m=np.asarray([r['mask'] for r in rows],np.float32)
    p=np.asarray([r['nn_probabilities'] for r in rows],np.float32)
    sampled=np.asarray([math.ACTIONS.index(r['sampled_action']) for r in rows])
    if len(rows)==0 or p.shape!=m.shape or p.shape!=(len(rows),5) or np.any(p[m==0]!=0):
        raise RuntimeError('cached complete masked NN targets missing')
    x=encode_actor_tree_features(obs,m);names=tuple(FEATURE_NAMES)
    tree,capacity=math.structure(raw.get('structure',dict(raw,leaf_count=len(raw['leaf_logits']))),names,enforce_limits=False)
    if capacity['depth']>(5 if old16 else 6) or capacity['leaf_count']>(16 if old16 else 32) or capacity['maximum_source_features_per_node']>3:
        raise RuntimeError('posthoc program capacity/binding differs')
    leaves=math.route(tree,x);z=np.asarray(raw['leaf_logits'],np.float32)[leaves]
    q=math.softmax(z,m);top=q.argmax(-1);raw_top=z.argmax(-1);nn_top=p.argmax(-1)
    proof=math.certificates(tree,names,explicit_crossing=not old16)
    legal=m[np.arange(len(x)),raw_top]>.5
    proven=np.asarray([int(a) in proof[int(l)] for a,l in zip(raw_top,leaves,strict=True)])
    valid=legal&proven;agree=top==nn_top
    f=np.isin(leaves,raw['feedback_leaves']);e=np.isin(leaves,raw['explanation_leaves'])
    matched=valid&(top==sampled)&(m[np.arange(len(x)),sampled]>.5)
    expected=valid*p[np.arange(len(x)),top]
    kl=np.sum(p*(np.log(np.maximum(p,1e-8))-np.log(np.maximum(q,1e-8))),axis=-1)
    values=np.stack([np.ones(len(x)),f,e,f&matched&agree,e&matched,
        f*expected*agree,e*expected,f&agree,e&agree,f*kl,e*kl,~legal,~proven],axis=-1).astype(np.float64)
    return values,_strata(x,obs,p,sampled,task,names,tuple(OBSERVATION_FEATURE_NAMES))


class Evidence:
    def __init__(self):self.hashes={}
    def raw(self,path,expected=None):
        p=Path(path).resolve();raw=p.read_bytes();actual=sha256(raw).hexdigest()
        if expected is not None and actual!=expected:raise RuntimeError('posthoc evidence SHA differs: '+str(p))
        if str(p) in self.hashes and self.hashes[str(p)]!=actual:raise RuntimeError('evidence changed during reporting')
        self.hashes[str(p)]=actual;return raw
    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def pointer(self,ref):return self.json(ref['path'],ref['sha256'])
    def verify(self):
        for path,digest in self.hashes.items():
            if _sha(path)!=digest:raise RuntimeError('posthoc input changed after read: '+path)


def completed_inputs(plan_paths,evidence):
    """Guard all batches before reading any program, traces, or NN arrays."""
    from experiments.warehouse import v622_protocol as v
    from scripts.run_warehouse_v622_capacity import load_plan
    plans=[]
    for path in plan_paths:
        path,plan=load_plan(path);evidence.raw(path)
        summary=evidence.json(path.parent/'summary.json');audit=evidence.json(path.parent/'independent-audit.json')
        expected={j['job_id'] for j in plan['jobs']}
        if (set(summary.get('jobs',{}))!=expected or audit.get('passed') is not True
                or audit.get('summary_sha256')!=_sha(path.parent/'summary.json')
                or audit.get('plan_sha256')!=_sha(path) or audit.get('source_closure_sha256')!=plan['identity']['source_closure_sha256']
                or audit.get('job_count')!=len(expected)):
            raise RuntimeError('posthoc requires every registered job and its completed independent batch audit')
        results=[]
        for job in plan['jobs']:
            folder=path.parent/'jobs'/job['job_id']
            if (folder/'failed.json').exists():raise RuntimeError('engineering-failed batch cannot be pooled')
            done=evidence.json(folder/'completed.json');result=evidence.pointer(summary['jobs'][job['job_id']])
            if (done['result_sha256']!=summary['jobs'][job['job_id']]['sha256'] or result['job']!=job
                    or result['plan_sha256']!=_sha(path) or result['status'] not in ('completed','scientific_budget_exhausted')):
                raise RuntimeError('completed posthoc job binding differs')
            results.append(result)
        plans.append((path,plan,results))
    if not plans or any(p[1]['folds']!=plans[0][1]['folds'] for p in plans):raise RuntimeError('plans must share identical frozen CV cases')
    stages=[p[1]['stage'] for p in plans]
    if len(stages)!=len(set(stages)):raise RuntimeError('duplicate reporting stage')
    if 'fallback' in stages and 'primary' not in stages:raise RuntimeError('D32 report must include audited primary for shared paired draws')
    return plans


def _public(estimate):return {k:v for k,v in estimate.items() if not k.startswith('_')}


def prepare_report(plan_paths):
    from experiments.warehouse import v622_protocol as v
    e=Evidence();plans=completed_inputs(plan_paths,e);plan=plans[0][1]
    episodes,manifest=v.checked_inputs(ROOT)
    if manifest!=plan['identity']['inputs']:raise RuntimeError('cached episode manifest changed')
    e.hashes.update(manifest['files_sha256'])
    cases=sorted([{**c,'fold':f['fold']} for f in plan['folds'] for c in f['splits']['validation']],key=lambda c:c['base'])
    if len(cases)!=225:raise RuntimeError('posthoc requires all225 unique natural CV cases')
    weights=bootstrap_weights(cases);case_index={c['base']:i for i,c in enumerate(cases)}
    old=e.pointer(plan['baseline_A16']['summary']);programs={};unavailable=[]
    for ref in old['jobs'].values():
        row=e.pointer(ref)
        if row['job']['arm']=='A':programs[('A16',row['job']['role'],row['job']['fold'])]=row
    for _,current,results in plans:
        for row in results:
            key=(row['job']['arm'],row['job']['role'],row['job']['fold'])
            if key in programs:raise RuntimeError('duplicate program family')
            programs[key]=row
    statistics={};families=sorted({(a,r) for a,r,_ in programs})
    for arm,role in families:
        rows=[programs.get((arm,role,fold)) for fold in range(3)]
        if any(r is None or r['status']!='completed' for r in rows):
            unavailable.append({'arm':arm,'role':role,'reason':'no complete three-fold program family; not pooled'})
            continue
        by_fold={fold:e.pointer(rows[fold]['program']) for fold in range(3)}
        grouped={}
        for c in cases:
            raw=by_fold[c['fold']]
            if raw['actor_role_sha256']!=v.ROLE_SHA256[role]:raise RuntimeError('posthoc program bound to another NN')
            values,strata=frame_statistics(raw,episodes[c['base']][role],c['task'],old16=arm=='A16')
            for label,take in strata.items():
                grouped.setdefault(label,np.zeros((len(cases),len(COLS)),np.float64))[case_index[c['base']]]=values[take].sum(0)
        for label,table in grouped.items():statistics[(arm,role,label)]=table
    estimates={};rows=[]
    for (arm,role,label),table in sorted(statistics.items()):
        values={}
        for metric,(num,den) in METRICS.items():
            value=ratio_estimate(table[:,COLS.index(num)],table[:,COLS.index(den)],weights)
            estimates[(arm,role,label,metric)]=value;values[metric]=_public(value)
        rows.append({'arm':arm,'role':role,'stratum':label,'physical_frames':int(table[:,0].sum()),
            'scenarios_with_stratum_frames':int((table[:,0]>0).sum()),'metrics':values,
            'whole_tree_raw_illegal_count':int(table[:,COLS.index('raw_illegal')].sum()),
            'whole_tree_path_unproven_count':int(table[:,COLS.index('path_unproven')].sum())})
    differences=[]
    for (arm,role,label,metric),value in sorted(estimates.items()):
        key=('A16',role,label,metric)
        if arm!='A16' and metric in PAIRED_METRICS and key in estimates:
            differences.append({'arm':arm,'reference':'A16','role':role,'stratum':label,
                'metric':metric,**paired_difference(value,estimates[key])})
    # Outcome files are already pinned by the scientific input manifest. Team
    # quality is counted once per episode, never duplicated by the two roles.
    outcomes={};original_validation=set()
    for path,digest in manifest['files_sha256'].items():
        if re.search(r'/collection/(fit|calibration|eligibility|validation)/episode-\d+\.json$',path):
            item=e.json(path,digest);case=item['case'];scene=case['scenario_seed']
            outcomes[scene]=item['evidence']['episode_outcomes'][0]
            if '/collection/validation/' in path:original_validation.add(scene)
    quality=[]
    for cohort in ('all225_natural_cached_episodes','original75_validation_cached_episodes'):
        cohort_cases=[c for c in cases if cohort.startswith('all225') or c['scenario_seed'] in original_validation]
        cohort_weights=bootstrap_weights(cohort_cases)
        for task in (None,1,2,3):
            keep=np.asarray([task is None or c['task']==task for c in cohort_cases])
            values={}
            for name,field in [('mean_deliveries','deliveries'),('mean_collisions','collisions'),('shutdown_episode_rate','shutdown_episode')]:
                nums=np.asarray([float(outcomes[c['scenario_seed']][field]) for c in cohort_cases])*keep
                values[name]=_public(ratio_estimate(nums,keep.astype(float),cohort_weights))
            quality.append({'cohort':cohort,'task':task,'episodes':int(keep.sum()),'metrics':values,
                'source':'carried original v620 outcomes; unchanged NN; not a candidate quality change'})
    e.raw(Path(__file__));e.verify()
    return {'schema':'warehouse-v622-posthoc-scenario-strata.v1','candidate_id':v.CANDIDATE_ID,
        'frozen_actor_sha256':v.ACTOR_SHA256,'actor_weights_changed':False,
        'data_scope':'225 already-spent natural episodes; each base validation exactly once; 50 argmax fit episodes excluded',
        'bootstrap':{'unit':'whole scenario/base episode','stratification':'fold x task','draws':BOOTSTRAP_DRAWS,
            'statistical_RNG_seed':STATISTICAL_RNG_SEED,'seed_is_not_an_environment_training_or_evaluation_seed':True,
            'interval':'pointwise percentile95; conditional on cached fitted programs; not simultaneous or training-seed CI'},
        'definitions':{'NN_top_action':'argmax of cached complete masked NN distribution; not collision-adjusted WAIT',
            'sampled_NN_proposal':'stored sampled NN action before collision resolution',
            'coverage_denominator':'all untouched physical frames in each stratum',
            'F_actual':'qualified and raw-legal/path-proven and program masked top=NN masked top=stored sampled proposal',
            'E_actual':'qualified and raw-legal/path-proven and program masked top=stored sampled proposal',
            'E_qualification_is_not_action_agreement':'qualified leaf membership alone does not authorize an explanation; E_actual additionally matches the actual cached NN proposal, while E_agreement compares its masked argmax',
            'agreement':'qualified frames only; conditional regions differ between programs',
            'low_battery':'posthoc battery fraction bins <=10%,(10%,20%],>20%; distinct from return margin',
            'low_return_margin':'existing public feature times50 <=4 movement steps',
            'zero_denominator':'point and CI null; invalid bootstrap replicates counted',
            'degenerate_zero_interval':'empirical zero-event interval is not a guarantee of population zero'},
        'strata':rows,'paired_changes_from_A16':differences,'carried_quality':quality,'unavailable_families':unavailable,
        'used_for_selection':False,'admission_evidence':False,'human_effect_claimed':False,
        'new_environment_resets':0,'new_NN_forward_calls':0,'fit_calls':0,'actor_updates':0,'PPO_joint_steps':0,
        'pilot_opened':False,'bound_evidence_sha256':dict(sorted(e.hashes.items()))}


def markdown(report):
    lines=['# Warehouse 程序抽取事后分层报告','','本报告仅描述既有缓存；不参与选模，不新增训练或评估。',
        '配送、碰撞和停机来自旧 v6.20 记录；当前 NN 未改变，不能称为候选模型提升。','',
        '区间按完整场景在 fold×task 内重采样；条件于已拟合程序，不覆盖训练随机性。',
        '详细 task、NN 动作、采样动作、电量、返程余量、冲突及逐方法配对区间见同目录 JSON。','',
        '| 方法 | 角色 | 场景数 | 帧数 | E覆盖率及95%CI | 实际F及95%CI |','|---|---|---:|---:|---|---|']
    def fmt(m):
        if m['point'] is None:return '无分母'
        ci=m['ci95'];return f"{m['point']:.2%}"+(f" [{ci[0]:.2%}, {ci[1]:.2%}]" if ci else ' [样本不足]')
    for row in report['strata']:
        if row['stratum']=='all:all':lines.append(f"| {row['arm']} | {row['role']} | {row['scenarios_with_stratum_frames']} | {row['physical_frames']} | {fmt(row['metrics']['E_coverage'])} | {fmt(row['metrics']['F_actual_coverage'])} |")
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,action='append',required=True)
    p.add_argument('--report-id',required=True);args=p.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',args.report_id):p.error('safe new report-id required')
    report=prepare_report(args.plan);folder=ROOT/'output/warehouse_v622/reports'/args.report_id
    folder.mkdir(parents=True,exist_ok=False)
    with (folder/'posthoc.json').open('x') as f:json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    (folder/'posthoc.zh.md').write_text(markdown(report))
    print(json.dumps({'report':str(folder/'posthoc.zh.md'),'used_for_selection':False,'new_environment_resets':0}))


if __name__=='__main__':main()
