"""Fixed-asset confirmation arithmetic. No collection or fitting authority.

This module implements the stage AFTER v629's six selected units pass. Fresh
scene clearance and a finite collector remain mandatory before actual use.
Program qualification/temperatures are held fixed; confirmation never refits
or requalifies a leaf. Pressure's three action draws are one scenario cluster.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json, re, math
import numpy as np
from . import v629_protocol as v
from backend.training.warehouse_v629_program import Program, measure
from backend.training.warehouse_v622_program import digest, gate_metrics
from scripts.audit_warehouse_v628 import forward
from scripts.audit_warehouse_v626 import basic, same_metrics

ROLES = ('robot_1', 'robot_2')
SPLITS = ('validation', 'pressure')
ENDPOINTS = ('robot_1_explanation', 'robot_1_expected_feedback',
             'robot_2_explanation', 'robot_2_expected_feedback')


def require(value, message):
    if not value: raise RuntimeError(message)


def validate_cases(cases):
    """300 new confirmation bases + 72 different pressure bases, 3 draws."""
    required = {'case_id','split','base','scenario_seed','task','action_seed','sampling_mode','draw'}
    require(isinstance(cases,list) and len(cases)==516, 'confirmation requires exactly 300+216 full episodes')
    for c in cases:
        require(isinstance(c,dict) and set(c)==required, 'unknown confirmation case schema')
        require(isinstance(c['case_id'],str) and re.fullmatch(r'[A-Za-z0-9_-]+',c['case_id']), 'unsafe case ID')
        require(c['split'] in SPLITS and c['sampling_mode']=='sampled', 'confirmation contains training/argmax rows')
        require(all(type(c[k]) is int and 0<=c[k]<2**63 for k in ('base','scenario_seed','task','action_seed','draw'))
            and c['task'] in (1,2,3) and c['scenario_seed']==10*c['base']+c['task'], 'invalid scene/action arguments')
    require(len({c['case_id'] for c in cases})==516 and len({c['action_seed'] for c in cases})==516, 'episode or action stream reused')
    from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
    require(not any(is_sealed_scenario_seed(c[k]) for c in cases for k in ('base','scenario_seed','action_seed')),
        'confirmation cannot consume sealed audit/holdout arguments')
    bysplit = {s:[c for c in cases if c['split']==s] for s in SPLITS}
    require(len(bysplit['validation'])==300 and len(bysplit['pressure'])==216, 'confirmation split counts differ')
    bases = {}
    for split, expected, draws in (('validation',300,{0}),('pressure',72,{0,1,2})):
        groups=defaultdict(list)
        for c in bysplit[split]:groups[c['base']].append(c)
        require(len(groups)==expected, 'wrong independent scenario count')
        for rows in groups.values():
            require(len(rows)==len(draws) and {c['draw'] for c in rows}==draws
                and len({(c['task'],c['scenario_seed']) for c in rows})==1, 'pressure draws not paired within one scene')
        require(all(sum(rows[0]['task']==task for rows in groups.values())==expected//3 for task in (1,2,3)), 'tasks not balanced by scene')
        bases[split]=set(groups)
    require(not bases['validation'] & bases['pressure'], 'confirmation and pressure scenes overlap')
    sets=({c[k] for c in cases} for k in ('base','scenario_seed','action_seed'))
    a,b,c=sets
    require(not a&b and not a&c and not b&c, 'physical and action RNG namespaces overlap')
    return cases


@dataclass(frozen=True)
class Unit:
    role: str
    fold: int
    program: Program
    reference: Program
    program_ref: dict
    reference_ref: dict
    program_digest: str
    reference_digest: str


@dataclass(frozen=True)
class Bundle:
    units: tuple
    selected_capacity: int
    actor: dict
    provenance: dict

    def verify(self):
        require(len(self.units)==6 and {(u.role,u.fold) for u in self.units}=={(r,f) for r in ROLES for f in range(3)}, 'all six fixed units required')
        require(v.pointer(self.actor['path'])==self.actor, 'fixed complete NN changed')
        require(all(v.pointer(ref['path'])==ref for ref in self.provenance.values()), 'fixed selection provenance changed')
        for unit in self.units:
            require(v.pointer(unit.program_ref['path'])==unit.program_ref
                and v.pointer(unit.reference_ref['path'])==unit.reference_ref, 'fixed evidence file changed')
            require(unit.program.role==unit.role and unit.reference.role==unit.role
                and unit.program.maximum_leaves==unit.reference.maximum_leaves==self.selected_capacity,
                'fixed role/capacity differs')
            require(digest(unit.program.to_dict())==unit.program_digest
                and digest(unit.reference.to_dict())==unit.reference_digest, 'confirmation attempted model or qualification mutation')
        return self


def load_locked_bundle():
    """Fail before any environment/RNG access unless complete audited selection."""
    for name in ('summary.json','method-lock.json','independent-audit.json'):
        require((v.OUTPUT/name).is_file(), 'v629 has no locked audited winner; confirmation remains closed')
    plan_path,plan=v.checked_plan()
    summary=v.bound(v.pointer(v.OUTPUT/'summary.json'));lock=v.bound(v.pointer(v.OUTPUT/'method-lock.json'))
    audit=v.bound(summary['independent_audit'])
    require(summary['scientific_pass'] is True and summary['all_24_settled'] is True
        and summary['selected_capacity'] in (64,128,256) and audit['integrity_passed'] is True, 'failed/partial study cannot confirm')
    require(lock['summary']==v.pointer(v.OUTPUT/'summary.json') and lock['capacity']==summary['selected_capacity']
        and lock['method']==v.recipe()['method'], 'method lock differs')
    results=[]
    from scripts.run_warehouse_v629 import gate
    for job in plan['jobs']:
        folder=v.OUTPUT/'jobs'/job['job_id'];result=v.bound(v.pointer(folder/'result.json'))
        require(json.loads((folder/'completed.json').read_bytes())==dict(result_sha256=v.file_sha(folder/'result.json')),
            'missing or changed settled result')
        require(result['job']==job, 'wrong settled unit')
        if result['status']=='completed':
            require(result['plan_sha256']==v.file_sha(plan_path)
                and gate(result['metrics']['validation'],v.bound(result['search_report']))==result['gate'], 'changed gate or source binding')
        results.append(result)
    passing=[n for n in (64,128,256) if all(r['status']=='completed' and r['gate']['passed']
        for r in results if r['job']['name']==f'G{n}')]
    require(passing and min(passing)==summary['selected_capacity'], 'selection did not use smallest all-six passing capacity')
    selected=[r for r in results if r['job']['name']==f"G{summary['selected_capacity']}"]
    require(lock['programs']==[r['program'] for r in selected], 'locked six programs differ')
    require([r['job'] for r in audit['jobs']]==plan['jobs'], 'audit missing units')
    units=[]
    for result in selected:
        job=result['job'];record=next(row for row in audit['jobs'] if row['job']==job)
        require(record['status']=='completed' and record['program']==result['program']
            and record['protected_exact_forward'] and record['independent_reference_metrics'], 'selected unit audit incomplete')
        p=Program.from_dict(v.bound(result['program']),plan['actor_role_sha256'][job['role']])
        ref=Program.from_dict(v.bound(result['reference']),plan['actor_role_sha256'][job['role']])
        units.append(Unit(job['role'],job['fold'],p,ref,result['program'],result['reference'],digest(p.to_dict()),digest(ref.to_dict())))
    return Bundle(tuple(units),summary['selected_capacity'],plan['actor'],
        dict(plan=v.pointer(plan_path),summary=v.pointer(v.OUTPUT/'summary.json'),lock=v.pointer(v.OUTPUT/'method-lock.json'),
            independent_audit=summary['independent_audit'])).verify()


def episode_endpoints(program, arrays, predictions):
    """Match the existing coverage definitions, then average by full episode."""
    route,z,q=predictions;p=arrays['p'];mask=arrays['m'];top=q.argmax(1);raw=z.argmax(1);nn=p.argmax(1)
    proofs=program.certificates()
    valid=(mask[np.arange(len(mask)),raw]==1)&np.array([int(a) in proofs[int(i)] for a,i in zip(raw,route,strict=True)])
    e=np.isin(route,list(program.explanation_leaves)).astype(float)
    f=np.isin(route,list(program.feedback_leaves))&valid&(top==nn)
    expected=f*p[np.arange(len(p)),top]
    groups=defaultdict(list)
    for i,row in enumerate(arrays['rows']):groups[str(row['episode'])].append(i)
    return {case:(float(e[ids].mean()),math.fsum(map(float,expected[ids]))/len(ids)) for case,ids in groups.items()}


def paired_bootstrap(records, cases, split, *, iterations=20000):
    """Fold-average -> paired trajectory difference -> scene cluster average.

    Equal independent scenes are the sampling units; neither frames, folds nor
    the three pressure action draws are counted as independent replications.
    Task-stratified resampling keeps task weights fixed. RNG0 is a declared
    numerical control, not a new experimental scenario/action random stream.
    """
    require(split in SPLITS and type(iterations) is int and iterations>=1000, 'invalid bootstrap contract')
    selected=[c for c in cases if c['split']==split];byid={c['case_id']:c for c in selected}
    require(len(byid)==len(selected) and bool(byid), 'duplicate or absent bootstrap cases')
    all_ids={c['case_id'] for c in cases};values={}
    for row in records:
        require(row['case_id'] in all_ids, 'unregistered paired metric episode')
        if row['case_id'] not in byid:continue
        case=byid[row['case_id']];key=(row['case_id'],row['role'],row['fold'])
        require(row['role'] in ROLES and row['fold'] in (0,1,2)
            and row['base']==case['base'] and row['task']==case['task'], 'paired metric identity differs')
        require(key not in values, 'duplicate fold/role/episode pair')
        require(all(type(row[k]) in (float,int) and math.isfinite(row[k]) and 0<=row[k]<=1
            for k in ('program_explanation','reference_explanation','program_expected_feedback','reference_expected_feedback')),
            'endpoint inputs outside bounded physical fractions')
        delta=np.array([row['program_explanation']-row['reference_explanation'],
            row['program_expected_feedback']-row['reference_expected_feedback']],float)
        require(np.isfinite(delta).all() and np.max(np.abs(delta))<=1, 'invalid endpoint difference')
        values[key]=delta
    trajectories={}
    for case in selected:
        trajectory=[]
        for role in ROLES:
            require(all((case['case_id'],role,fold) in values for fold in range(3)), 'missing paired role/fold')
            trajectory.extend(np.mean([values[(case['case_id'],role,fold)] for fold in range(3)],axis=0))
        trajectories[case['case_id']]=np.asarray(trajectory)
    clusters=defaultdict(list);tasks={}
    for case in selected:
        clusters[case['base']].append(trajectories[case['case_id']]);tasks[case['base']]=case['task']
    data=np.array([np.mean(clusters[base],axis=0) for base in sorted(clusters)])
    labels=np.array([tasks[base] for base in sorted(clusters)])
    require(all(np.sum(labels==task)>=2 for task in (1,2,3)), 'not enough independent task-stratified scenes')
    rng=np.random.default_rng(0);samples=np.zeros((iterations,4),float)
    for task in (1,2,3):
        block=data[labels==task];indices=rng.integers(0,len(block),size=(iterations,len(block)))
        samples+=block[indices].mean(axis=1)*(len(block)/len(data))
    raw_lower=np.quantile(samples,.0125,axis=0,method='linear');mean=data.mean(axis=0)
    # A balanced +.3/0/-.3 draw fixture exposed roundoff-only positive bounds.
    # Bound error conservatively for bounded-fraction subtraction, fold/draw
    # means, stratified sums and quantile interpolation, then require a truly
    # positive lower bound. This is a numerical bound, not a fitted effect size.
    eps=np.finfo(np.float64).eps;operations=64+8*len(data)+8*max(map(len,clusters.values()))
    radius=4*(operations*eps/(1-operations*eps))+8*eps
    lower=raw_lower-radius
    return dict(schema='warehouse-v629-scene-paired-bootstrap.v1',split=split,
        confidence_one_sided=.9875,iterations=iterations,numerical_control_rng=0,
        independent_scenes=len(data),trajectory_episodes=len(selected),folds_not_independent=True,
        pressure_draws_clustered=True,numeric_roundoff_radius=float(radius),endpoints={name:dict(mean_difference=float(mean[i]),
            raw_lower=float(raw_lower[i]),lower=float(lower[i]),
            passed=bool(lower[i]>0)) for i,name in enumerate(ENDPOINTS)},passed=bool(np.all(lower>0)))


def quality(outcomes, cases):
    require(len(outcomes)==len(cases) and {r['case']['case_id'] for r in outcomes}=={c['case_id'] for c in cases},
        'quality requires all complete registered episodes')
    byid={c['case_id']:c for c in cases};reports={}
    for row in outcomes:
        require(row['case']==byid[row['case']['case_id']] and row.get('complete') is True
            and type(row['shutdown']) is bool and all(type(row[k]) is int and row[k]>=0 for k in ('deliveries','robot_collisions')),
            'quality cannot relabel partial episodes or missing shutdown events')
    for split in SPLITS:
        natural=[r for r in outcomes if r['case']['split']==split];n=len(natural)
        q=dict(episodes=n,independent_scenes=len({r['case']['base'] for r in natural}),
            mean_deliveries=sum(r['deliveries'] for r in natural)/n,
            mean_collisions=sum(r['robot_collisions'] for r in natural)/n,
            shutdown_rate=sum(r['shutdown'] for r in natural)/n)
        checks=dict(deliveries=q['mean_deliveries']>=5.5,collisions=q['mean_collisions']<=6,shutdown=q['shutdown_rate']<=.05)
        reports[split]=dict(**q,checks=checks,passed=all(checks.values()))
    return reports


def evaluate_fixed(bundle, datasets, cases, outcomes):
    """No fit/calibrate/qualify API is imported or called in this stage."""
    bundle.verify();validate_cases(cases)
    require(set(datasets)==set(SPLITS) and all(set(a)==set(ROLES) for a in datasets.values()), 'both roles in both fresh splits required')
    for split in SPLITS:
        require(datasets[split]['robot_1']['rows']==datasets[split]['robot_2']['rows'], 'joint physical frames differ between roles')
    reports=[];paired=[];byid={c['case_id']:c for c in cases}
    for unit in bundle.units:
        item=dict(role=unit.role,fold=unit.fold,program=unit.program_ref,reference=unit.reference_ref,metrics={},gates={})
        for split in SPLITS:
            a=datasets[split][unit.role]
            require(len(a['rows'])==len(a['x'])==len(a['m'])==len(a['p'])==len(a['sampled'])>0, 'all physical rows required')
            require(a['p'].shape==a['m'].shape==(len(a['x']),5) and np.isfinite(a['p']).all() and np.all((a['p']>=0)&(a['p']<=1))
                and np.allclose(a['p'].sum(1),1,atol=2e-6,rtol=0) and np.all(a['p'][a['m']==0]==0), 'invalid complete-NN target distribution')
            require(a['sampled'].shape==(len(a['x']),) and np.issubdtype(a['sampled'].dtype,np.integer)
                and np.all((a['sampled']>=0)&(a['sampled']<5))
                and np.all(a['m'][np.arange(len(a['x'])),a['sampled']]==1), 'invalid actually sampled physical action')
            ids={r['episode'] for r in a['rows']};expected={c['case_id'] for c in cases if c['split']==split}
            require(ids==expected and all(str(r['base_seed'])==str(byid[r['episode']]['base']) and r['task']==byid[r['episode']]['task']
                for r in a['rows']), 'fresh scene membership differs')
            pred=forward(unit.program,a['x'],a['m']);refpred=forward(unit.reference,a['x'],a['m'])
            for p,prediction in ((unit.program,pred),(unit.reference,refpred)):
                require(all(np.array_equal(x,y) for x,y in zip(prediction,p.distributions(a['x'],a['m']),strict=True)), 'independent frozen forward differs')
            m=measure(unit.program,a);b=measure(unit.reference,a)
            for p,prediction,metric in ((unit.program,pred,m),(unit.reference,refpred,b)):
                for purpose,selected in (('feedback',p.feedback_leaves),('explanation',p.explanation_leaves)):
                    same_metrics(basic(p,a,prediction,selected,purpose),metric[purpose])
            item['metrics'][split]=m;item['gates'][split]=gate_metrics(m,phase='development')
            item.setdefault('reference_metrics',{})[split]=b
            now,old=episode_endpoints(unit.program,a,pred),episode_endpoints(unit.reference,a,refpred)
            for case_id in sorted(expected):
                case=byid[case_id];paired.append(dict(case_id=case_id,base=case['base'],task=case['task'],
                    role=unit.role,fold=unit.fold,program_explanation=now[case_id][0],reference_explanation=old[case_id][0],
                    program_expected_feedback=now[case_id][1],reference_expected_feedback=old[case_id][1]))
        reports.append(item)
    q=quality(outcomes,cases);bootstrap={s:paired_bootstrap(paired,cases,s) for s in SPLITS}
    passed=all(g['passed'] for r in reports for g in r['gates'].values()) and all(r['passed'] for r in q.values()) and bootstrap['validation']['passed']
    bundle.verify()
    return dict(schema='warehouse-v629-fixed-confirmation-metrics.v1',metrics_passed=bool(passed),
        scope='fixed_asset_metrics_only_requires_independent_collection_audit',full_confirmation_admission=False,
        units=reports,quality=q,bootstrap=bootstrap,primary_bootstrap_family='validation_four_endpoints',
        pressure_bootstrap='secondary_diagnostic_not_a_replacement_for_strict_pressure_gates',
        fixed_assets=bundle.provenance,all_qualification_and_temperatures_unchanged=True,
        scoring_environment_resets=0,fit_calls=0,LP_calls=0,actor_updates=0,PPO_joint_steps=0,
        sealed_audit_used=False,heldout_used=False,pilot_opened=False,publishable=False)
