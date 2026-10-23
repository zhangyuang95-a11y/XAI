"""Descriptive post-training intervals and strata; never grants admission.

Bootstrap units are whole scenes, stratified by task. All probabilities and
program routes below are freshly recomputed; no previous model's scores enter.
"""
import numpy as np
from .contracts import file_sha, write_new
from .program import kl_rows


def clustered_ratios(scene, task, columns, *, draws=10000, seed=20261024):
    scene = np.asarray(scene).astype(str); task = np.asarray(task)
    groups = sorted(set(zip(task.tolist(), scene.tolist())))
    if not groups: raise ValueError('empty scene bootstrap')
    counts = np.zeros((draws, len(groups)), dtype=np.int32)
    rng = np.random.default_rng(seed)
    for t in sorted(set(task)):
        indices = [i for i,g in enumerate(groups) if g[0] == t]
        counts[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1/len(indices)), size=draws)
    result = {}
    for name,(numerator,denominator) in columns.items():
        numerator, denominator = np.asarray(numerator,float), np.asarray(denominator,float)
        if numerator.shape != scene.shape or denominator.shape != scene.shape or (denominator<0).any():
            raise ValueError('invalid descriptive ratio rows')
        n = np.array([numerator[(task==t)&(scene==s)].sum() for t,s in groups])
        d = np.array([denominator[(task==t)&(scene==s)].sum() for t,s in groups])
        dn = np.einsum('ij,j->i',counts,d,optimize=False)
        nn = np.einsum('ij,j->i',counts,n,optimize=False)
        valid = dn > 0
        values = nn[valid]/dn[valid]
        result[name] = dict(estimate=float(n.sum()/d.sum()) if d.sum() else None,
            lower=float(np.quantile(values,.025)) if len(values) else None,
            upper=float(np.quantile(values,.975)) if len(values) else None,
            independent_scenes=len(groups), draws=draws, undefined_draws=int((~valid).sum()),
            unit='whole_scene_cluster_stratified_by_task', seed=seed)
    return result


def program_summary(program, data, scene_tasks):
    q, top, ids = program.distributions(data['x'], data['mask'])
    p = data['p']; agree = top == p.argmax(-1); kl = kl_rows(p,q)
    one = np.ones(len(p)); columns = dict(full_tree_agreement=(agree,one), full_tree_kl=(kl,one))
    for purpose,leaves in (('feedback',program.feedback),('explanation',program.explanation)):
        selected = np.isin(ids,leaves)
        columns[purpose+'_coverage'] = (selected,one)
        columns[purpose+'_agreement'] = (selected*agree,selected)
        columns[purpose+'_kl'] = (selected*kl,selected)
        columns[purpose+'_available_coverage'] = (selected*agree*(top==data['sampled']),one)
        if purpose=='feedback':
            columns['expected_feedback_coverage'] = (selected*agree*p.max(-1),one)
    tasks = np.array([scene_tasks[str(s)] for s in data['scene']])
    intervals = clustered_ratios(data['scene'],tasks,columns)
    strata = {}
    def summarize(name,rows):
        if not rows.any(): return
        strata[name] = dict(frames=int(rows.sum()),scenes=len(set(data['scene'][rows])),
            values={n:float(np.asarray(v[0])[rows].sum()/np.asarray(v[1])[rows].sum())
                if np.asarray(v[1])[rows].sum() else None for n,v in columns.items()})
    for task in (1,2,3): summarize('task:'+str(task), tasks==task)
    for i,action in enumerate(program.role.actions): summarize('sampled_action:'+action,data['sampled']==i)
    risks = []
    for i,f in enumerate(program.role.features):
        if f.name.endswith('_slack') or f.name.endswith('_freshness') or f.name=='self_charger_return_margin':
            threshold = 0.; label = f.name+'<=0'
        elif f.name in ('self_battery','partner_battery') and f.lower==0 and f.upper==1:
            threshold = .20; label = f.name+'<=0.20'
        else: continue
        summarize('public_risk:'+label,data['x'][:,i]<=threshold)
        risks.append(dict(feature=f.name,operator='<=',threshold=threshold,descriptive_only=True))
    return dict(confidence_intervals=intervals,strata=strata,public_risk_definitions=risks,
                fresh_full_NN_labels=True,admission_decision=False)


def quality_summary(rows):
    result = {}
    # Pairings are separately labelled, never pooled into a human-effect claim.
    for pairing in sorted(set(r.get('pairing','nn_nn') for r in rows)):
        selected = [r for r in rows if r.get('pairing','nn_nn')==pairing]
        metrics = [k for k in ('deliveries','collisions','failed','capacity','wait','completed') if k in selected[0]]
        columns = {k:(np.array([r[k] for r in selected]),np.ones(len(selected))) for k in metrics}
        if 'completed' in metrics: columns['all_five_completion']=(np.array([r['completed']==5 for r in selected]),np.ones(len(selected)))
        result[pairing] = dict(confidence_intervals=clustered_ratios(
            [r['seed'] for r in selected],[r['task'] for r in selected],columns),
            task_strata={str(task):{k:float(np.mean([r[k] for r in selected if r['task']==task]))
                for k in metrics} for task in sorted(set(r['task'] for r in selected))})
    return result


def calibration_report(adapter, run_folder, results, programs):
    import json
    from pathlib import Path
    folder=Path(run_folder); path=folder/'fresh-20k-quality/result.json'
    fresh=json.loads(path.read_text())
    tasks={str(r['seed']):r['task'] for r in fresh['rows']['reference']}
    program_reports={}
    for variant in ('raw','regularized'):
        program_reports[variant]={}
        for role in adapter.spec.roles:
            with np.load(folder/'fresh-20k-quality'/f'{variant}-{role.name}-fresh-program.npz',allow_pickle=False) as data:
                program_reports[variant][role.name]=program_summary(programs[variant][role.name],dict(data),tasks)
    result=dict(version='three-domain-unified-post-20k-descriptive-report.v1',domain=adapter.spec.name,
        actual_quality=dict(path=str(path.resolve()),sha256=file_sha(path)),
        joint_steps={v:results[v]['joint_steps'] for v in results},
        task_quality=fresh['task_metrics'],program_metrics=fresh['program_metrics'],
        quality_confidence_intervals={v:quality_summary(rows) for v,rows in fresh['rows'].items()},
        program_confidence_intervals_and_strata=program_reports,
        actual_training={v:{k:results[v][k] for k in ('training_gates','program_updates','program_gradient_norm_sum',
            'program_feedback_frames','program_evaluated_frames','accepted_actor_change','timing')} for v in results},
        reference_scores_are_not_post_training_scores=True,human_participants=0,
        model_acceptance_passed=False,pilot_opened=False)
    write_new(folder/'post-20k-metrics-and-intervals.json',result)
    return result
