"""Finite model-bound evaluation for development, pressure, audit and holdout.

This controller never chooses actions. Runtime and the same physical adapter
produce every transition; all summaries are independently reconstructed.
"""
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import numpy as np
from .contracts import MethodSpec, digest, file_sha, program_gates, write_new
from .evaluation import quality
from .program import Program, metrics
from .reporting import quality_summary, program_summary
from .runtime import Binding, Runtime


def binding_for(adapter, policy, asset):
    return Binding(adapter.spec.sha256, f'{adapter.spec.name}-{policy}-v7',
        'raw' if policy=='raw' else 'regularized',asset['path'],asset['sha256'],
        tuple((r.name,asset['role_sha256'][r.name]) for r in adapter.spec.roles))


def execute(adapter, owner, *, purpose, assets, directory, formal=False, programs=None, quality_required=True):
    directory=Path(directory);directory.mkdir(exist_ok=False,parents=True)
    cases=[c for c in owner.plan['cases'] if c['purpose']==purpose]
    if not cases or set(c['policy'] for c in cases)!=set(assets):raise ValueError('finite evaluation policy cases differ')
    rows={p:[] for p in assets}; traces=[]
    for policy,asset in assets.items():
        binding=binding_for(adapter,policy,asset);runtime=Runtime(adapter,binding)
        for case in [c for c in cases if c['policy']==policy]:
            native=owner.reset(adapter,case,purpose=purpose);initial=adapter.snapshot(native)
            state=runtime.wrap(native,seed=case['seed'],task=case['task'],action_seed=case['action_seed']);records=[]
            while not adapter.terminal(native):
                decision=runtime.decide(state,binding)
                human=adapter.proxy(native) if case['pairing']=='offline_human_proxy' else None
                after=runtime.step(state,human,binding,decision=decision)
                records.append(dict(before=state,decision=decision,human=human,after=after))
                state=after;native=runtime.restore(state,binding)
            outcome=dict(seed=case['seed'],task=case['task'],trajectory=case.get('trajectory',0),
                pairing=case['pairing'],complete=True,**adapter.outcome(native,adapter.restore(initial)))
            rows[policy].append(outcome)
            path=directory/(case['case_id'].replace(':','_')+'.json.gz')
            with gzip.open(path,'xt') as f:json.dump(dict(case=case,binding=asdict(binding),initial=initial,
                records=records,outcome=outcome),f,separators=(',',':'),allow_nan=False)
            traces.append(dict(path=str(path.resolve()),sha256=file_sha(path)))
    if not formal and 'reference' not in rows:raise ValueError('development requires actual same-scene BC reference')
    task={p:quality(adapter.spec.name,rows[p],rows[p] if formal else rows['reference'],formal=formal)
          for p in ('raw','regularized')} if quality_required else {}
    result=dict(version='three-domain-unified-finite-evaluation.v1',domain=adapter.spec.name,
        phase=owner.plan['phase'],purpose=purpose,formal_thresholds=formal,quality_required=quality_required,
        method_sha256=MethodSpec().sha256,
        plan_sha256=owner.plan_hash,cases=cases,assets=assets,rows=rows,traces=traces,task_metrics=task,
        reference_evaluated='reference' in rows,programs={} if programs is None else
            {p:{n:v.as_dict() for n,v in values.items()} for p,values in programs.items()},
        programs_control_actions=False,human_participants=0,model_acceptance_passed=False,pilot_opened=False)
    path=directory/'result.json';write_new(path,result)
    verified=verify(adapter,path,expected_cases=cases,expected_plan=owner.plan_hash)
    write_new(directory/'verification.json',verified)
    return dict(result=dict(path=str(path.resolve()),sha256=file_sha(path)),
        verification=dict(path=str((directory/'verification.json').resolve()),sha256=file_sha(directory/'verification.json')))


def verify(adapter,path,*,expected_cases,expected_plan):
    """Recompute actual role actions, physics, outcomes, metrics and proof use."""
    path=Path(path);h=file_sha(path);result=json.loads(path.read_text())
    if (result['version']!='three-domain-unified-finite-evaluation.v1'
            or result['domain']!=adapter.spec.name or result['method_sha256']!=MethodSpec().sha256
            or result['plan_sha256']!=expected_plan or result['cases']!=expected_cases
            or result['programs_control_actions'] is not False):raise ValueError('evaluation protocol binding differs')
    cases={c['case_id']:c for c in expected_cases}
    if len(cases)!=len(expected_cases):raise ValueError('duplicated finite evaluation cases')
    rows={p:[] for p in result['assets']};seen=set();role_actions=0
    program_rows={p:{r.name:[] for r in adapter.spec.roles} for p in result['programs']}
    for ref in result['traces']:
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('evaluation trace hash differs')
        with gzip.open(ref['path'],'rt') as f:trace=json.load(f)
        case=trace['case'];ident=case['case_id']
        if ident in seen or cases.get(ident)!=case:raise ValueError('unexpected or repeated evaluation reset')
        seen.add(ident);policy=case['policy'];asset=result['assets'][policy]
        binding=binding_for(adapter,policy,asset);runtime=Runtime(adapter,binding)
        if trace['binding']!=json.loads(json.dumps(asdict(binding))):raise ValueError('trace policy binding differs')
        native=adapter.restore(trace['initial']);initial=adapter.restore(trace['initial'])
        state=runtime.wrap(native,seed=case['seed'],task=case['task'],action_seed=case['action_seed'])
        for frame in trace['records']:
            if frame['before']!=state:raise ValueError('evaluation frame chain differs')
            decision=runtime.decide(state,binding)
            if decision!=frame['decision']:raise ValueError('executed action differs from frozen NN')
            if case['pairing']=='offline_human_proxy':
                if frame['human']!=adapter.proxy(native):raise ValueError('unregistered offline proxy')
            elif frame['human'] is not None:raise ValueError('unregistered human takeover')
            if policy in program_rows and case['pairing']=='nn_nn':
                observations=adapter.observe_public(adapter.public(native))
                for role in adapter.spec.roles:
                    o=observations[role.name]
                    program_rows[policy][role.name].append(dict(x=o['x'],mask=o['mask'],
                        p=decision['probabilities'][role.name],sampled=decision['indices'][role.name],
                        scene=str(case['seed']),task=case['task']))
            state=runtime.step(state,frame['human'],binding,decision=decision)
            if state!=frame['after']:raise ValueError('physical legal action, collision or score differs')
            native=runtime.restore(state,binding);role_actions+=2
        if not adapter.terminal(native):raise ValueError('partial episode cannot enter evaluation')
        outcome=dict(seed=case['seed'],task=case['task'],trajectory=case.get('trajectory',0),
            pairing=case['pairing'],complete=True,**adapter.outcome(native,initial))
        if outcome!=trace['outcome']:raise ValueError('evaluation outcome differs')
        rows[policy].append(outcome)
    if seen!=set(cases) or rows!=result['rows']:raise ValueError('missing complete finite evaluation cases')
    formal=result['formal_thresholds']
    if formal is not (result['phase'] in ('formal','independent_audit','heldout')):
        raise ValueError('evaluation threshold phase differs')
    if not formal and 'reference' not in rows:raise ValueError('actual BC reference missing')
    if result['quality_required'] is not True and result['phase']!='independent_audit':
        raise ValueError('only the independent program audit omits task quality gates')
    task={p:quality(adapter.spec.name,rows[p],rows[p] if formal else rows['reference'],formal=formal)
        for p in ('raw','regularized')} if result['quality_required'] else {}
    if task!=result['task_metrics']:raise ValueError('quality summary is not reproducible')
    program_metrics={};program_descriptive={}
    for policy,roles in result['programs'].items():
        program_metrics[policy]={};program_descriptive[policy]={}
        for role in adapter.spec.roles:
            program=Program.from_dict(role,roles[role.name],actor_sha256=result['assets'][policy]['role_sha256'][role.name])
            source=program_rows[policy][role.name]
            data={k:np.array([r[k] for r in source]) for k in ('x','mask','p','sampled','scene')}
            actual=metrics(program,data);actual['passed']=program_gates(actual,formal=formal)
            program_metrics[policy][role.name]=actual
            program_descriptive[policy][role.name]=program_summary(program,data,{r['scene']:r['task'] for r in source})
    if file_sha(path)!=h:raise ValueError('evaluation changed during independent verification')
    return dict(version='three-domain-unified-finite-evaluation-verification.v1',
        result=dict(path=str(path.resolve()),sha256=h),method_sha256=MethodSpec().sha256,
        plan_sha256=expected_plan,replayed_role_actions=role_actions,complete_episodes=len(seen),
        task_metrics=task,program_metrics=program_metrics,program_descriptive=program_descriptive,
        quality_intervals={p:quality_summary(r) for p,r in rows.items()},
        quality_passed=bool(task) and all(v['passed'] for v in task.values()),
        programs_passed=bool(program_metrics) and all(m['passed'] for r in program_metrics.values() for m in r.values()),
        physical_action_overrides=0,human_participants=0,model_acceptance_passed=False,pilot_opened=False)
