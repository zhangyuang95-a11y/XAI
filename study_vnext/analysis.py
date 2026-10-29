"""Participant-level extraction and explicit-assumption analysis/planning tools.

Only caller-provided exports are read. This module neither obtains participant
files nor recomputes Actors, authorizes enrollment, chooses N, or asserts benefit.
The upstream exporter must independently verify NN/physics; its verification flag
is checked here but cannot be authenticated by a statistics function alone.
"""
import json
import math

import numpy as np

from runtime_v642.contracts import PolicyBinding, RuntimeUnavailable, digest, is_sha
from .protocol import DOMAINS, GROUPS, policy_variant, protocol_sha256, analysis_contract, protocol_document

CONTACT_CAPACITY_DEFINITION_SHA256 = '714299b2afc14305396c3fa3b575a9d4090be9e9ea3f1aae9ebb511f12c99d7f'


class AnalysisError(ValueError):
    pass


def _check(ok, message):
    if not ok: raise AnalysisError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _decode(value):
    if type(value) is dict: return value
    _check(type(value) is str, 'export JSON object required')
    def pairs(items):
        out = {}
        for key, item in items:
            _check(key not in out, 'duplicate export JSON key'); out[key] = item
        return out
    return json.loads(value, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(AnalysisError('nonfinite export JSON')))


def _contact_capacity(native):
    """Independently reproduce the registered instantaneous-contact upper bound."""
    _check(type(native) is dict and type(native.get('lanes')) is int and native['lanes'] == 9
           and type(native.get('_schedule')) is list, 'Pong capacity needs its complete fixed initial schedule')
    arrivals = {}
    for item in native['_schedule']:
        _check(type(item) is dict and type(item.get('arrival_turn')) is int and item['arrival_turn'] >= 0,
               'invalid fixed Pong arrival')
        ball = item.get('ball', {})
        _check(ball.get('kind') in ('ordinary', 'cooperative') and type(ball.get('contacts')) is list
               and len(ball['contacts']) == (2 if ball['kind']=='cooperative' else 1)
               and all(type(x) is int and 0 <= x < 9 for x in ball['contacts'])
               and len(set(ball['contacts'])) == len(ball['contacts']), 'invalid fixed Pong ball')
        arrivals.setdefault(item['arrival_turn'], []).append(ball)
    return sum(max(sum((3 if b['kind']=='cooperative' else 1) for b in balls
                       if ({human, ai} == set(b['contacts']) if b['kind']=='cooperative' else b['contacts'][0] in (human, ai)))
                   for human in range(9) for ai in range(9)) for balls in arrivals.values())


def _metric(domain, run, frames, capacity_evidence):
    _check(run['status'] == 'complete' and frames, 'final metric needs a complete run')
    frames = sorted(frames, key=lambda f: f['turn'])
    initial = _decode(run['initial_state_json'])
    _check(digest(initial) == run['initial_state_sha256'] and frames[0]['state'] == initial
           and [f['turn'] for f in frames] == list(range(len(frames)))
           and frames[-1]['state'].get('terminal') is True, 'metric frame/origin continuity differs')
    previous = None
    for frame in frames:
        body = {k:v for k,v in frame.items() if k not in ('frame_sha256','run_id')}
        _check(is_sha(frame.get('frame_sha256')) and digest(body)==frame['frame_sha256']
               and frame.get('previous_frame_sha256')==previous, 'export frame hash chain differs')
        previous = frame['frame_sha256']
    final = frames[-1]
    _check(_decode(run['state_json'])==final['state'], 'export run final state differs')
    score = final['public']['score']
    if domain == 'warehouse': value = score.get('task_score')
    elif domain == 'kitchen':
        value = score.get('metrics', {}).get('completed_orders')
        _check(type(value) is int and 0 <= value <= 5, 'Kitchen completed order count required')
    else:
        evidence = capacity_evidence.get(run['id'])
        _check(type(evidence) is dict and set(evidence) == {'capacity', 'initial_state_sha256',
            'definition_sha256', 'verification_receipt_sha256'}, 'independent Pong capacity evidence required')
        _check(evidence['initial_state_sha256'] == run['initial_state_sha256']
               and evidence['definition_sha256'] == CONTACT_CAPACITY_DEFINITION_SHA256
               and is_sha(evidence['verification_receipt_sha256'])
               and type(evidence['capacity']) is int and evidence['capacity'] > 0,
               'Pong capacity evidence identity differs')
        capacity = _contact_capacity(initial.get('snapshot'))
        _check(capacity == evidence['capacity'], 'Pong denominator does not match independent fixed-schedule bound')
        raw = score.get('raw_score')
        _check(_finite(raw) and 0 <= raw <= capacity, 'Pong raw points exceed registered contact bound')
        value = 100.0 * raw / capacity
    _check(_finite(value), 'missing or nonfinite primary metric')
    return float(value)


def participant_rows(exports, *, expected_release_id, expected_binding_sha256, capacity_evidence=None):
    """Extract one row/person; preview/test are excluded and missing outcomes kept.

    Inputs must be outputs from the independent runtime export verifier. The
    returned exclusions/attrition inventory is not an outcome-based exclusion rule.
    Missing outcomes make contrast fitting fail until a registered policy handles
    them; they are never silently filled with zero or dropped here.
    """
    _check(type(exports) in (list, tuple) and type(expected_release_id) is str and expected_release_id,
           'explicit release and export sequence required')
    _check(type(expected_binding_sha256) is dict and set(expected_binding_sha256) == set(DOMAINS),
           'locked three-domain binding inventory required')
    for domain, binding_hashes in expected_binding_sha256.items():
        _check(type(binding_hashes) is dict and set(binding_hashes) == set(GROUPS)
               and all(is_sha(x) for x in binding_hashes.values())
               and binding_hashes['A'] == binding_hashes['B'] != binding_hashes['C'], 'invalid locked A/B/C binding inventory')
    capacity_evidence = {} if capacity_evidence is None else capacity_evidence
    rows, excluded, seen = [], [], set()
    for exported in exports:
        _check(type(exported) is dict, 'verified study export required')
        instance = exported.get('instance', {})
        if instance.get('mode') in ('preview', 'test'):
            excluded.append(dict(instance_id=instance.get('id'), reason='non_scientific_mode')); continue
        _check(instance.get('mode') == 'pilot' and exported.get('all_AI_frames_recomputed') is True
               and exported.get('protocol_sha256') == protocol_sha256()
               and exported.get('study_protocol') == protocol_document(), 'formal-mode verified protocol export required')
        records = exported.get('records', {}); people = records.get('instances', [])
        _check(type(people) is list and len(people) == 1 and people[0]['id'] == instance['id'], 'one participant instance per export required')
        person = people[0]; key = person.get('participant_key')
        _check(is_sha(key) and key not in seen, 'duplicate or unhashed participant identity')
        seen.add(key)
        domain, group = instance.get('domain'), instance.get('group')
        _check(domain in DOMAINS and group in GROUPS and instance.get('release_id') == expected_release_id
               and all(person.get(k)==instance.get(k) for k in ('id','release_id','domain','mode','group','scenario_seed','binding_sha256')), 'mixed release/domain/group exports')
        try: bound = PolicyBinding.from_dict(exported['binding'])
        except (RuntimeUnavailable, TypeError, KeyError, ValueError):
            raise AnalysisError('invalid exported policy binding') from None
        _check(bound.domain == domain and bound.variant == policy_variant(group)
               and bound.release_id == expected_release_id and instance.get('binding') == bound.to_dict()
               and instance.get('binding_sha256') == expected_binding_sha256[domain][group] == digest(bound.to_dict()),
               'participant policy differs from locked group binding')
        metrics, runs = {}, records.get('runs', [])
        _check(type(runs) is list and len({r['task'] for r in runs}) == len(runs)
               and all(type(r['task']) is int and r['task'] in (1,2,3) and r['instance_id']==instance['id'] for r in runs),
               'duplicate or foreign task runs')
        for task in (1,2,3):
            run = next((r for r in runs if r['task']==task), None)
            if run is None or run['status'] != 'complete': metrics[task] = None; continue
            frames = [f for f in records.get('frames', []) if f.get('run_id') == run['id']]
            _check(all(f.get('binding') == bound.to_dict() and f.get('binding_sha256') == digest(bound.to_dict())
                       and f['state'].get('task') == task for f in frames), 'frame model/task binding differs')
            metrics[task] = _metric(domain, run, frames, capacity_evidence)
        _check(instance.get('stage')!='complete' or all(metrics[t] is not None for t in (1,2,3)), 'completed instance lacks three terminal outcomes')
        metric = protocol_document()['analysis']['primary_metrics'][domain]
        rows.append(dict(participant_key=key,instance_id=instance['id'],release_id=expected_release_id,
            domain=domain,group=group,scenario_seed=instance['scenario_seed'],binding_sha256=digest(bound.to_dict()),
            task1=metrics[1],task2=metrics[2],task3=metrics[3],metric=metric['name'],unit=metric['unit'],
            source_export_sha256=digest(exported)))
    return dict(rows=rows,excluded=excluded,missing_primary_outcomes=[r['participant_key'] for r in rows if r['task3'] is None],
                unit='participant',formal_enrollment_allowed=False,scientific_acceptance_claimed=False)


def contrast_design(rows, *, domain, contrast):
    contract = analysis_contract(contrast)
    _check(domain in DOMAINS and type(rows) in (list,tuple), 'known domain and participant rows required')
    _check(len({r['participant_key'] for r in rows}) == len(rows), 'participant pseudoreplication is prohibited')
    selected = sorted((r for r in rows if r['domain']==domain and r['group'] in contrast.split('-')),
                      key=lambda r: r['participant_key'])
    first, second = contrast.split('-')
    _check(selected and {r['group'] for r in selected} == {first,second}
           and len({r['release_id'] for r in selected})==1, 'both groups in one release are required')
    _check(all(_finite(r['task3']) and (not contract['task1_baseline_adjustment'] or _finite(r['task1'])) for r in selected),
           'missing outcomes require the preregistered missing-data policy')
    expected_metric = protocol_document()['analysis']['primary_metrics'][domain]
    _check(all(r.get('metric')==expected_metric['name'] and r.get('unit')==expected_metric['unit'] for r in selected),
           'mixed primary metric definitions or units')
    _check(all(len({r['binding_sha256'] for r in selected if r['group']==group})==1 for group in (first,second)),
           'mixed frozen bindings within a group')
    if contrast == 'A-B':
        _check(len({r['binding_sha256'] for r in selected})==1, 'A-B must compare the same frozen model')
    columns = ['intercept','group_'+first]; vectors = [np.ones(len(selected)), np.array([r['group']==first for r in selected],float)]
    if contract['task1_baseline_adjustment']:
        baseline = np.array([r['task1'] for r in selected],float);vectors.append(baseline-baseline.mean());columns.append('task1_primary_metric')
    blocks = sorted({str(r['scenario_seed']) for r in selected})
    for block in blocks[1:]:
        vectors.append(np.array([str(r['scenario_seed'])==block for r in selected],float));columns.append('scenario_'+block)
    X, y = np.column_stack(vectors), np.array([r['task3'] for r in selected],float)
    _check(len(selected)>len(columns) and np.linalg.matrix_rank(X)==len(columns) and np.linalg.cond(X)<1e10, 'contrast design is underidentified or ill-conditioned')
    return dict(X=X,y=y,columns=columns,rows=selected,contrast=contrast,domain=domain,
                unit=protocol_document()['analysis']['primary_metrics'][domain]['unit'])


def hc3_contrast(rows, *, domain, contrast):
    design = contrast_design(rows, domain=domain, contrast=contrast);X,y=design['X'],design['y']
    inverse = np.linalg.inv(X.T@X); beta = inverse@X.T@y; residual = y-X@beta
    leverage = np.einsum('ij,jk,ik->i',X,inverse,X)
    _check(np.all(1-leverage>1e-10), 'HC3 leverage is degenerate')
    sandwich = inverse@(X.T@(((residual/(1-leverage))**2)[:,None]*X))@inverse
    estimate, se = float(beta[1]), math.sqrt(max(0.,float(sandwich[1,1])))
    pvalue = None if se==0 else math.erfc(abs(estimate/se)/math.sqrt(2))
    return dict(domain=domain,contrast=contrast,n=len(y),estimate_first_minus_second=estimate,
        unit=design['unit'],standard_error_HC3=se,CI95=[estimate-1.95996398454*se,estimate+1.95996398454*se],
        p_value_asymptotic_normal=pvalue,analysis='OLS_HC3_asymptotic_normal',columns=design['columns'],
        effect_claim='not_automatically_determined',relative_improvement_not_computed=True,
        limitations=['asymptotic_intervals_need_registered_assumptions','not_a_power_certificate'],
        formal_enrollment_allowed=False)


def holm_three_domains(pvalues):
    _check(type(pvalues) is dict and set(pvalues)==set(DOMAINS)
           and all(_finite(v) and 0<=v<=1 for v in pvalues.values()), 'all three primary-domain p-values required')
    adjusted, previous = {}, 0.
    for index, domain in enumerate(sorted(DOMAINS,key=lambda d:(pvalues[d],d))):
        previous=max(previous,min(1.,(3-index)*pvalues[domain]));adjusted[domain]=previous
    return {d:dict(raw_p=pvalues[d],holm_p=adjusted[d],rejected_at_FWER_0_05=adjusted[d]<=0.05) for d in DOMAINS}


def power_candidates(assumptions, per_group_candidates, *, simulations=10000, seed=20261029):
    """Normal-outcome planning approximation with joint three-domain Holm tests.

    Effect/SD/correlation are caller-owned registered assumptions. This does not
    estimate them from pilot data, select a final N or certify computed trial power.
    """
    _check(type(assumptions) is dict and set(assumptions)==set(DOMAINS)
           and type(per_group_candidates) in (list,tuple) and per_group_candidates
           and all(type(n) is int and n>=4 for n in per_group_candidates)
           and list(per_group_candidates)==sorted(set(per_group_candidates)), 'explicit unique ordered sample candidates required')
    _check(type(simulations) is int and 100<=simulations<=1000000 and type(seed) is int and seed>=0, 'bounded explicit power simulation required')
    for domain, a in assumptions.items():
        _check(type(a) is dict and set(a)=={'meaningful_effect','outcome_sd','task1_correlation','unit'}
               and all(_finite(a[k]) for k in ('meaningful_effect','outcome_sd','task1_correlation'))
               and a['meaningful_effect']>0 and a['outcome_sd']>0 and abs(a['task1_correlation'])<1
               and a['unit']==protocol_document()['analysis']['primary_metrics'][domain]['unit'], 'invalid native-unit planning assumptions')
    rng=np.random.default_rng(seed); result=[]
    for n in per_group_candidates:
        means=[assumptions[d]['meaningful_effect']/(assumptions[d]['outcome_sd']*math.sqrt(1-assumptions[d]['task1_correlation']**2)*math.sqrt(2/n)) for d in DOMAINS]
        z=rng.normal(size=(simulations,3))+np.array(means)
        p=np.array([math.erfc(abs(v)/math.sqrt(2)) for v in z.flat]).reshape(z.shape)
        order=np.argsort(p,axis=1);sorted_p=np.take_along_axis(p,order,axis=1)
        ordered_reject=np.logical_and.accumulate(sorted_p<=np.array([.05/3,.05/2,.05]),axis=1)
        reject=np.zeros_like(ordered_reject);np.put_along_axis(reject,order,ordered_reject,axis=1)
        powers=reject.mean(axis=0)
        result.append(dict(participants_per_domain_group=n,total_participants=9*n,
            estimated_primary_power={d:float(powers[i]) for i,d in enumerate(DOMAINS)},
            monte_carlo_standard_error={d:math.sqrt(float(powers[i]*(1-powers[i]))/simulations) for i,d in enumerate(DOMAINS)}))
    return dict(method='normal_outcome_joint_Holm_planning_approximation',contrast='A-B',assumptions=json.loads(json.dumps(assumptions)),
        simulations=simulations,seed=seed,candidates=result,selected_formal_N=None,
        formal_enrollment_allowed=False,computed_trial_power_verified=False)
