"""Predeclared paired task-retention arithmetic; no collection or RNG.

The 20k calibration is not a development-budget selection or release audit.
All frozen policies must run the same newly declared complete scenes.
"""
from collections import Counter
from .v626_protocol import require,digest
from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed

VERSION='warehouse-v642-20k-task-retention.v1'
VARIANTS=('reference','raw','regularized')


def recipe():
    return dict(version=VERSION,independent_scenes=75,scenes_per_task=25,
        variants=list(VARIANTS),sampling='complete_frozen_dual_NN_with_physical_action_mask',
        maximum_environment_resets=225,minimum_deliveries=5.5,minimum_reference_delivery_fraction=.95,
        maximum_collisions=6.,maximum_reference_collision_increase=1.,maximum_shutdown_rate=.05,
        gates_on_each_variant_not_pooled=True,task_strata_reported_separately=True,
        quality_evaluation_only=True,program_fidelity_gate_separate=True,
        training_updates=0,sealed_evaluation=False,pilot_opened=False)


def cases(base_start,action_start):
    require(type(base_start) is type(action_start) is int,'explicit scene/action arguments required')
    out=[dict(case_id=f'calibration-quality-{i:03d}',base=base_start+i,task=i%3+1,
        scenario_seed=10*(base_start+i)+i%3+1,action_seed=action_start+i) for i in range(75)]
    validate_cases(out);return out


def validate_cases(rows):
    require(type(rows) is list and len(rows)==75,'exactly 75 independent calibration scenes required')
    seen=set();ids=set();tasks=Counter()
    for row in rows:
        require(set(row)=={'case_id','base','task','scenario_seed','action_seed'}
            and type(row['case_id']) is str and row['case_id'] and row['case_id'] not in ids
            and type(row['task']) is int and row['task'] in (1,2,3), 'invalid calibration scene identity')
        values=[row[k] for k in ('base','scenario_seed','action_seed')]
        require(all(type(v) is int and 0<=v<2**63 and not is_sealed_scenario_seed(v) for v in values)
            and len(set(values))==3 and not seen.intersection(values)
            and row['scenario_seed']==10*row['base']+row['task'],
            'reused, sealed or invalid original calibration scene argument')
        seen.update(values);ids.add(row['case_id']);tasks[row['task']]+=1
    require(tasks=={1:25,2:25,3:25},'calibration task balance changed')


def evaluate(outcomes,case_rows,actors):
    validate_cases(case_rows)
    require(set(actors)==set(VARIANTS) and len({r['sha256'] for r in actors.values()})==3,
        'reference/raw/regularized must have three explicit distinct frozen asset hashes')
    expected={(v,c['case_id']) for v in VARIANTS for c in case_rows};seen=set();groups={v:[] for v in VARIANTS}
    byid={r['case_id']:r for r in case_rows}
    for row in outcomes:
        key=(row['variant'],row['case']['case_id'])
        require(key in expected and key not in seen and row['case']==byid[key[1]]
            and row['actor']==actors[key[0]] and row['complete'] is True,
            'missing, duplicated, incomplete or differently bound policy/scene result')
        require(type(row['physical_frames']) is int and 0<row['physical_frames']<=120
            and row['NN_action_checks']==2*row['physical_frames'] and row['action_overrides']==0,
            'all executed role actions must be checked against the complete frozen NN')
        require(type(row['shutdown']) is bool
            and all(type(row[k]) is int and row[k]>=0 for k in ('deliveries','robot_collisions')),
            'invalid complete-episode quality result')
        seen.add(key);groups[key[0]].append(row)
    require(seen==expected,'all three policies must finish every matched scene')
    def stats(rows):
        n=len(rows)
        return dict(episodes=n,deliveries=sum(r['deliveries'] for r in rows)/n,
            collisions=sum(r['robot_collisions'] for r in rows)/n,
            shutdown_episodes=sum(r['shutdown'] for r in rows),shutdown_rate=sum(r['shutdown'] for r in rows)/n)
    quality={v:stats(rows) for v,rows in groups.items()};ref=quality['reference'];gates={}
    for variant in ('raw','regularized'):
        current=quality[variant]
        checks=dict(deliveries=current['deliveries']>=max(5.5,.95*ref['deliveries']),
            collisions=current['collisions']<=min(6.,ref['collisions']+1.),shutdown_rate=current['shutdown_rate']<=.05)
        gates[variant]=dict(checks=checks,passed=all(checks.values()),
            minimum_deliveries=max(5.5,.95*ref['deliveries']),maximum_collisions=min(6.,ref['collisions']+1.))
    return dict(version=VERSION,recipe=recipe(),cases_sha256=digest(case_rows),actors=actors,quality=quality,
        by_task={v:{str(t):stats([r for r in rows if r['case']['task']==t]) for t in (1,2,3)} for v,rows in groups.items()},
        gates=gates,task_retention_passed=all(g['passed'] for g in gates.values()),
        scope='arithmetic_requires_separate_hash_bound_collection_and_NN_replay',
        program_fidelity_evaluation_required=True,model_acceptance_passed=False,pilot_opened=False)
