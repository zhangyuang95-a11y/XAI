"""Fresh post-training evaluation; reference scores never substitute for trained NN scores."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from .contracts import digest, file_sha, write_new
from .evaluation import quality
from .program import metrics
from .runtime import Binding, Runtime


def collect_quality(adapter, run, assets, programs, *, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = {n: [] for n in ('reference', 'raw', 'regularized')}
    program_data = {v: {r.name: [] for r in adapter.spec.roles} for v in ('raw', 'regularized')}
    traces = []
    for policy in ('reference', 'raw', 'regularized'):
        asset = assets[policy]
        binding = Binding(adapter.spec.sha256, f'{adapter.spec.name}-{policy}-v7',
            'raw' if policy == 'raw' else 'regularized', asset['path'], asset['sha256'], tuple(asset['role_sha256'].items()))
        runtime = Runtime(adapter, binding)
        for case in [c for c in run.plan['cases'] if c['purpose'] == 'calibration_quality' and c['policy'] == policy]:
            native = run.reset(adapter, case, purpose='calibration_quality'); initial = adapter.snapshot(native)
            state = runtime.wrap(native, seed=case['seed'], task=case['task'], action_seed=case['action_seed']); records = []
            while not adapter.terminal(native):
                decision = runtime.decide(state, binding, action_seed=case['action_seed'])
                human = adapter.proxy(native) if case['pairing'] == 'offline_human_proxy' else None
                before = deepcopy(state); after = runtime.step(state, human, binding, decision=decision)
                if policy != 'reference' and case['pairing'] == 'nn_nn':
                    obs = adapter.observe_public(adapter.public(native))
                    for role in adapter.spec.roles:
                        program_data[policy][role.name].append(dict(x=obs[role.name]['x'], mask=obs[role.name]['mask'],
                            p=decision['probabilities'][role.name], sampled=decision['indices'][role.name], scene=str(case['seed'])))
                records.append(dict(before=before, decision=decision, human=human, after=after))
                state = after; native = runtime.restore(state, binding)
            outcome = adapter.outcome(native, adapter.restore(initial))
            row = dict(seed=case['seed'], task=case['task'], pairing=case['pairing'], complete=True, **outcome)
            rows[policy].append(row)
            path = output/(case['case_id'].replace(':', '_') + '.json.gz')
            with gzip.open(path, 'xt') as f:
                json.dump(dict(case=case, binding=__import__('dataclasses').asdict(binding), initial=initial,
                               records=records, outcome=row), f, separators=(',', ':'), allow_nan=False)
            traces.append(dict(path=str(path.resolve()), sha256=file_sha(path)))
    task_metrics = {v: quality(adapter.spec.name, rows[v], rows['reference']) for v in ('raw', 'regularized')}
    program_metrics = {}
    for variant in ('raw', 'regularized'):
        program_metrics[variant] = {}
        for role in adapter.spec.roles:
            data = {k: np.array([r[k] for r in program_data[variant][role.name]]) for k in ('x', 'mask', 'p', 'scene', 'sampled')}
            program = programs[variant][role.name]
            if program.actor_sha256 != assets[variant]['role_sha256'][role.name]:
                raise ValueError('fresh quality program and final NN hash differ')
            program_metrics[variant][role.name] = metrics(program, data)
            with (output/f'{variant}-{role.name}-fresh-program.npz').open('xb') as f: np.savez(f, **data)
    result = dict(version='three-domain-unified-calibration-quality.v1', domain=adapter.spec.name,
        assets=assets, task_metrics=task_metrics, program_metrics=program_metrics,
        rows=rows, traces=traces, independent_audit_completed=False,
        calibration_metrics_passed=all(m['passed'] for m in task_metrics.values()) and
            all(m['passed'] for roles in program_metrics.values() for m in roles.values()),
        model_acceptance_passed=False, pilot_opened=False, human_participants=0)
    write_new(output/'result.json', result)
    return result


def audit_quality(adapter, result_path):
    """Replay immutable frames and recompute outcomes independently of summaries."""
    result_path = Path(result_path); result = json.loads(result_path.read_text()); original_sha = file_sha(result_path)
    rows = {n: [] for n in result['rows']}; checked_actions = 0
    for ref in result['traces']:
        if file_sha(ref['path']) != ref['sha256']: raise ValueError('quality trace hash differs')
        with gzip.open(ref['path'], 'rt') as f: trace = json.load(f)
        case = trace['case']; policy = case['policy']; asset = result['assets'][policy]
        from dataclasses import asdict
        binding = Binding(adapter.spec.sha256, f'{adapter.spec.name}-{policy}-v7',
            'raw' if policy == 'raw' else 'regularized', asset['path'], asset['sha256'], tuple(asset['role_sha256'].items()))
        expected_binding = json.loads(json.dumps(asdict(binding)))
        if trace['binding'] != expected_binding: raise ValueError('quality trace immutable binding differs')
        from .storage import load_frozen
        actors = load_frozen(asset['path'], asset['sha256'], adapter.spec, asset['role_sha256'])
        native = adapter.restore(trace['initial']); initial = adapter.restore(trace['initial'])
        previous_after = None
        for frame in trace['records']:
            if previous_after is not None and previous_after != frame['before']: raise ValueError('quality full frame chain differs')
            for state in (frame['before'], frame['after']):
                if (state['binding'] != expected_binding or state['seed'] != case['seed'] or state['task'] != case['task']
                        or state['action_seed'] != case['action_seed']
                        or digest({k: v for k, v in state.items() if k != 'state_sha256'}) != state['state_sha256']):
                    raise ValueError('quality state identity, random stream or digest differs')
            if digest(adapter.snapshot(native)) != digest(frame['before']['snapshot']): raise ValueError('quality frame chain differs')
            obs = adapter.observe_public(adapter.public(native))
            rng = np.random.default_rng(case['action_seed']); rng.bit_generator.advance(adapter.frame(native)*2)
            uniforms = rng.random(2); indices = {}
            for i, role in enumerate(adapter.spec.roles):
                o = obs[role.name]; z = actors[role.name].logits(o['x'], o['action_features'], o['mask']).astype(float)
                p = np.exp(z-z.max()); p /= p.sum()
                cdf = np.cumsum(p); cdf /= cdf[-1]
                expected = int(np.searchsorted(cdf, uniforms[i], side='right'))
                if expected != frame['decision']['indices'][role.name]: raise ValueError('executed quality action differs from bound NN')
                np.testing.assert_array_equal(p, frame['decision']['probabilities'][role.name])
                indices[role.name] = expected; checked_actions += 1
            if frame['human'] is not None:
                if case['pairing'] != 'offline_human_proxy' or frame['human'] != adapter.proxy(native):
                    raise ValueError('unregistered human replacement')
                indices[adapter.spec.roles[0].name] = frame['human']
            native, _ = adapter.step(native, indices)
            if digest(adapter.snapshot(native)) != digest(frame['after']['snapshot']): raise ValueError('quality physics, collision or scoring differs')
            previous_after = frame['after']
        outcome = dict(seed=case['seed'], task=case['task'], pairing=case['pairing'], complete=True,
                       **adapter.outcome(native, initial))
        if outcome != trace['outcome']: raise ValueError('quality final counters differ')
        rows[policy].append(outcome)
    reproduced = {v: quality(adapter.spec.name, rows[v], rows['reference']) for v in ('raw', 'regularized')}
    if reproduced != result['task_metrics'] or rows != result['rows'] or file_sha(result_path) != original_sha:
        raise ValueError('quality metrics or immutable result changed')
    # This is an execution/quality audit only. Program proofs and full training
    # provenance require their own independent admission; never emit release.
    return dict(version='three-domain-unified-independent-quality-replay.v1',
        result_sha256=original_sha, replayed_role_actions=checked_actions, physical_action_overrides=0,
        task_metrics=reproduced, task_quality_passed=all(v['passed'] for v in reproduced.values()),
        full_scientific_acceptance_passed=False, pilot_opened=False)
