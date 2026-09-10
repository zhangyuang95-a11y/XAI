"""Development-only RCPD from genuine frozen-Actor intervention labels.

This producer neither changes NN parameters nor grants participant eligibility.
The old trajectory-only producers and their failed acceptance stay unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from core.rcpd import RCPDConfig
from core.program import ExecutableProgram
from core.policy_program_regularizer import program_complexity
from env.warehouse_native.feedback import _NativeRCPD
from env.warehouse_native.policy import ACTIONS
from . import warehouse_native_expanded_rcpd as original
from . import warehouse_native_program_batch as prediction
from . import warehouse_native_shutdown_stage_rcpd as original_producer
from .warehouse_native_common import ROOT
from .warehouse_native_continuation_rcpd import _isolated_host_rng

VERSION = 'warehouse-family-development-intervention-rcpd.v1'
GROUPS = ('narrow_passage', 'shared_pickup', 'shared_charger')
CANDIDATES = original.CANDIDATES


def contract():
    return {'version': VERSION, 'role': 'robot_2',
        'candidates': [list(x) for x in CANDIDATES],
        'maximum_rcpd_calls': len(CANDIDATES),
        'maximum_sklearn_fits': sum(d for d, _ in CANDIDATES),
        'action_structure_weight': 0.0, 'counterfactual_changed_pair_weight': 1.0,
        'counterfactual_loss_weight': 0.2, 'minimum_fidelity': .9,
        'minimum_base_fidelity': .9, 'minimum_non_wait_fidelity': .9,
        'minimum_critical_fidelity': .85, 'minimum_direction_fidelity': .85,
        'minimum_critical_scenarios': 10, 'maximum_mean_kl': .35,
        'selection': 'simplest_program_passing_all_development_gates',
        'labels': 'same_frozen_NN_soft_distributions_only',
        'pair_definition': 'physical_effect_and_different_NN_argmax; both_endpoints_must_match',
        'pair_presentation': 'retain_all_unique_rows; append_each_physical_pair_as_two_tagged_views',
        'fit_view_weights': 'one_per_view; repeated_endpoints_have_greater_aggregate_fit_weight',
        'independent_acceptance_thresholds_modified': False,
        'pair_duplication_is_not_new_observation_or_NN_query': True,
        'environment_steps': 0, 'ppo_steps': 0, 'torch_loads': 0,
        'feedback_enabled': False, 'explanation_qualified': False, 'release_ready': False}


def hashed(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def _rate(values, fingerprints):
    return {'rows': len(values), 'scenarios': len(set(fingerprints)),
        'fidelity': float(np.mean(values)) if values else None}


def statistics(data, predicted):
    """Unweighted original rows and original pairs; duplication never raises rates."""
    truth = np.asarray(data['actions'])
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError('Prediction rows differ')
    if not set(truth) <= set(ACTIONS) or not set(predicted) <= set(ACTIONS):
        raise ValueError('Unknown action')
    same = predicted == truth
    fp = data['scene_fingerprints']

    def subset(indices):
        return _rate([bool(same[i]) for i in indices], [fp[i] for i in indices])

    result = {'overall': subset(range(len(truth))),
        'base': subset([i for i, k in enumerate(data['kind']) if k == 'base']),
        'counterfactual': subset([i for i, k in enumerate(data['kind']) if k == 'counterfactual']),
        'non_wait': subset([i for i, a in enumerate(truth) if a != 'WAIT']),
        'by_action': {a: subset([i for i, x in enumerate(truth) if x == a]) for a in ACTIONS},
        'critical': {}, 'direction': {}}
    for name in GROUPS:
        indices = [i for i, groups in enumerate(data['groups']) if name in groups]
        result['critical'][name] = {**subset(indices),
            'non_wait': subset([i for i in indices if truth[i] != 'WAIT'])}
    pairs = data['pairs']
    for pair in pairs:
        if bool(truth[pair['baseline_index']] != truth[pair['changed_index']]) != pair['nn_changed']:
            raise ValueError('Saved pair eligibility differs from genuine NN actions')
    valid = [p for p in pairs if p['physical_effect'] and p['nn_changed']]

    def direction(selected):
        return _rate([bool(same[p['baseline_index']] and same[p['changed_index']]) for p in selected],
            [p['scene_fingerprint'] for p in selected])

    result['direction']['all'] = direction(valid)
    for name in GROUPS:
        result['direction'][name] = direction([p for p in valid if name in p['groups']])
    result['direction']['all_pairs'] = len(pairs)
    result['direction']['valid_pairs'] = len(valid)
    return result


def gates(metrics):
    def meets(value, minimum, scenes=1):
        return (value['rows'] > 0 and value['scenarios'] >= scenes
            and value['fidelity'] is not None and value['fidelity'] >= minimum)
    checks = {name: meets(metrics[name], .9) for name in ('overall', 'base', 'non_wait')}
    checks['mean_kl'] = metrics['mean_kl'] <= .35
    for name in GROUPS:
        checks[name] = (meets(metrics['critical'][name], .85, 10)
            and meets(metrics['critical'][name]['non_wait'], .85, 10))
    for name in ('all', *GROUPS):
        checks['direction_' + name] = meets(metrics['direction'][name], .85, 10)
    return {'checks': checks, 'passed': all(checks.values())}


def samples(data):
    """Expose paired views to existing RCPD, preserving unique-row reporting."""
    x, y = data['observations'], data['probabilities']
    values = [{'obs': a, 'probabilities': b, 'episode': e, 'pair': None}
        for a, b, e in zip(x, y, data['episode_ids'])]
    for index, pair in enumerate(data['pairs']):
        if not pair['physical_effect']:
            continue
        a, b = pair['baseline_index'], pair['changed_index']
        if data['episode_ids'][a] != data['episode_ids'][b]:
            raise ValueError('A physical pair crosses episodes')
        for i in (a, b):
            values.append({'obs': x[i], 'probabilities': y[i],
                'episode': data['episode_ids'][i], 'pair': f"{data['episode_ids'][i]}:pair:{index}"})
    return values


def evaluate(program, data, feature_names):
    probabilities = prediction.predict(program, data['observations'], feature_names)
    result = statistics(data, [ACTIONS[i] for i in probabilities.argmax(axis=1)])
    target = data['probabilities']
    result['mean_kl'] = float(np.mean(np.sum(target * (
        np.log(target.clip(1e-8)) - np.log(probabilities.clip(1e-8))), axis=-1)))
    return result


def canonical_program(program, feature_names, metadata=None):
    # RCPD may sort encoder keys; predicates use names, while batch input uses
    # the frozen NN column order. This does not change any split or probability.
    if (len(set(feature_names)) != len(feature_names)
            or set(program.feature_names) != set(feature_names)
            or tuple(program.action_names) != tuple(ACTIONS)):
        raise ValueError('RCPD changed the real NN feature/action schema')
    return ExecutableProgram(program.action_names, tuple(feature_names), program.root,
        program.metadata if metadata is None else metadata)


def fit_collected(root, *, expected_plan_sha256, expected_manifest_sha256, output):
    from . import warehouse_family_intervention_dataset as reader
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or root.is_relative_to(output) or output.is_relative_to(root):
        raise ValueError('Use a separate new fit output; pending fits are never retried')
    data = reader.read_completed(root, expected_plan_sha256=expected_plan_sha256,
        expected_manifest_sha256=expected_manifest_sha256)
    if data['plan'].get('test_fixture') is not False:
        raise ValueError('Production fitting refuses fixture data')
    train, selection = data['data']['train'], data['data']['selection']
    feature_names = data['plan']['feature_names']
    if len(feature_names) != 197 or tuple(ACTIONS) != ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT'):
        raise ValueError('Frozen feature or action schema differs')
    training_samples, selection_samples = samples(train), samples(selection)
    sources = {str(ROOT / p): value for p, value in original.execution_sources(original_producer).items()}
    sources.update({str(Path(p).resolve()): hashed(p) for p in
        (__file__, reader.__file__, original.__file__, prediction.__file__)})
    output.mkdir(parents=True, exist_ok=False)
    request = {'version': VERSION, 'contract': contract(), 'source': str(root),
        'plan_sha256': expected_plan_sha256, 'collection_manifest_sha256': expected_manifest_sha256,
        'actor_bindings': data['plan']['bindings'], 'source_hashes': sources,
        'input_bindings': data['input_bindings'], 'test_fixture': False,
        'unique_rows': {k: len(v['actions']) for k, v in data['data'].items()},
        'fit_presentation_rows': {'train': len(training_samples), 'selection': len(selection_samples)},
        'automatic_retry': False}
    put(output/'request.json', request)
    records = []
    encoder = lambda item: dict(zip(feature_names, map(float, item['obs'])))
    oracle = lambda item: dict(zip(ACTIONS, map(float, item['probabilities'])))
    try:
        for index, (depth, leaves) in enumerate(CANDIDATES):
            cfg = RCPDConfig(max_depth=depth, max_leaf_nodes=leaves, max_predicates=None,
                min_samples_leaf=8, complexity_penalty=.001, random_seed=260908,
                regularization_lambda=.01, action_structure_weight=0.,
                counterfactual_changed_pair_weight=1., counterfactual_loss_weight=.2)
            put(output/f'fit_{index:02d}.request.json', {'index': index, 'config': asdict(cfg),
                'maximum_sklearn_fits': depth, 'retry_allowed': False})
            with _isolated_host_rng():
                result = _NativeRCPD(cfg).fit(training_samples, oracle, encoder,
                    validation_states=selection_samples,
                    split_group_provider=lambda item: item['episode'],
                    counterfactual_pair_provider=lambda item: item['pair'],
                    program_metadata={'native_source_actor_sha256': data['plan']['bindings']['actor_sha256'],
                        'native_feedback_version': VERSION, 'runtime_controller': 'native_neural_actor_only',
                        'role_scope': 'robot_2_only', 'feedback_eligible': False,
                        'explanation_eligible': False, 'prediction_semantics': prediction.VERSION})
            metadata = deepcopy(result.program.metadata)
            metadata['development_intervention_bindings'] = deepcopy(data['plan']['bindings'])
            metadata['metrics'] = {**metadata.get('metrics', {}),
                'feedback_eligible': False, 'explanation_eligible': False, 'feedback_weight': 0.,
                'feedback_ineligibility_reasons': ['development_robot_2_only_posthoc'],
                'explanation_ineligibility_reasons': ['fresh_independent_intervention_acceptance_pending']}
            program = canonical_program(result.program, feature_names, metadata)
            metrics = evaluate(program, selection, feature_names)
            complexity = program_complexity(program, max_depth=16, max_leaf_count=256,
                max_predicate_count=255).to_dict()
            entry = {'index': index, 'depth_cap': depth, 'leaf_cap': leaves,
                'selection_metrics': metrics, 'gates': gates(metrics), 'complexity': complexity,
                'core_extraction_summary': list(result.extraction_summary)}
            put(output/f'program_{index:02d}.json', program.to_dict())
            put(output/f'fit_{index:02d}.report.json', entry)
            records.append(entry)
            print(json.dumps({'event': 'intervention_fit_ack', 'index': index,
                'passed': entry['gates']['passed'], 'direction': metrics['direction']['all'],
                'base': metrics['base']}), flush=True)
        passing = [r for r in records if r['gates']['passed']]
        selected = min(passing, key=lambda r: (r['complexity']['loss'],
            r['selection_metrics']['mean_kl'], -r['selection_metrics']['overall']['fidelity'])) if passing else None
        if any(hashed(path) != value for path, value in sources.items()):
            raise ValueError('Fitting source changed')
        report = {'version': VERSION, 'status': 'completed', 'development_passed': bool(passing),
            'selected': selected, 'candidates': records, 'request_sha256': hashed(output/'request.json'),
            'actual_rcpd_calls': len(records), 'maximum_sklearn_fits': sum(d for d, _ in CANDIDATES),
            'new_environment_steps': 0, 'new_NN_queries': 0, 'ppo_updates': 0,
            'role_scope': 'robot_2_only', 'independent_acceptance_executed': False,
            'feedback_enabled': False, 'explanation_qualified': False, 'release_ready': False}
        put(output/'report.json', report)
        return report
    except BaseException as error:
        put(output/'failure.json', {'version': VERSION, 'error_type': type(error).__name__,
            'message': str(error), 'completed_rcpd_calls': len(records), 'pending_fit_not_retried': True,
            'partial_fit_count_may_be_unknown': True, 'release_ready': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source'); parser.add_argument('--plan-sha256')
    parser.add_argument('--manifest-sha256'); parser.add_argument('--output')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(contract(), sort_keys=True)); return
    if not all((args.source, args.plan_sha256, args.manifest_sha256, args.output)):
        parser.error('Execution requires source, both actual hashes, and a new output')
    result = fit_collected(args.source, expected_plan_sha256=args.plan_sha256,
        expected_manifest_sha256=args.manifest_sha256, output=args.output)
    print(json.dumps({k: result[k] for k in ('status', 'development_passed', 'actual_rcpd_calls', 'release_ready')}))


if __name__ == '__main__':
    main()
