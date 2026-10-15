"""Read the fixed 500k/arm feedback cycle without restoring its training chain.

The external completion SHA anchors an immutable ledger snapshot. Default
inspection hashes checkpoints as opaque bytes; optional CPU deserialization is
registered in advance for two terminal boundary states and the first positive
lambda update only. Original gradient metrics are checked, never recomputed.
Failed capability and tree results are retained. This module grants no release.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import gzip
from hashlib import sha256
import io
import json
from pathlib import Path

import numpy as np

from . import warehouse_family_feedback_cycle_run as runner
from . import warehouse_family_feedback_cycle_source as source
from . import warehouse_native_shutdown_feedback_result as previous
from . import warehouse_native_shutdown_feedback_trainer as trainer
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_shutdown_result import _Inputs, _json_bytes, _same, _int, _configuration
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic

VERSION = 'warehouse-family-feedback-cycle-result.v1'
BRANCHES = runner.BRANCHES
evaluation = previous.evaluation
expanded = runner.expanded
OVERLAPS = ('scene_fingerprints', 'episode_ids', 'public_states', 'exact_observations')


def sources():
    value = runner.sources()
    for path in (Path(previous.__file__), Path(__file__)):
        value[str(path.relative_to(ROOT))] = file_hash(path)
    return value


def _require(condition, message):
    if not condition: raise ValueError(message)


def _contract(p, protocol, scenes, completion, fixture):
    """Pure outer identity/schedule checks; no source loader or live ledger."""
    _require(p.get('version') == runner.VERSION and p.get('test_fixture') is fixture,
             'The genuine new feedback cycle and explicit fixture scope are required')
    _same(p['identity'], {k: v for k, v in p.items() if k not in
        ('identity', 'created_unix', 'formal_ready', 'explanation_qualified')}, 'Prepared identity differs')
    _same(p['guard'], runner.GUARD, 'Registered strict development guard differs')
    endpoints, probe = runner._schedule(p['primary_endpoint'], p['validation_interval'],
        p['environment_batch_size'], fixture, p['fixture_probe_endpoint'])
    _same(p['validation_endpoints'], endpoints, 'Fixed validation boundaries differ')
    _same(p['probe_endpoint'], probe, 'Registered within-cap probe differs')
    _require(protocol.get('version') == trainer.PROTOCOL_VERSION and protocol.get('test_fixture') is fixture
        and protocol.get('cycle_id') == p['cycle_id'] and protocol.get('branch') == 'own_credit'
        and protocol.get('shutdown_arm') == p['shutdown_arm'] == 'beta1'
        and protocol.get('own_shutdown_beta') == p['own_shutdown_beta'] == 1.
        and protocol.get('delivery_credit_alpha') == .5 and protocol.get('public_feedback_mode') == 'observed'
        and protocol.get('feedback_training_enabled') is True
        and protocol.get('explanation_qualification_granted') is False,
        'Actual retained-beta feedback protocol differs')
    _same(protocol['feedback'], trainer.feedback_contract(), 'PPO/KL contract differs')
    _same(protocol['feedback_branches'], list(BRANCHES), 'Paired conditions differ')
    _same(protocol['feedback_config'], asdict(expanded.feedback_config()), 'Expanded feedback configuration differs')
    _same(protocol['budget'], {'maximum_ppo_joint_steps_per_arm': p['primary_endpoint'],
        'maximum_ppo_joint_steps': p['primary_endpoint'], 'curriculum_generation_steps': 0}, 'Finite PPO cap differs')
    _same(protocol['evaluation']['checkpoints_ppo_steps'], endpoints, 'Registered validation matrix differs')
    base = deepcopy(protocol)
    for key in ('native_protocol', 'feedback_branches', 'feedback_config', 'feedback', 'explanation_qualification_granted'):
        base.pop(key)
    base.update(version=trainer.native.PROTOCOL_VERSION, feedback_training_enabled=False)
    _same(protocol['native_protocol'], base, 'Inner learner contract was changed or relabeled')
    _same(protocol['source_lineage'][-1], protocol['source'], 'Actual source lineage differs')
    for key, expected in (('checkpoint_sha256', p['source_checkpoint_sha256']),
            ('state_sha256', p['source_state_sha256']), ('cumulative_joint_steps', p['source_cumulative_steps'])):
        _same(protocol['source'][key], expected, 'Actual source binding differs: '+key)
    _require(protocol['training']['environments'] == p['environment_batch_size']
        and protocol['evaluation']['read_final_test'] is False, 'Training matrix or final pool boundary differs')
    _configuration(scenes, fixture)
    _require(len(scenes['splits']['validation']) > 0, 'Fixed validation pool is empty')
    if not fixture:
        _require(p['source_run'] == str(source.SOURCE) and p['source_completion_sha256'] == source.SOURCE_COMPLETION_SHA
            and p['source_cumulative_steps'] == source.SOURCE_CLOCK
            and p['initial_tree']['root'] == str(source.TREE)
            and p['initial_tree']['manifest_sha256'] == source.TREE_MANIFEST_SHA
            and len(scenes['splits']['validation']) == 50, 'Production must retain the registered 3.38m source and original tree')
    for key, expected in (('version', runner.VERSION), ('status', 'fixed_endpoint_completed'),
            ('until', p['primary_endpoint']), ('cycle_id', p['cycle_id']),
            ('source_cumulative_steps', p['source_cumulative_steps']), ('formal_ready', False),
            ('explanation_qualified', False), ('used_final_test', False), ('program_runtime_controller', False),
            ('total_interaction_equal', False), ('initial_tree_budget_is_separate', True)):
        _same(completion.get(key), expected, 'Incomplete or different fixed cycle: '+key)


def _prepare(root, read, completion, fixture):
    p = read.json(root, 'prepared.json'); protocol = read.json(root, 'protocol.json'); scenes = read.json(root, 'scenarios.json')
    _contract(p, protocol, scenes, completion, fixture)
    _same(p['runtime_sources'], runner.sources(), 'Current frozen training closure differs')
    for name, expected in p['runtime_sources'].items():
        read.raw(ROOT, name, expected); read.raw(root/'source_snapshot', name, expected)
    for value, key in ((protocol, 'protocol_sha256'), (scenes, 'scenario_manifest_sha256')):
        _same(digest(value), p[key], 'Bound input differs: '+key)
    _same(p['authorization_record_sha256'], runner.AUTHORIZATION_SHA, 'Original authorization differs')
    for name, key in (('authorization.json', 'authorization_record_sha256'), ('baselines.json', 'baselines_sha256'),
                      ('source_validation.json', 'source_report_sha256')): read.raw(root, name, p[key])
    admission = read.json(root, 'initial_admission.json')
    _same(digest(admission), p['initial_tree']['binding_sha256'], 'Initial admission receipt differs')
    _same(admission['descriptor'], p['source_descriptor'], 'Original current-endpoint source identity differs')
    _same(admission['descriptor']['checkpoint']['sha256'], p['source_checkpoint_sha256'], 'Original source checkpoint differs')
    for name, item in admission['input_bindings'].items():
        path = Path(name); _require(path.is_absolute(), 'Original source binding must be absolute')
        read.raw(path.parent, path.name, item['sha256'], item['size'])
    fit = read.json(root, 'initial_fit.json')
    _same(digest(fit), p['initial_tree']['fit_sha256'], 'Original admitted tree differs')
    _require(fit.get('version') == expanded.VERSION and fit.get('reliable') is True,
             'Initial training-reliable tree is missing')
    initial = read.json(root, 'initialization_check.json')
    _require(initial.get('version') == runner.VERSION and initial.get('identical_native_learning_state') is True
        and initial.get('new_environment_steps') == 0 and initial.get('program_runtime_controller') is False
        and initial.get('source_state_sha256') == p['source_state_sha256'], 'Complete same-state initialization is missing')
    required = {'model', 'optimizers', 'envs', 'rng', 'python_rng', 'numpy_rng', 'torch_rng', 'partner_kinds',
        'program_roles', 'scenario_ids', 'episode_context', 'episode_returns', 'episode_reward_components'}
    _require(set(initial['matched']) == required and all(evaluation.compact._sha(x) for x in initial['matched'].values()),
             'Initial dual Adam, environment or RNG record is incomplete')
    pools = read.json(root, 'refresh_pools.json'); contexts = read.json(root, 'refresh_contexts.json')
    _same(digest(pools), p['refresh_pools_sha256'], 'Original refresh pools differ')
    _same(digest(contexts), p['refresh_contexts_sha256'], 'Original refresh contexts differ')
    expected, caps, aux = runner._caps(p['validation_endpoints'], pools, scenes)
    _same(contexts, expected, 'Extraction ordering or seeds differ')
    _same(p['budget_caps'], caps, 'PPO/evaluation caps differ')
    _same(p['auxiliary_caps'], aux, 'Ten feedback refreshes plus one terminal control tree are required')
    _same(p['maximum_auxiliary_steps'], sum(sum(v.values()) for v in aux.values()), 'Auxiliary total differs')
    if not fixture:
        _require((len(pools['train']), len(pools['selection']), len(contexts)) == (192, 32, 896), 'Fixed extraction matrix differs')
    _same(p['extraction_contract'], expanded.contract(), 'Extraction contract differs')
    _require(p['collection_primitive_version'] == runner.primitive.VERSION
        and p['maximum_rcpd_calls_per_extraction'] == 13 and p['maximum_sklearn_fits_per_extraction'] == 104,
        'Actual extraction producer or search bounds differ')
    start = read.json(root, 'source_validation.json'); baseline = read.json(root, 'baselines.json')
    guard = source.gate_values(start, start, team_reference=baseline['reference'])
    _require(guard['capability_eligible'], 'The fixed starting Actor was not admitted by both gates')
    _same(initial['initial_gate']['strict_guard'], guard['diagnostics'], 'Initial strict gate receipt differs')
    return p, protocol, scenes, initial, start, baseline


def _fit_record(fit, plan, features, row_counts):
    """Check original fit selection arithmetic, without fitting or NN queries."""
    _require(fit.get('version') == expanded.VERSION and fit.get('test_fixture') is plan['test_fixture']
        and type(fit.get('reliable')) is bool and fit.get('prediction_semantics') == expanded.program_batch.VERSION
        and fit.get('explanation_qualified') is False and fit.get('release_eligible') is False
        and fit.get('actual_joint_steps') == 0 and fit.get('neural_training_updates') == 0,
        'Original expanded fit identity or qualification scope differs')
    fr = fit['fit_report']
    if 'input_rejection' in fit:
        _require(fit['reliable'] is False and fit.get('program') is None and fit.get('manager_state') is None,
                 'Rejected fit cannot supply a reliable program')
        return {'reliable': False, 'input_rejection': deepcopy(fit['input_rejection']), 'reason': fr.get('reason'),
            'candidate_fits_verified_from_reports': 0, 'candidate_fit_count_scope': 'input rejection; actual partial fit count unavailable'}
    binding = fr['observed197_bindings']; expected = plan['actor_bindings']
    for key in ('actor_sha256', 'actor_parameters_sha256'):
        _same(binding[key], expected[key], 'Fitted tree belongs to another Actor')
    _same(binding['cumulative_fit_step'], plan['cumulative_fit_step'], 'Tree uses another cumulative clock')
    _same(binding['config_sha256'], digest(expanded.contract()), 'Tree uses another configuration')
    _same(fit['evidence_sha256'], digest({'binding': binding, 'fit_report': fr}), 'Original tree evidence digest differs')
    _same(fr['overlap'], dict.fromkeys(OVERLAPS, 0), 'Fit silently accepts cross-pool overlap')
    _same(fr['extraction_config'], expanded.contract(), 'Fit extraction contract differs')
    _require(fr.get('rows_removed') == 0 and fr.get('collector_producer') == runner.extraction.VERSION
        and fr.get('intervention_direction_not_tested_here') is True and fr.get('explanation_qualified') is False
        and fr.get('step') == plan['cumulative_fit_step'], 'Fit provenance or interpretation differs')
    _same([fr['train_rows'], fr['validation_rows']], [row_counts['train'], row_counts['selection']], 'Fit rows differ from actual saved NN rows')
    candidates = fr['candidates']
    _same([(c['depth_cap'], c['leaf_cap']) for c in candidates], list(expanded.CANDIDATES), 'The thirteen registered candidates differ')
    for candidate in candidates:
        _same(candidate['gates'], expanded.gates(candidate['selection_metrics']), 'Candidate gate arithmetic differs')
        m = candidate['selection_metrics']
        _same(candidate['selection_objective'], 1-m['overall']['fidelity']+.2*m['mean_kl']+.001*candidate['complexity']['loss'],
              'Candidate objective arithmetic differs')
    reliable = [c for c in candidates if c['gates']['reliable']]
    selected = min(reliable, key=lambda c: (c['complexity']['loss'], c['selection_metrics']['mean_kl'],
        -c['selection_metrics']['overall']['fidelity'], c['depth_cap'], c['leaf_cap'])) if reliable else min(candidates, key=lambda c: c['selection_objective'])
    _same(fr['selected'], selected, 'Registered simplest reliable selection differs')
    _same(fr['selection_metrics'], selected['selection_metrics'], 'Selected metrics differ')
    _same(fr['reliable'], selected['gates']['reliable'], 'Recorded fit reliability differs')
    _same(fit['reliable'], fr['reliable'], 'Outer fit reliability differs')
    manager = fit['manager_state']; program = fit['program']; meta = program['metadata']
    _same(manager['program'], program, 'Program and manager differ')
    _same(manager['last_fit_report'], fr, 'Manager fit report differs')
    _same(manager['config'], plan['expanded_fit_contract']['feedback_config'], 'Manager configuration differs')
    _same(meta['metrics']['reliable'], fit['reliable'], 'Program reliability differs')
    for names in (manager['feature_names'], program['feature_names']): _same(names, features, 'Actual 197 feature order differs')
    for key, value in (('reliable', fit['reliable']), ('current_lambda', 0.), ('last_step', plan['cumulative_fit_step']),
                        ('last_fit_step', plan['cumulative_fit_step'])):
        _same(manager[key], value, 'Fresh tree manager differs: '+key)
    _same(meta['observed197_bindings'], binding, 'Program provenance differs')
    _same(meta['native_feedback_config'], plan['expanded_fit_contract']['feedback_config'], 'Program configuration differs')
    _require(meta.get('native_feedback_version') == expanded.VERSION
        and meta.get('native_source_actor_sha256') == expected['actor_sha256']
        and meta.get('prediction_semantics') == expanded.program_batch.VERSION
        and meta.get('metrics', {}).get('explanation_eligible') is False,
        'Program incorrectly advertises explanation qualification')
    return {'reliable': fit['reliable'], 'selected': deepcopy(selected), 'selection_metrics': deepcopy(fr['selection_metrics']),
        'train_rows': fr['train_rows'], 'selection_rows': fr['validation_rows'], 'overlap': fr['overlap'],
        'candidate_fits_verified_from_reports': len(candidates), 'candidate_fit_count_scope': 'thirteen saved candidate results; no fit replay',
        'independent_intervention_tested': False}


def _extraction(root, read, p, arm, step, report, marker, features):
    folder = f'branches/{arm}/refresh/step_{step:07d}'; saved = {}
    for name in ('plan.json', 'manifest.json', 'fit_result.json', 'fit_request.json', 'auxiliary_budget.json'):
        item = marker['evidence']['tree_'+name.replace('.', '_')]
        _require(item['path'] == folder+'/'+name, 'Tree evidence belongs to another arm or boundary')
        saved[name] = _json_bytes(read.bound(root, item))
    plan, manifest, fit, request, allocation = (saved[n] for n in
        ('plan.json', 'manifest.json', 'fit_result.json', 'fit_request.json', 'auxiliary_budget.json'))
    _require(plan.get('version') == runner.primitive.VERSION and plan.get('parent_driver') == runner.VERSION
        and plan.get('pair_version') == runner.VERSION and plan.get('feedback_branch') == arm
        and manifest.get('version') == runner.primitive.VERSION and manifest.get('parent_driver') == runner.VERSION
        and manifest.get('status') == 'completed' and manifest.get('ppo_steps') == 0
        and plan.get('ppo_steps') == 0 and plan.get('test_fixture') is p['test_fixture']
        and plan.get('explanation_qualified') is False, 'True nested collection producer or completion differs')
    for key, value in (('pair_identity_sha256', digest(p['identity'])), ('protocol_sha256', p['protocol_sha256']),
            ('scene_pools_sha256', p['refresh_pools_sha256']), ('runtime_sources', p['runtime_sources']),
            ('cumulative_fit_step', p['source_cumulative_steps']+step), ('expanded_fit_contract', expanded.contract()),
            ('extraction_config', runner.extraction.extraction_config()), ('maximum_rcpd_calls', 13), ('maximum_sklearn_fits', 104),
            ('purpose', 'feedback_refresh' if arm == 'feedback' else 'control_terminal_tree')):
        _same(plan[key], value, 'Extraction contract differs: '+key)
    _same(plan['contexts'], read.json(root, 'refresh_contexts.json'), 'Extraction changed its matrix')
    _same(plan['actor_bindings'], report['actor_bindings'], 'Validation and extraction use different actual Actors')
    _same(manifest['plan_sha256'], digest(plan), 'Collection plan binding differs')
    _require(len(manifest['episodes']) == len(plan['contexts']), 'All registered extraction ACKs are required')
    reserved = actual = 0; row_counts = dict(train=0, selection=0)
    for entry, context in zip(manifest['episodes'], plan['contexts']):
        _same(entry['context'], context, 'Extraction episode order differs')
        _require(entry['status'] == 'completed', 'Pending extraction cannot be retried or omitted')
        data = _json_bytes(gzip.decompress(read.bound(root/folder, entry['record'])))
        with np.load(io.BytesIO(read.bound(root/folder, entry['arrays'])), allow_pickle=False) as values:
            _require(len(values.files) == 2 and set(values.files) == {'observations', 'probabilities'}, 'Original extraction arrays differ')
            data.update({key: values[key].copy() for key in values.files})
        runner.extraction._validate_data(data, p['test_fixture'])
        episode = data['episode']; amount = _int(data['joint_transitions'], 1)
        _same(data['actor_bindings'], plan['actor_bindings'], 'Actual collected labels use another Actor')
        _same(data['feature_names'], features, 'Collected feature order differs')
        _same(data['actor_training_clock'], plan['cumulative_fit_step'], 'Collected clock differs')
        _same(data['data_sha256'], entry['data_sha256'], 'Acknowledged data digest differs')
        _same(digest(data['trace']), episode['trace_sha256'], 'Original raw extraction trace differs')
        _require(amount == entry['actual_joint_steps'] and amount <= context['horizon']
            and len(data['observations']) == entry['neural_rows'] and data['pool'] == context['pool']
            and episode['id'] == context['episode_id'] and episode['fingerprint'] == context['scenario_fingerprint'],
            'Saved collection rows or original episode identity differ')
        for key in ('profile', 'program_role', 'seed', 'sampling_mode', 'horizon'):
            _same(episode[key], context[key], 'Original collection context differs: '+key)
        reserved += context['horizon']; actual += amount; row_counts[context['pool']] += entry['neural_rows']
    cap = p['auxiliary_caps'][arm][f'step_{step:07d}']
    _same([reserved, manifest['reserved_auxiliary_steps'], allocation['reserved_joint_steps'], allocation['cap'], plan['maximum_auxiliary_steps']],
          [cap]*5, 'Permanent auxiliary reservations differ')
    _same(actual, manifest['actual_auxiliary_steps'], 'Actual auxiliary ACK count differs')
    for name, value in (('fit_request', request), ('fit_result', fit)):
        _same(_json_bytes(read.bound(root/folder, manifest[name])), value, 'Original fit receipt differs')
    for key, value in (('version', runner.primitive.VERSION), ('plan_sha256', digest(plan)),
            ('acknowledged_episodes_sha256', digest(manifest['episodes'])), ('prior_manager_state_sha256', plan['prior_manager_sha256']),
            ('feedback_config_sha256', digest(plan['expanded_fit_contract']['feedback_config'])),
            ('actor_sha256', plan['actor_bindings']['actor_sha256']), ('cumulative_fit_step', plan['cumulative_fit_step']),
            ('maximum_candidate_fits', 13), ('maximum_sklearn_fits', 104),
            ('extraction_contract_sha256', digest(expanded.contract())), ('retry_allowed', False)):
        _same(request[key], value, 'Original finite fit request differs: '+key)
    if 'input_rejection' in fit: _same(fit['input_rejection']['request_sha256'], digest(request), 'Rejected fit request differs')
    tree = _fit_record(fit, plan, features, row_counts)
    _same(manifest['reliable'], fit['reliable'], 'Collection/fit reliability differs')
    terminal = {'reliable': fit['reliable'], 'actor_sha256': plan['actor_bindings']['actor_sha256'],
        'actor_parameters_sha256': plan['actor_bindings']['actor_parameters_sha256'],
        'cumulative_fit_step': plan['cumulative_fit_step'], 'fit_result': marker['evidence']['tree_fit_result_json'],
        'explanation_qualified': False}
    _same(marker['terminal_tree'], terminal, 'Same-Actor terminal tree receipt differs')
    cell = {'reserved': reserved, 'acknowledged': actual, 'status': 'completed', 'reliable': fit['reliable'],
        'manifest': marker['evidence']['tree_manifest_json']}
    return cell, {**tree, 'episodes': len(manifest['episodes']), 'actual_auxiliary_steps': actual,
        'reserved_auxiliary_steps': reserved, 'fit_result_sha256': terminal['fit_result']['sha256'], 'actor_binding': terminal}


def _boundary(root, read, p, arm, step, account, report, start, baseline):
    prefix = f'branches/{arm}/boundaries/step_{step:07d}'; marker = read.json(root, prefix+'.json')
    matches = [op for op in account['operations'].values() if op['request']['kind'] == 'ppo'
        and op['request']['branch'] == arm and op['completion']['checkpoint']['step'] == step]
    _require(len(matches) == 1, 'Boundary has no unique acknowledged PPO predecessor')
    _same(marker['predecessor_head'], matches[0]['completion']['checkpoint'], 'Boundary predecessor differs')
    _require(marker.get('version') == runner.VERSION and marker.get('branch') == arm
        and marker.get('new_training_steps') == 0 and marker['checkpoint']['path'] == prefix+'.pt', 'Boundary identity differs')
    read.bound(root, marker['checkpoint'])
    needs_tree = arm == 'feedback' or step == p['primary_endpoint']
    names = {'validation_report', 'validation_manifest'}
    if needs_tree: names |= {'tree_'+n.replace('.', '_') for n in
        ('fit_result.json', 'fit_request.json', 'manifest.json', 'plan.json', 'auxiliary_budget.json')}
    _same(sorted(marker['evidence']), sorted(names), 'Boundary evidence scope differs')
    folder = f'branches/{arm}/validation/step_{step:07d}'
    for key, name in (('validation_report', 'report.json'), ('validation_manifest', 'manifest.json')):
        _require(marker['evidence'][key]['path'] == folder+'/'+name, 'Validation belongs to another boundary')
        read.bound(root, marker['evidence'][key])
    _same(read.json(root, folder+'/report.json'), report, 'Saved and recomputed validation report differ')
    raw = read.raw(root, f'branches/{arm}/actors/actor_{step:07d}.npz', report['actor_bindings']['actor_sha256'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = _json_bytes(str(archive['metadata_json'].item()))
    _same(metadata['source_counters']['joint_steps']+metadata['joint_steps'], p['source_cumulative_steps']+step, 'Actual Actor cumulative clock differs')
    guard = source.gate_values(report, start, team_reference=baseline['reference'])
    _same(marker['gate']['strict_guard'], guard['diagnostics'], 'Saved full AND warmup / NN90% guard differs')
    cell = tree = None
    if needs_tree:
        cell, tree = _extraction(root, read, p, arm, step, report, marker, metadata['feature_names'])
    else: _same(marker['terminal_tree'], None, 'Control has an unregistered intermediate tree')
    expected_eligible = guard['capability_eligible'] and (arm == 'control' or tree['reliable'])
    _same(marker['gate']['capability_eligible'], expected_eligible, 'Tree/strict feedback admission differs')
    manager_gate = marker['gate']['manager']; strength = manager_gate.get('lambda')
    _require(type(strength) in (int, float) and np.isfinite(strength) and 0 <= strength <= .01, 'Boundary lambda is invalid')
    if arm == 'control':
        _same(manager_gate, {'active': False, 'lambda': 0., 'reason': 'control'}, 'Control acquired feedback')
    elif not expected_eligible:
        _require(strength == 0 and manager_gate.get('active') is False, 'Failed guard or tree retained active feedback')
    return marker, cell, tree


def _training_scope(root, read, training, boundaries, p):
    """Count actual positive-lambda NN rows and reject use after a closed guard."""
    for arm, summary in training.items():
        summary.update(positive_lambda_updates=0, positive_lambda_neural_rows=0)
        for operation in summary['operations']:
            if operation['lambda'] <= 0: continue
            before = operation['step']-operation['steps']
            prior = [s for s in p['validation_endpoints'] if s <= before]
            if prior:
                gate = boundaries[(arm, prior[-1])]['gate']
                _require(gate['capability_eligible'] is True,
                         'Positive feedback follows a failed fixed guard or unreliable refresh')
            with np.load(io.BytesIO(read.bound(root, operation['arrays'])), allow_pickle=False) as arrays:
                mask = arrays['trainable']
                _require(np.isfinite(mask).all() and np.isin(mask, [0, 1]).all(), 'Saved actual-NN mask is invalid')
                rows = int(np.count_nonzero(mask))
            operation['positive_lambda_neural_rows'] = rows
            summary['positive_lambda_updates'] += 1; summary['positive_lambda_neural_rows'] += rows
        summary['gradient_scope'] = 'No per-update PT decoding; only explicitly registered first-positive checkpoint metrics'


def _registration(p, boundaries, first_positive):
    result = [{'branch': arm, 'kind': 'final_effective_boundary',
        'checkpoint': boundaries[(arm, p['primary_endpoint'])]['checkpoint']} for arm in BRANCHES]
    if first_positive is not None:
        result.append({'branch': 'feedback', 'kind': 'first_positive_lambda_update', 'checkpoint': first_positive['checkpoint']})
    _require(len(result) <= 3 and len({v['checkpoint']['path'] for v in result}) == len(result), 'At most three distinct registered CPU checkpoint decodes')
    return result


def _state_identity(state, p, protocol, arm, step):
    _require(state.get('version') == trainer.VERSION and state.get('feedback_branch') == arm
        and state.get('feedback_enabled') is (arm == 'feedback') and state.get('test_fixture') is p['test_fixture'], 'Actual trainer identity differs')
    _same(state['protocol'], protocol, 'Actual paired checkpoint protocol differs')
    _same(state['protocol_sha256'], digest(protocol), 'Paired protocol digest differs')
    _same(state['execution_sources'], trainer.execution_sources(), 'Actual trainer execution closure differs')
    _same(state['source_sha256'], digest(state['execution_sources']), 'Actual trainer source digest differs')
    native = state['native_state']
    _require(native['version'] == trainer.native.VERSION and native['joint_steps'] == step
        and native['source_counters']['joint_steps'] == p['source_cumulative_steps'], 'Actual native clock differs')
    _same(native['protocol'], protocol['native_protocol'], 'Actual retained native contract differs')
    for key, expected in (('cycle_id', p['cycle_id']), ('branch', 'own_credit'), ('shutdown_arm', 'beta1'),
            ('own_shutdown_beta', 1.), ('delivery_credit_alpha', .5), ('public_feedback_mode', 'observed'),
            ('obs_dim', 197), ('state_dim', 354), ('test_fixture', p['test_fixture']), ('feedback_enabled', False),
            ('training_device', p['device']), ('source_lineage', protocol['source_lineage']),
            ('protocol_sha256', digest(protocol['native_protocol'])), ('scenario_manifest_sha256', p['scenario_manifest_sha256'])):
        _same(native[key], expected, 'Retained native checkpoint binding differs: '+key)
    _same(native['sources'], trainer.native.execution_sources(), 'Native checkpoint sources differ')
    _same(native['source_sha256'], digest(native['sources']), 'Native source digest differs')
    _same(native['initialization_sha256'], p['source_state_sha256'], 'Checkpoint came from another source state')
    _same(native['source_checkpoint_sha256'], p['source_checkpoint_sha256'], 'Checkpoint came from another source artifact')
    _require(all(key in native for key in ('envs', 'rng', 'python_rng', 'numpy_rng', 'torch_rng', 'optimizers', 'model')),
             'Complete native learning state is missing')
    _require(len(native['envs']) == p['environment_batch_size'], 'Saved environment batch differs')
    if p['device'] == 'mps': _require('mps_rng' in native, 'Owned MPS RNG is missing')
    # This frozen validator inspects CPU tensor shapes, finiteness, both Adam
    # moment sets, hyperparameters and absolute steps. It constructs no model.
    trainer.native._validate_learning_state(native['model'], native['optimizers'], protocol['training'],
        {role: native['source_counters'][role+'_optimizer_steps']+native[role+'_optimizer_steps'] for role in ('actor', 'critic')})
    return native


def _positive_metrics(metrics, strength):
    _require(all(type(v) in (int, float) and np.isfinite(v) for v in metrics.values()), 'Nonfinite saved update metrics')
    _require(metrics['feedback_lambda'] == strength > 0 and metrics['feedback_gradient_norm'] >= 0
        and metrics['feedback_rows'] > 0 and metrics['feedback_kl'] >= -1e-7
        and np.isclose(metrics['feedback_loss'], strength*metrics['feedback_kl'], rtol=1e-5, atol=1e-8),
        'Committed positive KL metrics are inconsistent')
    return {'metrics': deepcopy(metrics), 'nonzero_feedback_gradient_observed': metrics['feedback_gradient_norm'] > 0,
        'scope': 'Original committed minibatch-average gradient norm/loss; no gradient recomputation'}


def _checkpoint_audits(root, read, p, protocol, boundaries, reports, first_positive, completion, registered):
    audited = []
    for entry in registered:
        payload = previous._decode_cpu(read, root, entry['checkpoint']); arm = entry['branch']
        _require(payload.get('version') == runner.VERSION and payload.get('branch') == arm
            and payload.get('cycle_id') == p['cycle_id'], 'Checkpoint outer producer differs')
        if entry['kind'] == 'first_positive_lambda_update':
            native = _state_identity(payload['trainer'], p, protocol, arm, first_positive['step'])
            _require(payload['operation_id'] == first_positive['operation_id']
                and payload['audit']['neural_overrides'] == 0 and payload['actual_steps'] == first_positive['steps'], 'First positive update is not the registered acknowledged update')
            for kind in ('arrays', 'trace'): _same(payload['evidence'][kind], first_positive[kind], 'Positive update batch binding differs')
            audited.append({**entry, **_positive_metrics(payload['metrics'], first_positive['lambda'])})
        else:
            marker = boundaries[(arm, p['primary_endpoint'])]; state = payload['trainer']
            native = _state_identity(state, p, protocol, arm, p['primary_endpoint'])
            for key in ('predecessor_head', 'evidence', 'gate', 'terminal_tree', 'new_training_steps'):
                _same(payload[key], marker[key], 'Terminal effective checkpoint and receipt differ')
            binding = reports[p['primary_endpoint']][arm]['actor_bindings']
            weights = {k[len('actor.'):]: v for k, v in native['model'].items() if k.startswith('actor.')}
            _require(set(weights) == set(source.FIELDS) and semantic(weights) == binding['actor_parameters_sha256'], 'Terminal six-array semantic hash differs')
            with np.load(io.BytesIO(read.raw(root, f'branches/{arm}/actors/actor_{p["primary_endpoint"]:07d}.npz', binding['actor_sha256'])), allow_pickle=False) as archive:
                for name, value in weights.items():
                    _require(str(value.dtype) == 'torch.float32' and np.array_equal(value.numpy(), archive[name]), 'Actual terminal Actor export differs')
            counters = {key: native[key] for key in ('optimizer_updates', 'actor_optimizer_steps', 'critic_optimizer_steps')}
            _same({'ppo_steps': native['joint_steps'], **counters}, completion['counts'][arm], 'Terminal actual learning counters differ')
            adam = {}
            for role in ('actor', 'critic'):
                entries = native['optimizers'][role]['state']; _require(bool(entries), 'Dual Adam state is empty')
                steps = {int(v['step'].item()) for v in entries.values()}
                expected = native['source_counters'][role+'_optimizer_steps']+native[role+'_optimizer_steps']
                _same(sorted(steps), [expected], 'Actual retained dual Adam steps differ'); adam[role] = expected
                for item in entries.values():
                    _require(all(np.isfinite(item[k].numpy()).all() for k in ('exp_avg', 'exp_avg_sq')), 'Nonfinite Adam moments')
            if arm == 'control':
                _require(all(state.get(k) is None for k in ('feedback_state', 'feedback_evidence', 'capability_evidence', 'refresh_failure'))
                    and state.get('refresh_pending') is False, 'Control acquired feedback state')
            else:
                manager = expanded.ExactProgramManager(list(reports[p['primary_endpoint']][arm]['feature_names'])
                    if 'feature_names' in reports[p['primary_endpoint']][arm] else list(state['feedback_state']['feature_names']),
                    trainer.validated_feedback_config(protocol['feedback_config'], p['test_fixture']))
                manager.load_state_dict(state['feedback_state']); trainer.validate_program(manager, p['test_fixture'])
                trainer.algorithm._validate_refresh_state(manager, state['capability_evidence'], state['refresh_pending'], state['refresh_failure'])
                _same(manager.current_lambda, marker['gate']['manager']['lambda'], 'Terminal manager lambda differs')
                if marker['terminal_tree']['reliable']:
                    _require(state['refresh_pending'] is False and manager.reliable is True, 'Reliable terminal refresh was not installed')
                    _same(state['feedback_evidence']['source_actor_sha256'], binding['actor_sha256'], 'Installed final tree uses an old Actor')
                    _same(state['feedback_evidence']['source_actor_parameters_sha256'], binding['actor_parameters_sha256'], 'Installed final tree uses old parameters')
                    _same(manager.last_fit_step, p['source_cumulative_steps']+p['primary_endpoint'], 'Installed final tree clock differs')
                else:
                    _require(state['refresh_pending'] is True and manager.current_lambda == 0 and not manager.reliable,
                             'Failed final refresh did not close the previous tree')
            audited.append({**entry, 'actor_parameters_sha256': semantic(weights), 'all_six_export_arrays_equal': True,
                'dual_adam_absolute_steps': adam, 'complete_state_fields_present': True,
                'scope': 'Current checkpoint fields and CPU tensors; environment/RNG are not replayed'})
        del payload, native
    return audited


def compare_fixed(output, *, expected_completion_sha256, decode_checkpoints=False, allow_test_fixture=False):
    """Inspect the completed fixed endpoint; never invoke run/admit/read_prepared."""
    source._sha(expected_completion_sha256)
    _require(type(decode_checkpoints) is bool and type(allow_test_fixture) is bool, 'Explicit decoding and fixture scope required')
    root = Path(output).expanduser().resolve(); read = _Inputs(); closure = sources()
    cap = read.json(root, 'prepared.json')['primary_endpoint'] if allow_test_fixture else runner.PPO_CAP
    completion = read.json(root, f'completion_{cap:07d}.json', expected_completion_sha256)
    p, protocol, scenes, initial, start, baseline = _prepare(root, read, completion, allow_test_fixture)
    account = completion['ledger']; operations = previous._ledger(root, read, p, account)
    training, first_positive = previous._ppo_records(root, read, p, protocol, operations)
    for arm in BRANCHES:
        c = completion['counts'][arm]
        _require(c['ppo_steps'] == cap and c['optimizer_updates'] == training[arm]['updates'], 'Final PPO/update counters differ from actual ACKs')
    _same(completion['counts']['control'], completion['counts']['feedback'], 'PPO/dual-optimizer allocations differ')
    reports = {}; boundaries = {}; trees = {arm: {} for arm in BRANCHES}; aux = {arm: {} for arm in BRANCHES}; actual_eval = 0
    for step in p['validation_endpoints']:
        reports[step] = {}
        for arm in BRANCHES:
            # The original stage reader recomputes saved rows/gates. It creates
            # a NumPy Actor object, but never calls its forward or an environment.
            report = runner._report_record(root, arm, step, p, protocol, scenes, account, allow_test_fixture)
            reports[step][arm] = report; actual_eval += report['environment_steps']
            marker, cell, tree = _boundary(root, read, p, arm, step, account, report, start, baseline)
            boundaries[(arm, step)] = marker
            if tree is not None: aux[arm][f'step_{step:07d}'] = cell; trees[arm][str(step)] = tree
            folder = f'branches/{arm}/validation/step_{step:07d}'; manifest = read.json(root, folder+'/manifest.json')
            for entry in manifest['episodes']:
                for kind in ('row', 'trace'): read.bound(root/folder, entry[kind])
            read.raw(root, f'branches/{arm}/actors/actor_{step:07d}.json')
    _same(actual_eval, account['totals']['evaluation']['acknowledged'], 'Validation reports omit actual evaluation ACKs')
    for arm in BRANCHES: _same(sorted(aux[arm]), sorted(p['auxiliary_caps'][arm]), 'A registered extraction boundary is missing')
    auxiliary = {'cap': p['maximum_auxiliary_steps'], 'reserved': sum(c['reserved'] for v in aux.values() for c in v.values()),
        'acknowledged': sum(c['acknowledged'] for v in aux.values() for c in v.values()), 'branches': aux}
    _same(auxiliary, completion['auxiliary'], 'Completed auxiliary accounting differs')
    terminal = {arm: boundaries[(arm, cap)]['terminal_tree'] for arm in BRANCHES}
    _same(terminal, completion['terminal_trees'], 'Both fixed terminal tree attempts are required')
    _training_scope(root, read, training, boundaries, p)
    registered = _registration(p, boundaries, first_positive)
    decoded = _checkpoint_audits(root, read, p, protocol, boundaries, reports, first_positive, completion, registered) if decode_checkpoints else []
    _same(sources(), closure, 'Result inspection sources changed'); read.unchanged()
    counts = read.counts(); counts.pop('actor_constructions', None)
    counts.update(checkpoint_decodes=len(decoded), numpy_actor_loads=2*len(p['validation_endpoints']), torch_nn_constructions=0, fits=0)
    comparison = previous._comparison(reports[cap])
    for arm in BRANCHES:
        comparison[arm]['strict_development_guard'] = source.gate_values(reports[cap][arm], start, team_reference=baseline['reference'])
        comparison[arm]['terminal_tree'] = trees[arm][str(cap)]
    return {'version': VERSION, 'runner_version': runner.VERSION, 'status': 'fixed_endpoint_complete', 'cycle_id': p['cycle_id'],
        'primary_endpoint': cap, 'source_cumulative_steps': p['source_cumulative_steps'],
        'final_cumulative_steps_per_arm': p['source_cumulative_steps']+cap,
        'primary': comparison, 'descriptive_endpoints': {str(s): previous._comparison(r) for s, r in reports.items() if s != cap},
        'training': training, 'completed_learning_counts': completion['counts'], 'budget': account['totals'], 'auxiliary': auxiliary,
        'trees': trees, 'registered_checkpoint_audits': registered, 'checkpoint_audits': decoded,
        'first_positive_feedback_operation': first_positive, 'same_source_initialization': initial,
        'initialization_scope': 'Bound original same-state preparation record; initial PTs hashed, not decoded',
        'feedback_gradient_scope': 'Only the first positive update committed metrics when explicitly decoded; no full gradient history or gradient recomputation',
        'tree_metric_scope': 'Original 13-candidate metrics and gate/selection arithmetic bound to actual ACKed labels; no fitting, NN or physics replay',
        'completion_sha256': expected_completion_sha256, 'result_sources': closure, 'input_bindings': read.records,
        'inspection_counts': counts, 'test_fixture': allow_test_fixture, 'source_selection': None,
        'total_interaction_equal': False, 'initial_tree_budget_is_separate': True,
        'scope': 'Single-seed fixed endpoint development comparison; failed gates retained, no causal or independent explanation claim',
        'release_ready': False, 'formal_ready': False, 'explanation_qualified': False, 'used_final_test': False}


def write_result(output, *, expected_completion_sha256, destination, decode_checkpoints=False, allow_test_fixture=False):
    """Write into a new directory only after the complete readonly inspection."""
    target = Path(destination).expanduser().resolve(); root = Path(output).expanduser().resolve()
    _require(not target.exists() and target != root and root not in target.parents,
             'Use a new independent result directory; original run artifacts are immutable')
    result = compare_fixed(root, expected_completion_sha256=expected_completion_sha256,
        decode_checkpoints=decode_checkpoints, allow_test_fixture=allow_test_fixture)
    primary = result['primary']; lines = [f"固定两支各 {result['primary_endpoint']:,} PPO 步结果。", '']
    lines.append(f"NN 配送均值：control {primary['control']['primary_value']:.6f}；feedback {primary['feedback']['primary_value']:.6f}；差 {primary['feedback_minus_control']:+.6f}。")
    for arm in BRANCHES:
        value = primary[arm]
        lines.append(f"{arm}：完整能力 {value['full_capability']['eligible']}；warmup {value['warmup_capability']['eligible']}；严格保留门槛 {value['strict_development_guard']['capability_eligible']}；末端树可靠 {value['terminal_tree']['reliable']}。")
    lines += ['', f"PPO 确认 {result['budget']['ppo']['acknowledged']:,}；验证确认/预留 {result['budget']['evaluation']['acknowledged']:,}/{result['budget']['evaluation']['reserved']:,}；抽树辅助确认/预留 {result['auxiliary']['acknowledged']:,}/{result['auxiliary']['reserved']:,}。",
        '全部已保存 NN 训练行原样提交已核对；正 λ 与梯度核验范围见 JSON。保存梯度指标不等于重新计算梯度。',
        '中途检查仅作描述；失败不删除、不择优替代固定终点。单 seed 开发比较，不授予解释、研究或发布资格。']
    target.mkdir(parents=True, exist_ok=False)
    runner.write_json(target/'result.json', result); runner.write_bytes(target/'result.md', ('\n'.join(lines)+'\n').encode())
    return result
