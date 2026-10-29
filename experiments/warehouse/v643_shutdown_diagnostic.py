"""Read-only analysis of spent v6.42 development evidence; no execution authority.

This module does not load a model, construct an environment, fit a program,
allocate seeds, update an optimizer, or change any historical result. A next
candidate proposal is a declaration, never an admission token.
"""
from collections import Counter
from hashlib import sha256
from math import comb, sqrt
from pathlib import Path
import gzip
import json
import statistics

import numpy as np

VERSION = 'warehouse-v643-spent-development-shutdown-diagnostic.v2'
QUALITY_VERSION = 'warehouse-v642-matched-quality-bracketed-verification.v3'
TRAINING_VERSION = 'warehouse-v642-accelerated-calibration-recovery.v3'
VARIANTS = ('reference', 'raw', 'regularized')
ACTIONS = ('UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT')
CHARGER_POSITION = (5, 3)
MAP_LAYOUT_ID = 'warehouse_staggered_aisles_6x7_v2_three_cell_exit_no_cross'
FEATURES = {'self_battery_fraction': 47, 'self_charger_shortest_distance': 97,
            'self_charger_return_margin': 99}


def check(ok, message):
    if not ok:
        raise ValueError(message)


def pointer(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha256(p.read_bytes()).hexdigest()}


def read_bound(ref, inputs, *, binary=False):
    check(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'invalid evidence pointer')
    p = Path(ref['path'])
    check(p.is_file() and not p.is_symlink(), 'missing or symlinked evidence')
    raw = p.read_bytes()
    check(sha256(raw).hexdigest() == ref['sha256'], 'evidence hash mismatch')
    inputs[str(p.resolve())] = dict(ref)
    return raw if binary else json.loads(raw)


def wilson_interval(events, episodes, z=1.959963984540054):
    check(type(events) is int and type(episodes) is int and 0 <= events <= episodes and episodes > 0,
          'invalid complete-episode counts')
    p = events / episodes
    den = 1 + z*z/episodes
    mid = (p + z*z/(2*episodes))/den
    half = z*sqrt(p*(1-p)/episodes + z*z/(4*episodes*episodes))/den
    return [max(0., mid-half), min(1., mid+half)]


def paired_failure_difference(raw, regularized):
    check(set(raw) == set(regularized) and raw, 'paired scenes differ')
    raw_only = sum(raw[k] and not regularized[k] for k in raw)
    reg_only = sum(regularized[k] and not raw[k] for k in raw)
    n = raw_only + reg_only
    exact = min(1., 2*sum(comb(n, j) for j in range(min(raw_only, reg_only)+1))/2**n) if n else 1.
    return {'raw_only_shutdown_scenes': raw_only, 'regularized_only_shutdown_scenes': reg_only,
            'both_shutdown_scenes': sum(raw[k] and regularized[k] for k in raw),
            'paired_discordant_scenes': n, 'two_sided_exact_discordance_p': exact,
            'exploratory_only': True, 'independent_training_pairs': 1,
            'causal_program_effect_established': False}


def summarize_updates(refs, inputs):
    check(refs, 'no actual training update receipts')
    batches = [read_bound(ref, inputs) for ref in refs]
    check(batches[0]['start_joint_step'] == 0 and batches[-1]['end_joint_step'] == 20000,
          'diagnostic requires exact completed 20k history')
    for before, after in zip(batches, batches[1:]):
        check(before['end_joint_step'] == after['start_joint_step'], 'training update history has a gap')
    active = [b for b in batches if any(u['actor_updated'] for e in b['epochs'] for u in e['updates'])]
    updates = [u for b in active for e in b['epochs'] for u in e['updates'] if u['actor_updated']]
    for b in batches:
        check(b['partial_episodes_labeled_negative'] == 0, 'partial episode was assigned a negative label')
        check(0 <= b['shutdown_episodes'] <= b['complete_episodes'], 'invalid complete-episode shutdown label count')
    first_increase = next((b['end_joint_step'] for b in batches if b['risk']['phase'] == 'increase'), None)
    role_counts = {role: sum(b['accepted_positive_program_gradient_updates'][role] for b in batches)
                   for role in ('robot_1', 'robot_2')}
    return {'joint_steps': 20000, 'update_batches': len(batches),
            'complete_training_episodes': sum(b['complete_episodes'] for b in batches),
            'shutdown_training_episodes': sum(b['shutdown_episodes'] for b in batches),
            'Actor_update_batches': len(active),
            'Actor_update_complete_episodes': sum(b['complete_episodes'] for b in active),
            'Actor_update_shutdown_episodes': sum(b['shutdown_episodes'] for b in active),
            'Actor_update_batches_with_shutdown': sum(b['shutdown_episodes'] > 0 for b in active),
            'partial_episodes_labeled_negative': 0, 'first_dual_increase_joint_step': first_increase,
            'risk_multiplier_initial': batches[0]['risk']['previous_multiplier'],
            'risk_multiplier_final': batches[-1]['risk']['next_multiplier'],
            'risk_phase_counts': dict(Counter(b['risk']['phase'] for b in batches)),
            'median_combined_Actor_gradient_norm': statistics.median(u['actor']['gradient_norm'] for u in updates),
            'median_common_gradient_clip_factor': statistics.median(u['common_gradient_clip_factor'] for u in updates),
            'accepted_external_program_updates': role_counts,
            'individual_cost_gradient_or_critic_calibration_not_identifiable_from_receipts': True}


def failure_detail(outcome, entry, inputs):
    check(entry['case'] == outcome['case'] and entry['variant'] == outcome['variant'], 'trace episode identity differs')
    raw = read_bound(entry['trace'], inputs, binary=True)
    trace = [json.loads(line) for line in gzip.decompress(raw).splitlines()]
    check(len(trace) == outcome['physical_frames'] and trace, 'trace horizon differs')
    previous = None
    for i, row in enumerate(trace):
        check(row['frame'] == i and row['pre_state_public']['frame'] == i
              and row['post_state_public']['frame'] == i+1 and row['case'] == outcome['case'],
              'trace frame identity differs')
        check(not row['action_override_agents'] and row['actor_sha256'] == outcome['actor']['sha256'],
              'trace action override or Actor binding differs')
        if previous is not None:
            check(previous == row['pre_state_public'], 'public trace is discontinuous')
        previous = row['post_state_public']
    raw_arrays = read_bound(entry['arrays'], inputs, binary=True)
    import io
    with np.load(io.BytesIO(raw_arrays), allow_pickle=False) as archive:
        check(set(archive.files) == {'x','m','p','sampled'}, 'unknown diagnostic feature archive')
        x, mask, probabilities, sampled = (archive[k] for k in ('x','m','p','sampled'))
    n = len(trace)
    check(x.shape == (n,2,173) and mask.shape == probabilities.shape == (n,2,5)
          and sampled.shape == (n,2) and np.isfinite(x).all() and np.isfinite(probabilities).all(),
          'diagnostic array shape or finiteness differs')
    check(np.issubdtype(sampled.dtype,np.integer) and np.isin(sampled,range(5)).all(), 'saved NN action index is invalid')
    check((np.take_along_axis(mask,sampled[...,None],axis=-1) > 0).all(), 'saved NN action is physically illegal')
    check(np.isin(mask, (0,1)).all() and (probabilities >= 0).all()
          and np.allclose(probabilities.sum(-1),1,atol=1e-5)
          and (probabilities[mask == 0] == 0).all(), 'invalid saved physical action distribution')
    agents = []
    for index, role in enumerate(('robot_1','robot_2')):
        check(all(row['proposed_actions'][role] == ACTIONS[int(sampled[i,index])] for i,row in enumerate(trace)),
              'saved NN sample and physical trace differ')
        before = next(a for a in trace[-1]['pre_state_public']['agents'] if a['agent_id'] == role)
        after = next(a for a in trace[-1]['post_state_public']['agents'] if a['agent_id'] == role)
        if before['active'] and not after['active']:
            margin = x[:,index,FEATURES['self_charger_return_margin']]
            negative = np.flatnonzero(margin < 0)
            nonpositive = np.flatnonzero(margin <= 0)
            preferred = int(probabilities[-1,index].argmax())
            low = np.flatnonzero(x[:,index,FEATURES['self_battery_fraction']] <= .20 + 1e-7)
            agents.append({'role': role, 'pre_battery': before['battery'], 'post_battery': after['battery'],
                'pre_position': before['position'], 'post_position': after['position'],
                'selected_action': trace[-1]['proposed_actions'][role],
                'NN_preferred_action': ACTIONS[preferred],
                'NN_preferred_action_probability': float(probabilities[-1,index,preferred]),
                'sampled_action_differs_from_NN_preference': preferred != int(sampled[-1,index]),
                'pre_position_is_charger': tuple(before['position']) == CHARGER_POSITION,
                'carrying_task_at_shutdown': after['carrying_task_id'],
                'first_negative_charger_return_margin_decision_turn': int(negative[0]) if len(negative) else None,
                'first_nonpositive_charger_return_margin_decision_turn': int(nonpositive[0]) if len(nonpositive) else None,
                'negative_return_margin_decision_frames': int(len(negative)),
                'minimum_normalized_charger_return_margin': float(margin.min()),
                'first_battery_at_most_20_percent_turn': int(low[0]) if len(low) else None,
                'public_recent_positions_before_shutdown': before['recent_positions'],
                'terminal_action_probability': float(probabilities[-1,index,int(sampled[-1,index])])})
    check(agents, 'shutdown outcome has no actual agent shutdown transition')
    return {'variant': outcome['variant'], 'case_id': outcome['case']['case_id'],
            'task': outcome['case']['task'], 'physical_frames': n, 'deliveries': outcome['deliveries'],
            'robot_collisions': outcome['robot_collisions'], 'shutdown_agents': agents,
            'archived_trace_consistency_checked': True, 'new_NN_forward_or_environment_replay': False}


def diagnose(quality_ref, training_ref):
    inputs = {}
    quality, training = read_bound(quality_ref, inputs), read_bound(training_ref, inputs)
    check(quality['version'] == QUALITY_VERSION and training['version'] == TRAINING_VERSION,
          'unexpected archived scientific evidence version')
    check(quality['calibration_precheck_passed'] is False and quality['pilot_opened'] is False
          and training['numerical_training_completed'] is True and training['pilot_opened'] is False,
          'diagnostic requires the original failed completed calibration')
    entries = {(e['variant'],e['case']['case_id']): e for e in quality['entries']}
    check(len(entries) == len(quality['entries']) == len(quality['outcomes']), 'duplicate or missing quality episode')
    metrics, failures = {}, []
    binary = {}
    for variant in VARIANTS:
        rows = [o for o in quality['outcomes'] if o['variant'] == variant]
        check(rows and all(o['complete'] and o['action_overrides'] == 0 for o in rows),
              'quality data has incomplete episodes or action overrides')
        n, k = len(rows), sum(o['shutdown'] for o in rows)
        check(len({o['case']['case_id'] for o in rows}) == n, 'duplicate quality scene')
        binary[variant] = {o['case']['case_id']: o['shutdown'] for o in rows}
        for o in rows:
            if o['shutdown']:
                failures.append(failure_detail(o, entries[(variant,o['case']['case_id'])], inputs))
        metrics[variant] = {'episodes': n, 'shutdown_episodes': k, 'shutdown_rate': k/n,
            'shutdown_rate_wilson_95_percent': wilson_interval(k,n),
            'deliveries_mean': sum(o['deliveries'] for o in rows)/n,
            'robot_collisions_mean': sum(o['robot_collisions'] for o in rows)/n,
            'shutdown_roles': dict(Counter(a['role'] for f in failures if f['variant'] == variant for a in f['shutdown_agents'])),
            'per_task': {str(t): {'episodes': sum(o['case']['task'] == t for o in rows),
                               'shutdown_episodes': sum(o['shutdown'] for o in rows if o['case']['task'] == t)} for t in (1,2,3)}}
    update_metrics = {v: summarize_updates(training['updates'][v], inputs) for v in ('raw','regularized')}
    check(all(n == 0 for n in update_metrics['raw']['accepted_external_program_updates'].values()),
          'raw external program gradient is nonzero')
    check(all(n > 0 for n in update_metrics['regularized']['accepted_external_program_updates'].values()),
          'regularized role external program update is absent')
    return {'version': VERSION, 'scope': 'spent_development_evidence_diagnostic_only',
        'evidence_interpretation': 'hash_bound_receipt_recount_and_public_trace_consistency_not_new_independent_NN_or_training_replay',
        'fixed_feature_columns': FEATURES, 'fixed_charger_position': list(CHARGER_POSITION), 'quality': metrics, 'training_cost_signal': update_metrics, 'shutdown_details': failures,
        'paired_raw_regularized_shutdown_comparison': paired_failure_difference(binary['raw'],binary['regularized']),
        'input_refs': sorted(inputs.values(),key=lambda r:r['path']),
        'existing_failed_calibration_preserved': True, 'new_environment_resets': 0,
        'new_Actor_updates': 0, 'new_NN_forwards': 0, 'new_seed_allocations': 0,
        'training_authority': False, 'model_acceptance_passed': False, 'pilot_opened': False}
