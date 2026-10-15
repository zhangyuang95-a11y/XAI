#!/usr/bin/env python3
"""Finalize completed Pong v5 pairs using the selected-Actor development audit.

This command does not train or change any Actor weights. If a previous
acceptance receipt has already examined held-out seeds, it cannot select a
different pair or repeat model selection from those results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from domains.pong import nn_policy
from scripts import train_rolling_pong_v5 as protocol


def _schedule_hash(schedule):
    return hashlib.sha256(json.dumps(schedule, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def main():
    variants = ('raw', 'regularized')
    source = {}
    for variant in variants:
        path = protocol.ROOT / nn_policy._v5_default_artifacts(variant)['training_report_path']
        source[variant] = json.loads(path.read_text())
    pair_seeds = source['raw']['pair_seeds']
    if (len(pair_seeds) != 3 or len(set(pair_seeds)) != 3
            or source['regularized']['pair_seeds'] != pair_seeds):
        raise RuntimeError('Pong finalization needs exactly three matched pair seeds')
    reports = {seed: {} for seed in pair_seeds}
    snapshots = {seed: {} for seed in pair_seeds}
    for variant in variants:
        for row in source[variant]['paired_runs']:
            seed = row['pair_seed']
            if seed not in reports or variant in reports[seed]:
                raise RuntimeError('Pong paired report is incomplete or duplicated')
            row['seed_schedule_sha256'] = _schedule_hash(row['seed_schedule'])
            reports[seed][variant] = row
            snapshots[seed][variant] = torch.load(
                protocol.ROOT / row['checkpoint_path'], map_location='cpu', weights_only=False)
    for seed in pair_seeds:
        raw, regularized = reports[seed]['raw'], reports[seed]['regularized']
        if (raw['joint_steps'] != protocol.STEPS or regularized['joint_steps'] != protocol.STEPS
                or raw['seed_schedule_sha256'] != regularized['seed_schedule_sha256']
                or raw['initial_actor_sha256'] != regularized['initial_actor_sha256']
                or raw['bc_checkpoint_sha256'] != regularized['bc_checkpoint_sha256']):
            raise RuntimeError(f'Pong pair {seed} is not matched at 200,000 steps')
        if any(run.get('post_policy_override_count') != 0
               for run in (raw, regularized)):
            raise RuntimeError('Pong pair has a post-policy action override')
    qualifying = [seed for seed in pair_seeds
                  if all(reports[seed][variant]['development']['passed'] for variant in variants)]
    audits = {}
    provenance = {}
    ready = []
    for seed in pair_seeds:
        audit, selection = protocol.prepare_candidate_programs(
            snapshots[seed]['regularized'], seed)
        audits[seed] = audit
        provenance[seed] = selection
        reg = reports[seed]['regularized']
        if (seed in qualifying
                and all(reg['program_feedback_rows'][role] > 0 for role in ('human', 'ai'))
                and all(snapshots[seed]['regularized']['programs'].get(role) is not None
                        for role in ('human', 'ai'))
                and all(audit[role]['passed'] for role in ('human', 'ai'))):
            ready.append(seed)
    # Program audit cannot steer seed selection toward a weaker model pair.
    # With no model-qualified pair, the diagnostic selection remains closed.
    preference = qualifying or pair_seeds
    selected = max(preference, key=lambda seed: min(
        protocol.development_score(reports[seed][variant]['development'])
        for variant in variants))
    acceptance_path = protocol.ROOT / nn_policy._v5_default_artifacts('raw')['acceptance_path']
    prior = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    already_evaluated = prior.get('heldout_evaluated') is True
    if already_evaluated:
        selected = prior['selected_pair_seed']
        if selected not in pair_seeds or prior.get('pair_seeds') != pair_seeds:
            raise RuntimeError('prior Pong held-out selection is not a completed pair')
        for variant in variants:
            entry = nn_policy._v5_default_artifacts(variant)
            if (prior.get('artifact_sha256', {}).get(variant)
                    != protocol.sha(protocol.ROOT / entry['actor_path'])
                    or prior.get('training_report_sha256', {}).get(variant)
                    != protocol.sha(protocol.ROOT / entry['training_report_path'])
                    or source[variant].get('selected_pair_seed') != selected
                    or variant not in prior.get('heldout', {})):
                raise RuntimeError('prior Pong held-out receipt does not match frozen assets')
    development_audit = audits.get(selected)
    if development_audit is None:
        development_audit = protocol.audit_selected_programs(snapshots[selected]['regularized'])
    snapshots[selected]['regularized']['development_program_audit'] = development_audit
    feedback_ok = all(reports[selected]['regularized']['program_feedback_rows'][role] > 0
                      for role in ('human', 'ai'))
    program_ok = all(snapshots[selected]['regularized']['programs'].get(role) is not None
                     and development_audit[role]['passed'] for role in ('human', 'ai'))
    heldout_evaluated = (already_evaluated or
                         (selected in ready and feedback_ok and program_ok))
    if heldout_evaluated and not already_evaluated:
        protocol.reserve_heldout_once(acceptance_path, pair_seeds, selected, reports)
    heldout = {}
    parity = {}
    entries = {}
    for variant in variants:
        models = protocol.restore_models(snapshots[selected][variant])
        if already_evaluated:
            heldout[variant] = prior['heldout'][variant]
        elif heldout_evaluated:
            heldout[variant] = protocol.base.evaluate(models, protocol.HELDOUT_SEEDS)
        else:
            heldout[variant] = {'passed': False, 'reason': 'no_qualified_development_pair'}
        report = dict(source[variant])
        report['selected_pair_seed'] = selected
        report['development_tuning'] = protocol.DEVELOPMENT_TUNING
        report['paired_runs'] = [reports[seed][variant] for seed in pair_seeds]
        report['candidate_program_audits'] = {str(seed): audits[seed] for seed in pair_seeds}
        report['candidate_program_provenance'] = {str(seed): provenance[seed]
                                                  for seed in pair_seeds}
        report['heldout'] = heldout[variant]
        report['selected_actor_development_program_audit'] = (
            development_audit if variant == 'regularized' else {})
        report['finalization'] = {'selected_actor_audit_basis': 'fresh_development_states',
                                  'heldout_reused_from_prior_receipt': already_evaluated,
                                  'model_weights_unchanged': True}
        entries[variant] = protocol.export_variant(variant, snapshots[selected][variant], report)
        if heldout_evaluated:
            payload = json.loads((protocol.ROOT / entries[variant]['actor_path']).read_text())
            parity[variant] = protocol.runtime_parity(
                models, payload, protocol.PARITY_SEEDS)
        else:
            parity[variant] = False
    candidate_training = []
    for seed in pair_seeds:
        candidate = {'pair_seed': seed}
        for variant in variants:
            row = reports[seed][variant]
            candidate[variant] = {key: row[key] for key in (
                'joint_steps', 'bc_checkpoint_sha256', 'initial_actor_sha256',
                'seed_schedule_sha256', 'actor_weight_change_l2',
                'program_feedback_rows', 'post_policy_override_count')}
        candidate_training.append(candidate)
    passed = (heldout_evaluated and selected in ready and feedback_ok and program_ok
              and all(parity.values()) and all(heldout[v]['passed'] for v in variants))
    acceptance = {
        'release': 'policylens-three-domain-20261014.v5.0', 'domain': 'pong',
        'passed': passed, 'heldout_evaluated': heldout_evaluated,
        'pair_seeds': pair_seeds, 'selected_pair_seed': selected,
        'candidate_training': candidate_training,
        'candidate_program_audits': {str(seed): audits[seed] for seed in pair_seeds},
        'candidate_program_provenance': {str(seed): provenance[seed]
                                          for seed in pair_seeds},
        'development_tuning': protocol.DEVELOPMENT_TUNING,
        'joint_steps': {variant: reports[selected][variant]['joint_steps'] for variant in variants},
        'development': {variant: reports[selected][variant]['development'] for variant in variants},
        'heldout': heldout, 'runtime_parity': parity,
        'feedback_positive': feedback_ok, 'explanation_eligible': program_ok,
        'selected_actor_development_program_audit': development_audit,
        'artifact_sha256': {variant: entries[variant]['actor_sha256'] for variant in variants},
        'training_report_sha256': {variant: entries[variant]['training_report_sha256']
                                   for variant in variants},
        'program_sha256': {'regularized': {
            role: entries['regularized']['program_sha256'].get(path)
            for role, path in zip(('human', 'ai'), entries['regularized']['program_paths'])}},
        'raw_program_feedback_rows': reports[selected]['raw']['program_feedback_rows'],
        'regularized_program_feedback_rows': reports[selected]['regularized']['program_feedback_rows'],
        'action_override_count': 0, 'post_policy_override_count': 0,
    }
    protocol.write_json(acceptance_path, acceptance)
    protocol.write_json(nn_policy.V5_MANIFEST_PATH, {
        'format': 'pong-paired-policies-v5', 'release': acceptance['release'],
        'variants': entries})
    print(json.dumps({'passed': passed, 'selected_pair_seed': selected,
                      'heldout_evaluated': heldout_evaluated,
                      'development_program_audit': development_audit,
                      'runtime_parity': parity}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
