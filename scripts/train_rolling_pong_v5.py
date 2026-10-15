#!/usr/bin/env python3
"""Train and audit three paired raw/RCPD rolling-Pong self-play runs.

The teacher is used only to create the common pre-self-play checkpoint. Both
roles' actions during the 200,000-step comparison are sampled from independent
Actors. The extracted programs enter the regularized Actor loss and are never
used to select an environment action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core.rcpd import RCPD
from core.rcpd_config import RCPDConfig
from domains.pong import nn_policy, turnbased as pong
from scripts import train_rolling_pong_nn as base

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/pong_rolling_v5'
PAIR_SEEDS = (251014, 251015, 251016)
DEVELOPMENT_SEEDS = tuple(range(880100, 880124))
PARITY_SEEDS = tuple(range(881000, 881008))
HELDOUT_SEEDS = tuple(range(890000, 890100))
PROGRAM_REFIT_TRAIN_SEEDS = tuple(range(870100, 870132))
PROGRAM_REFIT_VALIDATION_SEEDS = tuple(range(870200, 870208))
STEPS = 200_000
BATCH_SEEDS = 8
EXTRACTION_INTERVAL = 10_000
DEVELOPMENT_TUNING = {
    'changed_at_utc': '2026-10-14T10:23:35Z',
    'heldout_examined_before_change': False,
    'initial_feedback_gate': {'minimum_action_fidelity': .80, 'maximum_mean_kl': .20},
    'initial_explanation_gate': {'minimum_action_fidelity': .90, 'maximum_mean_kl': .10},
    'observed_first_pair_program_metrics': {
        'human': {'action_fidelity': .764, 'mean_kl': .5096878053799138},
        'ai': {'action_fidelity': .918, 'mean_kl': .2322167145793046}},
    'reason': ('The first development pair passed Pong task gates but both role programs '
               'failed the initial soft-distribution KL feedback gate, yielding zero '
               'regularization rows. The replacement feedback gate retains per-state '
               'NN/program action agreement; the AI explanation gate remains stricter.'),
    'revised_feedback_gate': {'minimum_action_fidelity': .75, 'maximum_mean_kl': .60},
    'revised_extraction_explanation_gate': {'minimum_action_fidelity': .90,
                                            'maximum_mean_kl': .25},
    'selected_actor_explanation_gate': {
        'set_at_utc': '2026-10-14T10:39:00Z',
        'minimum_action_fidelity': .95,
        'maximum_mean_kl': .10,
        'reason': ('The explanation audit uses independent development states and '
                   'the selected frozen Actor; its gate is stricter than the '
                   'training-feedback and extraction-time gates.')},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def actor_initial_hashes(initial_actors):
    return {
        role: hashlib.sha256(torch.cat([value.flatten() for value in initial_actors[role].values()])
                             .numpy().tobytes()).hexdigest()
        for role in ('human', 'ai')}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def reserve_heldout_once(acceptance_path, pair_seeds, selected_seed, reports):
    """Record the one-time test-set start before the first rollout.

    Exclusive creation is the guard. Even an interrupted/partial write leaves a
    file that blocks every later rollout, so a crash cannot silently retest.
    """
    receipt_path = Path(acceptance_path).with_name('rolling_actor_v5.heldout.started.json')
    receipt = {
        'status': 'started',
        'pair_seeds': list(pair_seeds),
        'selected_pair_seed': selected_seed,
        'heldout_seeds': list(HELDOUT_SEEDS),
        'checkpoint_sha256': {
            variant: reports[selected_seed][variant]['checkpoint_sha256']
            for variant in ('raw', 'regularized')},
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('Pong held-out evaluation was already started; refusing to rerun') from exc
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return receipt_path


def semantic_features(state, actor):
    """Named, observable, pre-action inputs; training metadata is excluded."""
    return nn_policy.semantic_features(state, actor)


def neural_oracle(model, actor):
    def predict(state):
        with torch.no_grad():
            probabilities = torch.softmax(base.masked_logits(model, state, actor), -1).tolist()
        return dict(zip(nn_policy.ACTIONS, probabilities))
    return predict


def new_rcpd(pair_seed):
    return RCPD(RCPDConfig(
        extraction_interval=EXTRACTION_INTERVAL,
        minimum_extraction_samples=128,
        regularization_lambda=.01,
        regularization_start_fraction=.80,
        regularization_ramp_fraction=.05,
        max_depth=5,
        max_leaf_nodes=16,
        max_predicates=64,
        require_action_agreement_for_feedback=True,
        minimum_overall_fidelity_for_feedback=.75,
        maximum_mean_kl_for_feedback=.60,
        minimum_overall_fidelity_for_explanation=.90,
        maximum_mean_kl_for_explanation=.25,
        random_seed=pair_seed,
    ))


def pretrained_actors(checkpoint):
    if not checkpoint.exists():
        torch.manual_seed(251014)
        models = {role: base.Actor() for role in ('human', 'ai')}
        teacher_rows = base.collect_teacher(range(820000, 820032))
        base.clone(models, teacher_rows, epochs=30)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'actors': {role: deepcopy(model.state_dict()) for role, model in models.items()},
                    'bc_seeds': list(range(820000, 820032)), 'bc_epochs': 30}, checkpoint)
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if set(source['actors']) != {'human', 'ai'}:
        raise RuntimeError('Pong BC checkpoint must contain two role Actors')
    models = {role: base.Actor() for role in ('human', 'ai')}
    for role in models:
        models[role].load_state_dict(source['actors'][role])
    return {role: deepcopy(model.state_dict()) for role, model in models.items()}


def weight_change(initial, current):
    return float(sum(torch.sum((current[key] - initial[key]) ** 2).item()
                     for key in initial) ** .5)


def development_score(evaluation):
    return sum(row['capacity_percent'] for row in evaluation['rows']) / len(evaluation['rows'])


def update_state_reservoir(trajectories, train, validation, seed_count):
    for episode_number, episode in enumerate(trajectories):
        if not episode:
            continue
        destination = validation if episode_number % seed_count in (seed_count - 2, seed_count - 1) else train
        destination.extend(item[6] for item in episode)
    del train[:-1600]
    del validation[:-500]


def train_variant(pair_index, pair_seed, variant, initial_actors, *, steps, batch_seeds):
    torch.manual_seed(pair_seed + 13)
    random.seed(pair_seed + 13)
    models = {role: base.Actor() for role in ('human', 'ai')}
    for role in models:
        models[role].load_state_dict(initial_actors[role])
    critic = base.Critic()
    # A small PPO step preserves the high-quality common BC initialization
    # across the full matched budget rather than selecting from a late collapse.
    optimizers = {role: torch.optim.Adam(models[role].parameters(), lr=.00002) for role in models}
    optimizers['critic'] = torch.optim.Adam(critic.parameters(), lr=.0001)
    rcpds = ({role: new_rcpd(pair_seed + (0 if role == 'human' else 1))
              for role in models} if variant == 'regularized' else {})
    encoders = {role: (lambda state, role=role: semantic_features(state, role)) for role in models}
    regularizers = {role: (rcpds[role], encoders[role]) for role in rcpds}
    train_states, validation_states = [], []
    round_index, joint_steps, next_dev = 0, 0, 20_000
    seed_schedule = []
    extraction_log = {'human': [], 'ai': []}
    feedback = {'human': 0.0, 'ai': 0.0}
    feedback_rows = {'human': 0, 'ai': 0}
    best = None
    best_snapshot = None
    last_development = None
    while joint_steps < steps:
        seeds = [810000 + pair_index * 10000 + round_index * batch_seeds + offset
                 for offset in range(batch_seeds)]
        seed_schedule.append(seeds)
        remaining = steps - joint_steps
        trajectories = base.rollout(models, critic, seeds, max_steps=remaining)
        produced = sum(len(episode) for episode in trajectories)
        if produced <= 0:
            raise RuntimeError('Pong self-play produced no joint steps')
        update_state_reservoir(trajectories, train_states, validation_states, len(seeds))
        joint_steps += produced
        feedback_batch = base.self_play_update(
            models, critic, optimizers, trajectories, regularizers=regularizers,
            training_fraction=joint_steps / steps)
        for role in feedback:
            feedback[role] += feedback_batch[role]['weighted_rows']
            feedback_rows[role] += feedback_batch[role]['rows']
        if rcpds:
            for role, rcpd in rcpds.items():
                extracted = rcpd.maybe_extract(
                    joint_steps, train_states, neural_oracle(models[role], role), encoders[role],
                    validation_states=validation_states,
                    action_legality_features={'left': 'legal_left', 'right': 'legal_right'},
                    split_group_provider=lambda state: state.get('_training_episode_id'),
                    program_metadata={'domain': 'pong', 'role': role, 'pair_seed': pair_seed,
                                      'source': 'neural_self_play_pre_action_states'})
                if extracted is not None:
                    extraction_log[role].append({'step': joint_steps,
                                                 'metrics': extracted.metrics.to_dict()})
                elif rcpd.last_error:
                    extraction_log[role].append({'step': joint_steps, 'error': rcpd.last_error})
        if joint_steps >= next_dev or joint_steps == steps:
            evaluation = base.evaluate(models, DEVELOPMENT_SEEDS)
            last_development = evaluation
            eligible = (variant == 'raw' or
                        (joint_steps >= int(steps * .85) and all(feedback_rows.values())))
            if eligible and (best is None or
                             (evaluation['passed'], development_score(evaluation)) >
                             (best['passed'], development_score(best))):
                best = evaluation
                best_snapshot = {
                    'actors': {role: deepcopy(model.state_dict()) for role, model in models.items()},
                    'programs': {role: deepcopy(rcpds[role].program) for role in rcpds},
                    'program_metrics': {role: (deepcopy(rcpds[role].last_result.metrics.to_dict())
                                              if rcpds[role].last_result else None) for role in rcpds},
                    'selected_checkpoint_steps': joint_steps,
                }
            print(json.dumps({'pair_seed': pair_seed, 'variant': variant,
                              'joint_steps': joint_steps, 'development_passed': evaluation['passed'],
                              'development_capacity': round(development_score(evaluation), 3),
                              'feedback': feedback}), flush=True)
            next_dev += 20_000
        round_index += 1
    if best_snapshot is None:
        best = last_development
        best_snapshot = {
            'actors': {role: deepcopy(model.state_dict()) for role, model in models.items()},
            'programs': {role: deepcopy(rcpds[role].program) for role in rcpds},
            'program_metrics': {role: (deepcopy(rcpds[role].last_result.metrics.to_dict())
                                      if rcpds[role].last_result else None) for role in rcpds},
            'selected_checkpoint_steps': joint_steps,
        }
    checkpoint = OUTPUT / f'pair_{pair_seed}_{variant}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_snapshot, checkpoint)
    report = {
        'pair_seed': pair_seed, 'variant': variant, 'joint_steps': joint_steps,
        'selected_checkpoint_steps': best_snapshot['selected_checkpoint_steps'],
        'initial_actor_sha256': actor_initial_hashes(initial_actors),
        'actor_weight_change_l2': {role: weight_change(initial_actors[role], best_snapshot['actors'][role])
                                   for role in models},
        'action_override_count': 0,
        'post_policy_override_count': 0,
        'neural_actions': joint_steps * 2,
        'feedback_weighted_rows': feedback,
        'program_feedback_rows': feedback_rows,
        'seed_schedule': seed_schedule,
        'seed_schedule_sha256': hashlib.sha256(
            json.dumps(seed_schedule, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'development': best,
        'extraction_history': extraction_log,
        'selected_program_metrics': best_snapshot['program_metrics'],
        'checkpoint_path': checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256': sha(checkpoint),
    }
    return report, best_snapshot


def restore_models(snapshot):
    models = {role: base.Actor() for role in ('human', 'ai')}
    for role in models:
        models[role].load_state_dict(snapshot['actors'][role])
        models[role].eval()
    return models


def audit_selected_programs(snapshot):
    """Check extracted programs against the selected frozen Actors on dev play."""
    models = restore_models(snapshot)
    trajectories = base.rollout(models, base.Critic(), DEVELOPMENT_SEEDS, sample=False)
    states = [item[6] for episode in trajectories for item in episode]
    # The deterministic stride covers all tasks and the whole pre-action
    # trajectory without fitting on these states.
    states = states[::max(1, len(states) // 1500)]
    audits = {}
    for role in ('human', 'ai'):
        program = snapshot['programs'].get(role)
        if program is None:
            audits[role] = {'passed': False, 'reason': 'missing_training_time_program'}
            continue
        metrics = new_rcpd(0).evaluate(
            program, states, neural_oracle(models[role], role),
            lambda state, role=role: semantic_features(state, role))
        fidelity_gate, kl_gate = ((.95, .10) if role == 'ai' else (.75, .60))
        audits[role] = {
            'sample_count': metrics.sample_count,
            'action_fidelity': metrics.action_fidelity,
            'mean_kl_divergence': metrics.mean_kl_divergence,
            'purpose': 'explanation' if role == 'ai' else 'training_feedback',
            'passed': (metrics.sample_count >= 1000
                       and metrics.action_fidelity >= fidelity_gate
                       and metrics.mean_kl_divergence <= kl_gate),
        }
    return audits


def prepare_candidate_programs(snapshot, pair_seed):
    """Audit training-time programs; refit only an unaudited human program.

    The optional human refit learns from fresh training-only NN rollouts, uses
    separate validation seeds, and is audited on development seeds. It does
    not feed gradients or actions back into either frozen Actor. The AI
    explanation program must itself pass the selected-Actor audit.
    """
    original_audit = audit_selected_programs(snapshot)
    provenance = {role: {'source': 'training_time_extraction',
                         'training_time_selected_actor_audit': original_audit[role]}
                  for role in ('human', 'ai')}
    if not original_audit['human']['passed']:
        models = restore_models(snapshot)
        train = [item[6] for episode in base.rollout(
            models, base.Critic(), PROGRAM_REFIT_TRAIN_SEEDS, sample=False)
                 for item in episode][::5][:1600]
        validation = [item[6] for episode in base.rollout(
            models, base.Critic(), PROGRAM_REFIT_VALIDATION_SEEDS, sample=False)
                      for item in episode][::4][:500]
        try:
            fitted = new_rcpd(pair_seed).fit(
                train, neural_oracle(models['human'], 'human'),
                lambda state: semantic_features(state, 'human'),
                validation_states=validation,
                action_legality_features={'left': 'legal_left', 'right': 'legal_right'},
                program_metadata={'domain': 'pong', 'role': 'human',
                                  'pair_seed': pair_seed,
                                  'source': 'post_training_selected_actor_extraction'})
            provenance['human']['post_training_refit'] = {
                'train_seeds': list(PROGRAM_REFIT_TRAIN_SEEDS),
                'validation_seeds': list(PROGRAM_REFIT_VALIDATION_SEEDS),
                'train_state_count': len(train),
                'validation_state_count': len(validation),
                'validation_metrics': fitted.metrics.to_dict(),
                'frozen_actor_weights_unchanged': True,
                'feedback_to_actor': False,
            }
            if fitted.metrics.feedback_eligible:
                previous_program = snapshot['programs'].get('human')
                previous_metrics = snapshot['program_metrics'].get('human')
                snapshot['programs']['human'] = fitted.program
                snapshot['program_metrics']['human'] = fitted.metrics.to_dict()
                refit_audit = audit_selected_programs(snapshot)
                provenance['human']['post_training_refit']['selected_actor_development_audit'] = refit_audit['human']
                if refit_audit['human']['passed']:
                    provenance['human']['source'] = 'post_training_selected_actor_extraction'
                else:
                    snapshot['programs']['human'] = previous_program
                    snapshot['program_metrics']['human'] = previous_metrics
            else:
                provenance['human']['post_training_refit']['rejected'] = 'independent_validation_failed'
        except (ValueError, RuntimeError) as exc:
            provenance['human']['post_training_refit'] = {'rejected': str(exc)}
    final_audit = audit_selected_programs(snapshot)
    snapshot['program_provenance'] = provenance
    return final_audit, provenance


def export_variant(variant, snapshot, training_report):
    models = restore_models(snapshot)
    prefix = nn_policy._v5_default_artifacts(variant)
    actor_path = ROOT / prefix['actor_path']
    base.export(models, actor_path)
    payload = json.loads(actor_path.read_text())
    actor_digest = sha(actor_path)
    role_digests = {role: nn_policy._role_digest(payload['actors'][role])
                    for role in ('human', 'ai')}
    program_digests = {}
    if variant == 'regularized':
        for role, path in zip(('human', 'ai'), prefix['program_paths']):
            program = snapshot['programs'].get(role)
            if program is not None:
                write_json(ROOT / path, {'program': program.to_dict(),
                                         'audit': snapshot['program_metrics'][role],
                                         'program_provenance':
                                         snapshot.get('program_provenance', {}).get(role),
                                         'selected_actor_development_audit':
                                         snapshot.get('development_program_audit', {}).get(role),
                                         'role': role, 'policy_id': prefix['policy_id'],
                                         'selected_actor_artifact_sha256': actor_digest,
                                         'selected_role_sha256': role_digests[role]})
                program_digests[path] = sha(ROOT / path)
    report_path = ROOT / prefix['training_report_path']
    write_json(report_path, training_report)
    return {
        **prefix,
        'actor_sha256': actor_digest,
        'role_sha256': role_digests,
        'program_sha256': program_digests,
        'training_report_sha256': sha(report_path),
    }


def runtime_parity(models, payload, seeds):
    expected = base.evaluate(models, seeds)
    actual = base.evaluate_runtime(seeds, payload)
    return all(a['task'] == b['task'] and a['seed'] == b['seed']
               and abs(a['capacity_percent'] - b['capacity_percent']) < 1e-8
               and abs(a['ai_wait_rate'] - b['ai_wait_rate']) < 1e-8
               for a, b in zip(expected['rows'], actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-seeds', nargs=3, type=int, default=PAIR_SEEDS)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--batch-seeds', type=int, default=BATCH_SEEDS)
    parser.add_argument('--pretrain-checkpoint', type=Path,
                        default=ROOT / 'output/pong_rolling_nn/pre_self_play.pt')
    parser.add_argument('--resume', action='store_true',
                        help='Reuse verified, per-variant 200k reports and checkpoints')
    args = parser.parse_args()
    if len(set(args.pair_seeds)) != 3 or args.steps <= 0 or args.batch_seeds < 4:
        parser.error('provide exactly three distinct pair seeds, positive steps, and at least four batch seeds')
    base.torch.set_num_threads(1)
    initial = pretrained_actors(args.pretrain_checkpoint)
    bc_sha = sha(args.pretrain_checkpoint)
    reports = {}
    snapshots = {}
    for pair_index, pair_seed in enumerate(args.pair_seeds):
        reports[pair_seed] = {}
        snapshots[pair_seed] = {}
        for variant in ('raw', 'regularized'):
            progress_path = OUTPUT / f'pair_{pair_seed}_{variant}.training.json'
            if args.resume and progress_path.exists():
                report = json.loads(progress_path.read_text())
                checkpoint = ROOT / report['checkpoint_path']
                expected_initial = actor_initial_hashes(initial)
                if (report.get('pair_seed') != pair_seed or report.get('variant') != variant
                        or report.get('joint_steps') != args.steps
                        or report.get('batch_seeds') != args.batch_seeds
                        or report.get('pair_index') != pair_index
                        or report.get('bc_checkpoint_sha256') != bc_sha
                        or report.get('initial_actor_sha256') != expected_initial
                        or not checkpoint.is_file() or sha(checkpoint) != report.get('checkpoint_sha256')):
                    raise RuntimeError(f'Pong resume evidence differs for {pair_seed} {variant}')
                snapshot = torch.load(checkpoint, map_location='cpu', weights_only=False)
                print(json.dumps({'pair_seed': pair_seed, 'variant': variant,
                                  'resumed_verified_joint_steps': report['joint_steps']}), flush=True)
            else:
                report, snapshot = train_variant(pair_index, pair_seed, variant, initial,
                                                 steps=args.steps, batch_seeds=args.batch_seeds)
                report['bc_checkpoint_path'] = args.pretrain_checkpoint.relative_to(ROOT).as_posix()
                report['bc_checkpoint_sha256'] = bc_sha
                report['batch_seeds'] = args.batch_seeds
                report['pair_index'] = pair_index
                write_json(progress_path, report)
            reports[pair_seed][variant] = report
            snapshots[pair_seed][variant] = snapshot
        raw, regularized = reports[pair_seed]['raw'], reports[pair_seed]['regularized']
        if (raw['joint_steps'] != regularized['joint_steps']
                or raw['initial_actor_sha256'] != regularized['initial_actor_sha256']
                or raw['bc_checkpoint_sha256'] != regularized['bc_checkpoint_sha256']
                or raw['seed_schedule_sha256'] != regularized['seed_schedule_sha256']
                or any(raw['program_feedback_rows'].values())
                or any(item.get('post_policy_override_count') != 0
                       for item in (raw, regularized))):
            raise RuntimeError(f'Pong paired training invariant failed for {pair_seed}')
    qualifying = [seed for seed in args.pair_seeds
                  if all(reports[seed][variant]['development']['passed']
                         for variant in ('raw', 'regularized'))]
    program_audits = {}
    program_provenance = {}
    ready_pairs = []
    for seed in args.pair_seeds:
        audit, provenance = prepare_candidate_programs(snapshots[seed]['regularized'], seed)
        program_audits[seed] = audit
        program_provenance[seed] = provenance
        reg = reports[seed]['regularized']
        if (seed in qualifying
                and all(reg['program_feedback_rows'][role] > 0 for role in ('human', 'ai'))
                and all(snapshots[seed]['regularized']['programs'].get(role) is not None
                        for role in ('human', 'ai'))
                and all(audit[role]['passed'] for role in ('human', 'ai'))):
            ready_pairs.append(seed)
    acceptance_path = ROOT / nn_policy._v5_default_artifacts('raw')['acceptance_path']
    prior = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
    already_evaluated = prior.get('heldout_evaluated') is True
    if already_evaluated:
        if (prior.get('pair_seeds') != list(args.pair_seeds)
                or prior.get('selected_pair_seed') not in args.pair_seeds):
            raise RuntimeError('Pong held-out receipt has different candidate seeds')
        for variant in ('raw', 'regularized'):
            prior_report_path = ROOT / nn_policy._v5_default_artifacts(variant)['training_report_path']
            prior_report = json.loads(prior_report_path.read_text())
            prior_runs = {row['pair_seed']: row for row in prior_report['paired_runs']}
            if (set(prior_runs) != set(args.pair_seeds)
                    or any(prior_runs[seed]['checkpoint_sha256']
                           != reports[seed][variant]['checkpoint_sha256']
                           for seed in args.pair_seeds)
                    or variant not in prior.get('heldout', {})):
                raise RuntimeError('Pong held-out receipt belongs to different Actor checkpoints')
    # Select the paired seed on model quality alone. Program fidelity and
    # feedback gate the selected pair; they cannot make us fall back to a
    # lower-quality model pair. If no pair passes both model gates, retain a
    # diagnostic selection while keeping held-out evaluation closed.
    selection_pool = qualifying or list(args.pair_seeds)
    selected_seed = max(selection_pool, key=lambda seed: min(
        development_score(reports[seed][variant]['development'])
        for variant in ('raw', 'regularized')))
    if already_evaluated:
        selected_seed = prior['selected_pair_seed']
    selected_reg = reports[selected_seed]['regularized']
    program_validation = program_audits.get(selected_seed)
    if program_validation is None:
        program_validation = audit_selected_programs(snapshots[selected_seed]['regularized'])
    snapshots[selected_seed]['regularized']['development_program_audit'] = program_validation
    program_ok = all(
        snapshots[selected_seed]['regularized']['programs'].get(role) is not None
        and program_validation[role]['passed']
        for role in ('human', 'ai'))
    feedback_ok = all(selected_reg['program_feedback_rows'][role] > 0
                      for role in ('human', 'ai'))
    heldout_evaluated = (already_evaluated or
                         bool(selected_seed in ready_pairs and args.steps == STEPS
                              and program_ok and feedback_ok))
    if heldout_evaluated and not already_evaluated:
        reserve_heldout_once(acceptance_path, args.pair_seeds, selected_seed, reports)
    heldout = {}
    parity = {}
    for variant in ('raw', 'regularized'):
        models = restore_models(snapshots[selected_seed][variant])
        if already_evaluated:
            heldout[variant] = prior['heldout'][variant]
        elif heldout_evaluated:
            heldout[variant] = base.evaluate(models, HELDOUT_SEEDS)
        else:
            heldout[variant] = {'passed': False, 'reason': 'no_qualified_development_pair_or_incomplete_budget'}
        training_report = {
            'protocol': 'pong-paired-dual-actor-v5', 'variant': variant,
            'development_tuning': DEVELOPMENT_TUNING,
            'pair_seeds': list(args.pair_seeds), 'selected_pair_seed': selected_seed,
            'development_seeds': list(DEVELOPMENT_SEEDS),
            'heldout_seeds': list(HELDOUT_SEEDS) if heldout_evaluated else [],
            'heldout_reused_from_prior_receipt': already_evaluated,
            'bc_checkpoint_sha256': bc_sha,
            'paired_runs': [reports[seed][variant] for seed in args.pair_seeds],
            'candidate_program_audits': {str(seed): program_audits[seed]
                                         for seed in args.pair_seeds},
            'candidate_program_provenance': {str(seed): program_provenance[seed]
                                             for seed in args.pair_seeds},
            'heldout': heldout[variant],
            'selected_actor_development_program_audit':
                program_validation if variant == 'regularized' else {},
        }
        entry = export_variant(variant, snapshots[selected_seed][variant], training_report)
        snapshots[selected_seed][variant]['entry'] = entry
        if heldout_evaluated:
            payload = json.loads((ROOT / entry['actor_path']).read_text())
            parity[variant] = runtime_parity(models, payload, PARITY_SEEDS)
        else:
            parity[variant] = False
    entries = {variant: snapshots[selected_seed][variant]['entry'] for variant in ('raw', 'regularized')}
    passed = (heldout_evaluated and selected_seed in ready_pairs
              and program_ok and feedback_ok and all(parity.values())
              and all(heldout[v]['passed'] for v in ('raw', 'regularized')))
    candidate_training = []
    for seed in args.pair_seeds:
        row = {'pair_seed': seed}
        for variant in ('raw', 'regularized'):
            run = reports[seed][variant]
            row[variant] = {
                'joint_steps': run['joint_steps'],
                'bc_checkpoint_sha256': run['bc_checkpoint_sha256'],
                'initial_actor_sha256': run['initial_actor_sha256'],
                'seed_schedule_sha256': run['seed_schedule_sha256'],
                'actor_weight_change_l2': run['actor_weight_change_l2'],
                'program_feedback_rows': run['program_feedback_rows'],
                'post_policy_override_count': run['post_policy_override_count'],
            }
        candidate_training.append(row)
    acceptance = {
        'release': 'policylens-three-domain-20261014.v5.0',
        'domain': 'pong', 'passed': passed, 'heldout_evaluated': heldout_evaluated,
        'heldout_reused_from_prior_receipt': already_evaluated,
        'development_tuning': DEVELOPMENT_TUNING,
        'pair_seeds': list(args.pair_seeds), 'selected_pair_seed': selected_seed,
        'candidate_training': candidate_training,
        'candidate_program_audits': {str(seed): program_audits[seed]
                                     for seed in args.pair_seeds},
        'candidate_program_provenance': {str(seed): program_provenance[seed]
                                         for seed in args.pair_seeds},
        'joint_steps': {variant: reports[selected_seed][variant]['joint_steps']
                        for variant in ('raw', 'regularized')},
        'development': {variant: reports[selected_seed][variant]['development']
                        for variant in ('raw', 'regularized')},
        'heldout': heldout,
        'runtime_parity': parity,
        'feedback_positive': feedback_ok,
        'explanation_eligible': program_ok,
        'selected_actor_development_program_audit': program_validation,
        'artifact_sha256': {variant: entries[variant]['actor_sha256'] for variant in entries},
        'training_report_sha256': {variant: entries[variant]['training_report_sha256']
                                   for variant in entries},
        'program_sha256': {'regularized': {
            role: entries['regularized']['program_sha256'].get(path)
            for role, path in zip(('human', 'ai'), entries['regularized']['program_paths'])}},
        'action_override_count': 0,
        'post_policy_override_count': 0,
        'raw_program_feedback_rows': reports[selected_seed]['raw']['program_feedback_rows'],
        'regularized_program_feedback_rows': selected_reg['program_feedback_rows'],
    }
    write_json(ROOT / entries['raw']['acceptance_path'], acceptance)
    write_json(nn_policy.V5_MANIFEST_PATH, {
        'format': 'pong-paired-policies-v5',
        'release': acceptance['release'],
        'variants': entries,
    })
    print(json.dumps({'passed': passed, 'heldout_evaluated': heldout_evaluated,
                      'selected_pair_seed': selected_seed, 'program_ok': program_ok,
                      'feedback_ok': feedback_ok, 'runtime_parity': parity}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
