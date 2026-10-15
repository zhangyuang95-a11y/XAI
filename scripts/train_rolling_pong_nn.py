#!/usr/bin/env python3
"""Train two independent rolling-Pong actors, then export pure-Python inference.

The old visible-state planner supplies *labels only* during pretraining. Every
reinforcement-learning transition is controlled by the two sampled neural
actors and settled by the unchanged physical Pong engine.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from torch.distributions import Categorical

from domains.pong import nn_policy, turnbased as pong

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/pong_rolling_nn'


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nn_policy.FEATURE_SIZE, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 3))

    def forward(self, observation):
        return self.net(observation)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nn_policy.FEATURE_SIZE * 2, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, observation):
        return self.net(observation).squeeze(-1)


def observation(state, actor):
    return torch.tensor(nn_policy.features(state, actor), dtype=torch.float32)


def masked_logits(model, state, actor):
    scores = model(observation(state, actor))
    return scores.masked_fill(torch.tensor(nn_policy.legal_mask(state, actor)) == 0, -1e9)


def choose(model, state, actor, sample=False):
    with torch.no_grad():
        distribution = Categorical(logits=masked_logits(model, state, actor))
        index = int(distribution.sample()) if sample else int(distribution.logits.argmax())
    return nn_policy.ACTIONS[index]


def teacher_decision(state):
    return pong._decision_from_observation(pong._observation(state))


def collect_teacher(seeds, models=None, *, sample=False):
    examples = {'human': [], 'ai': []}
    for task in (1, 2, 3):
        for seed in seeds:
            state = pong.initial_state(seed, task)
            while not state['terminal']:
                teacher = teacher_decision(state)
                for actor in ('human', 'ai'):
                    action = teacher['human_action'] if actor == 'human' else teacher['action']
                    examples[actor].append((nn_policy.features(state, actor), nn_policy.ACTIONS.index(action),
                                            nn_policy.legal_mask(state, actor)))
                if models is None:
                    state = pong.step_pair(state, teacher['human_action'], teacher['action'])
                else:
                    actions = {a: choose(models[a], state, a, sample=sample) for a in ('human', 'ai')}
                    state = pong.step_pair(state, actions['human'], actions['ai'])
    return examples


def clone(models, examples, epochs, learning_rate=0.001):
    for actor in ('human', 'ai'):
        model = models[actor]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        rows = examples[actor]
        x = torch.tensor([row[0] for row in rows], dtype=torch.float32)
        y = torch.tensor([row[1] for row in rows], dtype=torch.long)
        mask = torch.tensor([row[2] for row in rows], dtype=torch.bool)
        for epoch in range(epochs):
            order = torch.randperm(len(rows))
            for batch in order.split(512):
                logits = model(x[batch]).masked_fill(~mask[batch], -1e9)
                loss = nn.functional.cross_entropy(logits, y[batch])
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            accuracy = (model(x).masked_fill(~mask, -1e9).argmax(-1) == y).float().mean().item()
        print(f'{actor} cloning examples={len(rows)} accuracy={accuracy:.4f}', flush=True)


def rollout(models, critic, seeds, *, sample=True, max_steps=None):
    trajectories = []
    total_steps = 0
    for task in (1, 2, 3):
        for seed in seeds:
            if max_steps is not None and total_steps >= max_steps:
                return trajectories
            state = pong.initial_state(seed, task)
            episode = []
            while not state['terminal']:
                if max_steps is not None and total_steps >= max_steps:
                    break
                visible_state = pong._observation(state)
                visible_state['_training_episode_id'] = f'{task}:{seed}'
                observations = {a: observation(state, a) for a in ('human', 'ai')}
                legal = {a: torch.tensor(nn_policy.legal_mask(state, a), dtype=torch.bool) for a in ('human', 'ai')}
                with torch.no_grad():
                    distributions = {a: Categorical(logits=models[a](observations[a]).masked_fill(~legal[a], -1e9))
                                     for a in ('human', 'ai')}
                    indices = {a: (distributions[a].sample() if sample else distributions[a].logits.argmax())
                               for a in ('human', 'ai')}
                    logprobs = {a: distributions[a].log_prob(indices[a]) for a in ('human', 'ai')}
                    value = critic(torch.cat((observations['human'], observations['ai'])))
                actions = {a: nn_policy.ACTIONS[int(indices[a])] for a in ('human', 'ai')}
                before = state['raw_score']
                state = pong.step_pair(state, actions['human'], actions['ai'])
                episode.append((observations, legal, indices, logprobs, value,
                                state['raw_score'] - before, visible_state))
                total_steps += 1
            trajectories.append(episode)
    return trajectories


def self_play_update(models, critic, optimizers, trajectories, *, regularizers=None,
                     training_fraction=0.0):
    regularizers = regularizers or {}
    feedback = {'human': {'weighted_rows': 0.0, 'rows': 0, 'batches': 0},
                'ai': {'weighted_rows': 0.0, 'rows': 0, 'batches': 0}}
    flat, advantages, returns = [], [], []
    for episode in trajectories:
        gae = 0.0
        next_value = 0.0
        episode_advantages = []
        for item in reversed(episode):
            delta = item[5] + 0.99 * next_value - float(item[4])
            gae = delta + 0.99 * 0.95 * gae
            episode_advantages.append(gae)
            next_value = float(item[4])
        flat.extend(episode)
        episode_advantages.reverse()
        advantages.extend(episode_advantages)
        returns.extend(a + float(item[4]) for a, item in zip(episode_advantages, episode))
    advantages = torch.tensor(advantages, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    returns = torch.tensor(returns, dtype=torch.float32)
    indices = torch.randperm(len(flat))
    for _ in range(3):
        for batch in indices.split(512):
            items = [flat[int(i)] for i in batch]
            joint = torch.stack([torch.cat((item[0]['human'], item[0]['ai'])) for item in items])
            value_loss = nn.functional.mse_loss(critic(joint), returns[batch])
            optimizers['critic'].zero_grad(); value_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); optimizers['critic'].step()
            for actor in ('human', 'ai'):
                x = torch.stack([item[0][actor] for item in items])
                mask = torch.stack([item[1][actor] for item in items])
                actions = torch.stack([item[2][actor] for item in items])
                old = torch.stack([item[3][actor] for item in items])
                logits = models[actor](x).masked_fill(~mask, -1e9)
                distribution = Categorical(logits=logits)
                ratio = torch.exp(distribution.log_prob(actions) - old)
                gain = torch.minimum(ratio * advantages[batch], torch.clamp(ratio, .8, 1.2) * advantages[batch])
                loss = -gain.mean() - 0.005 * distribution.entropy().mean()
                if actor in regularizers:
                    rcpd, encoder = regularizers[actor]
                    weight = rcpd.scheduled_regularization_weight(training_fraction)
                    if weight > 0.0:
                        semantic = [encoder(item[6]) for item in items]
                        probabilities = distribution.probs.detach().cpu().numpy()
                        targets = rcpd.program_feedback_targets(semantic,
                                                                actor_probabilities=probabilities)
                        target_weights = rcpd.program_target_weights(
                            semantic, actor_probabilities=probabilities)
                        if targets is not None and target_weights is not None:
                            loss = loss + weight * rcpd.regularization_loss(
                                logits, targets, target_weights)
                            feedback[actor]['weighted_rows'] += float(target_weights.sum())
                            feedback[actor]['rows'] += int((target_weights > 0).sum())
                            feedback[actor]['batches'] += 1
                optimizers[actor].zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(models[actor].parameters(), 1.0); optimizers[actor].step()
    return feedback


def _layers(model):
    return [{'weight': layer.weight.detach().cpu().tolist(), 'bias': layer.bias.detach().cpu().tolist()}
            for layer in model.net if isinstance(layer, nn.Linear)]


def export(models, path):
    actors = {actor: _layers(model) for actor, model in models.items()}
    digest = hashlib.sha256(json.dumps(actors, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    payload = {'version': nn_policy.MODEL_VERSION, 'feature_size': nn_policy.FEATURE_SIZE,
               'actions': list(nn_policy.ACTIONS), 'actors': actors, 'actors_sha256': digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(',', ':')) + '\n')


def evaluate(models, seeds):
    rows = []
    for task in (1, 2, 3):
        for seed in seeds:
            state = pong.initial_state(seed, task)
            initial = deepcopy(state)
            while not state['terminal']:
                actions = {a: choose(models[a], state, a) for a in ('human', 'ai')}
                state = pong.step_pair(state, actions['human'], actions['ai'])
            capacity = contact_capacity(initial)
            rows.append({'task': task, 'seed': seed, 'score': pong.score(state)['task_score'],
                         'capacity_percent': 100 * state['raw_score'] / capacity,
                         'ai_wait_rate': state['metrics']['waiting_turns'] / state['turn']})
    summaries = []
    for task in (1, 2, 3):
        subset = [row for row in rows if row['task'] == task]
        summaries.append({'task': task, 'mean_capacity_percent': sum(r['capacity_percent'] for r in subset) / len(subset),
                          'min_capacity_percent': min(r['capacity_percent'] for r in subset),
                          'mean_ai_wait_rate': sum(r['ai_wait_rate'] for r in subset) / len(subset),
                          'max_ai_wait_rate': max(r['ai_wait_rate'] for r in subset)})
    passed = all(s['mean_capacity_percent'] >= 90 and s['min_capacity_percent'] >= 80
                 and s['mean_ai_wait_rate'] <= .10 and s['max_ai_wait_rate'] <= .20 for s in summaries)
    return {'passed': passed, 'summaries': summaries, 'rows': rows}


def evaluate_human_replacement(models, seeds):
    """Stress the frozen AI with non-neural human proxies; never a human effect."""
    rows = []
    for policy in ('wait', 'greedy', 'history', 'random'):
        for task in (1, 2, 3):
            for seed in seeds:
                state = pong.initial_state(seed, task)
                rng = random.Random(seed * 17 + task)
                history = []
                while not state['terminal']:
                    human = ('wait' if policy == 'wait' else
                             pong.greedy_human(state) if policy == 'greedy' else
                             pong.history_learning_human(state, history) if policy == 'history' else
                             rng.choice(pong.legal_actions(state)))
                    ai = choose(models['ai'], state, 'ai')
                    state = pong.step_pair(state, human, ai)
                    history.append({'events': state['events']})
                rows.append({'policy': policy, 'task': task, 'seed': seed,
                             'task_score': pong.score(state)['task_score'],
                             'ai_wait_rate': state['metrics']['waiting_turns'] / state['turn']})
    summaries = []
    for policy in ('wait', 'greedy', 'history', 'random'):
        for task in (1, 2, 3):
            subset = [row for row in rows if row['policy'] == policy and row['task'] == task]
            summaries.append({'policy': policy, 'task': task,
                              'mean_task_score': sum(r['task_score'] for r in subset) / len(subset),
                              'min_task_score': min(r['task_score'] for r in subset)})
    return {'evidence_type': 'fixed_ai_with_simulated_human_proxy', 'summaries': summaries, 'rows': rows}


def evaluate_runtime(seeds, payload):
    rows = []
    for task in (1, 2, 3):
        for seed in seeds:
            state = pong.initial_state(seed, task)
            capacity = contact_capacity(state)
            while not state['terminal']:
                human = nn_policy.choose_from_payload(state, 'human', payload)
                ai = nn_policy.choose_from_payload(state, 'ai', payload)
                state = pong.step_pair(state, human, ai)
            rows.append({'task': task, 'seed': seed,
                         'capacity_percent': 100 * state['raw_score'] / capacity,
                         'ai_wait_rate': state['metrics']['waiting_turns'] / state['turn']})
    return rows


def contact_capacity(state):
    arrivals = {}
    for entry in state['_schedule']:
        arrivals.setdefault(entry['arrival_turn'], []).append(entry['ball'])
    return sum(max(sum(pong.WEIGHTS[b['kind']] for b in balls
                       if ({human, ai} == set(b['contacts']) if b['kind'] == 'cooperative'
                           else b['contacts'][0] in (human, ai)))
                   for human in range(9) for ai in range(9)) for balls in arrivals.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clone-epochs', type=int, default=30)
    parser.add_argument('--dagger-rounds', type=int, default=2)
    parser.add_argument('--self-play-rounds', type=int, default=12)
    parser.add_argument('--train-seeds', type=int, default=160,
                        help='Additional disjoint training seeds, starting at 740000')
    parser.add_argument('--resume-from', type=Path,
                        help='Resume self-play from a previously cloned two-Actor checkpoint')
    parser.add_argument('--parent-report', type=Path,
                        help='Original cloning report to record when resuming a checkpoint')
    parser.add_argument('--seed', type=int, default=260927)
    args = parser.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    models = {a: Actor() for a in ('human', 'ai')}
    critic = Critic()
    development = pong.CONFIG['development_seeds']
    training_seeds = [740000 + i for i in range(args.train_seeds)]
    examples = {'human': [], 'ai': []}
    parent_sha = None
    parent_training = None
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        for actor in models:
            models[actor].load_state_dict(checkpoint['actors'][actor])
        parent_sha = hashlib.sha256(args.resume_from.read_bytes()).hexdigest()
        if args.parent_report:
            parent_training = json.loads(args.parent_report.read_text()).get('training')
    else:
        examples = collect_teacher(training_seeds)
        clone(models, examples, args.clone_epochs)
        for round_index in range(args.dagger_rounds):
            additional = collect_teacher(training_seeds, models, sample=bool(round_index % 2))
            for actor in examples:
                examples[actor].extend(additional[actor])
            clone(models, examples, max(8, args.clone_epochs // 2), learning_rate=.0005)
        torch.save({'actors': {a: deepcopy(model.state_dict()) for a, model in models.items()},
                    'seed': args.seed, 'training_seed_count': len(training_seeds)},
                   OUTPUT / 'pre_self_play.pt')
    before_self_play = evaluate(models, development)
    print('pre-self-play', json.dumps(before_self_play['summaries']), flush=True)
    best = None
    best_models = None
    best_round = None
    optimizers = {a: torch.optim.Adam(models[a].parameters(), lr=0.0001) for a in models}
    optimizers['critic'] = torch.optim.Adam(critic.parameters(), lr=0.0003)
    self_play_seed_schedule = []
    for round_index in range(args.self_play_rounds):
        # Development seeds select checkpoints; gradients use disjoint training seeds.
        round_seeds = [training_seeds[(round_index * len(development) + offset) % len(training_seeds)]
                       for offset in range(len(development))]
        self_play_seed_schedule.append(round_seeds)
        trajectories = rollout(models, critic, round_seeds)
        self_play_update(models, critic, optimizers, trajectories)
        result = evaluate(models, development)
        print('self-play', round_index + 1, json.dumps(result['summaries']), flush=True)
        if result['passed'] and (best is None or sum(row['capacity_percent'] for row in result['rows']) >
                                 sum(row['capacity_percent'] for row in best['rows'])):
            best = result
            best_models = {a: deepcopy(model.state_dict()) for a, model in models.items()}
            best_round = round_index + 1
    if best is None:
        (OUTPUT / 'self_play_failure.json').write_text(json.dumps({
            'reason': 'no_self_play_checkpoint_passed_development_gates',
            'pre_self_play': before_self_play,
            'rounds': args.self_play_rounds}, indent=2) + '\n')
        print('No self-play-updated checkpoint passed development gates.', flush=True)
        return 1
    for actor in models:
        models[actor].load_state_dict(best_models[actor])
    torch.save({'actors': best_models, 'seed': args.seed,
                'self_play_round': best_round, 'parent_checkpoint_sha256': parent_sha}, OUTPUT / 'best.pt')
    candidate_path = OUTPUT / 'candidate_actor.json'
    export(models, candidate_path)
    heldout = evaluate(models, pong.CONFIG['heldout_seeds'])
    model_sha = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    replacement = evaluate_human_replacement(models, pong.CONFIG['heldout_seeds'])
    report = {'model_sha256': model_sha, 'development': best, 'heldout': heldout,
              'human_replacement_proxies': replacement,
              'training': {'clone_examples_per_role': len(examples['ai']),
                           'clone_epochs': 0 if args.resume_from else args.clone_epochs,
                           'dagger_rounds': 0 if args.resume_from else args.dagger_rounds,
                           'self_play_rounds': args.self_play_rounds,
                           'selected_self_play_round': best_round,
                           'parent_checkpoint_sha256': parent_sha,
                           'parent_training': parent_training,
                           'training_seed_count': len(training_seeds),
                           'self_play_seed_schedule': self_play_seed_schedule,
                           'neural_joint_steps': args.self_play_rounds * len(development) * sum(pong.CONFIG['task_turns'].values())}}
    if best['passed'] and heldout['passed']:
        candidate_payload = json.loads(candidate_path.read_text())
        runtime_dev = evaluate_runtime(development, candidate_payload)
        runtime_hold = evaluate_runtime(pong.CONFIG['heldout_seeds'], candidate_payload)
        reference = [{k: r[k] for k in ('task', 'seed', 'capacity_percent', 'ai_wait_rate')}
                     for r in best['rows'] + heldout['rows']]
        parity = all(a['task'] == b['task'] and a['seed'] == b['seed']
                     and abs(a['capacity_percent'] - b['capacity_percent']) < 1e-9
                     and abs(a['ai_wait_rate'] - b['ai_wait_rate']) < 1e-9
                     for a, b in zip(runtime_dev + runtime_hold, reference))
        report['runtime_parity'] = parity
        (OUTPUT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
        if not parity:
            print('exported runtime actor diverged from evaluated checkpoint', flush=True)
            return 1
        release_paths = (nn_policy.MODEL_PATH, nn_policy.EVALUATION_PATH, nn_policy.ACCEPTANCE_PATH)
        previous = {path: path.read_bytes() if path.exists() else None for path in release_paths}
        try:
            nn_policy.MODEL_PATH.write_bytes(candidate_path.read_bytes())
            nn_policy.EVALUATION_PATH.write_bytes((OUTPUT / 'evaluation.json').read_bytes())
            evaluation_sha = hashlib.sha256(nn_policy.EVALUATION_PATH.read_bytes()).hexdigest()
            nn_policy.ACCEPTANCE_PATH.write_text(json.dumps({
                'model_sha256': model_sha, 'development_passed': True,
                'heldout_passed': True, 'model_version': nn_policy.MODEL_VERSION,
                'evaluation_sha256': evaluation_sha}, indent=2) + '\n')
            nn_policy.load_model.cache_clear()
            assert nn_policy.model_sha256() == model_sha
        except BaseException:
            for path, data in previous.items():
                if data is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(data)
            nn_policy.load_model.cache_clear()
            raise
    else:
        report['runtime_parity'] = False
        (OUTPUT / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'development_passed': best['passed'], 'heldout_passed': heldout['passed'],
                      'model_sha256': report['model_sha256']}), flush=True)
    return 0 if best['passed'] and heldout['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
