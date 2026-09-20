#!/usr/bin/env python3
"""V5 simulation-only calibration against unavoidable contact conflicts.

The contact-capacity bound ignores travel and is not an attainable-score claim.
All fixed seeds, including failed waiting targets, remain in the report. These
proxies do not stand for experimental groups or establish a human effect.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.pong import turnbased as pong


def contact_capacity(state):
    """Independent upper bound: instant repositioning before every arrival."""
    arrivals = {}
    for entry in state['_schedule']:
        arrivals.setdefault(entry['arrival_turn'], []).append(entry['ball'])
    upper_bound = 0
    for balls in arrivals.values():
        best = 0
        for human in range(state['lanes']):
            for ai in range(state['lanes']):
                points = 0
                for ball in balls:
                    caught = ({human, ai} == set(ball['contacts'])) if ball['kind'] == 'cooperative' else ball['contacts'][0] in (human, ai)
                    points += pong.WEIGHTS[ball['kind']] if caught else 0
                best = max(best, points)
        upper_bound += best
    return upper_bound


def run(include_baselines=False):
    rows = []
    for split in ('development_seeds', 'heldout_seeds'):
        for task in (1, 2, 3):
            for seed in pong.CONFIG[split]:
                initial = pong.initial_state(seed, task)
                capacity = contact_capacity(initial)
                policies = ('cooperative_visible', 'wait', 'greedy', 'random', 'history') if include_baselines else ('cooperative_visible',)
                for policy in policies:
                    state = pong.initial_state(seed, task)
                    history, timing = [], []
                    rng = random.Random(seed + task * 17)
                    while not state['terminal']:
                        before = perf_counter()
                        action = (pong.human_advisor(state) if policy == 'cooperative_visible' else 'wait' if policy == 'wait'
                                  else pong.greedy_human(state) if policy == 'greedy' else rng.choice(pong.legal_actions(state)) if policy == 'random'
                                  else pong.history_learning_human(state, history))
                        state = pong.step(state, action)
                        timing.append((perf_counter() - before) * 1000)
                        history.append({'events': state['events']})
                    rows.append({'split': split, 'task': task, 'seed': seed, 'policy': policy,
                                 'score': pong.score(state)['task_score'], 'raw_score': state['raw_score'],
                                 'total_possible_points': state['total_possible_points'],
                                 'contact_capacity_upper_bound': capacity,
                                 'contact_capacity_score_upper_bound': 100 * capacity / state['total_possible_points'],
                                 'percent_of_contact_capacity': 100 * state['raw_score'] / max(1, capacity),
                                 'wait_rate': pong.score(state)['metrics']['waiting_rate'],
                                 'caught_team': state['metrics']['cooperative_caught'], 'total_team': state['metrics']['cooperative_total'],
                                 'holding_turns': state['metrics']['holding_turns'], 'no_job_waiting_turns': state['metrics']['no_job_waiting_turns'],
                                 'max_step_ms': max(timing), 'mean_step_ms': statistics.mean(timing)})
    summaries = []
    for split in ('development_seeds', 'heldout_seeds'):
        for task in (1, 2, 3):
            for policy in sorted(set(r['policy'] for r in rows)):
                subset = [r for r in rows if r['split'] == split and r['task'] == task and r['policy'] == policy]
                summaries.append({'split':split, 'task':task, 'policy':policy, 'n':len(subset),
                                  'mean_score': statistics.mean(r['score'] for r in subset), 'minimum_score':min(r['score'] for r in subset),
                                  'mean_percent_of_contact_capacity':statistics.mean(r['percent_of_contact_capacity'] for r in subset),
                                  'minimum_percent_of_contact_capacity':min(r['percent_of_contact_capacity'] for r in subset),
                                  'mean_wait_rate':statistics.mean(r['wait_rate'] for r in subset), 'maximum_wait_rate':max(r['wait_rate'] for r in subset),
                                  'maximum_step_ms':max(r['max_step_ms'] for r in subset)})
    cooperative = [r for r in summaries if r['policy'] == 'cooperative_visible']
    score_gate = all(r['mean_percent_of_contact_capacity'] >= 90 and r['minimum_percent_of_contact_capacity'] >= 80 for r in cooperative)
    waiting_met = all(r['mean_wait_rate'] <= .10 and r['maximum_wait_rate'] <= .20 for r in cooperative)
    return {'engine_version':pong.VERSION, 'scenario_version':pong.CONFIG['scenario_version'],
            'evidence_type':'simulation_with_actual_fixed_ai_not_human_study', 'future_schedule_available_to_proxy':False,
            'normalization':'displayed scores divide by all scheduled raw points, including incompatible opportunities',
            'capacity_scope':'offline instantaneous-contact upper bound; ignores travel and does not claim attainability',
            'thresholds':{'mean_percent_of_contact_capacity_min':90, 'each_percent_of_contact_capacity_min':80,
                          'mean_wait_rate_target':0.10, 'each_wait_rate_target':0.20},
            'score_vs_capacity_gates_passed':score_gate, 'waiting_targets_met':waiting_met,
            'gates_passed':score_gate and waiting_met,
            'waiting_failures':[{'split':r['split'],'task':r['task'],'mean_wait_rate':r['mean_wait_rate'],'maximum_wait_rate':r['maximum_wait_rate']}
                                for r in cooperative if r['mean_wait_rate'] > .10 or r['maximum_wait_rate'] > .20],
            'summaries':summaries, 'runs':rows, 'human_task2_improvement':'not_measured'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run(args.baselines)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2))
    # Waiting target failures remain failures, not relabelled as a perfect pass.
    sys.exit(0 if result['gates_passed'] else 1)
