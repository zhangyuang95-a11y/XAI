#!/usr/bin/env python3
"""Reproducible simulation-only rolling Pong validation; never human results."""
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


def run(include_baselines=False):
    rows = []
    for split in ('development_seeds', 'heldout_seeds'):
        for task in (1, 2, 3):
            for seed in pong.CONFIG[split]:
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
                                 'score': pong.score(state)['task_score'], 'wait_rate': pong.score(state)['metrics']['waiting_rate'],
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
                                  'mean_wait_rate':statistics.mean(r['wait_rate'] for r in subset), 'maximum_wait_rate':max(r['wait_rate'] for r in subset),
                                  'maximum_step_ms':max(r['max_step_ms'] for r in subset)})
    cooperative = [r for r in summaries if r['policy'] == 'cooperative_visible']
    return {'engine_version':pong.VERSION, 'scenario_version':pong.CONFIG['scenario_version'],
            'evidence_type':'simulation_with_actual_fixed_ai_not_human_study', 'future_schedule_available_to_proxy':False,
            'thresholds':{'mean_score_min':90, 'each_score_min':80, 'mean_wait_rate_max':0.10, 'each_wait_rate_max':0.20},
            'gates_passed':all(r['mean_score'] >= 90 and r['minimum_score'] >= 80 and r['mean_wait_rate'] <= .10 and r['maximum_wait_rate'] <= .20 for r in cooperative),
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
    sys.exit(0 if result['gates_passed'] else 1)
