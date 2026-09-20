#!/usr/bin/env python3
"""Frozen-seed evaluation of the actual AI-led Pong controller.

These rollouts are simulation evidence, not A/B human observations. The helper
human follows the engine's visible-state advice. The unchanged schedule has
incompatible team-ball arrivals, so its contact-capacity upper bound is reported
separately from the displayed score; no perfect-score or treatment-effect claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.pong import turnbased as pong
from scripts.calibrate_rolling_pong import contact_capacity


def run():
    rows = []
    for split in ('development_seeds', 'heldout_seeds'):
        for task in (1, 2, 3):
            for seed in pong.CONFIG[split]:
                state = pong.initial_state(seed, task)
                capacity = contact_capacity(state)
                timings, overlaps, locked_transitions = [], 0, 0
                while not state['terminal']:
                    started = perf_counter()
                    before = state
                    decision = pong.decide(state)
                    state = pong.step(state, pong.human_advisor(state))
                    timings.append((perf_counter() - started) * 1000)
                    if before['policy_memory'].get('ai_route'):
                        assert state['ai']['x'] == before['policy_memory']['ai_route'][0]
                        locked_transitions += 1
                    caught = [event for event in state['events'] if event['type'] == 'caught']
                    if any(e['kind'] == 'cooperative' for e in caught) and any(
                            e['kind'] == 'ordinary' and state['ai']['x'] in e['contacts'] for e in caught):
                        overlaps += 1
                rows.append({'split': split, 'seed': seed, 'task': task,
                             'score': pong.score(state)['task_score'], 'raw_score': state['raw_score'],
                             'turns': state['turn'], 'cooperative_caught': state['metrics']['cooperative_caught'],
                             'ordinary_caught': state['metrics']['ordinary_caught'],
                             'ai_small_and_team_same_arrival': overlaps,
                             'checked_locked_transitions': locked_transitions,
                             'contact_capacity_upper_bound': capacity,
                             'percent_of_contact_capacity': 100 * state['raw_score'] / capacity,
                             'wait_rate': state['metrics']['waiting_turns'] / state['turn'],
                             'mean_step_ms': statistics.mean(timings), 'max_step_ms': max(timings)})
    summaries = []
    for split in ('development_seeds', 'heldout_seeds'):
        for task in (1, 2, 3):
            sample = [r for r in rows if r['split'] == split and r['task'] == task]
            summaries.append({'split': split, 'task': task, 'runs': len(sample),
                              'mean_score': statistics.mean(r['score'] for r in sample),
                              'min_score': min(r['score'] for r in sample),
                              'max_score': max(r['score'] for r in sample),
                              'mean_percent_of_contact_capacity': statistics.mean(r['percent_of_contact_capacity'] for r in sample),
                              'ai_simultaneous_small_team_catches': sum(r['ai_small_and_team_same_arrival'] for r in sample),
                              'team_catches': sum(r['cooperative_caught'] for r in sample),
                              'mean_wait_rate': statistics.mean(r['wait_rate'] for r in sample),
                              'max_wait_rate': max(r['wait_rate'] for r in sample),
                              'max_step_ms': max(r['max_step_ms'] for r in sample)})
    return {'engine_version': pong.VERSION, 'scenario_version': pong.CONFIG['scenario_version'],
            'evidence_type': 'simulation_actual_fixed_ai_with_visible_state_cooperative_proxy',
            'human_task2_improvement': 'not_measured', 'future_spawn_information_used': False,
            'score_optimality_claimed': False, 'seed_selection': 'all_24_development_and_24_heldout_seeds_all_3_tasks',
            'route_lock_assertions_passed': True,
            'checked_locked_transitions': sum(r['checked_locked_transitions'] for r in rows),
            'summaries': summaries, 'runs': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'runs'}, indent=2))
