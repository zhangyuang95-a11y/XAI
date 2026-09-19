"""Freeze and validate kitchen scenes using a human-only deterministic proxy.

This is a feasibility check, not an A/B experiment or human effect estimate.
Run: python -m domains.kitchen.calibrate_v4
Only development seeds can justify calibration; held-out seeds validate the
frozen artifact. The approved base budgets stay unchanged when feasible.
"""
from copy import deepcopy
import json
from pathlib import Path
from . import engine as e


def rollout(seed, task):
    state = e.initial_state(seed, task)
    event_types, used_pans = set(), set()
    while not state['terminal']:
        human_action = e.human_advisor(state)
        state = e.step(state, human_action)
        for event in state['events']:
            event_types.add(event['type'])
            if event['type'] == 'pot_loaded': used_pans.add(event['pot'])
    return {'seed': seed, 'task': task, 'completed': state['metrics']['completed_orders'], 'assigned': state['total_orders'],
            'score': e.score(state)['task_score'], 'turns': state['turn'], 'burnt': state['metrics']['burnt'],
            'parallel_recipe_turns': state['metrics']['parallel_recipe_turns'], 'parallel_cooking_turns': state['metrics']['parallel_cooking_turns'],
            'used_pans': sorted(used_pans), 'served_turns': {order['id']: order['served_turn'] for order in state['orders']},
            'event_types': sorted(event_types)}


def build():
    path = Path(__file__).resolve().parents[2] / 'configs' / 'study_v3_kitchen.json'
    configuration = json.loads(path.read_text())
    # Existing frozen scenarios are intentionally retained on rerun. A new
    # protocol version is required before changing an existing study snapshot.
    snapshots = {str(seed): {str(task): e._scenario(seed, task) for task in (1, 2, 3)}
                 for seed in configuration['development_seeds'] + configuration['held_out_seeds']}
    development = [rollout(seed, task) for seed in configuration['development_seeds'] for task in (1, 2, 3)]
    if not all(row['completed'] == row['assigned'] for row in development):
        raise AssertionError('Development feasibility failed. Recalibrate before freezing; never tune using held-out outcomes.')
    held_out = [rollout(seed, task) for seed in configuration['held_out_seeds'] for task in (1, 2, 3)]
    if not all(row['completed'] == row['assigned'] for row in held_out):
        raise AssertionError('Held-out feasibility failed; retain the failed result and revise the development protocol as a new version.')
    summary = {}
    for split, rows in [('development', development), ('held_out', held_out)]:
        summary[split] = {str(task): {'runs': len([r for r in rows if r['task'] == task]),
                                    'all_complete': all(r['completed'] == r['assigned'] for r in rows if r['task'] == task),
                                    'minimum_turns': min(r['turns'] for r in rows if r['task'] == task),
                                    'maximum_turns': max(r['turns'] for r in rows if r['task'] == task),
                                    'minimum_parallel_recipe_turns': min(r['parallel_recipe_turns'] for r in rows if r['task'] == task),
                                    'burnt_total': sum(r['burnt'] for r in rows if r['task'] == task)} for task in (1, 2, 3)}
    configuration.update(scenarios=snapshots,
        calibration_log=[{'version': e.VERSION, 'change': 'Replace soups with two compound recipes, four independent cupboards, facing-aware direct interactions and dedicated temporary plates.',
                          'budgets': {'1': 240, '2': 360, '3': 360}, 'budget_extension_applied': False,
                          'reason': 'All development trajectories complete within the approved base budgets. Deadlines are common to all groups: [130,160,225,240] for Task 1; [130,170,245,280,345,360] for Tasks 2 and 3. No score or controller accepts group identity.',
                          'parallelism': 'Both pans have overlapping active recipe stages. Simultaneously cooking pans are not claimed: moving between the opposite pans, facing, and loading needs six turns, equal to the longest cooking stage.',
                          'human_effect': 'not_measured'}],
        calibration_results={'evidence_type': 'simulated human-action proxy with the unchanged AI', 'summary': summary,
                             'development_runs': development, 'held_out_runs': held_out, 'human_relative_gain': None})
    path.write_text(json.dumps(configuration, ensure_ascii=False, indent=2) + '\n')
    e._configuration.cache_clear()
    return summary


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
