#!/usr/bin/env python3
"""Validate v6.2 Kitchen with the fixed AI and a deterministic proxy.

Seeds 4000–4023 were declared before any v6.2 rollout on 2026-09-22. Inspect
only development/regression until code is frozen. Every failure is retained.
This is software feasibility evidence, not human performance evidence.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.kitchen import engine as e

DEVELOPMENT_SEEDS = tuple(range(1000, 1024))
EXISTING_REGRESSION_SEEDS = tuple(range(2000, 2024)) + tuple(range(3000, 3024))
UNTOUCHED_VALIDATION_SEEDS = tuple(range(4000, 4024))
SPLITS = {'development': DEVELOPMENT_SEEDS, 'existing_regression': EXISTING_REGRESSION_SEEDS,
          'untouched_validation': UNTOUCHED_VALIDATION_SEEDS}
PROTOCOL_DECLARED = '2026-09-22; before any v6.2 run of seeds 4000–4023'


def intervals(turns):
    result = []
    for turn in turns:
        if result and turn == result[-1][1] + 1:
            result[-1][1] = turn
        else:
            result.append([turn, turn])
    return result


def rollout(seed, task):
    state = e.initial_state(seed, task)
    heating_steps, joint_steps, example, issues = [], [], None, []
    initial_menu = [dict(order) for order in state['orders']]
    event_score = 0
    while not state['terminal']:
        before = state
        human = e.simulation_partner(before)
        decision = e.decide(before)
        state = e.step(before, human, decision)
        event_score += sum(event['delta'] for event in state['events'] if event['type'] == 'score_delta')
        heating = all(p['status'] == 'cooking' for p in state['pots'])
        distinct_orders = len({p['order_id'] for p in state['pots']}) == 2
        if heating and distinct_orders:
            heating_steps.append(state['turn'])
        if (heating and distinct_orders and all(p['status'] == 'cooking' for p in before['pots'])
                and all(b['order_id'] == a['order_id'] and b['phase'] == a['phase']
                        and b['remaining'] == a['remaining'] - 1
                        for a, b in zip(before['pots'], state['pots']))):
            joint_steps.append(state['turn'])
            if example is None:
                example = {'before': before, 'human_action': human, 'ai_decision': decision, 'after': state}
    metrics = state['metrics']
    menu = [order['recipe'] for order in initial_menu]
    expected_score = 30 * metrics['completed_orders'] - state['turn'] - metrics['discard_penalty']
    if metrics['completed_orders'] != 5: issues.append('not_all_five_completed')
    if state['raw_score'] != expected_score: issues.append('wrong_v62_score_formula')
    if state['raw_score'] != event_score: issues.append('score_event_sum_mismatch')
    if state['turn'] > 360 or state['max_turns'] != 360: issues.append('wrong_turn_limit')
    if len(menu) != 5 or set(menu) != set(e.RECIPES): issues.append('wrong_menu_composition')
    if any(menu[i] == menu[i+1] == menu[i+2] for i in range(len(menu)-2)): issues.append('three_identical_dishes_in_sequence')
    if [o['deadline'] for o in initial_menu] != [100, 140, 240, 280, 360]: issues.append('wrong_deadlines')
    if len(heating_steps) != metrics['parallel_cooking_turns']: issues.append('heating_metric_mismatch')
    for key in ('burnt', 'spoiled', 'waste', 'expired'):
        if metrics[key]: issues.append(key)
    row = {'seed': seed, 'task': task, 'turns': state['turn'], 'score': state['raw_score'],
           'expected_score': expected_score, 'completed': metrics['completed_orders'], 'metrics': metrics,
           'menu': menu, 'deadlines': [o['deadline'] for o in initial_menu],
           'served_turns': {o['id']: o['served_turn'] for o in state['orders']},
           'heating_intervals_inclusive': intervals(heating_steps), 'joint_timer_decrement_turns': joint_steps,
           'joint_timer_decrement_intervals_inclusive': intervals(joint_steps),
           'issues': issues, 'passed': not issues,
           'final_state_sha256': sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()}
    return row, example


def run(selected='all'):
    config = e._configuration()
    if e.VERSION != 'kitchen-v6.2.0' or config['rules_version'] != e.VERSION:
        raise ValueError('This protocol requires synchronized Kitchen v6.2.0 engine/config')
    if e.SERVE_POINTS != 30 or set(e.PREPARED_FRESH_TURNS.values()) != {20}:
        raise ValueError('This protocol requires +30 serving and uniform prepared freshness 20')
    if e.COOK_TURNS != {'egg':8, 'meat':10, 'tomato':6, 'pepper':8} or e.MIX_TURNS != 2:
        raise ValueError('This protocol requires 16/20 pure heating')
    engine_path = Path(e.__file__)
    frozen_digest = sha256(engine_path.read_bytes()).hexdigest()
    config_path = engine_path.parents[2] / 'configs' / 'study_v3_kitchen.json'
    frozen_config_digest = sha256(config_path.read_bytes()).hexdigest()
    chosen = SPLITS if selected == 'all' else {selected: SPLITS[selected]}
    rows, examples = [], {}
    for split, seeds in chosen.items():
        for seed in seeds:
            menus = []
            for task in (1, 2, 3):
                row, pair = rollout(seed, task)
                rows.append(dict(row, split=split)); menus.append(tuple(row['menu']))
                if split not in examples and pair is not None:
                    examples[split] = dict(pair, seed=seed, task=task)
            if len(set(menus)) != 3:
                for row in rows[-3:]:
                    row['issues'].append('tasks_share_identical_menu'); row['passed'] = False
    summaries = {}
    for split in chosen:
        sample = [r for r in rows if r['split'] == split]
        summaries[split] = {
            'runs': len(sample), 'passed_runs': sum(r['passed'] for r in sample),
            'all_five_completed_runs': sum(r['completed'] == 5 for r in sample),
            'completed_orders': sum(r['completed'] for r in sample), 'assigned_orders': 5 * len(sample),
            'turn_range': [min(r['turns'] for r in sample), max(r['turns'] for r in sample)],
            'raw_score_range': [min(r['score'] for r in sample), max(r['score'] for r in sample)],
            'runs_with_joint_timer_decrements': sum(bool(r['joint_timer_decrement_turns']) for r in sample),
            'joint_timer_decrement_steps': sum(len(r['joint_timer_decrement_turns']) for r in sample),
            'joint_timer_decrement_step_range': [min(len(r['joint_timer_decrement_turns']) for r in sample), max(len(r['joint_timer_decrement_turns']) for r in sample)],
            'heating_snapshot_turn_range': [min(r['metrics']['parallel_cooking_turns'] for r in sample), max(r['metrics']['parallel_cooking_turns'] for r in sample)],
            **{key: sum(r['metrics'][key] for r in sample) for key in ('burnt', 'spoiled', 'waste', 'expired', 'discarded_ingredients', 'discarded_dishes', 'discard_penalty')},
            'failed_scenes': [{'seed': r['seed'], 'task': r['task'], 'issues': r['issues']} for r in sample if not r['passed']]}
    if (sha256(engine_path.read_bytes()).hexdigest() != frozen_digest
            or sha256(config_path.read_bytes()).hexdigest() != frozen_config_digest):
        raise RuntimeError('Engine/config changed during validation; cannot claim frozen-code evidence')
    return {'engine_version': e.VERSION, 'scenario_version': e.SCENARIO_VERSION, 'engine_sha256': frozen_digest, 'config_sha256': frozen_config_digest,
            'protocol_declared': PROTOCOL_DECLARED, 'predeclared_seed_lists': SPLITS,
            'evidence_type': 'actual fixed AI with deterministic simulation_partner; no human participants',
            'seed_note': 'Development 1000–1023; existing regression 2000–2023 plus 3000–3023 (48 distinct seeds, not the interval 2000–3023); new validation 4000–4023 declared before v6.2 evaluation. Tuning after inspection removes untouched status for subsequent runs.',
            'cook_turns': e.COOK_TURNS, 'mix_turns': e.MIX_TURNS, 'prepared_fresh_turns': e.PREPARED_FRESH_TURNS,
            'serve_points': e.SERVE_POINTS, 'menu_max_identical_run': 2,
            'task_budgets': config['task_budgets'], 'order_deadlines': config['order_deadlines'],
            'human_effect': 'not measured', 'passed': all(row['passed'] for row in rows),
            'summaries': summaries, 'runs': rows, 'actual_joint_step_examples': examples}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('all', *SPLITS), default='all')
    args = parser.parse_args()
    result = run(args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summaries'], ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
