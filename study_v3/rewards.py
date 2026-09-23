"""Versioned GBP performance rewards from authoritative final task scores.

Calculations are payment recommendations, not evidence of a Prolific transfer.
All monetary amounts are integer pence; each task is rounded half up once.
"""
from decimal import Decimal, ROUND_HALF_UP
import json

VERSION = 'gbp-280-base-20-per-task-v1'
BOUNDS = {'warehouse': (0, 100), 'pong': (45, 65), 'kitchen': (60, 120)}


def policy(domain):
    low, high = BOUNDS[domain]
    return {'version': VERSION, 'currency': 'GBP', 'base_pence': 280,
            'task_max_pence': 20, 'tasks': [1, 2, 3],
            'score_lower': low, 'score_upper': high,
            'score_field': 'task_score', 'rounding': 'nearest_penny_half_up'}


def task_bonus(score, rules):
    value = Decimal(str(score))
    if not value.is_finite():
        raise ValueError('non_finite_reward_score')
    low, high = Decimal(str(rules['score_lower'])), Decimal(str(rules['score_upper']))
    fraction = max(Decimal(0), min(Decimal(1), (value-low)/(high-low)))
    return int((fraction*rules['task_max_pence']).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def summary(db, instance_id):
    config = db.one('SELECT policy_json FROM pl3_reward_settings WHERE instance_id=?', (instance_id,))
    if not config:
        return None  # Never retrofit a different promise onto an old participant.
    rules = json.loads(config['policy_json'])
    rows = db.all('SELECT task,score_json FROM pl3_runs WHERE instance_id=? AND status=? ORDER BY task',
                  (instance_id, 'completed'))
    tasks = []
    for row in rows:
        if row['task'] not in rules['tasks']:
            continue
        score = json.loads(row['score_json'])['task_score']
        tasks.append({'task': row['task'], 'score': score, 'bonus_pence': task_bonus(score, rules)})
    total = sum(task['bonus_pence'] for task in tasks)
    return {**rules, 'completed_tasks': tasks, 'earned_bonus_pence': total,
            'total_pence': rules['base_pence']+total,
            'maximum_total_pence': rules['base_pence']+len(rules['tasks'])*rules['task_max_pence'],
            'payment_status': 'not_tracked_here'}
