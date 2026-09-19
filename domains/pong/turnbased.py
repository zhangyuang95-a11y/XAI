"""Deterministic rolling cooperative Pong; controllers see visible balls only.

The complete spawn schedule is private and fixed before play. One input advances
one physical step; animation never changes this engine's clock. Displayed lanes
are 1–9, coordinates 0–8, and the contact line is at vertical coordinate 12.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import random

DOMAIN = 'pong'
CONFIG = json.loads((Path(__file__).resolve().parents[2] / 'configs/study_v3_pong.json').read_text())
VERSION = CONFIG['version']
LANES, HEIGHT, WEIGHTS = CONFIG['lanes'], CONFIG['height'], CONFIG['weights']
_DELTAS = {'left': -1, 'wait': 0, 'right': 1}


def _ball(identifier, kind, contacts, remaining):
    vy = 2 if kind == 'ordinary' else 1
    return {'id': identifier, 'kind': kind, 'contacts': list(contacts),
            'remaining': remaining, 'initial_remaining': remaining,
            'y': HEIGHT - vy * remaining, 'vy': vy}


def _schedule(seed, task):
    """Construct a finite fair schedule, independently of play or group.

    Adjacent ordinary arrivals are two lanes apart. Cooperative contacts include
    the intermediate lane and a separate reachable partner lane. This gives a
    purposeful moving route without fake idle-prevention movements. The latent
    construction path is never supplied to either controller or explanation.
    """
    rng = random.Random(int(seed) * 1009 + task * 71)
    duration = CONFIG['task_turns'][str(task)]
    odd = {1: rng.choice((5, 7))}
    for at in range(3, duration + 2, 2):
        odd[at] = rng.choice([x for x in (odd[at - 2] - 2, odd[at - 2] + 2) if 0 <= x < LANES])
    entries = []
    for index, at in enumerate(range(1, duration, 2), 1):
        spawn = max(0, at - 6)
        entries.append({'spawn_turn': spawn, 'arrival_turn': at,
                        'ball': _ball(f's{index}', 'ordinary', [odd[at]], at - spawn)})
    human = 2
    for index, at in enumerate(range(6, duration + 1, 6), 1):
        own = (odd[at - 1] + odd[at + 1]) // 2
        human = rng.choice([x for x in range(LANES) if abs(x - own) >= 3 and abs(x - human) <= 6])
        spawn = max(0, at - 12)
        entries.append({'spawn_turn': spawn, 'arrival_turn': at,
                        'ball': _ball(f't{index}', 'cooperative', sorted((own, human)), at - spawn)})
    if task == 3:
        for entry in entries:
            entry['ball']['contacts'] = sorted(8 - x for x in entry['ball']['contacts'])
    return sorted(entries, key=lambda row: (row['spawn_turn'], row['arrival_turn'], row['ball']['id']))


def _new_state(schedule, *, seed, task, human, ai):
    # Historical short wave fixtures remain useful for independent QA tests.
    # Production initial_state always supplies the finite rolling schedule.
    if schedule and isinstance(schedule[0], list):
        entries, offset = [], 0
        for wave in schedule:
            for ball in wave:
                normalized = _ball(ball['id'], ball['kind'], ball['contacts'], ball['remaining'])
                entries.append({'spawn_turn': offset, 'arrival_turn': offset + ball['remaining'], 'ball': normalized})
            offset += max(b['remaining'] for b in wave)
    else:
        entries = deepcopy(schedule)
    all_balls = [e['ball'] for e in entries]
    duration = max((e['arrival_turn'] for e in entries), default=0)
    return {'domain': DOMAIN, 'version': VERSION, 'scenario_version': CONFIG['scenario_version'],
            'task': task, 'turn': 0, 'max_turns': duration, 'height': HEIGHT, 'lanes': LANES,
            'terminal': not entries, 'events': [], 'policy_memory': {}, 'seed': seed,
            'human': {'x': human}, 'ai': {'x': ai},
            'wave': 1, 'wave_count': max(1, (duration + 5) // 6),
            'balls': deepcopy([e['ball'] for e in entries if e['spawn_turn'] == 0]),
            '_schedule': deepcopy(entries), 'raw_score': 0,
            'total_possible_points': sum(WEIGHTS[b['kind']] for b in all_balls),
            'metrics': {'cooperative_caught': 0, 'ordinary_caught': 0,
                        'cooperative_total': sum(b['kind'] == 'cooperative' for b in all_balls),
                        'ordinary_total': sum(b['kind'] == 'ordinary' for b in all_balls),
                        'misses': 0, 'assignment_switches': 0, 'assignment_releases': 0,
                        'waiting_turns': 0, 'holding_turns': 0, 'no_job_waiting_turns': 0}}


def initial_state(seed, task):
    if task not in (1, 2, 3):
        raise ValueError('task must be 1, 2 or 3')
    human, ai = (6, 2) if task == 3 else (2, 6)
    return _new_state(_schedule(seed, task), seed=int(seed), task=task, human=human, ai=ai)


def legal_actions(state, actor='human'):
    if actor not in ('human', 'ai'):
        raise ValueError('actor must be human or ai')
    if state['terminal']:
        return []
    return [a for a in ('wait', 'left', 'right') if 0 <= state[actor]['x'] + _DELTAS[a] < LANES]


def _move_towards(position, target):
    return 'left' if target < position else 'right' if target > position else 'wait'


def _observation(state):
    return {key: deepcopy(state[key]) for key in ('human', 'ai', 'balls', 'policy_memory', 'terminal')}


def _route(start, appointments):
    """Exact feasibility/minimal distance for fixed one-dimensional deadlines."""
    previous_time, position, distance = 0, start, 0
    ordered = sorted(appointments)
    for at, target, _identifier in ordered:
        gap = abs(target - position)
        if gap > at - previous_time:
            return None
        distance += gap
        previous_time, position = at, target
    return distance, ordered


def _commitment(ball, own, partner, *, locked=True):
    return {'ball_id': ball['id'], 'ai_contact': own, 'human_contact': partner, 'locked': locked}


def _same_assignment(left, right):
    return bool(left and right and all(left.get(k) == right.get(k) for k in ('ball_id', 'ai_contact', 'human_contact')))


def _plan(obs):
    """Enumerate <=9 team choices and <=27 small allocations, no future data.

    This is event-time dynamic planning: between deadlines shortest monotone
    paths suffice, so checking ordered appointments exactly covers all feasible
    joint catch routes without enumerating every animation frame. Small-ball
    work is preferentially assigned to the AI, after preserving team catches
    and total points, so motion always serves a real selected catch.
    """
    teams = sorted((b for b in obs['balls'] if b['kind'] == 'cooperative'), key=lambda b: (b['remaining'], b['id']))
    smalls = sorted((b for b in obs['balls'] if b['kind'] == 'ordinary'), key=lambda b: (b['remaining'], b['id']))
    previous = obs['policy_memory'].get('commitment')
    options = []
    for index, ball in enumerate(teams):
        left, right = ball['contacts']
        choices = [_commitment(ball, right, left, locked=index == 0), _commitment(ball, left, right, locked=index == 0), None]
        if index == 0 and previous and previous.get('ball_id') == ball['id']:
            retained = next((c for c in choices if _same_assignment(c, previous)), None)
            if retained and abs(obs['ai']['x'] - retained['ai_contact']) <= ball['remaining'] and abs(obs['human']['x'] - retained['human_contact']) <= ball['remaining']:
                choices = [retained]
        options.append(choices)
    best = None
    for assignments in product(*options):
        human_points, ai_points = [], []
        selected = []
        for ball, assignment in zip(teams, assignments):
            if assignment:
                selected.append(assignment)
                human_points.append((ball['remaining'], assignment['human_contact'], ball['id']))
                ai_points.append((ball['remaining'], assignment['ai_contact'], ball['id']))
        if _route(obs['human']['x'], human_points) is None or _route(obs['ai']['x'], ai_points) is None:
            continue
        # Ignore / human / AI for each ordinary ball. A ball can score only once.
        for owners in product((0, 1, 2), repeat=len(smalls)):
            hp, ap = list(human_points), list(ai_points)
            for ball, owner in zip(smalls, owners):
                if owner:
                    (hp if owner == 1 else ap).append((ball['remaining'], ball['contacts'][0], ball['id']))
            hr, ar = _route(obs['human']['x'], hp), _route(obs['ai']['x'], ap)
            if hr is None or ar is None:
                continue
            count = sum(owner != 0 for owner in owners)
            # Team priority, then all visible points, then useful AI small catches.
            # Earlier teams win when not all are feasible; right side is last tie.
            rank = (len(selected), tuple(int(a is not None) for a in assignments),
                    count, owners.count(2), -(hr[0] + ar[0]),
                    tuple(a['ai_contact'] if a else -1 for a in assignments), tuple(owners))
            if best is None or rank > best[0]:
                best = (rank, selected, hr[1], ar[1], owners)
    assert best is not None
    return teams, smalls, best[1], best[2], best[3], best[4]


def _decision_from_observation(obs):
    # Cache immutable serialized observations, returning isolated results.
    return deepcopy(_cached_decision(json.dumps(obs, sort_keys=True, separators=(',', ':'))))


@lru_cache(maxsize=12000)
def _cached_decision(key):
    obs = json.loads(key)
    if obs['terminal']:
        return {'action': 'wait', 'human_action': 'wait', 'reason_code': 'task_complete',
                'reason_en': 'This task has finished. There is no next action to execute.',
                'reason_zh': '本任务已经结束，没有下一步需要执行。', 'goal': 'task_complete',
                'memory': {}, 'alternatives': [], 'assignment_changed': False,
                'assignment_released': False, 'target_lane': None, 'assignments': [], 'planned_catches': []}
    teams, smalls, assignments, human_path, ai_path, owners = _plan(obs)
    previous = obs['policy_memory'].get('commitment')
    first = next((a for a in assignments if teams and a['ball_id'] == teams[0]['id']), None)
    released = bool(previous and any(b['id'] == previous['ball_id'] for b in teams) and not _same_assignment(previous, first))
    switched = bool(released and first and first['ball_id'] == previous['ball_id'])
    target = ai_path[0][1] if ai_path else None
    goal = ai_path[0][2] if ai_path else 'hold_position'
    action = _move_towards(obs['ai']['x'], target) if target is not None else 'wait'
    human_action = _move_towards(obs['human']['x'], human_path[0][1]) if human_path else 'wait'
    current_small = next((b for b in smalls if b['id'] == goal), None)
    code = ('small_then_return' if current_small and assignments else 'ordinary_after_infeasible_team' if current_small and teams
            else 'catch_ordinary' if current_small else 'switch_unreachable_assignment' if switched
            else 'keep_assignment' if first and _same_assignment(first, previous) else 'assign_team_ball' if first
            else 'release_infeasible_assignment' if released else 'no_reachable_ball')
    en, zh = [], []
    if current_small:
        en.append(f"I will cover lane {target + 1} for {goal}, which arrives in {current_small['remaining']} turns.")
        zh.append(f"我会到第{target + 1}道接{goal}，它将在{current_small['remaining']}回合后到达。")
    for assignment in assignments:
        ball = next(b for b in teams if b['id'] == assignment['ball_id'])
        own, partner = assignment['ai_contact'] + 1, assignment['human_contact'] + 1
        if assignment['locked']:
            en.append(f"For {ball['id']} in {ball['remaining']} turns, I will cover lane {own}; you need to cover lane {partner}. I will keep this division while both sides remain reachable.")
            zh.append(f"对于{ball['remaining']}回合后到达的{ball['id']}，我负责第{own}道，你需要覆盖第{partner}道。只要双方仍然可达，我就保持这个分工。")
        else:
            en.append(f"After that, my current plan for {ball['id']} in {ball['remaining']} turns is lane {own}, with you at lane {partner}; this later plan may change when new balls appear.")
            zh.append(f"之后，关于{ball['remaining']}回合后到达的{ball['id']}，当前计划是我去第{own}道、你去第{partner}道；新球出现后，这个后续计划可能调整。")
    if current_small and assignments:
        en.append('The selected small-ball route leaves enough moves to reach each planned team-ball contact, provided you reach your contacts too.')
        zh.append('按当前选定的普通球路线，之后仍有足够步数到达每个计划中的合作接球点；你也需要及时到达你的接球点。')
    if switched:
        en.append('The previous division is no longer reachable, so I changed sides.')
        zh.append('原来的分工已经无法及时到达，因此我换了另一侧。')
    if not ai_path:
        en.append('No currently visible catching job is reachable for me, so I will stay here.')
        zh.append('当前没有我能及时完成的可见接球任务，因此我会留在这里。')
    elif action == 'wait':
        en.append(f'I am already at lane {target + 1}, the next selected catch position, so I will hold it this turn.')
        zh.append(f'我已在下一个选定接球位置第{target + 1}道，因此本回合会守住这里。')
    tentative_changes = []
    for assignment in assignments:
        old = next((a for a in obs['policy_memory'].get('commitments', []) if a['ball_id'] == assignment['ball_id']), None)
        if old and not old.get('locked', False) and not _same_assignment(old, assignment):
            tentative_changes.append({'ball_id': assignment['ball_id'], 'previous_ai_contact': old['ai_contact'],
                                      'previous_human_contact': old['human_contact'], 'ai_contact': assignment['ai_contact'],
                                      'human_contact': assignment['human_contact'], 'basis': 'current_visible_balls_and_positions'})
            en.append(f"My earlier tentative division for {assignment['ball_id']} changed after reassessing the visible balls and our current positions.")
            zh.append(f"根据当前可见的球和双方实际位置重新检查后，我调整了{assignment['ball_id']}先前的暂定分工。")
    comparisons = []
    for ball in teams:
        for own, partner in (ball['contacts'], list(reversed(ball['contacts']))):
            ok = abs(obs['ai']['x'] - own) <= ball['remaining'] and abs(obs['human']['x'] - partner) <= ball['remaining']
            comparisons.append({'action': _move_towards(obs['ai']['x'], own),
                'en': f"For {ball['id']} alone, lane {own + 1} needs {abs(obs['ai']['x'] - own)} moves from me and lane {partner + 1} needs {abs(obs['human']['x'] - partner)} moves from you; {ball['remaining']} turns remain. " + ('Both contacts are individually reachable; other catches may still conflict.' if ok else 'At least one contact cannot be reached in time.'),
                'zh': f"仅看{ball['id']}，我到第{own + 1}道需要{abs(obs['ai']['x'] - own)}步，你到第{partner + 1}道需要{abs(obs['human']['x'] - partner)}步；还剩{ball['remaining']}回合。" + ('两侧单独看都可达，但仍可能与其他接球冲突。' if ok else '至少一侧已经来不及。')})
    for ball in smalls:
        appointments = [(b['remaining'], a['ai_contact'], b['id']) for a in assignments for b in teams if a['ball_id'] == b['id']]
        appointments.append((ball['remaining'], ball['contacts'][0], ball['id']))
        safe = _route(obs['ai']['x'], appointments) is not None
        comparisons.append({'action': _move_towards(obs['ai']['x'], ball['contacts'][0]),
            'en': f"Catching {ball['id']} at lane {ball['contacts'][0] + 1} requires being there in {ball['remaining']} turns. " + ('There is a route from my current position through that catch and all my planned team-ball deadlines.' if safe else 'I cannot reach that catch and all my planned team-ball contacts by their deadlines.'),
            'zh': f"接{ball['id']}需要在{ball['remaining']}回合后位于第{ball['contacts'][0] + 1}道。" + ('从我当前位置出发，存在能接它并及时到达所有计划中合作接球点的路线。' if safe else '我无法接它并按时到达所有计划中的合作接球点。')})
    memory = {'commitments': deepcopy(assignments)} if assignments else {}
    if first:
        memory['commitment'] = deepcopy(first)
    return {'action': action, 'human_action': human_action, 'reason_code': code,
            'reason_en': ' '.join(en), 'reason_zh': ''.join(zh), 'goal': goal, 'memory': memory,
            'alternatives': comparisons, 'assignment_changed': switched, 'assignment_released': released,
            'target_lane': target, 'assignments': deepcopy(assignments), 'tentative_replans': tentative_changes,
            'planned_catches': [{'ball_id': b['id'], 'actor': 'human' if owner == 1 else 'ai'} for b, owner in zip(smalls, owners) if owner]}


def decide(state):
    return _decision_from_observation(_observation(state))


def _event(event_type, en, zh, **fields):
    return {'type': event_type, 'en': en, 'zh': zh, **fields}


def _settle_visible(obs, human_action, decision):
    nxt = deepcopy(obs)
    nxt['human']['x'] += _DELTAS[human_action]
    nxt['ai']['x'] += _DELTAS[decision['action']]
    nxt['policy_memory'] = deepcopy(decision['memory'])
    points, events, remaining = 0, [], []
    for ball in nxt['balls']:
        ball['remaining'] -= 1
        ball['y'] = ball.get('y', HEIGHT - ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1) * (ball['remaining'] + 1)) + ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1)
        if ball['remaining'] > 0:
            remaining.append(ball)
            continue
        human, ai = nxt['human']['x'], nxt['ai']['x']
        caught = ({human, ai} == set(ball['contacts'])) if ball['kind'] == 'cooperative' else ball['contacts'][0] in (human, ai)
        earned = WEIGHTS[ball['kind']] if caught else 0
        points += earned
        events.append(_event('caught' if caught else 'missed', f"{ball['id']} {'caught' if caught else 'missed'}: +{earned} points.",
                             f"{ball['id']}{'接住' if caught else '漏接'}：+{earned}分。", ball_id=ball['id'], kind=ball['kind'], points=earned,
                             contacts=list(ball['contacts']), human_x=human, ai_x=ai))
    nxt['balls'] = remaining
    ids = {b['id'] for b in remaining}
    if nxt['policy_memory'].get('commitment', {}).get('ball_id') not in ids:
        nxt['policy_memory'].pop('commitment', None)
    nxt['policy_memory']['commitments'] = [a for a in nxt['policy_memory'].get('commitments', []) if a['ball_id'] in ids]
    if not nxt['policy_memory'].get('commitments'):
        nxt['policy_memory'] = {}
    return nxt, points, events


def step(state, human_action, decision=None):
    if state['terminal']:
        raise ValueError('This task is complete.')
    if human_action not in legal_actions(state):
        raise ValueError('Illegal human action.')
    expected = decide(state)
    if decision is not None and decision != expected:
        raise ValueError('The AI decision does not match the submitted state.')
    decision = expected
    result = deepcopy(state)
    projected, gained, events = _settle_visible(_observation(state), human_action, decision)
    for key in ('human', 'ai', 'balls', 'policy_memory'):
        result[key] = projected[key]
    result['turn'] += 1
    result['raw_score'] += gained
    result['events'] = events
    metrics = result['metrics']
    metrics['waiting_turns'] += decision['action'] == 'wait'
    metrics['holding_turns'] += decision['action'] == 'wait' and decision['target_lane'] is not None
    metrics['no_job_waiting_turns'] += decision['action'] == 'wait' and decision['target_lane'] is None
    metrics['assignment_switches'] += bool(decision['assignment_changed'])
    metrics['assignment_releases'] += bool(decision['assignment_released'])
    for event in events:
        if event['type'] == 'caught':
            metrics[event['kind'] + '_caught'] += 1
        else:
            metrics['misses'] += 1
    spawned = [deepcopy(e['ball']) for e in state['_schedule'] if e['spawn_turn'] == result['turn']]
    result['balls'].extend(spawned)
    if spawned:
        result['events'].append(_event('balls_spawned', 'New balls entered at the top.', '新球从顶部进入。', ball_ids=[b['id'] for b in spawned]))
    result['wave'] = min(result['wave_count'], result['turn'] // 6 + 1)
    result['terminal'] = result['turn'] >= result['max_turns']
    return result


def score(state):
    denominator = state['total_possible_points']
    return {'task_score': round(100 * state['raw_score'] / denominator, 6) if denominator else 0.0,
            'raw_score': state['raw_score'], 'metrics': deepcopy(state['metrics']) | {
                'total_possible_points': denominator,
                'cooperative_success_rate': state['metrics']['cooperative_caught'] / max(1, state['metrics']['cooperative_total']),
                'ordinary_success_rate': state['metrics']['ordinary_caught'] / max(1, state['metrics']['ordinary_total']),
                'waiting_rate': state['metrics']['waiting_turns'] / max(1, state['turn']), 'elapsed_turns': state['turn']}}


def public_state(state):
    public = {key: deepcopy(state[key]) for key in ('domain', 'version', 'task', 'turn', 'max_turns', 'height', 'lanes', 'terminal', 'events', 'human', 'ai', 'wave', 'wave_count', 'balls')}
    public['score'] = score(state)
    for key in ('assignment_switches', 'assignment_releases', 'holding_turns', 'no_job_waiting_turns'):
        public['score']['metrics'].pop(key, None)
    return public


def rules(language='en'):
    en = ['The court has nine lanes and twelve vertical cells. A and D move one lane left and right; Space waits. One action advances one step. Paddles may share a lane.',
          'Both paddles move together, then small balls fall two cells and team balls fall one. Balls reaching the contact line are scored immediately.',
          'A small ball earns 1 point if either paddle covers its lane. Covering it twice still earns only 1 point.',
          'A team ball earns 3 points only when the two paddles cover its two different contact lanes at the same arrival step.',
          'Up to three small balls and two team balls are visible. New balls enter at the top after earlier balls finish. The supply ends before the task ends; every scheduled ball can arrive before the limit.',
          'Task score is 100 times points earned divided by all scheduled ball points. Reading, questions and replay never advance the game. The teammate plans using only currently visible balls.']
    zh = ['球场有九道、高十二格。A、D分别左移和右移一道，空格等待；每次操作推进一步。双方球拍可以重叠。',
          '双方球拍同时移动，随后小球下降两格、合作球下降一格；到达接球线的球立即结算。',
          '小球由任一球拍覆盖所在道即可得1分；双方同时覆盖仍然只得1分。',
          '合作球必须在同一步由两个球拍分别覆盖两个不同的接触位置，才能得3分。',
          '画面最多同时有三个小球、两个合作球。旧球结束后新球从顶部进入；任务结束前停止补球，所有预定球都能在步数上限前到达。',
          '任务分数为100乘实得分除以全部预定球的分值。阅读、提问和回放不推进游戏。队友只根据当前可见的球制定计划。']
    return zh if language.startswith('zh') else en


def facts(state, decision=None):
    actual = decide(state)
    if decision is not None and decision != actual:
        raise ValueError('Decision evidence does not match this state.')
    evidence = [{'id': 'position', 'en': f"You are at lane {state['human']['x'] + 1}; your teammate is at lane {state['ai']['x'] + 1}.",
                 'zh': f"你在第{state['human']['x'] + 1}道，队友在第{state['ai']['x'] + 1}道。"},
                {'id': 'decision', 'en': actual['reason_en'], 'zh': actual['reason_zh']},
                {'id': 'next_action', 'en': 'This task is finished; there is no next action.' if state['terminal'] else 'The teammate\'s next action is ' + {'left':'move left','right':'move right','wait':'wait'}[actual['action']] + '.',
                 'zh': '任务已经结束，没有下一步动作。' if state['terminal'] else '队友下一步将' + {'left':'左移','right':'右移','wait':'等待'}[actual['action']] + '。'}]
    for ball in state['balls']:
        lanes = ', '.join(str(x + 1) for x in ball['contacts'])
        evidence.append({'id': 'ball:' + ball['id'],
            'en': f"{ball['id']} is a {'team' if ball['kind'] == 'cooperative' else 'small'} ball worth {WEIGHTS[ball['kind']]} points, arriving at lane(s) {lanes} in {ball['remaining']} turns; it falls {ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1)} cells each step.",
            'zh': f"{ball['id']}是{'合作' if ball['kind'] == 'cooperative' else '小'}球，值{WEIGHTS[ball['kind']]}分，将在{ball['remaining']}回合后到达第{lanes}道，每步下降{ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1)}格。"})
    for i, assignment in enumerate(actual['assignments']):
        evidence.append({'id': f'assignment:{i}', 'en': f"For {assignment['ball_id']}, the {'committed' if assignment['locked'] else 'tentative later'} plan has me at lane {assignment['ai_contact'] + 1} and you at lane {assignment['human_contact'] + 1}.",
                         'zh': f"关于{assignment['ball_id']}，{'当前承诺的' if assignment['locked'] else '暂定后续'}分工是我在第{assignment['ai_contact'] + 1}道，你在第{assignment['human_contact'] + 1}道。"})
    evidence.extend({'id': f'comparison:{i}', 'en': row['en'], 'zh': row['zh']} for i, row in enumerate(actual['alternatives']))
    evidence.extend({'id': f'public_rule:{i}', 'en': en, 'zh': zh} for i, (en, zh) in enumerate(zip(rules(), rules('zh'))))
    evidence.extend({'id': f'event:{i}', 'en': e['en'], 'zh': e['zh']} for i, e in enumerate(state['events']))
    return evidence


def human_advisor(state):
    """Visible-state cooperative proxy; the real AI executes independently."""
    return decide(state)['human_action'] if not state['terminal'] else 'wait'


def greedy_human(state):
    if not state['balls']:
        return 'wait'
    x = state['human']['x']
    ball = min(state['balls'], key=lambda b: (b['remaining'], b['id']))
    return _move_towards(x, min(ball['contacts'], key=lambda at: (abs(x - at), at)))


def history_learning_human(state, history):
    teams = sorted((b for b in state['balls'] if b['kind'] == 'cooperative'), key=lambda b: (b['remaining'], b['id']))
    if not teams:
        return greedy_human(state)
    examples = [e for f in history for e in f.get('events', []) if e.get('kind') == 'cooperative' and e.get('ai_x') in e.get('contacts', [])]
    if not examples:
        return greedy_human(state)
    ai_right = sum(e['ai_x'] == max(e['contacts']) for e in examples) * 2 >= len(examples)
    return _move_towards(state['human']['x'], min(teams[0]['contacts']) if ai_right else max(teams[0]['contacts']))


def wave_feasibility(state):
    """Small-fixture exact fixed-AI search, with unseen future balls excluded.

    Full rolling studies use observed proxy rollouts, not a claimed exact global
    optimum. Bounding this diagnostic avoids exponential work in live QA.
    """
    obs = _observation(state)
    horizon = max((b['remaining'] for b in obs['balls']), default=0)
    if horizon > 6:
        raise ValueError('Exact fixed-AI diagnostic is limited to six visible steps; use the visible planning proxy for full tasks.')
    def solve(current):
        if not current['balls']:
            return 0, ()
        decision = _decision_from_observation(current)
        best = (-1, ())
        for action in legal_actions(current):
            nxt, earned, _ = _settle_visible(current, action, decision)
            later, path = solve(nxt)
            candidate = (earned + later, (action,) + path)
            if candidate[0] > best[0]:
                best = candidate
        return best
    points, actions = solve(obs)
    return {'reachable_points': points, 'public_wave_points': sum(WEIGHTS[b['kind']] for b in obs['balls']),
            'human_actions': list(actions), 'scope': 'current_visible_balls_without_future_spawns', 'ai_policy': VERSION}


def global_reachable_upper_bound(state):
    if any(e['spawn_turn'] > state['turn'] for e in state['_schedule']):
        raise ValueError('A global optimum for rolling schedules is not claimed; use recorded fixed-AI calibration.')
    result = wave_feasibility(state)
    return {'reachable_points': result['reachable_points'],
            'task_score_upper_bound': 100 * (state['raw_score'] + result['reachable_points']) / state['total_possible_points'],
            'scope': 'offline_small_fixture_upper_bound', 'ai_policy': VERSION, 'participant_visible': False}


def demonstration():
    # A separate twelve-step finite scene with actual successes and a miss.
    entries = [{'spawn_turn': 0, 'arrival_turn': at, 'ball': _ball(identifier, kind, contacts, at)} for identifier, kind, contacts, at in (
        ('demo-small-1','ordinary',[5],1), ('demo-small-2','ordinary',[7],3), ('demo-small-3','ordinary',[5],5),
        ('demo-team-1','cooperative',[2,6],6), ('demo-team-2','cooperative',[1,6],12))]
    entries.append({'spawn_turn': 1, 'arrival_turn': 7, 'ball': _ball('demo-small-4','ordinary',[7],6)})
    state = _new_state(entries, seed=0, task=1, human=2, ai=6)
    frames = [public_state(state)]
    while not state['terminal']:
        action = human_advisor(state) if state['turn'] < 6 else 'wait'
        state = step(state, action)
        frames.append(public_state(state))
    captions = [(0, 'Three small balls and two team balls are already on court. Their arrival times differ.', '三个小球和两个合作球已经在场上，到达时间各不相同。'),
                (1, 'One input moved both paddles; small balls fell two cells and team balls one. A caught small ball earned 1 point and a new ball entered at the top.', '一次操作使双方球拍移动；小球下降两格，合作球下降一格。接住小球得到1分，新球从顶部进入。'),
                (3, 'The next small ball was caught. Every ball is counted once, and the other balls keep falling.', '下一颗小球被接住了。每颗球只计分一次，其他球继续下落。'),
                (6, 'Two paddles covered the different team-ball contacts together, earning 3 points.', '两块球拍同时覆盖合作球的两个不同接触点，得到3分。'),
                (7, 'The replacement small ball reached the bottom after six steps from the top.', '补入的小球从顶部经过六步到达了底部。'),
                (12, 'A later team ball was missed because one contact was uncovered. The demonstration is complete; its points do not count toward your tasks.', '之后的合作球有一个接触点未被覆盖，因此漏接。演示结束，这些分数不计入正式任务。')]
    return {'frames': frames, 'captions': [{'index': index, 'en': en, 'zh': zh} for index, en, zh in captions]}


def comprehension(language='en'):
    en = [
        {'id':'pong-speed','text':'A small ball and a team ball start twelve cells above the catch line. After three steps, how many cells remain for each?', 'options':['Small: six; team: nine','Small: nine; team: six','Both: nine'],'answer':0},
        {'id':'pong-two-contacts','text':'A team ball reaches lanes 3 and 7. Both paddles are in lane 3. What happens?', 'options':['The team earns 3 points','The team earns 0 points because lane 7 is uncovered','The team earns 1 point'],'answer':1},
        {'id':'pong-two-deadlines','text':'Your teammate is in lane 6. A small ball arrives at lane 5 in one step, then its assigned team-ball contact is lane 7 one step later. Can it catch the small ball and keep that team-ball assignment?', 'options':['Yes, because small balls move faster','Yes, because one paddle can cover both lanes','No: moving from lane 5 to lane 7 needs two steps'],'answer':2}]
    zh = [
        {'id':'pong-speed','text':'一个小球和一个合作球都从接球线上方十二格出发。三步后，它们各剩多少格？','options':['小球六格，合作球九格','小球九格，合作球六格','都剩九格'],'answer':0},
        {'id':'pong-two-contacts','text':'合作球到达第3道和第7道，双方球拍都在第3道，会怎样？','options':['团队得到3分','第7道没有被覆盖，因此得0分','团队得到1分'],'answer':1},
        {'id':'pong-two-deadlines','text':'队友在第6道。小球一步后到达第5道，再过一步合作球到达队友负责的第7道。队友能先接小球并保持合作球分工吗？','options':['可以，因为小球移动更快','可以，因为一块球拍可以覆盖两道','不可以：从第5道到第7道需要两步'],'answer':2}]
    return zh if language.startswith('zh') else en
