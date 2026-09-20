"""Deterministic rolling cooperative Pong; controllers see visible balls only.

The complete spawn schedule is private and fixed before play. One input advances
one physical step; animation never changes this engine's clock. Displayed lanes
are 1–9, coordinates 0–8, and the contact line is at vertical coordinate 12.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
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
    """Finite seeded simultaneous arrivals, independent of group and play.

    Every two turns bring two distinct ordinary contacts. Every six turns also
    bring two incompatible team balls: each shares one of the small contacts
    and has a different second contact. No pair of paddles can collect both
    team balls, so a missed opportunity is not necessarily poor coordination.
    """
    rng = random.Random(int(seed) * 1009 + task * 71)
    duration = CONFIG['task_turns'][str(task)]
    entries, small_index, team_index = [], 0, 0
    previous = (2, 6)
    for at in range(2, duration + 1, 2):
        choices = [(left, right) for left in range(LANES) for right in range(LANES)
                   if left != right and abs(left - previous[0]) == 2 and abs(right - previous[1]) == 2
                   and (left != previous[0] or right != previous[1])]
        contacts = rng.choice(choices)
        previous = contacts
        for lane in contacts:
            small_index += 1
            spawn = max(0, at - 6)
            entries.append({'spawn_turn': spawn, 'arrival_turn': at,
                            'ball': _ball(f's{small_index}', 'ordinary', [lane], at - spawn)})
        if at % 6 == 0:
            alternatives = [(left, right) for left in range(LANES) for right in range(LANES)
                            if len({*contacts, left, right}) == 4
                            and abs(left - contacts[0]) >= 3 and abs(right - contacts[1]) >= 3]
            other_contacts = rng.choice(alternatives)
            for lane, other in zip(contacts, other_contacts):
                team_index += 1
                spawn = max(0, at - 12)
                entries.append({'spawn_turn': spawn, 'arrival_turn': at,
                                'ball': _ball(f't{team_index}', 'cooperative', sorted((lane, other)), at - spawn)})
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


# All legal simultaneous one-lane moves. This keeps planning bounded by
# twelve time layers x 81 position pairs, not 3^(number of visible balls).
_PAIR_MOVES = tuple(tuple((nh * LANES + na, int(nh != h) + int(na != a),
                          (2 if na > a else 1 if na < a else 0) * 3 + (2 if nh < h else 1 if nh > h else 0))
                         for nh in range(max(0, h - 1), min(LANES, h + 2))
                         for na in range(max(0, a - 1), min(LANES, a + 2)))
                    for h in range(LANES) for a in range(LANES))


def _retained_commitments(obs, teams):
    candidates = [a for a in obs['policy_memory'].get('commitments', []) if a.get('locked')]
    old = obs['policy_memory'].get('commitment')
    if old and not any(a['ball_id'] == old['ball_id'] for a in candidates):
        candidates.append(old)
    by_id = {b['id']:b for b in teams}
    retained, hp, ap = [], [], []
    for assignment in sorted(candidates, key=lambda a: (by_id.get(a['ball_id'], {}).get('remaining', 999), a['ball_id'])):
        ball = by_id.get(assignment['ball_id'])
        if not ball or {assignment['human_contact'], assignment['ai_contact']} != set(ball['contacts']):
            continue
        h = (ball['remaining'], assignment['human_contact'], ball['id'])
        a = (ball['remaining'], assignment['ai_contact'], ball['id'])
        if _route(obs['human']['x'], hp + [h]) is not None and _route(obs['ai']['x'], ap + [a]) is not None:
            retained.append(assignment); hp.append(h); ap.append(a)
    return retained


def _plan(obs):
    """Exact visible joint route under retained feasible commitments.

    Rank by real catch points, then team catches, useful AI small catches,
    shortest total movement, and a fixed tie order. Incompatible simultaneous
    contacts are settled by the same reward table as the physical engine.
    No future spawns or submitted human actions enter this planner.
    """
    teams = sorted((b for b in obs['balls'] if b['kind'] == 'cooperative'), key=lambda b: (b['remaining'], b['id']))
    smalls = sorted((b for b in obs['balls'] if b['kind'] == 'ordinary'), key=lambda b: (b['remaining'], b['id']))
    retained = _retained_commitments(obs, teams)
    horizon = max((b['remaining'] for b in obs['balls']), default=0)
    deadlines = {}
    by_id = {b['id']:b for b in teams}
    for assignment in retained:
        deadlines[by_id[assignment['ball_id']]['remaining']] = assignment['human_contact'] * LANES + assignment['ai_contact']
    arrivals = {}
    for ball in obs['balls']:
        arrivals.setdefault(ball['remaining'], []).append(ball)
    rewards = {}
    for at, balls in arrivals.items():
        table = []
        for h in range(LANES):
            for a in range(LANES):
                points = team_count = ai_small = 0
                for ball in balls:
                    if ball['kind'] == 'cooperative':
                        caught = {h, a} == set(ball['contacts'])
                        team_count += caught
                    else:
                        caught = ball['contacts'][0] in (h, a)
                        ai_small += ball['contacts'][0] == a
                    points += WEIGHTS[ball['kind']] if caught else 0
                table.append((points, team_count, ai_small))
        rewards[at] = table
    initial = obs['human']['x'] * LANES + obs['ai']['x']
    # Value = rank, actual position trace. No state mutation and no neural model.
    frontier = {initial: ((0, 0, 0, 0, 0), ())}
    zero_reward = [(0, 0, 0)] * (LANES * LANES)
    for at in range(1, horizon + 1):
        table, required = rewards.get(at, zero_reward), deadlines.get(at)
        following = {}
        for position, (rank, trace) in frontier.items():
            for nxt, movement, tie in _PAIR_MOVES[position]:
                if required is not None and nxt != required:
                    continue
                points, team_count, ai_small = table[nxt]
                candidate = (rank[0] + points, rank[1] + team_count, rank[2] + ai_small,
                             rank[3] - movement, rank[4] * 9 + tie)
                previous = following.get(nxt)
                if previous is None or candidate > previous[0]:
                    following[nxt] = (candidate, trace + (nxt,))
        frontier = following
    assert frontier, 'A retained commitment must have a feasible joint route.'
    _rank, trace = max(frontier.values(), key=lambda item:item[0])
    selected, hp, ap, owners = [], [], [], []
    retained_ids = {a['ball_id'] for a in retained}
    for ball in teams:
        h, a = divmod(trace[ball['remaining'] - 1], LANES)
        if {h, a} == set(ball['contacts']):
            selected.append(_commitment(ball, a, h, locked=ball['id'] in retained_ids))
            hp.append((ball['remaining'], h, ball['id']))
            ap.append((ball['remaining'], a, ball['id']))
    if selected:
        earliest = min(by_id[a['ball_id']]['remaining'] for a in selected)
        for assignment in selected:
            if by_id[assignment['ball_id']]['remaining'] == earliest:
                assignment['locked'] = True
    for ball in smalls:
        h, a = divmod(trace[ball['remaining'] - 1], LANES)
        owner = 2 if a == ball['contacts'][0] else 1 if h == ball['contacts'][0] else 0
        owners.append(owner)
        if owner:
            (ap if owner == 2 else hp).append((ball['remaining'], ball['contacts'][0], ball['id']))
    return teams, smalls, selected, sorted(hp), sorted(ap), owners


def _visible_arrival_score(balls, human_lane, ai_lane):
    """Score one simultaneous visible arrival, exactly as physical settlement."""
    return sum(WEIGHTS[ball['kind']] for ball in balls
               if ({human_lane, ai_lane} == set(ball['contacts']) if ball['kind'] == 'cooperative'
                   else ball['contacts'][0] in (human_lane, ai_lane)))


def _arrival_capacity_with_commitments(obs, arrival, ai_first_action=None):
    """Single-arrival reachable bound, not a whole-task optimum or forecast.

    Only already visible balls and previously retained feasible commitments are
    considered. Forcing the next AI move is an explanation-only comparison.
    """
    teams = [ball for ball in obs['balls'] if ball['kind'] == 'cooperative']
    by_id = {ball['id']:ball for ball in teams}
    retained = _retained_commitments(obs, teams)
    human_deadlines = [(by_id[a['ball_id']]['remaining'], a['human_contact'], a['ball_id']) for a in retained]
    ai_deadlines = [(by_id[a['ball_id']]['remaining'], a['ai_contact'], a['ball_id']) for a in retained]
    if ai_first_action is not None:
        next_lane = obs['ai']['x'] + _DELTAS[ai_first_action]
        if not 0 <= next_lane < LANES:
            return None
        ai_deadlines.append((1, next_lane, 'next-action'))
    arriving = [ball for ball in obs['balls'] if ball['remaining'] == arrival]
    feasible_human = [lane for lane in range(LANES)
                      if _route(obs['human']['x'], human_deadlines + [(arrival,lane,'arrival')]) is not None]
    feasible_ai = [lane for lane in range(LANES)
                   if _route(obs['ai']['x'], ai_deadlines + [(arrival,lane,'arrival')]) is not None]
    if not feasible_human or not feasible_ai:
        return None
    return max(_visible_arrival_score(arriving,human,ai) for human in feasible_human for ai in feasible_ai)


def _planned_arrival_payoffs(obs, assignments, planned_catches):
    """Conditional scores for the actual selected plan; never read _schedule."""
    by_id = {ball['id']:ball for ball in obs['balls']}
    selected = {}
    for assignment in assignments:
        ball = by_id[assignment['ball_id']]
        row = selected.setdefault(ball['remaining'], {'arrival_turns':ball['remaining'], 'human_lane':None,
                                  'ai_lane':None, 'catches':[], 'team_assignments':[]})
        row['human_lane'],row['ai_lane'] = assignment['human_contact'],assignment['ai_contact']
        row['team_assignments'].append(deepcopy(assignment))
        row['catches'].append({'ball_id':ball['id'], 'kind':'cooperative', 'actor':'team', 'points':WEIGHTS['cooperative']})
    for catch in planned_catches:
        ball = by_id[catch['ball_id']]
        row = selected.setdefault(ball['remaining'], {'arrival_turns':ball['remaining'], 'human_lane':None,
                                  'ai_lane':None, 'catches':[], 'team_assignments':[]})
        row[catch['actor'] + '_lane'] = ball['contacts'][0]
        row['catches'].append({'ball_id':ball['id'], 'kind':'ordinary', 'actor':catch['actor'], 'points':WEIGHTS['ordinary']})
    rows = []
    for arrival,row in sorted(selected.items()):
        row['catches'].sort(key=lambda catch:(catch['kind']=='cooperative',catch['actor']!='ai',catch['ball_id']))
        row['raw_points'] = sum(catch['points'] for catch in row['catches'])
        row['maximum_at_arrival'] = _arrival_capacity_with_commitments(obs,arrival)
        row['is_highest_at_arrival'] = row['maximum_at_arrival'] == row['raw_points']
        row['scope'] = 'currently_visible_balls_at_this_arrival_preserving_prior_feasible_commitments'
        rows.append(row)
    return rows


def _payoff_text(row, language='en', include_maximum=True):
    """A short conditional statement: placement, individually counted balls, sum."""
    en_parts,zh_parts = [],[]
    for actor in ('ai','human'):
        lane = row[actor+'_lane']
        if lane is not None:
            en_parts.append(f"{'I' if actor=='ai' else 'you'} reach lane {lane+1}")
            zh_parts.append(f"{'我' if actor=='ai' else '你'}到第{lane+1}道")
    owners_en = {'ai':'my', 'human':'your', 'team':'our'}
    owners_zh = {'ai':'我接', 'human':'你接', 'team':'合作接'}
    items_en = ' + '.join(f"{owners_en[catch['actor']]} {catch['ball_id']}({catch['points']})" for catch in row['catches'])
    items_zh = '＋'.join(f"{owners_zh[catch['actor']]}{catch['ball_id']}（{catch['points']}分）" for catch in row['catches'])
    # The actor's point is explicit even when the partner's small ball contributes.
    unit = 'turn' if row['arrival_turns'] == 1 else 'turns'
    en = f"If {' and '.join(en_parts)} in {row['arrival_turns']} {unit}, these visible catches give {items_en} = {row['raw_points']} points."
    zh = f"{row['arrival_turns']}回合后若{'、'.join(zh_parts)}，当前可见来球可同时得{items_zh}，合计{row['raw_points']}分。"
    if include_maximum and row['is_highest_at_arrival']:
        en += ' This is the highest reachable total for this arrival while keeping earlier assignments that remain reachable.'
        zh += '这是保留仍可达的原有承诺时，这一到达时刻双方可得的最高分。'
    return zh if language.startswith('zh') else en


def _decision_from_observation(obs):
    # Cache immutable serialized observations, returning isolated results.
    return deepcopy(_cached_decision(json.dumps(obs, sort_keys=True, separators=(',', ':'))))


@lru_cache(maxsize=2048)
def _cached_decision(key):
    obs = json.loads(key)
    if obs['terminal']:
        return {'action': 'wait', 'human_action': 'wait', 'reason_code': 'task_complete',
                'reason_en': 'This task has finished. There is no next action to execute.',
                'reason_zh': '本任务已经结束，没有下一步需要执行。', 'goal': 'task_complete',
                'memory': {}, 'alternatives': [], 'assignment_changed': False,
                'assignment_released': False, 'target_lane': None, 'assignments': [], 'planned_catches': [], 'action_alternatives': {}}
    teams, smalls, assignments, human_path, ai_path, owners = _plan(obs)
    previous = obs['policy_memory'].get('commitment')
    first = next((a for a in assignments if previous and a['ball_id'] == previous.get('ball_id')), None)
    if first is None:
        first = next((a for a in assignments if a['locked']), None)
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
    planned_catches = [{'ball_id': b['id'], 'actor': 'human' if owner == 1 else 'ai'} for b, owner in zip(smalls, owners) if owner]
    payoffs = _planned_arrival_payoffs(obs,assignments,planned_catches)
    en, zh = [], []
    goal_ball = next((b for b in obs['balls'] if b['id'] == goal), None)
    focused_payoff = next((row for row in payoffs if goal_ball and row['arrival_turns'] == goal_ball['remaining']),None)
    same_arrival_team = next((a for a in assignments if goal_ball and
                             next(b['remaining'] for b in teams if b['id']==a['ball_id']) == goal_ball['remaining']),None)
    if goal_ball:
        target_en,target_zh = f"lane {target+1} for {goal}",f"{goal}的第{target+1}道"
        if same_arrival_team:
            team = next(b for b in teams if b['id']==same_arrival_team['ball_id'])
            side = 'left' if target == min(team['contacts']) else 'right'
            target_en = f"{team['id']}'s {side} contact (lane {target+1})"
            target_zh = f"{team['id']}的{'左' if side=='left' else '右'}接点（第{target+1}道）"
        if action == 'wait':
            en.append(f"I will hold {target_en}.")
            zh.append(f"我会守住{target_zh}。")
        else:
            en.append(f"I will move {'left' if action=='left' else 'right'} toward {target_en}.")
            zh.append(f"我将{'左移' if action=='left' else '右移'}，前往{target_zh}。")
    else:
        en.append('I have no catch to cover in the current shared plan, so I will wait here.')
        zh.append('当前协作计划没有分配给我的接球任务，因此我会在这里等待。')
    if focused_payoff:
        en.append(_payoff_text(focused_payoff))
        zh.append(_payoff_text(focused_payoff,'zh'))
    nearest_assignment = next((a for a in assignments if a['ball_id'] == goal), assignments[0] if assignments else None)
    if nearest_assignment and not same_arrival_team:
        ball = next(b for b in teams if b['id'] == nearest_assignment['ball_id'])
        own, partner = nearest_assignment['ai_contact']+1,nearest_assignment['human_contact']+1
        en.append(f"For {ball['id']} in {ball['remaining']} turns, {'my side is' if nearest_assignment['locked'] else 'my tentative side is'} lane {own}; you need lane {partner}.")
        zh.append(f"对于{ball['remaining']}回合后的{ball['id']}，我{'负责' if nearest_assignment['locked'] else '暂定去'}第{own}道，你需要第{partner}道。")
    if switched:
        en[-1] += ' Our previous sides are no longer reachable.'
        zh[-1] += ' 原来的双方分工已经来不及。'
    tentative_changes = []
    for assignment in assignments:
        old = next((a for a in obs['policy_memory'].get('commitments', []) if a['ball_id'] == assignment['ball_id']), None)
        if old and not old.get('locked', False) and not _same_assignment(old, assignment):
            tentative_changes.append({'ball_id': assignment['ball_id'], 'previous_ai_contact': old['ai_contact'],
                                      'previous_human_contact': old['human_contact'], 'ai_contact': assignment['ai_contact'],
                                      'human_contact': assignment['human_contact'], 'basis': 'current_visible_balls_and_positions'})
    comparisons = []
    for ball in teams:
        for own, partner in (ball['contacts'], list(reversed(ball['contacts']))):
            own_distance, partner_distance = abs(obs['ai']['x'] - own), abs(obs['human']['x'] - partner)
            ok = own_distance <= ball['remaining'] and partner_distance <= ball['remaining']
            comparisons.append({'action': _move_towards(obs['ai']['x'], own), 'ball_id':ball['id'],
                'en': f"{ball['id']}: I need {own_distance} moves to lane {own + 1}; you need {partner_distance} to lane {partner + 1}. {ball['remaining']} turns remain; " + ('both can reach, before considering other balls.' if ok else 'at least one side is too far away.'),
                'zh': f"{ball['id']}：我到第{own + 1}道需{own_distance}步，你到第{partner + 1}道需{partner_distance}步。还剩{ball['remaining']}回合；" + ('只考虑这颗球时双方都来得及。' if ok else '至少一侧已经来不及。')})
    for ball in smalls:
        appointments = [(b['remaining'], a['ai_contact'], b['id']) for a in assignments for b in teams if a['ball_id'] == b['id']]
        appointments.append((ball['remaining'], ball['contacts'][0], ball['id']))
        safe = _route(obs['ai']['x'], appointments) is not None
        comparisons.append({'action': _move_towards(obs['ai']['x'], ball['contacts'][0]), 'ball_id':ball['id'],
            'en': f"{ball['id']} reaches lane {ball['contacts'][0] + 1} in {ball['remaining']} turns. " + ('I can include it without missing my planned team contacts.' if safe else 'Taking it would miss at least one planned team contact or its own deadline.'),
            'zh': f"{ball['id']}在{ball['remaining']}回合后到第{ball['contacts'][0] + 1}道。" + ('我能接它并赶上计划中的合作接球点。' if safe else '接它会错过至少一个计划中的合作接球点，或它本身就来不及。')})
    action_alternatives = {}
    for candidate in ('left', 'right', 'wait'):
        position = obs['ai']['x'] + _DELTAS[candidate]
        if not 0 <= position < LANES:
            en_alt, zh_alt = 'That move leaves the court and is unavailable.', '这个方向会离开球场，不能执行。'
        elif not goal_ball:
            en_alt, zh_alt = 'There is no selected visible catch that requires this move.', '没有已选定的可见接球任务需要这样移动。'
        else:
            gap, time = abs(position - target), goal_ball['remaining'] - 1
            preserves = all((lane == position if at == 1 else abs(lane - position) <= at - 1)
                            for at, lane, _ in ai_path)
            en_alt = f"After {candidate}, I would be in lane {position + 1}, {gap} moves from {goal}'s lane {target + 1}, with {time} turns left. " + ('This keeps the planned catch route reachable.' if preserves else 'This would miss a selected catch deadline.')
            zh_alt = f"如果{'左移' if candidate == 'left' else '右移' if candidate == 'right' else '等待'}，我会在第{position + 1}道，离{goal}所在第{target + 1}道还需{gap}步，剩{time}回合。" + ('计划中的接球路线仍然可达。' if preserves else '这会错过已选定的接球时限。')
        if focused_payoff and 0 <= position < LANES:
            possible = _arrival_capacity_with_commitments(obs,focused_payoff['arrival_turns'],candidate)
            if possible is None:
                en_alt += ' This first move cannot keep our earlier team-ball assignments.'
                zh_alt += '这样走第一步无法保持先前的合作球承诺。'
            else:
                unit = 'turn' if focused_payoff['arrival_turns'] == 1 else 'turns'
                en_alt += f" For the visible balls arriving in {focused_payoff['arrival_turns']} {unit} only, this first move allows at most {possible} points while keeping earlier assignments."
                zh_alt += f"只看{focused_payoff['arrival_turns']}回合后这批可见来球，先这样走且保持原承诺，最多可得{possible}分。"
        action_alternatives[candidate] = {'en':en_alt, 'zh':zh_alt}
    memory = {'commitments': deepcopy(assignments)} if assignments else {}
    if first:
        memory['commitment'] = deepcopy(first)
    return {'action': action, 'human_action': human_action, 'reason_code': code,
            'reason_en': ' '.join(en), 'reason_zh': ''.join(zh), 'goal': goal, 'memory': memory,
            'alternatives': comparisons, 'assignment_changed': switched, 'assignment_released': released,
            'target_lane': target, 'assignments': deepcopy(assignments), 'tentative_replans': tentative_changes, 'action_alternatives': action_alternatives,
            'planned_catches': planned_catches}


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
          'Up to six small balls and four team balls are visible. Several balls can arrive together; incompatible contacts force a choice. Replacements enter at the top, and the supply ends before the task limit.',
          'Task score is 100 times earned points divided by all scheduled ball points. Some simultaneous balls cannot all be caught, so 100 may be impossible. Reading, questions and replay do not advance the game.']
    zh = ['球场有九道、高十二格。A、D分别左移和右移一道，空格等待；每次操作推进一步。双方球拍可以重叠。',
          '双方球拍同时移动，随后小球下降两格、合作球下降一格；到达接球线的球立即结算。',
          '小球由任一球拍覆盖所在道即可得1分；双方同时覆盖仍然只得1分。',
          '合作球必须在同一步由两个球拍分别覆盖两个不同的接触位置，才能得3分。',
          '画面最多同时有六个小球、四个合作球。同一回合可能有多个球到达，不兼容的接触位置必须取舍。新球从顶部补入，结束前停止补球。',
          '任务分数为100乘实得分除以全部预定球的分值。部分同时到达的球无法全部接住，因此可能无法得到100分。阅读、提问和回放不推进游戏。']
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
    payoffs = _planned_arrival_payoffs(_observation(state),actual['assignments'],actual['planned_catches'])
    human_arrival = next((row for row in payoffs if row['human_lane'] is not None), None)
    if human_arrival is not None and not state['terminal']:
        lane, remaining = human_arrival['human_lane'], human_arrival['arrival_turns']
        selected = [catch['ball_id'] for catch in human_arrival['catches'] if catch['actor'] in ('human', 'team')]
        distance = abs(state['human']['x'] - lane)
        next_x = max(0, min(LANES - 1, state['human']['x'] + {'left':-1, 'right':1, 'wait':0}[actual['human_action']]))
        advised_reachable = abs(next_x - lane) <= remaining - 1
        wait_reachable = distance <= remaining - 1
        en = f"Your next selected catch is {' and '.join(selected)} at lane {lane + 1}, {distance} moves away and arriving in {remaining} turns. "
        zh = f"当前计划中你下一次接{'、'.join(selected)}，接点在第{lane + 1}道，相距{distance}步，{remaining}回合后到达。"
        if advised_reachable and not wait_reachable:
            en += f"Moving {'left' if actual['human_action']=='left' else 'right'} now keeps that contact reachable. Waiting now leaves only {remaining - 1} {'turn' if remaining == 2 else 'turns'} for {distance} {'move' if distance == 1 else 'moves'}, so you would miss this assigned contact."
            zh += f"现在{'左移' if actual['human_action']=='left' else '右移'}仍来得及；若先等待，之后只剩{remaining - 1}回合却还要移动{distance}步，就赶不上这个已分配接点。"
        else:
            en += f"Waiting now {'still leaves enough time to reach' if wait_reachable else 'would not leave enough time to reach'} this contact; this distance check alone does not establish a score advantage."
            zh += f"先等待后{'仍来得及' if wait_reachable else '来不及'}到这个接点；仅凭这项距离检查，不能断言得分更高。"
        evidence.append({'id':'human_catch_deadline', 'ball_ids':selected, 'contact_lane':lane+1,
                         'horizontal_distance':distance, 'arrival_turns':remaining,
                         'advised_action':actual['human_action'], 'advised_reachable':advised_reachable,
                         'wait_reachable':wait_reachable, 'en':en, 'zh':zh})
    for payoff in payoffs:
        evidence.append({'id': 'arrival_payoff:' + str(payoff['arrival_turns']),
                         'arrival_turns':payoff['arrival_turns'], 'raw_points':payoff['raw_points'],
                         'maximum_at_arrival':payoff['maximum_at_arrival'], 'is_highest_at_arrival':payoff['is_highest_at_arrival'],
                         'human_contact_lane':None if payoff['human_lane'] is None else payoff['human_lane']+1,
                         'ai_contact_lane':None if payoff['ai_lane'] is None else payoff['ai_lane']+1, 'scope':payoff['scope'],
                         'catches':deepcopy(payoff['catches']), 'en':_payoff_text(payoff), 'zh':_payoff_text(payoff,'zh')})
        for catch in payoff['catches']:
            evidence.append({'id':'ball_payoff:'+catch['ball_id'], 'ball_id':catch['ball_id'],
                             'arrival_turns':payoff['arrival_turns'], 'raw_points':payoff['raw_points'],
                             'en':_payoff_text(payoff), 'zh':_payoff_text(payoff,'zh')})
    balls_by_id = {ball['id']: ball for ball in state['balls']}
    assignments = {item['ball_id']: item for item in actual['assignments']}
    ordinary_owners = {item['ball_id']: item['actor'] for item in actual['planned_catches']}
    plan_rows = {}
    for ball in state['balls']:
        lanes = ', '.join(str(x + 1) for x in ball['contacts'])
        evidence.append({'id': 'ball:' + ball['id'],
            'en': f"{ball['id']}: {'team' if ball['kind'] == 'cooperative' else 'small'}, {WEIGHTS[ball['kind']]} points, lane(s) {lanes}, {ball['remaining']} turns away; falls {ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1)} cells each step.",
            'zh': f"{ball['id']}是{'合作' if ball['kind'] == 'cooperative' else '小'}球，值{WEIGHTS[ball['kind']]}分，将在{ball['remaining']}回合后到达第{lanes}道，每步下降{ball.get('vy', 2 if ball['kind'] == 'ordinary' else 1)}格。"})
        assignment = assignments.get(ball['id'])
        owner = ordinary_owners.get(ball['id'])
        if assignment:
            status = 'committed' if assignment['locked'] else 'tentative'
            en_plan = f"{ball['id']} is selected: I cover lane {assignment['ai_contact'] + 1}; you cover lane {assignment['human_contact'] + 1}, in {ball['remaining']} turns. "
            zh_plan = f"当前选择接{ball['id']}：{ball['remaining']}回合后，我覆盖第{assignment['ai_contact'] + 1}道，你覆盖第{assignment['human_contact'] + 1}道。"
            en_plan += 'These sides stay the same while both remain reachable.' if assignment['locked'] else 'This later plan can change.'
            zh_plan += '只要双方仍然来得及，这个分工就保持不变。' if assignment['locked'] else '这个后续计划可能调整。'
        elif owner:
            status = 'selected'
            en_plan = f"{ball['id']} is selected for {'me' if owner == 'ai' else 'you'} to catch at lane {ball['contacts'][0] + 1} in {ball['remaining']} turns."
            zh_plan = f"当前选择由{'我' if owner == 'ai' else '你'}在{ball['remaining']}回合后，于第{ball['contacts'][0] + 1}道接{ball['id']}。"
        else:
            status = 'not_selected'
            en_plan = f"{ball['id']} is not selected in the current catch plan."
            zh_plan = f"当前接球计划没有选择{ball['id']}。"
            competitors = [item for item in actual['assignments']
                           if balls_by_id[item['ball_id']]['remaining'] == ball['remaining']
                           and set(balls_by_id[item['ball_id']]['contacts']) != set(ball['contacts'])]
            if competitors:
                competing = competitors[0]
                other = balls_by_id[competing['ball_id']]
                qualifier_en = 'We are taking' if competing['locked'] else 'We currently plan to take'
                qualifier_zh = '我们选择接' if competing['locked'] else '我们暂定接'
                en_plan += f" {qualifier_en} {other['id']} at the same arrival: I cover lane {competing['ai_contact'] + 1}; you cover lane {competing['human_contact'] + 1}."
                zh_plan += f"同一时刻，{qualifier_zh}{other['id']}：我覆盖第{competing['ai_contact'] + 1}道，你覆盖第{competing['human_contact'] + 1}道。"
                if len(set(other['contacts']) | set(ball['contacts'])) > 2:
                    en_plan += ' We cannot catch both with two paddles.'
                    zh_plan += '两块球拍无法同时接住这两颗球。'
                if not competing['locked']:
                    en_plan += ' This later plan can change.'
                    zh_plan += '这个后续计划可能调整。'
        plan_row = {'id': 'ball_plan:' + ball['id'], 'ball_id': ball['id'],
                    'plan_status': status, 'en': en_plan, 'zh': zh_plan}
        plan_rows[ball['id']] = plan_row
        evidence.append(plan_row)
    for actor in ('human', 'ai'):
        identifier = actor + '_nearest_ball'
        if not state['balls']:
            evidence.append({'id': identifier, 'en': 'There are no visible balls to compare.', 'zh': '当前没有可见的球可供比较。'})
            continue
        position = state[actor]['x']
        candidates = [(min(abs(position - lane) for lane in ball['contacts']), ball['remaining'], ball['id'], ball)
                      for ball in state['balls']]
        distance, remaining, _, nearest = min(candidates, key=lambda item: item[:3])
        lane = min(nearest['contacts'], key=lambda contact: (abs(position - contact), contact))
        plan_status = plan_rows[nearest['id']]['plan_status']
        en_nearest = f"By horizontal lane distance, {'your' if actor == 'human' else 'my'} closest visible contact is {nearest['id']} at lane {lane + 1}: {distance} moves away, arriving in {remaining} turns. "
        zh_nearest = f"按横向道距离，离{'你' if actor == 'human' else '我'}最近的可见接触点是{nearest['id']}的第{lane + 1}道，相距{distance}步，{remaining}回合后到达。"
        if plan_status == 'not_selected':
            en_nearest += 'This ball is not selected in the current catch plan.'
            zh_nearest += '当前接球计划没有选择这颗球。'
        elif nearest['kind'] == 'ordinary':
            owner = ordinary_owners[nearest['id']]
            en_nearest += f"The current plan assigns it to {'me' if owner == 'ai' else 'you'}."
            zh_nearest += f"当前计划由{'我' if owner == 'ai' else '你'}接这颗球。"
        else:
            assigned = assignments[nearest['id']]
            en_nearest += f"Our {'committed' if assigned['locked'] else 'tentative'} sides are me at lane {assigned['ai_contact'] + 1}, you at lane {assigned['human_contact'] + 1}; both contacts are needed."
            zh_nearest += f"{'已承诺' if assigned['locked'] else '暂定'}分工是我在第{assigned['ai_contact'] + 1}道、你在第{assigned['human_contact'] + 1}道；双方都需要到位。"
        evidence.append({'id': identifier, 'ball_id': nearest['id'], 'contact_lane': lane + 1,
                         'horizontal_distance': distance, 'arrival_turns': remaining, 'plan_status': plan_status,
                         'en': en_nearest, 'zh': zh_nearest})
    for i, assignment in enumerate(actual['assignments']):
        evidence.append({'id': f'assignment:{i}', 'en': f"For {assignment['ball_id']}, the {'committed' if assignment['locked'] else 'tentative later'} plan has me at lane {assignment['ai_contact'] + 1} and you at lane {assignment['human_contact'] + 1}.",
                         'zh': f"关于{assignment['ball_id']}，{'当前承诺的' if assignment['locked'] else '暂定后续'}分工是我在第{assignment['ai_contact'] + 1}道，你在第{assignment['human_contact'] + 1}道。"})
    evidence.extend({'id': 'alternative_' + action, 'en': row['en'], 'zh': row['zh']} for action, row in actual.get('action_alternatives', {}).items())
    for i, row in enumerate(actual['alternatives']):
        plan_row = plan_rows.get(row.get('ball_id'))
        en_comparison, zh_comparison = row['en'], row['zh']
        if plan_row and plan_row['plan_status'] == 'not_selected':
            en_comparison = f"{row['ball_id']} is not selected; the following checks reachability only, not our current plan. " + en_comparison
            zh_comparison = f"当前没有选择{row['ball_id']}；下面只检查是否来得及，不是当前分工。" + zh_comparison
        elif plan_row and row.get('ball_id') in assignments:
            assigned = assignments[row['ball_id']]
            en_comparison = f"Our {'committed' if assigned['locked'] else 'tentative'} plan for {row['ball_id']} is me at lane {assigned['ai_contact'] + 1}, you at lane {assigned['human_contact'] + 1}. This checks one possible division: " + en_comparison
            zh_comparison = f"对于{row['ball_id']}，{'已承诺' if assigned['locked'] else '暂定'}分工是我在第{assigned['ai_contact'] + 1}道、你在第{assigned['human_contact'] + 1}道。下面检查一种可能分工：" + zh_comparison
        evidence.append({'id': f'comparison:{i}', 'en': en_comparison, 'zh': zh_comparison})
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
    # Use the actual seeded concurrent schedule, truncated to a finite 12 steps.
    entries = [e for e in _schedule(730100, 1) if e['arrival_turn'] <= 12]
    state = _new_state(entries, seed=0, task=1, human=2, ai=6)
    frames = [public_state(state)]
    while not state['terminal']:
        state = step(state, human_advisor(state))
        frames.append(public_state(state))
    captions = [
        (0, 'Six small balls and four team balls are visible. Several share an arrival time.', '场上有六个小球和四个合作球，其中多颗球会同时到达。'),
        (1, 'One input advances one step: small balls fall two cells, team balls one. Reading does not advance them.', '一次操作推进一步：小球下降两格，合作球下降一格。阅读不会让球继续下降。'),
        (2, 'Two small balls arrive together. Each covered contact earns 1 point; each missed contact earns 0.', '两颗小球同时到达。覆盖一颗的接触位置得1分，漏接则得0分。'),
        (4, 'The next pair arrives while replacement balls continue down from the top.', '下一对球到达，同时补入的球继续从顶部下降。'),
        (6, 'Two small balls and two team balls arrive together. The team balls require four distinct lanes, so two paddles must choose which team ball to catch.', '两颗小球和两颗合作球同时到达。两颗合作球需要覆盖四条不同的道，因此两块球拍必须选择接哪颗合作球。'),
        (12, 'The finite supply has ended. Every ball was scored once; these demonstration points do not count toward your tasks.', '演示的有限供球已经结束，每颗球只计分一次。这些分数不计入正式任务。')]
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
