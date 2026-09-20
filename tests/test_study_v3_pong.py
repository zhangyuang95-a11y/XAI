"""Independent physical and protocol checks, never human-effect evidence."""
from copy import deepcopy
import itertools
import json
import random
import statistics

import pytest

from domains.pong import turnbased as pong


def scenario(*, human=2, ai=6, contacts=(2, 6), turns=4, ordinary=None, small_turns=None):
    balls = [pong._ball('fixture-team', 'cooperative', list(contacts), turns)]
    if ordinary is not None:
        balls.append(pong._ball('fixture-small', 'ordinary', [ordinary], small_turns or turns))
    return pong._new_state([balls], seed=999, task=2, human=human, ai=ai)


def play(state, policy=pong.human_advisor):
    while not state['terminal']:
        state = pong.step(state, policy(state))
    return state


@pytest.mark.parametrize('task,turns,small,team', [(1,60,60,20),(2,90,90,30),(3,90,90,30)])
def test_finite_schedule_and_public_geometry(task, turns, small, team):
    state = pong.initial_state(730100, task)
    assert (state['max_turns'], state['height'], state['lanes']) == (turns,12,9)
    assert len(state['_schedule']) == small + team
    assert state['total_possible_points'] == small + 3 * team
    assert sorted(b['remaining'] for b in state['balls'] if b['kind'] == 'ordinary') == [2,2,4,4,6,6]
    assert sorted(b['remaining'] for b in state['balls'] if b['kind'] == 'cooperative') == [6,6,12,12]
    for entry in state['_schedule']:
        ball = entry['ball']
        assert entry['arrival_turn'] <= turns
        assert entry['arrival_turn'] == entry['spawn_turn'] + ball['remaining']
        assert ball['y'] + ball['vy'] * ball['remaining'] == 12
        assert 0 <= ball['y'] <= 12
        assert ball['vy'] == (2 if ball['kind'] == 'ordinary' else 1)
        if entry['spawn_turn']:
            assert ball['y'] == 0
        assert entry['arrival_turn'] % 2 == 0
    public = pong.public_state(state)
    assert {'human','ai','height','lanes','balls','score'} <= public.keys()
    for ball in public['balls']:
        assert set(ball) == {'id','kind','contacts','remaining','initial_remaining','y','vy'}
    serialized = json.dumps(public)
    for private in ('policy_memory','_schedule','"seed"','reason_en','alternatives','assignment_switches'):
        assert private not in serialized
    assert len(pong.rules()) == len(pong.rules('zh')) == 6
    with pytest.raises(ValueError):
        pong.initial_state(1,4)


def test_speed_and_first_spawn_boundary():
    state = pong.initial_state(730100,2)
    before = {b['id']:b for b in pong.public_state(state)['balls']}
    nxt = pong.step(state,'wait')
    after = {b['id']:b for b in pong.public_state(nxt)['balls']}
    assert before.keys() == after.keys()
    for identifier in before:
        assert after[identifier]['y'] - before[identifier]['y'] == before[identifier]['vy']
        assert after[identifier]['remaining'] == before[identifier]['remaining'] - 1
    final = pong.step(nxt,'wait')
    fresh = {b['id']:b for b in final['balls']}
    assert set(before) - set(fresh) == {'s1','s2'}
    assert set(fresh) - set(before) == {'s7','s8'}
    assert all(fresh[name]['y'] == 0 and fresh[name]['remaining'] == 6 for name in ('s7','s8'))
    assert final['events'][-1]['type'] == 'balls_spawned'
    assert set(final['events'][-1]['ball_ids']) == {'s7','s8'}


def test_simultaneous_move_then_contact_settlement_scores_once():
    state = scenario(human=1,ai=7,turns=1,ordinary=2)
    nxt = pong.step(state,'right')
    assert (nxt['human']['x'],nxt['ai']['x']) == (2,6)
    assert nxt['raw_score'] == 4
    assert [e['points'] for e in nxt['events']] == [3,1]
    assert nxt['terminal'] and not nxt['balls']
    with pytest.raises(ValueError):
        pong.step(nxt,'wait')


def test_two_distinct_team_contacts_and_no_double_ordinary_score():
    missed = pong.step(scenario(human=2,ai=2,turns=1),'wait')
    assert missed['raw_score'] == 0 and missed['metrics']['misses'] == 1
    overlapping = pong.step(scenario(human=2,ai=2,turns=1,ordinary=2),'wait')
    assert overlapping['raw_score'] == 1
    assert overlapping['metrics']['ordinary_caught'] == 1
    assert overlapping['metrics']['cooperative_caught'] == 0


def test_boundary_and_stale_decision_rejected():
    state = scenario(human=0)
    assert pong.legal_actions(state) == ['wait','right']
    for action in ('left','teleport'):
        with pytest.raises(ValueError):
            pong.step(state,action)
    forged = pong.decide(state)
    forged['action'] = 'left' if forged['action'] != 'left' else 'right'
    with pytest.raises(ValueError):
        pong.step(state,'wait',forged)


def test_ai_cannot_see_unsubmitted_human_action():
    state = scenario(human=4,ai=4)
    decision = pong.decide(state)
    assert len({pong.step(state,a,decision)['ai']['x'] for a in pong.legal_actions(state)}) == 1


def test_locked_first_assignment_survives_human_motion():
    state = scenario(human=4,ai=4)
    first = pong.decide(state)
    assert first['memory']['commitment']['ai_contact'] == 6
    nxt = pong.step(state,'left')
    following = pong.decide(nxt)
    assert following['memory']['commitment'] == first['memory']['commitment']
    assert not following['assignment_changed']


def test_two_team_deadlines_are_jointly_reachable_and_later_is_tentative():
    balls = [pong._ball('early','cooperative',[2,6],2),pong._ball('later','cooperative',[1,7],4),pong._ball('small','ordinary',[5],1)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    decision = pong.decide(state)
    assert len(decision['assignments']) == 2
    assert [a['locked'] for a in decision['assignments']] == [True,False]
    first,later = decision['assignments']
    assert abs(later['ai_contact']-first['ai_contact']) <= 2
    assert abs(later['human_contact']-first['human_contact']) <= 2
    assert len(decision['reason_en']) < 300
    assert 'possible later' in next(f['en'] for f in pong.facts(state) if f['id'] == 'assignment:1')
    result = play(state)
    assert result['raw_score'] == 7


def test_unreachable_partner_does_not_change_already_selected_ai_side():
    state = scenario(human=6,ai=4,contacts=(3,6),turns=2)
    state['policy_memory'] = {'commitment':{'ball_id':'fixture-team','ai_contact':6,'human_contact':3}}
    decision = pong.decide(state)
    assert not decision['assignment_changed']
    assert decision['memory']['commitment']['ai_contact'] == 6
    assert play(state,lambda _:'wait')['raw_score'] == 0


def test_ordinary_detour_is_rejected_if_return_cannot_meet_deadline():
    state = scenario(ordinary=4,small_turns=3,turns=4)
    decision = pong.decide(state)
    assert decision['action'] == 'wait'
    assert play(state,lambda _:'wait')['raw_score'] == 3
    chasing = deepcopy(state)
    for action in ('right','right','wait','left'):
        chasing = pong.step(chasing,action)
    assert chasing['raw_score'] == 1
    assert pong.wave_feasibility(state)['reachable_points'] == 3


def test_safe_small_route_and_no_job_wait_are_truthful():
    state = scenario(ordinary=5,small_turns=2)
    decision = pong.decide(state)
    assert decision['reason_code'] == 'small_then_return'
    assert play(state,lambda _:'wait')['raw_score'] == 4
    no_job = pong.decide(scenario(human=0,ai=8,contacts=(3,5),turns=1))
    assert no_job['action'] == 'wait' and no_job['goal'] == 'hold_position'
    assert 'wait' in no_job['reason_en']


def test_rolling_pool_bound_and_final_drain():
    state = pong.initial_state(730100,2)
    seen = set()
    for turn in range(90):
        assert sum(b['kind']=='ordinary' for b in state['balls']) <= 6
        assert sum(b['kind']=='cooperative' for b in state['balls']) <= 4
        if turn < 84:
            assert len(state['balls']) == 10
        state = pong.step(state,pong.human_advisor(state))
        for event in state['events']:
            if event['type'] in ('caught','missed'):
                assert event['ball_id'] not in seen
                seen.add(event['ball_id'])
    assert state['terminal'] and not state['balls']
    assert len(seen) == 120
    assert 0 < state['raw_score'] <= 120
    assert state['total_possible_points'] == 180


def test_future_schedule_mutations_cannot_change_current_views_or_decisions():
    first = pong.initial_state(730103,2)
    second = deepcopy(first)
    second['seed'] = -999999
    for entry in second['_schedule']:
        if entry['spawn_turn'] > 0:
            entry['ball']['contacts'] = [8-x for x in entry['ball']['contacts']]
            entry['ball']['id'] = 'SECRET-UNSEEN-'+entry['ball']['id']
            entry['ball']['remaining'] = 99
    assert pong.decide(first) == pong.decide(second)
    assert pong.facts(first) == pong.facts(second)
    assert pong.human_advisor(first) == pong.human_advisor(second)
    assert pong.public_state(first) == pong.public_state(second)
    assert 'SECRET' not in json.dumps(pong.facts(second))


def test_observers_do_not_mutate_state_or_memory():
    state = scenario()
    original = deepcopy(state)
    decision = pong.decide(state)
    pong.facts(state,decision); pong.public_state(state); pong.wave_feasibility(state); pong.human_advisor(state)
    assert state == original
    assert not state['policy_memory'] and pong.step(state,'wait')['policy_memory']


def test_schedule_and_ball_physics_do_not_adapt_to_actions_or_score():
    first = pong.initial_state(730108,2)
    second = deepcopy(first)
    for _ in range(30):
        first = pong.step(first,pong.human_advisor(first))
        second = pong.step(second,'wait')
        assert first['_schedule'] == second['_schedule']
        assert first['balls'] == second['balls']
        assert first['turn'] == second['turn']


def test_all_wait_terminates_and_random_replay_is_exact():
    result = play(pong.initial_state(730104,2),lambda _:'wait')
    assert result['turn'] == 90 and not result['balls']
    assert 0 <= pong.score(result)['task_score'] <= 100
    first = pong.initial_state(730117,3)
    second = deepcopy(first)
    rng = random.Random(112)
    while not first['terminal']:
        action = rng.choice(pong.legal_actions(first))
        first,second = pong.step(first,action),pong.step(second,action)
        assert first == second


def test_short_fixed_ai_search_matches_independent_action_enumeration():
    state = scenario(human=3,ai=6,contacts=(2,6),turns=4,ordinary=4,small_turns=3)
    scores = []
    for actions in itertools.product(('wait','left','right'),repeat=4):
        candidate = deepcopy(state)
        try:
            for action in actions:
                candidate = pong.step(candidate,action)
        except ValueError:
            continue
        scores.append(candidate['raw_score'])
    assert pong.wave_feasibility(state)['reachable_points'] == max(scores)
    assert pong.global_reachable_upper_bound(state)['reachable_points'] == max(scores)
    with pytest.raises(ValueError):
        pong.global_reachable_upper_bound(pong.initial_state(730100,2))


@pytest.mark.parametrize('split',('development_seeds','heldout_seeds'))
@pytest.mark.parametrize('task',(1,2,3))
def test_frozen_seed_rollouts_respect_conflict_capacity_and_keep_real_misses(split,task):
    seeds = pong.CONFIG[split]
    assert len(seeds) == len(set(seeds)) == 24
    assert set(pong.CONFIG['development_seeds']).isdisjoint(pong.CONFIG['heldout_seeds'])
    scores,waits = [],[]
    for seed in seeds:
        result = play(pong.initial_state(seed,task))
        scores.append(pong.score(result)['task_score'])
        waits.append(result['metrics']['waiting_turns']/result['turn'])
        # Each simultaneous pair has four distinct lanes: at most one team ball.
        assert result['metrics']['cooperative_caught'] * 2 == result['metrics']['cooperative_total']
        assert result['raw_score'] <= result['turn'] * 4 // 3
        assert result['metrics']['misses'] >= result['metrics']['cooperative_total'] // 2
        assert result['metrics']['holding_turns'] + result['metrics']['no_job_waiting_turns'] == result['metrics']['waiting_turns']
    # Real flow should remain playable, not pretend incompatible points exist.
    assert min(scores) >= 50 and max(scores) <= 200/3 + 1e-6
    assert all(0 <= wait <= 1 for wait in waits)


def test_demo_captions_match_actual_recorded_events():
    demo = pong.demonstration()
    assert len(demo['captions']) == 6
    frames = demo['frames']
    assert [f['turn'] for f in frames] == list(range(13))
    assert len(frames[0]['balls']) == 10
    for turn in (2,4):
        events = [e for e in frames[turn]['events'] if e.get('kind') == 'ordinary']
        assert len(events) == 2
    for turn in (6,12):
        events = [e for e in frames[turn]['events'] if e.get('kind') == 'cooperative']
        assert len(events) == 2
        assert sorted(e['points'] for e in events) == [0,3]
        assert len({lane for e in events for lane in e['contacts']}) == 4
    assert frames[-1]['terminal']
    assert all('reason' not in json.dumps(f) and 'policy_memory' not in json.dumps(f) for f in frames)


def test_comprehension_answers_follow_physics_and_reachability():
    assert 12-3*2 == 6 and 12-3 == 9
    assert pong.step(scenario(human=2,ai=2,turns=1),'wait')['raw_score'] == 0
    assert abs(4-6) > 1
    assert [q['answer'] for q in pong.comprehension()] == [0,1,2]
    assert [q['answer'] for q in pong.comprehension('zh')] == [0,1,2]


def test_evidence_reports_actual_distances_assignments_and_speed_without_jargon():
    state = scenario(human=1,ai=7,turns=3,ordinary=4,small_turns=2)
    evidence = pong.facts(state)
    assert any('3 turns' in row['en'] for row in evidence)
    assert any('lane 8' in row['en'] for row in evidence)
    assert any('2 cells each step' in row['en'] for row in evidence)
    text = ' '.join(row[lang] for row in evidence for lang in ('en','zh'))
    for term in ('NN','embedding','checkpoint','logit','reward shaping','神经网络'):
        assert term not in text


def test_score_is_unmodified_fraction_and_waiting_breakdown_adds_up():
    result = play(scenario(ordinary=4,small_turns=3),lambda _:'wait')
    scored = pong.score(result)
    assert scored['raw_score'] == 3
    assert scored['metrics']['total_possible_points'] == 4
    assert scored['task_score'] == 75
    assert result['metrics']['waiting_turns'] == result['metrics']['holding_turns'] + result['metrics']['no_job_waiting_turns']


def test_simultaneous_team_conflict_chooses_reachable_second_id_and_preserves_it():
    balls = [pong._ball('first','cooperative',[1,6],2), pong._ball('second','cooperative',[2,7],2),
             pong._ball('small-left','ordinary',[1],2), pong._ball('small-right','ordinary',[7],2)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    decision = pong.decide(state)
    assert len(decision['assignments']) == 1
    assert decision['memory']['commitment']['ball_id'] == 'second'
    after = pong.step(state,'wait')
    assert pong.decide(after)['memory']['commitment']['ball_id'] == 'second'
    result = pong.step(after,pong.human_advisor(after))
    assert result['raw_score'] == 4
    assert result['metrics']['cooperative_caught'] == 1
    assert result['metrics']['ordinary_caught'] == 1


def test_same_contacts_allow_two_simultaneous_team_scores_but_not_double_small_credit():
    balls = [pong._ball('a','cooperative',[2,6],1),pong._ball('b','cooperative',[2,6],1),pong._ball('s','ordinary',[2],1)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    assert len(pong.decide(state)['assignments']) == 2
    result = pong.step(state,'wait')
    assert result['raw_score'] == 7
    assert result['metrics']['cooperative_caught'] == 2
    assert result['metrics']['ordinary_caught'] == 1


def test_three_simultaneous_small_contacts_can_score_at_most_two():
    balls = [pong._ball(f's{x}','ordinary',[x],1) for x in (1,4,7)]
    state = pong._new_state([balls],seed=1,task=2,human=0,ai=8)
    result = pong.step(state,pong.human_advisor(state))
    assert result['raw_score'] == 2
    assert result['metrics']['misses'] == 1


def test_planner_maximizes_real_points_not_artificial_team_count_priority():
    balls = [pong._ball('team','cooperative',[3,6],1)] + [pong._ball(f's{i}','ordinary',[2],1) for i in range(4)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    result = pong.step(state,pong.human_advisor(state))
    assert result['raw_score'] == 4
    assert result['metrics']['cooperative_caught'] == 0


def test_independent_exhaustive_two_step_actions_match_concurrent_plan():
    balls = [pong._ball('a','cooperative',[1,6],2),pong._ball('b','cooperative',[2,7],2),
             pong._ball('c','ordinary',[1],2),pong._ball('d','ordinary',[7],2)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    best = 0
    for human_moves in itertools.product((-1,0,1),repeat=2):
        for ai_moves in itertools.product((-1,0,1),repeat=2):
            h,a = 2+sum(human_moves),6+sum(ai_moves)
            if not (0 <= h < 9 and 0 <= a < 9):
                continue
            earned = sum(3 if {h,a} == set(b['contacts']) else 0 for b in balls if b['kind']=='cooperative')
            earned += sum(b['contacts'][0] in (h,a) for b in balls if b['kind']=='ordinary')
            best = max(best,earned)
    assert play(state)['raw_score'] == best == 4


def test_generated_mixed_arrivals_have_four_distinct_team_contacts_and_capacity_bound():
    state = pong.initial_state(730100,2)
    by_time = {}
    for entry in state['_schedule']:
        by_time.setdefault(entry['arrival_turn'],[]).append(entry['ball'])
    ceiling = 0
    for at, balls in by_time.items():
        small = [b for b in balls if b['kind']=='ordinary']
        teams = [b for b in balls if b['kind']=='cooperative']
        assert len(small) == 2
        assert len(teams) == (2 if at % 6 == 0 else 0)
        if teams:
            assert len({x for b in teams for x in b['contacts']}) == 4
        capacity = max(sum((3 if {h,a} == set(b['contacts']) else 0) if b['kind']=='cooperative' else int(b['contacts'][0] in (h,a)) for b in balls)
                       for h in range(9) for a in range(9))
        assert capacity == (4 if teams else 2)
        ceiling += capacity
    assert ceiling == 120 < state['total_possible_points'] == 180


def test_why_not_direction_facts_are_short_and_grounded_in_current_deadline():
    state = scenario(human=1,ai=7,turns=1)
    decision = pong.decide(state)
    facts = {f['id']:f for f in pong.facts(state)}
    assert 'alternative_left' in facts and 'alternative_right' in facts and 'alternative_wait' in facts
    assert 'too far to catch' in facts['alternative_wait']['en']
    assert 'still leaves time' in facts['alternative_left']['en']
    assert len(decision['reason_en']) < 300
    assert 'lane 7' in decision['reason_en']


def production_t3_assignment_snapshot():
    """Fixed-seed v6 rollout; v5 exported snapshots remain historical evidence."""
    state = pong.initial_state(731100,2)
    for _ in range(8):
        state = pong.step(state,pong.human_advisor(state))
    return state


def test_fixed_seed_unselected_ball_names_actual_selected_competitor():
    state = production_t3_assignment_snapshot()
    before = deepcopy(state)
    decision = pong.decide(state)
    facts = {row['id']:row for row in pong.facts(state,decision)}
    t3,t4,t6 = (facts['ball_plan:'+name] for name in ('t3','t4','t6'))
    assert t4['plan_status'] == 'not_selected'
    assert 't4 is not selected' in t4['en']
    assert 'taking t3 at the same arrival' in t4['en']
    assert 'I cover lane 3; you cover lane 9' in t4['en']
    assert 'cannot catch both' in t4['en']
    assert '当前接球计划没有选择t4' in t4['zh']
    assert '我覆盖第3道，你覆盖第9道' in t4['zh']
    assert t3['plan_status'] == 'committed'
    assert t6['plan_status'] == 'tentative' and 'later plan can change' in t6['en']
    for index,row in enumerate(decision['alternatives']):
        if row.get('ball_id') == 't4':
            checked=facts[f'comparison:{index}']
            assert checked['en'].startswith('t4 is not selected;')
            assert 'not our current plan' in checked['en']
            assert '不是当前分工' in checked['zh']
    assert state == before
    assert pong.decide(state) == decision


def test_production_nearest_contact_is_not_mistaken_for_the_chosen_ball():
    state = production_t3_assignment_snapshot()
    facts = {row['id']:row for row in pong.facts(state)}
    human,ai = facts['human_nearest_ball'],facts['ai_nearest_ball']
    # The v6 AI now chooses the overlapping small/team side on the left.
    assert (human['ball_id'],human['contact_lane'],human['horizontal_distance'],human['arrival_turns']) == ('s12',7,0,4)
    assert human['plan_status'] == 'not_selected'
    assert 'not selected' in human['en'] and '横向道距离' in human['zh']
    assert (ai['ball_id'],ai['contact_lane'],ai['horizontal_distance'],ai['arrival_turns']) == ('s11',3,0,4)
    assert ai['plan_status'] == 'selected' and 'assigns it to me' in ai['en']
    assert set(facts) >= {'ball_plan:'+b['id'] for b in state['balls']}


def test_nearest_uses_horizontal_distance_then_arrival_then_id_not_ai_assignment():
    balls=[pong._ball('far-soon','ordinary',[0],1),pong._ball('near-later','ordinary',[4],3),
           pong._ball('z-near-earlier','ordinary',[4],2),pong._ball('a-near-earlier','ordinary',[4],2)]
    state=pong._new_state([balls],seed=1,task=2,human=4,ai=8)
    nearest=next(row for row in pong.facts(state) if row['id']=='human_nearest_ball')
    assert nearest['ball_id']=='a-near-earlier'
    assert nearest['contact_lane']==5 and nearest['horizontal_distance']==0 and nearest['arrival_turns']==2
    team=scenario(human=2,ai=2,turns=6)
    decision=pong.decide(team)
    nearest=next(row for row in pong.facts(team) if row['id']=='ai_nearest_ball')
    assert nearest['contact_lane']==3 and nearest['horizontal_distance']==0
    assert decision['assignments'][0]['ai_contact']==6
    assert 'me at lane 7, you at lane 3' in nearest['en']
    assert 'both contacts are needed' in nearest['en']


def test_unselected_ball_without_selected_same_arrival_does_not_invent_a_competitor():
    state=scenario(human=0,ai=0,turns=1,contacts=(7,8))
    facts={row['id']:row for row in pong.facts(state)}
    plan=facts['ball_plan:fixture-team']
    assert plan['plan_status']=='not_selected'
    assert plan['en']=='fixture-team is not selected in the current catch plan.'
    assert 'same arrival' not in plan['en']
    finished=pong.step(state,'wait')
    finalfacts={row['id']:row for row in pong.facts(finished)}
    assert 'no visible balls' in finalfacts['human_nearest_ball']['en']
    assert 'no visible balls' in finalfacts['ai_nearest_ball']['en']



def test_real_turn76_explains_right_contact_when_moving_left_and_actual_four_points():
    state = pong.initial_state(731100,2)
    for _ in range(76):
        state = pong.step(state,pong.human_advisor(state))
    decision = pong.decide(state)
    facts = {row['id']:row for row in pong.facts(state,decision)}
    payoff = facts['arrival_payoff:2']
    assert decision['action'] == 'left'
    assert 'lane 5 to cover small ball s78 and my end of team ball t26 together in 2 turns' in decision['reason_en']
    assert '同时接2回合后的小球s78和大球t26的一端' in decision['reason_zh']
    assert 'You need lane 2' in decision['reason_en']
    assert 'points' not in decision['reason_en'] and '最高' not in decision['reason_zh']
    assert payoff['raw_points'] == payoff['maximum_at_arrival'] == 4
    assert {row['ball_id'] for row in payoff['catches']} == {'s78','t26'}
    assert 'ball_payoff:t25' not in facts  # A reachable but unselected ball is not an actual payoff.
    assert facts['ball_plan:t25']['plan_status'] == 'not_selected'
    assert 'too far to catch' in facts['alternative_wait']['en']
    initial_points = state['raw_score']
    for _ in range(2):
        state = pong.step(state,pong.human_advisor(state))
    assert state['raw_score'] - initial_points == 4
    assert {e['ball_id']:e['points'] for e in state['events'] if e['type'] == 'caught'} == {'s78':1,'t26':3}


def test_payoff_counts_both_actors_small_balls_and_team_once_each():
    balls = [pong._ball('team','cooperative',[2,6],1),
             pong._ball('small-ai','ordinary',[6],1),pong._ball('small-human','ordinary',[2],1)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    facts = {row['id']:row for row in pong.facts(state)}
    payoff = facts['arrival_payoff:1']
    assert payoff['raw_points'] == payoff['maximum_at_arrival'] == 5
    assert {row['ball_id']:row['actor'] for row in payoff['catches']} == {'team':'team','small-ai':'ai','small-human':'human'}
    assert 'my small-ai(1) + your small-human(1) + our team(3) = 5 points' in payoff['en']
    assert '你接small-human（1分）' in payoff['zh']
    assert pong.step(state,'wait')['raw_score'] == payoff['raw_points']
    assert facts['ball_payoff:team']['en'] == facts['ball_payoff:small-ai']['en'] == payoff['en']


def test_payoff_maximum_respects_prior_feasible_commitment_not_global_points():
    balls = [pong._ball('prior','cooperative',[2,6],2)] + [
        pong._ball(f'rich{i}','ordinary',[4],2) for i in range(4)]
    state = pong._new_state([balls],seed=1,task=2,human=2,ai=6)
    state['policy_memory'] = {'commitment':{'ball_id':'prior','ai_contact':6,'human_contact':2}}
    payoff = next(row for row in pong.facts(state) if row['id'] == 'arrival_payoff:2')
    assert payoff['raw_points'] == payoff['maximum_at_arrival'] == 3
    assert 'highest' not in payoff['en']
    assert payoff['scope'] == 'currently_visible_balls_at_this_arrival_preserving_selected_ai_route'
    unconstrained = deepcopy(state)
    unconstrained['policy_memory'] = {}
    assert pong._arrival_capacity_with_commitments(pong._observation(unconstrained),2) == 4
    assert play(state)['raw_score'] == 3
    assert play(unconstrained)['raw_score'] == 4


def test_selected_payoff_below_single_arrival_bound_never_claims_highest():
    state = pong.initial_state(730101,2)
    for _ in range(16):
        state = pong.step(state,pong.human_advisor(state))
    payoff = next(row for row in pong.facts(state) if row['id'] == 'arrival_payoff:6')
    assert payoff['raw_points'] == 1 < payoff['maximum_at_arrival'] == 2
    assert 'highest' not in payoff['en'] and '最高' not in payoff['zh']
    assert 'If I reach lane 5' in payoff['en']


def test_why_not_reports_conditional_single_arrival_bound_not_guaranteed_reward():
    state = scenario(human=1,ai=7,turns=1,ordinary=6)
    facts = {row['id']:row for row in pong.facts(state)}
    assert 'still leaves time' in facts['alternative_left']['en']
    assert 'too far to catch' in facts['alternative_wait']['en']
    assert 'points' not in facts['alternative_left']['en']
    # A saved commitment is a constraint; failure to preserve it is not proof of zero possible points.
    state['policy_memory'] = {'commitment':{'ball_id':'fixture-team','ai_contact':6,'human_contact':2}}
    facts = {row['id']:row for row in pong.facts(state)}
    assert 'too far to catch' in facts['alternative_wait']['en']
    assert 'at most 0 points' not in facts['alternative_wait']['en']


def test_payoff_evidence_preserves_visible_information_boundary_and_snapshot():
    state = pong.initial_state(731100,2)
    for _ in range(76):
        state = pong.step(state,pong.human_advisor(state))
    before = deepcopy(state)
    altered = deepcopy(state)
    for entry in altered['_schedule']:
        if entry['spawn_turn'] > state['turn']:
            entry['ball']['id'] = 'unseen-secret'
            entry['ball']['contacts'] = [0] if entry['ball']['kind'] == 'ordinary' else [0,8]
    assert pong.decide(state) == pong.decide(altered)
    assert pong.facts(state) == pong.facts(altered)
    assert 'unseen-secret' not in json.dumps(pong.facts(altered))
    assert state == before


def test_human_catch_deadline_uses_actual_selected_contacts_and_does_not_claim_slack_is_urgent():
    state = pong.initial_state(731100,2)
    for _ in range(8):
        state = pong.step(state,pong.human_advisor(state))
    row = next(f for f in pong.facts(state) if f['id']=='human_catch_deadline')
    assert row['ball_ids']==['s10'] and row['contact_lane']==9
    assert row['horizontal_distance']==row['arrival_turns']==2
    assert row['advised_action']=='right' and row['advised_reachable'] and not row['wait_reachable']
    # A single ordinary selected catch with slack does not prove waiting worse.
    slack = pong._new_state([[pong._ball('slack','ordinary',[0],4)]],seed=200,task=2,human=1,ai=8)
    row = next(f for f in pong.facts(slack) if f['id']=='human_catch_deadline')
    assert row['wait_reachable']
    assert 'score' not in row['en']
    assert 'Go to lane 1' in row['en']
