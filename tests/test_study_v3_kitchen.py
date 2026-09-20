"""Independent mechanism boundaries and real fixed-controller trajectories.

Fixtures are synthetic engine unit states. No test/proxy is a human sample.
"""
from collections import deque
from copy import deepcopy
import json
import random
import unittest
from domains.kitchen import engine as e


def fixture():
    state = e.initial_state(1000, 2)
    state['max_turns'] = 1000
    state['orders'][0]['recipe'] = 'egg_tomato'
    state['orders'][1]['recipe'] = 'pepper_meat'
    for order in state['orders']:
        order['deadline'] = 1000
    return state


def item(ingredient='egg', stage='prepared', iid='test-egg', order='order1', pot=None):
    return {'id': iid, 'ingredient': ingredient, 'ingredients': [ingredient], 'recipe': e.INGREDIENT_RECIPE[ingredient], 'stage': stage,
            'prepare_progress': 0 if stage == 'raw' else e.PREPARE_TURNS[ingredient], 'components': [iid], 'order_id': order, 'pot_id': pot,
            'container': 'temporary_plate' if stage == 'cooked_protein' else None, 'was_buffered': False}


def dish(order='order1', pot='pot1'):
    result = item(stage='finished', order=order, pot=pot)
    result.update(ingredients=['egg', 'tomato'], components=['test-egg', 'test-tomato'], container='output_container', was_buffered=True)
    return result


def tick(state, human='wait', ai='wait', slot=None):
    return e.step(state, human, {'action': ai, 'memory': {}, 'slot': slot, 'reason_code': 'independent_fixture'})


def run(seed, task, human=None):
    state = e.initial_state(seed, task)
    frames, events = [deepcopy(state)], []
    while not state['terminal']:
        action = human(state) if human else e.human_advisor(state)
        state = e.step(state, action)
        frames.append(state)
        events.extend(state['events'])
    return frames, events


def ready_pot(state, phase='protein', ingredient='egg', age=0, pot_id='pot1'):
    pot = e._pot(state, pot_id)
    portion = dish(pot=pot_id) if phase == 'mix' else item(ingredient, 'cooked_vegetable' if phase == 'vegetable' else 'cooked_protein', pot=pot_id)
    pot.update(phase=phase, status='ready', item=portion, order_id='order1', recipe='egg_tomato', remaining=0, ready_age=age)
    return pot


class FacingAndInteraction(unittest.TestCase):
    def test_exact_map_and_connected_work_areas(self):
        expected = {'egg': (1, 1), 'tomato': (3, 1), 'meat': (1, 2), 'pepper': (3, 2), 'prep': (1, 3),
                    'human_buffer': (1, 4), 'plate': (1, 5), 'serve': (3, 5), 'handoff': (4, 3), 'trash': (4, 4),
                    'pot1': (7, 1), 'protein1': (7, 2), 'ai_raw': (7, 3), 'protein2': (7, 4), 'pot2': (7, 5)}
        self.assertEqual({key: (value['x'], value['y']) for key, value in e.STATION_BY_ID.items()}, expected)
        self.assertEqual(sum(x < 4 for x, y in e.FLOOR), 7)
        self.assertEqual(sum(x > 4 for x, y in e.FLOOR), 10)
        for station in e.STATIONS:
            actor = 'human' if station['x'] < 4 else 'ai'
            start = (2, 3) if actor == 'human' else (6, 3)
            route = e._route(start, 'right', station['id'])
            x, y, face = e._route_end(start, 'right', route)
            self.assertEqual(e._front({'x': x, 'y': y, 'facing': face})['id'], station['id'])

    def test_move_changes_facing_and_blocked_turn_costs_one(self):
        state = fixture()
        state['human'].update(x=2, y=1, facing='down')
        for action, station in [('left', 'egg'), ('right', 'tomato')]:
            nxt = tick(state, action)
            self.assertEqual((nxt['human']['x'], nxt['human']['y']), (2, 1))
            self.assertEqual(nxt['human']['facing'], action)
            self.assertEqual(nxt['turn'], 1)
            self.assertEqual(e._front(nxt['human'])['id'], station)
        nxt = tick(state, 'down')
        self.assertEqual((nxt['human']['x'], nxt['human']['y'], nxt['human']['facing']), (2, 2, 'down'))

    def test_only_facing_station_interacts_and_invalid_does_not_mutate(self):
        state = fixture()
        state['human'].update(x=2, y=1, facing='down')
        before = deepcopy(state)
        self.assertNotIn('interact', e.legal_actions(state))
        with self.assertRaises(ValueError): e.step(state, 'interact')
        self.assertEqual(before, state)
        for face, ingredient in [('left', 'egg'), ('right', 'tomato')]:
            state['human']['facing'] = face
            nxt = tick(state, 'interact')
            self.assertEqual(nxt['human']['holding']['ingredient'], ingredient)
            self.assertEqual(nxt['turn'], 1)

    def test_four_separate_cupboards_and_no_selection_menu(self):
        for ingredient, point, face in [('egg', (2, 1), 'left'), ('tomato', (2, 1), 'right'), ('meat', (2, 2), 'left'), ('pepper', (2, 2), 'right')]:
            with self.subTest(ingredient=ingredient):
                state = fixture()
                state['human'].update(x=point[0], y=point[1], facing=face)
                state = tick(state, 'interact')
                self.assertEqual(state['human']['holding']['ingredient'], ingredient)
                self.assertNotIn('interact', e.legal_actions(state))
                self.assertIn('full', e.interaction_label(state))

    def test_wait_advances_once_without_movement_or_facing_change(self):
        state = fixture(); before = deepcopy(state['human'])
        nxt = tick(state)
        self.assertEqual(before, nxt['human'])
        self.assertEqual(nxt['turn'], 1)
        self.assertEqual(nxt['metrics']['human_waits'], 1)

    def test_raw_requires_ingredient_specific_preparations(self):
        for ingredient, required in {'tomato':3,'pepper':3,'egg':4,'meat':5}.items():
            state = fixture(); state['human'].update(x=2, y=3, facing='left', holding=item(ingredient, 'raw', order=None))
            self.assertEqual('Whisk egg' if ingredient == 'egg' else 'Prepare ingredient', e.interaction_label(state))
            for progress in range(1,required+1):
                state=tick(state,'interact')
                self.assertEqual(state['human']['holding']['prepare_progress'],progress)
                self.assertEqual(state['human']['holding']['stage'],'prepared' if progress==required else 'raw')
            self.assertNotIn('interact',e.legal_actions(state))

    def test_role_boundaries(self):
        state = fixture()
        state['human'].update(x=6, y=1, facing='right', holding=item())
        self.assertNotIn('interact', e.legal_actions(state))
        state['ai'].update(x=2, y=1, facing='left')
        self.assertNotIn('interact', e.legal_actions(state, 'ai'))

    def test_full_counter_never_swaps_or_overwrites(self):
        state = fixture(); state['human'].update(x=3, y=3, facing='right', holding=item(iid='held'))
        state['handoff'] = item('tomato', iid='counter')
        before = deepcopy(state)
        self.assertNotIn('interact', e.legal_actions(state))
        self.assertIn('cannot be swapped', e.interaction_label(state))
        with self.assertRaises(ValueError): e.step(state, 'interact')
        self.assertEqual(before, state)

    def test_same_turn_handoff_conflict_is_simultaneous(self):
        state = fixture(); state['human'].update(x=3, y=3, facing='right'); state['ai'].update(x=5, y=3, facing='left')
        state['handoff'] = item()
        before = deepcopy(state['handoff'])
        nxt = tick(state, 'interact', 'interact')
        self.assertEqual(nxt['handoff'], before)
        self.assertIsNone(nxt['human']['holding']); self.assertIsNone(nxt['ai']['holding'])
        self.assertEqual(nxt['metrics']['handoff_conflicts'], 1)
        self.assertEqual(nxt['turn'], 1)

    def test_newly_placed_item_not_taken_same_turn(self):
        state = fixture(); state['human'].update(x=3, y=3, facing='right', holding=item()); state['ai'].update(x=5, y=3, facing='left')
        self.assertNotIn('interact', e.legal_actions(state, 'ai'))
        nxt = tick(state, 'interact')
        self.assertIsNone(nxt['ai']['holding']); self.assertIsNotNone(nxt['handoff'])
        later = tick(nxt, ai='interact')
        self.assertEqual(later['ai']['holding']['id'], 'test-egg')

    def test_physical_interaction_text_names_item_and_counter(self):
        state = fixture(); state['ai'].update(x=5, y=3, facing='left'); state['handoff'] = item()
        label = e.interaction_label(state, 'ai')
        self.assertIn('prepared egg', label); self.assertIn('handoff', label)
        self.assertIn('备好的鸡蛋', e.interaction_label(state, 'ai', 'zh'))


class RecipeStateMachine(unittest.TestCase):
    def test_exact_protein_cooking_times_loading_excluded(self):
        for ingredient, order, duration in [('egg', 'order1', 4), ('meat', 'order2', 6)]:
            state = fixture(); state['ai'].update(x=6, y=1, facing='right', holding=item(ingredient, order=order))
            state = tick(state, ai='interact')
            self.assertEqual(state['pots'][0]['remaining'], duration)
            for remaining in range(duration - 1, 0, -1):
                state = tick(state); self.assertEqual(state['pots'][0]['remaining'], remaining)
                self.assertEqual(state['pots'][0]['status'], 'cooking')
            state = tick(state); self.assertEqual(state['pots'][0]['status'], 'ready')
            self.assertEqual(state['pots'][0]['ready_age'], 0)

    def test_vegetable_cannot_start_first(self):
        state = fixture(); state['ai'].update(x=6, y=1, facing='right', holding=item('tomato'))
        self.assertNotIn('interact', e.legal_actions(state, 'ai'))
        self.assertEqual(e.decide(state)['reason_code'], 'store_early_vegetable')

    def test_protein_requires_real_temporary_counter_visit(self):
        state = fixture(); ready_pot(state); state['ai'].update(x=6, y=1, facing='right')
        state = tick(state, ai='interact')
        self.assertEqual(state['ai']['holding']['container'], 'temporary_plate')
        self.assertEqual(state['pots'][0]['phase'], 'await_protein_store')
        self.assertFalse(state['ai']['holding']['was_buffered'])
        state['buffers']['ai_raw'][0] = item('tomato', iid='tomato')
        self.assertEqual(e._loadable(state, state['buffers']['ai_raw'][0]), [])
        state['ai'].update(x=6, y=2, facing='right')
        state = tick(state, ai='interact')
        self.assertIsNone(state['ai']['holding'])
        self.assertTrue(state['buffers']['protein']['pot1']['was_buffered'])
        self.assertEqual(state['pots'][0]['phase'], 'await_vegetable')
        self.assertEqual(e._loadable(state, state['buffers']['ai_raw'][0])[0]['id'], 'pot1')

    def test_dedicated_counter_rejects_other_pan_protein(self):
        state = fixture(); state['ai'].update(x=6, y=4, facing='right', holding=item(stage='cooked_protein', pot='pot1'))
        self.assertNotIn('interact', e.legal_actions(state, 'ai'))

    def test_vegetable_requires_matching_order_and_same_pan(self):
        state = fixture(); state['pots'][0].update(phase='await_vegetable', order_id='order1', recipe='egg_tomato')
        state['ai'].update(x=6, y=1, facing='right', holding=item('pepper', order='order2'))
        self.assertNotIn('interact', e.legal_actions(state, 'ai'))

    def test_returned_protein_starts_exact_two_turn_mix_preserves_ids(self):
        state = fixture(); ready_pot(state, 'vegetable', 'tomato')
        state['pots'][0]['item']['id'] = 'vegetable'; state['pots'][0]['item']['components'] = ['vegetable']
        protein = item(stage='cooked_protein', pot='pot1'); protein['was_buffered'] = True
        state['ai'].update(x=6, y=1, facing='right', holding=protein)
        state = tick(state, ai='interact')
        pot = state['pots'][0]
        self.assertEqual((pot['phase'], pot['status'], pot['remaining']), ('mix', 'cooking', 2))
        self.assertEqual(pot['item']['components'], ['test-egg', 'vegetable'])
        self.assertIsNone(state['ai']['holding'])
        state = tick(state); self.assertEqual(state['pots'][0]['remaining'], 1)
        state = tick(state); self.assertEqual(state['pots'][0]['status'], 'ready')
        state = tick(state, ai='interact')
        self.assertEqual(state['ai']['holding']['stage'], 'finished')
        self.assertEqual(state['ai']['holding']['container'], 'output_container')
        self.assertEqual(state['pots'][0]['phase'], 'idle')

    def test_unbuffered_or_wrong_order_protein_cannot_combine(self):
        for buffered, order in [(False, 'order1'), (True, 'order2')]:
            state = fixture(); ready_pot(state, 'vegetable', 'tomato')
            protein = item(stage='cooked_protein', pot='pot1', order=order); protein['was_buffered'] = buffered
            state['ai'].update(x=6, y=1, facing='right', holding=protein)
            self.assertNotIn('interact', e.legal_actions(state, 'ai'))

    def test_eight_ready_turn_boundary_including_safe_last_action(self):
        state = fixture(); ready_pot(state); state['ai'].update(x=6, y=1, facing='right')
        for age in range(1, 8):
            state = tick(state); self.assertEqual(state['pots'][0]['ready_age'], age)
            self.assertEqual(state['pots'][0]['status'], 'ready')
        saved = tick(state, ai='interact'); burned = tick(state)
        self.assertEqual(saved['metrics']['burnt'], 0)
        self.assertEqual(burned['pots'][0]['status'], 'burnt')
        self.assertEqual(burned['metrics']['burnt'], 1)
        cleared = tick(burned, ai='interact')
        self.assertEqual(cleared['pots'][0]['phase'], 'idle')
        self.assertEqual(cleared['metrics']['waste'], 0)
        self.assertEqual(cleared['ai']['holding']['stage'], 'waste')
        self.assertEqual(cleared['ai']['holding']['components'], ['test-egg'])

    def test_removed_food_does_not_burn(self):
        state = fixture(); state['ai']['holding'] = item(stage='cooked_protein', pot='pot1')
        for _ in range(20): state = tick(state)
        self.assertEqual(state['ai']['holding']['stage'], 'cooked_protein')
        self.assertEqual(state['metrics']['burnt'], 0)

    def test_two_raw_slots_are_distinct_and_preserve_other_slot(self):
        state = fixture(); state['ai'].update(x=6, y=3, facing='right', holding=item('tomato', iid='held'))
        state['buffers']['ai_raw'][0] = item('pepper', iid='slot0', order='order2')
        nxt = tick(state, ai='interact', slot=1)
        self.assertEqual([it['id'] for it in nxt['buffers']['ai_raw']], ['slot0', 'held'])
        nxt = tick(nxt, ai='interact', slot=0)
        self.assertEqual(nxt['ai']['holding']['id'], 'slot0')
        self.assertEqual(nxt['buffers']['ai_raw'][1]['id'], 'held')

    def test_output_needs_distinct_human_final_plating(self):
        state = fixture(); state['human'].update(x=2, y=5, facing='right', holding=dish())
        self.assertNotIn('interact', e.legal_actions(state))
        self.assertIn('serving plate', e.interaction_label(state))
        state['human']['facing'] = 'left'; state = tick(state, 'interact')
        self.assertEqual(state['human']['holding']['container'], 'serving_plate')
        state['human']['facing'] = 'right'; state = tick(state, 'interact')
        self.assertEqual(state['metrics']['completed_orders'], 1)
        self.assertIsNone(state['human']['holding'])

    def test_partial_or_mismatched_dish_is_not_served(self):
        state = fixture(); bad = dish(); bad.update(stage='plated', ingredients=['egg'], components=['test-egg'])
        state['human'].update(x=2, y=5, facing='right', holding=bad)
        nxt = tick(state, 'interact')
        self.assertEqual(nxt['metrics']['completed_orders'], 0)
        self.assertIsNotNone(nxt['human']['holding'])
        self.assertIn('serve_rejected', [event['type'] for event in nxt['events']])

    def test_bound_order_cannot_be_stolen_by_same_recipe_order(self):
        state = fixture(); state['orders'][1]['recipe'] = 'egg_tomato'
        food = dish(order='order2'); food.update(stage='plated', container='serving_plate')
        state['human'].update(x=2, y=5, facing='right', holding=food)
        nxt = tick(state, 'interact')
        self.assertEqual(nxt['orders'][0]['status'], 'pending')
        self.assertEqual(nxt['orders'][1]['status'], 'completed')

    def test_serve_on_deadline_precedes_expiry(self):
        state = fixture(); state['orders'][0]['deadline'] = 1
        food = dish(); food.update(stage='plated', container='serving_plate')
        state['human'].update(x=2, y=5, facing='right', holding=food)
        self.assertEqual(tick(state, 'interact')['orders'][0]['status'], 'completed')
        self.assertEqual(tick(state)['orders'][0]['status'], 'expired')

    def test_discard_recovery_frees_hand_with_explicit_penalty(self):
        state = fixture(); state['human'].update(x=3, y=4, facing='right', holding=item())
        nxt = tick(state, 'interact')
        self.assertIsNone(nxt['human']['holding']); self.assertEqual(nxt['metrics']['waste'], 1)
        self.assertEqual(e.score(nxt)['task_score'], e.score(state)['task_score'] - 4)

    def test_clearing_burnt_vegetable_keeps_stored_protein_and_job(self):
        state = fixture(); ready_pot(state, 'vegetable', 'tomato', age=7)
        protein = item(stage='cooked_protein', pot='pot1'); protein['was_buffered'] = True
        state['buffers']['protein']['pot1'] = protein; state['ai'].update(x=6, y=1, facing='right')
        state = tick(state); state = tick(state, ai='interact')
        self.assertEqual(state['pots'][0]['phase'], 'await_vegetable')
        self.assertEqual(state['buffers']['protein']['pot1']['id'], protein['id'])


class ControllerAndEvidence(unittest.TestCase):
    def test_actual_wrong_vegetable_first_is_buffered_then_recovered(self):
        state = e.initial_state(1000, 2)
        # Drive only human actions to fetch, chop and deliver tomato first.
        targets = [('tomato', 1), ('prep', 3), ('handoff', 1)]
        events = []
        for target, count in targets:
            for _ in range(count):
                while e._front(state['human']) is None or e._front(state['human'])['id'] != target:
                    state = e.step(state, e._approach(state, 'human', target)); events.extend(state['events'])
                state = e.step(state, 'interact'); events.extend(state['events'])
        while not any(it for it in state['buffers']['ai_raw']):
            state = e.step(state, 'wait'); events.extend(state['events'])
        self.assertEqual(state['buffers']['ai_raw'][0]['ingredient'], 'tomato')
        original_id = state['buffers']['ai_raw'][0]['id']
        self.assertEqual(state['metrics']['wrong_order_buffered'], 1)
        while not state['terminal']:
            state = e.step(state, e.human_advisor(state)); events.extend(state['events'])
        self.assertTrue(any(event['type'] == 'served' and original_id in event['item']['components'] for event in events))
        self.assertEqual(state['metrics']['completed_orders'], 5)

    def test_raw_delivery_recovery_for_all_four_ingredients_from_initial_state(self):
        for ingredient in e.LABELS:
            with self.subTest(ingredient=ingredient):
                state = e.initial_state(1000, 2)
                events = []
                for target in (ingredient, 'handoff'):
                    while e._front(state['human']) is None or e._front(state['human'])['id'] != target:
                        state = e.step(state, e._approach(state, 'human', target)); events.extend(state['events'])
                    state = e.step(state, 'interact'); events.extend(state['events'])
                wrong_id = state['handoff']['id']
                self.assertEqual(e.decide(state)['reason_code'], 'await_preparation')
                self.assertEqual(e.human_advisor(state), 'interact')
                evidence = {row['id']: row for row in e.facts(state)}
                self.assertIn('Take it back', evidence['handoff_recovery']['en'])
                self.assertIn(ingredient, evidence['next_input_ingredient']['en'])
                while not state['terminal']:
                    state = e.step(state, e.human_advisor(state)); events.extend(state['events'])
                self.assertEqual(state['metrics']['completed_orders'], 5)
                self.assertTrue(any(event['type'] in ('served','waste') and wrong_id in event['item']['components'] for event in events))
                if ingredient != 'pepper': self.assertEqual(state['metrics']['waste'],0)

    def test_invalid_handoff_recovery_with_full_human_hand_and_storage(self):
        state = fixture()
        state['human'].update(x=3, y=3, facing='right', holding=item('meat', iid='held', order='order2'))
        state['handoff'] = item('egg', 'raw', iid='raw-blocker', order=None)
        state['buffers']['human'] = item('tomato', iid='buffered', order=None)
        before_counter = deepcopy(state['handoff'])
        self.assertEqual(e.human_advisor(state), 'down')
        while state['human']['holding'] is not None:
            state = e.step(state, e.human_advisor(state))
        self.assertEqual(state['handoff'], before_counter)
        self.assertEqual(state['buffers']['human']['id'], 'buffered')
        while state['human']['holding'] is None:
            state = e.step(state, e.human_advisor(state))
        self.assertIsNone(state['handoff'])
        self.assertEqual(state['human']['holding']['id'], 'raw-blocker')
        self.assertEqual(e._next_input_portion(state)['ingredient'], 'egg')

    def test_expired_handoff_is_retrieved_and_discarded(self):
        state = fixture(); state['orders'][0]['status'] = 'expired'
        state['handoff'] = item(order='order1')
        state['human'].update(x=3, y=3, facing='right')
        self.assertEqual(e.human_advisor(state), 'interact')
        state = e.step(state, 'interact')
        while state['human']['holding'] is not None:
            self.assertNotEqual(e.human_advisor(state), 'discard')
            state = e.step(state, e.human_advisor(state))
        self.assertEqual(next(event for event in state['events'] if event['type']=='waste')['station'], 'trash')
        self.assertIsNone(state['human']['holding']); self.assertIsNone(state['handoff'])

    def test_ingredient_locations_and_pan_contents_are_container_accurate(self):
        frames, _ = run(1000, 2)
        ready = next(frame for frame in frames if frame['pots'][0]['status'] == 'ready' and frame['pots'][0]['phase'] == 'protein')
        rows = {row['id']: row for row in e.facts(ready)}
        self.assertIn("stove 1's pan", rows['ingredient_location_egg']['en'])
        self.assertNotIn('temporary plate', rows['pot1_contents']['en'])
        self.assertIn('tomato and egg stir-fry', rows['pot1_contents']['en'])
        self.assertIn('original portions: 1', rows['pot1_contents']['en'])
        stored = next(frame for frame in frames if frame['buffers']['protein']['pot1'])
        rows = {row['id']: row for row in e.facts(stored)}
        self.assertIn('temporary plate counter', rows['ingredient_location_egg']['en'])
        mixed = next(frame for frame in frames if frame['pots'][0]['phase'] == 'mix')
        rows = {row['id']: row for row in e.facts(mixed)}
        self.assertIn('ingredients: egg, tomato', rows['pot1_contents']['en'])
        self.assertIn("stove 1's pan", rows['ingredient_location_egg']['en'])
        self.assertIn("stove 1's pan", rows['ingredient_location_tomato']['en'])

    def test_ingredient_goal_is_distinct_from_next_movement_and_ignores_future(self):
        state = e.initial_state(1000, 3)
        before = deepcopy(state)
        rows = {row['id']: row for row in e.facts(state)}
        self.assertIn('next input', rows['next_input_ingredient']['en'])
        self.assertIn('tomato and egg stir-fry', rows['next_input_ingredient']['en'])
        self.assertNotIn('order3', rows['missing_ingredients']['en'])
        self.assertNotEqual(e.human_advisor(state), 'interact')
        self.assertEqual(before, state)
        changed = deepcopy(state); changed['_future_orders'] = []
        self.assertEqual(rows, {row['id']:row for row in e.facts(changed)})

    def test_terminal_evidence_has_no_executable_wait_or_input_advice(self):
        frames, _ = run(1000, 1)
        rows = {row['id']: row for row in e.facts(frames[-1])}
        self.assertIn('no executable next action', rows['ai_next_action']['en'])
        self.assertIn('no next ingredient', rows['next_input_ingredient']['en'])
        self.assertNotIn('human_available_option', rows)

    def test_rescue_frees_hand_then_commits_to_ready_pan(self):
        state = fixture(); ready_pot(state); state['ai'].update(x=6, y=3, facing='right', holding=item('tomato', iid='tomato'))
        d = e.decide(state); self.assertEqual(d['reason_code'], 'free_for_rescue')
        self.assertEqual(d['action'], 'interact')
        state = e.step(state, 'wait')
        self.assertEqual(state['policy_memory']['rescue_pot'], 'pot1')
        self.assertIsNone(state['ai']['holding'])
        for _ in range(4): state = e.step(state, 'wait')
        self.assertEqual(state['ai']['holding']['stage'], 'cooked_protein')
        self.assertEqual(state['metrics']['burnt'], 0)

    def test_actual_route_distance_includes_facing(self):
        state = fixture(); state['ai'].update(x=6, y=1, facing='up')
        self.assertEqual(e._distance(state, 'ai', 'pot1'), 1)
        state['ai']['facing'] = 'right'
        self.assertEqual(e._distance(state, 'ai', 'pot1'), 0)
        route = e._route((6, 1), 'right', 'pot2')
        self.assertEqual(len(route) + 1, 6)

    def test_impossible_rescue_is_not_claimed_possible(self):
        state = fixture(); ready_pot(state, age=7); state['ai'].update(x=5, y=5, facing='down')
        self.assertIn('cannot reach', e.decide(state)['reason_en'])

    def test_queries_are_pure_and_hidden_future_is_not_observed(self):
        state = e.initial_state(1000, 3); original = deepcopy(state)
        decision, facts, public, advice = e.decide(state), e.facts(state), e.public_state(state), e.human_advisor(state)
        self.assertEqual(original, state)
        changed = deepcopy(state); changed['_future_orders'] = [{'id': 'secret', 'recipe': 'pepper_meat', 'arrival': 1, 'deadline': 9}]
        self.assertEqual(e.decide(changed), decision); self.assertEqual(e.facts(changed), facts)
        self.assertEqual(e.public_state(changed), public); self.assertEqual(e.human_advisor(changed), advice)
        for forbidden in ('policy_memory', '_future_orders', 'seed', 'reason_en', 'reason_code', 'human_advisor'):
            self.assertNotIn(forbidden, public)

    def test_counterfactual_branch_does_not_change_source_or_decision(self):
        state = fixture(); decision = e.decide(state); before, frozen = deepcopy(state), deepcopy(decision)
        one, two = e.step(state, 'up', decision), e.step(state, 'wait', decision)
        self.assertEqual(state, before); self.assertEqual(decision, frozen)
        self.assertNotEqual(one['human'], two['human'])
        self.assertEqual(one['ai'], two['ai'])

    def test_actual_interaction_decision_and_evidence_are_specific(self):
        state = fixture(); state['handoff'] = item(); state['ai'].update(x=5, y=3, facing='left')
        decision = e.decide(state)
        self.assertEqual(decision['action'], 'interact')
        self.assertIn('prepared egg', decision['action_label_en'])
        fact = next(row for row in e.facts(state) if row['id'] == 'ai_next_action')
        self.assertIn('prepared egg', fact['en'])

    def test_terminal_state_rejects_all_new_actions(self):
        state = fixture(); state['terminal'] = True
        self.assertEqual(e.legal_actions(state), [])
        for action in ('wait', 'left', 'interact'):
            with self.assertRaises(ValueError): e.step(state, action)
        self.assertEqual(e.decide(state)['reason_code'], 'terminal')

    def test_unusable_raw_and_expired_items_have_recovery_paths(self):
        state = fixture(); state['orders'][0]['status'] = 'expired'
        state['ai']['holding'] = item()
        self.assertEqual(e.decide(state)['goal'], 'trash')
        while state['ai']['holding'] is not None:
            self.assertNotEqual(e.decide(state)['action'], 'discard')
            state = e.step(state, 'wait')
        self.assertEqual(next(event for event in state['events'] if event['type']=='waste')['station'], 'trash')
        self.assertIsNone(state['ai']['holding'])


class PhysicalDisposalAndVisibility(unittest.TestCase):
    def test_no_remote_discard_for_either_actor(self):
        state = fixture()
        state['human']['holding'] = item(iid='human-food')
        state['ai']['holding'] = item(iid='ai-food')
        before = deepcopy(state)
        for actor in ('human', 'ai'):
            self.assertNotIn('discard', e.legal_actions(state, actor))
        with self.assertRaises(ValueError): tick(state, human='discard')
        with self.assertRaises(ValueError): tick(state, ai='discard')
        self.assertEqual(state, before)

    def test_trash_is_reachable_from_both_disjoint_work_areas(self):
        self.assertEqual(e.STATION_BY_ID['trash']['kind'], 'trash')
        self.assertNotIn([4, 4], e.WALLS)
        self.assertNotIn((4, 4), e.FLOOR)
        for actor, start in [('human', (2, 3)), ('ai', (6, 3))]:
            route = e._route(start, 'right', 'trash')
            x, y, facing = e._route_end(start, 'right', route)
            self.assertEqual((x, y, facing), (3, 4, 'right') if actor == 'human' else (5, 4, 'left'))

    def test_trash_requires_held_item_and_correct_facing(self):
        for actor, x, face in [('human', 3, 'right'), ('ai', 5, 'left')]:
            state = fixture(); state[actor].update(x=x, y=4, facing=face)
            self.assertNotIn('interact', e.legal_actions(state, actor))
            state[actor]['holding'] = item(iid=actor+'-food')
            self.assertIn('interact', e.legal_actions(state, actor))
            state[actor]['facing'] = 'up'
            self.assertNotIn('interact', e.legal_actions(state, actor))

    def test_simultaneous_trash_disposal_is_explicit_and_conserves_components(self):
        state = fixture()
        state['human'].update(x=3, y=4, facing='right', holding=item(iid='human-food'))
        state['ai'].update(x=5, y=4, facing='left', holding=item(iid='ai-food'))
        state = tick(state, 'interact', 'interact')
        self.assertIsNone(state['human']['holding']); self.assertIsNone(state['ai']['holding'])
        waste = [event for event in state['events'] if event['type'] == 'waste']
        self.assertEqual(len(waste), 2)
        self.assertEqual({event['station'] for event in waste}, {'trash'})
        self.assertEqual(state['metrics']['waste'], 2)
        self.assertEqual(state['metrics']['handoff_conflicts'], 0)

    def test_burnt_pan_contents_move_to_hand_before_actual_bin_disposal(self):
        state = fixture(); ready_pot(state, age=7); state['ai'].update(x=6, y=1, facing='right')
        state = tick(state)
        before = e._component_inventory(state)
        state = tick(state, ai='interact')
        self.assertEqual(e._component_inventory(state), before)
        self.assertEqual(state['ai']['holding']['stage'], 'waste')
        self.assertEqual(state['metrics']['waste'], 0)
        seen_positions = []
        while state['ai']['holding']:
            seen_positions.append(e._pos(state['ai']))
            decision = e.decide(state)
            self.assertEqual(decision['goal'], 'trash')
            state = e.step(state, 'wait')
        self.assertGreater(len(set(seen_positions)), 1)
        self.assertEqual(state['metrics']['waste'], 1)
        self.assertEqual(next(event for event in state['events'] if event['type']=='waste')['station'], 'trash')
        self.assertEqual(e._pos(state['ai']), (5, 4))

    def test_real_cooked_dish_expiry_regression_stays_visible_and_identical(self):
        # Previous engine: this identical legal trajectory discarded item1
        # remotely at turn 134 after order1's turn-130 expiry.
        state = e.initial_state(1000, 2)
        while not (state['ai']['holding'] and state['ai']['holding']['stage'] == 'finished'):
            state = e.step(state, e.human_advisor(state))
        self.assertLess(state['turn'], 100)
        cooked = deepcopy(state['ai']['holding'])
        self.assertEqual(cooked['id'], 'item1')
        self.assertEqual(len(cooked['components']),2)
        for _ in range(150):
            state = e.step(state, 'wait')
            self.assertFalse(any(event['type'] == 'waste' and event['item']['id'] == cooked['id'] for event in state['events']))
            for component in cooked['components']: self.assertEqual(e._component_inventory(state)[component],1)
        self.assertEqual(state['orders'][0]['status'], 'expired')
        public = e.public_state(state)
        visible = [public['human']['holding'], public['ai']['holding'], public['handoff'], public['buffers']['human'],
                   *public['buffers']['ai_raw'], *public['buffers']['protein'].values(), *(pan['item'] for pan in public['pots'])]
        saved = next(food for food in visible if food and food['id'] == cooked['id'])
        self.assertEqual(saved, cooked)

    def test_expiry_during_final_mixing_preserves_complete_dish(self):
        state = fixture(); state['orders'][0]['deadline'] = 1
        food = dish(); food.update(stage='mixing', container=None)
        state['pots'][0].update(phase='mix', status='cooking', remaining=2, ready_age=0,
                               recipe='egg_tomato', order_id='order1', item=food)
        state['ai'].update(x=6, y=1, facing='right')
        before = e._component_inventory(state)
        for _ in range(3): state = e.step(state, 'wait')
        self.assertEqual(state['orders'][0]['status'], 'expired')
        self.assertEqual(state['ai']['holding']['stage'], 'finished')
        self.assertEqual(state['ai']['holding']['components'], ['test-egg', 'test-tomato'])
        self.assertEqual(e._component_inventory(state), before)
        self.assertEqual(e.decide(state)['reason_code'], 'deliver_expired_finished')
        while state['handoff'] is None: state = e.step(state, 'wait')
        self.assertEqual(state['handoff']['stage'], 'finished')
        self.assertEqual(state['metrics']['waste'], 0)

    def test_expired_finished_dish_stays_held_when_serving_is_rejected(self):
        state = fixture(); state['orders'][0]['status'] = 'expired'
        food = dish(); food.update(stage='plated', container='serving_plate')
        state['human'].update(x=2, y=5, facing='right', holding=food)
        before = deepcopy(food)
        state = tick(state, 'interact')
        self.assertEqual(state['human']['holding'], before)
        self.assertEqual(state['metrics']['completed_orders'], 0)
        self.assertIn('serve_rejected', [event['type'] for event in state['events']])

    def test_preparation_progress_is_public_exact_and_immutable(self):
        state = fixture(); state['human'].update(x=2, y=3, facing='left', holding=item('egg', 'raw', order=None))
        for progress in range(5):
            before = deepcopy(state)
            public = e.public_state(state)
            self.assertEqual(state, before)
            prep = public['human']['preparation']
            self.assertEqual((prep['completed'], prep['remaining'], prep['required']), (progress, 4-progress, 4))
            self.assertEqual(prep['ready'], progress == 4)
            self.assertTrue(prep['at_station'])
            if progress < 4: state = tick(state, 'interact')

    def test_preparation_progress_follows_item_through_storage(self):
        state = fixture(); state['human'].update(x=2, y=3, facing='left', holding=item('egg', 'raw', order=None))
        state = tick(state, 'interact')
        state['human'].update(x=2, y=4, facing='left')
        state = tick(state, 'interact')
        self.assertIsNone(e.public_state(state)['human']['preparation'])
        self.assertEqual(state['buffers']['human']['prepare_progress'], 1)
        state = tick(state, 'interact')
        self.assertEqual(e.public_state(state)['human']['preparation']['remaining'], 3)

    def test_actual_cooking_dishes_and_held_output_are_public(self):
        frames, _ = run(1000, 2)
        both = next(frame for frame in frames if all(p['phase'] != 'idle' for p in frame['pots']))
        jobs = e.public_state(both)['ai']['current_cooking']
        self.assertEqual([job['recipe'] for job in jobs], [pot['recipe'] for pot in both['pots']])
        self.assertEqual(len(jobs),2)
        self.assertTrue(all(job['recipe_label_en'] and job['recipe_label_zh'] for job in jobs))
        held = next(frame for frame in frames if frame['ai']['holding'] and frame['ai']['holding']['stage'] == 'finished')
        jobs = e.public_state(held)['ai']['current_cooking']
        output = next(job for job in jobs if job['location'] == 'held')
        self.assertEqual(output['item_id'], held['ai']['holding']['id'])
        self.assertEqual(output['phase'], 'finished')
        self.assertEqual(output['remaining'], 0)

    def test_conservation_guard_rejects_silent_removal_and_remote_waste_event(self):
        before = fixture(); before['human']['holding'] = item()
        after = deepcopy(before); after['human']['holding'] = None
        with self.assertRaisesRegex(AssertionError, 'conserved'): e._verify_food_conservation(before, after)
        after['events'] = [{'type':'waste','station':'serve','item':deepcopy(before['human']['holding'])}]
        with self.assertRaisesRegex(AssertionError, 'trash-bin'): e._verify_food_conservation(before, after)


class KitchenFeasibility(unittest.TestCase):
    def test_all_frozen_scenarios_with_actual_fixed_ai(self):
        cfg = e._configuration()
        for seed in cfg['development_seeds'] + cfg['held_out_seeds']:
            for task in (1, 2, 3):
                with self.subTest(seed=seed, task=task):
                    frames, events = run(seed, task)
                    final = frames[-1]
                    self.assertEqual(final['metrics']['completed_orders'],5)
                    self.assertEqual(e.score(final)['task_score'],500-final['turn'])
                    self.assertLessEqual(final['turn'],cfg['task_budgets'][str(task)])
                    self.assertEqual(final['metrics']['burnt'], 0)
                    self.assertGreater(final['metrics']['parallel_recipe_turns'], 0)
                    self.assertEqual(final['metrics']['parallel_cooking_turns'], 0)
                    self.assertEqual({event['pot'] for event in events if event['type'] == 'pot_loaded'}, {'pot1', 'pot2'})
                    served = [event['item'] for event in events if event['type'] == 'served']
                    component_ids = [identifier for food in served for identifier in food['components']]
                    taken = [event['item']['id'] for event in events if event['type'] == 'ingredient_taken']
                    self.assertEqual(sorted(component_ids), sorted(taken))
                    self.assertEqual(len(component_ids), len(set(component_ids)))
                    for food in served:
                        self.assertEqual(food['container'], 'serving_plate')
                        self.assertEqual(len(food['components']), 2)

    def test_deterministic_replay_and_save_resume(self):
        initial = e.initial_state(1000, 2); state = deepcopy(initial)
        actions, decisions = [], []
        for _ in range(90):
            actions.append(e.human_advisor(state)); decisions.append(e.decide(state))
            state = e.step(state, actions[-1], decisions[-1])
        restored = json.loads(json.dumps(initial))
        for action, decision in zip(actions, decisions): restored = e.step(restored, action, decision)
        self.assertEqual(restored, state)
        restored = json.loads(json.dumps(restored))
        self.assertEqual(e.step(restored, e.human_advisor(restored)), e.step(state, e.human_advisor(state)))

    def test_random_legal_human_actions_do_not_break_controller(self):
        for seed in range(5):
            rng = random.Random(seed); state = e.initial_state(1000 + seed, 2)
            while not state['terminal']:
                before = deepcopy(state)
                state = e.step(state, rng.choice(e.legal_actions(state)))
                self.assertEqual(state['turn'], before['turn'] + 1)
                ids = [component for item_ in e._all_items(state) if item_ for component in item_['components']]
                self.assertEqual(len(ids), len(set(ids)))

    def test_demo_has_six_real_captions_and_complete_recipe(self):
        demo = e.demonstration()
        self.assertEqual(len(demo['captions']), 6)
        self.assertEqual([frame['turn'] for frame in demo['frames']], list(range(len(demo['frames']))))
        types = {event['type'] for frame in demo['frames'] for event in frame['events']}
        self.assertTrue({'prepared', 'components_combined', 'plated', 'served'} <= types)
        self.assertTrue(any(event['type'] == 'item_placed' and event.get('station', '').startswith('protein') for frame in demo['frames'] for event in frame['events']))
        self.assertGreater(demo['frames'][-1]['score']['raw_score'], 0)

    def test_comprehension_choices_match_independent_mechanisms(self):
        en, zh = e.comprehension(), e.comprehension('zh')
        self.assertEqual([row['answer'] for row in en], [0, 1, 0])
        self.assertEqual([row['answer'] for row in zh], [0, 1, 0])
        state = fixture(); state['ai']['holding'] = item('tomato')
        self.assertEqual(e.decide(state)['reason_code'], 'store_early_vegetable')
        state = fixture(); state['orders'][0]['recipe'] = 'pepper_meat'; ready_pot(state, 'vegetable', 'pepper')
        state['pots'][0]['recipe'] = 'pepper_meat'
        protein = item('meat', 'cooked_protein', pot='pot1'); protein['was_buffered'] = True
        state['buffers']['protein']['pot1'] = protein
        self.assertEqual(e.decide(state)['reason_code'], 'fetch_protein')


if __name__ == '__main__': unittest.main()
