"""Storage, reversible delivery grace and actual two-order overlap boundaries."""
from copy import deepcopy
import unittest
from domains.kitchen import engine as e
from tests.test_study_v3_kitchen import fixture, item, dish, tick, run


class StorageClock(unittest.TestCase):
    def stored(self, station, ingredient, stage):
        state = fixture()
        actor = 'human' if station == 'human_buffer' else 'ai'
        state[actor].update(x=2 if actor == 'human' else 6, y=4 if actor == 'human' else 3,
                            facing='left' if actor == 'human' else 'right',
                            holding=item(ingredient, stage, order=None))
        state = tick(state, 'interact' if actor == 'human' else 'wait', 'interact' if actor == 'ai' else 'wait')
        return state, actor

    def test_four_raw_ingredients_retain_both_storage_place_limits(self):
        for station in ('ai_raw', 'human_buffer'):
            for ingredient in e.LABELS:
                for stage in ('raw',):
                    with self.subTest(station=station, ingredient=ingredient, stage=stage):
                        state, actor = self.stored(station, ingredient, stage)
                        at = state['turn']
                        get = lambda s: s['buffers']['human'] if actor == 'human' else s['buffers']['ai_raw'][0]
                        original = deepcopy(get(state))
                        for _ in range(10): state = tick(state)
                        self.assertEqual(state['turn'] - at, 10)
                        self.assertEqual(get(state)['stage'], stage)
                        state = tick(state)
                        self.assertEqual(get(state)['stage'], 'spoiled')
                        self.assertEqual(get(state)['id'], original['id'])
                        self.assertEqual(state['metrics']['spoiled'], 1)
                        self.assertEqual(state['metrics']['discard_penalty'], 0)
                        state = tick(state, 'interact' if actor == 'human' else 'wait', 'interact' if actor == 'ai' else 'wait')
                        self.assertEqual(state[actor]['holding']['stage'], 'spoiled')
                        while (e._front(state[actor]) or {}).get('id') != 'trash':
                            move = e._approach(state, actor, 'trash')
                            state = tick(state, move if actor == 'human' else 'wait', move if actor == 'ai' else 'wait')
                        state = tick(state, 'interact' if actor == 'human' else 'wait', 'interact' if actor == 'ai' else 'wait')
                        self.assertIsNone(state[actor]['holding'])
                        self.assertEqual(state['metrics']['discard_penalty'], 5)

    def test_pickup_on_first_overdue_turn_cannot_restore_freshness(self):
        state, actor = self.stored('human_buffer', 'tomato', 'raw')
        for _ in range(10): state = tick(state)
        state = tick(state, 'interact')
        self.assertEqual(state['human']['holding']['stage'], 'spoiled')
        self.assertEqual(state['human']['holding']['spoilage_reason'], 'storage_limit')

    def test_timely_pickup_stops_storage_clock_but_keeps_original_oxidation(self):
        state, _ = self.stored('human_buffer', 'egg', 'prepared')
        food = state['buffers']['human']; food.update(prepared_turn=0, expires_turn=20, fresh_until=20, freshness_basis='prepared')
        for _ in range(9): state = tick(state)
        state = tick(state, 'interact')
        self.assertEqual(state['human']['holding']['stage'], 'prepared')
        self.assertEqual(state['human']['holding']['fresh_until'], 20)
        while state['turn'] < 19: state = tick(state)
        self.assertEqual(state['human']['holding']['stage'], 'prepared')
        state = tick(state)
        self.assertEqual(state['human']['holding']['stage'], 'spoiled')
        self.assertEqual(state['human']['holding']['spoilage_reason'], 'oxidation')

    def test_cooked_temporary_plate_is_recipe_stage_not_raw_storage(self):
        state = fixture(); food=item(stage='cooked_protein',pot='pot1')
        state['ai'].update(x=6,y=2,facing='right',holding=food)
        state = tick(state,ai='interact')
        for _ in range(15): state=tick(state)
        self.assertEqual(state['buffers']['protein']['pot1']['stage'],'cooked_protein')
        self.assertNotIn('storage_since_turn',state['buffers']['protein']['pot1'])


class DeliveryGrace(unittest.TestCase):
    def blocked(self, near=True):
        state=fixture(); state['ai'].update(x=5 if near else 6,y=3 if near else 1,facing='left' if near else 'right',holding=dish())
        state['handoff']=item('meat',iid='blocking-meat',order='order2')
        state['human'].update(x=3,y=3,facing='right')
        return state

    def test_arrival_then_two_complete_waits_then_third_turn_disposal(self):
        state=self.blocked(False)
        approach_steps=0
        while e._distance(state,'ai','handoff'):
            self.assertEqual(e.decide(state)['reason_code'],'approach_blocked_output')
            self.assertEqual(state['policy_memory'].get('handoff_wait_turns',0),0)
            state=e.step(state,'wait'); approach_steps+=1
        self.assertGreater(approach_steps,0)
        for waited in (1,2):
            self.assertEqual(e.decide(state)['reason_code'],'wait_blocked_output')
            self.assertEqual(e.decide(state)['action'],'wait')
            state=e.step(state,'wait')
            self.assertEqual(state['policy_memory']['handoff_wait_turns'],waited)
            self.assertEqual(state['metrics']['discarded_dishes'],0)
        self.assertEqual(e.decide(state)['reason_code'],'discard_blocked_output')
        while not state['metrics']['discarded_dishes']: state=e.step(state,'wait')
        self.assertEqual(e._front(state['ai'])['id'],'trash')
        self.assertEqual(state['metrics']['discard_penalty'],20)

    def test_cleared_during_either_wait_or_walk_resumes_delivery(self):
        for delay in (0,1,2,3):
            state=self.blocked()
            for _ in range(delay): state=e.step(state,'wait')
            state=e.step(state,'interact')
            self.assertIsNone(state['handoff'])
            self.assertEqual(state['metrics']['discarded_dishes'],0)
            self.assertEqual(e.decide(state)['reason_code'],'deliver')
            for _ in range(12):
                if state['handoff']: break
                state=e.step(state,'wait')
            self.assertEqual(state['handoff']['stage'],'finished')
            self.assertEqual(state['metrics']['discarded_dishes'],0)

    def test_cleared_on_bin_interaction_turn_keeps_finished_dish(self):
        state=self.blocked()
        while not (e.decide(state)['reason_code']=='discard_blocked_output' and e.decide(state)['action']=='interact'):
            state=e.step(state,'wait')
        state=e.step(state,'interact')
        self.assertIsNone(state['handoff'])
        self.assertEqual(state['ai']['holding']['stage'],'finished')
        self.assertEqual(state['metrics']['discarded_dishes'],0)
        self.assertTrue(any(ev['type']=='delivery_resumed' for ev in state['events']))


class RealParallelTraces(unittest.TestCase):
    def two_available_inputs(self, stored_at=0):
        state=fixture(); state['ai'].update(x=6,y=3,facing='right')
        state['pots'][0].update(phase='await_vegetable',order_id='order1',recipe='egg_tomato')
        protein=item(stage='cooked_protein',pot='pot1'); protein['was_buffered']=True
        state['buffers']['protein']['pot1']=protein
        state['buffers']['ai_raw']=[item('tomato',iid='first-tomato',order='order1'),item('meat',iid='second-meat',order='order2')]
        for food in state['buffers']['ai_raw']:
            food.update(storage_since_turn=stored_at,storage_station='ai_raw',prepared_turn=stored_at,
                        fresh_until=stored_at+20,expires_turn=stored_at+20,freshness_basis='prepared')
        return state

    def test_controller_starts_second_pan_before_first_vegetable_when_safe(self):
        state=self.two_available_inputs()
        self.assertEqual(e.decide(state)['reason_code'],'start_parallel_recipe')
        self.assertEqual(e.decide(state)['slot'],1)
        for _ in range(5): state=e.step(state,'wait')
        self.assertEqual(state['pots'][1]['order_id'],'order2')
        self.assertEqual(state['pots'][1]['status'],'cooking')
        self.assertEqual(state['pots'][0]['phase'],'await_vegetable')
        self.assertEqual(state['metrics']['completed_orders'],0)
        self.assertEqual(state['metrics']['spoiled'],0)

    def test_second_pan_detour_yields_to_first_ingredient_expiry(self):
        state=self.two_available_inputs(stored_at=-10)
        # Loading the second pan, fetching slot 1, and walking/loading its pan
        # cannot complete within the ten turns left. Pickup does not refresh it.
        self.assertEqual(e.decide(state)['reason_code'],'accept_ingredient')
        self.assertEqual(e.decide(state)['slot'],0)

    def test_two_real_orders_overlap_before_first_output_and_four_dishes_complete(self):
        for seed in (1000,1001,2000,2001):
            for task in (1,2,3):
                with self.subTest(seed=seed,task=task):
                    frames,events=run(seed,task)
                    self.assertEqual(frames[-1]['metrics']['completed_orders'],4)
                    self.assertGreater(frames[-1]['metrics']['parallel_recipe_turns'],0)
                    overlap=[s for s in frames if all(p['phase']!='idle' for p in s['pots'])]
                    self.assertTrue(overlap)
                    self.assertNotEqual(overlap[0]['pots'][0]['order_id'],overlap[0]['pots'][1]['order_id'])
                    self.assertEqual(overlap[0]['metrics']['completed_orders'],0)
                    self.assertTrue(any(e.decide(s)['reason_code']=='start_parallel_recipe' for s in frames))
                    self.assertEqual(frames[-1]['metrics']['discarded_dishes'],0)
                    self.assertEqual(frames[-1]['metrics']['burnt'],0)

if __name__=='__main__': unittest.main()
