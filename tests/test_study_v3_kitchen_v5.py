"""Independent v5 boundary checks; synthetic fixtures are not human evidence."""
from copy import deepcopy
import json
from pathlib import Path
import re
import unittest

from domains.kitchen import engine as e
from tests.test_study_v3_kitchen import fixture, item, dish, tick, run


def collect(state, ingredient):
    while (e._front(state['human']) or {}).get('id') != ingredient:
        state = tick(state, e._approach(state, 'human', ingredient))
    return tick(state, 'interact')


def prepare(state):
    while (e._front(state['human']) or {}).get('id') != 'prep':
        state = tick(state, e._approach(state, 'human', 'prep'))
    while state['human']['holding']['stage'] == 'raw':
        state = tick(state, 'interact')
    return state


class KitchenV5Mechanisms(unittest.TestCase):
    def test_full_fixed_menu_differs_by_task_and_initial_intention_is_public(self):
        menus = []
        counts = []
        for task in (1,2,3):
            state = e.initial_state(1000,task)
            view = e.public_state(state)
            self.assertEqual(len(view['menu']),4)
            self.assertEqual([o['ordinal'] for o in view['menu']],[1,2,3,4])
            self.assertTrue(all(o['status']=='pending' for o in view['menu']))
            self.assertEqual(state['_future_orders'],[])
            self.assertEqual(state,e.initial_state(1000,task))
            self.assertEqual(view['max_turns'],280)
            self.assertEqual([o['deadline'] for o in view['menu']],[100,140,240,280])
            menus.append(tuple(o['recipe'] for o in view['menu']))
            counts.append(menus[-1].count('egg_tomato'))
            job=view['ai']['current_cooking'][0]
            self.assertEqual(job['status'],'waiting_for_ingredient')
            self.assertEqual(job['recipe'],state['orders'][0]['recipe'])
            self.assertEqual(job['order_id'],'order1')
        self.assertEqual(len(set(menus)),3)
        self.assertEqual(len(set(counts)),3)

    def test_wait_negative_score_invalid_e_no_charge_and_explicit_score_events(self):
        state=fixture()
        with self.assertRaises(ValueError): e.step(state,'interact')
        self.assertEqual(state['raw_score'],0)
        state=tick(state)
        self.assertEqual(state['raw_score'],-1)
        self.assertEqual(e.score(state)['task_score'],-1)
        self.assertEqual(e.score(state)['score_scale'],'raw')
        self.assertIsNone(e.score(state)['score_max'])
        self.assertEqual([(v['delta'],v['reason']) for v in state['events'] if v['type']=='score_delta'],[(-1,'turn')])

    def test_serve_and_both_bin_penalties_are_separate_from_step_cost(self):
        for food,penalty in ((item(),5),(dish(),20)):
            state=fixture(); state['human'].update(x=3,y=4,facing='right',holding=food)
            state=tick(state,'interact')
            self.assertEqual(state['raw_score'],-1-penalty)
            self.assertEqual(sum(ev['delta'] for ev in state['events'] if ev['type']=='score_delta'),-1-penalty)
            event=next(ev for ev in state['events'] if ev['type']=='waste')
            self.assertEqual(event['station'],'trash'); self.assertEqual(event['penalty'],penalty)
            self.assertEqual(event['item']['components'],food['components'])
        state=fixture(); food=dish(); food.update(stage='plated',container='serving_plate')
        state['human'].update(x=2,y=5,facing='right',holding=food)
        state=tick(state,'interact')
        self.assertEqual(state['raw_score'],99)
        self.assertEqual(state['metrics']['completed_orders'],1)
        self.assertEqual(sum(ev['delta'] for ev in state['events'] if ev['type']=='score_delta'),99)

    def test_occupied_output_can_be_cancelled_before_reaching_handoff(self):
        state=fixture(); output=dish()
        state['ai'].update(x=6,y=1,facing='right',holding=output)
        state['human'].update(x=3,y=3,facing='right')
        state['handoff']=item('meat',iid='blocking-meat',order='order2')
        self.assertEqual(e.decide(state)['reason_code'],'approach_blocked_output')
        state=e.step(state,'interact')
        self.assertIsNone(state['handoff'])
        self.assertEqual(e.decide(state)['reason_code'],'deliver')
        self.assertEqual(state['metrics']['discarded_dishes'],0)
        while not state['handoff']: state=e.step(state,'wait')
        self.assertEqual(state['handoff']['components'],output['components'])
        self.assertEqual(state['metrics']['discarded_dishes'],0)

    def test_after_delivery_collects_next_finished_and_never_retrieves_own_output(self):
        state=fixture(); first=dish(); second=dish(order='order3',pot='pot2')
        second.update(id='second-egg',components=['second-egg','second-tomato'])
        state['orders'][2]['recipe']='egg_tomato'
        state['ai'].update(x=5,y=3,facing='left',holding=first)
        state['buffers']['protein']['pot2']=second
        state=e.step(state,'wait')
        self.assertEqual(state['handoff']['id'],first['id'])
        self.assertEqual(e.decide(state)['reason_code'],'collect_finished')
        events=[]
        for _ in range(15):
            if state['metrics']['discarded_dishes']: break
            state=e.step(state,'wait'); events.extend(state['events'])
            self.assertEqual(state['handoff']['components'],first['components'])
        self.assertEqual(state['metrics']['discarded_dishes'],1)
        self.assertEqual(next(ev['item']['components'] for ev in events if ev['type']=='waste'),second['components'])

    def test_raw_clock_exact_and_spoiled_item_remains_unusable_until_bin(self):
        state=collect(fixture(),'egg'); acquired=state['turn']; food_id=state['human']['holding']['id']
        self.assertEqual(state['human']['holding']['fresh_until'],acquired+120)
        while state['turn']<acquired+119: state=tick(state)
        self.assertEqual(state['human']['holding']['stage'],'raw')
        state=tick(state)
        self.assertEqual(state['human']['holding']['stage'],'spoiled')
        self.assertEqual(state['human']['holding']['id'],food_id)
        self.assertEqual(state['metrics']['spoiled'],1)
        self.assertEqual(state['metrics']['discard_penalty'],0)
        while (e._front(state['human']) or {}).get('id')!='prep':
            state=tick(state,e._approach(state,'human','prep'))
        self.assertNotIn('interact',e.legal_actions(state))
        self.assertIn('spoiled',e.interaction_label(state))
        while (e._front(state['human']) or {}).get('id')!='trash':
            state=tick(state,e._approach(state,'human','trash'))
        state=tick(state,'interact')
        self.assertIsNone(state['human']['holding'])
        self.assertEqual(state['metrics']['discard_penalty'],5)

    def test_preparation_clock_starts_only_at_completion_and_has_real_boundaries(self):
        for ingredient,required,lifetime in [('tomato',3,20),('pepper',3,20),('egg',4,20),('meat',5,20)]:
            with self.subTest(ingredient=ingredient):
                state=collect(fixture(),ingredient); raw_expiry=state['human']['holding']['fresh_until']
                while (e._front(state['human']) or {}).get('id')!='prep': state=tick(state,e._approach(state,'human','prep'))
                for i in range(required-1):
                    state=tick(state,'interact')
                    self.assertEqual(state['human']['holding']['fresh_until'],raw_expiry)
                    self.assertEqual(e.public_state(state)['human']['preparation']['remaining'],required-i-1)
                state=tick(state,'interact'); prepared_at=state['turn']; food_id=state['human']['holding']['id']
                self.assertEqual(state['human']['holding']['fresh_until'],prepared_at+lifetime)
                self.assertTrue(e.public_state(state)['human']['preparation']['ready'])
                while (e._front(state['human']) or {}).get('id')!='human_buffer': state=tick(state,e._approach(state,'human','human_buffer'))
                state=tick(state,'interact')
                while state['turn']<prepared_at+19: state=tick(state)
                self.assertEqual(state['buffers']['human']['stage'],'prepared')
                self.assertEqual(e.public_state(state)['food_freshness'][0]['remaining'],1)
                state=tick(state)
                self.assertEqual(state['buffers']['human']['stage'],'spoiled')
                self.assertEqual(state['buffers']['human']['id'],food_id)
                self.assertEqual(e.public_state(state)['food_freshness'][0]['remaining'],0)

    def test_last_fresh_turn_loading_ends_oxidation_but_does_not_prevent_burning(self):
        state=prepare(collect(fixture(),'egg')); food=state['human']['holding']; state['human']['holding']=None
        state['turn']=food['prepared_turn']+18  # Commit the load at age 19, not age 20.
        state['ai'].update(x=6,y=1,facing='right',holding=food)
        state=tick(state,ai='interact')
        self.assertIsNone(state['pots'][0]['item']['fresh_until'])
        for _ in range(e.COOK_TURNS['egg'] + e.BURN_TURNS): state=tick(state)
        self.assertEqual(state['metrics']['spoiled'],0)
        self.assertEqual(state['pots'][0]['status'],'burnt')
        self.assertEqual(state['pots'][0]['item']['id'],food['id'])

    def test_early_vegetable_really_spoils_then_ai_physically_discards_it(self):
        state=prepare(collect(fixture(),'tomato')); original=deepcopy(state['human']['holding'])
        state['ai']['holding']=state['human']['holding']; state['human']['holding']=None
        for _ in range(15):
            if state['buffers']['ai_raw'][0]: break
            state=e.step(state,'wait')
        self.assertEqual(state['buffers']['ai_raw'][0]['id'],original['id'])
        while state['turn']<original['prepared_turn']+20: state=e.step(state,'wait')
        self.assertEqual(state['buffers']['ai_raw'][0]['stage'],'spoiled')
        self.assertEqual(state['metrics']['discard_penalty'],0)
        for _ in range(15):
            if state['metrics']['discard_penalty']: break
            state=e.step(state,'wait')
        waste=next(ev for ev in state['events'] if ev['type']=='waste')
        self.assertEqual(waste['item']['id'],original['id'])
        self.assertEqual(waste['station'],'trash')
        self.assertEqual(state['metrics']['discard_penalty'],5)

    def test_observation_is_pure_and_human_facing_explanations_use_recipe_names(self):
        frames,events=run(1000,2)
        self.assertEqual(sum(ev['delta'] for ev in events if ev['type']=='score_delta'),frames[-1]['raw_score'])
        for state in frames:
            before=deepcopy(state)
            facts=e.facts(state); decision=e.decide(state); e.public_state(state)
            self.assertEqual(state,before)
            texts=[f[lang] for f in facts for lang in ('en','zh')]
            texts.extend([decision['reason_en'],decision['reason_zh']])
            self.assertFalse(any(re.search(r'\border\d+\b',text) for text in texts))
        self.assertEqual(frames[-1]['metrics']['completed_orders'],4)

    def test_frozen_bilingual_cases_rebuild_from_the_current_real_trajectories(self):
        from domains.kitchen.build_qa_cases import build
        recorded=json.loads((Path(e.__file__).parent/'qa_cases.json').read_text())
        self.assertEqual(build(),recorded)
        self.assertEqual(len(recorded),156)
        self.assertEqual(len({case['question'] for case in recorded}),156)


if __name__=='__main__': unittest.main()
