"""Build bilingual questions from actual v4 fixed-AI trajectories.

Plans below are hand-specified test oracles, not responses from a language model.
Explicit mechanism claims are independent expectations and the shared QA test
checks them against state/decision/simulation. Running this script is not evidence
of unrestricted language understanding or human performance improvement.
"""
from copy import deepcopy
import json
from pathlib import Path
from . import engine as e


def regular_trace():
    state = e.initial_state(1000, 2)
    frames = [state]
    while not state['terminal']:
        state = e.step(state, e.human_advisor(state)); frames.append(state)
    return frames


def vegetable_first_trace():
    state = e.initial_state(1000, 2)
    frames = [state]
    for target, count in [('tomato', 1), ('prep', 2), ('handoff', 1)]:
        for _ in range(count):
            while e._front(state['human']) is None or e._front(state['human'])['id'] != target:
                state = e.step(state, e._approach(state, 'human', target)); frames.append(state)
            state = e.step(state, 'interact'); frames.append(state)
    # A legal simultaneous pickup failure is part of this executed trajectory.
    assert e.decide(state)['action'] == 'interact'
    state = e.step(state, 'interact'); frames.append(state)
    while not any(it for it in state['buffers']['ai_raw']):
        state = e.step(state, 'wait'); frames.append(state)
    while not state['terminal']:
        state = e.step(state, e.human_advisor(state)); frames.append(state)
    return frames


def raw_handoff_state():
    state = e.initial_state(1000, 2)
    for target in ('egg', 'handoff'):
        while e._front(state['human']) is None or e._front(state['human'])['id'] != target:
            state = e.step(state, e._approach(state, 'human', target))
        state = e.step(state, 'interact')
    return state


def build():
    frames, early = regular_trace(), vegetable_first_trace()
    first = lambda predicate: next(state for state in frames if predicate(state))
    early_first = lambda predicate: next(state for state in early if predicate(state))
    states = {
        'start': frames[0],
        'raw_handoff': raw_handoff_state(),
        'egg_raw': first(lambda s: s['human']['holding'] and s['human']['holding']['ingredient'] == 'egg' and s['human']['holding']['stage'] == 'raw'),
        'one_whisk': first(lambda s: s['human']['holding'] and s['human']['holding']['ingredient'] == 'egg' and s['human']['holding']['prepare_progress'] == 1),
        'egg_prepared': first(lambda s: s['human']['holding'] and s['human']['holding']['ingredient'] == 'egg' and s['human']['holding']['stage'] == 'prepared'),
        'handoff_input': first(lambda s: s['handoff'] and s['handoff']['stage'] == 'prepared'),
        'ai_input': first(lambda s: s['ai']['holding'] and s['ai']['holding']['stage'] == 'prepared'),
        'egg_load': first(lambda s: s['pots'][0]['phase'] == 'protein' and s['pots'][0]['status'] == 'cooking' and s['pots'][0]['remaining'] == 4),
        'egg_ready': first(lambda s: s['pots'][0]['phase'] == 'protein' and s['pots'][0]['status'] == 'ready'),
        'egg_removed': first(lambda s: s['ai']['holding'] and s['ai']['holding']['ingredient'] == 'egg' and s['ai']['holding']['stage'] == 'cooked_protein' and not s['ai']['holding']['was_buffered']),
        'egg_stored': first(lambda s: s['buffers']['protein']['pot1'] and s['buffers']['protein']['pot1']['ingredient'] == 'egg'),
        'meat_load': first(lambda s: s['pots'][1]['phase'] == 'protein' and s['pots'][1]['status'] == 'cooking' and s['pots'][1]['remaining'] == 6),
        'both_recipes': first(lambda s: all(p['phase'] != 'idle' for p in s['pots'])),
        'tomato_load': first(lambda s: s['pots'][0]['phase'] == 'vegetable' and s['pots'][0]['status'] == 'cooking' and s['pots'][0]['remaining'] == 4),
        'tomato_ready': first(lambda s: s['pots'][0]['phase'] == 'vegetable' and s['pots'][0]['status'] == 'ready'),
        'protein_return': first(lambda s: s['ai']['holding'] and s['ai']['holding']['stage'] == 'cooked_protein' and s['ai']['holding']['was_buffered']),
        'mix': first(lambda s: s['pots'][0]['phase'] == 'mix' and s['pots'][0]['remaining'] == 2),
        'output': first(lambda s: s['ai']['holding'] and s['ai']['holding']['stage'] == 'finished'),
        'handoff_output': first(lambda s: s['handoff'] and s['handoff']['stage'] == 'finished'),
        'human_output': first(lambda s: s['human']['holding'] and s['human']['holding']['stage'] == 'finished'),
        'plated': first(lambda s: s['human']['holding'] and s['human']['holding']['stage'] == 'plated'),
        'served': first(lambda s: s['metrics']['completed_orders'] == 1),
        'early_held': early_first(lambda s: s['ai']['holding'] and e.decide(s)['reason_code'] == 'store_early_vegetable'),
        'early_stored': early_first(lambda s: any(s['buffers']['ai_raw'])),
        'conflict': early_first(lambda s: any(ev['type'] == 'handoff_conflict' for ev in s['events'])),
    }
    # At least thirty distinct bilingual prompts, rather than repeated variants
    # of the same start-state request. State claims use fixed mechanical values.
    specs = [
        ('empty_hands', 'start', 'What am I holding?', '我现在手里拿着什么？', ['human_holding'], [{'path':'human.holding','equals':None}], 'human', 'observation'),
        ('roles', 'start', 'Can you complete a dish if I never prepare or serve anything?', '如果我不备料也不上菜，你能独自完成一道菜吗？', ['public_rule0'], [], 'shared', 'rule'),
        ('controls', 'start', 'How do W, A, S, D, E and Space affect the kitchen?', 'WASD、E 和空格分别怎样操作厨房？', ['public_rule1'], [], 'shared', 'rule'),
        ('four_cupboards', 'start', 'Which four ingredients can I collect, and are they in separate cupboards?', '我能拿哪四种原料，它们在不同的柜子吗？', ['public_rule2'], [], 'shared', 'rule'),
        ('no_clock', 'egg_load', 'Does food keep cooking while I stop pressing keys to read or ask?', '我停下按键阅读或提问时，锅还会继续烹饪吗？', ['public_rule1','public_rule8'], [], 'shared', 'rule'),
        ('remaining_turns', 'start', 'How many steps are available in this task?', '这个任务一共有多少步？', ['current_turn'], [{'path':'max_turns','equals':360}], 'shared', 'observation'),
        ('raw_egg', 'egg_raw', 'Is this egg already prepared for the stove?', '我手里的鸡蛋已经能下锅了吗？', ['human_holding','public_rule2'], [{'path':'human.holding.stage','equals':'raw'}], 'human', 'observation'),
        ('one_whisk', 'one_whisk', 'I whisked once. How much preparation remains?', '鸡蛋已经打散一次，还要备料几次？', ['human_holding','public_rule2'], [{'path':'human.holding.prepare_progress','equals':1}], 'human', 'observation'),
        ('prepared_egg', 'egg_prepared', 'Is the ingredient in my hand ready for handing over?', '我手里的原料现在可以交接了吗？', ['human_holding'], [{'path':'human.holding.stage','equals':'prepared'}], 'human', 'observation'),
        ('front_only', 'egg_raw', 'Why can I not take another portion from the cupboard while holding this?', '我拿着这一份时，为什么不能再从柜子拿一份？', ['front_interaction','public_rule5'], [], 'human', 'observation'),
        ('facing', 'start', 'Can E use a worktop beside me or behind me?', 'E 能操作我侧面或身后的工作台吗？', ['public_rule1'], [], 'shared', 'rule'),
        ('blocked_turn', 'start', 'If a direction points into a cupboard, does it change my facing and spend a step?', '朝柜子方向按移动键，会转向并消耗一步吗？', ['public_rule1'], [], 'shared', 'rule'),
        ('handoff_contents', 'handoff_input', 'What is on the handoff counter?', '交接台上现在是什么？', ['handoff'], [{'path':'handoff.stage','equals':'prepared'}], 'shared', 'observation'),
        ('ai_pickup', 'handoff_input', 'What are you going to do next, and why?', '你下一步会做什么，为什么？', ['system:ai_action','system:ai_reason'], [], 'ai', 'reason'),
        ('ai_destination', 'ai_input', 'Where are you taking the prepared ingredient?', '你要把手里的备料拿到哪里？', ['system:ai_reason','ai_holding'], [{'path':'ai.holding.stage','equals':'prepared'}], 'ai', 'reason'),
        ('egg_timer', 'egg_load', 'How many full cooking steps does this egg still need?', '这份鸡蛋还需要几步完整烹饪？', ['pot1'], [{'path':'pots.0.remaining','equals':4}], 'shared', 'observation'),
        ('load_not_tick', 'egg_load', 'Did loading the egg already use one of its four cooking steps?', '把鸡蛋下锅的这一步，已经算进四步烹饪了吗？', ['pot1','public_rule6'], [{'path':'pots.0.remaining','equals':4}], 'shared', 'rule'),
        ('meat_timer', 'meat_load', 'Does meat need the same four steps as egg?', '肉和鸡蛋一样只要四步吗？', ['pot2','public_rule3'], [{'path':'pots.1.remaining','equals':6}], 'shared', 'rule'),
        ('ready_eight', 'egg_ready', 'The egg just became ready. What is its remaining safe window?', '鸡蛋刚炒好，安全窗口还剩多少回合？', ['pot1','public_rule6'], [{'path':'pots.0.ready_age','equals':0}], 'shared', 'observation'),
        ('store_protein', 'egg_removed', 'Why do you take the cooked egg to that temporary plate counter?', '鸡蛋炒熟后为什么要拿到那个临时盘位？', ['system:ai_reason','public_rule3'], [{'path':'ai.holding.container','equals':'temporary_plate'},{'path':'ai.holding.was_buffered','equals':False}], 'ai', 'reason'),
        ('protein_stored', 'egg_stored', 'What is waiting on stove 1\'s temporary plate counter?', '炉灶 1 的临时盘位上暂存着什么？', ['pot1_temporary_plate'], [{'path':'buffers.protein.pot1.stage','equals':'cooked_protein'},{'path':'buffers.protein.pot1.was_buffered','equals':True}], 'shared', 'observation'),
        ('plate_not_served', 'egg_stored', 'The egg has a temporary plate. Does that already complete a dish?', '鸡蛋已经有临时盘了，这样就完成一道菜了吗？', ['public_rule3','public_rule4'], [{'path':'metrics.completed_orders','equals':0}], 'shared', 'rule'),
        ('two_recipes', 'both_recipes', 'What stage is each pan working on?', '两口锅现在各处在哪一道工序？', ['pot1','pot2'], [{'path':'pots.0.phase','equals':'await_vegetable'},{'path':'pots.1.phase','equals':'protein'}], 'shared', 'observation'),
        ('no_false_parallel', 'both_recipes', 'Are both pans actively heating food at this exact step?', '这一刻两口锅都正在加热食物吗？', ['pot1','pot2'], [{'path':'pots.0.status','equals':'empty'},{'path':'pots.1.status','equals':'cooking'}], 'shared', 'observation'),
        ('tomato_timer', 'tomato_load', 'How long does the tomato stage take?', '炒番茄这个阶段需要多久？', ['pot1','public_rule3'], [{'path':'pots.0.remaining','equals':4},{'path':'pots.0.phase','equals':'vegetable'}], 'shared', 'observation'),
        ('ready_vegetable', 'tomato_ready', 'The tomato is ready. Does it go directly to serving now?', '番茄炒好了，现在直接上菜吗？', ['pot1','public_rule3'], [{'path':'pots.0.phase','equals':'vegetable'}], 'shared', 'rule'),
        ('return_protein', 'protein_return', 'Why are you carrying the egg back to its pan?', '为什么又把鸡蛋拿回那口锅？', ['system:ai_reason','ai_holding'], [{'path':'ai.holding.was_buffered','equals':True}], 'ai', 'reason'),
        ('mix_timer', 'mix', 'After combining the two cooked parts, how many steps remain?', '把两部分熟食倒到一起后，还要合炒几步？', ['pot1','public_rule3'], [{'path':'pots.0.remaining','equals':2},{'path':'pots.0.phase','equals':'mix'}], 'shared', 'observation'),
        ('two_components', 'mix', 'Is this the combined dish rather than a single ingredient?', '锅里现在是合成菜，还是只有一种原料？', ['public_rule3','pot1'], [{'path':'pots.0.item.ingredients','equals':['egg','tomato']}], 'shared', 'observation'),
        ('output_container', 'output', 'Is the container in your hand already my final serving plate?', '你手里的容器已经是我可以上菜的正式餐盘了吗？', ['ai_holding','public_rule4'], [{'path':'ai.holding.container','equals':'output_container'}], 'shared', 'rule'),
        ('handoff_output', 'handoff_output', 'What is now on the handoff counter, and how can I help?', '交接台上现在是什么，我可以怎么配合？', ['handoff','system:human_advice'], [{'path':'handoff.stage','equals':'finished'}], 'human', 'advice'),
        ('human_plate', 'human_output', 'What remains before I can serve what I am carrying?', '我拿着这份成品，上菜前还差什么？', ['human_holding','public_rule4','system:human_advice'], [{'path':'human.holding.container','equals':'output_container'}], 'human', 'advice'),
        ('ready_to_serve', 'plated', 'Has this dish been transferred onto a serving plate?', '这道菜已经转装到正式餐盘了吗？', ['human_holding'], [{'path':'human.holding.stage','equals':'plated'},{'path':'human.holding.container','equals':'serving_plate'}], 'human', 'observation'),
        ('score', 'served', 'How many orders have we actually completed so far?', '目前我们实际完成了几个订单？', ['system:score'], [{'path':'metrics.completed_orders','equals':1}], 'shared', 'observation'),
        ('early_vegetable', 'early_held', 'I gave you tomato first. Why are you putting it aside?', '我先给了番茄，你为什么把它放到一边？', ['system:ai_reason','ai_holding'], [{'decision_path':'reason_code','equals':'store_early_vegetable'},{'path':'ai.holding.ingredient','equals':'tomato'}], 'ai', 'reason'),
        ('stored_vegetable', 'early_stored', 'Where did the prepared tomato go?', '备好的番茄被放到哪里了？', ['raw_slot1'], [{'path':'buffers.ai_raw.0.ingredient','equals':'tomato'}], 'shared', 'observation'),
        ('early_not_waste', 'early_stored', 'Did giving the vegetable first cause it to disappear or lose points immediately?', '先交蔬菜会让原料消失，或者立刻扣分吗？', ['raw_slot1','public_rule8'], [{'path':'metrics.waste','equals':0},{'path':'metrics.completed_orders','equals':0}], 'shared', 'observation'),
        ('conflict', 'conflict', 'We both reached for the counter. Why did neither transfer happen?', '双方都去拿交接台的东西，为什么交接没有发生？', ['public_rule7','event0'], [{'path':'metrics.handoff_conflicts','equals':1}], 'shared', 'rule'),
        ('bound_order', 'start', 'If two orders request the same recipe, which order does a cooked dish complete?', '如果两张订单要同一道菜，一份做好的菜完成哪张订单？', ['public_rule8'], [], 'shared', 'rule'),
        ('deadline', 'start', 'An order is due on turn 130. May it be served during turn 130?', '订单截止第 130 回合，在第 130 回合上菜还来得及吗？', ['public_rule8','order1'], [{'path':'orders.0.deadline','equals':130}], 'shared', 'rule'),
        ('discard', 'egg_prepared', 'Can I discard my held item if I need to free my hand, and is there a hidden penalty?', '需要腾出手时能丢弃手中物品吗，有隐藏扣分吗？', ['public_rule5','public_rule8'], [], 'shared', 'rule'),
        ('safe_off_heat', 'egg_removed', 'Will the egg still burn while it is in your temporary plate?', '鸡蛋已经在你的临时盘里，还会继续烧糊吗？', ['ai_holding','public_rule6'], [{'path':'ai.holding.stage','equals':'cooked_protein'}], 'shared', 'rule'),
        ('false_cooked', 'ai_input', 'Why are you already holding the finished stir-fry?', '你为什么已经拿着最终成品了？', ['ai_holding'], [{'path':'ai.holding.stage','equals':'prepared'}], 'ai', 'observation'),
        ('mixed_question', 'egg_load', 'What am I holding, and what is stove 1 cooking?', '我手里是什么，炉灶 1 正在炒什么？', ['human_holding','pot1'], [{'path':'pots.0.item.ingredient','equals':'egg'}], 'shared', 'observation'),
    ]
    specs.extend([
        ('where_egg_in_pan', 'egg_ready', 'Where is the egg right now, and is it on a plate yet?', '鸡蛋现在在哪里，已经装到盘里了吗？', ['ingredient_location_egg','pot1_contents'], [{'path':'pots.0.item.container','equals':None},{'path':'pots.0.item.ingredient','equals':'egg'}], 'shared', 'observation'),
        ('where_egg_stored', 'egg_stored', 'Where did you put the cooked egg?', '炒好的鸡蛋被你放到哪里了？', ['ingredient_location_egg'], [{'path':'buffers.protein.pot1.container','equals':'temporary_plate'}], 'shared', 'observation'),
        ('where_egg_mixed', 'mix', 'I cannot see the separate egg plate. Where is the egg now?', '我看不到单独的鸡蛋盘了，鸡蛋现在在哪里？', ['ingredient_location_egg','pot1_contents'], [{'path':'pots.0.item.ingredients','equals':['egg','tomato']}], 'shared', 'observation'),
        ('next_ingredient', 'start', 'Which ingredient should I hand over next, beyond just moving?', '除了往哪边走，我下一份应该递什么原料？', ['next_input_ingredient','missing_ingredients'], [{'path':'orders.0.recipe','equals':'egg_tomato'}], 'shared', 'observation'),
        ('raw_blocker_reason', 'raw_handoff', 'I gave you an egg. Why are you not taking it to the stove?', '我已经给了鸡蛋，你为什么不拿去下锅？', ['system:ai_reason','handoff_recovery'], [{'path':'handoff.stage','equals':'raw'},{'decision_path':'reason_code','equals':'await_preparation'}], 'ai', 'reason'),
        ('raw_blocker_recovery', 'raw_handoff', 'How can I recover after placing an unprepared egg on the handoff counter?', '把没备好的鸡蛋放到交接台以后，怎么恢复？', ['handoff_recovery','next_input_ingredient','system:human_advice'], [{'path':'handoff.ingredient','equals':'egg'},{'path':'handoff.prepare_progress','equals':0}], 'human', 'advice'),
    ])
    cases = []
    for name, state_key, en, zh, identifiers, claims, subject, purpose in specs:
        state = states[state_key]
        available = {fact['id'] for fact in e.facts(state)} | {'system:ai_action','system:ai_reason','system:human_advice','system:score'}
        assert set(identifiers) <= available, (name, set(identifiers) - available)
        for language, question in [('en', en), ('zh', zh)]:
            plan = {'language':language, 'binding':{'task':state['task'],'turn':state['turn']}, 'premise':'contradicted' if name == 'false_cooked' else 'supported',
                    'clarification':None, 'intents':[{'kind':'facts','subject':subject,'purpose':purpose,'evidence_ids':identifiers}]}
            cases.append({'case_id':f'kitchen_v4_{name}_{language}', 'domain':'kitchen', 'question':question, 'language':language, 'state':deepcopy(state),
                          'expected_fact_ids':identifiers, 'expected_claims':deepcopy(claims), 'expected_kind':'facts', 'expected_plan':plan})
    for name, state_key, en, zh, actions, claims in [
        ('wait_cf','egg_load','If I wait one step, will that step alone earn a completed-order point?','如果我等待一步，仅这一步会获得完成订单分吗？',['wait'],[{'simulation_path':'raw_score_delta','equals':0}]),
        ('movement_cf','start','If I press W now, how many game steps advance?','如果我现在按 W，游戏会推进几步？',['up'],[{'simulation_path':'steps_completed','equals':1},{'simulation_path':'raw_score_delta','equals':0}]),
        ('unavailable_e_cf','start','What if I press E now while facing no work station?','如果我现在没有面朝工位却按 E，会怎么样？',['interact'],[{'simulation_path':'steps_completed','equals':0}]),
    ]:
        state = states[state_key]
        for language, question in [('en',en),('zh',zh)]:
            cases.append({'case_id':f'kitchen_v4_{name}_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
                'expected_fact_ids':[],'expected_claims':claims,'expected_kind':'counterfactual',
                'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'supported','clarification':None,
                    'intents':[{'kind':'counterfactual','subject':'human','purpose':'comparison','actions':actions,'horizon':1,'evidence_ids':[]}]}})
    for name, reason, en, zh in [
        ('ambiguous','ambiguous_object','What about that other thing?','那另一个东西呢？'),
        ('unbound_history','select_frame','Why did you change your plan at an earlier unspecified point in Task 1?','你为什么在 Task 1 之前某个没指出的时刻改变计划？'),
        ('outside','unsupported_question','Who won the last World Cup?','上一届世界杯谁夺冠了？')]:
        state = states['start']
        for language, question in [('en',en),('zh',zh)]:
            cases.append({'case_id':f'kitchen_v4_{name}_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
                'expected_fact_ids':[],'expected_claims':[],'expected_kind':'clarification',
                'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'unclear','clarification':reason,'intents':[]}})
    for language, question, previous in [('en','And the other pan?','What is stove 1 doing?'),('zh','那另一口锅呢？','炉灶 1 正在做什么？')]:
        state = states['both_recipes']
        cases.append({'case_id':f'kitchen_v4_followup_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
            'previous_dialogue':[{'question':previous,'answer':next(f[language] for f in e.facts(state) if f['id']=='pot1')}],
            'expected_fact_ids':['pot2'],'expected_claims':[{'path':'pots.1.phase','equals':'protein'}],'expected_kind':'facts',
            'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'supported','clarification':None,
                'intents':[{'kind':'facts','subject':'shared','purpose':'observation','evidence_ids':['pot2']}]}})
    past = frames[0]
    state = states['egg_load']
    for language, question in [('en','Where was I at turn 0?'),('zh','第 0 回合我在哪里？')]:
        identifier = 'history:task2:turn0:human_position'
        cases.append({'case_id':f'kitchen_v4_bound_history_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
            'public_history':[e.public_state(past)],'expected_fact_ids':[identifier],'expected_claims':[],'expected_kind':'facts',
            'expected_plan':{'language':language,'binding':{'task':2,'turn':0},'premise':'supported','clarification':None,
                'intents':[{'kind':'facts','subject':'human','purpose':'observation','evidence_ids':[identifier]}]}})
    return cases


if __name__ == '__main__':
    cases = build()
    Path(__file__).with_name('qa_cases.json').write_text(json.dumps(cases, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'Wrote {len(cases)} bilingual v4 Kitchen cases from executed fixed-controller trajectories; not a semantic-model evaluation.')
