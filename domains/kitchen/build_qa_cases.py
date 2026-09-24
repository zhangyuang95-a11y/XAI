"""Build bilingual questions from current fixed-AI Kitchen trajectories.

Plans below are hand-specified test oracles, not responses from a language model.
Explicit mechanism claims are independent expectations and the shared QA test
checks them against state/decision/simulation. Running this script is not evidence
of unrestricted language understanding or human performance improvement.
"""
from copy import deepcopy
import json
from pathlib import Path
from . import engine as e

# Fixed development scene whose menu begins with tomato-and-egg. This choice
# is for named-ingredient QA coverage, never a held-out performance claim.
QA_SEED = 1006
HEATING_SEED = 1005

def regular_trace(seed=QA_SEED):
    state = e.initial_state(seed, 2)
    frames = [state]
    while not state['terminal']:
        state = e.step(state, e.human_advisor(state)); frames.append(state)
    return frames


def vegetable_first_trace():
    state = e.initial_state(QA_SEED, 2)
    frames = [state]
    for target, count in [('tomato', 1), ('prep', e.PREPARE_TURNS['tomato']), ('handoff', 1)]:
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
    state = e.initial_state(QA_SEED, 2)
    for target in ('egg', 'handoff'):
        while e._front(state['human']) is None or e._front(state['human'])['id'] != target:
            state = e.step(state, e._approach(state, 'human', target))
        state = e.step(state, 'interact')
    return state


def occupied_output_trace(frames):
    # After handing over the tomato needed by stove 1, bring an unnecessary raw
    # egg instead of collecting its upcoming dish. Select that actual handoff
    # state by its contents, so heating-duration changes cannot shift the case.
    state=deepcopy(next(s for s in frames
                        if s['human']['holding'] is None
                        and s['handoff'] and s['handoff']['ingredient']=='tomato'
                        and s['handoff']['stage']=='prepared'
                        and s['pots'][0]['phase']=='await_vegetable'
                        and any(p['id']=='pot1' for p in e._loadable(s,s['handoff']))))
    result=[state]
    for target in ('egg','handoff'):
        while (e._front(state['human']) or {}).get('id') != target:
            state=e.step(state,e._approach(state,'human',target)); result.append(state)
        # The AI may finish a safe pan rescue before taking the tomato. Wait
        # for that actual transfer instead of assuming the counter is clear.
        while target=='handoff' and state['handoff'] is not None and not state['terminal']:
            state=e.step(state,'wait'); result.append(state)
        assert not state['terminal'], 'The ingredient handoff never cleared'
        state=e.step(state,'interact'); result.append(state)
    while not state['terminal']:
        if state['metrics']['discarded_dishes']: break
        state=e.step(state,'wait'); result.append(state)
    assert state['metrics']['discarded_dishes']==1
    return result


def build():
    frames, early = regular_trace(), vegetable_first_trace()
    heating_frames = regular_trace(HEATING_SEED)
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
        'egg_load': first(lambda s: s['pots'][0]['phase'] == 'protein' and s['pots'][0]['status'] == 'cooking' and s['pots'][0]['item']['ingredient'] == 'egg' and s['pots'][0]['remaining'] == e.COOK_TURNS['egg']),
        'egg_ready': first(lambda s: s['pots'][0]['phase'] == 'protein' and s['pots'][0]['status'] == 'ready' and s['pots'][0]['item']['ingredient'] == 'egg'),
        'egg_removed': first(lambda s: s['ai']['holding'] and s['ai']['holding']['ingredient'] == 'egg' and s['ai']['holding']['stage'] == 'cooked_protein' and not s['ai']['holding']['was_buffered']),
        'egg_stored': first(lambda s: s['buffers']['protein']['pot1'] and s['buffers']['protein']['pot1']['ingredient'] == 'egg'),
        'meat_load': first(lambda s: s['pots'][1]['phase'] == 'protein' and s['pots'][1]['status'] == 'cooking' and s['pots'][1]['item']['ingredient'] == 'meat' and s['pots'][1]['remaining'] == e.COOK_TURNS['meat']),
        'both_recipes': first(lambda s: s['pots'][0]['phase'] == 'await_vegetable' and s['pots'][0]['status'] == 'empty' and s['pots'][1]['phase'] == 'protein' and s['pots'][1]['status'] == 'cooking'),
        'both_heating': next(s for s in heating_frames if all(p['status'] == 'cooking' and p['remaining'] > 1 for p in s['pots']) and s['pots'][0]['order_id'] != s['pots'][1]['order_id']),
        'tomato_load': first(lambda s: s['pots'][0]['phase'] == 'vegetable' and s['pots'][0]['status'] == 'cooking' and s['pots'][0]['item']['ingredient'] == 'tomato' and s['pots'][0]['remaining'] == e.COOK_TURNS['tomato']),
        'tomato_ready': first(lambda s: s['pots'][0]['phase'] == 'vegetable' and s['pots'][0]['status'] == 'ready' and s['pots'][0]['item']['ingredient'] == 'tomato'),
        'protein_return': first(lambda s: s['ai']['holding'] and s['ai']['holding']['stage'] == 'cooked_protein' and s['ai']['holding']['ingredient'] == 'egg' and s['ai']['holding']['was_buffered']),
        'mix': first(lambda s: s['pots'][0]['phase'] == 'mix' and s['pots'][0]['remaining'] == e.MIX_TURNS and s['pots'][0]['recipe'] == 'egg_tomato'),
        'output': first(lambda s: s['ai']['holding'] and s['ai']['holding']['stage'] == 'finished' and s['ai']['holding']['recipe'] == 'egg_tomato'),
        'handoff_output': first(lambda s: s['handoff'] and s['handoff']['stage'] == 'finished'),
        'human_output': first(lambda s: s['human']['holding'] and s['human']['holding']['stage'] == 'finished'),
        'plated': first(lambda s: s['human']['holding'] and s['human']['holding']['stage'] == 'plated'),
        'serve_ready': first(lambda s: s['human']['holding'] and s['human']['holding']['stage'] == 'plated' and (e._front(s['human']) or {}).get('id') == 'serve'),
        'served': first(lambda s: s['metrics']['completed_orders'] == 1),
        'early_held': early_first(lambda s: s['ai']['holding'] and e.decide(s)['reason_code'] == 'store_early_vegetable'),
        'early_stored': early_first(lambda s: any(s['buffers']['ai_raw'])),
        'conflict': early_first(lambda s: any(ev['type'] == 'handoff_conflict' for ev in s['events'])),
    }
    state=deepcopy(states['early_stored'])
    while not any(it and it['stage']=='spoiled' for it in state['buffers']['ai_raw']):
        state=e.step(state,'wait')
    states['spoiled_tomato']=state
    blocked=occupied_output_trace(frames)
    states['blocked_output']=next(s for s in blocked if e.decide(s)['reason_code']=='discard_blocked_output')
    states['output_discarded']=blocked[-1]
    # These boundary frames are executed from a real prepared portion; every
    # elapsed turn goes through the same engine as the participant's game.
    state=deepcopy(states['egg_prepared'])
    prepared_at=state['human']['holding']['prepared_turn']
    while state['turn'] < prepared_at + e.PREPARED_FRESH_TURNS['egg'] - 1:
        state=e.step(state,'wait')
    states['freshness_last_turn']=state
    states['freshness_expired']=e.step(state,'wait')
    # At least thirty distinct bilingual prompts, rather than repeated variants
    # of the same start-state request. Timing claims use the current rules.
    specs = [
        ('empty_hands', 'start', 'What am I holding?', '我现在手里拿着什么？', ['human_holding'], [{'path':'human.holding','equals':None}], 'human', 'observation'),
        ('roles', 'start', 'Can you complete a dish if I never prepare or serve anything?', '如果我不备料也不上菜，你能独自完成一道菜吗？', ['public_rule0'], [], 'shared', 'rule'),
        ('controls', 'start', 'How do W, A, S, D, E and Space affect the kitchen?', 'WASD、E 和空格分别怎样操作厨房？', ['public_rule1'], [], 'shared', 'rule'),
        ('four_cupboards', 'start', 'Which four ingredients can I collect, and are they in separate cupboards?', '我能拿哪四种原料，它们在不同的柜子吗？', ['public_rule2'], [], 'shared', 'rule'),
        ('no_clock', 'egg_load', 'Does food keep cooking while I stop pressing keys to read or ask?', '我停下按键阅读或提问时，锅还会继续烹饪吗？', ['public_rule1','public_rule8'], [], 'shared', 'rule'),
        ('remaining_turns', 'start', 'How many steps are available in this task?', '这个任务一共有多少步？', ['current_turn'], [{'path':'max_turns','equals':states['start']['max_turns']}], 'shared', 'observation'),
        ('raw_egg', 'egg_raw', 'Is this egg already prepared for the stove?', '我手里的鸡蛋已经能下锅了吗？', ['human_holding','public_rule2'], [{'path':'human.holding.stage','equals':'raw'}], 'human', 'observation'),
        ('one_whisk', 'one_whisk', 'I whisked once. How much preparation remains?', '鸡蛋已经打散一次，还要备料几次？', ['human_holding','public_rule2'], [{'path':'human.holding.prepare_progress','equals':1}], 'human', 'observation'),
        ('prepared_egg', 'egg_prepared', 'Is the ingredient in my hand ready for handing over?', '我手里的原料现在可以交接了吗？', ['human_holding'], [{'path':'human.holding.stage','equals':'prepared'}], 'human', 'observation'),
        ('front_only', 'egg_raw', 'Why can I not take another portion from the cupboard while holding this?', '我拿着这一份时，为什么不能再从柜子拿一份？', ['front_interaction','public_rule5'], [], 'human', 'observation'),
        ('facing', 'start', 'Can E use a worktop beside me or behind me?', 'E 能操作我侧面或身后的工作台吗？', ['public_rule1'], [], 'shared', 'rule'),
        ('blocked_turn', 'start', 'If a direction points into a cupboard, does it change my facing and spend a step?', '朝柜子方向按移动键，会转向并消耗一步吗？', ['public_rule1'], [], 'shared', 'rule'),
        ('handoff_contents', 'handoff_input', 'What is on the handoff counter?', '交接台上现在是什么？', ['handoff'], [{'path':'handoff.stage','equals':'prepared'}], 'shared', 'observation'),
        ('ai_pickup', 'handoff_input', 'What are you going to do next, and why?', '你下一步会做什么，为什么？', ['system:ai_action','system:ai_reason'], [], 'ai', 'reason'),
        ('ai_destination', 'ai_input', 'Where are you taking the prepared ingredient?', '你要把手里的备料拿到哪里？', ['system:ai_reason','ai_holding'], [{'path':'ai.holding.stage','equals':'prepared'}], 'ai', 'reason'),
        ('egg_timer', 'egg_load', 'How many full cooking steps does this egg still need?', '这份鸡蛋还需要几步完整烹饪？', ['pot1'], [{'path':'pots.0.remaining','equals':e.COOK_TURNS['egg']}], 'shared', 'observation'),
        ('load_not_tick', 'egg_load', 'Does loading the egg count as a cooking step?', '鸡蛋下锅当回合计入烹饪时间吗？', ['pot1','public_rule6'], [{'path':'pots.0.remaining','equals':e.COOK_TURNS['egg']}], 'shared', 'rule'),
        ('meat_timer', 'meat_load', 'How many cooking steps does meat need compared with egg?', '肉和鸡蛋分别需要炒几回合？', ['pot2','public_rule3'], [{'path':'pots.1.remaining','equals':e.COOK_TURNS['meat']}], 'shared', 'rule'),
        ('recipe_heating_totals', 'start', 'How many heating turns does each dish need, excluding preparation, travel and transfers?', '不算备料、移动和取放，每道菜的纯加热共需几回合？', ['public_rule3'], [{'fact_id':'public_rule3','contains':str(e.COOK_TURNS[recipe['protein']]+e.COOK_TURNS[recipe['vegetable']]+e.MIX_TURNS)} for recipe in e.RECIPES.values()], 'shared', 'rule'),
        ('ready_eight', 'egg_ready', 'The egg just became ready. What is its remaining safe window?', '鸡蛋刚炒好，安全窗口还剩多少回合？', ['pot1','public_rule6'], [{'path':'pots.0.ready_age','equals':0}], 'shared', 'observation'),
        ('store_protein', 'egg_removed', 'Why do you take the cooked egg to that temporary plate counter?', '鸡蛋炒熟后为什么要拿到那个临时盘位？', ['system:ai_reason','public_rule3'], [{'path':'ai.holding.container','equals':'temporary_plate'},{'path':'ai.holding.was_buffered','equals':False}], 'ai', 'reason'),
        ('protein_stored', 'egg_stored', 'What is waiting on stove 1\'s temporary plate counter?', '炉灶 1 的临时盘位上暂存着什么？', ['pot1_temporary_plate'], [{'path':'buffers.protein.pot1.stage','equals':'cooked_protein'},{'path':'buffers.protein.pot1.was_buffered','equals':True}], 'shared', 'observation'),
        ('plate_not_served', 'egg_stored', 'The egg has a temporary plate. Does that already complete a dish?', '鸡蛋已经有临时盘了，这样就完成一道菜了吗？', ['public_rule3','public_rule4'], [{'path':'metrics.completed_orders','equals':0}], 'shared', 'rule'),
        ('two_recipes', 'both_recipes', 'What stage is each pan working on?', '两口锅现在各处在哪一道工序？', ['pot1','pot2'], [{'path':'pots.0.phase','equals':'await_vegetable'},{'path':'pots.1.phase','equals':'protein'}], 'shared', 'observation'),
        ('no_false_parallel', 'both_recipes', 'Are both pans actively heating food at this exact step?', '这一刻两口锅都正在加热食物吗？', ['pot1','pot2'], [{'path':'pots.0.status','equals':'empty'},{'path':'pots.1.status','equals':'cooking'}], 'shared', 'observation'),
        ('simultaneous_heating', 'both_heating', 'Are both pans cooking now, and how many turns remain in each?', '现在两口锅都在加热吗，各还需几回合？', ['pot1','pot2'], [{'path':'pots.0.status','equals':'cooking'},{'path':'pots.1.status','equals':'cooking'},{'path':'pots.0.remaining','equals':states['both_heating']['pots'][0]['remaining']},{'path':'pots.1.remaining','equals':states['both_heating']['pots'][1]['remaining']}], 'shared', 'observation'),
        ('tomato_timer', 'tomato_load', 'How long does the tomato stage take?', '炒番茄这个阶段需要多久？', ['pot1','public_rule3'], [{'path':'pots.0.remaining','equals':e.COOK_TURNS['tomato']},{'path':'pots.0.phase','equals':'vegetable'}], 'shared', 'observation'),
        ('ready_vegetable', 'tomato_ready', 'The tomato is ready. Does it go directly to serving now?', '番茄炒好了，现在直接上菜吗？', ['pot1','public_rule3'], [{'path':'pots.0.phase','equals':'vegetable'}], 'shared', 'rule'),
        ('return_protein', 'protein_return', 'Why are you carrying the egg back to its pan?', '为什么又把鸡蛋拿回那口锅？', ['system:ai_reason','ai_holding'], [{'path':'ai.holding.was_buffered','equals':True}], 'ai', 'reason'),
        ('mix_timer', 'mix', 'After combining the two cooked parts, how many steps remain?', '把两部分熟食倒到一起后，还要合炒几步？', ['pot1','public_rule3'], [{'path':'pots.0.remaining','equals':e.MIX_TURNS},{'path':'pots.0.phase','equals':'mix'}], 'shared', 'observation'),
        ('two_components', 'mix', 'Is this the combined dish rather than a single ingredient?', '锅里现在是合成菜，还是只有一种原料？', ['public_rule3','pot1'], [{'path':'pots.0.item.ingredients','equals':['egg','tomato']}], 'shared', 'observation'),
        ('output_container', 'output', 'Is the container in your hand already my final serving plate?', '你手里的容器已经是我可以上菜的正式餐盘了吗？', ['ai_holding','public_rule4'], [{'path':'ai.holding.container','equals':'output_container'}], 'shared', 'rule'),
        ('handoff_output', 'handoff_output', 'What is now on the handoff counter, and how can I help?', '交接台上现在是什么，我可以怎么配合？', ['handoff','system:human_advice'], [{'path':'handoff.stage','equals':'finished'}], 'human', 'advice'),
        ('human_plate', 'human_output', 'What remains before I can serve what I am carrying?', '我拿着这份成品，上菜前还差什么？', ['human_holding','public_rule4','system:human_advice'], [{'path':'human.holding.container','equals':'output_container'}], 'human', 'advice'),
        ('ready_to_serve', 'plated', 'Has this dish been transferred onto a serving plate?', '这道菜已经转装到正式餐盘了吗？', ['human_holding'], [{'path':'human.holding.stage','equals':'plated'},{'path':'human.holding.container','equals':'serving_plate'}], 'human', 'observation'),
        ('score', 'served', 'How many orders have we actually completed so far?', '目前我们实际完成了几个订单？', ['system:completed_orders'], [{'path':'metrics.completed_orders','equals':1}], 'shared', 'observation'),
        ('early_vegetable', 'early_held', 'I gave you tomato first. Why are you putting it aside?', '我先给了番茄，你为什么把它放到一边？', ['system:ai_reason','ai_holding'], [{'decision_path':'reason_code','equals':'store_early_vegetable'},{'path':'ai.holding.ingredient','equals':'tomato'}], 'ai', 'reason'),
        ('stored_vegetable', 'early_stored', 'Where did the prepared tomato go?', '备好的番茄被放到哪里了？', ['raw_slot1'], [{'path':'buffers.ai_raw.0.ingredient','equals':'tomato'}], 'shared', 'observation'),
        ('early_not_waste', 'early_stored', 'Did giving the vegetable first cause it to disappear or lose points immediately?', '先交蔬菜会让原料消失，或者立刻扣分吗？', ['raw_slot1','public_rule8'], [{'path':'metrics.waste','equals':0},{'path':'metrics.completed_orders','equals':0}], 'shared', 'observation'),
        ('conflict', 'conflict', 'We both reached for the counter. Why did neither transfer happen?', '双方都去拿交接台的东西，为什么交接没有发生？', ['public_rule7','event1'], [{'path':'metrics.handoff_conflicts','equals':1}], 'shared', 'rule'),
        ('bound_order', 'start', 'If two orders request the same recipe, which order does a cooked dish complete?', '如果两张订单要同一道菜，一份做好的菜完成哪张订单？', ['public_rule8'], [], 'shared', 'rule'),
        ('deadline', 'start', 'An order is due on turn 100. May it be served during turn 100?', '订单截止第 100 回合，在第 100 回合上菜还来得及吗？', ['public_rule8','order1'], [{'path':'orders.0.deadline','equals':100}], 'shared', 'rule'),
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
    specs.extend([
        ('waiting_dish','start','Which dish are you going to cook while waiting for my first ingredient?','你现在等我给原料，准备做什么菜？',['ai_current_dishes','next_input_ingredient'],[{'path':'orders.0.recipe','equals':'egg_tomato'}],'ai','observation'),
        ('full_menu','start','How many dishes are in the menu, and can I see the full list now?','这次菜单一共几道菜，现在能看到全部吗？',['public_rule8','order1','order2','order3'],[{'path':'total_orders','equals':3}],'shared','rule'),
        ('protein_first','start','Why must egg and meat be cooked before tomato and pepper?','为什么鸡蛋和肉必须在番茄和辣椒之前炒？',['recipe_sequence_egg_tomato','recipe_sequence_pepper_meat'],[],'ai','rule'),
        ('prep_counts','start','How many preparation interactions do egg, meat, tomato and pepper each require?','鸡蛋、肉、番茄和辣椒分别需要备料几次？',['public_rule2'],[],'shared','rule'),
        ('prepared_freshness','egg_prepared','When was this egg prepared, and when does it expire?','这份鸡蛋哪回合备好，哪回合过期？',['freshness_egg'],[{'path':'human.holding.freshness_basis','equals':'prepared'},{'path':'human.holding.prepared_turn','equals':prepared_at},{'path':'human.holding.expires_turn','equals':prepared_at+e.PREPARED_FRESH_TURNS['egg']}],'shared','observation'),
        ('early_oxidation','early_stored','Does moving the prepared tomato to another counter restart its freshness clock?','备好的番茄换个台面会重新计时吗？',['prepared_freshness_rule'],[{'path':'buffers.ai_raw.0.stage','equals':'prepared'}],'shared','rule'),
        ('prepared_twenty','start','How long do prepared ingredients stay fresh?','备好的原料能保鲜几回合？',['prepared_freshness_rule'],[{'fact_id':'prepared_freshness_rule','contains':str(e.PREPARED_FRESH_TURNS['egg'])}],'shared','rule'),
        ('freshness_last_turn','freshness_last_turn','How much freshness remains for the egg in my hand?','我手里的鸡蛋还剩几回合保鲜？',['freshness_egg'],[{'path':'human.holding.stage','equals':'prepared'},{'fact_id':'freshness_egg','contains':'1'}],'shared','observation'),
        ('freshness_expired','freshness_expired','Has this egg spoiled, or did it disappear?','这份鸡蛋变质了，还是消失了？',['freshness_egg','human_holding'],[{'path':'human.holding.stage','equals':'spoiled'},{'path':'human.holding.id','equals':states['egg_prepared']['human']['holding']['id']},{'path':'metrics.discard_penalty','equals':0}],'shared','observation'),
        ('spoiled_tomato','spoiled_tomato','Why can this tomato no longer be cooked, and did it disappear?','为什么这份番茄不能再下锅，它消失了吗？',['freshness_tomato','raw_slot1','ingredient_location_tomato','public_rule9','public_rule11'],[{'path':'buffers.ai_raw.0.stage','equals':'spoiled'},{'path':'metrics.discard_penalty','equals':0}],'shared','observation'),
        ('blocked_finished','blocked_output','Why are you taking this finished dish to the trash, and how many points will that lose?','为什么你要把这份成品拿去垃圾桶，会扣几分？',['system:ai_reason','ai_holding','handoff','public_rule10','kitchen_score'],[{'decision_path':'reason_code','equals':'discard_blocked_output'},{'path':'ai.holding.stage','equals':'finished'},{'path':'metrics.discarded_dishes','equals':0}],'ai','reason'),
        ('can_cancel_disposal','blocked_output','If I clear the handoff now, will you cancel the disposal trip?','我现在清空交接台，你会取消这次丢弃吗？',['system:ai_reason','public_rule10'],[{'decision_path':'reason_code','equals':'discard_blocked_output'}],'ai','rule'),
        ('dish_discarded_penalty','output_discarded',f'What exactly caused the {e.DISH_DISCARD_COST}-point disposal penalty just now?',f'刚才扣掉的{e.DISH_DISCARD_COST}分丢弃罚分具体是什么原因？',['kitchen_score','public_rule10'],[{'path':'metrics.discarded_dishes','equals':1},{'path':'metrics.discard_penalty','equals':e.DISH_DISCARD_COST}],'shared','observation'),
        ('every_step_cost','start','Does waiting or turning against a counter cost any points?','等待或者朝柜子原地转向会扣分吗？',['kitchen_score','public_rule1'],[{'path':'raw_score','equals':0}],'shared','rule'),
        ('serve_net_points','served',f'Why did a +{e.SERVE_POINTS} serving reward increase our score by only {e.SERVE_POINTS-e.STEP_COST}?',f'上菜奖励{e.SERVE_POINTS}分，为什么总分只增加{e.SERVE_POINTS-e.STEP_COST}？',['serving_score_rule'],[{'fact_id':'serving_score_rule','contains':str(e.SERVE_POINTS)},{'fact_id':'serving_score_rule','contains':str(e.SERVE_POINTS-e.STEP_COST)}],'shared','rule'),
        ('finished_head_label','output','Which completed dish are you actually holding now?','你现在手里实际拿着哪道成品？',['ai_current_dishes','ai_holding'],[{'path':'ai.holding.recipe','equals':'egg_tomato'},{'path':'ai.holding.stage','equals':'finished'}],'ai','observation'),
    ])
    cases = []
    for name, state_key, en, zh, identifiers, claims, subject, purpose in specs:
        state = states[state_key]
        available = {fact['id'] for fact in e.facts(state)} | {'system:ai_action','system:ai_reason','system:human_advice','system:score','system:completed_orders'}
        assert set(identifiers) <= available, (name, set(identifiers) - available)
        for language, question in [('en', en), ('zh', zh)]:
            plan = {'language':language, 'binding':{'task':state['task'],'turn':state['turn']}, 'premise':'contradicted' if name == 'false_cooked' else 'supported',
                    'clarification':None, 'intents':[{'kind':'facts','subject':subject,'purpose':purpose,'evidence_ids':identifiers}]}
            cases.append({'case_id':f'kitchen_v6_{name}_{language}', 'domain':'kitchen', 'question':question, 'language':language, 'state':deepcopy(state),
                          'expected_fact_ids':identifiers, 'expected_claims':deepcopy(claims), 'expected_kind':'facts', 'expected_plan':plan})
    for name, state_key, en, zh, actions, claims in [
        ('wait_cf','egg_load','If I wait one step, will that step alone earn a completed-order point?','如果我等待一步，仅这一步会获得完成订单分吗？',['wait'],[{'simulation_path':'raw_score_delta','equals':-1}]),
        ('movement_cf','start','If I press W now, how many game steps advance?','如果我现在按 W，游戏会推进几步？',['up'],[{'simulation_path':'steps_completed','equals':1},{'simulation_path':'raw_score_delta','equals':-1}]),
        ('unavailable_e_cf','start','What if I press E now while facing no work station?','如果我现在没有面朝工位却按 E，会怎么样？',['interact'],[{'simulation_path':'steps_completed','equals':0}]),
        ('serve_cf','serve_ready','If I serve now, how does the score change?','现在上菜，得分会怎样变化？',['interact'],[{'simulation_path':'raw_score_delta','equals':e.SERVE_POINTS-e.STEP_COST},{'simulation_path':'completed_orders_delta','equals':1}]),
        ('wait_then_serve_cf','serve_ready','What if I wait once and then serve?','先等一回合再上菜会怎样？',['wait','interact'],[{'simulation_path':'raw_score_delta','equals':e.SERVE_POINTS-2*e.STEP_COST},{'simulation_path':'completed_orders_delta','equals':1}]),
    ]:
        state = states[state_key]
        for language, question in [('en',en),('zh',zh)]:
            cases.append({'case_id':f'kitchen_v6_{name}_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
                'expected_fact_ids':[],'expected_claims':claims,'expected_kind':'counterfactual',
                'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'supported','clarification':None,
                    'intents':[{'kind':'counterfactual','subject':'human','purpose':'comparison','actions':actions,'horizon':len(actions),'evidence_ids':[]}]}})
    for name, reason, en, zh in [
        ('ambiguous','ambiguous_object','What about that other thing?','那另一个东西呢？'),
        ('unbound_history','select_frame','Why did you change your plan at an earlier unspecified point in Task 1?','你为什么在 Task 1 之前某个没指出的时刻改变计划？'),
        ('outside','unsupported_question','Who won the last World Cup?','上一届世界杯谁夺冠了？')]:
        state = states['start']
        for language, question in [('en',en),('zh',zh)]:
            cases.append({'case_id':f'kitchen_v6_{name}_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
                'expected_fact_ids':[],'expected_claims':[],'expected_kind':'clarification',
                'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'unclear','clarification':reason,'intents':[]}})
    for language, question, previous in [('en','And the other pan?','What is stove 1 doing?'),('zh','那另一口锅呢？','炉灶 1 正在做什么？')]:
        state = states['both_recipes']
        cases.append({'case_id':f'kitchen_v6_followup_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
            'previous_dialogue':[{'question':previous,'answer':next(f[language] for f in e.facts(state) if f['id']=='pot1')}],
            'expected_fact_ids':['pot2'],'expected_claims':[{'path':'pots.1.phase','equals':'protein'}],'expected_kind':'facts',
            'expected_plan':{'language':language,'binding':{'task':state['task'],'turn':state['turn']},'premise':'supported','clarification':None,
                'intents':[{'kind':'facts','subject':'shared','purpose':'observation','evidence_ids':['pot2']}]}})
    past = frames[0]
    state = states['egg_load']
    for language, question in [('en','Where was I at turn 0?'),('zh','第 0 回合我在哪里？')]:
        identifier = 'history:task2:turn0:human_position'
        cases.append({'case_id':f'kitchen_v6_bound_history_{language}','domain':'kitchen','question':question,'language':language,'state':deepcopy(state),
            'public_history':[e.public_state(past)],'expected_fact_ids':[identifier],'expected_claims':[],'expected_kind':'facts',
            'expected_plan':{'language':language,'binding':{'task':2,'turn':0},'premise':'supported','clarification':None,
                'intents':[{'kind':'facts','subject':'human','purpose':'observation','evidence_ids':[identifier]}]}})
    return cases


if __name__ == '__main__':
    cases = build()
    Path(__file__).with_name('qa_cases.json').write_text(json.dumps(cases, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(f'Wrote {len(cases)} bilingual {e.VERSION} cases from executed fixed-controller trajectories; not a semantic-model evaluation.')
