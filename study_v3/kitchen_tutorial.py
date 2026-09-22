"""Persisted, keyboard-driven operation practice; never a cooking demonstration.

Every action uses the real kitchen transition with a neutral waiting teammate.
Independent fixtures are explicitly labelled and never enter formal task runs.
"""
from copy import deepcopy
from domains.kitchen import engine as k

VERSION = 'kitchen-operations-tutorial.v2'
COUNT = 6

def public_help():
    r = k.rule_metadata()
    return [
        {'id':'controls','en':'WASD moves and turns. E uses only the adjacent station you face. A blocked move turns you and uses one turn. Space waits. An unavailable E explains the problem without using a turn. Reading, replay and permitted questions do not advance time.', 'zh':'WASD 移动及转向，E 只操作正前方相邻工位。撞到工位时原地转向并推进一回合。空格等待。无合法交互的 E 只提示原因，不推进时间。阅读、回看和获准的提问不消耗回合。'},
        {'id':'prep','en':f"Take ingredients from the four labelled cupboards. Face Preparation and press E repeatedly: tomato/pepper {k.PREPARE_TURNS['tomato']} times, egg {k.PREPARE_TURNS['egg']}, meat {k.PREPARE_TURNS['meat']}. Waiting does not prepare food; leaving keeps progress.", 'zh':f"从四个标明原料的柜子取料。面向备料台多次按 E：番茄／辣椒 {k.PREPARE_TURNS['tomato']} 次，鸡蛋 {k.PREPARE_TURNS['egg']} 次，肉 {k.PREPARE_TURNS['meat']} 次。等待不备料，中途离开保留进度。"},
        {'id':'capacity','en':'Each teammate carries one item. The handoff counter and your counter hold one each; the AI ingredient counter has two slots. Full counters cannot be overwritten or swapped. Simultaneous handoff interactions both fail; an item placed this turn cannot be taken by the other teammate in the same turn.', 'zh':'每人手持一件；交接台和你的暂存台各放一件，AI 原料台有两个槽位。满台不能覆盖或直接交换。双方同回合操作交接台都会失败；本回合放下的物品不能被对方同回合拿走。'},
        {'id':'containers','en':'Temporary plates hold cooked components. A completed dish arrives in an output container: take it, use Serving plates, then use the Serving hatch. Cooking alone does not complete an order.', 'zh':'熟料临时盘用于盛放熟配料。成品以出锅容器交接：取走后在正式装盘台装盘，再到上菜口上菜。炒好不等于完成订单。'},
        {'id':'freshness','en':f"Prepared ingredients spoil {r['prepared_fresh_turns']} turns after preparation, wherever they are held or stored. Transfers do not reset the clock. They must enter a pan before expiry; pan food follows cooking/burning rules instead. Ready pan food burns after {k.BURN_TURNS} full turns. Spoiled items remain in place; carry them to the bin and press E.", 'zh':f"备好原料从完成备料起 {r['prepared_fresh_turns']} 回合变质，不论手持或暂存，转移不重置计时。必须在过期前下锅；下锅后改用炒制及烧糊规则，炒好后留满 {k.BURN_TURNS} 回合烧糊。变质物品留在原处，需拿到垃圾桶前按 E 丢弃。"},
        {'id':'raw','en':f"Unprepared ingredients keep the existing limits: {k.RAW_FRESH_TURNS} turns after taking them, or more than {k.STORAGE_FRESH_TURNS} untouched turns on your counter or the AI ingredient slots. Completing preparation starts the prepared-ingredient clock.", 'zh':f"未备好生料沿用原规则：取出后 {k.RAW_FRESH_TURNS} 回合，或在你的暂存台／AI 原料槽连续存放超过 {k.STORAGE_FRESH_TURNS} 回合变质。完成备料后开始备好原料的计时。"},
        {'id':'menu','en':f"Five menu dishes; at most two identical dishes consecutively. Budget: {r['max_turns']} turns. Deadlines: {', '.join(map(str,r['order_deadlines']))}, counted from task start. Serving on the deadline is accepted; an unfinished order expires at that turn's end. Dishes remain bound to their orders; serving need not follow menu order.", 'zh':f"每轮五道菜，相同菜最多连续两道，上限 {r['max_turns']} 回合。截止回合为 {'、'.join(map(str,r['order_deadlines']))}，从本局开始累计。截止当回合仍可上菜，回合结束未完成才过期。菜与订单绑定，但不要求按菜单顺序上菜。"},
        {'id':'score','en':f"On-time serving +{k.SERVE_POINTS}; every turn −{k.STEP_COST}; discarding one component −{k.INGREDIENT_DISCARD_COST}, a combined dish −{k.DISH_DISCARD_COST}. Serving itself uses a turn, so its net change is +{k.SERVE_POINTS-k.STEP_COST}. Spoilage, burning and expiry have no extra penalty until actual disposal. Scores may be negative.", 'zh':f"按时上菜 +{k.SERVE_POINTS}，每回合 −{k.STEP_COST}，丢单份配料 −{k.INGREDIENT_DISCARD_COST}，丢合成菜品 −{k.DISH_DISCARD_COST}。上菜也消耗一回合，因此净加 {k.SERVE_POINTS-k.STEP_COST} 分。变质、烧糊和订单过期本身不额外扣分，实际丢弃才扣分；允许负分。"},
    ]

def _portion(ingredient, stage='raw', prepared_turn=0):
    return {'id':'practice-portion','ingredient':ingredient,'ingredients':[ingredient],
            'recipe':k.INGREDIENT_RECIPE[ingredient],'stage':stage,
            'prepare_progress':k.PREPARE_TURNS[ingredient] if stage=='prepared' else 0,
            'components':['practice-portion'],'order_id':None,'pot_id':None,'container':None,'was_buffered':False,
            'acquired_turn':0,'freshness_started_turn':prepared_turn if stage=='prepared' else 0,
            'fresh_until':prepared_turn+k.PREPARED_FRESH_TURNS[ingredient] if stage=='prepared' else k.RAW_FRESH_TURNS,
            **({'prepared_turn':prepared_turn,'expires_turn':prepared_turn+k.PREPARED_FRESH_TURNS[ingredient]} if stage=='prepared' else {}),
            'freshness_basis':'prepared' if stage=='prepared' else 'acquired'}

def _scene(index, ingredient='tomato'):
    state=k.initial_state(1000,1)
    state['human'].update(x=2,y=3,facing='right',holding=None)
    state['ai'].update(x=5,y=3,facing='left',holding=None)
    if index==2:
        state['human']['holding']=_portion(ingredient)
    elif index==3:
        state['human']['holding']=_portion(ingredient,'prepared')
    elif index==4:
        recipe=state['orders'][0]['recipe'];spec=k.RECIPES[recipe]
        food=_portion(spec['protein']);food.update(id='practice-dish',stage='finished',recipe=recipe,
            ingredients=[spec['protein'],spec['vegetable']],components=['practice-component-1','practice-component-2'],
            order_id=state['orders'][0]['id'],pot_id='pot1',container='output_container')
        state['handoff']=food
    elif index==5:
        # Explicit independent example, advance only by real input.
        state['turn']=k.PREPARED_FRESH_TURNS['tomato']-2;state['human']['holding']=_portion('tomato','prepared')
    return state

def initial():
    return {'version':VERSION,'index':0,'playing':False,'completed':False,'skipped':False,
            'progress':{},'ingredient':'tomato','state':_scene(0),'feedback':None}

def normalize(tutorial):
    """Retire v1's reading-only seventh segment without touching game state.

    Reads adapt a copy. The next participant command records the old and new
    tutorial snapshots, preserving the original practice history.
    """
    t=deepcopy(tutorial)
    if t['version']=='kitchen-operations-tutorial.v1':
        t['version']=VERSION
        if t['index']==COUNT:
            t.update(index=COUNT-1,completed=True,playing=False,feedback=None)
            t['retired_segment']='menu_and_scoring'
    return t

def _goal(index):
    fresh=k.rule_metadata()['prepared_fresh_turns']
    goals=[
        ('Move with WASD, face a counter by pressing toward it, then press Space to wait. E interacts only in front of your arrow.', '用 WASD 移动，再朝工位按方向键完成转向，然后按空格等待。E 只操作箭头前方。'),
        ('Find any of the four ingredient cupboards. Face it and press E to take one portion. No ingredient order is prescribed in this practice.', '找到任意一个原料柜，面向它按 E 取一份原料。本练习不规定原料顺序。'),
        ('Face the Preparation counter and press E until this portion is ready. Watch the progress label; waiting does not prepare it.', '面向备料台，多次按 E 直到这份原料备好。观察进度提示；等待不会自动备料。'),
        ('Practise placing and taking back an item on both the central handoff and your counter. The AI ingredient counter has two slots; your counter and the handoff hold one each.', '分别在中央交接台和你的暂存台练习放下、取回。AI 原料台有两槽；交接台和你的暂存台各放一件。'),
        ('Independent example: this finished dish already exists. Take it from the handoff, use Serving plates, then the Serving hatch. Its cooking process is not shown.', '独立操作示例：成品已放在交接台，不展示制作过程。取走成品，到正式装盘台装盘，再到上菜口上菜。'),
        (f'Independent example: this prepared tomato is already {fresh-2} turns old. Move or wait to see it spoil at age {fresh}, then face the trash bin and press E. Transfers never reset its age.', f'独立示例：这份备好的番茄已过 {fresh-2} 回合。移动或等待观察其在第 {fresh} 回合变质，然后面向垃圾桶按 E 丢弃。转移不重置寿命。'),
    ]
    en,zh=goals[index]
    return {'en':en,'zh':zh}

HIGHLIGHTS=[['prep'],['egg','tomato','meat','pepper'],['prep'],['handoff','human_buffer','ai_raw'],['handoff','plate','serve'],['trash']]

def view(tutorial):
    tutorial=normalize(tutorial)
    out={key:deepcopy(tutorial[key]) for key in ('version','index','playing','completed','skipped','progress','feedback')}
    state=k.public_state(tutorial['state'])
    state['ai']['current_cooking']=[]
    state['tutorial']=True
    out.update(state=state,total_segments=COUNT,goal=_goal(tutorial['index']),highlights=HIGHLIGHTS[tutorial['index']])
    return out

def apply(tutorial,command,action=None):
    t=normalize(tutorial)
    t['feedback']=None
    t.pop('last_transition',None)
    if command=='finish':
        if not (t['completed'] or t['skipped']):raise ValueError('tutorial_incomplete')
        return t
    if command=='start':
        if not t['completed']:t['playing']=True
        return t
    if command=='pause':
        t['playing']=False;return t
    if command=='retry':
        t.update(state=_scene(t['index'],t['ingredient']),progress={},completed=False,playing=True)
        return t
    if command=='skip':
        t.update(playing=False,skipped=True);return t
    if command!='action':raise ValueError('invalid_tutorial_command')
    if not t['playing'] or t['completed']:raise ValueError('tutorial_paused')
    state=t['state']
    if action not in (*k.MOVES,'wait','interact'):raise ValueError('illegal_action')
    if action not in k.legal_actions(state):
        t['feedback']={'en':k.interaction_label(state),'zh':k.interaction_label(state,language='zh')}
        return t
    if state['terminal']:
        t['playing']=False;t['feedback']={'en':'This practice scene ended. Retry this section to continue.','zh':'此练习场景已结束，请重练当前段继续。'}
        return t
    # Neutral teammate: no recipe policy, hidden strategy or cooking trajectory.
    nxt=k.step(state,action,{'action':'wait','memory':{},'reason_code':'tutorial_neutral'})
    p=t['progress'];i=t['index']
    if i==0:
        if (nxt['human']['x'],nxt['human']['y'])!=(state['human']['x'],state['human']['y']):p['moved']=True
        if action in k.MOVES and (nxt['human']['x'],nxt['human']['y'])==(state['human']['x'],state['human']['y']) and k._front(nxt['human']):p['faced_counter']=True
        if action=='wait':p['waited']=True
        done=all(p.get(key) for key in ('moved','faced_counter','waited'))
    elif i==1:
        done=any(e['type']=='ingredient_taken' for e in nxt['events'])
        if done:t['ingredient']=nxt['human']['holding']['ingredient']
    elif i==2:
        done=bool(nxt['human']['holding'] and nxt['human']['holding']['stage']=='prepared')
    elif i==3:
        for e in nxt['events']:
            if e['type'] in ('item_placed','item_taken') and e.get('station') in ('handoff','human_buffer'):
                p[e['station']+'_'+e['type']]=True
        done=all(p.get(station+'_'+event) for station in ('handoff','human_buffer') for event in ('item_placed','item_taken'))
    elif i==4:done=any(e['type']=='served' for e in nxt['events'])
    elif i==5:done=any(e['type']=='waste' and e['item']['stage']=='spoiled' for e in nxt['events'])
    else:raise ValueError('invalid_tutorial_segment')
    t['state']=nxt
    t['last_transition']={'before':state,'after':nxt,'human_action':action,'ai_action':'wait'}
    if done:
        if i==COUNT-1:t.update(completed=True,playing=False)
        else:t.update(index=i+1,state=_scene(i+1,t['ingredient']),progress={})
    return t
