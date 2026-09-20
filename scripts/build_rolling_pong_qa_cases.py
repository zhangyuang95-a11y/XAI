#!/usr/bin/env python3
"""Rebuild current bilingual composition fixtures; not a language-model evaluation."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.pong import turnbased as pong
ROOT = Path(__file__).resolve().parents[1]


def fixture(human=2, ai=6, turns=4, small=None, small_turns=2, contacts=(2,6)):
    balls = [pong._ball('fixture-team','cooperative',list(contacts),turns)]
    if small is not None:
        balls.append(pong._ball('fixture-small','ordinary',[small],small_turns))
    return pong._new_state([balls],seed=999,task=2,human=human,ai=ai)


def build():
    rolling = pong.initial_state(730100,2)
    approach = fixture(human=1,ai=7,turns=3)
    holding = fixture()
    multi = pong._new_state([[pong._ball('early','cooperative',[2,6],2),pong._ball('later','cooperative',[1,7],4),pong._ball('small','ordinary',[5],1)]],seed=1,task=2,human=2,ai=6)
    costly = fixture(small=4,small_turns=3)
    safe = fixture(small=5,small_turns=2)
    impossible = fixture(human=7,ai=6,turns=1,small=7,small_turns=1)
    no_job = fixture(human=0,ai=8,contacts=(3,5),turns=1)
    boundary = fixture(human=0)
    one = fixture(human=1,ai=7,turns=1)
    first = pong.step(rolling,pong.human_advisor(rolling))
    after = pong.step(first,pong.human_advisor(first))
    history = [pong.public_state(s) for s in (rolling,first,after)]
    snapshots = {}
    state = pong.initial_state(731100,2)
    for turn in range(77):
        if turn in (8,76):
            snapshots[turn] = deepcopy(state)
        if turn < 76:
            state = pong.step(state,pong.human_advisor(state))
    five = pong._new_state([[pong._ball("team","cooperative",[2,6],1),pong._ball("mine","ordinary",[6],1),pong._ball("yours","ordinary",[2],1)]],seed=1,task=2,human=2,ai=6)
    cases = []
    def add(name, state, en, zh, ids=(), claims=(), *, subject='shared', purpose='observation', premise='supported', actions=None, horizon=1, clarification=None, public_history=None, dialogue=None, object_id=None):
        for lang,question in [('en',en),('zh',zh)]:
            kind = 'clarification' if clarification else 'counterfactual' if actions else 'facts'
            intents = [] if clarification else [{'kind':kind,'subject':'human' if actions else subject,'purpose':'comparison' if actions else purpose,'evidence_ids':list(ids)}]
            if object_id:
                intents[0]['object_id'] = object_id
            if actions:
                intents[0].update(actions=actions,horizon=horizon)
            case = {'case_id':f'pong-rolling-{name}-{lang}','domain':'pong','question':question,'language':lang,'state':deepcopy(state),
                    'expected_kind':kind,'expected_fact_ids':list(ids),'expected_claims':deepcopy(list(claims)),
                    'expected_plan':{'language':lang,'binding':{'task':state['task'],'turn':state['turn']},'premise':'unclear' if clarification else premise,'clarification':clarification,'intents':intents}}
            if public_history:
                case['public_history'] = deepcopy(public_history)
            if dialogue:
                case['previous_dialogue'] = deepcopy(dialogue[lang])
            cases.append(case)
    add('geometry',rolling,'How tall is this court, and how many lanes does it have?','这个球场有多高、一共有几道？',['public_rule:0'],[{'path':'height','equals':12},{'path':'lanes','equals':9}],purpose='rule')
    add('small-speed',rolling,'How far does a small ball fall after each action?','每操作一次，小球会下降几格？',['ball:s1'],[{'path':'balls.0.vy','equals':2}])
    add('team-speed',rolling,'Do the team balls move as fast as the small balls?','合作球和小球下落得一样快吗？',['public_rule:1'],purpose='rule')
    add('ten-balls',rolling,'How many small and team balls can be on screen together?','屏幕上最多会同时有几个小球和合作球？',['public_rule:4'],purpose='rule')
    add('small-points',rolling,'What earns one point for a small ball?','怎样接小球才能得到一分？',['public_rule:2'],purpose='rule')
    add('team-points',rolling,'What must both of us do to earn three points from a team ball?','我们两个人怎样配合才能从合作球得到三分？',['public_rule:3'],purpose='rule')
    add('duplicate-small',rolling,'If both paddles cover the same small ball, do we earn two points?','如果两块球拍一起接同一个小球，会得两分吗？',['public_rule:2'],premise='contradicted',purpose='rule')
    add('same-team-side',holding,'We can both stay on one side of the team ball and still catch it, right?','我们都站在合作球同一侧也能接住，对吗？',['public_rule:3'],premise='contradicted',purpose='rule')
    add('controls',rolling,'Which keys move me, and which key makes me wait?','哪些按键可以移动，哪个按键是等待？',['public_rule:0'],purpose='rule')
    add('question-time',rolling,'Will the balls keep falling while I read your answer?','我阅读你的回答时，球还会继续下落吗？',['public_rule:5'],purpose='rule')
    add('rolling-supply',rolling,'Does the game wait for every ball to disappear before adding another?','游戏是不是必须等所有球消失后才补入下一颗？',['public_rule:4'],premise='contradicted',purpose='rule')
    add('ai-next',approach,'What will you do next, and why are you choosing that direction?','你下一步会做什么，为什么选择那个方向？',['system:ai_action','system:ai_reason'],[{'decision_path':'action','equals':'left'}],subject='ai',purpose='reason')
    add('human-advice',approach,'Which move should I choose now to support your catch?','我现在应该怎么移动才能配合你接球？',['system:human_advice'],[{'decision_path':'human_action','equals':'right'}],subject='human',purpose='advice')
    add('first-commitment',multi,'Which side have you selected for the earlier team ball?','对于先到的大球，你已经选定接哪一侧？',['assignment:0'],[{'decision_path':'assignments.0.locked','equals':True}])
    add('later-tentative',multi,'Is your side for the later team ball fixed already?','后面那颗合作球的分工现在就固定了吗？',['assignment:1'],[{'decision_path':'assignments.1.locked','equals':False}],premise='contradicted')
    add('two-deadlines',multi,'Can we reach the two planned team-ball contacts in sequence?','按当前分工，我们能依次赶到两个合作球的接球位置吗？',['assignment:0','assignment:1','decision'])
    add('safe-small',safe,'Why can you take this small ball without abandoning the team ball?','你为什么可以先接这颗小球而不放弃合作球？',['system:ai_reason','comparison:2'],[{'decision_path':'reason_code','equals':'small_then_return'}],subject='ai',purpose='reason')
    add('unsafe-small',costly,'Why are you holding your lane instead of chasing the small ball in the middle?','你为什么守住所在道，而不追中间的小球？',['system:ai_reason','comparison:2'],[{'decision_path':'action','equals':'wait'}],subject='ai',purpose='reason')
    add('missed-opportunity',impossible,'The team ball is no longer reachable from both sides. What are you doing instead?','合作球的两侧已经无法及时覆盖了，你会转而做什么？',['system:ai_reason','system:ai_action'],[{'decision_path':'action','equals':'right'}],subject='ai',purpose='reason')
    add('no-job',no_job,'Why are you staying still when balls are still visible?','还有球在画面上，你为什么却停在原地？',['system:ai_reason'],[{'decision_path':'goal','equals':'hold_position'}],subject='ai',purpose='reason')
    add('left-edge',boundary,'I am at the left edge. Which movement choices are available?','我已经在最左边了，现在有哪些合法移动选择？',['system:available_actions','position'])
    add('wrong-wait-premise',approach,'You are about to wait rather than move left, correct?','你接下来会等待而不是左移，对吧？',['system:ai_action'],[{'decision_path':'action','equals':'left'}],subject='ai',purpose='action',premise='contradicted')
    add('current-catch',after,'What happened to the first small ball on the previous move?','刚才那一步，第一颗小球发生了什么？',['event:0'],[{'path':'events.0.type','equals':'caught'},{'path':'events.0.points','equals':1}])
    add('history-position',after,'Where was I before either of those two steps?','在这两步发生之前，我在哪里？',['history:task2:turn0:human_position'],public_history=history)
    add('history-event',after,'How many points did the catch at Task 2 turn 2 earn?','Task 2 第2回合的那次接球获得了几分？',['history:task2:turn2:event0'],public_history=history)
    add('followup-other-side',multi,'For that earlier-ball plan, which contact do I need to cover?','沿用刚才先到球的那个计划，我需要覆盖哪个接触点？',['assignment:0'],dialogue={'en':[{'question':'Which ball are you handling first?','answer':'I am first handling early.'}],'zh':[{'question':'你先处理哪颗球？','answer':'我先处理 early。'}]})
    add('new-spawn-boundary',rolling,'If I wait for four steps, how far can you verify without using unseen balls?','如果我连续等待四步，在不使用尚未出现的球的情况下能验证到哪里？',claims=[{'simulation_path':'steps_completed','equals':2},{'simulation_path':'stopped_at_public_boundary','equals':True}],actions=['wait'],horizon=4)
    add('catch-once',one,'If I move right now, do we cover both contacts and gain three points?','如果我现在右移，我们能覆盖两侧并得到三分吗？',claims=[{'simulation_path':'raw_score_delta','equals':3}],actions=['right'])
    add('miss-human-side',one,'If I wait here instead, does the arriving team ball score?','如果我留在这里等待，到达的合作球会得分吗？',claims=[{'simulation_path':'raw_score_delta','equals':0}],actions=['wait'])
    add('four-step-safe',safe,'If I keep waiting for four steps, can you catch the small ball and our team ball?','如果我连续等待四步，你能接住小球并和我接住合作球吗？',claims=[{'simulation_path':'raw_score_delta','equals':4},{'simulation_path':'steps_completed','equals':4}],actions=['wait'],horizon=4)
    add('ambiguous-object',multi,'Why did you ignore that one?','你为什么忽略那一个？',clarification='ambiguous_object')
    add('unseen-turn',rolling,'Why did you switch sides on the turn I have not played yet?','在我还没玩到的那个回合，你为什么换边？',clarification='select_frame')
    add('real-turn76-why',snapshots[76],'Why will you move left next, and which small and team balls does that target cover together?','你下一步为什么左移，这个目标能同时接哪一个小球和大球？',['system:ai_reason'],[{'decision_path':'action','equals':'left'}],subject='ai',purpose='reason')
    add('real-turn76-why-not-right',snapshots[76],'Why not move right on your next step?','你下一步为什么不右移？',['alternative_right'],subject='ai',purpose='comparison')
    add('real-turn76-arrival-payoff',snapshots[76],'What is the selected plan’s total for the balls arriving in two turns?','当前所选计划中，两回合后这一批球一共能得几分？',['arrival_payoff:2'])
    add('real-turn76-team-contact',snapshots[76],'For t26, which lane should I cover and which will you cover?','对于t26，我该覆盖哪一道，你会覆盖哪一道？',['ball_plan:t26'],purpose='assignment',object_id='t26')
    add('real-turn8-skipped-team',snapshots[8],'For t4, which lane should I cover and which will you cover?','对于t4，我该覆盖哪一道，你会覆盖哪一道？',['ball_plan:t4'],purpose='assignment',object_id='t4')
    add('real-turn8-nearest',snapshots[8],'Which ball am I closest to, how many turns until it arrives, and how can I coordinate with you?','我离哪个球最近，还剩几回合，我怎样配合你？',['human_nearest_ball','system:human_advice'],subject='human',purpose='advice')
    add('five-point-both-smalls',five,'If we cover both team contacts, how are my small ball, your small ball, and the team ball counted?','我们覆盖合作球两侧时，我的小球、你的小球、合作球分别计几分？',['arrival_payoff:1'])
    add('five-point-physical-cf',five,'If I wait for this one arrival, can our two small balls plus the team ball really give five points?','如果我等待这一次到达，我们两个小球加合作球真的能得五分吗？',claims=[{'simulation_path':'raw_score_delta','equals':5}],actions=['wait'])
    assert len(cases) == 80 and len({c['question'] for c in cases}) == 80
    return cases


if __name__ == '__main__':
    path = ROOT/'configs/study_v3_qa_cases.json'
    existing = json.loads(path.read_text())
    existing['cases'] = [c for c in existing['cases'] if c.get('domain',c['state']['domain']) != 'pong'] + build()
    existing['evaluation_status'] = 'Injected composition fixtures only; they do not establish real-provider question understanding.'
    existing['version'] = 'three-domain-qa-cases.v3.7'
    path.write_text(json.dumps(existing,ensure_ascii=False,indent=2)+'\n')
    print('Rebuilt 80 current Pong composition fixtures; external domain case files retained.')
