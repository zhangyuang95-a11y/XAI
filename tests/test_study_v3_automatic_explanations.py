"""Treatment isolation, factual triggers, persistence, and exposure audit."""
import json
from dataclasses import replace
import pytest
from study_v3.automatic_explanations import candidate
from study_v3.config import Settings
from study_v3.store import Store, StudyError, encode
from tests.test_study_v3_store import Flow


def decision(**overrides):
    return {'action':'up','goal':'parcel','reason_code':'mission_progress',
            'reason_en':'I will move up toward the parcel.','reason_zh':'我会向上移动去取货。',**overrides}


def state(turn=5,**overrides):
    return {'turn':turn,'terminal':False,'events':[],**overrides}


def test_warehouse_events_and_episode_edges():
    ordinary=decision()
    detour=decision(controller_trace={'selected_ai_action':{'distance_before':3,'distance_after':4}})
    assert candidate('warehouse',state(),ordinary) is None
    assert candidate('warehouse',state(),detour)['trigger_types']==['detour']
    assert candidate('warehouse',state(),detour,detour) is None
    assert candidate('warehouse',state(),decision(action='wait',controller_trace=detour['controller_trace'])) is None
    going=decision(goal_details={'label':'charger'})
    charging=decision(action='wait',reason_code='charging',goal_details={'label':'charger'})
    assert candidate('warehouse',state(),going)['trigger_types']==['charge_travel']
    assert candidate('warehouse',state(),going,going) is None
    assert candidate('warehouse',state(),charging,going)['trigger_types']==['charge_charging']
    assert candidate('warehouse',state(),charging,charging) is None
    collision=state(events=[{'type':'collision','en':'Conflicting movements cancelled.','zh':'冲突移动已取消。'}],last_actions={'ai':'up','human':'left'})
    card=candidate('warehouse',collision,ordinary)
    assert card['trigger_types']==['collision']
    assert '4 → 5' in card['body']['en'] and 'you chose left; I chose up' in card['body']['en']
    assert card['body']['en'].endswith(ordinary['reason_en'])


def test_pong_only_current_team_target_and_stable_identity():
    target={'team_ids':['t1'],'lane':3,'human_lane':7,'remaining':4}
    first=candidate('pong',state(),decision(explanation_target=target))
    later=candidate('pong',state(6),decision(explanation_target={**target,'remaining':3}))
    assert first['trigger_key']==later['trigger_key']
    changed=candidate('pong',state(6),decision(explanation_target={**target,'human_lane':2}))
    assert first['trigger_key']!=changed['trigger_key']
    assert candidate('pong',state(),decision(explanation_target={'team_ids':[],'next_team':target})) is None


@pytest.mark.parametrize('turn,trigger',[(0,False),(1,False),(4,False),(5,True),(6,False),(10,True),(15,True)])
def test_kitchen_five_steps(turn,trigger):
    assert bool(candidate('kitchen',state(turn),decision())) is trigger
    assert candidate('kitchen',state(turn,terminal=True),decision()) is None


def task2(flow):
    flow.finish_demo()
    with flow.store.db.transaction() as db:
        db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?",(flow.view['run_id'],))
    flow.command('next')


def test_store_isolation_acknowledgement_and_resume(tmp_path):
    store=Store(Settings(database=str(tmp_path/'auto.db'),automatic_explanations=True))
    a=Flow(store,'kitchen','A');b=Flow(store,'kitchen','B')
    assert a.view['automatic_explanations_enabled'] and not a.view['automatic_explanations']
    task2(a);task2(b)
    for flow in (a,b):
        for _ in range(5):flow.step('wait')
    cards=a.view['automatic_explanations'];assert len(cards)==1
    assert cards[0]['turn']==5 and not cards[0]['displayed']
    assert b.view['automatic_explanations']==[] and not a.view['questions']
    # Visible exposure is separate from generation, survives refresh and process restart.
    store.acknowledge_automatic(a.token,a.view['instance_id'],cards[0]['id'])
    store.acknowledge_automatic(a.token,a.view['instance_id'],cards[0]['id'])
    resumed=Store(replace(store.settings,automatic_explanations=False))
    assert resumed.view(a.token,a.view['instance_id'])['automatic_explanations'][0]['displayed']
    with pytest.raises(StudyError):store.acknowledge_automatic(b.token,b.view['instance_id'],cards[0]['id'])
    # Mere visibility never unlocks a move, including after reconnecting.
    with pytest.raises(StudyError,match='explanation_confirmation_required'):a.step('wait')
    saved_turn=a.view['state']['turn']
    assert resumed.view(a.token,a.view['instance_id'])['state']['turn']==saved_turn
    with pytest.raises(StudyError):b.command('confirm_explanation',explanation_id=cards[0]['id'])
    with pytest.raises(StudyError):a.command('confirm_explanation',explanation_id='unknown')
    confirmed_payload=a.command('confirm_explanation',explanation_id=cards[0]['id'])
    assert a.view['automatic_explanations'][0]['confirmed']
    assert a.view['state']['turn']==saved_turn  # Clicking confirm does not play a move.
    store.command(a.token,'confirm_explanation',confirmed_payload)
    a.command('confirm_explanation',explanation_id=cards[0]['id'])
    for _ in range(5):payload=a.step('wait')
    retry=store.command(a.token,'action',payload)
    assert len(retry['automatic_explanations'])==2
    assert [c['turn'] for c in retry['automatic_explanations']]==[5,10]
    with pytest.raises(StudyError,match='explanation_confirmation_required'):a.step('wait')
    a.command('confirm_explanation',explanation_id=a.view['automatic_explanations'][-1]['id'])
    a.command('language',language='zh')
    assert '我' in a.view['automatic_explanations'][-1]['body']
    with store.db.transaction() as db:
        db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?",(a.view['run_id'],))
    assert not store.view(a.token,a.view['instance_id'])['automatic_explanations']
    a.command('next')
    assert a.view['stage']=='task3' and a.view['automatic_explanations']==[]
    with pytest.raises(StudyError):store.acknowledge_automatic(a.token,a.view['instance_id'],cards[0]['id'])
    exported=list(store.iter_export(mode='test'))
    automatic=[r['record'] for r in exported if r['table']=='auto_explanations']
    assert len(automatic)==2
    content=json.loads(automatic[0]['content_json'])
    assert content['version']=='event-nodes-confirm-v2' and content['decision_sha256']
    assert not any(r['table']=='questions' for r in exported)
    confirmations=[r['record'] for r in exported if r['table']=='auto_explanation_confirmations']
    assert len(confirmations)==2 and all(r['confirmed'] for r in confirmations)
    with store.db.transaction() as db:
        db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?",(a.view['run_id'],))
    a.command('next')
    items={q['id']:q for q in a.view['questionnaire']['items']}
    assert items['relevant']['allow_na'] and '解释' in items['relevant']['text']
    assert not items['predictable']['allow_na']
    resumed.db.close();store.db.close()


def test_existing_enrollment_never_changes_treatment_when_flag_enabled(tmp_path):
    settings=Settings(database=str(tmp_path/'legacy.db'))
    old=Store(settings);flow=Flow(old,'kitchen','A')
    new=Store(replace(settings,automatic_explanations=True));flow.store=new
    task2(flow)
    for _ in range(5):flow.step('wait')
    assert not flow.view['automatic_explanations_enabled']
    assert flow.view['automatic_explanations']==[]
    new.db.close();old.db.close()


def test_legacy_automatic_protocol_stays_nonblocking(tmp_path):
    store=Store(Settings(database=str(tmp_path/'v1.db'),automatic_explanations=True))
    flow=Flow(store,'kitchen','A')
    with store.db.transaction() as db:
        db.execute("UPDATE pl3_auto_explanation_settings SET version='event-nodes-v1' WHERE instance_id=?",(flow.view['instance_id'],))
    task2(flow)
    for _ in range(6):flow.step('wait')
    assert flow.view['automatic_explanations'][0]['requires_confirmation'] is False
    assert flow.view['state']['turn']==6
    store.db.close()


def test_only_first_charging_node_survives_restart_and_keeps_other_events(tmp_path):
    """Synthetic node records exercise persistence without playing a real cohort."""
    from study_v3.automatic_explanations import VERSION
    settings=Settings(database=str(tmp_path/'charge-once.db'),automatic_explanations=True)
    store=Store(settings)
    instance={'id':'synthetic-warehouse','domain':'warehouse','group_code':'A'}
    with store.db.transaction() as db:
        db.execute('INSERT INTO pl3_auto_explanation_settings VALUES(?,?,?)',(instance['id'],VERSION,0))
        going=decision(goal_details={'label':'charger'})
        store._record_automatic(db,instance,'run-1',state(5),going)
    store.db.close()
    store=Store(settings)
    charging=decision(action='wait',reason_code='charging',goal_details={'label':'charger'})
    with store.db.transaction() as db:
        # Arriving at the charger and a later new trip must not show again.
        store._record_automatic(db,instance,'run-1',state(6),charging,going)
        store._record_automatic(db,instance,'run-1',state(30),going,decision())
        collision=state(31,events=[{'type':'collision','en':'A collision occurred.','zh':'发生碰撞。'}])
        store._record_automatic(db,instance,'run-1',collision,charging,going)
        detour=decision(goal_details={'label':'charger'},controller_trace={'selected_ai_action':{'distance_before':3,'distance_after':4}})
        store._record_automatic(db,instance,'run-1',state(32),detour,decision())
        cards=[json.loads(r['content_json']) for r in db.all('SELECT content_json FROM pl3_auto_explanations WHERE run_id=? ORDER BY turn',('run-1',))]
        assert [c['trigger_types'] for c in cards]==[['charge_travel'],['collision'],['detour']]
        # A separate run has its own first charge prompt, even if it starts on the charger.
        store._record_automatic(db,instance,'run-2',state(1),charging)
        first=json.loads(db.one('SELECT content_json FROM pl3_auto_explanations WHERE run_id=?',('run-2',))['content_json'])
        assert first['trigger_types']==['charge_charging']
    store.db.close()
