"""Operation tutorial uses physical transitions but never formal task scores."""
import json
from copy import deepcopy
import pytest
from domains.kitchen import engine as k
from study_v3 import kitchen_tutorial as tut, RELEASE_ID
from study_v3.store import Store, StudyError
from study_v3.config import Settings
from tests.test_study_v3_store import Flow


def act(t,action):
    out=tut.apply(t,'action',action)
    assert all(p['status']=='empty' for p in out['state']['pots'])
    assert tut.view(out)['state']['ai']['current_cooking']==[]
    return out


def interact_at(t,station):
    state=t['state'];route=k._route(k._pos(state['human']),state['human']['facing'],station)
    for action in route:t=act(t,action)
    return act(t,'interact')


def complete_practice(command):
    command('start')
    command('action','down');command('action','left');command('action','wait')
    for _ in range(150):
        t=command('read');i=t['index'];state=t['state']
        if t['completed']:return
        if i==1:target='tomato'
        elif i==2:target='prep'
        elif i==3:
            target='handoff' if not t['progress'].get('handoff_item_taken') else 'human_buffer'
        elif i==4:
            held=state['human']['holding'];target='handoff' if held is None else 'plate' if held['stage']=='finished' else 'serve'
        elif i==5:target='trash'
        elif i==6:command('action','wait');continue
        else:raise AssertionError(t)
        route=k._route(k._pos(state['human']),state['human']['facing'],target)
        command('action',route[0] if route else 'interact')
    pytest.fail('Tutorial did not finish')


def test_all_seven_sections_are_actual_user_interactions_without_cooking():
    t=tut.initial();seen=[]
    def command(kind,action=None):
        nonlocal t
        if kind=='read':return t
        old=t['index'];t=tut.apply(t,kind,action)
        assert all(p['status']=='empty' for p in t['state']['pots'])
        if t['index']!=old:seen.append(old)
        return t
    complete_practice(command)
    assert seen==list(range(6))
    assert t['completed'] and not t['playing']
    assert t['last_transition']['after']['turn']==1


def test_pause_invalid_interact_and_retry_are_not_formal_actions():
    t=tut.initial();t=tut.apply(t,'start');before=deepcopy(t['state'])
    t=tut.apply(t,'action','interact')
    assert t['state']==before and t['feedback']
    t=tut.apply(t,'action','down');t=tut.apply(t,'pause')
    with pytest.raises(ValueError,match='tutorial_paused'):tut.apply(t,'action','wait')
    t=tut.apply(t,'retry')
    assert t['state']['turn']==0 and t['progress']=={} and t['playing']


def test_tutorial_records_are_persisted_idempotent_exported_and_isolated(tmp_path):
    settings=Settings(database=str(tmp_path/'practice.db'))
    store=Store(settings);f=Flow(store,'kitchen')
    payload=f.command('tutorial',command='start')
    f.command('tutorial',command='action',action='down')
    prior=deepcopy(f.view)
    # The first command is safely retried after later actions, with no duplication.
    assert store.command(f.token,'tutorial',payload)==prior
    before=f.view['state']['turn']
    f.command('language',language='zh');assert f.view['state']['turn']==before
    f.command('tutorial',command='pause')
    store.db.close();store=Store(settings);f.store=store
    recovered=store.recover_view(f.token,'kitchen')
    assert recovered['tutorial']['playing'] is False
    assert recovered['state']['turn']==before
    assert recovered['task_runs']==[] and recovered['can_ask'] is False
    rows=list(store.iter_export(RELEASE_ID,'test'))
    assert len([r for r in rows if r['table']=='tutorial_events'])==3
    assert not any(r['table'] in ('runs','frames') for r in rows)
    f.view=recovered;f.command('demo_skip')
    assert f.view['stage']=='task1' and f.view['state']['turn']==0
    assert f.view['state']['score']['raw_score']==0
    with store.db.transaction(read_only=True) as db:
        saved=json.loads(db.one('SELECT state_json FROM pl3_tutorials WHERE instance_id=?',(f.view['instance_id'],))['state_json'])
    assert saved['skipped'] and not saved['completed']
    store.db.close()


def test_complete_practice_advances_to_fresh_task_only_on_start(tmp_path):
    store=Store(Settings(database=str(tmp_path/'full.db')));f=Flow(store,'kitchen','B')
    def command(kind,action=None):
        if kind!='read':f.command('tutorial',command=kind,**({'action':action} if action else {}))
        with store.db.transaction(read_only=True) as db:
            return json.loads(db.one('SELECT state_json FROM pl3_tutorials WHERE instance_id=?',(f.view['instance_id'],))['state_json'])
    complete_practice(command)
    assert f.view['stage']=='demo' and not f.view['task_runs']
    f.command('next');assert f.view['stage']=='task1' and f.view['state']['turn']==0
    assert f.view['state']['rule_metadata']['score']['served']==30
    with pytest.raises(StudyError,match='wrong_stage'):f.command('tutorial',command='start')
    store.db.close()


def test_group_does_not_change_tutorial_and_cannot_bypass_with_old_playback(tmp_path):
    store=Store(Settings(database=str(tmp_path/'groups.db')))
    a=Flow(store,'kitchen','A');b=Flow(store,'kitchen','B')
    assert a.view['tutorial']==b.view['tutorial']
    assert a.view['public_help']==b.view['public_help']
    with pytest.raises(StudyError,match='interactive_tutorial_required'):a.command('demo_next')
    with pytest.raises(StudyError,match='finish_demo'):a.command('demo_finish')
    with pytest.raises(StudyError,match='finish_demo'):a.command('next')
    store.db.close()
