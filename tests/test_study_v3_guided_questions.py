"""Persisted guided-question onboarding and optional explanation choices."""
import json
import pytest
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from tests.test_study_v3_store import Flow
from tests.test_study_v3_automatic_explanations import task2


@pytest.mark.parametrize('domain',['warehouse','pong','kitchen'])
def test_first_question_is_required_once_and_persists(tmp_path,domain):
    settings=Settings(database=str(tmp_path/'guided.db'),automatic_explanations=True)
    store=Store(settings);a=Flow(store,domain,'A');b=Flow(store,domain,'B')
    assert not a.view['automatic_explanations']
    task2(a);task2(b)
    assert not b.view['automatic_explanations']
    assert a.view['state']['turn']==0
    for _ in range(100):
        if a.view['automatic_explanations']:break
        a.step(None if domain=='warehouse' else 'wait')
    node_turn=a.view['state']['turn']
    if domain=='kitchen':assert node_turn==5
    c=a.view['automatic_explanations'][0];identifier=c['id']
    assert c['guided'] and c['onboarding'] and c['turn']==node_turn
    assert not c['body'] and not c['displayed'] and not c['requested']
    with pytest.raises(StudyError,match='guided_question_required'):
        a.command('confirm_explanation',explanation_id=identifier,choice='understood')
    with pytest.raises(StudyError,match='explanation_confirmation_required'):a.step('wait')
    with pytest.raises(StudyError,match='guided_question_required'):
        store.acknowledge_automatic(a.token,a.view['instance_id'],identifier)
    with pytest.raises(StudyError):b.command('request_explanation',explanation_id=identifier,question_id='why')
    with pytest.raises(StudyError):a.command('request_explanation',explanation_id=identifier,question_id='unknown')
    payload=a.command('request_explanation',explanation_id=identifier,question_id='why')
    store.command(a.token,'request_explanation',payload)
    c=a.view['automatic_explanations'][0]
    assert c['requested'] and c['body'] and not c['confirmed'] and a.view['state']['turn']==node_turn
    with pytest.raises(StudyError,match='explanation_confirmation_required'):a.step('wait')
    resumed=Store(settings)
    assert resumed.view(a.token,a.view['instance_id'])['automatic_explanations'][0]==c
    resumed.db.close()
    a.command('language',language='zh');assert '你为什么' in a.view['automatic_explanations'][0]['question']
    store.acknowledge_automatic(a.token,a.view['instance_id'],identifier)
    a.command('confirm_explanation',explanation_id=identifier,choice='explanation')
    assert a.view['state']['turn']==node_turn
    assert a.view['automatic_explanations'][0]['response']=='read_explanation'
    a.step('wait');assert a.view['state']['turn']==node_turn+1
    assert sum(c['onboarding'] for c in a.view['automatic_explanations'])==1
    store.db.close()


def test_later_nodes_allow_understood_without_exposure_or_requested_answer(tmp_path):
    store=Store(Settings(database=str(tmp_path/'choices.db'),automatic_explanations=True))
    a=Flow(store,'kitchen','A');task2(a)
    assert not a.view['automatic_explanations']
    for _ in range(5):a.step('wait')
    first=a.view['automatic_explanations'][0]['id']
    a.command('request_explanation',explanation_id=first,question_id='why')
    a.command('confirm_explanation',explanation_id=first,choice='explanation')
    for _ in range(5):a.step('wait')
    c=a.view['automatic_explanations'][-1]
    assert c['turn']==10 and not c['onboarding'] and not c['body']
    a.command('confirm_explanation',explanation_id=c['id'],choice='understood')
    c=a.view['automatic_explanations'][-1]
    assert c['confirmed'] and not c['displayed'] and not c['requested'] and not c['body']
    assert c['response']=='self_reported_understood'
    for _ in range(5):a.step('wait')
    c=a.view['automatic_explanations'][-1]
    a.command('request_explanation',explanation_id=c['id'],question_id='why')
    assert a.view['automatic_explanations'][-1]['body']
    a.command('confirm_explanation',explanation_id=c['id'],choice='explanation')
    exported=[json.loads(r['record']['content_json']) for r in store.iter_export(mode='test') if r['table']=='auto_explanations']
    assert [c['response'] for c in sorted(exported,key=lambda c:c['turn'])]==['read_explanation','self_reported_understood','read_explanation']
    assert all(c.get('question_id')=='why' for c in exported if c.get('requested_at'))
    assert not a.view['questions'] # Guided selections do not masquerade as free-text questions.
    with store.db.transaction() as db:db.execute("UPDATE pl3_runs SET status='completed' WHERE id=?",(a.view['run_id'],))
    a.command('next');assert not a.view['automatic_explanations']
    with pytest.raises(StudyError):a.command('request_explanation',explanation_id=c['id'],question_id='why')
    store.db.close()


def test_warehouse_pre_collision_guide_and_legacy_protocol(tmp_path,monkeypatch):
    from study_v3 import automatic_explanations as auto
    settings=Settings(database=str(tmp_path/'early.db'),automatic_explanations=True)
    store=Store(settings)
    with monkeypatch.context() as m:
        m.setattr(auto,'VERSION','first-node-question-guide-v4')
        old=Flow(store,'warehouse','A');task2(old)
    new=Flow(store,'warehouse','A');task2(new)
    # The initial warehouse state already has a controller-verified collision
    # counterfactual. Teach asking before any risky participant move is accepted.
    assert not old.view['automatic_explanations']
    c=new.view['automatic_explanations'][0]
    assert c['turn']==0 and c['onboarding'] and c['trigger_types']==['collision_risk']
    assert not any(e['type']=='collision' for e in new.view['state']['events'])
    new.command('request_explanation',explanation_id=c['id'],question_id='why',source='ai_question_button')
    new.command('confirm_explanation',explanation_id=c['id'],choice='explanation')
    for _ in range(15):
        pending=next((c for c in new.view['automatic_explanations'] if not c['confirmed']),None)
        if pending:new.command('confirm_explanation',explanation_id=pending['id'],choice='understood')
        new.step('wait')
    assert sum('collision_risk' in c['trigger_types'] for c in new.view['automatic_explanations'])==1
    with store.db.transaction() as db:
        row=db.one('SELECT content_json FROM pl3_auto_explanations WHERE id=?',(c['id'],))
        assert json.loads(row['content_json'])['question_source']=='ai_question_button'
    store.db.close()
