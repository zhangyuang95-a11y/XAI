"""Repeated self-report scheduling, persistence and enforced response gates."""
import json
import pytest
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from study_v3 import understanding
from tests.test_study_v3_store import Flow
from tests.test_study_v3_automatic_explanations import task2


@pytest.mark.parametrize('domain,expected',[
    ('warehouse',[24,48,72,96,120]),('pong',[18,36,54,72,90]),('kitchen',[56,112,168,224,280])])
@pytest.mark.parametrize('group',['A','B'])
def test_task2_checkpoints_and_final_gate(tmp_path,domain,expected,group):
    settings=Settings(database=str(tmp_path/'ratings.db'),understanding_ratings=True)
    store=Store(settings);f=Flow(store,domain,group);task2(f)
    assert f.view['understanding_rating'] is None
    seen=[]
    while not f.view['state']['terminal']:
        f.step('wait');r=f.view['understanding_rating']
        if not r:continue
        seen.append((r['checkpoint'],r['turn']))
        assert r['rating'] is None and r['submitted'] is None and len(r['labels'])==5
        with pytest.raises(StudyError,match='understanding_rating_required'):f.step('wait')
        with pytest.raises(StudyError,match='understanding_rating_required'):f.command('next')
        if len(seen)==1:
            for bad in [0,6,True,2.5,'3',None]:
                with pytest.raises(StudyError,match='invalid_understanding_rating'):
                    f.command('understanding_rating',rating_id=r['id'],rating=bad)
            restored=Store(settings)
            assert restored.view(f.token,f.view['instance_id'])['understanding_rating']==r
            restored.db.close()
            f.command('language',language='zh')
            assert f.view['understanding_rating']['labels'][0]=='完全不理解'
        payload=f.command('understanding_rating',rating_id=r['id'],rating=4)
        assert f.view['understanding_rating'] is None
        store.command(f.token,'understanding_rating',payload) # exact retry is idempotent
    assert seen==list(zip([20,40,60,80,100],expected))
    exported=[x['record'] for x in store.iter_export(mode='test') if x['table']=='understanding_ratings']
    assert len(exported)==5 and all(r['rating']==4 and r['submitted']>=r['created'] for r in exported)
    f.command('next');assert f.view['stage']=='task3'
    for _ in range(25):f.step('wait')
    assert f.view['understanding_rating'] is None
    store.db.close()


def test_existing_instances_not_retrofitted_and_task1_not_measured(tmp_path):
    path=str(tmp_path/'legacy.db');old=Store(Settings(database=path));f=Flow(old,'pong','B');old.db.close()
    store=Store(Settings(database=path,understanding_ratings=True));f.store=store;task2(f)
    for _ in range(20):f.step('wait')
    assert f.view['understanding_rating'] is None
    new=Flow(store,'pong','B');new.finish_demo()
    for _ in range(20):new.step('wait')
    assert new.view['understanding_rating'] is None
    store.db.close()


def test_early_finish_is_one_end_rating_and_missing_checkpoints_are_not_backfilled():
    assert understanding.checkpoint({'turn':31,'max_turns':120,'terminal':True})==100
    assert understanding.checkpoint({'turn':0,'max_turns':90,'terminal':False}) is None
    assert understanding.checkpoint({'turn':19,'max_turns':90,'terminal':False}) is None
    assert understanding.checkpoint({'turn':19,'max_turns':91,'terminal':False})==20


def test_rating_precedes_explanations_and_is_owned(tmp_path):
    store=Store(Settings(database=str(tmp_path/'priority.db'),understanding_ratings=True,automatic_explanations=True))
    f=Flow(store,'kitchen','A');other=Flow(store,'kitchen','A');task2(f);task2(other)
    for _ in range(5):f.step('wait')
    c=f.view['automatic_explanations'][0]
    with store.db.transaction() as db:
        inst=db.one('SELECT * FROM pl3_instances WHERE id=?',(f.view['instance_id'],))
        run=db.one('SELECT * FROM pl3_runs WHERE id=?',(f.view['run_id'],))
        state=json.loads(run['state_json']);state['max_turns']=25 # fixture with a shared 20%/five-step node
        store._record_understanding(db,inst,run,state)
    f.view=store.view(f.token,f.view['instance_id']);r=f.view['understanding_rating'];assert r
    with pytest.raises(StudyError,match='understanding_rating_required'):
        f.command('request_explanation',explanation_id=c['id'],question_id='why')
    with pytest.raises(StudyError,match='understanding_rating_required'):store.ask(f.token,f.question())
    with pytest.raises(StudyError,match='understanding_rating_unavailable'):
        other.command('understanding_rating',rating_id=r['id'],rating=5)
    f.command('understanding_rating',rating_id=r['id'],rating=3)
    assert not f.view['understanding_rating'] and not f.view['automatic_explanations'][0]['requested']
    f.command('request_explanation',explanation_id=c['id'],question_id='why')
    assert f.view['automatic_explanations'][0]['body']
    store.db.close()
