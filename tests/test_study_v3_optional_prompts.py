"""Continuing gameplay never manufactures an understanding answer or confirmation."""
import pytest
from study_v3.config import Settings
from study_v3.store import Store, StudyError
from tests.test_study_v3_store import Flow
from tests.test_study_v3_automatic_explanations import task2

@pytest.mark.parametrize('domain',['warehouse','pong','kitchen'])
@pytest.mark.parametrize('group',['A','B'])
def test_explanations_optional_but_all_ratings_required(tmp_path,domain,group):
    settings=Settings(database=str(tmp_path/'optional.db'),automatic_explanations=True,understanding_ratings=True,optional_prompts=True)
    store=Store(settings);f=Flow(store,domain,group);task2(f)
    assert f.view['optional_prompts']
    seen=[]
    while not f.view['state']['terminal']:
        if f.view['understanding_rating']:
            r=f.view['understanding_rating'];seen.append(r['id'])
            # An invalid action must not mark anything skipped.
            with pytest.raises(StudyError):f.step('invalid')
            assert store.view(f.token,f.view['instance_id'])['understanding_rating']['id']==r['id']
            for name in ['skip_prompts','next']:
                with pytest.raises(StudyError,match='understanding_rating_required'):f.command(name)
            with pytest.raises(StudyError,match='understanding_rating_required'):f.step('wait')
            f.command('understanding_rating',rating_id=r['id'],rating=3)
        f.step('wait')
    last=f.view['understanding_rating'];assert last['checkpoint']==100
    with pytest.raises(StudyError,match='understanding_rating_required'):f.command('next')
    f.command('understanding_rating',rating_id=last['id'],rating=3)
    f.command('next');assert f.view['stage']=='task3'
    exported=store.export()
    ratings=exported['understanding_ratings'];assert len(ratings)==5
    assert all(r['rating']==3 and r['submitted'] is not None for r in ratings)
    skipped={r['prompt_id'] for r in exported['prompt_skips']}
    assert not (set(seen+[last['id']])&skipped)
    assert not exported['auto_explanation_confirmations']
    if group=='A':assert all(c['id'] in skipped for c in exported['auto_explanations'])
    store.db.close()

def test_optional_answers_remain_available_and_skips_persist(tmp_path):
    settings=Settings(database=str(tmp_path/'answers.db'),automatic_explanations=True,understanding_ratings=True,optional_prompts=True)
    store=Store(settings);f=Flow(store,'warehouse','A');task2(f)
    c=f.view['automatic_explanations'][0]
    f.command('request_explanation',explanation_id=c['id'],question_id='why',source='ai_question_button')
    f.step('wait')
    c=f.view['automatic_explanations'][0];assert c['skipped'] and not c['confirmed'] and c['response'] is None and c['requested']
    while not f.view['understanding_rating']:f.step('wait')
    r=f.view['understanding_rating'];f.command('understanding_rating',rating_id=r['id'],rating=4)
    while not f.view['understanding_rating']:f.step('wait')
    turn=f.view['state']['turn']
    with pytest.raises(StudyError,match='understanding_rating_required'):f.command('skip_prompts')
    r=f.view['understanding_rating'];payload=f.command('understanding_rating',rating_id=r['id'],rating=2);store.command(f.token,'understanding_rating',payload)
    assert f.view['state']['turn']==turn and not f.view['understanding_rating']
    store.db.close();store=Store(settings)
    assert store.view(f.token,f.view['instance_id'])['optional_prompts']
    assert not store.view(f.token,f.view['instance_id'])['understanding_rating']
    assert sum(r['rating']==4 for r in store.export()['understanding_ratings'])==1
    store.db.close()
