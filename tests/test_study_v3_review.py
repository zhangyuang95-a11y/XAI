"""Invitation boundaries and complete review flow use disposable test databases."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.cookies import SimpleCookie

import pytest

from study_v3 import review, prolific
from study_v3.config import Settings
from study_v3.server import make_server, COOKIE
from study_v3.store import Store, StudyError
from tests.test_study_v3_store import Flow, RecordingExplainer


@pytest.fixture
def store(tmp_path):
    settings=Settings(database=str(tmp_path/'review.sqlite'),mode='pilot',persistent=True,
        verified=True,admin_token='local-test-key',llm_base_url='https://test.invalid',llm_model='test',
        automatic_explanations=True,understanding_ratings=True,optional_prompts=True,
        prolific_study_id='f'*24,prolific_completion_code='TESTONLY')
    result=Store(settings,RecordingExplainer())
    result.set_qa_health(True)
    yield result
    result.db.close()


def invitation(store,domain):
    link=review.mint(store.settings,domain)
    return dict(invitation=link['url'].split('#invite=')[1],consent=True,age_21=True,
                consent_version=prolific.CONSENT_VERSION)


@pytest.mark.parametrize('domain',['warehouse','pong','kitchen'])
def test_complete_flow_matches_formal_protocol_and_excludes_recruitment(store,domain):
    payload=invitation(store,domain)
    token,view=review.enrol(store,domain,payload)
    assert view['group']=='A' and view['mode']=='preview' and view['stage']=='demo'
    assert view['task_runs']==[] and view['questionnaire_optional']
    assert view['automatic_explanations_enabled'] and view['rewards']['base_pence']==300
    assert review.enrol(store,domain,payload,token)[1]['instance_id']==view['instance_id']
    assert review.authorize(store,domain,token)==view['instance_id']
    f=object.__new__(Flow);f.store,f.token,f.view,f.domain=store,token,view,domain
    f.finish_demo()
    for task in (1,2,3):
        assert f.view['stage']=='task'+str(task)
        for _ in range(400):
            rating=f.view['understanding_rating']
            if rating:
                with pytest.raises(StudyError,match='understanding_rating_required'):f.command('next')
                f.command('understanding_rating',rating_id=rating['id'],rating=4)
            guide=next((c for c in f.view['automatic_explanations'] if c['four_step'] and not c['confirmed']),None)
            if guide:
                f.command('open_question_guide',explanation_id=guide['id'])
                f.command('request_explanation',explanation_id=guide['id'],question_id='why')
                f.command('confirm_explanation',explanation_id=guide['id'],choice='explanation')
            if f.view['state']['terminal']:break
            f.step('wait')
        assert f.view['state']['terminal']
        f.command('next')
    assert f.view['stage']=='questionnaire' and f.view['questionnaire_optional']
    f.command('questionnaire',answers={},feedback='')
    assert f.view['stage']=='completed' and 'completion_code' not in f.view
    assert [r['task'] for r in f.view['task_runs']]==[1,2,3]
    assert not prolific.payment_records(store)
    assert not store.export(mode='pilot')['instances']


def test_signed_scope_expiry_concurrent_capacity_and_cross_session(store,monkeypatch):
    p=invitation(store,'pong')
    for domain,payload in [('kitchen',p),('pong',{**p,'invitation':p['invitation']+'bad'})]:
        with pytest.raises(StudyError,match='invalid_review_invitation'):review.enrol(store,domain,payload)
    with pytest.raises(StudyError,match='consent_required'):
        review.enrol(store,'pong',{**p,'consent':False})
    with ThreadPoolExecutor(max_workers=6) as pool:
        sessions=list(pool.map(lambda _:review.enrol(store,'pong',p),range(6)))
    with pytest.raises(StudyError,match='review_invitation_full'):review.enrol(store,'pong',p)
    a,b=sessions[:2]
    with pytest.raises(StudyError,match='study_not_found'):review.authorize(store,'pong',a[0],b[1]['instance_id'])
    with pytest.raises(StudyError,match='review_session_required'):review.authorize(store,'kitchen',a[0])
    now=review.time.time();monkeypatch.setattr(review.time,'time',lambda:now+31*86400)
    with pytest.raises(StudyError,match='invalid_review_invitation'):review.enrol(store,'pong',p)


def test_http_separate_game_cookies_and_no_admin_or_real_session_access(store):
    server=make_server(store.settings,port=0,explainer=RecordingExplainer());server.store.set_qa_health(True)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    def request(method,path,payload=None,cookie='',admin=False):
        headers={'Content-Type':'application/json','Cookie':cookie}
        if admin:headers['Authorization']='Bearer '+store.settings.admin_token
        c=HTTPConnection('127.0.0.1',server.server_port)
        try:
            c.request(method,path,body=json.dumps(payload) if payload is not None else None,headers=headers)
            response=c.getresponse();body=response.read();h=dict(response.getheaders())
            return response.status,json.loads(body),h
        finally:c.close()
    try:
        assert request('POST','/api/study/admin/review-link',{'domain':'pong'})[0]==403
        assert request('GET','/api/review/pong/info')[0]==200
        cookies=[];ids=[]
        for domain in ('pong','kitchen'):
            status,link,_=request('POST','/api/study/admin/review-link',{'domain':domain},admin=True)
            assert status==200
            p=dict(invitation=link['url'].split('#invite=')[1],consent=True,age_21=True,consent_version=prolific.CONSENT_VERSION)
            status,v,h=request('POST','/api/review/'+domain+'/enrol',p)
            assert status==200 and v['stage']=='demo'
            parsed=SimpleCookie();parsed.load(h['Set-Cookie']);name=COOKIE+'_review_'+domain
            assert name in parsed and COOKIE not in parsed
            cookies.append(name+'='+parsed[name].value);ids.append(v['instance_id'])
        combined='; '.join(cookies)
        for domain,iid in zip(('pong','kitchen'),ids):
            assert request('GET','/api/review/'+domain+'/view',cookie=combined)[1]['instance_id']==iid
        assert request('GET','/api/prolific/session',cookie=combined)[0]==401
        assert request('GET','/api/review/pong/admin/export',cookie=combined)[0]==404
        assert request('POST','/api/review/pong/session',{},combined)[0]==404
        assert request('GET','/api/review/pong/view?instance_id='+ids[1],cookie=combined)[0]==404
        assert not prolific.payment_records(server.store)
    finally:
        server.shutdown();server.server_close();worker.join(5)
