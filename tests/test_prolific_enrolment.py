"""Synthetic participants in a temporary database; no paid submissions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import pytest

from study_v3.config import Settings
from study_v3.store import Store, StudyError
from study_v3 import prolific
from tests.test_study_v3_store import RecordingExplainer, Flow

STUDY='a'*24

@pytest.fixture
def store(tmp_path):
    s=Settings(database=str(tmp_path/'prolific.sqlite3'),persistent=True,admin_token='test-only',
        mode='pilot',verified=True,llm_base_url='https://test.invalid',llm_model='test',
        prolific_study_id=STUDY,prolific_completion_code='TESTONLY',prolific_launch_confirmed=True)
    result=Store(s,RecordingExplainer())
    yield result
    result.db.close()

def payload(n):
    return {'PROLIFIC_PID':f'{n:024x}','STUDY_ID':STUDY,'SESSION_ID':f'{1000+n:024x}',
        'consent':True,'age_21':True,'consent_version':prolific.CONSENT_VERSION}

def test_concurrent_blocks_cap_and_private_identity(store):
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda n:prolific.enrol(store,payload(n)),range(1,13)))
    assert Counter((v['domain'],v['group']) for _,v in results)==Counter({cell:2 for cell in prolific.CELLS})
    records=prolific.payment_records(store)
    for block in (records[:6],records[6:]):
        assert {(r['domain'],r['group_code']) for r in block}==set(prolific.CELLS)
    with pytest.raises(StudyError,match='prolific_full'):prolific.enrol(store,payload(13))
    exported=json.dumps(store.export())
    assert 'prolific_pid' not in exported and 'submission_id' not in exported
    for n,(_,view) in enumerate(results,1):
        assert payload(n)['PROLIFIC_PID'] not in exported
        assert view['participant_id'].startswith('pl-')
        assert 'completion_code' not in view

def test_main_cohort_has_ten_per_cell_and_separate_old_cohort(store):
    # The main study must start fresh without counting earlier allocations.
    prolific.enrol(store, payload(100))
    main_study = 'b' * 24
    store.settings = replace(store.settings, prolific_study_id=main_study, prolific_places=60)
    def enrol(n):
        return prolific.enrol(store, {**payload(n), 'STUDY_ID': main_study})
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(enrol, range(1, 61)))
    assert Counter((v['domain'], v['group']) for _, v in results) == Counter({cell: 10 for cell in prolific.CELLS})
    assert all(v['rewards']['base_pence'] == 300 for _, v in results)
    with pytest.raises(StudyError, match='prolific_full'):
        enrol(61)
    assert len(prolific.payment_records(store)) == 60
    assert len(store.export()['instances']) == 61
    with pytest.raises(StudyError, match='prolific_already_participated'):
        enrol(100)


def test_retry_restart_and_cross_identity_access(store):
    token,view=prolific.enrol(store,payload(1))
    same_token,same=prolific.enrol(store,payload(1),token)
    assert same_token==token and same['id']==view['id']
    reopened=Store(store.settings,RecordingExplainer())
    try:assert prolific.resume(reopened,token)['id']==view['id']
    finally:reopened.db.close()
    with pytest.raises(StudyError,match='session_required'):prolific.enrol(store,payload(1))
    with pytest.raises(StudyError,match='prolific_already_participated'):prolific.enrol(store,payload(2),token)
    assert len(prolific.payment_records(store))==1
    with pytest.raises(StudyError,match='prolific_identity_mismatch'):
        prolific.resume(store,token,payload(2))

def test_client_cannot_choose_cell_or_enrol_other_game(store):
    token,view=prolific.enrol(store,{**payload(1),'domain':'not-a-domain','group':'not-a-group','mode':'preview'})
    assert view['mode']=='pilot' and (view['domain'],view['group']) in prolific.CELLS
    with pytest.raises(StudyError,match='use_prolific_entry'):
        store.create({'participant_id':view['participant_id'],'domain':'pong','consent':True},token)
    with pytest.raises(StudyError,match='prolific_assignment_locked'):
        store.create({'participant_id':view['participant_id'],'domain':'pong','mode':'test','consent':True},token,admin=True)

@pytest.mark.parametrize('change,error',[
    ({'age_21':False},'consent_required'),({'consent':False},'consent_required'),
    ({'consent_version':'stale'},'consent_version_changed'),
    ({'PROLIFIC_PID':'invalid'},'invalid_prolific_link'),
    ({'STUDY_ID':'b'*24},'wrong_prolific_study')])
def test_invalid_entry_leaves_no_records(store,change,error):
    with pytest.raises(StudyError,match=error):prolific.enrol(store,{**payload(1),**change})
    assert not prolific.payment_records(store)
    assert not store.export()['participants']

def test_not_confirmed_remains_closed(store):
    store.settings=replace(store.settings,prolific_launch_confirmed=False)
    assert not prolific.ready(store)
    with pytest.raises(StudyError,match='study_not_ready'):prolific.enrol(store,payload(1))

def test_optional_questionnaire_completion_and_code(store):
    # Drive the existing state machine using actual engine transitions.
    token,view=prolific.enrol(store,payload(1))
    flow=object.__new__(Flow)
    flow.store,flow.token,flow.view,flow.domain=store,token,view,view['domain']
    flow.finish_demo()
    for _ in range(3):
        flow.finish_task()
        flow.command('next')
    assert flow.view['stage']=='questionnaire' and flow.view['questionnaire_optional']
    flow.command('questionnaire',answers={},feedback='')
    assert flow.view['stage']=='completed'
    assert flow.view['completion_url']=='https://app.prolific.com/submissions/complete?cc=TESTONLY'
    assert all(x is None for x in json.loads(store.export()['questionnaires'][0]['answers_json'])['ratings'].values())
    assert prolific.resume(store,token)['completion_code']=='TESTONLY'
    assert len(prolific.payment_records(store))==1

def test_http_entry_ids_recovery_and_restricted_payment_export(store):
    from study_v3.server import make_server
    from tests.test_study_v3_http import Client, ORIGIN
    server=make_server(replace(store.settings,origin=ORIGIN),port=0,explainer=RecordingExplainer())
    server.store.qa_healthy=True
    worker=threading.Thread(target=server.serve_forever,daemon=True)
    worker.start()
    client=Client(server.server_port)
    try:
        assert client.request('GET','/prolific/')[0]==200
        status,info,_=client.request('GET','/api/prolific/info')
        assert status==200 and info['ready']
        assert 'TESTONLY' not in json.dumps(info)
        status,view,_=client.request('POST','/api/prolific/enrol',payload(1))
        assert status==200 and view['prolific'] and client.cookie
        assert 'completion_url' not in view
        status,resumed,_=client.request('GET','/api/prolific/session')
        assert status==200 and resumed['id']==view['id']
        assert client.request('GET','/api/prolific/admin/payments')[0]==403
        status,records,_=client.request('GET','/api/prolific/admin/payments',headers={'Authorization':'Bearer test-only'})
        assert status==200 and records['records'][0]['prolific_pid']==payload(1)['PROLIFIC_PID']
        stranger=Client(server.server_port)
        assert stranger.request('POST','/api/prolific/enrol',payload(1))[0]==401
    finally:
        server.shutdown();server.server_close();worker.join(5)

def release_payload(record, status='RETURNED'):
    return {'instance_id':record['instance_id'],'submission_id':record['submission_id'],
        'submission_status':status,'reason':'Verified terminal status in Prolific study submissions.'}

def test_replacements_fill_only_released_cells_without_rewriting_history(store):
    sessions=[prolific.enrol(store,payload(n)) for n in range(1,13)]
    before=prolific.payment_records(store)
    missing={('warehouse','B'),('kitchen','A'),('kitchen','B')}
    released=[]
    for cell in sorted(missing):
        record=next(r for r in before if (r['domain'],r['group_code'])==cell)
        first=prolific.release_slot(store,release_payload(record))
        assert prolific.release_slot(store,release_payload(record))==first
        released.append(record)
    # Three parallel arrivals fill each vacant cell exactly once.
    with ThreadPoolExecutor(max_workers=3) as pool:
        replacements=list(pool.map(lambda n:prolific.enrol(store,payload(n)),range(13,16)))
    assert {(v['domain'],v['group']) for _,v in replacements}==missing
    with pytest.raises(StudyError,match='prolific_full'):prolific.enrol(store,payload(16))
    after=prolific.payment_records(store)
    assert len(after)==15 and [r['allocation_index'] for r in after]==list(range(1,16))
    assert Counter((r['domain'],r['group_code']) for r in after if not r['released'])==Counter({cell:2 for cell in prolific.CELLS})
    for old,new in zip(before,after):
        for key in ('prolific_pid','consented','instance_id','participant_id','allocation_index','stage'):
            assert old[key]==new[key]
    for record in released:
        token,view=sessions[record['allocation_index']-1]
        with pytest.raises(StudyError,match='prolific_submission_closed'):prolific.resume(store,token)
        with pytest.raises(StudyError,match='prolific_submission_closed'):store.recover_view(token)
        with pytest.raises(StudyError,match='prolific_submission_closed'):
            store.command(token,'demo_skip',{'instance_id':view['id'],'revision':view['revision'],'command_id':'closed-attempt'})
    assert 'release_reason' not in json.dumps(store.export())
    reopened=Store(store.settings,RecordingExplainer())
    try:assert sum(bool(r['released']) for r in prolific.payment_records(reopened))==3
    finally:reopened.db.close()

def test_release_rejects_wrong_submission_active_status_and_completed(store):
    token,view=prolific.enrol(store,payload(1))
    record=prolific.payment_records(store)[0]
    with pytest.raises(StudyError,match='invalid_slot_release'):
        prolific.release_slot(store,release_payload(record,'ACTIVE'))
    with pytest.raises(StudyError,match='prolific_submission_not_found'):
        prolific.release_slot(store,{**release_payload(record),'submission_id':'f'*24})
    flow=object.__new__(Flow)
    flow.store,flow.token,flow.view,flow.domain=store,token,view,view['domain']
    flow.finish_demo()
    for _ in range(3):
        flow.finish_task();flow.command('next')
    flow.command('questionnaire',answers={},feedback='')
    with pytest.raises(StudyError,match='completed_slot_cannot_be_released'):
        prolific.release_slot(store,release_payload(record))
    assert not prolific.payment_records(store)[0]['released']
    assert prolific.resume(store,token)['stage']=='completed'

def test_http_release_requires_researcher_and_is_idempotent(store):
    from study_v3.server import make_server
    from tests.test_study_v3_http import Client, ORIGIN
    server=make_server(replace(store.settings,origin=ORIGIN),port=0,explainer=RecordingExplainer())
    server.store.qa_healthy=True
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    client=Client(server.server_port)
    try:
        assert client.request('POST','/api/prolific/enrol',payload(1))[0]==200
        record=prolific.payment_records(server.store)[0]
        body=release_payload(record)
        path='/api/prolific/admin/release-slot'
        assert client.request('POST',path,body)[0]==403
        assert not prolific.payment_records(server.store)[0]['released']
        auth={'Authorization':'Bearer test-only'}
        first=client.request('POST',path,body,headers=auth)
        assert first[0]==200
        assert client.request('POST',path,body,headers=auth)[1]==first[1]
        assert client.request('GET','/api/prolific/session')[0]==409
        assert client.request('POST',path,release_payload(record,'TIMED_OUT'),headers=auth)[0]==409
    finally:
        server.shutdown();server.server_close();worker.join(5)
