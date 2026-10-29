"""Synthetic software-only flow, never model or human-study acceptance."""
from pathlib import Path
import importlib.util
import sqlite3
from threading import Thread
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import json
import pytest

from runtime_v642 import RuntimeRegistry,RuntimeUnavailable
from runtime_v642.store import StudyError
from study_vnext.store import StudyStore,CONSENT_VERSION,RATINGS
from study_vnext.server import make_server

@pytest.fixture
def helper():
    spec=importlib.util.spec_from_file_location('vnext_synthetic_core_fixture',Path(__file__).with_name('test_runtime_v642_store.py'))
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);return h

@pytest.fixture
def study(tmp_path,helper):
    _,_,catalog=helper.package_at(tmp_path/'package')
    registry=RuntimeRegistry(catalog,adapters={d:helper.FakeAdapter(d) for d in ('warehouse','pong','kitchen')})
    path=tmp_path/'new-study.sqlite'
    s=StudyStore(path,registry,scenario_pools={d:(11,) for d in registry.adapters},
        demonstration_seeds={d:10 for d in registry.adapters},mode='preview')
    yield s
    s.close()

def enroll(s,reference='synthetic',domain='warehouse',group='A',question_transmission_consent=False):
    r=s.enroll(domain=domain,participant_reference=reference,group=group,admin=True,
        consent=True,consent_version=CONSENT_VERSION,question_transmission_consent=question_transmission_consent)
    return r['session_token'],r['instance']['id']

def prepare_controls(s,t,i):
    domain=s.view(t,i)['instance']['domain']
    codes={'warehouse':['KeyW','KeyA','KeyS','KeyD','Space'],'pong':['KeyA','KeyD','Space'],
        'kitchen':['KeyW','KeyA','KeyS','KeyD','Space','KeyE']}[domain]
    return s.complete_preparation(t,i,kind='controls',observed_codes=codes)

def prepare(s,t,i):
    prepare_controls(s,t,i);s.demonstration(t,i)
    receipt=s.view(t,i)['study']['preparation']
    s.complete_preparation(t,i,kind='demonstration',bundle_sha256=receipt['demonstration_sha256'],
        final_frame_count=receipt['demonstration_frame_count'])
    s.acknowledge_demonstration(t,i)

def tasks(s,t,i):
    prepare(s,t,i)
    for task in (1,2,3):
        s.start_task(t,i)
        for turn in (0,1):s.action(t,i,command_id=f'{task}-{turn}',expected_turn=turn,human_action='wait')

def answer_understanding(s,t,i):
    bank=s.understanding(t,i)
    answers={q['id']:q['options'][0]['id'] for q in bank['questions']}
    return s.submit_understanding(t,i,answers=answers,version=bank['version'],sha256=bank['sha256'])

def test_atomic_consent_and_single_domain_enrollment(study):
    with pytest.raises(StudyError):study.enroll(domain='warehouse',participant_reference='x',consent=False,consent_version=CONSENT_VERSION)
    assert study.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0]==0
    t,i=enroll(study)
    assert study.db.execute('SELECT COUNT(*) FROM vnext_enrollments').fetchone()[0]==1
    with pytest.raises(StudyError,match='one_domain_per_participant'):enroll(study,domain='pong')
    assert study.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0]==1
    assert study.view(t,i)['study']['scientific_sample'] is False
    with pytest.raises(sqlite3.IntegrityError):study.db.execute('DELETE FROM vnext_enrollments')

@pytest.mark.parametrize('domain',['warehouse','pong','kitchen'])
@pytest.mark.parametrize('group',['A','B','C'])
def test_three_tasks_survey_export_and_restart(study,domain,group):
    t,i=enroll(study,domain=domain,group=group)
    with pytest.raises(StudyError):study.submit_survey(t,i,answers={k:4 for k in RATINGS})
    tasks(study,t,i)
    assert study.view(t,i)['study']['questionnaire_required']
    answer_understanding(study,t,i)
    result=study.submit_survey(t,i,answers={k:4 for k in RATINGS},feedback='synthetic test')
    assert result['study']['phase']=='finished'
    assert study.submit_survey(t,i,answers={k:4 for k in RATINGS},feedback='synthetic test')==result
    with pytest.raises(StudyError):study.submit_survey(t,i,answers={k:7 for k in RATINGS})
    exported=study.export_instance(t,i,admin=True)
    assert len(exported['records']['runs'])==3 and len(exported['records']['study_surveys'])==1
    assert len(exported['records']['study_understanding'])==1
    assert 'token' not in json.dumps(exported['records']['study_enrollments'])
    registry,path=study.registry,study.database
    reopened=StudyStore(path,registry,scenario_pools=study.scenarios,demonstration_seeds=study.demo_seeds,mode='preview')
    try:assert reopened.view(t,i)['study']['phase']=='finished'
    finally:reopened.close()

def test_event_idempotence_scope_and_immutable_record(study):
    t,i=enroll(study);prepare(study,t,i);study.start_task(t,i)
    data=dict(event_id='same-event',task=1,kind='active',seconds=1.25)
    assert study.event(t,i,**data)==study.event(t,i,**data)
    with pytest.raises(StudyError):study.event(t,i,**dict(data,seconds=2))
    with pytest.raises(StudyError):study.event(t,i,**dict(data,event_id='bad-event',seconds=float('nan')))
    with pytest.raises(StudyError):study.event(t,i,**dict(data,event_id='cross-task',task=2))

def test_no_synthetic_or_catalog_flag_can_open_pilot(study):
    study.mode='pilot'
    with pytest.raises(RuntimeUnavailable):enroll(study)
    assert study.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0]==0

def test_request_asset_failure_is_explicit_closed_HTTP_response(study):
    from contextlib import contextmanager
    @contextmanager
    def unavailable():
        raise RuntimeUnavailable('synthetic missing asset')
        yield
    study.request_context=unavailable
    server=make_server(study,admin_token='synthetic-administrator-secret-32-characters')
    worker=Thread(target=server.serve_forever,daemon=True);worker.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f'http://127.0.0.1:{server.server_port}/study-api/status',timeout=5)
        assert error.value.code==503
        assert json.loads(error.value.read())==dict(error='synthetic missing asset',pilot_opened=False)
        assert study.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0]==0
    finally:server.shutdown();server.server_close();worker.join(timeout=2)

def test_researcher_HTTP_controls_and_real_participant_projection(study):
    admin='synthetic-private-admin-at-least-32-characters'
    server=make_server(study,admin_token=admin);worker=Thread(target=server.serve_forever,daemon=True);worker.start()
    def call(path,payload=None,token=None,admin_header=False):
        headers={'Content-Type':'application/json'}
        if token:headers['Authorization']='Bearer '+token
        if admin_header:headers['X-Admin-Token']=admin
        req=Request(f'http://127.0.0.1:{server.server_port}'+path,headers=headers,data=None if payload is None else json.dumps(payload).encode())
        try:r=urlopen(req,timeout=10)
        except HTTPError as e:r=e
        with r:return r.status,r.read()
    try:
        for page in ('/','/warehouse/','/pong/','/kitchen/'):
            status,body=call(page);assert status==200 and b'question-form' in body
        assert call('/study-assets/input-queue.js')[0]==200
        data=dict(domain='warehouse',participant_reference='http-test',consent=True,consent_version=CONSENT_VERSION,group='A')
        assert call('/study-api/enroll',data)[0]==403
        status,raw=call('/study-api/enroll',data,admin_header=True);assert status==200
        r=json.loads(raw);t,i=r['session_token'],r['instance']['id']
        assert call(f'/study-api/{i}/view')[0]==401
        assert call(f'/study-api/{i}/view',token=t)[0]==200
        # Preparation API transport is covered separately; these are real store
        # receipts so the participant route cannot skip practice or playback.
        prepare(study,t,i);call(f'/study-api/{i}/next-task',{},t)
        public=json.loads(call(f'/study-api/{i}/view',token=t)[1])
        assert public['current']['decision']=={} and public['current']['last_executed_action'] is None
        assert call(f'/study-api/{i}/export',token=t)[0]==403
    finally:server.shutdown();server.server_close();worker.join(timeout=2)

def active_questions(study, *, question_transmission_consent=False, reference='synthetic'):
    # Toy evidence uses a real action name but remains a synthetic adapter test.
    from study_vnext.qa import EvidenceQuestionComposer
    adapter=study.registry.adapters['warehouse'];original=adapter.decide
    def decide(state,binding):
        result=original(state,binding);result['action']='wait';result['policy_actions']['ai']='wait';return result
    adapter.decide=decide;study.question_composer=EvidenceQuestionComposer()
    t,i=enroll(study,reference=reference,question_transmission_consent=question_transmission_consent)
    prepare(study,t,i);study.start_task(t,i)
    for turn in (0,1):study.action(t,i,command_id='first-'+str(turn),expected_turn=turn,human_action='wait')
    study.start_task(t,i)
    return t,i

def test_rendering_persisted_replayed_and_revoked(study):
    t,i=active_questions(study)
    answer=study.ask(t,i,question='刚才为什么等待？',target_task=1,decision_turn=0,human_actions=['wait'])
    rendered=answer['language_rendering'];assert rendered['production_provider_verified'] is False
    assert rendered['free_generated_policy_claims'] is False
    assert study.answers(t,i)['questions'][0]['language_rendering']==rendered
    assert len(study.export_instance(t,i,admin=True)['records']['study_answer_renderings'])==1
    for turn in (0,1):study.action(t,i,command_id='second-'+str(turn),expected_turn=turn,human_action='wait')
    study.start_task(t,i)
    with pytest.raises(StudyError):study.answers(t,i)
    assert len(study.export_instance(t,i,admin=True)['records']['study_answer_renderings'])==1

def test_late_provider_result_not_delivered_or_saved_after_task2(study):
    from study_vnext.qa import EvidenceQuestionComposer
    t,i=active_questions(study,question_transmission_consent=True)
    class DelayedProvider:
        configured=True;consented=True
        def select(self,_):
            for turn in (0,1):study.action(t,i,command_id='during-provider-'+str(turn),expected_turn=turn,human_action='wait')
            study.start_task(t,i)
            return dict(intent='reason',fact_ids=['recorded_action'])
    study.question_composer=EvidenceQuestionComposer(DelayedProvider())
    with pytest.raises(StudyError):study.ask(t,i,question='Why?',target_task=1,decision_turn=0)
    assert study.db.execute('SELECT COUNT(*) FROM vnext_answer_renderings').fetchone()[0]==0

def test_tampered_rendered_answer_refused_on_export(study):
    t,i=active_questions(study);answer=study.ask(t,i,question='Why?',target_task=1,decision_turn=0)
    raw=study.db.execute('SELECT rendering_json FROM vnext_answer_renderings').fetchone()[0]
    bad=json.loads(raw);bad['answer']='invented future plan'
    study.db.execute('DROP TRIGGER vnext_answer_renderings_update')
    study.db.execute('UPDATE vnext_answer_renderings SET rendering_json=?',(json.dumps(bad),))
    with pytest.raises(RuntimeUnavailable):study.export_instance(t,i,admin=True)


@pytest.mark.parametrize('value',[None,0,1,'true',[],{}])
def test_question_transmission_consent_requires_explicit_boolean(study,value):
    with pytest.raises(StudyError,match='invalid_question_transmission_consent'):
        study.enroll(domain='warehouse',participant_reference='invalid-transmission-consent',
            consent=True,consent_version=CONSENT_VERSION,question_transmission_consent=value)
    assert study.db.execute('SELECT COUNT(*) FROM v642_instances').fetchone()[0]==0
    assert study.db.execute('SELECT COUNT(*) FROM vnext_enrollments').fetchone()[0]==0


def test_question_transmission_consent_is_instance_bound_immutable_and_restored(study):
    t,declined=enroll(study,reference='declined')
    _,accepted=enroll(study,reference='accepted',question_transmission_consent=True)
    rows=study.db.execute('SELECT instance_id,question_transmission_consent FROM vnext_enrollments').fetchall()
    assert {r['instance_id']:r['question_transmission_consent'] for r in rows}=={declined:0,accepted:1}
    assert study.view(t,declined)['study']['question_transmission_consent'] is False
    with pytest.raises(sqlite3.IntegrityError):
        study.db.execute('UPDATE vnext_enrollments SET question_transmission_consent=1 WHERE instance_id=?',(declined,))
    assert study.export_instance(t,declined,admin=True)['records']['study_enrollments'][0]['question_transmission_consent']==0
    reopened=StudyStore(study.database,study.registry,scenario_pools=study.scenarios,
        demonstration_seeds=study.demo_seeds,mode='preview')
    try:assert reopened.view(t,declined)['study']['question_transmission_consent'] is False
    finally:reopened.close()


def test_global_provider_opt_in_does_not_replace_enrollment_consent(study):
    from study_vnext.qa import EvidenceQuestionComposer
    t,i=active_questions(study)
    class GlobalOptInProvider:
        configured=True;consented=True
        def __init__(self):self.calls=[]
        def select(self,payload):
            self.calls.append(payload)
            return dict(intent='reason',fact_ids=['recorded_action'])
    provider=GlobalOptInProvider()
    # Even configuring the transport after enrollment cannot grant consent.
    study.question_composer=EvidenceQuestionComposer(provider)
    result=study.ask(t,i,question='Do not send this question externally.',target_task=1,decision_turn=0)
    assert provider.calls==[]
    rendering=result['language_rendering']
    assert rendering['provider_status']=='not_configured_template_only'
    assert rendering['semantic_freeform_interpretation'] is False
    assert rendering['audit']['provider_used'] is False
    assert study.answers(t,i)['questions'][0]['language_rendering']==rendering
    assert study.export_instance(t,i,admin=True)['records']['study_answer_renderings']==[rendering]


def test_explicit_enrollment_consent_allows_only_constrained_provider_catalogue(study):
    from study_vnext.qa import EvidenceQuestionComposer
    t,i=active_questions(study,question_transmission_consent=True)
    class ConsentedProvider:
        configured=True;consented=True
        def __init__(self):self.calls=[]
        def select(self,payload):
            self.calls.append(payload)
            return dict(intent='reason',fact_ids=['recorded_action'])
    provider=ConsentedProvider();study.question_composer=EvidenceQuestionComposer(provider)
    result=study.ask(t,i,question='Why wait?',target_task=1,decision_turn=0)
    assert len(provider.calls)==1
    assert set(provider.calls[0])=={'question','language','facts'}
    assert provider.calls[0]['question']=='Why wait?'
    assert all(set(f)=={'id','kind','text'} for f in provider.calls[0]['facts'])
    assert result['language_rendering']['provider_status']=='validated_fact_selection'
    assert result['language_rendering']['audit']['provider_used'] is True
    assert study.view(t,i)['study']['question_transmission_consent'] is True
    assert study.answers(t,i)['questions'][0]['language_rendering']==result['language_rendering']
    assert len(provider.calls)==1  # Reading a stored answer never retransmits.


def test_legacy_enrollment_schema_is_refused_without_inventing_consent(study,tmp_path):
    path=tmp_path/'old-study-without-question-consent.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE vnext_enrollments(instance_id TEXT PRIMARY KEY,protocol_id TEXT NOT NULL,'
            'protocol_sha256 TEXT NOT NULL,consent_version TEXT NOT NULL,language TEXT NOT NULL,created REAL NOT NULL)')
    with pytest.raises(RuntimeUnavailable,match='explicit per-instance question consent'):
        StudyStore(path,study.registry,scenario_pools=study.scenarios,demonstration_seeds=study.demo_seeds,mode='preview')
    with sqlite3.connect(path) as db:
        assert 'question_transmission_consent' not in {r[1] for r in db.execute('PRAGMA table_info(vnext_enrollments)')}
        assert db.execute('SELECT COUNT(*) FROM vnext_enrollments').fetchone()[0]==0


def test_external_rendering_record_without_instance_consent_is_refused(study):
    t,i=active_questions(study)
    study.ask(t,i,question='Why?',target_task=1,decision_turn=0)
    raw=study.db.execute('SELECT rendering_json FROM vnext_answer_renderings').fetchone()[0]
    bad=json.loads(raw);bad['audit']['provider_used']=True
    study.db.execute('DROP TRIGGER vnext_answer_renderings_update')
    study.db.execute('UPDATE vnext_answer_renderings SET rendering_json=?',(json.dumps(bad),))
    with pytest.raises(RuntimeUnavailable,match='lacks enrollment consent'):
        study.export_instance(t,i,admin=True)


@pytest.mark.parametrize('domain,codes',[
    ('warehouse',['KeyW','KeyA','KeyS','KeyD','Space']),
    ('pong',['KeyA','KeyD','Space']),
    ('kitchen',['KeyW','KeyA','KeyS','KeyD','Space','KeyE'])])
def test_preparation_requires_domain_controls_and_bound_completed_demo(study,domain,codes):
    t,i=enroll(study,domain=domain)
    initial=study.view(t,i)['study']['preparation']
    assert initial==dict(controls=False,demonstration=False,demonstration_sha256=None,demonstration_frame_count=None)
    with pytest.raises(StudyError,match='controls_practice_required'):study.demonstration(t,i)
    with pytest.raises(StudyError,match='preparation_incomplete'):study.acknowledge_demonstration(t,i)
    assert study.db.execute('SELECT COUNT(*) FROM v642_demonstrations').fetchone()[0]==0
    with pytest.raises(StudyError):study.start_task(t,i)
    with pytest.raises(StudyError):study.complete_preparation(t,i,kind='controls',observed_codes=codes[:-1])
    first=study.complete_preparation(t,i,kind='controls',observed_codes=codes[::-1])
    assert first['study']['preparation']['controls'] is True
    assert study.complete_preparation(t,i,kind='controls',observed_codes=codes)==first
    with pytest.raises(StudyError,match='generate_bound_demonstration_first'):
        study.complete_preparation(t,i,kind='demonstration',bundle_sha256='0'*64,final_frame_count=1)
    demo=study.demonstration(t,i)
    info=study.view(t,i)['study']['preparation']
    from runtime_v642.contracts import digest
    assert info['demonstration_sha256']==digest(demo)
    assert info['demonstration_frame_count']==len(demo['records'])
    assert info['demonstration'] is False
    with pytest.raises(StudyError,match='preparation_incomplete'):study.acknowledge_demonstration(t,i)
    for wrong in (dict(bundle_sha256='0'*64,final_frame_count=len(demo['records'])),
                  dict(bundle_sha256=digest(demo),final_frame_count=len(demo['records'])-1),
                  dict(bundle_sha256=digest(demo),final_frame_count=True)):
        with pytest.raises(StudyError,match='complete_bound_demonstration_playback_required'):
            study.complete_preparation(t,i,kind='demonstration',**wrong)
    receipt=study.complete_preparation(t,i,kind='demonstration',bundle_sha256=digest(demo),final_frame_count=len(demo['records']))
    assert receipt['study']['preparation']['demonstration'] is True
    assert study.complete_preparation(t,i,kind='demonstration',bundle_sha256=digest(demo),
        final_frame_count=len(demo['records']))==receipt
    study.acknowledge_demonstration(t,i);study.start_task(t,i)
    with pytest.raises(StudyError,match='preparation_enrollment_stage_required'):
        study.complete_preparation(t,i,kind='controls',observed_codes=codes)
    exported=study.export_instance(t,i,admin=True)
    assert {row['kind'] for row in exported['records']['study_preparation']}=={'controls','demonstration'}
    assert study.view(t,i)['study']['scientific_sample'] is False
    with pytest.raises(sqlite3.IntegrityError):study.db.execute('DELETE FROM vnext_preparation')


def test_preparation_cannot_be_forged_from_another_instances_bundle(study):
    from runtime_v642.contracts import digest
    first_t,first_i=enroll(study,reference='first',group='A')
    second_t,second_i=enroll(study,reference='second',group='C')
    prepare_controls(study,first_t,first_i);first_demo=study.demonstration(first_t,first_i)
    prepare_controls(study,second_t,second_i);second_demo=study.demonstration(second_t,second_i)
    assert digest(first_demo)!=digest(second_demo)
    with pytest.raises(StudyError,match='complete_bound_demonstration_playback_required'):
        study.complete_preparation(second_t,second_i,kind='demonstration',bundle_sha256=digest(first_demo),
            final_frame_count=len(first_demo['records']))
    with pytest.raises(StudyError):
        study.complete_preparation(first_t,second_i,kind='controls',observed_codes=['KeyW','KeyA','KeyS','KeyD','Space'])
    assert study.view(second_t,second_i)['study']['preparation']['demonstration'] is False


def test_preparation_receipt_tamper_is_rejected_on_restore_and_export(study):
    t,i=enroll(study);prepare(study,t,i)
    study.db.execute('DROP TRIGGER vnext_preparation_update')
    study.db.execute("UPDATE vnext_preparation SET details_json=? WHERE kind='demonstration'",
        (json.dumps(dict(bundle_sha256='0'*64,final_frame_count=1)),))
    for read in (lambda:study.view(t,i),lambda:study.export_instance(t,i,admin=True)):
        with pytest.raises(RuntimeUnavailable,match='playback receipt differs'):read()


def test_reading_and_question_entry_events_are_separate_modes(study):
    t,i=active_questions(study)
    for kind in ('reading','question_entry'):
        assert study.event(t,i,event_id='mode-'+kind,task=2,kind=kind,seconds=2.5)['saved']
    exported=study.export_instance(t,i,admin=True)
    assert {r['kind'] for r in exported['records']['study_events']}=={'reading','question_entry'}


def test_understanding_bank_and_scoring_are_gated_by_actual_task3_completion(study,monkeypatch):
    import study_vnext.understanding as instrument
    calls=[];original=instrument.score
    def score(domain,answers):
        calls.append((domain,answers));return original(domain,answers)
    monkeypatch.setattr(instrument,'score',score)
    t,i=enroll(study)
    for read in (lambda:study.understanding(t,i),lambda:study.submit_understanding(t,i,answers={},
        version=instrument.VERSION,sha256=instrument.instrument_sha256())):
        with pytest.raises(StudyError,match='finish_three_tasks_before_understanding'):read()
    prepare(study,t,i)
    for task in (1,2,3):
        study.start_task(t,i)
        with pytest.raises(StudyError):study.understanding(t,i)
        with pytest.raises(StudyError):study.submit_understanding(t,i,answers={},
            version=instrument.VERSION,sha256=instrument.instrument_sha256())
        for turn in (0,1):study.action(t,i,command_id=f'gate-{task}-{turn}',expected_turn=turn,human_action='wait')
    assert calls==[]
    bank=study.understanding(t,i)
    assert bank['administration']=='after_task3_only'
    assert bank['construct']=='public_mechanics_and_control_comprehension'
    assert bank['status']=='candidate_instrument_requires_pilot_validation'
    assert all(set(q)=={'id','prompt','options'} for q in bank['questions'])
    assert all(set(o)=={'id','text'} for q in bank['questions'] for o in q['options'])
    assert calls==[]  # Delivering the key-free bank does not score responses.
    answer_understanding(study,t,i)
    assert calls
    assert study.db.execute('SELECT COUNT(*) FROM vnext_understanding').fetchone()[0]==1


def test_forged_complete_stage_cannot_unlock_questionnaire_bank(study):
    t,i=enroll(study)
    study.db.execute("UPDATE v642_instances SET stage='complete' WHERE id=?",(i,))
    with pytest.raises(RuntimeUnavailable,match='saved pre-task stage differs'):study.understanding(t,i)
    assert study.db.execute('SELECT COUNT(*) FROM vnext_understanding').fetchone()[0]==0


def test_understanding_same_bank_for_groups_and_enrollment_language(study):
    banks=[]
    for group in ('A','B','C'):
        result=study.enroll(domain='pong',participant_reference='bank-'+group,group=group,admin=True,
            consent=True,consent_version=CONSENT_VERSION,language='en')
        t,i=result['session_token'],result['instance']['id'];tasks(study,t,i)
        banks.append(study.understanding(t,i))
    assert banks[0]==banks[1]==banks[2]
    assert banks[0]['language']=='en'
    assert 'correct' not in {key for q in banks[0]['questions'] for key in q}


def test_questionnaire_partial_submission_recovery_and_private_scoring(study):
    from study_vnext.understanding import score
    t,i=enroll(study);tasks(study,t,i)
    ratings={k:5 for k in RATINGS}
    first=study.submit_survey(t,i,answers=ratings,feedback='saved before connection loss')
    assert first['study']['phase']=='questionnaire'
    assert first['study']['questionnaire_required'] and first['study']['understanding_required']
    assert first['study']['survey_required'] is False
    restored=study.view(t,i)['study']
    assert restored['survey_answers']==ratings and restored['survey_feedback']=='saved before connection loss'
    assert restored['understanding_answers'] is None
    bank=study.understanding(t,i)
    answers={q['id']:q['options'][0]['id'] for q in bank['questions']}
    for args in (dict(version='stale-version',sha256=bank['sha256']),dict(version=bank['version'],sha256='0'*64)):
        with pytest.raises(StudyError,match='current_understanding_instrument_required'):
            study.submit_understanding(t,i,answers=answers,**args)
    assert study.db.execute('SELECT COUNT(*) FROM vnext_understanding').fetchone()[0]==0
    final=study.submit_understanding(t,i,answers=answers,version=bank['version'],sha256=bank['sha256'])
    assert final['study']['phase']=='finished' and final['study']['questionnaire_required'] is False
    assert final['study']['understanding_answers']==answers
    assert 'result_json' not in final['study'] and 'correct_count' not in final['study']
    assert study.submit_understanding(t,i,answers=answers,version=bank['version'],sha256=bank['sha256'])==final
    changed=dict(answers);changed[bank['questions'][0]['id']]=bank['questions'][0]['options'][1]['id']
    with pytest.raises(StudyError,match='understanding_already_submitted'):
        study.submit_understanding(t,i,answers=changed,version=bank['version'],sha256=bank['sha256'])
    saved=study.export_instance(t,i,admin=True)['records']['study_understanding'][0]
    assert json.loads(saved['result_json'])==score('warehouse',answers)
    assert json.loads(saved['result_json'])['validated_scale'] is False
    with pytest.raises(sqlite3.IntegrityError):study.db.execute('DELETE FROM vnext_understanding')


@pytest.mark.parametrize('mutation',['missing','extra','unknown_option'])
def test_understanding_rejects_malformed_answers_without_saving(study,mutation):
    t,i=enroll(study);tasks(study,t,i)
    bank=study.understanding(t,i)
    answers={q['id']:q['options'][0]['id'] for q in bank['questions']}
    if mutation=='missing':answers.pop(bank['questions'][0]['id'])
    elif mutation=='extra':answers['unregistered_item']='a'
    else:answers[bank['questions'][0]['id']]='unregistered_option'
    with pytest.raises(StudyError,match='invalid_understanding_answers'):
        study.submit_understanding(t,i,answers=answers,version=bank['version'],sha256=bank['sha256'])
    assert study.db.execute('SELECT COUNT(*) FROM vnext_understanding').fetchone()[0]==0


@pytest.mark.parametrize('field,value',[
    ('sha256','0'*64),('answers_json','{}'),('result_json','{}')])
def test_understanding_export_revalidates_instrument_answers_and_score(study,field,value):
    t,i=enroll(study);tasks(study,t,i);answer_understanding(study,t,i)
    study.db.execute('DROP TRIGGER vnext_understanding_update')
    study.db.execute(f'UPDATE vnext_understanding SET {field}=?',(value,))
    for read in (lambda:study.view(t,i),lambda:study.export_instance(t,i,admin=True)):
        with pytest.raises(RuntimeUnavailable):read()
