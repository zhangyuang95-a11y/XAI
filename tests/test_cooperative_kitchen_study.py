"""Behavioral acceptance for the independent kitchen study ledger and HTTP boundary."""
import copy
import csv
import io
import json
import os
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, insert, select, text, update
from sqlalchemy.engine import make_url

from backend.cooperative_kitchen.study import KitchenStudy, DEFAULT_CONSENT, build_default_consent
from env.cooperative_kitchen import CooperativeKitchen, KitchenConfig
from ui.cooperative_kitchen_server import create_app, COOKIE
from ui.cooperative_kitchen_store import (KitchenStore, StudyError, encode, episodes,
                                         events, frames, invitations, participants, blocks, questions, runs, token_digest)


class FixturePolicy:
    policy_kind="fixture_neural"
    checkpoint_id="test-only-not-publishable"
    def act(self, observations):
        actions={k:"WAIT" for k in observations}
        return actions,{k:{"chosen_action":"WAIT","probabilities":[0,0,0,0,0,1],"logits":[0,0,0,0,0,1],"checkpoint_id":self.checkpoint_id} for k in observations}


class FixtureExplainer:
    def generate(self,snapshot,question,**kwargs):
        return {"title":"Fixture evidence","text":f"Selected turn {snapshot['turn']}","frame":snapshot["turn"],"kind":kwargs["kind"],"verified":True,"evidence":{"secret":True},"diagnostics":{"test_only":True}}


def release():
    return {"study_ready":True,"versions":{"fixture":"never-a-research-release"},
            "scenarios":{"X":["base_empty"]*3,"Y":["mirror_empty"]*3},
            "question_bank":[{"id":f"p{i}","type":"prediction" if i<4 else "counterfactual","prompt":{"zh":"测试题","en":"Test item"},"options":[{"value":"WAIT","label":{"zh":"等待","en":"Wait"}},{"value":"UP","label":{"zh":"向上","en":"Up"}}],"correct_answer":"WAIT"} for i in range(8)]}


@pytest.fixture
def study(tmp_path, request):
    configured=os.environ.get("KITCHEN_TEST_DATABASE_URL")
    admin_engine=None
    if configured:
        admin_engine=create_engine(configured)
        schema="kitchen_test_"+uuid.uuid4().hex
        with admin_engine.begin() as db: db.execute(text(f'CREATE SCHEMA "{schema}"'))
        url=make_url(configured).update_query_dict({"options":f"-csearch_path={schema}"})
        database_url=url.render_as_string(hide_password=False)
    else: database_url=f"sqlite:///{tmp_path/'test.sqlite3'}"
    app=KitchenStudy(tmp_path,database_url,namespace="test",allow_sqlite=not configured,policy=FixturePolicy(),explainer=FixtureExplainer(),release=release(),test_mode=True,enrollment_mode="formal",
                     qa_limits=getattr(request,"param",{"min_interval_seconds":0}))
    yield app
    app.store.engine.dispose()
    if admin_engine:
        with admin_engine.begin() as db: db.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin_engine.dispose()


def join(study,mode="pilot"):
    import uuid
    code="P_"+uuid.uuid4().hex[:20] if mode=="pilot" else ""
    token,view=study.join({"operation_id":str(uuid.uuid4()),"participant_id":code,"language":"zh","mode":mode})
    return token,view,code


def command(study,token,cmd,**fields):
    import uuid
    view=study.view(token)
    return study.command(token,{"operation_id":str(uuid.uuid4()),"version":view["run"]["version"],"command":cmd,**fields})


def to_practice(study,token):
    command(study,token,"consent",accepted=True)
    return command(study,token,"next")


def finish(study,token):
    """Exercise every real step; no production HTTP test bypass."""
    view=study.view(token)
    while view["can_act"]:
        view=study.command(token,{"operation_id":f"step-{view['run']['episode_id']}-{view['state']['turn']}","version":view["run"]["version"],"command":"action","action":"WAIT"})
    return view


def to_task1(study,token):
    to_practice(study,token);finish(study,token)
    return command(study,token,"next")


def condition(study,token):
    with study.store.transaction() as db:return study.store.run(db,token)["condition"]


def join_a(study):
    for _ in range(4):
        token,view,code=join(study)
        if condition(study,token)=="A":return token,view,code
    raise AssertionError("Four-person block has no A condition")


def test_sqlite_never_used_for_research(tmp_path):
    for ns in ["pilot","confirmatory"]:
        with pytest.raises(ValueError): KitchenStore("sqlite:///:memory:",ns,allow_sqlite=True)
    with pytest.raises(ValueError): KitchenStore("sqlite:///:memory:","development")
    with pytest.raises(ValueError): KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="development",allow_sqlite=True,test_mode=True)


def test_candidate_gate_allows_only_labeled_freeplay(tmp_path):
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,enrollment_mode="formal")
    assert not app.status()["study_ready"]
    with pytest.raises(StudyError) as exc: app.join({"operation_id":"x","mode":"pilot","participant_id":"P001"})
    assert exc.value.status==503
    token,view,_=join(app,"freeplay")
    assert view["policy_kind"]=="program_baseline"
    view=command(app,token,"next")
    assert view["can_restart"] and view["can_swap"] and view["can_ask"]


def test_random_four_person_block_and_same_cookie_participant_resume(study):
    entries=[join(study) for _ in range(4)]
    with study.store.transaction() as db:
        rows=[study.store.run(db,entry[0]) for entry in entries]
    assert {(r["condition"],r["task_order"]) for r in rows}=={("A","XY"),("A","YX"),("B","XY"),("B","YX")}
    token,view,code=entries[0]
    command(study,token,"consent",accepted=True)
    token2,resumed=study.join({"operation_id":"resume","participant_id":code,"language":"en","mode":"pilot"},existing_token=token)
    assert resumed["run"]["id"]==view["run"]["id"]
    assert resumed["run"]["phase"]=="instructions"
    assert token2==token and study.view(token)==resumed
    with study.store.transaction() as db:
        after=study.store.run(db,token2)
    assert after["condition"]==rows[0]["condition"] and after["task_order"]==rows[0]["task_order"]


def test_explicit_consent_permissions_and_phase_protection(study):
    token,view,_=join(study)
    for cmd in ["next","action","restart","swap","survey_submit"]:
        with pytest.raises(StudyError):command(study,token,cmd,action="WAIT",answers={})
    with pytest.raises(StudyError):command(study,token,"consent",accepted=False)
    view=to_practice(study,token)
    assert not view["can_ask"] and not view["can_restart"] and not view["can_swap"]
    assert view["state"]["actors"][0]["side"]=="left"
    assert "condition" not in view["run"]
    assert not any(k.startswith("_") for k in view["state"])


def test_idempotency_conflicting_payload_and_stale_version(study):
    token,_,_=join(study);view=to_practice(study,token)
    payload={"operation_id":"same-action","version":view["run"]["version"],"command":"action","action":"UP"}
    first=study.command(token,payload);second=study.command(token,payload)
    assert first["state"]["turn"]==second["state"]["turn"]==1
    with pytest.raises(StudyError) as exc:study.command(token,{**payload,"action":"DOWN"})
    assert exc.value.code=="operation_conflict"
    with pytest.raises(StudyError) as exc:study.command(token,{**payload,"operation_id":"different"})
    assert exc.value.code=="version_conflict"


def test_concurrent_duplicate_actions_advance_exactly_once(study):
    token,_,_=join(study);view=to_practice(study,token)
    payload={"operation_id":"parallel","version":view["run"]["version"],"command":"action","action":"WAIT"}
    with ThreadPoolExecutor(max_workers=8) as pool:results=list(pool.map(lambda _:study.command(token,payload),range(8)))
    assert all(r["state"]["turn"]==1 for r in results)


def test_twenty_sessions_independent_steps_and_cross_session_denial(study):
    entries=[join(study,"freeplay") for _ in range(20)]
    for token,_,_ in entries:command(study,token,"next")
    with ThreadPoolExecutor(max_workers=20) as pool:result=list(pool.map(lambda entry:command(study,entry[0],"action",action="UP"),entries))
    assert len({r["run"]["id"] for r in result})==20
    assert all(r["state"]["turn"]==1 for r in result)
    with pytest.raises(StudyError):study.history(entries[0][0],result[1]["run"]["episode_id"])


def test_restart_restores_committed_progress_and_swap_freeplay(study):
    token,_,_=join(study,"freeplay");command(study,token,"next")
    first=command(study,token,"action",action="UP")
    second=KitchenStudy(study.output,str(study.store.engine.url),namespace="test",allow_sqlite=True,policy=FixturePolicy(),explainer=FixtureExplainer(),release=release(),test_mode=True)
    assert second.view(token)==first
    switched=command(second,token,"swap")
    assert switched["state"]["actors"][0]["side"]=="right"
    assert switched["state"]["turn"]==0
    assert len(switched["episodes"])==2


def test_six_rounds_survey_draft_resume_and_export_scores(study):
    token,_,_=join(study);to_task1(study,token)
    for index in range(6):
        end=finish(study,token)
        assert end["state"]["score"]==-180
        next_view=command(study,token,"next")
        if index<5:assert next_view["run"]["phase"]==("task1" if index<2 else "task2")
    assert next_view["run"]["phase"]=="questionnaire"
    items=next_view["survey"]["items"]
    assert len(items)==11 and not any("correct_answer" in item for item in items)
    draft=command(study,token,"survey_save",answers={"p0":"WAIT"})
    assert study.view(token)["survey"]["draft"]=={"p0":"WAIT"}
    with pytest.raises(StudyError):command(study,token,"survey_submit",answers={})
    answers={item["id"]:item["options"][0]["value"] for item in items}
    completed=command(study,token,"survey_submit",answers=answers)
    assert completed["run"]["phase"]=="complete" and completed["completion_code"]
    rows=list(csv.DictReader(io.StringIO(study.store.export("csv"))))
    assert rows[0]["task2_mean_score"]=="-180.0" and rows[0]["prediction_accuracy"]=="1.0"
    raw=[json.loads(line) for line in study.store.export().splitlines()]
    assert len([r for r in raw if r["type"]=="episode" and r["document"]["phase"] in {"task1","task2"}])==6
    assert len([r for r in raw if r["type"]=="event" and r["kind"]=="joint_step"])==1260


def test_history_bound_questions_isolated_and_exposure_no_game_change(study):
    token,_,_=join_a(study);view=to_task1(study,token)
    command(study,token,"action",action="UP");before=study.view(token)
    payload={"operation_id":"why-history","version":before["run"]["version"],"episode_id":before["run"]["episode_id"],"frame":0,"kind":"why","question":"Why?"}
    job=study.ask(token,payload)
    assert study.process_one_question()
    answer=study.question(token,job["id"])
    assert answer["answer"]["frame"]==0
    assert "evidence" not in answer["answer"]
    after=study.view(token)
    assert before["state"]==after["state"]
    receipt=study.ask(token,payload)
    assert receipt["id"]==job["id"]
    exposed=study.exposure(token,{"operation_id":"shown","question_id":job["id"],"event":"shown"})
    assert exposed["version"]==after["run"]["version"]
    assert study.view(token)["run"]["version"]==after["run"]["version"]


def test_task2_blocks_old_answers_questions_and_cached_operation(study):
    token,_,_=join_a(study);view=to_task1(study,token)
    payload={"operation_id":"old-question","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"why"}
    job=study.ask(token,payload);study.process_one_question()
    for _ in range(3):finish(study,token);view=command(study,token,"next")
    assert view["run"]["phase"]=="task2" and view["questions"]==[] and not view["can_ask"]
    assert study.history(token,payload["episode_id"])["questions"]==[]
    for f in [lambda:study.question(token,job["id"]),lambda:study.ask(token,payload),lambda:study.exposure(token,{"operation_id":"old-exposure","question_id":job["id"],"event":"shown"})]:
        with pytest.raises(StudyError) as exc:f()
        assert exc.value.status==403


def test_lease_recovery_and_inference_outside_run_lock(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    job=study.ask(token,{"operation_id":"lease","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"why"})
    with study.store.transaction() as db:
        db.execute(update(questions).where(questions.c.id==job["id"]).values(status="running",lease_token="old",lease_until=time.time()-10))
    started=threading.Event();finish_explanation=threading.Event()
    class Blocking:
        def generate(self,*args,**kwargs):
            started.set();assert finish_explanation.wait(10)
            return {"title":"Ready","text":"Verified fixture","frame":0,"verified":True}
    study.explainer=Blocking()
    worker=threading.Thread(target=study.process_one_question);worker.start();assert started.wait(10)
    state=command(study,token,"action",action="WAIT")
    assert state["state"]["turn"]==1
    finish_explanation.set();worker.join(10);assert not worker.is_alive()
    assert study.question(token,job["id"])["status"]=="complete"


def test_technical_retry_preserves_assignment_and_prior_run(study):
    token,view,code=join(study);to_practice(study,token)
    with study.store.transaction() as db:before=study.store.run(db,token)
    retry=study.technical_retry({"operation_id":"admin-retry","run_id":before["id"],"reason":"Browser was closed during supervised test"})
    app=create_app(study,start_workers=False)
    with TestClient(app) as browser:
        browser.cookies.set(COOKIE,"r."+token,domain="testserver.local",path="/")
        closed=browser.get("/api/view")
        assert closed.status_code==200 and closed.json()["run"]["phase"]=="technical_retry_closed"
        response=browser.post("/api/session",json={"operation_id":"resumed-after-retry","mode":"pilot","participant_id":code})
        assert response.status_code==200
        resumed=response.json();token2=browser.cookies[COOKIE][2:]
        assert browser.get("/api/view").json()["run"]["id"]==retry["run_id"]
    with study.store.transaction() as db:after=study.store.run(db,token2)
    assert retry["retry_id"]==after["retry_id"]==1
    assert before["condition"]==after["condition"] and before["task_order"]==after["task_order"]
    assert before["id"]!=after["id"] and after["phase"]=="consent"
    with pytest.raises(StudyError) as revoked:
        study.view(token)
    assert revoked.value.code=="session_not_found"


def test_technical_retry_old_cookie_can_migrate_only_once(study):
    token,view,participant_id=join(study)
    retry=study.technical_retry({"operation_id":"admin-one-use-retry","run_id":view["run"]["id"],
                                 "reason":"Two-browser session migration regression"})
    app=create_app(study,start_workers=False)
    first=TestClient(app);second=TestClient(app)
    for browser in (first,second):
        browser.cookies.set(COOKIE,"r."+token,domain="testserver.local",path="/")
    migrated=first.post("/api/session",json={"operation_id":"first-migration","mode":"pilot",
                                              "participant_id":participant_id})
    assert migrated.status_code==200 and migrated.json()["run"]["id"]==retry["run_id"]
    first_token=first.cookies[COOKIE][2:]
    denied=second.post("/api/session",json={"operation_id":"second-migration","mode":"pilot",
                                             "participant_id":participant_id})
    assert denied.status_code==409 and denied.json()["code"]=="participant_id_taken"
    assert study.view(first_token)["run"]["id"]==retry["run_id"]
    # A lost first response is recoverable with the same idempotency key even
    # after the old migration credential has been revoked.
    replay=second.post("/api/session",json={"operation_id":"first-migration","mode":"pilot",
                                             "participant_id":participant_id})
    assert replay.status_code==200 and replay.json()["run"]["id"]==retry["run_id"]
    assert second.cookies[COOKIE][2:]==first_token


def test_only_direct_predecessor_cookie_can_migrate_retry_chain(study):
    original_token,view,participant_id=join(study)
    first=study.technical_retry({"operation_id":"admin-chain-one","run_id":view["run"]["id"],
                                 "reason":"First supervised recovery"})
    with pytest.raises(StudyError) as unclaimed:
        study.technical_retry({"operation_id":"admin-chain-too-soon","run_id":first["run_id"],
                               "reason":"Must not strand the participant before claim"})
    assert unclaimed.value.status==409 and unclaimed.value.code=="technical_retry_not_resumed"
    first_token,first_view=study.join({"operation_id":"claim-first-retry","mode":"pilot",
                                      "participant_id":participant_id},existing_token=original_token)
    assert first_view["run"]["id"]==first["run_id"]
    second=study.technical_retry({"operation_id":"admin-chain-two","run_id":first["run_id"],
                                  "reason":"Second supervised recovery"})
    with study.store.transaction() as db:
        # Simulate a legacy release that did not revoke an even older cookie.
        leaked="legacy-valid-older-cookie"
        db.execute(update(runs).where(runs.c.id==view["run"]["id"]).values(token_hash=token_digest(leaked)))
        active_hash=db.execute(select(runs.c.token_hash).where(runs.c.id==second["run_id"])).scalar_one()
    with pytest.raises(StudyError) as stale:
        study.join({"operation_id":"skip-retry-generation","mode":"pilot",
                    "participant_id":participant_id},existing_token=leaked)
    assert stale.value.status==409 and stale.value.code=="session_superseded"
    with study.store.transaction() as db:
        assert db.execute(select(runs.c.token_hash).where(runs.c.id==second["run_id"])).scalar_one()==active_hash
    final_token,final_view=study.join({"operation_id":"claim-second-retry","mode":"pilot",
                                      "participant_id":participant_id},existing_token=first_token)
    assert final_token!=first_token and final_view["run"]["id"]==second["run_id"]


def test_legacy_research_run_can_enter_current_release_through_audited_retry(study):
    token,view,participant_id=join(study)
    legacy_versions={"ui":"cooperative_kitchen_web_v2","protocol":"legacy_invitation_v2"}
    with study.store.transaction() as db:
        legacy=study.store.run(db,token);legacy.pop("enrollment_mode",None);legacy["versions"]=legacy_versions
        study.store.save_run(db,legacy)
    study.enrollment_mode="internal_pilot"
    with pytest.raises(StudyError) as old_release:
        study.view(token)
    assert old_release.value.code=="release_changed"
    retry=study.technical_retry({"operation_id":"legacy-v3-migration","run_id":view["run"]["id"],
                                 "reason":"Move a retained v2 run into the reviewed v3 retry flow"})
    assert study.view(token)["run"]["phase"]=="technical_retry_closed"
    new_token,new_view=study.join({"operation_id":"claim-v3-migration","mode":"pilot",
                                   "participant_id":participant_id},existing_token=token)
    assert new_view["run"]["id"]==retry["run_id"] and new_view["run"]["phase"]=="consent"
    with study.store.transaction() as db:
        previous=study.store.run_by_id(db,view["run"]["id"]);current=study.store.run(db,new_token)
        created=json.loads(db.execute(select(events.c.document).where(events.c.run_id==retry["run_id"],
            events.c.kind=="technical_retry_created")).scalar_one())
    assert previous["versions"]==legacy_versions and "enrollment_mode" not in previous
    assert current["versions"]==study.versions and current["enrollment_mode"]=="internal_pilot"
    assert created["previous_versions"]==legacy_versions and created["new_versions"]==study.versions
    assert created["previous_enrollment_mode"] is None and created["new_enrollment_mode"]=="internal_pilot"


def test_http_security_cookie_no_cache_admin_and_errors(study):
    app=create_app(study,start_workers=False,admin_key="test-admin-key")
    with TestClient(app) as client:
        assert client.get("/").headers["cache-control"]=="no-store"
        assert client.get("/api/view").status_code==401
        response=client.post("/api/session",json={"operation_id":"browser-join","mode":"freeplay"})
        assert response.status_code==200
        assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=strict" in response.headers["set-cookie"]
        assert response.headers["cache-control"]=="no-store, private"
        assert client.post("/api/command",headers={"origin":"https://evil.invalid"},json={}).status_code==403
        assert client.get("/api/admin/status?key=test-admin-key").status_code==401
        assert client.get("/api/admin/status",headers={"x-kitchen-admin-key":"test-admin-key"}).status_code==200
        assert client.post("/api/command",json={"operation_id":"next","version":0,"command":"next"}).status_code==200
        assert client.post("/api/command",json={"operation_id":"stale","version":0,"command":"action","action":"WAIT"}).status_code==409
        assert client.get("/api/admin/export",headers={"x-kitchen-admin-key":"test-admin-key"}).status_code==200


def test_old_release_is_rejected_without_erasing_data(study):
    payload={"operation_id":"original-create","mode":"freeplay"}
    token,view=study.join(payload)
    command(study,token,"next")
    study.versions={"fixture":"new-release"}
    recovery=study.view(token)
    assert recovery["requires_restart"] and recovery["can_restart"] and recovery["state"] is None
    with pytest.raises(StudyError) as exc:study.join(payload)
    assert exc.value.code=="release_changed"
    restarted=command(study,token,"restart")
    assert restarted["state"]["turn"]==0 and len(restarted["episodes"])==2
    assert view["run"]["id"] in study.store.export()


def test_program_counterfactual_does_not_mutate_real_state(tmp_path):
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True)
    token,_,_=join(app,"freeplay");view=command(app,token,"next")
    before=copy.deepcopy(view["state"])
    job=app.ask(token,{"operation_id":"cf","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"counterfactual"})
    assert app.process_one_question()
    answer=app.question(token,job["id"])["answer"]
    assert "三步" in answer["text"] and "真实回合没有推进" in answer["text"]
    assert app.view(token)["state"]==before


def test_inflight_answer_cancelled_on_task2_transition(study):
    token,_,_=join_a(study);view=to_task1(study,token)
    job=study.ask(token,{"operation_id":"inflight","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"why"})
    started=threading.Event();release_job=threading.Event()
    class Blocking:
        def generate(self,*args,**kwargs):
            started.set();assert release_job.wait(20)
            return {"title":"Old Task 1 answer","text":"Must never leak to Task 2","frame":0,"verified":True}
    study.explainer=Blocking()
    worker=threading.Thread(target=study.process_one_question);worker.start();assert started.wait(5)
    for _ in range(3):finish(study,token);view=command(study,token,"next")
    assert view["run"]["phase"]=="task2"
    release_job.set();worker.join(10)
    assert study.view(token)["questions"]==[]
    with pytest.raises(StudyError):study.question(token,job["id"])
    with study.store.transaction() as db:
        row=db.execute(select(questions.c.status).where(questions.c.id==job["id"])).first()
    assert row[0]=="cancelled"


def test_freeplay_namespace_is_separate_from_research_export(study,tmp_path):
    debug=KitchenStudy(tmp_path,str(study.store.engine.url),namespace="development",allow_sqlite=True,
        allow_freeplay_qa=False)
    app=create_app(study,freeplay_study=debug,start_workers=False,admin_key="admin")
    with TestClient(app) as client:
        response=client.post("/api/session",json={"operation_id":"isolated-freeplay","mode":"freeplay"})
        debug_id=response.json()["run"]["id"]
        assert client.cookies[COOKIE].startswith("d.")
        assert client.get("/api/view").json()["run"]["id"]==debug_id
        assert client.post("/api/command",json={"operation_id":"debug-start","version":0,"command":"next"}).status_code==200
        assert not client.get("/api/view").json()["can_ask"]
        blocked_question=client.post("/api/question",json={"operation_id":"no-paid-freeplay","version":1,
            "episode_id":client.get("/api/view").json()["run"]["episode_id"],"frame":0,"kind":"why"})
        assert blocked_question.status_code==403
        exported=client.get("/api/admin/export",headers={"x-kitchen-admin-key":"admin"}).text
        assert debug_id not in exported
        assert debug_id in debug.store.export()
        cookie=client.cookies[COOKIE]
        client.cookies.set(COOKIE,"r."+cookie[2:],domain="testserver.local",path="/")
        assert client.get("/api/view").status_code==401

    # A raw, pre-v3 free-play cookie lives in the old study namespace. It may
    # start a fresh d.* session without being mistaken for an enrolled person.
    legacy_token,legacy_view,_=join(study,"freeplay")
    with TestClient(app) as legacy_browser:
        legacy_browser.cookies.set(COOKIE,legacy_token,domain="testserver.local",path="/")
        moved=legacy_browser.post("/api/session",json={"operation_id":"move-legacy-freeplay",
            "mode":"freeplay","language":"zh"})
        assert moved.status_code==200 and legacy_browser.cookies[COOKIE].startswith("d.")
        assert moved.json()["run"]["id"]!=legacy_view["run"]["id"]
        assert moved.json()["run"]["id"] in debug.store.export()
        research_export=study.store.export()
        assert legacy_view["run"]["id"] in research_export and moved.json()["run"]["id"] not in research_export

    with TestClient(app) as participant_browser:
        enrolled=participant_browser.post("/api/session",json={"operation_id":"research-first","mode":"pilot","participant_id":"Tabs01"})
        assert enrolled.status_code==200
        run_id=enrolled.json()["run"]["id"]
        research_cookie=participant_browser.cookies[COOKIE]
        assert research_cookie.startswith("r.")
        blocked=participant_browser.post("/api/session",json={"operation_id":"freeplay-other-tab","mode":"freeplay"})
        assert blocked.status_code==409 and blocked.json()["code"]=="participant_session_conflict"
        assert participant_browser.cookies[COOKIE]==research_cookie
        assert participant_browser.get("/api/view").json()["run"]["id"]==run_id
        resumed=participant_browser.post("/api/session",json={"operation_id":"research-resume","mode":"pilot","participant_id":"tabs01"})
        assert resumed.status_code==200 and resumed.json()["run"]["id"]==run_id
    debug.store.engine.dispose()


def test_pending_old_release_job_is_cancelled_before_inference(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    job=study.ask(token,{"operation_id":"old-pending","version":view["run"]["version"],"episode_id":view["run"]["episode_id"],"frame":0,"kind":"why"})
    class ShouldNotCall:
        def generate(self,*args,**kwargs):raise AssertionError("A new release must not answer an old frame's pending job")
    study.explainer=ShouldNotCall();study.versions={"fixture":"updated"}
    assert study.process_one_question()
    with study.store.transaction() as db:
        row=db.execute(select(questions.c.status).where(questions.c.id==job["id"])).first()
    assert row[0]=="cancelled"


def test_corrupt_version_and_frame_types_rejected_without_progress(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    assert view["run"]["version"]==1
    with pytest.raises(StudyError):study.command(token,{"operation_id":"boolean-version","version":True,"command":"action","action":"UP"})
    with pytest.raises(StudyError):study.ask(token,{"operation_id":"boolean-frame","version":1,"episode_id":view["run"]["episode_id"],"frame":False,"kind":"why"})
    assert study.view(token)["state"]["turn"]==0


def test_freeplay_old_episode_answers_do_not_rebind_to_new_actor(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    old_episode=view["run"]["episode_id"]
    old_request={"operation_id":"old-actor-answer","version":view["run"]["version"],"episode_id":old_episode,"frame":0,"kind":"why"}
    answered=study.ask(token,old_request);assert study.process_one_question()
    current=study.view(token)
    pending=study.ask(token,{"operation_id":"pending-old-actor","version":current["run"]["version"],"episode_id":old_episode,"frame":0,"kind":"waiting"})
    study.versions={"fixture":"new-frozen-actor"}
    restarted=command(study,token,"restart")
    assert restarted["can_ask"] and restarted["questions"]==[]
    old_history=study.history(token,old_episode)
    assert not old_history["can_ask"] and old_history["questions"]==[] and len(old_history["frames"])==1
    for operation in [lambda:study.question(token,answered["id"]),lambda:study.ask(token,old_request),lambda:study.exposure(token,{"operation_id":"old-shown","question_id":answered["id"],"event":"shown"})]:
        with pytest.raises(StudyError) as exc:operation()
        assert exc.value.status==403
    assert study.process_one_question()
    with study.store.transaction() as db:
        row=db.execute(select(questions.c.status).where(questions.c.id==pending["id"])).first()
    assert row[0]=="cancelled"


def test_legacy_episode_without_version_is_replay_only(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    with study.store.transaction() as db:
        episode=study.store.episode(db,view["run"]["episode_id"],view["run"]["id"])
        episode.pop("versions");study.store.save_episode(db,episode)
    assert not study.view(token)["can_ask"]
    assert not study.history(token,episode["id"])["can_ask"]


def test_missing_or_malformed_consent_uses_draft_and_blocks_recruitment(tmp_path):
    for malformed in ({},{"version":"approved"},{"version":"approved","title":{"zh":"标题","en":"Title"},"text":{"zh":"","en":"Text"}}):
        app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,release={"consent":malformed})
        assert app.consent==DEFAULT_CONSENT
        assert "researcher_participant_information" in app.status()["missing_configuration"]


def frozen_rating_release():
    value=release()
    value.update(survey_scales=[
        {"id":"cooperation_understanding","prompt":{"zh":"冻结的合作题项。","en":"Frozen cooperation item."}},
        {"id":"predictability","prompt":{"zh":"冻结的预测题项。","en":"Frozen prediction item."}},
        {"id":"difficulty","prompt":{"zh":"冻结的难度题项。","en":"Frozen difficulty item."}}],
        scale_range=[1,7],scale_anchors={"zh":["冻结下限","冻结上限"],"en":["Frozen lower anchor","Frozen upper anchor"]})
    return value


def test_frozen_rating_ids_wording_range_and_anchors_are_served_exactly(tmp_path):
    value=frozen_rating_release()
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,test_mode=True,release=value)
    ratings=app._survey_items()[-3:]
    assert [item["id"] for item in ratings]==[item["id"] for item in value["survey_scales"]]
    assert [item["prompt"] for item in ratings]==[item["prompt"] for item in value["survey_scales"]]
    for item in ratings:
        assert [option["value"] for option in item["options"]]==list(map(str,range(1,8)))
        assert item["options"][0]["label"]=={"zh":"1 · 冻结下限","en":"1 · Frozen lower anchor"}
        assert item["options"][-1]["label"]=={"zh":"7 · 冻结上限","en":"7 · Frozen upper anchor"}
    assert "understanding" not in {item["id"] for item in app._survey_items()}


@pytest.mark.parametrize("damage",["missing","duplicate","wrong_id","range","boolean_range","anchors","wording"])
def test_invalid_frozen_ratings_never_fall_back_in_research(tmp_path,damage,monkeypatch):
    value=frozen_rating_release()
    if damage=="missing":value.pop("survey_scales")
    elif damage=="duplicate":value["survey_scales"][1]["id"]="cooperation_understanding"
    elif damage=="wrong_id":value["survey_scales"][0]["id"]="understanding"
    elif damage=="range":value["scale_range"]=[1,5]
    elif damage=="boolean_range":value["scale_range"]=[True,7]
    elif damage=="anchors":value["scale_anchors"]["en"]=["Low"]
    else:value["survey_scales"][0]["prompt"]["zh"]=""
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,release=value)
    assert app._rating_items is None
    assert "frozen_survey_scales" in app.status()["missing_configuration"]
    assert app._survey_items()[-3]["id"]=="understanding"  # development compatibility only
    with monkeypatch.context() as patch:
        patch.setattr(app.store,"namespace","pilot")
        with pytest.raises(StudyError) as exc:app._survey_items()
        assert exc.value.code=="survey_artifact_invalid"


def test_rating_csv_preserves_legacy_understanding_and_frozen_id(study):
    from sqlalchemy import insert
    from ui.cooperative_kitchen_store import surveys
    legacy,legacy_view,_=join(study,"freeplay")
    current,current_view,_=join(study,"freeplay")
    with study.store.transaction() as db:
        for view,answers in [(legacy_view,{"understanding":"2","predictability":"3","difficulty":"4"}),
                             (current_view,{"cooperation_understanding":"5","predictability":"6","difficulty":"7"})]:
            db.execute(insert(surveys).values(run_id=view["run"]["id"],document=encode({"answers":answers}),submitted=time.time()))
    rows={row["run_id"]:row for row in csv.DictReader(io.StringIO(study.store.export("csv")))}
    assert rows[legacy_view["run"]["id"]]["cooperation_understanding"]=="2"
    assert rows[current_view["run"]["id"]]["cooperation_understanding"]=="5"
    assert rows[current_view["run"]["id"]]["predictability"]=="6"
    assert rows[current_view["run"]["id"]]["difficulty"]=="7"


def test_auto_step_is_denied_for_all_research_rounds(study):
    token,view,_=join(study)
    assert not view["can_auto"]
    with pytest.raises(StudyError) as exc:command(study,token,"auto_step")
    assert exc.value.status==403
    view=to_practice(study,token)
    assert not view["can_auto"]
    before=copy.deepcopy(view)
    with pytest.raises(StudyError) as exc:command(study,token,"auto_step")
    assert exc.value.status==403
    assert study.view(token)==before


@pytest.mark.parametrize("swapped",[False,True])
def test_auto_step_preserves_before_state_neural_action_and_idempotency(study,swapped):
    from env.cooperative_kitchen import program_decision
    from ui.cooperative_kitchen_store import events
    token,view,_=join(study,"freeplay")
    assert not view["can_auto"]
    view=command(study,token,"next")
    if swapped:view=command(study,token,"swap")
    assert view["can_auto"]
    with study.store.transaction() as db:
        ep=study.store.episode(db,view["run"]["episode_id"],view["run"]["id"])
        before=ep["snapshot"]
    independent=CooperativeKitchen();independent.restore(before)
    human_action=program_decision(independent,"human")["action"]
    assert independent.snapshot()==before
    captured=[]
    class DistinctPolicy(FixturePolicy):
        def act(self,observations):
            captured.append({key:value.copy() for key,value in observations.items()})
            actions,dist=super().act(observations)
            actions["ai"]="RIGHT";dist["ai"]["chosen_action"]="RIGHT"
            return actions,dist
    study.policy=DistinctPolicy()
    expected_observations=independent.observations()
    independent.step({"human":human_action,"ai":"RIGHT"})
    payload={"operation_id":"automatic-confirmed","version":view["run"]["version"],"command":"auto_step","action":"DOWN"}
    confirmed=study.command(token,payload)
    repeated=study.command(token,payload)
    assert confirmed==repeated and len(captured)==1
    assert all((captured[0][key]==expected_observations[key]).all() for key in expected_observations)
    with study.store.transaction() as db:
        current=study.store.episode(db,view["run"]["episode_id"],view["run"]["id"])
        stored_before,_=study.store.frame(db,ep["id"],0)
        row=db.execute(select(events.c.document).where(events.c.run_id==view["run"]["id"],events.c.kind=="joint_step")).scalar_one()
    event=json.loads(row)
    assert stored_before==before and current["snapshot"]==independent.snapshot()
    assert event["human_command"]==human_action
    assert event["input_source"]=="program_demonstration"
    assert event["actual_actions"]["ai"]=="RIGHT"
    assert event["before"]==before and event["after"]==independent.snapshot()
    assert event["human_program_decision"]["facts"]["side"]==("right" if swapped else "left")


def test_auto_step_stops_at_terminal_and_release_boundary(study):
    token,_,_=join(study,"freeplay");view=command(study,token,"next")
    while not view["state"]["done"]:view=command(study,token,"auto_step")
    assert view["state"]["turn"]==180 and not view["can_auto"]
    with pytest.raises(StudyError):command(study,token,"auto_step")
    study.versions={"fixture":"new-release"}
    assert not study.view(token)["can_auto"]
    with pytest.raises(StudyError) as exc:command(study,token,"auto_step")
    assert exc.value.code=="release_changed"


def provider_release(provider):
    from backend.cooperative_kitchen.study import build_default_consent
    value=release()
    value["qa_configuration"]={"provider":provider,"model":"deepseek-v4-flash" if provider=="deepseek" else "qwen-plus-2025-12-01",
                               "model_version_pinned":provider=="qwen"}
    value["qa_configured"]=True
    value["qa_required_key_env"]="DEEPSEEK_API_KEY" if provider=="deepseek" else "DASHSCOPE_API_KEY"
    value["consent"]=build_default_consent(value["qa_configuration"])
    return value


def test_deepseek_consent_has_no_stale_qwen_region_or_snapshot_claim(tmp_path):
    from backend.cooperative_kitchen.study import build_default_consent
    cfg=provider_release("deepseek")
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,release=cfg)
    serialized=json.dumps(app.consent,ensure_ascii=False)
    assert "DeepSeek API" in app.consent["text"]["zh"] and "DeepSeek API" in app.consent["text"]["en"]
    assert not any(value in serialized for value in ("Qwen","千问","新加坡","Singapore","固定模型","pinned snapshot"))
    assert app.status()["qa_required_key_env"]=="DEEPSEEK_API_KEY"
    assert app.status()["qa_configuration"]["model_version_pinned"] is False
    assert "researcher_participant_information" in app.status()["missing_configuration"]
    # An otherwise valid old provider statement is never silently reused.
    cfg["consent"]=build_default_consent(provider_release("qwen")["qa_configuration"])
    changed=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,release=cfg)
    assert changed.consent["qa_provider"]=="deepseek"
    assert "DeepSeek API" in changed.consent["text"]["zh"]


def test_provider_change_invalidates_old_research_session_without_mutating_it(study):
    old=KitchenStudy(study.output,str(study.store.engine.url),namespace="test",allow_sqlite=study.store.is_sqlite,
                     policy=FixturePolicy(),explainer=FixtureExplainer(),release=provider_release("qwen"),test_mode=True,enrollment_mode="formal")
    token,_,_=join(old);to_practice(old,token)
    with old.store.transaction() as db:before=copy.deepcopy(old.store.run(db,token))
    new=KitchenStudy(study.output,str(study.store.engine.url),namespace="test",allow_sqlite=study.store.is_sqlite,
                     policy=FixturePolicy(),explainer=FixtureExplainer(),release=provider_release("deepseek"),test_mode=True)
    with pytest.raises(StudyError) as exc:new.view(token)
    assert exc.value.code=="release_changed"
    assert old.versions["qa_configuration_sha256"]!=new.versions["qa_configuration_sha256"]
    with new.store.transaction() as db:after=new.store.run(db,token)
    assert after==before
    assert "qwen" in after["consent"]["version"]


def test_legacy_draft_version_does_not_become_reviewed_information(tmp_path):
    old=copy.deepcopy(DEFAULT_CONSENT);old["version"]="kitchen-consent-draft-v1"
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True,release={"consent":old})
    assert "researcher_participant_information" in app.status()["missing_configuration"]


@pytest.mark.parametrize("value", ["P01", "a"*32, "P_A-9", "  P01  "])
def test_participant_id_accepts_documented_ascii_format(study, value):
    token, view = study.join({"operation_id":"valid-id", "mode":"pilot", "participant_id":value})
    assert view["run"]["participant_id"] == value.strip()
    with study.store.transaction() as db:
        participant = db.execute(select(participants)).mappings().one()
        assert participant["participant_key"] == value.strip().lower()
        assert participant["active_run"] == study.store.run(db, token)["id"]


@pytest.mark.parametrize("value", [None, "", "ab", "1abc", "a"*33, "中abc", "ab c", "abc/def", "abc\ndef", 123])
def test_invalid_participant_id_never_allocates(study, value):
    with pytest.raises(StudyError) as exc:
        study.join({"operation_id":"invalid-id", "mode":"pilot", "participant_id":value})
    assert exc.value.code == "participant_id_invalid"
    with study.store.transaction() as db:
        assert not db.execute(select(participants)).all()
        assert not db.execute(select(blocks)).all()


def test_participant_normalization_is_not_an_authentication_credential(study):
    payload = {"operation_id":"original", "mode":"pilot", "participant_id":"  Pilot_01  "}
    token, initial = study.join(payload)
    other, _, _ = join(study)
    with pytest.raises(StudyError) as exc:
        study.join({**payload, "operation_id":"new", "participant_id":"pilot_01"})
    assert exc.value.status == 409 and exc.value.code == "participant_id_taken"
    with pytest.raises(StudyError) as exc:
        study.join({**payload, "operation_id":"other-session", "participant_id":"pilot_01"},existing_token=other)
    assert exc.value.status == 409 and exc.value.code == "participant_session_conflict"
    # The exact operation ID is a high-entropy recovery nonce for a lost first response.
    replay, repeated = study.join(payload)
    assert replay == token and repeated == initial
    same, resumed = study.join({**payload,"operation_id":"same-browser","participant_id":"PILOT_01"},existing_token=token)
    assert same == token and resumed["run"]["id"] == initial["run"]["id"]
    with pytest.raises(StudyError) as exc:
        study.join({"operation_id":"second-id","mode":"pilot","participant_id":"Another01"},existing_token=token)
    assert exc.value.status == 409 and exc.value.code == "participant_session_conflict"
    with study.store.transaction() as db:
        assert len(db.execute(select(participants)).all()) == 2


def test_concurrent_normalized_registration_allocates_exactly_once(study):
    def register(index):
        try:
            return study.join({"operation_id":f"race-{index}","mode":"pilot",
                "participant_id":"Concurrent01" if index%2 else " concurrent01 "})
        except StudyError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=8) as pool: results=list(pool.map(register,range(8)))
    assert sum(isinstance(result,tuple) for result in results)==1
    assert results.count("participant_id_taken")==7
    with study.store.transaction() as db:
        rows=db.execute(select(participants)).mappings().all()
        assert len(rows)==1 and rows[0]["position"]==0
        assert len(db.execute(select(runs)).all())==1


def test_concurrent_distinct_registration_preserves_four_person_blocks(study):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda index:study.join({"operation_id":f"unique-{index}","mode":"pilot","participant_id":f"User{index:03}"}),range(8)))
    with study.store.transaction() as db:
        rows=db.execute(select(participants).order_by(participants.c.position)).mappings().all()
    assert [row["position"] for row in rows]==list(range(8))
    for block in (rows[:4],rows[4:]):
        assert {(row["condition"],row["task_order"]) for row in block}=={("A","XY"),("A","YX"),("B","XY"),("B","YX")}
    assert all("condition" not in view["run"] and "task_order" not in view["run"] for _,view in results)


def test_http_session_cookie_receipt_and_duplicate_browser(study):
    app=create_app(study,start_workers=False,admin_key="admin")
    with TestClient(app) as original, TestClient(app) as other:
        status=original.get("/api/status")
        assert set(status.json()["enrollment"])=={"mode","enabled","formal_ready","participant_id_pattern","participant_id_example"}
        assert status.json()["enrollment"]["enabled"]
        assert status.json()["enrollment"]["participant_id_example"]=="user_01"
        assert "set-cookie" not in status.headers
        payload={"operation_id":"http-initial", "mode":"pilot", "participant_id":"Browser01"}
        initial=original.post("/api/session",json=payload)
        assert initial.status_code==200
        original.cookies.delete(COOKIE)  # Simulate loss of the first session response.
        repeated=original.post("/api/session",json=payload)
        assert repeated.status_code==200 and repeated.json()==initial.json()
        other.get("/api/status")
        denied=other.post("/api/session",json={**payload,"operation_id":"other-op","participant_id":" browser01 "})
        assert denied.status_code==409 and denied.json()["code"]=="participant_id_taken"
        resumed=original.post("/api/session",json={**payload,"operation_id":"resume"})
        assert resumed.status_code==200 and resumed.json()==initial.json()
        conflict=original.post("/api/session",json={"operation_id":"another-id","mode":"pilot","participant_id":"Browser02"})
        assert conflict.status_code==409 and conflict.json()["code"]=="participant_session_conflict"
        with study.store.transaction() as db:
            assert len(db.execute(select(participants)).all())==1
        assert original.post("/api/admin/invitations",json={}).status_code==401
        retired=original.post("/api/admin/invitations",json={"count":4},headers={"x-kitchen-admin-key":"admin"})
        assert retired.status_code==410 and retired.json()["code"]=="invitations_retired"


def test_closed_enrollment_preserves_confirmed_progress(study):
    token, view, participant_id=join(study)
    study.enrollment_mode="closed"
    assert not study.status()["enrollment"]["enabled"]
    assert study.status()["enrollment"]["formal_ready"]
    with pytest.raises(StudyError) as exc:join(study)
    assert exc.value.code=="enrollment_closed"
    assert study.view(token)==view
    assert command(study,token,"consent",accepted=True)["run"]["phase"]=="instructions"
    resumed, _=study.join({"operation_id":"closed-resume","mode":"pilot","participant_id":participant_id},existing_token=token)
    assert resumed==token


def test_internal_candidate_admission_does_not_unlock_formal_gate(tmp_path):
    candidate=frozen_rating_release()
    candidate.update(study_ready=False,qa_configured=True,
        qa_configuration={"provider":"deepseek","model":"deepseek-v4-flash","model_version_pinned":False},
        missing_configuration=["extraction_gate","remote_load_gate","qa_model_snapshot_unpinned"])
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,
        policy=FixturePolicy(),explainer=FixtureExplainer(),release=candidate,enrollment_mode="internal_pilot")
    status=app.status()
    assert not status["study_ready"] and not status["enrollment"]["formal_ready"] and status["enrollment"]["enabled"]
    token, view, _=join(app)
    with app.store.transaction() as db:
        assert app.store.run(db,token)["enrollment_mode"]=="internal_pilot"
    exported=list(csv.DictReader(io.StringIO(app.store.export("csv"))))
    assert exported[0]["enrollment_mode"]=="internal_pilot"
    assert exported[0]["namespace"]=="test" and exported[0]["mode"]=="pilot"
    assert to_practice(app,token)["can_act"]
    app.enrollment_mode="formal"
    assert not app.status()["enrollment"]["enabled"]
    with pytest.raises(StudyError) as exc:join(app)
    assert exc.value.code=="study_not_ready"
    assert command(app,token,"action",action="WAIT")["state"]["turn"]==1
    app.release["qa_configured"]=False
    with pytest.raises(StudyError) as exc:app.view(token)
    assert exc.value.code=="study_not_ready"


@pytest.mark.parametrize("binding",[
    "qa_configuration_version","qa_configuration_binding",
    "remote_load_configuration_binding","qa_runtime_binding"])
def test_internal_pilot_rejects_unfrozen_qa_or_runtime_binding(tmp_path,binding):
    candidate=frozen_rating_release()
    candidate.update(study_ready=False,qa_configured=True,missing_configuration=["extraction_gate","remote_load_gate",binding])
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,
        policy=FixturePolicy(),explainer=FixtureExplainer(),release=candidate,test_mode=True,
        enrollment_mode="internal_pilot")
    app.test_mode=False;app.store.is_sqlite=False;app.store.namespace="pilot"
    status=app.status()
    assert not status["enrollment"]["enabled"]
    assert binding in app._internal_pilot_missing()


def test_enrollment_mode_is_strongly_bound_to_data_namespace(tmp_path):
    value=frozen_rating_release()
    frozen_qa={"provider":"qwen","model":"qwen-plus-2025-12-01","model_version_pinned":True}
    reviewed=build_default_consent(frozen_qa);reviewed["version"]="kitchen-consent-reviewed-v1-qwen"
    value.update(qa_configured=True,qa_configuration=frozen_qa,missing_configuration=[],consent=reviewed)
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,
        policy=FixturePolicy(),explainer=FixtureExplainer(),release=value,test_mode=True,
        enrollment_mode="formal")
    # Exercise production gating without requiring a second external database:
    # only status inputs are changed; no operation is written through this stub.
    app.test_mode=False
    app.store.is_sqlite=False
    app.store.namespace="pilot"
    status=app.status()
    assert not status["enrollment"]["formal_ready"]
    assert not status["enrollment"]["enabled"]
    assert "formal_database_namespace" in status["missing_configuration"]

    app.store.namespace="confirmatory"
    status=app.status()
    assert status["enrollment"]["formal_ready"]
    assert status["enrollment"]["enabled"]

    app.enrollment_mode="internal_pilot"
    status=app.status()
    assert status["enrollment"]["formal_ready"]
    assert not status["enrollment"]["enabled"]
    app.store.namespace="pilot"
    status=app.status()
    assert not status["enrollment"]["formal_ready"]
    assert status["enrollment"]["enabled"]


def test_default_and_invalid_enrollment_modes(tmp_path,monkeypatch):
    monkeypatch.delenv("KITCHEN_ENROLLMENT_MODE",raising=False)
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True)
    assert app.status()["enrollment"]["mode"]=="closed"
    monkeypatch.setenv("KITCHEN_ENROLLMENT_MODE","unexpected")
    with pytest.raises(ValueError,match="KITCHEN_ENROLLMENT_MODE"):
        KitchenStudy(tmp_path,"sqlite:///:memory:",allow_sqlite=True)


def test_internal_pilot_information_is_explicit_and_never_formal_approval(tmp_path):
    assert DEFAULT_CONSENT["version"]=="kitchen-consent-internal-pilot-v3"
    for phrase in ("内部预实验","记录操作","问卷","DeepSeek","去标识化","不纳入未来正式研究样本","姓名","邮箱","电话"):
        assert phrase in DEFAULT_CONSENT["text"]["zh"]
    for phrase in ("internal pilot","recorded","questionnaire","DeepSeek","de-identified","not be included in the future formal research sample","names","email addresses","phone numbers"):
        assert phrase in DEFAULT_CONSENT["text"]["en"]
    value=frozen_rating_release()
    value.update(qa_configured=True,consent=copy.deepcopy(DEFAULT_CONSENT))
    app=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,
        policy=FixturePolicy(),explainer=FixtureExplainer(),release=value,enrollment_mode="formal")
    assert "researcher_participant_information" in app.status()["missing_configuration"]
    assert not app.status()["enrollment"]["formal_ready"]
    value["consent"]["version"]="kitchen-consent-draft-v2"
    pilot=KitchenStudy(tmp_path,"sqlite:///:memory:",namespace="test",allow_sqlite=True,
        policy=FixturePolicy(),explainer=FixtureExplainer(),release=value,enrollment_mode="internal_pilot")
    assert pilot.consent["version"]=="kitchen-consent-internal-pilot-v3"


def test_legacy_invitation_migration_is_additive_idempotent_and_retry_safe(study):
    entries=[join(study) for _ in range(4)]
    with study.store.transaction() as db:
        original=[dict(row) for row in db.execute(select(participants).order_by(participants.c.position)).mappings()]
        for row in original:
            db.execute(insert(invitations).values(id="legacy-"+row["id"],namespace=study.store.namespace,
                code_hash=token_digest("old-code-"+row["id"]),participant=row["participant_id"],condition=row["condition"],
                task_order=row["task_order"],position=row["position"],active_run=row["active_run"],created=row["created"]))
        db.execute(delete(participants).where(participants.c.namespace==study.store.namespace))
        before=[dict(row) for row in db.execute(select(invitations).order_by(invitations.c.position)).mappings()]
    url=study.store.engine.url.render_as_string(hide_password=False)
    migrated=KitchenStore(url,study.store.namespace,allow_sqlite=study.store.is_sqlite)
    with migrated.transaction() as db:
        copied=[dict(row) for row in db.execute(select(participants).order_by(participants.c.position)).mappings()]
        assert [{key:row["participant_id"] if key=="participant" else row[key] for key in ("participant","condition","task_order","position","active_run","created")} for row in copied]==[
            {key:row[key] for key in ("participant","condition","task_order","position","active_run","created")} for row in before]
        assert all(row["updated"]>=row["created"] for row in copied)
    token,view,participant_id=entries[0]
    same,_=study.join({"operation_id":"legacy-resume","mode":"pilot","participant_id":participant_id.lower()},existing_token=token)
    assert same==token
    request={"operation_id":"legacy-retry","run_id":view["run"]["id"],"reason":"Technical recovery test"}
    retry=study.technical_retry(request)
    assert study.technical_retry(request)==retry
    again=KitchenStore(url,study.store.namespace,allow_sqlite=study.store.is_sqlite)
    with again.transaction() as db:
        after=[dict(row) for row in db.execute(select(invitations).order_by(invitations.c.position)).mappings()]
        assert after==before
        assert db.execute(select(participants.c.active_run).where(participants.c.participant_key==participant_id.lower())).scalar_one()==retry["run_id"]
        assert len(db.execute(select(participants)).all())==4
    new_token,new_view=study.join({"operation_id":"legacy-retry-resume","mode":"pilot","participant_id":participant_id},existing_token=token)
    assert new_view["run"]["id"]==retry["run_id"] and new_token!=token
    with study.store.transaction() as db:
        assert len(db.execute(select(participants)).all())==4
        assert len(db.execute(select(episodes).where(episodes.c.run_id==retry["run_id"])).all())==0
    migrated.engine.dispose();again.engine.dispose()


def test_legacy_case_collision_does_not_merge_or_erase_allocations(study):
    with study.store.transaction() as db:
        for index,name in enumerate(("Legacy01","legacy01")):
            db.execute(insert(invitations).values(id=f"collision-{index}",namespace=study.store.namespace,
                code_hash=token_digest(f"code-{index}"),participant=name,condition="A" if index == 0 else "B",
                task_order="XY" if index == 0 else "YX",position=index,active_run=f"legacy-run-{index}",created=time.time()))
    with pytest.raises(ValueError,match="collide after normalization"):
        KitchenStore(study.store.engine.url.render_as_string(hide_password=False),study.store.namespace,allow_sqlite=study.store.is_sqlite)
    with study.store.transaction() as db:
        assert len(db.execute(select(invitations)).all())==2
        assert not db.execute(select(participants)).all()


def test_unredeemed_legacy_invitations_are_retained_without_creating_participants(study):
    with study.store.transaction() as db:
        db.execute(insert(invitations).values(id="unredeemed",namespace=study.store.namespace,
            code_hash=token_digest("unused-code"),participant="UnusedLegacy",created=time.time()))
    migrated=KitchenStore(study.store.engine.url.render_as_string(hide_password=False),study.store.namespace,allow_sqlite=study.store.is_sqlite)
    with migrated.transaction() as db:
        assert db.execute(select(invitations.c.id).where(invitations.c.id=="unredeemed")).scalar_one()=="unredeemed"
        assert db.execute(select(participants.c.id).where(participants.c.participant_key=="unusedlegacy")).first() is None
    migrated.engine.dispose()
