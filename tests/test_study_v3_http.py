"""HTTP access-control and actual-process restart checks for the shared service.

All records are marked ``test``. Injected answers only test delivery authorization;
they do not count as the real-provider semantic QA release smoke test.
"""
import csv
import io
import json
import multiprocessing
import threading
import uuid
from contextlib import contextmanager
from http.client import HTTPConnection
from http.cookies import SimpleCookie

import pytest

from study_v3.config import Settings
from study_v3.registry import engine
from study_v3.server import COOKIE, make_server
from tests.test_study_v3_store import Flow, RecordingExplainer, assert_public


ORIGIN = "https://study-http-test.invalid"
ADMIN = "test-only-" + uuid.uuid4().hex


class Client:
    def __init__(self, port):
        self.port = port
        self.cookie = None

    def request(self, method, path, payload=None, headers=None, raw=None):
        merged = {}
        if self.cookie:
            merged["Cookie"] = self.cookie
        data = raw
        if payload is not None:
            data = json.dumps(payload).encode()
            merged.update({"Content-Type": "application/json", "Origin": ORIGIN})
        merged.update(headers or {})
        conn = HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request(method, path, body=data, headers=merged)
            response = conn.getresponse()
            content = response.read()
            out_headers = dict(response.getheaders())
            if "Set-Cookie" in out_headers:
                cookies = SimpleCookie()
                cookies.load(out_headers["Set-Cookie"])
                self.cookie = f"{COOKIE}={cookies[COOKIE].value}"
            value = json.loads(content) if "application/json" in out_headers.get("Content-Type", "") else content.decode()
            return response.status, value, out_headers
        finally:
            conn.close()

    def create(self, domain="pong", group="A", name=None):
        return self.request("POST", "/api/study/session", {"participant_id": name or "http-test-" + uuid.uuid4().hex,
            "domain": domain, "group": group, "mode": "test", "consent": True},
            {"Authorization": "Bearer " + ADMIN})

    def command(self, kind, view, **fields):
        return self.request("POST", "/api/study/" + kind,
            {"instance_id": view["instance_id"], "revision": view["revision"],
             "command_id": uuid.uuid4().hex, **fields})


@pytest.fixture
def http_service(tmp_path):
    settings = Settings(database=str(tmp_path / "http.sqlite3"), origin=ORIGIN, admin_token=ADMIN,
        llm_base_url="https://private-model-provider.invalid/v1", llm_model="private-model-name",
        llm_api_key="test-only-secret-api-key")
    server = make_server(settings, port=0, explainer=RecordingExplainer())
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, Client(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


@pytest.mark.parametrize("path", ["/", "/warehouse/", "/pong/", "/kitchen/"])
def test_same_origin_routes_default_english_and_security_headers(http_service, path):
    _, client = http_service
    status, html, headers = client.request("GET", path)
    assert status == 200 and '<html lang="en">' in html
    assert 'id="englishButton" lang="en"' in html and "中文" in html
    assert "no-store" in headers["Cache-Control"]
    assert headers["X-Frame-Options"] == "DENY"
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert "connect-src 'self'" in headers["Content-Security-Policy"]
    for asset in ("/study-assets/app.js", "/study-assets/styles.css", "/study-assets/favicon.svg"):
        assert client.request("GET", asset)[0] == 200


def test_release_liveness_and_public_metadata_do_not_expose_configuration(http_service):
    server, client = http_service
    status, health, _ = client.request("GET", "/health")
    assert status == 200 and health["status"] == "ok"
    assert health["study_ready"] is False
    assert set(health["domains"]) == {"warehouse", "pong", "kitchen"}
    status, release, _ = client.request("GET", "/api/release")
    assert status == 200
    assert release["source_sha256"] == server.release["source_sha256"]
    assert len(release["source_sha256"]) == 64
    assert release["human_effect_status"] == "not_measured"
    assert release["explanations"] == {"group": "A", "task": 2, "active_only": True,
        "automatic_for_new_enrollments": False, "automatic_version": "four-step-question-guide-v6"}
    serialized = json.dumps(release)
    for secret in (ADMIN, "private-model-provider", "private-model-name", "test-only-secret-api-key", "http.sqlite3"):
        assert secret not in serialized
    assert client.request("GET", "/kitchen")[2]["Location"] == "/kitchen/"
    assert client.request("GET", "/study-assets/../../study_v3/config.py")[0] == 404


def test_http_session_cookie_consent_modes_and_origin_protection(http_service):
    _, client = http_service
    assert client.request("GET", "/api/study/view")[0] == 401
    assert client.request("GET", "/api/study/view", headers={"Cookie": COOKIE + "=forged"})[0] == 401
    payload = {"participant_id": "http-denied", "domain": "pong", "mode": "test", "consent": True}
    assert client.request("POST", "/api/study/session", payload)[0:2] == (403, {"error": "researcher_access_required"})
    assert client.request("POST", "/api/study/session", {**payload, "mode": "pilot"})[0:2] == (503, {"error": "study_not_ready"})
    status, view, headers = client.create()
    assert status == 200 and view["stage"] == "demo"
    for flag in ("HttpOnly", "SameSite=Strict", "Secure", "Path=/"):
        assert flag in headers["Set-Cookie"]
    assert client.request("GET", "/api/study/view?domain=pong")[1]["instance_id"] == view["instance_id"]
    request = {"instance_id": view["instance_id"], "revision": 0, "command_id": uuid.uuid4().hex}
    for headers in ({"Origin": "https://attacker.invalid"}, {"Sec-Fetch-Site": "cross-site"}):
        assert client.request("POST", "/api/study/demo_next", request, headers)[0:2] == (403, {"error": "origin_not_allowed"})
    assert client.request("GET", "/api/study/view")[1]["revision"] == 0
    assert client.request("POST", "/api/study/demo_next", request, {"Content-Type": "text/plain"})[0] == 415
    assert client.request("POST", "/api/study/demo_next", raw=b"not-json", headers={"Content-Type": "application/json"})[0] == 400
    assert client.request("POST", "/api/study/demo_next", raw=b"x" * 100001, headers={"Content-Type": "application/json"})[0] == 413


def test_http_direct_ask_and_frame_access_cannot_bypass_group_or_phase(http_service):
    server, client = http_service
    # Drive actual flow through the authoritative store, then exercise only HTTP authorization.
    flow = Flow(server.store, "pong", "A")
    client.cookie = COOKIE + "=" + flow.token
    question = flow.question()
    assert client.request("POST", "/api/study/ask", question)[0] == 403
    flow.finish_demo()
    assert client.request("POST", "/api/study/ask", flow.question())[0] == 403
    flow.finish_task(); flow.command("next")
    question = flow.question(target_run=flow.view["task_runs"][0]["id"], turn=0)
    status, answer, headers = client.request("POST", "/api/study/ask", question)
    assert status == 200 and answer["status"] == "answered"
    assert_public(answer)
    assert "no-store" in headers["Cache-Control"]
    assert client.request("POST", "/api/study/answer-displayed", {"instance_id": flow.view["instance_id"], "question_id": answer["id"]})[0] == 200
    outsider = Flow(server.store, "kitchen", "B")
    outsider_client = Client(server.server_port)
    outsider_client.cookie = COOKIE + "=" + outsider.token
    assert outsider_client.request("GET", "/api/study/view?instance_id=" + flow.view["instance_id"])[0] == 404
    frame_query = f"/api/study/frame?instance_id={flow.view['instance_id']}&run_id={question['target_run']}&turn=0"
    assert outsider_client.request("GET", frame_query)[0] == 404
    assert_public(client.request("GET", frame_query)[1])
    outsider.start_task2()
    assert outsider_client.request("POST", "/api/study/ask", outsider.question())[0] == 403
    flow.finish_task()
    for stage in ("terminal-summary", "task3"):
        if stage == "task3": flow.command("next")
        assert client.request("POST", "/api/study/ask", question)[0] == 403
        assert client.request("POST", "/api/study/answer-displayed", {"instance_id": flow.view["instance_id"], "question_id": answer["id"]})[0] == 403
        visible = client.request("GET", "/api/study/view")[1]
        assert not visible["can_ask"] and not visible["questions"]


def test_admin_jsonl_csv_exports_require_auth_and_exclude_session_credentials(http_service):
    server, client = http_service
    flow = Flow(server.store, "pong", "A")
    flow.start_task2()
    server.store.ask(flow.token, flow.question())
    for header in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic " + ADMIN}):
        assert client.request("GET", "/api/study/admin/export", headers=header)[0] == 403
    auth = {"Authorization": "Bearer " + ADMIN}
    status, body, _ = client.request("GET", "/api/study/admin/export", headers=auth)
    assert status == 200
    records = [json.loads(line) for line in body.splitlines()]
    assert {r["table"] for r in records} >= {"participants", "instances", "runs", "frames", "questions", "releases"}
    assert "sessions" not in {r["table"] for r in records}
    assert "recovery_hash" not in body and "token_hash" not in body
    assert flow.token not in body and flow.recovery not in body and ADMIN not in body
    question = next(r["record"] for r in records if r["table"] == "questions")
    assert json.loads(question["result_json"])["audit"]["secret_trace"] == "must-stay-server-side"
    status, csv_body, headers = client.request("GET", "/api/study/admin/export?format=csv", headers=auth)
    assert status == 200 and "text/csv" in headers["Content-Type"]
    csv_records = list(csv.DictReader(io.StringIO(csv_body)))
    assert len(csv_records) == len(records)
    assert all(set(row) == {"table", "record_json"} for row in csv_records)
    filtered = client.request("GET", "/api/study/admin/export?mode=test&release_id=" + flow.view["release_id"], headers=auth)
    assert filtered[0] == 200
    assert filtered[1] == body
    assert not client.request("GET", "/api/study/admin/export?mode=preview", headers=auth)[1].strip()
    assert not client.request("GET", "/api/study/admin/export?release_id=missing-version", headers=auth)[1].strip()


@pytest.mark.parametrize("domain", ["warehouse", "pong", "kitchen"])
@pytest.mark.parametrize("group", ["A", "B"])
def test_all_domain_group_flows_through_http(http_service, domain, group):
    """No direct writes to the store: every stage/action/survey travels over HTTP."""
    server, client = http_service
    status, view, _ = client.create(domain, group)
    assert status == 200 and view["language"] == "en"
    iid = view["instance_id"]
    if domain=='kitchen':
        assert view['tutorial']['state']['turn']==0
        status,view,_=client.command('tutorial',view,command='start')
        assert status==200
        status,view,_=client.command('tutorial',view,command='action',action='wait')
        assert status==200 and view['state']['turn']==1 and not view['task_runs']
        status,view,_=client.command('demo_skip',view)
    else:
        for _ in range(len(view["demo"]["captions"])):
            status, view, _ = client.command("demo_next", view)
            assert status == 200
        status, view, _ = client.command("next", view)
    assert status == 200
    scores = []
    for task in (1, 2, 3):
        assert view["stage"] == f"task{task}"
        can_ask = group == "A" and task == 2
        assert view["can_ask"] is can_ask
        question = {"instance_id": iid, "authorized_run": view["run_id"],
            "target_run": view["run_id"], "turn": 0, "question_id": uuid.uuid4().hex,
            "question": "Why did you choose that?", "language": "en"}
        status, answer, _ = client.request("POST", "/api/study/ask", question)
        assert status == (200 if can_ask else 403)
        if can_ask:
            assert answer["status"] == "answered"
            assert_public(answer)
        while not view["state"]["terminal"]:
            # The test actor controls only the human. Controller state is read for
            # that actor's deterministic policy, never written or substituted.
            with server.store.db.transaction() as db:
                row = db.one("SELECT state_json FROM pl3_runs WHERE id=?", (view["run_id"],))
            state = json.loads(row["state_json"])
            human_action = engine(domain).human_advisor(state)
            status, view, _ = client.command("action", view, run_id=view["run_id"],
                turn=view["state"]["turn"], action=human_action)
            assert status == 200
        assert view["questions"] == [] and not view["can_ask"]
        assert client.request("POST", "/api/study/ask", question)[0] == 403
        scores.append(view["state"]["score"]["task_score"])
        status, view, _ = client.command("next", view)
        assert status == 200
    assert view["stage"] == "questionnaire"
    assert len(view["questionnaire"]["items"]) == (8 if group == "A" else 5)
    survey = view["questionnaire"]
    status, completed, _ = client.command("questionnaire", view,
        answers={q["id"]: "na" if q["allow_na"] else 4 for q in survey["items"]},
        comprehension={q["id"]: 0 for q in survey["comprehension"]},
        feedback="Automated HTTP test, not human research data.")
    assert status == 200 and completed["stage"] == "completed"
    assert [r["score"]["task_score"] for r in completed["task_runs"]] == scores
    assert client.request("GET", "/api/study/view?domain=" + domain)[1] == completed
    export = server.store.export()
    assert sum(r["instance_id"] == iid for r in export["questionnaires"]) == 1
    assert sum(r["instance_id"] == iid for r in export["runs"]) == 3
    assert all(r["mode"] == "test" for r in export["instances"])


def _http_process(database, admin, pipe):
    server = make_server(Settings(database=database, origin=ORIGIN, admin_token=admin),
        port=0, explainer=RecordingExplainer())
    pipe.send(server.server_port)
    pipe.close()
    server.serve_forever()


@contextmanager
def running_process(database):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_http_process, args=(database, ADMIN, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(15), "child HTTP process did not start"
        client = Client(receiver.recv())
        yield client, process.pid
    finally:
        receiver.close()
        process.terminate()
        process.join(10)
        if process.is_alive(): process.kill(); process.join(5)


def test_actual_application_process_restart_recovers_task_and_export(tmp_path):
    database = str(tmp_path / "restart.sqlite3")
    with running_process(database) as (first, first_pid):
        status, view, _ = first.create()
        assert status == 200
        for _ in range(len(view["demo"]["captions"])):
            status, view, _ = first.command("demo_next", view)
            assert status == 200
        status, view, _ = first.command("next", view)
        assert status == 200 and view["stage"] == "task1"
        status, view, _ = first.command("action", view, run_id=view["run_id"], turn=0, action="wait")
        assert status == 200 and view["state"]["turn"] == 1
        cookie = first.cookie
        old_export = first.request("GET", "/api/study/admin/export", headers={"Authorization": "Bearer " + ADMIN})[1]
    with running_process(database) as (restarted, second_pid):
        assert second_pid != first_pid
        restarted.cookie = cookie
        status, recovered, _ = restarted.request("GET", "/api/study/view?domain=pong")
        assert status == 200 and recovered == view
        new_export = restarted.request("GET", "/api/study/admin/export", headers={"Authorization": "Bearer " + ADMIN})[1]
        before = [json.loads(row) for row in old_export.splitlines()]
        after = [json.loads(row) for row in new_export.splitlines()]
        assert before == after
        status, advanced, _ = restarted.command("action", recovered, run_id=recovered["run_id"], turn=1, action="wait")
        assert status == 200 and advanced["state"]["turn"] == 2


def test_existing_render_start_command_dispatches_three_domain_service(tmp_path):
    import os
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    environment={**os.environ,'POLICYLENS_DATABASE_URL':str(tmp_path/'entry.sqlite3'),
        'POLICYLENS_MODE':'preview','POLICYLENS_ADMIN_TOKEN':ADMIN,'POLICYLENS_STUDY_VERIFIED':'0'}
    environment.pop('POLICYLENS_LEGACY_HUB',None)
    process=subprocess.Popen([sys.executable,'-m','ui.domain_hub_server','--host','127.0.0.1','--port',str(port)],
        cwd=Path(__file__).resolve().parents[1],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        client=Client(port)
        for attempt in range(50):
            try:
                status,health,_=client.request('GET','/health')
                break
            except OSError:
                if process.poll() is not None:pytest.fail('existing Render command exited early')
                time.sleep(.1)
        else:pytest.fail('existing Render command did not become ready')
        assert status==200 and health['service']=='policylens-three-domain'
        assert set(health['domains'])=={'warehouse','pong','kitchen'}
        assert health['study_ready'] is False
    finally:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)


def test_changed_source_cannot_reuse_a_pilot_release_id(http_service,monkeypatch):
    from dataclasses import replace
    import study_v3.server as service
    server,_=http_service
    actual=service.manifest(server.store.settings)
    monkeypatch.setattr(service,'manifest',lambda settings:{**actual,'source_sha256':'0'*64})
    with pytest.raises(RuntimeError,match='release_id_reused_with_changed_source'):
        make_server(replace(server.store.settings,mode='pilot'),port=0,explainer=RecordingExplainer())
