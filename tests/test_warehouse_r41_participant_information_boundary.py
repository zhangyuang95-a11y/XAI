"""Participant responses expose only observable r4.1 study information."""
from __future__ import annotations

from contextlib import closing
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
from uuid import uuid4

import pytest

from tests.test_warehouse_alignment_online_server import (
    FakeBank, FakeContext, FakeEnv, FakeExplainer, state,
)
from ui import warehouse_alignment_online_server as online


FORBIDDEN_KEYS = {
    "condition", "task_order", "seed", "scene_index", "scenario_id",
    "source_scenario", "source_frame", "actor_sha256", "program_sha256",
    "scenario_sha256", "runtime_signature", "observation_hashes",
    "probabilities", "logits", "decision", "policy_actions",
    "proposed_actions", "target_goal", "goal", "fingerprint", "signature",
    "sources", "model_version", "namespace", "policy_hidden",
    "score_breakdown", "episode_id", "ai_waits", "ai_blocked", "overrides",
}


def keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys(child)


def assert_public(value, secrets=()):
    found = set(keys(value)) & FORBIDDEN_KEYS
    assert not found, sorted(found)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    for secret in secrets:
        assert secret not in raw


class PreviewBank(FakeBank):
    def public_items(self):
        preview_state = state(frame=7)
        preview_state["tasks"][0]["task_id"] = "job_f7340265426dfe7a"
        preview_state["agents"][1]["carrying_task_id"] = "job_f7340265426dfe7a"
        return [{
            "id": f"prediction_{index}", "type": "choice",
            "prediction_kind": "next_action", "required": True,
            "prompt": {"zh": f"题目{index}", "en": f"Question {index}"},
            "options": [{"value": "WAIT", "label": {"zh": "等待", "en": "Wait"}}],
            "preview": {"state": preview_state,
                "map": online._public_map(FakeEnv()),
                "public_feedback": FakeEnv({"state": preview_state}).public_history()},
            "source_frame": 7,
            "source_scenario": "formal_seed_do_not_release_410001",
            "actor_sha256": "a" * 64,
        } for index in range(8)]


class ForbiddenMainAnswer(FakeExplainer):
    def answer(self, request, record, runtime):
        return {"answer": "NN probability was 80%.",
                "evidence_detail": "Actor SHA-256: " + "a" * 64}


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as directory:
        context = FakeContext(directory)
        context.question_bank = PreviewBank()
        context.scenarios["splits"]["play"] = [{
            "id": f"formal_seed_do_not_release_{410000 + index}",
            "fingerprint": f"private-fingerprint-{index}",
            "snapshot": {"state": state()},
        } for index in range(7)]
        context.release.update(actor_sha256="a" * 64,
                               program_sha256="b" * 64,
                               condition="A", task_order="XY",
                               seed=410001,
                               database_path="/tmp/private-participant-study.sqlite3",
                               internal={"probabilities": [0, 0, 0, 0, 1]})
        context.provenance["database_path"] = "/var/data/private-participant-study.sqlite3"
        value = online.OnlineAlignmentStudyStore(
            context, database=Path(directory) / "study.sqlite3",
            storage_mode="ephemeral")
        try:
            yield value
        finally:
            value.close()


def command(store, sid, view, kind, **fields):
    return store.command(sid, {"operation_id": uuid4().hex,
        "expected_version": view["version"], "kind": kind, **fields})


def enroll(store, participant="boundary_1", condition="A"):
    sid = store.session()
    view = store.view(sid)
    view = command(store, sid, view, "start", mode="study",
                   participant_id=participant, consent=True)
    with closing(store.connect()) as db:
        db.execute("UPDATE sessions SET condition=? WHERE id=?", (condition, sid))
    return sid, store.view(sid)


def begin(store, sid, view):
    return command(store, sid, view, "begin_task1")


def test_registration_tutorial_task_history_and_error_views_are_minimal(store):
    secrets = ("a" * 64, "b" * 64, store.tutorial_signature,
               "formal_seed_do_not_release", "private-fingerprint",
               store.runtime.signature, store.explainer.signature,
               "/tmp/private-participant-study.sqlite3",
               "/var/data/private-participant-study.sqlite3")
    sid = store.session()
    assert_public(store.view(sid), secrets)
    sid, tutorial = enroll(store)
    assert_public(tutorial, secrets)
    assert "signature" not in tutorial["tutorial"]
    assert set(tutorial["metrics"]) == {
        "deliveries", "score", "legacy_score", "steps", "collisions", "shutdowns"}
    task = begin(store, sid, tutorial)
    assert_public(task, secrets)
    assert all(row["task_id"].startswith("task_") for row in task["state"]["tasks"])
    moved = command(store, sid, task, "action", action="RIGHT")
    assert_public(moved, secrets)
    history = store.history(sid)
    assert_public(history, secrets)
    assert set(history["frames"][0]["metrics"]) == {
        "deliveries", "score", "legacy_score", "steps", "collisions", "shutdowns"}
    with pytest.raises(online.CommandError) as conflict:
        command(store, sid, task, "action", action="LEFT")
    assert conflict.value.reason == "state_version_conflict"
    assert_public(conflict.value.view, secrets)


def test_questionnaire_preview_drops_source_scene_frame_hashes_and_raw_task_ids(store):
    sid, view = enroll(store, "questionnaire_boundary")
    view = begin(store, sid, view)
    for _ in range(6):
        view = command(store, sid, view, "end")
        view = command(store, sid, view, "next")
    assert view["flow"]["stage"] == "questionnaire"
    assert_public(view, ("a" * 64, "formal_seed_do_not_release",
                         "job_f7340265426dfe7a"))
    predictions = [item for item in view["questionnaire"]["items"]
                   if item["id"].startswith("prediction_")]
    assert len(predictions) == 8
    assert all(set(item) == {"id", "type", "prediction_kind", "required",
                             "prompt", "options", "preview"}
               for item in predictions)
    assert all(item["preview"]["state"]["tasks"][0]["task_id"] == "task_1"
               for item in predictions)


def test_a_answer_has_one_technical_container_and_bad_main_answer_fails_closed(store):
    sid, view = enroll(store, "answer_boundary", "A")
    view = begin(store, sid, view)
    view = command(store, sid, view, "question", run_id=view["run_id"], frame=0,
                   question="机器人2下一步为什么等待？", language="zh", focus="next")
    for _ in range(100):
        view = store.view(sid)
        if view["answers"] and view["answers"][0]["status"] == "complete":
            break
        time.sleep(.01)
    answer = view["answers"][0]
    assert set(answer) == {"id", "status", "frame", "question", "text",
                           "run_id", "focus", "evidence_detail"}
    assert not online._MAIN_ANSWER_FORBIDDEN.search(answer["text"])
    assert answer["evidence_detail"]

    store.explainer = ForbiddenMainAnswer()
    view = command(store, sid, view, "question", run_id=view["run_id"], frame=0,
                   question="再解释一次", language="en", focus="next")
    for _ in range(100):
        view = store.view(sid)
        bad = next(row for row in view["answers"] if row["question"] == "再解释一次")
        if bad["status"] == "failed":
            break
        time.sleep(.01)
    assert bad["status"] == "failed"
    assert bad["text"] == ""
    assert bad["evidence_detail"] == ""


def test_participant_evidence_detail_removes_artifact_identity_but_keeps_evidence():
    actor_hash = "a" * 64
    stored = {"id":"answer-1", "status":"complete", "frame":12,
              "question":"为什么？", "answer":"机器人2向右，距离缩短了一格。",
              "run_id":"run-1", "focus":"executed",
              "evidence_detail":("绑定帧：12（已执行动作）\n"
                                 "动作概率：右 80%\n"
                                 f"Actor SHA-256：{actor_hash}")}
    projected = online._participant_answer(stored)
    assert projected["status"] == "complete"
    assert projected["evidence_detail"] == "绑定帧：12（已执行动作）\n动作概率：右 80%"
    assert actor_hash not in projected["evidence_detail"]
    assert actor_hash in stored["evidence_detail"]


def test_b_task1_task2_old_cache_and_cross_session_are_closed(store):
    sid_b, view_b = enroll(store, "group_b_boundary", "B")
    view_b = begin(store, sid_b, view_b)
    with closing(store.connect()) as db:
        db.execute("""INSERT INTO questions
            (id,session_id,run_id,frame,stage,question,language,status,answer,shown,created,focus,evidence_detail)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("cached-b", sid_b, view_b["run_id"], 0, "task1", "old", "zh",
             "complete", "internal cached answer", None,
             "2026-09-11T00:00:00+00:00", "next", "Actor SHA-256: " + "a" * 64))
    hidden_b = store.view(sid_b)
    assert hidden_b["answers"] == []
    assert not hidden_b["explain_allowed"]
    assert_public(hidden_b, ("internal cached answer", "a" * 64))

    sid_a, view_a = enroll(store, "group_a_task2_boundary", "A")
    view_a = begin(store, sid_a, view_a)
    with closing(store.connect()) as db:
        db.execute("""INSERT INTO questions
            (id,session_id,run_id,frame,stage,question,language,status,answer,shown,created,focus,evidence_detail)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ("cached-a", sid_a, view_a["run_id"], 0, "task1", "old", "zh",
             "complete", "task1 cached answer", None,
             "2026-09-11T00:00:00+00:00", "next", "technical detail"))
    for _ in range(3):
        view_a = command(store, sid_a, view_a, "end")
        view_a = command(store, sid_a, view_a, "next")
    assert view_a["flow"]["stage"] == "task2"
    assert view_a["answers"] == []
    assert not view_a["explain_allowed"]
    assert_public(view_a, ("task1 cached answer", "technical detail"))

    with pytest.raises(online.CommandError) as cross:
        store.history(sid_b, view_a["run_id"])
    assert cross.value.reason == "history_current_run_only"
    assert cross.value.view is None


def test_served_html_and_javascript_contain_no_release_identity(store):
    assets = (store.web_assets["index.html"] + store.web_assets["app.js"])
    for secret in ("a" * 64, "b" * 64, store.tutorial_signature,
                   b"formal_seed_do_not_release", b"private-fingerprint"):
        encoded = secret if isinstance(secret, bytes) else secret.encode()
        assert encoded not in assets


def test_http_error_does_not_echo_request_or_internal_identity(store):
    server = online.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        online.handler_class(store, public_origin="https://study.test"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_port, timeout=3)
        connection.request("GET", "/api/view")
        response = connection.getresponse()
        response.read()
        cookie = response.getheader("Set-Cookie").split(";", 1)[0]
        secret = "a" * 64
        body = json.dumps({"operation_id": "stale", "expected_version": 99,
                           "kind": "question", "debug": secret})
        connection.request("POST", "/api/study/command", body, {
            "Content-Type": "application/json", "Origin": "https://study.test",
            "Cookie": cookie})
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 409
        assert payload["error"] == "state_version_conflict"
        assert secret not in json.dumps(payload)
        assert_public(payload["view"], (secret, "formal_seed_do_not_release"))
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
