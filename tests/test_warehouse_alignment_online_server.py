from __future__ import annotations

import http.client
import json
import sqlite3
from contextlib import closing
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from uuid import uuid4


FORBIDDEN = {
    "ui.warehouse_alignment_server",
    "ui.warehouse_family_server",
    "ui.warehouse_public_history_server",
    "ui.warehouse_native_server",
    "torch",
}
before_import = set(sys.modules)
from ui import warehouse_alignment_online_server as online
IMPORTED_BY_SERVER = set(sys.modules) - before_import


def agent(agent_id, position):
    return {"agent_id":agent_id, "position":list(position), "battery":100.0,
        "carrying_task_id":None, "deliveries_completed":0, "active":True,
        "heading":"UP", "last_action":"WAIT", "last_executed_action":"WAIT"}


def state(frame=0):
    return {"episode_id":1, "frame":frame, "total_deliveries":0,
        "collision_count":0, "shutdown_count":0, "terminated":False,
        "truncated":False, "terminal_reason":None, "user_score":-float(frame),
        "score_breakdown":{"time":-float(frame)}, "robot_collision_events":0,
        "invalid_move_count":0, "agents":[agent("robot_1",(5,2)),agent("robot_2",(5,4))],
        "tasks":[{"task_id":"task_1","pickup_position":[1,2],"delivery_position":[4,2],
            "status":"available","carrier_agent_id":None,"created_frame":0,"claimed_frame":None}]}


class FakeEnv:
    def __init__(self, snapshot=None):
        self.state = json.loads(json.dumps((snapshot or {"state":state()})["state"]))
        self.config = SimpleNamespace(horizon=120)

    def snapshot(self):
        return {"state":json.loads(json.dumps(self.state)), "public_history":{"valid":self.state["frame"]>0}}

    def public_history(self):
        return {"valid":self.state["frame"]>0, "submitted_actions":{"robot_1":"WAIT","robot_2":"WAIT"},
            "executed_actions":{"robot_1":"WAIT","robot_2":"WAIT"}, "move_canceled":{},
            "consecutive_move_canceled":{}, "collision_kind":"none", "consecutive_collision":0}


class FakeRuntime:
    signature = "runtime-v1"
    actor_sha256 = "a" * 64
    protocol_sha256 = "b" * 64
    config = SimpleNamespace(horizon=120, map_layout_id="warehouse-6x7")

    def verify_binding(self):
        return self.signature

    def environment(self, _scene):
        return FakeEnv()

    def from_snapshot(self, snapshot):
        return FakeEnv(snapshot)

    def step(self, env, human_action):
        before = env.snapshot()
        env.state["frame"] += 1
        for row in env.state["agents"]:
            row["last_action"] = human_action if row["agent_id"] == "robot_1" else "WAIT"
            row["last_executed_action"] = row["last_action"]
        env.state["user_score"] -= 1
        actions = {"robot_1":human_action,"robot_2":"WAIT"}
        return {"before":before,"after":env.snapshot(),"decision":{"actor_sha256":self.actor_sha256,
            "frame":before["state"]["frame"],"policy_actions":{"robot_2":"WAIT"},
            "probabilities":{"robot_2":[0,0,0,0,1]}},"submitted_actions":actions,
            "executed_actions":actions,"policy_actions":{"robot_1":"WAIT","robot_2":"WAIT"},
            "runtime_signature":self.signature,"done":False}


class FakeExplainer:
    signature = "explainer-v1"

    def _assert_current(self, runtime):
        assert runtime.signature == "runtime-v1"

    def answer(self, request, record, runtime):
        assert request["frame"] == record["after"]["state"]["frame"]
        assert runtime.signature == record["runtime_signature"]
        return {
            "answer": ("这是绑定所选帧的核验回答。" if request["language"] == "zh"
                       else "Verified answer for the selected frame."),
            "evidence_detail": ("绑定帧与策略证据已核验。" if request["language"] == "zh"
                                else "Frame binding and policy evidence verified."),
        }


class FakeBank:
    signature = "bank-v1"

    def public_items(self):
        return [{"id":f"prediction_{i}","type":"choice","required":True,
            "prompt":{"zh":f"题目{i}","en":f"Question {i}"},
            "options":[{"value":"WAIT","label":{"zh":"等待","en":"Wait"}}]}
            for i in range(8)]

    def grade(self, answers):
        return {"correct":sum(answers.get(f"prediction_{i}")=="WAIT" for i in range(8)),"total":8}


class FakeContext:
    def __init__(self, root):
        self.runtime, self.explainer, self.question_bank = FakeRuntime(), FakeExplainer(), FakeBank()
        self.scenarios = {"splits":{"play":[{"id":f"scene-{i}","snapshot":{"state":state()}} for i in range(7)]}}
        self.release = {"status":"local_pilot_technically_verified","model_ready":True,
            "explanation_ready":True,"study_ready":True,"formal_ready":False,"test_fixture":False,
            "model_version":"fake-r2","online_portable":True}
        self.provenance = {"version":"fake"}
        self.source_binding = {"fake":"c"*64}
        self.signature = "context-v1"
        self.root = Path(root) / "release"
        self.root.mkdir(exist_ok=True)
        self.closed = False

    def close(self):
        self.closed = True


class StoreCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "study.sqlite3"
        self.context = FakeContext(self.tmp.name)
        self.store = online.OnlineAlignmentStudyStore(self.context, database=self.db, storage_mode="ephemeral")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def cmd(self, sid, view, kind, **values):
        return self.store.command(sid, {"operation_id":uuid4().hex,
            "expected_version":view["version"],"kind":kind,**values})

    def enroll(self, participant):
        sid = self.store.session()
        view = self.store.view(sid)
        self.assertNotEqual(view["session_id"], sid)
        view = self.cmd(sid, view, "start", mode="study", participant_id=participant,
                        consent=True)
        self.assertEqual(view["flow"]["stage"], "practice")
        return sid, view

    def end_next(self, sid, view):
        view = self.cmd(sid, view, "end")
        return self.cmd(sid, view, "next")

    def test_import_is_dependency_light(self):
        self.assertFalse(FORBIDDEN & IMPORTED_BY_SERVER)

    def test_id_consent_casefold_and_block_balance(self):
        allocations = set()
        for i in range(4):
            sid = self.store.session()
            view = self.store.view(sid)
            if i == 0:
                with self.assertRaises(online.CommandError) as error:
                    self.cmd(sid, view, "start", mode="study", participant_id=f"User_{i}")
                self.assertEqual(error.exception.reason, "explicit_consent_required")
            registered = self.cmd(sid, view, "start", mode="study",
                                  participant_id=f"User_{i}", consent=True)
            self.assertEqual(registered["flow"]["stage"], "practice")
            with closing(self.store.connect()) as db:
                row = db.execute("SELECT condition,task_order,consented FROM sessions WHERE id=?", (sid,)).fetchone()
                allocations.add((row["condition"],row["task_order"]))
                self.assertIsNotNone(row["consented"])
        self.assertEqual(allocations, {("A","XY"),("A","YX"),("B","XY"),("B","YX")})
        sid = self.store.session(); view = self.store.view(sid)
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid, view, "start", mode="study", participant_id="user_0", consent=True)
        self.assertEqual(error.exception.reason, "participant_id_taken")

    def test_ab_permissions_and_task2_hides_old_answers(self):
        enrolled = [self.enroll(f"p_{i}") for i in range(4)]
        selected = {}
        with closing(self.store.connect()) as db:
            for sid, view in enrolled:
                selected[db.execute("SELECT condition FROM sessions WHERE id=?",(sid,)).fetchone()[0]] = (sid,view)
        sid_a, a = selected["A"]
        sid_b, b = selected["B"]
        a = self.end_next(sid_a, a); b = self.end_next(sid_b, b)
        self.assertEqual(a["flow"]["stage"], "task1")
        a = self.cmd(sid_a, a, "question", run_id=a["run_id"], frame=0,
            question="下一步为什么等待？", language="zh", focus="next")
        for _ in range(50):
            a = self.store.view(sid_a)
            if a["answers"] and a["answers"][0]["status"] not in ("pending","running"):
                break
            time.sleep(.01)
        self.assertEqual(a["answers"][0]["status"], "complete")
        self.assertEqual(a["answers"][0]["evidence_detail"], "绑定帧与策略证据已核验。")
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid_b, b, "question", run_id=b["run_id"], frame=0,
                question="why", language="en", focus="next")
        self.assertEqual(error.exception.reason, "explanation_forbidden")
        for _ in range(3):
            a = self.end_next(sid_a, a)
        self.assertEqual(a["flow"]["stage"], "task2")
        self.assertFalse(a["explain_allowed"])
        self.assertEqual(a["answers"], [])
        with closing(self.store.connect()) as db:
            retained = db.execute(
                "SELECT status,answer,evidence_detail FROM questions WHERE session_id=?",
                (sid_a,),
            ).fetchone()
        self.assertEqual(retained["status"], "complete")
        self.assertEqual(retained["answer"], "这是绑定所选帧的核验回答。")
        self.assertEqual(retained["evidence_detail"], "绑定帧与策略证据已核验。")
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid_a, a, "question", run_id=a["run_id"], frame=0,
                question="why", language="en", focus="next")
        self.assertEqual(error.exception.reason, "explanation_forbidden")

    def test_six_rounds_questionnaire_and_completion(self):
        sid, view = self.enroll("study_1")
        view = self.end_next(sid, view)
        for _ in range(3):
            view = self.end_next(sid, view)
        self.assertEqual(view["flow"]["stage"], "task2")
        for _ in range(3):
            view = self.end_next(sid, view)
        self.assertEqual(view["flow"]["stage"], "questionnaire")
        items = view["questionnaire"]["items"]
        self.assertEqual(len(items), 11)
        answers = {x["id"]:(1 if x["type"]=="scale" else "WAIT") for x in items}
        view = self.cmd(sid, view, "questionnaire", answers=answers, submit=True)
        self.assertEqual(view["flow"]["stage"], "completed")
        self.assertEqual(view["answers"], [])

    def test_idempotency_conflict_history_scope_and_restart(self):
        sid, view = self.enroll("restore_1")
        body = {"operation_id":"same-op","expected_version":view["version"],"kind":"action","action":"RIGHT"}
        first = self.store.command(sid, body)
        replay = self.store.command(sid, body)
        self.assertEqual(first["version"], replay["version"])
        self.assertEqual(first["state"]["frame"], 1)
        with closing(self.store.connect()) as db:
            internal = json.loads(db.execute(
                "SELECT internal FROM frames WHERE run_id=? AND frame=1",
                (first["run_id"],),
            ).fetchone()[0])
        self.assertNotIn("run_signature", internal)
        changed = dict(body, action="LEFT")
        with self.assertRaises(online.CommandError) as error:
            self.store.command(sid, changed)
        self.assertEqual(error.exception.reason, "operation_id_reused")
        history = self.store.history(sid)
        self.assertEqual(len(history["frames"]), 2)
        other = self.store.session()
        with self.assertRaises(online.CommandError):
            self.store.history(other, first["run_id"])
        self.store.close()
        new_context = FakeContext(self.tmp.name)
        self.store = online.OnlineAlignmentStudyStore(new_context, database=self.db, storage_mode="ephemeral")
        restored = self.store.view(sid)
        self.assertEqual(restored["version"], first["version"])
        self.assertEqual(restored["state"]["frame"], 1)

    def test_ephemeral_label_and_secure_https_http_edge(self):
        view = self.store.view(self.store.session())
        self.assertFalse(view["release"]["data_persistent"])
        self.assertFalse(view["release"]["formal_ready"])
        self.assertIn("丢失", view["release"]["message"]["zh"])
        server = online.ThreadingHTTPServer(("127.0.0.1",0),
            online.handler_class(self.store, public_origin="https://study.test"))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=3)
            conn.request("GET","/health"); response=conn.getresponse()
            self.assertEqual(response.status,200); self.assertFalse(json.loads(response.read())["data_persistent"])
            body = json.dumps({"kind":"start"})
            conn.request("POST","/api/study/command",body,{"Content-Type":"application/json","Origin":"http://study.test"})
            response=conn.getresponse(); self.assertEqual(response.status,403); response.read()
            conn.request("GET","/api/view"); response=conn.getresponse(); response.read()
            cookie=response.getheader("Set-Cookie")
            self.assertIn("HttpOnly",cookie); self.assertIn("Secure",cookie); self.assertIn("SameSite=Strict",cookie)
            enrollment = json.dumps({"operation_id":"http-enroll","expected_version":0,
                "kind":"start","mode":"study","participant_id":"http_user_1","consent":True})
            conn.request("POST","/api/study/command",enrollment,{"Content-Type":"application/json",
                "Origin":"https://study.test","Cookie":cookie.split(";",1)[0]})
            response=conn.getresponse(); enrolled=json.loads(response.read())
            self.assertEqual(response.status,200)
            self.assertEqual(enrolled["flow"]["stage"],"practice")
            conn.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_online_assets_show_consent_before_registration(self):
        javascript = self.store.web_assets["app.js"].decode("utf-8")
        self.assertIn("registrationStage", javascript)
        self.assertIn('participant_id:id,consent:true', javascript)
        self.assertIn("线上试玩 · 已核验", javascript)
        self.assertIn("Online demo · Verified", javascript)
        self.assertIn("evidence_detail", javascript)
        self.assertIn("whyCollision", javascript)
        self.assertIn('request("/api/study/command",{method:"POST"', javascript)
        self.assertNotIn('request("/api/command",{method:"POST"', javascript)
        html = self.store.web_assets["index.html"].decode("utf-8")
        self.assertEqual(html.count("data-question="), 6)
        css = self.store.web_assets["styles.css"].decode("utf-8")
        self.assertIn("grid-template-columns: minmax(600px, 1.45fr)", css)

    def test_question_evidence_column_is_added_to_existing_database(self):
        enrolled = [self.enroll(f"legacy_{index}") for index in range(4)]
        with closing(self.store.connect()) as db:
            sid, view = next((sid, view) for sid, view in enrolled
                if db.execute("SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0] == "A")
        view = self.end_next(sid, view)
        run_id = view["run_id"]
        self.store.close()
        with sqlite3.connect(self.db) as db:
            db.execute("DROP TABLE questions")
            db.execute("""CREATE TABLE questions(
                id TEXT PRIMARY KEY,session_id TEXT NOT NULL,run_id TEXT NOT NULL,frame INTEGER NOT NULL,
                stage TEXT NOT NULL,question TEXT NOT NULL,language TEXT NOT NULL,status TEXT NOT NULL,
                answer TEXT,shown TEXT,created TEXT NOT NULL,focus TEXT NOT NULL DEFAULT 'executed')""")
            db.execute("""INSERT INTO questions
                (id,session_id,run_id,frame,stage,question,language,status,answer,shown,created,focus)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("legacy-answer", sid, run_id, 0, "task1", "旧问题", "zh", "complete",
                 "旧回答仍可读取。", None, "2026-09-10T00:00:00+00:00", "next"))
        self.store = online.OnlineAlignmentStudyStore(
            FakeContext(self.tmp.name), database=self.db, storage_mode="ephemeral")
        with closing(self.store.connect()) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(questions)")}
        self.assertIn("evidence_detail", columns)
        restored = self.store.view(sid)
        self.assertEqual(restored["answers"][0]["text"], "旧回答仍可读取。")
        self.assertEqual(restored["answers"][0]["evidence_detail"], "")

    def test_version_change_rotates_browser_session_without_deleting_old_records(self):
        sid, view = self.enroll("old_version_1")
        old_run = view["run_id"]
        with closing(self.store.connect()) as db:
            db.execute("UPDATE sessions SET study_signature='older-runtime' WHERE id=?", (sid,))
        fresh_sid = self.store.session(sid)
        self.assertNotEqual(fresh_sid, sid)
        fresh = self.store.view(fresh_sid)
        self.assertEqual(fresh["flow"]["stage"], "registration")
        with closing(self.store.connect()) as db:
            self.assertIsNotNone(db.execute("SELECT 1 FROM sessions WHERE id=?", (sid,)).fetchone())
            self.assertIsNotNone(db.execute("SELECT 1 FROM runs WHERE id=? AND session_id=?", (old_run, sid)).fetchone())


if __name__ == "__main__":
    unittest.main()
