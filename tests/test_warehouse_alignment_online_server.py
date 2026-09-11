from __future__ import annotations

import http.client
from hashlib import sha256
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
        self._last_actions = json.loads(json.dumps((snapshot or {}).get("last_actions", {
            "robot_1":"WAIT", "robot_2":"WAIT"})))

    def snapshot(self):
        return {"state":json.loads(json.dumps(self.state)), "last_actions":dict(self._last_actions),
            "public_history":{"valid":self.state["frame"]>0}}

    def public_history(self):
        return {"valid":self.state["frame"]>0, "submitted_actions":dict(self._last_actions),
            "executed_actions":dict(self._last_actions), "move_canceled":{},
            "consecutive_move_canceled":{}, "collision_kind":"none", "consecutive_collision":0}

    def step(self, actions):
        delta = {"UP":(-1,0),"DOWN":(1,0),"LEFT":(0,-1),"RIGHT":(0,1),"WAIT":(0,0)}
        self.state["frame"] += 1
        self._last_actions = dict(actions)
        for row in self.state["agents"]:
            action = actions[row["agent_id"]]
            dr, dc = delta[action]
            row["position"] = [row["position"][0]+dr,row["position"][1]+dc]
            row["last_action"] = action
            row["last_executed_action"] = action
        self.state["user_score"] -= 1
        truncated = self.state["frame"] >= 3
        self.state["truncated"] = truncated
        self.state["terminal_reason"] = "horizon" if truncated else None
        info = {"frame":self.state["frame"], "requested_actions":dict(actions),
            "executed_actions":dict(actions), "events":[]}
        return {}, {"robot_1":-1.0,"robot_2":-1.0}, False, truncated, info


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


class BlockingExplainer(FakeExplainer):
    """Keep one answer in flight while the participant continues the task."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def answer(self, request, record, runtime):
        self.started.set()
        if not self.release.wait(5):
            raise TimeoutError("blocking explainer was not released")
        return super().answer(request, record, runtime)


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
        env, frames = FakeEnv(), []
        neutral_actions = [
            {"robot_1":"UP", "robot_2":"LEFT"},
            {"robot_1":"WAIT", "robot_2":"WAIT"},
            {"robot_1":"RIGHT", "robot_2":"DOWN"},
        ]
        def append_frame(actions):
            public = online._public_state(env)
            public["public_feedback"] = env.public_history()
            frames.append({"state":public,"metrics":{"steps":env.state["frame"],
                "deliveries":0,"score":-float(env.state["frame"]),"collisions":0,"shutdowns":0},
                "actions":dict(actions)})
        append_frame({})
        for actions in neutral_actions:
            env.step(actions); append_frame(actions)
        self.tutorial = {"version":"warehouse-alignment-neutral-tutorial.v1",
            "source":"independent_neutral_ai_ai","uses_final_actor":False,
            "scene_id":"neutral-fixture-1","duration_ms":380,
            "map_sha256":online._digest(online._public_map(FakeEnv())),
            "bindings":{"scene_manifest_version":"warehouse-r41-conflict-scene-manifest.v1",
                "scene_manifest_content_sha256":"1"*64,
                "tutorial_scene_fingerprint":"2"*64,
                "tutorial_successor_state_sha256":"3"*64,
                "tutorial_snapshot_sha256":"4"*64,
                "conflict_contract_sha256":"5"*64,
                "conflict_graph_sha256":"6"*64,
                "producer_sources_sha256":"7"*64},
            "coverage":{"pickup_frames":[1],"delivery_frames":[2],
                "simultaneous_movement_frames":[1],"collision_frames":[1],
                "wait_frames":[2],"charge_frames":[3]},
            "frames":frames}
        self.tutorial_signature = online._digest(self.tutorial)
        self.provenance = {"version":"fake","tutorial_signature":self.tutorial_signature}
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
        self.assertEqual(view["flow"]["stage"], "instructions")
        self.assertIsNone(view["run_id"])
        return sid, view

    def begin_task1(self, sid, view):
        view = self.cmd(sid, view, "begin_task1")
        self.assertEqual(view["flow"]["stage"], "task1")
        self.assertIsNotNone(view["run_id"])
        return view

    def end_next(self, sid, view):
        view = self.cmd(sid, view, "end")
        return self.cmd(sid, view, "next")

    def test_import_is_dependency_light(self):
        self.assertFalse(FORBIDDEN & IMPORTED_BY_SERVER)

    def test_persistent_label_cannot_be_applied_to_ephemeral_sqlite_path(self):
        context = FakeContext(self.tmp.name)
        with self.assertRaisesRegex(
                ValueError, "persistent SQLite storage must be under /var/data"):
            online.OnlineAlignmentStudyStore(
                context,
                database=Path(self.tmp.name) / "operator-labelled-persistent.sqlite3",
                storage_mode="persistent",
            )
        context.close()

        # Local development and test remain SQLite-backed and are explicitly
        # advertised as ephemeral.
        self.assertFalse(self.store.data_persistent)
        self.assertEqual(self.store.view(self.store.session())["release"]["status"],
                         "online_demo_ephemeral")

    def test_tmp_auto_mode_stays_ephemeral_and_persistent_copy_is_explicit(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            context = FakeContext(directory)
            store = online.OnlineAlignmentStudyStore(
                context, database=Path(directory) / "tmp-study.sqlite3",
                storage_mode="auto")
            try:
                view = store.view(store.session())
                self.assertFalse(store.data_persistent)
                self.assertIn("没有持久磁盘", view["flow"]["consent_text"]["zh"])
                self.assertIn("no persistent disk", view["flow"]["consent_text"]["en"])
            finally:
                store.close()

        # Use a temporary stand-in for the Render mount so the test verifies
        # the persistent branch without writing to the host's /var/data.
        with tempfile.TemporaryDirectory() as directory:
            old_root = online.PERSISTENT_DATABASE_ROOT
            online.PERSISTENT_DATABASE_ROOT = Path(directory) / "render-disk"
            try:
                auto_context = FakeContext(directory)
                auto_store = online.OnlineAlignmentStudyStore(
                    auto_context,
                    database=online.PERSISTENT_DATABASE_ROOT / "auto.sqlite3",
                    storage_mode="auto")
                try:
                    self.assertFalse(auto_store.data_persistent)
                finally:
                    auto_store.close()
                context = FakeContext(directory)
                store = online.OnlineAlignmentStudyStore(
                    context,
                    database=online.PERSISTENT_DATABASE_ROOT / "study.sqlite3",
                    storage_mode="persistent")
                try:
                    view = store.view(store.session())
                    self.assertTrue(store.data_persistent)
                    self.assertIn("已配置的持久磁盘", view["flow"]["consent_text"]["zh"])
                    self.assertNotIn("免费", view["flow"]["consent_text"]["zh"])
                    self.assertIn("configured persistent disk",
                                  view["flow"]["consent_text"]["en"])
                    self.assertNotIn("may erase", view["flow"]["consent_text"]["en"])
                finally:
                    store.close()
            finally:
                online.PERSISTENT_DATABASE_ROOT = old_root

    def test_tutorial_must_be_neutral_public_and_release_hash_bound(self):
        for mutation in ("missing", "final_actor", "internal_evidence", "wrong_hash"):
            context = FakeContext(self.tmp.name)
            if mutation == "missing":
                del context.tutorial
            elif mutation == "final_actor":
                context.tutorial["uses_final_actor"] = True
                context.tutorial_signature = online._digest(context.tutorial)
                context.provenance["tutorial_signature"] = context.tutorial_signature
            elif mutation == "internal_evidence":
                context.tutorial["frames"][1]["probabilities"] = {"robot_2":[1,0,0,0,0]}
                context.tutorial_signature = online._digest(context.tutorial)
                context.provenance["tutorial_signature"] = context.tutorial_signature
            else:
                context.tutorial_signature = "f" * 64
                context.provenance["tutorial_signature"] = context.tutorial_signature
            with self.assertRaises(ValueError, msg=mutation):
                online.OnlineAlignmentStudyStore(context,
                    database=Path(self.tmp.name) / f"invalid-{mutation}.sqlite3",
                    storage_mode="ephemeral")
            context.close()

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
            self.assertEqual(registered["flow"]["stage"], "instructions")
            with closing(self.store.connect()) as db:
                row = db.execute("SELECT condition,task_order,consented FROM sessions WHERE id=?", (sid,)).fetchone()
                allocations.add((row["condition"],row["task_order"]))
                self.assertIsNotNone(row["consented"])
        self.assertEqual(allocations, {("A","XY"),("A","YX"),("B","XY"),("B","YX")})
        sid = self.store.session(); view = self.store.view(sid)
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid, view, "start", mode="study", participant_id="user_0", consent=True)
        self.assertEqual(error.exception.reason, "participant_id_taken")

    def test_ai_ai_tutorial_is_unscored_idempotent_and_restored(self):
        sid, view = self.enroll("tutorial_1")
        self.assertEqual(view["allowed_kinds"], ["tutorial_advance", "tutorial_restart",
            "tutorial_select", "begin_task1"])
        self.assertEqual(view["tutorial"]["frame_index"], 0)
        self.assertEqual(view["tutorial"]["total_frames"], 4)
        self.assertFalse(view["tutorial"]["scored"])
        self.assertTrue(view["tutorial"]["independent_from_formal_rounds"])
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM runs WHERE session_id=?", (sid,)).fetchone()[0], 0)

        body = {"operation_id":"tutorial-step", "expected_version":view["version"],
            "kind":"tutorial_advance"}
        advanced = self.store.command(sid, body)
        replayed = self.store.command(sid, body)
        self.assertEqual(replayed["version"], advanced["version"])
        self.assertEqual(advanced["tutorial"]["frame_index"], 1)
        self.assertEqual(advanced["tutorial"]["max_played_index"], 1)
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid, advanced, "tutorial_select", frame_index=2)
        self.assertEqual(error.exception.reason, "tutorial_frame_not_available")
        selected = self.cmd(sid, advanced, "tutorial_select", frame_index=0)

        self.store.close()
        self.store = online.OnlineAlignmentStudyStore(
            FakeContext(self.tmp.name), database=self.db, storage_mode="ephemeral")
        restored = self.store.view(sid)
        self.assertEqual(restored["tutorial"]["frame_index"], 0)
        self.assertEqual(restored["tutorial"]["max_played_index"], 1)
        self.assertEqual(restored["version"], selected["version"])
        task = self.begin_task1(sid, restored)
        self.assertEqual(task["flow"]["round_count"], 3)
        with closing(self.store.connect()) as db:
            runs = db.execute("SELECT scenario_id,stage FROM runs WHERE session_id=?", (sid,)).fetchall()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["stage"], "task1")
            self.assertNotEqual(runs[0]["scenario_id"], "scene-0")
            audit = json.loads(db.execute("SELECT payload FROM events WHERE session_id=? AND kind='tutorial_acknowledged'",
                (sid,)).fetchone()[0])
        self.assertTrue(audit["ended_early"])
        self.assertFalse(audit["scored"])
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid, task, "tutorial_advance")
        self.assertEqual(error.exception.reason, "tutorial_not_allowed_in_stage")

    def test_tutorial_can_play_to_completion_restart_and_begin_task1(self):
        sid, view = self.enroll("tutorial_complete_1")
        for expected in range(1, view["tutorial"]["total_frames"]):
            view = self.cmd(sid, view, "tutorial_advance")
            self.assertEqual(view["tutorial"]["frame_index"], expected)
        self.assertTrue(view["tutorial"]["complete"])
        maximum = view["tutorial"]["max_played_index"]
        view = self.cmd(sid, view, "tutorial_restart")
        self.assertEqual(view["tutorial"]["frame_index"], 0)
        self.assertEqual(view["tutorial"]["max_played_index"], maximum)
        view = self.cmd(sid, view, "tutorial_select", frame_index=maximum)
        self.assertEqual(view["tutorial"]["frame_index"], maximum)
        task = self.begin_task1(sid, view)
        with closing(self.store.connect()) as db:
            audit = json.loads(db.execute("SELECT payload FROM events WHERE session_id=? AND kind='tutorial_acknowledged'",
                (sid,)).fetchone()[0])
        self.assertFalse(audit["ended_early"])
        self.assertTrue(audit["tutorial_completed"])
        self.assertEqual(task["state"]["frame"], 0)
        self.assertEqual(task["metrics"]["score"], 0)

    def test_ab_permissions_and_task2_hides_old_answers(self):
        enrolled = [self.enroll(f"p_{i}") for i in range(4)]
        selected = {}
        with closing(self.store.connect()) as db:
            for sid, view in enrolled:
                selected[db.execute("SELECT condition FROM sessions WHERE id=?",(sid,)).fetchone()[0]] = (sid,view)
        sid_a, a = selected["A"]
        sid_b, b = selected["B"]
        a = self.begin_task1(sid_a, a); b = self.begin_task1(sid_b, b)
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

    def test_group_a_can_ask_about_any_confirmed_frame_during_ended_review(self):
        enrolled = [self.enroll(f"review_{i}") for i in range(4)]
        with closing(self.store.connect()) as db:
            sid, view = next((sid, view) for sid, view in enrolled
                if db.execute("SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0] == "A")
        view = self.begin_task1(sid, view)
        view = self.cmd(sid, view, "action", action="RIGHT")
        view = self.cmd(sid, view, "end")
        self.assertTrue(view["ended"])
        self.assertIn("question", view["allowed_kinds"])
        self.assertIn("next", view["allowed_kinds"])
        for frame in (0, 1):
            view = self.cmd(sid, view, "question", run_id=view["run_id"], frame=frame,
                question=f"请解释第 {frame} 帧", language="zh", focus="executed" if frame else "next")
        for _ in range(50):
            view = self.store.view(sid)
            if len(view["answers"]) == 2 and all(a["status"] == "complete" for a in view["answers"]):
                break
            time.sleep(.01)
        self.assertEqual({a["frame"] for a in view["answers"]}, {0, 1})
        self.assertTrue(all(a["evidence_detail"] for a in view["answers"]))

    def test_in_flight_task1_answer_stays_bound_to_original_frame_while_play_continues(self):
        blocker = BlockingExplainer()
        self.store.explainer = blocker
        enrolled = [self.enroll(f"bound_async_{i}") for i in range(4)]
        with closing(self.store.connect()) as db:
            sid, view = next((sid, view) for sid, view in enrolled
                if db.execute("SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0] == "A")
        view = self.begin_task1(sid, view)
        question_run = view["run_id"]

        try:
            view = self.cmd(sid, view, "question", run_id=question_run, frame=0,
                question="机器人2当前在朝哪个任务前进？", language="zh", focus="next")
            self.assertTrue(blocker.started.wait(2))

            # Gameplay can advance while the answer is running. Releasing the
            # answer afterwards must not rebind it to the newer live frame.
            view = self.cmd(sid, view, "action", action="RIGHT")
            self.assertEqual(view["run_id"], question_run)
            self.assertEqual(view["state"]["frame"], 1)
            blocker.release.set()

            for _ in range(100):
                view = self.store.view(sid)
                if view["answers"] and view["answers"][0]["status"] == "complete":
                    break
                time.sleep(.01)

            self.assertEqual(len(view["answers"]), 1)
            answer = view["answers"][0]
            self.assertEqual(answer["status"], "complete")
            self.assertEqual(answer["run_id"], question_run)
            self.assertEqual(answer["frame"], 0)
            self.assertEqual(view["run_id"], question_run)
            self.assertEqual(view["state"]["frame"], 1)

            history = self.store.history(sid, question_run)
            self.assertEqual(len(history["frames"]), 2)
            self.assertEqual(history["frames"][1]["state"]["frame"], 1)
            self.assertEqual(history["frames"][1]["actions"]["robot_1"], "RIGHT")
        finally:
            blocker.release.set()

    def test_in_flight_task1_answer_does_not_block_play_or_cross_into_task2(self):
        blocker = BlockingExplainer()
        self.store.explainer = blocker
        enrolled = [self.enroll(f"async_{i}") for i in range(4)]
        with closing(self.store.connect()) as db:
            sid, view = next((sid, view) for sid, view in enrolled
                if db.execute("SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0] == "A")
        view = self.begin_task1(sid, view)
        question_run = view["run_id"]
        view = self.cmd(sid, view, "question", run_id=question_run, frame=0,
            question="机器人2当前在朝哪个任务前进？", language="zh", focus="next")
        self.assertTrue(blocker.started.wait(2))

        # Explanation work runs outside the gameplay transaction. A confirmed
        # action can therefore advance while the answer is still running.
        view = self.cmd(sid, view, "action", action="RIGHT")
        self.assertEqual(view["state"]["frame"], 1)
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT status FROM questions WHERE run_id=?",
                (question_run,)).fetchone()[0], "running")

        # Advance all remaining Task 1 rounds before releasing the answer. The
        # Task 2 response must never expose the old pending/running answer.
        view = self.end_next(sid, view)
        view = self.end_next(sid, view)
        view = self.end_next(sid, view)
        self.assertEqual(view["flow"]["stage"], "task2")
        self.assertFalse(view["explain_allowed"])
        self.assertEqual(view["answers"], [])
        self.assertNotIn("question", view["allowed_kinds"])

        blocker.release.set()
        for _ in range(100):
            with closing(self.store.connect()) as db:
                row = db.execute("SELECT status,answer,evidence_detail FROM questions WHERE run_id=?",
                    (question_run,)).fetchone()
            if row["status"] == "expired":
                break
            time.sleep(.01)
        self.assertEqual(tuple(row), ("expired", None, None))
        self.assertEqual(self.store.view(sid)["answers"], [])
        with self.assertRaises(online.CommandError) as error:
            self.cmd(sid, view, "answer_seen", answer_id="unknown")
        self.assertEqual(error.exception.reason, "explanation_forbidden")

    def test_six_rounds_questionnaire_and_completion(self):
        sid, view = self.enroll("study_1")
        view = self.begin_task1(sid, view)
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
        view = self.begin_task1(sid, view)
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
        self.assertEqual(view["release"]["release_version"], "legacy")
        self.assertIn("丢失", view["release"]["message"]["zh"])
        server = online.ThreadingHTTPServer(("127.0.0.1",0),
            online.handler_class(self.store, public_origin="https://study.test"))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1",server.server_port,timeout=3)
            conn.request("GET","/health"); response=conn.getresponse()
            self.assertEqual(response.status,200); health=json.loads(response.read())
            self.assertFalse(health["data_persistent"])
            self.assertEqual(health["version"], online.VERSION)
            self.assertNotIn(".v3", health["version"])
            self.assertEqual(health["release_version"], "legacy")
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
            self.assertEqual(enrolled["flow"]["stage"],"instructions")
            conn.close()
        finally:
            server.shutdown(); server.server_close(); thread.join(2)

    def test_online_assets_show_consent_before_registration(self):
        javascript = self.store.web_assets["app.js"].decode("utf-8")
        self.assertIn("registrationStage", javascript)
        self.assertIn('participant_id:id,consent:true', javascript)
        self.assertIn("内部预实验 · r4.1", javascript)
        self.assertIn("Internal pilot · r4.1", javascript)
        self.assertIn("evidence_detail", javascript)
        self.assertIn("whyCollision", javascript)
        self.assertIn('FRONTEND_VERSION="warehouse-alignment-online.r4.1"', javascript)
        self.assertIn('kind:"tutorial_advance"', javascript)
        self.assertIn('kind:"begin_task1"', javascript)
        self.assertIn('request("/api/study/command",{method:"POST"', javascript)
        self.assertNotIn('request("/api/command",{method:"POST"', javascript)
        self.assertIn("local(r.message,ui.language)", javascript)
        html = self.store.web_assets["index.html"].decode("utf-8")
        self.assertEqual(html.count("data-question="), 6)
        self.assertEqual(html.count('data-i18n="tutorialRule'), 8)
        for element_id in ("instructionsPanel", "tutorialPlayButton", "tutorialPreviousButton",
                           "tutorialNextButton", "beginTask1Button"):
            self.assertIn(f'id="{element_id}"', html)
        css = self.store.web_assets["styles.css"].decode("utf-8")
        self.assertIn("grid-template-columns: minmax(600px, 1.45fr)", css)

    def test_r41_release_label_is_public_but_deployment_hashes_stay_private(self):
        context = FakeContext(self.tmp.name)
        context.provenance.update(version=online.R41_RELEASE_CONTEXT_VERSION,
            package_sha256="8" * 64, manifest_sha256="9" * 64)
        for index, scene in enumerate(context.scenarios["splits"]["play"]):
            scene["fingerprint"] = format(index + 1, "064x")
        store = online.OnlineAlignmentStudyStore(context,
            database=Path(self.tmp.name) / "r41-identity.sqlite3",
            storage_mode="ephemeral")
        try:
            view = store.view(store.session())
            self.assertEqual(view["release"]["release_version"], "r4.1")
            serialized = online._canonical(view)
            for secret in ("a" * 64, "8" * 64, "9" * 64,
                           *(format(index, "064x") for index in range(2, 8))):
                self.assertNotIn(secret, serialized)
            identity = store.deployment_identity()
            self.assertEqual(identity["release_version"], "r4.1")
            self.assertEqual(identity["actor_sha256"], "a" * 64)
            self.assertEqual(identity["package_sha256"], "8" * 64)
            self.assertEqual(identity["manifest_sha256"], "9" * 64)
            self.assertEqual(identity["scene_fingerprints"], {
                "X": [format(index, "064x") for index in range(2, 5)],
                "Y": [format(index, "064x") for index in range(5, 8)],
            })
            secret = Path(self.tmp.name) / "warehouse_release.b64"
            secret.write_bytes(b"bound-secret-file-bytes\n")
            startup = online._startup_identity(store, base64_path=secret)
            self.assertEqual(startup["secret_file_sha256"],
                sha256(secret.read_bytes()).hexdigest())
        finally:
            store.close()

    def test_tutorial_coverage_cannot_reference_a_missing_frame(self):
        context = FakeContext(self.tmp.name)
        context.tutorial["coverage"]["pickup_frames"] = [len(context.tutorial["frames"])]
        context.tutorial_signature = online._digest(context.tutorial)
        context.provenance["tutorial_signature"] = context.tutorial_signature
        with self.assertRaisesRegex(ValueError, "coverage references an unknown frame"):
            online.OnlineAlignmentStudyStore(
                context, database=self.tmp.name + "/bad-tutorial.sqlite3",
                storage_mode="ephemeral")

    def test_question_evidence_column_is_added_to_existing_database(self):
        enrolled = [self.enroll(f"legacy_{index}") for index in range(4)]
        with closing(self.store.connect()) as db:
            sid, view = next((sid, view) for sid, view in enrolled
                if db.execute("SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0] == "A")
        view = self.begin_task1(sid, view)
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

    def test_tutorial_progress_columns_are_added_to_existing_database(self):
        sid = self.store.session()
        self.store.close()
        with sqlite3.connect(self.db) as db:
            db.executescript("""
                ALTER TABLE sessions RENAME TO sessions_current;
                CREATE TABLE sessions(id TEXT PRIMARY KEY,version INTEGER NOT NULL DEFAULT 0,active_run TEXT,
                 participant_id TEXT,participant_key TEXT UNIQUE,position INTEGER UNIQUE,condition TEXT,task_order TEXT,
                 mode TEXT NOT NULL DEFAULT 'enrollment',stage TEXT NOT NULL DEFAULT 'registration',round_index INTEGER NOT NULL DEFAULT 0,
                 questionnaire TEXT NOT NULL DEFAULT '{}',questionnaire_scores TEXT,namespace TEXT NOT NULL,study_signature TEXT,
                 questionnaire_bank_signature TEXT,consented TEXT);
                INSERT INTO sessions SELECT id,version,active_run,participant_id,participant_key,position,condition,task_order,
                 mode,stage,round_index,questionnaire,questionnaire_scores,namespace,study_signature,
                 questionnaire_bank_signature,consented FROM sessions_current;
                DROP TABLE sessions_current;
            """)
        self.store = online.OnlineAlignmentStudyStore(
            FakeContext(self.tmp.name), database=self.db, storage_mode="ephemeral")
        with closing(self.store.connect()) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            row = db.execute("SELECT tutorial_index,tutorial_max_index,tutorial_complete FROM sessions WHERE id=?",
                (sid,)).fetchone()
        self.assertTrue({"tutorial_index", "tutorial_max_index", "tutorial_complete"} <= columns)
        self.assertEqual(tuple(row), (0, 0, 0))
        self.assertEqual(self.store.view(sid)["flow"]["stage"], "registration")

    def test_version_change_rotates_browser_session_without_deleting_old_records(self):
        sid, view = self.enroll("old_version_1")
        view = self.begin_task1(sid, view)
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
