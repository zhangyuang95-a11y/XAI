"""Local native-NN warehouse sessions, isolated from all legacy study records."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sqlite3
from threading import Lock
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.runtime import NativeRuntime
from ui.warehouse_view import serialize_warehouse_state, warehouse_map_payload

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).with_name("warehouse_native")
COOKIE = "warehouse_native_session_v1"


def service_sources():
    # Native physics delegates to legacy warehouse code. Bind that dependency,
    # the serializer and the participant UI as well as the native Actor runtime.
    paths = list((ROOT / "env/warehouse").glob("*.py"))
    paths += list((ROOT / "env/warehouse_native").glob("*.py"))
    paths += list((ROOT / "core").glob("*.py"))
    paths += [Path(__file__), ROOT / "ui/warehouse_view.py", ROOT / "ui/warehouse_native_bank.py"]
    paths += [p for p in WEB.iterdir() if p.is_file()]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


class CommandError(ValueError):
    def __init__(self, reason, status=400, view=None):
        super().__init__(reason)
        self.reason, self.status, self.view = reason, status, view


class NativeStore:
    def __init__(self, database, actor_path, scenarios, *, verification=False, explainer=None, question_bank=None, capability_report=None, release_manifest=None):
        self.database = Path(database)
        if verification and release_manifest is not None:
            raise ValueError("Verification cannot load an accepted local-study release")
        self.runtime = NativeRuntime(actor_path)
        self.scenarios = scenarios
        self.verification = bool(verification)
        self.namespace = "verification" if self.verification else "local_pilot"
        self.test_fixture = self.runtime.actor.metadata.get("test_fixture") is True
        if self.test_fixture and not self.verification:
            raise ValueError("Test fixtures require an isolated verification server")
        self.cookie_name = "warehouse_native_verification_v1" if self.verification else COOKIE
        self.explainer = explainer
        self.question_bank = question_bank
        if question_bank is not None and (question_bank.actor_sha256 != self.runtime.actor.artifact_sha256 or question_bank.runtime_signature != self.runtime.signature or question_bank.foundation_manifest_sha256 != digest(scenarios)):
            raise ValueError("Question bank Actor/runtime/scenario binding mismatch")
        self.capability_status = "unknown"
        if capability_report is not None:
            if capability_report.get("actor_sha256") != self.runtime.actor.artifact_sha256 or type(capability_report.get("joint_steps")) is not int or capability_report.get("joint_steps") != self.runtime.actor.metadata.get("joint_steps"):
                raise ValueError("Capability report Actor/hash or checkpoint step mismatch")
            eligible = capability_report.get("capability", {}).get("eligible")
            if type(eligible) is not bool:
                raise ValueError("Capability report lacks an explicit eligibility result")
            self.capability_status = "passed" if eligible else "failed"
        if getattr(question_bank, "test_fixture", False) and not self.verification:
            raise ValueError("Test question-bank fixtures require an isolated verification server")
        if getattr(explainer, "test_fixture", False) and not self.verification:
            raise ValueError("Test explanation fixtures require an isolated verification server")
        self.release_audit = None
        if release_manifest is not None:
            from ui.warehouse_native_release import load_release
            self.release_audit = load_release(release_manifest, actor_path, scenarios, explainer, question_bank)
            if not self.release_audit.eligible:
                raise ValueError("Local-study release checks failed; inspect the release manifest with ui.warehouse_native_release")
            bound_capability = json.loads(Path(self.release_audit.artifacts["validation"]["path"]).read_text())
            if capability_report is not None and digest(capability_report) != digest(bound_capability):
                raise ValueError("Capability report differs from the verified release validation artifact")
            capability_report = bound_capability
            self.capability_status = "passed"
        self.source_binding = service_sources()
        self.web_assets = {name: (WEB / name).read_bytes() for name in ("index.html", "app.js", "styles.css")}
        if any(sha256(content).hexdigest() != self.source_binding["ui/warehouse_native/" + name]
               for name, content in self.web_assets.items()):
            raise ValueError("Participant assets changed while the service version was being bound")
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="native-explanation")
        self.scheduled = set()
        self.schedule_lock = Lock()
        self.signature = digest({"runtime": self.runtime.signature, "server": file_hash(Path(__file__)),
                                 "scenarios": digest(scenarios), "verification": verification,
                                 "explanation": getattr(explainer, "signature", None), "question_bank": getattr(question_bank, "signature", None),
                                 "capability_report": digest(capability_report) if capability_report is not None else None,
                                 "service_sources": self.source_binding,
                                 "local_study_release": getattr(self.release_audit, "signature", None)})
        # Technical qualification opens a local pilot only. It neither changes
        # the data namespace nor certifies human efficacy or formal sampling.
        self.release = {"status": "test_fixture" if self.test_fixture else "candidate", "model_ready": self.runtime.actor.metadata.get("joint_steps", 0) > 0,
                        "explanation_ready": bool(explainer is not None and explainer.eligible),
                        "study_ready": self.release_audit is not None,
                        "formal_ready": False, "capability_status": self.capability_status, "model_version": self.runtime.actor.metadata.get("policy_version"), "test_fixture": self.test_fixture, "joint_steps": self.runtime.actor.metadata.get("joint_steps", 0),
                        "message": {"zh": "候选模型：能力与解释验收尚未全部完成。", "en": "Candidate model: capability and explanation validation are not all complete."}}
        if self.release_audit is not None:
            self.release["status"] = "qualified_local_pilot"
            self.release["message"] = {"zh": "本地预实验版本已核验；不作为正式研究样本。", "en": "Verified local pilot release; records are not formal study samples."}
            self.release["local_study"] = self.release_audit.public_summary
        self.provenance = {
            "namespace": self.namespace, "verification_only": self.verification,
            "actor_sha256": self.runtime.actor.artifact_sha256, "runtime_signature": self.runtime.signature,
            "run_signature": self.signature, "server_sha256": file_hash(Path(__file__)),
            "service_sources": self.source_binding,
            "local_study_release_signature": getattr(self.release_audit, "signature", None),
            "local_study_artifact_sha256s": {name: item["sha256"] for name, item in self.release_audit.artifacts.items()} if self.release_audit is not None else None,
            "analysis_protocol": self.release_audit.analysis if self.release_audit is not None else None,
            "scenario_manifest_sha256": digest(scenarios),
            "explanation_signature": getattr(explainer, "signature", None), "release": dict(self.release),
            "question_bank_signature": getattr(question_bank, "signature", None),
            "capability_report_sha256": digest(capability_report) if capability_report is not None else None,
            "question_bank_status": question_bank.summary() if question_bank else {"status":"unavailable","formal_ready":False},
            "study_configuration": {"task1_rounds": 3, "task2_rounds": 3,
                "assignment": "A/B x XY/YX randomized blocks of four", "task2_primary": "mean_deliveries",
                "questionnaire_kind": "candidate_prediction_and_self_report" if question_bank and question_bank.eligible else "self_report_only", "questionnaire_items": questionnaire_items(),
                "behavior_prediction_bank_ready": bool(question_bank and question_bank.eligible),
                "behavior_prediction_bank_formal_ready": False, "formal_enrollment_ready": False},
        }
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
                    active_run TEXT, participant_id TEXT, participant_key TEXT UNIQUE, position INTEGER UNIQUE,
                    condition TEXT, task_order TEXT, mode TEXT NOT NULL DEFAULT 'freeplay',
                    stage TEXT NOT NULL DEFAULT 'practice', round_index INTEGER NOT NULL DEFAULT 0,
                    questionnaire TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,scenario_id TEXT NOT NULL,
                    signature TEXT NOT NULL,stage TEXT NOT NULL,round_index INTEGER NOT NULL,snapshot TEXT NOT NULL,
                    metrics TEXT NOT NULL,ended INTEGER NOT NULL DEFAULT 0,created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS frames(run_id TEXT NOT NULL,frame INTEGER NOT NULL,public TEXT NOT NULL,
                    internal TEXT NOT NULL,PRIMARY KEY(run_id,frame));
                CREATE TABLE IF NOT EXISTS operations(session_id TEXT NOT NULL,id TEXT NOT NULL,digest TEXT NOT NULL,
                    version INTEGER NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(session_id,id));
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,run_id TEXT,
                    kind TEXT,payload TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,run_id TEXT NOT NULL,
                    frame INTEGER NOT NULL,stage TEXT NOT NULL,question TEXT NOT NULL,language TEXT NOT NULL,
                    status TEXT NOT NULL,answer TEXT,shown TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,allocation TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)
            # Additive migrations preserve prior local runs and question records.
            for table, column, declaration in (("questions", "focus", "TEXT NOT NULL DEFAULT 'executed'"),
                                                ("runs", "feedback", "TEXT NOT NULL DEFAULT '{}'"),
                                                ("runs", "end_reason", "TEXT"),
                                                ("runs", "provenance", "TEXT"),
                                                ("sessions", "namespace", "TEXT"),
                                                ("sessions", "study_signature", "TEXT"),
                                                ("sessions", "questionnaire_bank_signature", "TEXT"),
                                                ("sessions", "questionnaire_scores", "TEXT")):
                if column not in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}:
                    db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
            db.execute("BEGIN IMMEDIATE")
            try:
                record = db.execute("SELECT value FROM metadata WHERE key='namespace'").fetchone()
                if record is None:
                    has_history = db.execute("SELECT count(*) FROM sessions").fetchone()[0] > 0
                    recorded_namespace = "unknown_legacy" if has_history else self.namespace
                    db.execute("INSERT INTO metadata VALUES('namespace',?)", (canonical(recorded_namespace),))
                    db.execute("INSERT INTO metadata VALUES('namespace_origin',?)", (canonical("legacy_missing" if has_history else "service_initialization"),))
                else:
                    recorded_namespace = json.loads(record[0])
                    if recorded_namespace not in ("unknown_legacy", self.namespace):
                        raise ValueError("Database namespace differs from the requested server mode")
                db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", ("service_context:"+self.signature, canonical(self.provenance)))
                db.commit()
            except Exception:
                db.rollback()
                raise
            db.execute("UPDATE sessions SET stage='freeplay' WHERE mode='freeplay' AND stage='practice'")
            db.execute("UPDATE runs SET stage='freeplay' WHERE session_id IN (SELECT id FROM sessions WHERE mode='freeplay') AND stage='practice'")
            db.execute("UPDATE questions SET status='pending' WHERE status='running'")

    def close(self):
        self.executor.shutdown(wait=True)

    def connect(self):
        db = sqlite3.connect(self.database, isolation_level=None, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        return db

    def session(self, candidate):
        with closing(self.connect()) as db:
            if candidate and db.execute("SELECT id FROM sessions WHERE id=?", (candidate,)).fetchone():
                return candidate
            sid = uuid4().hex
            db.execute("INSERT INTO sessions(id,stage,namespace) VALUES(?,'freeplay',?)", (sid, self.namespace))
            return sid

    def _session(self, db, sid):
        row = db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if row is None:
            raise CommandError("session_not_found", 401)
        return row

    def _run(self, db, sid, rid=None):
        rid = rid or self._session(db, sid)["active_run"]
        row = db.execute("SELECT * FROM runs WHERE id=? AND session_id=?", (rid, sid)).fetchone()
        if row is None:
            raise CommandError("run_not_found", 404)
        return row

    def _permitted(self, session):
        return not self._study_mismatch(session) and self.release["explanation_ready"] and (session["mode"] == "freeplay" or (
            session["stage"] == "task1" and session["condition"] == "A"))

    def _study_mismatch(self, session):
        # Missing historical bindings remain unknown; never silently certify
        # an old participant under the currently loaded model or protocol.
        return session["mode"] == "study" and session["study_signature"] != self.signature

    def _can_explain(self, db, session):
        if not self._permitted(session) or not session["active_run"]:
            return False
        run = self._run(db, session["id"])
        return run["signature"] == self.signature and run["stage"] == session["stage"]

    def _frame(self, env, metrics, actions=None):
        state = serialize_warehouse_state(env.get_state(), selected_agent="robot_2", reveal_policy=False)
        state.pop("human_route_regret_units", None)
        state.pop("policy_hidden", None)
        for agent in state["agents"]:
            agent.pop("proposed_action", None)
            agent.pop("reward", None)
        return {"state": state, "metrics": dict(metrics), "actions": actions or {}}

    def _metrics(self, env, old=None, actions=None):
        s = env.state
        result = dict(old or {"ai_waits": 0, "ai_blocked": 0, "overrides": 0})
        result.update(steps=s.frame, deliveries=s.total_deliveries, score=s.user_score,
                      native_score=s.user_score, legacy_score=None, collisions=s.robot_collision_events,
                      shutdowns=s.shutdown_count, individual_deliveries=[a.deliveries_completed for a in s.agents])
        if actions:
            result["ai_waits"] += int(actions["submitted_actions"]["robot_2"] == "WAIT")
            result["ai_blocked"] += int(actions["submitted_actions"]["robot_2"] != "WAIT" and actions["executed_actions"]["robot_2"] == "WAIT")
        return result

    def _start_run(self, db, sid, scene_index):
        session = self._session(db, sid)
        scene = self.scenarios["splits"]["play"][scene_index]
        env = self.runtime.environment(scene)
        rid = uuid4().hex
        metrics = self._metrics(env)
        db.execute("INSERT INTO runs(id,session_id,scenario_id,signature,stage,round_index,snapshot,metrics,provenance) VALUES(?,?,?,?,?,?,?,?,?)",
            (rid, sid, scene["id"], self.signature, session["stage"], session["round_index"], canonical(env.snapshot()), canonical(metrics), canonical({**self.provenance, "scenario_sha256": digest(scene)})))
        db.execute("INSERT INTO frames VALUES(?,?,?,?)", (rid, 0, canonical(self._frame(env, metrics)), canonical({"after": env.snapshot(), "actor_sha256": self.runtime.actor.artifact_sha256, "runtime_signature": self.runtime.signature, "run_signature": self.signature, "scenario_sha256": digest(scene)})))
        db.execute("UPDATE sessions SET active_run=? WHERE id=?", (rid, sid))
        return rid

    def _bound_bank(self, session):
        signature = session["questionnaire_bank_signature"]
        if signature is None:
            return None
        if self.question_bank is None or self.question_bank.signature != signature or not self.question_bank.eligible:
            raise CommandError("questionnaire_bank_version_mismatch", 409)
        return self.question_bank

    def _questionnaire(self, session):
        blocked = self._study_mismatch(session)
        try:
            bank = self._bound_bank(session)
        except CommandError:
            bank = None
            blocked = True
        visible = session["stage"] == "questionnaire" and not blocked
        return {"items": questionnaire_items() + (bank.public_items() if visible and bank else []),
                "draft": json.loads(session["questionnaire"]), "blocked": blocked,
                "bank": {"status": "version_mismatch" if blocked else "candidate_ready" if bank else "candidate_failed" if self.question_bank and not self.question_bank.eligible else "unavailable",
                         "available": bool(bank), "formal_ready": False, "bound_at_enrollment": bool(session["questionnaire_bank_signature"])}}

    def _view(self, db, sid):
        session = self._session(db, sid)
        permitted = self._can_explain(db, session)
        result = {"session_id": sid, "version": session["version"], "run_id": session["active_run"],
            "release": self.release, "study_allowed": (self.verification or self.release["study_ready"]) and not self._study_mismatch(session),
            "study_version_mismatch": self._study_mismatch(session),
            "verification_only": self.verification,
            "flow": {"mode": session["mode"], "stage": session["stage"], "round_index": session["round_index"]+1,
                "round_count": 3 if session["stage"] in ("task1", "task2") else 1, "participant_id": session["participant_id"]},
            "map": warehouse_map_payload(NativeWarehouseEnv().layout), "horizon": 120,
            "explain_allowed": permitted, "answers": [], "runs": [], "history_count": 0,
            "state": None, "metrics": {}, "ended": False, "play_scene_count": len(self.scenarios["splits"]["play"]),
            "questionnaire": self._questionnaire(session)}
        if session["active_run"]:
            run = self._run(db, sid)
            frame = db.execute("SELECT public FROM frames WHERE run_id=? ORDER BY frame DESC LIMIT 1", (run["id"],)).fetchone()
            result.update(json.loads(frame[0]))
            scene_index = next((i for i, scene in enumerate(self.scenarios["splits"]["play"]) if scene["id"] == run["scenario_id"]), None)
            result.update(ended=bool(run["ended"]), done=bool(run["ended"]), end_reason=run["end_reason"],
                          version_mismatch=run["signature"] != self.signature, seed=scene_index,
                          feedback=json.loads(run["feedback"]), history_count=result["state"]["frame"]+1)
            if permitted:
                result["answers"] = [{"id": q["id"], "status": q["status"], "frame": q["frame"],
                    "question": q["question"], "text": q["answer"] or "", "run_id": q["run_id"], "focus": q["focus"]} for q in db.execute(
                    "SELECT * FROM questions WHERE session_id=? AND run_id=? ORDER BY created,id", (sid, run["id"]))]
        if session["mode"] == "freeplay":
            result["runs"] = [{"run_id": row["id"], "steps": json.loads(row["metrics"])["steps"],
                "score": json.loads(row["metrics"])["score"], "mode": "freeplay", "created": row["created"],
                "seed": next((i for i, scene in enumerate(self.scenarios["splits"]["play"]) if scene["id"] == row["scenario_id"]), None)} for row in db.execute(
                "SELECT * FROM runs WHERE session_id=? ORDER BY rowid DESC LIMIT 100", (sid,))]
        kinds = []
        if session["mode"] == "freeplay":
            kinds.append("start")
            if session["active_run"]:
                kinds.append("select_run")
                if result["ended"]:
                    kinds.append("feedback")
        elif session["stage"] == "consent":
            kinds.append("next")
        if session["active_run"] and session["stage"] in ("freeplay", "practice", "task1", "task2"):
            if not result["ended"]:
                kinds.append("end")
                if not result["version_mismatch"] and self.release["model_ready"]:
                    kinds.append("action")
            elif session["mode"] == "study":
                kinds.append("next")
        if permitted and session["active_run"]:
            kinds += ["question", "answer_seen"]
        if session["stage"] == "questionnaire":
            kinds.append("questionnaire")
        result["allowed_kinds"] = [] if self._study_mismatch(session) else kinds
        return result

    def view(self, sid):
        self.schedule_pending(sid)
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            return self._view(db, sid)

    def history(self, sid, run_id=None):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            run = self._run(db, sid)
            if run_id is not None and run_id != run["id"]:
                raise CommandError("history_current_run_only", 403)
            return {"run_id": run["id"], "version": self._session(db, sid)["version"],
                    "map": warehouse_map_payload(NativeWarehouseEnv().layout),
                    "frames": [json.loads(row[0]) for row in db.execute(
                        "SELECT public FROM frames WHERE run_id=? ORDER BY frame", (run["id"],))]}

    def command(self, sid, payload):
        op = payload.get("operation_id")
        kind = payload.get("kind")
        if not isinstance(op, str) or not 1 <= len(op) <= 128:
            raise CommandError("operation_id_required")
        signature = digest(payload)
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                session = self._session(db, sid)
                old = db.execute("SELECT * FROM operations WHERE session_id=? AND id=?", (sid, op)).fetchone()
                if old:
                    if old["digest"] != signature:
                        raise CommandError("operation_id_reused", 409, self._view(db, sid))
                    if self._study_mismatch(session):
                        raise CommandError("study_version_changed", 409, self._view(db, sid))
                    # Return the current permission-filtered view, never an old
                    # cached Task 1 answer after entering Task 2.
                    return self._view(db, sid)
                if type(payload.get("expected_version")) is not int or payload["expected_version"] != session["version"]:
                    raise CommandError("state_version_conflict", 409, self._view(db, sid))
                if self._study_mismatch(session):
                    raise CommandError("study_version_changed", 409, self._view(db, sid))
                if kind == "start":
                    mode = payload.get("mode", "freeplay")
                    if session["mode"] == "study":
                        raise CommandError("study_cannot_restart_or_switch_mode", 403)
                    if not self.release["model_ready"] and not self.verification:
                        raise CommandError("model_not_trained", 409)
                    if mode == "study":
                        if len(self.scenarios["splits"]["play"]) < 7:
                            raise CommandError("study_scenes_incomplete", 409)
                        if not (self.verification or self.release["study_ready"]):
                            raise CommandError("formal_release_not_ready", 403)
                        participant = str(payload.get("participant_id", "")).strip()
                        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,31}", participant):
                            raise CommandError("invalid_participant_id")
                        if db.execute("SELECT id FROM sessions WHERE participant_key=?", (participant.lower(),)).fetchone():
                            raise CommandError("participant_id_taken", 409)
                        position = db.execute("SELECT count(*) FROM sessions WHERE position IS NOT NULL").fetchone()[0]
                        block_index = position // 4
                        block = db.execute("SELECT allocation FROM blocks WHERE id=?", (block_index,)).fetchone()
                        if block is None:
                            cells = [["A", "XY"], ["A", "YX"], ["B", "XY"], ["B", "YX"]]
                            random.SystemRandom().shuffle(cells)
                            db.execute("INSERT INTO blocks VALUES(?,?)", (block_index, canonical(cells)))
                        else:
                            cells = json.loads(block[0])
                        condition, order = cells[position % 4]
                        stage = "practice" if payload.get("consent") is True else "consent"
                        db.execute("UPDATE sessions SET mode='study',stage=?,participant_id=?,participant_key=?,position=?,condition=?,task_order=?,round_index=0,active_run=NULL,questionnaire_bank_signature=?,study_signature=? WHERE id=?",
                            (stage, participant, participant.lower(), position, condition, order, self.question_bank.signature if self.question_bank and self.question_bank.eligible else None, self.signature, sid))
                        if stage == "practice":
                            self._start_run(db, sid, 0)
                    elif mode == "freeplay":
                        db.execute("UPDATE sessions SET mode='freeplay',stage='freeplay' WHERE id=?", (sid,))
                        index = payload.get("seed", 0)
                        if type(index) is not int or not 0 <= index < len(self.scenarios["splits"]["play"]):
                            raise CommandError("invalid_play_scene")
                        self._start_run(db, sid, index)
                    else:
                        raise CommandError("invalid_mode")
                elif kind == "action":
                    if session["stage"] not in ("freeplay", "practice", "task1", "task2"):
                        raise CommandError("actions_not_allowed_in_stage", 403)
                    run = self._run(db, sid)
                    if run["signature"] != self.signature:
                        raise CommandError("runtime_version_changed", 409)
                    if run["ended"]:
                        raise CommandError("round_ended", 409)
                    env = NativeWarehouseEnv()
                    env.restore(json.loads(run["snapshot"]))
                    if payload.get("action") not in ("UP", "DOWN", "LEFT", "RIGHT", "WAIT"):
                        raise CommandError("invalid_action")
                    outcome = self.runtime.step(env, payload["action"])
                    metrics = self._metrics(env, json.loads(run["metrics"]), outcome)
                    db.execute("UPDATE runs SET snapshot=?,metrics=?,ended=? WHERE id=?",
                        (canonical(env.snapshot()), canonical(metrics), int(outcome["done"]), run["id"]))
                    db.execute("INSERT INTO frames VALUES(?,?,?,?)", (run["id"], env.state.frame,
                        canonical(self._frame(env, metrics, outcome["executed_actions"])), canonical(outcome)))
                elif kind == "select_run":
                    if session["mode"] != "freeplay":
                        raise CommandError("study_round_selection_forbidden", 403)
                    run = self._run(db, sid, payload.get("run_id"))
                    db.execute("UPDATE sessions SET active_run=? WHERE id=?", (run["id"], sid))
                elif kind == "end":
                    if session["stage"] not in ("freeplay", "practice", "task1", "task2"):
                        raise CommandError("end_not_allowed_in_stage", 403)
                    run = self._run(db, sid)
                    if run["ended"]:
                        raise CommandError("round_ended", 409)
                    db.execute("UPDATE runs SET ended=1,end_reason='participant_ended' WHERE id=?", (run["id"],))
                elif kind == "next":
                    self._next(db, session, payload)
                elif kind == "question":
                    if not self._can_explain(db, session):
                        raise CommandError("explanation_forbidden", 403)
                    run = self._run(db, sid, payload.get("run_id"))
                    if run["id"] != session["active_run"] or run["stage"] != session["stage"]:
                        raise CommandError("explanation_round_mismatch", 403)
                    frame = payload.get("frame")
                    if type(frame) is not int or not db.execute("SELECT 1 FROM frames WHERE run_id=? AND frame=?", (run["id"], frame)).fetchone():
                        raise CommandError("frame_not_found", 404)
                    question = str(payload.get("question", "")).strip()
                    if not 1 <= len(question) <= 1000:
                        raise CommandError("invalid_question")
                    language, focus = payload.get("language", "zh"), payload.get("focus", "executed")
                    if language not in ("zh", "en") or focus not in ("executed", "next"):
                        raise CommandError("invalid_question_reference")
                    db.execute("INSERT INTO questions(id,session_id,run_id,frame,stage,question,language,status,focus) VALUES(?,?,?,?,?,?,?,'pending',?)",
                        (uuid4().hex, sid, run["id"], frame, session["stage"], question, language, focus))
                elif kind == "answer_seen":
                    if not self._can_explain(db, session):
                        raise CommandError("explanation_forbidden", 403)
                    question = db.execute("SELECT * FROM questions WHERE id=? AND session_id=? AND run_id=?",
                        (payload.get("answer_id"), sid, session["active_run"])).fetchone()
                    if not question:
                        raise CommandError("answer_not_found", 404)
                    db.execute("UPDATE questions SET shown=coalesce(shown,CURRENT_TIMESTAMP) WHERE id=?", (question["id"],))
                elif kind == "questionnaire":
                    if session["stage"] != "questionnaire":
                        raise CommandError("questionnaire_not_allowed", 403)
                    values = payload.get("answers", {})
                    if not isinstance(values, dict) or len(canonical(values)) > 10000:
                        raise CommandError("invalid_questionnaire")
                    draft = json.loads(session["questionnaire"])
                    draft.update(values)
                    bank = self._bound_bank(session)
                    items = questionnaire_items() + (bank.public_items() if bank else [])
                    ids = {item["id"] for item in items}
                    if set(draft) - ids:
                        raise CommandError("unknown_questionnaire_item")
                    if type(payload.get("submit", False)) is not bool:
                        raise CommandError("invalid_questionnaire_submit")
                    for item in items:
                        value = draft.get(item["id"])
                        if value is None:
                            continue
                        if item["type"] == "scale":
                            valid = type(value) is int and 1 <= value <= 7
                        else:
                            valid = isinstance(value, str) and value in {option["value"] for option in item["options"]}
                        if not valid:
                            raise CommandError("invalid_questionnaire_value")
                    if payload.get("submit") and (not ids <= draft.keys() or any(draft.get(key) is None for key in ids)):
                        raise CommandError("questionnaire_incomplete")
                    scores = bank.grade(draft) if payload.get("submit") and bank else None
                    db.execute("UPDATE sessions SET questionnaire=?,stage=?,questionnaire_scores=? WHERE id=?",
                        (canonical(draft), "completed" if payload.get("submit") else "questionnaire", canonical(scores) if scores else None, sid))
                elif kind == "feedback":
                    if session["mode"] != "freeplay":
                        raise CommandError("feedback_requires_freeplay", 403)
                    run = self._run(db, sid)
                    if not run["ended"]:
                        raise CommandError("feedback_requires_ended_run", 409)
                    feedback = payload.get("feedback")
                    if not isinstance(feedback, dict) or set(feedback) - {"clarity", "cooperation", "note"}:
                        raise CommandError("invalid_feedback")
                    if any(type(feedback.get(key)) is not int or not 1 <= feedback[key] <= 5 for key in ("clarity", "cooperation")):
                        raise CommandError("invalid_feedback_rating")
                    if not isinstance(feedback.get("note", ""), str) or len(feedback.get("note", "")) > 2000:
                        raise CommandError("invalid_feedback_note")
                    db.execute("UPDATE runs SET feedback=? WHERE id=?", (canonical(feedback), run["id"]))
                else:
                    raise CommandError("unknown_command")
                db.execute("UPDATE sessions SET version=version+1 WHERE id=?", (sid,))
                version = self._session(db, sid)["version"]
                db.execute("INSERT INTO operations VALUES(?,?,?,?,?)", (sid, op, signature, version, kind))
                db.execute("INSERT INTO events(session_id,run_id,kind,payload) VALUES(?,?,?,?)",
                    (sid, self._session(db, sid)["active_run"], kind, canonical(payload)))
                result = self._view(db, sid)
                db.commit()
            except Exception:
                db.rollback()
                raise
        self.schedule_pending(sid)
        return result

    def _next(self, db, session, payload):
        sid = session["id"]
        if session["mode"] != "study":
            raise CommandError("next_requires_study")
        stage, index = session["stage"], session["round_index"]
        if stage == "consent":
            if payload.get("consent") is not True:
                raise CommandError("explicit_consent_required")
            db.execute("UPDATE sessions SET stage='practice' WHERE id=?", (sid,))
            self._start_run(db, sid, 0)
            return
        if stage not in ("practice", "task1", "task2"):
            raise CommandError("cannot_advance_stage")
        run = self._run(db, sid)
        if not run["ended"]:
            raise CommandError("finish_or_end_round_first", 409)
        if stage == "practice":
            stage, index = "task1", 0
        elif index < 2:
            index += 1
        elif stage == "task1":
            stage, index = "task2", 0
        else:
            stage, index = "questionnaire", 0
        db.execute("UPDATE sessions SET stage=?,round_index=?,active_run=NULL WHERE id=?", (stage, index, sid))
        if stage in ("task1", "task2"):
            bank = 0 if (stage == "task1") == (session["task_order"] == "XY") else 1
            self._start_run(db, sid, 1 + bank*3 + index)

    def schedule_pending(self, sid):
        if self.explainer is None:
            return
        with closing(self.connect()) as db:
            rows = list(db.execute("SELECT id FROM questions WHERE session_id=? AND status='pending'", (sid,)))
        for row in rows:
            with self.schedule_lock:
                if row[0] in self.scheduled:
                    continue
                self.scheduled.add(row[0])
            self.executor.submit(self._answer, row[0])

    def _answer(self, qid):
        try:
            with closing(self.connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                q = db.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
                s = self._session(db, q["session_id"])
                if not self._can_explain(db, s) or s["active_run"] != q["run_id"] or s["stage"] != q["stage"]:
                    db.execute("UPDATE questions SET status='expired' WHERE id=?", (qid,))
                    db.commit()
                    return
                frame = db.execute("SELECT internal FROM frames WHERE run_id=? AND frame=?", (q["run_id"], q["frame"])).fetchone()
                db.execute("UPDATE questions SET status='running' WHERE id=?", (qid,))
                db.commit()
            # Any slow inference is outside the game transaction.
            answer = self.explainer.answer(dict(q), json.loads(frame[0]), self.runtime)
            with closing(self.connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                s = self._session(db, q["session_id"])
                permitted = self._can_explain(db, s) and s["active_run"] == q["run_id"] and s["stage"] == q["stage"]
                db.execute("UPDATE questions SET status=?,answer=? WHERE id=?",
                    ("complete" if permitted else "expired", answer, qid))
                db.commit()
        except Exception:
            with closing(self.connect()) as db:
                db.execute("UPDATE questions SET status='failed',answer=NULL WHERE id=?", (qid,))
        finally:
            with self.schedule_lock:
                self.scheduled.discard(qid)


def questionnaire_items():
    # Self-report only until a separately frozen behavioral prediction bank exists.
    return [{"id": key, "type": "scale", "prompt": {"zh": zh, "en": en}, "options": [1,2,3,4,5,6,7]}
            for key, zh, en in [("cooperation", "我理解如何与队友配合。", "I understand how to cooperate with the teammate."),
                ("predictability", "我能预测队友的行为。", "I can predict the teammate's behavior."),
                ("difficulty", "任务的配合难度很高。", "Coordination in the task was difficult.")]]


def handler_class(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload, sid=None, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else canonical(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if sid:
                self.send_header("Set-Cookie", f"{store.cookie_name}={sid}; Path=/; HttpOnly; SameSite=Strict")
            self.end_headers()
            self.wfile.write(body)

        def sid(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
            except Exception:
                pass
            return store.session(cookie[store.cookie_name].value if store.cookie_name in cookie else None)

        def do_GET(self):
            path = urlparse(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/assets/styles.css": ("styles.css", "text/css; charset=utf-8")}
            if path in assets:
                name, mime = assets[path]
                self.reply(200, store.web_assets[name], content_type=mime)
                return
            sid = self.sid()
            try:
                if path == "/api/view":
                    value = store.view(sid)
                elif path == "/api/history":
                    value = store.history(sid, parse_qs(urlparse(self.path).query).get("run_id", [None])[0])
                else:
                    raise CommandError("not_found", 404)
                self.reply(200, value, sid)
            except CommandError as error:
                self.reply(error.status, {"error": error.reason, "view": error.view}, sid)

        def do_POST(self):
            sid = self.sid()
            try:
                if urlparse(self.path).path != "/api/command":
                    raise CommandError("not_found", 404)
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    raise CommandError("origin_forbidden", 403)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20000:
                    raise CommandError("invalid_body_size", 413)
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise CommandError("invalid_body")
                self.reply(200, store.command(sid, payload), sid)
            except CommandError as error:
                self.reply(error.status, {"error": error.reason, "view": error.view}, sid)
            except (ValueError, TypeError):
                self.reply(400, {"error": "invalid_request"}, sid)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=ROOT / "output/warehouse_native/local_play.sqlite3")
    parser.add_argument("--port", type=int, default=8007)
    parser.add_argument("--capability-report", type=Path, help="Selected-checkpoint validation report; never grants formal release")
    parser.add_argument("--question-bank", type=Path, help="Server-private candidate prediction bank, bound to this Actor and scenario foundation")
    parser.add_argument("--program", type=Path, help="Actor-bound extracted program (optional, requires --explanation-report)")
    parser.add_argument("--explanation-report", type=Path, help="Matching explanation acceptance report")
    parser.add_argument("--verification", action="store_true", help="Isolated test flow only; never a formal release")
    parser.add_argument("--release-manifest", type=Path, help="Independently verified, hash-bound local A/B pilot release; does not certify formal study samples")
    args = parser.parse_args(argv)
    if args.verification and args.release_manifest is not None:
        parser.error("Verification and an accepted local-study release cannot be combined")
    if args.verification and args.database == ROOT / "output/warehouse_native/local_play.sqlite3":
        parser.error("Verification requires its own explicitly named database")
    if (args.program is None) != (args.explanation_report is None):
        parser.error("--program and --explanation-report must be supplied together")
    explainer = None
    if args.program is not None:
        from env.warehouse_native.explanation import NativeExplainer
        explainer = NativeExplainer(args.program, args.explanation_report, file_hash(args.actor))
    scenes = json.loads(args.scenarios.read_text())
    bank = None
    if args.question_bank is not None:
        from ui.warehouse_native_bank import NativeQuestionBank
        bank = NativeQuestionBank(args.question_bank, NativeRuntime(args.actor), scenes)
    store = NativeStore(args.database, args.actor, scenes, verification=args.verification, explainer=explainer, question_bank=bank, capability_report=json.loads(args.capability_report.read_text()) if args.capability_report else None, release_manifest=args.release_manifest)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_class(store))
    print(f"Native warehouse candidate at http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        store.close()


if __name__ == "__main__":
    main()
