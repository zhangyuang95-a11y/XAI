"""Portable, dependency-light online A/B server for Warehouse alignment.

The release loader performs artifact admission.  This module owns only the
participant protocol, the SQLite journal, public projection, and HTTP edge.
It deliberately does not import any of the local study server modules so that
starting the Render process does not import PyTorch or training code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
from threading import Lock
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "ui/warehouse_family_feedback_research"
VERSION = "warehouse-alignment-online-study-server.v2"
SERVICE_FAMILY = "warehouse_alignment_online_r2"
NAMESPACE = "online_demo"
COOKIE = "warehouse_alignment_online_session_v1"
DEFAULT_PORT = 8000
DEFAULT_ORIGIN = "https://policylens-warehouse-study.onrender.com"
DEFAULT_DATABASE = Path(os.environ.get("WAREHOUSE_ONLINE_DATABASE", "/tmp/warehouse_alignment_online.sqlite3"))
MAX_BODY = 20_000
_ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT", "WAIT"}
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,31}")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return sha256(_canonical(value).encode()).hexdigest()


def _utcnow():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    """Convert runtime dataclasses/tuples to JSON values without importing env code."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _attr(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _point(value):
    value = list(value)
    return [int(value[0]), int(value[1])]


def _public_map(env):
    layout = getattr(env, "layout", None)
    if layout is not None:
        try:
            rows, cols = int(layout.rows), int(layout.cols)
            blocked = getattr(layout, "blocked_positions", ())
            charger = getattr(layout, "charger_position")
            exits = getattr(layout, "robot_exit_positions", ())
            starts = getattr(layout, "robot_start_positions", ())
            passable = getattr(layout, "passable_positions", ())
            return {
                "layout_id": str(getattr(layout, "layout_id", "warehouse-6x7")),
                "rows": rows, "cols": cols,
                "shelves": [_point(p) for p in blocked],
                "charger_position": _point(charger),
                "robot_exit_positions": [_point(p) for p in exits],
                "waiting_zone": [_point(p) for p in passable if int(p[0]) == rows - 1 and tuple(p) != tuple(charger)],
                "robot_start_positions": [_point(p) for p in starts],
                "shared_delivery_tasks": True,
            }
        except (AttributeError, TypeError, ValueError):
            pass
    # Frozen 6x7 participant geometry; model state still comes from the release.
    shelves = [[0,0],[0,1],[0,2],[0,4],[0,5],[0,6],[1,4],[1,5],[1,6],
        [2,0],[2,1],[3,3],[3,4],[3,5],[3,6],[4,0],[4,1],[4,5],[4,6],
        [5,0],[5,1],[5,5],[5,6]]
    return {"layout_id":"warehouse-6x7","rows":6,"cols":7,"shelves":shelves,
        "charger_position":[5,3],"robot_exit_positions":[[4,2],[4,3],[4,4]],
        "waiting_zone":[[5,2],[5,4]],"robot_start_positions":[[5,2],[5,4]],
        "shared_delivery_tasks":True}


def _state(env):
    getter = getattr(env, "get_state", None)
    return getter() if callable(getter) else env.state


def _public_state(env):
    state = _state(env)
    tasks = sorted(list(_attr(state, "tasks", ())), key=lambda t: str(_attr(t, "task_id", "")))
    slots = {str(_attr(task, "task_id")): i for i, task in enumerate(tasks, 1)}
    agents = []
    for agent in _attr(state, "agents", ()):
        agent_id = str(_attr(agent, "agent_id", _attr(agent, "id", "")))
        carrying = _attr(agent, "carrying_task_id")
        agents.append({
            "id": agent_id,
            "position": _point(_attr(agent, "position")),
            "battery": float(_attr(agent, "battery", 0)),
            "carrying_task_id": carrying,
            "carrying_label": f"A{slots[str(carrying)]}" if carrying is not None and str(carrying) in slots else None,
            "deliveries_completed": int(_attr(agent, "deliveries_completed", 0)),
            "active": bool(_attr(agent, "active", True)),
            "heading": str(_attr(agent, "heading", "WAIT")),
            "last_action": str(_attr(agent, "last_action", "WAIT")),
            "last_executed_action": str(_attr(agent, "last_executed_action", "WAIT")),
            "selected": agent_id == "robot_2",
        })
    return {
        "episode_id": int(_attr(state, "episode_id", 0)),
        "frame": int(_attr(state, "frame", 0)),
        "total_deliveries": int(_attr(state, "total_deliveries", 0)),
        "active_count": sum(bool(a["active"]) for a in agents),
        "collision_count": int(_attr(state, "collision_count", 0)),
        "shutdown_count": int(_attr(state, "shutdown_count", 0)),
        "terminated": bool(_attr(state, "terminated", False)),
        "truncated": bool(_attr(state, "truncated", False)),
        "terminal_reason": _attr(state, "terminal_reason"),
        "selected_agent": "robot_2",
        "agents": agents,
        "tasks": [{
            "task_id": str(_attr(task, "task_id")),
            "pickup_position": _point(_attr(task, "pickup_position")),
            "delivery_position": _point(_attr(task, "delivery_position")),
            "status": str(_attr(task, "status", "available")),
            "carrier_agent_id": _attr(task, "carrier_agent_id"),
            "created_frame": int(_attr(task, "created_frame", 0)),
            "claimed_frame": (int(_attr(task, "claimed_frame")) if _attr(task, "claimed_frame") is not None else None),
        } for task in tasks],
        "user_score": float(_attr(state, "user_score", 0)),
        "score_breakdown": {str(k): float(v) for k, v in dict(_attr(state, "score_breakdown", {})).items()},
        "robot_collision_events": int(_attr(state, "robot_collision_events", _attr(state, "collision_count", 0))),
        "invalid_move_count": int(_attr(state, "invalid_move_count", 0)),
        "events": None,
        "policy_hidden": True,
    }


def _public_history(env, outcome=None):
    method = getattr(env, "public_history", None)
    if callable(method):
        return _plain(method())
    outcome = outcome or {}
    submitted = outcome.get("submitted_actions", outcome.get("policy_actions", {}))
    executed = outcome.get("executed_actions", submitted)
    return {"valid": bool(outcome), "submitted_actions": submitted,
        "executed_actions": executed, "move_canceled": {},
        "consecutive_move_canceled": {}, "collision_kind": "none",
        "consecutive_collision": 0}


def _questionnaire_items():
    return [{"id": key, "type": "scale", "required": True,
        "prompt": {"zh": zh, "en": en}, "options": [1,2,3,4,5,6,7]}
        for key, zh, en in (
            ("cooperation", "我理解如何与队友配合。", "I understand how to cooperate with the teammate."),
            ("predictability", "我能预测队友的行为。", "I can predict the teammate's behavior."),
            ("difficulty", "任务的配合难度很高。", "Coordination in the task was difficult."),
        )]


def _assets():
    data = {name: (WEB / name).read_bytes()
            for name in ("index.html", "app.js", "styles.css", "favicon.svg")}
    js = data["app.js"].decode("utf-8")
    js = js.replace('const FRONTEND_VERSION="warehouse-family-feedback-research.r4";',
                    'const FRONTEND_VERSION="warehouse-alignment-online.r4";')
    js = re.sub(r'const PENDING_KEY="[^"]+", LANGUAGE_KEY="[^"]+";',
        'const PENDING_KEY="warehouse-alignment-online.r4.pending", LANGUAGE_KEY="warehouse-alignment-online.r4.lang";', js, count=1)
    js = js.replace('localStudy:"本地预实验 · 已核验"',
                    'localStudy:"线上试玩 · 已核验"')
    js = js.replace('localStudyMessage:"本地预实验 · 已核验。当前记录不作为正式研究样本。"',
                    'localStudyMessage:"线上试玩已核验。当前免费实例重启后可能丢失记录，不作为正式研究样本。"')
    js = js.replace('localStudy:"Local pilot · Verified"',
                    'localStudy:"Online demo · Verified"')
    js = js.replace('localStudyMessage:"Local pilot · Verified. These records are not formal research samples."',
                    'localStudyMessage:"Verified online demo. The free instance may lose records after a restart; these are not formal research samples."')
    js = js.replace('request("/api/command",{method:"POST"',
                    'request("/api/study/command",{method:"POST"')
    js = js.replace(
        'const consentStage=isStudy && p==="consent";const entryAllowed=!isStudy || consentStage;',
        'const consentStage=isStudy && p==="consent",registrationStage=p==="registration" || consentStage;const entryAllowed=!isStudy || registrationStage;',
    )
    js = js.replace('document.querySelector(".agreement").hidden=!consentStage;',
                    'document.querySelector(".agreement").hidden=!registrationStage;')
    js = js.replace('$("consentText").hidden=!consentStage;',
                    '$("consentText").hidden=!registrationStage;')
    js = js.replace(
        '$("startButton").textContent=consentStage?tr("startStudy"):(ui.language==="zh"?"登记用户 ID":"Register participant ID");',
        '$("startButton").textContent=registrationStage?tr("startStudy"):(ui.language==="zh"?"登记用户 ID":"Register participant ID");',
    )
    js = js.replace(
        '    const id=$("participantInput").value.trim();',
        '    if(!$("consentInput").checked){ui.error="consentRequired";render();return;}\n    const id=$("participantInput").value.trim();',
    )
    js = js.replace(
        'void execute({kind:"start",mode:"study",participant_id:id});',
        'void execute({kind:"start",mode:"study",participant_id:id,consent:true});',
    )
    required = ("registrationStage", "participant_id:id,consent:true",
                'request("/api/study/command",{method:"POST"',
                "线上试玩 · 已核验", "Online demo · Verified")
    if any(value not in js for value in required):
        raise ValueError("online frontend transformation anchor changed")
    data["app.js"] = js.encode("utf-8")
    data["styles.css"] += (b"\n#freeplayTab,#freeplaySetup,#savedRunsField,#feedbackPanel,.technical{display:none!important}"
                            b" .agreement[hidden]{display:none!important}\n")
    return data


class CommandError(ValueError):
    def __init__(self, reason, status=400, view=None):
        super().__init__(reason)
        self.reason, self.status, self.view = reason, status, view


_SCHEMA = """PRAGMA journal_mode=WAL;
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,version INTEGER NOT NULL DEFAULT 0,active_run TEXT,
 participant_id TEXT,participant_key TEXT UNIQUE,position INTEGER UNIQUE,condition TEXT,task_order TEXT,
 mode TEXT NOT NULL DEFAULT 'enrollment',stage TEXT NOT NULL DEFAULT 'registration',round_index INTEGER NOT NULL DEFAULT 0,
 questionnaire TEXT NOT NULL DEFAULT '{}',questionnaire_scores TEXT,namespace TEXT NOT NULL,study_signature TEXT,
 questionnaire_bank_signature TEXT,consented TEXT);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,scenario_id TEXT NOT NULL,signature TEXT NOT NULL,
 stage TEXT NOT NULL,round_index INTEGER NOT NULL,snapshot TEXT NOT NULL,metrics TEXT NOT NULL,ended INTEGER NOT NULL DEFAULT 0,
 created TEXT NOT NULL,end_reason TEXT,provenance TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS frames(run_id TEXT NOT NULL,frame INTEGER NOT NULL,public TEXT NOT NULL,internal TEXT NOT NULL,
 PRIMARY KEY(run_id,frame));
CREATE TABLE IF NOT EXISTS operations(session_id TEXT NOT NULL,id TEXT NOT NULL,digest TEXT NOT NULL,version INTEGER NOT NULL,
 kind TEXT NOT NULL,PRIMARY KEY(session_id,id));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT NOT NULL,run_id TEXT,kind TEXT NOT NULL,
 payload TEXT NOT NULL,created TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,run_id TEXT NOT NULL,frame INTEGER NOT NULL,
 stage TEXT NOT NULL,question TEXT NOT NULL,language TEXT NOT NULL,status TEXT NOT NULL,answer TEXT,shown TEXT,created TEXT NOT NULL,
 focus TEXT NOT NULL DEFAULT 'executed',evidence_detail TEXT);
CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,allocation TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
COMMIT;
"""


class OnlineAlignmentStudyStore:
    """Authoritative study state backed by one identity-bound SQLite file."""

    def __init__(self, context, *, database=DEFAULT_DATABASE, storage_mode="auto"):
        self._closed = False
        self._context = context
        if not callable(getattr(context, "close", None)):
            raise ValueError("online release context must own its artifact lifetime")
        self.runtime = context.runtime
        self.explainer = context.explainer
        self.question_bank = context.question_bank
        self.scenarios = deepcopy(context.scenarios)
        for name in ("environment", "from_snapshot", "step"):
            if not callable(getattr(self.runtime, name, None)):
                raise ValueError("online runtime contract is incomplete")
        if callable(getattr(self.runtime, "verify_binding", None)):
            self.runtime.verify_binding()
        if callable(getattr(self.explainer, "_assert_current", None)):
            self.explainer._assert_current(self.runtime)
        release = deepcopy(context.release)
        if (not isinstance(release, dict) or release.get("test_fixture") is True
                or any(release.get(k) is not True for k in ("model_ready", "explanation_ready", "study_ready"))
                or release.get("formal_ready") is not False):
            raise ValueError("a genuine technically verified local-pilot release is required")
        play = self.scenarios.get("splits", {}).get("play")
        if not isinstance(play, list) or len(play) < 7 or len({s.get("id") for s in play}) != len(play):
            raise ValueError("practice and six unique online study scenes are required")
        public_items = self.question_bank.public_items()
        if not isinstance(public_items, list) or len(public_items) != 8 or len({x.get("id") for x in public_items}) != 8:
            raise ValueError("the frozen eight-item prediction bank is required")
        raw_database = Path(database).expanduser().absolute()
        if raw_database.is_symlink():
            raise ValueError("database cannot be a symlink")
        requested = raw_database.resolve(strict=False)
        root = Path(getattr(context, "root", ROOT)).resolve()
        if requested == root or root in requested.parents:
            raise ValueError("database must be outside the immutable release")
        mode = storage_mode
        if mode == "auto":
            mode = "persistent" if str(requested).startswith("/var/data/") else "ephemeral"
        if mode not in ("persistent", "ephemeral"):
            raise ValueError("storage_mode must be auto, persistent, or ephemeral")
        self.data_persistent = mode == "persistent"
        self.database = requested
        self.namespace, self.cookie_name = NAMESPACE, COOKIE
        self.web_assets = _assets()
        self.asset_sha256 = {k: sha256(v).hexdigest() for k, v in self.web_assets.items()}
        self.signature = _digest({"version": VERSION, "context": str(context.signature),
            "runtime": str(self.runtime.signature), "explainer": str(self.explainer.signature),
            "bank": str(self.question_bank.signature), "scenarios": _digest(self.scenarios),
            "assets": self.asset_sha256})
        message = ({
            "zh": "线上试玩服务已核验；当前 Render 实例没有持久磁盘，服务重启可能丢失记录，因此不能作为正式持久化人类实验。",
            "en": "Verified online demo. This Render instance has no persistent disk; a restart may erase records, so it is not a formally persistent human study.",
        } if not self.data_persistent else {
            "zh": "线上预实验服务已核验；记录写入持久存储。本版本仍不是正式研究版本。",
            "en": "Verified online pilot with persistent storage. This is still not a formal-study release.",
        })
        self.release = {**release, "status": "online_demo_ephemeral" if not self.data_persistent else "online_pilot_persistent",
            "namespace": NAMESPACE, "online": True, "online_demo": not self.data_persistent,
            "data_persistent": self.data_persistent, "formal_ready": False,
            "human_explanation_effect_validated": False, "message": message}
        self.provenance = {**deepcopy(context.provenance), "service_family": SERVICE_FAMILY,
            "service_version": VERSION, "run_signature": self.signature, "namespace": NAMESPACE,
            "data_persistent": self.data_persistent, "formal_ready": False,
            "release": deepcopy(self.release)}
        template = self.runtime.environment(play[0])
        self._map = _public_map(template)
        self.horizon = int(_attr(getattr(self.runtime, "config", {}), "horizon", 120))
        self._initialize_database()
        self.scheduled, self.schedule_lock = set(), Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="online-alignment-explanation")

    def _initialize_database(self):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        exists = self.database.exists()
        if exists and not self.database.is_file():
            raise ValueError("database must be a regular file")
        if not exists:
            fd = os.open(self.database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        with closing(self.connect()) as db:
            if exists:
                try:
                    values = {r["key"]: json.loads(r["value"]) for r in db.execute(
                        "SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')")}
                except sqlite3.Error as exc:
                    raise ValueError("existing database lacks the online service identity") from exc
                if values != {"service_family": SERVICE_FAMILY, "namespace": NAMESPACE}:
                    raise ValueError("database family or namespace differs")
            db.executescript(_SCHEMA)
            question_columns = {row[1] for row in db.execute("PRAGMA table_info(questions)")}
            if "evidence_detail" not in question_columns:
                db.execute("ALTER TABLE questions ADD COLUMN evidence_detail TEXT")
            if not exists:
                db.executemany("INSERT INTO metadata VALUES(?,?)", [
                    ("service_family", _canonical(SERVICE_FAMILY)), ("namespace", _canonical(NAMESPACE))])
            db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)",
                ("service_context:" + self.signature, _canonical(self.provenance)))
            db.execute("UPDATE questions SET status='pending' WHERE status='running'")

    def connect(self):
        db = sqlite3.connect(self.database, isolation_level=None, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.executor.shutdown(wait=True)
        finally:
            self._context.close()

    def session(self, candidate=None):
        with closing(self.connect()) as db:
            if candidate:
                existing = db.execute("SELECT * FROM sessions WHERE id=?", (candidate,)).fetchone()
                if existing is not None and not self._mismatch(existing):
                    return candidate
                # A deployed runtime change starts a fresh browser session while
                # retaining the old session, runs, frames, and answers as audit
                # records.  This also prevents an old pending request from being
                # replayed into the newly bound experiment version.
            sid = uuid4().hex
            db.execute("INSERT INTO sessions(id,namespace) VALUES(?,?)", (sid, NAMESPACE))
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

    def _mismatch(self, session):
        return session["mode"] == "study" and session["study_signature"] != self.signature

    def _can_explain(self, db, session):
        if (self._closed or self._mismatch(session) or session["mode"] != "study"
                or session["stage"] != "task1" or session["condition"] != "A"
                or not session["active_run"]):
            return False
        run = self._run(db, session["id"])
        return run["signature"] == self.signature and run["stage"] == "task1"

    def _metrics(self, env, old=None, outcome=None):
        state = _state(env)
        result = dict(old or {"ai_waits": 0, "ai_blocked": 0, "overrides": 0})
        agents = list(_attr(state, "agents", ()))
        result.update(steps=int(_attr(state, "frame", 0)),
            deliveries=int(_attr(state, "total_deliveries", 0)),
            score=float(_attr(state, "user_score", 0)), native_score=float(_attr(state, "user_score", 0)),
            legacy_score=None,
            collisions=int(_attr(state, "robot_collision_events", _attr(state, "collision_count", 0))),
            shutdowns=int(_attr(state, "shutdown_count", 0)),
            individual_deliveries=[int(_attr(a, "deliveries_completed", 0)) for a in agents])
        if outcome:
            submitted = outcome.get("submitted_actions", outcome.get("policy_actions", {}))
            executed = outcome.get("executed_actions", submitted)
            ai_submitted, ai_executed = submitted.get("robot_2"), executed.get("robot_2")
            result["ai_waits"] = int(result.get("ai_waits", 0)) + int(ai_submitted == "WAIT")
            result["ai_blocked"] = int(result.get("ai_blocked", 0)) + int(ai_submitted not in (None, "WAIT") and ai_executed == "WAIT")
        return result

    def _frame(self, env, metrics, outcome=None):
        actions = (outcome or {}).get("executed_actions", {})
        state = _public_state(env)
        state["public_feedback"] = _public_history(env, outcome)
        return {"state": state, "metrics": deepcopy(metrics), "actions": deepcopy(actions)}

    def _initial_record(self, env, scene):
        snapshot = _plain(env.snapshot())
        return {"after": snapshot, "runtime_signature": str(self.runtime.signature),
            "run_signature": self.signature, "scenario_sha256": _digest(scene),
            "actor_sha256": str(getattr(self.runtime, "actor_sha256", ""))}

    def _start_run(self, db, sid, scene_index):
        session = self._session(db, sid)
        scene = self.scenarios["splits"]["play"][scene_index]
        env = self.runtime.environment(scene)
        snapshot = _plain(env.snapshot())
        rid, metrics = uuid4().hex, self._metrics(env)
        provenance = {**self.provenance, "scenario_sha256": _digest(scene), "scene_index": scene_index}
        db.execute("INSERT INTO runs(id,session_id,scenario_id,signature,stage,round_index,snapshot,metrics,created,provenance) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (rid, sid, scene["id"], self.signature, session["stage"], session["round_index"],
             _canonical(snapshot), _canonical(metrics), _utcnow(), _canonical(provenance)))
        db.execute("INSERT INTO frames VALUES(?,?,?,?)", (rid, int(_attr(_state(env), "frame", 0)),
            _canonical(self._frame(env, metrics)), _canonical(self._initial_record(env, scene))))
        db.execute("UPDATE sessions SET active_run=? WHERE id=?", (rid, sid))
        return rid

    def _questionnaire(self, session):
        blocked = self._mismatch(session)
        bound = session["questionnaire_bank_signature"] == str(self.question_bank.signature)
        visible = session["stage"] == "questionnaire" and bound and not blocked
        return {"items": _questionnaire_items() + (deepcopy(self.question_bank.public_items()) if visible else []),
            "draft": json.loads(session["questionnaire"]), "blocked": blocked or (session["mode"] == "study" and not bound),
            "complete_prediction_bank_available": True,
            "scope": "four_next_four_wait_three_plus_self_report",
            "bank": {"status": "version_mismatch" if blocked or not bound else "candidate_ready",
                "available": bool(bound), "formal_ready": False, "bound_at_enrollment": bool(bound)}}

    def _view(self, db, sid):
        session = self._session(db, sid)
        permitted = self._can_explain(db, session)
        result = {"session_id": _digest({"service": self.signature, "session": sid}),
            "version": session["version"], "run_id": session["active_run"],
            "release": deepcopy(self.release), "study_allowed": not self._mismatch(session),
            "study_version_mismatch": self._mismatch(session), "verification_only": False,
            "verification_flow_allowed": False, "study_only": True,
            "enrollment": {"mode": "online_demo" if not self.data_persistent else "online_pilot",
                "enabled": True, "id_pattern": _ID.pattern, "formal_ready": False,
                "data_persistent": self.data_persistent},
            "flow": {"mode": session["mode"], "stage": session["stage"],
                "round_index": session["round_index"] + 1,
                "round_count": 3 if session["stage"] in ("task1", "task2") else 1,
                "participant_id": session["participant_id"],
                "consent_text": {"zh": "本线上预实验记录用户 ID、操作、问答和问卷。当前免费 Render 实例的本地记录可能在重启时丢失，因此本服务不是正式持久化实验。请使用不含姓名、邮箱或电话号码的研究编号。",
                    "en": "This online pilot records the study ID, actions, questions, and questionnaire. Local records on the current free Render instance may be lost after a restart, so this is not a formally persistent study. Use an ID without a name, email, or telephone number."}},
            "map": deepcopy(self._map), "horizon": self.horizon,
            "explain_allowed": permitted, "answers": [], "runs": [], "history_count": 0,
            "state": None, "metrics": {}, "ended": False,
            "play_scene_count": len(self.scenarios["splits"]["play"]),
            "questionnaire": self._questionnaire(session), "provenance": {"service_version": VERSION,
                "data_persistent": self.data_persistent, "formal_ready": False}}
        if session["active_run"]:
            run = self._run(db, sid)
            row = db.execute("SELECT public FROM frames WHERE run_id=? ORDER BY frame DESC LIMIT 1", (run["id"],)).fetchone()
            if row is None:
                raise ValueError("run has no authoritative frame")
            result.update(json.loads(row["public"]))
            scene_index = next(i for i, scene in enumerate(self.scenarios["splits"]["play"])
                               if scene["id"] == run["scenario_id"])
            result.update(ended=bool(run["ended"]), done=bool(run["ended"]), end_reason=run["end_reason"],
                version_mismatch=run["signature"] != self.signature, seed=scene_index,
                history_count=db.execute("SELECT count(*) FROM frames WHERE run_id=?", (run["id"],)).fetchone()[0])
            if permitted:
                result["answers"] = [{"id": q["id"], "status": q["status"], "frame": q["frame"],
                    "question": q["question"], "text": q["answer"] or "", "run_id": q["run_id"],
                    "focus": q["focus"], "evidence_detail": q["evidence_detail"] or "",
                    "sources": ["frozen NN", "validated policy program", "isolated counterfactual"]}
                    for q in db.execute("SELECT * FROM questions WHERE session_id=? AND run_id=? ORDER BY created,id", (sid, run["id"]))]
        kinds = []
        if session["mode"] == "enrollment":
            kinds = ["start"]
        elif session["stage"] == "consent":
            kinds = ["next"]
        elif session["stage"] == "questionnaire":
            kinds = ["questionnaire"]
        elif session["active_run"] and session["stage"] in ("practice", "task1", "task2"):
            if result["ended"]:
                kinds = ["next"]
            elif not result.get("version_mismatch"):
                kinds = ["action", "end"]
        if permitted:
            kinds += ["question", "answer_seen"]
        result["allowed_kinds"] = [] if self._mismatch(session) else list(dict.fromkeys(kinds))
        return result

    def view(self, sid):
        self.schedule_pending(sid)
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            return self._view(db, sid)

    def history(self, sid, run_id=None):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            session = self._session(db, sid)
            if self._mismatch(session):
                raise CommandError("study_version_changed", 409)
            run = self._run(db, sid)
            if run_id is not None and run_id != run["id"]:
                raise CommandError("history_current_run_only", 403)
            return {"run_id": run["id"], "version": session["version"], "map": deepcopy(self._map),
                "frames": [json.loads(row[0]) for row in db.execute(
                    "SELECT public FROM frames WHERE run_id=? ORDER BY frame", (run["id"],))]}

    def _record_operation(self, db, sid, op, request_hash, kind, payload):
        db.execute("UPDATE sessions SET version=version+1 WHERE id=?", (sid,))
        version = self._session(db, sid)["version"]
        db.execute("INSERT INTO operations VALUES(?,?,?,?,?)", (sid, op, request_hash, version, kind))
        current = self._session(db, sid)
        db.execute("INSERT INTO events(session_id,run_id,kind,payload,created) VALUES(?,?,?,?,?)",
            (sid, current["active_run"], kind, _canonical(payload), _utcnow()))

    def _register(self, db, sid, payload):
        if payload.get("mode") != "study":
            raise CommandError("study_only_no_freeplay", 403)
        if payload.get("consent") is not True:
            raise CommandError("explicit_consent_required")
        session = self._session(db, sid)
        if session["mode"] != "enrollment":
            raise CommandError("study_cannot_restart_or_switch_mode", 403)
        participant = payload.get("participant_id", "")
        if not isinstance(participant, str) or not _ID.fullmatch(participant.strip()):
            raise CommandError("invalid_participant_id")
        participant = participant.strip()
        if db.execute("SELECT 1 FROM sessions WHERE participant_key=?", (participant.lower(),)).fetchone():
            raise CommandError("participant_id_taken", 409)
        position = db.execute("SELECT count(*) FROM sessions WHERE position IS NOT NULL").fetchone()[0]
        block_id = position // 4
        block = db.execute("SELECT allocation FROM blocks WHERE id=?", (block_id,)).fetchone()
        if block is None:
            cells = [["A","XY"],["A","YX"],["B","XY"],["B","YX"]]
            random.SystemRandom().shuffle(cells)
            db.execute("INSERT INTO blocks VALUES(?,?)", (block_id, _canonical(cells)))
        else:
            cells = json.loads(block[0])
        condition, task_order = cells[position % 4]
        consented = _utcnow()
        db.execute("UPDATE sessions SET mode='study',stage='practice',participant_id=?,participant_key=?,position=?,condition=?,task_order=?,round_index=0,active_run=NULL,questionnaire_bank_signature=?,study_signature=?,consented=? WHERE id=?",
            (participant, participant.lower(), position, condition, task_order,
             str(self.question_bank.signature), self.signature, consented, sid))
        self._start_run(db, sid, 0)

    def command(self, sid, payload):
        if not isinstance(payload, dict):
            raise CommandError("invalid_body")
        op, kind = payload.get("operation_id"), payload.get("kind")
        if not isinstance(op, str) or not 1 <= len(op) <= 128:
            raise CommandError("operation_id_required")
        request_hash = _digest(payload)
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                session = self._session(db, sid)
                old = db.execute("SELECT * FROM operations WHERE session_id=? AND id=?", (sid, op)).fetchone()
                if old:
                    if old["digest"] != request_hash:
                        raise CommandError("operation_id_reused", 409, self._view(db, sid))
                    if self._mismatch(session):
                        raise CommandError("study_version_changed", 409, self._view(db, sid))
                    result = self._view(db, sid)
                    db.commit()
                    return result
                if type(payload.get("expected_version")) is not int or payload["expected_version"] != session["version"]:
                    raise CommandError("state_version_conflict", 409, self._view(db, sid))
                if self._mismatch(session):
                    raise CommandError("study_version_changed", 409, self._view(db, sid))
                if kind == "start":
                    self._register(db, sid, payload)
                elif kind == "next":
                    self._next(db, session, payload)
                elif kind == "action":
                    self._action(db, session, payload)
                elif kind == "end":
                    if session["stage"] not in ("practice", "task1", "task2"):
                        raise CommandError("end_not_allowed_in_stage", 403)
                    run = self._run(db, sid)
                    if run["ended"]:
                        raise CommandError("round_ended", 409)
                    db.execute("UPDATE runs SET ended=1,end_reason='participant_ended' WHERE id=?", (run["id"],))
                elif kind == "question":
                    self._question(db, session, payload)
                elif kind == "answer_seen":
                    if not self._can_explain(db, session):
                        raise CommandError("explanation_forbidden", 403)
                    row = db.execute("SELECT id FROM questions WHERE id=? AND session_id=? AND run_id=?",
                        (payload.get("answer_id"), sid, session["active_run"])).fetchone()
                    if row is None:
                        raise CommandError("answer_not_found", 404)
                    db.execute("UPDATE questions SET shown=coalesce(shown,?) WHERE id=?", (_utcnow(), row["id"]))
                elif kind == "questionnaire":
                    self._save_questionnaire(db, session, payload)
                else:
                    raise CommandError("unknown_command")
                self._record_operation(db, sid, op, request_hash, str(kind), payload)
                result = self._view(db, sid)
                db.commit()
            except BaseException:
                db.rollback()
                raise
        self.schedule_pending(sid)
        return result

    def _action(self, db, session, payload):
        if session["stage"] not in ("practice", "task1", "task2"):
            raise CommandError("actions_not_allowed_in_stage", 403)
        if payload.get("action") not in _ACTIONS:
            raise CommandError("invalid_action")
        run = self._run(db, session["id"])
        if run["signature"] != self.signature:
            raise CommandError("runtime_version_changed", 409)
        if run["ended"]:
            raise CommandError("round_ended", 409)
        env = self.runtime.from_snapshot(json.loads(run["snapshot"]))
        outcome = _plain(self.runtime.step(env, payload["action"]))
        metrics = self._metrics(env, json.loads(run["metrics"]), outcome)
        snapshot = _plain(env.snapshot())
        done = bool(outcome.get("done", _attr(_state(env), "terminated", False) or _attr(_state(env), "truncated", False)))
        reason = outcome.get("terminal_reason", _attr(_state(env), "terminal_reason")) if done else None
        db.execute("UPDATE runs SET snapshot=?,metrics=?,ended=?,end_reason=? WHERE id=?",
            (_canonical(snapshot), _canonical(metrics), int(done), reason, run["id"]))
        frame = int(_attr(_state(env), "frame", 0))
        record = dict(outcome)
        record.setdefault("after", snapshot)
        record.setdefault("runtime_signature", str(self.runtime.signature))
        db.execute("INSERT INTO frames VALUES(?,?,?,?)", (run["id"], frame,
            _canonical(self._frame(env, metrics, outcome)), _canonical(record)))

    def _next(self, db, session, payload):
        sid, stage, index = session["id"], session["stage"], session["round_index"]
        if session["mode"] != "study":
            raise CommandError("next_requires_study")
        if stage == "consent":
            if payload.get("consent") is not True:
                raise CommandError("explicit_consent_required")
            db.execute("UPDATE sessions SET stage='practice',consented=?,round_index=0 WHERE id=?", (_utcnow(), sid))
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
            # Task 2 cannot expose or create explanations, but completed Task 1
            # answers are research records and must remain recoverable from the
            # database/export.  The view and command gates below enforce the
            # participant boundary; only unfinished work is retired here.
            db.execute("""UPDATE questions SET status='expired'
                WHERE session_id=? AND status IN ('pending','running')""", (sid,))
        else:
            stage, index = "questionnaire", 0
        db.execute("UPDATE sessions SET stage=?,round_index=?,active_run=NULL WHERE id=?", (stage, index, sid))
        if stage in ("task1", "task2"):
            bank = 0 if (stage == "task1") == (session["task_order"] == "XY") else 1
            self._start_run(db, sid, 1 + bank * 3 + index)

    def _question(self, db, session, payload):
        if not self._can_explain(db, session):
            raise CommandError("explanation_forbidden", 403)
        run = self._run(db, session["id"], payload.get("run_id"))
        if run["id"] != session["active_run"] or run["stage"] != "task1":
            raise CommandError("explanation_round_mismatch", 403)
        frame = payload.get("frame")
        if type(frame) is not int or not db.execute("SELECT 1 FROM frames WHERE run_id=? AND frame=?", (run["id"], frame)).fetchone():
            raise CommandError("frame_not_found", 404)
        question = payload.get("question")
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
            raise CommandError("invalid_question")
        language, focus = payload.get("language", "zh"), payload.get("focus", "executed")
        if language not in ("zh", "en") or focus not in ("executed", "next"):
            raise CommandError("invalid_question_reference")
        db.execute("""INSERT INTO questions
            (id,session_id,run_id,frame,stage,question,language,status,answer,shown,created,focus,evidence_detail)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (uuid4().hex, session["id"], run["id"], frame, "task1", question.strip(), language,
             "pending", None, None, _utcnow(), focus, None))

    def _save_questionnaire(self, db, session, payload):
        if session["stage"] != "questionnaire":
            raise CommandError("questionnaire_not_allowed", 403)
        answers = payload.get("answers", {})
        if not isinstance(answers, dict) or len(_canonical(answers)) > 10_000:
            raise CommandError("invalid_questionnaire")
        draft = json.loads(session["questionnaire"])
        draft.update(answers)
        items = _questionnaire_items() + deepcopy(self.question_bank.public_items())
        by_id = {item["id"]: item for item in items}
        if set(draft) - set(by_id):
            raise CommandError("unknown_questionnaire_item")
        for key, value in draft.items():
            if value is None:
                continue
            item = by_id[key]
            options = item.get("options", [])
            allowed = {str(x.get("value", x.get("id"))) if isinstance(x, dict) else str(x) for x in options}
            if item.get("type") == "scale":
                valid = type(value) is int and str(value) in allowed
            else:
                valid = isinstance(value, str) and value in allowed
            if not valid:
                raise CommandError("invalid_questionnaire_value")
        submit = payload.get("submit", False)
        if type(submit) is not bool:
            raise CommandError("invalid_questionnaire_submit")
        required = {x["id"] for x in items if x.get("required", True)}
        if submit and any(draft.get(key) in (None, "") for key in required):
            raise CommandError("questionnaire_incomplete")
        scores = self.question_bank.grade(draft) if submit else None
        db.execute("UPDATE sessions SET questionnaire=?,questionnaire_scores=?,stage=? WHERE id=?",
            (_canonical(draft), _canonical(scores) if scores is not None else None,
             "completed" if submit else "questionnaire", session["id"]))

    def schedule_pending(self, sid):
        if self._closed:
            return
        with closing(self.connect()) as db:
            rows = list(db.execute("SELECT id FROM questions WHERE session_id=? AND status='pending'", (sid,)))
        for row in rows:
            with self.schedule_lock:
                if row["id"] in self.scheduled:
                    continue
                self.scheduled.add(row["id"])
            self.executor.submit(self._answer, row["id"])

    def _answer(self, qid):
        try:
            with closing(self.connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                q = db.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
                if q is None:
                    db.rollback(); return
                session = self._session(db, q["session_id"])
                if not self._can_explain(db, session) or session["active_run"] != q["run_id"]:
                    db.execute("UPDATE questions SET status='expired',answer=NULL,evidence_detail=NULL WHERE id=?", (qid,))
                    db.commit(); return
                row = db.execute("SELECT internal FROM frames WHERE run_id=? AND frame=?", (q["run_id"], q["frame"])).fetchone()
                db.execute("UPDATE questions SET status='running' WHERE id=?", (qid,))
                db.commit()
            record = json.loads(row[0])
            answer_result = self.explainer.answer(dict(q), record, self.runtime)
            evidence_detail = ""
            if isinstance(answer_result, dict):
                evidence_detail = answer_result.get("evidence_detail", "")
                answer = answer_result.get("answer", answer_result.get("text"))
            else:
                answer = answer_result
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("explainer returned no verified answer")
            if not isinstance(evidence_detail, str):
                raise ValueError("explainer returned invalid evidence detail")
            with closing(self.connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                session = self._session(db, q["session_id"])
                permitted = self._can_explain(db, session) and session["active_run"] == q["run_id"]
                db.execute("UPDATE questions SET status=?,answer=?,evidence_detail=? WHERE id=?",
                    ("complete" if permitted else "expired", answer if permitted else None,
                     evidence_detail if permitted else None, qid))
                db.commit()
        except Exception:
            with closing(self.connect()) as db:
                db.execute("UPDATE questions SET status='failed',answer=NULL,evidence_detail=NULL WHERE id=?", (qid,))
        finally:
            with self.schedule_lock:
                self.scheduled.discard(qid)


def handler_class(store, *, public_origin=DEFAULT_ORIGIN):
    expected_origin = public_origin.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def _headers(self, status, length, content_type, sid=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            if sid:
                self.send_header("Set-Cookie", f"{store.cookie_name}={sid}; Path=/; HttpOnly; Secure; SameSite=Strict")
            self.end_headers()

        def reply(self, status, payload, sid=None, content_type="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else _canonical(payload).encode("utf-8")
            self._headers(status, len(body), content_type, sid)
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
            if path in ("/health", "/api/health"):
                self.reply(200, {"status":"ok", "service":SERVICE_FAMILY, "version":VERSION,
                    "data_persistent":store.data_persistent, "formal_ready":False})
                return
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/assets/favicon.svg": ("favicon.svg", "image/svg+xml")}
            if path in assets:
                name, mime = assets[path]
                self.reply(200, store.web_assets[name], content_type=mime)
                return
            sid = self.sid()
            try:
                if path == "/api/view":
                    result = store.view(sid)
                elif path == "/api/history":
                    run_id = parse_qs(urlparse(self.path).query).get("run_id", [None])[0]
                    result = store.history(sid, run_id)
                else:
                    raise CommandError("not_found", 404)
                self.reply(200, result, sid)
            except CommandError as exc:
                self.reply(exc.status, {"error":exc.reason, "view":exc.view}, sid)

        def do_POST(self):
            sid = self.sid()
            try:
                if urlparse(self.path).path not in ("/api/study/command", "/api/command"):
                    raise CommandError("not_found", 404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise CommandError("invalid_body_size", 413)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise CommandError("invalid_body")
                origin = self.headers.get("Origin")
                if origin != expected_origin or not origin.startswith("https://"):
                    raise CommandError("https_origin_required", 403)
                self.reply(200, store.command(sid, payload), sid)
            except CommandError as exc:
                self.reply(exc.status, {"error":exc.reason, "view":exc.view}, sid)
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                self.reply(400, {"error":"invalid_request"}, sid)

    return Handler


def load_online_context(*, expected_package_sha256, expected_manifest_sha256,
        package_path=None, base64_path=None):
    module = importlib.import_module("ui.warehouse_alignment_online_release")
    return module.load_online_release(expected_package_sha256=expected_package_sha256,
        expected_manifest_sha256=expected_manifest_sha256, package_path=package_path,
        base64_path=base64_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--base64", type=Path, default=os.environ.get("WAREHOUSE_RELEASE_BASE64"))
    parser.add_argument("--expected-package-sha256", default=os.environ.get("WAREHOUSE_RELEASE_PACKAGE_SHA256"), required=False)
    parser.add_argument("--expected-manifest-sha256", default=os.environ.get("WAREHOUSE_RELEASE_MANIFEST_SHA256"), required=False)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--storage-mode", choices=("auto","persistent","ephemeral"), default=os.environ.get("WAREHOUSE_STORAGE_MODE", "auto"))
    parser.add_argument("--public-origin", default=os.environ.get("WAREHOUSE_PUBLIC_ORIGIN", DEFAULT_ORIGIN))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", DEFAULT_PORT)))
    args = parser.parse_args(argv)
    if not args.expected_package_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.expected_package_sha256):
        parser.error("expected package SHA256 is required")
    if not args.expected_manifest_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.expected_manifest_sha256):
        parser.error("expected manifest SHA256 is required")
    if bool(args.package) == bool(args.base64):
        parser.error("provide exactly one of --package or --base64")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not args.public_origin.startswith("https://"):
        parser.error("public origin must use HTTPS")
    context = load_online_context(expected_package_sha256=args.expected_package_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        package_path=args.package, base64_path=args.base64)
    store = OnlineAlignmentStudyStore(context, database=args.database, storage_mode=args.storage_mode)
    server = ThreadingHTTPServer((args.host, args.port), handler_class(store, public_origin=args.public_origin))
    server.daemon_threads = False
    handlers = {}
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop)
        print(f"PolicyLens Warehouse online: {args.public_origin}", flush=True)
        server.serve_forever(poll_interval=.25)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        store.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
