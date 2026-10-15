"""Observed197 Store component, deliberately unavailable as a production service.

The explicit fixture entry exercises real public-history dynamics and durable A/B
flow. It grants no model, explanation, prediction-bank or study qualification.
No HTTP listener or production release-loader is provided by this module.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from threading import Lock
from uuid import uuid4

from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.warehouse_public_history_runtime import PublicHistoryRuntime, runtime_sources
from ui import warehouse_native_server as inherited
from ui.warehouse_view import serialize_warehouse_state, warehouse_map_payload

VERSION = "warehouse-native-observed197-store-component.v1"
SERVICE_FAMILY = "warehouse_public_history_verification"
COOKIE = "warehouse_public_history_verification_v1"
CommandError = inherited.CommandError
questionnaire_items = inherited.questionnaire_items


def service_sources():
    return {**inherited.service_sources(), **runtime_sources(),
            str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__))}


def _database_identity(database):
    if database.is_symlink():
        raise ValueError("Database cannot be a symlink")
    if not database.exists():
        return False
    if not database.is_file():
        raise ValueError("Database must be a regular file")
    try:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            rows = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')"))
        if (json.loads(rows.get("service_family", "null")) != SERVICE_FAMILY
                or json.loads(rows.get("namespace", "null")) != "verification"):
            raise ValueError("Existing database is not an observed197 verification database")
    except (sqlite3.Error, json.JSONDecodeError) as error:
        raise ValueError("Existing database has no verified observed197 identity") from error
    return True


class PublicHistoryStore(inherited.NativeStore):
    """An isolated component fixture; never a substitute for a qualified release.

    Snapshots, not visible-state reconstructions, are the restoration contract.
    Questionnaire previews are fixture-only saved snapshots, not a prediction
    bank. They have no answer key and are never scored as research measurements.
    """
    def __init__(self, database, runtime, scenarios, *, verification_fixture=False,
                 explainer=None, questionnaire_previews=None):
        # This check precedes Actor access, source reads and database creation.
        if verification_fixture is not True:
            raise ValueError("public_history_qualified_release_unavailable")
        if type(runtime) is not PublicHistoryRuntime or runtime.test_fixture is not True:
            raise ValueError("Explicit genuine synthetic observed197 runtime required")
        runtime.verify_binding()
        if explainer is not None:
            if getattr(explainer, "test_fixture", None) is not True or not callable(getattr(explainer, "answer", None)):
                raise ValueError("Only an explicit fixture explanation component can be injected")
            if not isinstance(getattr(explainer, "signature", None), str) or not explainer.signature:
                raise ValueError("Fixture explanation needs a stable binding")
            checker = getattr(explainer, "_assert_current", None)
            if callable(checker):
                checker(runtime)
        scenes = deepcopy(scenarios)
        play = scenes.get("splits", {}).get("play") if isinstance(scenes, dict) else None
        if not isinstance(play, list) or not play or len({x.get("id") for x in play}) != len(play):
            raise ValueError("Unique registered play scenes are required")
        # Validate raw scene starts once; no reset or step from guessed history.
        template = None
        for scene in play:
            env = runtime.environment(scene)
            if env.state.frame != 0:
                raise ValueError("Store play scenes must begin at frame zero")
            template = env
        self.runtime, self.scenarios = runtime, scenes
        self._map = warehouse_map_payload(template.layout)
        self._previews = deepcopy(questionnaire_previews or {})
        if not isinstance(self._previews, dict) or any(not isinstance(k, str) for k in self._previews):
            raise ValueError("Questionnaire preview snapshots must be keyed objects")
        for snapshot in self._previews.values():
            self.runtime.from_snapshot(snapshot)
        requested = Path(database).expanduser().absolute()
        if requested.is_symlink() or requested.parent.resolve() != requested.parent:
            raise ValueError("Database path must not traverse a symlink")
        self.database = requested
        existing = _database_identity(self.database)
        self.explainer, self.question_bank = explainer, None
        self.verification = self.test_fixture = True
        self.namespace, self.cookie_name = "verification", COOKIE
        self.source_binding = service_sources()
        self.signature = digest({"version": VERSION, "runtime": runtime.signature,
            "scenarios": digest(scenes), "sources": self.source_binding,
            "explainer": explainer.signature if explainer else None,
            "fixture_preview_snapshots": digest(self._previews), "test_fixture": True})
        self.release = {"status": "verification_fixture_only", "model_ready": False,
            "explanation_ready": False, "study_ready": False, "formal_ready": False,
            "test_fixture": True, "qualification_evaluated": False,
            "message": {"zh": "仅隔离组件测试，未开放参与者实验。",
                        "en": "Isolated component verification; participant enrollment is unavailable."}}
        self.provenance = {"version": VERSION, "service_family": SERVICE_FAMILY,
            "namespace": self.namespace, "test_fixture": True, "runtime_signature": runtime.signature,
            "actor_sha256": runtime.actor_sha256, "protocol_sha256": runtime.protocol_sha256,
            "scenario_manifest_sha256": digest(scenes), "run_signature": self.signature,
            "service_sources": self.source_binding, "release": deepcopy(self.release)}
        self._closed = False
        self._initialize_database(existing)
        self.scheduled, self.schedule_lock = set(), Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="observed197-fixture-explanation")

    def _permitted(self, session):
        # Technical A/B permission checks do not make the component qualified.
        return self.explainer is not None and not self._study_mismatch(session) and (
            session["mode"] == "freeplay" or (session["stage"] == "task1" and session["condition"] == "A"))

    def _frame(self, env, metrics, actions=None):
        state = serialize_warehouse_state(env.get_state(), selected_agent="robot_2", reveal_policy=False)
        state.pop("human_route_regret_units", None)
        state.pop("policy_hidden", None)
        for agent in state["agents"]:
            agent.pop("proposed_action", None)
            agent.pop("reward", None)
        state["public_feedback"] = deepcopy(env.public_history())
        return {"state": state, "metrics": dict(metrics), "actions": dict(actions or {})}

    def _verified_frame(self, run, row):
        if run["signature"] != self.signature:
            raise CommandError("runtime_version_changed_restart_required", 409)
        record, public = json.loads(row["internal"]), json.loads(row["public"])
        if record.get("runtime_signature") != self.runtime.signature:
            raise ValueError("Stored frame runtime binding differs")
        env = self.runtime.from_snapshot(record["after"])
        if "before" in record:
            previous = self.runtime.from_snapshot(record["before"])
            if previous.state.frame + 1 != env.state.frame:
                raise ValueError("Stored frame transition is not consecutive")
        if env.state.frame != row["frame"]:
            raise ValueError("Stored frame number differs")
        rebuilt = self._frame(env, public["metrics"], public.get("actions"))
        if canonical(rebuilt) != canonical(public):
            raise ValueError("Stored public frame differs from its complete observed snapshot")
        return rebuilt

    def preview(self, snapshot):
        env = self.runtime.from_snapshot(snapshot)
        return {**self._frame(env, self._metrics(env)), "map": deepcopy(self._map),
                "scope": "fixture_snapshot_preview", "formal_ready": False}

    def _questionnaire(self, session):
        result = super()._questionnaire(session)
        result["prediction_bank_available"] = False
        result["fixture_previews"] = ({name: self.preview(snapshot) for name, snapshot in self._previews.items()}
            if session["stage"] == "questionnaire" and not result["blocked"] else {})
        return result

    def history(self, sid, run_id=None):
        with closing(self.connect()) as db:
            db.execute("BEGIN")
            session = self._session(db, sid)
            if self._study_mismatch(session):
                raise CommandError("study_version_changed", 409)
            run = self._run(db, sid)
            if run_id is not None and run_id != run["id"]:
                raise CommandError("history_current_run_only", 403)
            result, previous = [], None
            for row in db.execute("SELECT * FROM frames WHERE run_id=? ORDER BY frame", (run["id"],)):
                public = self._verified_frame(run, row)
                internal = json.loads(row["internal"])
                if previous is not None and canonical(internal.get("before")) != canonical(previous):
                    raise ValueError("Stored history snapshot chain differs")
                previous = internal["after"]
                result.append(public)
            if canonical(previous) != run["snapshot"]:
                raise ValueError("Run head and history head differ")
            return {"run_id": run["id"], "version": session["version"],
                    "map": deepcopy(self._map), "frames": result}

    def _view(self, db, sid):
        session = self._session(db, sid)
        permitted = self._can_explain(db, session)
        result = {"session_id": sid, "version": session["version"], "run_id": session["active_run"],
            "release": self.release, "study_allowed": False, "verification_flow_allowed": not self._study_mismatch(session),
            "study_version_mismatch": self._study_mismatch(session),
            "verification_only": self.verification,
            "flow": {"mode": session["mode"], "stage": session["stage"], "round_index": session["round_index"]+1,
                "round_count": 3 if session["stage"] in ("task1", "task2") else 1, "participant_id": session["participant_id"]},
            "map": deepcopy(self._map), "horizon": self.runtime.config.horizon,
            "explain_allowed": permitted, "answers": [], "runs": [], "history_count": 0,
            "state": None, "metrics": {}, "ended": False, "play_scene_count": len(self.scenarios["splits"]["play"]),
            "questionnaire": self._questionnaire(session)}
        if session["active_run"]:
            run = self._run(db, sid)
            frame = db.execute("SELECT * FROM frames WHERE run_id=? ORDER BY frame DESC LIMIT 1", (run["id"],)).fetchone()
            result.update(self._verified_frame(run, frame))
            current = self.runtime.from_snapshot(json.loads(run["snapshot"]))
            if canonical(current.snapshot()) != canonical(json.loads(frame["internal"])["after"]):
                raise ValueError("Run and frame heads differ")
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
                if not result["version_mismatch"] and self.verification:
                    kinds.append("action")
            elif session["mode"] == "study":
                kinds.append("next")
        if permitted and session["active_run"]:
            kinds += ["question", "answer_seen"]
        if session["stage"] == "questionnaire":
            kinds.append("questionnaire")
        result["allowed_kinds"] = [] if self._study_mismatch(session) else kinds
        return result

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
                    env = self.runtime.from_snapshot(json.loads(run["snapshot"]))
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
            record = json.loads(frame[0])
            if record.get("runtime_signature") != self.runtime.signature:
                raise ValueError("Explanation frame runtime differs")
            env = self.runtime.from_snapshot(record["after"])
            if env.state.frame != q["frame"]:
                raise ValueError("Explanation frame number differs")
            if "before" in record:
                self.runtime.from_snapshot(record["before"])
            else:
                # Enrollment provenance is owned by the Store; the explanation
                # component consumes the original initial-frame runtime shape.
                record = {"after": record["after"], "runtime_signature": record["runtime_signature"]}
            answer = self.explainer.answer(dict(q), record, self.runtime)
            if not isinstance(answer, str):
                raise ValueError("Explanation must return verified text")
            with closing(self.connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                s = self._session(db, q["session_id"])
                permitted = self._can_explain(db, s) and s["active_run"] == q["run_id"] and s["stage"] == q["stage"]
                db.execute("UPDATE questions SET status=?,answer=? WHERE id=?",
                    ("complete" if permitted else "expired", answer if permitted else None, qid))
                db.commit()
        except Exception:
            with closing(self.connect()) as db:
                db.execute("UPDATE questions SET status='failed',answer=NULL WHERE id=?", (qid,))
        finally:
            with self.schedule_lock:
                self.scheduled.discard(qid)

    def _initialize_database(self, existing):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        if not existing:
            # A file appearing after the read-only preflight must not be
            # adopted as an unlabelled new database.
            descriptor = os.open(self.database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(descriptor)
        # The DB identity was checked without opening a writable connection.
        # Recheck before DDL; never import unlabelled legacy state.
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=rw", uri=True,
                                     isolation_level=None, timeout=15)) as db:
            db.execute("PRAGMA synchronous=FULL")
            if existing:
                rows = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')"))
                if (json.loads(rows.get("service_family", "null")) != SERVICE_FAMILY
                        or json.loads(rows.get("namespace", "null")) != self.namespace):
                    raise ValueError("Database identity changed before initialization")
            db.executescript("""
                PRAGMA journal_mode=WAL;
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
                    active_run TEXT, participant_id TEXT, participant_key TEXT UNIQUE, position INTEGER UNIQUE,
                    condition TEXT, task_order TEXT, mode TEXT NOT NULL DEFAULT 'freeplay',
                    stage TEXT NOT NULL DEFAULT 'practice', round_index INTEGER NOT NULL DEFAULT 0,
                    questionnaire TEXT NOT NULL DEFAULT '{}', namespace TEXT, study_signature TEXT,
                    questionnaire_bank_signature TEXT, questionnaire_scores TEXT);
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,scenario_id TEXT NOT NULL,
                    signature TEXT NOT NULL,stage TEXT NOT NULL,round_index INTEGER NOT NULL,snapshot TEXT NOT NULL,
                    metrics TEXT NOT NULL,ended INTEGER NOT NULL DEFAULT 0,created TEXT DEFAULT CURRENT_TIMESTAMP,
                    feedback TEXT NOT NULL DEFAULT '{}',end_reason TEXT,provenance TEXT);
                CREATE TABLE IF NOT EXISTS frames(run_id TEXT NOT NULL,frame INTEGER NOT NULL,public TEXT NOT NULL,
                    internal TEXT NOT NULL,PRIMARY KEY(run_id,frame));
                CREATE TABLE IF NOT EXISTS operations(session_id TEXT NOT NULL,id TEXT NOT NULL,digest TEXT NOT NULL,
                    version INTEGER NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(session_id,id));
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,run_id TEXT,
                    kind TEXT,payload TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP);
                CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,run_id TEXT NOT NULL,
                    frame INTEGER NOT NULL,stage TEXT NOT NULL,question TEXT NOT NULL,language TEXT NOT NULL,
                    status TEXT NOT NULL,answer TEXT,shown TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP,
                    focus TEXT NOT NULL DEFAULT 'executed');
                CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,allocation TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            """)
            try:
                if not existing:
                    db.executemany("INSERT INTO metadata VALUES(?,?)", [
                        ("service_family", canonical(SERVICE_FAMILY)), ("namespace", canonical(self.namespace)),
                        ("namespace_origin", canonical("observed197_verification_initialization"))])
                db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)",
                           ("service_context:" + self.signature, canonical(self.provenance)))
                db.execute("UPDATE questions SET status='pending' WHERE status='running'")
                db.commit()
                # Persist the identity in the main file for a future immutable
                # preflight. A busy checkpoint is not silently called durable.
                checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint is not None and checkpoint[0] != 0:
                    raise ValueError("Database identity checkpoint is busy")
            except BaseException:
                db.rollback()
                raise

    def close(self):
        if not getattr(self, "_closed", True):
            self._closed = True
            self.executor.shutdown(wait=True)


def main(argv=None):
    raise RuntimeError("public_history_qualified_release_unavailable: no production listener is implemented")


if __name__ == "__main__":
    main()
