"""Local A/B research service for an externally verified retained-beta release.

Only the independent release loader admits production material. Component
renderers and bank views retain their own false study-qualification flags.
No freeplay, model download, answer-key or private export HTTP endpoint exists.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from threading import Lock
from uuid import uuid4

from backend import warehouse_runtime_family as registry
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.warehouse_family_explanation import FamilyExplainer
from backend.warehouse_shutdown_runtime import ShutdownRuntime
from env.warehouse.layouts import get_map_layout
from ui import warehouse_native_server as native
from ui.warehouse_family_bank_view import FrozenFamilyBank
from ui.warehouse_native_release import analysis_protocol
from ui.warehouse_public_history_server import PublicHistoryStore
from ui.warehouse_view import warehouse_map_payload

VERSION = "warehouse-retained-beta-local-study-server.v1"
SERVICE_FAMILY = "warehouse_family_retained_beta197"
COOKIE = "warehouse_family_retained_beta197_session_v1"
NAMESPACE = "local_pilot"
DEFAULT_DATABASE = ROOT / "output/warehouse_native/family_local_pilot.sqlite3"
DEFAULT_PORT = 8009
WEB = ROOT / "ui/warehouse_public_history_research"
CommandError = native.CommandError


def load_family_release(root, *, expected_manifest_sha256):
    from ui.warehouse_family_release import load_family_release as loader
    return loader(root, expected_manifest_sha256=expected_manifest_sha256)


def service_sources():
    # The loader additionally binds the runtime, renderer and evidence closures.
    paths = [Path(__file__), ROOT / "ui/warehouse_family_release.py",
             ROOT / "ui/warehouse_public_history_server.py", ROOT / "ui/warehouse_native_server.py",
             ROOT / "ui/warehouse_native_release.py", ROOT / "ui/warehouse_native_export.py",
             ROOT / "ui/warehouse_family_bank_view.py", ROOT / "ui/warehouse_view.py"]
    paths += [WEB / name for name in ("index.html", "app.js", "styles.css")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def _sha(value):
    if type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("Explicit lowercase SHA256 is required")
    return value


def _sources(binding, required):
    if type(binding) is not dict or not binding:
        raise ValueError("Complete release source binding is required")
    for name, value in binding.items():
        p = Path(name)
        if (p.is_absolute() or p.as_posix() != name or ".." in p.parts
                or (ROOT / p).resolve() != ROOT / p or file_hash(ROOT / p) != _sha(value)):
            raise ValueError("Qualified release source changed")
    if any(binding.get(name) != value for name, value in required.items()):
        raise ValueError("Release does not bind the complete study service and components")
    return deepcopy(binding)


def _database_identity(database):
    if database.is_symlink(): raise ValueError("Database cannot be a symlink")
    if not database.exists(): return False
    if not database.is_file(): raise ValueError("Database must be a regular file")
    try:
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            values = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')"))
        if (json.loads(values.get("service_family", "null")) != SERVICE_FAMILY
                or json.loads(values.get("namespace", "null")) != NAMESPACE):
            raise ValueError("Database family or namespace differs")
    except (sqlite3.Error, json.JSONDecodeError) as error:
        raise ValueError("Existing database lacks a verified study identity") from error
    return True


def _assets(binding):
    """Small, explicit study-entry adapter; frozen Canvas/transport stay intact."""
    data = {name: (WEB / name).read_bytes() for name in ("index.html", "app.js", "styles.css")}
    for name, raw in data.items():
        if sha256(raw).hexdigest() != binding[str((WEB / name).relative_to(ROOT))]:
            raise ValueError("Bound participant asset changed during loading")
    js = data["app.js"].decode()
    lines = js.splitlines()
    start = [i for i, line in enumerate(lines) if '$("startButton").addEventListener("click"' in line]
    if len(start) != 1: raise ValueError("Study entry adapter no longer matches the frozen frontend")
    lines[start[0]] = '''  $("startButton").addEventListener("click",()=>{
    if(!canEnter(ui.view,"study"))return;
    if(study(ui.view) && phase(ui.view)==="consent"){
      if(!$("consentInput").checked){ui.error="consentRequired";render();return;}
      void execute({kind:"next",consent:true});return;
    }
    const id=$("participantInput").value.trim();
    if(!/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(id)){ui.error="participantInvalid";render();return;}
    void execute({kind:"start",mode:"study",participant_id:id});
  });'''
    js = "\n".join(lines) + "\n"
    old = '    const consentStage=isStudy && p==="consent";const entryAllowed=!isStudy || consentStage;'
    addition = '''
    ui.entry="study";
    document.querySelector(".agreement").hidden=!consentStage;
    $("participantInput").value=v?.flow?.participant_id || $("participantInput").value;
    $("consentText").hidden=!consentStage;'''
    if js.count(old) != 1: raise ValueError("Consent adapter no longer matches the frozen frontend")
    js = js.replace(old, old + addition)
    marker = '$("startButton").textContent=tr(ui.entry==="study"?"startStudy":"startFreeplay");'
    if js.count(marker) != 1: raise ValueError("Entry caption adapter differs")
    js = js.replace(marker, '$("startButton").textContent=consentStage?tr("startStudy"):(ui.language==="zh"?"登记用户 ID":"Register participant ID");')
    # The private request key cannot accidentally resume another service's input.
    js = re.sub(r'const PENDING_KEY="[^"]+", LANGUAGE_KEY="[^"]+";',
        'const PENDING_KEY="warehouse-family-study.v1.pending", LANGUAGE_KEY="warehouse-family-study.v1.lang";', js, count=1)
    if 'warehouse-family-study.v1.pending' not in js: raise ValueError("Request-cache adapter differs")
    data["app.js"] = js.encode()
    data["styles.css"] += b'\n#freeplayTab,#freeplaySetup,#savedRunsField,#feedbackPanel,.technical{display:none!important} .agreement[hidden]{display:none!important}\n'
    return data


class FamilyStudyStore(PublicHistoryStore):
    def __init__(self, release_root, *, expected_manifest_sha256, database=DEFAULT_DATABASE):
        expected_manifest_sha256 = _sha(expected_manifest_sha256)
        root = Path(release_root).expanduser().absolute()
        requested = Path(database).expanduser().absolute()
        if root.resolve() != root or requested.resolve() != requested:
            raise ValueError("Release and database paths must not traverse symlinks")
        if requested == root or root in requested.parents:
            raise ValueError("Database must be outside the immutable release")
        self._closed, self._release_context = False, None
        context = load_family_release(root, expected_manifest_sha256=expected_manifest_sha256)
        self._release_context = context
        try:
            if not callable(getattr(context, "close", None)):
                raise ValueError("Release context must own its private evidence lifetime")
            if context.manifest_sha256 != expected_manifest_sha256:
                raise ValueError("Loader returned another release")
            _sha(context.signature)
            if (type(context.runtime) is not ShutdownRuntime or type(context.explainer) is not FamilyExplainer
                    or type(context.question_bank) is not FrozenFamilyBank):
                raise ValueError("Genuine retained-beta runtime, renderer and frozen bank are required")
            identity = registry.verify(context.runtime, expected_family="retained_beta197")
            context.explainer._assert_current(context.runtime)
            bank = context.question_bank
            if (bank.test_fixture is not False or bank.content_eligible is not True or bank.eligible is not False
                    or bank.actor_sha256 != identity["actor_sha256"]
                    or bank.runtime_signature != identity["runtime_signature"]):
                raise ValueError("Bank does not bind the admitted genuine runtime")
            release = context.release
            if (type(release) is not dict or release.get("status") != "local_pilot_technically_verified"
                    or release.get("namespace") != NAMESPACE
                    or any(release.get(key) is not True for key in ("model_ready", "explanation_ready", "study_ready"))
                    or release.get("formal_ready") is not False or release.get("test_fixture") is not False):
                raise ValueError("Completed genuine local-pilot evidence is required")
            if canonical(context.provenance.get("analysis")) != canonical(analysis_protocol()):
                raise ValueError("Frozen local-pilot analysis differs")
            self.runtime, self.explainer, self.question_bank = context.runtime, context.explainer, bank
            self.scenarios = deepcopy(context.scenarios)
            play = self.scenarios.get("splits", {}).get("play")
            if (self.scenarios.get("test_fixture") is True or type(play) is not list or len(play) < 7
                    or len({s["id"] for s in play}) != len(play)):
                raise ValueError("Practice and six frozen task scenarios are required")
            for key, actual in (("runtime_signature", self.runtime.signature),
                    ("actor_sha256", self.runtime.actor_sha256), ("protocol_sha256", self.runtime.protocol_sha256),
                    ("scenario_manifest_sha256", digest(self.scenarios)), ("manifest_sha256", expected_manifest_sha256)):
                if context.provenance.get(key) != actual: raise ValueError("Release provenance differs: " + key)
            required = {**service_sources(), **identity["registry_sources"], **self.explainer.sources, **bank.sources}
            self.source_binding = _sources(context.source_binding, required)
            self.web_assets = _assets(self.source_binding)
            # Loading the fixed layout is pure; no environment reset or policy call.
            self._map = warehouse_map_payload(get_map_layout(self.runtime.config.map_layout_id))
            self.database, self.namespace, self.cookie_name = requested, NAMESPACE, COOKIE
            existing = _database_identity(self.database)
            self.verification = self.test_fixture = False
            self.release = deepcopy(release)
            self.signature = digest({"version": VERSION, "release": context.signature,
                "manifest_sha256": expected_manifest_sha256, "sources": self.source_binding,
                "runtime": self.runtime.signature, "bank": bank.signature, "explainer": self.explainer.signature,
                "scenarios": digest(self.scenarios), "study_only": True})
            self.provenance = {**deepcopy(context.provenance), "service_family": SERVICE_FAMILY,
                "service_version": VERSION, "run_signature": self.signature, "service_sources": self.source_binding,
                "verification_only": False, "study_only": True}
            self._initialize_database(existing)
            self.scheduled, self.schedule_lock = set(), Lock()
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="family-study-explanation")
        except BaseException:
            self.close()
            raise

    def session(self, candidate):
        with closing(self.connect()) as db:
            if candidate and db.execute("SELECT id FROM sessions WHERE id=?", (candidate,)).fetchone():
                return candidate
            sid = uuid4().hex
            db.execute("INSERT INTO sessions(id,mode,stage,namespace) VALUES(?,'enrollment','registration',?)", (sid, NAMESPACE))
            return sid

    def _permitted(self, session):
        return (not self._closed and self.release["study_ready"] is True and self.release["explanation_ready"] is True
            and not self._study_mismatch(session) and session["mode"] == "study"
            and session["stage"] == "task1" and session["condition"] == "A")

    def _bound_bank(self, session):
        if session["mode"] != "study": return None
        if (session["questionnaire_bank_signature"] != self.question_bank.signature
                or not self.question_bank.content_eligible):
            raise CommandError("questionnaire_bank_version_mismatch", 409)
        return self.question_bank

    def _questionnaire(self, session):
        # Original pure questionnaire projection, using the independently bound
        # bank above rather than modifying its component-qualification property.
        return native.NativeStore._questionnaire(self, session)

    def _view(self, db, sid):
        result = super()._view(db, sid)
        session = self._session(db, sid)
        result.update(study_allowed=not self._study_mismatch(session), verification_flow_allowed=False,
                      verification_only=False, study_only=True, runs=[])
        kinds = result["allowed_kinds"]
        if not self._study_mismatch(session):
            if session["mode"] == "enrollment": kinds = ["start"]
            elif result["run_id"] and not result["ended"] and not result.get("version_mismatch"):
                if session["stage"] in ("practice", "task1", "task2"): kinds += ["action"]
        result["allowed_kinds"] = [k for k in dict.fromkeys(kinds) if k not in ("select_run", "feedback")]
        result["flow"]["consent_text"] = {
            "zh": "本地预实验会保存用户 ID、操作、问答和问卷，用于研究人机合作；不纳入正式研究样本。问题由本机已核验的证据系统处理。您可随时停止参加；已确认的回合记录会保留。请使用不含姓名、邮箱或电话号码的研究编号，并阅读研究者提供的完整参与说明。",
            "en": "This local pilot records your study ID, actions, questions and questionnaire responses to study human–AI cooperation. These are not formal study samples. Answers use the verified local evidence system. You may stop at any time; confirmed records are retained. Use an ID without your name, email or telephone number and read the researcher's full participation information."}
        return result

    def command(self, sid, payload):
        if type(payload) is not dict: raise CommandError("invalid_body")
        if payload.get("kind") == "start": return self._register(sid, payload)
        if payload.get("kind") in ("select_run", "feedback"):
            raise CommandError("study_only_no_freeplay", 403)
        return super().command(sid, payload)

    def _register(self, sid, payload):
        if payload.get("mode") != "study": raise CommandError("study_only_no_freeplay", 403)
        # Registration never advances to practice, even if consent is supplied.
        op = payload.get("operation_id")
        if type(op) is not str or not 1 <= len(op) <= 128: raise CommandError("operation_id_required")
        request_hash = digest(payload)
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                s = self._session(db, sid)
                old = db.execute("SELECT * FROM operations WHERE session_id=? AND id=?", (sid, op)).fetchone()
                if old:
                    if old["digest"] != request_hash: raise CommandError("operation_id_reused", 409, self._view(db, sid))
                    if self._study_mismatch(s): raise CommandError("study_version_changed", 409)
                    return self._view(db, sid)
                if type(payload.get("expected_version")) is not int or payload["expected_version"] != s["version"]:
                    raise CommandError("state_version_conflict", 409, self._view(db, sid))
                if s["mode"] != "enrollment": raise CommandError("study_cannot_restart_or_switch_mode", 403)
                participant = payload.get("participant_id", "")
                if type(participant) is not str: raise CommandError("invalid_participant_id")
                participant = participant.strip()
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,31}", participant): raise CommandError("invalid_participant_id")
                if db.execute("SELECT id FROM sessions WHERE participant_key=?", (participant.lower(),)).fetchone():
                    raise CommandError("participant_id_taken", 409)
                position = db.execute("SELECT count(*) FROM sessions WHERE position IS NOT NULL").fetchone()[0]
                block_id = position // 4
                old_block = db.execute("SELECT allocation FROM blocks WHERE id=?", (block_id,)).fetchone()
                if old_block is None:
                    cells = [["A", "XY"], ["A", "YX"], ["B", "XY"], ["B", "YX"]]
                    random.SystemRandom().shuffle(cells)
                    db.execute("INSERT INTO blocks VALUES(?,?)", (block_id, canonical(cells)))
                else: cells = json.loads(old_block[0])
                condition, order = cells[position % 4]
                db.execute("UPDATE sessions SET mode='study',stage='consent',participant_id=?,participant_key=?,position=?,condition=?,task_order=?,questionnaire_bank_signature=?,study_signature=?,version=version+1 WHERE id=?",
                    (participant, participant.lower(), position, condition, order, self.question_bank.signature, self.signature, sid))
                version = self._session(db, sid)["version"]
                db.execute("INSERT INTO operations VALUES(?,?,?,?,?)", (sid, op, request_hash, version, "start"))
                db.execute("INSERT INTO events(session_id,kind,payload) VALUES(?,'start',?)", (sid, canonical(payload)))
                result = self._view(db, sid)
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise

    def _initialize_database(self, existing):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        if not existing:
            fd = os.open(self.database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        with closing(sqlite3.connect(self.database.as_uri()+"?mode=rw", uri=True, isolation_level=None, timeout=15)) as db:
            db.execute("PRAGMA synchronous=FULL")
            if existing:
                values = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')"))
                if json.loads(values.get("service_family", "null")) != SERVICE_FAMILY or json.loads(values.get("namespace", "null")) != NAMESPACE:
                    raise ValueError("Database identity changed before initialization")
            db.executescript(_SCHEMA)
            try:
                if not existing:
                    db.executemany("INSERT INTO metadata VALUES(?,?)", [("service_family", canonical(SERVICE_FAMILY)),
                        ("namespace", canonical(NAMESPACE)), ("namespace_origin", canonical(VERSION))])
                db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)", ("service_context:"+self.signature, canonical(self.provenance)))
                db.execute("UPDATE questions SET status='pending' WHERE status='running'")
                db.commit()
                status = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if status and status[0] != 0: raise ValueError("Database identity checkpoint is busy")
            except BaseException:
                db.rollback()
                raise

    def close(self):
        if not getattr(self, "_closed", True):
            self._closed = True
            try:
                if hasattr(self, "executor"): self.executor.shutdown(wait=True)
            finally:
                context = getattr(self, "_release_context", None)
                if context is not None and callable(getattr(context, "close", None)): context.close()


# Same transaction schema as the frozen Store; the separate metadata identity
# prevents adopting its verification or legacy records as newly accepted data.
_SCHEMA = """PRAGMA journal_mode=WAL;
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
 active_run TEXT,participant_id TEXT,participant_key TEXT UNIQUE,position INTEGER UNIQUE,condition TEXT,task_order TEXT,
 mode TEXT NOT NULL DEFAULT 'enrollment',stage TEXT NOT NULL DEFAULT 'registration',round_index INTEGER NOT NULL DEFAULT 0,
 questionnaire TEXT NOT NULL DEFAULT '{}',namespace TEXT,study_signature TEXT,questionnaire_bank_signature TEXT,questionnaire_scores TEXT);
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,scenario_id TEXT NOT NULL,signature TEXT NOT NULL,
 stage TEXT NOT NULL,round_index INTEGER NOT NULL,snapshot TEXT NOT NULL,metrics TEXT NOT NULL,ended INTEGER NOT NULL DEFAULT 0,
 created TEXT DEFAULT CURRENT_TIMESTAMP,feedback TEXT NOT NULL DEFAULT '{}',end_reason TEXT,provenance TEXT);
CREATE TABLE IF NOT EXISTS frames(run_id TEXT NOT NULL,frame INTEGER NOT NULL,public TEXT NOT NULL,internal TEXT NOT NULL,PRIMARY KEY(run_id,frame));
CREATE TABLE IF NOT EXISTS operations(session_id TEXT NOT NULL,id TEXT NOT NULL,digest TEXT NOT NULL,version INTEGER NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(session_id,id));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,run_id TEXT,kind TEXT,payload TEXT,created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS questions(id TEXT PRIMARY KEY,session_id TEXT NOT NULL,run_id TEXT NOT NULL,frame INTEGER NOT NULL,
 stage TEXT NOT NULL,question TEXT NOT NULL,language TEXT NOT NULL,status TEXT NOT NULL,answer TEXT,shown TEXT,
 created TEXT DEFAULT CURRENT_TIMESTAMP,focus TEXT NOT NULL DEFAULT 'executed');
CREATE TABLE IF NOT EXISTS blocks(id INTEGER PRIMARY KEY,allocation TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535: parser.error("port must be between 1 and 65535")
    store = FamilyStudyStore(args.release_root, expected_manifest_sha256=args.expected_manifest_sha256, database=args.database)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), native.handler_class(store))
        try: server.serve_forever()
        finally: server.server_close()
    finally: store.close()


if __name__ == "__main__": main()
