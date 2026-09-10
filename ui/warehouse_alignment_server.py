"""Independent alignment A/B service using the unchanged study transactions.

The public constructor requires a completed independent alignment release.
The separate in-process component-verification factory admits a real frozen
NN without claiming participant, explanation, bank or research qualification.
There is no HTTP or command-line switch for that private verification mode.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from http.server import ThreadingHTTPServer
import argparse
import signal
import json
import os
from pathlib import Path
import random
import re
import sqlite3
from threading import Lock
from uuid import uuid4

from backend import warehouse_alignment_runtime as alignment
from backend.warehouse_alignment_explanation import AlignmentExplainer
from backend.warehouse_alignment_diverse_explanation import DiverseAlignmentExplainer
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from env.warehouse.layouts import get_map_layout
from ui import warehouse_family_server as previous
from ui.warehouse_native_release import analysis_protocol
from ui.warehouse_view import warehouse_map_payload

VERSION = "warehouse-alignment-ab-study-server.v1"
SERVICE_FAMILY = "warehouse_alignment197"
CONTEXT_VERSION = "warehouse-alignment-component-verification-context.v1"
RELEASE_VERSION = "warehouse-alignment-local-pilot-release.v1"
WEB = ROOT / "ui/warehouse_family_feedback_research"
DEFAULT_DATABASE = ROOT / "output/warehouse_native/alignment_local_pilot.sqlite3"
CommandError = previous.CommandError
_CLOSED_FLAGS = ("model_ready", "explanation_ready", "study_ready", "formal_ready",
                 "participant_enabled", "release_ready", "explanation_qualified")


def service_sources():
    result = previous.service_sources()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    for name in ("index.html", "app.js", "styles.css"):
        path = WEB / name
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _assets(binding, verification):
    assets = {name: (WEB / name).read_bytes() for name in ("index.html", "app.js", "styles.css")}
    for name, raw in assets.items():
        if sha256(raw).hexdigest() != binding.get(str((WEB / name).relative_to(ROOT))):
            raise ValueError("Bound alignment study asset changed")
    js = assets["app.js"].decode()
    replacements = {
        'const FRONTEND_VERSION="warehouse-family-feedback-research.v2";':
            'const FRONTEND_VERSION="warehouse-alignment-research.v1";',
        'const PENDING_KEY="warehouse-family-feedback-fresh.v2.pending", LANGUAGE_KEY="warehouse-family-feedback-fresh.v2.lang";':
            'const PENDING_KEY="warehouse-alignment.' + ("verification" if verification else "pilot") + '.v1.pending", LANGUAGE_KEY="warehouse-alignment.v1.lang";',
        'function verificationFlow(view){return view?.verification_only===true && view?.verification_flow_allowed===true && view?.release?.test_fixture===true;}':
            'function verificationFlow(view){return view?.verification_only===true && view?.verification_flow_allowed===true && view?.real_nn_component_verification===true && view?.release?.test_fixture===false && view?.release?.study_ready===false && view?.release?.participant_enabled===false;}',
        'const consentStage=isStudy && p==="consent";const entryAllowed=!isStudy || consentStage;':
            'const consentStage=isStudy && p==="consent",consentEntry=!isStudy && p==="registration",showConsent=consentStage || consentEntry;const entryAllowed=!isStudy || consentStage;',
        'document.querySelector(".agreement").hidden=!consentStage;':
            'document.querySelector(".agreement").hidden=!showConsent;',
        '$("consentText").hidden=!consentStage;':
            '$("consentText").hidden=!showConsent;',
        '$("startButton").textContent=consentStage?tr("startStudy"):(ui.language==="zh"?"登记用户 ID":"Register participant ID");':
            '$("startButton").textContent=consentStage?tr("startStudy"):(ui.language==="zh"?"同意并登记用户 ID":"Consent and register participant ID");',
        'const id=$("participantInput").value.trim();\n    if(!/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(id)){ui.error="participantInvalid";render();return;}\n    void execute({kind:"start",mode:"study",participant_id:id});':
            'const id=$("participantInput").value.trim();\n    if(!$("consentInput").checked){ui.error="consentRequired";render();return;}\n    if(!/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(id)){ui.error="participantInvalid";render();return;}\n    void execute({kind:"start",mode:"study",participant_id:id,consent:true});',
    }
    if verification:
        replacements.update({
            'verification:"流程验证 · 非正式实验"': 'verification:"真实 NN 技术验收 · 非参与者"',
            'verification:"Flow verification · Not a study"': 'verification:"Real NN technical verification · no participants"',
            'verificationMessage:"隔离流程验证环境；若为测试权重，不代表训练或能力验收。"':
                'verificationMessage:"真实冻结 NN 的隔离流程技术验收；未开放参与者实验或解释资格。"',
            'verificationMessage:"Isolated flow verification. Test weights are not training or capability evidence."':
                'verificationMessage:"Isolated technical flow with a real frozen NN; participant and explanation qualification remain closed."',
            'agree:"我已阅读参与说明并同意参加。"': 'agree:"我确认这是内部技术验收，不是参与者实验。"',
            'agree:"I have read the participation information and consent to take part."': 'agree:"I understand this is internal technical verification, not participant enrollment."',
        })
    for old, new in replacements.items():
        if js.count(old) != 1:
            raise ValueError("Frozen frontend adapter no longer matches: " + old)
        js = js.replace(old, new)
    assets["app.js"] = js.encode()
    return assets


def _bank_identity(bank, runtime, *, required, full_verification=True):
    if bank is None:
        if required:
            raise ValueError("A completed same-Actor alignment prediction bank is required")
        return None
    from ui.warehouse_alignment_metadata_bank_view import FrozenAlignmentMetadataBank
    if type(bank) is not FrozenAlignmentMetadataBank:
        raise ValueError("Only the genuine independently verified alignment metadata bank is supported")
    if full_verification:
        bank.verify_binding()
    if (bank.test_fixture is not False or bank.content_eligible is not True
            or any(getattr(bank, name, None) is not False for name in
                   ("eligible", "participant_enabled", "formal_ready", "release_ready"))
            or bank.actor_sha256 != runtime.actor_sha256
            or bank.runtime_signature != runtime.signature
            or bank.protocol_sha256 != runtime.protocol_sha256):
        raise ValueError("Alignment bank source, replay or component scope differs")
    previous._sha(bank.signature)
    if full_verification:
        previous._sources(bank.sources, bank.sources)
    items = bank.public_items()
    if len(items) != 8 or len({item["id"] for item in items}) != 8:
        raise ValueError("The complete four next-action and four wait-three prediction items are required")
    return {"signature": bank.signature, "sources": deepcopy(bank.sources),
            "public_items_sha256": digest(items), "private_items_sha256": digest(bank.items),
            "summary_sha256": digest(bank.summary())}


def _components(runtime, explainer, scenarios, *, expected_scenarios, bank, required_bank):
    identity = alignment.verify(runtime, allow_test_fixture=False, expected_family=alignment.FAMILY)
    expected_explainer = DiverseAlignmentExplainer if required_bank else AlignmentExplainer
    if type(explainer) is not expected_explainer:
        raise ValueError("The genuine qualified alignment renderer is required")
    explainer._assert_current(runtime)
    if required_bank and (explainer.explanation_qualified is not True
            or explainer.study_ready is not False
            or explainer.participant_enabled is not False
            or explainer.release_ready is not False):
        raise ValueError("Qualified renderer component scope differs")
    if (type(scenarios) is not dict or digest(scenarios) != previous._sha(expected_scenarios)
            or scenarios.get("test_fixture") is True):
        raise ValueError("Externally bound original non-fixture scenarios are required")
    play = scenarios.get("splits", {}).get("play")
    if (type(play) is not list or len(play) < 7 or any(type(s) is not dict for s in play)
            or any(type(s.get("id")) is not str or not s["id"] for s in play)
            or len({s["id"] for s in play}) != len(play)):
        raise ValueError("Practice and six unique ordered task scenarios are required")
    bank_identity = _bank_identity(bank, runtime, required=required_bank)
    required = {**service_sources(), **identity["runtime_sources"], **explainer.sources,
                **(bank.sources if bank is not None else {})}
    return identity, bank_identity, previous._sources(required, required)


@dataclass(frozen=True)
class _ComponentVerificationContext:
    runtime: object
    explainer: object
    question_bank: object
    scenarios: dict
    source_binding: dict
    provenance: dict
    signature: str
    evidence: dict

    def close(self):
        """This context borrows immutable caller-owned component objects."""


def make_component_verification_context(runtime, explainer, scenarios, *,
        expected_scenario_manifest_sha256, component_report_path,
        expected_component_report_sha256, question_bank=None):
    """Zero-forward admission of saved real component evidence for technical QA."""
    identity, bank_identity, sources = _components(runtime, explainer, scenarios,
        expected_scenarios=expected_scenario_manifest_sha256, bank=question_bank, required_bank=False)
    path = Path(component_report_path).expanduser().resolve()
    if file_hash(path) != previous._sha(expected_component_report_sha256):
        raise ValueError("Completed component report differs from its external hash")
    report = json.loads(path.read_bytes())
    if (report.get("version") != "warehouse-alignment-answer-component-audit.v1"
            or report.get("status") != "completed" or report.get("component_checks_passed") is not True
            or any(report.get(k) is not False for k in ("explanation_qualified", "participant_enabled",
                "release_ready", "independent_explanation_acceptance", "holdout_or_final_test_used",
                "whole_research_goal_completed", "live_server_changed"))
            or digest(report.get("explanation_contract")) != digest(explainer.contract_report)
            or digest(report.get("runtime_contract")) != digest(runtime.contract_report)):
        raise ValueError("Actual component evidence or closed qualification differs")
    artifacts = {}
    for name in ("request", "answers", "records"):
        artifact = path.parent / (name + ".json")
        actual = file_hash(artifact)
        if actual != previous._sha(report.get(name + "_sha256")):
            raise ValueError("Saved component evidence changed: " + name)
        artifacts[str(artifact)] = actual
    request = json.loads((path.parent / "request.json").read_bytes())
    for key in ("input_sha256", "source_sha256"):
        previous._sources(request[key], request[key])
    evidence = {"version": CONTEXT_VERSION, "report_path": str(path),
        "report_sha256": expected_component_report_sha256, "artifacts": artifacts,
        "runtime": identity, "explainer_signature": explainer.signature,
        "program_sha256": explainer.program_sha256, "bank": bank_identity,
        "scenario_manifest_sha256": expected_scenario_manifest_sha256,
        "source_binding": sources, "real_nn": True, "participant_enabled": False,
        "independent_acceptance": False}
    provenance = {"version": CONTEXT_VERSION, "namespace": "verification",
        "actor_sha256": runtime.actor_sha256, "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature, "program_sha256": explainer.program_sha256,
        "scenario_manifest_sha256": expected_scenario_manifest_sha256,
        "component_report_sha256": expected_component_report_sha256,
        "question_bank_signature": question_bank.signature if question_bank is not None else None,
        "analysis": analysis_protocol(), "participant_enabled": False,
        "real_nn_component_verification": True, "formal_ready": False}
    return _ComponentVerificationContext(runtime, explainer, question_bank, deepcopy(scenarios),
        sources, provenance, digest(evidence), evidence)


class AlignmentStudyStore(previous.FamilyStudyStore):
    def __init__(self, release_root, *, expected_manifest_sha256, database=DEFAULT_DATABASE):
        """Production stays closed until its independent release reader exists."""
        root = Path(release_root).expanduser().absolute()
        if root.resolve() != root:
            raise ValueError("Release path must not traverse symlinks")
        previous._sha(expected_manifest_sha256)
        reader_path = ROOT / "ui/warehouse_alignment_release.py"
        if not reader_path.is_file():
            raise ValueError("alignment_qualified_release_unavailable: independent alignment release integration is unfinished")
        reader = import_module("ui.warehouse_alignment_release")
        if Path(reader.__file__).resolve() != reader_path or reader.VERSION != RELEASE_VERSION:
            raise ValueError("Independent alignment release reader identity differs")
        context = reader.load_alignment_release(root, expected_manifest_sha256=expected_manifest_sha256)
        self._closed, self._release_context = False, context
        try:
            if (not callable(getattr(context, "close", None))
                    or context.manifest_sha256 != expected_manifest_sha256
                    or file_hash(root / "manifest.json") != expected_manifest_sha256):
                raise ValueError("Completed alignment release context differs")
            release = context.release
            if (type(release) is not dict or release.get("status") != "local_pilot_technically_verified"
                    or release.get("namespace") != "local_pilot"
                    or any(release.get(k) is not True for k in ("model_ready", "explanation_ready", "study_ready"))
                    or release.get("formal_ready") is not False or release.get("test_fixture") is not False):
                raise ValueError("Complete genuine local-pilot acceptance is required")
            self._initialize(context, database, verification=False, release=release,
                             release_root=root, manifest_sha256=expected_manifest_sha256)
        except BaseException:
            self.close()
            raise

    @classmethod
    def for_component_verification(cls, context, *, database):
        """Explicit process-local QA only; never a production release shortcut."""
        if type(context) is not _ComponentVerificationContext:
            raise ValueError("An independently bound private component-verification context is required")
        if digest(context.evidence) != context.signature:
            raise ValueError("Private component-verification context changed")
        rechecked = make_component_verification_context(context.runtime, context.explainer, context.scenarios,
            expected_scenario_manifest_sha256=context.evidence["scenario_manifest_sha256"],
            component_report_path=context.evidence["report_path"],
            expected_component_report_sha256=context.evidence["report_sha256"],
            question_bank=context.question_bank)
        if (rechecked.signature != context.signature or rechecked.source_binding != context.source_binding
                or rechecked.provenance != context.provenance):
            raise ValueError("Private component evidence or source identity changed")
        self = cls.__new__(cls)
        self._closed, self._release_context = False, context
        release = {**{name: False for name in _CLOSED_FLAGS},
            "status": "real_nn_component_verification_only", "namespace": "verification",
            "test_fixture": False, "qualification_evaluated": False,
            "message": {"zh": "真实冻结 NN 的内部技术验收；未开放参与者实验或解释资格。",
                        "en": "Internal technical verification with a real frozen NN; no participant or explanation qualification."}}
        try:
            self._initialize(context, database, verification=True, release=release)
        except BaseException:
            self.close()
            raise
        return self

    def _initialize(self, context, database, *, verification, release, release_root=None, manifest_sha256=None):
        requested = Path(database).expanduser().absolute()
        if requested.resolve() != requested or (release_root is not None
                and (requested == release_root or release_root in requested.parents)):
            raise ValueError("Database must be a separate path without symlinks")
        previous._sha(context.signature)
        expected_scenes = context.provenance["scenario_manifest_sha256"]
        identity, bank_identity, required = _components(context.runtime, context.explainer, context.scenarios,
            expected_scenarios=expected_scenes, bank=context.question_bank, required_bank=not verification)
        if not verification:
            path = ROOT / "ui/warehouse_alignment_release.py"
            required[str(path.relative_to(ROOT))] = file_hash(path)
            if (context.provenance.get("manifest_sha256") != manifest_sha256
                    or context.provenance.get("program_sha256") != context.explainer.program_sha256):
                raise ValueError("Qualified alignment release manifest or program binding differs")
        self.source_binding = previous._sources(context.source_binding, required)
        for key, value in (("actor_sha256", context.runtime.actor_sha256),
                ("runtime_signature", context.runtime.signature), ("protocol_sha256", context.runtime.protocol_sha256)):
            if context.provenance.get(key) != value:
                raise ValueError("Alignment context provenance differs: " + key)
        if digest(context.provenance.get("analysis")) != digest(analysis_protocol()):
            raise ValueError("Frozen A/B analysis contract differs")
        self.runtime, self.explainer, self.question_bank = context.runtime, context.explainer, context.question_bank
        self.scenarios = deepcopy(context.scenarios)
        self._bank_binding = bank_identity
        self._bank_object = self.question_bank
        self.web_assets = _assets(self.source_binding, verification)
        self.asset_sha256 = {name: sha256(raw).hexdigest() for name, raw in self.web_assets.items()}
        self._map = warehouse_map_payload(get_map_layout(self.runtime.config.map_layout_id))
        self.database = requested
        self.namespace = "verification" if verification else "local_pilot"
        self.cookie_name = "warehouse_alignment_" + self.namespace + "_session_v1"
        self.verification, self.test_fixture = verification, False
        self.release = deepcopy(release)
        self.signature = digest({"version": VERSION, "context": context.signature,
            "manifest_sha256": manifest_sha256, "sources": self.source_binding,
            "served_assets_sha256": self.asset_sha256, "runtime": self.runtime.signature,
            "explainer": self.explainer.signature, "bank": bank_identity,
            "scenarios": expected_scenes, "namespace": self.namespace, "verification_only": verification})
        self.provenance = {**deepcopy(context.provenance), "service_family": SERVICE_FAMILY,
            "service_version": VERSION, "namespace": self.namespace, "run_signature": self.signature,
            "service_sources": self.source_binding, "served_assets_sha256": self.asset_sha256,
            "verification_only": verification, "real_nn_component_verification": verification,
            "participant_enabled": not verification, "study_only": True,
            "questionnaire_bank_complete": self.question_bank is not None,
            "questionnaire_scope": "four_next_four_wait_three_plus_self_report" if self.question_bank else "self_report_only_bank_unavailable",
            "release": deepcopy(self.release)}
        existing = self._database_identity()
        self._initialize_database(existing)
        self.scheduled, self.schedule_lock = set(), Lock()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="alignment-study-explanation")

    def _database_identity(self):
        if self.database.is_symlink():
            raise ValueError("Database cannot be a symlink")
        if not self.database.exists():
            return False
        if not self.database.is_file():
            raise ValueError("Database must be a regular file")
        try:
            with closing(sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)) as db:
                self._check_database(db)
        except (sqlite3.Error, json.JSONDecodeError) as error:
            raise ValueError("Existing database lacks this alignment service identity") from error
        return True

    def _check_database(self, db):
        values = dict(db.execute("SELECT key,value FROM metadata WHERE key IN ('service_family','namespace')"))
        if (json.loads(values.get("service_family", "null")) != SERVICE_FAMILY
                or json.loads(values.get("namespace", "null")) != self.namespace):
            raise ValueError("Alignment database family or namespace differs")

    def _initialize_database(self, existing):
        self.database.parent.mkdir(parents=True, exist_ok=True)
        if not existing:
            fd = os.open(self.database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        with closing(sqlite3.connect(self.database.as_uri() + "?mode=rw", uri=True,
                                    isolation_level=None, timeout=15)) as db:
            db.execute("PRAGMA synchronous=FULL")
            if existing:
                self._check_database(db)
            db.executescript(previous._SCHEMA)
            try:
                if not existing:
                    db.executemany("INSERT INTO metadata VALUES(?,?)", [
                        ("service_family", canonical(SERVICE_FAMILY)),
                        ("namespace", canonical(self.namespace)), ("namespace_origin", canonical(VERSION))])
                db.execute("INSERT OR IGNORE INTO metadata VALUES(?,?)",
                    ("service_context:" + self.signature, canonical(self.provenance)))
                db.execute("UPDATE questions SET status='pending' WHERE status='running'")
                db.commit()
                status = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if status and status[0] != 0:
                    raise ValueError("Database identity checkpoint is busy")
            except BaseException:
                db.rollback()
                raise

    def session(self, candidate):
        with closing(self.connect()) as db:
            if candidate and db.execute("SELECT id FROM sessions WHERE id=?", (candidate,)).fetchone():
                return candidate
            sid = uuid4().hex
            db.execute("INSERT INTO sessions(id,mode,stage,namespace) VALUES(?,'enrollment','registration',?)", (sid, self.namespace))
            return sid

    def _permitted(self, session):
        if self.verification:
            return (not self._closed and all(self.release.get(k) is False for k in _CLOSED_FLAGS)
                and not self._study_mismatch(session) and session["mode"] == "study"
                and session["stage"] == "task1" and session["condition"] == "A")
        return super()._permitted(session)

    def _bound_bank(self, session):
        if session["mode"] != "study":
            return None
        if self.question_bank is None:
            if self.verification and session["questionnaire_bank_signature"] is None:
                return None
            raise CommandError("questionnaire_bank_version_mismatch", 409)
        if (self.question_bank is not self._bank_object
                or session["questionnaire_bank_signature"] != self.question_bank.signature):
            raise CommandError("questionnaire_bank_version_mismatch", 409)
        # The bank's complete journal/files were verified during admission.
        # Polling checks the bound object, identities and all eight private and
        # public items in memory; it never rehashes the 576-step source history.
        if _bank_identity(self.question_bank, self.runtime, required=True,
                          full_verification=False) != self._bank_binding:
            raise CommandError("questionnaire_bank_version_mismatch", 409)
        return self.question_bank

    def command(self, sid, payload):
        if isinstance(payload, dict) and payload.get("kind") == "questionnaire":
            # Every draft write/submission verifies the complete frozen bank
            # before the inherited transaction can grade or persist answers.
            self.explainer._assert_current(self.runtime)
            if self.question_bank is not None:
                if (self.question_bank is not self._bank_object
                        or _bank_identity(self.question_bank, self.runtime, required=True) != self._bank_binding):
                    raise CommandError("questionnaire_bank_version_mismatch", 409)
        return super().command(sid, payload)

    def close(self):
        if getattr(self, "_closed", True):
            return
        try:
            if getattr(self, "question_bank", None) is not None:
                if (self.question_bank is not self._bank_object
                        or _bank_identity(self.question_bank, self.runtime, required=True) != self._bank_binding):
                    raise ValueError("Alignment bank changed before service close")
            if hasattr(self, "explainer"):
                self.explainer._assert_current(self.runtime)
        finally:
            super().close()

    def _view(self, db, sid):
        result = super()._view(db, sid)
        session = self._session(db, sid)
        if self.verification:
            result.update(study_allowed=False, verification_only=True,
                verification_flow_allowed=not self._study_mismatch(session),
                real_nn_component_verification=True, participant_enabled=False)
            result["flow"]["consent_text"] = {
                "zh": "这是使用真实冻结 NN 的内部 A/B 技术验收，不招募参与者、不授予解释资格。ID、操作、问答及问卷保存在独立 verification 数据库，仅用于检查流程。",
                "en": "This is internal A/B technical verification with a real frozen NN, not participant enrollment or explanation qualification. IDs, actions, answers and questionnaire entries remain in a separate verification database for flow checks only."}
        else:
            result.update(real_nn_component_verification=False, participant_enabled=True)
        result["questionnaire"]["complete_prediction_bank_available"] = self.question_bank is not None
        result["questionnaire"]["scope"] = self.provenance["questionnaire_scope"]
        return result

    def _register(self, sid, payload):
        # Identical registration transaction; only the absent private-QA bank
        # is represented honestly by a NULL enrollment binding.
        if payload.get("mode") != "study":
            raise CommandError("study_only_no_freeplay", 403)
        if payload.get("consent") is not True:
            raise CommandError("explicit_consent_required")
        op = payload.get("operation_id")
        if type(op) is not str or not 1 <= len(op) <= 128:
            raise CommandError("operation_id_required")
        request_hash = digest(payload)
        with closing(self.connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                s = self._session(db, sid)
                old = db.execute("SELECT * FROM operations WHERE session_id=? AND id=?", (sid, op)).fetchone()
                if old:
                    if old["digest"] != request_hash:
                        raise CommandError("operation_id_reused", 409, self._view(db, sid))
                    if self._study_mismatch(s):
                        raise CommandError("study_version_changed", 409)
                    return self._view(db, sid)
                if type(payload.get("expected_version")) is not int or payload["expected_version"] != s["version"]:
                    raise CommandError("state_version_conflict", 409, self._view(db, sid))
                if s["mode"] != "enrollment":
                    raise CommandError("study_cannot_restart_or_switch_mode", 403)
                participant = payload.get("participant_id", "")
                if type(participant) is not str:
                    raise CommandError("invalid_participant_id")
                participant = participant.strip()
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{2,31}", participant):
                    raise CommandError("invalid_participant_id")
                if db.execute("SELECT id FROM sessions WHERE participant_key=?", (participant.lower(),)).fetchone():
                    raise CommandError("participant_id_taken", 409)
                position = db.execute("SELECT count(*) FROM sessions WHERE position IS NOT NULL").fetchone()[0]
                block_id = position // 4
                old_block = db.execute("SELECT allocation FROM blocks WHERE id=?", (block_id,)).fetchone()
                if old_block is None:
                    cells = [["A", "XY"], ["A", "YX"], ["B", "XY"], ["B", "YX"]]
                    random.SystemRandom().shuffle(cells)
                    db.execute("INSERT INTO blocks VALUES(?,?)", (block_id, canonical(cells)))
                else:
                    cells = json.loads(old_block[0])
                condition, order = cells[position % 4]
                db.execute("UPDATE sessions SET mode='study',stage='consent',participant_id=?,participant_key=?,position=?,condition=?,task_order=?,questionnaire_bank_signature=?,study_signature=?,version=version+1 WHERE id=?",
                    (participant, participant.lower(), position, condition, order,
                     self.question_bank.signature if self.question_bank is not None else None, self.signature, sid))
                version = self._session(db, sid)["version"]
                db.execute("INSERT INTO operations VALUES(?,?,?,?,?)", (sid, op, request_hash, version, "start"))
                db.execute("INSERT INTO events(session_id,kind,payload) VALUES(?,'start',?)", (sid, canonical(payload)))
                result = self._view(db, sid)
                db.commit()
                return result
            except BaseException:
                db.rollback()
                raise


DEFAULT_PORT = 8012


def main(argv=None):
    """Serve only a completed independently accepted release on local loopback.

    No verification context or fixture flag is accepted here. Signals interrupt
    the main serve loop; calling HTTPServer.shutdown from that same thread would
    deadlock. Both the listener and the store are always closed before returning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--expected-manifest-sha', '--expected-manifest-sha256',
                        dest='expected_manifest_sha256', required=True)
    parser.add_argument('--db', '--database', dest='database', type=Path,
                        default=DEFAULT_DATABASE)
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    try:
        previous._sha(args.expected_manifest_sha256)
    except ValueError:
        parser.error('expected manifest hash must be a lowercase SHA256')

    store, server = None, None
    original_handlers = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        if not stopping:
            stopping = True
            raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, stop)
        # This constructor performs the existing full release verification.
        # There is deliberately no fallback to for_component_verification.
        store = AlignmentStudyStore(args.release_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
            database=args.database)
        server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                     previous.native.handler_class(store))
        # Drain in-flight HTTP handlers before releasing their store/context.
        server.daemon_threads = False
        print(f'PolicyLens alignment pilot: http://127.0.0.1:{args.port}/', flush=True)
        server.serve_forever(poll_interval=0.25)
        return 0
    except KeyboardInterrupt:
        stopping = True
        return 0
    finally:
        stopping = True
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if store is not None:
                    store.close()
            finally:
                for signum, handler in original_handlers.items():
                    signal.signal(signum, handler)


if __name__ == '__main__':
    raise SystemExit(main())
