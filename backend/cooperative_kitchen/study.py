"""Authoritative study state machine and durable explanation jobs for the kitchen."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import threading
import time
from types import MappingProxyType

from sqlalchemy import case, func, insert, select, update
from env.cooperative_kitchen import CooperativeKitchen, KitchenConfig, ACTIONS, program_decision
from ui.cooperative_kitchen_store import (KitchenStore, StudyError, admin_receipts, blocks,
    creation_receipts, digest, encode, episodes, events, frames, participants, operations,
    questions, runs, surveys, token_digest)

DEFAULT_CONSENT = {
    "version": "kitchen-consent-internal-pilot-v3",
    "title": {"zh": "内部预实验参与说明与同意", "en": "Internal pilot information and consent"},
    "text": {
        "zh": "本次为内部预实验，用于检查人与 AI 队友合作任务的流程和体验。您将练习厨房操作，完成两阶段各三局任务并填写问卷。系统记录操作、任务状态、提问、回答展示与问卷；这些数据不纳入未来正式研究样本。解释服务将必要的去标识化问题与任务证据发送至 DeepSeek API，以生成行为解释。请勿在用户 ID 或问题中填写姓名、邮箱、电话等个人信息。参与自愿，可随时停止。研究负责人联系方式、数据保存期限及适用审批信息须在正式招募前由研究者补充。",
        "en": "This is an internal pilot to check the workflow and experience of cooperation with an AI teammate. You will practise the controls, complete three rounds in each of two tasks, and answer a questionnaire. Actions, task states, questions, answer exposure and questionnaire responses are recorded. These data will not be included in the future formal research sample. Necessary de-identified questions and task evidence are sent to DeepSeek API to generate behavior explanations. Do not enter names, email addresses, phone numbers or other personal information in your user ID or questions. Participation is voluntary and you may stop at any time. The researcher must supply contact, retention and applicable review information before formal recruitment."
    }
}


def build_default_consent(qa_configuration=None):
    """Create provider-bound internal pilot information; never rewrite saved consent."""
    config=copy.deepcopy(qa_configuration or {})
    consent=copy.deepcopy(DEFAULT_CONSENT)
    provider=config.get("provider")
    if not provider: return consent
    names={"deepseek":("DeepSeek API","DeepSeek API"),"qwen":("通义千问 API","Qwen API")}
    zh,en=names.get(provider,("所配置的云端问答服务","the configured cloud question-answering service"))
    consent["version"]+=f"-{provider}"
    consent["text"]["zh"]=consent["text"]["zh"].replace("DeepSeek API",zh)
    consent["text"]["en"]=consent["text"]["en"].replace("DeepSeek API",en)
    consent["qa_provider"]=provider
    consent["qa_configuration_sha256"]=digest(config)
    return consent


class ProgramBaseline:
    policy_kind = "program_baseline"
    checkpoint_id = "kitchen-development-program-v1"


PARTICIPANT_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{2,31}$"
ENROLLMENT_MODES = {"closed", "internal_pilot", "formal"}
INTERNAL_PILOT_ALLOWED_RELEASE_GAPS = {"extraction_gate", "remote_load_gate",
                                       "qa_model_snapshot_unpinned"}
DEFAULT_QA_LIMITS = MappingProxyType({"per_episode": 8, "per_run": 24,
                                   "per_namespace": 500, "min_interval_seconds": 2})


class KitchenStudy:
    def __init__(self, output, database_url, *, namespace="development", allow_sqlite=False,
                 policy=None, explainer=None, release=None, test_mode=False, workers=4,
                 enrollment_mode=None, allow_freeplay_qa=None, qa_limits=None):
        if test_mode and namespace != "test":
            raise ValueError("Fixture injection is restricted to the test namespace")
        if qa_limits is not None and not test_mode:
            raise ValueError("QA limit overrides are restricted to test_mode construction")
        if qa_limits is not None and (not isinstance(qa_limits, dict) or set(qa_limits) - set(DEFAULT_QA_LIMITS)):
            raise ValueError("Unknown QA limit configuration")
        limits = {**DEFAULT_QA_LIMITS, **(qa_limits or {})}
        if any(type(limits[key]) is not int or limits[key] < 1
               for key in ("per_episode", "per_run", "per_namespace")):
            raise ValueError("QA count limits must be positive integers")
        interval = limits["min_interval_seconds"]
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval < 0:
            raise ValueError("QA minimum interval must be a finite nonnegative number")
        self.qa_limits = MappingProxyType(limits)
        self.enrollment_mode = enrollment_mode if enrollment_mode is not None else os.environ.get("KITCHEN_ENROLLMENT_MODE", "closed")
        if self.enrollment_mode not in ENROLLMENT_MODES:
            raise ValueError("KITCHEN_ENROLLMENT_MODE must be closed, internal_pilot, or formal")
        self.output = Path(output)
        self.store = KitchenStore(database_url, namespace, allow_sqlite=allow_sqlite)
        self.policy = policy or ProgramBaseline()
        self.explainer = explainer
        self.test_mode = test_mode
        self.allow_freeplay_qa = (namespace in {"development", "test"}
            if allow_freeplay_qa is None else bool(allow_freeplay_qa))
        self.release = copy.deepcopy(release or {})
        self.qa_configuration=copy.deepcopy(self.release.get("qa_configuration",{}))
        self.versions = copy.deepcopy(self.release.get("versions", {"environment": "cooperative-kitchen-v1", "ui": "cooperative_kitchen_web_v3_id_pilot", "protocol": "cooperative_kitchen_user_id_pilot_v3", "policy": getattr(self.policy, "checkpoint_id", "unknown")}))
        if self.qa_configuration: self.versions["qa_configuration_sha256"]=digest(self.qa_configuration)
        self.workers = max(1, min(int(workers), 20))
        self._stop = threading.Event()
        self._thread = None
        self._pool = None
        supplied_consent = self.release.get("consent")
        valid_consent = (isinstance(supplied_consent,dict) and isinstance(supplied_consent.get("version"),str)
            and bool(supplied_consent["version"].strip())
            and all(isinstance(supplied_consent.get(key),dict)
                and all(isinstance(supplied_consent[key].get(lang),str) and bool(supplied_consent[key][lang].strip()) for lang in ("zh","en"))
                for key in ("title","text")))
        if valid_consent and self.qa_configuration:
            valid_consent=(supplied_consent.get("qa_provider")==self.qa_configuration.get("provider")
                and supplied_consent.get("qa_configuration_sha256")==digest(self.qa_configuration))
        if valid_consent and self.enrollment_mode == "internal_pilot":
            valid_consent = supplied_consent["version"].startswith("kitchen-consent-internal-pilot-")
        self.consent = copy.deepcopy(supplied_consent if valid_consent else build_default_consent(self.qa_configuration))
        self._rating_items = self._validated_scales()

    def _validated_scales(self):
        scales=self.release.get("survey_scales")
        scale_range=self.release.get("scale_range")
        anchors=self.release.get("scale_anchors")
        expected={"cooperation_understanding","predictability","difficulty"}
        if not isinstance(scales,list) or len(scales)!=3: return None
        if not all(isinstance(item,dict) and isinstance(item.get("id"),str) for item in scales): return None
        if {item["id"] for item in scales} != expected: return None
        if not isinstance(scale_range,list) or scale_range!=[1,7] or not all(type(value) is int for value in scale_range): return None
        if not isinstance(anchors,dict) or not all(isinstance(anchors.get(lang),list) and len(anchors[lang])==2
            and all(isinstance(value,str) and value.strip() for value in anchors[lang]) for lang in ("zh","en")): return None
        result=[]
        for item in scales:
            prompt=item.get("prompt")
            if not isinstance(prompt,dict) or not all(isinstance(prompt.get(lang),str) and prompt[lang].strip() for lang in ("zh","en")): return None
            options=[{"value":str(n),"label":{lang:f"{n}"+(f" · {anchors[lang][0]}" if n==1 else f" · {anchors[lang][1]}" if n==7 else "") for lang in ("zh","en")}} for n in range(1,8)]
            result.append({"id":item["id"],"type":"likert","prompt":copy.deepcopy(prompt),"options":options})
        return result

    def status(self):
        missing = list(self.release.get("missing_configuration", []))
        if not self.release.get("study_ready", False): missing.append("validated_release")
        if getattr(self.policy, "policy_kind", "neural") == "program_baseline": missing.append("validated_neural_policy")
        if self.explainer is None: missing.append("explanation_backend")
        if self.store.namespace not in {"pilot", "confirmatory", "test"}: missing.append("research_database_namespace")
        if self.store.is_sqlite and self.store.namespace != "test": missing.append("postgresql")
        if not self.release.get("scenarios"): missing.append("calibrated_scenario_pairs")
        if len(self.release.get("question_bank", [])) != 8: missing.append("frozen_question_bank")
        if self._rating_items is None: missing.append("frozen_survey_scales")
        if self.consent.get("version", "").startswith(("kitchen-consent-draft-", "kitchen-consent-internal-pilot-")):
            missing.append("researcher_participant_information")
        if self.release.get("qa_configured") and self.qa_configuration.get("model_version_pinned") is not True:
            missing.append("qa_model_snapshot_unpinned")
        # Formal enrollment has its own data boundary.  A release that passes
        # every model/protocol gate must still fail closed while it points at
        # the pilot namespace, otherwise changing only the enrollment mode can
        # silently mix confirmatory samples with pilot records.
        if not self.test_mode and self.store.namespace != "confirmatory":
            missing.append("formal_database_namespace")
        if self.test_mode:
            missing = [] if self.release.get("study_ready") else ["fixture_release_not_ready"]
        formal_ready = not missing
        enabled = (self.enrollment_mode == "formal" and formal_ready) or (
            self.enrollment_mode == "internal_pilot" and not self._internal_pilot_missing())
        return {"policy_kind": getattr(self.policy, "policy_kind", "neural"), "study_ready": formal_ready,
                "enrollment": {"mode": self.enrollment_mode, "enabled": enabled,
                    "formal_ready": formal_ready, "participant_id_pattern": PARTICIPANT_ID_PATTERN,
                    "participant_id_example": "user_01"},
                "missing_configuration": sorted(set(missing)), "versions": self.versions,
                "namespace": self.store.namespace, "storage": "sqlite_development" if self.store.is_sqlite else "postgresql",
                "consent": self.consent, "test_mode": self.test_mode,
                "qa_configuration":copy.deepcopy(self.qa_configuration),
                "qa_configured":bool(self.release.get("qa_configured",False)),
                "qa_limits":dict(self.qa_limits),
                "freeplay_qa_enabled":self.allow_freeplay_qa,
                "qa_required_key_env":self.release.get("qa_required_key_env")}

    def _internal_pilot_missing(self):
        """Candidate audits may fail; the actual study components must still exist."""
        if self.test_mode:
            return []
        missing = []
        if self.store.namespace not in {"pilot", "test"}: missing.append("pilot_database_namespace")
        if self.store.is_sqlite and self.store.namespace != "test": missing.append("postgresql")
        if getattr(self.policy, "policy_kind", "neural") == "program_baseline": missing.append("neural_policy")
        if self.explainer is None: missing.append("explanation_backend")
        if not self.release.get("qa_configured", False): missing.append("qa_configuration")
        scenarios = self.release.get("scenarios", {})
        if not isinstance(scenarios, dict) or any(not isinstance(scenarios.get(k), list) or len(scenarios[k]) != 3 for k in ("X", "Y")):
            missing.append("scenario_pairs")
        if len(self.release.get("question_bank", [])) != 8: missing.append("question_bank")
        if self._rating_items is None: missing.append("survey_scales")
        # Internal pilots may proceed with the two explicitly documented
        # research-performance gaps. Integrity, runtime, artifact, credential
        # and LLM configuration bindings still fail closed.
        missing.extend(item for item in self.release.get("missing_configuration", [])
            if item not in INTERNAL_PILOT_ALLOWED_RELEASE_GAPS)
        return missing

    def _check_release(self, run):
        if run["mode"] != "freeplay":
            # Closing enrollment stops new registrations, not confirmed sessions.
            # Legacy runs remain subject to the original formal gate.
            ready = not self._internal_pilot_missing() if run.get("enrollment_mode") == "internal_pilot" else self.status()["study_ready"]
            if not ready:
                raise StudyError("Research release is not validated or configured", 503, "study_not_ready")
        if run.get("versions") != self.versions:
            raise StudyError("This session belongs to another release. Contact the researcher; historical data are retained.", 409, "release_changed")

    def create_invitations(self, payload):
        raise StudyError("Invitation enrollment has been retired; participants enter a user ID", 410, "invitations_retired")

    def _assign(self, db, participant):
        if participant["condition"]:
            return participant["condition"], participant["task_order"]
        assigned = list(db.execute(select(participants.c.position).where(participants.c.namespace==self.store.namespace, participants.c.position.is_not(None))))
        position=max((row[0] for row in assigned), default=-1)+1
        block_index, offset=divmod(position,4)
        row=db.execute(select(blocks.c.cells).where(blocks.c.namespace==self.store.namespace,blocks.c.block_index==block_index)).first()
        if row: cells=json.loads(row[0])
        else:
            cells=[["A","XY"],["A","YX"],["B","XY"],["B","YX"]]
            random.SystemRandom().shuffle(cells)
            db.execute(insert(blocks).values(namespace=self.store.namespace,block_index=block_index,cells=encode(cells)))
        condition, order=cells[offset]
        db.execute(update(participants).where(participants.c.id==participant["id"]).values(condition=condition,task_order=order,position=position,updated=time.time()))
        return condition,order

    @staticmethod
    def _participant_id(value):
        if not isinstance(value, str) or re.fullmatch(PARTICIPANT_ID_PATTERN, value.strip()) is None:
            raise StudyError("Participant ID must contain 3–32 ASCII letters, digits, underscores or hyphens and start with a letter", 400, "participant_id_invalid")
        return value.strip(), value.strip().lower()

    def join(self, payload, *, existing_token=None):
        self.store.validate_operation(payload.get("operation_id"))
        mode=payload.get("mode","freeplay")
        language=payload.get("language","zh")
        if mode not in {"freeplay","pilot"} or language not in {"zh","en"}:
            raise StudyError("Unsupported mode or language")
        participant_id, participant_key = self._participant_id(payload.get("participant_id")) if mode == "pilot" else (None, None)
        with self.store.transaction() as db:
            self.store.namespace_lock(db)
            existing_run = None
            if existing_token:
                try:
                    existing_run = self.store.run(db, existing_token)
                except StudyError:
                    existing_token = None
                else:
                    if (existing_run["mode"] != mode or
                            mode == "pilot" and existing_run["participant_id"].strip().lower() != participant_key):
                        raise StudyError("This browser already has another participant session", 409, "participant_session_conflict")
            participant = db.execute(select(participants).where(participants.c.namespace == self.store.namespace,
                participants.c.participant_key == participant_key)).mappings().first() if mode == "pilot" else None
            old=db.execute(select(creation_receipts).where(creation_receipts.c.namespace==self.store.namespace,creation_receipts.c.operation_id==payload["operation_id"])).mappings().first()
            if old:
                if old["request_hash"] != digest(payload): raise StudyError("Operation ID reused",409,"operation_conflict")
                run=self.store.run_by_id(db,old["run_id"])
                self._check_release(run)
                if existing_run and existing_run["id"] != run["id"]:
                    raise StudyError("This browser already has another participant session", 409, "participant_session_conflict")
                if participant and participant["active_run"] != run["id"]:
                    raise StudyError("A technical retry superseded this registration; submit a new operation ID", 409, "session_superseded")
                current=db.execute(select(runs.c.token_hash).where(runs.c.id==run["id"])).scalar_one()
                if current != token_digest(old["token"]): raise StudyError("Session was resumed elsewhere; use a new operation ID",409,"session_rotated")
                return old["token"], self._view(db,run)
            if participant and existing_run is None:
                raise StudyError("This participant ID is already in use; use the original browser or contact the researcher", 409, "participant_id_taken")
            if participant:
                if participant["active_run"]:
                    run=self.store.run_by_id(db,participant["active_run"],locked=True)
                    self._check_release(run)
                    # A closed run's cookie is a one-time migration credential for
                    # its active technical retry.  Revoke it in the same
                    # transaction that issues the retry token so another client
                    # holding the old cookie cannot repeatedly rotate the active
                    # session token and lock out the participant.
                    if existing_run and existing_run["id"] != run["id"]:
                        if (existing_run.get("phase") != "technical_retry_closed" or
                                run.get("previous_run_id") != existing_run["id"]):
                            raise StudyError("This older session cannot resume the active technical retry",409,"session_superseded")
                        db.execute(update(runs).where(runs.c.id==existing_run["id"]).values(
                            token_hash=token_digest(secrets.token_urlsafe(32))))
                    current = db.execute(select(runs.c.token_hash).where(runs.c.id == run["id"])).scalar_one()
                    token = existing_token if existing_token and current == token_digest(existing_token) else secrets.token_urlsafe(32)
                    if current != token_digest(token):
                        db.execute(update(runs).where(runs.c.id==run["id"]).values(token_hash=token_digest(token)))
                    db.execute(insert(creation_receipts).values(namespace=self.store.namespace,operation_id=payload["operation_id"],request_hash=digest(payload),run_id=run["id"],token=token))
                    if existing_run and existing_run["id"] != run["id"]:
                        self.store.event(db,run["id"],None,payload["operation_id"],"technical_retry_resumed",{
                            "previous_run_id":existing_run["id"]})
                    return token,self._view(db,run)
            if mode == "pilot":
                if self.enrollment_mode == "closed":
                    raise StudyError("Participant enrollment is closed", 403, "enrollment_closed")
                if not self.status()["enrollment"]["enabled"]:
                    raise StudyError("The selected enrollment mode is not ready or configured", 503, "study_not_ready")
                participant = {"id": secrets.token_hex(16), "namespace": self.store.namespace,
                    "participant_id": participant_id, "participant_key": participant_key,
                    "condition": None, "task_order": None, "created": time.time(), "updated": time.time()}
                db.execute(insert(participants).values(**participant))
            condition,order=self._assign(db,participant) if participant else (None,None)
            token=secrets.token_urlsafe(32)
            run={"id":secrets.token_hex(16),"participant_id":participant["participant_id"] if participant else "D-"+secrets.token_hex(5),
                 "mode":mode,"enrollment_mode":self.enrollment_mode if participant else "freeplay",
                 "namespace":self.store.namespace,"phase":"consent" if participant else "instructions",
                 "language":language,"condition":condition,"task_order":order,"version":0,
                 "episode_index":-1,"episode_id":None,"retry_id":0,"versions":copy.deepcopy(self.versions),
                 "created":time.time(),"consent":None,"preset":"supply"}
            db.execute(insert(runs).values(id=run["id"],namespace=self.store.namespace,token_hash=token_digest(token),document=encode(run),version=0,created=time.time(),updated=time.time()))
            if participant: db.execute(update(participants).where(participants.c.id==participant["id"]).values(active_run=run["id"],updated=time.time()))
            db.execute(insert(creation_receipts).values(namespace=self.store.namespace,operation_id=payload["operation_id"],request_hash=digest(payload),run_id=run["id"],token=token))
            self.store.event(db,run["id"],None,payload["operation_id"],"run_created",{"mode":mode,"enrollment_mode":run["enrollment_mode"],"condition":condition,"task_order":order,"versions":self.versions})
            return token,self._view(db,run)

    def _env(self, snapshot=None, scenario_id="base_empty", preset="supply", seed=0):
        env=CooperativeKitchen(KitchenConfig(horizon=180,target_orders=2), seed=seed,scenario_id=scenario_id)
        if snapshot: env.restore(snapshot)
        elif preset=="cook":
            snap=env.snapshot()
            # Environment supplies the same role-swap option as the demo when available.
            if hasattr(env,"reset"): env.reset(seed=0,scenario_id=scenario_id)
            if hasattr(env,"swap_roles"): env.swap_roles()
            else:
                state=snap.get("state",snap)
                a,b=state["actors"]
                for key in ("position","side"):
                    a[key],b[key]=b[key],a[key]
                state["preset"]="cook"
                env.restore(snap)
        return env

    def _new_episode(self, db, run, phase, scenario):
        descriptor=scenario if isinstance(scenario,dict) else {"scenario_id":scenario,"seed":0}
        env=self._env(scenario_id=descriptor["scenario_id"],seed=descriptor.get("seed",0),preset=run.get("preset","supply") if run["mode"]=="freeplay" else "supply")
        index=run["episode_index"]+1
        ep={"id":secrets.token_hex(16),"run_id":run["id"],"index":index,"phase":phase,"scenario_id":descriptor["scenario_id"],"scenario":descriptor,
            "done":False,"summary":None,"snapshot":env.snapshot(),"attempt_id":secrets.token_hex(16),"created":time.time(),"versions":copy.deepcopy(self.versions)}
        self.store.save_episode(db,ep)
        self.store.save_frame(db,ep["id"],env.snapshot(),env.public_view())
        run.update(phase=phase,episode_id=ep["id"],episode_index=index)
        return ep

    def _scenario(self, run, phase, number):
        groups=self.release.get("scenarios", {"X":["base_empty"]*3,"Y":["base_empty"]*3})
        group=run["task_order"][0 if phase=="task1" else 1]
        return groups[group][number]

    def _permission(self, run, ep=None):
        # Legacy or previous-release frames remain replayable, but cannot be
        # explained with the newly loaded Actor/program as if it acted then.
        if ep is not None and ep.get("versions") != self.versions: return False
        if run.get("versions") != self.versions: return False
        if run["mode"]=="freeplay": return self.allow_freeplay_qa
        return run["condition"]=="A" and run["phase"]=="task1" and (ep is None or ep["phase"]=="task1")

    def _public_question(self, row, version):
        doc=json.loads(row["document"])
        answer=doc.get("answer") if row["status"] in {"complete","failed"} else None
        if answer:
            answer={k:v for k,v in answer.items() if k in {"title","text","frame","kind","verified","source_summary","assumptions","clarification"}}
            answer.setdefault("source_summary", {"zh":"所选帧、策略与可核验证据", "en":"Selected frame, policy and verified evidence"}[doc["language"]])
        return {"id":row["id"],"status":row["status"],"episode_id":row["episode_id"],"frame":row["frame"],"kind":doc["kind"],"question":doc["question"],"answer":answer,"version":version}

    def _survey_items(self):
        items=[]
        for item in self.release.get("question_bank",[]):
            items.append({k:copy.deepcopy(v) for k,v in item.items() if k in {"id","type","prompt","options","state","frame","assumption"}})
        if self._rating_items is not None:
            items.extend(copy.deepcopy(self._rating_items))
        elif self.store.namespace in {"development","test"}:
            for key,zh,en in [("understanding","我理解如何与队友合作。","I understand how to cooperate with the teammate."),("predictability","我可以预测队友接下来的行为。","I can predict the teammate's next behavior."),("difficulty","我觉得任务很困难。","I found the task difficult.")]:
                items.append({"id":key,"type":"likert","prompt":{"zh":zh,"en":en},"options":[{"value":str(n),"label":{"zh":f"{n}"+(" · 非常不同意" if n==1 else " · 非常同意" if n==7 else ""),"en":f"{n}"+(" · Strongly disagree" if n==1 else " · Strongly agree" if n==7 else "")}} for n in range(1,8)]})
        else:
            raise StudyError("Frozen questionnaire scales are missing or invalid",503,"survey_artifact_invalid")
        return items

    def _view(self, db, run):
        eps=[json.loads(row[0]) for row in db.execute(select(episodes.c.document).where(episodes.c.run_id==run["id"]).order_by(episodes.c.episode_index))]
        current=next((ep for ep in eps if ep["id"]==run["episode_id"]),None)
        if run["mode"]=="freeplay" and run.get("versions") != self.versions:
            return {"run":{k:run[k] for k in ["id","participant_id","phase","language","mode","version","episode_index","episode_id"]},
                "state":None,"episodes":[{k:ep[k] for k in ["id","index","phase","done","summary"]} for ep in eps],
                "can_ask":False,"can_act":False,"can_auto":False,"can_next":False,"can_restart":True,"can_swap":False,"questions":[],"survey":None,
                "consent":self.consent,"completion_code":None,"policy_kind":getattr(self.policy,"policy_kind","neural"),"requires_restart":True,
                "notice":{"zh":"试玩版本已更新，请重新开始。历史记录已保留。","en":"The freeplay version has changed. Restart to use it; previous records are retained."}}
        state=self._env(current["snapshot"]).public_view() if current else None
        if state: state["episode_id"]=current["id"]
        can_ask=self._permission(run,current) and current is not None
        shown=[]
        if can_ask:
            permitted_ids=[item["id"] for item in eps if self._permission(run,item)]
            shown=[self._public_question(row,run["version"]) for row in db.execute(select(questions).where(questions.c.run_id==run["id"],questions.c.episode_id.in_(permitted_ids)).order_by(questions.c.created)).mappings()]
        survey=None
        if run["phase"] in {"questionnaire","complete"}:
            row=db.execute(select(surveys.c.document).where(surveys.c.run_id==run["id"])).first()
            document=json.loads(row[0]) if row else {}
            survey={"items":self._survey_items(),"draft":document.get("answers",{})}
        can_next=run["phase"]=="instructions" or bool(current and (current["done"] or (run["phase"]=="practice" and state["orders"]>=1)))
        if run["mode"]=="freeplay" and current: can_next=False
        return {"run":{k:run[k] for k in ["id","participant_id","phase","language","mode","version","episode_index","episode_id"]},
                "state":state,"episodes":[{k:ep[k] for k in ["id","index","phase","done","summary"]} for ep in eps],
                "can_ask":can_ask,"can_act":bool(current and not current["done"] and run["phase"] in {"practice","task1","task2","freeplay"}),
                "can_auto":bool(run["mode"]=="freeplay" and current and not current["done"] and run["phase"]=="freeplay"),
                "can_next":can_next,"can_restart":run["mode"]=="freeplay", "can_swap":run["mode"]=="freeplay",
                "questions":shown,"survey":survey,"consent":self.consent,
                "completion_code":run.get("completion_code"),"policy_kind":getattr(self.policy,"policy_kind","neural")}

    def view(self, token):
        with self.store.transaction() as db:
            run=self.store.run(db,token,locked=True)
            # A closed predecessor is exposed only as the technical-retry
            # handoff page. This lets a legacy cookie claim the audited new run;
            # commands, history and explanations remain version-gated.
            if run.get("phase")!="technical_retry_closed" and (run["mode"]!="freeplay" or run.get("versions")==self.versions):
                self._check_release(run)
            return self._view(db,run)

    def command(self, token, payload):
        with self.store.transaction() as db:
            run=self.store.run(db,token,locked=True)
            release_restart=run["mode"]=="freeplay" and run.get("versions")!=self.versions and payload.get("command")=="restart"
            if not release_restart: self._check_release(run)
            previous=self.store.receipt(db,run,payload)
            if previous is not None: return self._view(db,run)
            cmd=payload.get("command")
            if release_restart:
                self.store.event(db,run["id"],run["episode_id"],payload["operation_id"],"release_restart",{"previous_versions":run["versions"],"new_versions":self.versions})
                run.setdefault("release_history",[]).append(copy.deepcopy(run["versions"]))
                run["versions"]=copy.deepcopy(self.versions)
            ep=self.store.episode(db,run["episode_id"],run["id"]) if run["episode_id"] else None
            if cmd=="language":
                if payload.get("language") not in {"zh","en"}: raise StudyError("Unsupported language")
                run["language"]=payload["language"]
            elif cmd=="consent":
                if run["phase"]!="consent" or payload.get("accepted") is not True: raise StudyError("Explicit consent is required",403,"consent_required")
                run["consent"]={"version":self.consent["version"],"accepted":True,"at":time.time()}
                run["phase"]="instructions"
            elif cmd=="next":
                phase=run["phase"]
                if phase=="instructions": self._new_episode(db,run,"freeplay" if run["mode"]=="freeplay" else "practice","base_empty")
                elif phase=="practice" and ep and (ep["done"] or self._env(ep["snapshot"]).public_view()["orders"]>=1):
                    if not ep["done"]:
                        ep["done"]=True; ep["summary"]=self._summary(self._env(ep["snapshot"]).public_view(),"practice_finished"); self.store.save_episode(db,ep)
                    self._new_episode(db,run,"task1",self._scenario(run,"task1",0))
                elif phase in {"task1","task2"} and ep and ep["done"]:
                    count=db.execute(select(episodes.c.id).where(episodes.c.run_id==run["id"],episodes.c.phase==phase)).all()
                    if len(count)<3: self._new_episode(db,run,phase,self._scenario(run,phase,len(count)))
                    elif phase=="task1":
                        self._new_episode(db,run,"task2",self._scenario(run,"task2",0))
                        self._cancel_questions(db,run["id"])
                    else: run["phase"]="questionnaire"
                else: raise StudyError("The current phase cannot be advanced",403,"phase_permission")
            elif cmd in {"restart","swap"}:
                if run["mode"]!="freeplay": raise StudyError("Research rounds cannot be restarted or swapped",403,"phase_permission")
                if ep and not ep["done"]:
                    _,previous_public=self.store.frame(db,ep["id"],ep["snapshot"]["turn"])
                    ep["done"]=True;ep["summary"]=self._summary(previous_public,"freeplay_restart"); self.store.save_episode(db,ep)
                if cmd=="swap": run["preset"]="cook" if run.get("preset")=="supply" else "supply"
                self._new_episode(db,run,"freeplay","base_empty")
            elif cmd in {"action","auto_step"}:
                if cmd=="auto_step" and run["mode"]!="freeplay": raise StudyError("Automatic demonstration is restricted to freeplay",403,"phase_permission")
                if ep is None or ep["done"] or run["phase"] not in {"practice","task1","task2","freeplay"}: raise StudyError("Actions are disabled in this phase",403,"phase_permission")
                env=self._env(ep["snapshot"])
                before=env.snapshot()
                human_program=program_decision(env,"human") if cmd=="auto_step" else None
                action=human_program["action"] if human_program else payload.get("action")
                if action not in ACTIONS: raise StudyError("Unknown kitchen action")
                if isinstance(self.policy,ProgramBaseline):
                    decision=program_decision(env,"ai")
                    ai_action=decision["action"]
                    distribution={"ai":{"chosen_action":ai_action,"policy_kind":"program_baseline","decision":decision}}
                else:
                    selected,distribution=self.policy.act(env.observations())
                    ai_action=selected["ai"]
                result=env.step({"human":action,"ai":ai_action})
                public=env.public_view()
                ep["snapshot"]=env.snapshot();ep["done"]=bool(public["done"])
                if ep["done"]: ep["summary"]=self._summary(public); ep["summary"]["first_delivery"]=env.snapshot().get("_first_serve_turn")
                self.store.save_episode(db,ep);self.store.save_frame(db,ep["id"],env.snapshot(),public)
                self.store.event(db,run["id"],ep["id"],payload["operation_id"],"joint_step",{"before":before,"after":env.snapshot(),"human_command":action,"input_source":"program_demonstration" if cmd=="auto_step" else "participant","human_program_decision":human_program,"distributions":distribution,"proposed_actions":result.get("proposed_actions",{"human":action,"ai":ai_action}),"actual_actions":result.get("actual_actions"),"events":result["events"],"versions":self.versions})
            elif cmd in {"survey_save","survey_submit"}:
                if run["phase"]!="questionnaire": raise StudyError("Questionnaire is unavailable",403,"phase_permission")
                answers=payload.get("answers")
                if not isinstance(answers,dict): raise StudyError("Answers must be an object")
                items=self._survey_items();allowed={i["id"]:{str(o["value"]) for o in i["options"]} for i in items}
                for key,value in answers.items():
                    if key not in allowed or str(value) not in allowed[key]: raise StudyError("Invalid questionnaire response")
                row=db.execute(select(surveys.c.document).where(surveys.c.run_id==run["id"])).first()
                document=json.loads(row[0]) if row else {"answers":{}}
                document["answers"].update({k:str(v) for k,v in answers.items()})
                submitted=None
                if cmd=="survey_submit":
                    if set(document["answers"]) != set(allowed): raise StudyError("Complete every questionnaire item")
                    keys={i["id"]:str(i["correct_answer"]) for i in self.release.get("question_bank",[]) if "correct_answer" in i}
                    document["prediction_accuracy"]=sum(document["answers"].get(k)==v for k,v in keys.items())/len(keys) if keys else None
                    for kind in ("prediction","counterfactual"):
                        subset={i["id"]:str(i["correct_answer"]) for i in self.release.get("question_bank",[]) if i.get("type")==kind and "correct_answer" in i}
                        document[f"{kind}_item_accuracy"]=sum(document["answers"].get(k)==v for k,v in subset.items())/len(subset) if subset else None
                    submitted=time.time();run["phase"]="complete";run["completion_code"]="KITCHEN-"+secrets.token_hex(4).upper()
                document["versions"]=self.versions
                if row: db.execute(update(surveys).where(surveys.c.run_id==run["id"]).values(document=encode(document),submitted=submitted))
                else: db.execute(insert(surveys).values(run_id=run["id"],document=encode(document),submitted=submitted))
            else: raise StudyError("Unknown command")
            run["version"]+=1;self.store.save_run(db,run)
            self.store.event(db,run["id"],run["episode_id"],payload["operation_id"],"command",{"command":cmd,"phase":run["phase"],"version":run["version"],"consent":run.get("consent") if cmd=="consent" else None})
            response=self._view(db,run)
            self.store.record_receipt(db,run,payload,{"version":run["version"],"command":cmd})
            return response

    @staticmethod
    def _summary(state, reason=None):
        return {"orders":state["orders"],"steps":state["turn"],"score":100*state["orders"]-state["turn"],"completed":state["orders"]>=2,"reason":reason or state.get("reason"),"first_delivery":state.get("first_delivery_turn")}

    def history(self, token, episode_id):
        with self.store.transaction() as db:
            run=self.store.run(db,token,locked=True);self._check_release(run)
            ep=self.store.episode(db,episode_id,run["id"])
            can_ask=self._permission(run,ep)
            states=[json.loads(r[0]) for r in db.execute(select(frames.c.public).where(frames.c.episode_id==episode_id).order_by(frames.c.turn))]
            qs=[self._public_question(row,run["version"]) for row in db.execute(select(questions).where(questions.c.episode_id==episode_id,questions.c.run_id==run["id"])).mappings()] if can_ask else []
            return {"episode_id":episode_id,"phase":ep["phase"],"frames":states,"questions":qs,"can_ask":can_ask,"version":run["version"]}

    def ask(self, token, payload):
        if not isinstance(payload.get("question", ""),str): raise StudyError("Question must be text")
        question=payload.get("question", "").strip()
        kind=payload.get("kind","why")
        if kind not in {"why","waiting","counterfactual","free","freeform"}: raise StudyError("Unknown question kind")
        if not question and kind not in {"why","waiting","counterfactual"}: raise StudyError("Enter a question")
        if len(question)>2000: raise StudyError("Question is too long")
        rejection = None
        with self.store.transaction() as db:
            # Use the same namespace -> run lock order as enrollment and retries.
            # Database locking also serializes admissions across server processes.
            self.store.namespace_lock(db)
            run=self.store.run(db,token,locked=True);self._check_release(run)
            ep=self.store.episode(db,str(payload.get("episode_id","")),run["id"])
            if not self._permission(run,ep): raise StudyError("Explanations are disabled in this phase",403,"question_permission")
            prior=self.store.receipt(db,run,payload)
            if prior:
                rejection = prior.get("rejection")
                if rejection is None:
                    row=db.execute(select(questions).where(questions.c.id==prior["id"],questions.c.run_id==run["id"])).mappings().one()
                    return self._public_question(row,run["version"])
            else:
                turn=payload.get("frame")
                if type(turn) is not int or turn<0: raise StudyError("A valid frame is required")
                snapshot,_=self.store.frame(db,ep["id"],turn)
                now = time.time()
                usage = self._question_usage(db, run, ep)
                rejection = self._question_rejection(usage, now)
                if rejection:
                    # No question text or snapshot is retained for rejected work.
                    self.store.record_receipt(db, run, payload, {"rejection": rejection})
                    self.store.event(db, run["id"], ep["id"], payload["operation_id"], "question_rejected",
                        {"code": rejection["code"], "scope": rejection["scope"], "frame": turn,
                         "version": run["version"], "usage": usage, "limits": dict(self.qa_limits)})
                else:
                    qid=secrets.token_hex(16)
                    doc={"question":question,"kind":kind,"language":run["language"],"snapshot":snapshot,"answer":None,"versions":self.versions,"requested_phase":run["phase"],"created":now}
                    db.execute(insert(questions).values(id=qid,run_id=run["id"],episode_id=ep["id"],frame=turn,status="pending",attempts=0,created=now,updated=now,document=encode(doc)))
                    run["version"]+=1;self.store.save_run(db,run)
                    response={"id":qid,"status":"pending","version":run["version"],"episode_id":ep["id"],"frame":turn,"kind":kind,"question":question,"answer":None}
                    self.store.record_receipt(db,run,payload,{"id":qid})
                    self.store.event(db,run["id"],ep["id"],payload["operation_id"],"question_queued",{"question_id":qid,"frame":turn,"question":question,"kind":kind})
                    return response
        # Raising inside the transaction would roll back the rejection receipt.
        raise StudyError(rejection["message"], rejection["status"], rejection["code"])

    def _question_usage(self, db, run, ep):
        """Count every admitted job, including failed/cancelled jobs and old releases."""
        same_run = questions.c.run_id == run["id"]
        row = db.execute(select(
            func.count().label("per_namespace"),
            func.count().filter(same_run).label("per_run"),
            func.count().filter(same_run, questions.c.episode_id == ep["id"]).label("per_episode"),
            func.count().filter(same_run, questions.c.status.in_(["pending", "running"])).label("pending"),
            func.max(case((same_run, questions.c.created), else_=None)).label("last_accepted_at")
        ).select_from(questions.join(runs, questions.c.run_id == runs.c.id))
            .where(runs.c.namespace == self.store.namespace)).mappings().one()
        return dict(row)

    def _question_rejection(self, usage, now):
        for scope in ("per_episode", "per_run", "per_namespace"):
            if usage[scope] >= self.qa_limits[scope]:
                label = scope.removeprefix("per_")
                return {"status": 429, "code": "question_budget_exhausted" if scope == "per_namespace" else f"question_{label}_limit", "scope": scope,
                        "message": f"The question limit for this {label} has been reached"}
        last = usage["last_accepted_at"]
        if last is not None and now - last < self.qa_limits["min_interval_seconds"]:
            return {"status": 429, "code": "question_rate_limit", "scope": "min_interval_seconds",
                    "message": "Please wait before submitting another question"}
        if usage["pending"] >= 2:
            return {"status": 429, "code": "question_limit", "scope": "pending",
                    "message": "Wait for the pending answer"}
        return None

    def question(self, token, question_id):
        with self.store.transaction() as db:
            run=self.store.run(db,token,locked=True);self._check_release(run)
            row=db.execute(select(questions).where(questions.c.id==question_id,questions.c.run_id==run["id"])).mappings().first()
            if row is None: raise StudyError("Question not found",404,"question_not_found")
            ep=self.store.episode(db,row["episode_id"],run["id"])
            if not self._permission(run,ep): raise StudyError("Explanations are disabled in this phase",403,"question_permission")
            return self._public_question(row,run["version"])

    def exposure(self, token, payload):
        self.store.validate_operation(payload.get("operation_id"))
        if payload.get("event") not in {"shown","closed"}: raise StudyError("Unknown exposure event")
        with self.store.transaction() as db:
            run=self.store.run(db,token,locked=True)
            self._check_release(run)
            row=db.execute(select(questions).where(questions.c.id==payload.get("question_id"),questions.c.run_id==run["id"])).mappings().first()
            if row is None: raise StudyError("Question not found",404,"question_not_found")
            ep=self.store.episode(db,row["episode_id"],run["id"])
            if not self._permission(run,ep): raise StudyError("Explanations are disabled in this phase",403,"question_permission")
            old=db.execute(select(operations).where(operations.c.run_id==run["id"],operations.c.operation_id==payload["operation_id"])).mappings().first()
            if old and old["request_hash"]!=digest(payload): raise StudyError("Operation ID reused",409,"operation_conflict")
            if not old:
                if row["status"] not in {"complete","failed"}: raise StudyError("The answer has not been delivered",409,"answer_pending")
                self.store.event(db,run["id"],ep["id"],payload["operation_id"],"answer_exposure",{"question_id":row["id"],"event":payload["event"],"frame":row["frame"]})
                self.store.record_receipt(db,run,payload,{"ok":True})
            return {"ok":True,"version":run["version"]}

    @staticmethod
    def _cancel_questions(db, run_id):
        db.execute(update(questions).where(questions.c.run_id==run_id,questions.c.status.in_(["pending","running"])).values(status="cancelled",lease_until=None,lease_token=None,updated=time.time()))

    def _claim_question(self):
        now=time.time()
        with self.store.transaction() as db:
            stmt=select(questions).join(runs,runs.c.id==questions.c.run_id).where(runs.c.namespace==self.store.namespace, ((questions.c.status=="pending") | ((questions.c.status=="running") & (questions.c.lease_until<now)))).order_by(questions.c.created).with_for_update(skip_locked=True,of=questions).limit(1)
            row=db.execute(stmt).mappings().first()
            if not row: return None
            row=dict(row);doc=json.loads(row["document"])
            queued_run=self.store.run_by_id(db,row["run_id"])
            queued_episode=self.store.episode(db,row["episode_id"],row["run_id"])
            if queued_run.get("versions") != self.versions or not self._permission(queued_run,queued_episode):
                db.execute(update(questions).where(questions.c.id==row["id"]).values(status="cancelled",lease_until=None,lease_token=None,updated=now))
                self.store.event(db,row["run_id"],row["episode_id"],row["id"],"question_cancelled",{"reason":"release_or_phase_changed"})
                return False
            if row["attempts"]>=3:
                doc["answer"]={"title":"解释暂不可用" if doc["language"]=="zh" else "Explanation unavailable",
                    "text":"服务恢复重试仍未完成，请重新提问。" if doc["language"]=="zh" else "Service recovery could not complete this answer. Please ask again.",
                    "verified":False,"frame":row["frame"],"kind":doc["kind"],"diagnostics":{"error_type":"lease_retries_exhausted"}}
                db.execute(update(questions).where(questions.c.id==row["id"]).values(status="failed",document=encode(doc),lease_until=None,lease_token=None,updated=now))
                self.store.event(db,row["run_id"],row["episode_id"],row["id"],"question_finished",{"status":"failed","reason":"lease_retries_exhausted"})
                return False
            lease=secrets.token_hex(16)
            db.execute(update(questions).where(questions.c.id==row["id"]).values(status="running",lease_until=now+300,lease_token=lease,attempts=row["attempts"]+1,updated=now))
        return row,doc,lease

    def _process_claimed_question(self, row, doc, lease):
        # Inference is deliberately outside every database transaction and run lock.
        try:
            if self.explainer is None:
                answer=self._development_answer(doc)
            else:
                answer=self.explainer.generate(copy.deepcopy(doc["snapshot"]),doc["question"],kind=doc["kind"],language=doc["language"],anchor="next")
            state="complete"
        except Exception as exc:
            answer={"title":"解释暂不可用" if doc["language"]=="zh" else "Explanation unavailable", "text":"暂时无法核验回答。您可以继续任务或稍后重试。" if doc["language"]=="zh" else "The answer could not be verified. You may continue or retry later.","verified":False,"frame":row["frame"],"kind":doc["kind"],"diagnostics":{"error_type":type(exc).__name__}}
            state="failed"
        with self.store.transaction() as db:
            run=self.store.run_by_id(db,row["run_id"],locked=True)
            current=db.execute(select(questions).where(questions.c.id==row["id"])).mappings().one()
            if current["lease_token"]!=lease or current["status"]!="running": return True
            ep=self.store.episode(db,row["episode_id"],run["id"])
            doc["answer"]=answer;doc["completed"]=time.time();doc["elapsed_seconds"]=doc["completed"]-row["created"]
            if not self._permission(run,ep) or run.get("versions")!=self.versions: state="cancelled"
            db.execute(update(questions).where(questions.c.id==row["id"]).values(status=state,document=encode(doc),lease_until=None,lease_token=None,updated=time.time()))
            self.store.event(db,run["id"],ep["id"],row["id"],"question_finished",{"question_id":row["id"],"status":state,"elapsed_seconds":doc["elapsed_seconds"],"diagnostics":answer.get("diagnostics",{})})
        return True

    def process_one_question(self):
        claim=self._claim_question()
        if claim is None: return False
        if claim is False: return True
        return self._process_claimed_question(*claim)

    def _development_answer(self, doc):
        env=self._env(doc["snapshot"])
        decision=program_decision(env,"ai")
        language=doc["language"];turn=env.public_view()["turn"]
        names={"UP":("向上","move up"),"DOWN":("向下","move down"),"LEFT":("向左","move left"),"RIGHT":("向右","move right"),"WAIT":("等待","wait"),"INTERACT":("交互","interact")}
        reasons={
            "serve_soup":("手中有汤，先送到出餐口。","It is carrying soup and prioritises serving it."),
            "collect_soup":("共享工作台上已有汤，先取走并出餐。","Soup is on a shared counter, so it prioritises collecting and serving it."),
            "handoff_soup":("手中有成汤，共享工作台有空位，准备交回。","It is carrying soup and a shared counter is free for handoff."),
            "handoff_onion":("手中有洋葱，当前锅还需要食材，准备放到空工作台。","It is carrying an onion needed by the current pot and will place it on a free counter."),
            "get_onion":("当前锅需要更多洋葱，且共享工作台有空位。","The current pot needs more onions and a shared counter is free."),
            "collect_onion":("共享工作台有洋葱，先取走用于入锅。","An onion is available on a shared counter to load into the pot."),
            "load_pot":("手中有洋葱，锅内还未装满三份。","It holds an onion and the pot contains fewer than three."),
            "get_plate":("汤已煮熟，需要先取盘子装汤。","The soup is ready and a plate is needed to collect it."),
            "get_counter_plate":("汤已煮熟，共享工作台上有可用盘子。","The soup is ready and a plate is available on a shared counter."),
            "plate_soup":("手中有盘子，锅内的汤已煮熟。","It holds a plate and the pot's soup is ready."),
            "wait_space":("两张共享工作台都被占用，需要先腾出交接空间。","Both shared counters are occupied; handoff space must be cleared first."),
            "wait_cooking":("锅正在烹饪，等待后续联合步使汤煮熟。","The pot is cooking and needs subsequent joint steps to become ready."),
            "wait_onion":("当前没有可取的洋葱，等待左侧交接食材。","No onion is available to collect; it is waiting for an ingredient handoff."),
            "wait_pickup":("共享工作台被占用，等待物品被取走。","The shared counters are occupied; it is waiting for an item to be collected."),
            "wait_soup":("当前已补足食材，等待烹饪和汤的交回。","Enough ingredients have been supplied; it is waiting for cooking and soup handoff."),
            "wait_soup_handoff":("右侧拿着成汤，等待汤被放到共享工作台。","The right-side chef holds soup; it is waiting for the soup handoff."),
            "clear_for_handoff":("右侧拿着成汤，需要先清理共享工作台。","The right-side chef holds soup and a shared counter needs clearing."),
            "clear_extra_onion":("需要先取走多余洋葱，腾出共享工作台。","An excess onion must be collected to free shared-counter space."),
            "clear_plate":("共享工作台上的盘子占用了交接位置，准备取走。","A plate occupies shared-counter space and is selected for collection."),
            "discard_plate":("当前手中的盘子不能用于眼前任务，准备送到垃圾桶。","Its held plate is not usable for the current task, so it will take it to a bin."),
            "discard_extra_onion":("当前食材已足够，准备把多余洋葱送到垃圾桶。","Enough ingredients are supplied, so it will take the excess onion to a bin."),
            "discard_for_handoff":("右侧正在交回成汤，先处理手中洋葱以便接汤。","The right-side chef is returning soup; it clears its held onion to receive it."),
            "finished":("本回合已经结束。","This round has ended.")}
        idx=0 if language=="zh" else 1
        reason=reasons.get(decision["rule"],("该程序分支没有额外解释。","No additional explanation is available for this program branch."))[idx]
        if doc["kind"]=="counterfactual":
            future=[]
            for _ in range(3):
                if env.public_view()["done"]: break
                d=program_decision(env,"ai")
                env.step({"human":"WAIT","ai":d["action"]})
                future.append({"turn":env.public_view()["turn"],"action":d["action"]})
            end=env.public_view()
            actions="、".join(names[f["action"]][0] for f in future) if language=="zh" else ", ".join(names[f["action"]][1] for f in future)
            text=(f"以第 {turn} 步的状态为起点，假设你接下来连续等待三步（若回合提前结束则停止）。程序队友依次：{actions or '无后续动作'}。模拟结束于第 {end['turn']} 步，已出餐 {end['orders']} 份。真实回合没有推进。" if language=="zh" else f"Starting from step {turn}, assume you wait for three consecutive steps, stopping if the round ends. The programmed teammate would: {actions or 'take no further actions'}. The simulation ends at step {end['turn']} with {end['orders']} orders served. The real round is unchanged.")
        else:
            text=(f"第 {turn} 步之后，程序队友的下一次动作是{names[decision['action']][0]}。{reason}" if language=="zh" else f"After step {turn}, the programmed teammate's next action is to {names[decision['action']][1]}. {reason}")
            if doc["kind"]=="waiting" and decision["action"]!="WAIT":
                text+=("它此刻没有原地等待。" if language=="zh" else "It is not currently waiting in place.")
            if doc["question"]:
                text+=("此处仅返回可核验的程序记录；自由问答服务尚未配置。" if language=="zh" else "Only the verified program record is shown here; free-text explanations are not configured.")
        return {"title":"程序决策记录" if language=="zh" else "Program decision record","text":text,"frame":turn,"kind":doc["kind"],"verified":True,"source_summary":"程序决策记录（开发试玩）" if language=="zh" else "Program decision record (development freeplay)"}

    def start_workers(self):
        if self._thread and self._thread.is_alive(): return
        if self.store.namespace == "development" and not self.allow_freeplay_qa: return
        self._stop.clear();self._pool=ThreadPoolExecutor(max_workers=self.workers,thread_name_prefix="kitchen-question")
        def loop():
            active=set();backoff=.25
            while not self._stop.is_set():
                active={future for future in active if not future.done()}
                if len(active)>=self.workers:
                    self._stop.wait(.05);continue
                claim=self._claim_question()
                if claim is None:
                    self._stop.wait(backoff);backoff=min(backoff*2,1.0);continue
                backoff=.25
                if claim is False: continue
                active.add(self._pool.submit(self._process_claimed_question,*claim))
        self._thread=threading.Thread(target=loop,name="kitchen-question-dispatch",daemon=True);self._thread.start()

    def stop_workers(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=3)
        if self._pool: self._pool.shutdown(wait=True,cancel_futures=True)

    def admin_status(self):
        with self.store.transaction() as db:
            data=[json.loads(r[0]) for r in db.execute(select(runs.c.document).where(runs.c.namespace==self.store.namespace))]
            return {"service":self.status(),"runs":[{k:r[k] for k in ["id","participant_id","phase","version","condition","task_order","retry_id","created"]} for r in data]}

    def technical_retry(self, payload):
        self.store.validate_operation(payload.get("operation_id"))
        reason=str(payload.get("reason", "")).strip()
        if not reason or len(reason)>1000: raise StudyError("An audit reason is required")
        with self.store.transaction() as db:
            self.store.namespace_lock(db)
            old=db.execute(select(admin_receipts).where(admin_receipts.c.namespace==self.store.namespace,admin_receipts.c.operation_id==payload["operation_id"])).mappings().first()
            if old:
                if old["request_hash"]!=digest(payload): raise StudyError("Operation ID reused",409,"operation_conflict")
                return json.loads(old["response"])
            previous=self.store.run_by_id(db,str(payload.get("run_id","")),locked=True)
            participant=db.execute(select(participants).where(participants.c.active_run==previous["id"],participants.c.namespace==self.store.namespace)).mappings().first()
            if participant is None: raise StudyError("An active participant registration is required")
            if previous.get("retry_id",0)>0:
                claimed=db.execute(select(events.c.id).where(events.c.run_id==previous["id"],
                    events.c.kind=="technical_retry_resumed").limit(1)).first()
                if claimed is None:
                    raise StudyError("The active technical retry has not been resumed by the participant",409,"technical_retry_not_resumed")
            previous_versions=copy.deepcopy(previous.get("versions"))
            previous_enrollment_mode=previous.get("enrollment_mode")
            next_enrollment_mode=previous_enrollment_mode if previous_enrollment_mode in {"internal_pilot","formal"} else (
                self.enrollment_mode if self.enrollment_mode in {"internal_pilot","formal"} else
                "internal_pilot" if self.store.namespace in {"pilot","test"} else "formal")
            self._cancel_questions(db,previous["id"])
            previous["phase"]="technical_retry_closed";previous["version"]+=1;self.store.save_run(db,previous)
            new=copy.deepcopy(previous);new.update(id=secrets.token_hex(16),phase="consent",episode_index=-1,episode_id=None,retry_id=previous["retry_id"]+1,version=0,created=time.time(),consent=None,previous_run_id=previous["id"],enrollment_mode=next_enrollment_mode,versions=copy.deepcopy(self.versions))
            db.execute(insert(runs).values(id=new["id"],namespace=self.store.namespace,token_hash=token_digest(secrets.token_urlsafe(32)),document=encode(new),version=0,created=time.time(),updated=time.time()))
            db.execute(update(participants).where(participants.c.id==participant["id"]).values(active_run=new["id"],updated=time.time()))
            version_audit={"previous_versions":previous_versions,"new_versions":self.versions,
                "previous_enrollment_mode":previous_enrollment_mode,"new_enrollment_mode":next_enrollment_mode}
            self.store.event(db,previous["id"],previous["episode_id"],payload["operation_id"],"technical_retry",{"new_run_id":new["id"],"reason":reason,**version_audit})
            self.store.event(db,new["id"],None,payload["operation_id"],"technical_retry_created",{"previous_run_id":previous["id"],"reason":reason,**version_audit})
            response={"run_id":new["id"],"retry_id":new["retry_id"],"participant_id":new["participant_id"]}
            db.execute(insert(admin_receipts).values(namespace=self.store.namespace,operation_id=payload["operation_id"],request_hash=digest(payload),response=encode(response)))
            return response
