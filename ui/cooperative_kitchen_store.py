"""Transactional kitchen ledger. PostgreSQL in research; SQLite only explicit development/test."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import hashlib
import io
import json
import secrets
import time
from typing import Any

from sqlalchemy import (Column, Float, Integer, MetaData, String, Table, Text,
                        UniqueConstraint, create_engine, delete, insert, or_, select, update)
from sqlalchemy.pool import StaticPool


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class StudyError(ValueError):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status, self.code = status, code


metadata = MetaData()
lock_table = Table("kitchen_namespace_locks", metadata,
    Column("namespace", String(30), primary_key=True), Column("created", Float, nullable=False))
blocks = Table("kitchen_assignment_blocks", metadata,
    Column("namespace", String(30), primary_key=True), Column("block_index", Integer, primary_key=True),
    Column("cells", Text, nullable=False))
invitations = Table("kitchen_invitations", metadata,
    Column("id", String(64), primary_key=True), Column("namespace", String(30), nullable=False, index=True),
    Column("code_hash", String(64), unique=True, nullable=False), Column("participant", String(120), nullable=False),
    Column("condition", String(1)), Column("task_order", String(2)), Column("position", Integer),
    Column("active_run", String(64)), Column("created", Float, nullable=False),
    UniqueConstraint("namespace", "participant"), UniqueConstraint("namespace", "position"))
# Additive enrollment ledger: invitation rows and historical run documents stay intact.
participants = Table("kitchen_participants", metadata,
    Column("id", String(64), primary_key=True), Column("namespace", String(30), nullable=False, index=True),
    Column("participant_id", String(120), nullable=False), Column("participant_key", String(120), nullable=False),
    Column("condition", String(1)), Column("task_order", String(2)), Column("position", Integer),
    Column("active_run", String(64)), Column("legacy_invitation_id", String(64), unique=True),
    Column("created", Float, nullable=False), Column("updated", Float, nullable=False),
    UniqueConstraint("namespace", "participant_key"), UniqueConstraint("namespace", "position"))
runs = Table("kitchen_runs", metadata,
    Column("id", String(64), primary_key=True), Column("namespace", String(30), nullable=False, index=True),
    Column("token_hash", String(64), unique=True, nullable=False), Column("document", Text, nullable=False),
    Column("version", Integer, nullable=False), Column("created", Float, nullable=False), Column("updated", Float, nullable=False))
creation_receipts = Table("kitchen_creation_receipts", metadata,
    Column("namespace", String(30), primary_key=True), Column("operation_id", String(120), primary_key=True),
    Column("request_hash", String(64), nullable=False), Column("run_id", String(64), nullable=False),
    Column("token", String(120), nullable=False))
episodes = Table("kitchen_episodes", metadata,
    Column("id", String(64), primary_key=True), Column("run_id", String(64), nullable=False, index=True),
    Column("episode_index", Integer, nullable=False), Column("phase", String(30), nullable=False),
    Column("document", Text, nullable=False), UniqueConstraint("run_id", "episode_index"))
frames = Table("kitchen_frames", metadata,
    Column("episode_id", String(64), primary_key=True), Column("turn", Integer, primary_key=True),
    Column("snapshot", Text, nullable=False), Column("public", Text, nullable=False))
events = Table("kitchen_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True), Column("run_id", String(64), nullable=False, index=True),
    Column("episode_id", String(64)), Column("operation_id", String(120), nullable=False),
    Column("kind", String(60), nullable=False), Column("created", Float, nullable=False), Column("document", Text, nullable=False))
operations = Table("kitchen_operations", metadata,
    Column("run_id", String(64), primary_key=True), Column("operation_id", String(120), primary_key=True),
    Column("request_hash", String(64), nullable=False), Column("response", Text, nullable=False),
    Column("created", Float, nullable=False))
questions = Table("kitchen_questions", metadata,
    Column("id", String(64), primary_key=True), Column("run_id", String(64), nullable=False, index=True),
    Column("episode_id", String(64), nullable=False), Column("frame", Integer, nullable=False),
    Column("status", String(30), nullable=False, index=True), Column("lease_until", Float),
    Column("lease_token", String(64)), Column("attempts", Integer, nullable=False),
    Column("created", Float, nullable=False), Column("updated", Float, nullable=False), Column("document", Text, nullable=False))
surveys = Table("kitchen_surveys", metadata,
    Column("run_id", String(64), primary_key=True), Column("document", Text, nullable=False), Column("submitted", Float))
admin_receipts = Table("kitchen_admin_receipts", metadata,
    Column("namespace", String(30), primary_key=True), Column("operation_id", String(120), primary_key=True),
    Column("request_hash", String(64), nullable=False), Column("response", Text, nullable=False))


class KitchenStore:
    def __init__(self, database_url: str, namespace: str = "development", *, allow_sqlite: bool = False):
        if namespace not in {"development", "test", "pilot", "confirmatory"}:
            raise ValueError("Unknown kitchen data namespace")
        self.namespace = namespace
        if database_url.startswith("postgres://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+psycopg://" + database_url[len("postgresql://"):]
        is_sqlite = database_url.startswith("sqlite")
        if is_sqlite and (not allow_sqlite or namespace not in {"development", "test"}):
            raise ValueError("SQLite is restricted to explicit development/test use")
        if not is_sqlite and not database_url.startswith("postgresql+psycopg://"):
            raise ValueError("Use PostgreSQL for research persistence")
        options = {"pool_pre_ping": True}
        if is_sqlite:
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if database_url.endswith(":memory:"):
                options["poolclass"] = StaticPool
        else:
            options.update(pool_size=8, max_overflow=16, pool_timeout=30)
        self.engine = create_engine(database_url, **options)
        self.is_sqlite = is_sqlite
        metadata.create_all(self.engine)
        with self.transaction() as db:
            # Concurrent process startup must not race while creating the lock row.
            if is_sqlite:
                from sqlalchemy.dialects.sqlite import insert as lock_insert
            else:
                from sqlalchemy.dialects.postgresql import insert as lock_insert
            db.execute(lock_insert(lock_table).values(namespace=namespace, created=time.time()).on_conflict_do_nothing())
            self.namespace_lock(db)
            self.migrate_invitations(db)

    def migrate_invitations(self, db):
        """Copy legacy allocation once, without moving a participant back after retry."""
        rows = db.execute(select(invitations).where(
                          invitations.c.namespace == self.namespace,
                          or_(invitations.c.position.is_not(None), invitations.c.condition.is_not(None),
                              invitations.c.task_order.is_not(None), invitations.c.active_run.is_not(None)))
                          .order_by(invitations.c.created, invitations.c.id)).mappings().all()
        for row in rows:
            if db.execute(select(participants.c.id).where(participants.c.legacy_invitation_id == row["id"])).first():
                continue
            key = row["participant"].strip().lower()
            collision = db.execute(select(participants.c.id).where(
                participants.c.namespace == self.namespace, participants.c.participant_key == key)).first()
            if collision:
                raise ValueError("Legacy kitchen participant IDs collide after normalization; researcher reconciliation is required")
            db.execute(insert(participants).values(id=secrets.token_hex(16), namespace=self.namespace,
                participant_id=row["participant"], participant_key=key, condition=row["condition"],
                task_order=row["task_order"], position=row["position"], active_run=row["active_run"],
                legacy_invitation_id=row["id"], created=row["created"], updated=time.time()))

    @contextmanager
    def transaction(self):
        with self.engine.connect() as db:
            if self.is_sqlite:
                db.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                db.begin()
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def namespace_lock(self, db):
        db.execute(select(lock_table).where(lock_table.c.namespace == self.namespace).with_for_update()).first()

    def run(self, db, token: str, *, locked=False) -> dict:
        stmt = select(runs).where(runs.c.token_hash == token_digest(token), runs.c.namespace == self.namespace)
        row = db.execute(stmt.with_for_update() if locked else stmt).mappings().first()
        if row is None:
            raise StudyError("Session not found", 401, "session_not_found")
        return json.loads(row["document"])

    def run_by_id(self, db, run_id: str, *, locked=False) -> dict:
        stmt = select(runs).where(runs.c.id == run_id, runs.c.namespace == self.namespace)
        row = db.execute(stmt.with_for_update() if locked else stmt).mappings().first()
        if row is None:
            raise StudyError("Session not found", 404, "session_not_found")
        return json.loads(row["document"])

    def save_run(self, db, run: dict):
        db.execute(update(runs).where(runs.c.id == run["id"]).values(document=encode(run), version=run["version"], updated=time.time()))

    @staticmethod
    def episode(db, episode_id: str, run_id: str) -> dict:
        row = db.execute(select(episodes.c.document).where(episodes.c.id == episode_id, episodes.c.run_id == run_id)).first()
        if row is None:
            raise StudyError("Episode not found", 404, "episode_not_found")
        return json.loads(row[0])

    @staticmethod
    def save_episode(db, episode: dict):
        if db.execute(select(episodes.c.id).where(episodes.c.id == episode["id"])).first():
            db.execute(update(episodes).where(episodes.c.id == episode["id"]).values(document=encode(episode)))
        else:
            db.execute(insert(episodes).values(id=episode["id"], run_id=episode["run_id"], episode_index=episode["index"], phase=episode["phase"], document=encode(episode)))

    @staticmethod
    def save_frame(db, episode_id: str, snapshot: dict, public: dict):
        db.execute(insert(frames).values(episode_id=episode_id, turn=public["turn"], snapshot=encode(snapshot), public=encode(public)))

    @staticmethod
    def frame(db, episode_id: str, turn: int):
        row = db.execute(select(frames).where(frames.c.episode_id == episode_id, frames.c.turn == turn)).mappings().first()
        if row is None:
            raise StudyError("Frame not found", 404, "frame_not_found")
        return json.loads(row["snapshot"]), json.loads(row["public"])

    @staticmethod
    def event(db, run_id, episode_id, operation_id, kind, document):
        db.execute(insert(events).values(run_id=run_id, episode_id=episode_id, operation_id=operation_id, kind=kind, created=time.time(), document=encode(document)))

    @staticmethod
    def validate_operation(operation_id):
        if not isinstance(operation_id, str) or not 1 <= len(operation_id) <= 120:
            raise StudyError("A unique operation_id of 1–120 characters is required")

    def receipt(self, db, run, payload):
        self.validate_operation(payload.get("operation_id"))
        row = db.execute(select(operations).where(operations.c.run_id == run["id"], operations.c.operation_id == payload["operation_id"])).mappings().first()
        if row:
            if row["request_hash"] != digest(payload):
                raise StudyError("Operation ID reused for another request", 409, "operation_conflict")
            return json.loads(row["response"])
        if type(payload.get("version")) is not int or payload["version"] != run["version"]:
            raise StudyError("State changed; refresh and retry with a new operation ID", 409, "version_conflict")
        return None

    def record_receipt(self, db, run, payload, response):
        db.execute(insert(operations).values(run_id=run["id"], operation_id=payload["operation_id"], request_hash=digest(payload), response=encode(response), created=time.time()))

    def export(self, format: str = "jsonl") -> str:
        with self.transaction() as db:
            run_rows = [json.loads(r[0]) for r in db.execute(select(runs.c.document).where(runs.c.namespace == self.namespace))]
            if format == "csv":
                out = io.StringIO()
                columns = ["participant_id", "run_id", "retry_id", "namespace", "mode", "enrollment_mode", "condition", "task_order", "phase", "task1_mean_score", "task2_mean_score", "task2_orders", "task2_completion_rate", "prediction_accuracy", "cooperation_understanding", "predictability", "difficulty"]
                writer = csv.DictWriter(out, fieldnames=columns)
                writer.writeheader()
                for run in run_rows:
                    eps = [json.loads(r[0]) for r in db.execute(select(episodes.c.document).where(episodes.c.run_id == run["id"]))]
                    one = [e for e in eps if e["phase"] == "task1" and e["done"]]
                    two = [e for e in eps if e["phase"] == "task2" and e["done"]]
                    survey_row = db.execute(select(surveys.c.document,surveys.c.submitted).where(surveys.c.run_id == run["id"])).first()
                    survey = json.loads(survey_row[0]) if survey_row else {}
                    final_answers = survey.get("answers",{}) if survey_row and survey_row[1] is not None else {}
                    writer.writerow({"participant_id": run["participant_id"], "run_id": run["id"], "retry_id": run["retry_id"],
                        "namespace": self.namespace, "mode": run["mode"],
                        "enrollment_mode": run.get("enrollment_mode", "legacy_invitation" if run["mode"]=="pilot" else "freeplay"),
                        "condition": run["condition"], "task_order": run["task_order"], "phase": run["phase"], "task1_mean_score": sum(e["summary"]["score"] for e in one)/3 if len(one)==3 else "", "task2_mean_score": sum(e["summary"]["score"] for e in two)/3 if len(two)==3 else "", "task2_orders": sum(e["summary"]["orders"] for e in two) if len(two)==3 else "", "task2_completion_rate": sum(e["summary"]["completed"] for e in two)/3 if len(two)==3 else "", "prediction_accuracy": survey.get("prediction_accuracy", ""), "cooperation_understanding": final_answers.get("cooperation_understanding",final_answers.get("understanding","")), "predictability": final_answers.get("predictability",""), "difficulty": final_answers.get("difficulty","")})
                return out.getvalue()
            if format != "jsonl":
                raise StudyError("Export format must be jsonl or csv")
            output=[]
            for run in run_rows:
                output.append(encode({"type":"run", "namespace":self.namespace, "document":run}))
                for table, name in [(episodes,"episode"),(events,"event"),(questions,"question"),(surveys,"survey")]:
                    for row in db.execute(select(table).where(table.c.run_id == run["id"])).mappings():
                        safe=dict(row)
                        safe.pop("lease_token",None)
                        if "document" in safe: safe["document"]=json.loads(safe["document"])
                        output.append(encode({"type":name, **safe}))
                epids=[r[0] for r in db.execute(select(episodes.c.id).where(episodes.c.run_id==run["id"]))]
                if epids:
                    for row in db.execute(select(frames).where(frames.c.episode_id.in_(epids))).mappings():
                        output.append(encode({"type":"frame","episode_id":row["episode_id"],"turn":row["turn"],"snapshot":json.loads(row["snapshot"]),"public":json.loads(row["public"])}))
            return "\n".join(output)+("\n" if output else "")
