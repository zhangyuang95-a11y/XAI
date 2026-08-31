"""Transactional persistence for the current collaborative Warehouse study.

SQLite is authoritative; JSONL is an export-only analysis format.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, Callable, Iterable, Mapping

from env.warehouse.contracts import SQLITE_SCHEMA_VERSION, STUDY_LOG_VERSION


ACTIVE_STAGES = {
    "instructions",
    "task1",
    "task1_complete",
    "task2",
    "survey",
}
TERMINAL_STAGES = {"completed", "abandoned"}
_PARTICIPANT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")


class StudyConflict(RuntimeError):
    """A recoverable stale, duplicate, or cross-session command conflict."""

    def __init__(self, message: str, *, code: str, current: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.current = dict(current or {})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_participant_id(value: str) -> tuple[str, str]:
    display = " ".join(str(value).strip().split())
    if not display:
        raise ValueError("Participant ID is required.")
    if not _PARTICIPANT_RE.fullmatch(display):
        raise ValueError(
            "Participant ID must contain 1-80 printable characters."
        )
    return display, display.casefold()


class StudyStore:
    """Small SQLite repository with idempotent command transactions."""

    sqlite_schema_version = SQLITE_SCHEMA_VERSION

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str,
        abandon_on_start: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace).strip() or "pilot"
        self._lock = threading.RLock()
        self._initialize()
        if abandon_on_start:
            self.abandon_interrupted_runs(reason="server_restart")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=15.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS participants (
                    namespace TEXT NOT NULL,
                    participant_key TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    assignment_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, participant_key)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    participant_key TEXT NOT NULL,
                    participant_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    browser_session_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    assignment_json TEXT NOT NULL,
                    tutorial_index INTEGER NOT NULL DEFAULT 0,
                    tutorial_max_index INTEGER NOT NULL DEFAULT 0,
                    tutorial_complete INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    abandoned_reason TEXT,
                    FOREIGN KEY(namespace, participant_key)
                        REFERENCES participants(namespace, participant_key)
                );
                CREATE INDEX IF NOT EXISTS idx_runs_participant
                    ON runs(namespace, participant_key, attempt);
                CREATE TABLE IF NOT EXISTS explanations (
                    run_id TEXT NOT NULL,
                    exposure_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, exposure_index),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS surveys (
                    run_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    namespace TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    command TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, operation_id)
                );
                CREATE TABLE IF NOT EXISTS pending_operations (
                    namespace TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    browser_session_id TEXT NOT NULL,
                    command TEXT NOT NULL,
                    expected_stage TEXT NOT NULL,
                    expected_version INTEGER NOT NULL,
                    accepted_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, operation_id),
                    FOREIGN KEY(run_id) REFERENCES runs(run_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, sequence)
                );
                """
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(self.sqlite_schema_version),),
            )

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        with self._lock:
            db = self._connect()
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            finally:
                db.close()

    def participant_assignment(self, participant_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT assignment_json FROM participants WHERE namespace=? AND participant_key=?",
                (self.namespace, participant_key),
            ).fetchone()
        return json.loads(row["assignment_json"]) if row else None

    def assignments(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT assignment_json FROM participants WHERE namespace=? ORDER BY created_at, participant_key",
                (self.namespace,),
            ).fetchall()
        return [json.loads(row["assignment_json"]) for row in rows]

    def run_assignments(self) -> list[dict[str, Any]]:
        """Return per-run assignments for the isolated development namespace."""

        with self._connect() as db:
            rows = db.execute(
                "SELECT assignment_json FROM runs WHERE namespace=? "
                "ORDER BY created_at, run_id",
                (self.namespace,),
            ).fetchall()
        return [json.loads(row["assignment_json"]) for row in rows]

    def ensure_participant(
        self,
        db: sqlite3.Connection,
        *,
        participant_id: str,
        participant_key: str,
        assignment: Mapping[str, Any],
    ) -> None:
        db.execute(
            """
            INSERT INTO participants(namespace, participant_key, participant_id, assignment_json, created_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(namespace, participant_key) DO NOTHING
            """,
            (
                self.namespace,
                participant_key,
                participant_id,
                json.dumps(dict(assignment), ensure_ascii=False, sort_keys=True),
                utc_now(),
            ),
        )

    def next_attempt(self, db: sqlite3.Connection, participant_key: str) -> int:
        row = db.execute(
            "SELECT COALESCE(MAX(attempt), 0) AS value FROM runs WHERE namespace=? AND participant_key=?",
            (self.namespace, participant_key),
        ).fetchone()
        return int(row["value"]) + 1

    def abandon_participant_runs(
        self,
        db: sqlite3.Connection,
        participant_key: str,
        *,
        reason: str,
    ) -> None:
        now = utc_now()
        db.execute(
            f"""
            UPDATE runs
            SET stage='abandoned', status='abandoned', abandoned_reason=?,
                state_version=state_version+1, updated_at=?
            WHERE namespace=? AND participant_key=?
              AND stage IN ({','.join('?' for _ in ACTIVE_STAGES)})
            """,
            (reason, now, self.namespace, participant_key, *sorted(ACTIVE_STAGES)),
        )

    def abandon_interrupted_runs(self, *, reason: str) -> int:
        with self.transaction() as db:
            rows = db.execute(
                f"SELECT run_id FROM runs WHERE namespace=? AND stage IN ({','.join('?' for _ in ACTIVE_STAGES)})",
                (self.namespace, *sorted(ACTIVE_STAGES)),
            ).fetchall()
            now = utc_now()
            for row in rows:
                run_id = str(row["run_id"])
                sequence = self._next_event_sequence(db, run_id)
                payload = {
                    "schema_version": STUDY_LOG_VERSION,
                    "timestamp": now,
                    "event": "study_abandoned",
                    "run_id": run_id,
                    "reason": reason,
                }
                db.execute(
                    "INSERT INTO events(namespace, run_id, sequence, payload_json, created_at) VALUES(?,?,?,?,?)",
                    (self.namespace, run_id, sequence, json.dumps(payload, ensure_ascii=False), now),
                )
            db.execute(
                f"""
                UPDATE runs SET stage='abandoned', status='abandoned', abandoned_reason=?,
                    state_version=state_version+1, updated_at=?
                WHERE namespace=? AND stage IN ({','.join('?' for _ in ACTIVE_STAGES)})
                """,
                (reason, now, self.namespace, *sorted(ACTIVE_STAGES)),
            )
            db.execute(
                "DELETE FROM pending_operations WHERE namespace=?",
                (self.namespace,),
            )
        return len(rows)

    def run_row(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def cached_operation(self, operation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT run_id, response_json FROM operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
        if row is None:
            return None
        response = json.loads(row["response_json"])
        response.setdefault("run_id", str(row["run_id"]))
        return response

    def reserve_long_operation(
        self,
        *,
        operation_id: str,
        run_id: str,
        browser_session_id: str,
        expected_stage: str,
        expected_version: int,
        command: str,
    ) -> dict[str, Any] | None:
        """Validate and reserve a slow command without holding a write lock.

        A completed replay returns its original response.  A genuinely
        concurrent duplicate gets a recoverable conflict and cannot execute a
        second model call or create a second exposure.
        """

        with self.transaction() as db:
            cached = db.execute(
                "SELECT response_json FROM operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
            if cached:
                response = json.loads(cached["response_json"])
                response.setdefault("run_id", run_id)
                return response
            pending = db.execute(
                "SELECT run_id FROM pending_operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
            if pending:
                raise StudyConflict(
                    "This operation is still being processed.",
                    code="operation_in_progress",
                    current=self._authoritative_view(db, run_id),
                )
            row = self._validate_run(
                db,
                run_id=run_id,
                browser_session_id=browser_session_id,
                expected_stage=expected_stage,
                expected_version=expected_version,
            )
            db.execute(
                """
                INSERT INTO pending_operations(namespace, operation_id, run_id,
                    browser_session_id, command, expected_stage,
                    expected_version, accepted_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    self.namespace, operation_id, run_id, browser_session_id,
                    command, expected_stage, int(expected_version), utc_now(),
                ),
            )
        return None

    def complete_long_operation(
        self,
        *,
        operation_id: str,
        run_id: str,
        browser_session_id: str,
        expected_stage: str,
        expected_version: int,
        command: str,
        mutation: tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically commit a previously reserved slow command."""

        with self.transaction() as db:
            cached = db.execute(
                "SELECT response_json FROM operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
            if cached:
                response = json.loads(cached["response_json"])
                response.setdefault("run_id", run_id)
                return response
            pending = db.execute(
                """
                SELECT * FROM pending_operations
                WHERE namespace=? AND operation_id=?
                """,
                (self.namespace, operation_id),
            ).fetchone()
            if pending is None or str(pending["run_id"]) != run_id:
                raise StudyConflict(
                    "The reserved operation is no longer available.",
                    code="operation_not_reserved",
                    current=self._authoritative_view(db, run_id),
                )
            row = self._validate_run(
                db,
                run_id=run_id,
                browser_session_id=browser_session_id,
                expected_stage=expected_stage,
                expected_version=expected_version,
            )
            response = self._commit_mutation(
                db,
                row=row,
                run_id=run_id,
                operation_id=operation_id,
                command=command,
                mutation=mutation,
            )
            db.execute(
                "DELETE FROM pending_operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            )
            return response

    def cancel_long_operation(self, operation_id: str) -> None:
        with self.transaction() as db:
            db.execute(
                "DELETE FROM pending_operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            )

    @staticmethod
    def _authoritative_view(db: sqlite3.Connection, run_id: str) -> dict[str, Any]:
        row = db.execute(
            "SELECT run_id, stage, state_version FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else {}

    def _validate_run(
        self,
        db: sqlite3.Connection,
        *,
        run_id: str,
        browser_session_id: str,
        expected_stage: str,
        expected_version: int,
    ) -> sqlite3.Row:
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None or str(row["namespace"]) != self.namespace:
            raise StudyConflict("Unknown study run.", code="unknown_run")
        current = {
            "run_id": run_id,
            "stage": str(row["stage"]),
            "state_version": int(row["state_version"]),
        }
        if str(row["browser_session_id"]) != browser_session_id:
            raise StudyConflict(
                "This run is active in another browser tab or session.",
                code="run_owned_elsewhere",
                current=current,
            )
        if str(row["stage"]) != expected_stage or int(row["state_version"]) != int(expected_version):
            raise StudyConflict(
                "The page state is stale. Reload the latest study state.",
                code="stale_state",
                current=current,
            )
        return row

    def _commit_mutation(
        self,
        db: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        run_id: str,
        operation_id: str,
        command: str,
        mutation: tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        response, events, persisted = mutation
        new_stage = str(persisted["stage"])
        new_version = int(row["state_version"]) + 1
        response["run_id"] = run_id
        response["state_version"] = new_version
        if isinstance(response.get("view"), dict):
            response["view"].setdefault("study", {})["state_version"] = new_version
        now = utc_now()
        db.execute(
            """
            UPDATE runs SET stage=?, locale=?, state_version=?, status=?,
                tutorial_index=?, tutorial_max_index=?, tutorial_complete=?,
                updated_at=? WHERE run_id=?
            """,
            (
                new_stage, str(persisted["locale"]), new_version,
                "completed" if new_stage == "completed" else (
                    "abandoned" if new_stage == "abandoned" else "active"
                ),
                int(persisted.get("tutorial_index", 0)),
                int(persisted.get("tutorial_max_index", 0)),
                int(bool(persisted.get("tutorial_complete", False))),
                now, run_id,
            ),
        )
        self._persist_events(db, run_id, events)
        self._persist_domain_records(db, run_id, events, persisted)
        db.execute(
            "INSERT INTO operations(namespace, operation_id, run_id, command, response_json, created_at) VALUES(?,?,?,?,?,?)",
            (
                self.namespace, operation_id, run_id, command,
                json.dumps(response, ensure_ascii=False, default=str), now,
            ),
        )
        return response

    def execute(
        self,
        *,
        operation_id: str,
        run_id: str,
        browser_session_id: str,
        expected_stage: str,
        expected_version: int,
        command: str,
        mutate: Callable[[], tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        if not operation_id.strip():
            raise ValueError("operation_id is required.")
        with self.transaction() as db:
            cached = db.execute(
                "SELECT response_json FROM operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
            if cached:
                response = json.loads(cached["response_json"])
                response.setdefault("run_id", run_id)
                return response
            row = self._validate_run(
                db,
                run_id=run_id,
                browser_session_id=browser_session_id,
                expected_stage=expected_stage,
                expected_version=expected_version,
            )
            return self._commit_mutation(
                db,
                row=row,
                run_id=run_id,
                operation_id=operation_id,
                command=command,
                mutation=mutate(),
            )

    def create_run(
        self,
        *,
        operation_id: str,
        run_id: str,
        browser_session_id: str,
        participant_id: str,
        participant_key: str,
        assignment: Mapping[str, Any],
        locale: str,
        force_restart: bool = False,
        mutate: Callable[[int], tuple[dict[str, Any], list[Mapping[str, Any]], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        with self.transaction() as db:
            cached = db.execute(
                "SELECT response_json FROM operations WHERE namespace=? AND operation_id=?",
                (self.namespace, operation_id),
            ).fetchone()
            if cached:
                return json.loads(cached["response_json"])
            self.ensure_participant(
                db,
                participant_id=participant_id,
                participant_key=participant_key,
                assignment=assignment,
            )
            active = db.execute(
                f"SELECT run_id, browser_session_id, stage, state_version FROM runs "
                f"WHERE namespace=? AND participant_key=? AND stage IN "
                f"({','.join('?' for _ in ACTIVE_STAGES)}) ORDER BY attempt DESC LIMIT 1",
                (self.namespace, participant_key, *sorted(ACTIVE_STAGES)),
            ).fetchone()
            if active is not None and not force_restart:
                raise StudyConflict(
                    "This participant already has an active run in another page.",
                    code="participant_active_elsewhere",
                    current={
                        "run_id": str(active["run_id"]),
                        "stage": str(active["stage"]),
                        "state_version": int(active["state_version"]),
                    },
                )
            if active is not None:
                active_rows = db.execute(
                    f"SELECT run_id FROM runs WHERE namespace=? AND participant_key=? "
                    f"AND stage IN ({','.join('?' for _ in ACTIVE_STAGES)})",
                    (self.namespace, participant_key, *sorted(ACTIVE_STAGES)),
                ).fetchall()
                self.abandon_participant_runs(db, participant_key, reason="participant_restarted")
                for active_row in active_rows:
                    self._persist_events(
                        db,
                        str(active_row["run_id"]),
                        ({"event": "study_abandoned", "reason": "participant_restarted"},),
                    )
            attempt = self.next_attempt(db, participant_key)
            now = utc_now()
            db.execute(
                """
                INSERT INTO runs(run_id, namespace, participant_key, participant_id,
                    attempt, browser_session_id, stage, locale, state_version,
                    status, assignment_json, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, self.namespace, participant_key, participant_id,
                    attempt, browser_session_id, "instructions", locale, 0,
                    "active", json.dumps(dict(assignment), ensure_ascii=False), now, now,
                ),
            )
            response, events, persisted = mutate(attempt)
            response["state_version"] = 1
            if isinstance(response.get("view"), dict):
                response["view"].setdefault("study", {})["state_version"] = 1
            db.execute(
                """
                UPDATE runs SET stage=?, locale=?, state_version=1,
                    tutorial_index=?, tutorial_max_index=?, tutorial_complete=?, updated_at=?
                WHERE run_id=?
                """,
                (
                    str(persisted["stage"]), str(persisted["locale"]),
                    int(persisted.get("tutorial_index", 0)),
                    int(persisted.get("tutorial_max_index", 0)),
                    int(bool(persisted.get("tutorial_complete", False))),
                    now, run_id,
                ),
            )
            self._persist_events(db, run_id, events)
            db.execute(
                "INSERT INTO operations(namespace, operation_id, run_id, command, response_json, created_at) VALUES(?,?,?,?,?,?)",
                (self.namespace, operation_id, run_id, "start", json.dumps(response, ensure_ascii=False, default=str), now),
            )
            return response

    def _next_event_sequence(self, db: sqlite3.Connection, run_id: str) -> int:
        row = db.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS value FROM events WHERE run_id=?",
            (run_id,),
        ).fetchone()
        return int(row["value"]) + 1

    def _persist_events(
        self,
        db: sqlite3.Connection,
        run_id: str,
        events: Iterable[Mapping[str, Any]],
    ) -> None:
        sequence = self._next_event_sequence(db, run_id)
        for raw in events:
            payload = {
                "schema_version": STUDY_LOG_VERSION,
                "timestamp": utc_now(),
                "run_id": run_id,
                **dict(raw),
            }
            db.execute(
                "INSERT INTO events(namespace, run_id, sequence, payload_json, created_at) VALUES(?,?,?,?,?)",
                (self.namespace, run_id, sequence, json.dumps(payload, ensure_ascii=False, default=str), utc_now()),
            )
            sequence += 1

    def _persist_domain_records(
        self,
        db: sqlite3.Connection,
        run_id: str,
        events: Iterable[Mapping[str, Any]],
        persisted: Mapping[str, Any],
    ) -> None:
        for event in events:
            kind = str(event.get("event", ""))
            if kind == "explanation_presented":
                db.execute(
                    "INSERT INTO explanations(run_id, exposure_index, payload_json, created_at) VALUES(?,?,?,?)",
                    (run_id, int(event["exposure_index"]), json.dumps(dict(event), ensure_ascii=False, default=str), utc_now()),
                )
        survey = persisted.get("survey")
        if survey is not None:
            db.execute(
                "INSERT INTO surveys(run_id, payload_json, created_at) VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET payload_json=excluded.payload_json",
                (run_id, json.dumps(survey, ensure_ascii=False, default=str), utc_now()),
            )

    def export_jsonl(self, path: str | Path) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT e.payload_json, r.participant_id, r.attempt, r.status
                FROM events e JOIN runs r ON r.run_id=e.run_id
                WHERE e.namespace=? ORDER BY e.event_id
                """,
                (self.namespace,),
            ).fetchall()
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = json.loads(row["payload_json"])
                payload.setdefault("participant_id", row["participant_id"])
                payload["attempt"] = int(row["attempt"])
                payload["run_status"] = str(row["status"])
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        return len(rows)
