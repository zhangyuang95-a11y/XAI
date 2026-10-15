"""Private, read-only native warehouse export and SQLite online backup.

Examples (destinations must not exist):
  python -m ui.warehouse_native_export export --database local.sqlite3 \
      --output private/export_01 --namespace local_pilot
  python -m ui.warehouse_native_export backup --database local.sqlite3 \
      --output private/backup_01.sqlite3 --namespace local_pilot
  python -m ui.warehouse_native_export restore --backup private/backup_01.sqlite3 \
      --output private/restored_01.sqlite3

A restore creates a NEW database, never overwrites a live database. Stop a service
before explicitly changing its --database argument to a restored path. The source
is opened mode=ro; SQLite backup() captures committed WAL data. Main/WAL file
hashes describe sampled source files; the standalone snapshot hash is the binding
for the coherent exported data. Stored namespace metadata is checked against the caller declaration. Legacy
databases without metadata remain explicitly unknown. Neither verification nor
local_pilot is formal data.

Questionnaire exports include three 1–7 self-reports and, only when a candidate
bank was bound and privately scored, next-action and three-step prediction
accuracy. Missing banks/scores remain blank; no formal acceptance is inferred.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

VERSION = "warehouse-native-private-export.v1"
NAMESPACES = ("verification", "local_pilot")
TABLES = ("sessions", "runs", "frames", "events", "questions", "operations", "blocks")
JSON_COLUMNS = {"sessions": {"questionnaire", "questionnaire_scores"}, "runs": {"snapshot", "metrics", "feedback", "provenance"},
                "frames": {"public", "internal"}, "events": {"payload"}, "blocks": {"allocation"}}
LIMITATIONS = [
    "Stored namespace metadata is checked when present. Missing legacy metadata remains unknown; the operator declaration does not retroactively certify old records.",
    "Verification and local_pilot data are not formal research samples or evidence that model/explanation acceptance passed.",
    "Self-report items are always retained. Prediction scores exist only when a specific candidate bank was bound at enrollment and privately scored on submission; missing banks/scores remain blank.",
    "Self-reported predictability is not behavioral prediction performance. Candidate prediction-bank checks do not establish model capability or formal study approval.",
    "Answer shown timestamps are the server's persisted acknowledgements, not an independent attention/reading measurement.",
    "An Actor hash is recoverable from recorded decisions; a zero-action run may have no separately recorded Actor hash.",
    "Task 2's primary mean includes recorded participant-ended rounds; aborts remain explicitly identifiable. Missing or duplicate round slots invalidate the complete-stage mean.",
]


class ExportError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_only(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ExportError("Source must be an existing SQLite file")
    db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _source_files(path):
    result = {}
    for suffix in ("", "-wal"):
        item = Path(str(path)+suffix)
        try:
            result["main" if not suffix else "wal"] = {"sha256": sha256(item), "bytes": item.stat().st_size}
        except FileNotFoundError:
            result["main" if not suffix else "wal"] = None
    return result


def _new_path(path):
    path = Path(path).absolute()
    if any(os.path.lexists(str(path)+suffix) for suffix in ("", "-wal", "-shm")):
        raise FileExistsError(f"Refusing to overwrite existing destination or SQLite sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _private_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        handle.flush(); os.fsync(handle.fileno())


def _snapshot(source, destination):
    """Use SQLite's backup API, not a main-file copy that can drop WAL commits."""
    source = Path(source).resolve(strict=True)
    destination = _new_path(destination)
    before = _source_files(source)
    fd, temporary_name = tempfile.mkstemp(prefix=".warehouse-backup-", suffix=".sqlite3", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with closing(read_only(source)) as src, closing(sqlite3.connect(temporary)) as dst:
            src.execute("BEGIN")
            src.execute("SELECT count(*) FROM sqlite_master").fetchone()  # Pin the read snapshot.
            src.backup(dst)
            dst.commit()
            dst.execute("PRAGMA journal_mode=DELETE")
            if dst.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ExportError("SQLite snapshot integrity check failed")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses an existing destination.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(temporary)+suffix).unlink(missing_ok=True)
    return {"source_path": str(source), "source_files_before": before,
            "source_files_after": _source_files(source),
            "snapshot_sha256": sha256(destination), "snapshot_bytes": destination.stat().st_size,
            "method": "SQLite read-only transaction + Connection.backup; includes committed WAL",
            "source_file_hashes_are_point_samples": True}


def namespace_info(database, declared):
    if declared not in NAMESPACES:
        raise ExportError("Namespace must be verification or local_pilot")
    with closing(read_only(database)) as db:
        present = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone()
        metadata = {row["key"]: json.loads(row["value"]) for row in db.execute("SELECT * FROM metadata")} if present else {}
    recorded = metadata.get("namespace", "unknown_legacy")
    if recorded not in (*NAMESPACES, "unknown_legacy"):
        raise ExportError("Stored database namespace is not recognized")
    if recorded in NAMESPACES and recorded != declared:
        raise ExportError("Declared namespace does not match stored database metadata")
    return {"namespace": recorded, "declared_namespace": declared,
            "namespace_origin": "stored_metadata" if recorded in NAMESPACES else "unknown_legacy_not_inferred",
            "stored_metadata": metadata}


def backup_database(database, output, namespace):
    if namespace not in NAMESPACES:
        raise ExportError("Namespace must be verification or local_pilot")
    namespace_binding = namespace_info(database, namespace)
    destination = _new_path(output)
    sidecar = Path(str(destination)+".manifest.json")
    if os.path.lexists(sidecar):
        raise FileExistsError("Refusing to overwrite backup manifest")
    binding = _snapshot(database, destination)
    manifest = {"version": VERSION, "kind": "sqlite_backup", "created_utc": _now(),
                **namespace_binding, "formal_sample": False,
                **binding, "backup_path": str(destination), "backup_sha256": binding["snapshot_sha256"]}
    _private_json(sidecar, manifest)
    return manifest


def restore_database(backup, output):
    source = Path(backup).resolve(strict=True)
    sidecar = Path(str(source)+".manifest.json")
    if not sidecar.is_file():
        raise ExportError("Restore requires the backup's manifest sidecar")
    manifest = json.loads(sidecar.read_text())
    if manifest.get("kind") != "sqlite_backup" or manifest.get("backup_sha256") != sha256(source):
        raise ExportError("Backup manifest/hash mismatch")
    # A standalone backup has no active WAL; accepting a sidecar WAL would import
    # content that is outside the checked backup hash.
    if Path(str(source)+"-wal").exists() and Path(str(source)+"-wal").stat().st_size:
        raise ExportError("Backup has a WAL file and is not the frozen standalone artifact")
    receipt = Path(str(Path(output).absolute())+".restore.json")
    if os.path.lexists(receipt):
        raise FileExistsError("Refusing to overwrite an existing restore receipt")
    result = backup_database(source, output, manifest.get("declared_namespace", manifest["namespace"]))
    result["kind"] = "sqlite_restore_to_new_path"
    result["restored_from_sha256"] = manifest["backup_sha256"]
    # The standalone backup manifest remains a reusable backup artifact. A separate
    # restore receipt identifies the operational action without rewriting it.
    _private_json(receipt, result)
    return result


def _now():
    return datetime.now(timezone.utc).isoformat()


def _decode(row, table):
    result = dict(row)
    for key in JSON_COLUMNS.get(table, ()):
        if key in result and result[key] is not None:
            try:
                result[key] = json.loads(result[key])
            except (TypeError, json.JSONDecodeError) as error:
                raise ExportError(f"Malformed stored JSON in {table}.{key}") from error
    return result


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _metric(run, name):
    metrics = run.get("metrics") or {}
    if name in ("human_deliveries", "ai_deliveries"):
        values = metrics.get("individual_deliveries")
        index = 0 if name == "human_deliveries" else 1
        return _number(values[index]) if isinstance(values, list) and len(values) > index else None
    return _number(metrics.get(name))


def end_status(run, end_events, final_public=None):
    """Operational abort evidence takes precedence over artificial terminal tags."""
    if not run.get("ended"):
        return {"category": "active", "reason": None, "evidence": "runs.ended=false"}
    if run.get("end_reason") == "participant_ended" or run["id"] in end_events:
        return {"category": "aborted", "reason": "participant_ended", "evidence": "runs.end_reason or persisted end event"}
    state = (run.get("snapshot") or {}).get("state") or {}
    public_state = (final_public or {}).get("state") or {}
    reason = run.get("end_reason") or state.get("terminal_reason") or public_state.get("terminal_reason")
    if reason == "horizon":
        return {"category": "horizon", "reason": reason, "evidence": "stored terminal reason"}
    if reason:
        return {"category": "terminal", "reason": reason, "evidence": "stored terminal reason"}
    return {"category": "unknown", "reason": None, "evidence": "ended without a stored reason; not assumed horizon"}


def _stage_summary(runs, stage):
    rows = [r for r in runs if r["stage"] == stage]
    slots = {index: [r for r in rows if r["round_index"] == index] for index in range(3)}
    ended = [r for r in rows if r["ended"]]
    structurally_complete = len(rows) == 3 and all(len(slots[i]) == 1 and slots[i][0]["ended"] for i in slots)
    result = {f"{stage}_started_rounds": len(rows), f"{stage}_ended_rounds": len(ended),
              f"{stage}_missing_slots": canonical([i+1 for i, values in slots.items() if not values]),
              f"{stage}_duplicate_slots": canonical([i+1 for i, values in slots.items() if len(values) > 1]),
              f"{stage}_complete": structurally_complete}
    for metric in ("deliveries", "score", "legacy_score", "steps", "collisions", "shutdowns", "human_deliveries", "ai_deliveries"):
        values = [_metric(r, metric) for r in ended]
        present = [v for v in values if v is not None]
        complete = structurally_complete and len(present) == 3
        result[f"{stage}_mean_{metric}"] = sum(present)/3 if complete else None
        result[f"{stage}_total_{metric}"] = sum(present) if complete else None
        result[f"{stage}_observed_{metric}_rounds"] = len(present)
        if metric == "deliveries":
            result[f"{stage}_observed_mean_deliveries"] = sum(present)/len(present) if present else None
    return result


def participant_summary(session, runs, questions, namespace):
    formal = [r for r in runs if r["stage"] in ("task1", "task2")]
    one, two = _stage_summary(formal, "task1"), _stage_summary(formal, "task2")
    draft = session.get("questionnaire") or {}
    prediction = session.get("questionnaire_scores") or {}
    result = {"namespace": namespace, "formal_sample": False,
              **{key: session.get(key) for key in ("participant_id", "participant_key", "condition", "task_order", "position", "mode", "stage")},
              "session_id": session["id"], "flow_completed": session["stage"] == "completed",
              "six_round_completion": one["task1_complete"] and two["task2_complete"],
              "formal_rounds_started": len(formal), "formal_rounds_ended": sum(bool(r["ended"]) for r in formal),
              **one, **two,
              "task2_primary_available": two["task2_mean_deliveries"] is not None,
              "aborted_rounds": sum(r["end_status"]["category"] == "aborted" for r in formal),
              "horizon_rounds": sum(r["end_status"]["category"] == "horizon" for r in formal),
              "other_terminal_rounds": sum(r["end_status"]["category"] in ("terminal", "unknown") for r in formal),
              "questions": len(questions), "completed_answers": sum(q["status"] == "complete" for q in questions),
              "answers_shown": sum(bool(q.get("shown")) for q in questions),
              "questionnaire_self_report_only": not bool(session.get("questionnaire_bank_signature")),
              "questionnaire_bank_signature": session.get("questionnaire_bank_signature"),
              "prediction_scoring_available": bool(prediction),
              "prediction_scores_json": canonical(prediction) if prediction else None, "questionnaire_json": canonical(draft)}
    for kind in ("next_action", "wait_three"):
        category = prediction.get(kind) or {}
        for metric in ("correct", "total", "accuracy"):
            result[f"prediction_{kind}_{metric}"] = category.get(metric)
    for key in ("cooperation", "predictability", "difficulty"):
        result[f"self_report_{key}"] = draft.get(key)
    return result


def _jsonl(path, rows):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    count = 0
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row)+"\n"); count += 1
        handle.flush(); os.fsync(handle.fileno())
    return count


def export_database(database, output, namespace, actor_path=None):
    if namespace not in NAMESPACES:
        raise ExportError("Namespace must be verification or local_pilot")
    namespace_binding = namespace_info(database, namespace)
    destination = Path(output).absolute()
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    binding = _snapshot(database, destination/"source_snapshot.sqlite3")
    namespace_binding = namespace_info(destination/"source_snapshot.sqlite3", namespace)
    files, counts = {}, {}
    with closing(read_only(destination/"source_snapshot.sqlite3")) as db:
        present = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if set(TABLES) - present:
            raise ExportError("Not a native warehouse research database: required tables are missing")
        sessions = [_decode(r, "sessions") for r in db.execute("SELECT * FROM sessions ORDER BY id")]
        runs = [_decode(r, "runs") for r in db.execute("SELECT * FROM runs ORDER BY session_id,stage,round_index,id")]
        questions = [_decode(r, "questions") for r in db.execute("SELECT * FROM questions ORDER BY session_id,created,id")]
        end_events = {row[0] for row in db.execute("SELECT DISTINCT run_id FROM events WHERE kind='end'")}
        finals = {r["run_id"]: json.loads(r["public"]) for r in db.execute("SELECT f.run_id,f.public FROM frames f WHERE f.frame=(SELECT max(f2.frame) FROM frames f2 WHERE f2.run_id=f.run_id)")}
        actor_bindings = {r["id"]: set() for r in runs}
        for frame in db.execute("SELECT run_id,internal FROM frames ORDER BY run_id,frame"):
            internal = json.loads(frame["internal"])
            actor = internal.get("actor_sha256") or internal.get("decision", {}).get("actor_sha256")
            if actor is not None:
                actor_bindings[frame["run_id"]].add(actor)
        for run in runs:
            if (run.get("provenance") or {}).get("actor_sha256"):
                actor_bindings[run["id"]].add(run["provenance"]["actor_sha256"])
            run["end_status"] = end_status(run, end_events, finals.get(run["id"]))
            run["recorded_actor_sha256s"] = sorted(actor_bindings[run["id"]])
        datasets = {
            "participants.jsonl": iter(sessions), "runs.jsonl": iter(runs), "questions.jsonl": iter(questions),
            "answer_shown.jsonl": ({"question_id": q["id"], "session_id": q["session_id"], "run_id": q["run_id"], "frame": q["frame"], "shown": q["shown"]} for q in questions if q.get("shown")),
        }
        for table in ("events", "frames", "operations", "blocks"):
            datasets[table+".jsonl"] = (_decode(row, table) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid")) if table in JSON_COLUMNS else (dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY rowid"))
            # Consume each cursor now; late-bound table variables must not cross iterations.
            counts[table] = _jsonl(destination/(table+".jsonl"), datasets.pop(table+".jsonl"))
        for name, rows in datasets.items():
            counts[name.removesuffix(".jsonl")] = _jsonl(destination/name, rows)
        summaries = [participant_summary(s, [r for r in runs if r["session_id"] == s["id"]],
                          [q for q in questions if q["session_id"] == s["id"]], s.get("namespace") or namespace_binding["namespace"])
                     for s in sessions if s.get("participant_id")]
        # An empty export still has the full stable CSV schema.
        prototype = participant_summary({"id": "", "stage": "", "questionnaire": {}}, [], [], namespace_binding["namespace"])
        with os.fdopen(os.open(destination/"participants.csv", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prototype)); writer.writeheader(); writer.writerows(summaries)
            handle.flush(); os.fsync(handle.fileno())
        run_provenance = [{"run_id": r["id"], "run_signature": r["signature"], "scenario_id": r["scenario_id"],
                           "environment_version": r["snapshot"].get("version"),
                           "recorded_actor_sha256s": r["recorded_actor_sha256s"],
                           "actor_binding_status": "conflicting_recorded_actors" if len(r["recorded_actor_sha256s"]) > 1 else "recorded_artifacts" if r["recorded_actor_sha256s"] else "not_separately_recorded",
                           "provenance": r.get("provenance"),
                           "namespace": (r.get("provenance") or {}).get("namespace", "unknown_legacy"),
                           "end_status": r["end_status"]} for r in runs]
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {"version": VERSION, "kind": "native_warehouse_private_export", "complete": True, "created_utc": _now(),
                **namespace_binding, "formal_sample": False,
                "primary_metric": {"column": "task2_mean_deliveries", "definition": "Arithmetic mean across the three unique ended Task 2 rounds; blank unless all three delivery values are present", "missing_round_policy": "no zero imputation; observed means and counts are separate"},
                "six_round_completion_definition": "All six unique expected task rounds have ended; this is protocol completion, not delivery success",
                "questionnaire": {"kind": "session_bound_optional_candidate_prediction_bank", "self_report_items": ["cooperation", "predictability", "difficulty"], "range": [1, 7], "prediction_scored_sessions": sum(bool(s.get("questionnaire_scores")) for s in sessions), "formal_bank_approval_asserted": False},
                "limitations": LIMITATIONS, "source": binding, "run_bindings": run_provenance,
                "counts": {**counts, "participant_csv_rows": len(summaries)}, "files": files,
                "exporter_sha256": sha256(Path(__file__))}
    if actor_path is not None:
        manifest["operator_supplied_actor"] = {"path": str(Path(actor_path).resolve(strict=True)), "sha256": sha256(actor_path), "historical_binding": "not substituted for recorded per-run decisions"}
    _private_json(destination/"manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export", "backup"):
        item = commands.add_parser(name)
        item.add_argument("--database", type=Path, required=True)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--namespace", choices=NAMESPACES, required=True)
        if name == "export":
            item.add_argument("--actor", type=Path, help="Optional provenance input; does not fill unknown historical Actor bindings")
    item = commands.add_parser("restore")
    item.add_argument("--backup", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_database(args.database, args.output, args.namespace, args.actor)
        elif args.command == "backup":
            result = backup_database(args.database, args.output, args.namespace)
        else:
            result = restore_database(args.backup, args.output)
    except (ExportError, FileExistsError, FileNotFoundError, sqlite3.DatabaseError) as error:
        parser.exit(2, str(error)+"\n")
    print(canonical({"kind": result["kind"], "namespace": result["namespace"], "output": str(args.output), "formal_sample": False}))


if __name__ == "__main__":
    main()
