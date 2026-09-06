"""Validate de-identified remote evidence; preview by default, write only with --write.

No network, environment configuration, credentials, cookies or answer bodies are
used. The strict evidence contract is documented in normalize_canary.md. This
checks the consistency of supplied measurements, not their independent origin.
Reports remain candidate evidence; a four-session canary never passes the
twenty-person capacity gate. Manifest freezing and deployment are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.cooperative_kitchen.artifacts import runtime_hash

SHA = re.compile(r"^[a-f0-9]{64}$")
ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
BINDING_KEYS = {"runtime_sha256", "manifest_sha256", "actor_sha256",
                "program_sha256", "qa_configuration_sha256"}
REPORT_NAMES = ("recovery", "remote_load")


class EvidenceError(ValueError):
    """Only fixed diagnostic codes may be shown to the caller."""


def require(ok, code):
    if not ok:
        raise EvidenceError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def fields(value, expected, code="evidence_fields_mismatch"):
    require(isinstance(value, dict) and set(value) == set(expected), code)


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, "invalid_measurement")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid_integer")


def identifier(value):
    require(isinstance(value, str) and ID.fullmatch(value), "invalid_identifier")


def hash_value(value):
    require(isinstance(value, str) and SHA.fullmatch(value), "invalid_hash")


def endpoint(value):
    require(isinstance(value, str), "invalid_endpoint")
    parsed = urlsplit(value)
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and parsed.path in ("", "/")
            and parsed.hostname not in ("localhost", "127.0.0.1", "::1"), "invalid_endpoint")
    return value.rstrip("/")


def safe_path(path):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "symlink_not_allowed")
    return path


def read_json(path):
    path = safe_path(path)
    require(path.suffix == ".json" and path.is_file() and path.stat().st_size <= 2_000_000, "invalid_json_file")
    data = path.read_bytes()
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(EvidenceError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError):
        raise EvidenceError("invalid_json") from None
    return data, value


def checkpoint(value, binding, expected_endpoint, phase, session):
    fields(value, {"endpoint", "binding", "run_id", "episode_id", "phase", "version", "turn", "state_sha256",
                   "captured_at", "capture_sha256", "record_index"})
    require(endpoint(value["endpoint"]) == expected_endpoint, "checkpoint_endpoint_mismatch")
    require(value["binding"] == binding, "checkpoint_release_mismatch")
    require(all(value[key] == session[key] for key in ("run_id", "episode_id")), "checkpoint_session_mismatch")
    require(value["phase"] == phase, "checkpoint_phase_mismatch")
    integer(value["version"], 1); integer(value["turn"], 1)
    hash_value(value["state_sha256"]); number(value["captured_at"], 1)
    hash_value(value["capture_sha256"]); integer(value["record_index"])
    return {key: value[key] for key in ("version", "turn", "state_sha256", "phase")}


def validate_evidence(evidence, binding, expected_endpoint):
    kind = evidence.get("kind") if isinstance(evidence, dict) else None
    require(kind in {"closed_restart", "internal_four"}, "invalid_canary_kind")
    expected = {"schema", "kind", "endpoint", "binding", "started_at", "finished_at",
                "enrollment_mode", "freeplay_qa_enabled", "sessions", "questions"}
    if kind == "closed_restart": expected.add("restart")
    fields(evidence, expected)
    require(evidence["schema"] == "kitchen_canary_evidence_v1", "evidence_schema_mismatch")
    expected_endpoint = endpoint(expected_endpoint)
    require(endpoint(evidence["endpoint"]) == expected_endpoint, "evidence_endpoint_mismatch")
    fields(evidence["binding"], BINDING_KEYS)
    for value in evidence["binding"].values(): hash_value(value)
    require(evidence["binding"] == binding, "evidence_release_mismatch")
    start, end = evidence["started_at"], evidence["finished_at"]
    number(start, 1); number(end, 1)
    require(start < end <= time.time() + 300, "evidence_time_order")
    closed = kind == "closed_restart"
    require(evidence["enrollment_mode"] == ("closed" if closed else "internal_pilot"), "enrollment_mismatch")
    require(evidence["freeplay_qa_enabled"] is closed, "freeplay_qa_setting_mismatch")
    sessions = evidence["sessions"]
    require(isinstance(sessions, list) and len(sessions) == (1 if closed else 4), "session_count_mismatch")
    restart = evidence.get("restart")
    if closed:
        fields(restart, {"provider", "operation", "dashboard_event", "operator_confirmed", "evidence_sha256"})
        require(restart["provider"] == "render" and restart["operation"] == "restart", "actual_render_restart_required")
        require(restart["operator_confirmed"] is True, "restart_operator_confirmation_required")
        hash_value(restart["evidence_sha256"])
        event = restart["dashboard_event"]
        fields(event, {"service_id", "dashboard_url", "event_text", "displayed_timezone",
                       "displayed_event_minute", "time_precision", "source"})
        identifier(event["service_id"])
        require(event["service_id"].startswith("srv-")
                and event["dashboard_url"] == "https://dashboard.render.com/web/" + event["service_id"] + "/events",
                "invalid_dashboard_reference")
        require(event["source"] in {"authenticated_render_dashboard_accessibility_tree", "authenticated_render_dashboard_screenshot"}
                and isinstance(event["event_text"], str) and event["event_text"].startswith("Service restarted by you ")
                and len(event["event_text"]) <= 200 and "\n" not in event["event_text"], "invalid_dashboard_event")
        require(event["time_precision"] == "minute" and isinstance(event["displayed_timezone"], str)
                and 0 < len(event["displayed_timezone"]) <= 64, "invalid_event_time_precision")
        try: event_time = datetime.fromisoformat(event["displayed_event_minute"])
        except (ValueError, TypeError): raise EvidenceError("invalid_event_time") from None
        require(event_time.tzinfo is not None and event_time.second == event_time.microsecond == 0, "invalid_event_time")
        event_start = event_time.timestamp()
        # A minute-precision dashboard label is an interval, never an invented
        # request timestamp. Capture order and database evidence supply the rest.
        require(start < event_start + 60 and event_start <= end, "restart_time_order")
    seen_runs, seen_episodes, seen_ops, by_run, latencies = set(), set(), set(), {}, []
    for session in sessions:
        fields(session, {"run_id", "episode_id", "namespace", "mode", "condition", "task_order",
                         "before", "after", "replay", "operation"})
        for name, seen in (("run_id", seen_runs), ("episode_id", seen_episodes)):
            identifier(session[name]); require(session[name] not in seen, "duplicate_session_identity"); seen.add(session[name])
        require(session["namespace"] == ("development" if closed else "pilot")
                and session["mode"] == ("freeplay" if closed else "pilot"), "session_namespace_or_mode_mismatch")
        require((session["condition"] is None and session["task_order"] is None) if closed else
                (session["condition"] in {"A", "B"} and session["task_order"] in {"XY", "YX"}), "invalid_assignment")
        phase = "freeplay" if closed else "task1"
        states = [checkpoint(session[name], binding, expected_endpoint, phase, session) for name in ("before", "after", "replay")]
        require(states[0] == states[1] == states[2], "confirmed_state_or_version_changed")
        before, after, replay = [session[name]["captured_at"] for name in ("before", "after", "replay")]
        require(start <= before < after <= replay <= end, "session_time_order")
        if session["after"]["capture_sha256"] == session["replay"]["capture_sha256"]:
            require(session["after"]["record_index"] < session["replay"]["record_index"], "recovery_transcript_order_mismatch")
        if closed: require(before < event_start + 60 and event_start <= after, "session_does_not_span_restart")
        op = session["operation"]
        fields(op, {"operation_id_sha256", "replayed_operation_id_sha256", "request_sha256", "replayed_request_sha256", "first_response_sha256", "replayed_response_sha256",
                    "request_version", "first_response_version", "replayed_response_version", "first_http_status", "replayed_http_status",
                    "first_state_sha256", "replayed_state_sha256", "database", "first_ack_seconds", "replay_ack_seconds"})
        for key in ("operation_id_sha256", "replayed_operation_id_sha256", "request_sha256", "replayed_request_sha256", "first_response_sha256", "replayed_response_sha256", "first_state_sha256", "replayed_state_sha256"): hash_value(op[key])
        require(op["operation_id_sha256"] == op["replayed_operation_id_sha256"]
                and op["request_sha256"] == op["replayed_request_sha256"], "replayed_request_mismatch")
        require(op["operation_id_sha256"] not in seen_ops, "duplicate_operation_identity"); seen_ops.add(op["operation_id_sha256"])
        for key in ("request_version", "first_response_version", "replayed_response_version"): integer(op[key])
        require(op["request_version"] + 1 == op["first_response_version"] <= states[0]["version"]
                and op["replayed_response_version"] == states[2]["version"], "operation_version_mismatch")
        require(op["first_http_status"] == op["replayed_http_status"] == 200
                and type(op["first_http_status"]) is int and type(op["replayed_http_status"]) is int, "operation_not_acknowledged")
        require(op["first_state_sha256"] == op["replayed_state_sha256"] == states[0]["state_sha256"], "replayed_game_state_mismatch")
        database = op["database"]
        fields(database, {"evidence_source", "record_sha256", "operation_receipt_count", "joint_step_event_count",
                          "request_sha256", "recorded_response_version"})
        require(database["evidence_source"] in {"postgresql", "authenticated_admin_export"}, "database_receipt_evidence_required")
        hash_value(database["record_sha256"]); hash_value(database["request_sha256"])
        integer(database["operation_receipt_count"]); integer(database["joint_step_event_count"])
        integer(database["recorded_response_version"])
        require(database["operation_receipt_count"] == database["joint_step_event_count"] == 1, "nonunique_database_action")
        require(database["request_sha256"] == op["request_sha256"]
                and database["recorded_response_version"] == op["first_response_version"], "database_receipt_mismatch")
        for key in ("first_ack_seconds", "replay_ack_seconds"): number(op[key]); latencies.append(op[key])
        by_run[session["run_id"]] = session
    if not closed:
        require({session["condition"] for session in sessions} == {"A", "B"}, "both_conditions_required")
    questions = evidence["questions"]
    require(isinstance(questions, list) and 2 <= len(questions) <= 24, "bilingual_qa_required")
    seen_questions, languages, qa_seconds, asked_runs, question_versions = set(), set(), [], set(), set()
    for question in questions:
        fields(question, {"question_id", "run_id", "episode_id", "frame", "language", "provider", "status",
                          "llm_success", "verified", "evidence_source", "record_sha256", "request_version",
                          "completed_version", "game_before_sha256", "game_after_sha256", "elapsed_seconds"})
        identifier(question["question_id"])
        require(question["question_id"] not in seen_questions, "duplicate_question_identity"); seen_questions.add(question["question_id"])
        require(question["run_id"] in by_run, "question_session_mismatch")
        session = by_run[question["run_id"]]; asked_runs.add(question["run_id"])
        require(question["episode_id"] == session["episode_id"], "question_episode_mismatch")
        require(closed or session["condition"] == "A", "unauthorized_condition_question")
        integer(question["frame"]); require(question["frame"] <= session["before"]["turn"], "question_frame_mismatch")
        require(question["language"] in {"zh", "en"}, "invalid_question_language"); languages.add(question["language"])
        require(question["provider"] == "deepseek" and question["status"] == "complete"
                and question["llm_success"] is True and question["verified"] is True, "real_verified_cloud_qa_required")
        require(question["evidence_source"] in {"postgresql", "authenticated_admin_export"}, "persisted_qa_evidence_required")
        hash_value(question["record_sha256"]); hash_value(question["game_before_sha256"]); hash_value(question["game_after_sha256"])
        require(question["game_before_sha256"] == question["game_after_sha256"], "question_changed_game_state")
        integer(question["request_version"]); integer(question["completed_version"])
        require(question["request_version"] + 1 <= question["completed_version"]
                <= session["before"]["version"], "question_state_version_mismatch")
        version_key = (question["run_id"], question["request_version"])
        require(version_key not in question_versions, "duplicate_question_state_version"); question_versions.add(version_key)
        number(question["elapsed_seconds"]); qa_seconds.append(question["elapsed_seconds"])
    require(languages == {"zh", "en"}, "bilingual_qa_required")
    require(closed or asked_runs == {s["run_id"] for s in sessions if s["condition"] == "A"}, "each_A_session_needs_qa")
    def p95(values): return sorted(values)[math.ceil(len(values) * .95) - 1]
    action_p95, qa_p95 = p95(latencies), p95(qa_seconds)
    return {"passed": action_p95 <= 1 and qa_p95 <= 30, "functional_checks_passed": True,
            "kind": kind, "mode": "real_remote", "endpoint": expected_endpoint, "binding": binding,
            "started_at": start, "finished_at": end, "sessions": len(sessions), "run_ids": sorted(seen_runs),
            "question_ids": sorted(seen_questions), "cloud_questions": len(questions), "languages": sorted(languages),
            "action_p95_seconds": action_p95, "question_p95_seconds": qa_p95,
            "zero_duplicate_steps": True, "zero_confirmed_record_loss": True,
            "evidence_sha256": sha(canonical(evidence)), "restart": restart,
            "scope": "Controlled canary and confirmed-state replay only; not twenty-person capacity or human effects."}


def prepare(release, evidence_path, expected_url):
    release = safe_path(release)
    require(release.name == "v3-id-pilot" and release.is_dir(), "v3_release_required")
    manifest_bytes, manifest = read_json(release / "manifest.json")
    require(manifest.get("schema") == "cooperative_kitchen_release_manifest_v1"
            and manifest.get("status") == "candidate", "candidate_manifest_required")
    require(manifest.get("runtime_sha256") == runtime_hash(), "current_runtime_mismatch")
    config = manifest.get("qa_configuration")
    require(isinstance(config, dict) and config.get("provider") == "deepseek", "deepseek_configuration_required")
    binding = {"runtime_sha256": manifest["runtime_sha256"], "manifest_sha256": sha(manifest_bytes),
               "actor_sha256": manifest["artifacts"]["actor"]["sha256"],
               "program_sha256": manifest["artifacts"]["program"]["sha256"],
               "qa_configuration_sha256": sha(canonical(config))}
    evidence_bytes, evidence = read_json(evidence_path)
    summary = validate_evidence(evidence, binding, expected_url)
    summary["evidence_file_sha256"] = sha(evidence_bytes)
    originals, reports = {release / "manifest.json": manifest_bytes}, {}
    for name in REPORT_NAMES:
        entry = manifest["artifacts"][name + "_report"]
        require(entry["path"] == f"acceptance/{name}_report.json", "report_path_mismatch")
        path = release / entry["path"]
        original, report = read_json(path)
        require(sha(original) == entry["sha256"] and report.get("runtime_sha256") == binding["runtime_sha256"], "report_binding_mismatch")
        require(report.get("schema") == f"kitchen_{name}_audit_v3", "report_schema_mismatch")
        originals[path] = original; reports[name] = report
    remote, recovery = reports["remote_load"], reports["recovery"]
    require(isinstance(recovery.get("local_postgresql"), dict)
            and recovery["local_postgresql"].get("passed") is True, "local_recovery_evidence_required")
    previous = remote.get("previous_20_session_result")
    require(isinstance(previous, dict) and previous.get("passed") is False
            and previous.get("action_p95_seconds") == 6.76 and previous.get("question_p95_seconds") == 37.97,
            "historical_twenty_session_result_mismatch")
    if evidence["kind"] == "closed_restart":
        recovery.update(passed=True, remote_restart_tested=True, render_restart_canary=summary,
                        blocker=None, scope="Actual Render restart restored confirmed canary state and idempotent replay; local recovery evidence is retained.")
        remote["closed_canary"] = summary
    else:
        require(recovery.get("passed") is True and recovery.get("remote_restart_tested") is True
                and isinstance(recovery.get("render_restart_canary"), dict)
                and recovery["render_restart_canary"].get("functional_checks_passed") is True,
                "validated_closed_restart_required_first")
        old_binding = recovery["render_restart_canary"].get("binding", {})
        require(all(old_binding.get(key) == binding[key] for key in BINDING_KEYS - {"manifest_sha256"})
                and recovery["render_restart_canary"].get("endpoint") == endpoint(expected_url), "earlier_restart_release_mismatch")
        remote["four_session_canary"] = summary
    remote.update(passed=False, mode="real_remote", runtime_sha256=binding["runtime_sha256"],
                  actor_sha256=binding["actor_sha256"], program_sha256=binding["program_sha256"],
                  qa_configuration=config, blocker="The twenty-session capacity gate remains unpassed.",
                  scope="Controlled canary evidence is nested below. Historical twenty-session latency failure is retained; no twenty-participant support claim.")
    replacements = {release / f"acceptance/{name}_report.json":
                    (json.dumps(reports[name], ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
                    for name in REPORT_NAMES}
    return originals, replacements, summary


def atomic_write(path, data):
    handle, pending = tempfile.mkstemp(prefix=".canary-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as target:
            target.write(data); target.flush(); os.fsync(target.fileno())
        os.chmod(pending, 0o600); os.replace(pending, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
    finally:
        Path(pending).unlink(missing_ok=True)


def normalize(release, evidence_path, expected_url, *, write=False):
    originals, replacements, summary = prepare(release, evidence_path, expected_url)
    preview = {"validated": True, "written": False, "kind": summary["kind"],
               "canary_passed": summary["passed"], "remote_load_passed": False,
               "evidence_sha256": summary["evidence_sha256"], "runtime_sha256": summary["binding"]["runtime_sha256"],
               "evidence_file_sha256": summary["evidence_file_sha256"],
               "report_sha256": {path.name: sha(data) for path, data in replacements.items()},
               "manifest_requires_refreeze": bool(write)}
    if not write: return preview
    release = safe_path(release)
    provenance = safe_path(release / "provenance"); provenance.mkdir(mode=0o700, exist_ok=True)
    lock = safe_path(provenance / ".normalize-canary.lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "r+") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        require(all(safe_path(path).read_bytes() == data for path, data in originals.items()), "release_changed_before_write")
        backup = provenance / ("canary_" + summary["evidence_sha256"][:12] + "_" + uuid.uuid4().hex[:8])
        backup.mkdir(mode=0o700)
        for path, data in originals.items(): atomic_write(backup / path.name, data)
        journal = {"status": "prepared", "original_sha256": {p.name: sha(v) for p, v in originals.items()},
                   "replacement_sha256": preview["report_sha256"], "evidence_sha256": summary["evidence_sha256"]}
        atomic_write(backup / "transaction.json", canonical(journal))
        try:
            for path, data in replacements.items(): atomic_write(path, data)
        except BaseException:
            # Normal I/O errors restore both old reports. A process kill between
            # replacements leaves old manifest hashes invalid: fail closed, with
            # the complete owner-only backup available for recovery.
            for path in replacements: atomic_write(path, originals[path])
            raise
        journal["status"] = "complete"; atomic_write(backup / "transaction.json", canonical(journal))
    preview.update(written=True, backup_directory=str(backup))
    return preview


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "output/cooperative_kitchen/v3-id-pilot")
    parser.add_argument("--evidence", type=Path, required=True, help="Strict de-identified JSON only; never a raw export or environment file.")
    parser.add_argument("--expected-url", required=True, help="Exact HTTPS application origin; no credentials or query string.")
    parser.add_argument("--write", action="store_true", help="Back up old reports and atomically replace each report; does not freeze or deploy.")
    args = parser.parse_args(argv)
    try:
        result = normalize(args.release, args.evidence, args.expected_url, write=args.write)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"validated": False, "written": False,
                          "error": str(error) if isinstance(error, EvidenceError) else "canary_validation_or_io_failed"}))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
