"""Explicit legacy remote capacity diagnostic, using development freeplay only.

Example (the private file contains KITCHEN_URL, KITCHEN_ADMIN_KEY and preferably
KITCHEN_DIRECT_DATABASE_URL):
  python scripts/cooperative_kitchen/remote_acceptance.py --env-file /private/file \
      --run --output output/cooperative_kitchen/v3-id-pilot/remote-load-<date>.json

Twenty sessions create 200 joint steps and 40 bilingual free questions. All IDs
are retained for exclusion from research data; no records are deleted. Real
cloud success is checked in persisted private diagnostics, never inferred from
the public answer's verified flag. This does not test research effectiveness,
formal A/B protocol, cold-start capacity, or process-restart recovery.

This diagnostic requires KITCHEN_FREEPLAY_QA=1. Run it only in a short,
enrollment-closed maintenance canary and restore the value to 0 immediately;
anonymous cloud QA stays disabled in the public internal-pilot configuration.
Use controlled A-condition pilot IDs for the final deployment check.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import threading
import time
from urllib.parse import urlsplit
import uuid

import httpx

ENV_KEYS = {"KITCHEN_ACCEPTANCE_URL", "KITCHEN_URL", "KITCHEN_ADMIN_KEY",
            "KITCHEN_DIRECT_DATABASE_URL", "DATABASE_URL"}
SESSIONS, STEPS, QUESTIONS = 20, 10, 2


class AcceptanceError(Exception):
    """Only a safe, fixed diagnostic code is allowed into a report."""


def require(condition, code):
    if not condition:
        raise AcceptanceError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def snapshot_digest(value):
    # The authority's snapshot contract uses the standard JSON separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def percentile(values, fraction=.95):
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)] if values else None


def private_environment(path=None):
    """Parse literal dotenv values without evaluating a shell or exposing values."""
    result = {}
    if path:
        location = Path(path).expanduser()
        info = location.stat()
        require(stat.S_ISREG(info.st_mode) and not info.st_mode & 0o077, "env_file_requires_owner_only_permissions")
        for line in location.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            if key.strip() not in ENV_KEYS:
                continue
            require(bool(separator), "invalid_private_environment")
            tokens = shlex.split(value, comments=True, posix=True)
            require(len(tokens) <= 1, "invalid_private_environment")
            result[key.strip()] = tokens[0] if tokens else ""
    # An explicitly configured process value wins; nothing is copied into the
    # process environment, so API credentials cannot reach child processes.
    result.update({key: os.environ[key] for key in ENV_KEYS if key in os.environ})
    return result


def validate_endpoint(url, allow_local=False):
    parsed = urlsplit(url)
    require(bool(parsed.hostname) and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment and parsed.path in {"", "/"}, "invalid_endpoint")
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    require(parsed.scheme == "https" or (allow_local and local and parsed.scheme == "http"), "remote_requires_https")
    require(not local or allow_local, "local_requires_explicit_fixture_flag")
    return url.rstrip("/"), local


class Client:
    def __init__(self, base, *, transport=None, cookies=None, admin_key=None, retries=1):
        self.base, self.retries = base, retries
        self.client = httpx.Client(timeout=30, follow_redirects=False, transport=transport, cookies=cookies)
        self.admin_key = admin_key
        self.retry_log = []

    def close(self):
        self.client.close()

    def request(self, path, payload=None, *, allowed=(200,), raw=False, timeout=None):
        require(path.startswith("/api/"), "invalid_request_path")
        began = time.perf_counter()
        # No default authorization header: the admin key can only be sent to
        # protected endpoints at the validated origin, never with participant
        # requests, cookies from other sessions, or a followed redirect.
        headers = {}
        if path.startswith("/api/admin/"):
            headers["X-Kitchen-Admin-Key"] = self.admin_key or ""
        body = canonical(payload).encode() if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(self.retries + 1):
            try:
                response = self.client.request("POST" if payload is not None else "GET", self.base + path,
                    content=body, headers=headers, timeout=timeout or 30)
                break
            except httpx.TransportError as exc:
                self.retry_log.append({"route": path.split("?")[0].split("/api/question/")[0],
                    "attempt": attempt + 1, "error": type(exc).__name__, "elapsed_seconds": time.perf_counter() - began,
                    "retried": attempt < self.retries})
                if attempt >= self.retries:
                    raise AcceptanceError("transport_retries_exhausted") from None
                time.sleep(.1 * (attempt + 1))
        require(response.status_code in allowed, "http_" + str(response.status_code))
        try:
            value = response.text if raw else response.json()
        except ValueError:
            raise AcceptanceError("non_json_response") from None
        return value, time.perf_counter() - began, response.status_code


def database_records(dsn, run_ids):
    """Read only the created runs in one consistent PostgreSQL snapshot."""
    import sqlalchemy as sa
    if dsn.startswith("postgres://"):
        dsn = "postgresql+psycopg://" + dsn[len("postgres://"):]
    elif dsn.startswith("postgresql://"):
        dsn = "postgresql+psycopg://" + dsn[len("postgresql://"):]
    require(dsn.startswith("postgresql+psycopg://"), "audit_requires_postgresql")
    engine = sa.create_engine(dsn, pool_pre_ping=True)
    output = {key: [] for key in ("run", "episode", "frame", "event", "question")}
    try:
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as db, db.begin():
            db.exec_driver_sql("SET TRANSACTION READ ONLY")
            queries = {
                "run": "SELECT id, namespace, document FROM kitchen_runs WHERE id IN :ids",
                "episode": "SELECT id, run_id, document FROM kitchen_episodes WHERE run_id IN :ids",
                "event": "SELECT id, run_id, episode_id, operation_id, kind, document FROM kitchen_events WHERE run_id IN :ids",
                "question": "SELECT id, run_id, episode_id, frame, status, document FROM kitchen_questions WHERE run_id IN :ids",
                "frame": "SELECT f.episode_id, f.turn, f.snapshot, f.public FROM kitchen_frames f JOIN kitchen_episodes e ON f.episode_id=e.id WHERE e.run_id IN :ids",
            }
            for kind, query in queries.items():
                stmt = sa.text(query).bindparams(sa.bindparam("ids", expanding=True))
                for row in db.execute(stmt, {"ids": run_ids}).mappings():
                    value = dict(row)
                    for field in ("document", "snapshot", "public"):
                        if field in value and isinstance(value[field], str):
                            value[field] = json.loads(value[field])
                    output[kind].append(value)
    finally:
        engine.dispose()
    return output


def exported_records(client, run_ids):
    """Admin export fallback. Ignore every record outside this test's IDs.

    A pilot primary namespace exports no development freeplay rows. In that
    setup an independent read-only PostgreSQL audit is required; absence fails
    acceptance instead of silently dropping the audit or changing namespaces.
    """
    content, _, _ = client.request("/api/admin/export?format=jsonl", raw=True)
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    selected = set(run_ids)
    output = {key: [] for key in ("run", "episode", "frame", "event", "question")}
    for row in rows:
        kind, doc = row.get("type"), row.get("document", {})
        if kind == "run" and doc.get("id") in selected:
            output[kind].append({"id": doc["id"], "namespace": row.get("namespace"), "document": doc})
        elif kind in {"episode", "event", "question"} and row.get("run_id", doc.get("run_id")) in selected:
            output[kind].append(row)
    episodes = {row["id"] for row in output["episode"]}
    output["frame"] = [row for row in rows if row.get("type") == "frame" and row.get("episode_id") in episodes]
    return output


def public_state(value):
    return {key: val for key, val in value.items() if key != "episode_id"}


def audit_records(records, participants, binding):
    require(len(records["run"]) == SESSIONS, "persisted_run_count")
    require(len({r["document"]["participant_id"] for r in records["run"]}) == SESSIONS, "participant_identity_collision")
    config = binding["qa_configuration"]
    audit = {"mode": "persisted_private_diagnostics", "runs": 0, "joint_steps": 0, "frames": 0,
             "cloud_questions": 0, "shown_exposures": 0, "token_usage": {}, "returned_models": []}
    models = set()
    for participant in participants:
        run_id, episode_id = participant["run_id"], participant["episode_id"]
        run = next((row for row in records["run"] if row["id"] == run_id), None)
        require(run is not None and run["namespace"] == "development", "test_namespace_not_development")
        require(run["document"]["mode"] == "freeplay" and run["document"]["phase"] == "freeplay", "test_not_freeplay")
        require(run["document"]["versions"] == binding["versions"], "persisted_version_mismatch")
        episodes = [row for row in records["episode"] if row["run_id"] == run_id]
        require(len(episodes) == 1 and episodes[0]["id"] == episode_id, "episode_identity_mismatch")
        events = [row for row in records["event"] if row["run_id"] == run_id and row["kind"] == "joint_step"]
        require(len(events) == STEPS and {row["operation_id"] for row in events} == set(participant["action_operation_ids"]), "duplicate_or_missing_joint_steps")
        require(sorted(row["document"]["after"]["turn"] for row in events) == list(range(1, STEPS + 1)), "step_sequence_mismatch")
        frames = sorted((row for row in records["frame"] if row["episode_id"] == episode_id), key=lambda row: row["turn"])
        require([row["turn"] for row in frames] == list(range(STEPS + 1)), "duplicate_or_missing_frames")
        require(public_state(frames[-1]["public"]) == participant["final_state"], "confirmed_state_not_persisted")
        questions = [row for row in records["question"] if row["run_id"] == run_id]
        require(len(questions) == QUESTIONS and {row["id"] for row in questions} == set(participant["question_ids"]), "question_identity_mismatch")
        for row in questions:
            expected = participant["question_bindings"][row["id"]]
            doc, frame = row["document"], expected["frame"]
            answer = doc.get("answer") or {}
            diagnostics = answer.get("diagnostics") or {}
            require(row["status"] == "complete" and answer.get("verified") is True, "question_not_verified")
            require(row["episode_id"] == episode_id and row["frame"] == frame and answer.get("frame") == frame, "question_frame_mismatch")
            require(answer.get("kind") == expected["intent"], "free_question_intent_mismatch")
            require(doc.get("language") == participant["language"], "question_language_mismatch")
            require(diagnostics.get("llm_success") is True and diagnostics.get("parser_verified") is True
                    and not diagnostics.get("fallback"), "cloud_fallback_is_not_success")
            require(diagnostics.get("configuration") == config and diagnostics.get("actor_sha256") == binding["versions"]["actor_sha256"], "question_provenance_mismatch")
            require(diagnostics.get("source_sha256") == snapshot_digest(frames[frame]["snapshot"]), "question_source_mismatch")
            calls = diagnostics.get("calls", [])
            for stage in ("parse", "answer"):
                require(any(call.get("stage") == stage and call.get("http_status") == 200 and not call.get("error") for call in calls), "missing_real_cloud_stage")
            accepted = config.get("model_identity_policy", {}).get("accepted_returned_models", [])
            for call in calls:
                returned = call.get("returned_model")
                if call.get("http_status") == 200:
                    require(returned in accepted, "unexpected_cloud_model_identity")
                    models.add(returned)
                for key, value in (call.get("usage") or {}).items():
                    if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        audit["token_usage"][key] = audit["token_usage"].get(key, 0) + value
            if expected["intent"] == "counterfactual":
                branch = (answer.get("evidence") or {}).get("counterfactual") or {}
                require(branch.get("assumptions", {}).get("human_actions") == ["WAIT"] * 3
                        and branch.get("source_sha256") == diagnostics["source_sha256"]
                        and branch.get("final_state", {}).get("turn") == frame + 3, "counterfactual_binding_mismatch")
            audit["cloud_questions"] += 1
        exposures = [row for row in records["event"] if row["run_id"] == run_id and row["kind"] == "answer_exposure"]
        require(len(exposures) == QUESTIONS and {row["document"]["question_id"] for row in exposures} == set(participant["question_ids"])
                and all(row["document"]["event"] == "shown" for row in exposures), "exposure_idempotency_failure")
        audit["runs"] += 1; audit["joint_steps"] += len(events); audit["frames"] += len(frames); audit["shown_exposures"] += len(exposures)
    audit["returned_models"] = sorted(models)
    return audit


def run_acceptance(url, admin_key, *, dsn=None, allow_local=False, transport=None,
                   audit_reader=None, poll_interval=.15, question_timeout=90, checkpoint=None):
    base, local = validate_endpoint(url, allow_local)
    require(bool(admin_key), "admin_key_required")
    require(transport is None or audit_reader is not None, "fixture_requires_private_audit")
    report = {"schema_version": 1, "mode": "local_fixture" if local or transport is not None else "real_remote",
              "passed": False, "endpoint": base, "test_id": "remote-" + uuid.uuid4().hex,
              "started_at": datetime.now(timezone.utc).isoformat(), "scope": "Candidate freeplay technical capacity; not human effects, formal recruitment approval, cold-start or restart acceptance.",
              "expected_sessions": SESSIONS, "actions_per_session": STEPS, "questions_per_session": QUESTIONS,
              "test_namespace": "development", "thresholds": {"warm_action_p95_seconds": 1, "question_p95_seconds": 30},
              "errors": [], "participants": [], "transport_retries": []}
    admin = Client(base, transport=transport, admin_key=admin_key)
    clients = []
    start = time.perf_counter()
    try:
        if checkpoint:
            checkpoint(report)
        binding, warmup, _ = admin.request("/api/status", timeout=90)
        require(binding.get("policy_kind") == "neural", "neural_policy_required")
        require(binding.get("qa_configured") is True and binding.get("qa_configuration", {}).get("provider") == "deepseek", "configured_deepseek_required")
        require(binding.get("storage") == "postgresql" and not binding.get("test_mode"), "real_postgresql_runtime_required")
        require(all(binding.get("versions", {}).get(key) for key in ("actor_sha256", "program_sha256", "runtime_sha256")), "release_binding_required")
        report.update(versions=copy.deepcopy(binding["versions"]), qa_configuration=copy.deepcopy(binding["qa_configuration"]),
                      actor_sha256=binding["versions"]["actor_sha256"], program_sha256=binding["versions"]["program_sha256"],
                      runtime_sha256=binding["versions"]["runtime_sha256"], policy_kind=binding["policy_kind"],
                      study_ready_at_start=binding["study_ready"], service_namespace=binding["namespace"], warmup_seconds=warmup)
        admin_status, _, _ = admin.request("/api/admin/status")
        require(admin_status.get("service", {}).get("versions") == binding["versions"], "admin_release_binding_mismatch")
        for number in range(SESSIONS):
            client = Client(base, transport=transport)
            clients.append(client)
            language = "en" if number % 2 else "zh"
            create_id = report["test_id"] + "-join-" + str(number)
            participant = {"number": number, "language": language, "creation_operation_id": create_id,
                           "action_operation_ids": [], "question_ids": [], "question_bindings": {}, "action_seconds": [], "duplicate_action_seconds": [], "question_seconds": [],
                           "duplicate_actions": 0, "duplicate_exposures": 0, "completed": False}
            report["participants"].append(participant)
            view, _, _ = client.request("/api/session", {"operation_id": create_id, "mode": "freeplay", "language": language})
            participant["run_id"] = view["run"]["id"]
            if checkpoint:
                checkpoint(report)
            require(view["run"]["mode"] == "freeplay", "join_not_freeplay")
            view, _, _ = client.request("/api/command", {"operation_id": uuid.uuid4().hex, "version": view["run"]["version"], "command": "next"})
            participant["episode_id"] = view["run"]["episode_id"]
            participant["initial_view"] = view
            if checkpoint:
                checkpoint(report)
        require(len({p["run_id"] for p in report["participants"]}) == SESSIONS, "session_identity_collision")
        barrier = threading.Barrier(SESSIONS)

        def participant_work(number):
            client, result = clients[number], report["participants"][number]
            view = result.pop("initial_view")
            run_id, episode_id = result["run_id"], result["episode_id"]
            try:
                barrier.wait(timeout=30)
                for step in range(STEPS):
                    payload = {"operation_id": uuid.uuid4().hex, "version": view["run"]["version"], "command": "action",
                               "action": ["UP", "INTERACT", "WAIT", "RIGHT", "DOWN"][step % 5]}
                    result["action_operation_ids"].append(payload["operation_id"])
                    view, elapsed, _ = client.request("/api/command", payload)
                    result["action_seconds"].append(elapsed)
                    require(view["run"]["id"] == run_id and view["run"]["episode_id"] == episode_id, "session_mixup")
                    require(view["state"]["turn"] == step + 1, "wrong_confirmed_step")
                    if step in (0, 9):
                        repeated, duplicate_elapsed, _ = client.request("/api/command", payload)
                        result["duplicate_action_seconds"].append(duplicate_elapsed)
                        require(repeated["run"] == view["run"] and repeated["state"] == view["state"], "duplicate_request_advanced_state")
                        result["duplicate_actions"] += 1
                    if step not in (3, 7):
                        continue
                    before = copy.deepcopy(view["state"])
                    frame, intent = (0, "why") if step == 3 else (step + 1, "counterfactual")
                    question = (("请解释所选画面中神经队友下一次会选择什么动作，以及依据。" if step == 3 else "如果我从所选画面开始连续等待三步，队友和厨房会怎样变化？")
                                if result["language"] == "zh" else ("What action will the neural teammate choose next at this selected frame, and why?" if step == 3 else "What will happen to the teammate and kitchen if I wait for three consecutive steps from this selected frame?"))
                    begun = time.perf_counter()
                    job, _, _ = client.request("/api/question", {"operation_id": uuid.uuid4().hex, "version": view["run"]["version"],
                        "episode_id": episode_id, "frame": frame, "kind": "free", "question": question})
                    result["question_ids"].append(job["id"])
                    result["question_bindings"][job["id"]] = {"frame": frame, "intent": intent}
                    while True:
                        answer, _, _ = client.request("/api/question/" + job["id"])
                        if answer["status"] in {"complete", "failed", "cancelled"}:
                            break
                        require(time.perf_counter() - begun < question_timeout, "question_deadline_exceeded")
                        time.sleep(poll_interval)
                    result["question_seconds"].append(time.perf_counter() - begun)
                    require(answer["status"] == "complete" and (answer.get("answer") or {}).get("verified") is True, "public_question_failed")
                    require(answer.get("episode_id") == episode_id and answer["answer"].get("frame") == frame, "public_answer_binding_mismatch")
                    exposure = {"operation_id": uuid.uuid4().hex, "question_id": job["id"], "event": "shown"}
                    client.request("/api/exposure", exposure); client.request("/api/exposure", exposure)
                    result["duplicate_exposures"] += 1
                    view, _, _ = client.request("/api/view")
                    require(view["run"]["id"] == run_id and view["state"] == before, "question_changed_game_or_session")
                history, _, _ = client.request("/api/history?episode_id=" + episode_id)
                require(history["episode_id"] == episode_id and [frame["turn"] for frame in history["frames"]] == list(range(STEPS + 1)), "history_not_confirmed_sequence")
                # A new HTTP client carries this session's cookie exactly as a
                # browser refresh does. Cookies never enter the report.
                refresh = Client(base, transport=transport, cookies=client.client.cookies)
                try:
                    restored, _, _ = refresh.request("/api/view")
                    require(restored["run"] == view["run"] and restored["state"] == view["state"], "refresh_lost_confirmed_state")
                    result["refresh_verified"] = True
                    result["refresh_retries"] = refresh.retry_log
                finally:
                    refresh.close()
                result["final_state"] = public_state(view["state"])
                result["final_state_sha256"] = digest(result["final_state"])
                result["completed"] = True
            except Exception as exc:
                result["error"] = str(exc) if isinstance(exc, AcceptanceError) else type(exc).__name__

        with ThreadPoolExecutor(max_workers=SESSIONS) as pool:
            list(pool.map(participant_work, range(SESSIONS)))
        for result in report["participants"]:
            if result.get("error"):
                report["errors"].append({"participant": result["number"], "code": result["error"]})
        require(all(p["completed"] for p in report["participants"]), "incomplete_sessions")
        for number, client in enumerate(clients):
            other = report["participants"][(number + 1) % SESSIONS]
            client.request("/api/history?episode_id=" + other["episode_id"], allowed=(403, 404))
            client.request("/api/question/" + other["question_ids"][0], allowed=(403, 404))
        report["cross_session_denials"] = SESSIONS * 2
        final_status, _, _ = admin.request("/api/status")
        require(final_status.get("versions") == binding["versions"] and final_status.get("qa_configuration") == binding["qa_configuration"], "release_changed_during_test")
        ids = [p["run_id"] for p in report["participants"]]
        if audit_reader:
            rows = audit_reader(ids)
            report["audit_source"] = "injected_fixture"
        elif dsn:
            rows = database_records(dsn, ids)
            report["audit_source"] = "read_only_postgresql_snapshot"
        else:
            rows = exported_records(admin, ids)
            report["audit_source"] = "authenticated_namespace_export"
        report["audit"] = audit_records(rows, report["participants"], binding)
        report.update(zero_duplicate_steps=True, zero_session_mixups=True, zero_confirmed_data_loss=True)
    except Exception as exc:
        report["errors"].append({"code": str(exc) if isinstance(exc, AcceptanceError) else type(exc).__name__})
    finally:
        report["transport_retries"] = [{"client": "admin", **item} for item in admin.retry_log]
        admin.close()
        for number, client in enumerate(clients):
            report["transport_retries"] += [{"client": number, **item} for item in client.retry_log]
            client.close()
        actions = [value for p in report["participants"] for value in p["action_seconds"]]
        duplicate_actions = [value for p in report["participants"] for value in p["duplicate_action_seconds"]]
        questions = [value for p in report["participants"] for value in p["question_seconds"]]
        report.update(sessions=sum(p["completed"] for p in report["participants"]), actions=len(actions), questions=len(questions),
                      action_acknowledgements=len(actions) + len(duplicate_actions),
                      action_p95_seconds=percentile(actions + duplicate_actions), question_p95_seconds=percentile(questions),
                      duplicate_action_retries=sum(p["duplicate_actions"] for p in report["participants"]),
                      duplicate_exposure_retries=sum(p["duplicate_exposures"] for p in report["participants"]),
                      duration_seconds=time.perf_counter() - start)
        for p in report["participants"]:
            report["transport_retries"] += [{"client": str(p["number"]) + "-refresh", **item} for item in p.pop("refresh_retries", [])]
            p.pop("initial_view", None); p.pop("final_state", None)
        if not report["errors"]:
            if report["action_p95_seconds"] > 1:
                report["errors"].append({"code": "warm_action_p95_over_one_second"})
            if report["question_p95_seconds"] > 30:
                report["errors"].append({"code": "question_p95_over_thirty_seconds"})
        report["passed"] = not report["errors"] and report["sessions"] == SESSIONS and report.get("audit", {}).get("cloud_questions") == SESSIONS * QUESTIONS
        report["transport_retry_count"] = sum(bool(item["retried"]) for item in report["transport_retries"])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, help="Private owner-only dotenv file; values are never printed.")
    parser.add_argument("--output", type=Path, required=True, help="New report path; existing files are never overwritten.")
    parser.add_argument("--run", action="store_true", help="Explicitly authorize this invocation's 20 sessions / 40 billed cloud questions.")
    parser.add_argument("--allow-local-test", action="store_true", help="Loopback only; results are marked local_fixture.")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("No requests were made. Pass --run to start the real, potentially billable test.")
    try:
        values = private_environment(args.env_file)
        url = values.get("KITCHEN_ACCEPTANCE_URL") or values.get("KITCHEN_URL", "")
        validate_endpoint(url, args.allow_local_test)
        require(bool(values.get("KITCHEN_ADMIN_KEY")), "admin_key_required")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except Exception as exc:
        parser.exit(2, "Acceptance configuration failed: " + (str(exc) if isinstance(exc, AcceptanceError) else type(exc).__name__) + ". No credentials were logged.\n")
    # Reserve the output before network work so failure cannot replace an older
    # pass. If interrupted, this initial record explicitly remains unpassed.
    with os.fdopen(descriptor, "w") as destination:
        json.dump({"passed": False, "status": "running_or_interrupted"}, destination); destination.flush(); os.fsync(destination.fileno())
        def save_progress(value):
            safe = copy.deepcopy(value)
            for participant in safe["participants"]:
                participant.pop("initial_view", None)
                participant.pop("final_state", None)
            safe["status"] = "running_or_interrupted"
            destination.seek(0); json.dump(safe, destination, ensure_ascii=False, indent=2); destination.truncate()
            destination.flush(); os.fsync(destination.fileno())
        report = run_acceptance(url, values["KITCHEN_ADMIN_KEY"],
            dsn=values.get("KITCHEN_DIRECT_DATABASE_URL") or values.get("DATABASE_URL"), allow_local=args.allow_local_test,
            checkpoint=save_progress)
        destination.seek(0); json.dump(report, destination, ensure_ascii=False, indent=2); destination.truncate()
        destination.flush(); os.fsync(destination.fileno())
    print(json.dumps({key: report.get(key) for key in ("mode", "passed", "sessions", "actions", "questions", "action_p95_seconds", "question_p95_seconds", "transport_retry_count", "errors")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
