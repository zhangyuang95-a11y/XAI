#!/usr/bin/env python3
"""Two-phase real-HTTPS acceptance for the r4.1-diagnostic v8 service.

Run ``before`` against a fresh QA database, restart the exact same service,
then run ``after`` with the same output directory.  The checker uses four
independent cookie jars, public HTTP commands, and read-only SQLite audit
queries.  It never imports a model, environment, explainer, or release loader.

``--allow-synthetic-fixture`` exists only to test this checker on loopback.  A
report produced with that flag is explicitly ineligible as release evidence.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from copy import deepcopy
import gzip
from hashlib import sha256
import http.cookiejar
import json
import os
from pathlib import Path
import re
import sqlite3
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4


VERSION = "warehouse-r41-diagnostic-v8-real-https-flow.v1"
SERVICE_FAMILY = "warehouse_alignment_online_r2"
NAMESPACE = "online_diagnostic"
CONTEXT_VERSION = "warehouse-r41-diagnostic-online-release.v8"
COOKIE = "warehouse_alignment_r41_diagnostic_session_v1"
ENDPOINT = "/api/study/command"
HEX = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,31}\Z")
CAP = {"participants": 4, "http_requests": 900, "gameplay_actions": 40,
       "questions": 16}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def save_new(path, value):
    path = Path(path)
    raw = (canonical(value) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return {"path": str(path), "sha256": sha256(raw).hexdigest(),
            "size": len(raw)}


def public_only(value):
    forbidden = {"probabilities", "logits", "decision", "policy_actions",
                 "proposed_actions", "snapshot", "actor_sha256",
                 "package_sha256", "manifest_sha256", "fingerprint", "seed"}
    if isinstance(value, dict):
        require(not forbidden.intersection(value),
                "participant response leaked operator or policy structure")
        for child in value.values():
            public_only(child)
    elif isinstance(value, list):
        for child in value:
            public_only(child)


def physical(view):
    return {"run_id": view.get("run_id"), "state": view.get("state"),
            "metrics": view.get("metrics"), "ended": view.get("ended"),
            "history_count": view.get("history_count"),
            "stage": view.get("flow", {}).get("stage"),
            "round_index": view.get("flow", {}).get("round_index")}


def validate_view(view):
    public_only(view)
    release = view.get("release", {})
    require(release.get("release_version") == "r4.1-diagnostic",
            "wrong public release version")
    require(release.get("pilot_class") == "internal_diagnostic",
            "wrong diagnostic classification")
    require(release.get("model_ready") is True
            and release.get("explanation_ready") is True
            and release.get("study_ready") is True,
            "diagnostic capability is not admitted")
    require(release.get("formal_ready") is False
            and release.get("formal_sample_eligible") is False
            and release.get("data_persistent") is False,
            "diagnostic release overstates study eligibility or durability")
    require(view.get("enrollment", {}).get("mode") == "internal_diagnostic",
            "wrong enrollment mode")
    require(view.get("enrollment", {}).get("formal_sample_eligible") is False,
            "participant view claims formal eligibility")
    require("condition" not in view.get("flow", {})
            and "task_order" not in view.get("flow", {}),
            "participant view leaked allocation")
    return view


def database(path):
    requested = Path(path).expanduser().absolute()
    require(not requested.is_symlink() and requested.is_file(),
            "QA database must be a canonical regular file")
    requested = requested.resolve()
    require(requested.name.endswith("_diagnostic_v8_http_qa.sqlite3"),
            "use a separate *_diagnostic_v8_http_qa.sqlite3 database")
    db = sqlite3.connect(requested.as_uri() + "?mode=ro", uri=True, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def database_identity(path, *, package_sha256, manifest_sha256,
                      actor_sha256, empty=False):
    with closing(database(path)) as db:
        metadata = {row["key"]: json.loads(row["value"])
                    for row in db.execute("SELECT key,value FROM metadata")}
        require(metadata.get("service_family") == SERVICE_FAMILY
                and metadata.get("namespace") == NAMESPACE,
                "QA database belongs to another service")
        contexts = [value for key, value in metadata.items()
                    if key.startswith("service_context:")]
        require(len(contexts) == 1, "QA database must bind one service context")
        context = contexts[0]
        require(context.get("version") == CONTEXT_VERSION,
                "QA database does not bind v8")
        require(context.get("package_sha256") == package_sha256
                and context.get("manifest_sha256") == manifest_sha256
                and context.get("actor_sha256") == actor_sha256,
                "QA database identity differs from declared release")
        require(context.get("pilot_class") == "internal_diagnostic"
                and context.get("formal_sample_eligible") is False
                and context.get("data_persistent") is False,
                "stored context misclassifies the diagnostic release")
        if empty:
            for table in ("sessions", "runs", "frames", "questions",
                          "operations", "events", "blocks"):
                require(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0,
                        f"fresh QA database is not empty: {table}")
        return context


def allocation(path, participant_id):
    with closing(database(path)) as db:
        row = db.execute(
            "SELECT id,participant_id,condition,task_order,position,stage "
            "FROM sessions WHERE participant_id=?", (participant_id,)).fetchone()
        require(row is not None, "registered participant is absent from QA database")
        return dict(row)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        raise ValueError("acceptance checker does not follow redirects")


class Harness:
    def __init__(self, args, output, *, append=False):
        self.args = args
        self.base = args.base.rstrip("/")
        self.output = Path(output)
        self.requests = self.actions = self.questions = 0
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_EXCL)
        descriptor = os.open(self.output / "events.jsonl", flags, 0o600)
        self.events = os.fdopen(descriptor, "a", encoding="utf-8")
        self.ssl_context = ssl.create_default_context(cafile=str(args.ca_certificate))

    def close(self):
        self.events.close()

    def record(self, value):
        self.events.write(canonical({"time": time.time(), **value}) + "\n")
        self.events.flush(); os.fsync(self.events.fileno())

    def request(self, client, path, payload=None, *, expected=200, error=None,
                adopt=False):
        require(self.requests < CAP["http_requests"], "HTTP request cap exhausted")
        self.requests += 1
        request_id = f"{self.args.phase}_{self.requests:04d}"
        data = None if payload is None else canonical(payload).encode()
        request = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json", "Origin": self.base})
        self.record({"event": "request", "id": request_id,
                     "client": client.name, "path": path,
                     "method": "GET" if data is None else "POST",
                     "payload_sha256": None if payload is None else digest(payload)})
        started = time.monotonic()
        try:
            response = client.opener.open(request, timeout=60)
        except urllib.error.HTTPError as exception:
            response = exception
        with response:
            status = response.status
            headers = response.headers
            raw = response.read()
        response_file = self.output / f"response_{request_id}.json.gz"
        descriptor = os.open(response_file,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1,
                               mtime=0) as zipped:
                zipped.write(raw)
            stream.flush(); os.fsync(stream.fileno())
        value = json.loads(raw)
        public_only(value)
        require(headers.get("Cache-Control") == "no-store",
                "response is cacheable")
        require(headers.get("X-Frame-Options") == "DENY",
                "frame protection is absent")
        require(status == expected,
                f"unexpected HTTP {status}, expected {expected}: {value}")
        if error is not None:
            require(value.get("error") == error,
                    f"unexpected rejection: {value}")
        if status == 200 and path.startswith("/api/"):
            if path == "/api/view" or path == ENDPOINT:
                validate_view(value)
        self.record({"event": "response", "id": request_id,
                     "status": status, "elapsed_seconds": time.monotonic() - started,
                     "response_file": response_file.name,
                     "response_sha256": file_hash(response_file)})
        client.save_cookies()
        if adopt and status == 200:
            client.view = value
        return value


class Client:
    def __init__(self, harness, name, *, load=False):
        self.harness, self.name = harness, name
        self.cookie_path = harness.output / f"{name}.cookies.txt"
        self.jar = http.cookiejar.MozillaCookieJar(str(self.cookie_path))
        if load:
            require(self.cookie_path.is_file()
                    and self.cookie_path.stat().st_mode & 0o077 == 0,
                    "saved cookie jar is missing or not private")
            self.jar.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=harness.ssl_context),
            urllib.request.HTTPCookieProcessor(self.jar), NoRedirect())
        self.view = None

    @property
    def cookie(self):
        values = [item.value for item in self.jar if item.name == COOKIE]
        require(len(values) == 1, "client does not own exactly one diagnostic cookie")
        return values[0]

    def save_cookies(self):
        if len(self.jar):
            self.jar.save(ignore_discard=True, ignore_expires=True)
            os.chmod(self.cookie_path, 0o600)

    def get(self):
        return self.harness.request(self, "/api/view", adopt=True)

    def refresh_client(self):
        before = physical(self.view)
        self.save_cookies()
        replacement = Client(self.harness, self.name, load=True)
        value = replacement.get()
        require(physical(value) == before, "fresh HTTPS client changed confirmed state")
        self.jar, self.opener, self.view = (
            replacement.jar, replacement.opener, replacement.view)
        return value

    def command(self, kind, *, expected=200, error=None, adopt=True,
                operation_id=None, expected_version=None, **fields):
        require(self.view is not None, "client view is not initialized")
        payload = {"operation_id": operation_id or uuid4().hex,
                   "expected_version": (self.view["version"] if expected_version is None
                                        else expected_version),
                   "kind": kind, **fields}
        value = self.harness.request(
            self, ENDPOINT, payload, expected=expected, error=error,
            adopt=adopt and expected == 200)
        return payload, value

    def action(self, action="WAIT"):
        require(self.harness.actions < CAP["gameplay_actions"],
                "gameplay action cap exhausted")
        before = deepcopy(self.view)
        payload, value = self.command("action", action=action)
        require(value["run_id"] == before["run_id"]
                and value["state"]["frame"] == before["state"]["frame"] + 1,
                "one action did not advance exactly one joint step")
        self.harness.actions += 1
        return payload, value


def history(client, run_id=None, *, expected=200, error=None):
    before = physical(client.view)
    query = "" if run_id is None else "?" + urllib.parse.urlencode({"run_id": run_id})
    value = client.harness.request(
        client, "/api/history" + query, expected=expected, error=error)
    if expected == 200:
        require(value["run_id"] == client.view["run_id"],
                "history belongs to another run")
        frames = value["frames"]
        require([row["state"]["frame"] for row in frames]
                == list(range(client.view["state"]["frame"] + 1)),
                "history has missing or duplicate frames")
        require(frames[-1]["state"] == client.view["state"],
                "history head differs from live state")
    require(physical(client.view) == before, "history read changed live state")
    return value


def ask(client, question, *, language, frame, focus):
    require(client.harness.questions < CAP["questions"], "question cap exhausted")
    before = physical(client.view)
    previous_ids = {row.get("id") for row in client.view.get("answers", [])}
    payload, value = client.command(
        "question", run_id=client.view["run_id"], frame=frame,
        question=question, language=language, focus=focus)
    client.harness.questions += 1
    matches = [row for row in value.get("answers", [])
               if row.get("id") not in previous_ids
               and row.get("question") == question and row.get("frame") == frame]
    require(len(matches) == 1, "submitted question has no unique public identity")
    answer_id = matches[0]["id"]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        client.get()
        match = next((row for row in client.view.get("answers", [])
                      if row.get("id") == answer_id), None)
        if match is not None and match.get("status") not in {"pending", "running"}:
            break
        time.sleep(0.05)
    require(match is not None and match.get("status") == "complete"
            and match.get("text", "").strip(), "verified answer did not complete")
    require(match.get("run_id") == payload["run_id"]
            and match.get("frame") == frame, "answer rebound to another frame or run")
    require(isinstance(match.get("evidence_detail"), str),
            "answer evidence detail has the wrong public type")
    require(physical(client.view) == before,
            "question or explanation changed physical game state")
    client.command("answer_seen", answer_id=answer_id)
    return {"id": answer_id, "run_id": payload["run_id"], "frame": frame,
            "language": language, "text_sha256": sha256(
                match["text"].encode()).hexdigest(),
            "evidence_sha256": sha256(
                match.get("evidence_detail", "").encode()).hexdigest()}


def forbidden_question(client, *, answer_id=None):
    before = physical(client.view)
    client.command(
        "question", expected=403, error="explanation_forbidden", adopt=False,
        run_id=client.view["run_id"], frame=client.view["state"]["frame"],
        question="Why did Robot 2 choose that action?", language="en",
        focus="executed")
    if answer_id:
        client.command("answer_seen", expected=403,
                       error="explanation_forbidden", adopt=False,
                       answer_id=answer_id)
    client.get()
    require(not client.view.get("explain_allowed")
            and client.view.get("answers") == [],
            "forbidden stage exposed explanation data")
    require(physical(client.view) == before,
            "forbidden explanation request changed physical state")


def _question(language, kind):
    values = {
        "zh": {"next": "机器人2当前在朝哪个任务前进？",
               "executed": "机器人2刚才为什么这样行动？",
               "ended": "本局结束前，机器人2刚才为什么这样行动？"},
        "en": {"next": "Which task is Robot 2 moving toward?",
               "executed": "Why did Robot 2 choose that action?",
               "ended": "Before this round ended, why did Robot 2 take that action?"},
    }
    return values[language][kind]


def _tutorial(client, mode):
    require(client.view["flow"]["stage"] == "instructions"
            and client.view.get("tutorial"), "registration did not enter tutorial")
    require(client.view["tutorial"]["duration_ms"] == 380
            and client.view["tutorial"]["scored"] is False,
            "tutorial timing or score boundary differs")
    if mode == "complete":
        client.command("tutorial_advance")
        client.command("tutorial_select", frame_index=0)
        client.command("tutorial_advance")
        client.command("tutorial_restart")
        while not client.view["tutorial"]["complete"]:
            client.command("tutorial_advance")
    elif mode == "partial":
        client.command("tutorial_advance")
        client.refresh_client()
    client.command("begin_task1")
    require(client.view["flow"]["stage"] == "task1"
            and client.view["flow"]["round_index"] == 1,
            "tutorial did not enter Task 1 round 1")


def _plan(args, output):
    return {
        "version": VERSION, "phase": "before", "base": args.base.rstrip("/"),
        "database": str(Path(args.database).resolve()),
        "ca_certificate": str(Path(args.ca_certificate).resolve()),
        "ca_certificate_sha256": file_hash(args.ca_certificate),
        "expected_package_sha256": args.expected_package_sha256,
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "expected_actor_sha256": args.expected_actor_sha256,
        "script_sha256": file_hash(__file__), "caps": CAP,
        "synthetic_fixture": bool(args.allow_synthetic_fixture),
        "release_acceptance_eligible": not args.allow_synthetic_fixture,
        "restart_protocol": "run before, restart the exact same service, run after",
        "output": str(Path(output).resolve()),
    }


def run_before(args):
    output = Path(args.output).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    plan = _plan(args, output)
    save_new(output / "plan.json", plan)
    database_identity(
        args.database, package_sha256=args.expected_package_sha256,
        manifest_sha256=args.expected_manifest_sha256,
        actor_sha256=args.expected_actor_sha256, empty=True)
    harness = Harness(args, output)
    clients = []
    try:
        token = uuid4().hex[:8]
        for index in range(4):
            participant = f"qa_v8_{token}_{index + 1}"
            require(ID.fullmatch(participant), "generated QA participant ID is invalid")
            client = Client(harness, f"client_{index + 1}")
            view = client.get()
            require(view["flow"]["stage"] == "registration"
                    and view["version"] == 0, "new cookie did not enter registration")
            client.command("start", mode="study", participant_id=participant,
                           consent=True)
            client.participant_id = participant
            _tutorial(client, ("complete", "partial", "skip", "skip")[index])
            clients.append(client)
        cells = [allocation(args.database, c.participant_id) for c in clients]
        require({(row["condition"], row["task_order"]) for row in cells}
                == {("A", "XY"), ("A", "YX"), ("B", "XY"), ("B", "YX")},
                "first four participants do not cover the randomized block")
        by_id = {row["participant_id"]: row for row in cells}
        for client in clients:
            client.allocation = by_id[client.participant_id]
            expected = client.allocation["condition"] == "A"
            require(client.view["explain_allowed"] is expected
                    and (("question" in client.view["allowed_kinds"]) is expected),
                    "Task 1 public permission differs from stored allocation")
        other_run = clients[1].view["run_id"]
        history(clients[0], other_run, expected=403,
                error="history_current_run_only")
        answers = {client.name: [] for client in clients}
        for index, client in enumerate(clients):
            language = "zh" if index % 2 == 0 else "en"
            if client.allocation["condition"] == "A":
                answers[client.name].append(ask(
                    client, _question(language, "next"), language=language,
                    frame=0, focus="next"))
            else:
                forbidden_question(client)
        lost_client = next(c for c in clients
                           if c.allocation["condition"] == "B")
        lost_before = deepcopy(lost_client.view)
        lost_payload = {"operation_id": "lost_" + uuid4().hex,
                        "expected_version": lost_before["version"],
                        "kind": "action", "action": "RIGHT"}
        lost_response = harness.request(
            lost_client, ENDPOINT, lost_payload, adopt=False)
        harness.actions += 1
        require(lost_response["state"]["frame"]
                == lost_before["state"]["frame"] + 1,
                "lost-response action did not commit one step")
        for client in clients:
            if client is not lost_client:
                client.action("WAIT")
        for index, client in enumerate(clients):
            language = "zh" if index % 2 == 0 else "en"
            if client.allocation["condition"] == "A":
                history(client)
                answers[client.name].append(ask(
                    client, _question(language, "next"), language=language,
                    frame=0, focus="next"))
        restored = {c.name: (lost_response if c is lost_client else c.view)
                    for c in clients}
        for client in clients:
            client.save_cookies()
        checkpoint = {
            "version": VERSION, "plan_sha256": file_hash(output / "plan.json"),
            "service_context_sha256": digest(database_identity(
                args.database, package_sha256=args.expected_package_sha256,
                manifest_sha256=args.expected_manifest_sha256,
                actor_sha256=args.expected_actor_sha256)),
            "clients": [{"name": c.name, "participant_id": c.participant_id,
                         "allocation": c.allocation,
                         "restored_physical": physical(restored[c.name]),
                         "restored_version": restored[c.name]["version"],
                         "answers": answers[c.name]}
                        for c in clients],
            "lost_response": {"client": lost_client.name,
                              "payload": lost_payload,
                              "committed_view_sha256": digest(lost_response),
                              "committed_physical": physical(lost_response)},
            "before_counts": {"requests": harness.requests,
                              "actions": harness.actions,
                              "questions": harness.questions},
            "restart_required": True,
        }
        save_new(output / "restart_checkpoint.json", checkpoint)
        result = {"version": VERSION,
                  "status": "awaiting_exact_same_service_restart",
                  "release_acceptance_eligible": False,
                  "synthetic_fixture": bool(args.allow_synthetic_fixture),
                  "participants": [{"participant_id": row["participant_id"],
                                    "condition": row["condition"],
                                    "task_order": row["task_order"]}
                                   for row in cells],
                  "checkpoint_sha256": file_hash(output / "restart_checkpoint.json")}
        save_new(output / "before_report.json", result)
        print(canonical(result))
        return result
    except BaseException as error:
        save_new(output / "before_failure.json", {
            "version": VERSION, "error": repr(error),
            "requests": harness.requests, "actions": harness.actions,
            "questions": harness.questions, "passed": False})
        raise
    finally:
        harness.close()


def _answer_questionnaire(client):
    items = client.view.get("questionnaire", {}).get("items", [])
    require(len(items) == 11, "questionnaire is not frozen 8+3")
    answers = {}
    for item in items:
        options = item.get("options", [])
        require(options, "questionnaire item has no allowed option")
        first = options[0]
        value = first.get("value", first.get("id")) if isinstance(first, dict) else first
        answers[item["id"]] = int(value) if item.get("type") == "scale" else str(value)
    partial = dict(list(answers.items())[:4])
    client.command("questionnaire", answers=partial, submit=False)
    client.refresh_client()
    require(client.view["questionnaire"]["draft"] == partial,
            "questionnaire draft did not survive refresh")
    client.command("questionnaire", answers=answers, submit=True)
    require(client.view["flow"]["stage"] == "completed"
            and client.view.get("answers") == [],
            "questionnaire did not complete or exposed old answers")


def _finish_round(client, *, action=True):
    if action:
        client.action("WAIT")
    if not client.view["ended"]:
        client.command("end")
    require(client.view["ended"] is True, "round did not end")


def _database_final(path, checkpoint):
    result = []
    with closing(database(path)) as db:
        require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok",
                "SQLite integrity check failed")
        for item in checkpoint["clients"]:
            session = db.execute(
                "SELECT * FROM sessions WHERE participant_id=?",
                (item["participant_id"],)).fetchone()
            require(session is not None and session["stage"] == "completed",
                    "participant flow is incomplete")
            require((session["condition"], session["task_order"])
                    == (item["allocation"]["condition"],
                        item["allocation"]["task_order"]),
                    "allocation changed across restart")
            runs = list(db.execute(
                "SELECT rowid,* FROM runs WHERE session_id=? ORDER BY rowid",
                (session["id"],)))
            require(len(runs) == 6, "study does not contain exactly six rounds")
            require([(row["stage"], row["round_index"]) for row in runs]
                    == [("task1", 0), ("task1", 1), ("task1", 2),
                        ("task2", 0), ("task2", 1), ("task2", 2)],
                    "six-round stage order differs")
            expected_indices = ([1, 2, 3, 4, 5, 6]
                                if session["task_order"] == "XY"
                                else [4, 5, 6, 1, 2, 3])
            actual_indices = [json.loads(row["provenance"])["scene_index"]
                              for row in runs]
            require(actual_indices == expected_indices,
                    "X/Y scene order differs from allocation")
            action_count = 0
            for run in runs:
                require(run["ended"] == 1, "saved round is not ended")
                metrics = json.loads(run["metrics"])
                require(metrics.get("overrides") == 0,
                        "saved metrics report an action override")
                frames = list(db.execute(
                    "SELECT frame,internal FROM frames WHERE run_id=? ORDER BY frame",
                    (run["id"],)))
                require([row["frame"] for row in frames]
                        == list(range(metrics["steps"] + 1)),
                        "saved frames have a gap or duplicate")
                action_count += metrics["steps"]
                for frame in frames[1:]:
                    internal = json.loads(frame["internal"])
                    policy = internal.get("decision", {}).get(
                        "policy_actions", {}).get("robot_2")
                    submitted = internal.get("submitted_actions", {}).get("robot_2")
                    require(policy is not None and policy == submitted,
                            "NN action was not submitted unchanged")
                    requested = internal.get("info", {}).get(
                        "requested_actions", {}).get("robot_2")
                    if requested is not None:
                        require(requested == policy,
                                "environment received another robot action")
            questions = list(db.execute(
                "SELECT * FROM questions WHERE session_id=? ORDER BY created,id",
                (session["id"],)))
            expected_question_count = (4 if session["condition"] == "A" else 0)
            require(len(questions) == expected_question_count,
                    "questions crossed A/B boundaries or are missing")
            require(all(row["stage"] == "task1" and row["status"] == "complete"
                        and row["shown"] and row["answer"]
                        and row["evidence_detail"] is not None
                        for row in questions),
                    "stored Task 1 answer evidence is incomplete")
            tutorial = db.execute(
                "SELECT payload FROM events WHERE session_id=? "
                "AND kind='tutorial_acknowledged'", (session["id"],)).fetchone()
            require(tutorial is not None
                    and json.loads(tutorial[0]).get("scored") is False,
                    "tutorial is missing or scored")
            require(len(json.loads(session["questionnaire"])) == 11
                    and session["questionnaire_scores"] is not None,
                    "questionnaire record is incomplete")
            result.append({"participant_id": session["participant_id"],
                           "condition": session["condition"],
                           "task_order": session["task_order"],
                           "run_ids_sha256": digest([row["id"] for row in runs]),
                           "confirmed_actions": action_count,
                           "questions": len(questions)})
    return result


def run_after(args):
    output = Path(args.output).expanduser().resolve(strict=False)
    require(output.is_dir(), "before-phase output directory is missing")
    checkpoint_path = output / "restart_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    expected_plan = _plan(args, output)
    require(plan == expected_plan, "after-phase arguments differ from before phase")
    require(checkpoint["version"] == VERSION
            and checkpoint["plan_sha256"] == file_hash(output / "plan.json"),
            "restart checkpoint binding differs")
    context = database_identity(
        args.database, package_sha256=args.expected_package_sha256,
        manifest_sha256=args.expected_manifest_sha256,
        actor_sha256=args.expected_actor_sha256)
    require(digest(context) == checkpoint["service_context_sha256"],
            "service context changed across restart")
    harness = Harness(args, output, append=True)
    clients = []
    try:
        by_name = {item["name"]: item for item in checkpoint["clients"]}
        for item in checkpoint["clients"]:
            client = Client(harness, item["name"], load=True)
            client.participant_id = item["participant_id"]
            client.allocation = item["allocation"]
            client.get()
            require(physical(client.view) == item["restored_physical"]
                    and client.view["version"] == item["restored_version"],
                    "same-version process restart did not restore state")
            clients.append(client)
        lost = checkpoint["lost_response"]
        lost_client = next(client for client in clients
                           if client.name == lost["client"])
        before_retry = deepcopy(lost_client.view)
        retried = harness.request(
            lost_client, ENDPOINT, lost["payload"], adopt=True)
        require(digest(retried) == lost["committed_view_sha256"]
                and physical(retried) == lost["committed_physical"]
                and retried["version"] == before_retry["version"],
                "lost-response retry was not idempotent")
        answers = {item["name"]: list(item["answers"])
                   for item in checkpoint["clients"]}
        for index, client in enumerate(clients):
            language = "zh" if index % 2 == 0 else "en"
            if client.allocation["condition"] == "A":
                answers[client.name].append(ask(
                    client, _question(language, "executed"), language=language,
                    frame=client.view["state"]["frame"], focus="executed"))
            else:
                forbidden_question(client)
            first_run = client.view["run_id"]
            if not client.view["ended"]:
                client.command("end")
            if client.allocation["condition"] == "A":
                answers[client.name].append(ask(
                    client, _question(language, "ended"), language=language,
                    frame=client.view["state"]["frame"], focus="executed"))
            else:
                forbidden_question(client)
            client.first_task1_run = first_run
            client.old_answer = answers[client.name][0]["id"] if answers[client.name] else None
            client.command("next")
            for task1_round in (2, 3):
                require(client.view["flow"]["stage"] == "task1"
                        and client.view["flow"]["round_index"] == task1_round,
                        "Task 1 round progression differs")
                _finish_round(client, action=True)
                client.command("next")
            require(client.view["flow"]["stage"] == "task2"
                    and client.view["flow"]["round_index"] == 1,
                    "Task 2 did not start after three Task 1 rounds")
            require(client.view.get("answers") == []
                    and client.view.get("explain_allowed") is False,
                    "Task 2 exposed a Task 1 answer")
            forbidden_question(client, answer_id=client.old_answer)
            history(client, client.first_task1_run, expected=403,
                    error="history_current_run_only")
            for task2_round in (1, 2, 3):
                require(client.view["flow"]["stage"] == "task2"
                        and client.view["flow"]["round_index"] == task2_round,
                        "Task 2 round progression differs")
                _finish_round(client, action=True)
                client.command("next")
            require(client.view["flow"]["stage"] == "questionnaire",
                    "six rounds did not enter questionnaire")
            _answer_questionnaire(client)
        participants = _database_final(args.database, checkpoint)
        report = {
            "version": VERSION,
            "status": ("passed_synthetic_harness_validation"
                       if args.allow_synthetic_fixture
                       else "passed_r41_diagnostic_v8_real_https_acceptance"),
            "release_acceptance_eligible": not args.allow_synthetic_fixture,
            "synthetic_fixture": bool(args.allow_synthetic_fixture),
            "exact_same_service_restart_restored": True,
            "lost_response_retry_idempotent": True,
            "four_cell_allocation": sorted(
                [[row["condition"], row["task_order"]] for row in participants]),
            "participants": participants,
            "answer_bindings": answers,
            "after_counts": {"requests": harness.requests,
                             "actions": harness.actions,
                             "questions": harness.questions},
            "total_counts": {
                "requests": checkpoint["before_counts"]["requests"] + harness.requests,
                "actions": checkpoint["before_counts"]["actions"] + harness.actions,
                "questions": checkpoint["before_counts"]["questions"] + harness.questions},
            "checks": [
                "A_XY_A_YX_B_XY_B_YX", "tutorial_complete_partial_skip",
                "task1_live_historical_and_post_round_questions",
                "B_and_task2_question_denial", "task2_old_answer_denial",
                "same_run_frame_binding", "refresh_cookie_restore",
                "cross_session_history_denial", "process_restart_restore",
                "lost_response_idempotency", "six_round_scene_order",
                "questionnaire_draft_restore", "NN_action_authority",
            ],
        }
        save_new(output / "report.json", report)
        print(canonical(report))
        return report
    except BaseException as error:
        save_new(output / "after_failure.json", {
            "version": VERSION, "error": repr(error),
            "requests": harness.requests, "actions": harness.actions,
            "questions": harness.questions, "passed": False})
        raise
    finally:
        harness.close()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execute", action="store_true")
    value.add_argument("--phase", choices=("before", "after"), default="before")
    value.add_argument("--base", required=True)
    value.add_argument("--database", type=Path, required=True)
    value.add_argument("--ca-certificate", type=Path, required=True)
    value.add_argument("--expected-package-sha256", required=True)
    value.add_argument("--expected-manifest-sha256", required=True)
    value.add_argument("--expected-actor-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--allow-synthetic-fixture", action="store_true")
    return value


def validate_args(args):
    parsed = urllib.parse.urlsplit(args.base.rstrip("/"))
    require(parsed.scheme == "https" and parsed.hostname in
            {"127.0.0.1", "localhost", "::1"}
            and parsed.path in ("", "/") and not parsed.query
            and not parsed.fragment and not parsed.username
            and not parsed.password,
            "use a credential-free loopback HTTPS origin")
    for field in ("expected_package_sha256", "expected_manifest_sha256",
                  "expected_actor_sha256"):
        require(HEX.fullmatch(getattr(args, field) or ""),
                f"{field} must be an exact lowercase SHA-256")
    ca = Path(args.ca_certificate).expanduser().absolute()
    require(not ca.is_symlink() and ca.is_file(),
            "CA certificate must be a canonical regular file")
    if args.allow_synthetic_fixture:
        require(parsed.hostname in {"127.0.0.1", "localhost", "::1"},
                "synthetic fixture is loopback-only")
    require(args.execute, "plan-only: add --execute after reviewing arguments")
    return args


def main(argv=None):
    args = validate_args(parser().parse_args(argv))
    return 0 if (run_before(args) if args.phase == "before"
                 else run_after(args)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
