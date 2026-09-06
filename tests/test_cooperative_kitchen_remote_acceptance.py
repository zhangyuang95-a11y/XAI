"""The acceptance runner is tested with a transport fixture; no cloud API calls."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import threading
import uuid

import httpx
import pytest

from scripts.cooperative_kitchen import remote_acceptance as runner


class RemoteFixture:
    def __init__(self, *, fallback=False, drift=False, disconnect=False, persistent=False):
        self.lock = threading.RLock()
        self.fallback, self.drift, self.disconnect, self.persistent = fallback, drift, disconnect, persistent
        self.lost_response = None
        self.identities = {}
        self.operations = {}
        self.rows = {key: [] for key in ("run", "episode", "frame", "event", "question")}
        self.requests = []
        self.status_count = 0
        self.config = {"provider": "deepseek", "model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com",
                       "model_version_pinned": False, "model_identity_policy": {"kind": "rolling_alias", "accepted_returned_models": ["deepseek-v4-flash"]}}
        self.versions = {"actor_sha256": "a" * 64, "program_sha256": "b" * 64, "runtime_sha256": "c" * 64}

    def status(self):
        return {"versions": copy.deepcopy(self.versions), "policy_kind": "neural", "qa_configured": True,
                "qa_configuration": copy.deepcopy(self.config), "storage": "postgresql", "test_mode": False,
                "study_ready": False, "namespace": "pilot"}

    def view(self, run):
        document = run["document"]
        episode = document.get("episode_id")
        frames = [row for row in self.rows["frame"] if row["episode_id"] == episode]
        state = {**frames[-1]["public"], "episode_id": episode} if frames else None
        return {"run": {key: document.get(key) for key in ("id", "participant_id", "episode_id", "mode", "version", "phase")}, "state": state}

    def response(self, data, status=200, headers=None):
        return httpx.Response(status, json=copy.deepcopy(data), headers=headers)

    def __call__(self, request):
        with self.lock:
            path = request.url.path
            body = json.loads(request.content) if request.content else None
            self.requests.append((path, copy.deepcopy(body), dict(request.headers)))
            if path == "/api/status":
                self.status_count += 1
                data = self.status()
                if self.drift and self.status_count > 1:
                    data["versions"]["actor_sha256"] = "d" * 64
                return self.response(data)
            if path.startswith("/api/admin/"):
                if request.headers.get("X-Kitchen-Admin-Key") != "PRIVATE-ADMIN-SECRET":
                    return self.response({}, 401)
                return self.response({"service": self.status()})
            if path == "/api/session":
                identity = "r" + str(len(self.identities))
                run = {"id": identity, "namespace": "development", "document": {"id": identity, "participant_id": "p" + identity,
                    "mode": "freeplay", "phase": "instructions", "versions": copy.deepcopy(self.versions), "version": 0, "language": body["language"]}}
                self.identities[identity] = run
                self.rows["run"].append(run)
                return self.response(self.view(run), headers={"set-cookie": "session=" + identity + "; Path=/; HttpOnly; Secure"})
            identity = request.headers.get("cookie", "").partition("session=")[2].split(";")[0]
            if identity not in self.identities:
                return self.response({}, 401)
            run = self.identities[identity]
            doc = run["document"]
            episode = doc.get("episode_id")
            if path == "/api/view":
                return self.response(self.view(run))
            if path == "/api/command":
                key = identity, body["operation_id"]
                if key in self.operations:
                    if self.operations[key] != body:
                        return self.response({}, 409)
                    return self.response(self.view(run))
                if self.persistent and body["command"] == "action" and identity == "r0":
                    raise httpx.RemoteProtocolError("DO-NOT-LEAK-SECRET URL")
                if body["version"] != doc["version"]:
                    return self.response({}, 409)
                self.operations[key] = body
                doc["version"] += 1
                if body["command"] == "next":
                    episode = "ep-" + identity
                    doc.update(episode_id=episode, phase="freeplay")
                    self.rows["episode"].append({"id": episode, "run_id": identity, "document": {}})
                    state = {"turn": 0, "orders": 0}
                else:
                    before = self.view(run)["state"]
                    state = {"turn": before["turn"] + 1, "orders": 0}
                    self.rows["event"].append({"id": "event-" + body["operation_id"], "run_id": identity, "episode_id": episode,
                        "kind": "joint_step", "operation_id": body["operation_id"], "document": {"after": state}})
                self.rows["frame"].append({"episode_id": episode, "turn": state["turn"], "snapshot": copy.deepcopy(state), "public": copy.deepcopy(state)})
                if self.disconnect and self.lost_response is None and body["command"] == "action":
                    self.lost_response = copy.deepcopy(body)
                    raise httpx.RemoteProtocolError("DO-NOT-LEAK-SECRET")
                return self.response(self.view(run))
            if path == "/api/question":
                if body["version"] != doc["version"]:
                    return self.response({}, 409)
                doc["version"] += 1
                question_id = "q-" + body["operation_id"]
                frame = body["frame"]
                source = runner.snapshot_digest({"turn": frame, "orders": 0})
                kind = "why" if frame == 0 else "counterfactual"
                answer = {"verified": True, "frame": frame, "kind": kind, "diagnostics": {
                    "parser_verified": True, "llm_success": not self.fallback, "configuration": copy.deepcopy(self.config),
                    "actor_sha256": self.versions["actor_sha256"], "source_sha256": source,
                    "calls": [{"stage": stage, "http_status": 200, "returned_model": "deepseek-v4-flash", "usage": {"total_tokens": 50}} for stage in ("parse", "answer")]},
                    "evidence": {"counterfactual": {"source_sha256": source, "assumptions": {"human_actions": ["WAIT"] * 3}, "final_state": {"turn": frame + 3}} if kind == "counterfactual" else None}}
                self.rows["question"].append({"id": question_id, "run_id": identity, "episode_id": episode, "frame": frame,
                    "status": "complete", "document": {"answer": answer, "language": doc["language"]}})
                return self.response({"id": question_id, "status": "pending", "version": doc["version"]})
            if path.startswith("/api/question/"):
                row = next((q for q in self.rows["question"] if q["id"] == path.rsplit("/", 1)[1] and q["run_id"] == identity), None)
                if row is None:
                    return self.response({}, 404)
                return self.response({**row, "answer": {key: val for key, val in row["document"]["answer"].items() if key != "diagnostics"}})
            if path == "/api/exposure":
                key = identity, body["operation_id"]
                if key not in self.operations:
                    self.operations[key] = body
                    self.rows["event"].append({"id": "exposure-" + body["operation_id"], "kind": "answer_exposure", "run_id": identity,
                        "episode_id": episode, "document": {"question_id": body["question_id"], "event": body["event"]}})
                return self.response({"ok": True})
            if path == "/api/history":
                if request.url.params.get("episode_id") != episode:
                    return self.response({}, 404)
                return self.response({"episode_id": episode, "frames": [r["public"] for r in self.rows["frame"] if r["episode_id"] == episode]})
            return self.response({}, 404)

    def audit(self, ids):
        assert set(ids) == set(self.identities)
        return copy.deepcopy(self.rows)


def execute(fixture, **kwargs):
    return runner.run_acceptance("https://fixture.invalid", "PRIVATE-ADMIN-SECRET", transport=httpx.MockTransport(fixture),
                                 audit_reader=fixture.audit, poll_interval=0, **kwargs)


def test_full_twenty_client_workload_and_private_cloud_audit():
    fixture = RemoteFixture()
    report = execute(fixture)
    assert report["passed"], report["errors"]
    assert report["mode"] == "local_fixture"
    assert (report["sessions"], report["actions"], report["questions"]) == (20, 200, 40)
    assert report["duplicate_action_retries"] == report["duplicate_exposure_retries"] == 40
    assert report["audit"]["cloud_questions"] == 40
    assert report["audit"]["frames"] == 220
    assert report["audit"]["token_usage"]["total_tokens"] == 4000
    assert report["cross_session_denials"] == 40
    assert all(p["refresh_verified"] for p in report["participants"])
    assert {p["language"] for p in report["participants"]} == {"zh", "en"}
    assert all("final_state" not in p and "initial_view" not in p for p in report["participants"])
    for path, _, headers in fixture.requests:
        assert (headers.get("x-kitchen-admin-key") == "PRIVATE-ADMIN-SECRET") == path.startswith("/api/admin/")
    assert "PRIVATE-ADMIN-SECRET" not in json.dumps(report)


def test_public_verified_fallback_is_not_counted_as_real_cloud():
    report = execute(RemoteFixture(fallback=True))
    assert not report["passed"]
    assert {e["code"] for e in report["errors"]} == {"cloud_fallback_is_not_success"}


def test_lost_confirmation_retries_same_payload_without_double_step():
    fixture = RemoteFixture(disconnect=True)
    report = execute(fixture)
    assert report["passed"], report["errors"]
    assert report["transport_retry_count"] == 1
    requests = [payload for path, payload, _ in fixture.requests if path == "/api/command" and payload.get("operation_id") == fixture.lost_response["operation_id"]]
    assert len(requests) == 3  # initial lost confirmation, bounded retry, deliberate replay
    assert all(payload == fixture.lost_response for payload in requests)
    assert max(p["action_seconds"][0] for p in report["participants"]) >= .1
    assert "DO-NOT-LEAK-SECRET" not in json.dumps(report)


def test_persistent_disconnect_keeps_created_ids_and_fails():
    report = execute(RemoteFixture(persistent=True))
    assert not report["passed"]
    assert len({p["run_id"] for p in report["participants"]}) == 20
    assert report["transport_retry_count"] == 1
    assert any(e["code"] == "transport_retries_exhausted" for e in report["errors"])
    assert "DO-NOT-LEAK-SECRET" not in json.dumps(report)


def test_release_drift_fails():
    report = execute(RemoteFixture(drift=True))
    assert not report["passed"]
    assert {e["code"] for e in report["errors"]} == {"release_changed_during_test"}


def test_redirect_never_forwards_admin_key():
    requests = []
    def redirect(request):
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://other.invalid/admin"})
    client = runner.Client("https://fixture.invalid", admin_key="secret", transport=httpx.MockTransport(redirect))
    with pytest.raises(runner.AcceptanceError, match="http_307"):
        client.request("/api/admin/status")
    client.close()
    assert len(requests) == 1
    assert requests[0].url.host == "fixture.invalid"


@pytest.mark.parametrize("url", ["http://remote.invalid", "https://user:secret@remote.invalid", "https://remote.invalid?key=secret", "https://remote.invalid/#fragment", "http://127.0.0.1:8003"])
def test_bad_endpoints_fail_before_io(url):
    with pytest.raises(runner.AcceptanceError):
        runner.validate_endpoint(url)


def test_private_env_is_literal_and_owner_only(tmp_path, monkeypatch):
    for key in runner.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "private.env"
    path.write_text("KITCHEN_URL=https://fixture.invalid\nexport KITCHEN_ADMIN_KEY='literal$HOME`noexec`'\nUNKNOWN=ignored\n")
    path.chmod(0o600)
    assert runner.private_environment(path)["KITCHEN_ADMIN_KEY"] == "literal$HOME`noexec`"
    monkeypatch.setenv("KITCHEN_ADMIN_KEY", "override")
    assert runner.private_environment(path)["KITCHEN_ADMIN_KEY"] == "override"
    path.chmod(0o644)
    with pytest.raises(runner.AcceptanceError, match="owner_only"):
        runner.private_environment(path)


def test_explicit_run_required_and_no_output_created(tmp_path):
    path = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        runner.main(["--output", str(path)])
    assert exc.value.code == 2 and not path.exists()


def test_snapshot_hash_matches_authoritative_format():
    from backend.cooperative_kitchen.explanations import snapshot_hash
    snapshot = {"turn": 3, "label": "测试", "actors": {"human": [1, 3]}}
    assert runner.snapshot_digest(snapshot) == snapshot_hash(snapshot)


def test_checkpoint_records_all_created_ids_before_load():
    saved = []
    report = execute(RemoteFixture(), checkpoint=lambda data: saved.append(copy.deepcopy(data)))
    assert report["passed"]
    assert {p["run_id"] for p in saved[-1]["participants"]} == {p["run_id"] for p in report["participants"]}
    assert saved[-1]["passed"] is False


def test_export_filters_other_participants_and_handles_freeplay_namespace():
    rows = [
        {"type": "run", "namespace": "development", "document": {"id": "test-run"}},
        {"type": "run", "namespace": "development", "document": {"id": "unrelated-run"}},
        {"type": "episode", "id": "test-ep", "run_id": "test-run", "document": {}},
        {"type": "episode", "id": "other-ep", "run_id": "unrelated-run", "document": {}},
        {"type": "frame", "episode_id": "test-ep", "turn": 0},
        {"type": "frame", "episode_id": "other-ep", "turn": 0},
    ]
    class Export:
        def request(self, path, **kwargs):
            assert path == "/api/admin/export?format=jsonl" and kwargs["raw"]
            return "\n".join(json.dumps(row) for row in rows), 0, 200
    selected = runner.exported_records(Export(), ["test-run"])
    assert [row["id"] for row in selected["run"]] == ["test-run"]
    assert [row["episode_id"] for row in selected["frame"]] == ["test-ep"]


@pytest.mark.skipif(not os.environ.get("KITCHEN_REMOTE_RUNNER_TEST_DSN"), reason="Explicit isolated local PostgreSQL fixture DSN is required")
def test_real_postgresql_read_only_audit_filters_exact_created_runs():
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url
    from ui import cooperative_kitchen_store as store
    dsn = os.environ["KITCHEN_REMOTE_RUNNER_TEST_DSN"]
    parsed = make_url(dsn)
    # This test never creates a fixture in the remotely deployed database.
    assert parsed.host in {None, "localhost", "127.0.0.1", "::1"}
    schema = "kitchen_remote_runner_fixture_" + uuid.uuid4().hex
    control = sa.create_engine(dsn)
    with control.begin() as db:
        db.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    isolated = parsed.update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    engine = sa.create_engine(isolated)
    try:
        store.metadata.create_all(engine)
        fixture = RemoteFixture()
        report = execute(fixture)
        assert report["passed"]
        with engine.begin() as db:
            for row in fixture.rows["run"]:
                db.execute(store.runs.insert().values(**{**row, "document": json.dumps(row["document"]), "token_hash": row["id"], "version": 0, "created": 0, "updated": 0}))
            db.execute(store.runs.insert().values(id="unrelated-private-run", namespace="pilot", token_hash="unrelated", document='{"id":"private"}', version=0, created=0, updated=0))
            for row in fixture.rows["episode"]:
                db.execute(store.episodes.insert().values(**{**row, "document": json.dumps(row["document"]), "episode_index": 0, "phase": "freeplay"}))
            for row in fixture.rows["frame"]:
                db.execute(store.frames.insert().values(**{**row, "snapshot": json.dumps(row["snapshot"]), "public": json.dumps(row["public"])}))
            for row in fixture.rows["event"]:
                value = {key: value for key, value in row.items() if key != "id"}
                value.update(document=json.dumps(value["document"]), operation_id=value.get("operation_id", uuid.uuid4().hex), created=0)
                db.execute(store.events.insert().values(**value))
            for row in fixture.rows["question"]:
                db.execute(store.questions.insert().values(**{**row, "document": json.dumps(row["document"]), "attempts": 1, "created": 0, "updated": 0}))
        ids = [p["run_id"] for p in report["participants"]]
        rows = runner.database_records(isolated, ids)
        assert {row["id"] for row in rows["run"]} == set(ids)
        assert len(rows["frame"]) == 220 and len(rows["question"]) == 40
        participants = copy.deepcopy(report["participants"])
        for p in participants:
            p["final_state"] = {"turn": 10, "orders": 0}
        audited = runner.audit_records(rows, participants, fixture.status())
        assert audited["runs"] == 20 and audited["cloud_questions"] == 40
        with engine.connect() as db:
            assert db.scalar(sa.select(sa.func.count()).select_from(store.runs)) == 21
    finally:
        engine.dispose()
        with control.begin() as db:
            db.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        control.dispose()
