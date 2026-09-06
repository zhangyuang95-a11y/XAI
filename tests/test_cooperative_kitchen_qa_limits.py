"""Paid-QA admission budgets, using SQLite or an isolated real PostgreSQL schema."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from backend.cooperative_kitchen.study import DEFAULT_QA_LIMITS, KitchenStudy
from ui.cooperative_kitchen_server import create_app, COOKIE
from ui.cooperative_kitchen_store import StudyError, events, operations, questions, runs
from tests.test_cooperative_kitchen_study import (
    FixtureExplainer, FixturePolicy, command, join, release, study,
)


def ready(study):
    token, _, _ = join(study, "freeplay")
    command(study, token, "next")
    return token


def payload(study, token, operation_id, text="Why this action?"):
    view = study.view(token)
    return {"operation_id": operation_id, "version": view["run"]["version"],
            "episode_id": view["run"]["episode_id"], "frame": 0,
            "question": text, "kind": "why"}


def peer(study, tmp_path, **kwargs):
    return KitchenStudy(tmp_path, study.store.engine.url.render_as_string(hide_password=False),
                        namespace="test", allow_sqlite=study.store.is_sqlite,
                        policy=FixturePolicy(), explainer=FixtureExplainer(), release=release(),
                        test_mode=True, enrollment_mode="formal",
                        qa_limits=dict(study.qa_limits), **kwargs)


def reject(study, token, request, code):
    with pytest.raises(StudyError) as caught:
        study.ask(token, request)
    assert (caught.value.status, caught.value.code) == (429, code)
    return str(caught.value)


def test_default_limits_are_fixed_and_status_is_a_copy(tmp_path):
    app = KitchenStudy(tmp_path, "sqlite:///:memory:", allow_sqlite=True)
    try:
        expected = {"per_episode": 8, "per_run": 24, "per_namespace": 500,
                    "min_interval_seconds": 2}
        assert app.status()["qa_limits"] == dict(DEFAULT_QA_LIMITS) == expected
        app.status()["qa_limits"]["per_run"] = 999
        assert app.qa_limits["per_run"] == 24
        with pytest.raises(TypeError):
            app.qa_limits["per_run"] = 999
        with TestClient(create_app(app, start_workers=False)) as client:
            assert client.get("/api/status").json()["qa_limits"] == expected
    finally:
        app.store.engine.dispose()
    for overrides in ({}, expected):
        with pytest.raises(ValueError, match="test_mode"):
            KitchenStudy(tmp_path, "sqlite:///:memory:", allow_sqlite=True, qa_limits=overrides)


@pytest.mark.parametrize("limits", [[], {"per_run": 0}, {"per_episode": True},
    {"per_namespace": 1.5}, {"min_interval_seconds": -1}, {"min_interval_seconds": True},
    {"min_interval_seconds": float("nan")}, {"min_interval_seconds": float("inf")}, {"hourly": 1}])
def test_invalid_test_limit_overrides_fail_before_database_creation(tmp_path, limits):
    with pytest.raises(ValueError):
        KitchenStudy(tmp_path, "not-a-database", namespace="test", test_mode=True, qa_limits=limits)


@pytest.mark.parametrize("study", [{"per_episode": 2, "per_run": 3,
                                  "min_interval_seconds": 0}], indirect=True)
def test_episode_and_run_budgets_include_failed_jobs_and_survive_restart(study, tmp_path):
    token = ready(study)
    for n, status in enumerate(("failed", "cancelled")):
        job = study.ask(token, payload(study, token, f"accepted-{n}"))
        with study.store.transaction() as db:
            db.execute(update(questions).where(questions.c.id == job["id"]).values(status=status))
    before = study.view(token)
    request = payload(study, token, "episode-refused", "PRIVATE-REJECTED-QUESTION")
    error = reject(study, token, request, "question_episode_limit")
    assert study.view(token) == before
    # A committed rejection is visible through a newly constructed service.
    reopened = peer(study, tmp_path)
    try:
        command(reopened, token, "action", action="WAIT")
        assert reject(reopened, token, request, "question_episode_limit") == error
        with pytest.raises(StudyError) as conflict:
            reopened.ask(token, {**request, "question": "changed"})
        assert conflict.value.code == "operation_conflict"
        with reopened.store.transaction() as db:
            rejected = db.execute(select(events).where(events.c.operation_id == "episode-refused")).mappings().all()
            receipt = db.execute(select(operations).where(operations.c.operation_id == "episode-refused")).mappings().one()
            assert len(rejected) == 1 and rejected[0]["kind"] == "question_rejected"
            assert "PRIVATE-REJECTED-QUESTION" not in json.dumps([dict(rejected[0]), dict(receipt)])
            assert set(json.loads(rejected[0]["document"])) == {"code", "scope", "frame", "version", "usage", "limits"}
            saved_receipt = json.loads(receipt["response"])
            assert set(saved_receipt) == {"rejection"}
            assert set(saved_receipt["rejection"]) == {"status", "code", "scope", "message"}
        command(reopened, token, "restart")
        reopened.ask(token, payload(reopened, token, "new-episode-last-credit"))
        reject(reopened, token, payload(reopened, token, "run-refused"), "question_run_limit")
        command(reopened, token, "restart")
        reject(reopened, token, payload(reopened, token, "run-still-refused"), "question_run_limit")
    finally:
        reopened.store.engine.dispose()


@pytest.mark.parametrize("study", [{"min_interval_seconds": 2}], indirect=True)
def test_interval_boundary_and_replay_do_not_consume_new_allowance(study, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("backend.cooperative_kitchen.study.time", SimpleNamespace(time=lambda: clock[0]))
    token = ready(study)
    first_request = payload(study, token, "accepted-once")
    first = study.ask(token, first_request)
    clock[0] = 101.999
    denied = payload(study, token, "too-soon")
    reject(study, token, denied, "question_rate_limit")
    assert study.ask(token, first_request)["id"] == first["id"]
    clock[0] = 102.0
    # A rejected operation never becomes a new paid job after its cooldown expires.
    reject(study, token, denied, "question_rate_limit")
    study.ask(token, payload(study, token, "accepted-at-boundary"))
    with study.store.transaction() as db:
        assert len(db.execute(select(questions.c.id)).all()) == 2
        assert len(db.execute(select(events.c.id).where(events.c.kind == "question_rejected")).all()) == 1


def test_pending_job_rejection_has_durable_idempotent_http_error(study):
    token = ready(study)
    for n in range(2):
        study.ask(token, payload(study, token, f"pending-{n}"))
    request = payload(study, token, "pending-rejection", "do-not-store-this-question")
    with TestClient(create_app(study, start_workers=False)) as client:
        client.cookies.set(COOKIE, token)
        a = client.post("/api/question", json=request)
        b = client.post("/api/question", json=request)
        assert a.status_code == b.status_code == 429
        assert a.json() == b.json()
        assert a.json()["code"] == "question_limit"
    assert "do-not-store-this-question" not in study.store.export()
    with study.store.transaction() as db:
        assert len(db.execute(select(events.c.id).where(events.c.kind == "question_rejected")).all()) == 1


@pytest.mark.parametrize("study", [{"per_namespace": 5, "min_interval_seconds": 0}], indirect=True)
def test_twenty_concurrent_sessions_cannot_exceed_namespace_budget(study, tmp_path):
    """Separate engines prove the limit is a database invariant, not a Python lock."""
    other = peer(study, tmp_path)
    try:
        tokens = [ready(study) for _ in range(20)]
        requests = [payload(study, token, f"namespace-race-{i}") for i, token in enumerate(tokens)]
        barrier = threading.Barrier(20)
        def attempt(i):
            service = study if i % 2 else other
            barrier.wait(timeout=15)
            try:
                return ("accepted", service.ask(tokens[i], requests[i])["id"])
            except StudyError as error:
                return ("rejected", error.status, error.code)
        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(attempt, range(20)))
        assert sum(result[0] == "accepted" for result in results) == 5
        assert all(result == ("rejected", 429, "question_budget_exhausted")
                   for result in results if result[0] == "rejected")
        with study.store.transaction() as db:
            assert len(db.execute(select(questions.c.id)).all()) == 5
            assert len(db.execute(select(events.c.id).where(events.c.kind == "question_rejected")).all()) == 15
            assert len(db.execute(select(operations.c.operation_id).where(operations.c.operation_id.like("namespace-race-%"))).all()) == 20
        for i, result in enumerate(results):
            if result[0] == "accepted":
                assert other.ask(tokens[i], requests[i])["id"] == result[1]
            else:
                reject(other, tokens[i], requests[i], "question_budget_exhausted")
    finally:
        other.store.engine.dispose()


@pytest.mark.parametrize("study", [{"per_namespace": 1, "min_interval_seconds": 0}], indirect=True)
def test_concurrent_duplicate_operations_charge_exactly_once(study, tmp_path):
    token = ready(study)
    request = payload(study, token, "same-operation-twenty-times")
    other = peer(study, tmp_path)
    barrier = threading.Barrier(20)
    try:
        def attempt(i):
            barrier.wait(timeout=15)
            return (study if i % 2 else other).ask(token, request)["id"]
        with ThreadPoolExecutor(max_workers=20) as pool:
            assert len(set(pool.map(attempt, range(20)))) == 1
        with study.store.transaction() as db:
            assert len(db.execute(select(questions.c.id)).all()) == 1
            assert len(db.execute(select(events.c.id).where(events.c.kind == "question_rejected")).all()) == 0
    finally:
        other.store.engine.dispose()


@pytest.mark.parametrize("study", [{"per_namespace": 1, "min_interval_seconds": 0}], indirect=True)
def test_budgets_are_scoped_to_database_namespace(study, tmp_path):
    development = KitchenStudy(tmp_path, study.store.engine.url.render_as_string(hide_password=False),
        namespace="development", allow_sqlite=study.store.is_sqlite,
        policy=FixturePolicy(), explainer=FixtureExplainer())
    try:
        dev_token = ready(development)
        development.ask(dev_token, payload(development, dev_token, "other-namespace"))
        token = ready(study)
        study.ask(token, payload(study, token, "one-test-credit"))
        rejected_token = ready(study)
        reject(study, rejected_token, payload(study, rejected_token, "no-test-credit"), "question_budget_exhausted")
        with study.store.transaction() as db:
            row = db.execute(select(events.c.document).where(events.c.kind == "question_rejected")).scalar_one()
            assert json.loads(row)["usage"]["per_namespace"] == 1
    finally:
        development.store.engine.dispose()
