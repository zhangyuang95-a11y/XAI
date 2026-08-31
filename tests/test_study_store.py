from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from ui.study_store import StudyConflict, StudyStore


def _assignment(participant: str = "P1") -> dict[str, object]:
    return {
        "participant_id": participant,
        "enrollment_index": 0,
        "block_index": 0,
        "condition": "no_trace",
        "controlled_agent": "robot_1",
        "target_agent": "robot_2",
        "form_id": 0,
        "demo_seed": 42026,
        "task1_seed": 41000,
        "task2_seed": 51000,
        "study_phase": "test",
        "randomization_seed": 41000,
    }


def _create(store: StudyStore, *, operation: str = "start-op") -> dict:
    return store.create_run(
        operation_id=operation,
        run_id="run-1",
        browser_session_id="browser-a",
        participant_id="P1",
        participant_key="p1",
        assignment=_assignment(),
        locale="en",
        mutate=lambda attempt: (
            {"view": {"study": {"stage": "instructions"}}},
            [{"event": "study_started", "assignment": _assignment()}],
            {"stage": "instructions", "locale": "en"},
        ),
    )


def test_store_is_idempotent_and_rejects_stale_version(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    started = _create(store)
    assert started["state_version"] == 1
    calls = 0

    def mutate():
        nonlocal calls
        calls += 1
        return (
            {"view": {"study": {"stage": "instructions"}}},
            [{"event": "tutorial_advanced"}],
            {"stage": "instructions", "locale": "en", "tutorial_index": 1},
        )

    first = store.execute(
        operation_id="advance-1", run_id="run-1", browser_session_id="browser-a",
        expected_stage="instructions", expected_version=1,
        command="tutorial_advance", mutate=mutate,
    )
    replay = store.execute(
        operation_id="advance-1", run_id="run-1", browser_session_id="browser-a",
        expected_stage="instructions", expected_version=1,
        command="tutorial_advance", mutate=mutate,
    )
    assert first == replay
    assert calls == 1
    with pytest.raises(StudyConflict, match="stale"):
        store.execute(
            operation_id="advance-stale", run_id="run-1", browser_session_id="browser-a",
            expected_stage="instructions", expected_version=1,
            command="tutorial_advance", mutate=mutate,
        )
    assert calls == 1


def test_completed_operation_replay_survives_later_stage_progress(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    first = store.execute(
        operation_id="first", run_id="run-1", browser_session_id="browser-a",
        expected_stage="instructions", expected_version=1, command="advance",
        mutate=lambda: (
            {"view": {"study": {"stage": "task1"}}}, [],
            {"stage": "task1", "locale": "en"},
        ),
    )
    store.execute(
        operation_id="second", run_id="run-1", browser_session_id="browser-a",
        expected_stage="task1", expected_version=2, command="collaborative_step",
        mutate=lambda: (
            {"view": {"study": {"stage": "task1_complete"}}}, [],
            {"stage": "task1_complete", "locale": "en"},
        ),
    )
    replay = store.execute(
        operation_id="first", run_id="run-1", browser_session_id="browser-a",
        expected_stage="instructions", expected_version=1, command="advance",
        mutate=lambda: pytest.fail("a replay must not mutate"),
    )
    assert replay == first


def test_begin_task2_operation_is_idempotent_after_stage_changes(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    store.execute(
        operation_id="task1-finished", run_id="run-1",
        browser_session_id="browser-a", expected_stage="instructions",
        expected_version=1, command="finish_task1",
        mutate=lambda: (
            {"view": {"study": {"stage": "task1_complete"}}},
            [{"event": "task1_completion_presented"}],
            {"stage": "task1_complete", "locale": "en"},
        ),
    )
    calls = 0

    def begin_task2():
        nonlocal calls
        calls += 1
        return (
            {"view": {"study": {"stage": "task2"}}, "fresh_frame": 0},
            [{"event": "task1_completion_acknowledged"}],
            {"stage": "task2", "locale": "en"},
        )

    first = store.execute(
        operation_id="begin-task2-1", run_id="run-1",
        browser_session_id="browser-a", expected_stage="task1_complete",
        expected_version=2, command="begin_task2", mutate=begin_task2,
    )
    replay = store.execute(
        operation_id="begin-task2-1", run_id="run-1",
        browser_session_id="browser-a", expected_stage="task1_complete",
        expected_version=2, command="begin_task2", mutate=begin_task2,
    )

    assert first == replay
    assert first["fresh_frame"] == 0
    assert calls == 1
    assert store.run_row("run-1")["stage"] == "task2"


def test_transaction_rolls_back_all_records(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)

    def fail():
        raise OSError("simulated material failure")

    with pytest.raises(OSError):
        store.execute(
            operation_id="broken", run_id="run-1", browser_session_id="browser-a",
            expected_stage="instructions", expected_version=1,
            command="tutorial_advance", mutate=fail,
        )
    assert store.run_row("run-1")["state_version"] == 1
    assert store.cached_operation("broken") is None


def test_slow_operation_reservation_does_not_hold_the_write_lock(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    assert store.reserve_long_operation(
        operation_id="explain-1", run_id="run-1",
        browser_session_id="browser-a", expected_stage="instructions",
        expected_version=1, command="ask_explanation",
    ) is None

    # A separate short transaction remains writable while model work happens.
    with store.transaction() as db:
        db.execute(
            "INSERT INTO metadata(key, value) VALUES(?, ?)",
            ("concurrent-write", "ok"),
        )
    with pytest.raises(StudyConflict) as duplicate:
        store.reserve_long_operation(
            operation_id="explain-1", run_id="run-1",
            browser_session_id="browser-a", expected_stage="instructions",
            expected_version=1, command="ask_explanation",
        )
    assert duplicate.value.code == "operation_in_progress"

    mutation = (
        {"view": {"study": {"stage": "instructions"}}},
        [{"event": "explanation_presented", "exposure_index": 1}],
        {"stage": "instructions", "locale": "en"},
    )
    completed = store.complete_long_operation(
        operation_id="explain-1", run_id="run-1",
        browser_session_id="browser-a", expected_stage="instructions",
        expected_version=1, command="ask_explanation", mutation=mutation,
    )
    assert completed["state_version"] == 2
    assert store.cached_operation("explain-1") == completed


def test_failed_slow_operation_can_be_cancelled_without_state_change(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    store.reserve_long_operation(
        operation_id="explain-fail", run_id="run-1",
        browser_session_id="browser-a", expected_stage="instructions",
        expected_version=1, command="ask_explanation",
    )
    store.cancel_long_operation("explain-fail")
    assert store.run_row("run-1")["state_version"] == 1
    assert store.cached_operation("explain-fail") is None
    assert store.reserve_long_operation(
        operation_id="explain-fail", run_id="run-1",
        browser_session_id="browser-a", expected_stage="instructions",
        expected_version=1, command="ask_explanation",
    ) is None


def test_restart_abandons_active_run_and_keeps_assignment(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    with pytest.raises(StudyConflict) as error:
        store.create_run(
            operation_id="start-2", run_id="run-2", browser_session_id="browser-b",
            participant_id="P1", participant_key="p1", assignment=_assignment(), locale="en",
            mutate=lambda attempt: ({}, [], {"stage": "instructions", "locale": "en"}),
        )
    assert error.value.code == "participant_active_elsewhere"
    restarted = store.create_run(
        operation_id="restart-2", run_id="run-2", browser_session_id="browser-a",
        participant_id="P1", participant_key="p1", assignment=_assignment(), locale="zh-CN",
        force_restart=True,
        mutate=lambda attempt: (
            {"view": {"study": {"stage": "instructions"}}}, [],
            {"stage": "instructions", "locale": "zh-CN"},
        ),
    )
    assert restarted["state_version"] == 1
    assert store.run_row("run-1")["stage"] == "abandoned"
    assert json.loads(store.run_row("run-2")["assignment_json"]) == _assignment()
    assert len(store.run_assignments()) == 2


def test_current_log_export_is_non_destructive(tmp_path: Path) -> None:
    store = StudyStore(tmp_path / "study.sqlite3", namespace="test")
    _create(store)
    output = tmp_path / "export.jsonl"
    assert store.export_jsonl(output) == 1
    assert store.run_row("run-1")["stage"] == "instructions"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_only_server_startup_abandons_interrupted_runs(tmp_path: Path) -> None:
    path = tmp_path / "study.sqlite3"
    store = StudyStore(path, namespace="test")
    _create(store)
    StudyStore(path, namespace="test")
    assert store.run_row("run-1")["stage"] == "instructions"
    StudyStore(path, namespace="test", abandon_on_start=True)
    assert store.run_row("run-1")["stage"] == "abandoned"
