"""Synthetic release-wiring checks; no protected evaluation artifact is read."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from contextlib import closing
from uuid import uuid4

import pytest

from tests.test_warehouse_r41_diagnostic_online_server import diagnostic_context
from ui import warehouse_alignment_online_server as online
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release


class GateAwareSyntheticExplainer:
    signature = "v9-gate-aware-synthetic-explainer"

    def __init__(self):
        self.calls: list[dict] = []

    def _assert_current(self, runtime):
        assert runtime.signature == "runtime-v1"

    def answer_study(self, request, record, runtime, *, access_context):
        assert request["frame"] == record["after"]["state"]["frame"]
        assert runtime.signature == record["runtime_signature"]
        self.calls.append(deepcopy(access_context))
        return {
            "answer": ("机器人2的动作已绑定到你选择的这一步。"
                       if request["language"] == "zh" else
                       "Robot 2's action is bound to the selected step."),
            "evidence_detail": ("所选帧、提交动作和物理结果已核验。"
                                if request["language"] == "zh" else
                                "The selected frame, submitted action, and physical result were verified."),
        }


def v9_context(root: Path):
    context = diagnostic_context(root)
    context.provenance["version"] = release.VERSION
    context.explainer = GateAwareSyntheticExplainer()
    context.release["runtime_action_override"] = False
    return context


def command(store, sid, view, kind, **values):
    return store.command(sid, {
        "operation_id": uuid4().hex,
        "expected_version": view["version"],
        "kind": kind,
        **values,
    })


def wait_for_answers(store, sid, count):
    for _ in range(100):
        view = store.view(sid)
        if (len(view["answers"]) == count
                and all(row["status"] not in ("pending", "running")
                        for row in view["answers"])):
            return view
        time.sleep(0.01)
    raise AssertionError("synthetic v9 answer did not finish")


def test_v9_release_loader_is_registered_and_unadmitted_input_fails_closed(tmp_path):
    assert release.VERSION == online.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSION_V9
    assert release.VERSION in online.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSIONS
    assert online.R41_DIAGNOSTIC_RELEASE_MODULE_V9 in online.RELEASE_MODULES
    projection = release.release_projection()
    assert projection["release_version"] == "r4.1-diagnostic"
    assert projection["pilot_class"] == "internal_diagnostic"
    assert projection["formal_ready"] is False
    assert projection["formal_sample_eligible"] is False
    assert projection["model_ready"] is False
    assert projection["explanation_ready"] is False
    assert projection["study_ready"] is False
    assert projection["runtime_action_override"] is False
    assert projection["animation_duration_ms"] == 380
    sources = release.release_sources()
    assert {
        "ui/warehouse_alignment_r41_diagnostic_release_v9.py",
        "ui/warehouse_alignment_online_server.py",
        "backend/warehouse_r41_diagnostic_online_explanation_v9.py",
        "ui/warehouse_family_feedback_research/index.html",
        "ui/warehouse_family_feedback_research/app.js",
        "ui/warehouse_family_feedback_research/styles.css",
        "ui/warehouse_family_feedback_research/favicon.svg",
    }.issubset(sources)
    with pytest.raises((ValueError, FileNotFoundError)):
        online.load_online_context(
            expected_package_sha256="1" * 64,
            expected_manifest_sha256="2" * 64,
            package_path=tmp_path / "not-opened.zip",
            release_module=online.R41_DIAGNOSTIC_RELEASE_MODULE_V9,
        )


def test_v9_server_preserves_flow_animation_and_a_task1_bound_qa(tmp_path):
    context = v9_context(tmp_path)
    explainer = context.explainer
    database = tmp_path / "v9-study.sqlite3"
    store = online.OnlineAlignmentStudyStore(
        context, database=database, storage_mode="ephemeral")
    try:
        enrolled = []
        for index in range(4):
            sid = store.session()
            view = store.view(sid)
            body = {
                "operation_id": f"enroll-{index}",
                "expected_version": view["version"],
                "kind": "start",
                "mode": "study",
                "participant_id": f"v9_user_{index}",
                "consent": True,
            }
            first = store.command(sid, body)
            repeated = store.command(sid, body)
            assert repeated["version"] == first["version"]
            assert first["flow"]["stage"] == "instructions"
            assert first["tutorial"]["duration_ms"] == 380
            enrolled.append((sid, first))

        with closing(store.connect()) as db:
            by_condition = {
                db.execute("SELECT condition FROM sessions WHERE id=?",
                           (sid,)).fetchone()[0]: (sid, view)
                for sid, view in enrolled
            }
        sid_a, view_a = by_condition["A"]
        sid_b, view_b = by_condition["B"]
        view_a = command(store, sid_a, view_a, "begin_task1")
        view_b = command(store, sid_b, view_b, "begin_task1")
        assert view_a["release"]["release_version"] == "r4.1-diagnostic"
        assert view_a["release"]["pilot_class"] == "internal_diagnostic"
        assert view_a["release"]["formal_ready"] is False
        assert view_a["enrollment"]["mode"] == "internal_diagnostic"

        view_a = command(
            store, sid_a, view_a, "question", run_id=view_a["run_id"],
            frame=0, question="机器人2刚才为什么这样行动？",
            language="zh", focus="next")
        view_a = wait_for_answers(store, sid_a, 1)
        assert view_a["answers"][0]["evidence_detail"]
        assert explainer.calls[-1]["surface"] == "live"

        view_a = command(store, sid_a, view_a, "action", action="RIGHT")
        view_a = command(
            store, sid_a, view_a, "question", run_id=view_a["run_id"],
            frame=0, question="请解释第0帧", language="zh", focus="next")
        view_a = wait_for_answers(store, sid_a, 2)
        assert explainer.calls[-1]["surface"] == "history"

        view_a = command(store, sid_a, view_a, "end")
        view_a = command(
            store, sid_a, view_a, "question", run_id=view_a["run_id"],
            frame=1, question="机器人2刚才为什么这样行动？",
            language="zh", focus="executed")
        view_a = wait_for_answers(store, sid_a, 3)
        assert explainer.calls[-1]["surface"] == "round_review"
        assert all(call["condition"] == "A" and call["stage"] == "task1"
                   and call["active_run_id"] == call["bound_run_id"]
                   for call in explainer.calls)

        with pytest.raises(online.CommandError, match="explanation_forbidden"):
            command(
                store, sid_b, view_b, "question", run_id=view_b["run_id"],
                frame=0, question="why", language="en", focus="next")

        # Finish Task 1: the ended first round advances directly; the next two
        # rounds are explicitly ended. Task 2 must hide every old answer.
        view_a = command(store, sid_a, view_a, "next")
        for _ in range(2):
            view_a = command(store, sid_a, view_a, "end")
            view_a = command(store, sid_a, view_a, "next")
        assert view_a["flow"]["stage"] == "task2"
        assert view_a["answers"] == []
        assert view_a["explain_allowed"] is False
        assert "question" not in view_a["allowed_kinds"]
        with pytest.raises(online.CommandError, match="explanation_forbidden"):
            command(
                store, sid_a, view_a, "question", run_id=view_a["run_id"],
                frame=0, question="why", language="en", focus="next")

        for _ in range(3):
            view_a = command(store, sid_a, view_a, "end")
            view_a = command(store, sid_a, view_a, "next")
        assert view_a["flow"]["stage"] == "questionnaire"
        assert len(view_a["questionnaire"]["items"]) == 11

        javascript = store.web_assets["app.js"].decode("utf-8")
        html = store.web_assets["index.html"].decode("utf-8")
        assert "const MOTION_DURATION_MS=380;" in javascript
        assert html.count("data-question=") == 6
    finally:
        store.close()

    # A browser refresh or process restart reuses the confirmed SQLite state.
    restored_context = v9_context(tmp_path)
    restored = online.OnlineAlignmentStudyStore(
        restored_context, database=database, storage_mode="ephemeral")
    try:
        view = restored.view(sid_a)
        assert view["flow"]["stage"] == "questionnaire"
        assert view["answers"] == []
    finally:
        restored.close()
