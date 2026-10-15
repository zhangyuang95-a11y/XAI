#!/usr/bin/env python3
"""End-to-end local admission check for the r4.2 internal-pilot package."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import time
from uuid import uuid4
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.warehouse_alignment_r42_release import load_online_release
from ui.warehouse_alignment_r42_server import CommandError, OnlineAlignmentStudyStore


DEFAULT_RELEASE = ROOT / "output/warehouse_native/r42_delivery_release_20260915"


def _command(store, sid, kind, **values):
    view = store.view(sid)
    return store.command(sid, {
        "operation_id": uuid4().hex,
        "expected_version": view["version"],
        "kind": kind,
        **values,
    })


def _fresh(view):
    assert view["round_ready"] is True
    assert view["allowed_kinds"] == ["begin_round"]
    assert view["state"]["frame"] == 0
    assert all(view["metrics"][key] == 0 for key in (
        "deliveries", "score", "steps", "collisions", "shutdowns"))
    assert len(view["state"]["tasks"]) == 2
    assert all(agent["battery"] == 100
               and agent["carrying_task_id"] is None
               and agent["deliveries_completed"] == 0
               for agent in view["state"]["agents"])


def _wait_for_answer(store, sid, *, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        view = store.view(sid)
        if (view["answers"] and
                view["answers"][0]["status"] in ("complete", "failed")):
            return view
        time.sleep(.05)
    raise AssertionError("timed out waiting for bound explanation")


def _run_condition(store, sid, condition, view=None):
    if view is None:
        view = _command(store, sid, "start", mode="study",
                        participant_id=f"r42_check_{condition.lower()}_{uuid4().hex[:6]}",
                        consent=True)
    assert view["flow"]["stage"] == "instructions"
    assert view["tutorial"]["total_frames"] == 121
    assert view["tutorial"]["duration_ms"] == 380
    view = _command(store, sid, "begin_task1")
    _fresh(view)
    view = _command(store, sid, "begin_round")
    assert "action" in view["allowed_kinds"]

    if condition == "A":
        view = _command(
            store, sid, "question", question="机器人2现在需要充电吗？",
            focus="next", language="zh", run_id=view["run_id"], frame=0,
            intent_id="charging_need")
        view = _wait_for_answer(store, sid)
        answer = view["answers"][0]
        assert answer["status"] == "complete"
        assert answer["text"].startswith((
            "现在需要充电。", "暂时不需要充电。",
            "现有证据无法可靠判断是否需要充电。"))
        first_id = answer["id"]
        view = _command(
            store, sid, "question", question="机器人2当前在朝哪个任务前进？",
            focus="next", language="zh", run_id=view["run_id"], frame=0,
            intent_id="task_direction")
        assert len(view["answers"]) == 1
        assert view["answers"][0]["id"] != first_id
    else:
        try:
            _command(store, sid, "question", question="Why?", focus="next",
                     language="en", run_id=view["run_id"], frame=0,
                     intent_id="action_reason")
        except CommandError as error:
            assert error.reason == "explanation_forbidden"
        else:
            raise AssertionError("condition B received an explanation")

    for index in range(6):
        view = store.view(sid)
        if not view["ended"]:
            view = _command(store, sid, "end")
        if index == 5:
            view = _command(store, sid, "next")
            break
        view = _command(store, sid, "next")
        _fresh(view)
        if view["flow"]["stage"] == "task2":
            assert view["answers"] == [] and view["explain_allowed"] is False
            try:
                _command(store, sid, "question", question="为什么？",
                         focus="executed", language="zh",
                         run_id=view["run_id"], frame=0,
                         intent_id="action_reason")
            except CommandError as error:
                assert error.reason == "explanation_forbidden"
            else:
                raise AssertionError("Task 2 exposed an explanation")
        view = _command(store, sid, "begin_round")
    assert view["flow"]["stage"] == "questionnaire"
    assert view["answers"] == [] and view["explain_allowed"] is False

    with store.connect() as database:
        scene_ids = [row[0] for row in database.execute(
            "SELECT scenario_id FROM runs WHERE session_id=? ORDER BY created",
            (sid,))]
        stored_questions = database.execute(
            "SELECT count(*) FROM questions WHERE session_id=?", (sid,)).fetchone()[0]
    assert scene_ids == [scene["id"]
                         for scene in store.scenarios["splits"]["play"][1:7]]
    assert stored_questions == (2 if condition == "A" else 0)
    return {"condition": condition, "scene_ids": scene_ids,
            "stored_questions": stored_questions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = json.loads((args.release_dir / "release_receipt.json").read_text())
    package = args.release_dir / "warehouse_r42_delivery_release.zip"
    encoded = args.release_dir / "warehouse_r42_delivery_release.b64"
    assert sha256(package.read_bytes()).hexdigest() == receipt["package_sha256"]
    assert sha256(encoded.read_bytes()).hexdigest() == receipt["base64_sha256"]
    assert encoded.stat().st_size < 1_000_000

    context = load_online_release(
        expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=receipt["manifest_sha256"],
        package_path=package)
    database_path = Path(tempfile.mkstemp(prefix="warehouse-r42-", suffix=".sqlite3")[1])
    database_path.unlink()
    store = OnlineAlignmentStudyStore(
        context, database=database_path, storage_mode="ephemeral")
    try:
        identity = store.deployment_identity()
        assert identity["release_version"] == "r4.2-internal-pilot"
        assert identity["actor_sha256"] == receipt["actor_sha256"]
        behavior = context.evidence["behavior"]
        assert behavior["passed"] is True
        assert behavior["actor_sha256"] == receipt["actor_sha256"]
        assert behavior["gates"]["skilled_each_robot_2_deliveries_at_least_2"] is True
        assert behavior["gates"]["no_action_override"] is True
        tutorial = context.evidence["tutorial"]
        assert tutorial["passed"] is True and tutorial["frame_count"] == 121
        assert min(tutorial["individual_deliveries"]) >= 1
        assert max(tutorial["maximum_consecutive_waits"].values()) <= 16
        javascript = store.web_assets["app.js"].decode()
        html = store.web_assets["index.html"].decode()
        assert 'FRONTEND_VERSION="warehouse-alignment-online.r4.2"' in javascript
        assert "QUICK_INTENTS" in javascript and 'kind:"begin_round"' in javascript
        assert "roundPreviewPanel" in html and html.count("data-question=") == 6

        sessions = [store.session(), store.session()]
        enrolled = []
        for sid in sessions:
            view = _command(store, sid, "start", mode="study",
                            participant_id=f"r42_probe_{uuid4().hex[:8]}", consent=True)
            with store.connect() as database:
                condition = database.execute(
                    "SELECT condition FROM sessions WHERE id=?", (sid,)).fetchone()[0]
            enrolled.append((sid, condition, view))
        assert {condition for _, condition, _ in enrolled} == {"A", "B"}
        traces = [_run_condition(store, sid, condition, view)
                  for sid, condition, view in enrolled]

        report = {
            "status": "passed",
            "release_version": "r4.2-internal-pilot",
            "package": receipt,
            "deployment_identity": identity,
            "tutorial": tutorial,
            "ab_traces": traces,
            "behavior": behavior,
        }
    finally:
        store.close()
        database_path.unlink(missing_ok=True)
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
