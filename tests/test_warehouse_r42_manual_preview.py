"""Protocol fixtures only: manually selected internal previews stay distinct."""
from copy import deepcopy
from contextlib import closing
import json
from uuid import uuid4

import pytest

from tests.test_warehouse_alignment_online_server import FakeContext, FakeEnv
from ui import warehouse_alignment_r42_server as online


def context_for_protocol(root):
    context = FakeContext(root)
    context.provenance["version"] = online.R42_RELEASE_CONTEXT_VERSION
    context.release.update(release_version="r4.2-internal-pilot",
        pilot_class="internal_pilot", formal_sample_eligible=False,
        human_explanation_effect_validated=False,
        behavior_performance_gate_passed=True,
        behavior_performance_gate_waived=False, data_persistent=False)
    def environment(scene):
        env = FakeEnv()
        task = deepcopy(env.state["tasks"][0])
        task.update(task_id="task_2", pickup_position=[1,4], delivery_position=[4,4])
        env.state["tasks"].append(task)
        return env
    context.runtime.environment = environment
    template = deepcopy(context.tutorial["frames"][0])
    frames = []
    for index in range(121):
        frame = deepcopy(template)
        frame["state"]["frame"] = index
        frame["metrics"].update(steps=index, individual_deliveries=[1,1])
        frame["actions"] = {} if index == 0 else {"robot_1":"WAIT","robot_2":"WAIT"}
        frames.append(frame)
    context.tutorial.update(version="warehouse-alignment-r42-neutral-tutorial.v2",
        bindings={key:str(index)*64 for index,key in enumerate((
            "parent_scene_fingerprint", "parent_snapshot_sha256",
            "producer_sources_sha256"),1)}, frames=frames)
    context.tutorial_signature = online._digest(context.tutorial)
    context.provenance["tutorial_signature"] = context.tutorial_signature
    return context


@pytest.fixture
def store(tmp_path):
    value = online.OnlineAlignmentStudyStore(context_for_protocol(tmp_path),
        database=tmp_path / "preview.sqlite3", storage_mode="ephemeral")
    yield value
    value.close()


def command(store, sid, kind, **kwargs):
    return store.command(sid, {"operation_id":uuid4().hex,
        "expected_version":store.view(sid)["version"], "kind":kind, **kwargs})


def enroll(store, choice="auto"):
    sid = store.session()
    view = command(store, sid, "start", mode="study", consent=True,
        participant_id="test_"+uuid4().hex[:8], group_choice=choice)
    return sid, view


@pytest.mark.parametrize("condition", ["A", "B"])
def test_manual_choice_is_persisted_and_cannot_switch(store, condition):
    sid, view = enroll(store, condition)
    assert view["flow"]["assignment_source"] == "manual_preview"
    assert view["flow"]["preview_condition"] == condition
    assert store.view(sid)["flow"]["preview_condition"] == condition
    with closing(store.connect()) as db:
        record = db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        assert record["condition"] == condition
        assert record["position"] is None
        assert record["task_order"] == "XY"
    with pytest.raises(online.CommandError, match="study_cannot_restart_or_switch_mode"):
        command(store,sid,"start",mode="study",consent=True,
            participant_id="changed_id",group_choice="B" if condition=="A" else "A")
    preview = command(store,sid,"begin_task1")
    assert preview["explain_allowed"] is False
    with closing(store.connect()) as db:
        provenance=json.loads(db.execute("SELECT provenance FROM runs WHERE id=?",(preview["run_id"],)).fetchone()[0])
        event=json.loads(db.execute("SELECT payload FROM events WHERE session_id=? AND kind='start'",(sid,)).fetchone()[0])
        assert provenance["assignment_source"] == event["assignment_source"] == "manual_preview"
        assert provenance["condition"] == event["assigned_condition"] == condition
    view = command(store,sid,"begin_round")
    assert view["explain_allowed"] is (condition=="A")
    for index in range(3):
        command(store,sid,"end")
        view = command(store,sid,"next")
        if index<2:
            view = command(store,sid,"begin_round")
    assert view["flow"]["stage"] == "task2"
    view = command(store,sid,"begin_round")
    assert view["explain_allowed"] is False
    assert view["answers"] == []
    with pytest.raises(online.CommandError, match="explanation_forbidden"):
        command(store,sid,"question",question="Why?",run_id=view["run_id"],
            frame=0,focus="next",language="en")


def test_manual_previews_do_not_consume_balanced_allocation_positions(store):
    ids=[]
    for choice in ["auto","A","A","B","auto","B","auto","auto"]:
        sid,view=enroll(store,choice)
        if choice=="auto":
            ids.append(sid)
            assert "preview_condition" not in view["flow"]
            assert view["flow"]["assignment_source"] == "random_block"
    with closing(store.connect()) as db:
        rows=[dict(db.execute("SELECT * FROM sessions WHERE id=?",(sid,)).fetchone()) for sid in ids]
        assert [r["position"] for r in rows] == [0,1,2,3]
        assert {r["condition"] for r in rows[:2]} == {"A","B"}
        assert {r["condition"] for r in rows[2:]} == {"A","B"}


@pytest.mark.parametrize("choice", ["C", "a", "", 1, None, ["A"]])
def test_invalid_group_rejected_without_registration(store, choice):
    sid=store.session()
    with pytest.raises(online.CommandError, match="invalid_group_choice"):
        command(store,sid,"start",mode="study",consent=True,
            participant_id="test_invalid",group_choice=choice)
    assert store.view(sid)["flow"]["stage"] == "registration"


def test_r42_assets_wire_explicit_preview_choice_only():
    assets=online._assets(r42=True)
    assert b'id="groupChoice"' in assets["index.html"]
    assert b'group_choice:$("groupChoice").value' in assets["app.js"]
    assert b'locked || !registrationStage' in assets["app.js"]
    assert b'id="groupChoice"' not in online._assets(r42=False)["index.html"]


def test_retried_registration_and_restart_keep_manual_assignment(tmp_path):
    path=tmp_path / "restart.sqlite3"
    first=online.OnlineAlignmentStudyStore(context_for_protocol(tmp_path),
        database=path,storage_mode="ephemeral")
    sid=first.session()
    payload={"operation_id":uuid4().hex,"expected_version":first.view(sid)["version"],
        "kind":"start","mode":"study","consent":True,
        "participant_id":"manual_restart","group_choice":"A"}
    view=first.command(sid,payload)
    retried=first.command(sid,payload)
    assert retried["version"] == view["version"]
    assert retried["flow"]["preview_condition"] == "A"
    first.close()
    second=online.OnlineAlignmentStudyStore(context_for_protocol(tmp_path),
        database=path,storage_mode="ephemeral")
    try:
        restored=second.view(sid)
        assert restored["flow"]["assignment_source"] == "manual_preview"
        assert restored["flow"]["preview_condition"] == "A"
        assert restored["version"] == view["version"]
        with closing(second.connect()) as db:
            assert db.execute("SELECT count(*) FROM sessions WHERE participant_id='manual_restart'").fetchone()[0] == 1
            assert db.execute("SELECT count(*) FROM blocks").fetchone()[0] == 0
    finally:
        second.close()


def test_existing_database_gains_assignment_source_without_changing_group(tmp_path):
    path=tmp_path / "migration.sqlite3"
    first=online.OnlineAlignmentStudyStore(context_for_protocol(tmp_path),
        database=path,storage_mode="ephemeral")
    sid,_=enroll(first)
    with closing(first.connect()) as db:
        original=dict(db.execute("SELECT * FROM sessions WHERE id=?",(sid,)).fetchone())
        db.execute("ALTER TABLE sessions DROP COLUMN assignment_source")
    first.close()
    second=online.OnlineAlignmentStudyStore(context_for_protocol(tmp_path),
        database=path,storage_mode="ephemeral")
    try:
        with closing(second.connect()) as db:
            restored=dict(db.execute("SELECT * FROM sessions WHERE id=?",(sid,)).fetchone())
        assert restored["assignment_source"] == "random_block"
        for key in ("condition","task_order","position","participant_id"):
            assert restored[key] == original[key]
    finally:
        second.close()
