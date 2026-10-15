"""Research-flow and isolation checks using test identities and simulated humans.

The explainer here is deliberately a test double: these tests make no claim about
semantic question answering, participant usability, or a human treatment effect.
"""
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from study_v3.config import Settings
from study_v3.registry import engine, scenario_config
from study_v3.store import Store, StudyError, encode


class RecordingExplainer:
    def __init__(self):
        self.calls = []

    def answer(self, eng, state, decision, question, language, previous_dialogue, public_history, *, action_context=None):
        self.calls.append(json.loads(encode({"domain": eng.DOMAIN, "state": state,
            "decision": decision, "question": question, "language": language,
            "previous": previous_dialogue, "history": public_history, "action_context": action_context})))
        return {"status": "answered", "answer": "A test-only verified answer.",
            "evidence_ids": ["test-evidence"], "language": language,
            "audit": {"secret_trace": "must-stay-server-side"},
            "raw_answer": "must-stay-server-side", "policy_memory": {"private": True}}


class BlockingExplainer(RecordingExplainer):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def answer(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(20), "test failed to release the explanation worker"
        return super().answer(*args, **kwargs)


class Flow:
    def __init__(self, store, domain="pong", group="A", name=None):
        self.store = store
        self.domain = domain
        self.group = group
        self.name = name or "test-" + uuid.uuid4().hex
        self.token, self.view = store.create({"participant_id": self.name,
            "domain": domain, "mode": "test", "group": group, "consent": True}, admin=True)
        self.recovery = self.view["recovery_code"]

    def command(self, command_name, **fields):
        payload = {"instance_id": self.view["instance_id"],
            "revision": self.view["revision"], "command_id": uuid.uuid4().hex, **fields}
        self.view = self.store.command(self.token, command_name, payload)
        return payload

    def internal_state(self):
        with self.store.db.transaction() as db:
            run = db.one("SELECT * FROM pl3_runs WHERE id=?", (self.view["run_id"],))
        return json.loads(run["state_json"])

    def step(self, action=None):
        state = self.internal_state()
        if action is None:
            action = engine(self.domain).human_advisor(state)
        return self.command("action", run_id=self.view["run_id"], turn=state["turn"], action=action)

    def finish_demo(self):
        if self.domain=='kitchen':
            self.command('demo_skip')
            return
        for _ in range(len(self.view["demo"]["captions"])):
            self.command("demo_next")
        self.command("next")

    def finish_task(self):
        budget = self.view["state"]["max_turns"]
        for _ in range(budget + 1):
            if self.view["state"]["terminal"]:
                return
            self.step()
        pytest.fail("task did not end within its fixed turn budget")

    def start_task2(self):
        self.finish_demo()
        self.finish_task()
        self.command("next")

    def question(self, **fields):
        return {"instance_id": self.view["instance_id"], "authorized_run": self.view["run_id"],
            "target_run": self.view["run_id"], "turn": self.view["state"]["turn"] if self.view["state"] else 0,
            "question_id": uuid.uuid4().hex, "question": "Why is my teammate doing this?",
            "language": "en", **fields}

    def submit_survey(self):
        items = self.view["questionnaire"]
        return self.command("questionnaire",
            answers={q["id"]: "na" if q["allow_na"] else 4 for q in items["items"]},
            comprehension={q["id"]: 0 for q in items["comprehension"]},
            feedback="Automated flow test; this is not a participant response.")


@pytest.fixture
def store(tmp_path):
    return Store(Settings(database=str(tmp_path / "flow.sqlite3")), RecordingExplainer())


def denied(call, code, status):
    with pytest.raises(StudyError) as error:
        call()
    assert (error.value.code, error.value.status) == (code, status)


def all_keys(value):
    if isinstance(value, dict):
        for key, sub in value.items():
            yield key
            yield from all_keys(sub)
    elif isinstance(value, list):
        for sub in value:
            yield from all_keys(sub)


def assert_public(value):
    forbidden = {"audit", "secret_trace", "raw_answer", "decision", "decision_json",
        "policy_memory", "private", "rng_state", "seed", "scenario_seed", "future_schedule",
        "schedule", "q_values", "recovery_hash", "token_hash"}
    assert not forbidden.intersection(all_keys(value))
    assert "must-stay-server-side" not in encode(value)


@pytest.mark.parametrize("domain", ["warehouse", "pong", "kitchen"])
@pytest.mark.parametrize("group", ["A", "B"])
def test_complete_three_task_flow_and_explanation_permission_matrix(store, domain, group):
    """Real engine transitions, fixed AI; only the simulated human is controlled."""
    flow = Flow(store, domain, group)
    assert flow.view["language"] == "en"
    assert flow.view["stage"] == "demo" and not flow.view["can_ask"]
    denied(lambda: store.ask(flow.token, flow.question()), "explanations_unavailable", 403)
    denied(lambda: flow.command("next"), "finish_demo", 409)
    flow.finish_demo()
    completed = []
    task2_question = None
    for task in (1, 2, 3):
        assert flow.view["stage"] == f"task{task}"
        allowed = group == "A" and task == 2
        assert flow.view["can_ask"] is allowed
        if allowed:
            before = flow.internal_state()
            task2_question = flow.question()
            response = store.ask(flow.token, task2_question)
            assert response["status"] == "answered"
            assert_public(response)
            assert flow.internal_state() == before
            store.acknowledge_answer(flow.token, flow.view["instance_id"], response["id"])
            task1 = completed[0]
            past = store.ask(flow.token, flow.question(target_run=task1, turn=0))
            assert past["status"] == "answered"
            assert store.explainer.calls[-1]["state"]["task"] == 1
            task1_final_turn = flow.view["task_runs"][0]["turn"]
            past_final = store.ask(flow.token, flow.question(target_run=task1, turn=task1_final_turn))
            assert past_final["status"] == "answered"
            assert store.explainer.calls[-1]["state"]["terminal"] is True
        else:
            denied(lambda: store.ask(flow.token, flow.question()), "explanations_unavailable", 403)
            assert not flow.view["questions"]
        assert_public(flow.view)
        if task == 3 and task2_question:
            denied(lambda: store.ask(flow.token, task2_question), "explanations_unavailable", 403)
            denied(lambda: store.acknowledge_answer(flow.token, flow.view["instance_id"],
                task2_question["question_id"]), "explanations_unavailable", 403)
        denied(lambda: flow.command("next"), "finish_task", 409)
        flow.finish_task()
        # The summary page is still Task 2; authorization must already be revoked.
        assert flow.view["run_status"] == "completed"
        assert not flow.view["can_ask"] and flow.view["questions"] == []
        denied(lambda: store.ask(flow.token, flow.question()), "explanations_unavailable", 403)
        score = flow.view["state"]["score"]["task_score"]
        if domain in ("warehouse", "kitchen"):
            assert score == flow.view["state"]["score"]["raw_score"]
            assert flow.view["state"]["score"]["score_max"] is None
        else:
            assert 0 <= score <= 100
        if domain == "kitchen":
            metrics = flow.view["state"]["score"]["metrics"]
            assert flow.view["state"]["score"]["score_scale"] == "raw"
            assert metrics["completed_orders"] == metrics["total_orders"] == 4
            assert metrics["step_penalty"] == flow.view["state"]["turn"]
            assert metrics["discard_penalty"] == 5 * metrics["discarded_ingredients"] + 20 * metrics["discarded_dishes"]
            assert score == 100 * metrics["completed_orders"] - metrics["step_penalty"] - metrics["discard_penalty"]
            assert metrics["burnt"] == metrics["spoiled"] == metrics["waste"] == 0
        if domain == "pong":
            assert score >= 50, f"{domain} task {task} cooperation fixture became infeasible"
        completed.append(flow.view["run_id"])
        flow.command("next")
    assert flow.view["stage"] == "questionnaire"
    assert not flow.view["can_ask"] and not flow.view["questions"]
    denied(lambda: store.ask(flow.token, flow.question()), "explanations_unavailable", 403)
    assert "answer" not in set(all_keys(flow.view["questionnaire"]))
    assert len(flow.view["questionnaire"]["items"]) == (8 if group == "A" else 5)
    assert flow.view["questionnaire"]["comprehension"] == []
    submitted = flow.submit_survey()
    assert flow.view["stage"] == "completed"
    assert store.command(flow.token, "questionnaire", submitted)["stage"] == "completed"
    new_attempt = {**submitted, "command_id": uuid.uuid4().hex, "revision": flow.view["revision"]}
    denied(lambda: store.command(flow.token, "questionnaire", new_attempt), "wrong_stage", 409)
    exported = store.export()
    assert len(exported["runs"]) == 3 and len(exported["questionnaires"]) == 1
    assert all(row["mode"] == "test" for row in exported["instances"])
    for rid in completed:
        frames = sorted((r for r in exported["frames"] if r["run_id"] == rid), key=lambda r:r["turn"])
        assert [r["turn"] for r in frames] == list(range(len(frames)))
        for before, after in zip(frames, frames[1:]):
            actual = engine(domain).step(json.loads(before["state_json"]), before["human_action"],
                json.loads(before["decision_json"]))
            assert actual == json.loads(after["state_json"])
        assert_public(store.frame(flow.token, flow.view["instance_id"], rid, 0))


def test_actions_are_idempotent_stale_safe_and_precommit_decisions_match(store):
    flow = Flow(store)
    flow.finish_demo()
    initial = flow.internal_state()
    expected = engine("pong").decide(initial)
    payload = flow.step("wait")
    once = flow.internal_state()
    duplicate = store.command(flow.token, "action", payload)
    assert duplicate["state"]["turn"] == 1 and flow.internal_state() == once
    denied(lambda: store.command(flow.token, "action", {**payload, "action": "left"}),
        "idempotency_conflict", 409)
    denied(lambda: store.command(flow.token, "action", {**payload, "command_id": uuid.uuid4().hex}),
        "stale_state", 409)
    denied(lambda: flow.command("action", action="wait", run_id="other-run", turn=1), "wrong_run", 409)
    denied(lambda: flow.command("action", action="teleport", run_id=flow.view["run_id"], turn=1),
        "illegal_action", 400)
    with store.db.transaction() as db:
        recorded = db.one("SELECT decision_json FROM pl3_frames WHERE run_id=? AND turn=0", (flow.view["run_id"],))
    assert json.loads(recorded["decision_json"]) == expected


def test_ask_language_history_replay_and_public_serialization_do_not_mutate_task(store):
    flow = Flow(store)
    flow.start_task2()
    flow.step("wait")
    baseline = flow.internal_state()
    question = flow.question(turn=0, language="zh")
    first = store.ask(flow.token, question)
    assert store.ask(flow.token, question) == {"id": first["id"], "status": "answered", "result": first["result"]}
    assert len(store.explainer.calls) == 1
    assert store.explainer.calls[0]["language"] == "zh"
    assert [frame["turn"] for frame in store.explainer.calls[0]["history"]] == [0]
    second = store.ask(flow.token, flow.question(question="What about that choice?"))
    assert store.explainer.calls[-1]["previous"] == [{"question": question["question"], "answer": first["result"]["answer"]}]
    flow.command("language", language="zh")
    assert flow.view["language"] == "zh"
    assert_public(store.frame(flow.token, flow.view["instance_id"], flow.view["run_id"], 0))
    assert_public(store.view(flow.token, flow.view["instance_id"]))
    assert_public(second)
    assert flow.internal_state() == baseline
    private = store.export()["questions"]
    assert json.loads(private[0]["result_json"])["audit"]["secret_trace"] == "must-stay-server-side"
    denied(lambda: store.ask(flow.token, {**question, "question": "Different content"}), "idempotency_conflict", 409)
    denied(lambda: store.ask(flow.token, flow.question(language="xx")), "invalid_language", 400)


def test_participant_instance_domain_and_run_boundaries(store):
    owner, outsider = Flow(store, "pong"), Flow(store, "warehouse")
    owner.start_task2()
    outsider.start_task2()
    denied(lambda: store.view(outsider.token, owner.view["instance_id"]), "study_not_found", 404)
    denied(lambda: store.frame(outsider.token, owner.view["instance_id"], owner.view["run_id"], 0), "study_not_found", 404)
    denied(lambda: store.ask(outsider.token, owner.question()), "study_not_found", 404)
    denied(lambda: store.ask(owner.token, owner.question(target_run=outsider.view["run_id"])), "frame_not_found", 404)
    denied(lambda: store.ask(owner.token, owner.question(authorized_run=owner.view["task_runs"][0]["id"])), "wrong_run", 403)
    denied(lambda: store.ask(owner.token, owner.question(turn=999)), "frame_not_found", 404)
    denied(lambda: store.view(None, owner.view["instance_id"]), "session_required", 401)
    denied(lambda: store.view("invalid-session", owner.view["instance_id"]), "session_required", 401)
    denied(lambda: store.create({"participant_id": owner.name, "domain": "pong", "consent": True,
        "mode": "test", "group": "B"}, admin=True), "participant_exists_use_recovery", 409)
    preserved = store.view(owner.token, owner.view["instance_id"])
    same_token, second_domain = store.create({"participant_id": owner.name, "domain": "kitchen", "consent": True,
        "mode": "test", "group": "B"}, token=owner.token, admin=True)
    assert same_token == owner.token and second_domain["group"] == "B"
    assert second_domain["stage"] == "demo"
    assert store.view(owner.token, owner.view["instance_id"]) == preserved


@pytest.mark.parametrize("advance_to_task3", [False, True])
def test_late_explanation_is_revoked_at_terminal_and_after_transition(tmp_path, advance_to_task3):
    explainer = BlockingExplainer()
    store = Store(Settings(database=str(tmp_path / "late.sqlite3")), explainer)
    flow = Flow(store)
    flow.start_task2()
    question = flow.question()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store.ask, flow.token, question)
        assert explainer.entered.wait(5)
        flow.finish_task()
        if advance_to_task3:
            flow.command("next")
        explainer.release.set()
        denied(lambda: future.result(timeout=10), "explanations_unavailable", 403)
    assert not store.view(flow.token, flow.view["instance_id"])["questions"]
    record = store.export()["questions"][0]
    assert record["status"] == "revoked" and record["displayed"] is None
    assert json.loads(record["result_json"])["audit"]["secret_trace"] == "must-stay-server-side"
    denied(lambda: store.ask(flow.token, question), "explanations_unavailable", 403)
    denied(lambda: store.acknowledge_answer(flow.token, flow.view["instance_id"], question["question_id"]),
        "explanations_unavailable", 403)


def test_old_command_response_never_restores_task2_chat(store):
    flow = Flow(store)
    flow.start_task2()
    store.ask(flow.token, flow.question())
    payload = flow.command("language", language="zh")
    assert len(flow.view["questions"]) == 1
    flow.finish_task()
    flow.command("next")
    replayed = store.command(flow.token, "language", payload)
    assert replayed["stage"] == "task3"
    assert replayed["questions"] == [] and not replayed["can_ask"]


def test_storage_restart_recovery_and_separate_domain_instance(tmp_path):
    path = str(tmp_path / "persistent.sqlite3")
    store = Store(Settings(database=path), RecordingExplainer())
    flow = Flow(store)
    flow.start_task2()
    flow.step()
    store.ask(flow.token, flow.question())
    expected = store.view(flow.token, flow.view["instance_id"])
    # A distinct Store/Database object opens fresh SQLite connections, as a process restart does.
    restarted = Store(Settings(database=path), RecordingExplainer())
    assert restarted.recover_view(flow.token, "pong") == expected
    new_token, recovered = restarted.create({"participant_id": flow.name, "domain": "pong",
        "mode": "test", "consent": True, "group": "B", "recovery_code": flow.recovery}, admin=True)
    assert new_token != flow.token
    assert recovered["instance_id"] == expected["instance_id"] and recovered["group"] == "A"
    assert recovered["state"] == expected["state"]
    flow.store = restarted
    flow.finish_task(); flow.command("next"); flow.finish_task(); flow.command("next"); flow.submit_survey()
    token, next_domain = restarted.create({"participant_id": flow.name, "domain": "warehouse",
        "mode": "test", "group": "B", "consent": True}, token=flow.token, admin=True)
    assert token == flow.token and next_domain["group"] == "B"
    assert next_domain["instance_id"] != flow.view["instance_id"]
    assert restarted.recover_view(token)["domain"] == "warehouse"
    assert restarted.recover_view(token, "pong")["stage"] == "completed"
    assert len(restarted.export()["questionnaires"]) == 1


def test_mutating_from_two_tabs_only_commits_one_action(store):
    flow = Flow(store)
    flow.finish_demo()
    state = flow.view["state"]
    payload = {"instance_id": flow.view["instance_id"], "revision": flow.view["revision"],
        "run_id": flow.view["run_id"], "turn": state["turn"], "action": "wait"}
    barrier = threading.Barrier(2)
    def submit():
        barrier.wait()
        try:
            return store.command(flow.token, "action", {**payload, "command_id": uuid.uuid4().hex})
        except StudyError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert sum(isinstance(r, dict) for r in results) == 1
    assert "stale_state" in results and flow.internal_state()["turn"] == 1


def test_consent_modes_input_and_failure_are_honest(store):
    payload = {"participant_id": "test-mode", "domain": "pong", "mode": "test", "consent": True}
    denied(lambda: store.create(payload), "researcher_access_required", 403)
    denied(lambda: store.create({**payload, "consent": False}, admin=True), "consent_required", 400)
    denied(lambda: store.create({**payload, "mode": "pilot"}), "study_not_ready", 503)
    denied(lambda: store.create({**payload, "participant_id": "x'; DROP TABLE participants;"}, admin=True), "invalid_participant_id", 400)
    flow = Flow(store)
    flow.start_task2()
    store.explainer = None
    unavailable = store.ask(flow.token, flow.question())
    assert unavailable["status"] == "unavailable"
    assert unavailable["result"]["status"] == "unavailable"
    assert store.export()["questions"][0]["status"] == "unavailable"


def test_questionnaire_validation_and_timings_are_transactional(store):
    flow = Flow(store)
    flow.finish_demo()
    before = flow.internal_state()
    flow.command("timing", kind="reading", seconds=2.5)
    assert flow.internal_state() == before
    denied(lambda: flow.command("language", language="xx", timings=[{"kind": "active", "seconds": 3}]), "invalid_language", 400)
    assert len(store.export()["timings"]) == 1
    denied(lambda: flow.command("timing", kind="private_state", seconds=2), "invalid_timing", 400)
    for task in (1, 2, 3):
        flow.finish_task(); flow.command("next")
    denied(lambda: flow.command("questionnaire", answers={}, comprehension={}), "incomplete_questionnaire", 400)
    assert not store.export()["questionnaires"]
    flow.submit_survey()
    assert len(store.export()["questionnaires"]) == 1


@pytest.mark.parametrize("domain", ["warehouse", "pong", "kitchen"])
def test_group_assignment_never_changes_scene_ai_decision_or_result(store, domain):
    a, b = Flow(store, domain, "A"), Flow(store, domain, "B")
    a.start_task2(); b.start_task2()
    assert a.internal_state() == b.internal_state()
    store.ask(a.token, a.question())
    for _ in range(8):
        left, right = a.internal_state(), b.internal_state()
        assert left == right
        assert engine(domain).decide(left) == engine(domain).decide(right)
        action = engine(domain).human_advisor(left)
        a.step(action); b.step(action)
    assert a.internal_state() == b.internal_state()
    assert a.view["state"]["score"] == b.view["state"]["score"]


def test_release_mismatch_never_resumes_old_session_under_new_rules(store, monkeypatch):
    flow = Flow(store)
    flow.start_task2()
    question = flow.question()
    store.ask(flow.token, question)
    monkeypatch.setattr("study_v3.store.RELEASE_ID", "test-new-release")
    monkeypatch.setattr("study_v3.store.SUPPORTED_RELEASE_IDS", frozenset({"test-new-release"}))
    denied(lambda: store.view(flow.token, flow.view["instance_id"]), "release_changed", 409)
    assert store.recover_view(flow.token, "pong")["previous_version_saved"] is True
    denied(lambda: store.ask(flow.token, question), "release_changed", 409)
    denied(lambda: store.frame(flow.token, flow.view["instance_id"], flow.view["run_id"], 0), "release_changed", 409)


@pytest.mark.parametrize("requested_domain", ["pong", "kitchen"])
def test_unsupported_release_can_start_fresh_without_changing_archived_rows(store, monkeypatch, requested_domain):
    flow = Flow(store, "pong")
    flow.finish_demo()
    original_records = store.export()
    monkeypatch.setattr("study_v3.store.RELEASE_ID", "test-new-release")
    monkeypatch.setattr("study_v3.store.SUPPORTED_RELEASE_IDS", frozenset({"test-new-release"}))

    payload = {"participant_id": flow.name, "domain": requested_domain,
        "consent": True, "mode": "test"}
    _, new_domain = store.create(payload, token=flow.token, admin=True)
    assert new_domain["domain"] == requested_domain and new_domain["release_id"] == "test-new-release"
    assert new_domain["stage"] == "demo"
    after = store.export()
    for table, records in original_records.items():
        assert all(row in after[table] for row in records)



@pytest.mark.parametrize("domain", ["warehouse", "pong", "kitchen"])
def test_enrollment_preserves_consent_language_assignment_and_heldout_scenario(store, domain):
    flow = Flow(store, domain)
    enrollment = store.export()["enrollments"][0]
    instance = store.export()["instances"][0]
    frozen = scenario_config(domain)
    assert enrollment["instance_id"] == flow.view["instance_id"]
    assert enrollment["consent"] == 1 and enrollment["initial_language"] == "en"
    assert enrollment["assignment_source"] == "researcher_override"
    assert enrollment["scenario_version"] == frozen.get("scenario_version", frozen["version"])
    assert instance["scenario_seed"] in (frozen.get("heldout_seeds") or frozen["held_out_seeds"])
    flow.command("language", language="zh")
    assert flow.view["language"] == "zh"
    assert store.export()["enrollments"][0] == enrollment
    denied(lambda: store.create({"participant_id": flow.name, "domain": domain, "mode": "preview",
        "consent": True}, token=flow.token, admin=True), "participant_mode_conflict", 409)
    assert len(store.export()["enrollments"]) == 1


def test_export_filter_keeps_all_child_records_with_their_selected_instances(store):
    current = Flow(store)
    current.start_task2()
    store.ask(current.token, current.question())
    older = Flow(store)
    older.start_task2()
    store.ask(older.token, older.question())
    with store.db.transaction() as db:
        db.execute("UPDATE pl3_instances SET release_id=? WHERE id=?", ("test-older-release", older.view["instance_id"]))
    _, preview = store.create({"participant_id": "preview-filter-" + uuid.uuid4().hex, "domain": "kitchen",
        "consent": True, "mode": "preview", "group": "B"}, admin=True)
    current_release = current.view["release_id"]
    selected = store.export(release_id=current_release, mode="test")
    assert [r["id"] for r in selected["instances"]] == [current.view["instance_id"]]
    assert [r["id"] for r in selected["participants"]] == [current.name]
    for table in ("enrollments", "runs", "questions", "questionnaires", "timings"):
        assert all(r["instance_id"] == current.view["instance_id"] for r in selected[table])
    runs = {r["id"] for r in selected["runs"]}
    assert selected["frames"] and all(r["run_id"] in runs for r in selected["frames"])
    assert [r["id"] for r in store.export(mode="preview")["instances"]] == [preview["instance_id"]]
    assert [r["id"] for r in store.export(release_id="test-older-release")["instances"]] == [older.view["instance_id"]]
    assert all(not rows for rows in store.export(release_id="does-not-exist").values())


def test_unverified_configuration_and_recent_qa_failure_block_new_pilot_admission(store):
    configured = replace(store.settings, persistent=True, admin_token="test-admin", llm_base_url="https://example.invalid",
        llm_model="test-model",mode="pilot")
    assert configured.llm_configured and not configured.ready
    verified = Store(replace(configured, verified=True), RecordingExplainer())
    assert verified.ready
    flow = Flow(verified)
    flow.start_task2()
    verified.explainer = None
    assert verified.ask(flow.token, flow.question())["status"] == "unavailable"
    assert not verified.ready
    denied(lambda: verified.create({"participant_id": "pilot-admission-fixture", "domain": "pong",
        "mode": "pilot", "consent": True}), "study_not_ready", 503)
    assert len(verified.export()["participants"]) == 1


def test_interrupted_question_lease_recovers_after_restart(store):
    import time
    flow=Flow(store);flow.start_task2()
    payload=flow.question(language="zh")
    with store.db.transaction() as db:
        db.execute("INSERT INTO pl3_questions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(payload['question_id'],flow.view['instance_id'],flow.view['run_id'],flow.view['run_id'],flow.view['state']['turn'],payload['question'],'zh','{}','pending',time.time()-121,None,None))
    restored=Store(store.settings,RecordingExplainer())
    previous=restored.ask(flow.token,payload)
    assert previous['status']=='unavailable'
    assert '中断' in previous['result']['answer']
    answer=restored.ask(flow.token,flow.question())
    assert answer['status']=='answered'
    rows=restored.export()['questions']
    assert len(rows)==2
    completed = next(row for row in rows if row['id'] == answer['id'])
    assert json.loads(completed['result_json'])['audit']['authorization_at_completion'] is True
