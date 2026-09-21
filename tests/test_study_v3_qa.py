"""Semantic-provider protocol, verified evidence and real simulation tests.

Injected plans test composition; they are not scored as model understanding.
The --smoke CLI exercises an actually configured provider independently.
"""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from domains.pong import turnbased as pong
from study_v3.config import Settings
from study_v3.qa import Explainer, PlanError, _catalog, evaluate_case, simulate
from study_v3.registry import engine as get_engine

ROOT = Path(__file__).resolve().parents[1]


def fixture(*, human=2, ai=6, turns=4, small=None, small_turns=2):
    balls = [pong._ball("fixture-team", "cooperative", [2, 6], turns)]
    if small is not None:
        balls.append(pong._ball("fixture-small", "ordinary", [small], small_turns))
    return pong._new_state([balls], seed=200, task=2, human=human, ai=ai)


def plan(state, *, language="en", ids=("decision",), intents=None, clarification=None):
    return {"language": language, "binding": {"task": state["task"], "turn": state["turn"]},
            "premise": "supported", "clarification": clarification,
            "intents": intents if intents is not None else [{"kind": "facts", "evidence_ids": list(ids)}]}


def explainer_for_plan(selected_plan, capture=None):
    instance = Explainer(Settings(database=":memory:", llm_base_url="http://test.invalid/v1", llm_model="protocol-fixture"))
    def request(payload):
        if capture is not None:
            capture.append(deepcopy(payload))
        return deepcopy(selected_plan), {"provider_response_id": "injected-protocol-plan", "reported_model": "not-a-semantic-evaluation"}
    instance._request_plan = request
    return instance


def ask(instance, state, **kwargs):
    return instance.answer(pong, state, pong.decide(state), kwargs.pop("question", "What happens?"), **kwargs)


def test_unconfigured_service_is_honestly_unavailable_without_keyword_answers():
    state = fixture()
    instance = Explainer(Settings(database=":memory:"))
    for question, expected_language in (("Why are you waiting?", "en"), ("你为什么等待？", "zh")):
        result = ask(instance, state, question=question)
        assert result["status"] == "unavailable"
        assert not result["evidence_ids"]
        assert result["language"] == expected_language
        assert result["audit"]["failure_code"] == "not_configured"


def test_final_factual_text_is_selected_verified_evidence_not_model_prose():
    state = fixture()
    selected = plan(state, ids=("position", "decision"))
    result = ask(explainer_for_plan(selected), state)
    evidence = {r["id"]: r for r in pong.facts(state)}
    assert result["status"] == "answered"
    assert result["answer"] == "Task 2 · Turn 0\n\n" + evidence["position"]["en"] + "\n\n" + evidence["decision"]["en"]


def test_duplicate_fact_aliases_render_once_but_preserve_audit_ids():
    state = fixture()
    decision = pong.decide(state)
    selected = plan(state, ids=("decision", "system:ai_reason"))
    result = ask(explainer_for_plan(selected), state)
    assert result["status"] == "answered"
    assert result["answer"].count(decision["reason_en"]) == 1
    assert result["evidence_ids"] == ["decision", "system:ai_reason"]


def test_model_cannot_add_assertions_or_select_nonexistent_evidence():
    state = fixture()
    wrong = plan(state, ids=("fake:guaranteed-win",))
    assert ask(explainer_for_plan(wrong), state)["status"] == "unavailable"
    wrong = plan(state)
    wrong["answer"] = "I guarantee the human will win."
    result = ask(explainer_for_plan(wrong), state)
    assert result["status"] == "unavailable"
    assert "guarantee" not in result["answer"]


def test_unknown_evidence_gets_only_one_model_repair_never_heuristic_substitution():
    state = fixture()
    bad = plan(state, ids=("invented-fact",))
    good = plan(state, ids=("position",))
    instance = explainer_for_plan(good)
    calls = []
    def request(payload):
        calls.append(deepcopy(payload))
        return deepcopy(bad if len(calls) == 1 else good), {}
    instance._request_plan = request
    result = ask(instance, state)
    assert result["status"] == "answered"
    assert len(calls) == 2
    assert "repair_request" in calls[1]
    assert result["audit"]["repair_attempt"]["successful"]
    calls.clear()
    instance._request_plan = lambda payload: (calls.append(payload) or deepcopy(bad), {})
    result = ask(instance, state)
    assert result["status"] == "unavailable"
    assert len(calls) == 2


def test_bilingual_question_language_overrides_interface():
    state = fixture()
    result = ask(explainer_for_plan(plan(state, language="zh")), state, question="为什么这样做？", language="en")
    assert result["status"] == "answered"
    assert pong.decide(state)["reason_zh"] in result["answer"]
    assert result["language"] == "zh"


def test_context_and_history_sent_as_data_without_hidden_schedule_or_identity():
    state = pong.initial_state(730100, 2)
    captured = []
    prior = [{"question": "What about the other side?", "answer": "A verified earlier response."}]
    result = ask(explainer_for_plan(plan(state), captured), state, previous_dialogue=prior, public_history=[pong.public_state(state)])
    assert result["status"] == "answered"
    payload = captured[0]
    assert payload["prior_dialogue"] == prior
    text = json.dumps(payload)
    for forbidden in ("_schedule", "policy_memory", '"seed"', "participant_id", "llm_api_key"):
        assert forbidden not in text
    assert "w2-team" not in text


def test_hidden_future_changes_leave_semantic_provider_payload_identical():
    state = pong.initial_state(730105, 2)
    other = deepcopy(state)
    other["_schedule"][1] = [pong._ball("SECRET-UNANNOUNCED", "ordinary", [8], 99)]
    one, two = [], []
    ask(explainer_for_plan(plan(state), one), state)
    ask(explainer_for_plan(plan(state), two), other)
    assert one == two


def test_history_facts_are_bound_and_future_or_other_domain_frames_excluded():
    state = fixture(turns=3)
    old = pong.public_state(state)
    state = pong.step(state, "wait")
    future = deepcopy(old)
    future["turn"] = 99
    future["events"] = [{"type": "secret", "en": "Future secret", "zh": "未来秘密"}]
    other = deepcopy(old)
    other["domain"] = "kitchen"
    rows = _catalog(pong, state, pong.decide(state), [old, future, other])
    assert "history:task2:turn0:human_position" in rows
    assert "Future secret" not in json.dumps(rows)


def test_explicit_other_turn_requires_selection_without_fabricated_old_reason():
    state = fixture()
    selected = plan(state)
    selected["binding"]["turn"] = 9
    result = ask(explainer_for_plan(selected), state)
    assert result["status"] == "clarification"
    assert "select" in result["answer"].lower()


def test_available_historical_position_does_not_need_unnecessary_frame_reselection():
    first = fixture()
    current = pong.step(first, "right")
    selected = plan(current, ids=("history:task2:turn0:human_position",))
    selected["binding"]["turn"] = 0
    result = ask(explainer_for_plan(selected), current,
                 public_history=[pong.public_state(first), pong.public_state(current)])
    assert result["status"] == "answered"
    assert result["answer"].startswith("Task 2 · Turn 0")
    assert "lane 3" in result["answer"]


def test_ambiguous_question_can_clarify_and_wrong_premise_uses_correcting_facts():
    state = fixture()
    clarification = plan(state, intents=[], clarification="ambiguous_object")
    assert ask(explainer_for_plan(clarification), state)["status"] == "clarification"
    selected = plan(state, ids=("next_action",))
    selected["premise"] = "contradicted"
    result = ask(explainer_for_plan(selected), state)
    assert "differs" in result["answer"]
    assert "wait" in result["answer"]


def test_multintent_combines_why_and_actual_counterfactual_results():
    state = fixture(human=1, ai=7, turns=1)
    selected = plan(state, intents=[{"kind": "facts", "evidence_ids": ["decision"]},
        {"kind": "counterfactual", "evidence_ids": [], "actions": ["right"], "horizon": 1}])
    result = ask(explainer_for_plan(selected), state)
    assert result["status"] == "answered"
    assert "caught: +3 points" in result["answer"]
    assert result["audit"]["simulations"][0]["raw_score_delta"] == 3
    assert "In the selected recorded state:" in result["answer"]
    assert result["answer"].index(pong.decide(state)["reason_en"]) < result["answer"].index("If you move right")


def test_two_counterfactuals_compare_real_different_outcomes():
    state = fixture(human=1, ai=7, turns=1)
    selected = plan(state, intents=[{"kind": "counterfactual", "evidence_ids": [], "actions": [a], "horizon": 1} for a in ("right", "wait")])
    result = ask(explainer_for_plan(selected), state)
    assert [s["raw_score_delta"] for s in result["audit"]["simulations"]] == [3, 0]


def test_first_simulated_ai_action_matches_recorded_decision_and_state_unchanged():
    state = fixture(human=4, ai=4)
    original = deepcopy(state)
    decision = pong.decide(state)
    for human_action in ("left", "right", "wait"):
        result = simulate(pong, state, decision, [human_action], 2)
        assert result["trace"][0]["ai_action"] == decision["action"]
        next_real = pong.step(state, human_action, decision)
        assert result["trace"][1]["ai_action"] == pong.decide(next_real)["action"]
    assert state == original


def test_multi_step_wait_assumption_is_expressed_and_simulates():
    state = fixture(human=1, ai=7, turns=3)
    selected = plan(state, intents=[{"kind": "counterfactual", "evidence_ids": [], "actions": ["right"], "horizon": 3}])
    result = ask(explainer_for_plan(selected), state)
    simulation = result["audit"]["simulations"][0]
    assert simulation["executed_actions"] == ["right", "wait", "wait"]
    assert "explicit assumption" in result["answer"]
    assert simulation["raw_score_delta"] == 3


@pytest.mark.parametrize("horizon", (0, 13, -1, 1.5, True))
def test_unbounded_or_invalid_horizon_rejected(horizon):
    state = fixture()
    with pytest.raises(PlanError):
        simulate(pong, state, pong.decide(state), ["wait"], horizon)


def test_illegal_action_is_not_executed_and_no_game_state_changes():
    state = fixture(human=0)
    result = simulate(pong, state, pong.decide(state), ["left"], 1)
    assert result["steps_completed"] == 0
    assert result["illegal_action"] == {"action": "left", "step": 1}
    assert result["human"] == state["human"]


def test_pong_forward_simulation_stops_before_exposing_hidden_next_wave():
    waves = [[pong._ball("known", "cooperative", [2, 6], 1)],
             [pong._ball("TOP-SECRET-NEXT-BALL", "cooperative", [0, 8], 6)]]
    state = pong._new_state(waves, seed=200, task=2, human=2, ai=6)
    result = simulate(pong, state, pong.decide(state), ["wait"], 5)
    assert result["steps_completed"] == 1
    assert result["stopped_at_public_boundary"]
    assert result["raw_score_delta"] == 3
    assert "TOP-SECRET" not in json.dumps(result)
    assert "wave_started" not in json.dumps(result)


def test_kitchen_public_menu_is_fully_known_without_an_artificial_future_order_boundary():
    kitchen = get_engine("kitchen")
    state = kitchen.initial_state(730100, 3)
    before = deepcopy(state)
    public = kitchen.public_state(state)
    assert state["_future_orders"] == []
    assert len(public["orders"]) == 5
    assert {order["id"] for order in public["orders"]} == {order["id"] for order in state["orders"]}
    result = simulate(kitchen, state, kitchen.decide(state), ["wait"], 4)
    assert result["steps_completed"] == 4
    assert not result["stopped_at_public_boundary"]
    assert "_future_orders" not in json.dumps(result)
    assert state == before


def test_restored_warehouse_counterfactual_uses_actual_unbounded_score():
    warehouse = get_engine("warehouse")
    state = warehouse.initial_state(730100, 2)
    selected = plan(state, intents=[{"kind": "counterfactual", "evidence_ids": [], "actions": ["wait"], "horizon": 1}])
    result = explainer_for_plan(selected).answer(warehouse, state, warehouse.decide(state), "What is the net score change if I wait?", "en", [], [])
    simulation = result["audit"]["simulations"][0]
    assert result["status"] == "answered"
    assert simulation['raw_score_delta'] == simulation['task_score_delta']
    assert f"Task score changes by {simulation['raw_score_delta']:g} points" in result["answer"]
    assert f"Task score changes by {simulation['task_score_delta']:g}" in result["answer"]
    assert simulation['raw_score_delta'] == simulation['task_score_delta']
    assert '/100' not in result['answer']


def test_simulation_returns_public_actor_projection_not_internal_actor_fields():
    warehouse = get_engine("warehouse")
    state = warehouse.initial_state(730100, 2)
    state["human"]["private_note"] = "NEVER-EXPOSE-THIS"
    result = simulate(warehouse, state, warehouse.decide(state), ["wait"], 1)
    assert "NEVER-EXPOSE-THIS" not in json.dumps(result)


def test_recorded_terminal_frame_has_no_next_action_even_with_empty_decision():
    state = pong.step(fixture(turns=1), "wait")
    result = explainer_for_plan(plan(state, ids=("next_action",))).answer(pong, state, {}, "What happens next?", "en", [], [])
    assert result["status"] == "answered"
    assert "no next action" in result["answer"]


@pytest.mark.parametrize("semantics", ({"subject": "ai", "purpose": "observation"}, {},
    {"subject": None, "purpose": "observation"}, {"subject": "ai", "purpose": None},
    {"subject": "unknown", "purpose": "observation"}, {"subject": "ai", "purpose": "unknown"}))
def test_real_http_transport_structure_auth_and_no_secret_audit(semantics):
    received = []
    state = fixture()
    selected = plan(state)
    selected["intents"][0].update(semantics)
    include_semantics = semantics == {"subject": "ai", "purpose": "observation"}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            received.append({"path": self.path, "auth": self.headers.get("Authorization"),
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            body = json.dumps({"id": "test-secret", "model": "fixture", "choices": [{"message": {"content": json.dumps(selected)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        instance = Explainer(Settings(database=":memory:", llm_base_url=f"http://127.0.0.1:{server.server_port}/v1", llm_model="fixture", llm_api_key="test-secret"))
        result = ask(instance, state)
        assert result["status"] == ("answered" if include_semantics else "unavailable")
        if not include_semantics:
            assert result["audit"]["failure_code"] == "missing_provider_semantics"
        assert received[0]["path"] == "/v1/chat/completions"
        assert received[0]["auth"] == "Bearer test-secret"
        assert received[0]["body"]["response_format"] == {"type": "json_object"}
        assert "test-secret" not in json.dumps(result)
        assert "test-secret" not in json.dumps(received[0]["body"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_provider_protocol_failure_is_unavailable_not_fake_success():
    state = fixture()
    instance = Explainer(Settings(database=":memory:", llm_base_url="http://127.0.0.1:1/v1", llm_model="unreachable"), timeout=.1)
    result = ask(instance, state)
    assert result["status"] == "unavailable"
    assert result["audit"]["failure_code"] == "connection_or_timeout"


def _case_files():
    config = json.loads((ROOT / "configs/study_v3_qa_cases.json").read_text())
    cases = list(config["cases"])
    for name in config["external_case_files"]:
        path = ROOT / name
        if path.exists():
            data = json.loads(path.read_text())
            cases.extend(data if isinstance(data, list) else data["cases"])
    return cases


def _lookup(value, path):
    for component in path.split("."):
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


@pytest.mark.parametrize("case", _case_files(), ids=lambda c: c["case_id"])
def test_recorded_question_case_independent_expectations_and_composition(case):
    engine = get_engine(case.get("domain", case["state"]["domain"]))
    state = deepcopy(case["state"])
    decision = engine.decide(state)
    catalog = _catalog(engine, state, decision, case.get("public_history", []))
    for identifier in case["expected_fact_ids"]:
        assert identifier in catalog
    selected = case.get("expected_plan")
    if not selected:
        selected = plan(state, language=case["language"], ids=case["expected_fact_ids"])
    instance = explainer_for_plan(selected)
    result = instance.answer(engine, state, decision, case["question"], case["language"], case.get("previous_dialogue", []), case.get("public_history", []))
    assert result["status"] == ("clarification" if selected.get("clarification") else "answered")
    assert set(case["expected_fact_ids"]) <= set(result["evidence_ids"])
    for claim in case.get("expected_claims", []):
        if isinstance(claim, str):
            assert claim in result["answer"]
        elif "path" in claim:
            assert _lookup(state, claim["path"]) == claim["equals"]
        elif "decision_path" in claim:
            assert _lookup(decision, claim["decision_path"]) == claim["equals"]
        elif "simulation_path" in claim:
            assert _lookup(result["audit"]["simulations"][0], claim["simulation_path"]) == claim["equals"]
        elif "fact_id" in claim:
            assert claim["contains"] in catalog[claim["fact_id"]][case["language"]]
        else:
            raise AssertionError(f"Unknown independent claim format: {claim}")


def test_pong_case_coverage_has_sixty_distinct_questions_and_expected_claims():
    cases = [c for c in _case_files() if c.get("domain", c["state"]["domain"]) == "pong"]
    assert len(cases) >= 60
    assert {c["language"] for c in cases} == {"en", "zh"}
    assert len({c["case_id"] for c in cases}) == len(cases)
    assert len({c["question"] for c in cases}) >= 60
    assert {c["expected_kind"] for c in cases} >= {"facts", "counterfactual", "clarification"}


@pytest.mark.parametrize("domain", ("pong", "warehouse", "kitchen"))
def test_all_three_domains_have_at_least_sixty_bilingual_cases(domain):
    cases = [c for c in _case_files() if c.get("domain", c["state"]["domain"]) == domain]
    assert len(cases) >= 60
    assert {c["language"] for c in cases} == {"en", "zh"}


def test_real_provider_evaluator_does_not_count_string_outputs_as_correctness():
    state = fixture()
    case = {"state": state, "language": "en", "expected_kind": "facts", "expected_fact_ids": ["position"], "expected_claims": []}
    unrelated = {"status": "answered", "language": "en", "answer": "This is a string.", "evidence_ids": [], "audit": {"simulations": []}}
    report = evaluate_case(case, unrelated, pong)
    assert not report["passed"]
    assert "missing_evidence:position" in report["issues"]
    actual = ask(explainer_for_plan(plan(state, ids=("position",))), state)
    report = evaluate_case(case, actual, pong)
    assert report["passed"] and report["requires_human_review"]


@pytest.mark.parametrize("domain", ("warehouse", "pong", "kitchen"))
def test_concrete_human_advice_and_ai_intention_have_authoritative_separate_roles(domain):
    engine = get_engine(domain)
    state = engine.initial_state(730100, 2)
    original = deepcopy(state)
    decision = engine.decide(state)
    evidence = _catalog(engine, state, decision, [])
    advice = evidence["system:human_advice"]
    assert advice["subject"] == "human"
    assert advice["action"] == engine.human_advisor(state)
    assert advice["action"] in engine.legal_actions(state)
    assert evidence["system:ai_reason"]["en"] == decision["reason_en"]
    assert evidence["system:ai_reason"]["zh"] == decision["reason_zh"]
    assert evidence["system:ai_action"]["subject"] == "ai"
    assert state == original


def test_historical_ai_reason_cannot_be_answered_with_human_event_evidence():
    engine = get_engine("warehouse")
    state = engine.initial_state(730100, 1)
    for _ in range(10):
        state = engine.step(state, engine.human_advisor(state))
    wrong = plan(state, intents=[{"kind": "facts", "subject": "ai", "purpose": "reason",
                                 "evidence_ids": ["human_state", "human_charger_distance"]}])
    result = explainer_for_plan(wrong).answer(engine, state, engine.decide(state), "What were you trying to do and why?")
    assert result["status"] == "unavailable"
    assert result["audit"]["failure_code"] == "missing_subject_evidence"
    assert result["audit"]["repair_attempt"]["failure_code"] == "missing_subject_evidence"


def test_human_advice_cannot_be_replaced_with_legal_action_list():
    state = fixture()
    wrong = plan(state, intents=[{"kind": "facts", "subject": "human", "purpose": "advice",
                                 "evidence_ids": ["system:available_actions"]}])
    result = ask(explainer_for_plan(wrong), state, question="Which move should I choose?")
    assert result["status"] == "unavailable"
    assert result["audit"]["failure_code"] == "missing_subject_evidence"


def test_human_comparison_rejects_ai_alternative_and_preserves_two_real_branches():
    engine = get_engine("warehouse")
    state = engine.initial_state(730100, 2)
    wrong = plan(state, intents=[{"kind": "facts", "subject": "human", "purpose": "comparison",
                                 "evidence_ids": ["alternative_wait"]}])
    result = explainer_for_plan(wrong).answer(engine, state, engine.decide(state), "Why is that better than waiting here?")
    assert result["status"] == "unavailable"
    assert result["audit"]["failure_code"] == "wrong_actor_evidence"
    recommended = engine.human_advisor(state)
    correct = plan(state, intents=[{"kind": "counterfactual", "subject": "human", "purpose": "comparison",
        "evidence_ids": [], "actions": [action], "horizon": 1} for action in (recommended, "wait")])
    original = deepcopy(state)
    result = explainer_for_plan(correct).answer(engine, state, engine.decide(state), "Why is that better than waiting here?")
    assert result["status"] == "answered"
    assert [s["requested_actions"] for s in result["audit"]["simulations"]] == [[recommended], ["wait"]]
    assert state == original


def test_provider_payload_disambiguates_speakers_and_carries_concrete_action():
    state = fixture()
    captured = []
    selected = plan(state, intents=[{"kind": "facts", "subject": "human", "purpose": "advice",
        "evidence_ids": ["system:human_advice"]}])
    result = ask(explainer_for_plan(selected, captured), state, question="How should I help?")
    assert result["status"] == "answered"
    assert captured[0]["speaker_roles"]["participant_question_you"] == "ai"
    assert captured[0]["speaker_roles"]["assistant_answer_you"] == "human"
    assert captured[0]["recommended_human_action"] == pong.human_advisor(state)


def test_simulation_cannot_be_bound_to_ai_actor():
    state = fixture()
    selected = plan(state, intents=[{"kind": "counterfactual", "subject": "ai", "purpose": "comparison",
        "evidence_ids": [], "actions": ["wait"], "horizon": 1}])
    result = ask(explainer_for_plan(selected), state)
    assert result["status"] == "unavailable"
    assert result["audit"]["failure_code"] == "wrong_simulation_actor"


@pytest.mark.parametrize("language", ["en", "zh"])
def test_position_hypothesis_recomputes_actual_controller_without_advancing_game(language):
    state = fixture(human=0, ai=4, turns=4)
    original = deepcopy(state)
    actual = pong.decide(state)
    assert actual['action'] == 'right'
    intent = {"kind": "counterfactual", "subject": "human", "purpose": "comparison",
              "evidence_ids": [], "intervention": {"human_lane": 9}, "actions": [], "horizon": 0}
    capture = []
    result = ask(explainer_for_plan(plan(state, language=language, intents=[intent]), capture), state,
                 question="If I were at lane 9, how would you move?" if language == 'en' else "如果我在第9道，你会怎么移动？")
    assert result['status'] == 'answered'
    simulation = result['audit']['simulations'][0]
    assert simulation['intervention']['ai_action'] == 'left'
    assert simulation['intervention']['actual_human_lane'] == 1
    assert simulation['intervention']['human_lane'] == 9
    assert simulation['steps_completed'] == 0 and simulation['trace'] == []
    assert simulation['human']['x'] == 8 and simulation['ai']['x'] == 4
    assert simulation['raw_score_delta'] == simulation['task_score_delta'] == 0
    assert state == original and pong.decide(state) == actual
    assert result['evidence_ids'] == ['simulation:1']
    assert capture[0]['public_observation']['human']['x'] == 0
    assert capture[0]['counterfactual_interventions']['human_lane']['maximum'] == 9
    if language == 'en':
        assert 'If you were in lane 9: ' in result['answer']
        assert result['answer'].count('move left') == 1
        assert 'hypothetical position' in result['answer'] and 'game has not changed' in result['answer']
    else:
        assert '假设你在第9道：' in result['answer']
        assert result['answer'].count('左移') == 1
        assert '实际游戏没有改变' in result['answer']


def test_position_hypothesis_keeps_ai_commitment_even_when_human_can_no_longer_reach():
    state = fixture(human=0, ai=4, turns=4)
    state['policy_memory'] = deepcopy(pong.decide(state)['memory'])
    original = deepcopy(state)
    # Lane 7 makes the other split shorter, but the actual fixed controller
    # keeps an existing still-reachable agreement instead of erasing memory.
    maintained = simulate(pong, state, pong.decide(state), [], 0, intervention={'human_lane': 7})
    assert maintained['intervention']['ai_action'] == 'right'
    no_commitment = deepcopy(state)
    no_commitment['policy_memory'] = {}
    changed = simulate(pong, no_commitment, pong.decide(no_commitment), [], 0,
                       intervention={'human_lane': 7})
    assert changed['intervention']['ai_action'] == 'left'
    maintained_after_deviation = simulate(pong, state, pong.decide(state), [], 0, intervention={'human_lane': 9})
    assert maintained_after_deviation['intervention']['ai_action'] == 'right'
    assert maintained_after_deviation['intervention']['target_unchanged']
    assert 'no longer reach' in maintained_after_deviation['intervention']['reason_en']
    assert state == original


def test_position_and_action_counterfactual_steps_real_physics_from_modified_state():
    state = fixture(human=0, ai=4, turns=4)
    original = deepcopy(state)
    branch = deepcopy(state)
    branch['human']['x'] = 6
    expected = []
    for action in ['wait', 'wait']:
        decision = pong.decide(branch)
        branch = pong.step(branch, action, decision)
        expected.append((decision['action'], branch['human']['x'], branch['ai']['x']))
    result = simulate(pong, state, pong.decide(state), ['wait'], 2,
                      intervention={'human_lane': 7})
    assert [(row['ai_action'], row['human']['x'], row['ai']['x']) for row in result['trace']] == expected
    assert expected == [('left', 6, 3), ('left', 6, 2)]
    assert result['executed_actions'] == ['wait', 'wait'] and result['assumed_wait_turns'] == 1
    assert result['human'] == branch['human'] and result['ai'] == branch['ai']
    assert state == original


def test_position_hypothesis_with_explicit_window_discloses_assumed_waits():
    state = fixture(human=0, ai=4, turns=4)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'intervention': {'human_lane': 7},
        'horizon': 2}])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'answered'
    assert result['audit']['simulations'][0]['executed_actions'] == ['wait', 'wait']
    assert 'explicit assumption' in result['answer']
    assert 'From that hypothetical position' in result['answer']


@pytest.mark.parametrize('intervention', [{'human_lane': 0}, {'human_lane': 10},
    {'human_lane': -1}, {'human_lane': True}, {'human_lane': 1.5}, {'human_lane': '3'},
    {'human_lane': 2, 'ai_lane': 6}, {'ai_lane': 6}, {}, []])
def test_unvalidated_position_changes_are_never_simulated(intervention):
    state = fixture()
    original = deepcopy(state)
    with pytest.raises(PlanError):
        simulate(pong, state, pong.decide(state), [], 0, intervention=intervention)
    assert state == original
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'intervention': intervention,
        'actions': [], 'horizon': 0}])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'unavailable'
    assert result['audit']['simulations'] == []


@pytest.mark.parametrize('domain', ['warehouse', 'kitchen'])
def test_lane_hypotheses_are_not_applied_to_other_domains(domain):
    engine = get_engine(domain)
    state = engine.initial_state(730100, 2)
    original = deepcopy(state)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'intervention': {'human_lane': 3},
        'actions': [], 'horizon': 0}])
    captured = []
    result = explainer_for_plan(selected, captured).answer(engine, state, engine.decide(state), 'If I were in lane 3?')
    assert result['status'] == 'unavailable'
    assert captured[0]['counterfactual_interventions'] == {}
    assert result['audit']['simulations'] == [] and state == original


def test_position_hypothesis_does_not_leak_new_balls_or_follow_their_schedule():
    state = pong._new_state([[pong._ball('known-team', 'cooperative', [2, 6], 1)],
                            [pong._ball('HIDDEN-BALL', 'ordinary', [8], 3)]],
                           seed=200, task=2, human=0, ai=6)
    other = deepcopy(state)
    other['_schedule'][1]['ball']['contacts'] = [0]
    one = simulate(pong, state, pong.decide(state), ['wait'], 5, intervention={'human_lane': 3})
    two = simulate(pong, other, pong.decide(other), ['wait'], 5, intervention={'human_lane': 3})
    assert one['steps_completed'] == two['steps_completed'] == 1
    assert one['stopped_at_public_boundary'] and two['stopped_at_public_boundary']
    assert one['trace'] == two['trace'] and one['raw_score_delta'] == 3
    assert 'HIDDEN-BALL' not in json.dumps(one)
    assert one['intervention']['ai_action'] == two['intervention']['ai_action']


def test_position_hypothesis_requires_exact_selected_historical_state():
    state = fixture(human=0, ai=4, turns=4)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'intervention': {'human_lane': 9}}])
    selected['binding']['turn'] = 8
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'clarification' and 'select' in result['answer'].lower()
    assert result['audit']['simulations'] == []


def test_actual_reason_cannot_masquerade_as_hypothetical_position_evidence():
    state = fixture(human=0, ai=4, turns=4)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': ['decision'],
        'intervention': {'human_lane': 9}, 'actions': [], 'horizon': 0}])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'unavailable' and result['audit']['simulations'] == []
    assert result['audit']['failure_code'] == 'intervention_has_actual_evidence'


def test_focused_pong_counterfactual_keeps_consequence_without_duplicate_score_prose():
    state = fixture(human=1, ai=7, turns=1)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'actions': ['right'], 'horizon': 1}])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'answered'
    assert 'caught: +3 points' in result['answer']
    assert len(result['answer'].split()) < 65
    assert 'Catch points change by' not in result['answer']


def test_action_already_present_in_selected_reason_is_not_repeated():
    state = pong.initial_state(730100, 2)
    decision = pong.decide(state)
    assert 'move ' + decision['action'] in decision['reason_en']
    selected = plan(state, ids=('system:ai_action', 'system:ai_reason'))
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'answered'
    assert result['answer'] == 'Task 2 · Turn 0\n\n' + decision['reason_en']
    assert result['evidence_ids'] == ['system:ai_action', 'system:ai_reason']


def test_pong_equal_window_comparison_states_no_score_advantage_once():
    state = fixture(human=2, ai=6, turns=4)
    selected = plan(state, intents=[{'kind': 'counterfactual', 'subject': 'human',
        'purpose': 'comparison', 'evidence_ids': [], 'actions': [action], 'horizon': 1}
        for action in ('right', 'wait')])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'answered'
    assert 'no score advantage' in result['answer']
    assert result['answer'].count('This window') == 1
    assert 'Only this window' not in result['answer']


def test_kitchen_catalog_uses_facing_station_labels_and_offers_no_remote_discard():
    kitchen = get_engine('kitchen')
    state = kitchen.initial_state(730100, 2)
    interactions = []
    for _ in range(24):
        decision = kitchen.decide(state)
        evidence = _catalog(kitchen, state, decision, [])
        suggestion = evidence['system:human_advice']['action']
        assert suggestion in kitchen.legal_actions(state)
        if suggestion == 'interact':
            interaction = kitchen.public_state(state)['interaction']
            assert interaction['available']
            assert interaction['label_en'] in evidence['system:human_advice']['en']
            assert interaction['label_zh'] in evidence['system:human_advice']['zh']
            interactions.append(interaction['station'])
        state = kitchen.step(state, suggestion, decision)
    first_protein = kitchen.RECIPES[state['orders'][0]['recipe']]['protein']
    assert {first_protein, 'prep', 'handoff'} <= set(interactions)
    captured = []
    selected = plan(state, ids=('system:human_advice',))
    result = explainer_for_plan(selected, captured).answer(kitchen, state, kitchen.decide(state), 'What can I do?')
    assert result['status'] == 'answered'
    assert 'discard' not in captured[0]['allowed_action_ids']


def simultaneous_ball_fixture():
    # Both large balls are individually reachable, but only t4 shares the AI's
    # small-ball lane. Four different large contacts cannot all be covered.
    return pong._new_state([[pong._ball('t3', 'cooperative', [2, 8], 4),
                            pong._ball('t4', 'cooperative', [0, 6], 4),
                            pong._ball('s11', 'ordinary', [0], 4),
                            pong._ball('s12', 'ordinary', [6], 4)]],
                           seed=99, task=2, human=0, ai=6)


@pytest.mark.parametrize('language', ['en', 'zh'])
@pytest.mark.parametrize('ball,selected', [('t3', False), ('t4', True)])
def test_named_ball_assignment_distinguishes_actual_plan_from_individual_reachability(language, ball, selected):
    state = simultaneous_ball_fixture()
    original = deepcopy(state)
    actual = {assignment['ball_id']: assignment for assignment in pong.decide(state)['assignments']}
    assert ('t3' in actual) is False
    assert actual['t4']['ai_contact'] == 6 and actual['t4']['human_contact'] == 0
    chosen = plan(state, language=language, intents=[{'kind': 'facts', 'subject': 'shared',
        'purpose': 'assignment', 'object_id': ball, 'evidence_ids': ['ball_plan:' + ball]}])
    captured = []
    result = ask(explainer_for_plan(chosen, captured), state,
                 question=f'For {ball}, which lane should I cover, and which will you cover?')
    assert result['status'] == 'answered'
    assert result['evidence_ids'] == ['ball_plan:' + ball]
    fact = next(row for row in captured[0]['evidence'] if row['id'] == 'ball_plan:' + ball)
    assert fact['subject'] == 'shared' and fact['purpose'] == 'assignment' and fact['object_id'] == ball
    assert (fact['plan_status'] != 'not_selected') is selected
    if language == 'en':
        assert 'I cover lane 7; you cover lane 1' in result['answer']
        assert 'I cover lane 9' not in result['answer']
        if not selected:
            assert 'not selected' in result['answer'] and 'cannot catch both' in result['answer']
    elif not selected:
        assert '没有选择t3' in result['answer'] and '选择接t4' in result['answer']
    assert state == original


@pytest.mark.parametrize('object_id,ids,error', [
    ('t3', ['ball:t3', 'comparison:1'], 'missing_ball_assignment_evidence'),
    ('t3', ['ball_plan:t4'], 'missing_ball_assignment_evidence'),
    (None, ['ball_plan:t3'], 'missing_ball_assignment_evidence'),
    ('t3', ['ball_plan:t3', 'comparison:1'], 'isolated_reachability_is_not_assignment'),
])
def test_assignment_intents_require_the_named_balls_actual_plan_fact(object_id, ids, error):
    state = simultaneous_ball_fixture()
    selected = plan(state, intents=[{'kind': 'facts', 'subject': 'shared',
        'purpose': 'assignment', 'object_id': object_id, 'evidence_ids': ids}])
    result = ask(explainer_for_plan(selected), state)
    assert result['status'] == 'unavailable'
    assert result['audit']['failure_code'] == error
    assert result['audit']['repair_attempt']['failure_code'] == error


def test_invalid_assignment_plan_gets_one_semantic_repair_with_exact_plan_evidence():
    state = simultaneous_ball_fixture()
    bad = plan(state, intents=[{'kind': 'facts', 'subject': 'shared', 'purpose': 'assignment',
        'object_id': 't3', 'evidence_ids': ['comparison:1']}])
    good = deepcopy(bad)
    good['intents'][0]['evidence_ids'] = ['ball_plan:t3']
    instance = explainer_for_plan(good)
    requests = []
    def request(payload):
        requests.append(deepcopy(payload))
        return deepcopy(bad if len(requests) == 1 else good), {}
    instance._request_plan = request
    result = ask(instance, state)
    assert result['status'] == 'answered'
    assert len(requests) == 2 and result['audit']['repair_attempt']['successful']
    assert 'not selected' in result['answer']


def test_nearest_ball_preset_answers_distance_arrival_and_actual_coordination_separately():
    state = simultaneous_ball_fixture()
    selected = plan(state, intents=[{'kind': 'facts', 'subject': 'human', 'purpose': 'observation',
        'evidence_ids': ['human_nearest_ball']}, {'kind': 'facts', 'subject': 'human', 'purpose': 'advice',
        'evidence_ids': ['system:human_advice', 'assignment:0']}])
    result = ask(explainer_for_plan(selected), state,
                 question='Which ball am I closest to, how many turns until it arrives, and how can I coordinate with you?')
    assert result['status'] == 'answered'
    assert 's11' in result['answer'] and '0 moves away' in result['answer']
    assert 'arriving in 4 turns' in result['answer']
    assert 'assigns it to you' in result['answer']
    assert 'wait' in result['answer'] and 't4' in result['answer']


@pytest.mark.parametrize('language',('en','zh'))
@pytest.mark.parametrize('order',(('system:ai_reason','arrival_payoff:4'),('arrival_payoff:4','system:ai_reason')))
def test_pong_payoff_is_separate_from_short_reason_when_explicitly_requested(language,order):
    state = simultaneous_ball_fixture()
    decision = pong.decide(state)
    selected = plan(state,language=language,ids=order)
    result = ask(explainer_for_plan(selected),state,question='为什么这样接球，一共能得几分？' if language=='zh' else 'Why choose this catch, and how many points can it earn?')
    assert result['status'] == 'answered'
    assert decision['reason_'+language] in result['answer']
    assert result['evidence_ids'] == list(order)
    assert result['answer'].count('合计5分' if language=='zh' else '= 5 points') == 1
    assert ('合计' if language=='zh' else 'points') not in decision['reason_'+language]


def test_current_pong_recorded_cases_reproduce_from_their_builder():
    from scripts.build_rolling_pong_qa_cases import build
    recorded = [c for c in json.loads((ROOT/'configs/study_v3_qa_cases.json').read_text())['cases'] if c.get('domain') == 'pong']
    assert recorded == build()
    assert len(recorded) == 80


def kitchen_production_trajectory(target):
    """Reproduce the production test's legal tomato-first inputs, no saved identity."""
    kitchen = get_engine('kitchen')
    state = kitchen.initial_state(2000, 2)
    supplied = False
    while state['turn'] < target:
        held = state['human']['holding']
        if supplied:
            action = kitchen.human_advisor(state)
        elif held is None:
            action = kitchen._approach(state, 'human', 'tomato')
        elif held['stage'] == 'raw':
            action = kitchen._approach(state, 'human', 'prep')
        elif state['handoff'] is None:
            action = kitchen._approach(state, 'human', 'handoff')
        else:
            action = 'wait'
        nxt = kitchen.step(state, action)
        if held and held['ingredient'] == 'tomato' and held['stage'] == 'prepared' and nxt['human']['holding'] is None:
            supplied = True
        state = nxt
    return state


@pytest.mark.parametrize('language', ['en', 'zh'])
def test_kitchen_counterfactual_time_penalty_is_points_not_negative_completed_orders(language):
    kitchen = get_engine('kitchen')
    state = kitchen_production_trajectory(168)
    before = deepcopy(state)
    selected = plan(state, language=language, intents=[{'kind':'counterfactual', 'subject':'human',
        'purpose':'comparison', 'evidence_ids':[], 'actions':['wait','wait'], 'horizon':2}])
    result = explainer_for_plan(selected).answer(kitchen, state, kitchen.decide(state), 'If I wait for the next two turns, what changes?')
    assert result['status'] == 'answered'
    simulation = result['audit']['simulations'][0]
    assert simulation['raw_score_delta'] == simulation['task_score_delta'] == -2
    assert simulation['completed_orders_delta'] == 0
    assert ('Task score changes by -2 points' if language == 'en' else '任务分数变化-2分') in result['answer']
    assert ('completed orders change by 0' if language == 'en' else '完成订单数变化0') in result['answer']
    assert result['answer'].count('-2') == 1
    assert state == before


@pytest.mark.parametrize('wait_first,points', [(False,29),(True,28)])
@pytest.mark.parametrize('language', ['en','zh'])
def test_kitchen_delivery_counterfactual_counts_real_completed_orders_independently_of_net_points(wait_first, points, language):
    from domains.kitchen.build_qa_cases import regular_trace
    kitchen = get_engine('kitchen')
    state = next(s for s in regular_trace() if s['human']['holding'] and s['human']['holding']['stage'] == 'plated'
                 and (kitchen._front(s['human']) or {}).get('id') == 'serve')
    actions = ['wait','interact'] if wait_first else ['interact']
    selected = plan(state, language=language, intents=[{'kind':'counterfactual','subject':'human','purpose':'comparison',
        'evidence_ids':[],'actions':actions,'horizon':len(actions)}])
    result = explainer_for_plan(selected).answer(kitchen,state,kitchen.decide(state),'What if I serve?')
    assert result['status'] == 'answered'
    sim = result['audit']['simulations'][0]
    assert sim['raw_score_delta'] == sim['task_score_delta'] == points
    assert sim['completed_orders_delta'] == 1
    assert sum(e['type']=='served' for e in sim['events']) == 1
    assert (f'Task score changes by {points} points' if language=='en' else f'任务分数变化{points}分') in result['answer']
    assert ('completed orders change by 1' if language=='en' else '完成订单数变化1') in result['answer']


@pytest.mark.parametrize('domain,seed,task,turn',[('pong',731100,2,0),('kitchen',2000,1,10),('kitchen',2000,2,None)])
@pytest.mark.parametrize('language',['en','zh'])
def test_actual_wait_reason_synonyms_render_once_without_question_keyword_binding(domain,seed,task,turn,language):
    module = get_engine(domain)
    state = module.initial_state(seed,task)
    if turn is None:
        # Select the late-task waiting situation by behavior rather than an
        # old cooking-speed-dependent frame number.
        while not state['terminal']:
            if state['metrics']['completed_orders'] == 4 and module.decide(state)['action'] == 'wait':
                break
            state = module.step(state, module.human_advisor(state))
        assert not state['terminal']
        turn = state['turn']
    if domain == 'pong':
        state = fixture(human=2, ai=6, turns=4)
    while state['turn'] < turn:
        state = module.step(state,module.human_advisor(state))
    decision = module.decide(state)
    assert decision['action'] == 'wait'
    selected = plan(state,language=language,ids=['system:ai_action','system:ai_reason'])
    result = explainer_for_plan(selected).answer(module,state,decision,'What were you trying to do and why?')
    assert result['status'] == 'answered'
    assert result['answer'].split('\n\n',1)[1] == decision['reason_'+language]
    assert result['evidence_ids'] == ['system:ai_action','system:ai_reason']


def test_waiting_destination_reason_does_not_hide_actual_movement():
    from study_v3.qa import _reason_states_action
    kitchen = get_engine('kitchen')
    state = kitchen.initial_state(2000,2)
    decision = kitchen.decide(state)
    decision.update(action='left', reason_en='I am waiting near the handoff counter.', reason_zh='我在交接台附近等待。')
    assert not _reason_states_action(kitchen,state,decision,'en')
    assert not _reason_states_action(kitchen,state,decision,'zh')


@pytest.mark.parametrize('domain,turn',[('warehouse',12),('kitchen',168)])
def test_human_advice_can_use_real_ai_reason_as_coordination_context_but_not_ai_action(domain,turn):
    module = get_engine(domain)
    state = kitchen_production_trajectory(turn) if domain=='kitchen' else module.initial_state(1000,2)
    while state['turn'] < turn:
        state = module.step(state,module.human_advisor(state))
    selected = plan(state,language='zh',intents=[{'kind':'facts','subject':'human','purpose':'advice',
        'evidence_ids':['system:human_advice','system:ai_reason']}])
    result = explainer_for_plan(selected).answer(module,state,module.decide(state),'如果我现在想与你配合，下一步应该做什么？为什么？')
    assert result['status'] == 'answered'
    assert module.decide(state)['reason_zh'] in result['answer']
    selected['intents'][0]['evidence_ids'].append('system:ai_action')
    assert explainer_for_plan(selected).answer(module,state,module.decide(state),'How can I help?')['status']=='unavailable'


@pytest.mark.parametrize('language',['en','zh'])
def test_one_step_human_comparison_explains_visible_deadline_without_extending_window(language):
    state = pong._new_state([[pong._ball('team', 'cooperative', [0, 6], 2),
                             pong._ball('s9', 'ordinary', [0], 2),
                             pong._ball('mine', 'ordinary', [6], 2)]],
                            seed=123, task=2, human=2, ai=6)
    intents = [{'kind':'counterfactual','subject':'human','purpose':'comparison','evidence_ids':[],
        'actions':[action],'horizon':1} for action in ['left','wait']]
    intents.append({'kind':'facts','subject':'human','purpose':'observation','evidence_ids':['human_catch_deadline']})
    result = ask(explainer_for_plan(plan(state,language=language,intents=intents)),state,question='Why is that better than waiting here?')
    assert result['status'] == 'answered'
    assert [s['steps_completed'] for s in result['audit']['simulations']] == [1,1]
    assert [s['raw_score_delta'] for s in result['audit']['simulations']] == [0,0]
    assert 's9' in result['answer']
    assert ('waiting now would miss' if language=='en' else '如果先等待，就会错过') in result['answer']
    assert ('This window shows no score advantage' if language=='en' else '在这个窗口内，两种选择没有得分优势') in result['answer']


@pytest.mark.parametrize('language',['en','zh'])
def test_old_kitchen_slot_event_uses_exact_metadata_in_historical_evidence_without_rewriting_history(language):
    kitchen = get_engine('kitchen')
    saved = kitchen_production_trajectory(15)
    index,event = next((i,event) for i,event in enumerate(saved['events']) if event['type']=='item_placed' and event.get('station')=='ai_raw')
    assert event['slot']==0
    event.update(en='AI put prepared tomato on two ingredient slots.',zh='AI放下了备好的番茄（双槽原料暂存台）。')
    history=[kitchen.public_state(saved)]
    before=deepcopy(history)
    current=kitchen.step(saved,'wait')
    identifier=f'history:task2:turn15:event{index}'
    historical_fact=_catalog(kitchen,current,kitchen.decide(current),history)[identifier]
    assert historical_fact['purpose']=='observation'
    selected=plan(current,language=language,intents=[{'kind':'facts','subject':'ai','purpose':'observation','evidence_ids':[identifier]}])
    result=explainer_for_plan(selected).answer(kitchen,current,kitchen.decide(current),'Where did you put that tomato at turn 15?',language,[],history)
    assert result['status']=='answered'
    assert ('ingredient slot 1' if language=='en' else '原料槽1') in result['answer']
    assert 'two ingredient slots' not in result['answer'] and '双槽原料暂存台' not in result['answer']
    assert history==before and saved['events'][index]['en']=='AI put prepared tomato on two ingredient slots.'


def test_historical_event_action_misclassification_requires_semantic_repair_not_validation_bypass():
    kitchen=get_engine('kitchen')
    earlier=kitchen_production_trajectory(15)
    current=kitchen.step(earlier,'wait')
    index=next(i for i,e in enumerate(earlier['events']) if e['type']=='item_placed' and e.get('station')=='ai_raw')
    bad=plan(current,intents=[{'kind':'facts','subject':'ai','purpose':'action',
        'evidence_ids':[f'history:task2:turn15:event{index}']}])
    bad['binding']['turn']=15
    good=deepcopy(bad);good['intents'][0]['purpose']='observation'
    service=explainer_for_plan(good);calls=[]
    def request(payload):
        calls.append(deepcopy(payload))
        return deepcopy(bad if len(calls)==1 else good),{}
    service._request_plan=request
    result=service.answer(kitchen,current,kitchen.decide(current),'At turn 15, where did you put the prepared tomato?',
        'en',[],[kitchen.public_state(earlier)])
    assert result['status']=='answered' and 'ingredient slot 1' in result['answer']
    assert result['audit']['repair_attempt']['failure_code']=='missing_subject_evidence'
    assert 'Historical observed events and positions use purpose=observation' in calls[1]['repair_request']['error']


def test_provider_extra_top_level_field_requires_one_validated_model_repair():
    state=fixture()
    good=plan(state,intents=[{'kind':'facts','subject':'ai','purpose':'reason','evidence_ids':['system:ai_reason']}])
    bad=dict(deepcopy(good),type='json_object')
    service=explainer_for_plan(good);requests=[]
    def request(payload):
        requests.append(deepcopy(payload))
        return deepcopy(bad if len(requests)==1 else good),{}
    service._request_plan=request
    result=ask(service,state,question='At this selected earlier turn, what were you trying to do and why?')
    assert result['status']=='answered' and len(requests)==2
    assert result['audit']['repair_attempt']['failure_code']=='unexpected_plan_fields'
    assert result['audit']['repair_attempt']['original_plan']['type']=='json_object'
    assert result['audit']['plan']==good
    assert 'type' in requests[1]['repair_request']['previous_plan']
    # No local field stripping and no second repair: repeating the invalid
    # response remains unavailable, even if all the selected fact IDs are valid.
    service=explainer_for_plan(bad,requests:=[])
    rejected=ask(service,state)
    assert rejected['status']=='unavailable' and len(requests)==2
    assert rejected['audit']['failure_code']=='unexpected_plan_fields'
