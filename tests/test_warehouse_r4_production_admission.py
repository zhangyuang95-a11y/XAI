from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from backend.training import warehouse_r4_explanation_audit as explanation
from backend.training import warehouse_r4_conflict_scene_selection as scene_selector
from backend.training import warehouse_r4_production_admission as admission
from backend.training import warehouse_r4_training_ledger as training_ledger


def h(value) -> str:
    return sha256(str(value).encode()).hexdigest()


def test_admission_binds_the_current_high_conflict_scene_producer():
    assert admission.SCENE_VERSION == scene_selector.VERSION


def material():
    required = {"train": 512, "calibration": 100, "validation": 50,
                "extraction": 100, "explanation_test": 100,
                "final_test": 100, "play": 12}
    serial = 0; splits = {}
    for name, count in required.items():
        rows = []
        for index in range(count):
            serial += 1
            rows.append({"id": f"{name}_{index:04d}", "fingerprint": h((name, index)),
                         "snapshot": {"state": {"frame": 0}}})
        splits[name] = rows
    scenarios = {"version": "warehouse-native-physical-splits-v1",
                 "counts": required, "splits": splits}
    context = {key: h(key) for key in (
        "actor_sha256", "actor_parameters_sha256", "actor_scenario_manifest_sha256",
        "training_actor_sha256", "protocol_sha256", "runtime_signature", "program_sha256",
        "explainer_signature", "selected_scenes_sha256", "selected_scenes_file_sha256",
        "question_bank_sha256", "question_bank_signature", "scenario_manifest_semantic_sha256",
        "scenario_manifest_file_sha256", "validation_entries_sha256")}
    context["scenario_manifest_semantic_sha256"] = admission.digest(scenarios)
    context["actor_scenario_manifest_sha256"] = context["scenario_manifest_semantic_sha256"]
    context["validation_entries_sha256"] = admission.digest(splits["validation"])

    def scene(index):
        return {"id": f"selected_{index}", "seed": 700000 + index,
                "fingerprint": h(("selected", index)), "task_signature": h(("task", index)),
                "observed_state_signature": h(("observed", index)),
                "snapshot": {"state": {"frame": 0}}}
    scenes = [scene(index) for index in range(7)]
    balance = {"X_mean_initial_workload": 20, "Y_mean_initial_workload": 20,
        "workload_relative_difference": 0.,
        "X_mean_conflict_opportunity": .30, "Y_mean_conflict_opportunity": .30,
        "conflict_relative_difference": 0.}
    selection = {"version": admission.SCENE_VERSION, "actor_sha256": context["actor_sha256"],
        "source_scenario_manifest_sha256": context["scenario_manifest_file_sha256"],
        "practice": scenes[0], "X": scenes[1:4], "Y": scenes[4:7],
        "pairs": [[scenes[index]["id"], scenes[index + 3]["id"]] for index in range(1, 4)],
        "balance": balance, "selection_score": 0.1}
    context["selected_scenes_sha256"] = admission.digest(selection)

    replay = [{"id": f"q{index}", "answer_match": True, "evidence_match": True,
               "source_unchanged": True} for index in range(8)]
    categories = {name: {"count": 4, "independent_scenarios": 4, "distinct_outcomes": 4}
                  for name in ("next_action", "wait_three")}
    question_checks = {"passed": True, "categories": categories,
                       "independent_replay": replay, "formal_ready": False}
    items = [{"id": f"q{index}", "scenario_id": f"question_scene_{index}",
              "kind": "next_action" if index < 4 else "wait_three"} for index in range(8)]
    question = {"version": admission.QUESTION_PAYLOAD_VERSION, "test_fixture": False,
        "actor_sha256": context["actor_sha256"], "protocol_sha256": context["protocol_sha256"],
        "runtime_signature": context["runtime_signature"],
        "source_bank_signature": context["question_bank_signature"],
        "checks": question_checks, "items": items}
    context["question_bank_sha256"] = admission.digest(question)

    ledger = {"version": admission.TRAINING_LEDGER_VERSION, "status": "complete",
        "test_fixture": False, "formal_ready": False, "admission_eligible": True,
        "failure_reason": None,
        "source_r3": {"actor_sha256": admission.R3_ACTOR_SHA256},
        "maximum_additional_joint_steps": 1_000_000,
        "total_actual_additional_joint_steps": 50_000,
        "remaining_additional_joint_steps": 950_000,
        "attempts": [{"id": "selected"}],
        "selection": {"training_actor_sha256": context["training_actor_sha256"],
            "active_policy_report_sha256": h("active-report"),
            "earliest_full_pass": True},
        "invariants": {"parent_dag_valid": True, "fresh_steps_counted_once": True}}
    baseline_summary = {"episodes": 300, "action_override_count": 0,
        "policy_action_equality_rate": 1.0, "environment_steps": 100,
        "active_non_wait_rate": .70, "productive_action_rate": .50,
        "full_battery_non_charger_wait_rate": .10,
        "first_productive_latency_median": 5., "first_productive_latency_p90": 10.,
        "mean_ai_deliveries": 2., "ai_delivery_share": .30,
        "no_task_progress_streak_p95": 25., "collision_cancellation_rate": .05,
        "mean_longest_collision_streak": 2., "static_wall_command_rate": 0.,
        "ai_shutdown_count": 0}
    candidate_summary = {**baseline_summary,
        "active_non_wait_rate": .90, "productive_action_rate": .70,
        "full_battery_non_charger_wait_rate": .01,
        "first_productive_latency_median": 1., "first_productive_latency_p90": 4.,
        "mean_ai_deliveries": 3.2, "ai_delivery_share": .40,
        "no_task_progress_streak_p95": 15., "collision_cancellation_rate": .06}
    actor_result = lambda actor_sha, summary: {"actor": {"artifact_sha256": actor_sha,
        "runtime_action_override": False}, "partners": list(admission.PARTNERS),
        "validation_entries_sha256": context["validation_entries_sha256"],
        "summary": deepcopy(summary)}
    absolute = {key: True for key in admission.ABSOLUTE_CHECKS}
    relative = {key: True for key in admission.RELATIVE_CHECKS}
    active = {"version": admission.ACTIVE_VERSION, "status": "passed", "selected": True,
        "absolute_checks": absolute, "relative_checks": relative,
        "relative_checks_passed": 5,
        "absolute_gate": deepcopy(admission.ABSOLUTE_GATE),
        "relative_gate": deepcopy(admission.RELATIVE_GATE),
        "baseline": actor_result(admission.R3_ACTOR_SHA256, baseline_summary),
        "candidate": actor_result(context["training_actor_sha256"], candidate_summary),
        "scenario_manifest": {"sha256": context["scenario_manifest_file_sha256"],
            "validation_entries_sha256": context["validation_entries_sha256"]},
        "definitions": {"fixture": "test"}}
    runtime = {"version": admission.RUNTIME_VERSION, "status": "runtime_components_verified",
        "candidate_selected_by_r4_gate": True, "formal_ready": False,
        "actor": {"sha256": context["actor_sha256"],
                  "parameters_sha256": context["actor_parameters_sha256"]},
        "program": {"sha256": context["program_sha256"], "feature_names_equal_actor": True,
                    "native_source_actor_sha256": context["actor_sha256"],
                    "source_training_actor_sha256": context["training_actor_sha256"],
                    "final_rcpd_binding_sha256": h("final-rcpd-binding")},
        "protocol": {"sha256": context["protocol_sha256"]},
        "evaluation": {"sha256": h("active-report"), "status": "passed"},
        "final_rcpd": {"version": admission.FINAL_RCPD_VERSION,
            "path": "/tmp/final_rcpd/report.json", "sha256": h("final-rcpd-report"),
            "status": "passed", "binding_sha256": h("final-rcpd-binding"),
            "source_training_actor_path": "/tmp/final_rcpd/actor.npz",
            "source_training_actor_sha256": context["training_actor_sha256"],
            "scenario_manifest_path": "/tmp/final_rcpd/scenarios.json",
            "scenario_manifest_sha256": context["scenario_manifest_file_sha256"],
            "program_source_sha256": h("final-rcpd-program"), "candidate_count": 25,
            "ppo_joint_steps": 0, "optimizer_updates": 0,
            "program_feedback_into_actor": False},
        "runtime": {"signature": context["runtime_signature"], "load_verified": True,
                    "action_masks": False, "post_policy_overrides": 0,
                    "actual_observed197_action_parity": True,
                    "actual_step_policy_equals_submitted": True},
        "explainer": {"signature": context["explainer_signature"], "load_verified": True,
                      "independent_r4_explanation_acceptance_pending": True},
        "numpy_parity": {"observations": 64, "maximum_absolute_logit_error": 0.0,
                         "deterministic_actions_equal": True}}
    scene_rows = []
    for value in scenes[1:]:
        geometry_checks = {
            "route_overlap_band": True,
            "shared_bottleneck_or_intersection": True,
            "opposing_and_same_direction_opportunities": True,
        }
        dynamic_checks = {key: True for key in admission.SCENE_DYNAMIC_CHECKS}
        scene_rows.append({"id": value["id"], "fingerprint": value["fingerprint"],
            "geometry": {"passed": True, "checks": geometry_checks,
                        "shortest_task_route_shared_edge_ratio_min": .30,
                        "shortest_task_route_shared_edge_ratio_max": .50,
                        "shared_bridge_edges": [], "shared_intersections": [[2, 2]],
                        "same_direction_mission_edges": 1, "opposing_mission_edges": 1,
                        "initial_joint_work_steps": 20},
            "dynamic": {"passed": True, "checks": dynamic_checks,
                        "conflict_opportunity_fraction": .30,
                        "player_risky_action_mass": .15,
                        "collision_cancellation_fraction": .10,
                        "collision_recovery_within_10_rate": .95,
                        "compatible_reference_mean_deliveries": 5.,
                        "simple_profile_mean_deliveries": {
                            name: 2. for name in admission.SCENE_SIMPLE_BASELINES},
                        "best_simple_profile": "right_of_way",
                        "best_simple_mean_deliveries": 2.,
                        "reference_absolute_delivery_gap": 3.,
                        "reference_relative_delivery_gap": .60,
                        "max_consecutive_collisions": 2,
                        "max_no_progress_streak": 20,
                        "actor_submission_frames": 100, "actor_action_override_frames": 0}})
    conflicts = {"version": admission.SCENE_VERSION, "status": "accepted_six_scene_selection",
        "participant_data_read": False, "final_test_rollouts": 0,
        "actor": {"sha256": context["actor_sha256"], "deterministic": True,
                  "post_policy_overrides": 0},
        "actor_protocol": {"semantic_sha256": context["protocol_sha256"]},
        "source_scenario_manifest_sha256": context["scenario_manifest_file_sha256"],
        "actor_action_override_frames": 0, "actor_submission_frames": 600,
        "geometry_evaluated": 60, "dynamic_evaluated": 6, "dynamic_passed": 6,
        "scene_metrics": scene_rows,
        "selection": {"pairs": selection["pairs"], "balance": balance,
                      "all_dynamic_pass_count": 6},
        "deployment_runtime_validation": {"passed": True,
            "runtime_signature": context["runtime_signature"], "actor_sha256": context["actor_sha256"],
            "protocol_sha256": context["protocol_sha256"],
            "scenes": [{"id": f"play_{index:04d}", "seed": 800000 + index,
                "frame": 0, "expected_fingerprint": h(("play", index)),
                "actual_fingerprint": h(("play", index)), "fingerprint_equal": True,
                "public_history_valid": False, "inference_state_unchanged": True,
                "robot_2_policy_action": "UP", "step_policy_action": "UP",
                "step_submitted_action": "UP", "policy_action_submitted_unchanged": True,
                "post_policy_overrides": 0} for index in range(7)]},
        "evidence_artifacts": {"selected_scenes.json": context["selected_scenes_file_sha256"]}}
    stat = lambda fidelity=.9: {"rows": 100, "scenes": 100, "fidelity": fidelity}
    explanation_report = {"version": admission.EXPLANATION_VERSION, "status": "passed",
        "test_fixture": False, "formal_ready": False,
        "bindings": {"actor_sha256": context["actor_sha256"],
            "protocol_sha256": context["protocol_sha256"], "runtime_signature": context["runtime_signature"],
            "program_sha256": context["program_sha256"], "explainer_signature": context["explainer_signature"],
            "scenario_manifest_sha256": context["scenario_manifest_semantic_sha256"], "test_fixture": False,
            "contract_sha256": h("contract"), "producer_sources_sha256": h("sources"),
            "holdout_fingerprints_sha256": h("holdout")},
        "statistics": {"fidelity": {"overall": stat(.9), "nonwait": stat(.85),
            "by_group": {group: stat(.85) for group in admission.CRITICAL_GROUPS},
            "by_group_nonwait": {group: stat(.85) for group in admission.CRITICAL_GROUPS}},
            "intervention_direction": {"overall": stat(.85),
                "by_group": {group: stat(.85) for group in admission.CRITICAL_GROUPS}},
            "checks": {"all": True}, "passed": True, "evaluated_role": "robot_2",
            "no_effect_pairs_counted_as_success": False},
        "base_rows": 100, "branch_anchor_count": 10, "episodes": 300,
        "zero_nn_overrides": True, "source_state_unchanged": True,
        "counterfactual_isolated": True, "program_never_controls_action": True,
        "execution": {"accounting_complete": True, "pending_operation": None,
            "counts": {"base_steps": 100, "counterfactual_steps": 50,
                       "language_runtime_steps": 140,
                       "acknowledged_steps": 290, "neural_updates": 0,
                       "tree_fits": 0, "torch_loads": 0}},
        "evidence_artifacts": {name: h(name) for name in
            ("inputs.json", "ordinary_rows.jsonl", "intervention_rows.jsonl",
             "language_rows.jsonl")}}
    questionnaire = {"version": admission.QUESTION_VERSION, "status": "candidate_ready",
        "formal_ready": False, "actor_sha256": context["actor_sha256"],
        "protocol_sha256": context["protocol_sha256"], "runtime_signature": context["runtime_signature"],
        "source_bank_signature": context["question_bank_signature"],
        "payload_sha256": context["question_bank_sha256"],
        "selected_scenarios": 8, "candidate_frames": 80, "checks": question_checks}
    reports = {"training_budget": ledger, "validation_scenarios": scenarios,
        "active_policy": active, "runtime_components": runtime,
        "high_conflict_scenes": conflicts, "explanation_program": explanation_report,
        "questionnaire": questionnaire}
    hashes = {name: h(name + "-report") for name in admission.REPORT_NAMES}
    hashes["active_policy"] = h("active-report")
    paths = {name: Path("/tmp") / f"{name}.json" for name in admission.REPORT_NAMES}
    return reports, hashes, paths, context, selection, question, scenarios


def validate(monkeypatch, values):
    reports, hashes, paths, context, selection, question, scenarios = values
    monkeypatch.setattr(explanation, "read_saved_report",
                        lambda *args, **kwargs: deepcopy(reports["explanation_program"]))
    monkeypatch.setattr(admission, "_validate_active_evidence",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(admission, "_validate_final_rcpd_evidence",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(training_ledger, "read_saved_ledger",
                        lambda *args, **kwargs: deepcopy(reports["training_budget"]))
    return admission.validate_report_set(reports=reports, report_hashes=hashes,
        report_paths=paths, context=context, selection=selection,
        question=question, scenarios=scenarios)


def test_complete_real_report_contract_recomputes_all_six_gates(monkeypatch):
    result = validate(monkeypatch, material())
    assert result == {name: True for name in admission.GATE_NAMES}


@pytest.mark.parametrize("case", ["budget", "active", "runtime", "scene", "explanation", "question"])
def test_each_component_fails_closed(monkeypatch, case):
    values = material(); reports = values[0]
    if case == "budget":
        reports["training_budget"]["total_actual_additional_joint_steps"] = 1_000_001
    elif case == "active":
        # A producer-side true boolean cannot hide a metric below threshold.
        reports["active_policy"]["candidate"]["summary"]["productive_action_rate"] = .64
    elif case == "runtime":
        reports["runtime_components"]["runtime"]["post_policy_overrides"] = 1
    elif case == "scene":
        reports["high_conflict_scenes"]["scene_metrics"][0]["dynamic"]["checks"][
            "collision_recovery_at_least_90_percent"] = False
    elif case == "explanation":
        reports["explanation_program"]["statistics"]["fidelity"]["overall"]["fidelity"] = .899
    else:
        reports["questionnaire"]["checks"]["independent_replay"][0]["source_unchanged"] = False
        reports["questionnaire"]["checks"] = deepcopy(reports["questionnaire"]["checks"])
        reports["validation_scenarios"] = values[-1]
        values[5]["checks"] = reports["questionnaire"]["checks"]
    with pytest.raises(ValueError):
        validate(monkeypatch, values)


def test_diagnostic_failed_training_ledger_is_never_admissible(monkeypatch):
    values = material(); ledger_report = values[0]["training_budget"]
    ledger_report.update(
        status="failed_no_eligible_actor", admission_eligible=False,
        failure_reason="no_passing_full_paired_audit", selection=None,
    )
    with pytest.raises(ValueError, match="complete r4 PPO budget ledger"):
        validate(monkeypatch, values)


def test_explanation_summary_cannot_replace_saved_row_recalculation(monkeypatch):
    values = material(); reports = values[0]
    changed = deepcopy(reports["explanation_program"]); changed["status"] = "failed"
    monkeypatch.setattr(explanation, "read_saved_report", lambda *args, **kwargs: changed)
    monkeypatch.setattr(admission, "_validate_active_evidence",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(admission, "_validate_final_rcpd_evidence",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(training_ledger, "read_saved_ledger",
                        lambda *args, **kwargs: deepcopy(reports["training_budget"]))
    with pytest.raises(ValueError, match="recalculation"):
        admission.validate_report_set(reports=reports, report_hashes=values[1],
            report_paths=values[2], context=values[3], selection=values[4],
            question=values[5], scenarios=values[6])


def test_paired_active_summary_is_recomputed_from_episode_rows(tmp_path):
    from backend.training import warehouse_r4_active_evaluation as producer

    scenes = [{"id": f"validation_{index:04d}", "fingerprint": h(index)}
              for index in range(50)]
    reports = {}
    for label, directory in (("baseline", "r3"), ("candidate", "candidate")):
        seed = 1000
        rows = []
        for partner_index, partner in enumerate(admission.PARTNERS):
            for scene_index, scene in enumerate(scenes):
                rows.append({
                    "scenario_id": scene["id"], "scenario_fingerprint": scene["fingerprint"],
                    "partner": partner, "seed": seed + partner_index * 10_000 + scene_index,
                    "steps": 1, "active_frames": 1, "active_non_wait": 1,
                    "productive_actions": 1,
                    "action_counts": {"UP": 1, "DOWN": 0, "LEFT": 0, "RIGHT": 0, "WAIT": 0},
                    "charge_needed_frames": 0, "charge_needed_waits": 0,
                    "charge_needed_nonproductive": 0, "full_noncharger_frames": 1,
                    "full_noncharger_waits": 0, "first_productive_latency": 1,
                    "ai_deliveries": 1, "team_deliveries": 2, "ai_delivery_share": .5,
                    "collision_steps": 0, "collision_cancellations": 0,
                    "longest_collision_streak": 0, "longest_no_task_progress_streak": 0,
                    "static_wall_commands": 0, "shutdowns": 0, "ai_shutdown": 0,
                    "player_shutdown": 0, "ai_active_end": True, "policy_actions": 1,
                    "submitted_actions": 1, "action_equal": 1, "action_overrides": 0,
                    "terminal_reason": "horizon",
                })
        result = {"version": admission.ACTIVE_VERSION, "actor": {},
                  "partners": list(admission.PARTNERS), "validation_entries_sha256": h("v"),
                  "seed": seed, "summary": producer.summarize(rows), "elapsed_seconds": 1.0}
        location = tmp_path / directory
        location.mkdir()
        (location / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (location / "report.json").write_text(json.dumps(result), encoding="utf-8")
        reports[label] = result
    paired = {"baseline": reports["baseline"], "candidate": reports["candidate"]}
    path = tmp_path / "paired_report.json"
    path.write_text(json.dumps(paired), encoding="utf-8")
    admission._validate_active_evidence(paired, path, scenes)

    candidate_rows = (tmp_path / "candidate" / "episodes.jsonl").read_text().splitlines()
    changed = json.loads(candidate_rows[0]); changed["productive_actions"] = 0
    candidate_rows[0] = json.dumps(changed)
    (tmp_path / "candidate" / "episodes.jsonl").write_text(
        "\n".join(candidate_rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="summary differs"):
        admission._validate_active_evidence(paired, path, scenes)


def test_final_rcpd_receipt_reopens_semantic_evidence(monkeypatch, tmp_path):
    from backend.training import warehouse_r4_final_rcpd as final_producer

    root = tmp_path.resolve()
    monkeypatch.setattr(admission, "ROOT", root)
    evidence = root / "final_rcpd"
    evidence.mkdir()
    actor_path = evidence / "actor.npz"
    scenarios_path = evidence / "scenarios.json"
    report_path = evidence / "report.json"
    program_path = evidence / "program.json"
    for path, raw in ((actor_path, b"actor"), (scenarios_path, b"scenarios"),
                      (report_path, b"report"), (program_path, b"program")):
        path.write_bytes(raw)
    actor_sha, scenario_sha = admission.file_hash(actor_path), admission.file_hash(scenarios_path)
    report_sha, program_sha = admission.file_hash(report_path), admission.file_hash(program_path)
    binding = h("binding")
    receipt = {
        "source_training_actor_path": str(actor_path),
        "source_training_actor_sha256": actor_sha,
        "scenario_manifest_path": str(scenarios_path),
        "scenario_manifest_sha256": scenario_sha,
        "path": str(report_path), "sha256": report_sha,
        "program_source_sha256": program_sha, "binding_sha256": binding,
    }
    seen = []

    def replay(*args, **kwargs):
        seen.append((args, kwargs))
        return {"final_rcpd_binding_sha256": binding,
                "program_file_sha256": program_sha,
                "bindings": {"actor_file_sha256": actor_sha}}

    monkeypatch.setattr(final_producer, "read_saved_report", replay)
    admission._validate_final_rcpd_evidence(
        receipt, {"scenario_manifest_file_sha256": scenario_sha})
    assert len(seen) == 1 and seen[0][1]["require_passed"] is True

    report_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        admission._validate_final_rcpd_evidence(
            receipt, {"scenario_manifest_file_sha256": scenario_sha})


def test_report_set_and_paths_are_exact(monkeypatch):
    values = material(); values[0].pop("training_budget")
    with pytest.raises(ValueError, match="Exact"):
        validate(monkeypatch, values)


def test_conflict_thresholds_and_runtime_rows_are_recomputed(monkeypatch):
    values = material()
    row = values[0]["high_conflict_scenes"]["scene_metrics"][0]
    row["dynamic"]["conflict_opportunity_fraction"] = .24
    # A stale producer-side pass matrix cannot relax the frozen 25% floor.
    with pytest.raises(ValueError, match="frozen dynamic"):
        validate(monkeypatch, values)

    values = material()
    online = values[0]["high_conflict_scenes"]["deployment_runtime_validation"]
    online["scenes"][0]["policy_action_submitted_unchanged"] = False
    with pytest.raises(ValueError, match="genuine online-runtime"):
        validate(monkeypatch, values)


def test_conflict_pair_and_group_balance_are_recomputed(monkeypatch):
    values = material()
    selection = values[4]
    report = values[0]["high_conflict_scenes"]
    # Keep both copied balance objects self-consistent but inconsistent with
    # the six immutable scene metrics.
    changed = deepcopy(selection["balance"])
    changed["X_mean_initial_workload"] = 21
    selection["balance"] = changed
    report["selection"]["balance"] = deepcopy(changed)
    values[3]["selected_scenes_sha256"] = admission.digest(selection)
    with pytest.raises(ValueError, match="pairing or X/Y balance"):
        validate(monkeypatch, values)


def test_explanation_saved_reader_recomputes_rows(tmp_path):
    holdout = [h(i) for i in range(100)]
    base = [{"fingerprint": fingerprint, "partner": partner, "frame": 0,
             "after_frame": 1, "done": True, "groups": list(explanation.GROUPS),
             "action": "UP", "tree_action": "UP", "correct": True,
             "submitted_equal": True, "decision_source_unchanged": True,
             "policy_controller": "frozen_actor"}
            for fingerprint in holdout for partner in explanation.PARTNERS]
    pairs = []
    for fingerprint in holdout[:10]:
        for player_action in ("UP", "DOWN", "LEFT", "RIGHT"):
            pairs.append({"fingerprint": fingerprint, "partner": "skilled", "frame": 0,
                "groups": list(explanation.GROUPS), "player_action": player_action,
                "active": True, "wait_physical": h((fingerprint, "wait")),
                "changed_physical": h((fingerprint, player_action)),
                "wait_next_action": "UP", "changed_next_action": "RIGHT",
                "wait_next_tree": "UP", "changed_next_tree": "RIGHT",
                "physical_effect": True, "nn_changed": True, "correct": True,
                "wait_submitted_equal": True, "changed_submitted_equal": True,
                "submitted_equal": True, "source_unchanged": True,
                "policy_controller": "frozen_actor"})
    language = []
    for i in range(explanation.LANGUAGE_SCENES):
        for intent, *_ in explanation.QUESTIONS:
            for lang in ("zh", "en"):
                language.append({"fingerprint": h(i), "intent": intent, "language": lang,
                    "answer": "已确认。" if lang == "zh" else "Confirmed.",
                    "evidence_detail": "detail", "action": "UP", "tree_action": "UP",
                    "main_terms_clean": True, "sentence_count_ok": True,
                    "sentence_count": 1, "runtime_step_count": 1,
                    "source_unchanged": True, "evidence_present": True,
                    "tree_mismatch_case": False, "tree_mismatch_hidden": True})
    bindings = {"actor_sha256": h("actor"),
        "contract_sha256": explanation.digest(explanation.contract()),
        "producer_sources_sha256": explanation.digest(explanation.producer_sources()),
        "holdout_fingerprints_sha256": explanation.digest(sorted(holdout))}
    inputs = {"version": explanation.VERSION, "bindings": bindings,
        "contract": explanation.contract(), "sources": explanation.producer_sources(),
        "input_file_sha256": {}, "holdout_fingerprints": sorted(holdout),
        "formal_ready": False}
    explanation._write_json(tmp_path / "inputs.json", inputs)
    explanation._write_jsonl(tmp_path / "ordinary_rows.jsonl", base)
    explanation._write_jsonl(tmp_path / "intervention_rows.jsonl", pairs)
    explanation._write_jsonl(tmp_path / "language_rows.jsonl", language)
    stats = explanation._summarize(base, pairs, language, set(holdout))
    artifacts = {name: admission.file_hash(tmp_path / name) for name in
        ("inputs.json", "ordinary_rows.jsonl", "intervention_rows.jsonl", "language_rows.jsonl")}
    counts = {"base_steps": len(base), "counterfactual_steps": 10 * len(explanation.ACTIONS),
        "language_runtime_steps": len(language), "neural_updates": 0,
        "tree_fits": 0, "torch_loads": 0}
    counts["acknowledged_steps"] = (counts["base_steps"] + counts["counterfactual_steps"]
                                     + counts["language_runtime_steps"])
    report = {"bindings": bindings, "statistics": stats, "episodes": 300,
        "base_rows": len(base), "branch_anchor_count": 10, "zero_nn_overrides": True,
        "source_state_unchanged": True, "counterfactual_isolated": True,
        "program_never_controls_action": True,
        "execution": {"accounting_complete": True, "pending_operation": None,
                      "counts": counts},
        "evidence_artifacts": artifacts}
    explanation._write_json(tmp_path / "report.json", report)
    assert explanation.read_saved_report(tmp_path,
        expected_report_sha256=admission.file_hash(tmp_path / "report.json"),
        expected_bindings=bindings) == report
    rows = (tmp_path / "ordinary_rows.jsonl").read_text().replace('"correct":true', '"correct":false', 1)
    (tmp_path / "ordinary_rows.jsonl").write_text(rows)
    with pytest.raises(ValueError, match="artifact changed"):
        explanation.read_saved_report(tmp_path,
            expected_report_sha256=admission.file_hash(tmp_path / "report.json"),
            expected_bindings=bindings)


def test_explanation_summary_rejects_partial_episode_and_raw_term_leak(tmp_path):
    holdout = {h(i) for i in range(100)}
    base = [{"fingerprint": fingerprint, "partner": partner, "frame": 0,
             "after_frame": 1, "done": True, "groups": [], "action": "UP",
             "tree_action": "UP", "correct": True, "submitted_equal": True,
             "decision_source_unchanged": True, "policy_controller": "frozen_actor"}
            for fingerprint in holdout for partner in explanation.PARTNERS]
    with pytest.raises(ValueError, match="matrix"):
        explanation._summarize(base[:-1], [], [], holdout)

    language = [{"fingerprint": next(iter(holdout)), "intent": "reason", "language": "zh",
        "answer": "NN 选择向上。", "evidence_detail": "detail", "action": "UP",
        "tree_action": "UP", "main_terms_clean": True, "sentence_count": 1,
        "sentence_count_ok": True, "evidence_present": True, "source_unchanged": True,
        "runtime_step_count": 1, "tree_mismatch_case": False,
        "tree_mismatch_hidden": True}]
    with pytest.raises(ValueError, match="raw answer"):
        explanation._validate_language_rows(language, holdout)
