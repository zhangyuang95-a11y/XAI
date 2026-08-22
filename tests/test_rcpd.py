from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from backend.adapters.base import SemanticPolicyContext
from core.rcpd import (
    ExecutableProgram,
    OracleOutput,
    ProgramNode,
    RCPD,
    RCPDConfig,
    _Sample,
    _counterfactual_changed_training_indices,
    _is_relational_predicate,
    implementation_audit,
)


def _features(state: tuple[float, float, float]) -> dict[str, float]:
    distance, resource_margin, conflict = state
    return {
        "goal.distance": distance,
        "self.resource_margin": resource_margin,
        "other.conflict_probability": conflict,
    }


def _oracle(state: tuple[float, float, float]) -> OracleOutput:
    distance, resource_margin, conflict = state
    logits = np.asarray(
        [
            2.0 - distance + 0.4 * resource_margin - 2.5 * conflict,
            distance - 0.2 * resource_margin + 1.5 * conflict,
        ],
        dtype=float,
    )
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    actions = ("ADVANCE", "YIELD")
    return OracleOutput(
        probabilities=dict(zip(actions, probabilities)),
        q_values=dict(zip(actions, logits)),
    )


def test_relational_predicate_audit_excludes_single_agent_geometry() -> None:
    assert _is_relational_predicate("candidate.RIGHT.blocked_by_robot")
    assert _is_relational_predicate(
        "candidate.RIGHT.predicted_same_cell_conflict"
    )
    assert _is_relational_predicate("charger.occupant_has_lower_battery")
    assert _is_relational_predicate("other.lower_battery_gap")
    assert not _is_relational_predicate(
        "candidate.RIGHT.blocked_by_static_obstacle"
    )
    assert not _is_relational_predicate("charger.distance")
    assert not _is_relational_predicate("goal.distance")


def _states(seed: int, count: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    return [
        (
            float(rng.uniform(0.0, 4.0)),
            float(rng.uniform(-1.0, 1.0)),
            float(rng.uniform(0.0, 1.0)),
        )
        for _ in range(count)
    ]


def test_rcpd_is_environment_independent() -> None:
    audit = implementation_audit()
    assert audit["single_file"] is True
    assert audit["environment_independent"] is True
    assert audit["forbidden_environment_imports"] == []


def test_rcpd_fit_trace_export_and_reload(tmp_path: Path) -> None:
    training = _states(7, 240)
    validation = _states(8, 120)
    result = RCPD(
        RCPDConfig(
            max_depth=6,
            min_samples_leaf=4,
            random_seed=11,
        )
    ).fit(
        training,
        _oracle,
        _features,
        validation_states=validation,
    )

    assert result.metrics.action_fidelity >= 0.90
    assert result.metrics.extractability_loss is not None
    assert result.metrics.extractability_score is not None
    assert result.metrics.program_size > 1
    assert result.metrics.program_depth <= 6
    assert result.extraction_summary
    assert result.program.metadata["runtime_controller"] == "neural_policy_only"
    assert result.program.metadata["program_roles"] == (
        "training_regularity_signal",
        "local_explanation_audit",
    )

    probe = validation[0]
    action = result.program.predict(_features(probe))
    trace = result.program.trace(_features(probe))
    assert action in {"ADVANCE", "YIELD"}
    assert trace
    assert all(not step.feature.startswith("feature_") for step in trace)

    output = result.program.export_python(tmp_path / "distilled_policy.py")
    spec = importlib.util.spec_from_file_location("distilled_policy", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "deployed controller remains the neural policy" in (
        module.__doc__ or ""
    ).casefold()
    exported_action, exported_probabilities = module.distilled_policy(_features(probe))
    assert exported_action == action
    assert set(exported_probabilities) == {"ADVANCE", "YIELD"}
    assert isinstance(module.PROGRAM_METADATA, dict)

    json_output = result.program.save_json(tmp_path / "distilled_policy.json")
    restored = ExecutableProgram.load_json(json_output)
    assert restored.predict_proba(_features(probe)) == result.program.predict_proba(_features(probe))
    assert restored.trace(_features(probe)) == trace


def test_rcpd_rejects_anonymous_features() -> None:
    with np.testing.assert_raises_regex(ValueError, "semantic relational feature names"):
        RCPD(RCPDConfig()).fit(
            [(0.0, 0.0, 0.0)],
            _oracle,
            lambda _state: {"feature_17": 1.0},
        )


def test_required_semantic_groups_are_used_by_final_program() -> None:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(2027)
    for index in range(320):
        route_signal = float(index % 8)
        # Cargo, energy, and relational values deliberately do not determine
        # the oracle action.  An unconstrained CART may therefore omit them;
        # the semantic contract must retain and structurally include them
        # without inventing any new action labels.
        records.append(
            {
                "features": {
                    "goal.route_signal": route_signal,
                    "self.carrying_shared_task": float(index % 2),
                    "self.battery_percent": float(rng.uniform(5.0, 100.0)),
                    "other.nearest_distance": float(rng.integers(1, 12)),
                },
                "probabilities": (
                    {"A": 0.92, "B": 0.08}
                    if int(route_signal) % 2 == 0
                    else {"A": 0.08, "B": 0.92}
                ),
                "split_group": f"episode_{index // 8}",
            }
        )

    result = RCPD(
        RCPDConfig(
            max_depth=6,
            max_leaf_nodes=24,
            max_predicates=4,
            min_samples_leaf=2,
            random_seed=2027,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        split_group_provider=lambda record: record["split_group"],
        required_predicate_groups={
            "shared_task_state": ("self.carrying_shared_task",),
            "energy_state": ("self.battery_percent",),
            "multiagent_relation": ("other.nearest_distance",),
        },
    )

    used = result.program.root.used_predicates()
    assert "self.carrying_shared_task" in used
    assert "self.battery_percent" in used
    assert "other.nearest_distance" in used
    assert result.metrics.semantic_predicate_coverage_complete is True
    assert result.metrics.required_predicate_group_coverage == {
        "shared_task_state": ("self.carrying_shared_task",),
        "energy_state": ("self.battery_percent",),
        "multiagent_relation": ("other.nearest_distance",),
    }
    assert result.program.metadata[
        "semantic_predicate_coverage_complete"
    ] is True


def test_required_semantic_group_without_varying_feature_fails_visibly() -> None:
    records = [
        {
            "features": {
                "goal.distance": float(index),
                "self.carrying_shared_task": 0.0,
            },
            "probabilities": (
                {"A": 0.9, "B": 0.1}
                if index < 20
                else {"A": 0.1, "B": 0.9}
            ),
        }
        for index in range(40)
    ]

    with pytest.raises(ValueError, match="has no varying feature"):
        RCPD(
            RCPDConfig(
                max_depth=3,
                max_leaf_nodes=8,
                min_samples_leaf=2,
            )
        ).fit(
            records,
            lambda record: record["probabilities"],
            lambda record: record["features"],
            required_predicate_groups={
                "shared_task_state": ("self.carrying_shared_task",),
            },
        )


def test_changed_pair_weighting_selects_both_nn_changing_endpoints() -> None:
    samples = (
        _Sample(
            state="baseline",
            features={"self.value": 0.0},
            oracle=OracleOutput({"A": 0.9, "B": 0.1}),
            weight=1.0,
            pair_id="changed",
        ),
        _Sample(
            state="edited",
            features={"self.value": 1.0},
            oracle=OracleOutput({"A": 0.1, "B": 0.9}),
            weight=1.0,
            pair_id="changed",
        ),
        _Sample(
            state="same_left",
            features={"self.value": 2.0},
            oracle=OracleOutput({"A": 0.8, "B": 0.2}),
            weight=1.0,
            pair_id="unchanged",
        ),
        _Sample(
            state="same_right",
            features={"self.value": 3.0},
            oracle=OracleOutput({"A": 0.7, "B": 0.3}),
            weight=1.0,
            pair_id="unchanged",
        ),
    )

    selected = _counterfactual_changed_training_indices(
        samples,
        np.arange(len(samples)),
        ("A", "B"),
    )

    assert selected.tolist() == [0, 1]


def test_action_structure_auxiliary_keeps_soft_leaf_probabilities() -> None:
    result = RCPD(
        RCPDConfig(
            max_depth=3,
            min_samples_leaf=4,
            action_structure_weight=1.0,
            random_seed=19,
        )
    ).fit(
        _states(19, 160),
        _oracle,
        _features,
    )

    assert result.program.metadata["action_structure_weight"] == 1.0
    assert result.program.metadata[
        "counterfactual_changed_training_samples"
    ] == 0
    leaf_probabilities = result.program.predict_proba(
        _features(_states(20, 1)[0])
    )
    assert 0.0 < min(leaf_probabilities.values())
    assert max(leaf_probabilities.values()) < 1.0


def test_executable_program_and_export_apply_adapter_legality_features(tmp_path: Path) -> None:
    program = ExecutableProgram(
        action_names=("ADVANCE", "YIELD"),
        feature_names=("candidate.ADVANCE.legal", "candidate.YIELD.legal"),
        root=ProgramNode(probabilities=(0.9, 0.1)),
        metadata={
            "action_legality_features": {
                "ADVANCE": "candidate.ADVANCE.legal",
                "YIELD": "candidate.YIELD.legal",
            }
        },
    )
    features = {
        "candidate.ADVANCE.legal": 0.0,
        "candidate.YIELD.legal": 1.0,
    }

    assert program.predict(features) == "YIELD"
    assert program.predict_proba(features) == {"ADVANCE": 0.0, "YIELD": 1.0}

    output = program.export_python(tmp_path / "legal_program.py")
    spec = importlib.util.spec_from_file_location("legal_program", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    action, probabilities = module.distilled_policy(features)
    assert action == "YIELD"
    assert probabilities == {"ADVANCE": 0.0, "YIELD": 1.0}


def test_program_v2_trace_records_observable_exclusion_and_binding(
    tmp_path: Path,
) -> None:
    program = ExecutableProgram(
        action_names=("ADVANCE", "YIELD"),
        feature_names=(
            "candidate.ADVANCE.legal",
            "candidate.ADVANCE.blocked_by_robot",
            "candidate.YIELD.legal",
        ),
        root=ProgramNode(probabilities=(0.8, 0.2)),
        metadata={
            "action_legality_features": {
                "ADVANCE": "candidate.ADVANCE.legal",
                "YIELD": "candidate.YIELD.legal",
            },
            "action_constraint_reason_features": {
                "ADVANCE": {
                    "blocked_by_robot": (
                        "candidate.ADVANCE.blocked_by_robot"
                    )
                }
            },
        },
    )
    features = {
        "candidate.ADVANCE.legal": 0.0,
        "candidate.ADVANCE.blocked_by_robot": 1.0,
        "candidate.YIELD.legal": 1.0,
    }
    context = SemanticPolicyContext(
        features=features,
        entity_bindings={
            "candidate.ADVANCE.blocker": "agent_b",
            "nearest_agent": "agent_c",
        },
    )

    execution = program.execute(features, context)
    assert execution.action == "YIELD"
    assert execution.trace.pre_mask_distribution == {
        "ADVANCE": 0.8,
        "YIELD": 0.2,
    }
    assert execution.trace.post_mask_distribution == {
        "ADVANCE": 0.0,
        "YIELD": 1.0,
    }
    assert len(execution.trace.excluded_actions) == 1
    exclusion = execution.trace.excluded_actions[0]
    assert exclusion.action == "ADVANCE"
    assert exclusion.active_reason_features == (
        "candidate.ADVANCE.blocked_by_robot",
    )
    assert exclusion.bound_entities == {
        "candidate.ADVANCE.blocker": "agent_b"
    }
    assert execution.trace.regularization_version is True
    assert execution.trace.program_complexity == {
        "depth": 0,
        "leaves": 1,
        "predicates": 0,
    }

    output = program.export_python(tmp_path / "program_v2.py")
    spec = importlib.util.spec_from_file_location("program_v2", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exported_action, exported_probabilities, exported_trace = (
        module.distilled_policy_with_trace(features)
    )
    assert exported_action == execution.action
    assert exported_probabilities == execution.probabilities
    assert exported_trace["excluded_actions"][0][
        "active_reason_features"
    ] == ["candidate.ADVANCE.blocked_by_robot"]
    assert exported_trace["regularization_version"] is True
    assert exported_trace["program_complexity"] == {
        "depth": 0,
        "leaves": 1,
        "predicates": 0,
    }


def test_program_export_matches_in_memory_when_all_actions_are_masked(
    tmp_path: Path,
) -> None:
    program = ExecutableProgram(
        action_names=("A", "B"),
        feature_names=("candidate.A.legal", "candidate.B.legal"),
        root=ProgramNode(probabilities=(0.9, 0.1)),
        metadata={
            "action_legality_features": {
                "A": "candidate.A.legal",
                "B": "candidate.B.legal",
            }
        },
    )
    features = {"candidate.A.legal": 0.0, "candidate.B.legal": 0.0}
    expected = program.execute(features)
    output = program.export_python(tmp_path / "all_masked.py")
    spec = importlib.util.spec_from_file_location("all_masked", output)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.distilled_policy(features) == (
        expected.action,
        expected.probabilities,
    )
    assert module.distilled_policy_with_trace(features)[:2] == (
        expected.action,
        expected.probabilities,
    )


def test_program_v1_and_v2_json_are_both_loadable() -> None:
    program = ExecutableProgram(
        action_names=("A", "B"),
        feature_names=("goal.distance",),
        root=ProgramNode(probabilities=(0.4, 0.6)),
    )
    v2 = program.to_dict()
    v1 = {**v2, "format": "rcpd_executable_program_v1"}

    assert ExecutableProgram.from_dict(v2).predict({}) == "B"
    assert ExecutableProgram.from_dict(v1).predict({}) == "B"


def test_relational_extraction_reports_group_and_counterfactual_metrics() -> None:
    records = []
    for pair_index in range(30):
        for blocked, role in ((0.0, "baseline"), (1.0, "counterfactual")):
            records.append(
                {
                    "features": {
                        "candidate.A.legal": 1.0,
                        "candidate.B.legal": 1.0,
                        "candidate.A.blocked_by_robot": blocked,
                        "goal.distance": float(pair_index % 5),
                    },
                    "probabilities": (
                        {"A": 0.95, "B": 0.05}
                        if blocked == 0.0
                        else {"A": 0.05, "B": 0.95}
                    ),
                    "groups": (
                        ("ordinary",)
                        if blocked == 0.0
                        else ("occupied_progress",)
                    ),
                    "pair_id": f"pair_{pair_index}",
                    # Deliberately different: pair linkage must still keep
                    # both rows on the same side of the split.
                    "split_group": f"episode_{pair_index}_{role}",
                }
            )
    algorithm = RCPD(
        RCPDConfig(
            max_depth=3,
            max_leaf_nodes=8,
            max_predicates=4,
            min_samples_leaf=2,
            minimum_overall_fidelity_for_feedback=0.85,
            minimum_interaction_fidelity_for_feedback=0.75,
            minimum_interaction_validation_samples=8,
            regularization_lambda=0.1,
            random_seed=91,
        )
    )
    result = algorithm.fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        validation_states=records,
        action_legality_features={
            "A": "candidate.A.legal",
            "B": "candidate.B.legal",
        },
        action_constraint_reason_features={
            "A": {
                "blocked_by_robot": "candidate.A.blocked_by_robot"
            }
        },
        group_provider=lambda record: record["groups"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
        interaction_groups=("occupied_progress",),
    )

    assert "candidate.A.blocked_by_robot" in result.program.feature_names
    assert "candidate.A.blocked_by_robot" in result.program.root.used_predicates()
    assert result.metrics.action_fidelity == 1.0
    assert result.metrics.interaction_macro_fidelity == 1.0
    assert result.metrics.counterfactual_delta_error == pytest.approx(0.0)
    assert result.metrics.counterfactual_direction_fidelity == 1.0
    assert result.metrics.relational_predicate_count >= 1
    assert result.metrics.feedback_eligible is True
    assert result.program.metadata["data_split_audit"][
        "split_group_overlap"
    ] == []
    assert result.program.metadata["data_split_audit"][
        "counterfactual_pair_overlap"
    ] == []


def test_internal_validation_split_preserves_rare_pairs_and_interactions() -> None:
    records: list[dict[str, object]] = [
        {
            "features": {
                "goal.distance": float(index % 4),
                "candidate.A.blocked_by_robot": 0.0,
            },
            "probabilities": {"A": 0.9, "B": 0.1},
            "groups": ("ordinary",),
            # One large ordinary component used to consume the complete
            # validation budget before any paired evidence was selected.
            "split_group": "ordinary_episode",
            "pair_id": None,
        }
        for index in range(100)
    ]
    for pair_index in range(10):
        for blocked in (0.0, 1.0):
            records.append(
                {
                    "features": {
                        "goal.distance": 1.0,
                        "candidate.A.blocked_by_robot": blocked,
                    },
                    "probabilities": (
                        {"A": 0.9, "B": 0.1}
                        if blocked == 0.0
                        else {"A": 0.1, "B": 0.9}
                    ),
                    "groups": ("occupied_progress",),
                    "split_group": f"probe_episode_{pair_index}",
                    "pair_id": f"pair_{pair_index}",
                }
            )

    result = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=2,
            validation_fraction=0.20,
            minimum_interaction_validation_samples=4,
            random_seed=7,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        group_provider=lambda record: record["groups"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
        interaction_groups=("occupied_progress",),
    )

    assert result.metrics.counterfactual_validation_pairs > 0
    assert result.metrics.counterfactual_changed_pairs > 0
    assert (
        result.metrics.group_validation_samples["occupied_progress"]
        >= 4
    )
    audit = result.program.metadata["data_split_audit"]
    assert audit["validation_counterfactual_pairs"] > 0
    assert audit["counterfactual_pair_overlap"] == []
    assert audit["split_group_overlap"] == []


def test_counterfactual_direction_requires_both_pair_endpoints_to_match() -> None:
    records = [
        {
            "probabilities": {"A": 0.9, "B": 0.1},
            "pair_id": "pair",
        },
        {
            "probabilities": {"A": 0.1, "B": 0.9},
            "pair_id": "pair",
        },
    ]
    always_b = ExecutableProgram(
        action_names=("A", "B"),
        feature_names=("goal.distance",),
        root=ProgramNode(probabilities=(0.1, 0.9)),
    )
    metrics = RCPD().evaluate(
        always_b,
        records,
        lambda record: record["probabilities"],
        lambda _record: {"goal.distance": 1.0},
        counterfactual_pair_provider=lambda record: record["pair_id"],
    )

    assert metrics.counterfactual_validation_pairs == 1
    assert metrics.counterfactual_changed_pairs == 1
    assert metrics.counterfactual_direction_fidelity == 0.0


def test_counterfactual_direction_gate_blocks_an_explanation_wrong_teacher() -> None:
    records = []
    for pair_index in range(20):
        for role, probabilities in (
            ("baseline", {"A": 0.9, "B": 0.1}),
            ("counterfactual", {"A": 0.1, "B": 0.9}),
        ):
            records.append(
                {
                    # The intervention is deliberately absent from the
                    # program features, so no bounded tree can reproduce the
                    # NN's causal response despite seeing both labels.
                    "features": {"goal.distance": 1.0},
                    "probabilities": probabilities,
                    "pair_id": f"pair_{pair_index}",
                    "split_group": f"episode_{pair_index}_{role}",
                }
            )
    result = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=2,
            validation_fraction=0.50,
            minimum_counterfactual_direction_fidelity_for_explanation=0.75,
            minimum_counterfactual_changed_pairs_for_explanation=8,
            regularization_lambda=0.1,
            random_seed=19,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
    )

    assert result.metrics.counterfactual_changed_pairs >= 8
    assert result.metrics.counterfactual_direction_fidelity == 0.0
    assert result.metrics.feedback_eligible is True
    assert result.metrics.feedback_weight == 0.1
    assert result.metrics.explanation_eligible is False
    assert (
        "counterfactual_direction_fidelity_below_threshold"
        in result.metrics.explanation_ineligibility_reasons
    )


def test_training_feedback_and_explanation_quality_have_independent_gates() -> None:
    records = [
        {
            "features": {"constant": 1.0},
            "probabilities": (
                {"A": 0.9, "B": 0.1}
                if index % 2 == 0
                else {"A": 0.1, "B": 0.9}
            ),
        }
        for index in range(40)
    ]
    result = RCPD(
        RCPDConfig(
            max_depth=1,
            max_leaf_nodes=2,
            min_samples_leaf=2,
            validation_fraction=0.5,
            regularization_lambda=0.01,
            minimum_overall_fidelity_for_feedback=0.4,
            maximum_mean_kl_for_feedback=0.5,
            minimum_overall_fidelity_for_explanation=0.85,
            maximum_mean_kl_for_explanation=0.1,
            random_seed=23,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        validation_states=records,
    )

    assert result.metrics.feedback_eligible is True
    assert result.metrics.feedback_weight == pytest.approx(0.01)
    assert result.metrics.explanation_eligible is False
    assert (
        "overall_fidelity_below_explanation_threshold"
        in result.metrics.explanation_ineligibility_reasons
    )
    assert (
        "program_fit_kl_above_explanation_threshold"
        in result.metrics.explanation_ineligibility_reasons
    )


def test_causal_feedback_gate_prevents_a_wrong_explanation_teacher() -> None:
    records = []
    for pair_index in range(20):
        for role, probabilities in (
            ("baseline", {"A": 0.9, "B": 0.1}),
            ("counterfactual", {"A": 0.1, "B": 0.9}),
        ):
            records.append(
                {
                    "features": {"goal.distance": 1.0},
                    "probabilities": probabilities,
                    "pair_id": f"pair_{pair_index}",
                    "split_group": f"episode_{pair_index}_{role}",
                }
            )
    result = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=2,
            validation_fraction=0.50,
            minimum_counterfactual_direction_fidelity_for_explanation=0.75,
            minimum_counterfactual_changed_pairs_for_explanation=8,
            require_explanation_eligibility_for_feedback=True,
            regularization_lambda=0.1,
            random_seed=19,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
    )

    assert result.metrics.explanation_eligible is False
    assert result.metrics.feedback_eligible is False
    assert result.metrics.feedback_weight == 0.0
    assert (
        "explanation_gate:counterfactual_direction_fidelity_below_threshold"
        in result.metrics.feedback_ineligibility_reasons
    )


def test_causal_gate_only_withholds_counterfactual_feedback_rows() -> None:
    records = []
    for pair_index in range(20):
        for role, probabilities in (
            ("baseline", {"A": 0.9, "B": 0.1}),
            ("counterfactual", {"A": 0.1, "B": 0.9}),
        ):
            records.append(
                {
                    "features": {"goal.distance": 1.0},
                    "probabilities": probabilities,
                    "pair_id": f"pair_{pair_index}",
                    "split_group": f"episode_{pair_index}_{role}",
                }
            )
    algorithm = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=2,
            validation_fraction=0.50,
            minimum_counterfactual_direction_fidelity_for_explanation=0.75,
            minimum_counterfactual_changed_pairs_for_explanation=8,
            require_explanation_eligibility_for_counterfactual_feedback=True,
            regularization_lambda=0.1,
            random_seed=19,
        )
    )
    result = algorithm.maybe_extract(
        1,
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
        force=True,
    )
    assert result is not None
    assert result.metrics.feedback_eligible is True
    assert result.metrics.explanation_eligible is False

    features = [{"goal.distance": 1.0}, {"goal.distance": 1.0}]
    targets = algorithm.program_targets(features)
    assert targets is not None
    weights = algorithm.program_target_weights(
        features,
        actor_probabilities=targets,
        counterfactual_mask=(False, True),
    )

    assert weights is not None
    assert weights[0] > 0.0
    assert weights[1] == 0.0


def test_program_fit_kl_gate_blocks_a_distributionally_wrong_teacher() -> None:
    records = []
    for pair_index in range(20):
        for role, probabilities in (
            ("baseline", {"A": 0.99, "B": 0.01}),
            ("counterfactual", {"A": 0.01, "B": 0.99}),
        ):
            records.append(
                {
                    "features": {"goal.distance": 1.0},
                    "probabilities": probabilities,
                    "pair_id": f"pair_{pair_index}",
                    "split_group": f"episode_{pair_index}_{role}",
                }
            )
    result = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            min_samples_leaf=2,
            validation_fraction=0.50,
            maximum_mean_kl_for_feedback=0.0,
            regularization_lambda=0.1,
            random_seed=19,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
        split_group_provider=lambda record: record["split_group"],
    )

    assert result.metrics.mean_kl_divergence > 0.0
    assert result.metrics.feedback_eligible is False
    assert result.metrics.feedback_weight == 0.0
    assert (
        "program_fit_kl_above_threshold"
        in result.metrics.feedback_ineligibility_reasons
    )


def test_irrelevant_required_relation_is_available_but_not_forced_into_tree() -> None:
    records = [
        {
            "goal.distance": float(index % 2),
            "candidate.A.blocked_by_robot": float((index // 2) % 2),
        }
        for index in range(120)
    ]

    result = RCPD(
        RCPDConfig(
            max_depth=2,
            max_leaf_nodes=4,
            max_predicates=2,
            min_samples_leaf=4,
            random_seed=17,
        )
    ).fit(
        records,
        lambda record: (
            {"A": 0.9, "B": 0.1}
            if record["goal.distance"] == 0.0
            else {"A": 0.1, "B": 0.9}
        ),
        lambda record: record,
        action_constraint_reason_features={
            "A": {
                "blocked_by_robot": "candidate.A.blocked_by_robot"
            }
        },
        validation_states=records,
    )

    assert "candidate.A.blocked_by_robot" in result.program.feature_names
    assert result.program.root.used_predicates() == frozenset(
        {"goal.distance"}
    )


def test_paired_counterfactual_change_can_preserve_a_rare_predicate() -> None:
    records: list[dict[str, object]] = []
    for index in range(120):
        records.append(
            {
                "features": {
                    "diagnostic.ordinary_signal": float(index % 2),
                    "diagnostic.paired_signal": 0.0,
                },
                "probabilities": (
                    {"A": 0.9, "B": 0.1}
                    if index % 2 == 0
                    else {"A": 0.1, "B": 0.9}
                ),
                "pair_id": None,
            }
        )
    for pair_index in range(12):
        for value, probabilities in (
            (0.0, {"A": 0.95, "B": 0.05}),
            (1.0, {"A": 0.05, "B": 0.95}),
        ):
            records.append(
                {
                    "features": {
                        "diagnostic.ordinary_signal": 0.0,
                        "diagnostic.paired_signal": value,
                    },
                    "probabilities": probabilities,
                    "pair_id": f"pair_{pair_index}",
                }
            )

    result = RCPD(
        RCPDConfig(
            max_depth=1,
            max_leaf_nodes=2,
            max_predicates=1,
            min_samples_leaf=2,
            counterfactual_feature_selection_weight=2.0,
            random_seed=73,
        )
    ).fit(
        records,
        lambda record: record["probabilities"],
        lambda record: record["features"],
        counterfactual_pair_provider=lambda record: record["pair_id"],
    )

    assert result.program.feature_names == ("diagnostic.paired_signal",)


def test_group_reliability_zeroes_only_unreliable_interaction_feedback() -> None:
    algorithm = RCPD(
        RCPDConfig(
            minimum_target_weight=0.2,
            minimum_interaction_fidelity_for_feedback=0.75,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(101, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    assert algorithm.last_result is not None
    metrics = algorithm.last_result.metrics
    algorithm.last_result = type(algorithm.last_result)(
        program=algorithm.last_result.program,
        metrics=type(metrics)(
            **{
                **metrics.__dict__,
                "group_action_fidelity": {
                    "ordinary": 0.95,
                    "occupied_progress": 0.60,
                    "charger_competition": 0.80,
                },
            }
        ),
        extraction_summary=algorithm.last_result.extraction_summary,
    )
    features = [_features(state) for state in _states(102, 3)]
    weights = algorithm.program_target_weights(
        features,
        sample_groups=(
            ("ordinary",),
            ("occupied_progress",),
            ("charger_competition",),
        ),
    )

    assert weights is not None
    assert weights[0] > 0.0
    assert weights[1] == 0.0
    assert weights[2] > 0.0


def test_training_time_extraction_repeats_and_supplies_actor_targets() -> None:
    states = _states(21, 80)
    algorithm = RCPD(
        RCPDConfig(
            max_depth=4,
            min_samples_leaf=2,
            extraction_interval=2,
            minimum_extraction_samples=8,
            regularization_lambda=0.07,
            random_seed=23,
        )
    )
    first = algorithm.maybe_extract(
        1,
        states,
        _oracle,
        _features,
        force=False,
    )
    assert first is None
    first = algorithm.maybe_extract(
        2,
        states,
        _oracle,
        _features,
        force=False,
    )
    assert first is not None
    assert algorithm.regularization_weight == 0.07
    assert algorithm.maybe_extract(
        3,
        states,
        _oracle,
        _features,
    ) is None
    second = algorithm.maybe_extract(
        4,
        states,
        _oracle,
        _features,
    )
    assert second is not None
    assert [item["step"] for item in algorithm.extraction_history] == [
        2,
        4,
    ]

    features = [_features(state) for state in states[:8]]
    targets = algorithm.program_targets(features)
    weights = algorithm.program_target_weights(features)
    assert targets is not None
    assert weights is not None
    assert targets.shape == (8, 2)
    assert np.allclose(targets.sum(axis=1), 1.0)
    assert np.all(weights >= algorithm.config.minimum_target_weight)


def test_program_feedback_protects_confident_actor_disagreements() -> None:
    algorithm = RCPD(
        RCPDConfig(
            minimum_target_weight=0.2,
            maximum_disagreement_actor_margin=0.10,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(51, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(52, 3)]
    targets = algorithm.program_targets(features)
    assert targets is not None
    actor = targets.copy()

    # Row 0 agrees and keeps its program-confidence weight.
    # Row 1 confidently disagrees and is completely protected.
    # Row 2 weakly disagrees and may still be regularised.
    actor[1] = actor[1][::-1]
    actor[1, np.argmax(actor[1])] = 0.99
    actor[1, np.argmin(actor[1])] = 0.01
    actor[2] = actor[2][::-1]
    actor[2] = np.asarray([0.49, 0.51], dtype=np.float32)
    if np.argmax(actor[2]) == np.argmax(targets[2]):
        actor[2] = actor[2][::-1]

    weights = algorithm.program_target_weights(
        features,
        actor_probabilities=actor,
    )

    assert weights is not None
    assert weights[0] >= algorithm.config.minimum_target_weight
    assert weights[1] == 0.0
    assert weights[2] > 0.0


def test_program_feedback_can_require_nn_program_action_agreement() -> None:
    algorithm = RCPD(
        RCPDConfig(
            minimum_target_weight=0.2,
            maximum_disagreement_actor_margin=1.0,
            maximum_feedback_actor_margin=1.0,
            require_action_agreement_for_feedback=True,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(152, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(153, 2)]
    targets = algorithm.program_targets(features)
    assert targets is not None
    actor = targets.copy()
    actor[1] = actor[1][::-1]

    weights = algorithm.program_target_weights(
        features,
        actor_probabilities=actor,
    )

    assert weights is not None
    assert weights[0] >= algorithm.config.minimum_target_weight
    assert weights[1] == 0.0


def test_action_anchor_feedback_preserves_nn_distribution_and_action_order() -> None:
    algorithm = RCPD(
        RCPDConfig(
            feedback_target_mode="action_anchor",
            feedback_target_strength=0.10,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(154, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(155, 3)]
    program = algorithm.program_targets(features)
    assert program is not None
    actor = np.asarray(
        [
            [0.52, 0.48]
            if int(np.argmax(target)) == 0
            else [0.48, 0.52]
            for target in program
        ],
        dtype=np.float32,
    )

    anchored = algorithm.program_feedback_targets(
        features,
        actor_probabilities=actor,
    )

    assert anchored is not None
    expected = actor * 0.90
    expected[np.arange(len(program)), np.argmax(program, axis=1)] += 0.10
    assert np.allclose(anchored, expected)
    assert np.array_equal(np.argmax(anchored, axis=1), np.argmax(actor, axis=1))
    assert np.allclose(anchored.sum(axis=1), 1.0)


def test_program_blend_feedback_is_proximal_to_the_current_nn() -> None:
    algorithm = RCPD(
        RCPDConfig(
            feedback_target_mode="program_blend",
            feedback_target_strength=0.10,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(156, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(157, 3)]
    program = algorithm.program_targets(features)
    assert program is not None
    actor = np.asarray(
        [[0.52, 0.48], [0.48, 0.52], [0.55, 0.45]],
        dtype=np.float32,
    )

    blended = algorithm.program_feedback_targets(
        features,
        actor_probabilities=actor,
    )

    assert blended is not None
    assert np.allclose(blended, 0.90 * actor + 0.10 * program)
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_disagreement_margin_one_disables_confidence_protection() -> None:
    algorithm = RCPD(
        RCPDConfig(
            minimum_target_weight=0.2,
            maximum_disagreement_actor_margin=1.0,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(55, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(56, 4)]
    targets = algorithm.program_targets(features)
    assert targets is not None
    actor = targets[:, ::-1].copy()

    weights = algorithm.program_target_weights(
        features,
        actor_probabilities=actor,
    )

    assert weights is not None
    assert np.all(weights >= algorithm.config.minimum_target_weight)


def test_program_feedback_only_regularises_the_low_margin_nn_boundary() -> None:
    algorithm = RCPD(
        RCPDConfig(
            minimum_target_weight=0.2,
            maximum_disagreement_actor_margin=0.10,
            maximum_feedback_actor_margin=0.10,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(57, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(58, 2)]
    targets = algorithm.program_targets(features)
    assert targets is not None

    actor = np.zeros_like(targets)
    for index, target in enumerate(targets):
        selected = int(np.argmax(target))
        other = 1 - selected
        if index == 0:
            # A confident agreeing NN decision must remain entirely under PPO.
            actor[index, selected] = 0.90
            actor[index, other] = 0.10
        else:
            # Only the uncertain decision boundary receives program feedback.
            actor[index, selected] = 0.53
            actor[index, other] = 0.47

    weights = algorithm.program_target_weights(
        features,
        actor_probabilities=actor,
    )

    assert weights is not None
    assert weights[0] == 0.0
    assert weights[1] >= algorithm.config.minimum_target_weight


def test_program_feedback_rejects_mismatched_actor_probabilities() -> None:
    algorithm = RCPD(RCPDConfig())
    assert algorithm.maybe_extract(
        1,
        _states(53, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(_states(54, 1)[0])]

    with pytest.raises(ValueError, match="match the program-target shape"):
        algorithm.program_target_weights(
            features,
            actor_probabilities=np.ones((2, 2), dtype=np.float32),
        )


def test_program_targets_are_sharpened_without_changing_the_action() -> None:
    algorithm = RCPD(
        RCPDConfig(
            program_target_temperature=0.5,
        )
    )
    assert algorithm.maybe_extract(
        1,
        _states(55, 80),
        _oracle,
        _features,
        force=True,
    ) is not None
    features = [_features(state) for state in _states(56, 8)]
    raw = algorithm._raw_program_targets(features)
    sharpened = algorithm.program_targets(features)

    assert raw is not None
    assert sharpened is not None
    assert np.array_equal(
        np.argmax(raw, axis=1),
        np.argmax(sharpened, axis=1),
    )
    assert np.all(
        np.max(sharpened, axis=1) >= np.max(raw, axis=1) - 1e-7
    )

    softened_algorithm = RCPD(
        RCPDConfig(
            program_target_temperature=2.0,
        )
    )
    softened_algorithm.program = algorithm.program
    softened_algorithm.regularization_weight = algorithm.regularization_weight
    softened = softened_algorithm.program_targets(features)
    assert softened is not None
    assert np.array_equal(
        np.argmax(raw, axis=1),
        np.argmax(softened, axis=1),
    )
    assert np.all(
        np.max(softened, axis=1) <= np.max(raw, axis=1) + 1e-7
    )

    hard = RCPD(
        RCPDConfig(
            program_target_temperature=0.0,
        )
    )
    hard.program = algorithm.program
    hard.regularization_weight = algorithm.regularization_weight
    hard_targets = hard.program_targets(features)
    assert hard_targets is not None
    assert np.allclose(hard_targets.sum(axis=1), 1.0)
    assert set(np.unique(hard_targets)).issubset({0.0, 1.0})

    sharpened_metrics = algorithm.evaluate(
        algorithm.program,
        _states(57, 32),
        _oracle,
        _features,
    )
    softened_metrics = softened_algorithm.evaluate(
        softened_algorithm.program,
        _states(57, 32),
        _oracle,
        _features,
    )
    assert sharpened_metrics.action_fidelity == pytest.approx(
        softened_metrics.action_fidelity
    )
    assert not np.isclose(
        sharpened_metrics.mean_kl_divergence,
        softened_metrics.mean_kl_divergence,
    )


def test_program_feedback_lambda_is_delayed_ramped_and_then_held() -> None:
    algorithm = RCPD(
        RCPDConfig(
            regularization_lambda=0.2,
            regularization_start_fraction=0.50,
            regularization_ramp_fraction=0.25,
        )
    )
    # A successful extraction activates the configured target lambda.
    result = algorithm.maybe_extract(
        1,
        _states(29, 80),
        _oracle,
        _features,
        force=True,
    )
    assert result is not None

    assert algorithm.scheduled_regularization_weight(0.49) == 0.0
    assert algorithm.scheduled_regularization_weight(0.50) == 0.0
    assert np.isclose(
        algorithm.scheduled_regularization_weight(0.625),
        0.1,
    )
    assert np.isclose(
        algorithm.scheduled_regularization_weight(0.75),
        0.2,
    )
    assert np.isclose(
        algorithm.scheduled_regularization_weight(1.0),
        0.2,
    )


def test_rcpd_resume_restores_program_but_uses_new_lambda() -> None:
    source = RCPD(
        RCPDConfig(
            regularization_lambda=0.0,
            minimum_extraction_samples=8,
        )
    )
    result = source.maybe_extract(
        17,
        _states(41, 80),
        _oracle,
        _features,
        force=True,
    )
    assert result is not None

    branch = RCPD(
        RCPDConfig(
            regularization_lambda=0.0625,
            regularization_start_fraction=0.8,
            regularization_ramp_fraction=0.05,
        )
    )
    branch.restore_training_state(source.training_state())

    assert branch.program is not None
    assert branch.last_result is not None
    assert branch.last_extract_step == 17
    assert branch.regularization_weight == 0.0625
    assert branch.program.to_dict() == source.program.to_dict()
    assert branch.scheduled_regularization_weight(0.825) == pytest.approx(
        0.03125
    )


def test_rcpd_resume_refits_when_program_bound_changes() -> None:
    source = RCPD(
        RCPDConfig(
            regularization_lambda=0.0,
            max_depth=5,
            max_leaf_nodes=16,
            minimum_extraction_samples=8,
        )
    )
    assert source.maybe_extract(
        17,
        _states(43, 80),
        _oracle,
        _features,
        force=True,
    ) is not None

    compact_branch = RCPD(
        RCPDConfig(
            regularization_lambda=0.05,
            max_depth=4,
            max_leaf_nodes=8,
        )
    )
    compact_branch.restore_training_state(source.training_state())

    assert compact_branch.program is None
    assert compact_branch.last_result is None
    assert compact_branch.last_extract_step is None
    assert compact_branch.regularization_weight == 0.0
    assert "structure changed" in (compact_branch.last_error or "")


def test_rcpd_resume_refits_when_split_selection_protocol_changes() -> None:
    source = RCPD(
        RCPDConfig(
            regularization_lambda=0.0,
            action_structure_weight=0.0,
            counterfactual_feature_selection_weight=0.0,
            minimum_extraction_samples=8,
        )
    )
    assert source.maybe_extract(
        17,
        _states(47, 80),
        _oracle,
        _features,
        force=True,
    ) is not None

    branch = RCPD(
        RCPDConfig(
            regularization_lambda=0.02,
            action_structure_weight=0.25,
            counterfactual_feature_selection_weight=1.0,
        )
    )
    branch.restore_training_state(source.training_state())

    assert branch.program is None
    assert branch.last_result is None
    assert branch.last_extract_step is None
    assert branch.regularization_weight == 0.0
    assert "extraction configuration changed" in (
        branch.last_error or ""
    )


def test_rcpd_resume_refits_when_program_selection_penalties_change() -> None:
    source = RCPD(
        RCPDConfig(
            regularization_lambda=0.0,
            complexity_penalty=0.001,
            distribution_penalty=0.2,
            minimum_extraction_samples=8,
        )
    )
    assert source.maybe_extract(
        17,
        _states(49, 80),
        _oracle,
        _features,
        force=True,
    ) is not None

    branch = RCPD(
        RCPDConfig(
            regularization_lambda=0.05,
            complexity_penalty=0.005,
            distribution_penalty=0.3,
        )
    )
    branch.restore_training_state(source.training_state())

    assert branch.program is None
    assert branch.last_result is None
    assert branch.last_extract_step is None
    assert branch.regularization_weight == 0.0
    assert "extraction configuration changed" in (
        branch.last_error or ""
    )


def test_rcpd_resume_refits_when_program_target_temperature_changes() -> None:
    source = RCPD(
        RCPDConfig(
            program_target_temperature=1.0,
            minimum_extraction_samples=8,
        )
    )
    assert source.maybe_extract(
        17,
        _states(51, 80),
        _oracle,
        _features,
        force=True,
    ) is not None

    branch = RCPD(RCPDConfig(program_target_temperature=2.0))
    branch.restore_training_state(source.training_state())

    assert branch.program is None
    assert branch.last_result is None
    assert branch.last_extract_step is None
    assert branch.regularization_weight == 0.0
    assert "extraction configuration changed" in (
        branch.last_error or ""
    )


def test_rcpd_can_be_disabled_and_extraction_failure_is_fail_open() -> None:
    states = _states(31, 12)
    disabled = RCPD(
        RCPDConfig(
            enabled=False,
            minimum_extraction_samples=1,
        )
    )
    assert (
        disabled.maybe_extract(
            1,
            states,
            _oracle,
            _features,
            force=True,
        )
        is None
    )
    assert disabled.program is None
    assert disabled.regularization_weight == 0.0

    failing = RCPD(
        RCPDConfig(
            minimum_extraction_samples=1,
        )
    )
    assert (
        failing.maybe_extract(
            1,
            states,
            lambda _state: (_ for _ in ()).throw(
                RuntimeError("oracle unavailable")
            ),
            _features,
            force=True,
        )
        is None
    )
    assert failing.regularization_weight == 0.0
    assert "oracle unavailable" in (failing.last_error or "")


def test_weighted_program_kl_regularization() -> None:
    actor_logits = np.asarray(
        [[2.0, 0.0], [0.0, 2.0]],
        dtype=float,
    )
    program = np.asarray(
        [[0.8, 0.2], [0.8, 0.2]],
        dtype=float,
    )
    first_only = RCPD.regularization_loss(
        actor_logits,
        program,
        weights=np.asarray([1.0, 0.0]),
    )
    second_upweighted = RCPD.regularization_loss(
        actor_logits,
        program,
        weights=np.asarray([1.0, 3.0]),
    )

    assert first_only >= 0.0
    assert second_upweighted > first_only


def test_program_kl_gate_reduces_absolute_feedback_strength() -> None:
    actor_logits = np.asarray(
        [[2.0, 0.0], [2.0, 0.0]],
        dtype=float,
    )
    program = np.asarray(
        [[0.2, 0.8], [0.2, 0.8]],
        dtype=float,
    )

    full = RCPD.regularization_loss(
        actor_logits,
        program,
        weights=np.asarray([1.0, 1.0]),
    )
    half_coverage = RCPD.regularization_loss(
        actor_logits,
        program,
        weights=np.asarray([1.0, 0.0]),
    )
    quarter_confidence = RCPD.regularization_loss(
        actor_logits,
        program,
        weights=np.asarray([0.25, 0.25]),
    )

    assert half_coverage == pytest.approx(full * 0.5)
    assert quarter_confidence == pytest.approx(full * 0.25)


def test_exported_program_is_a_bounded_safe_python_subset() -> None:
    result = RCPD(
        RCPDConfig(
            max_depth=3,
            max_leaf_nodes=5,
            max_predicates=3,
            min_samples_leaf=2,
        )
    ).fit(
        _states(41, 80),
        _oracle,
        _features,
        validation_states=_states(42, 30),
    )
    tree = ast.parse(result.program.to_python())
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.While,
        ast.AsyncFor,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.With,
        ast.Try,
    )

    assert not any(isinstance(node, forbidden) for node in ast.walk(tree))
    assert result.program.complexity()["depth"] <= 3
    assert result.program.complexity()["leaf_nodes"] <= 5
    assert result.program.complexity()["predicates"] <= 3
    assert result.metrics.extractability_score is not None
    assert result.metrics.extraction_time_seconds is not None
    assert result.metrics.extraction_time_seconds >= 0.0
    assert 0.0 <= result.metrics.extractability_score <= 1.0
