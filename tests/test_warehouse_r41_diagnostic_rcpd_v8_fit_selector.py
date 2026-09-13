from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v8 as v8
from backend.training import warehouse_r41_diagnostic_rcpd_v8_fit_selector as subject


FAMILIES = tuple(subject.INNER_HOLDOUT_FAMILY_QUOTAS)


def _fingerprint(number: int) -> str:
    return f"{number:064x}"


def _scope(
    *, eligible: list[str], exposed: list[str], families: dict[str, str],
    fresh: list[str] | None = None,
) -> dict:
    value = {
        "version": subject.SCOPE_VERSION,
        "source_report_sha256": "a" * 64,
        "source_rows_sha256": "b" * 64,
        "eligible_fit_scene_fingerprints": eligible,
        "inner_candidate_scenes": [
            {"fingerprint": scene, "family_id": families[scene]}
            for scene in eligible
        ],
        "previously_exposed_outer_scene_fingerprints": exposed,
        "fresh_outer_scene_fingerprints": [] if fresh is None else fresh,
        "label_blind": True,
        "final_rows_accessed": False,
        "final_labels_accessed": False,
    }
    value["content_sha256"] = subject.digest(value)
    return value


def _rows(specifications: list[tuple[str, bool, int]]) -> dict[str, np.ndarray]:
    count = len(specifications)
    observations = np.zeros((count, v8.BASE_FEATURE_COUNT), dtype=np.float32)
    probabilities = np.full((count, len(v8.ACTIONS)), 0.01, dtype=np.float32)
    for index, (_, _, action) in enumerate(specifications):
        probabilities[index, action] = 0.96
    observation_hashes = np.asarray(
        [_fingerprint(10_000 + index) for index in range(count)], dtype="S64")
    scenes = np.asarray([row[0] for row in specifications], dtype="S64")
    split = np.asarray([row[1] for row in specifications], dtype=np.bool_)
    return {
        "observations": observations,
        "probabilities": probabilities,
        "action_indices": np.asarray(
            [row[2] for row in specifications], dtype=np.uint8),
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": observation_hashes,
        "scene_fingerprints": scenes,
        "episode_ids": np.asarray(
            [f"episode-{index}:skilled" for index in range(count)], dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.zeros(count, dtype=np.uint8),
        "kinds": np.full(count, "ordinary", dtype="S16"),
        "anchor_ids": np.full(count, "", dtype="S240"),
        "branch_actions": np.full(count, "", dtype="S8"),
        "physical_hashes": np.full(count, "", dtype="S64"),
        "source_state_hashes": np.asarray(
            [_fingerprint(20_000 + index) for index in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": split,
    }


def _config() -> dict:
    model = {
        "learning_rate": 0.1,
        "max_iter": 70,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 10,
        "l2_regularization": 0.1,
        "max_depth": None,
        "max_bins": 255,
        "random_state": 941,
    }
    return {
        "version": v8.CONFIG_VERSION,
        "pair_pool_multiplier": 16.0,
        "use_action_factor": True,
        "models": {
            "base": deepcopy(model),
            "narrow_passage": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                               "random_state": 947},
            "shared_pickup": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                              "random_state": 953},
            "shared_charger": {**model, "max_iter": 1, "max_leaf_nodes": 2,
                               "random_state": 967},
        },
        "mix_weights": {group: 0.0 for group in v8.GROUPS},
    }


def _metrics(*, charger_direction: float, other: float = 0.92) -> dict:
    return {
        "overall": {"rows": 100, "scenes": 12, "fidelity": other},
        "nonwait": {"rows": 80, "scenes": 12, "fidelity": other},
        "critical": {
            group: {"rows": 30, "scenes": 12, "fidelity": other}
            for group in v8.GROUPS
        },
        "effective_intervention_direction": {
            "pairs": 50,
            "scenes": 12,
            "fidelity": other,
            "by_group": {
                group: {
                    "pairs": 20,
                    "scenes": 12,
                    "fidelity": charger_direction
                    if group == "shared_charger" else other,
                }
                for group in v8.GROUPS
            },
        },
        "mean_kl": 0.1,
    }


def test_contract_freezes_fit_only_search_and_has_no_outer_or_final_metric_input():
    value = subject.contract()
    assert value["source"][
        "previously_exposed_outer_rows_removed_before_label_validation"] is True
    assert value["source"][
        "fresh_outer_fingerprints_forbidden_from_source_rows"] is True
    assert value["source"]["outer_labels_accessed"] is False
    assert value["source"]["outer_probabilities_accessed"] is False
    assert value["source"]["final_rows_accessed"] is False
    assert value["source"]["final_labels_accessed"] is False
    assert value["frozen_model_change"]["mix_candidates"] == [0.0, 0.25, 0.5, 1.0]
    assert value["inner_split"]["holdout_scenes"] == 64
    assert sum(value["inner_split"]["family_quotas"].values()) == 64


def test_candidate_registry_changes_only_frozen_charger_capacity_and_mix():
    source = _config()
    candidates = subject.candidate_configs(source)
    assert [row["mix_weights"]["shared_charger"] for row in candidates] == [
        0.0, 0.25, 0.5, 1.0]
    for row in candidates:
        assert row["models"]["base"] == source["models"]["base"]
        assert row["models"]["narrow_passage"] == source["models"]["narrow_passage"]
        assert row["models"]["shared_pickup"] == source["models"]["shared_pickup"]
        assert row["models"]["shared_charger"] == subject.CHARGER_MODEL
        assert row["mix_weights"]["narrow_passage"] == 0.0
        assert row["mix_weights"]["shared_pickup"] == 0.0
        assert row["pair_pool_multiplier"] == 16.0
        assert row["use_action_factor"] is True


def test_outer_label_and_probability_mutation_cannot_change_fit_only_projection():
    eligible = [_fingerprint(index) for index in range(1, 5)]
    exposed = [_fingerprint(99)]
    family_map = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=family_map)
    rows = _rows([
        (eligible[0], False, 0), (eligible[0], False, 1),
        (eligible[1], False, 2), (eligible[1], False, 3),
        (eligible[2], False, 0), (eligible[2], False, 1),
        (eligible[3], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    changed = {name: value.copy() for name, value in rows.items()}
    changed["action_indices"][-1] = 0
    changed["probabilities"][-1] = [0.96, 0.01, 0.01, 0.01, 0.01]
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    first, first_audit = subject._project_fit_only(
        rows, scope, quotas=quotas, salt="unit-test-salt")
    second, second_audit = subject._project_fit_only(
        changed, scope, quotas=quotas, salt="unit-test-salt")
    assert subject._arrays_digest(first) == subject._arrays_digest(second)
    assert first_audit == second_audit
    assert exposed[0] not in set(map(str, v8._decode(
        first["scene_fingerprints"], "test scenes")))
    assert first_audit["outer_labels_or_probabilities_used_for_projection"] is False


def test_projection_is_whole_scene_and_validation_wins_exact_observation_overlap():
    eligible = [_fingerprint(index) for index in range(10, 14)]
    exposed = [_fingerprint(199)]
    family_map = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=family_map)
    rows = _rows([
        (scene, False, index % 5)
        for index, scene in enumerate(eligible)
        for _ in range(2)
    ] + [(exposed[0], True, 4)])
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    chosen, _ = subject._inner_holdout(
        scope["inner_candidate_scenes"], quotas=quotas, salt="overlap-salt")
    chosen_set = set(chosen)
    decoded = v8._decode(rows["scene_fingerprints"], "fixture scenes")
    validation_index = int(np.flatnonzero(np.isin(decoded, list(chosen_set)))[0])
    training_index = int(np.flatnonzero(~np.isin(decoded, list(chosen_set)))[0])
    rows["observation_hashes"][training_index] = rows["observation_hashes"][
        validation_index]
    projected, audit = subject._project_fit_only(
        rows, scope, quotas=quotas, salt="overlap-salt")
    projected_scenes = v8._decode(projected["scene_fingerprints"], "projected")
    assert set(projected_scenes[projected["split_validation"]]) == chosen_set
    assert not (set(map(bytes, projected["observation_hashes"][
        ~projected["split_validation"]])) & set(map(bytes, projected[
            "observation_hashes"][projected["split_validation"]])))
    assert audit["inner_fit_rows_removed_for_exact_holdout_overlap"] == 1
    assert audit["scene_identity_overlap"] == 0
    assert audit["episode_identity_overlap"] == 0
    assert audit["nonempty_anchor_identity_overlap"] == 0
    for scene in eligible:
        flags = projected["split_validation"][projected_scenes == scene]
        assert len(set(map(bool, flags))) == 1


def test_projection_rejects_episode_or_anchor_identity_crossing_inner_split():
    eligible = [_fingerprint(index) for index in range(20, 24)]
    exposed = [_fingerprint(219)]
    families = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(eligible=eligible, exposed=exposed, families=families)
    quotas = {FAMILIES[0]: 1, FAMILIES[1]: 1}
    rows = _rows([
        (eligible[0], False, 0), (eligible[1], False, 1),
        (eligible[2], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    chosen, _ = subject._inner_holdout(
        scope["inner_candidate_scenes"], quotas=quotas, salt="identity-salt")
    decoded = v8._decode(rows["scene_fingerprints"], "identity fixture scenes")
    validation_index = int(np.flatnonzero(np.isin(decoded, chosen))[0])
    fit_index = int(np.flatnonzero(
        (~rows["split_validation"]) & ~np.isin(decoded, chosen))[0])

    crossed_episode = {name: value.copy() for name, value in rows.items()}
    crossed_episode["episode_ids"][fit_index] = crossed_episode["episode_ids"][
        validation_index]
    with pytest.raises(ValueError, match="projection isolation differs"):
        subject._project_fit_only(
            crossed_episode, scope, quotas=quotas, salt="identity-salt")

    crossed_anchor = {name: value.copy() for name, value in rows.items()}
    crossed_anchor["anchor_ids"][[fit_index, validation_index]] = "shared-anchor"
    with pytest.raises(ValueError, match="projection isolation differs"):
        subject._project_fit_only(
            crossed_anchor, scope, quotas=quotas, salt="identity-salt")


def test_fresh_outer_must_be_absent_from_source_rows():
    eligible = [_fingerprint(index) for index in range(30, 34)]
    exposed = [_fingerprint(299)]
    families = {
        eligible[0]: FAMILIES[0], eligible[1]: FAMILIES[0],
        eligible[2]: FAMILIES[1], eligible[3]: FAMILIES[1],
    }
    scope = _scope(
        eligible=eligible, exposed=exposed, families=families,
        fresh=[_fingerprint(400)],
    )
    rows = _rows([
        (eligible[0], False, 0), (eligible[1], False, 1),
        (eligible[2], False, 2), (eligible[3], False, 3),
        (exposed[0], True, 4),
    ])
    projected, audit = subject._project_fit_only(
        rows, scope, quotas={FAMILIES[0]: 1, FAMILIES[1]: 1},
        salt="fresh-absent")
    assert audit["fresh_outer_scenes_registered_absent_from_source"] == 1
    contaminated = {name: value.copy() for name, value in rows.items()}
    extra = _rows([(_fingerprint(400), False, 0)])
    contaminated = {
        name: np.concatenate([contaminated[name], extra[name]], axis=0)
        for name in contaminated
    }
    with pytest.raises(ValueError, match="scene identity differs"):
        subject._project_fit_only(
            contaminated, scope,
            quotas={FAMILIES[0]: 1, FAMILIES[1]: 1}, salt="fresh-absent")


def test_selection_uses_all_gates_gain_guardrail_and_smallest_mix():
    baseline = _metrics(charger_direction=0.84)
    quarter = _metrics(charger_direction=0.851, other=0.919)
    half = _metrics(charger_direction=0.87, other=0.917)
    full = _metrics(charger_direction=0.88, other=0.919)
    result = subject.choose_candidate([
        {"mix_weight": mix, "metrics": metrics}
        for mix, metrics in zip(subject.MIX_CANDIDATES,
                                (baseline, quarter, half, full))
    ])
    assert result["status"] == subject.STATUS_SELECTED
    assert result["selected_mix_weight"] == 0.25
    by_mix = {row["mix_weight"]: row for row in result["candidates"]}
    assert not by_mix[0.0]["eligible"]
    assert by_mix[0.25]["eligible"]
    assert not by_mix[0.5]["selection_checks"][
        "other_metrics_within_degradation_limit"]
    assert by_mix[1.0]["eligible"]
    assert result["outer_evaluation_performed"] is False


def test_weight_audit_recovers_one_true_family_per_fit_scene():
    scenes = {_fingerprint(501), _fingerprint(502)}
    audit = {"balance": {"combined_training": {"scene_totals": [
        {"scene": sorted(scenes)[0], "family": FAMILIES[0], "mass": 10.0},
        {"scene": sorted(scenes)[1], "family": FAMILIES[1], "mass": 20.0},
    ]}}}
    result = subject._scene_families_from_weight_audit(
        audit, expected_scenes=scenes)
    assert result == {
        sorted(scenes)[0]: FAMILIES[0], sorted(scenes)[1]: FAMILIES[1],
    }
    audit["balance"]["combined_training"]["scene_totals"].append({
        "scene": sorted(scenes)[0], "family": FAMILIES[2], "mass": 1.0,
    })
    with pytest.raises(ValueError, match="two scene families"):
        subject._scene_families_from_weight_audit(audit, expected_scenes=scenes)


def test_source_closure_contains_no_release_or_final_evaluator():
    sources = subject.producer_sources()
    assert "scripts/build_warehouse_r41_diagnostic_rcpd_v8_fit_selector.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_pair_weights_v8.py" in sources
    assert not [path for path in sources if any(token in path for token in (
        "admission", "release", "preflight", "fresh_final_holdout",
        "final_once", "explanation_audit",
    ))]
