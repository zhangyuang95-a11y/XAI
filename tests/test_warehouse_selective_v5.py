"""Selective Warehouse program audit contracts; no held-out episodes."""
from copy import deepcopy

import numpy as np
import pytest

from backend.training.warehouse_selective_v5 import (
    FEATURE_NAMES, SelectiveLeafProgram, audit_frozen_program, fit_selective_program,
    observable_features,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.pure_nn import action_masks, observations
from env.warehouse.pure_nn import observation_feature_names


ROLE_HASH = "a" * 64


def _template():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=82000)
    return {"obs": observations(env.state)[1], "mask": action_masks(env.state)[1]}


def _rows(template, prefix, wait_count=100, *, illegal_wait=0):
    result = []
    for index in range(100):
        row = deepcopy(template)
        row["episode"] = f"{prefix}-{index}"
        row["label"] = "WAIT" if index < wait_count else "UP"
        if index < illegal_wait:
            row["mask"] = row["mask"].copy()
            row["mask"][4] = 0
            row["label"] = "UP"
        result.append(row)
    return result


def _oracle(row):
    return np.eye(5)[{"UP": 0, "WAIT": 4}[row["label"]]]


def test_selective_feedback_and_ai_explanation_have_distinct_audits(tmp_path):
    template = _template()
    assert set(observable_features(template)) == set(FEATURE_NAMES)
    parts = [_rows(template, f"split-{index}", 89) for index in range(4)]
    program = fit_selective_program("robot_2", *parts, _oracle,
                                    actor_role_sha256=ROLE_HASH)
    assert program.report["max_depth"] <= 5
    assert program.report["leaf_count"] <= 16
    assert program.report["validation"]["feedback_passed"] is True
    assert program.report["validation"]["explanation_passed"] is False
    assert program.report["validation"]["feedback"]["agreement"] == pytest.approx(.89)
    assert program.report["validation"]["feedback"]["coverage"] == 1.0
    assert program.covered(parts[-1][0], actor_action="WAIT", purpose="feedback") is not None
    assert program.covered(parts[-1][0], actor_action="UP", purpose="feedback") is None
    assert program.covered(parts[-1][0], actor_action="WAIT", purpose="explanation") is None
    path = tmp_path / "program.json"
    digest = program.save(path)
    assert digest == program.sha256()
    restored = SelectiveLeafProgram.load_json(path, expected_sha256=digest,
                                               actor_role_sha256=ROLE_HASH)
    assert restored.execute(parts[-1][0])["leaf_id"] == program.lookup(parts[-1][0])["leaf_id"]
    assert SelectiveLeafProgram.from_dict(program.to_dict(), actor_role_sha256=ROLE_HASH).sha256() == digest
    corrupted = deepcopy(program.to_dict())
    corrupted["feedback_leaves"] = []
    with pytest.raises(RuntimeError, match="eligibility audit"):
        SelectiveLeafProgram.from_dict(corrupted, actor_role_sha256=ROLE_HASH)
    with pytest.raises(RuntimeError, match="binding mismatch"):
        SelectiveLeafProgram.load(path, expected_sha256=digest,
                                  actor_role_sha256="b" * 64)


def test_selective_explanation_passes_only_with_covered_ai_fidelity():
    template = _template()
    parts = [_rows(template, f"split-{index}", 95) for index in range(4)]
    program = fit_selective_program("robot_2", *parts, _oracle,
                                    actor_role_sha256=ROLE_HASH)
    assert program.report["validation"]["explanation_passed"] is True
    evidence = program.covered(parts[-1][0], actor_action="WAIT", purpose="explanation")
    assert evidence is not None
    assert evidence["eligible"] is True
    assert evidence["leaf_id"] in program.explanation_leaves
    assert evidence["program_sha256"] == program.sha256()
    assert program.covered(parts[-1][-1], actor_action="UP", purpose="explanation") is None
    digest = program.sha256()
    independent = _rows(template, "90000", 95)
    for index, row in enumerate(independent):
        row["episode"] = str(90000 + index)
    audit = audit_frozen_program(program, independent, _oracle)
    assert audit["passed"] is True
    assert audit["development_seeds"] == list(range(90000, 90100))
    assert audit["validation"]["global"]["rows"] == audit["sample_count"] == 100
    assert audit["feedback_eligible"] and audit["explanation_eligible"]
    assert program.sha256() == digest
    with pytest.raises(ValueError, match="reused"):
        audit_frozen_program(program, parts[-1], _oracle)
    with pytest.raises(ValueError, match="numeric scenario seeds"):
        audit_frozen_program(program, _rows(template, "nonnumeric", 95), _oracle)


def test_episode_leakage_and_physical_illegality_fail_closed():
    template = _template()
    fit, calibration, eligibility, validation = (
        _rows(template, f"split-{index}") for index in range(4))
    overlap = deepcopy(eligibility)
    overlap[0]["episode"] = fit[0]["episode"]
    with pytest.raises(ValueError, match="overlap"):
        fit_selective_program("robot_2", fit, calibration, overlap, validation,
                              _oracle, actor_role_sha256=ROLE_HASH)
    eligibility = _rows(template, "audit", illegal_wait=5)
    program = fit_selective_program("robot_2", fit, calibration, eligibility,
                                    validation, _oracle, actor_role_sha256=ROLE_HASH)
    leaf = program.lookup(fit[0])["leaf_id"]
    assert program.leaf_audit[leaf]["illegal_count"] == 5
    assert program.report["validation"]["feedback_passed"] is False
    assert program.covered(validation[0], actor_action="WAIT", purpose="feedback") is None


def test_online_feedback_requires_each_leaf_on_both_disjoint_splits():
    template = _template()
    battery = observation_feature_names().index("self_battery_fraction")

    def rows(prefix, *, noisy_majority):
        result = []
        for index in range(200):
            row = deepcopy(template)
            row["episode"] = f"{prefix}-{index}"
            rare = index < 25
            row["obs"] = row["obs"].copy()
            row["obs"][battery] = .1 if rare else .9
            row["label"] = ("WAIT" if rare or (noisy_majority and index % 2 == 0)
                            else "UP")
            result.append(row)
        return result

    fit = rows("fit", noisy_majority=False)
    calibration = rows("calibration", noisy_majority=False)
    eligibility = rows("eligibility", noisy_majority=True)
    validation = rows("validation", noisy_majority=True)
    program = fit_selective_program("robot_1", fit, calibration, eligibility,
                                    validation, _oracle, actor_role_sha256=ROLE_HASH)
    assert len(program.feedback_leaves) == 1
    assert program.report["validation"]["feedback"]["coverage"] == pytest.approx(.125)
    assert program.report["validation"]["feedback_online_passed"] is True
    assert program.report["validation"]["feedback_passed"] is False
    assert program.covered(validation[0], actor_action="WAIT", purpose="feedback")
    assert program.covered(validation[-1], actor_action="UP", purpose="feedback") is None
    assert SelectiveLeafProgram.from_dict(program.to_dict(),
                                          actor_role_sha256=ROLE_HASH).feedback_leaves == program.feedback_leaves

    failed_validation = rows("failed-validation", noisy_majority=True)
    for row in failed_validation[:10]:
        row["label"] = "UP"
    gated = fit_selective_program("robot_1", fit, calibration, eligibility,
                                  failed_validation, _oracle, actor_role_sha256=ROLE_HASH)
    assert gated.feedback_leaves == frozenset()
    assert gated.report["validation"]["feedback_online_passed"] is False
    assert gated.covered(failed_validation[11], actor_action="WAIT", purpose="feedback") is None
    forged = deepcopy(gated.to_dict())
    forged["feedback_leaves"] = sorted(program.feedback_leaves)
    with pytest.raises(RuntimeError, match="eligibility audit"):
        SelectiveLeafProgram.from_dict(forged, actor_role_sha256=ROLE_HASH)

    final = rows("unused-final", noisy_majority=False)
    for index, row in enumerate(final):
        row["episode"] = str(91000 + index)
        if 25 <= index < 50:
            row["obs"][battery] = .1
            row["label"] = "WAIT"
    independent = audit_frozen_program(program, final, _oracle)
    assert independent["validation"]["feedback"]["coverage"] == pytest.approx(.25)
    assert independent["feedback_eligible"] is True

    ai = fit_selective_program("robot_2", fit, calibration, eligibility,
                               validation, _oracle, actor_role_sha256=ROLE_HASH)
    assert ai.report["validation"]["explanation_online_passed"] is True
    assert ai.report["validation"]["explanation_passed"] is False
    assert ai.covered(validation[0], actor_action="WAIT", purpose="explanation")
    ai_independent = audit_frozen_program(ai, final, _oracle)
    assert ai_independent["explanation_eligible"] is True
    forged_ai = deepcopy(ai.to_dict())
    forged_ai["report"]["validation"]["explanation_online_passed"] = False
    with pytest.raises(RuntimeError, match="eligibility audit"):
        SelectiveLeafProgram.from_dict(forged_ai, actor_role_sha256=ROLE_HASH)
