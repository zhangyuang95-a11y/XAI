"""Warehouse v6 protocol cannot pass by changing labels or reusing sealed data."""
from copy import deepcopy

import pytest

from experiments.warehouse import protocol as p


def _quality(deliveries=6.0, collisions=5.0, shutdown=.02):
    return {"mean_deliveries": deliveries, "mean_collisions": collisions,
            "shutdown_episode_rate": shutdown}


def _program(coverage=.35, agreement=.97, kl=.10):
    return {"rows": 1_000, "coverage": coverage, "agreement": agreement,
            "kl": kl, "illegal_count": 0}


def _paired_training(joint_steps=200_000):
    initial = {"robot_1": "a" * 64, "robot_2": "b" * 64}
    variants = {}
    for variant in p.VARIANTS:
        variants[variant] = {
            "joint_steps": joint_steps, "actor_decisions": 2 * joint_steps,
            "post_policy_interventions": 0, "physical_no_legal_action_rows": 0,
            "scenario_schedule_sha256": "s" * 64,
            "initial_role_sha256": initial,
            "selected_role_sha256": {"robot_1": ("c" if variant == "raw" else "e") * 64,
                                     "robot_2": ("d" if variant == "raw" else "f") * 64},
            "role_weight_change_norm": {"robot_1": 1.0, "robot_2": 1.0},
            "feedback_rows": {"robot_1": 0 if variant == "raw" else 10,
                              "robot_2": 0 if variant == "raw" else 10},
            "feedback_gradient_norm": {"robot_1": 0.0 if variant == "raw" else .1,
                                       "robot_2": 0.0 if variant == "raw" else .1},
        }
    return variants


def _development_report(pair_seed, budget):
    training = _paired_training(budget)
    return {"pair_seed": pair_seed, "budget": budget,
            "training": training,
            "stress_episodes": len(p.stress_schedule()),
            "stress_schedule_sha256": p._hash_json(p.stress_schedule()),
            "evaluation": {variant: _quality() for variant in p.VARIANTS},
            "program_audit": {role: {
                "actor_role_sha256": training["regularized"]["selected_role_sha256"][role],
                "program_sha256": "9" * 64, "sample_count": 2304,
                "feedback": _program(), "explanation": _program(),
            } for role in p.ROLES}}


def test_seed_blocks_and_stress_streams_are_sealed_and_disjoint():
    assert not set(p.DEVELOPMENT_PAIR_SEEDS) & set(p.FORMAL_PAIR_SEEDS)
    blocks = (p.DEVELOPMENT_SELECTION_SEEDS, p.DEVELOPMENT_STRESS_BASE_SEEDS,
              p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
              p.FORMAL_SELECTION_SEEDS,
              p.FORMAL_PROGRAM_PREAUDIT_SEEDS, p.FINAL_PROGRAM_AUDIT_SEEDS,
              p.HELDOUT_BASE_SEEDS)
    for index, block in enumerate(blocks):
        assert not set(block) & set().union(*map(set, blocks[index + 1:]))
    schedule = p.stress_schedule()
    assert len(schedule) == 256 * 3 * 3
    assert len(set(schedule)) == len(schedule)
    assert len({row[2] for row in schedule}) == len(schedule)
    assert all(len(seeds) == 100 for seeds in p.heldout_schedule().values())
    assert p.sealed_seed_blocks()["heldout"] == p.HELDOUT_BASE_SEEDS
    assert p.final_audit_seeds() == p.FINAL_PROGRAM_AUDIT_SEEDS
    assert p.heldout_base_seeds() == p.HELDOUT_BASE_SEEDS
    next_blocks = p.sealed_seed_blocks("policylens-three-domain-20261016.v6.1")
    assert not set(next_blocks["final_program_audit"]) & set(p.FINAL_PROGRAM_AUDIT_SEEDS)
    assert not set(next_blocks["heldout"]) & set(p.HELDOUT_BASE_SEEDS)
    assert not set(next_blocks["heldout"]) & set(next_blocks["final_program_audit"])
    assert set(p.heldout_schedule("policylens-three-domain-20261016.v6.1")[1]) == {
        p.task_scenario_seed(seed, 1) for seed in next_blocks["heldout"]}
    spec = p.PairRunSpec("warehouse-v6-c1", 6110, 200_000, "/tmp/run-a",
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    moved = p.PairRunSpec("warehouse-v6-c1", 6110, 200_000, "/tmp/run-b",
                          checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    assert spec.sha256() == moved.sha256()
    with pytest.raises(ValueError, match="checkpoint seeds"):
        p.PairRunSpec("warehouse-v6-c1", 6110, 200_000, "/tmp/run-a",
                      checkpoint_seeds=p.FORMAL_SELECTION_SEEDS)


def test_quality_and_program_margins_are_stricter_only_in_development():
    assert p.quality_pass(_quality(), development=True)
    assert not p.quality_pass(_quality(shutdown=.06), development=True)
    assert p.quality_pass(_quality(shutdown=.06), development=False)
    assert not p.quality_pass(_quality(deliveries=float("nan")), development=False)
    assert p.program_pass(_program(), purpose="feedback", development=True)
    assert p.program_pass(_program(), purpose="explanation", development=True)
    assert not p.program_pass(_program(coverage=.28), purpose="feedback", development=True)
    assert p.program_pass(_program(coverage=.28), purpose="feedback", development=False)
    assert not p.program_pass({**_program(), "illegal_count": 1},
                              purpose="feedback", development=False)


def test_pair_requires_identical_bc_and_scenario_schedule_and_real_gradients():
    reports = _paired_training()
    assert p.paired_training_pass(reports, joint_steps=200_000)
    bad = deepcopy(reports)
    bad["regularized"]["scenario_schedule_sha256"] = "different"
    assert not p.paired_training_pass(bad, joint_steps=200_000)
    bad = deepcopy(reports)
    bad["regularized"]["feedback_gradient_norm"]["robot_2"] = 0.0
    assert not p.paired_training_pass(bad, joint_steps=200_000)
    bad = deepcopy(reports)
    bad["raw"]["feedback_rows"]["robot_1"] = 1
    assert not p.paired_training_pass(bad, joint_steps=200_000)
    bad = deepcopy(reports)
    bad["regularized"]["post_policy_interventions"] = 1
    assert not p.paired_training_pass(bad, joint_steps=200_000)


def test_formal_selection_uses_both_variant_quality_and_never_program_metrics():
    reports = []
    for seed, delivery in zip(p.FORMAL_PAIR_SEEDS, (6.0, 6.4, 6.1)):
        reports.append({"pair_seed": seed,
                        "selection_seeds": list(p.FORMAL_SELECTION_SEEDS),
                        "variants": {variant: {"quality": _quality(deliveries=delivery)}
                                     for variant in p.VARIANTS},
                        "program": {"arbitrary": seed == 6102}})
    assert p.select_formal_pair(reports)["pair_seed"] == 6102
    reports[1]["variants"]["regularized"]["quality"]["shutdown_episode_rate"] = .20
    assert p.select_formal_pair(reports)["pair_seed"] == 6103
    reports[0]["selection_seeds"] = list(p.FINAL_PROGRAM_AUDIT_SEEDS)
    with pytest.raises(ValueError, match="seed mismatch"):
        p.select_formal_pair(reports)


def test_development_locks_smallest_budget_only_after_all_three_pairs_pass():
    rows = {budget: [_development_report(seed, budget)
                     for seed in p.DEVELOPMENT_PAIR_SEEDS]
            for budget in p.BUDGETS}
    assert p.choose_development_budget(rows) == 200_000
    rows[200_000][1]["evaluation"]["regularized"]["shutdown_episode_rate"] = .08
    assert p.choose_development_budget(rows) == 300_000
    rows[300_000][2]["program_audit"]["robot_2"]["feedback"]["agreement"] = .90
    assert p.choose_development_budget(rows) == 400_000
    rows[400_000][0]["training"]["raw"]["feedback_rows"]["robot_1"] = 1
    assert p.choose_development_budget(rows) is None


def test_once_only_claim_is_created_before_evaluation_and_never_reused(tmp_path):
    path = tmp_path / "final-audit.claim.json"
    hashes = {"raw": "a" * 64, "regularized": "b" * 64}
    receipt = p.claim_once(path, phase="final_program_audit",
                           candidate_id="warehouse-v6-c1", actor_hashes=hashes,
                           seed_rows=p.FINAL_PROGRAM_AUDIT_SEEDS)
    assert receipt["actor_hashes"] == hashes
    with pytest.raises(FileExistsError):
        p.claim_once(path, phase="final_program_audit",
                     candidate_id="warehouse-v6-c1", actor_hashes=hashes,
                     seed_rows=p.FINAL_PROGRAM_AUDIT_SEEDS)
    with pytest.raises(ValueError, match="wrong seeds"):
        p.claim_once(tmp_path / "bad.json", phase="heldout",
                     candidate_id="warehouse-v6-c1", actor_hashes=hashes,
                     seed_rows=p.FINAL_PROGRAM_AUDIT_SEEDS)
    assert not (tmp_path / "bad.json").exists()
    v61 = "policylens-three-domain-20261016.v6.1"
    with pytest.raises(ValueError, match="wrong seeds"):
        p.claim_once(tmp_path / "wrong-version.json", phase="heldout",
                     candidate_id="warehouse-v6-c2", actor_hashes=hashes,
                     seed_rows=p.HELDOUT_BASE_SEEDS, release_id=v61)
    assert p.claim_once(tmp_path / "v61.json", phase="heldout",
                        candidate_id="warehouse-v6-c2", actor_hashes=hashes,
                        seed_rows=p.sealed_seed_blocks(v61)["heldout"],
                        release_id=v61)["release_id"] == v61
