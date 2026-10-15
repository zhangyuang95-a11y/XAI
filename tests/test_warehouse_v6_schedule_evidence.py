"""A paired receipt must expose the assigned blocks and actual reset history."""
from __future__ import annotations

from copy import deepcopy

from backend.training.warehouse_v6_runner import (
    SCENARIO_BLOCK_STEPS, scenario_schedule_sha256, scenario_seed)
from experiments.warehouse import protocol as p
from experiments.warehouse.acceptance import _digest, _paired_schedule_evidence


def _resets(pair_seed: int, steps: int, *, split_first: bool) -> list[dict]:
    namespace = "warehouse-v6-matched-120-step-blocks"
    rows, cursor = [], 0
    while cursor < steps:
        limit = min(SCENARIO_BLOCK_STEPS - cursor % SCENARIO_BLOCK_STEPS,
                    steps - cursor)
        length = 60 if split_first and cursor == 0 else limit
        seed = scenario_seed(namespace, pair_seed, cursor)
        index = len(rows)
        rows.append({
            "seed": seed, "episode_index": index,
            "episode_key": f"{seed}:{index}",
            "start_joint_step": cursor, "joint_steps": length,
            "complete": bool(split_first and cursor == 0),
            "shutdown_episode": False,
        })
        cursor += length
    return rows


def _pair() -> dict:
    pair_seed, steps = p.DEVELOPMENT_PAIR_SEEDS[0], p.BUDGETS[0]
    namespace = "warehouse-v6-matched-120-step-blocks"
    spec = p.PairRunSpec("schedule-test", pair_seed, steps, "/unused",
                         checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    digest = scenario_schedule_sha256(namespace, pair_seed, steps)
    raw = _resets(pair_seed, steps, split_first=False)
    regularized = _resets(pair_seed, steps, split_first=True)
    variants = {}
    for variant, resets in (("raw", raw), ("regularized", regularized)):
        variants[variant] = {
            "variant": variant, "pair_seed": pair_seed, "joint_steps": steps,
            "run_spec_sha256": spec.sha256(),
            "scenario_schedule_sha256": digest,
            "assigned_scenario_schedule_kind": "fixed_120_joint_step_blocks",
            "actual_episode_resets": resets,
            "actual_episode_resets_sha256": _digest(resets),
        }
    return {
        "candidate_id": spec.candidate_id, "release_id": spec.release_id,
        "pair_seed": pair_seed, "joint_steps": steps,
        "run_spec_sha256": spec.sha256(),
        "checkpoint_seeds": list(spec.checkpoint_seeds),
        "assigned_scenario_schedule_sha256": digest,
        "actual_episode_reset_count": {variant: len(row["actual_episode_resets"])
                                       for variant, row in variants.items()},
        "actual_episode_resets_equal": False,
        "variants": variants,
    }


def test_paired_schedule_allows_different_early_resets_but_rejects_forgery():
    pair = _pair()
    assert _paired_schedule_evidence(pair)
    assert pair["variants"]["raw"]["actual_episode_resets"] != \
           pair["variants"]["regularized"]["actual_episode_resets"]

    forged = deepcopy(pair)
    forged["actual_episode_resets_equal"] = True
    assert not _paired_schedule_evidence(forged)

    forged = deepcopy(pair)
    forged["variants"]["regularized"]["actual_episode_resets"][0]["seed"] += 1
    forged["variants"]["regularized"]["actual_episode_resets_sha256"] = _digest(
        forged["variants"]["regularized"]["actual_episode_resets"])
    assert not _paired_schedule_evidence(forged)

    forged = deepcopy(pair)
    forged["variants"]["regularized"]["actual_episode_resets"][0]["complete"] = False
    forged["variants"]["regularized"]["actual_episode_resets_sha256"] = _digest(
        forged["variants"]["regularized"]["actual_episode_resets"])
    assert not _paired_schedule_evidence(forged)

    forged = deepcopy(pair)
    forged["assigned_scenario_schedule_sha256"] = "0" * 64
    assert not _paired_schedule_evidence(forged)

    forged = deepcopy(pair)
    forged["variants"]["raw"]["actual_episode_resets"].pop()
    forged["variants"]["raw"]["actual_episode_resets_sha256"] = _digest(
        forged["variants"]["raw"]["actual_episode_resets"])
    forged["actual_episode_reset_count"]["raw"] -= 1
    assert not _paired_schedule_evidence(forged)
