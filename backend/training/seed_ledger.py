"""Seed-family provenance for warehouse training and formal evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Any, Mapping

from reference_trajectory import TUTORIAL_SEED


def evaluation_seed_span(
    seed_ranges: Mapping[str, Any],
) -> tuple[int, int]:
    """Return one conservative interval covering nested evaluation ranges."""

    ranges: list[tuple[int, int]] = []
    for value in seed_ranges.values():
        items = value.values() if isinstance(value, Mapping) else (value,)
        for item in items:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("Every evaluation seed range must be [start, count].")
            start, count = int(item[0]), int(item[1])
            if count <= 0:
                raise ValueError("Evaluation seed range counts must be positive.")
            ranges.append((start, start + count))
    if not ranges:
        raise ValueError("Evaluation declares no seed ranges.")
    return min(item[0] for item in ranges), max(item[1] for item in ranges)


def reserve_evaluation_seed_span(
    ledger: Mapping[str, Any],
    seed_ranges: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Return an idempotently extended seed ledger for prior evaluation use."""

    updated = deepcopy(dict(ledger))
    intervals = updated.setdefault("reserved_intervals", [])
    if not isinstance(intervals, list):
        raise ValueError("Seed ledger reserved_intervals must be a list.")
    start, end_exclusive = evaluation_seed_span(seed_ranges)
    interval = {
        "name": name,
        "start": start,
        "end_exclusive": end_exclusive,
    }
    existing = next(
        (
            item
            for item in intervals
            if isinstance(item, Mapping) and item.get("name") == name
        ),
        None,
    )
    if existing is None:
        intervals.append(interval)
    elif dict(existing) != interval:
        raise ValueError("Seed interval name already has another span.")
    return updated


def training_seed_ledger(args: argparse.Namespace) -> dict[str, object]:
    """Disclose every seed family touched before formal evaluation."""

    first_family_end = max(
        int(args.seed) + int(args.episodes) + 1,
        int(args.seed) + 200_000 + int(args.eval_episodes),
        int(args.seed) + 300_001,
        int(args.seed) + 400_000 + int(args.parallel_seed_candidates),
    )
    return {
        "schema": "warehouse-training-seed-ledger.v1",
        "interval_semantics": "start_inclusive_end_exclusive",
        "reserved_intervals": [
            {
                "name": "mappo_training_final_eval_trajectory_and_seed_calibration",
                "start": int(args.seed),
                "end_exclusive": first_family_end,
            },
            {
                "name": "fixed_reference_calibration",
                "start": int(TUTORIAL_SEED),
                "end_exclusive": int(TUTORIAL_SEED)
                + int(args.reference_seed_candidates),
            },
            {
                "name": "offline_actor_curricula_periodic_eval_and_relabeling",
                "start": int(args.seed) + 700_000,
                "end_exclusive": int(args.seed) + 10_000_000,
            },
            {
                "name": "final_posthoc_rcpd_rollouts",
                "start": int(args.seed) + 11_000_000,
                "end_exclusive": int(args.seed) + 11_100_000,
            },
            {
                "name": "rejected_candidate_development_evaluation",
                "start": 500_004,
                "end_exclusive": 10_500_204,
            },
        ],
    }
