"""Focused CLI options for balancing offline teaching and PPO learning."""

from __future__ import annotations

import argparse


def add_teacher_balance_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--behavior-cloning-samples", type=int, default=65536)
    parser.add_argument("--behavior-cloning-epochs", type=int, default=30)
    parser.add_argument("--behavior-cloning-batch-size", type=int, default=512)
    parser.add_argument("--behavior-cloning-lr", type=float, default=0.0003)
    parser.add_argument("--skill-retention-samples", type=int, default=32768)
    parser.add_argument("--skill-retention-weight", type=float, default=1.0)
    parser.add_argument("--skill-retention-fixed", action="store_true")
    parser.add_argument("--legacy-teacher-supervision", action="store_true")
    parser.add_argument(
        "--skill-retention-fade-end",
        type=float,
        default=0.50,
        help=(
            "Fraction of total training by which offline teacher retention "
            "decays linearly to zero."
        ),
    )


def skill_retention_weight(args: argparse.Namespace, episode: int) -> float:
    """Decay offline imitation completely before PPO's second half."""

    if args.skill_retention_fixed:
        return float(args.skill_retention_weight)
    progress = episode / max(1, int(args.episodes))
    if progress >= float(args.skill_retention_fade_end):
        return 0.0
    return float(
        args.skill_retention_weight
        * (1.0 - progress / float(args.skill_retention_fade_end))
    )
