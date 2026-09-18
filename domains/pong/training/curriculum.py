"""Quota-tracked, physical-state-only curriculum for Cooperative Pong v2.1/v2.2.

The curriculum may choose a legal reset state. It never supplies a target
ball, an action label, or an action override to the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping

from ..config import PongConfig


@dataclass(frozen=True)
class CurriculumEpisode:
    stage: str
    scenario: str
    active_ball_ids: frozenset[str]


class PongCurriculum:
    """Choose scenarios from deficits measured in real collected decisions."""

    _STAGE_SCENARIOS = {
        "foundation": (
            ("large_both_ready", .40), ("large_both_approach", .25),
            ("large_one_holds", .15), ("full", .20),
        ),
        "coordination": (
            ("small_then_large", .25), ("dual_large", .20),
            ("dual_small", .15), ("large_both_ready", .10), ("full", .30),
        ),
        "full": (("full", 1.0),),
    }
    _LEGACY_STAGE_SCENARIOS = {
        "foundation": (("single_small", .40), ("single_large", .40), ("full", .20)),
        "coordination": (("two_small", .30), ("small_large_conflict", .30), ("hold_large", .20), ("full", .20)),
        "full": (("full", 1.0),),
    }
    _V22_SCENARIOS = {
        "foundation": (("full", .30), ("large_both_ready", .10),
                       ("large_both_approach", .20), ("large_one_holds", .20),
                       ("dual_small", .20)),
        "coordination": (("full", .60), ("small_then_large", .15),
                         ("dual_large", .15), ("dual_small", .10)),
        "full": (("full", .90), ("small_then_large", .04),
                 ("dual_large", .03), ("large_one_holds", .03)),
    }

    def __init__(self, config: PongConfig, settings: Mapping[str, object], *, seed: int) -> None:
        self.config = config
        self.settings = dict(settings)
        self._rng = random.Random(seed)
        self.v21 = "foundation_until_joint_steps" in self.settings
        self.v22 = bool(self.settings.get("v22", False))
        self._stage_scenarios = (self._V22_SCENARIOS if self.v22 else
                                 self._STAGE_SCENARIOS if self.v21 else self._LEGACY_STAGE_SCENARIOS)
        self.step_counts: dict[str, dict[str, int]] = {
            stage: {name: 0 for name, _ in choices}
            for stage, choices in self._stage_scenarios.items()
        }

    def stage_for_steps(self, joint_steps: int) -> str:
        if self.v21:
            foundation = int(self.settings.get("foundation_until_joint_steps", 50_000))
            coordination = int(self.settings.get("coordination_until_joint_steps", 125_000))
        else:
            maximum = int(self.settings.get("max_joint_steps", 500_000))
            foundation = int(maximum * float(self.settings.get("foundation_until_fraction", .20)))
            coordination = int(maximum * float(self.settings.get("coordination_until_fraction", .50)))
        if joint_steps < foundation:
            return "foundation"
        if joint_steps < coordination:
            return "coordination"
        return "full"

    def stage_for_progress(self, progress: float) -> str:
        """Compatibility helper used by older callers and status displays."""
        maximum = int(self.settings.get("max_joint_steps", 250_000))
        return self.stage_for_steps(int(max(0.0, progress) * maximum))

    def sample(self, joint_steps: int) -> CurriculumEpisode:
        stage = self.stage_for_steps(joint_steps)
        if self.v22 and joint_steps >= int(self.settings.get("full_only_from_joint_steps", 400_000)):
            return CurriculumEpisode(stage, "full", frozenset(self.config.ball_ids))
        choices = self._stage_scenarios[stage]
        counts = self.step_counts[stage]
        observed = sum(counts.values())
        # Choose a proportional deficit. This converges to the stated mix in
        # collected decision steps, rather than merely counting episode starts.
        deficits = [(name, max(0.0, weight * (observed + 1) - counts[name]))
                    for name, weight in choices]
        weighted = choices if sum(value for _, value in deficits) <= 1e-12 else tuple(deficits)
        scenario = self._weighted(*weighted)
        return CurriculumEpisode(stage, scenario, frozenset(self._ball_ids_for(scenario)))

    def note_steps(self, stage: str, scenario: str, count: int = 1) -> None:
        if stage not in self.step_counts or scenario not in self.step_counts[stage]:
            raise ValueError(f"unknown curriculum sample {stage}/{scenario}")
        self.step_counts[stage][scenario] += int(count)

    def actual_proportions(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for stage, counts in self.step_counts.items():
            total = sum(counts.values())
            result[stage] = {name: (value / total if total else 0.0) for name, value in counts.items()}
        return result

    def _weighted(self, *choices: tuple[str, float]) -> str:
        point = self._rng.random() * sum(weight for _, weight in choices)
        total = 0.0
        for name, weight in choices:
            total += weight
            if point <= total:
                return name
        return choices[-1][0]

    def _ball_ids_for(self, scenario: str) -> tuple[str, ...]:
        small = self.config.small_ball_ids
        large = self.config.large_ball_ids
        if scenario == "single_small":
            return (self._rng.choice(small),)
        if scenario == "single_large" or scenario == "hold_large":
            return (self._rng.choice(large),)
        if scenario == "two_small":
            return tuple(self._rng.sample(list(small), 2))
        if scenario == "small_large_conflict":
            return (self._rng.choice(small), self._rng.choice(large))
        if scenario in {"large_both_ready", "large_both_approach", "large_one_holds"}:
            return (self._rng.choice(large),)
        if scenario == "small_then_large":
            return (self._rng.choice(small), self._rng.choice(large))
        if scenario == "dual_large":
            return tuple(large)
        if scenario == "dual_small":
            return tuple(self._rng.sample(list(small), 2))
        if scenario == "full":
            return tuple(self.config.ball_ids)
        raise ValueError(f"unknown Pong curriculum scenario: {scenario}")

    def state_dict(self) -> dict[str, object]:
        return {"rng": self._rng.getstate(), "settings": dict(self.settings),
                "step_counts": {stage: dict(values) for stage, values in self.step_counts.items()}}

    def restore_state(self, payload: Mapping[str, object]) -> None:
        self._rng.setstate(_tupleize(payload["rng"]))
        stored = payload.get("step_counts", {})
        for stage, counts in dict(stored).items():
            if stage in self.step_counts:
                for scenario, value in dict(counts).items():
                    if scenario in self.step_counts[stage]:
                        self.step_counts[stage][scenario] = int(value)


def _tupleize(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value
