"""Deterministic calibration of the four parallel Task 1/Task 2 seed pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .contracts import (
    ACTION_EXECUTION_VERSION,
    RUNTIME_CONTROLLER,
    SEED_LIBRARY_VERSION,
)
from .environment import WarehouseMultiAgentEnv, shortest_path_distance
from .policy import MAPPOPolicy


@dataclass(frozen=True)
class SeedBaseline:
    seed: int
    initial_path_workload: int
    ai_ai_deliveries: int
    ai_ai_score: float
    shutdowns: int
    robot_collisions: int


@dataclass(frozen=True)
class ParallelSeedPair:
    form_id: int
    task1_seed: int
    task2_seed: int
    task1: SeedBaseline
    task2: SeedBaseline

    @property
    def delivery_gap(self) -> int:
        return abs(self.task2.ai_ai_deliveries - self.task1.ai_ai_deliveries)

    @property
    def score_gap(self) -> float:
        return abs(self.task2.ai_ai_score - self.task1.ai_ai_score)

    @property
    def workload_gap(self) -> int:
        return abs(
            self.task2.initial_path_workload
            - self.task1.initial_path_workload
        )


def _initial_workload(environment: WarehouseMultiAgentEnv) -> int:
    state = environment.get_state()
    agents = sorted(state.agents, key=lambda item: item.agent_id)
    tasks = sorted(state.tasks, key=lambda item: item.task_id)
    if len(agents) != 2 or len(tasks) != 2:
        raise ValueError("Seed calibration requires two robots and two active jobs.")

    def cost(agent_index: int, task_index: int) -> int:
        agent = agents[agent_index]
        task = tasks[task_index]
        return shortest_path_distance(
            agent.position,
            task.pickup_position,
        ) + shortest_path_distance(
            task.pickup_position,
            task.delivery_position,
        )

    return min(cost(0, 0) + cost(1, 1), cost(0, 1) + cost(1, 0))


def evaluate_ai_ai_seed(policy: MAPPOPolicy, seed: int) -> SeedBaseline:
    inference = policy.fork_for_inference(seed=int(seed))
    environment = WarehouseMultiAgentEnv(inference.environment_config)
    observations, _ = environment.reset(seed=int(seed))
    workload = _initial_workload(environment)
    while True:
        actions, _ = inference.act(
            observations,
            environment.global_state(),
            deterministic=True,
        )
        observations, _, terminated, truncated, _ = environment.step(actions)
        if terminated or truncated:
            break
    state = environment.get_state()
    return SeedBaseline(
        seed=int(seed),
        initial_path_workload=int(workload),
        ai_ai_deliveries=int(state.total_deliveries),
        ai_ai_score=float(state.user_score),
        shutdowns=int(state.shutdown_count),
        robot_collisions=int(state.robot_collision_events),
    )


def calibrate_parallel_seed_pairs(
    policy: MAPPOPolicy,
    candidate_seeds: Iterable[int],
    *,
    pair_count: int = 4,
    maximum_delivery_gap: int = 1,
    maximum_score_gap: float = 50.0,
) -> tuple[ParallelSeedPair, ...]:
    baselines = [
        evaluate_ai_ai_seed(policy, seed)
        for seed in sorted({int(value) for value in candidate_seeds})
    ]
    candidates: list[tuple[tuple[float, ...], SeedBaseline, SeedBaseline]] = []
    for index, left in enumerate(baselines):
        for right in baselines[index + 1 :]:
            delivery_gap = abs(left.ai_ai_deliveries - right.ai_ai_deliveries)
            score_gap = abs(left.ai_ai_score - right.ai_ai_score)
            if delivery_gap > maximum_delivery_gap or score_gap > maximum_score_gap:
                continue
            workload_gap = abs(
                left.initial_path_workload - right.initial_path_workload
            )
            quality = (
                float(delivery_gap),
                float(score_gap),
                float(workload_gap),
                float(left.shutdowns + right.shutdowns),
                float(left.robot_collisions + right.robot_collisions),
                float(left.seed),
                float(right.seed),
            )
            candidates.append((quality, left, right))
    candidates.sort(key=lambda item: item[0])
    used: set[int] = set()
    selected: list[ParallelSeedPair] = []
    for _, left, right in candidates:
        if left.seed in used or right.seed in used:
            continue
        selected.append(
            ParallelSeedPair(
                form_id=len(selected),
                task1_seed=left.seed,
                task2_seed=right.seed,
                task1=left,
                task2=right,
            )
        )
        used.update((left.seed, right.seed))
        if len(selected) == int(pair_count):
            break
    if len(selected) != int(pair_count):
        raise RuntimeError(
            "Could not calibrate four disjoint seed pairs within the "
            f"delivery-gap <= {maximum_delivery_gap} and score-gap <= "
            f"{maximum_score_gap:g} constraints."
        )
    return tuple(selected)


def save_parallel_seed_library(
    path: str | Path,
    pairs: Sequence[ParallelSeedPair],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SEED_LIBRARY_VERSION,
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "runtime_controller": RUNTIME_CONTROLLER,
        "rollout_action_source": "mappo_actor",
        "post_policy_action_interventions": 0,
        "maximum_delivery_gap": 1,
        "maximum_score_gap": 50.0,
        "pairs": [
            {
                **{
                    key: value
                    for key, value in asdict(pair).items()
                    if key not in {"task1", "task2"}
                },
                "task1": asdict(pair.task1),
                "task2": asdict(pair.task2),
                "delivery_gap": pair.delivery_gap,
                "score_gap": pair.score_gap,
                "workload_gap": pair.workload_gap,
            }
            for pair in pairs
        ],
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def load_parallel_seed_library(
    path: str | Path,
) -> tuple[ParallelSeedPair, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != SEED_LIBRARY_VERSION:
        raise ValueError("Incompatible collaborative parallel-seed library.")
    if payload.get("action_execution_version") != ACTION_EXECUTION_VERSION:
        raise ValueError("Incompatible collaborative action-execution contract.")
    if payload.get("runtime_controller") != RUNTIME_CONTROLLER:
        raise ValueError("Incompatible collaborative runtime controller.")
    if payload.get("rollout_action_source") != "mappo_actor":
        raise ValueError("Parallel seeds were not calibrated by the MAPPO Actor.")
    if int(payload.get("post_policy_action_interventions", -1)) != 0:
        raise ValueError("Parallel seeds contain post-policy action interventions.")
    pairs: list[ParallelSeedPair] = []
    for raw in payload.get("pairs", ()):  # type: ignore[union-attr]
        pair = ParallelSeedPair(
            form_id=int(raw["form_id"]),
            task1_seed=int(raw["task1_seed"]),
            task2_seed=int(raw["task2_seed"]),
            task1=SeedBaseline(**raw["task1"]),
            task2=SeedBaseline(**raw["task2"]),
        )
        if pair.delivery_gap > 1 or pair.score_gap > 50.0:
            raise ValueError("Parallel seed pair violates preregistered balance limits.")
        pairs.append(pair)
    if len(pairs) != 4 or {pair.form_id for pair in pairs} != set(range(4)):
        raise ValueError("The parallel-seed library must contain forms 0 through 3.")
    seeds = [
        seed
        for pair in pairs
        for seed in (pair.task1_seed, pair.task2_seed)
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Parallel task seeds must be unique across all forms.")
    return tuple(sorted(pairs, key=lambda item: item.form_id))
