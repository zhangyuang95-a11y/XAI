"""Fail-closed workload screening for diagnostic r4.1 warehouse scenes.

Every replacement task must remain in the robust conflict closure, must not
recreate a just-delivered task, and must place neither endpoint on either robot.
A scene is admitted to a manifest split only after replaying the exact
Actor/partner/branch workload that will consume that split.

This module is a generator/audit component.  It never selects or replaces an
Actor action and it is not imported by the participant runtime.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.warehouse_alignment_online_runtime import OnlineNumPyActor
from backend.warehouse_r41_diagnostic_online_runtime import (
    DEFAULT_REWARD_CONFIG,
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action
from env.warehouse_native.r41_conflict import R41ConflictSamplingError
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario


VERSION = "warehouse-r41-diagnostic-workload-screen.v2"
FROZEN_ACTOR_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
PARTNERS = ("skilled", "assertive", "noisy")
SPLIT_WORKLOADS = {
    "train": ("rcpd_collection",),
    "conflict_validation": ("rcpd_collection",),
    "final_test": ("explanation_final_audit",),
    "question_bank": ("question_bank_collection",),
    "tutorial": ("neutral_tutorial_choreography",),
}


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "frozen_actor_sha256": FROZEN_ACTOR_SHA256,
        "actor_role": "robot_2 deterministic raw argmax",
        "runtime_action_override": False,
        "strict_successor_failure_policy": "reject scene; never weaken or fall back",
        "replacement_endpoint_policy": (
            "pickup and delivery both clear of both post-motion robots; "
            "no immediate recreation"
        ),
        "split_workloads": {key: list(value) for key, value in SPLIT_WORKLOADS.items()},
        "rcpd_collection": {
            "partners": list(PARTNERS),
            "horizon": 120,
            "rng": "numpy.default_rng(41900000 + concatenated_fit_index*101 + partner_index)",
            "concatenated_fit_order": "train then conflict_validation",
            "intervention_anchor": "every fifth critical pre-action frame",
            "intervention_actions": list(ACTIONS),
            "branch_steps": 1,
        },
        "explanation_final_audit": {
            "partners": list(PARTNERS),
            "horizon": 120,
            "rng": "numpy.default_rng(17000 + partner_index*1000 + final_test_index)",
            "intervention_anchor": "every tenth critical pre-action frame",
            "intervention_actions": list(ACTIONS),
            "branch_steps": 1,
            "language_scenes": 10,
            "language_counterfactual": "after first skilled-partner transition, player WAIT for up to three steps",
        },
        "question_bank_collection": {
            "partner": "assertive",
            "horizon": 80,
            "rng": "random.Random(260910800 + question_bank_index)",
            "candidate_frames": "each nonterminal frame >= 1",
            "counterfactual": "player WAIT for up to three steps before the trajectory step",
        },
        "neutral_tutorial_choreography": {
            "source": "ui.warehouse_alignment_r41_diagnostic_tutorial.build_neutral_tutorial",
            "prelude_joint_actions": [
                ["RIGHT", "LEFT"],
                ["WAIT", "WAIT"],
                ["RIGHT", "WAIT"],
                ["WAIT", "UP"],
                ["UP", "DOWN"],
            ],
            "then": "robot_1 shortest physical pickup and delivery; robot_2 WAIT",
        },
    }


CONTRACT_SHA256 = digest(contract())


class WorkloadScreenFailure(RuntimeError):
    """A strict successor was unavailable in an exact consumer workload."""

    def __init__(self, evidence: Mapping[str, Any]):
        self.evidence = deepcopy(dict(evidence))
        super().__init__(str(self.evidence.get("error", "workload screen failed")))


def load_frozen_actor(path: str | Path) -> OnlineNumPyActor:
    path = Path(path).expanduser().resolve()
    if file_hash(path) != FROZEN_ACTOR_SHA256:
        raise ValueError("Workload screen requires the exact final two-million-step Actor")
    actor = OnlineNumPyActor(path)
    if (
        actor.obs_dim != 197
        or actor.metadata.get("runtime_action_override") is not False
        or actor.metadata.get("action_masks") is not False
        or tuple(actor.metadata.get("actions", ())) != tuple(ACTIONS)
    ):
        raise ValueError("Frozen Actor contract differs from diagnostic workload screen")
    return actor


def _environment(scene: Mapping[str, Any]) -> R41DiagnosticConflictWarehouseEnv:
    env = R41DiagnosticConflictWarehouseEnv(
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG)
    )
    reset_diagnostic_scenario(env, deepcopy(scene))
    return env


def _from_snapshot(snapshot: Mapping[str, Any]) -> R41DiagnosticConflictWarehouseEnv:
    env = R41DiagnosticConflictWarehouseEnv(
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG)
    )
    env.restore(deepcopy(snapshot), require_feedback=True)
    return env


def _policy_action(
    actor: OnlineNumPyActor, env: R41DiagnosticConflictWarehouseEnv
) -> str:
    before = digest(env.snapshot())
    actions, _ = actor.act(env.observations(), deterministic=True)
    if digest(env.snapshot()) != before:
        raise RuntimeError("Frozen Actor inference changed the workload source state")
    return str(actions["robot_2"])


def _step(
    actor: OnlineNumPyActor,
    env: R41DiagnosticConflictWarehouseEnv,
    player_action: str,
    *,
    workload: str,
    partner: str,
    branch_action: str | None = None,
) -> dict[str, Any]:
    actor_action = _policy_action(actor, env)
    source_frame = int(env.state.frame)
    try:
        _, _, terminated, truncated, info = env.step(
            {"robot_1": player_action, "robot_2": actor_action},
            decision_metadata={
                "policy_action": actor_action,
                "submitted_action": actor_action,
                "post_policy_overrides": 0,
            },
        )
    except R41ConflictSamplingError as error:
        raise WorkloadScreenFailure(
            {
                "workload": workload,
                "partner": partner,
                "frame": source_frame,
                "player_action": player_action,
                "actor_action": actor_action,
                "branch_action": branch_action,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ) from error
    if (
        info["requested_actions"]["robot_2"] != actor_action
        or info.get("decision_metadata", {}).get("post_policy_overrides", 0) != 0
    ):
        raise RuntimeError("Workload screen observed an Actor action override")
    created = info["r41_diagnostic_conflict"]["created"]
    endpoint_fields = (
        "new_pickup_on_agent",
        "new_delivery_on_agent",
        "new_endpoint_on_agent",
        "immediate_task_recreation",
    )
    created_counts = {
        field: sum(bool(row[field]) for row in created)
        for field in endpoint_fields
    }
    if any(created_counts.values()):
        raise RuntimeError(
            "Workload screen observed an invalid replacement endpoint or recreation"
        )
    return {
        "source_frame": source_frame,
        "after_frame": int(env.state.frame),
        "player_action": player_action,
        "actor_action": actor_action,
        "created": len(created),
        **created_counts,
        "done": bool(terminated or truncated),
    }


def _wait_branch(
    actor: OnlineNumPyActor,
    snapshot: Mapping[str, Any],
    *,
    steps: int,
    workload: str,
    partner: str,
) -> tuple[int, tuple[int, int] | None, dict[str, int]]:
    env = _from_snapshot(snapshot)
    start = tuple(env.state.by_id("robot_2").position)
    executed = 0
    replacement_counts = {
        "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0,
        "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0,
    }
    for _ in range(steps):
        if env.done:
            break
        transition = _step(
            actor, env, "WAIT", workload=workload, partner=partner,
            branch_action="WAIT",
        )
        for field in replacement_counts:
            replacement_counts[field] += transition[field]
        executed += 1
    finish = tuple(env.state.by_id("robot_2").position) if executed else None
    displacement = (
        None
        if finish is None
        else (int(finish[0] - start[0]), int(finish[1] - start[1]))
    )
    return executed, displacement, replacement_counts


def _trajectory_screen(
    actor: OnlineNumPyActor,
    scene: Mapping[str, Any],
    *,
    workload: str,
    scene_index: int,
    branch_modulus: int,
    rng_factory,
    language_counterfactual: bool = False,
) -> dict[str, Any]:
    episodes = trajectory_steps = branch_anchors = branch_steps = 0
    language_steps = actor_submission_frames = 0
    replacement_counts = {
        "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0,
        "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0,
    }
    for partner_index, partner in enumerate(PARTNERS):
        env = _environment(scene)
        rng = rng_factory(scene_index, partner_index)
        first_after: Mapping[str, Any] | None = None
        while not env.done:
            source = env.snapshot()
            groups = tuple(critical_groups(env, "robot_2"))
            if int(env.state.frame) % branch_modulus == 0 and groups:
                branch_anchors += 1
                for player_action in ACTIONS:
                    branch = _from_snapshot(source)
                    branch_transition = _step(
                        actor,
                        branch,
                        player_action,
                        workload=workload,
                        partner=partner,
                        branch_action=player_action,
                    )
                    for field in replacement_counts:
                        replacement_counts[field] += branch_transition[field]
                    branch_steps += 1
                    actor_submission_frames += 1
            before = digest(env.snapshot())
            player_action = partner_action(env, "robot_1", partner, rng)
            if digest(env.snapshot()) != before:
                raise RuntimeError("Workload partner changed the source state")
            transition = _step(
                actor, env, player_action, workload=workload, partner=partner
            )
            for field in replacement_counts:
                replacement_counts[field] += transition[field]
            trajectory_steps += 1
            actor_submission_frames += 1
            if first_after is None:
                first_after = env.snapshot()
        episodes += 1
        if (
            language_counterfactual
            and partner_index == 0
            and scene_index < 10
            and first_after is not None
        ):
            executed, _, branch_counts = _wait_branch(
                actor,
                first_after,
                steps=3,
                workload=workload,
                partner="skilled_language_wait_three",
            )
            for field in replacement_counts:
                replacement_counts[field] += branch_counts[field]
            language_steps += executed
            actor_submission_frames += executed
    return {
        "episodes": episodes,
        "trajectory_steps": trajectory_steps,
        "branch_anchors": branch_anchors,
        "branch_steps": branch_steps,
        "language_counterfactual_steps": language_steps,
        "actor_submission_frames": actor_submission_frames,
        "actor_action_override_frames": 0,
        **replacement_counts,
    }


def _screen_rcpd(
    actor: OnlineNumPyActor, scene: Mapping[str, Any], *, fit_index: int
) -> dict[str, Any]:
    return _trajectory_screen(
        actor,
        scene,
        workload="rcpd_collection",
        scene_index=fit_index,
        branch_modulus=5,
        rng_factory=lambda index, partner_index: np.random.default_rng(
            41_900_000 + index * 101 + partner_index
        ),
    )


def _screen_final(
    actor: OnlineNumPyActor, scene: Mapping[str, Any], *, scene_index: int
) -> dict[str, Any]:
    return _trajectory_screen(
        actor,
        scene,
        workload="explanation_final_audit",
        scene_index=scene_index,
        branch_modulus=10,
        rng_factory=lambda index, partner_index: np.random.default_rng(
            17_000 + partner_index * 1_000 + index
        ),
        language_counterfactual=True,
    )


def _screen_question_bank(
    actor: OnlineNumPyActor, scene: Mapping[str, Any], *, scene_index: int
) -> dict[str, Any]:
    env = _environment(scene)
    rng = random.Random(260_910_800 + scene_index)
    trajectory_steps = counterfactual_steps = candidate_frames = 0
    replacement_counts = {
        "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0,
        "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0,
    }
    next_actions: set[str] = set()
    wait_displacements: set[tuple[int, int]] = set()
    for _ in range(min(80, int(env.config.horizon))):
        if int(env.state.frame) >= 1 and not env.done:
            candidate_frames += 1
            next_actions.add(_policy_action(actor, env))
            executed, displacement, branch_counts = _wait_branch(
                actor,
                env.snapshot(),
                steps=3,
                workload="question_bank_collection",
                partner="assertive_wait_three",
            )
            for field in replacement_counts:
                replacement_counts[field] += branch_counts[field]
            counterfactual_steps += executed
            if executed == 3 and displacement is not None:
                wait_displacements.add(displacement)
        if env.done:
            break
        before = digest(env.snapshot())
        player_action = partner_action(env, "robot_1", "assertive", rng)
        if digest(env.snapshot()) != before:
            raise RuntimeError("Question-bank partner changed the source state")
        transition = _step(
            actor,
            env,
            player_action,
            workload="question_bank_collection",
            partner="assertive",
        )
        for field in replacement_counts:
            replacement_counts[field] += transition[field]
        trajectory_steps += 1
    return {
        "episodes": 1,
        "trajectory_steps": trajectory_steps,
        "candidate_frames": candidate_frames,
        "counterfactual_steps": counterfactual_steps,
        "actor_submission_frames": trajectory_steps + counterfactual_steps,
        "actor_action_override_frames": 0,
        **replacement_counts,
        "next_actions": sorted(next_actions, key=ACTIONS.index),
        "wait_displacements": [
            list(value) for value in sorted(wait_displacements)
        ],
    }


def _screen_tutorial(scene: Mapping[str, Any]) -> dict[str, Any]:
    # Use the same physical helpers and exact choreography as the final
    # tutorial builder, without importing its manifest validator recursively.
    from ui import warehouse_alignment_r41_tutorial as base

    env = _environment(scene)
    metrics = base._metrics(env)
    frames = [base._public_frame(env, metrics, {})]
    coverage = base._empty_coverage()
    prelude = (
        {"robot_1": "RIGHT", "robot_2": "LEFT"},
        {"robot_1": "WAIT", "robot_2": "WAIT"},
        {"robot_1": "RIGHT", "robot_2": "WAIT"},
        {"robot_1": "WAIT", "robot_2": "UP"},
        {"robot_1": "UP", "robot_2": "DOWN"},
    )
    try:
        for joint in prelude:
            metrics, _ = base._append_step(env, frames, coverage, metrics, joint)
        worker = env.state.by_id("robot_1")
        partner = env.state.by_id("robot_2")
        candidates = []
        for task in env.state.tasks:
            to_pickup = base._path_actions(
                env, worker.position, task.pickup_position,
                blocked=(partner.position,),
            )
            to_delivery = base._path_actions(
                env, task.pickup_position, task.delivery_position,
                blocked=(partner.position,),
            )
            candidates.append(
                (len(to_pickup) + len(to_delivery), task.task_id,
                 task, to_pickup, to_delivery)
            )
        _, task_id, task, to_pickup, _ = min(candidates, key=lambda row: row[:2])
        for action in to_pickup:
            metrics, _ = base._append_step(
                env, frames, coverage, metrics,
                {"robot_1": action, "robot_2": "WAIT"},
            )
        if env.state.by_id("robot_1").carrying_task_id != task_id:
            raise RuntimeError("Tutorial screen did not physically pick up its task")
        to_delivery = base._path_actions(
            env, env.state.by_id("robot_1").position, task.delivery_position,
            blocked=(env.state.by_id("robot_2").position,),
        )
        delivery_seen = False
        for action in to_delivery:
            metrics, info = base._append_step(
                env, frames, coverage, metrics,
                {"robot_1": action, "robot_2": "WAIT"},
            )
            delivery_seen |= any(
                event.get("event") == "delivery" and event.get("task_id") == task_id
                for event in info["events"]
            )
    except R41ConflictSamplingError as error:
        raise WorkloadScreenFailure(
            {
                "workload": "neutral_tutorial_choreography",
                "partner": "scripted_ai_ai",
                "frame": int(env.state.frame),
                "player_action": None,
                "actor_action": None,
                "branch_action": None,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        ) from error
    if not delivery_seen or any(not coverage[name] for name in base._COVERAGE_FIELDS):
        raise RuntimeError("Tutorial workload coverage differs from the final builder")
    return {
        "episodes": 1,
        "trajectory_steps": len(frames) - 1,
        "coverage": deepcopy(coverage),
        "coverage_sha256": digest(coverage),
        "actor_submission_frames": 0,
        "actor_action_override_frames": 0,
        "new_pickup_on_agent": 0,
        "new_delivery_on_agent": 0,
        "new_endpoint_on_agent": 0,
        "immediate_task_recreation": 0,
    }


def screen_scene(
    scene: Mapping[str, Any],
    *,
    split: str,
    scene_index: int,
    actor: OnlineNumPyActor | None,
    train_count: int = 128,
) -> dict[str, Any]:
    """Replay an exact split workload and return a compact immutable receipt.

    Failed receipts are returned rather than raised so the deterministic scene
    generator can reject the seed while preserving aggregate failure evidence.
    Unexpected implementation/contract errors still raise.
    """
    if split not in SPLIT_WORKLOADS:
        raise ValueError("Unknown diagnostic workload split")
    if type(scene_index) is not int or scene_index < 0:
        raise ValueError("Diagnostic workload scene index must be non-negative")
    if split != "tutorial" and actor is None:
        raise ValueError("Actor-backed workload screen requires the frozen Actor")
    if actor is not None and actor.artifact_sha256 != FROZEN_ACTOR_SHA256:
        raise ValueError("Workload screen Actor bytes differ")
    try:
        if split == "train":
            metrics = _screen_rcpd(actor, scene, fit_index=scene_index)
        elif split == "conflict_validation":
            metrics = _screen_rcpd(
                actor, scene, fit_index=train_count + scene_index
            )
        elif split == "final_test":
            metrics = _screen_final(actor, scene, scene_index=scene_index)
        elif split == "question_bank":
            metrics = _screen_question_bank(actor, scene, scene_index=scene_index)
        else:
            metrics = _screen_tutorial(scene)
    except WorkloadScreenFailure as error:
        failure = deepcopy(error.evidence)
        receipt = {
            "version": VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "frozen_actor_sha256": (
                FROZEN_ACTOR_SHA256 if actor is not None else None
            ),
            "scene_fingerprint": scene.get("fingerprint"),
            "split": split,
            "scene_index": scene_index,
            "passed": False,
            "failure": failure,
            "metrics": None,
        }
        receipt["receipt_sha256"] = digest(receipt)
        return receipt
    receipt = {
        "version": VERSION,
        "contract_sha256": CONTRACT_SHA256,
        "frozen_actor_sha256": FROZEN_ACTOR_SHA256 if actor is not None else None,
        "scene_fingerprint": scene.get("fingerprint"),
        "split": split,
        "scene_index": scene_index,
        "passed": True,
        "failure": None,
        "metrics": metrics,
    }
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    scene: Mapping[str, Any],
    split: str,
    scene_index: int,
) -> None:
    expected_actor = None if split == "tutorial" else FROZEN_ACTOR_SHA256
    copy = deepcopy(dict(receipt))
    claimed = copy.pop("receipt_sha256", None)
    if (
        claimed != digest(copy)
        or receipt.get("version") != VERSION
        or receipt.get("contract_sha256") != CONTRACT_SHA256
        or receipt.get("frozen_actor_sha256") != expected_actor
        or receipt.get("scene_fingerprint") != scene.get("fingerprint")
        or receipt.get("split") != split
        or receipt.get("scene_index") != scene_index
        or receipt.get("passed") is not True
        or receipt.get("failure") is not None
        or not isinstance(receipt.get("metrics"), Mapping)
        or receipt["metrics"].get("actor_action_override_frames") != 0
        or receipt["metrics"].get("new_pickup_on_agent") != 0
        or receipt["metrics"].get("new_delivery_on_agent") != 0
        or receipt["metrics"].get("new_endpoint_on_agent") != 0
        or receipt["metrics"].get("immediate_task_recreation") != 0
    ):
        raise ValueError("Diagnostic scene workload receipt differs")


def replay_and_compare(
    scene: Mapping[str, Any],
    *,
    split: str,
    scene_index: int,
    actor: OnlineNumPyActor | None,
    train_count: int = 128,
) -> dict[str, Any]:
    saved = scene.get("workload_screen")
    validate_receipt(
        saved, scene=scene, split=split, scene_index=scene_index
    )
    replayed = screen_scene(
        scene,
        split=split,
        scene_index=scene_index,
        actor=actor,
        train_count=train_count,
    )
    if replayed != saved:
        raise ValueError("Diagnostic workload screen did not replay exactly")
    return replayed


def source_sha256() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "VERSION",
    "FROZEN_ACTOR_SHA256",
    "PARTNERS",
    "SPLIT_WORKLOADS",
    "CONTRACT_SHA256",
    "WorkloadScreenFailure",
    "contract",
    "load_frozen_actor",
    "screen_scene",
    "validate_receipt",
    "replay_and_compare",
    "source_sha256",
]
