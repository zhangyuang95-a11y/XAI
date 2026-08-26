"""Verified, condition-blind tutorial material from the real warehouse policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from backend.adapters.base import RolloutFrame
from backend.adapters.warehouse import WAREHOUSE_PROGRAM_VERSION, WarehouseAdapter
from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    REFERENCE_TRAJECTORY_FORMAT,
    RUNTIME_CONTROLLER,
)
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOPolicy


TUTORIAL_SEED = 40_221
@dataclass(frozen=True)
class TutorialTrajectory:
    """One uninterrupted, successful warehouse mission used for onboarding."""

    seed: int
    focus_agent: str
    frames: tuple[RolloutFrame, ...]
    milestones: tuple[tuple[str, int, str], ...]


def build_verified_tutorial(
    policy: MAPPOPolicy,
    *,
    seed: int = TUTORIAL_SEED,
) -> TutorialTrajectory:
    """Generate and validate one complete real-policy tutorial mission.

    The browser receives every consecutive state from reset through mission
    success.  Absence of a required semantic event, a discontinuity, or an
    incomplete mission is a startup error.  There is deliberately no fallback
    scene because that would silently change what participants learn.
    """

    inference_policy = policy.fork_for_inference(seed=seed)
    environment = WarehouseMultiAgentEnv(
        replace(
            inference_policy.environment_config,
            participant_detour_scoring=False,
        )
    )
    environment.reset(seed=seed)
    # The standardized demonstration deliberately includes a low-battery
    # teammate so the shared-charger behavior is observable within 120 steps.
    demo_state = environment.get_state()
    demo_state.by_id("robot_2").battery = 35.0
    environment.set_state(demo_state)
    rollout = WarehouseAdapter(environment).rollout(
        inference_policy,
        horizon=environment.config.horizon,
        deterministic=True,
    )
    frames = tuple(rollout.frames)
    _validate_continuous_mission(frames, terminal_reason=rollout.terminal_reason)

    pickup = _first_transition(frames, "pickup")
    delivery = _first_transition(frames, "delivery")
    delivering_agents = {
        item[1]
        for item in (
            _first_transition_for_agent(frames, "delivery", "robot_1"),
            _first_transition_for_agent(frames, "delivery", "robot_2"),
        )
        if item is not None
    }
    coordination = _first_coordination(frames)
    charge_start, _, charge_agent = _charging_sequence(frames)
    required = {
        "pickup": pickup,
        "delivery": delivery,
        "coordination": coordination,
        "charging": (
            (charge_start, charge_agent)
            if charge_start is not None and charge_agent is not None
            else None
        ),
        "both_robots_deliver": (
            (0, "robot_2") if delivering_agents == {"robot_1", "robot_2"} else None
        ),
    }
    missing = tuple(key for key, value in required.items() if value is None)
    if missing:
        raise RuntimeError(
            "Tutorial seed does not contain required verified events: "
            + ", ".join(missing)
        )

    assert pickup is not None and delivery is not None and coordination is not None
    assert charge_start is not None and charge_agent is not None
    milestones = (
        ("pickup", pickup[0], pickup[1]),
        ("delivery", delivery[0], delivery[1]),
        ("coordination", coordination[0], coordination[1]),
        ("charging", charge_start, charge_agent),
        ("demonstration_complete", len(frames) - 1, charge_agent),
    )
    return TutorialTrajectory(
        seed=int(seed),
        focus_agent=charge_agent,
        frames=frames,
        milestones=milestones,
    )


def reference_event_frames(
    trajectory: TutorialTrajectory,
) -> dict[str, list[int]]:
    mapping = {
        "claimed": "pickup",
        "delivered": "delivery",
        "charging": "charging",
        "charger_queue": "charger_queue",
        "coordination_yield": "coordination_yield",
        "head_on_conflict_risk": "head_on_conflict_risk",
    }
    result = {value: [] for value in mapping.values()}
    for index, frame in enumerate(trajectory.frames, start=1):
        for event in frame.environment_events:
            label = mapping.get(str(event.get("event", "")))
            if label is not None:
                result[label].append(index)
    return result


def calibrate_reference_trajectory(
    policy: MAPPOPolicy,
    *,
    start_seed: int = TUTORIAL_SEED,
    maximum_candidates: int = 2_000,
) -> TutorialTrajectory:
    """Return the first non-shutdown seed containing every v5 teaching event."""

    required = {
        "pickup", "delivery", "charging", "charger_queue",
        "coordination_yield", "head_on_conflict_risk",
    }
    failures: list[str] = []
    for seed in range(int(start_seed), int(start_seed) + int(maximum_candidates)):
        try:
            trajectory = build_verified_tutorial(policy, seed=seed)
        except RuntimeError as exc:
            if len(failures) < 5:
                failures.append(f"{seed}: {exc}")
            continue
        frames = reference_event_frames(trajectory)
        present = {name for name, indices in frames.items() if indices}
        shutdown = any(
            str(frame.info.get("terminal_reason", "")) == "battery_shutdown"
            or any(
                str(event.get("event", "")) == "battery_shutdown"
                for event in frame.environment_events
            )
            for frame in trajectory.frames
        )
        charger_return_cycle = any(
            any(
                str(event.get("event", "")) == "charger_return_cycle"
                for event in frame.environment_events
            )
            for frame in trajectory.frames
        )
        task_starvation = any(
            bool(frame.info.get("starving_task_ids", ()))
            for frame in trajectory.frames
        )
        detour_penalty = any(
            any(float(value) > 0.0 for value in frame.info.get("route_regret", {}).values())
            for frame in trajectory.frames
        )
        if (
            required.issubset(present)
            and not shutdown
            and not charger_return_cycle
            and not task_starvation
            and not detour_penalty
        ):
            return trajectory
    detail = "; ".join(failures)
    raise RuntimeError(
        f"No eligible v5 reference trajectory was found in {maximum_candidates} seeds."
        + (f" First failures: {detail}" if detail else "")
    )


def save_reference_trajectory_manifest(
    path: str | Path,
    trajectory: TutorialTrajectory,
    policy: MAPPOPolicy,
) -> Path:
    events = reference_event_frames(trajectory)
    identity = {
        "format": REFERENCE_TRAJECTORY_FORMAT,
        "seed": int(trajectory.seed),
        "frame_count": len(trajectory.frames) + 1,
        "model_version": policy.model_version,
        "environment_version": WarehouseMultiAgentEnv.environment_name,
        "warehouse_program_version": WAREHOUSE_PROGRAM_VERSION,
        "map_layout_id": policy.environment_config.map_layout_id,
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "runtime_controller": RUNTIME_CONTROLLER,
        "rollout_action_source": "mappo_actor",
        "post_policy_action_interventions": 0,
        "agent_control": {"robot_1": "ai", "robot_2": "ai"},
        "event_frames": events,
        "battery_shutdown": False,
        "charger_departure_return_cycles": 0,
        "task_starvation_events": 0,
        "participant_detour_scoring": False,
        "frozen": True,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {**identity, "trajectory_manifest_hash": sha256(encoded).hexdigest()}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def validate_tutorial_seed_isolated(
    tutorial_seed: int,
    *,
    task1_seeds: Sequence[int],
    task2_seeds: Sequence[int],
) -> None:
    """Reject overlap between the fixed demonstration and scored rounds."""

    scored = {
        *(int(seed) for seed in task1_seeds),
        *(int(seed) for seed in task2_seeds),
    }
    if int(tutorial_seed) in scored:
        raise RuntimeError(
            f"Tutorial seed {tutorial_seed} overlaps a scored study round seed."
        )


def _validate_continuous_mission(
    frames: Sequence[RolloutFrame],
    *,
    terminal_reason: str | None,
) -> None:
    if not frames:
        raise RuntimeError("Verified tutorial trajectory is empty.")
    first = frames[0]
    if first.snapshot.frame != 0:
        raise RuntimeError("Tutorial must begin at the environment reset frame.")
    episode_id = first.snapshot.state.episode_id
    expected_frame = 0
    for frame in frames:
        if frame.next_snapshot is None:
            raise RuntimeError("Tutorial contains a transition without its next state.")
        if frame.snapshot.state.episode_id != episode_id:
            raise RuntimeError("Tutorial changes episode before the mission ends.")
        if frame.snapshot.frame != expected_frame:
            raise RuntimeError("Tutorial contains a skipped or repeated decision frame.")
        if frame.next_snapshot.frame != expected_frame + 1:
            raise RuntimeError("Tutorial contains a discontinuous state transition.")
        if frame.next_snapshot.state.episode_id != episode_id:
            raise RuntimeError("Tutorial transition crosses into another episode.")
        expected_frame += 1
    if terminal_reason not in {"horizon", "battery_shutdown"} or not frames[-1].done:
        raise RuntimeError("Tutorial trajectory does not finish a complete scored round.")
    final_state = frames[-1].next_snapshot.state
    if not (final_state.terminated or final_state.truncated):
        raise RuntimeError("Tutorial round did not reach a terminal boundary.")


def _agent_map(frame: RolloutFrame, *, next_state: bool = False):
    snapshot = frame.next_snapshot if next_state else frame.snapshot
    if snapshot is None:
        return {}
    return {agent.agent_id: agent for agent in snapshot.state.agents}


def _first_transition(
    frames: Sequence[RolloutFrame],
    kind: str,
) -> tuple[int, str] | None:
    for index, frame in enumerate(frames):
        before = _agent_map(frame)
        after = _agent_map(frame, next_state=True)
        for agent_id in sorted(before.keys() & after.keys()):
            if (
                kind == "pickup"
                and before[agent_id].carrying_task_id is None
                and after[agent_id].carrying_task_id is not None
            ):
                return index, agent_id
            if kind == "delivery" and after[agent_id].deliveries_completed > before[agent_id].deliveries_completed:
                return index, agent_id
    return None


def _first_transition_for_agent(
    frames: Sequence[RolloutFrame],
    kind: str,
    requested_agent: str,
) -> tuple[int, str] | None:
    result = _first_transition(
        tuple(
            frame
            for frame in frames
            if requested_agent in _agent_map(frame)
        ),
        kind,
    )
    if result is None or result[1] != requested_agent:
        for index, frame in enumerate(frames):
            before = _agent_map(frame)
            after = _agent_map(frame, next_state=True)
            if requested_agent not in before or requested_agent not in after:
                continue
            if (
                kind == "delivery"
                and after[requested_agent].deliveries_completed
                > before[requested_agent].deliveries_completed
            ):
                return index, requested_agent
        return None
    return result


def _first_coordination(
    frames: Sequence[RolloutFrame],
) -> tuple[int, str] | None:
    for index, frame in enumerate(frames):
        for event in frame.environment_events:
            if event.get("event") == "coordination_yield":
                return index, str(event.get("yielding_agent_id", "robot_1"))
        before = _agent_map(frame)
        if len(before) == 2:
            left, right = (before[key] for key in sorted(before))
            distance = abs(left.position[0] - right.position[0]) + abs(
                left.position[1] - right.position[1]
            )
            if distance <= 2:
                for agent_id in sorted(before):
                    other_id = next(key for key in before if key != agent_id)
                    if (
                        frame.executed_actions.get(agent_id) == "WAIT"
                        and frame.executed_actions.get(other_id) != "WAIT"
                    ):
                        return index, agent_id
    return None


def _charging_sequence(
    frames: Sequence[RolloutFrame],
) -> tuple[int | None, int | None, str | None]:
    for start, frame in enumerate(frames):
        before = _agent_map(frame)
        after = _agent_map(frame, next_state=True)
        charging = next(
            (
                agent_id
                for agent_id in sorted(before.keys() & after.keys())
                if after[agent_id].battery > before[agent_id].battery
            ),
            None,
        )
        if charging is None:
            continue
        end = start
        while end + 1 < len(frames):
            state = _agent_map(frames[end + 1])
            if charging not in state or state[charging].goal_kind != "charge":
                break
            end += 1
        return start, min(len(frames) - 1, end + 2), charging
    return None, None, None
