"""Domain state and immutable configuration for collaborative delivery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .layouts import DEFAULT_MAP_LAYOUT, STUDY_MAP_LAYOUT, get_map_layout
from .rewards import RewardConfig


@dataclass(frozen=True)
class WarehouseConfig:
    """Immutable contract for the collaborative experiment environment."""

    rows: int = DEFAULT_MAP_LAYOUT.rows
    cols: int = DEFAULT_MAP_LAYOUT.cols
    map_layout_id: str = DEFAULT_MAP_LAYOUT.layout_id
    num_agents: int = 2
    max_agents: int = 2
    horizon: int = 120
    active_task_count: int = 2
    minimum_task_distance: int = 4
    human_agent_id: str = "robot_1"
    delivery_points: float = 100.0
    robot_collision_points: float = -200.0
    shutdown_points: float = -50.0
    step_points: float = -1.0
    human_detour_points_per_unit: float = -2.0
    participant_detour_scoring: bool = True
    move_battery_cost: float = 2.0
    charge_per_wait: float = 10.0
    battery_safety_margin: float = 2.0
    # Expected clearance for one ordinary two-robot passing manoeuvre. This
    # is energy planning, not an action rule: it keeps a neural robot from
    # leaving the charger with a route that is safe only in an empty map.
    coordination_energy_reserve_steps: float = 2.0
    charge_release_hysteresis_steps: float = 2.0
    local_patch_radius: int = 2
    # Offline-supervision ablation only; the environment never invokes the
    # teacher and deployed Actor actions remain direct in either setting.
    teacher_efficiency_guard_enabled: bool = True
    seed: int = 2026
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        layout = get_map_layout(self.map_layout_id)
        if (self.rows, self.cols) != (layout.rows, layout.cols):
            raise ValueError("rows and cols must match the selected map layout.")
        if self.num_agents != 2 or self.max_agents != 2:
            raise ValueError(
                "The collaborative environment requires num_agents=max_agents=2."
            )
        if self.horizon < 1:
            raise ValueError("horizon must be positive.")
        if self.active_task_count != 2:
            raise ValueError(
                "The collaborative study requires exactly two active tasks."
            )
        if self.minimum_task_distance < 1:
            raise ValueError("minimum_task_distance must be positive.")
        if self.human_agent_id != "robot_1":
            raise ValueError("Participants must control robot_1.")
        if self.move_battery_cost != 2.0:
            raise ValueError(
                "The collaborative study requires a movement battery cost of 2."
            )
        if self.charge_per_wait <= 0:
            raise ValueError("The charging rate must be positive.")
        if self.battery_safety_margin < 0:
            raise ValueError("battery_safety_margin cannot be negative.")
        if self.coordination_energy_reserve_steps < 0:
            raise ValueError(
                "coordination_energy_reserve_steps cannot be negative."
            )
        if self.charge_release_hysteresis_steps < 0:
            raise ValueError("charge_release_hysteresis_steps cannot be negative.")

    @property
    def mission_reserve_steps(self) -> float:
        """Post-route safety plus one expected coordination clearance."""

        return float(
            self.battery_safety_margin
            + self.coordination_energy_reserve_steps
        )


def collaborative_study_config(**overrides: Any) -> WarehouseConfig:
    """Return the production 120-step staggered-aisle configuration."""

    values: dict[str, Any] = {
        "rows": STUDY_MAP_LAYOUT.rows,
        "cols": STUDY_MAP_LAYOUT.cols,
        "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
        # Preserve the calibrated reserve until the new staggered-map policy
        # evaluation proves that a different value is safer and more useful.
        "battery_safety_margin": 4.0,
    }
    values.update(overrides)
    return WarehouseConfig(**values)


@dataclass
class DeliveryTask:
    """A shared A-to-B task that is unowned until a robot reaches A."""

    task_id: str
    pickup_position: tuple[int, int]
    delivery_position: tuple[int, int]
    status: str = "available"
    carrier_agent_id: str | None = None
    created_frame: int = 0
    claimed_frame: int | None = None
    delivered_frame: int | None = None
    # Training/evaluation audit fields.  They never enter participant score or
    # task ownership and remain optional for archived scenario fixtures.
    claimed_battery: float | None = None
    shortest_safe_delivery_steps: float | None = None
    # Path-efficiency provenance.  The denominator starts with the exact
    # claim-time safe plan and may grow only for transition-audited mandatory
    # clearance work.  These fields never enter the Actor observation.
    safe_path_charge_planned: bool | None = None
    safe_path_clearance_extension_steps: float = 0.0
    safe_path_clearance_energy_budget: float = 0.0
    safe_path_unplanned_charge_active: bool = False
    safe_path_unplanned_charge_extension_steps: float = 0.0

    @property
    def active(self) -> bool:
        return self.status in {"available", "carried"}


@dataclass
class AgentState:
    """Robot state; cargo ownership is represented only by a task ID."""

    agent_id: str
    position: tuple[int, int]
    battery: float = 100.0
    carrying_task_id: str | None = None
    # A non-binding intent inferred from the robot's own executed movement.
    # It does not reserve a package: ownership still begins only at A.
    route_commitment_task_id: str | None = None
    active: bool = True
    heading: str = "UP"
    last_action: str = "WAIT"
    last_executed_action: str = "WAIT"
    last_battery_delta: float = 0.0
    steps_since_charging: int = 0
    charger_wait_streak: int = 0
    charge_mode_active: bool = False
    # Consecutive WAITs for which a collision-free, legal step toward the
    # transition-frozen mission existed.  This is separate from charging and
    # joint-stall memory so necessary waits never accumulate a penalty.
    avoidable_wait_streak: int = 0
    last_charger_departure_frame: int | None = None
    deliveries_at_last_charger_departure: int = 0
    team_deliveries_at_last_charger_departure: int = 0
    carrying_task_at_last_charger_departure: str | None = None
    # A public two-phase charger handoff can require this robot to vacate the
    # station even when the participant later deviates.  Persist that causal
    # evidence across the short absence so a safe recovery is not mislabeled
    # as an unproductive charger oscillation.
    last_charger_departure_was_coordination: bool = False
    last_charger_departure_plan_id: str | None = None
    deliveries_completed: int = 0
    navigation_goal_kind: str = "pickup"
    navigation_goal_position: tuple[int, int] = (0, 0)
    # Persistent, human-auditable decision state. Goal identity and age are
    # derived before the next joint decision and are therefore safe to expose
    # to both Actors without leaking either current action.
    goal_type: str = "SELECT_TASK"
    goal_id: str | None = None
    goal_since: int = 0
    goal_switch_reason: str = "episode_reset"
    charging_reason: str | None = None
    yielding_plan_id: str | None = None
    recent_positions: tuple[tuple[int, int], ...] = ()
    recent_goal_types: tuple[str, ...] = ()

    @property
    def goal_kind(self) -> str:
        return self.navigation_goal_kind

    @property
    def goal_position(self) -> tuple[int, int]:
        return self.navigation_goal_position


def empty_score_breakdown() -> dict[str, float]:
    return {
        "delivery": 0.0,
        "robot_collision": 0.0,
        "shutdown": 0.0,
        "time": 0.0,
        "human_detour": 0.0,
    }


@dataclass
class WarehouseState:
    episode_id: int
    frame: int
    agents: list[AgentState]
    tasks: list[DeliveryTask]
    completed_tasks: list[DeliveryTask] = field(default_factory=list)
    next_task_index: int = 1
    total_deliveries: int = 0
    collision_count: int = 0
    shutdown_count: int = 0
    terminated: bool = False
    truncated: bool = False
    terminal_reason: str | None = None
    last_rewards: dict[str, float] = field(default_factory=dict)
    user_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=empty_score_breakdown)
    human_route_regret_units: float = 0.0
    robot_collision_events: int = 0
    invalid_move_count: int = 0
    last_robot_collision_event: bool = False
    last_robot_collision_kind: str | None = None
    last_coordination_events: tuple[dict[str, Any], ...] = ()
    ineffective_joint_wait_streak: int = 0
    # A frozen multi-frame clearance contract.  The first transition clears
    # the occupied route cell; the second lets the original priority robot
    # enter it.  Persisting the contract prevents priority from flipping when
    # the geometry changes after the clearance move.
    active_coordination_plan: dict[str, Any] | None = None
    coordination_plan_cooldown_until: int = 0
    # Episode-level control provenance is known before a decision.  It never
    # contains the participant's current action.
    participant_controlled_agent_id: str | None = None

    def by_id(self, agent_id: str) -> AgentState:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        raise KeyError(agent_id)

    def task_by_id(self, task_id: str) -> DeliveryTask:
        for task in (*self.tasks, *self.completed_tasks):
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)
