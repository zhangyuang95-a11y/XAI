"""Warehouse r4.1 continuation on the shared high-conflict task dynamics.

This is a fresh child of the frozen 3.95M r3 learner.  It never loads an r4
diagnostic Actor.  The shared conflict environment owns initial and successor
task generation for training, validation, and online play.  RCPD supplies only
a bounded training regularizer and post-hoc evidence; every environment action
is the sampled neural command.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse_native.feedback import FeedbackManager
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor
from . import warehouse_r4_active_trainer as r4
from .warehouse_native import gae
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_partner_mix_trainer import _prefix
from .warehouse_native_public_feedback_evaluation import REWARD as TRAINING_REWARD
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import (
    COUNTERS, EPISODE_FIELDS, PublicFeedbackTrainer, _cpu,
)


VERSION = "warehouse-r41-active-trainer.v1"
SOURCE_ROOT = r4.SOURCE_ROOT
SOURCE_ACTOR_SHA256 = r4.SOURCE_ACTOR_SHA256
SOURCE_CHECKPOINT_SHA256 = r4.SOURCE_CHECKPOINT_SHA256
SOURCE_CUMULATIVE_STEPS = r4.SOURCE_CUMULATIVE_STEPS
SOURCE_ACTOR_PATH = ROOT / SOURCE_ROOT / "branches/feedback/actors/actor_0050000.npz"
MAXIMUM_ADDITIONAL_JOINT_STEPS = 2_000_000
BOUNDARY_INTERVAL = 50_000

PARTNER_MIX = {"selfplay": .20, "skilled": .20, "assertive": .50, "noisy": .10}
SHAPING = deepcopy(r4.SHAPING)
KL_MAX = r4.KL_MAX
TRAINING_KL_LAMBDA = r4.TRAINING_KL_LAMBDA
ACTOR_LR = r4.ACTOR_LR
CRITIC_LR = r4.CRITIC_LR
ENTROPY_COEFFICIENT = r4.ENTROPY_COEFFICIENT
TREE_CONFIG = r4.TREE_CONFIG
RESERVOIR_ROWS = r4.RESERVOIR_ROWS

ORDINARY_EPISODE_PROBABILITY = .80
REACHABLE_ENERGY_EPISODE_PROBABILITY = .20
PROGRAM_PARTNER_PROBABILITY = 1. - PARTNER_MIX["selfplay"]
CONDITIONAL_ENERGY_PROBABILITY = (
    REACHABLE_ENERGY_EPISODE_PROBABILITY / PROGRAM_PARTNER_PROBABILITY)
ENERGY_CURRICULUM_VERSION = "warehouse-r41-public-reachable-energy-curriculum.v1"
ENERGY_PARTNERS = ("skilled", "assertive", "noisy")
ENERGY_BATTERY_MAX = 70.
ENERGY_ROWS_PER_PARTNER = 64
ENERGY_MINIMUM_ROWS_PER_PARTNER = 16
ENERGY_MAXIMUM_SOURCE_EPISODES = 256

# The frozen r3 validation closure predates this additive r4.1 environment
# module.  Its original evaluator used a directory glob, so merely adding the
# new file would otherwise make the genuine saved r3 checkpoint unreadable.
# The compatibility scope below permits exactly this one additive file while
# still requiring every historical source path and byte hash to match the
# immutable r3 receipt.  It never edits an old artifact or ignores a changed
# historical source.
R41_ADDITIVE_HISTORICAL_SOURCE_PATHS = frozenset({
    "env/warehouse_native/r41_conflict.py",
})


@contextmanager
def _r3_historical_source_compatibility(root: Path | str = SOURCE_ROOT):
    """Temporarily reconstruct the exact pre-r4.1 r3 source inventory.

    ``warehouse_native_partner_mix_evaluation.execution_sources`` historically
    globbed every Python file in ``env/warehouse_native``.  R4.1 adds a separate
    conflict environment in that directory without changing any historical
    implementation.  Before filtering it from the old read path, this scope
    proves that the current alignment closure is byte-for-byte the saved r3
    closure plus only the explicit additive allowlist.
    """
    from . import warehouse_family_alignment_run as alignment_run
    from . import warehouse_native_partner_mix_evaluation as compact

    source_root = Path(root).resolve()
    prepared_path = source_root / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    expected = prepared.get("identity", {}).get("runtime_sources")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Frozen r3 runtime source receipt is missing")
    current = alignment_run.sources()
    missing = set(expected) - set(current)
    changed = {path for path in set(expected) & set(current)
               if expected[path] != current[path]}
    additive = set(current) - set(expected)
    if (missing or changed
            or additive != set(R41_ADDITIVE_HISTORICAL_SOURCE_PATHS)):
        raise ValueError(
            "Current sources are not the exact r3 closure plus the registered "
            "r4.1 additive environment"
        )
    for relative in additive:
        path = ROOT / relative
        if not path.is_file() or file_hash(path) != current[relative]:
            raise ValueError("Registered r4.1 additive source bytes differ")

    original = compact.execution_sources

    def historical_execution_sources():
        values = original()
        extras = set(values) & set(R41_ADDITIVE_HISTORICAL_SOURCE_PATHS)
        if extras != set(R41_ADDITIVE_HISTORICAL_SOURCE_PATHS):
            raise ValueError("R4.1 additive source is absent from the live closure")
        for relative in extras:
            if values[relative] != current[relative]:
                raise ValueError("R4.1 additive source changed during r3 restore")
            values.pop(relative)
        return values

    compact.execution_sources = historical_execution_sources
    try:
        if alignment_run.sources() != expected:
            raise ValueError("Historical r3 source reconstruction differs")
        yield {
            "historical_runtime_sources_sha256": digest(expected),
            "additive_sources": {path: current[path] for path in sorted(additive)},
        }
        if alignment_run.sources() != expected:
            raise ValueError("Historical r3 sources changed during reconstruction")
    finally:
        compact.execution_sources = original


def _conflict_api():
    from types import SimpleNamespace
    from env.warehouse_native import r41_conflict as contract
    from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
    from backend.training import warehouse_r41_conflict_scenarios as scenarios

    required = ("CONTRACT_VERSION", "CONTRACT_SHA256", "CONFLICT_GRAPH_SHA256")
    if (any(not hasattr(contract, name) for name in required)
            or not hasattr(scenarios, "validate_conflict_manifest")
            or not hasattr(scenarios, "reset_conflict_scenario")):
        raise RuntimeError("The shared r4.1 conflict runtime contract is incomplete")
    return SimpleNamespace(
        R41ConflictWarehouseEnv=R41ConflictWarehouseEnv,
        reset_conflict_scenario=scenarios.reset_conflict_scenario,
        validate_conflict_manifest=scenarios.validate_conflict_manifest,
        CONTRACT_VERSION=contract.CONTRACT_VERSION,
        CONTRACT_SHA256=contract.CONTRACT_SHA256,
        CONFLICT_GRAPH_SHA256=contract.CONFLICT_GRAPH_SHA256,
        MANIFEST_VERSION=scenarios.MANIFEST_VERSION,
    )


def _new_environment():
    api = _conflict_api()
    return api.R41ConflictWarehouseEnv(
        reward_config=deepcopy(TRAINING_REWARD), collision_cost=.05,
        mode="observed",
    )


def _implementation_sources():
    """Explicit training closure; no directory glob can absorb later files."""
    paths = (
        Path(__file__), Path(r4.__file__),
        ROOT / "backend/training/warehouse_r4_role_feedback_update.py",
        ROOT / "backend/training/warehouse_r41_conflict_scenarios.py",
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "env/warehouse_native/r41_conflict.py",
    )
    if any(not path.is_file() for path in paths):
        raise RuntimeError("R4.1 training implementation closure is incomplete")
    return {str(path.resolve().relative_to(ROOT)): file_hash(path)
            for path in paths}


def load_source(root: Path | str = SOURCE_ROOT):
    """Use the audited r4 loader, which reconstructs the genuine r3 learner."""
    with _r3_historical_source_compatibility(root):
        source, scenarios, baseline = r4.load_source(root)
    actor_path = Path(root).resolve() / "branches/feedback/actors/actor_0050000.npz"
    if file_hash(actor_path) != SOURCE_ACTOR_SHA256:
        raise ValueError("The r4.1 parent is not the exact frozen r3 Actor")
    return source, scenarios, baseline


def _actor_parameter_sha(model) -> str:
    return r4._actor_parameter_sha(model)


def _reachable_energy_state(snapshot) -> dict | None:
    """Admit an exact public state from which robot_2 can still reach charge."""
    state, config = snapshot["state"], snapshot["configuration"]
    actor = state["agents"][1]
    if not actor["active"] or float(actor["battery"]) > ENERGY_BATTERY_MAX:
        return None
    layout = get_map_layout(config["map_layout_id"])
    distance = shortest_path_distance(
        tuple(actor["position"]), tuple(layout.charger_position),
        config["map_layout_id"],
    )
    if not np.isfinite(distance) or tuple(actor["position"]) == tuple(layout.charger_position):
        return None
    move_cost = float(config["move_battery_cost"])
    required_to_charger = move_cost * float(distance)
    configured_margin = float(config["battery_safety_margin"])
    if float(actor["battery"]) < required_to_charger + configured_margin:
        return None
    _, charge_needed, _ = r4._goal_distance(snapshot, 1)
    if not charge_needed:
        return None
    return {
        "robot_2_battery": float(actor["battery"]),
        "charger_distance": int(distance),
        "required_battery_to_charger": required_to_charger,
        "public_safety_margin": float(actor["battery"]) - required_to_charger,
        "configured_safety_margin": configured_margin,
        "charge_needed": True,
        "reachable_without_shutdown": True,
    }


def build_reachable_energy_curriculum(entries, actor_path=SOURCE_ACTOR_PATH,
                                      *, maximum_source_episodes=ENERGY_MAXIMUM_SOURCE_EPISODES,
                                      rows_per_partner=ENERGY_ROWS_PER_PARTNER):
    """Collect exact train-only states under the online successor dynamics.

    No battery value, task, action, or label is edited.  A row is admitted only
    after the public shortest path proves that the active Actor can still reach
    the charger.  The source Actor command is used only to advance collection
    and is omitted from the curriculum artifact.
    """
    actor_path = Path(actor_path).resolve()
    if file_hash(actor_path) != SOURCE_ACTOR_SHA256:
        raise ValueError("Energy curriculum must use the exact frozen r3 Actor")
    if not entries or maximum_source_episodes <= 0 or rows_per_partner <= 0:
        raise ValueError("Energy curriculum requires conflict training scenes")
    api = _conflict_api()
    source_entries_sha256 = digest(entries)
    actor = NumPyNativeActor(actor_path)
    rows = []
    counts = Counter()
    seen = set()
    environment_steps = actor_commands = action_equal = 0
    source_episodes = 0
    round_index = 0
    while (source_episodes < maximum_source_episodes
           and any(counts[p] < rows_per_partner for p in ENERGY_PARTNERS)):
        partner = ENERGY_PARTNERS[source_episodes % len(ENERGY_PARTNERS)]
        entry_index = (source_episodes // len(ENERGY_PARTNERS)) % len(entries)
        entry = entries[entry_index]
        rng = np.random.default_rng(np.random.SeedSequence([
            260_911_401, source_episodes, int(entry.get("seed", entry_index)),
        ]))
        env = _new_environment()
        api.reset_conflict_scenario(env, entry)
        episode_id = f"{entry['id']}:{partner}:{round_index:03d}"
        source_episodes += 1
        if entry_index == len(entries) - 1:
            round_index += 1
        while not env.done:
            snapshot = env.snapshot()
            snapshot_sha = digest(snapshot)
            admission = _reachable_energy_state(snapshot)
            if admission is not None and counts[partner] < rows_per_partner:
                fingerprint = snapshot_sha
                if fingerprint not in seen:
                    row = {
                        "id": f"energy_{partner}_{counts[partner]:04d}",
                        "source_entry_id": entry["id"],
                        "source_split": "train",
                        "source_episode": episode_id,
                        "source_frame": int(env.state.frame),
                        "source_snapshot_sha256": snapshot_sha,
                        "snapshot": deepcopy(snapshot),
                        "partner": partner,
                        "category": "reachable_charge_needed",
                        "admission": admission,
                        "action_label_stored": False,
                    }
                    rows.append(row)
                    counts[partner] += 1
                    seen.add(fingerprint)
            before = digest(env.snapshot())
            player_action = partner_action(env, "robot_1", partner, rng)
            if digest(env.snapshot()) != before:
                raise RuntimeError("Energy curriculum partner mutated state")
            policy_actions, probabilities = actor.act(env.observations(), deterministic=True)
            if (digest(env.snapshot()) != before
                    or policy_actions["robot_2"]
                        != ACTIONS[int(np.argmax(probabilities["robot_2"]))]):
                raise RuntimeError("Energy curriculum Actor authority differs")
            submitted = {"robot_1": player_action,
                         "robot_2": policy_actions["robot_2"]}
            _, _, _, _, info = env.step(submitted)
            environment_steps += 1
            actor_commands += 1
            action_equal += int(info["requested_actions"]["robot_2"] == policy_actions["robot_2"])
            if action_equal != actor_commands:
                raise RuntimeError("Energy curriculum overwrote the Actor command")
    if digest(entries) != source_entries_sha256:
        raise RuntimeError("Conflict training entries changed during curriculum collection")
    if any(counts[partner] < ENERGY_MINIMUM_ROWS_PER_PARTNER for partner in ENERGY_PARTNERS):
        raise RuntimeError("Insufficient exact reachable-energy states for every partner")
    rows.sort(key=lambda row: (row["partner"], row["id"]))
    manifest = {
        "version": ENERGY_CURRICULUM_VERSION,
        "source_split": "train",
        "source_entries_sha256": source_entries_sha256,
        "conflict_contract_sha256": api.CONTRACT_SHA256,
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_episodes": source_episodes,
        "environment_steps": environment_steps,
        "actor_commands": actor_commands,
        "actor_action_equal": action_equal,
        "action_overrides": actor_commands - action_equal,
        "rows": len(rows),
        "rows_by_partner": {partner: counts[partner] for partner in ENERGY_PARTNERS},
        "snapshot_mutation": False,
        "action_labels_stored": False,
        "validation_and_play_excluded": True,
        "admission": {
            "battery_max": ENERGY_BATTERY_MAX,
            "charge_needed": True,
            "public_shortest_path_reachable": True,
            "minimum_rows_per_partner": ENERGY_MINIMUM_ROWS_PER_PARTNER,
        },
        "row_bindings": [{
            "id": row["id"], "partner": row["partner"],
            "source_entry_id": row["source_entry_id"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
        } for row in rows],
    }
    manifest["sha256"] = digest(manifest)
    return tuple(rows), manifest


class R41ActiveTrainer(r4.ActiveTrainer):
    """Fresh r3 continuation bound to the shared r4.1 conflict environment."""

    def __init__(self, source, conflict_manifest, *, conflict_manifest_file_sha256,
                 original_validation_binding, source_actor_path=SOURCE_ACTOR_PATH):
        api = _conflict_api()
        api.validate_conflict_manifest(conflict_manifest)
        # AlignmentTrainer.state_dict verifies the exact historical execution
        # closure.  Re-enter the same narrowly scoped compatibility audit here;
        # once copied, r4.1 owns its independent checkpoint schema.
        with _r3_historical_source_compatibility(SOURCE_ROOT) as historical_compatibility:
            source_state = source.state_dict()
        cumulative = source.source_counters["joint_steps"] + source.joint_steps
        if cumulative != SOURCE_CUMULATIVE_STEPS:
            raise ValueError("R4.1 must start from the exact 3.95M r3 source")
        self.__dict__.update(source.__dict__)
        self.source_state_sha256 = initialization_sha256(source_state)
        self.source_actor_sha256 = SOURCE_ACTOR_SHA256
        source_actor_path = Path(source_actor_path).resolve()
        if (file_hash(source_actor_path) != SOURCE_ACTOR_SHA256
                or _actor_parameter_sha(self.model) != NumPyNativeActor(
                    source_actor_path).metadata["actor_parameters_sha256"]):
            raise ValueError("R4.1 source checkpoint and Actor parameters differ")

        self.cfg = deepcopy(self.cfg)
        self.cfg["learning_rate"] = ACTOR_LR
        self.cfg["actor_learning_rate"] = ACTOR_LR
        self.cfg["critic_learning_rate"] = CRITIC_LR
        self.cfg["entropy"] = ENTROPY_COEFFICIENT
        self.optimizers.cfg = self.cfg
        self.optimizers.actor.param_groups[0]["lr"] = ACTOR_LR
        self.optimizers.critic.param_groups[0]["lr"] = CRITIC_LR
        self.source_counters = {
            key: int(source.source_counters.get(key, 0) + getattr(source, key, 0))
            for key in ("joint_steps", "optimizer_updates", "minibatch_updates",
                        "actor_optimizer_steps", "critic_optimizer_steps")
        }
        self.source_frames = [env.state.frame for env in source.envs]
        self.scenarios = deepcopy(conflict_manifest)
        self.training_entries = tuple(self.scenarios["splits"]["train"])
        if not self.training_entries:
            raise ValueError("R4.1 conflict manifest has no training split")
        self.training_entries_sha256 = digest(self.training_entries)
        self.envs = [_new_environment() for _ in range(self.cfg["environments"])]
        if tuple(self.envs[0].feature_names) != tuple(source.envs[0].feature_names):
            raise ValueError("Conflict environment changed the observed197 Actor schema")
        self.mode = "observed"
        self.test_fixture = False
        self._r41_manifest_file_sha256 = str(conflict_manifest_file_sha256)
        if len(self._r41_manifest_file_sha256) != 64:
            raise ValueError("Conflict manifest file SHA-256 is required")
        if not isinstance(original_validation_binding, dict):
            raise ValueError("Original validation binding is required")
        self._r41_original_validation_binding = deepcopy(original_validation_binding)
        (self.reachable_energy_curriculum,
         self.reachable_energy_curriculum_manifest) = build_reachable_energy_curriculum(
            self.training_entries, source_actor_path)
        self.energy_by_partner = {
            partner: tuple(row for row in self.reachable_energy_curriculum
                           if row["partner"] == partner)
            for partner in ENERGY_PARTNERS
        }
        if any(not rows for rows in self.energy_by_partner.values()):
            raise RuntimeError("Every program partner needs reachable-energy states")
        self.protocol = {
            "version": VERSION,
            "source": {
                "actor_sha256": SOURCE_ACTOR_SHA256,
                "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
                "state_sha256": self.source_state_sha256,
                "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
                "parent_version": "r3",
                "failed_r4_parent_used": False,
                "historical_source_compatibility": deepcopy(historical_compatibility),
            },
            "training": deepcopy(self.cfg),
            "partners": deepcopy(PARTNER_MIX),
            "shaping": deepcopy(SHAPING),
            "feedback": {
                "direction": "KL(nn||program)",
                "maximum_lambda": KL_MAX,
                "applied_lambda": TRAINING_KL_LAMBDA,
                "refresh_interval_joint_steps": BOUNDARY_INTERVAL,
                "role_scope": "robot_2",
                "runtime_action_override": False,
            },
            "role_assignment": {
                "program_partner_role": "robot_1",
                "deployed_nn_role": "robot_2",
                "selfplay_trains_both_roles": True,
            },
            "scenario_sampling": {
                "environment": "shared_r41_conflict_runtime",
                "conflict_contract_sha256": api.CONTRACT_SHA256,
                "conflict_contract_version": api.CONTRACT_VERSION,
                "conflict_graph_sha256": api.CONFLICT_GRAPH_SHA256,
                "manifest_version": api.MANIFEST_VERSION,
                "manifest_file_sha256": self._r41_manifest_file_sha256,
                "manifest_semantic_sha256": digest(self.scenarios),
                "train_entries_sha256": self.training_entries_sha256,
                "ordinary_probability": ORDINARY_EPISODE_PROBABILITY,
                "reachable_energy_probability": REACHABLE_ENERGY_EPISODE_PROBABILITY,
                "selfplay_uses_ordinary_start": True,
                "curriculum": deepcopy(self.reachable_energy_curriculum_manifest),
                "validation_and_play_excluded": True,
                "action_labels_stored": False,
            },
            "evaluation": {
                "interval_joint_steps": BOUNDARY_INTERVAL,
                "original_validation": deepcopy(original_validation_binding),
                "conflict_validation_entries_sha256": digest(
                    self.scenarios["splits"]["conflict_validation"]),
                "both_suites_must_pass": True,
            },
            "maximum_additional_joint_steps": MAXIMUM_ADDITIONAL_JOINT_STEPS,
            "implementation_sources": _implementation_sources(),
            "initial_feedback": None,
            "runtime_action_override": False,
        }
        count = len(self.envs)
        self.partner_kinds = [""] * count
        self.program_roles = [-1] * count
        self.scenario_ids = [""] * count
        self.episode_context = [None] * count
        self.episode_returns = np.zeros(count, dtype=np.float64)
        self.episode_reward_components = [Counter() for _ in range(count)]
        self.completed_episodes = []
        self.round_episode_prefixes = [None] * count
        for key in COUNTERS:
            setattr(self, key, 0)
        self.elapsed_seconds = 0.
        self.feedback_enabled = False
        self.feedback_manager = FeedbackManager(self.envs[0].feature_names, TREE_CONFIG)
        self.current_lambda = 0.
        self.feedback_rng = np.random.default_rng(260_911_402)
        self.reservoir_rng = np.random.default_rng(260_911_403)
        self.reservoir_seen = 0
        self.extraction_rows = []
        self.update_audits = []
        self.shaping_counts = Counter()
        self.action_authority = Counter()
        self.sampling_counts = Counter()
        for i in range(count):
            self.reset_one(i)
        if self.envs[0].observation_size != 197 or len(self.envs[0].global_state()) != 354:
            raise ValueError("Conflict environment changed Actor or Critic dimensions")

    def _environment(self):
        return _new_environment()

    def bind_r4_stage_source(self, *args, **kwargs):
        raise ValueError("R4.1 is a fresh r3 child; failed r4 stages are forbidden")

    def bind_initial_feedback(self, artifact, *, artifact_sha256, lambda_value):
        if self.joint_steps != 0 or self.current_lambda != 0:
            raise ValueError("Initial feedback can only bind at the r3 boundary")
        if (artifact.get("version") != "warehouse-r41-r3-preextraction.v1"
                or artifact.get("test_fixture") is not False
                or artifact.get("source_actor_sha256") != SOURCE_ACTOR_SHA256
                or artifact.get("conflict_contract_sha256")
                    != self.protocol["scenario_sampling"]["conflict_contract_sha256"]
                or artifact.get("conflict_contract_version")
                    != self.protocol["scenario_sampling"]["conflict_contract_version"]
                or artifact.get("conflict_graph_sha256")
                    != self.protocol["scenario_sampling"]["conflict_graph_sha256"]
                or artifact.get("conflict_manifest_version")
                    != self.protocol["scenario_sampling"]["manifest_version"]
                or artifact.get("conflict_manifest_semantic_sha256")
                    != self.protocol["scenario_sampling"]["manifest_semantic_sha256"]):
            raise ValueError("Initial RCPD is not bound to r3 and r4.1 conflict dynamics")
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise ValueError("Initial RCPD artifact SHA-256 is required")
        value = float(lambda_value)
        if not 0 < value <= KL_MAX:
            raise ValueError("Initial feedback lambda must be positive and bounded")
        self.feedback_manager.load_state_dict(artifact["feedback_manager"])
        program = self.feedback_manager.program
        if (program is None or not self.feedback_manager.reliable
                or tuple(program.feature_names) != tuple(self.envs[0].feature_names)
                or tuple(program.action_names) != tuple(ACTIONS)
                or program.metadata.get("native_source_actor_sha256") != SOURCE_ACTOR_SHA256
                or artifact.get("program_content_sha256") != digest(program.to_dict())):
            raise ValueError("Initial RCPD program/source/schema binding differs")
        self.current_lambda = value
        self.protocol["initial_feedback"] = {
            "artifact_sha256": artifact_sha256,
            "program_content_sha256": digest(program.to_dict()),
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "conflict_contract_sha256": artifact["conflict_contract_sha256"],
            "conflict_contract_version": artifact["conflict_contract_version"],
            "conflict_graph_sha256": artifact["conflict_graph_sha256"],
            "conflict_manifest_version": artifact["conflict_manifest_version"],
            "lambda": value,
            "extraction_environment_steps": int(artifact["extraction_environment_steps"]),
            "ppo_joint_steps": 0,
            "runtime_action_override": False,
        }

    def reset_one(self, i):
        api = _conflict_api()
        partner = str(self.rng.choice(list(PARTNER_MIX), p=list(PARTNER_MIX.values())))
        self.partner_kinds[i] = partner
        self.program_roles[i] = -1 if partner == "selfplay" else 0
        category = "conflict_train_initial"
        draw = float(self.rng.random())
        if partner != "selfplay" and draw < CONDITIONAL_ENERGY_PROBABILITY:
            rows = self.energy_by_partner[partner]
            row = rows[int(self.rng.integers(len(rows)))]
            self.envs[i].restore(deepcopy(row["snapshot"]), require_feedback=True)
            if (digest(self.envs[i].snapshot()) != row["source_snapshot_sha256"]
                    or not self.envs[i].state.agents[1].active
                    or _reachable_energy_state(self.envs[i].snapshot()) is None):
                raise RuntimeError("Exact reachable-energy snapshot failed validation")
            entry_id = row["id"]
            category = "reachable_charge_needed"
            self.sampling_counts["reachable_energy_episodes"] += 1
            self.sampling_counts[f"reachable_energy_{partner}_episodes"] += 1
        else:
            entry = self.training_entries[int(self.rng.integers(len(self.training_entries)))]
            api.reset_conflict_scenario(self.envs[i], entry)
            entry_id = entry["id"]
            self.sampling_counts["conflict_train_initial_episodes"] += 1
        if digest(self.training_entries) != self.training_entries_sha256:
            raise RuntimeError("Frozen conflict training split changed during reset")
        self.scenario_ids[i] = entry_id
        self.episode_returns[i] = 0.
        self.episode_reward_components[i] = Counter()
        self.episode_context[i] = {
            "source": "r41_conflict_train",
            "curriculum_id": (ENERGY_CURRICULUM_VERSION
                              if category == "reachable_charge_needed" else None),
            "category": category,
            "start_frame": int(self.envs[i].state.frame),
            "prefix": {
                "team_deliveries": self.envs[i].state.total_deliveries,
                "individual_deliveries": [
                    agent.deliveries_completed for agent in self.envs[i].state.agents],
                "shutdowns": self.envs[i].state.shutdown_count,
                "collisions": self.envs[i].state.robot_collision_events,
            },
        }
        self.round_episode_prefixes[i] = _prefix(self.envs[i], 0., {})
        self.episode_count += 1

    def collect(self, time_steps):
        increment = time_steps * len(self.envs)
        if self.joint_steps + increment > MAXIMUM_ADDITIONAL_JOINT_STEPS:
            raise ValueError("R4.1 continuation exceeds the 2M maximum")
        batch = PublicFeedbackTrainer.collect(self, time_steps)
        for record in batch["transition_records"]:
            t, i = record["rollout_index"], record["environment_index"]
            episode = f"{record['scenario_id']}:{i}:{record['before']['episode_counter']}"
            for role in range(2):
                agent_id = f"robot_{role + 1}"
                sampled = record["neural_sampled_actions"][role]
                if record["trainable"][role]:
                    self.action_authority["trainable"] += 1
                    equal = record["requested_actions"][agent_id] == sampled
                    self.action_authority["equal"] += int(equal)
                    self.action_authority[f"trainable_{agent_id}"] += 1
                    self.action_authority[
                        f"partner_{record['partner']}_trainable_{agent_id}"] += 1
                    value, flags = r4.shaping_for_transition(record, role)
                    base = float(batch["rewards"][t, i, role])
                    batch["rewards"][t, i, role] += value
                    self.shaping_counts["base_reward_sum"] += base
                    self.shaping_counts["base_reward_abs_sum"] += abs(base)
                    self.shaping_counts["shaping_reward_sum"] += value
                    self.shaping_counts["shaping_reward_abs_sum"] += abs(value)
                    for name, active in flags.items():
                        self.shaping_counts[name] += int(active)
                    self.shaping_counts["rows"] += 1
                    observation = np.asarray(
                        batch["observations"][t, i, role], dtype=np.float32).copy()
                    self._add_extraction_row({
                        "observation": observation, "episode": episode,
                        "groups": r4._critical_groups(record["before"], role),
                        "role": role,
                    })
                    if not equal:
                        raise RuntimeError("A trainable r4.1 neural action was overwritten")
        with self._rng_context():
            _, bootstrap = self.infer(*self.arrays())
        batch["advantages"], batch["returns"] = gae(
            batch["rewards"], batch["values"], batch["dones"], bootstrap,
            self.cfg["gamma"], self.cfg["gae_lambda"],
        )
        return batch

    def fit_program(self, actor_file_sha256):
        report = super().fit_program(actor_file_sha256)
        report["r41_conflict_binding"] = {
            "contract_sha256": self.protocol["scenario_sampling"]["conflict_contract_sha256"],
            "contract_version": self.protocol["scenario_sampling"]["conflict_contract_version"],
            "graph_sha256": self.protocol["scenario_sampling"]["conflict_graph_sha256"],
            "manifest_version": self.protocol["scenario_sampling"]["manifest_version"],
            "manifest_semantic_sha256": self.protocol["scenario_sampling"]["manifest_semantic_sha256"],
            "train_entries_sha256": self.training_entries_sha256,
            "runtime_action_override": False,
        }
        return report

    def state_dict_r41(self):
        return {
            "version": VERSION, "protocol": deepcopy(self.protocol),
            "source_state_sha256": self.source_state_sha256,
            "source_counters": deepcopy(self.source_counters),
            "model": _cpu(self.model.state_dict()),
            "optimizers": _cpu(self.optimizers.state_dict()),
            "rng": deepcopy(self.rng.bit_generator.state),
            "feedback_rng": deepcopy(self.feedback_rng.bit_generator.state),
            "reservoir_rng": deepcopy(self.reservoir_rng.bit_generator.state),
            "owned_rng": _cpu(self._owned_rng_state),
            "envs": [env.snapshot() for env in self.envs],
            "feedback_manager": self.feedback_manager.state_dict(),
            "current_lambda": self.current_lambda,
            "reservoir_seen": self.reservoir_seen,
            "extraction_rows": deepcopy(self.extraction_rows),
            "update_audits": deepcopy(self.update_audits),
            "shaping_counts": dict(self.shaping_counts),
            "action_authority": dict(self.action_authority),
            "sampling_counts": dict(self.sampling_counts),
            "elapsed_seconds": self.elapsed_seconds,
            "round_episode_prefixes": deepcopy(self.round_episode_prefixes),
            "completed_episodes": _cpu(self.completed_episodes),
            "episode_reward_components": [dict(x) for x in self.episode_reward_components],
            **{key: _cpu(getattr(self, key)) for key in (*COUNTERS, *EPISODE_FIELDS)},
        }

    def load_state_dict_r41(self, payload):
        if payload.get("version") != VERSION or payload.get("protocol") != self.protocol:
            raise ValueError("R4.1 checkpoint protocol differs")
        if payload.get("source_state_sha256") != self.source_state_sha256:
            raise ValueError("R4.1 checkpoint belongs to another r3 parent")
        self.model.load_state_dict(payload["model"])
        self.optimizers.load_state_dict(payload["optimizers"])
        self.rng.bit_generator.state = deepcopy(payload["rng"])
        self.feedback_rng.bit_generator.state = deepcopy(payload["feedback_rng"])
        self.reservoir_rng.bit_generator.state = deepcopy(payload["reservoir_rng"])
        self._owned_rng_state = _cpu(payload["owned_rng"])
        if len(payload["envs"]) != len(self.envs):
            raise ValueError("R4.1 checkpoint environment count differs")
        for env, snapshot in zip(self.envs, payload["envs"]):
            env.restore(snapshot, require_feedback=True)
        self.feedback_manager.load_state_dict(payload["feedback_manager"])
        for key in ("current_lambda", "reservoir_seen", "extraction_rows",
                    "update_audits", "elapsed_seconds", "round_episode_prefixes",
                    "completed_episodes"):
            setattr(self, key, deepcopy(payload[key]))
        self.shaping_counts = Counter(payload["shaping_counts"])
        self.action_authority = Counter(payload["action_authority"])
        self.sampling_counts = Counter(payload["sampling_counts"])
        self.episode_reward_components = [
            Counter(item) for item in payload["episode_reward_components"]]
        for key in (*COUNTERS, *EPISODE_FIELDS):
            setattr(self, key, deepcopy(payload[key]))
        if (self.joint_steps > MAXIMUM_ADDITIONAL_JOINT_STEPS
                or self.action_authority["equal"] != self.action_authority["trainable"]):
            raise ValueError("R4.1 saved budget or action authority differs")

    def export(self, path):
        metadata = {
            "experiment_version": VERSION,
            "joint_steps": self.joint_steps,
            "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + self.joint_steps,
            "optimizer_updates": self.optimizer_updates,
            "actor_parameters_sha256": _actor_parameter_sha(self.model),
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_counters": deepcopy(self.source_counters),
            "protocol_sha256": digest(self.protocol),
            "conflict_contract_sha256": self.protocol["scenario_sampling"]["conflict_contract_sha256"],
            "conflict_contract_version": self.protocol["scenario_sampling"]["conflict_contract_version"],
            "conflict_graph_sha256": self.protocol["scenario_sampling"]["conflict_graph_sha256"],
            "conflict_manifest_version": self.protocol["scenario_sampling"]["manifest_version"],
            "conflict_manifest_semantic_sha256": self.protocol["scenario_sampling"]["manifest_semantic_sha256"],
            "public_feedback_version": "warehouse-native-public-feedback.v1",
            "public_feedback_mode": "observed",
            "feature_names": list(self.envs[0].feature_names),
            "feedback_lambda": self.current_lambda,
            "action_controller": "neural_actor_only",
            "runtime_action_override": False,
            "failed_r4_parent_used": False,
            "candidate": True,
        }
        return self.model.export_npz(path, metadata)
