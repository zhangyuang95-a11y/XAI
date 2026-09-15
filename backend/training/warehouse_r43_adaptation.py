"""Bounded r4.3 adaptation on 3% energy and shared-charger physics.

This wrapper reuses the accepted r4.2 conservative DAgger/PPO procedure but
binds every rollout to the r4.3 authoritative environment.  The teacher is
training-only; the exported two-layer Actor remains the sole runtime action
source.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

from backend.training import warehouse_r42_delivery_conservative as trainer
from backend.training import warehouse_r42_delivery_dagger as dagger
from backend.training import warehouse_r42_delivery_finetune as base
from backend.warehouse_r41_diagnostic_online_runtime import DEFAULT_REWARD_CONFIG
from backend.warehouse_r43_runtime import R43WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from env.warehouse_native.r43_charger import R43_OBSERVATION_FEATURE_NAMES


VERSION = "warehouse-r43-bounded-adaptation.v2"
_ORIGINAL_COLLECT = dagger.collect_dagger_rows


def _expanded_parent(source: Path, destination: Path):
    """Zero-extend the frozen parent before learning the new rule features."""
    parent = NumPyNativeActor(source)
    model = NativeActorCritic(
        parent.obs_dim + len(R43_OBSERVATION_FEATURE_NAMES),
        parent.state_dim, parent.hidden,
    )
    actor_state = {}
    for name, value in parent.weights.items():
        if name == "0.weight":
            padded = np.zeros(
                (parent.hidden, model.obs_dim), dtype=np.float32)
            padded[:, :parent.obs_dim] = value
            value = padded
        actor_state[name] = torch.as_tensor(value, dtype=torch.float32)
    model.actor.load_state_dict(actor_state)
    metadata = {
        **parent.metadata,
        "obs_dim": model.obs_dim,
        "feature_names": list(parent.metadata["feature_names"])
            + list(R43_OBSERVATION_FEATURE_NAMES),
        "r43_observation_features": list(R43_OBSERVATION_FEATURE_NAMES),
        "r43_base_actor_sha256": parent.sha256,
        "r43_parent_expansion": "zero_initialized_then_trained",
        "runtime_action_override": False,
    }
    model.export_npz(destination, metadata)
    return parent.sha256


def _load_expanded_model(parent: Path, device: torch.device,
                         actor_parent: Path | None = None):
    """Restore the old critic while accepting the expanded Actor input."""
    checkpoint = torch.load(
        parent / "checkpoint.pt", map_location="cpu", weights_only=False)
    actor = NumPyNativeActor(actor_parent)
    model = NativeActorCritic(actor.obs_dim, actor.state_dim, actor.hidden)
    previous = checkpoint["trainer"]["model"]
    current = model.state_dict()
    migrated = {}
    for name, target in current.items():
        source = previous[name]
        if tuple(source.shape) == tuple(target.shape):
            migrated[name] = source
        elif name == "actor.0.weight" and source.shape[0] == target.shape[0]:
            value = torch.zeros_like(target)
            value[:, :source.shape[1]] = source
            migrated[name] = value
        else:
            raise ValueError("r4.3 checkpoint tensor cannot be expanded: " + name)
    model.load_state_dict(migrated)
    model.actor.load_state_dict({
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in actor.weights.items()
    })
    return model.to(device), actor, checkpoint


def actor_environment(actor, scene):
    env = R43WarehouseEnv(
        collaborative_study_config(move_battery_cost=3.0),
        reward_config=deepcopy(DEFAULT_REWARD_CONFIG),
    )
    reset_diagnostic_scenario(env, scene)
    if list(env.feature_names) != actor.metadata.get("feature_names"):
        raise ValueError("r4.3 Actor feature contract differs")
    return env


def _safe_release(env):
    learner = env.state.by_id("robot_2")
    if (learner.position != env.layout.charger_position
            or learner.battery < 60.0):
        return None
    other = env.state.by_id("robot_1")
    qualification = env._qualification(
        env.state, {"robot_1": "WAIT", "robot_2": "WAIT"}, "robot_2")
    if (not other.active or other.battery > 20.0
            or qualification["teammate_distance"] > 2):
        return None
    preferred = partner_action(env, "robot_2", "skilled", random.Random(0))
    return (preferred if preferred in qualification["safe_departures"]
            else qualification["safe_departures"][0]
            if qualification["safe_departures"] else None)


def _append(rows, seen, env, action):
    observation = env.observations()["robot_2"].astype(np.float32, copy=True)
    key = sha256(observation.tobytes()).digest()
    if key not in seen:
        seen.add(key)
        rows.append((observation, base.ACTION_INDEX[action]))


def collect_dagger_rows(model, contract, entries, **kwargs):
    rows = list(_ORIGINAL_COLLECT(model, contract, entries, **kwargs))
    seen = {sha256(row[0].tobytes()).digest() for row in rows}
    for scene in entries[:64]:
        probe = actor_environment(contract, scene)
        candidates = sorted(
            point for point in probe.layout.passable_positions
            if point != probe.layout.charger_position
            and shortest_path_distance(
                point, probe.layout.charger_position,
                probe.config.map_layout_id) <= 2
        )
        nearby = sorted(
            candidates,
            key=lambda point: abs(point[0] - probe.layout.charger_position[0])
            + abs(point[1] - probe.layout.charger_position[1]),
        )
        for other_position in nearby[:4]:
            for occupant_battery in (60.0, 72.0, 90.0, 100.0):
                env = actor_environment(contract, scene)
                state = env.get_state()
                learner, other = state.by_id("robot_2"), state.by_id("robot_1")
                learner.position = env.layout.charger_position
                learner.battery = occupant_battery
                learner.active = True
                other.position = other_position
                other.battery = 20.0
                other.active = True
                env.set_state(state)
                action = _safe_release(env)
                if action:
                    _append(rows, seen, env, action)
                    # One authoritative grace wait exposes the public previous
                    # WAIT feature before teaching the same safe release.
                    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
                    action = _safe_release(env)
                    if action:
                        _append(rows, seen, env, action)
    return rows


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        index = arguments.index("--actor-parent")
        source = Path(arguments[index + 1]).expanduser().resolve()
        output_index = arguments.index("--output")
        output = Path(arguments[output_index + 1]).expanduser().resolve()
    except (ValueError, IndexError):
        raise ValueError("r4.3 adaptation requires --actor-parent and --output")
    with tempfile.TemporaryDirectory(prefix="warehouse-r43-expand-") as name:
        expanded = Path(name) / "expanded_parent.npz"
        base_actor_sha = _expanded_parent(source, expanded)
        arguments[index + 1] = str(expanded)
        base.actor_environment = actor_environment
        dagger.actor_environment = actor_environment
        trainer.actor_environment = actor_environment
        dagger.collect_dagger_rows = collect_dagger_rows
        base._load_model = _load_expanded_model
        trainer.VERSION = VERSION
        status = trainer.main(arguments)
        report_path = output / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["parent"]["base_actor_path"] = str(source)
        report["parent"]["base_actor_sha256"] = base_actor_sha
        report["training"]["observation_features_added"] = list(
            R43_OBSERVATION_FEATURE_NAMES)
        report["training"]["rule_state_observed_by_actor"] = True
        base.write_json(report_path, report)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
