"""Export a validated MAPPO Actor for dependency-light Render inference."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from env.warehouse.contracts import (
    ACTION_EXECUTION_VERSION,
    ENVIRONMENT_VERSION,
    RUNTIME_CONTROLLER,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import observation_dim
from env.warehouse.policy import MAPPOPolicy
from env.warehouse.rewards import REWARD_VERSION


def export_numpy_actor(checkpoint: str | Path, output: str | Path) -> Path:
    source = Path(checkpoint)
    policy = MAPPOPolicy.load(source, device="cpu")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    network = policy.network
    weights = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in network.state_dict().items()
        if not name.startswith("critic.")
    }
    metadata = {
        "model_version": policy.model_version,
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": REWARD_VERSION,
        "action_execution_version": ACTION_EXECUTION_VERSION,
        "runtime_controller": RUNTIME_CONTROLLER,
        "map_layout_id": policy.environment_config.map_layout_id,
        "observation_dim": observation_dim(policy.environment_config),
        "action_names": list(ACTIONS),
        "action_dim": int(network.action_dim),
        "per_action_feature_dim": int(network.per_action_feature_dim),
        "own_action_features_start": int(network.own_action_features_start),
        "teammate_action_features_start": int(
            network.teammate_action_features_start
        ),
        "joint_collision_matrix_start": int(
            network.joint_collision_matrix_start
        ),
        "checkpoint_sha256": sha256(source.read_bytes()).hexdigest(),
    }
    np.savez_compressed(
        target,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        **weights,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    print(export_numpy_actor(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
