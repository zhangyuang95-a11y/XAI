"""Direct-Actor r4.6 runtime for the PPO-first warehouse policy."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from backend.training.warehouse_nn.environment import (
    ENVIRONMENT_VERSION,
    SNAPSHOT_KEY,
    WarehouseNNEnv,
)
from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_r44_runtime import R44WarehouseEnv
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario


VERSION = "warehouse-r46-ppo-direct-runtime.v1"
PROTOCOL_VERSION = "warehouse-nn-ppo-rcpd.v7-safety-targeted-final"


def _actor_metadata(path: str | Path) -> dict:
    with np.load(Path(path), allow_pickle=False) as payload:
        raw = payload["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
    value = json.loads(str(raw))
    if not isinstance(value, dict):
        raise ValueError("r4.6 Actor metadata differs")
    return value


def runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "backend/training/warehouse_nn/environment.py",
        root / "backend/training/warehouse_nn/energy.py",
        root / "backend/warehouse_r44_runtime.py",
        root / "env/warehouse_native/r44_charger.py",
        root / "env/warehouse_native/partners.py",
        root / "env/warehouse/domain.py",
    )
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def _with_history(snapshot: dict) -> dict:
    value = deepcopy(snapshot)
    if SNAPSHOT_KEY in value:
        return value
    env = WarehouseNNEnv({"move_battery_cost": 3.0, "observation_contract": "r4.6"})
    history = env._empty_history()
    for agent in value["state"]["agents"]:
        history[agent["agent_id"]]["positions"] = [tuple(agent["position"])]
    value[SNAPSHOT_KEY] = {
        "version": env._environment_version,
        "history": history,
    }
    return value


class R46WarehouseRuntime(R41DiagnosticOnlineAlignmentRuntime):
    """Serve the selected 800k Actor with its exact training observation."""

    def __init__(self, actor_path, *args, **kwargs):
        metadata = _actor_metadata(actor_path)
        features = tuple(metadata.get("feature_names", ()))
        kwargs.update(
            environment_config_overrides={"move_battery_cost": 3.0},
            exact_observation_feature_names=features,
            expected_hidden=128,
            expected_state_dim=514,
            expected_protocol_version=PROTOCOL_VERSION,
        )
        super().__init__(actor_path, *args, **kwargs)
        self._r46_sources = runtime_sources()
        self.signature = digest({
            "version": VERSION,
            "parent_runtime_signature": self.signature,
            "actor_parameters_sha256": metadata.get("actor_parameters_sha256"),
            "sources": self._r46_sources,
        })
        self.contract_report.update(
            version=VERSION,
            signature=self.signature,
            selected_joint_steps=800000,
            actor_parameters_sha256=metadata.get("actor_parameters_sha256"),
            behavior_performance_gate_passed=False,
            behavior_performance_gate_waived=True,
            runtime_action_override=False,
        )

    def verify_binding(self):
        base = super().verify_binding()
        if hasattr(self, "_r46_sources") and runtime_sources() != self._r46_sources:
            raise ValueError("r4.6 runtime sources changed")
        return getattr(self, "signature", base)

    def _new_environment(self):
        return WarehouseNNEnv({"move_battery_cost": 3.0, "observation_contract": "r4.6"})

    def _check_environment(self, env):
        if (type(env) is not WarehouseNNEnv
                or env.config != self.config
                or env.observation_size != self.actor.obs_dim
                or list(env.feature_names) != self.actor.metadata.get("feature_names")):
            raise ValueError("r4.6 online environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        value = deepcopy(scenario)
        validator = R44WarehouseEnv(
            self.config, deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )
        reset_diagnostic_scenario(validator, deepcopy(value))
        value["snapshot"] = _with_history(value["snapshot"])
        env = self._new_environment()
        env.restore(value["snapshot"])
        self._check_environment(env)
        return env


__all__ = ["VERSION", "PROTOCOL_VERSION", "R46WarehouseRuntime", "runtime_sources"]
