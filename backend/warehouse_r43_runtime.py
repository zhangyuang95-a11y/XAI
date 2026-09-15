"""Direct-Actor r4.3 runtime with 3% movement cost and shared-charger rule."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
    R41DiagnosticOnlineAlignmentRuntime,
)
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from env.warehouse_native.r43_charger import (
    R43_OBSERVATION_FEATURE_NAMES,
    R43SharedChargerMixin,
)


VERSION = "warehouse-r43-direct-runtime.v2"


def runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "env/warehouse_native/r43_charger.py",
        root / "env/warehouse/domain.py",
    )
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


class R43WarehouseEnv(R43SharedChargerMixin,
                      R41DiagnosticConflictWarehouseEnv):
    def __init__(self, config=None, reward_config=None, collision_cost=.05,
                 mode="observed"):
        super().__init__(config, reward_config, collision_cost, mode=mode)
        self._initialize_r43_charger()


class R43WarehouseRuntime(R41DiagnosticOnlineAlignmentRuntime):
    """Keep the Actor action authoritative while changing only public physics."""

    def __init__(self, *args: Any, **kwargs: Any):
        later_features = tuple(
            kwargs.get("additional_observation_feature_names", ())
        )
        kwargs["environment_config_overrides"] = {"move_battery_cost": 3.0}
        kwargs["additional_observation_feature_names"] = (
            tuple(R43_OBSERVATION_FEATURE_NAMES) + later_features
        )
        super().__init__(*args, **kwargs)
        self._r43_sources = runtime_sources()
        self.signature = digest({
            "version": VERSION,
            "parent_runtime_signature": self.signature,
            "configuration": {
                "move_battery_cost": self.config.move_battery_cost,
                "charge_per_wait": self.config.charge_per_wait,
            },
            "sources": self._r43_sources,
        })
        self.contract_report.update(
            version=VERSION,
            signature=self.signature,
            move_battery_cost=3.0,
            charge_per_wait=10.0,
            shared_charger_rule=True,
            runtime_action_override=False,
        )

    def verify_binding(self):
        base = super().verify_binding()
        if hasattr(self, "_r43_sources") and runtime_sources() != self._r43_sources:
            raise ValueError("r4.3 runtime sources changed")
        return getattr(self, "signature", base)

    def _new_environment(self):
        return R43WarehouseEnv(
            self.config,
            deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )

    def _check_environment(self, env):
        if (type(env) is not R43WarehouseEnv
                or env.config != self.config
                or env.reward_config != self._reward_config
                or env.observation_size != self.actor.obs_dim
                or list(env.feature_names)
                    != self.actor.metadata.get("feature_names")):
            raise ValueError("r4.3 online environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        env = self._new_environment()
        reset_diagnostic_scenario(env, deepcopy(scenario))
        return env


__all__ = ["VERSION", "R43WarehouseEnv", "R43WarehouseRuntime"]
