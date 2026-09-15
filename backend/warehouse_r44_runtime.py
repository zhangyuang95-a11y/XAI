"""Direct-Actor r4.4 runtime with 3% movement and immediate charger rule."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_r43_runtime import R43WarehouseEnv, R43WarehouseRuntime
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from env.warehouse_native.r43_charger import R43SharedChargerMixin
from env.warehouse_native.r44_charger import (
    R44_OBSERVATION_FEATURE_NAMES,
    R44SharedChargerMixin,
)


VERSION = "warehouse-r44-direct-runtime.v1"


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "env/warehouse_native/r44_charger.py",
        root / "env/warehouse_native/partners.py",
        root / "env/warehouse/domain.py",
    )
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


class R44WarehouseEnv(R44SharedChargerMixin,
                      R43WarehouseEnv):
    """Replace the r4.3 rule in the otherwise identical conflict environment."""

    def step(self, actions, *, decision_metadata=None):
        return R44SharedChargerMixin.step(
            self, actions, decision_metadata=decision_metadata
        )

    def snapshot(self):
        return R44SharedChargerMixin.snapshot(self)

    def restore(self, payload, **kwargs):
        return R44SharedChargerMixin.restore(self, payload, **kwargs)


class R44WarehouseRuntime(R43WarehouseRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._r44_sources = runtime_sources()
        self.signature = digest({
            "version": VERSION,
            "parent_runtime_signature": self.signature,
            "sources": self._r44_sources,
        })
        self.contract_report.update(
            version=VERSION,
            signature=self.signature,
            shared_charger_rule_version="r4.4",
            runtime_action_override=False,
        )

    def verify_binding(self):
        base = super().verify_binding()
        if hasattr(self, "_r44_sources") and runtime_sources() != self._r44_sources:
            raise ValueError("r4.4 runtime sources changed")
        return getattr(self, "signature", base)

    def _new_environment(self):
        return R44WarehouseEnv(
            self.config,
            deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )

    def _check_environment(self, env):
        if (type(env) is not R44WarehouseEnv
                or env.config != self.config
                or env.reward_config != self._reward_config
                or env.observation_size != self.actor.obs_dim
                or list(env.feature_names)
                    != self.actor.metadata.get("feature_names")):
            raise ValueError("r4.4 online environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        env = self._new_environment()
        reset_diagnostic_scenario(env, deepcopy(scenario))
        return env


__all__ = ["VERSION", "R44WarehouseEnv", "R44WarehouseRuntime"]
