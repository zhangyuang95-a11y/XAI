"""Direct-Actor r4.5 runtime with unified energy-cycle observations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from backend.warehouse_alignment_online_runtime import digest, file_hash
from backend.warehouse_r44_runtime import R44WarehouseEnv, R44WarehouseRuntime
from env.warehouse_native.r41_diagnostic_conflict import reset_diagnostic_scenario
from env.warehouse_native.r45_cycle import (
    R45EnergyCycleMixin, R45_OBSERVATION_FEATURE_NAMES,
)


VERSION = "warehouse-r45-direct-runtime.v1"


def runtime_sources():
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "env/warehouse_native/r45_cycle.py",
        root / "env/warehouse_native/r45_energy.py",
        root / "env/warehouse_native/r44_charger.py",
        root / "env/warehouse_native/partners.py",
        root / "env/warehouse/domain.py",
    )
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


class R45WarehouseEnv(R45EnergyCycleMixin, R44WarehouseEnv):
    pass


class R45WarehouseRuntime(R44WarehouseRuntime):
    def __init__(self, *args, **kwargs):
        kwargs["additional_observation_feature_names"] = (
            R45_OBSERVATION_FEATURE_NAMES
        )
        kwargs["expected_hidden"] = 512
        super().__init__(*args, **kwargs)
        self._r45_sources = runtime_sources()
        self.signature = digest({
            "version": VERSION,
            "parent_runtime_signature": self.signature,
            "sources": self._r45_sources,
        })
        self.contract_report.update(
            version=VERSION,
            signature=self.signature,
            energy_budget_version="warehouse-r45-energy-budget.v2",
            energy_cycle_features=list(R45_OBSERVATION_FEATURE_NAMES),
            runtime_action_override=False,
        )

    def verify_binding(self):
        base = super().verify_binding()
        if hasattr(self, "_r45_sources") and runtime_sources() != self._r45_sources:
            raise ValueError("r4.5 runtime sources changed")
        return getattr(self, "signature", base)

    def _new_environment(self):
        return R45WarehouseEnv(
            self.config,
            deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )

    def _check_environment(self, env):
        if (type(env) is not R45WarehouseEnv
                or env.config != self.config
                or env.reward_config != self._reward_config
                or env.observation_size != self.actor.obs_dim
                or list(env.feature_names)
                    != self.actor.metadata.get("feature_names")):
            raise ValueError("r4.5 online environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        env = self._new_environment()
        reset_diagnostic_scenario(env, deepcopy(scenario))
        return env


__all__ = ["VERSION", "R45WarehouseEnv", "R45WarehouseRuntime"]
