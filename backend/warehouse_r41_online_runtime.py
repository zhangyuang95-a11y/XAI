"""Observed197 runtime bound to the r4.1 conflict-task environment.

This module is additive: the r4 runtime remains available for its historical
failed evidence.  r4.1 training, evaluation, replay, and deployment must all
instantiate :class:`R41ConflictWarehouseEnv` from this module.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from backend.warehouse_alignment_online_runtime import (
    OnlineAlignmentRuntime,
    OnlinePublicFeedbackEnvironment,
    digest,
    file_hash,
    runtime_sources,
)
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH_SHA256,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    R41ConflictMixin,
    reset_r41_scenario,
)


RUNTIME_VERSION = "warehouse-r41-alignment-online-runtime.v1"
DEFAULT_REWARD_CONFIG = {
    "version": "warehouse-native-score-pbrs.v2",
    "native_score_scale": 0.01,
    "static_wall_command_cost": 0.02,
    "potential_scale": 0.25,
    "gamma": 0.99,
    "shared_reward": True,
    "pickup_bonus": 0.0,
    "charge_bonus": 0.0,
}


def r41_runtime_sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__),
        root / "env/warehouse_native/r41_conflict.py",
    )
    return {
        **{"r4_base/" + key: value for key, value in runtime_sources().items()},
        **{str(path.relative_to(root)): file_hash(path) for path in paths},
    }


class R41ConflictWarehouseEnv(R41ConflictMixin, OnlinePublicFeedbackEnvironment):
    """The sole observed197 environment contract for r4.1."""

    def __init__(
        self,
        config=None,
        reward_config=None,
        collision_cost: float = 0.05,
        mode: str = "observed",
    ) -> None:
        super().__init__(
            config,
            deepcopy(DEFAULT_REWARD_CONFIG if reward_config is None else reward_config),
            collision_cost,
            mode=mode,
        )
        self._initialize_r41_conflict()


class R41OnlineAlignmentRuntime(OnlineAlignmentRuntime):
    """Frozen-Actor runtime whose task creation is the r4.1 graph sampler."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._r41_sources = r41_runtime_sources()
        super().__init__(*args, **kwargs)
        base_signature = self.signature
        self.signature = digest(
            {
                "version": RUNTIME_VERSION,
                "base_runtime_signature": base_signature,
                "contract_version": CONTRACT_VERSION,
                "contract_sha256": CONTRACT_SHA256,
                "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
                "sources": self._r41_sources,
            }
        )
        self.contract_report.update(
            version=RUNTIME_VERSION,
            signature=self.signature,
            base_runtime_signature=base_signature,
            conflict_contract_version=CONTRACT_VERSION,
            conflict_contract_sha256=CONTRACT_SHA256,
            conflict_graph_sha256=CONFLICT_GRAPH_SHA256,
            r41_runtime_sources_sha256=digest(self._r41_sources),
        )

    def verify_binding(self):
        base = super().verify_binding()
        if r41_runtime_sources() != self._r41_sources:
            raise ValueError("r4.1 online runtime sources changed")
        return getattr(self, "signature", base)

    def _new_environment(self):
        return R41ConflictWarehouseEnv(
            self.config,
            deepcopy(self._reward_config),
            collision_cost=self._reward_config["collision_training_cost"],
            mode="observed",
        )

    def _check_environment(self, env):
        if (
            type(env) is not R41ConflictWarehouseEnv
            or env.config != self.config
            or env.reward_config != self._reward_config
        ):
            raise ValueError("r4.1 online runtime environment differs")
        env._require_state()

    def environment(self, scenario):
        self.verify_binding()
        env = self._new_environment()
        reset_r41_scenario(env, deepcopy(scenario))
        return env


AlignmentRuntime = R41OnlineAlignmentRuntime

