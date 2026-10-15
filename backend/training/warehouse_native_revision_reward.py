"""Untrained collision-cost revision of the frozen v2 training reward.

Only the scalar training reward changes. Public physics, participant scoring,
observations, submitted actions and RNG transitions stay in the frozen parent.
This component makes no claim that reducing collision cost improves learning.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

from .warehouse_native_v2 import REWARD_VERSION, V2RewardEnvironment


REWARD_REVISION = "warehouse-native-score-pbrs.collision-r1"


def _cost(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Collision training cost must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Collision training cost must be a finite nonnegative number")
    return result


class RevisedRewardEnvironment(V2RewardEnvironment):
    """Keep the v2 envelope, explicitly bind the revised collision component.

    Raw native snapshots remain valid scenario and curriculum starts. A snapshot
    carrying any training-reward metadata must instead carry the complete,
    matching revision. Original V2 environments using their original reward
    configuration reject these snapshots because the configurations differ.
    """

    def __init__(self, config=None, reward_config=None, collision_cost=0.05):
        collision_cost = _cost(collision_cost)
        super().__init__(config, reward_config)
        for key, expected in (("revision", REWARD_REVISION),
                              ("collision_training_cost", collision_cost)):
            if key in self.reward_config and self.reward_config[key] != expected:
                raise ValueError("Collision reward revision configuration mismatch")
        if "collision_training_cost" in self.reward_config:
            _cost(self.reward_config["collision_training_cost"])
        self._collision_cost = collision_cost
        self.reward_config.update(revision=REWARD_REVISION,
                                  collision_training_cost=collision_cost)

    @property
    def collision_cost(self):
        return self._collision_cost

    def _validate_revision(self):
        if (self.reward_config.get("version") != REWARD_VERSION
                or self.reward_config.get("revision") != REWARD_REVISION
                or _cost(self.reward_config.get("collision_training_cost"))
                != self.collision_cost):
            raise ValueError("Collision reward revision configuration mismatch")

    def step(self, actions, *, decision_metadata=None):
        self._validate_revision()
        obs, _, terminated, truncated, info = super().step(
            actions, decision_metadata=decision_metadata)
        # The native score increment retains the participant's original collision
        # deduction. This explicit correction replaces only its training weight.
        adjustment = (-self.collision_cost
                      - self.config.robot_collision_points
                      * self.reward_config["native_score_scale"])
        components = info["reward_components"]
        components["collision_risk_adjustment"] = (
            adjustment * int(info["robot_collision"]))
        reward = float(sum(components.values()))
        rewards = {key: reward for key in self.agent_ids}
        self.state.last_rewards = dict(rewards)
        info.update(reward_revision=REWARD_REVISION,
                    collision_training_cost=self.collision_cost)
        return obs, rewards, terminated, truncated, info

    def snapshot(self):
        self._validate_revision()
        payload = super().snapshot()
        payload["training_reward_revision"] = REWARD_REVISION
        return payload

    def restore(self, payload):
        self._validate_revision()
        # Fail before changing state or RNG. Partial/old training metadata must
        # not masquerade as an unannotated physical curriculum snapshot.
        annotated = any(key.startswith("training_reward_") for key in payload)
        if annotated:
            cfg = payload.get("training_reward_config")
            if (payload.get("training_reward_revision") != REWARD_REVISION
                    or payload.get("training_reward_version") != REWARD_VERSION
                    or not isinstance(cfg, dict)
                    or cfg != self.reward_config):
                raise ValueError("Collision reward snapshot revision mismatch")
            if _cost(cfg.get("collision_training_cost")) != self.collision_cost:
                raise ValueError("Collision reward snapshot cost mismatch")
        super().restore(payload)

    def branch(self):
        self._validate_revision()
        result = type(self)(self.config, copy.deepcopy(self.reward_config),
                            collision_cost=self.collision_cost)
        result.restore(self.snapshot())
        return result
