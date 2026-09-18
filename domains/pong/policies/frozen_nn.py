"""Read-only adapter for a frozen Pong neural policy in non-browser runners."""
from __future__ import annotations

from typing import Callable, Mapping

from ..adapters.core_adapter import make_public_features
from ..environment.engine import PongEnvironment
from .rule_demo import ControllerDecision


class FrozenNNController:
    version = "pong-frozen-nn.v1"

    def __init__(self, environment: PongEnvironment, policy: Callable[[Mapping[str, float]], Mapping[str, float]]) -> None:
        self.environment = environment
        self.policy = policy

    def reset(self) -> None:
        return None

    def choose(self, _frame) -> ControllerDecision:
        features = make_public_features(self.environment, "ai")
        probabilities = self.policy(features)
        action = max(("left", "right", "stay"), key=lambda name: float(probabilities.get(name, 0.0)))
        return ControllerDecision(action=action, reason="冻结NN根据当前公开观察选择该动作。", intent_type="nn_action")
