from __future__ import annotations

from typing import Any

from ..environment.engine import PongEnvironment


def make_public_features(environment: PongEnvironment, agent: str = "ai") -> dict[str, float]:
    return environment.features(agent)


def action_names() -> tuple[str, ...]:
    return PongEnvironment.ACTIONS


def observation_signature(environment: PongEnvironment) -> dict[str, Any]:
    features = make_public_features(environment)
    return {
        "domain_id": environment.config.domain_id,
        "version": environment.config.version,
        "action_names": list(action_names()),
        "feature_names": list(features),
        "feature_count": len(features),
    }


def make_nn_adapter(*args: Any, **kwargs: Any) -> None:
    raise NotImplementedError(
        "Pong NN training is not bundled yet. Use RuleDemoController for local physics checks."
    )
