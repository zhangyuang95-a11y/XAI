from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..environment.engine import PongEnvironment


def make_public_features(environment: PongEnvironment, agent: str = "ai") -> dict[str, float]:
    """Return the versioned, named observation used by Pong NN and RCPD.

    Values are public pre-action facts only.  The same keys are used by
    training, extraction and the browser export; do not insert a new key
    without changing the observation signature.
    """
    if agent not in {"ai", "player"}:
        raise ValueError("agent must be 'ai' or 'player'")
    values = environment.features(agent)
    values["role.is_ai"] = 1.0 if agent == "ai" else 0.0
    return values


def feature_names(environment: PongEnvironment) -> tuple[str, ...]:
    return tuple(make_public_features(environment, "ai").keys())


def feature_vector(environment: PongEnvironment, agent: str = "ai") -> np.ndarray:
    values = make_public_features(environment, agent)
    return np.asarray([values[name] for name in feature_names(environment)], dtype=np.float32)


def feature_mapping_from_vector(names: tuple[str, ...], row: np.ndarray) -> dict[str, float]:
    return {name: float(row[index]) for index, name in enumerate(names)}


def action_names() -> tuple[str, ...]:
    return PongEnvironment.ACTIONS


def observation_signature(environment: PongEnvironment) -> dict[str, Any]:
    features = make_public_features(environment)
    return {
        "domain_id": environment.config.domain_id,
        "version": environment.config.version,
        "action_names": list(action_names()),
        "feature_names": list(feature_names(environment)),
        "feature_count": len(features),
    }


def make_nn_adapter(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
    """Small public contract used by callers before they load a Pong Actor."""
    environment = kwargs.get("environment") or PongEnvironment()
    return observation_signature(environment)
