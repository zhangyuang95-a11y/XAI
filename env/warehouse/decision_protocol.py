"""Causal joint-decision hashing and order-independent sampling helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np

from core.policy_contracts import ActionDistribution


DECISION_AUDIT_SCHEMA = "warehouse-joint-decision-audit.v1"


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=repr)
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def independent_agent_seed(
    *,
    base_seed: int,
    episode_id: int,
    frame: int,
    agent_id: str,
) -> int:
    """Derive one stable stream without depending on Actor iteration order."""

    digest = sha256(
        f"{int(base_seed)}:{int(episode_id)}:{int(frame)}:{agent_id}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def serialized_action_distributions(
    distributions: Mapping[str, ActionDistribution],
) -> dict[str, dict[str, Any]]:
    return {
        agent_id: {
            "actions": list(distribution.actions),
            "probabilities": list(distribution.probabilities),
            "logits": list(distribution.logits),
            "action_mask": list(distribution.action_mask),
            "proposed_action": distribution.proposed_action,
        }
        for agent_id, distribution in sorted(distributions.items())
    }


def distribution_decision_metadata(
    distributions: Mapping[str, ActionDistribution],
    *,
    decision_source: str,
    participant_overrides: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    payload = serialized_action_distributions(distributions)
    return {
        "decision_source": str(decision_source),
        "action_distributions": payload,
        "action_distribution_hash": canonical_sha256(payload),
        "participant_overrides": {
            str(agent_id): str(action)
            for agent_id, action in sorted(
                (participant_overrides or {}).items()
            )
        },
    }
