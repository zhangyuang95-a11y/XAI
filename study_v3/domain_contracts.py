"""The five explicit layers of a frozen two-Actor study domain.

The participant adapter is the only layer exposed to the study store. A model
and an audited program may use the same pre-action observation; only the model
can choose an action. The physical engine remains responsible for legal moves,
collisions and scoring. Existing domain module paths remain importable shims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class DomainLayers:
    participant_adapter: str
    physical_engine: str
    observation_encoder: str
    frozen_actor: str
    audited_program: str


LAYERS = {
    "warehouse": DomainLayers(
        "domains.warehouse.neural_turnbased", "env.warehouse.environment",
        "env.warehouse.public_observation_v6", "backend.training.warehouse_v6_actor",
        "backend.training.warehouse_v6_program"),
    "pong": DomainLayers(
        "domains.pong.turnbased", "domains.pong.turnbased",
        "domains.pong.nn_policy", "domains.pong.nn_policy", "domains.pong.nn_policy"),
    "kitchen": DomainLayers(
        "domains.kitchen.neural_domain", "domains.kitchen.engine_v5_frozen",
        "domains.kitchen.neural_policy", "domains.kitchen.neural_policy",
        "domains.kitchen.neural_domain"),
}


class ParticipantAdapter(Protocol):
    DOMAIN: str
    VERSION: str
    RELEASE_VERSION: str

    def initial_state(self, seed: int, task: int, policy_id: str,
                      release_id: str | None = None) -> dict[str, Any]: ...
    def demonstration(self, policy_id: str,
                      release_id: str | None = None) -> dict[str, Any]: ...
    def decide(self, state: dict[str, Any]) -> dict[str, Any]: ...
    def step(self, state: dict[str, Any], human_action: str,
             decision: dict[str, Any] | None = None) -> dict[str, Any]: ...
    def legal_actions(self, state: dict[str, Any], actor: str = "human") -> list[str]: ...
    def public_state(self, state: dict[str, Any]) -> dict[str, Any]: ...
    def score(self, state: dict[str, Any]) -> dict[str, Any]: ...


def validate_adapter(domain: str, module: object) -> None:
    if domain not in LAYERS or getattr(module, "DOMAIN", None) != domain:
        raise RuntimeError("study_domain_adapter_identity_mismatch")
    required = ("initial_state", "demonstration", "decide", "step",
                "legal_actions", "public_state", "score", "rules")
    if any(not callable(getattr(module, name, None)) for name in required):
        raise RuntimeError("study_domain_adapter_incomplete")
