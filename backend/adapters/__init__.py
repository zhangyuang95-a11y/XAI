"""Environment-independent interfaces used by the XAI stack."""

from .base import (
    ActionDistribution,
    CandidateIntervention,
    EnvironmentAdapter,
    EnvironmentSnapshot,
    Intervention,
    PolicyProtocol,
    RolloutFrame,
    RolloutResult,
)

__all__ = [
    "ActionDistribution",
    "CandidateIntervention",
    "EnvironmentAdapter",
    "EnvironmentSnapshot",
    "Intervention",
    "PolicyProtocol",
    "RolloutFrame",
    "RolloutResult",
]
