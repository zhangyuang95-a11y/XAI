"""Cooperative Pong domain.

The package is intentionally independent from the Warehouse environment.  It
contains deterministic physics, a diagnostic controller, evidence-grounded
explanations, and a small local web adapter.
"""

from .config import PongConfig
from .environment.engine import PongEnvironment
from .study import PongStudySession

__all__ = ["PongConfig", "PongEnvironment", "PongStudySession"]
