"""Authoritative, deterministic cooperative kitchen environment.

Actions are simultaneous and both interaction intents read the previous state.
Snapshots include item provenance; public views retain the original demo shape.
No policy is invoked by ``step``. Program partners are training/demo baselines.
"""

from .environment import (
    ACTIONS, ACTOR_IDS, BASE_MAP, OBSERVATION_FEATURES, OBSERVATION_DIM,
    STATE_DIM, SCENARIOS, CooperativeKitchen, KitchenConfig, program_decision,
)

__all__ = [
    "ACTIONS", "ACTOR_IDS", "BASE_MAP", "OBSERVATION_FEATURES", "OBSERVATION_DIM",
    "STATE_DIM", "SCENARIOS", "CooperativeKitchen", "KitchenConfig", "program_decision",
]
