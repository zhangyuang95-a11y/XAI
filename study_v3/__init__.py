"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.4.1"
# v3.4.1 clarifies explanation evidence only: physics, fixed decisions, scenarios
# and state schemas match v3.4. Earlier gameplay stays archived without migration.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260920.v3.4"})
