"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.7"
# Pong now fixes AI-led catch targets. Preserve older records verbatim;
# new enrollments start fresh instead of applying new rules to saved games.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID})
