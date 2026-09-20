"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.4"
# Pong schedules and kitchen interactions changed. Prior runs remain archived
# verbatim; returning participants enroll afresh rather than migrating a state.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID})
