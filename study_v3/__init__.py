"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.5"
# Kitchen preparation, menu, freshness and scoring changed. Preserve earlier
# runs verbatim; enroll separately instead of reinterpreting their game states.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID})
