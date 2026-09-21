"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260922.v3.8"
# Kitchen heating, freshness, menus and scoring changed. Retain old study records without resuming them
# under different mechanics or mixing their scores with this release.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID})
