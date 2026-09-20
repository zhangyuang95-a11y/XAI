"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.3"
# These interface/enrollment revisions share identical gameplay and stored state.
# Existing records keep their original release IDs; no state migration is applied.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260920.v3.2"})
