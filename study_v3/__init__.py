"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260920.v3.5.2"
# Entry-readiness recovery leaves v3.5/v3.5.1 gameplay and saved records intact.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260920.v3.5.1",
                                   "policylens-three-domain-20260920.v3.5"})
