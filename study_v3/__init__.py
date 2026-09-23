"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260923.v3.13-auto-explanations"
KITCHEN_SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260923.v3.12-replacements", "policylens-three-domain-20260922.v3.11-consent", "policylens-three-domain-20260922.v3.10-prolific", "policylens-three-domain-20260922.v3.9"})
# Kitchen scoring changes require a new enrollment; older Kitchen runs remain
# archived under their original rules. Warehouse and Pong can still resume.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260923.v3.12-replacements", "policylens-three-domain-20260922.v3.11-consent", "policylens-three-domain-20260922.v3.10-prolific",
    "policylens-three-domain-20260922.v3.9",
    "policylens-three-domain-20260922.v3.8",
    "policylens-three-domain-20260922.v3.8.1",
    "policylens-three-domain-20260922.v3.8.2"})
