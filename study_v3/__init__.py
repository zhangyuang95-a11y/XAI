"""Versioned, turn-based PolicyLens human-AI study."""

RELEASE_ID = "policylens-three-domain-20260922.v3.8.1"
# v3.8.1 only completes Kitchen counterfactual answers; all v3.8 physical rules,
# menus, tutorial states and scoring are identical. Resume those snapshots as-is,
# retaining their original release IDs. Earlier rule versions remain isolated.
SUPPORTED_RELEASE_IDS = frozenset({RELEASE_ID, "policylens-three-domain-20260922.v3.8"})
