"""Versioned public context features for the risk-stratified neural Actor.

The context bit chooses no action. Its two source measurements are counted
against the three-feature predicate budget. The v6.14 encoder stays intact
for historical frozen models.
"""
from hashlib import sha256
import json
import numpy as np

from . import public_action_features_v614 as legacy
from .public_action_features_v614 import (  # unchanged physical measurements
    ACTION_FEATURE_DIM, ACTION_PHYSICS_CONFIG, ACTION_PHYSICS_CONFIG_SHA256,
    TREE_MASK_START, validate_study_action_physics,
)

TREE_FEATURE_NAMES = (*legacy.TREE_FEATURE_NAMES, "public_hard_risk_context",
                      "public_near_robots_context")
TREE_FEATURE_DIM = len(TREE_FEATURE_NAMES)
CONTEXT_INDEX = legacy.TREE_FEATURE_DIM
NEAR_CONTEXT_INDEX = CONTEXT_INDEX + 1
CONTEXT_SOURCES = ("same_target_collision_streak_capped", "self_charger_return_margin")
CONTEXT_RECIPE = {"version": "warehouse-public-context.v6.17",
                  "sources": CONTEXT_SOURCES,
                  "formula": "streak > 0 OR return_margin * 50 <= 4",
                  "source_feature_cost": 2}
ACTOR_FEATURE_SCHEMA_SHA256 = sha256(json.dumps({
    "base_schema": legacy.ACTOR_FEATURE_SCHEMA_SHA256,
    "features": TREE_FEATURE_NAMES, "context": CONTEXT_RECIPE,
    "near_formula": "robots_shortest_distance * 42 <= 2",
}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def append_context(base: np.ndarray) -> np.ndarray:
    x = np.asarray(base, dtype=np.float32)
    if x.shape[-1:] != (legacy.TREE_FEATURE_DIM,):
        raise ValueError("Warehouse v6.17 base feature dimensions differ")
    streak = x[..., legacy.TREE_FEATURE_NAMES.index(CONTEXT_SOURCES[0])]
    margin = x[..., legacy.TREE_FEATURE_NAMES.index(CONTEXT_SOURCES[1])]
    risk = ((streak > 0) | (margin * np.float32(50) <= np.float32(4)))
    distance = x[..., legacy.TREE_FEATURE_NAMES.index("robots_shortest_distance")]
    near = distance * np.float32(42) <= np.float32(2)
    return np.concatenate((x, risk[..., None].astype(np.float32),
                           near[..., None].astype(np.float32)), axis=-1)


def encode_actor_tree_features(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return append_context(legacy.encode_actor_tree_features(obs, mask))


def action_features_from_tree_encoded(encoded: np.ndarray) -> np.ndarray:
    x = np.asarray(encoded, dtype=np.float32)
    if x.shape[-1:] != (TREE_FEATURE_DIM,) or not np.array_equal(
            append_context(x[..., :legacy.TREE_FEATURE_DIM]), x):
        raise ValueError("Warehouse v6.17 public context was altered")
    return legacy.action_features_from_tree_encoded(x[..., :legacy.TREE_FEATURE_DIM])
