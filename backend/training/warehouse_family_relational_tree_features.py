"""Deterministic public comparisons for a tree, appended to genuine observed197.

These features never change the Actor input or choose a target/action. Distance
columns are *encoded* distances: 1 also represents unreachable destinations or
illegal neighbors, and missing task slots use 0. Subtracting those codes is not
always a difference of finite shortest-path lengths. We preserve the codes
without masking, inferring reachability or selecting a preferred direction.
"""
from itertools import combinations

import numpy as np

from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.observations import observation_names
from .warehouse_native_public_feedback import HISTORY_FEATURE_NAMES

VERSION = "warehouse-family-public-relational-tree-features.v1"
INPUT_SIZE = 197
ADDED_SIZE = 71
OUTPUT_SIZE = 268
TARGETS = ("charger", "task.0.pickup", "task.0.delivery", "task.1.pickup", "task.1.delivery")
ROLES = ("self", "other")
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

# Reading the original immutable configuration creates no environment.
_CONFIG = collaborative_study_config()
DISTANCE_SCALE = max(_CONFIG.rows * _CONFIG.cols - 1, 1)
MOVE_BATTERY_COST = _CONFIG.move_battery_cost
if (_CONFIG.rows, _CONFIG.cols, DISTANCE_SCALE, MOVE_BATTERY_COST) != (6, 7, 41, 2.):
    raise ValueError("Relational tree features require the original 6x7/41/2 configuration")
ORIGINAL_NAMES = tuple(observation_names(_CONFIG)) + tuple(HISTORY_FEATURE_NAMES)
if len(ORIGINAL_NAMES) != INPUT_SIZE or len(set(ORIGINAL_NAMES)) != INPUT_SIZE:
    raise ValueError("The original observed197 column contract differs")


def _relations():
    """Each entry is (readable name, left input column, right column, multiplier)."""
    values = []
    for target in TARGETS:
        for role in ROLES:
            for left, right in combinations(DIRECTIONS, 2):
                values.append((f"relation.{target}.{role}.neighbor_encoded_distance.{left}_minus_{right}",
                    f"{target}.{role}.neighbor_distance.{left}",
                    f"{target}.{role}.neighbor_distance.{right}", 1.))
    for target in TARGETS:
        values.append((f"relation.{target}.path_encoded_distance.self_minus_other",
            f"{target}.self.path_distance", f"{target}.other.path_distance", 1.))
    for role in ROLES:
        for destination in ("pickup", "delivery"):
            values.append((f"relation.{role}.{destination}.path_encoded_distance.task0_minus_task1",
                f"task.0.{destination}.{role}.path_distance",
                f"task.1.{destination}.{role}.path_distance", 1.))
    for role in ROLES:
        values.append((f"relation.charger.{role}.battery_minus_encoded_move_cost",
            f"{role}.battery", f"charger.{role}.path_distance",
            MOVE_BATTERY_COST * DISTANCE_SCALE / 100.))
    return tuple(values)


RELATIONS = _relations()
_OUTPUT_NAMES = ORIGINAL_NAMES + tuple(row[0] for row in RELATIONS)
_LEFT = tuple(ORIGINAL_NAMES.index(row[1]) for row in RELATIONS)
_RIGHT = tuple(ORIGINAL_NAMES.index(row[2]) for row in RELATIONS)
if len(RELATIONS) != ADDED_SIZE or len(set(_OUTPUT_NAMES)) != OUTPUT_SIZE:
    raise ValueError("Exactly 71 unique appended relational columns are required")


def _validate_names(feature_names):
    if feature_names is not None and tuple(feature_names) != ORIGINAL_NAMES:
        raise ValueError("Input names must match the original observed197 order exactly")


def names(feature_names=None):
    """Original 197 names followed by the fixed 60+5+4+2 comparisons."""
    _validate_names(feature_names)
    return _OUTPUT_NAMES


def transform(observations, feature_names=None):
    """Copy float32 (197,) or (N,197) into (268,) or (N,268), respectively.

    Finite values are required, but encoded values are not clipped or otherwise
    reinterpreted. The input is never mutated and the result owns its storage.
    """
    _validate_names(feature_names)
    if (not isinstance(observations, np.ndarray) or observations.dtype != np.dtype(np.float32)
            or observations.ndim not in (1, 2) or observations.shape[-1] != INPUT_SIZE):
        raise ValueError("Expected original float32 observed197 vector or matrix")
    if not np.isfinite(observations).all():
        raise ValueError("Observed197 values must all be finite")
    result = np.empty((*observations.shape[:-1], OUTPUT_SIZE), dtype=np.float32)
    result[..., :INPUT_SIZE] = observations
    factors = np.asarray([row[3] for row in RELATIONS], dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        result[..., INPUT_SIZE:] = observations[..., _LEFT] - factors * observations[..., _RIGHT]
    if not np.isfinite(result).all():
        raise ValueError("Relational feature arithmetic must remain finite in float32")
    return result


def contract(feature_names=None):
    """Serializable formula/column contract; no model, labels or state access."""
    _validate_names(feature_names)
    return dict(version=VERSION, input_size=INPUT_SIZE, added_size=ADDED_SIZE,
        output_size=OUTPUT_SIZE, dtype="float32", original_columns_preserved=True,
        actor_input_changed=False, target_or_action_selection=False,
        input_names=list(ORIGINAL_NAMES), output_names=list(_OUTPUT_NAMES),
        configuration=dict(map_layout_id=_CONFIG.map_layout_id, rows=_CONFIG.rows,
            cols=_CONFIG.cols, encoded_distance_scale=DISTANCE_SCALE,
            move_battery_cost=MOVE_BATTERY_COST, battery_normalization=100.),
        counts=dict(neighbor_distance_differences=60, self_other_path_differences=5,
            task0_task1_path_differences=4, charger_encoded_energy_margins=2),
        appended_columns=[dict(name=name, index=INPUT_SIZE+i, left=left, right=right,
            left_index=_LEFT[i], right_index=_RIGHT[i], right_multiplier=factor,
            formula="left - right_multiplier * right")
            for i, (name, left, right, factor) in enumerate(RELATIONS)],
        encoded_distance_semantics="min(1, finite_shortest_path_length/41); unreachable or illegal neighbor=1; missing task slot=0. Differences retain these codes and are not always finite shortest-path differences.",
        charger_margin_semantics="normalized_battery - (2*41/100)*encoded_charger_path_distance; an encoded movement-cost comparison, not a reachability certificate or a collision/charging/reserve-aware guarantee.",
        source_of_inputs="original public observed197 only; no Actor/logits/probability/hidden-target/oracle input",
        configuration_source="env/warehouse/domain.py:collaborative_study_config",
        distance_encoding_source="env/warehouse_native/observations.py:public_observations")
