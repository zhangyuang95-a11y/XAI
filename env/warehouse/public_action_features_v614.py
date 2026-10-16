"""Deterministic, pre-action geodesic measurements for the v6.14 Actor.

Each candidate is an isolated static-map hypothetical.  The encoder reads
only the current public observation and physical action mask.  It does not
choose an order or an action, predict the partner's action, or resolve a
simultaneous-motion collision.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json

import numpy as np

from .layouts import STUDY_MAP_LAYOUT
from .navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from .public_observation_v6 import (
    FEATURE_SCHEMA_SHA256 as PUBLIC_FEATURE_SCHEMA_SHA256,
    OBSERVATION_DIM, OBSERVATION_FEATURE_NAMES, OBSERVATION_VERSION,
)


ACTION_FEATURE_VERSION = "warehouse-public-action-geodesic.v6.14.0"
ACTION_FEATURE_NAMES = (
    "legal",
    "target_row",
    "target_column",
    "charger_shortest_distance",
    "order_0_pickup_shortest_distance",
    "order_0_delivery_shortest_distance",
    "order_1_pickup_shortest_distance",
    "order_1_delivery_shortest_distance",
    "remaining_feasible_move_steps",
    "charger_return_margin",
    "partner_shortest_distance",
    "partner_delta_row",
    "partner_delta_column",
)
ACTION_FEATURE_DIM = len(ACTION_FEATURE_NAMES)
_ENDPOINTS = ("order_0_pickup", "order_0_delivery",
              "order_1_pickup", "order_1_delivery")
TREE_EXTRA_FEATURE_NAMES = (
    *(f"progress_to_{endpoint}_if_{action.lower()}"
      for endpoint in _ENDPOINTS for action in ACTIONS),
    *(f"charger_return_margin_if_{action.lower()}" for action in ACTIONS),
)
TREE_EXTRA_DIM = len(TREE_EXTRA_FEATURE_NAMES)
TREE_MASK_START = OBSERVATION_DIM
TREE_EXTRA_START = TREE_MASK_START + len(ACTIONS)
TREE_FEATURE_NAMES = (
    *OBSERVATION_FEATURE_NAMES,
    *(f"legal_{action.lower()}" for action in ACTIONS),
    *TREE_EXTRA_FEATURE_NAMES,
)
TREE_FEATURE_DIM = len(TREE_FEATURE_NAMES)

_INDEX = {name: index for index, name in enumerate(OBSERVATION_FEATURE_NAMES)}
_ROWS, _COLS = STUDY_MAP_LAYOUT.rows, STUDY_MAP_LAYOUT.cols
_ROW_SCALE = float(max(1, _ROWS - 1))
_COL_SCALE = float(max(1, _COLS - 1))
_DISTANCE_SCALE = float(_ROWS * _COLS)
_CAPACITY_SCALE = 50.0
_MOVE_BATTERY_COST = 2.0
_CHARGE_PER_WAIT = 10.0
_CHARGER = STUDY_MAP_LAYOUT.charger_position
_WAIT_INDEX = ACTIONS.index("WAIT")
_WALLS = np.asarray(
    [float(tile == "#") for row in STUDY_MAP_LAYOUT.tiles for tile in row],
    dtype=np.float32,
)
_PASSABLE = np.asarray(
    [[tile != "#" for tile in row] for row in STUDY_MAP_LAYOUT.tiles],
    dtype=bool,
)
_DELTAS = np.asarray(
    [MOVE_DELTAS.get(action, (0, 0)) for action in ACTIONS], dtype=np.intp,
)
_IS_MOVE = np.asarray([action in MOVE_DELTAS for action in ACTIONS], dtype=bool)

if (OBSERVATION_DIM != 141 or ACTION_FEATURE_DIM != 13
        or TREE_EXTRA_DIM != 25 or TREE_FEATURE_DIM != 171):
    raise RuntimeError("Warehouse v6.14 public Actor feature dimensions drifted")

ACTION_PHYSICS_CONFIG = {
    "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
    "rows": STUDY_MAP_LAYOUT.rows,
    "cols": STUDY_MAP_LAYOUT.cols,
    "map_tiles": STUDY_MAP_LAYOUT.tiles,
    "charger_position": STUDY_MAP_LAYOUT.charger_position,
    "move_battery_cost": _MOVE_BATTERY_COST,
    "charge_per_wait": _CHARGE_PER_WAIT,
}
ACTION_PHYSICS_CONFIG_SHA256 = sha256(json.dumps(
    ACTION_PHYSICS_CONFIG, sort_keys=True,
    separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_study_action_physics(config) -> str:
    """Reject an environment whose physics the derived encoder cannot infer."""
    if (config.map_layout_id != STUDY_MAP_LAYOUT.layout_id
            or config.rows != STUDY_MAP_LAYOUT.rows
            or config.cols != STUDY_MAP_LAYOUT.cols
            or float(config.move_battery_cost) != _MOVE_BATTERY_COST
            or float(config.charge_per_wait) != _CHARGE_PER_WAIT):
        raise ValueError("Warehouse v6.14 Actor action physics differs from bound study config")
    return ACTION_PHYSICS_CONFIG_SHA256

ACTION_FEATURE_SCHEMA_SHA256 = sha256(json.dumps({
    "version": ACTION_FEATURE_VERSION,
    "feature_names": ACTION_FEATURE_NAMES,
    "action_order": ACTIONS,
    "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
    "map_tiles": STUDY_MAP_LAYOUT.tiles,
    "row_scale": _ROW_SCALE,
    "column_scale": _COL_SCALE,
    "distance_scale": _DISTANCE_SCALE,
    "capacity_scale": _CAPACITY_SCALE,
    "move_battery_cost": _MOVE_BATTERY_COST,
    "charge_per_wait": _CHARGE_PER_WAIT,
    "invalid_move": "retain_current_position_and_wait_energy",
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

ACTOR_FEATURE_SCHEMA_SHA256 = sha256(json.dumps({
    "public_observation_version": OBSERVATION_VERSION,
    "public_feature_schema_sha256": PUBLIC_FEATURE_SCHEMA_SHA256,
    "action_feature_schema_sha256": ACTION_FEATURE_SCHEMA_SHA256,
    "tree_feature_names": TREE_FEATURE_NAMES,
    "tree_extra_formula": {
        "progress": "WAIT_geodesic_distance_minus_candidate_geodesic_distance_divided_by_map_area",
        "return_margin": "post_action_battery_divided_by_move_cost_minus_candidate_charger_distance_divided_by_capacity_scale",
    },
    "tree_mask_start": TREE_MASK_START,
    "tree_extra_start": TREE_EXTRA_START,
    "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _geodesic_table() -> np.ndarray:
    """Cache study-map BFS distances without running or resetting an episode."""
    size = _ROWS * _COLS
    table = np.full((size, size), size, dtype=np.int16)
    for start in STUDY_MAP_LAYOUT.passable_positions:
        source = start[0] * _COLS + start[1]
        for end in STUDY_MAP_LAYOUT.passable_positions:
            destination = end[0] * _COLS + end[1]
            table[source, destination] = shortest_path_distance(
                start, end, STUDY_MAP_LAYOUT.layout_id)
    table.flags.writeable = False
    return table


def _decode_grid_coordinates(rows: np.ndarray, columns: np.ndarray,
                             name: str) -> np.ndarray:
    scaled = np.stack((rows * _ROW_SCALE, columns * _COL_SCALE), axis=-1)
    rounded = np.rint(scaled)
    if (np.any(np.abs(scaled - rounded) > 1e-4)
            or np.any(rounded[..., 0] < 0)
            or np.any(rounded[..., 0] >= _ROWS)
            or np.any(rounded[..., 1] < 0)
            or np.any(rounded[..., 1] >= _COLS)):
        raise ValueError(f"{name} is not a study-map grid coordinate")
    result = rounded.astype(np.intp)
    if not np.all(_PASSABLE[result[..., 0], result[..., 1]]):
        raise ValueError(f"{name} is not a passable study-map coordinate")
    return result


def _flat(position: np.ndarray) -> np.ndarray:
    return position[..., 0] * _COLS + position[..., 1]


def action_features_from_public_observation(
        obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate ``(..., 5, 13)`` and tree-extra ``(..., 25)`` arrays.

    The first twenty extras are ordered endpoint-major, action-minor and are
    signed WAIT-to-candidate geodesic progress.  The final five are action-
    ordered charger return margins.  Illegal moves retain the current cell;
    their legal bit remains zero and must match the supplied physical mask.
    """
    values = np.asarray(obs, dtype=np.float32)
    physical = np.asarray(mask, dtype=np.float32)
    if (values.ndim < 1 or values.shape[-1] != OBSERVATION_DIM
            or physical.shape != values.shape[:-1] + (len(ACTIONS),)
            or not np.isfinite(values).all()
            or not np.isfinite(physical).all()
            or not np.all((physical == 0.0) | (physical == 1.0))
            or np.any(physical.sum(axis=-1) < 1.0)):
        raise ValueError("invalid Warehouse v6.14 pre-action observation or mask")
    leading_shape = values.shape[:-1]
    base = values.reshape(-1, OBSERVATION_DIM)
    supplied_mask = physical.reshape(-1, len(ACTIONS))

    wall_start = _INDEX["wall_row_0_column_0"]
    if not np.array_equal(base[:, wall_start:wall_start + _WALLS.size],
                          np.broadcast_to(_WALLS, (len(base), len(_WALLS)))):
        raise ValueError("Warehouse public observation map differs from study map")
    charger = _decode_grid_coordinates(
        base[:, _INDEX["charger_row"]],
        base[:, _INDEX["charger_column"]], "charger")
    if not np.all(charger == _CHARGER):
        raise ValueError("Warehouse public charger differs from study map")
    own = _decode_grid_coordinates(
        base[:, _INDEX["self_row"]], base[:, _INDEX["self_column"]], "self")
    partner = _decode_grid_coordinates(
        base[:, _INDEX["partner_row"]],
        base[:, _INDEX["partner_column"]], "partner")
    endpoints = tuple(_decode_grid_coordinates(
        base[:, _INDEX[f"{endpoint}_row"]],
        base[:, _INDEX[f"{endpoint}_column"]], endpoint)
        for endpoint in _ENDPOINTS)
    active = base[:, _INDEX["self_active"]]
    battery = base[:, _INDEX["self_battery_fraction"]] * 100.0
    if (not np.all((active == 0.0) | (active == 1.0))
            or np.any(battery < -1e-4)
            or np.any(battery > 100.0 + 1e-4)):
        raise ValueError("invalid public active or battery value")
    battery = np.clip(battery, 0.0, 100.0)

    candidate = own[:, None, :] + _DELTAS[None, :, :]
    inside = ((candidate[..., 0] >= 0) & (candidate[..., 0] < _ROWS)
              & (candidate[..., 1] >= 0) & (candidate[..., 1] < _COLS))
    clipped = np.stack((np.clip(candidate[..., 0], 0, _ROWS - 1),
                        np.clip(candidate[..., 1], 0, _COLS - 1)), axis=-1)
    passable = inside & _PASSABLE[clipped[..., 0], clipped[..., 1]]
    legal = np.where(_IS_MOVE[None, :],
                     (active[:, None] > 0.5) & passable, True)
    if not np.array_equal(supplied_mask, legal.astype(np.float32)):
        raise ValueError("candidate legality differs from physical action mask")
    target = np.where(legal[..., None], candidate, own[:, None, :])
    executed_move = legal & _IS_MOVE[None, :]
    at_charger = np.all(target == _CHARGER, axis=-1)
    post_battery = np.where(
        executed_move, np.maximum(0.0, battery[:, None] - _MOVE_BATTERY_COST),
        np.where(at_charger,
                 np.minimum(100.0, battery[:, None] + _CHARGE_PER_WAIT),
                 battery[:, None]))
    remaining_steps = post_battery / _MOVE_BATTERY_COST

    table = _geodesic_table()
    target_flat = _flat(target)
    charger_distance = table[target_flat, _CHARGER[0] * _COLS + _CHARGER[1]]
    partner_distance = table[target_flat, _flat(partner)[:, None]]
    endpoint_distances = np.stack(
        [table[target_flat, _flat(endpoint)[:, None]]
         for endpoint in endpoints], axis=-1)
    relative = partner[:, None, :] - target
    return_margin = ((remaining_steps - charger_distance) /
                     _CAPACITY_SCALE).astype(np.float32)
    features = np.stack((
        legal.astype(np.float32),
        target[..., 0] / _ROW_SCALE,
        target[..., 1] / _COL_SCALE,
        charger_distance / _DISTANCE_SCALE,
        *(endpoint_distances[..., index] / _DISTANCE_SCALE
          for index in range(len(_ENDPOINTS))),
        remaining_steps / _CAPACITY_SCALE,
        return_margin,
        partner_distance / _DISTANCE_SCALE,
        relative[..., 0] / _ROW_SCALE,
        relative[..., 1] / _COL_SCALE,
    ), axis=-1).astype(np.float32)
    progress = ((endpoint_distances[:, _WAIT_INDEX, :][:, None, :]
                 - endpoint_distances) / _DISTANCE_SCALE).astype(np.float32)
    extras = np.concatenate((
        np.swapaxes(progress, 1, 2).reshape(len(base), len(_ENDPOINTS) * len(ACTIONS)),
        return_margin,
    ), axis=-1).astype(np.float32)
    return (features.reshape(leading_shape + (len(ACTIONS), ACTION_FEATURE_DIM)),
            extras.reshape(leading_shape + (TREE_EXTRA_DIM,)))


def encode_actor_tree_features(obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return ``obs141 + physical_mask5 + derived_tree_extra25``."""
    values = np.asarray(obs, dtype=np.float32)
    physical = np.asarray(mask, dtype=np.float32)
    _, extras = action_features_from_public_observation(values, physical)
    encoded = np.concatenate((values, physical, extras), axis=-1)
    if encoded.shape[-1] != TREE_FEATURE_DIM or not np.isfinite(encoded).all():
        raise RuntimeError("Warehouse v6.14 Actor tree feature schema drifted")
    return encoded


def action_features_from_tree_encoded(tree_encoded: np.ndarray) -> np.ndarray:
    """Recover the 13 candidate measurements from a validated 171-vector."""
    encoded = np.asarray(tree_encoded, dtype=np.float32)
    if (encoded.ndim < 1 or encoded.shape[-1] != TREE_FEATURE_DIM
            or not np.isfinite(encoded).all()):
        raise ValueError("invalid Warehouse v6.14 Actor tree features")
    features, extras = action_features_from_public_observation(
        encoded[..., :OBSERVATION_DIM],
        encoded[..., TREE_MASK_START:TREE_EXTRA_START])
    if not np.array_equal(encoded[..., TREE_EXTRA_START:], extras):
        raise ValueError("Warehouse v6.14 derived tree features were altered")
    return features


__all__ = [
    "ACTION_FEATURE_VERSION", "ACTION_FEATURE_NAMES", "ACTION_FEATURE_DIM",
    "ACTION_FEATURE_SCHEMA_SHA256", "TREE_EXTRA_FEATURE_NAMES", "TREE_EXTRA_DIM",
    "TREE_MASK_START", "TREE_EXTRA_START", "TREE_FEATURE_NAMES", "TREE_FEATURE_DIM",
    "ACTION_PHYSICS_CONFIG", "ACTION_PHYSICS_CONFIG_SHA256",
    "validate_study_action_physics",
    "ACTOR_FEATURE_SCHEMA_SHA256", "action_features_from_public_observation",
    "encode_actor_tree_features", "action_features_from_tree_encoded",
]
