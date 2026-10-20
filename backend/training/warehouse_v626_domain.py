"""Complete immutable public domains for the unchanged v6.20 Actor encoder.

No observed sample extrema are used. Bounds follow the bound study map,
encoder formulas, 120-frame horizon, and 2/10 energy physics only.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.public_action_features_v614 import _geodesic_table, ACTION_PHYSICS_CONFIG_SHA256
from .warehouse_v6_actor import FEATURE_NAMES, FEATURE_SCHEMA_SHA256
from .warehouse_v622_program import digest

VERSION = 'warehouse-public-domain.v6.26-complete'
NORMALIZATION = 'fp32-subtract-public-midpoint-then-multiply-public-scale.v1'


def outward(lo, hi):
    a, b = np.float32(lo), np.float32(hi)
    if float(a) > lo: a = np.nextafter(a, np.float32(-np.inf))
    if float(b) < hi: b = np.nextafter(b, np.float32(np.inf))
    return float(a), float(b)


@lru_cache(maxsize=1)
def domain_table():
    positions = tuple(STUDY_MAP_LAYOUT.passable_positions)
    flat = [r * STUDY_MAP_LAYOUT.cols + c for r, c in positions]
    diameter = int(_geodesic_table()[np.ix_(flat, flat)].max())
    area = STUDY_MAP_LAYOUT.rows * STUDY_MAP_LAYOUT.cols
    rows = []
    for name in FEATURE_NAMES:
        reason, constant = 'public_indicator_or_fraction', None
        if name.startswith('wall_row_'):
            bits = name.split('_'); r, c = int(bits[2]), int(bits[4])
            constant = float(STUDY_MAP_LAYOUT.tiles[r][c] == '#')
        elif name in ('charger_row', 'charger_column'):
            axis = 0 if name.endswith('row') else 1
            scale = STUDY_MAP_LAYOUT.rows - 1 if axis == 0 else STUDY_MAP_LAYOUT.cols - 1
            constant = float(np.float32(STUDY_MAP_LAYOUT.charger_position[axis] / scale))
        if constant is not None:
            lo = hi = constant; reason = 'bound_map_constant'
        elif name.startswith('progress_to_'):
            if name.endswith('_if_wait'):
                lo = hi = 0.; reason = 'identical_wait_distance'
            else:
                lo, hi = -1 / area, 1 / area; reason = 'geodesic_one_edge_triangle_inequality'
        elif name.endswith('last_battery_delta'):
            lo, hi = -.2, 1.; reason = 'move_cost_2_charge_10_divided_by_10'
        elif name.endswith('remaining_mission_distance'):
            lo, hi = 0., 3 * diameter / area; reason = 'at_most_three_map_paths'
        elif name.endswith('charger_shortest_distance') or name == 'robots_shortest_distance':
            lo, hi = 0., diameter / area; reason = 'finite_bound_map_diameter'
        elif 'charger_return_margin' in name:
            multiplier = 3 if name.startswith('order_') else 1
            lo, hi = -multiplier * diameter / 50., 1.; reason = 'capacity_0_50_minus_mission_distance'
        elif (name == 'elapsed_fraction' or name.endswith(('_row', '_column', '_battery_fraction', '_active',
                '_feasible_move_steps', '_steps_since_charging', '_charger_wait_streak'))
                or '_carrying_task_' in name or '_heading_' in name or '_previous_action_' in name
                or '_last_submitted_' in name or name.startswith('legal_')
                or name.startswith(('order_0_', 'order_1_'))
                or name in ('same_target_collision_streak_capped', 'public_hard_risk_context', 'public_near_robots_context')):
            lo, hi = 0., 1.
        else:
            raise ValueError('public feature has no derived domain: ' + name)
        lo, hi = outward(lo, hi)
        center = float(np.float32((lo + hi) / 2.)) if lo != hi else lo
        scale = float(np.float32(2. / (hi - lo))) if lo != hi else 0.
        rows.append(dict(name=name, lower=lo, upper=hi, center=center, scale=scale,
                         constant=lo == hi, reason=reason))
    return dict(version=VERSION, normalization=NORMALIZATION, feature_schema_sha256=FEATURE_SCHEMA_SHA256,
        action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256, horizon=120,
        map_diameter=diameter, map_area=area, features=rows)


def domain_sha(): return digest(domain_table())


def validate_rows(x):
    x = np.asarray(x, np.float32)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError('invalid complete public feature array')
    for i, item in enumerate(domain_table()['features']):
        if np.any((x[:, i] < item['lower']) | (x[:, i] > item['upper'])):
            raise ValueError('outside proven public domain: ' + item['name'])
    return x


def transform(x, support):
    x = np.asarray(x, np.float32)
    d = domain_table()['features']
    return np.column_stack([np.multiply(np.subtract(x[:, i], np.float32(d[i]['center']), dtype=np.float32),
        np.float32(d[i]['scale']), dtype=np.float32) for i in support]) if support else np.zeros((len(x), 0), np.float32)
