"""Deterministic, simultaneous kitchen dynamics and transparent program partners.

The observation vocabulary is geometric/state information, never teacher actions.
Snapshots are JSON objects with public fields and private ``_`` item provenance.
"""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any

import numpy as np

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT")
ACTOR_IDS = ("human", "ai")
DIRECTIONS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ITEMS = (None, "onion", "plate", "soup")
BASE_MAP = ("#########", "#I.X#X.D#", "#...C...#", "#H..#..A#", "#...C...#", "#S..#..P#", "#########")
COUNTER_KEYS = ("2,4", "4,4")
LAYOUTS = {
    "base": BASE_MAP,
    "mirror": tuple(reversed(BASE_MAP)),
    "detour": tuple(row if i != 3 else "#H#.#.#A#" for i, row in enumerate(BASE_MAP)),
    "asymmetric": tuple("#.#.C...#" if i == 2 else "#...C.#.#" if i == 4 else row for i, row in enumerate(BASE_MAP)),
}
SCENARIOS = tuple(f"{layout}_{initial}" for layout in LAYOUTS for initial in ("empty", "inprogress", "congestion"))
STATION_NAMES = ("ingredient", "plate", "pot", "serve", "left_trash", "right_trash", "upper_counter", "lower_counter")
FRONT_TILES = ("#", ".", "I", "D", "P", "S", "X", "C")

@dataclass(frozen=True)
class KitchenConfig:
    horizon: int = 180
    target_orders: int = 2
    cooking_steps: int = 4
    discount: float = 0.99
    time_cost: float = 0.01
    serve_reward: float = 10.0
    shaping_scale: float = 1.0

    def __post_init__(self):
        if self.horizon < 1 or self.target_orders < 1 or self.cooking_steps != 4:
            raise ValueError("Kitchen requires positive limits and four subsequent cooking steps")
        if not 0 < self.discount <= 1 or self.time_cost < 0 or self.shaping_scale < 0:
            raise ValueError("Invalid reward configuration")


def _features():
    names = ["self_row", "self_col", "partner_row", "partner_col", "role_left", "role_right"]
    for prefix in ("self_facing", "partner_facing"):
        names += [f"{prefix}_{action.lower()}" for action in DIRECTIONS]
    for prefix in ("self_holding", "partner_holding", "upper_counter_item", "lower_counter_item"):
        names += [f"{prefix}_{item or 'empty'}" for item in ITEMS]
    names += ["pot_ingredients", "pot_remaining", "pot_ready", "orders", "remaining_time"]
    names += [f"front_{t if t.isalpha() else 'wall' if t == '#' else 'floor'}" for t in FRONT_TILES]
    for name in STATION_NAMES:
        names += [f"{name}_row_delta", f"{name}_col_delta", f"{name}_interaction_distance"]
        names += [f"{name}_distance_after_{action.lower()}" for action in DIRECTIONS]
    names += [f"direction_{action.lower()}_walkable" for action in DIRECTIONS]
    names += ["supplied_onions", "needed_onions", "empty_counter_count", "done"]
    return tuple(names)

OBSERVATION_FEATURES = _features()
OBSERVATION_DIM = len(OBSERVATION_FEATURES)
STATE_DIM = OBSERVATION_DIM * 2


def _key(p): return f"{p[0]},{p[1]}"
def _pos(k): return tuple(int(x) for x in k.split(","))
def _add(p, d): return (p[0] + d[0], p[1] + d[1])
def _tile(grid, p): return grid[p[0]][p[1]] if 0 <= p[0] < 7 and 0 <= p[1] < 9 else "#"
def _walkable(grid, p): return _tile(grid, p) in (".", "H", "A")

@lru_cache(maxsize=1024)
def _stations(grid):
    found = {name: [] for name in ("I", "D", "P", "S", "X")}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell in found: found[cell].append((r, c))
    return dict(ingredient=found["I"][0], plate=found["D"][0], pot=found["P"][0], serve=found["S"][0],
                left_trash=next(p for p in found["X"] if p[1] < 4), right_trash=next(p for p in found["X"] if p[1] > 4),
                upper_counter=(2, 4), lower_counter=(4, 4))

@lru_cache(maxsize=500000)
def _route(grid, start, facing, target):
    """Demo-equivalent BFS with terminal turn cost; static geometry is cached."""
    queue = deque([(start, None, 0, facing)])
    visited = {start}
    best_action, best_distance = "WAIT", 99
    while queue:
        p, first, distance, direction_at_p = queue.popleft()
        for direction, delta in DIRECTIONS.items():
            if _add(p, delta) == target:
                turn_cost = int(direction_at_p != direction)
                if distance + turn_cost + 1 < best_distance:
                    best_action = first or (direction if turn_cost else "INTERACT")
                    best_distance = distance + turn_cost + 1
        for direction, delta in DIRECTIONS.items():
            q = _add(p, delta)
            if not _walkable(grid, q) or q in visited: continue
            visited.add(q)
            queue.append((q, first or direction, distance + 1, direction))
    return best_action, best_distance

@lru_cache(maxsize=256)
def _geometry_table(grid):
    result = {}
    stations = _stations(grid)
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell not in (".", "H", "A"): continue
            for facing in DIRECTIONS:
                p = (r, c)
                front = _tile(grid, _add(p, DIRECTIONS[facing]))
                if front in ("H", "A"): front = "."
                features = [float(front == t) for t in FRONT_TILES]
                for target in stations.values():
                    features += [(target[0] - r) / 6, (target[1] - c) / 8, _route(grid, p, facing, target)[1] / 20]
                    for action, delta in DIRECTIONS.items():
                        q = _add(p, delta)
                        if not _walkable(grid, q): q = p
                        features.append(_route(grid, q, action, target)[1] / 20)
                features += [float(_walkable(grid, _add(p, delta))) for delta in DIRECTIONS.values()]
                result[(p, facing)] = features
    return result


def _valid_layout(grid):
    """Every floor and required facility remains reachable within each region."""
    for left in (True, False):
        floors = {(r, c) for r in range(7) for c in range(9)
                  if _walkable(grid, (r, c)) and (c < 4) == left}
        if not floors: return False
        seen, queue = {min(floors)}, deque([min(floors)])
        while queue:
            p = queue.popleft()
            for delta in DIRECTIONS.values():
                q = _add(p, delta)
                if q in floors and q not in seen: seen.add(q); queue.append(q)
        if seen != floors: return False
        for r, row in enumerate(grid):
            for c, tile in enumerate(row):
                if tile not in "IDPSXC" or (tile != "C" and (c < 4) != left): continue
                if not any(_add((r, c), d) in floors for d in DIRECTIONS.values()): return False
    return True


@lru_cache(maxsize=4096)
def _generated_scenario(seed):
    rng = np.random.default_rng(seed)
    base = LAYOUTS["mirror" if rng.integers(2) else "base"]
    grid = [list(row.replace("H", ".").replace("A", ".")) for row in base]
    for left in (True, False):
        candidates = [(r, c) for r in range(1, 6) for c in range(1, 8)
                      if grid[r][c] == "." and (c < 4) == left]
        rng.shuffle(candidates)
        blocked = 0
        for r, c in candidates:
            if blocked >= int(seed % 3): break
            grid[r][c] = "#"
            if _valid_layout(tuple("".join(row) for row in grid)): blocked += 1
            else: grid[r][c] = "."
    grid = tuple("".join(row) for row in grid)
    starts = []
    facings = []
    for left in (True, False):
        positions = [(r, c) for r in range(7) for c in range(9)
                     if _walkable(grid, (r, c)) and (c < 4) == left]
        starts.append(positions[int(rng.integers(len(positions)))])
        facings.append(ACTIONS[int(rng.integers(4))])
    initial = ("empty", "inprogress", "congestion")[int(rng.integers(3))]
    return grid, tuple(starts), tuple(facings), initial


class CooperativeKitchen:
    def __init__(self, config=None, seed=0, scenario_id="base_empty"):
        self.config = config or KitchenConfig()
        self.reset(seed=seed, scenario_id=scenario_id)

    @property
    def state(self): return self._state

    def _new_item(self, kind, location, **extra):
        serial = self._state["_next_item_id"]
        self._state["_next_item_id"] += 1
        item_id = f"i{serial:06d}"
        self._state["_items"][item_id] = dict(id=item_id, kind=kind, location=location, created_turn=self._state["turn"], **extra)
        return item_id

    def _new_batch(self):
        serial = self._state["_next_batch_id"]
        self._state["_next_batch_id"] += 1
        batch = f"b{serial:06d}"
        self._state["_batches"][batch] = {"id": batch, "ingredient_ids": [], "created_turn": self._state["turn"], "served_turn": None}
        self._state["_pot_batch"] = batch
        return batch

    def reset(self, seed=None, scenario_id=None):
        seed = int(seed if seed is not None else getattr(self, "seed", 0))
        scenario_id = scenario_id or getattr(self, "scenario_id", "base_empty")
        if scenario_id == "train": scenario_id = SCENARIOS[seed % len(SCENARIOS)]
        if scenario_id not in SCENARIOS and scenario_id != "generated": raise ValueError(f"Unknown kitchen scenario: {scenario_id}")
        self.seed, self.scenario_id = seed, scenario_id
        if scenario_id == "generated":
            self.grid, starts, facings, initial = _generated_scenario(seed)
        else:
            layout, initial = scenario_id.split("_", 1)
            self.grid, starts, facings = LAYOUTS[layout], ((3, 1), (3, 7)), ("UP", "UP")
        self.stations = _stations(self.grid)
        self._geometry = _geometry_table(self.grid)
        self._state = {
            "schema": 1, "environment_version": "cooperative_kitchen_v1", "preset": "supply", "turn": 0,
            "maxSteps": self.config.horizon, "targetOrders": self.config.target_orders,
            "orders": 0, "done": False, "reason": None, "scenario_id": scenario_id, "seed": seed,
            "map": list(self.grid), "actors": [
                {"id": "human", "side": "left", "position": list(starts[0]), "facing": facings[0], "holding": None, "_held_id": None},
                {"id": "ai", "side": "right", "position": list(starts[1]), "facing": facings[1], "holding": None, "_held_id": None},
            ], "pot": {"ingredients": 0, "remaining": 0, "ready": False},
            "counters": {k: None for k in COUNTER_KEYS}, "_counter_item_ids": {k: None for k in COUNTER_KEYS},
            "_items": {}, "_batches": {}, "_pot_batch": None,
            "_next_item_id": 1, "_next_batch_id": 1, "_last_events": [], "_first_serve_turn": None,
            "_config": asdict(self.config),
        }
        if initial == "inprogress":
            batch = self._new_batch()
            for _ in range(2):
                item = self._new_item("onion", f"pot:{batch}")
                self._state["_batches"][batch]["ingredient_ids"].append(item)
            self._state["pot"]["ingredients"] = 2
            self._put_initial_counter("2,4", "onion")
        elif initial == "congestion":
            # A real, recoverable soup handoff with both counters occupied.
            batch = self._new_batch()
            for _ in range(3):
                onion = self._new_item("onion", f"consumed:{batch}")
                self._state["_batches"][batch]["ingredient_ids"].append(onion)
            plate = self._new_item("plate", f"consumed:{batch}")
            soup = self._new_item("soup", "actor:ai", batch_id=batch, plate_id=plate)
            self._state["actors"][1].update(holding="soup", _held_id=soup)
            self._state["_pot_batch"] = None
            self._put_initial_counter("2,4", "onion")
            self._put_initial_counter("4,4", "plate")
        return self.public_view()

    def _put_initial_counter(self, key, kind):
        self._state["counters"][key] = kind
        self._state["_counter_item_ids"][key] = self._new_item(kind, f"counter:{key}")

    def snapshot(self): return copy.deepcopy(self._state)

    def swap_roles(self):
        """Swap empty-round assignments for freeplay, never mid-round progress."""
        if self._state["turn"] != 0:
            raise ValueError("Reset before exchanging roles")
        human, ai = self._state["actors"]
        self._state["actors"] = [dict(ai, id="human"), dict(human, id="ai")]
        self._state["preset"] = "cook" if self._state["preset"] == "supply" else "supply"
        for item in self._state["_items"].values():
            if item["location"] == "actor:human": item["location"] = "actor:ai"
            elif item["location"] == "actor:ai": item["location"] = "actor:human"
        return self.public_view()

    def restore(self, snapshot):
        s = copy.deepcopy(snapshot)
        if not isinstance(s, dict) or s.get("environment_version") != "cooperative_kitchen_v1":
            raise ValueError("Invalid kitchen snapshot version")
        config = KitchenConfig(**s["_config"])
        scenario_id = s["scenario_id"]
        if scenario_id == "generated": expected_map = _generated_scenario(int(s["seed"]))[0]
        elif scenario_id in SCENARIOS: expected_map = LAYOUTS[scenario_id.split("_", 1)[0]]
        else: raise ValueError("Unknown snapshot scenario")
        if tuple(s["map"]) != expected_map:
            raise ValueError("Snapshot scenario/map mismatch")
        grid = tuple(s["map"])
        if s["maxSteps"] != config.horizon or s["targetOrders"] != config.target_orders:
            raise ValueError("Snapshot/config limits differ")
        integer = lambda x, lo, hi: type(x) is int and lo <= x <= hi
        if not integer(s["turn"], 0, config.horizon) or not integer(s["orders"], 0, config.target_orders):
            raise ValueError("Invalid snapshot progress")
        if [a["id"] for a in s["actors"]] != list(ACTOR_IDS): raise ValueError("Invalid actor order")
        active_ids = []
        if s.get("preset") not in ("supply", "cook"): raise ValueError("Invalid role preset")
        sides = ("left", "right") if s["preset"] == "supply" else ("right", "left")
        for actor, side in zip(s["actors"], sides):
            p = actor["position"]
            if len(p) != 2 or not all(type(x) is int for x in p) or not _walkable(grid, p): raise ValueError("Invalid position")
            if actor["side"] != side or (p[1] < 4) != (side == "left") or actor["facing"] not in DIRECTIONS or actor["holding"] not in ITEMS:
                raise ValueError("Invalid actor state")
            item = actor["_held_id"]
            if bool(item) != bool(actor["holding"]): raise ValueError("Held item provenance missing")
            if item:
                if s["_items"][item]["kind"] != actor["holding"] or s["_items"][item]["location"] != f"actor:{actor['id']}":
                    raise ValueError("Held item provenance mismatch")
                active_ids.append(item)
        if set(s["counters"]) != set(COUNTER_KEYS) or set(s["_counter_item_ids"]) != set(COUNTER_KEYS):
            raise ValueError("Invalid counters")
        for k in COUNTER_KEYS:
            kind, item = s["counters"][k], s["_counter_item_ids"][k]
            if kind not in ITEMS or bool(kind) != bool(item): raise ValueError("Counter provenance missing")
            if item:
                if s["_items"][item]["kind"] != kind or s["_items"][item]["location"] != f"counter:{k}":
                    raise ValueError("Counter provenance mismatch")
                active_ids.append(item)
        if len(active_ids) != len(set(active_ids)): raise ValueError("Duplicated item")
        p = s["pot"]
        if not integer(p["ingredients"], 0, 3) or not integer(p["remaining"], 0, 4) or type(p["ready"]) is not bool:
            raise ValueError("Invalid pot")
        if (p["ingredients"] < 3 and (p["remaining"] or p["ready"])) or (p["ingredients"] == 3 and (p["remaining"] == 0) != p["ready"]):
            raise ValueError("Inconsistent cooking state")
        batch = s["_pot_batch"]
        if bool(batch) != bool(p["ingredients"]): raise ValueError("Missing pot batch")
        if batch and len(s["_batches"][batch]["ingredient_ids"]) != p["ingredients"]: raise ValueError("Batch ingredient mismatch")
        expected_reason = "success" if s["orders"] == config.target_orders else "timeout" if s["turn"] == config.horizon else None
        if s["reason"] != expected_reason or s["done"] is not (expected_reason is not None): raise ValueError("Invalid terminal state")
        self.config, self.seed, self.scenario_id, self.grid = config, s["seed"], scenario_id, grid
        self.stations, self._geometry, self._state = _stations(grid), _geometry_table(grid), s

    def fork(self):
        other = object.__new__(CooperativeKitchen)
        other.config, other.seed, other.scenario_id = self.config, self.seed, self.scenario_id
        other.grid, other.stations, other._geometry = self.grid, self.stations, self._geometry
        other._state = self.snapshot()
        return other

    def public_view(self):
        result = {k: copy.deepcopy(v) for k, v in self._state.items() if not k.startswith("_")}
        for actor in result["actors"]:
            for key in list(actor):
                if key.startswith("_"): del actor[key]
        result["score"] = 100 * result["orders"] - result["turn"]
        result["first_delivery_turn"] = self._state["_first_serve_turn"]
        result["events"] = copy.deepcopy(self._state["_last_events"])
        # Item and batch identifiers belong in research logs, not participant APIs.
        for event in result["events"]:
            for key in ("item_id", "batch_id", "plate_id"):
                event.pop(key, None)
        return result

    def observations(self):
        s = self._state
        pot = s["pot"]
        output = {}
        right = next(actor for actor in s["actors"] if actor["side"] == "right")
        supplied = pot["ingredients"] + sum(x == "onion" for x in s["counters"].values()) + (right["holding"] == "onion")
        for i, actor in enumerate(s["actors"]):
            partner = s["actors"][1-i]
            values = [actor["position"][0]/6, actor["position"][1]/8, partner["position"][0]/6, partner["position"][1]/8,
                      float(actor["side"] == "left"), float(actor["side"] == "right")]
            for a in (actor, partner): values += [float(a["facing"] == direction) for direction in DIRECTIONS]
            for item in (actor["holding"], partner["holding"], s["counters"]["2,4"], s["counters"]["4,4"]):
                values += [float(item == kind) for kind in ITEMS]
            values += [pot["ingredients"]/3, pot["remaining"]/4, float(pot["ready"]), s["orders"]/s["targetOrders"], 1-s["turn"]/s["maxSteps"]]
            values += self._geometry[(tuple(actor["position"]), actor["facing"])]
            values += [supplied/5, max(0, 3-supplied)/3, sum(x is None for x in s["counters"].values())/2, float(s["done"])]
            output[actor["id"]] = np.asarray(values, dtype=np.float32)
        return output

    def global_state(self, observations=None):
        observations = self.observations() if observations is None else observations
        return np.concatenate([observations[id] for id in ACTOR_IDS])

    def _potential(self):
        s = self._state
        result = s["pot"]["ingredients"] * .18
        if s["pot"]["ingredients"] == 3: result += .20 * (4-s["pot"]["remaining"])/4
        if s["pot"]["ready"]: result += .10
        for actor in s["actors"]:
            if actor["holding"] == "onion": result += .03 if actor["side"] == "left" else .10
            if actor["holding"] == "soup": result += 1.10 if actor["side"] == "left" else .95
        result += sum(.08 if x == "onion" else 1.0 if x == "soup" else 0 for x in s["counters"].values())
        return min(result, 4.0)

    def _intent(self, actor):
        s = self._state
        target = _add(actor["position"], DIRECTIONS[actor["facing"]])
        cell, held = _tile(self.grid, target), actor["holding"]
        result = dict(actor=actor["id"], target=list(target), item=held, resource=_key(target))
        def finish(kind, **extra): return dict(result, type=kind, **extra)
        if cell == "C":
            item = s["counters"][_key(target)]
            if not held and item: return dict(result, type="pickup", item=item)
            if held and not item: return finish("drop")
            return finish("invalid_interaction", reason="counter_occupied" if held else "counter_empty")
        if cell in ("I", "D"):
            if not held: return dict(result, type="take_source", item="onion" if cell == "I" else "plate", resource=None)
            return finish("invalid_interaction", reason="hands_full")
        if cell == "X" and held: return dict(result, type="discard", resource=None)
        if cell == "S" and held == "soup": return dict(result, type="serve", resource="orders")
        if cell == "P":
            p = s["pot"]
            if held == "onion" and p["ingredients"] < 3: return finish("load")
            if held == "plate" and p["ready"]: return finish("plate")
            return finish("invalid_interaction", reason="pot_cooking" if p["remaining"] else "plate_needed" if p["ready"] else "pot_full" if p["ingredients"] == 3 else "onion_needed")
        return finish("invalid_interaction", reason="soup_needed" if cell == "S" else "hands_empty" if cell == "X" else "no_station")

    def step(self, actions, *, include_state=True):
        if set(actions) != set(ACTOR_IDS) or any(a not in ACTIONS for a in actions.values()):
            raise ValueError("Provide exactly human and ai actions from ACTIONS")
        s = self._state
        if s["done"]:
            return dict(state=self.public_view() if include_state else None, events=[], rewards={id: 0.0 for id in ACTOR_IDS}, done=True,
                        proposed_actions=dict(actions), actual_actions={id: "WAIT" for id in ACTOR_IDS})
        old_potential, old_remaining, old_orders = self._potential(), s["pot"]["remaining"], s["orders"]
        actors = {actor["id"]: actor for actor in s["actors"]}
        priority = ACTOR_IDS if s["turn"] % 2 == 0 else tuple(reversed(ACTOR_IDS))
        # Freeze intents before moving actors or changing inventory.
        intents = {id: self._intent(actors[id]) for id in ACTOR_IDS if actions[id] == "INTERACT"}
        origins = {id: tuple(actors[id]["position"]) for id in ACTOR_IDS}
        destinations = {}
        for id in ACTOR_IDS:
            q = _add(origins[id], DIRECTIONS[actions[id]]) if actions[id] in DIRECTIONS else origins[id]
            destinations[id] = q if _walkable(self.grid, q) else origins[id]
        if destinations["human"] == destinations["ai"]:
            stationary = next((id for id in ACTOR_IDS if origins[id] == destinations[id]), None)
            loser = next(id for id in ACTOR_IDS if id != stationary) if stationary else priority[1]
            destinations[loser] = origins[loser]
        if destinations["human"] == origins["ai"] and destinations["ai"] == origins["human"]:
            destinations = dict(origins)
        events = []
        for id in ACTOR_IDS:
            action, actor = actions[id], actors[id]
            if action in DIRECTIONS:
                old_facing = actor["facing"]
                actor["position"], actor["facing"] = list(destinations[id]), action
                moved = destinations[id] != origins[id]
                event_type = "move" if moved else "turn_in_place" if old_facing != action else "blocked"
                events.append(dict(type=event_type, actor=id, action=action, **{"from": list(origins[id]), "to": list(destinations[id])}, facing_before=old_facing))
            elif action == "WAIT": events.append(dict(type="wait", actor=id))
        claimed = set()
        for id in priority:
            intent = intents.get(id)
            if intent is None: continue
            actor = actors[id]
            kind, resource = intent["type"], intent["resource"]
            if kind == "invalid_interaction": events.append(intent); continue
            if resource and resource in claimed:
                events.append(dict(type="conflict", actor=id, target=intent["target"])); continue
            if resource: claimed.add(resource)
            item_id = actor["_held_id"]
            k = _key(intent["target"])
            if kind == "pickup":
                item_id = s["_counter_item_ids"][k]
                actor["holding"], actor["_held_id"] = intent["item"], item_id
                s["counters"][k], s["_counter_item_ids"][k] = None, None
                s["_items"][item_id]["location"] = f"actor:{id}"
            elif kind == "drop":
                s["counters"][k], s["_counter_item_ids"][k] = actor["holding"], item_id
                s["_items"][item_id]["location"] = f"counter:{k}"
                actor["holding"], actor["_held_id"] = None, None
            elif kind == "take_source":
                item_id = self._new_item(intent["item"], f"actor:{id}")
                actor["holding"], actor["_held_id"] = intent["item"], item_id
            elif kind in ("discard", "serve", "load"):
                actor["holding"], actor["_held_id"] = None, None
                if kind == "discard": s["_items"][item_id]["location"] = "discarded"
                elif kind == "serve":
                    s["orders"] += 1
                    s["_items"][item_id]["location"] = "served"
                    batch = s["_items"][item_id]["batch_id"]
                    s["_batches"][batch]["served_turn"] = s["turn"] + 1
                    intent["batch_id"] = batch
                    if s["_first_serve_turn"] is None: s["_first_serve_turn"] = s["turn"] + 1
                else:
                    batch = s["_pot_batch"] or self._new_batch()
                    s["_batches"][batch]["ingredient_ids"].append(item_id)
                    s["_items"][item_id]["location"] = f"pot:{batch}"
                    s["pot"]["ingredients"] += 1
                    intent["batch_id"] = batch
                    if s["pot"]["ingredients"] == 3: s["pot"]["remaining"] = 4
            elif kind == "plate":
                batch = s["_pot_batch"]
                s["_items"][item_id]["location"] = f"consumed:{batch}"
                for onion_id in s["_batches"][batch]["ingredient_ids"]:
                    s["_items"][onion_id]["location"] = f"consumed:{batch}"
                intent["plate_id"], intent["batch_id"] = item_id, batch
                item_id = self._new_item("soup", f"actor:{id}", batch_id=batch, plate_id=item_id)
                actor["holding"], actor["_held_id"] = "soup", item_id
                s["pot"], s["_pot_batch"] = dict(ingredients=0, remaining=0, ready=False), None
            intent["item_id"] = item_id
            events.append(intent)
            if kind == "load" and s["pot"]["ingredients"] == 3: events.append(dict(type="cooking_started", remaining=4))
        if old_remaining > 0 and s["pot"]["ingredients"] == 3:
            s["pot"]["remaining"] = old_remaining - 1
            if s["pot"]["remaining"] == 0:
                s["pot"]["ready"] = True
                events.append(dict(type="soup_ready"))
        s["turn"] += 1
        if s["orders"] >= s["targetOrders"]:
            s["orders"], s["done"], s["reason"] = s["targetOrders"], True, "success"
            events.append(dict(type="success"))
        elif s["turn"] >= s["maxSteps"]:
            s["done"], s["reason"] = True, "timeout"
            events.append(dict(type="timeout"))
        s["_last_events"] = events
        next_potential = 0 if s["done"] else self._potential()
        reward = (s["orders"]-old_orders)*self.config.serve_reward - self.config.time_cost
        reward += self.config.shaping_scale * (self.config.discount*next_potential-old_potential)
        return dict(state=self.public_view() if include_state else None, events=copy.deepcopy(events) if include_state else events, rewards={id: float(reward) for id in ACTOR_IDS},
                    done=s["done"], proposed_actions=dict(actions), actual_actions=dict(actions))


def program_decision(env: CooperativeKitchen, actor_id: str, profile="efficient", rng=None):
    if actor_id not in ACTOR_IDS: raise ValueError("Unknown actor")
    if profile not in ("efficient", "upper", "lower", "perturbed"): raise ValueError("Unknown program profile")
    s, grid = env.state, env.grid
    actor = s["actors"][ACTOR_IDS.index(actor_id)]
    right = next(a for a in s["actors"] if a["side"] == "right")
    pot = s["pot"]
    staged = sum(x == "onion" for x in s["counters"].values())
    right_held = int(right["holding"] == "onion")
    needed = max(0, 3-pot["ingredients"]-staged-right_held)
    facts = dict(side=actor["side"], holding=actor["holding"], orders=s["orders"], pot=dict(pot),
                 counterItems=dict(s["counters"]), stagedOnions=staged, rightHeldOnions=right_held,
                 neededOnions=needed, waitingFor=None)
    def result(action, rule, target=None, **extra):
        return dict(action=action, rule=rule, target=list(target) if target else None, facts=dict(facts, **extra))
    def go(rule, target, **extra):
        return result(_route(grid, tuple(actor["position"]), actor["facing"], tuple(target))[0], rule, target, **extra)
    def near(rule, targets):
        if not targets: return result("WAIT", "wait_space", waitingFor="space")
        target = min(targets, key=lambda p: _route(grid, tuple(actor["position"]), actor["facing"], p)[1])
        return go(rule, target)
    def wait(rule, target, condition):
        res = go(rule, target, waitingFor=condition)
        if res["action"] == "INTERACT": res["action"] = "WAIT"
        return res
    def matching(item): return [_pos(k) for k in COUNTER_KEYS if s["counters"][k] == item]
    empty, onions, plates, soups = [matching(item) for item in ITEMS]
    stations = env.stations
    input_row = 2 if profile == "upper" else 4 if profile == "lower" else 2 if stations["ingredient"][0] < 3 else 4
    input_counter, output_counter = (input_row, 4), (6-input_row, 4)
    def preferred(options, target): return target if target in options else options[0]
    trash = stations["left_trash"] if actor["side"] == "left" else stations["right_trash"]
    holding = actor["holding"]
    def choose():
        if s["done"]: return result("WAIT", "finished")
        if actor["side"] == "left":
            if holding == "soup": return go("serve_soup", stations["serve"])
            if holding == "plate": return go("discard_plate", trash)
            if holding == "onion":
                if right["holding"] == "soup": return go("discard_for_handoff", trash)
                if needed == 0: return go("discard_extra_onion", trash)
                if empty: return go("handoff_onion", preferred(empty, input_counter))
                return result("WAIT", "wait_space", waitingFor="space")
            if soups: return near("collect_soup", soups)
            if plates: return near("clear_plate", plates)
            if right["holding"] == "soup":
                if not empty and onions: return near("clear_for_handoff", onions)
                return wait("wait_soup_handoff", output_counter, "soup_handoff")
            if pot["ingredients"] == 3 and onions: return near("clear_extra_onion", onions)
            if needed > 0:
                if empty: return go("get_onion", stations["ingredient"])
                return wait("wait_pickup", output_counter, "counter_pickup")
            return wait("wait_soup", output_counter, "soup_handoff" if pot["ready"] else "cooking" if pot["remaining"] else "pot_loading")
        if holding == "soup":
            if empty: return go("handoff_soup", preferred(empty, output_counter))
            return result("WAIT", "wait_space", waitingFor="space")
        if holding == "onion": return go("load_pot", stations["pot"]) if pot["ingredients"] < 3 else go("discard_extra_onion", trash)
        if holding == "plate":
            if pot["ready"]: return go("plate_soup", stations["pot"])
            if pot["remaining"]: return wait("wait_cooking", stations["pot"], "cooking")
            return go("discard_plate", trash)
        if pot["ready"]:
            if not empty and onions: return near("clear_extra_onion", onions)
            if plates: return near("get_counter_plate", plates)
            if not empty: return result("WAIT", "wait_space", waitingFor="space")
            return go("get_plate", stations["plate"])
        if pot["remaining"]:
            if not empty and onions: return near("clear_extra_onion", onions)
            return wait("wait_cooking", stations["plate"], "cooking")
        if onions: return near("collect_onion", onions)
        if plates: return near("clear_plate", plates)
        return wait("wait_onion", input_counter, "onion")
    decision = choose()
    if profile == "perturbed" and not s["done"]:
        if rng is None: raise ValueError("Perturbed program requires an explicit caller-owned RNG")
        draw = float(rng.random())
        if draw < .10:
            return result("WAIT", "perturbed_wait", intendedAction=decision["action"])
        if draw < .15:
            action = ACTIONS[int(rng.integers(len(ACTIONS)))] if hasattr(rng, "integers") else rng.choice(ACTIONS)
            return result(action, "perturbed_action", intendedAction=decision["action"])
    return decision
