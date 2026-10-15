"""Observable Kitchen features and dependency-light frozen Actor inference.

Training lives in ``training.py``. This module contains no teacher or routing
policy: the only action selector is a masked neural-network forward pass.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from . import engine_v5_frozen as e

MODEL_VERSION = "kitchen-dual-nn-v1"
FEATURE_VERSION = "kitchen-visible-features-v1"
HUMAN_ACTIONS = ("up", "down", "left", "right", "wait", "interact")
AI_ACTIONS = HUMAN_ACTIONS + ("interact_slot0", "interact_slot1") + tuple(
    f"interact_order{i}" for i in range(1, 6)
)
STAGES = ("raw", "prepared", "cooked_protein", "cooked_vegetable", "mixing", "finished", "plated", "spoiled", "waste")
PHASES = ("idle", "protein", "await_protein_store", "await_vegetable", "vegetable", "mix")
STATUSES = ("empty", "cooking", "ready", "burnt")
FACES = ("up", "down", "left", "right")
CONTAINERS = ("temporary_plate", "output_container", "serving_plate", "disposal_container")


def _onehot(value, choices):
    return [float(value == choice) for choice in choices]


def _item(item, turn):
    if not item:
        return [0.0] * 33
    values = [1.0]
    values += _onehot(item.get("ingredient"), tuple(e.LABELS))
    values += _onehot(item.get("recipe"), tuple(e.RECIPES))
    values += _onehot(item.get("stage"), STAGES)
    values += _onehot(item.get("order_id"), tuple(f"order{i}" for i in range(1, 6)))
    values += _onehot(item.get("pot_id"), ("pot1", "pot2"))
    values += _onehot(item.get("container"), CONTAINERS)
    values += [float(item.get("prepare_progress", 0)) / 5.0,
               max(-1.0, min(1.0, (item.get("expires_turn", turn + 120) - turn) / 120.0)),
               min(1.0, max(0.0, turn - item.get("storage_since_turn", turn)) / 10.0),
               float(item.get("was_buffered", False)),
               len(item.get("components", ())) / 2.0,
               min(1.0, max(0.0, turn - item.get("acquired_turn", turn)) / 120.0)]
    assert len(values) == 33
    return values


def features(state):
    """Fixed-length full visible state; excludes seed, group and old policy memory."""
    turn = state["turn"]
    row = [turn / 360.0, state["max_turns"] / 360.0, state["raw_score"] / 500.0]
    row += _onehot(state["task"], (1, 2, 3))
    for order_id in range(1, 6):
        order = next((o for o in state["orders"] if o["id"] == f"order{order_id}"), None)
        row += [float(order is not None)]
        row += _onehot(order["recipe"] if order else None, tuple(e.RECIPES))
        row += _onehot(order["status"] if order else None, ("pending", "completed", "expired"))
        row += [max(0.0, (order["deadline"] - turn) / 360.0) if order else 0.0]
    for role in ("human", "ai"):
        actor = state[role]
        row += [actor["x"] / 8.0, actor["y"] / 6.0]
        row += _onehot(actor["facing"], FACES)
    items = [state["human"]["holding"], state["ai"]["holding"], state["handoff"],
             state["buffers"]["human"], *state["buffers"]["ai_raw"],
             state["buffers"]["protein"]["pot1"], state["buffers"]["protein"]["pot2"],
             state["pots"][0]["item"], state["pots"][1]["item"]]
    for item in items:
        row += _item(item, turn)
    for pot in state["pots"]:
        row += _onehot(pot["status"], STATUSES)
        row += _onehot(pot["phase"], PHASES)
        row += _onehot(pot["order_id"], tuple(f"order{i}" for i in range(1, 6)))
        row += [pot["remaining"] / 10.0, pot["ready_age"] / 8.0]
    return np.asarray(row, dtype=np.float32)


def action_spec(action_id):
    if action_id in HUMAN_ACTIONS:
        return {"action": action_id}
    if action_id.startswith("interact_slot"):
        return {"action": "interact", "slot": int(action_id[-1])}
    if action_id.startswith("interact_order"):
        return {"action": "interact", "order_id": "order" + action_id[-1]}
    raise ValueError("Unknown Kitchen NN action")


def legal_mask(state, role):
    actions = HUMAN_ACTIONS if role == "human" else AI_ACTIONS
    if state["terminal"]:
        return np.zeros(len(actions), dtype=bool)
    if role == "human":
        legal = set(e.legal_actions(state, "human"))
        return np.asarray([action in legal for action in actions], dtype=bool)
    station = e._front(state["ai"])
    is_raw = station is not None and station["id"] == "ai_raw"
    is_load = (station is not None and station["kind"] == "pot" and state["ai"]["holding"] is not None
               and state["ai"]["holding"]["stage"] == "prepared")
    mask = []
    for action_id in actions:
        if action_id in e.MOVES or action_id == "wait":
            mask.append(True)
        elif action_id == "interact":
            mask.append(not (is_raw or is_load) and e._descriptor(state, "ai", explicit=True) is not None)
        elif action_id.startswith("interact_slot"):
            slot = action_spec(action_id)["slot"]
            mask.append(is_raw and e._descriptor(state, "ai", slot=slot, explicit=True) is not None)
        else:
            order_id = action_spec(action_id)["order_id"]
            mask.append(is_load and e._descriptor(state, "ai", order_id=order_id, explicit=True) is not None)
    return np.asarray(mask, dtype=bool)


def program_features(state, role):
    """Named, pre-action observables for a role-specific bounded program.

    These are never used to select an action at execution.  They are a small,
    auditable subset of the same visible state supplied to the neural Actor.
    """
    if role not in ("human", "ai"):
        raise ValueError("Unknown Kitchen role")
    actor, other = state[role], state["ai" if role == "human" else "human"]
    turn = state["turn"]
    front = e._front(actor)
    held = actor["holding"]
    result = {
        "turn_fraction": turn / max(1, state["max_turns"]),
        "team_score_fraction": state["raw_score"] / 500.0,
        "actor_x": actor["x"] / 8.0,
        "actor_y": actor["y"] / 6.0,
        "partner_x": other["x"] / 8.0,
        "partner_y": other["y"] / 6.0,
        "partner_distance": (abs(actor["x"] - other["x"]) + abs(actor["y"] - other["y"])) / 14.0,
        "actor_holds_item": float(held is not None),
        "partner_holds_item": float(other["holding"] is not None),
        "handoff_occupied": float(state["handoff"] is not None),
        "human_buffer_occupied": float(state["buffers"]["human"] is not None),
        "ai_raw_slot0_occupied": float(state["buffers"]["ai_raw"][0] is not None),
        "ai_raw_slot1_occupied": float(state["buffers"]["ai_raw"][1] is not None),
        "held_freshness_fraction": max(-1.0, min(1.0, (held.get("expires_turn", turn + 120) - turn) / 120.0)) if held else 0.0,
        "held_prepare_progress": held.get("prepare_progress", 0) / 5.0 if held else 0.0,
    }
    for task in (1, 2, 3):
        result[f"task_{task}"] = float(state["task"] == task)
    for face in FACES:
        result[f"actor_facing_{face}"] = float(actor["facing"] == face)
        result[f"partner_facing_{face}"] = float(other["facing"] == face)
    # Exact observed floor positions let a small decision tree describe the
    # neural Actor's local navigation without supplying any route or action.
    for x, y in sorted(e.FLOOR):
        result[f"actor_at_{x}_{y}"] = float((actor["x"], actor["y"]) == (x, y))
        result[f"partner_at_{x}_{y}"] = float((other["x"], other["y"]) == (x, y))
    for station in ("handoff", "ai_raw", "output", "serving", "disposal", "prep", "pot"):
        station_id = front["id"] if front else ""
        result[f"front_station_{station}"] = float(station in station_id)
    for station_id in ("egg", "tomato", "meat", "pepper", "prep", "human_buffer",
                       "plate", "serve", "handoff", "trash", "pot1", "pot2",
                       "protein1", "protein2", "ai_raw"):
        result[f"front_exact_{station_id}"] = float(bool(front) and front["id"] == station_id)
    for stage in STAGES:
        result[f"held_stage_{stage}"] = float(bool(held) and held.get("stage") == stage)
        result[f"partner_held_stage_{stage}"] = float(bool(other["holding"]) and other["holding"].get("stage") == stage)
    for ingredient in e.LABELS:
        result[f"held_ingredient_{ingredient}"] = float(bool(held) and held.get("ingredient") == ingredient)
    for index, pot in enumerate(state["pots"], start=1):
        result[f"pot{index}_remaining_fraction"] = pot["remaining"] / 10.0
        result[f"pot{index}_ready_age_fraction"] = pot["ready_age"] / 8.0
        for recipe in e.RECIPES:
            result[f"pot{index}_recipe_{recipe}"] = float(pot["recipe"] == recipe)
        for order_index in range(1, 6):
            result[f"pot{index}_order{order_index}"] = float(pot["order_id"] == f"order{order_index}")
        for status in STATUSES:
            result[f"pot{index}_status_{status}"] = float(pot["status"] == status)
        for phase in PHASES:
            result[f"pot{index}_phase_{phase}"] = float(pot["phase"] == phase)
    # Keep the program input as observed pre-action state.  In particular,
    # current buffer contents matter to the neural Actor's navigation, while
    # a mere occupied flag loses their recipe and order identity.
    observed_items = {
        "actor_item": held,
        "partner_item": other["holding"],
        "handoff_item": state["handoff"],
        "human_buffer_item": state["buffers"]["human"],
        "ai_raw_slot0_item": state["buffers"]["ai_raw"][0],
        "ai_raw_slot1_item": state["buffers"]["ai_raw"][1],
        "protein1_item": state["buffers"]["protein"]["pot1"],
        "protein2_item": state["buffers"]["protein"]["pot2"],
        "pot1_item": state["pots"][0]["item"],
        "pot2_item": state["pots"][1]["item"],
    }
    for location, item in observed_items.items():
        result[f"{location}_present"] = float(item is not None)
        for stage in STAGES:
            result[f"{location}_stage_{stage}"] = float(bool(item) and item.get("stage") == stage)
        for ingredient in e.LABELS:
            result[f"{location}_ingredient_{ingredient}"] = float(bool(item) and item.get("ingredient") == ingredient)
        for recipe in e.RECIPES:
            result[f"{location}_recipe_{recipe}"] = float(bool(item) and item.get("recipe") == recipe)
        for order_index in range(1, 6):
            result[f"{location}_order{order_index}"] = float(bool(item) and item.get("order_id") == f"order{order_index}")
    for index in range(1, 6):
        order = next((row for row in state["orders"] if row["id"] == f"order{index}"), None)
        result[f"order{index}_present"] = float(order is not None)
        result[f"order{index}_deadline_fraction"] = max(0.0, (order["deadline"] - turn) / 360.0) if order else 0.0
        for status in ("pending", "completed", "expired"):
            result[f"order{index}_status_{status}"] = float(bool(order) and order["status"] == status)
        for recipe in e.RECIPES:
            result[f"order{index}_recipe_{recipe}"] = float(bool(order) and order["recipe"] == recipe)
    actions = HUMAN_ACTIONS if role == "human" else AI_ACTIONS
    for action, allowed in zip(actions, legal_mask(state, role)):
        result[f"legal_{action}"] = float(allowed)
    return result


def role_weight_sha256(weights, role):
    """Stable digest of one Actor independent of NPZ container metadata."""
    if role not in ("human", "ai"):
        raise ValueError("Unknown Kitchen role")
    digest = sha256()
    for layer in range(3):
        for kind in ("weight", "bias"):
            name = f"{role}.{kind}{layer}"
            tensor = np.ascontiguousarray(weights[name], dtype="<f4")
            digest.update(name.encode("ascii") + b"\0")
            digest.update(json.dumps(tensor.shape).encode("ascii") + b"\0")
            digest.update(tensor.tobytes())
    return digest.hexdigest()


class FrozenActors:
    def __init__(self, path, expected_sha256=None):
        path = Path(path)
        payload = path.read_bytes()
        self.sha256 = sha256(payload).hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise RuntimeError("Kitchen frozen Actor checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            self.role_sha256 = {
                role: role_weight_sha256(archive, role) for role in ("human", "ai")
            }
            # Float64 accumulation avoids Accelerate float32 matmul warnings
            # on this deployment platform while preserving Actor argmax.
            self.weights = {key: archive[key].astype(np.float64).copy() for key in archive.files if key != "metadata_json"}
        for tensor in self.weights.values():
            tensor.setflags(write=False)
        self.metadata = metadata
        if (metadata.get("model_version") != MODEL_VERSION or metadata.get("feature_version") != FEATURE_VERSION
                or metadata.get("observation_dim") != len(features(e.initial_state(1000, 1)))
                or tuple(metadata.get("human_actions", ())) != HUMAN_ACTIONS
                or tuple(metadata.get("ai_actions", ())) != AI_ACTIONS):
            raise RuntimeError("Incompatible Kitchen frozen Actor")
        hidden, observation = metadata.get("hidden"), metadata["observation_dim"]
        if type(hidden) is not int or hidden < 1:
            raise RuntimeError("Invalid Kitchen frozen Actor width")
        for role, count in (("human", len(HUMAN_ACTIONS)), ("ai", len(AI_ACTIONS))):
            dimensions = (observation, hidden, hidden, count)
            for layer in range(3):
                weight = self.weights.get(f"{role}.weight{layer}")
                bias = self.weights.get(f"{role}.bias{layer}")
                if (weight is None or bias is None or weight.shape != (dimensions[layer + 1], dimensions[layer])
                        or bias.shape != (dimensions[layer + 1],) or not np.isfinite(weight).all()
                        or not np.isfinite(bias).all()):
                    raise RuntimeError("Malformed Kitchen frozen Actor weights")

    def logits(self, state, role):
        vector = features(state).astype(np.float64)
        for layer in range(3):
            vector = (np.sum(self.weights[f"{role}.weight{layer}"] * vector[None, :], axis=1,
                             dtype=np.float64) + self.weights[f"{role}.bias{layer}"])
            if layer < 2:
                vector = np.tanh(vector)
        return vector

    def choose(self, state, role):
        mask = legal_mask(state, role)
        if not mask.any():
            raise ValueError("No legal action in a terminal Kitchen state")
        scores = self.logits(state, role)
        if not np.isfinite(scores).all():
            raise RuntimeError("Kitchen frozen Actor produced non-finite scores")
        index = int(np.argmax(np.where(mask, scores, -np.inf)))
        actions = HUMAN_ACTIONS if role == "human" else AI_ACTIONS
        return actions[index], scores, mask


@lru_cache(maxsize=2)
def load_frozen(path, expected_sha256):
    return FrozenActors(path, expected_sha256)
