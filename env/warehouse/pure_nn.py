"""Raw-state Warehouse Actor and dependency-light frozen NumPy inference.

No goal assignment, route heuristic, coordination plan, or fixed action bonus
enters this policy.  The only action filter is the environment's static mask.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from core.policy_contracts import ActionDistribution
from .decision_protocol import independent_agent_seed
from .layouts import STUDY_MAP_LAYOUT
from .navigation import ACTIONS, legal_action_mask

MODEL_VERSION = "warehouse-pure-nn.v1"
DUAL_MODEL_VERSION = "warehouse-dual-pure-nn.v5"
OBSERVATION_VERSION = "warehouse-public-raw.v1"
AGENTS = ("robot_1", "robot_2")
_ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
_WALLS = np.asarray(
    [float(tile == "#") for row in STUDY_MAP_LAYOUT.tiles for tile in row],
    dtype=np.float32,
)


def _one_hot(index: int, size: int) -> list[float]:
    return [float(index == position) for position in range(size)]


def observation(state, agent_id: str) -> np.ndarray:
    """Public pre-action state in ego/partner order; no inferred intentions."""
    if agent_id not in AGENTS:
        raise ValueError("unknown Warehouse agent")
    agents = [state.by_id(agent_id), state.by_id(AGENTS[1 - AGENTS.index(agent_id)])]
    tasks = sorted(state.tasks, key=lambda task: task.task_id)
    if len(tasks) != 2:
        raise ValueError("pure NN expects two active shared tasks")
    rows, cols = STUDY_MAP_LAYOUT.rows, STUDY_MAP_LAYOUT.cols
    features: list[float] = [
        float(state.frame) / 120.0,
        STUDY_MAP_LAYOUT.charger_position[0] / max(1, rows - 1),
        STUDY_MAP_LAYOUT.charger_position[1] / max(1, cols - 1),
        *_WALLS.tolist(),
    ]
    for agent in agents:
        carried = next((index for index, task in enumerate(tasks)
                        if task.task_id == agent.carrying_task_id), 2)
        features.extend((
            agent.position[0] / max(1, rows - 1),
            agent.position[1] / max(1, cols - 1),
            float(agent.battery) / 100.0,
            float(agent.active),
            *_one_hot(carried, 3),
            *_one_hot(_ACTION_INDEX.get(agent.heading, len(ACTIONS) - 1), len(ACTIONS)),
            *_one_hot(_ACTION_INDEX.get(agent.last_executed_action, len(ACTIONS) - 1), len(ACTIONS)),
        ))
    for task in tasks:
        carrier = (0 if task.carrier_agent_id == agents[0].agent_id else
                   1 if task.carrier_agent_id == agents[1].agent_id else 2)
        features.extend((
            task.pickup_position[0] / max(1, rows - 1),
            task.pickup_position[1] / max(1, cols - 1),
            task.delivery_position[0] / max(1, rows - 1),
            task.delivery_position[1] / max(1, cols - 1),
            float(task.status == "available"),
            float(task.status == "carried"),
            *_one_hot(carrier, 3),
        ))
    return np.asarray(features, dtype=np.float32)


def observations(state) -> np.ndarray:
    return np.stack([observation(state, agent_id) for agent_id in AGENTS])


def action_masks(state, layout_id: str = STUDY_MAP_LAYOUT.layout_id) -> np.ndarray:
    return np.asarray([
        legal_action_mask(state, state.by_id(agent_id), layout_id)
        for agent_id in AGENTS
    ], dtype=np.float32)


OBSERVATION_DIM = 3 + len(_WALLS) + 2 * (4 + 3 + 2 * len(ACTIONS)) + 2 * 9


def observation_feature_names() -> tuple[str, ...]:
    """Names for public, pre-action input coordinates used by RCPD."""
    names = ["elapsed_fraction", "charger_row", "charger_column"]
    names.extend(f"wall_row_{r}_column_{c}" for r in range(STUDY_MAP_LAYOUT.rows)
                 for c in range(STUDY_MAP_LAYOUT.cols))
    for role in ("self", "partner"):
        names.extend((f"{role}_row", f"{role}_column", f"{role}_battery_fraction",
                      f"{role}_active"))
        names.extend(f"{role}_carrying_task_{i}" for i in (0, 1, 2))
        names.extend(f"{role}_heading_{action.lower()}" for action in ACTIONS)
        names.extend(f"{role}_previous_action_{action.lower()}" for action in ACTIONS)
    for index in (0, 1):
        prefix = f"order_{index}"
        names.extend((f"{prefix}_pickup_row", f"{prefix}_pickup_column",
                      f"{prefix}_delivery_row", f"{prefix}_delivery_column",
                      f"{prefix}_available", f"{prefix}_carried"))
        names.extend(f"{prefix}_carrier_{role}" for role in ("self", "partner", "none"))
    if len(names) != OBSERVATION_DIM:
        raise RuntimeError("Warehouse semantic feature schema does not match the observation")
    return tuple(names)


def role_weight_sha256(weights: dict[str, np.ndarray]) -> str:
    """Stable digest of one role's six Actor tensors, independent of NPZ compression."""
    digest = sha256()
    for key in ("w0", "b0", "w1", "b1", "w2", "b2"):
        tensor = np.ascontiguousarray(weights[key], dtype="<f4")
        digest.update(key.encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


class FrozenPureNN:
    """Exactly the learned three-layer Actor, evaluated without PyTorch."""

    def __init__(self, weights: dict[str, np.ndarray], artifact_sha256: str):
        self.weights = weights
        self.artifact_sha256 = artifact_sha256

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None):
        try:
            raw = Path(path).read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Warehouse pure NN artifact missing: {path}") from exc
        digest = sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError("Warehouse pure NN artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("model_version") != MODEL_VERSION or
                metadata.get("observation_version") != OBSERVATION_VERSION or
                metadata.get("observation_dim") != OBSERVATION_DIM or
                metadata.get("map_layout_id") != STUDY_MAP_LAYOUT.layout_id or
                metadata.get("action_names") != list(ACTIONS)):
                raise RuntimeError("Warehouse pure NN contract mismatch")
            weights = {key: np.asarray(archive[key], dtype=np.float32)
                       for key in ("w0", "b0", "w1", "b1", "w2", "b2")}
        if (weights["w0"].shape[1] != OBSERVATION_DIM or
            weights["w2"].shape[0] != len(ACTIONS) or
            weights["w0"].shape[0] != weights["b0"].size or
            weights["w1"].shape[1] != weights["b0"].size or
            weights["w1"].shape[0] != weights["b1"].size or
            weights["w2"].shape[1] != weights["b1"].size or
            weights["b2"].size != len(ACTIONS)):
            raise RuntimeError("Warehouse pure NN tensor shape mismatch")
        return cls(weights, digest)

    def logits(self, features: np.ndarray) -> np.ndarray:
        data = np.asarray(features, dtype=np.float32)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for index in (0, 1):
                data = np.maximum(data @ self.weights[f"w{index}"].T + self.weights[f"b{index}"], 0.0)
            result = data @ self.weights["w2"].T + self.weights["b2"]
        if not np.isfinite(result).all():
            raise RuntimeError("Warehouse pure NN produced nonfinite logits")
        return result

    def act(self, state, *, base_seed: int, deterministic: bool = False,
            temperature: float = 1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        features = observations(state)
        masks = action_masks(state)
        raw_logits = self.logits(features) / temperature
        masked = np.where(masks > 0.5, raw_logits, -1.0e9)
        stable = masked - masked.max(axis=-1, keepdims=True)
        exp = np.exp(stable)
        probabilities = exp / exp.sum(axis=-1, keepdims=True)
        actions = {}
        distributions = {}
        for row, agent_id in enumerate(AGENTS):
            rng = np.random.default_rng(independent_agent_seed(
                base_seed=base_seed, episode_id=state.episode_id,
                frame=state.frame, agent_id=agent_id))
            index = (int(np.argmax(probabilities[row])) if deterministic else
                     int(rng.choice(len(ACTIONS), p=probabilities[row])))
            action = ACTIONS[index]
            actions[agent_id] = action
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id, actions=ACTIONS,
                probabilities=tuple(float(value) for value in probabilities[row]),
                logits=tuple(float(value) for value in masked[row]),
                action_mask=tuple(float(value) for value in masks[row]),
                proposed_action=action,
            )
        return actions, distributions


class FrozenDualPureNN:
    """Two independently parameterised frozen Actors in one pinned artifact."""

    def __init__(self, roles: dict[str, FrozenPureNN], artifact_sha256: str):
        if set(roles) != set(AGENTS):
            raise ValueError("Warehouse dual Actor requires both robot roles")
        self.roles = roles
        self.artifact_sha256 = artifact_sha256
        self.role_sha256 = {agent_id: role_weight_sha256(actor.weights)
                            for agent_id, actor in roles.items()}

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str | None = None,
             expected_role_sha256: dict[str, str] | None = None):
        try:
            raw = Path(path).read_bytes()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Warehouse dual NN artifact missing: {path}") from exc
        digest = sha256(raw).hexdigest()
        if not expected_sha256 or digest != expected_sha256:
            raise RuntimeError("Warehouse dual NN artifact hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if (metadata.get("model_version") != DUAL_MODEL_VERSION or
                metadata.get("observation_version") != OBSERVATION_VERSION or
                metadata.get("observation_dim") != OBSERVATION_DIM or
                metadata.get("map_layout_id") != STUDY_MAP_LAYOUT.layout_id or
                metadata.get("action_names") != list(ACTIONS) or
                metadata.get("agent_ids") != list(AGENTS)):
                raise RuntimeError("Warehouse dual NN contract mismatch")
            roles = {}
            for agent_id in AGENTS:
                weights = {key: np.asarray(archive[f"{agent_id}_{key}"], dtype=np.float32)
                           for key in ("w0", "b0", "w1", "b1", "w2", "b2")}
                actor = FrozenPureNN(weights, digest)
                if (weights["w0"].shape[1] != OBSERVATION_DIM or
                    weights["w2"].shape[0] != len(ACTIONS) or
                    weights["w0"].shape[0] != weights["b0"].size or
                    weights["w1"].shape[1] != weights["b0"].size or
                    weights["w1"].shape[0] != weights["b1"].size or
                    weights["w2"].shape[1] != weights["b1"].size or
                    weights["b2"].size != len(ACTIONS)):
                    raise RuntimeError("Warehouse dual NN tensor shape mismatch")
                roles[agent_id] = actor
        result = cls(roles, digest)
        pinned = expected_role_sha256 or metadata.get("role_sha256")
        if pinned != result.role_sha256:
            raise RuntimeError("Warehouse dual NN role weight hash mismatch")
        return result

    def logits(self, features: np.ndarray) -> np.ndarray:
        rows = np.asarray(features, dtype=np.float32)
        if rows.shape != (len(AGENTS), OBSERVATION_DIM):
            raise ValueError("Warehouse dual Actor expects one observation per role")
        return np.stack([self.roles[agent_id].logits(rows[index])
                         for index, agent_id in enumerate(AGENTS)])

    def act(self, state, *, base_seed: int, deterministic: bool = False,
            temperature: float = 1.0):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        masks = action_masks(state)
        raw_logits = self.logits(observations(state)) / temperature
        masked = np.where(masks > 0.5, raw_logits, -1.0e9)
        stable = masked - masked.max(axis=-1, keepdims=True)
        probabilities = np.exp(stable)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        actions = {}
        distributions = {}
        for index, agent_id in enumerate(AGENTS):
            rng = np.random.default_rng(independent_agent_seed(
                base_seed=base_seed, episode_id=state.episode_id,
                frame=state.frame, agent_id=agent_id))
            choice = (int(np.argmax(probabilities[index])) if deterministic else
                      int(rng.choice(len(ACTIONS), p=probabilities[index])))
            actions[agent_id] = ACTIONS[choice]
            distributions[agent_id] = ActionDistribution(
                agent_id=agent_id, actions=ACTIONS,
                probabilities=tuple(float(value) for value in probabilities[index]),
                logits=tuple(float(value) for value in masked[index]),
                action_mask=tuple(float(value) for value in masks[index]),
                proposed_action=ACTIONS[choice],
            )
        return actions, distributions
