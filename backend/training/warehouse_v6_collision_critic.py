"""Training-only Warehouse collision-risk critic fitted to real physics events.

Every label is the change in ``robot_collision_events`` after an actual
``WarehouseMultiAgentEnv.step`` with two individually legal actions. The
critic supplies a differentiable expected-risk feature for offline BC; it is
never part of the frozen Actor or the deployed action-selection interface.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    OBSERVATION_DIM, OBSERVATION_FEATURE_NAMES, encode_public_observation,
)


COLLISION_CRITIC_VERSION = "warehouse-v6-collision-critic.v1"
COLLISION_CRITIC_ENVIRONMENT_RESETS = 2
_POSITION_COLUMNS = tuple(OBSERVATION_FEATURE_NAMES.index(name) for name in (
    "self_row", "self_column", "partner_row", "partner_column"))
_GEOMETRY_DIM = 4 * 2 * 3 + 2 * len(ACTIONS)
_MOVE = torch.tensor([MOVE_DELTAS.get(action, (0, 0)) for action in ACTIONS],
                     dtype=torch.float32)
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    "backend/training/warehouse_v6_collision_critic.py",
    "core/policy_contracts.py",
    *(path.relative_to(_SOURCE_ROOT).as_posix()
      for path in sorted((_SOURCE_ROOT / "env" / "warehouse").glob("*.py"))),
)


@dataclass(frozen=True)
class _EventRows:
    observations: np.ndarray  # (rows, 2, public observation dimension)
    masks: np.ndarray         # (rows, 2, actions)
    joint_actions: np.ndarray  # (rows, 2) action indices
    collision_events: np.ndarray  # (rows,) labels from env.step only
    position_pairs: np.ndarray  # (rows, 2, 2) pre-action positions


def _hash_arrays(**arrays: np.ndarray) -> str:
    digest = sha256()
    for name, array in sorted(arrays.items()):
        item = np.ascontiguousarray(array)
        digest.update(name.encode("ascii"))
        digest.update(item.dtype.str.encode("ascii"))
        digest.update(np.asarray(item.shape, dtype="<i8").tobytes())
        digest.update(item.tobytes())
    return digest.hexdigest()


def _unordered_pair_key(pair: np.ndarray) -> tuple[int, int, int, int]:
    first, second = (tuple(int(value) for value in position) for position in pair)
    return tuple(value for position in sorted((first, second)) for value in position)


def _source_hashes() -> dict[str, str]:
    return {name: sha256((_SOURCE_ROOT / name).read_bytes()).hexdigest()
            for name in _SOURCE_FILES}


def collision_critic_sha256(critic: "CollisionCritic") -> str:
    """Content hash independent of torch.save serialization details."""
    arrays = {name: parameter.detach().cpu().numpy()
              for name, parameter in critic.state_dict().items()}
    return _hash_arrays(**arrays)


class CollisionCritic(nn.Module):
    """Learned event probability from public positions and joint actions.

    Candidate target vectors are input features. Their geometry is never
    treated as the collision label or used to replace Actor actions.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("move", _MOVE.clone())
        self.net = nn.Sequential(
            nn.Linear(_GEOMETRY_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor,
                joint_action_onehot: torch.Tensor) -> torch.Tensor:
        if (observations.ndim != 3
                or observations.shape[1:] != (len(AGENTS), OBSERVATION_DIM)
                or joint_action_onehot.shape != (
                    len(observations), len(AGENTS) * len(ACTIONS))):
            raise ValueError("Warehouse collision critic input shape mismatch")
        first = observations[:, 0, list(_POSITION_COLUMNS[:2])]
        second = observations[:, 0, list(_POSITION_COLUMNS[2:])]
        scale = observations.new_tensor((max(1, STUDY_MAP_LAYOUT.rows - 1),
                                         max(1, STUDY_MAP_LAYOUT.cols - 1)))
        first, second = first * scale, second * scale
        move_first = joint_action_onehot[:, :len(ACTIONS)] @ self.move
        move_second = joint_action_onehot[:, len(ACTIONS):] @ self.move
        target_first, target_second = first + move_first, second + move_second
        vectors = torch.cat((first - second, target_first - target_second,
                             target_first - second, target_second - first), -1)
        features = torch.cat((vectors, vectors.abs(), vectors.square(),
                              joint_action_onehot), -1)
        return self.net(features).squeeze(-1)


def _environment(seed: int) -> WarehouseMultiAgentEnv:
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=seed)
    state = env.get_state()
    state.participant_controlled_agent_id = AGENTS[0]
    env.set_state(state)
    return env


def _enumerate_event_rows(seed: int, *,
                          position_pairs: Sequence[
                              tuple[tuple[int, int], tuple[int, int]]] | None = None
                          ) -> _EventRows:
    """Enumerate pre-action public states and legal joint env.step events."""
    environments = tuple(_environment(seed) for _ in range(
        COLLISION_CRITIC_ENVIRONMENT_RESETS))
    template_env, branch = environments
    template = template_env.get_state()
    cells = STUDY_MAP_LAYOUT.passable_positions
    pairs = (position_pairs if position_pairs is not None else
             ((first, second) for first in cells for second in cells
              if first != second))
    observations, masks, choices, labels, groups = [], [], [], [], []
    for pair in pairs:
        first, second = pair
        if first == second or not all(STUDY_MAP_LAYOUT.is_passable(pos)
                                       for pos in pair):
            raise ValueError("collision critic requires distinct passable positions")
        state = deepcopy(template)
        for agent, position in zip(state.agents, pair, strict=True):
            agent.position = position
            agent.recent_positions = (position,)
            agent.battery = 100.0
            agent.last_battery_delta = 0.0
            agent.active = True
        public_obs = encode_public_observation(state)
        legal = action_masks(state)
        for first_action in np.flatnonzero(legal[0] > 0.5):
            for second_action in np.flatnonzero(legal[1] > 0.5):
                # Reset every candidate to the identical pre-action state.
                branch.state = deepcopy(state)
                before = branch.state.robot_collision_events
                actions = {AGENTS[0]: ACTIONS[int(first_action)],
                           AGENTS[1]: ACTIONS[int(second_action)]}
                _, _, _, _, info = branch.step(actions)
                if (info["proposed_actions"] != actions
                        or info["action_override_agents"]):
                    raise RuntimeError("environment rewrote a legal critic action")
                observations.append(public_obs)
                masks.append(legal)
                choices.append((first_action, second_action))
                labels.append(int(branch.state.robot_collision_events > before))
                groups.append(pair)
    rows = _EventRows(
        observations=np.asarray(observations, dtype=np.float32),
        masks=np.asarray(masks, dtype=np.float32),
        joint_actions=np.asarray(choices, dtype=np.int16),
        collision_events=np.asarray(labels, dtype=np.uint8),
        position_pairs=np.asarray(groups, dtype=np.int16),
    )
    if not len(rows.collision_events):
        raise RuntimeError("collision critic enumeration yielded no events")
    return rows


def _partition_position_pairs(position_pairs: np.ndarray,
                              seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Both role orientations of one physical position pair share a split."""
    if position_pairs.ndim != 3 or position_pairs.shape[1:] != (2, 2):
        raise ValueError("collision critic position-pair shape mismatch")
    cache: dict[tuple[int, ...], bool] = {}
    training = np.empty(len(position_pairs), dtype=bool)
    for index, pair in enumerate(position_pairs):
        key = _unordered_pair_key(pair)
        if key not in cache:
            digest = sha256(f"{COLLISION_CRITIC_VERSION}|{seed}|{key}".encode()).digest()
            # A quarter of physical position-pair groups provides enough
            # independent collision positives for a meaningful recall check.
            cache[key] = int.from_bytes(digest[:4], "little") % 4 != 0
        training[index] = cache[key]
    validation = ~training
    if not training.any() or not validation.any():
        raise ValueError("collision critic needs nonempty group-disjoint splits")
    return training, validation


def _joint_action_onehot(actions: np.ndarray) -> torch.Tensor:
    indices = torch.as_tensor(actions, dtype=torch.long)
    first = nn.functional.one_hot(indices[:, 0], len(ACTIONS))
    second = nn.functional.one_hot(indices[:, 1], len(ACTIONS))
    return torch.cat((first, second), -1).to(torch.float32)


def train_collision_critic(seed: int, *, epochs: int = 80,
                           batch_size: int = 256,
                           learning_rate: float = 1e-3,
                           positive_weight: float = 3.0
                           ) -> tuple[CollisionCritic, dict]:
    """Fit a training-only critic; return model and reproducibility receipt."""
    if (not isinstance(seed, int) or seed < 0
            or not isinstance(epochs, int) or epochs < 1
            or not isinstance(batch_size, int) or batch_size < 1
            or not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(positive_weight) or positive_weight <= 0):
        raise ValueError("collision critic training parameters must be positive")
    source_files = _source_hashes()
    rows = _enumerate_event_rows(seed)
    training, validation = _partition_position_pairs(rows.position_pairs, seed)
    observations = torch.from_numpy(rows.observations)
    actions = _joint_action_onehot(rows.joint_actions)
    labels = torch.from_numpy(rows.collision_events.astype(np.float32))
    train_indices = torch.from_numpy(np.flatnonzero(training))
    validation_indices = torch.from_numpy(np.flatnonzero(validation))
    if int(labels[train_indices].sum()) < 1 or int(labels[validation_indices].sum()) < 1:
        raise RuntimeError("collision critic group split lacks collision events")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        critic = CollisionCritic()
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    for _ in range(epochs):
        for batch in torch.randperm(len(train_indices), generator=generator).split(batch_size):
            indices = train_indices[batch]
            logits = critic(observations[indices], actions[indices])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels[indices],
                pos_weight=logits.new_tensor(positive_weight))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    critic.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(critic(
            observations[validation_indices], actions[validation_indices]))
        actual = labels[validation_indices].bool()
        predicted = probabilities >= 0.5
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        tn = int((~predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        validation_report = {
            "true_positive": tp, "false_positive": fp,
            "true_negative": tn, "false_negative": fn,
            "positive_rows": int(actual.sum()),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
            "brier": float(((probabilities - actual.to(torch.float32)) ** 2).mean()),
        }
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    group_train = {tuple(int(value) for value in pair.flat)
                   for pair in rows.position_pairs[training]}
    group_valid = {tuple(int(value) for value in pair.flat)
                   for pair in rows.position_pairs[validation]}
    if group_train & group_valid:
        raise RuntimeError("collision critic position-pair leakage")
    unordered_train = {_unordered_pair_key(pair)
                       for pair in rows.position_pairs[training]}
    unordered_valid = {_unordered_pair_key(pair)
                       for pair in rows.position_pairs[validation]}
    if unordered_train & unordered_valid:
        raise RuntimeError("collision critic reversed position-pair leakage")
    report = {
        "version": COLLISION_CRITIC_VERSION,
        "purpose": "offline_BC_gradient_only_not_execution",
        "labels_source": "env.step_robot_collision_events_delta",
        "seed": seed,
        "environment_reset_seed": seed,
        "environment_reset_count": COLLISION_CRITIC_ENVIRONMENT_RESETS,
        "epochs": epochs,
        "batch_size": batch_size,
        "dependencies": {"torch": torch.__version__, "numpy": np.__version__},
        "learning_rate": learning_rate,
        "positive_weight": positive_weight,
        "validation_group_fraction_target": 0.25,
        "data_sha256": _hash_arrays(
            observations=rows.observations, masks=rows.masks,
            joint_actions=rows.joint_actions,
            collision_events=rows.collision_events,
            position_pairs=rows.position_pairs),
        "split_sha256": _hash_arrays(training=training, validation=validation),
        "source_files_sha256": source_files,
        "source_sha256": sha256(json.dumps(
            source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model_sha256": collision_critic_sha256(critic),
        "training_rows": int(training.sum()),
        "validation_rows": int(validation.sum()),
        "training_positive_rows": int(labels[train_indices].sum()),
        "train_position_pairs": len(group_train),
        "validation_position_pairs": len(group_valid),
        "train_unordered_position_pair_groups": len(unordered_train),
        "validation_unordered_position_pair_groups": len(unordered_valid),
        "position_pair_overlap": 0,
        "legal_actions_rewritten_by_environment": 0,
        "validation": validation_report,
        "validation_passed": bool(
            validation_report["positive_rows"] >= 20
            and validation_report["precision"] >= 0.90
            and validation_report["recall"] >= 0.90
            and validation_report["brier"] <= 0.05),
    }
    if _source_hashes() != source_files:
        raise RuntimeError("collision critic source changed during training")
    return critic, report


def risk_matrix(critic: CollisionCritic, observations: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
    """Frozen estimated event probabilities for all legal joint actions.

    The output is for a differentiable *training loss*: it is never used to
    mask, rank, or replace a deployed Actor action. Illegal entries are zero.
    """
    if (not isinstance(observations, torch.Tensor)
            or not isinstance(masks, torch.Tensor)
            or observations.ndim != 3
            or observations.shape[1:] != (len(AGENTS), OBSERVATION_DIM)
            or masks.shape != (len(observations), len(AGENTS), len(ACTIONS))
            or observations.device != critic.move.device
            or masks.device != critic.move.device
            or not bool(torch.isfinite(observations).all())
            or not bool(torch.isfinite(masks).all())):
        raise ValueError("collision critic risk input mismatch")
    pairs = torch.cartesian_prod(torch.arange(len(ACTIONS), device=observations.device),
                                 torch.arange(len(ACTIONS), device=observations.device))
    action_onehot = torch.cat((
        nn.functional.one_hot(pairs[:, 0], len(ACTIONS)),
        nn.functional.one_hot(pairs[:, 1], len(ACTIONS))), -1).to(torch.float32)
    chunks = []
    with torch.no_grad():
        for batch in observations.split(128):
            expanded = batch[:, None].expand(-1, len(pairs), -1, -1).reshape(
                -1, len(AGENTS), OBSERVATION_DIM)
            repeated_actions = action_onehot[None].expand(
                len(batch), -1, -1).reshape(-1, len(AGENTS) * len(ACTIONS))
            chunks.append(torch.sigmoid(critic(expanded, repeated_actions)).reshape(
                len(batch), len(ACTIONS), len(ACTIONS)))
    if not chunks:
        return observations.new_empty((0, len(ACTIONS), len(ACTIONS)))
    estimated = torch.cat(chunks)
    return estimated * (masks[:, 0, :, None] > 0.5).to(estimated.dtype) * (
        masks[:, 1, None, :] > 0.5).to(estimated.dtype)


__all__ = ["COLLISION_CRITIC_VERSION", "CollisionCritic", "collision_critic_sha256",
           "train_collision_critic", "risk_matrix"]
