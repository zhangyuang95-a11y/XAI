"""Pure CPU checkpoint-state migration for a matched public-history experiment.

The caller must verify the original checkpoint bytes and decode them on CPU.
This module never reads/unpickles a file. The independently retained file hash,
semantic payload hash and frozen contracts are required inputs; passing a file
hash alone cannot prove that an in-memory payload came from those bytes.

Only Actor input columns and the matching two Adam moments are expanded. The
result has its own initialization schema, not a resumable r1 checkpoint. It
grants neither a training budget nor capability/release qualification.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import hmac
import json
import math
import re
import struct

import numpy as np
import torch

from backend.training.warehouse_native_public_feedback import (
    BASE_OBSERVATION_SIZE, OBSERVATION_SIZE, GLOBAL_STATE_SIZE,
    HISTORY_FEATURE_NAMES, VERSION as OBSERVER_VERSION,
)

VERSION = "warehouse-native-public-feedback-initialization.v1"
SOURCE_VERSION = "warehouse-native-foundation.v2"
SOURCE_REVISION = "warehouse-native-foundation.r1"
SOURCE_REWARD = "warehouse-native-score-pbrs.collision-r1"
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _hash_string(value, name):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _semantic_update(hasher, value):
    """Typed, length-delimited encoding; no pickle, hidden device copy or RNG."""
    def part(tag, raw):
        hasher.update(tag + struct.pack("!Q", len(raw)) + raw)
    if torch.is_tensor(value):
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Migration accepts dense CPU tensors only")
        try:
            array = value.detach().contiguous().numpy()
        except (TypeError, RuntimeError) as error:
            raise ValueError("Unsupported checkpoint tensor dtype") from error
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("Non-finite or unsupported checkpoint tensor")
        part(b"T", json.dumps([str(value.dtype), list(value.shape)]).encode())
        part(b"B", array.tobytes())
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("Non-finite or unsupported checkpoint array")
        part(b"A", json.dumps([value.dtype.str, list(value.shape)]).encode())
        part(b"B", np.ascontiguousarray(value).tobytes())
    elif isinstance(value, np.generic):
        part(b"N", str(value.dtype).encode()); _semantic_update(hasher, value.item())
    elif value is None:
        part(b"0", b"")
    elif type(value) is bool:
        part(b"b", b"1" if value else b"0")
    elif type(value) is int:
        part(b"i", str(value).encode())
    elif type(value) is float:
        if not math.isfinite(value): raise ValueError("Non-finite checkpoint scalar")
        part(b"f", struct.pack("!d", value))
    elif isinstance(value, str):
        part(b"s", value.encode("utf-8"))
    elif isinstance(value, bytes):
        part(b"r", value)
    elif isinstance(value, Mapping):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError("Checkpoint mapping keys must be strings or integers")
        part(b"d", str(len(value)).encode())
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            _semantic_update(hasher, key); _semantic_update(hasher, value[key])
    elif type(value) in (list, tuple):
        part(b"l" if type(value) is list else b"t", str(len(value)).encode())
        for item in value: _semantic_update(hasher, item)
    else:
        raise ValueError("Unsupported checkpoint value: " + type(value).__name__)


def initialization_sha256(value):
    """Hash complete semantic state. Keep the expected digest before migration."""
    hasher = hashlib.sha256(); _semantic_update(hasher, value)
    return hasher.hexdigest()


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")


def _tensor(value, shape, name):
    if (not torch.is_tensor(value) or value.device.type != "cpu" or value.dtype != torch.float32
            or value.layout != torch.strided or tuple(value.shape) != tuple(shape)
            or not torch.isfinite(value).all().item()):
        raise ValueError("Invalid CPU float32 tensor: " + name)


def _names(values, count, name):
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != count
            or any(not isinstance(value, str) or not value for value in values)
            or len(set(values)) != count):
        raise ValueError("Invalid feature-name contract: " + name)
    return tuple(values)


def migrate_public_feedback_initialization(
    payload, *, source_checkpoint_sha256, expected_source_checkpoint_sha256,
    expected_initialization_sha256, expected_protocol, expected_sources,
    source_feature_names, expected_feature_names, expected_joint_steps=280000,
    allow_test_fixture=False,
):
    """Return detached model/Adam state plus unchanged original resume evidence.

    Both experiment arms must independently clone this same returned state.
    source_checkpoint_sha256 is the caller's verified file digest, not a claim
    that this path-free helper authenticated checkpoint bytes. The semantic
    expected_initialization_sha256 must also be retained by that caller.
    """
    if type(allow_test_fixture) is not bool:
        raise ValueError("Fixture mode must be explicitly boolean")
    if (BASE_OBSERVATION_SIZE != 177 or GLOBAL_STATE_SIZE != 354
            or len(HISTORY_FEATURE_NAMES) != 20
            or OBSERVATION_SIZE != BASE_OBSERVATION_SIZE + len(HISTORY_FEATURE_NAMES)):
        raise ValueError("Public-history migration requires the reviewed 20-feature observer")
    for value, name in ((source_checkpoint_sha256, "source checkpoint hash"),
                        (expected_source_checkpoint_sha256, "expected checkpoint hash"),
                        (expected_initialization_sha256, "expected semantic hash")):
        _hash_string(value, name)
    if not hmac.compare_digest(source_checkpoint_sha256, expected_source_checkpoint_sha256):
        raise ValueError("Source checkpoint hash differs from the frozen external expectation")
    actual_source_hash = initialization_sha256(payload)
    if not hmac.compare_digest(actual_source_hash, expected_initialization_sha256):
        raise ValueError("Source checkpoint state changed from its frozen semantic digest")
    if not isinstance(payload, dict) or not isinstance(expected_protocol, dict) or not isinstance(expected_sources, dict):
        raise ValueError("Checkpoint and trusted contracts must be dictionaries")
    if payload.get("protocol") != expected_protocol or payload.get("revision_sources") != expected_sources:
        raise ValueError("Frozen source protocol or revision-source binding differs")
    if not expected_sources or any(not isinstance(key, str) or not key for key in expected_sources):
        raise ValueError("A trusted source-hash mapping is required")
    for value in expected_sources.values(): _hash_string(value, "source-code hash")
    if not allow_test_fixture and len(expected_sources) != 59:
        raise ValueError("Production r1 source contract must contain 59 frozen files")
    _positive_integer(expected_joint_steps, "Expected source step")
    if not allow_test_fixture and expected_joint_steps != 280000:
        raise ValueError("This production migration is fixed to the r1 280000-step endpoint")
    fixture_fields = (payload.get("test_fixture"), expected_protocol.get("test_fixture"))
    if (allow_test_fixture and fixture_fields != (True, True)) or (not allow_test_fixture and any(value is not None for value in fixture_fields)):
        raise ValueError("Fixture provenance must remain explicit and cannot become production")
    if (payload.get("version") != SOURCE_VERSION or payload.get("experiment_revision") != SOURCE_REVISION
            or payload.get("reward_revision") != SOURCE_REWARD
            or expected_protocol.get("version") != SOURCE_VERSION
            or expected_protocol.get("experiment_revision", {}).get("version") != SOURCE_REVISION
            or type(payload.get("joint_steps")) is not int or payload["joint_steps"] != expected_joint_steps
            or payload.get("training_device") not in ("cpu", "mps")
            or type(payload.get("obs_dim")) is not int or type(payload.get("state_dim")) is not int
            or payload.get("obs_dim") != BASE_OBSERVATION_SIZE or payload.get("state_dim") != GLOBAL_STATE_SIZE):
        raise ValueError("Source checkpoint version, dimensions or step differs")
    for name in ("optimizer_updates", "minibatch_updates"):
        _positive_integer(payload.get(name), name)
    if payload["optimizer_updates"] > payload["minibatch_updates"]:
        raise ValueError("Saved PPO updates cannot exceed saved Adam minibatch updates")
    old_names = _names(source_feature_names, BASE_OBSERVATION_SIZE, "source")
    new_names = _names(expected_feature_names, OBSERVATION_SIZE, "expanded")
    if new_names != old_names + tuple(HISTORY_FEATURE_NAMES):
        raise ValueError("The original feature prefix and appended history order must remain exact")
    architecture = expected_protocol.get("architecture", {})
    if (architecture.get("hidden") != [128, 128] or architecture.get("activation") != "tanh"
            or architecture.get("actions") != ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]
            or architecture.get("masks") is not False or architecture.get("runtime_override") is not False):
        raise ValueError("Source Actor architecture differs from the frozen plain MLP")
    shapes = {"0.weight": (128, BASE_OBSERVATION_SIZE), "0.bias": (128,),
              "2.weight": (128, 128), "2.bias": (128,), "4.weight": (5, 128), "4.bias": (5,)}
    critic_shapes = {**shapes, "0.weight": (128, GLOBAL_STATE_SIZE + 2), "4.weight": (1, 128), "4.bias": (1,)}
    model = payload.get("model", {})
    all_shapes = {**{"actor." + key: value for key, value in shapes.items()},
                  **{"critic." + key: value for key, value in critic_shapes.items()}}
    if set(model) != set(all_shapes):
        raise ValueError("Source checkpoint must contain exactly the Actor and Critic MLP tensors")
    for name, shape in all_shapes.items(): _tensor(model[name], shape, name)
    optimizers = payload.get("optimizers", {})
    if set(optimizers) != {"actor", "critic"}:
        raise ValueError("Both independent Adam states are required")
    for role, role_shapes in (("actor", shapes), ("critic", critic_shapes)):
        optimizer = optimizers[role]
        if not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}:
            raise ValueError("Incomplete Adam state: " + role)
        groups = optimizer["param_groups"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("Exactly one parameter group is required for each independent Adam")
        group = groups[0]
        if (group.get("params") != list(range(6)) or set(optimizer["state"]) != set(range(6))
                or any(type(key) is not int for key in group["params"])
                or any(type(key) is not int for key in optimizer["state"])
                or group.get("lr") != expected_protocol["training"][role + "_learning_rate"]
                or group.get("eps") != 1e-5 or tuple(group.get("betas", ())) != (.9, .999)
                or group.get("weight_decay") != 0
                or any(group.get(name, False) is not False for name in ("amsgrad", "maximize", "differentiable", "decoupled_weight_decay"))):
            raise ValueError("Source Adam parameter order or frozen configuration differs: " + role)
        for index, (name, shape) in enumerate(role_shapes.items()):
            state = optimizer["state"][index]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Incomplete Adam moments: " + role + "." + name)
            _tensor(state["step"], (), role + "." + name + ".step")
            if state["step"].item() != payload["minibatch_updates"]:
                raise ValueError("Adam step differs from actual saved minibatch count")
            for field in ("exp_avg", "exp_avg_sq"):
                _tensor(state[field], shape, role + "." + name + "." + field)
            if (state["exp_avg_sq"] < 0).any().item():
                raise ValueError("Adam variance cannot be negative")

    migrated_model = {key: value.detach().clone() for key, value in model.items()}
    migrated_optimizers = deepcopy(optimizers)
    def expand(value):
        result = value.new_zeros((value.shape[0], OBSERVATION_SIZE))
        result[:, :BASE_OBSERVATION_SIZE].copy_(value)
        return result
    migrated_model["actor.0.weight"] = expand(migrated_model["actor.0.weight"])
    for name in ("exp_avg", "exp_avg_sq"):
        migrated_optimizers["actor"]["state"][0][name] = expand(migrated_optimizers["actor"]["state"][0][name])
    original_state = deepcopy({key: value for key, value in payload.items() if key not in ("model", "optimizers")})
    result = {"version": VERSION, "observer_version": OBSERVER_VERSION,
        "obs_dim": OBSERVATION_SIZE, "state_dim": GLOBAL_STATE_SIZE,
        "feature_names": list(new_names), "source_feature_names": list(old_names),
        "model": migrated_model, "optimizers": migrated_optimizers,
        "source_resume_evidence": original_state,
        "source": {"checkpoint_sha256": source_checkpoint_sha256,
            "semantic_payload_sha256": actual_source_hash, "source_revision": SOURCE_REVISION,
            "joint_steps": payload["joint_steps"], "optimizer_updates": payload["optimizer_updates"],
            "minibatch_updates": payload["minibatch_updates"],
            "model_sha256": initialization_sha256(model), "optimizers_sha256": initialization_sha256(optimizers),
            "model_tensor_sha256": {key: initialization_sha256(value) for key, value in model.items()},
            "file_to_decoded_payload_binding": "verified by caller before this pure migration"},
        "migration": {"changed_model_tensors": ["actor.0.weight"],
            "changed_adam_tensors": ["actor.state.0.exp_avg", "actor.state.0.exp_avg_sq"],
            "old_actor_width": BASE_OBSERVATION_SIZE, "new_actor_width": OBSERVATION_SIZE,
            "zero_added_columns": len(HISTORY_FEATURE_NAMES), "critic_unchanged": True,
            "old_prefix_bitwise_preserved": True, "adam_steps_preserved": True},
        "test_fixture": allow_test_fixture, "training_authorized": False,
        "qualification_evaluated": False, "r1_resumable_checkpoint": False,
        "numerical_scope": "Zero columns preserve the old mathematical function; expanded float32 GEMM may differ in final rounding bits."}
    result["model_sha256"] = initialization_sha256(migrated_model)
    result["model_tensor_sha256"] = {key: initialization_sha256(value) for key, value in migrated_model.items()}
    result["optimizers_sha256"] = initialization_sha256(migrated_optimizers)
    if initialization_sha256(payload) != actual_source_hash:
        raise RuntimeError("Input checkpoint changed during pure migration")
    return result
