"""Versioned pure-neural Warehouse study adapter over unchanged physics."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import re

from backend.training.warehouse_selective_v5 import SelectiveLeafProgram as AxisLeafProgram
from backend.training.warehouse_sparse_v5 import (
    PROGRAM_VERSION as SPARSE_PROGRAM_VERSION,
    SparseObliqueProgram,
)
from backend.inference.warehouse_structured_v5 import (
    FrozenStructuredDualNN, STRUCTURED_FEATURE_SCHEMA_SHA256,
    STRUCTURED_MODEL_VERSION,
)
from backend.training.warehouse_v6_actor import (
    FrozenWarehouseV6Actor, FEATURE_SCHEMA_SHA256 as V6_FEATURE_SCHEMA_SHA256,
    LEGACY_MODEL_VERSION as V6_LEGACY_MODEL_VERSION,
    V616_MODEL_VERSION,
    MODEL_VERSION as V6_MODEL_VERSION,
)
from backend.training.warehouse_v6_program import WarehouseV6Program
from env.warehouse.public_action_features_v614 import (
    ACTOR_FEATURE_SCHEMA_SHA256 as LEGACY_V6_FEATURE_SCHEMA_SHA256,
    ACTION_PHYSICS_CONFIG_SHA256, validate_study_action_physics,
)
from env.warehouse.public_observation_v6 import (
    OBSERVATION_VERSION as V6_OBSERVATION_VERSION,
    encode_role_observation as v6_role_observation,
)
from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState, participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse.decision_protocol import distribution_decision_metadata
from env.warehouse.pure_nn import (
    FrozenPureNN, FrozenDualPureNN, MODEL_VERSION, DUAL_MODEL_VERSION,
    OBSERVATION_VERSION, action_masks, observation, observation_feature_names,
)
from . import turnbased as legacy

DOMAIN = "warehouse"
ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/study_v3_warehouse_neural.json").read_text())
V5_CONFIG = json.loads((ROOT / "configs/study_v3_warehouse_neural_v5.json").read_text())
V6_CONFIG_PATH = ROOT / "configs/study_v3_warehouse_neural_v6.json"
STRUCTURED_CONFIG_NAME = "configs/study_v3_warehouse_neural_structured_v54.json"
VERSION = CONFIG["version"]
RELEASE_VERSION = V5_CONFIG["version"]
_ACTORS = {"human": "robot_1", "ai": "robot_2"}
LEGACY_POLICY_ID = "warehouse-legacy-v4"
_V5_POLICY_IDS = {"warehouse-regularized-v5": "regularized", "warehouse-raw-v5": "raw"}
_STRUCTURED_POLICY_IDS = {"warehouse-structured-regularized-v54": "regularized",
                          "warehouse-structured-raw-v54": "raw"}
_V6_POLICY_IDS = {"warehouse-regularized-v6": "regularized",
                  "warehouse-raw-v6": "raw"}
V6_RELEASE_ID = "policylens-three-domain-20261016.v6.0"
V6_CATALOG_PATH = ROOT / "releases" / V6_RELEASE_ID / "catalog.json"
_V6_POLICY_PATTERN = re.compile(
    r"warehouse-(raw|regularized)-v6(?:\.([1-9][0-9]*))?\Z")


def _v6_identity(policy_id: str | None) -> tuple[str, str] | None:
    """Resolve an explicit v6.x policy to its unique versioned release."""
    if not isinstance(policy_id, str):
        return None
    matched = _V6_POLICY_PATTERN.fullmatch(policy_id)
    if matched is None:
        return None
    from study_v3.release_catalog import (
        ReleaseAssetUnavailable, warehouse_release_spec)
    variant, minor = matched.groups()
    release_id = (V6_RELEASE_ID if minor is None else
                  V6_RELEASE_ID.rsplit(".", 1)[0] + "." + minor)
    try:
        spec = warehouse_release_spec(release_id)
    except (ReleaseAssetUnavailable, ValueError):
        return None
    if spec.policy_ids[variant] != policy_id:
        raise ValueError("Warehouse v6 policy and release identity differ")
    return variant, release_id


def _v6_config():
    try:
        config = json.loads(V6_CONFIG_PATH.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Warehouse v6 policy configuration unavailable") from exc
    if (config.get("version") != config.get("model_version")
            or config.get("model_version") not in
               (V6_LEGACY_MODEL_VERSION, V616_MODEL_VERSION, V6_MODEL_VERSION)
            or config.get("observation_version") != V6_OBSERVATION_VERSION
            or config.get("feature_schema_sha256") != (V6_FEATURE_SCHEMA_SHA256 if config.get("model_version") == V6_MODEL_VERSION else LEGACY_V6_FEATURE_SCHEMA_SHA256)
            or config.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256
            or config.get("release_id") != V6_RELEASE_ID
            or config.get("pilot_scenario_seeds") != list(range(1000, 1024))
            or {row.get("policy_id") for row in config.get("variants", {}).values()}
               != set(_V6_POLICY_IDS)):
        raise RuntimeError("Warehouse v6 policy configuration schema mismatch")
    return config


def _checked_structured_config(config, policy_ids):
    """Require the same model contract for candidates and later promotion."""
    if (not isinstance(config, dict) or
            not isinstance(config.get("variants"), dict) or
            config.get("version") != STRUCTURED_MODEL_VERSION or
            config.get("model_version") != STRUCTURED_MODEL_VERSION or
            config.get("feature_schema_sha256") != STRUCTURED_FEATURE_SCHEMA_SHA256 or
            config.get("observation_version") != OBSERVATION_VERSION or
            set(config.get("variants") or {}) != {"raw", "regularized"} or
            any(not isinstance(config["variants"][variant], dict) or
                config["variants"][variant].get("policy_id") != policy_id
                for policy_id, variant in policy_ids.items())):
        raise RuntimeError("Warehouse structured policy configuration mismatch")
    for variant, record in config["variants"].items():
        if not isinstance(record, dict):
            raise RuntimeError("Warehouse structured policy configuration mismatch")
        paths = [record.get("actor_path"), record.get("training_report_path"),
                 record.get("acceptance_path"), *(record.get("program_paths") or [])]
        if (record.get("feature_schema_sha256") != STRUCTURED_FEATURE_SCHEMA_SHA256
                or not isinstance(record.get("program_paths"), list)
                or len(record["program_paths"]) != (2 if variant == "regularized" else 0)
                or any(not isinstance(value, str) or not value or
                       not (ROOT / value).resolve().is_relative_to(ROOT)
                       for value in paths)):
            raise RuntimeError("Warehouse structured policy paths or schema mismatch")
    return config


def _structured_config():
    """Load the isolated candidate only when its explicit policy ID is used."""
    path = ROOT / STRUCTURED_CONFIG_NAME
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Warehouse structured policy configuration unavailable") from exc
    return _checked_structured_config(config, _STRUCTURED_POLICY_IDS)


def _config_for_policy(policy_id):
    if policy_id == LEGACY_POLICY_ID:
        return CONFIG
    if policy_id in _V5_POLICY_IDS:
        return (_checked_structured_config(V5_CONFIG, _V5_POLICY_IDS)
                if V5_CONFIG.get("model_version") == STRUCTURED_MODEL_VERSION else
                V5_CONFIG)
    if policy_id in _STRUCTURED_POLICY_IDS:
        return _structured_config()
    v6 = _v6_identity(policy_id)
    if v6 is not None:
        config = _v6_config()
        if v6[1] == V6_RELEASE_ID:
            return config
        # The physical scenario config is shared; the release and policy IDs
        # are supplied by the versioned, content-addressed catalog.
        from study_v3.release_catalog import warehouse_release_spec
        spec = warehouse_release_spec(v6[1])
        return {**config, "release_id": v6[1],
                "variants": {variant: {**config["variants"][variant],
                                       "policy_id": spec.policy_ids[variant]}
                             for variant in ("raw", "regularized")}}
    raise ValueError("unknown Warehouse policy_id")


def _variant_for_policy(policy_id):
    if policy_id in _V5_POLICY_IDS:
        return _V5_POLICY_IDS[policy_id]
    if policy_id in _STRUCTURED_POLICY_IDS:
        return _STRUCTURED_POLICY_IDS[policy_id]
    v6 = _v6_identity(policy_id)
    if v6 is not None:
        return v6[0]
    raise ValueError("unknown Warehouse policy_id")


def policy_artifacts(release_id=None):
    """Immutable v5 policy inventory; incomplete entries keep pilot closed."""
    if release_id is not None:
        from study_v3.release_catalog import ReleaseAssetUnavailable, warehouse_release_spec
        try:
            warehouse_release_spec(release_id)
        except ReleaseAssetUnavailable as exc:
            raise ValueError("unknown Warehouse release_id") from exc
        return policy_artifacts_v6(release_id)
    return _policy_artifacts_from(_config_for_policy("warehouse-raw-v5"))


def policy_artifacts_v6(release_id: str = V6_RELEASE_ID):
    """Read only content-addressed v6 release entries; pending stays closed."""
    if release_id == V6_RELEASE_ID:
        _v6_config()
    from study_v3.release_catalog import (
        ReleaseAssetUnavailable, warehouse_policy_artifacts, warehouse_release_spec,
    )
    spec = warehouse_release_spec(release_id)
    try:
        entries = warehouse_policy_artifacts(release_id)
    except (OSError, ValueError, ReleaseAssetUnavailable):
        entries = {}
    result = {}
    for variant, policy_id in spec.policy_ids.items():
        entry = entries.get(variant) or {}
        roles = entry.get("role_sha256") or {}
        program_paths = entry.get("program_paths") or []
        program_hashes = entry.get("program_sha256") or {}
        paths = [entry.get("actor_path"), entry.get("training_report_path"),
                 entry.get("acceptance_path"), *program_paths]
        hashes = [entry.get("actor_sha256"), entry.get("training_report_sha256"),
                  entry.get("acceptance_sha256"),
                  *(program_hashes.get(role) for role in ("robot_1", "robot_2")
                    if variant == "regularized")]
        ready = bool(
            entries and entry.get("policy_id") == policy_id
            and entry.get("model_version") == V6_MODEL_VERSION
            and entry.get("feature_schema_sha256") == V6_FEATURE_SCHEMA_SHA256
            and entry.get("action_physics_config_sha256") == ACTION_PHYSICS_CONFIG_SHA256
            and set(roles) == {"human", "ai"}
            and all(isinstance(value, str) and len(value) == 64 for value in roles.values())
            and len(program_paths) == (2 if variant == "regularized" else 0)
            and len(paths) == len(hashes) + (0 if variant == "regularized" else 0)
            and all(isinstance(path, str) and path for path in paths)
            and all(isinstance(value, str) and len(value) == 64 for value in hashes))
        if ready:
            for path, expected in zip(paths, hashes, strict=True):
                target = (ROOT / path).resolve()
                if (not target.is_relative_to(ROOT) or not target.is_file()
                        or sha256(target.read_bytes()).hexdigest() != expected):
                    ready = False
                    break
        if ready:
            try:
                receipt = json.loads((ROOT / entry["acceptance_path"]).read_text())
                ready = bool(receipt.get("passed") is True
                             and receipt.get("heldout_evaluated") is True
                             and receipt.get("release_id") == release_id)
            except (OSError, ValueError, TypeError):
                ready = False
        result[variant] = {
            "policy_id": policy_id, "model_version": V6_MODEL_VERSION,
            "feature_schema_sha256": V6_FEATURE_SCHEMA_SHA256,
            "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256,
            "actor_path": entry.get("actor_path"),
            "actor_sha256": entry.get("actor_sha256"),
            "role_sha256": roles,
            "program_paths": program_paths,
            "program_sha256": program_hashes,
            "training_report_path": entry.get("training_report_path"),
            "training_report_sha256": entry.get("training_report_sha256"),
            "acceptance_path": entry.get("acceptance_path"),
            "acceptance_sha256": entry.get("acceptance_sha256"),
            "readiness": ready,
        }
    return result


def structured_policy_artifacts():
    """Candidate inventory, separate from the active A/B/C assignment."""
    return _policy_artifacts_from(_structured_config())


def _policy_artifacts_from(config):
    result = {}
    structured = config["model_version"] == STRUCTURED_MODEL_VERSION
    for variant, record in config["variants"].items():
        actor_path = record["actor_path"]
        actor_sha = record.get("actor_sha256")
        role_sha = record.get("role_sha256") or {}
        acceptance_path = record["acceptance_path"]
        passed = False
        target = ROOT / acceptance_path
        receipt = {}
        if target.is_file():
            try:
                receipt = json.loads(target.read_text())
                passed = bool(receipt.get("passed") and receipt.get("heldout_evaluated")
                              and receipt.get("artifact_sha256", {}).get(variant) == actor_sha
                              and receipt.get("role_sha256", {}).get(variant) == role_sha
                              and (not structured or
                                   (receipt.get("actor_model_version") == STRUCTURED_MODEL_VERSION
                                    and receipt.get("feature_schema_sha256") ==
                                        STRUCTURED_FEATURE_SCHEMA_SHA256)))
            except (AttributeError, TypeError, ValueError, OSError):
                passed = False
        program_hashes = receipt.get("program_sha256", {}).get(variant, {})
        report_hash = receipt.get("training_report_sha256", {}).get(variant)
        expected_paths = list(record.get("program_paths", []))
        actual_programs = (all((ROOT / path).is_file() and
                               sha256((ROOT / path).read_bytes()).hexdigest() == program_hashes.get(role)
                               for path, role in zip(expected_paths, ("robot_1", "robot_2"), strict=True))
                           if expected_paths else True)
        actual_report = bool(report_hash and (ROOT / record["training_report_path"]).is_file() and
                             sha256((ROOT / record["training_report_path"]).read_bytes()).hexdigest() == report_hash)
        result[variant] = {
            "policy_id": record["policy_id"],
            "model_version": config["model_version"],
            "feature_schema_sha256": (config.get("feature_schema_sha256")
                                      if structured else None),
            "actor_path": actor_path,
            "actor_sha256": actor_sha,
            "role_sha256": {"human": role_sha.get("robot_1"), "ai": role_sha.get("robot_2")},
            "program_paths": expected_paths,
            "program_sha256": program_hashes,
            "training_report_path": record["training_report_path"],
            "training_report_sha256": report_hash,
            "acceptance_path": acceptance_path,
            "acceptance_sha256": sha256(target.read_bytes()).hexdigest() if target.is_file() else None,
            "readiness": bool(passed and actor_sha and all(role_sha.get(role) for role in _ACTORS.values())
                              and (ROOT / actor_path).is_file() and
                              sha256((ROOT / actor_path).read_bytes()).hexdigest() == actor_sha and
                              actual_programs and actual_report),
        }
    return result


def _policy_id(policy_id):
    if policy_id is None:
        return LEGACY_POLICY_ID
    if (policy_id not in _V5_POLICY_IDS and
            policy_id not in _STRUCTURED_POLICY_IDS and
            _v6_identity(policy_id) is None and
            policy_id != LEGACY_POLICY_ID):
        raise ValueError("unknown Warehouse policy_id")
    return policy_id


def _verify_v5_artifact(policy_id):
    if policy_id == LEGACY_POLICY_ID:
        return
    v6 = _v6_identity(policy_id)
    if v6 is not None:
        variant, release_id = v6
        entry = policy_artifacts_v6(release_id)[variant]
        if not entry["readiness"]:
            raise RuntimeError("Warehouse v6 accepted policy binding is unavailable")
        from study_v3 import policies as study_policies
        try:
            catalog_entry = study_policies._catalog_entry(
                DOMAIN, variant, release_id=release_id)
            binding = study_policies.binding_for_group(
                DOMAIN, "C" if variant == "raw" else "A",
                release_id=release_id)
            if not study_policies._accepted(
                    catalog_entry, binding, release_id=release_id):
                raise RuntimeError("Warehouse v6 model acceptance gate failed")
        except (AttributeError, KeyError, TypeError, ValueError, OSError,
                RuntimeError, study_policies.PolicyUnavailable) as exc:
            raise RuntimeError("Warehouse v6 accepted policy binding is unavailable") from exc
        actor = _actor(policy_id)
        if (binding.get("actor_sha256") != actor.artifact_sha256
                or binding.get("human_actor_sha256") != actor.role_sha256["robot_1"]
                or binding.get("ai_actor_sha256") != actor.role_sha256["robot_2"]
                or binding.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256):
            raise RuntimeError("Warehouse v6 runtime Actor binding mismatch")
        return
    variant = _variant_for_policy(policy_id)
    config = _config_for_policy(policy_id)
    record = config["variants"][variant]
    path = ROOT / record["actor_path"]
    try:
        digest = sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Warehouse dual NN artifact missing: {path}") from exc
    if not record.get("actor_sha256") or digest != record["actor_sha256"]:
        raise RuntimeError("Warehouse dual NN artifact hash mismatch")
    # This runs on every direct v5 entrypoint, including restored states and
    # demonstrations.  The central release gate verifies the complete
    # three-pair evidence and both once-only final receipts; checking it here
    # prevents a direct domain call from bypassing enrollment readiness.
    from study_v3 import policies as study_policies
    try:
        entry = (structured_policy_artifacts()[variant]
                 if policy_id in _STRUCTURED_POLICY_IDS else
                 policy_artifacts()[variant])
        binding = (study_policies.binding_for_warehouse_structured(policy_id)
                   if policy_id in _STRUCTURED_POLICY_IDS else
                   study_policies.binding_for_group(
                       DOMAIN, "C" if variant == "raw" else "A"))
        accepted = entry["readiness"] and study_policies._accepted(entry, binding)
    except (AttributeError, KeyError, TypeError, ValueError, OSError,
            study_policies.PolicyUnavailable) as exc:
        raise RuntimeError("Warehouse v5 accepted policy binding is unavailable") from exc
    if not accepted:
        raise RuntimeError("Warehouse v5 accepted policy binding is unavailable")
    cached_actor = _actor(policy_id)
    if (cached_actor.artifact_sha256 != digest or
            cached_actor.role_sha256 != record.get("role_sha256")):
        raise RuntimeError("Warehouse cached dual Actor no longer matches the accepted binding")


@lru_cache(maxsize=4)
def _actor(policy_id=None):
    policy_id = _policy_id(policy_id)
    if policy_id == LEGACY_POLICY_ID:
        if CONFIG["model_version"] != MODEL_VERSION or CONFIG["observation_version"] != OBSERVATION_VERSION:
            raise RuntimeError("Warehouse pure NN configuration mismatch")
        return FrozenPureNN.load(ROOT / CONFIG["actor_path"], expected_sha256=CONFIG["actor_sha256"])
    v6 = _v6_identity(policy_id)
    if v6 is not None:
        entry = policy_artifacts_v6(v6[1])[v6[0]]
        if not entry["readiness"]:
            raise RuntimeError("Warehouse v6 Actor is not accepted")
        actor = FrozenWarehouseV6Actor.load(
            ROOT / entry["actor_path"], expected_sha256=entry["actor_sha256"],
            expected_role_sha256={"robot_1": entry["role_sha256"]["human"],
                                  "robot_2": entry["role_sha256"]["ai"]})
        if actor.model_version != entry["model_version"]:
            raise RuntimeError("Warehouse v6 Actor model version differs from release")
        return actor
    config = _config_for_policy(policy_id)
    if config["observation_version"] != OBSERVATION_VERSION:
        raise RuntimeError("Warehouse dual NN observation configuration mismatch")
    variant = _variant_for_policy(policy_id)
    record = config["variants"][variant]
    if record["policy_id"] != policy_id:
        raise RuntimeError("Warehouse policy configuration mismatch")
    if config["model_version"] == STRUCTURED_MODEL_VERSION:
        return FrozenStructuredDualNN.load(
            ROOT / record["actor_path"], expected_sha256=record.get("actor_sha256"),
            expected_role_sha256=record.get("role_sha256"))
    if config["model_version"] != DUAL_MODEL_VERSION:
        raise RuntimeError("Warehouse dual NN configuration mismatch")
    return FrozenDualPureNN.load(ROOT / record["actor_path"],
                                 expected_sha256=record.get("actor_sha256"),
                                 expected_role_sha256=record.get("role_sha256"))


def _snapshot(env):
    return {"version": CONFIG["snapshot_version"], "state": legacy._pack(asdict(env.state)),
            "rng": legacy._pack(env.get_rng_state()), "episode_counter": env._episode_counter}


def _v6_public_history(state):
    """Only earlier submitted commands and physical outcomes, in UI order."""
    previous = None if state.frame == 0 else {
        role: {"proposed": state.by_id(agent_id).last_action.lower(),
               "executed": state.by_id(agent_id).last_executed_action.lower()}
        for role, agent_id in _ACTORS.items()
    }
    return {"previous_joint_actions": previous,
            "previous_collision_kind": (state.last_robot_collision_kind
                                        if state.frame else None),
            "same_target_collision_streak": state.same_target_collision_streak}


def _restore(state):
    policy_id = _policy_id(state.get("policy_id"))
    _verify_v5_artifact(policy_id)
    v6 = _v6_identity(policy_id)
    if v6 is not None and (
            state.get("release_id") != v6[1] or
            state.get("feature_schema_sha256") != V6_FEATURE_SCHEMA_SHA256 or
            state.get("action_physics_config_sha256") != ACTION_PHYSICS_CONFIG_SHA256):
        raise RuntimeError("Warehouse v6 state release or feature binding mismatch")
    expected_version = _config_for_policy(policy_id)["version"]
    if state.get("version") != expected_version:
        raise ValueError("warehouse pure NN state version mismatch")
    data = state["snapshot"]
    if data["version"] != CONFIG["snapshot_version"]:
        raise ValueError("warehouse pure NN snapshot version mismatch")
    raw = legacy._unpack(data["state"])
    raw["agents"] = [AgentState(**item) for item in raw["agents"]]
    for field in ("tasks", "completed_tasks"):
        raw[field] = [DeliveryTask(**item) for item in raw[field]]
    env = WarehouseMultiAgentEnv(participant_study_config())
    if v6 is not None:
        validate_study_action_physics(env.config)
    env.state = WarehouseState(**raw)
    env._episode_counter = int(data["episode_counter"])
    env.set_rng_state(legacy._unpack(data["rng"]))
    errors = env.validate_state(env.state)
    if errors:
        raise ValueError("invalid warehouse pure NN snapshot: " + str(errors))
    if env.state.frame != state["turn"] or bool(env.state.terminated or env.state.truncated) != state["terminal"]:
        raise ValueError("warehouse pure NN snapshot boundary mismatch")
    if v6 is not None:
        history = _v6_public_history(env.state)
        if any(state.get(key) != value for key, value in history.items()):
            raise ValueError("Warehouse v6 public history differs from bound snapshot")
        prior = history["previous_joint_actions"]
        expected_actions = (None if prior is None else
                            {role: values["proposed"] for role, values in prior.items()})
        if state.get("last_actions") != expected_actions:
            raise ValueError("Warehouse v6 submitted actions differ from bound snapshot")
    if policy_id != LEGACY_POLICY_ID:
        actor = _actor(policy_id)
        if (state.get("actor_sha256") != actor.artifact_sha256 or
            state.get("role_sha256") != {"human": actor.role_sha256["robot_1"],
                                          "ai": actor.role_sha256["robot_2"]}):
            raise RuntimeError("Warehouse state policy binding mismatch")
    return env


def _state(env, task, seed, events=None, last_actions=None, charges=0, policy_id=None):
    s = env.state
    policy_id = _policy_id(policy_id)
    config = _config_for_policy(policy_id)
    result = {"domain": DOMAIN,
            "version": VERSION if policy_id == LEGACY_POLICY_ID else config["version"],
            "scenario_version": config["scenario_version"],
            "task": task, "seed": int(seed), "scenario_seed": _task_seed(seed, task, policy_id),
            "turn": s.frame, "max_turns": env.config.horizon, "terminal": bool(s.terminated or s.truncated),
            "events": legacy._plain(events or []), "policy_memory": {}, "snapshot": legacy._plain(_snapshot(env)),
            "human": legacy._actor_view(s.by_id("robot_1")), "ai": legacy._actor_view(s.by_id("robot_2")),
            "orders": legacy._orders(env), "deliveries": s.total_deliveries,
            "collision_events": s.robot_collision_events, "shutdown_events": s.shutdown_count,
            "charge_events": charges, "last_actions": last_actions}
    if policy_id != LEGACY_POLICY_ID:
        actor = _actor(policy_id)
        result.update(policy_id=policy_id, actor_sha256=actor.artifact_sha256,
                      role_sha256={"human": actor.role_sha256["robot_1"],
                                   "ai": actor.role_sha256["robot_2"]})
    v6 = _v6_identity(policy_id)
    if v6 is not None:
        result.update(release_id=v6[1],
                      feature_schema_sha256=V6_FEATURE_SCHEMA_SHA256,
                      action_physics_config_sha256=validate_study_action_physics(env.config),
                      **_v6_public_history(s))
    return result


def _task_seed(seed: int, task: int, policy_id: str | None) -> int:
    """Keep v4 fixed scenarios while balancing actual v5 task scenarios."""
    if task == 0:
        return int(seed)
    if policy_id == LEGACY_POLICY_ID:
        return int(CONFIG["task_seeds"][str(task)])
    return int(seed) * 10 + int(task)


def initial_state(seed: int, task: int, policy_id: str | None = None,
                  release_id: str | None = None):
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2, or 3")
    policy_id = _policy_id(policy_id)
    v6 = _v6_identity(policy_id)
    if v6 is not None and release_id not in (None, v6[1]):
        raise RuntimeError("Warehouse v6 release_id mismatch")
    if v6 is None and isinstance(release_id, str) and release_id.startswith(
            V6_RELEASE_ID.rsplit(".", 1)[0] + "."):
        raise RuntimeError("Warehouse v6 requires a v6 policy_id")
    _verify_v5_artifact(policy_id)
    _actor(policy_id)
    env = WarehouseMultiAgentEnv(participant_study_config())
    if v6 is not None:
        validate_study_action_physics(env.config)
    env.reset(seed=_task_seed(seed, task, policy_id))
    s = env.get_state()
    s.participant_controlled_agent_id = "robot_1"
    env.set_state(s)
    return _state(env, task, seed, policy_id=policy_id)


def legal_actions(state, actor="human"):
    if actor not in _ACTORS:
        raise ValueError("unknown actor")
    return [] if state["terminal"] else [action.lower() for action in ACTIONS]


def _selection(env, seed, policy_id=None):
    if _v6_identity(policy_id) is not None:
        validate_study_action_physics(env.config)
    actions, distributions = _actor(policy_id).act(env.state, base_seed=seed)
    ai_action = actions["robot_2"]
    if action_masks(env.state, env.config.map_layout_id)[1, ACTIONS.index(ai_action)] <= 0.5:
        raise RuntimeError("pure NN selected a physically illegal AI action")
    return ai_action, actions, distributions


def decide(state):
    if state["terminal"]:
        _restore(state)
        return {"action": "wait", "reason_code": "complete",
                "reason_en": "The task has finished.", "reason_zh": "任务已经结束。",
                "goal": "complete", "memory": {}, "alternatives": [], "facts": []}
    env = _restore(state)
    selected, actions, distributions = _selection(env, state["scenario_seed"], state.get("policy_id"))
    dist = distributions["robot_2"]
    probability = dist.probabilities[ACTIONS.index(selected)]
    en = (f"My frozen neural policy selected {selected.lower()} with probability {probability:.3f} "
          "from the visible pre-action state. This output alone does not establish a single causal motive.")
    zh = (f"冻结的神经网络根据行动前可见状态选择了{selected.lower()}，该动作概率为{probability:.3f}。"
          "仅凭输出不能确定唯一的决策原因。")
    alternatives = []
    for action, allowed in zip(ACTIONS, dist.action_mask):
        if action == selected or allowed <= 0.5:
            continue
        cases = legacy._collision_cases(env, action.lower())
        alternatives.append({"action": action.lower(),
            "en": (f"If the AI chose {action.lower()}, {len(cases)} of your five possible commands "
                   "would cause a robot collision in this step."),
            "zh": f"若AI选择{action.lower()}，你本步的五种可能指令中有{len(cases)}种会导致机器人碰撞。",
            "collision_cases": cases})
    trace = {"mode": "pure_neural",
             "actor_sha256": _actor(state.get("policy_id")).artifact_sha256,
             "observation_version": (_config_for_policy(state.get("policy_id"))["observation_version"]
                                     if state.get("policy_id") else OBSERVATION_VERSION),
             "policy_actions": actions, "selected_actions": dict(actions),
             "selection_changed_policy": False,
             "ai_action_distribution": {"actions": list(ACTIONS),
                "probabilities": list(dist.probabilities), "logits": list(dist.logits),
                "action_mask": list(dist.action_mask)}}
    if state.get("policy_id"):
        trace["policy_id"] = state["policy_id"]
        trace["role_sha256"] = dict(state["role_sha256"])
    result = {"action": selected.lower(), "reason_code": "neural_choice",
            "reason_en": en, "reason_zh": zh, "goal": "not_inferred",
            "memory": {}, "alternatives": alternatives, "facts": [],
            "controller_trace": trace,
            "snapshot_sha256": sha256(json.dumps(state["snapshot"], sort_keys=True).encode()).hexdigest()}
    if state.get("policy_id"):
        result.update(policy_id=state["policy_id"], actor_sha256=state["actor_sha256"],
                      role_sha256=dict(state["role_sha256"]))
    return result


def step(state, human_action, decision=None):
    if state["terminal"]:
        raise ValueError("task complete")
    if human_action not in legal_actions(state):
        raise ValueError("illegal human action")
    actual = decide(state)
    if decision is not None and decision != actual:
        raise ValueError("decision does not match this pre-action state")
    env = _restore(state)
    before = env.get_state()
    selected, proposed, distributions = _selection(env, state["scenario_seed"], state.get("policy_id"))
    if selected.lower() != actual["action"]:
        raise RuntimeError("Warehouse neural decision changed between decide and step")
    human = human_action.upper()
    actions = {"robot_1": human, "robot_2": selected}
    runtime = {"mode": "pure_neural",
               "actor_sha256": _actor(state.get("policy_id")).artifact_sha256,
               "policy_actions": proposed, "selected_actions": dict(actions),
               "selection_changed_policy": False, "ai_action_overrides": 0,
               "human_role_replacement": True}
    if state.get("policy_id"):
        runtime["policy_id"] = state["policy_id"]
        runtime["role_sha256"] = dict(state["role_sha256"])
    _, _, _, _, info = env.step(actions, decision_metadata=distribution_decision_metadata(
        distributions, decision_source="participant_plus_pure_neural_actor",
        participant_overrides={"robot_1": human}, policy_actions=proposed,
        selected_actions=actions, runtime_decision=runtime))
    if (info["proposed_actions"]["robot_2"] != selected or
        info["controller_selected_actions"]["robot_2"] != selected or
        "robot_2" in info["action_override_agents"]):
        raise RuntimeError("Warehouse pure NN AI action was rewritten before physics")
    events = legacy._events(before, env.state, info)
    result = _state(env, state["task"], state["seed"], events,
                    {"human": human.lower(), "ai": selected.lower()},
                    state["charge_events"] + sum(event["type"] == "charge" for event in events),
                    policy_id=state.get("policy_id"))
    result["scenario_seed"] = state["scenario_seed"]
    return result


def score(state):
    s = _restore(state).state
    return {"task_score": float(s.user_score), "raw_score": float(s.user_score),
            "score_max": None, "score_scale": "raw", "breakdown": dict(s.score_breakdown),
            "metrics": {"deliveries": s.total_deliveries,
                        "collision_events": s.robot_collision_events,
                        "shutdown_events": s.shutdown_count,
                        "charge_events": state["charge_events"],
                        "elapsed_turns": s.frame,
                        "human_detour_units": s.human_route_regret_units,
                        "shared_charger_penalties": int(-s.score_breakdown["shared_charger_occupancy"] / 5),
                        "shared_charger_penalty_points": s.score_breakdown["shared_charger_occupancy"],
                        "terminal_reason": s.terminal_reason}}


def public_state(state):
    env = _restore(state)
    public = {key: deepcopy(state[key]) for key in ("domain", "version", "task", "turn", "max_turns", "terminal", "events", "policy_id", "actor_sha256", "role_sha256", "release_id", "feature_schema_sha256", "action_physics_config_sha256") if key in state}
    public.update({key: legacy._actor_view(env.state.by_id(agent_id)) for key, agent_id in _ACTORS.items()})
    public.update(width=STUDY_MAP_LAYOUT.cols, height=STUDY_MAP_LAYOUT.rows,
        walls=[list(reversed(position)) for position in sorted(STUDY_MAP_LAYOUT.blocked_positions)],
        orders=legacy._orders(env), chargers=[list(reversed(STUDY_MAP_LAYOUT.charger_position))],
        stations=[{"id": "charger", "x": STUDY_MAP_LAYOUT.charger_position[1],
                   "y": STUDY_MAP_LAYOUT.charger_position[0], "kind": "charger",
                   "label_en": "Shared charger", "label_zh": "共享充电位置"}],
        score=score(state), map_id=STUDY_MAP_LAYOUT.layout_id)
    if _v6_identity(state.get("policy_id")) is not None:
        public.update(episode_id=env.state.episode_id,
                      **_v6_public_history(env.state))
    collision = next((event for event in state["events"] if event["type"] == "collision"), None)
    if collision and "collision_animation" in collision:
        public["collision_animation"] = deepcopy(collision["collision_animation"])
    return public


def rules(language="en"):
    return legacy.rules(language)


def facts(state, decision=None):
    env = _restore(state)
    result = [
        {"id": "time", "en": f"Step {state['turn']} of 120.", "zh": f"已用{state['turn']}步，共120步。"},
        {"id": "score", "en": f"Raw team score is {env.state.user_score:g}.",
         "zh": f"团队原始分为{env.state.user_score:g}。"},
    ]
    for actor, agent_id in _ACTORS.items():
        agent = env.state.by_id(agent_id)
        result.append({"id": actor + "_state",
            "en": f"{actor} is at column {agent.position[1]}, row {agent.position[0]}, with {agent.battery:g} battery and carrying {agent.carrying_task_id or 'nothing'}.",
            "zh": f"{actor}位于第{agent.position[1]}列、第{agent.position[0]}行，电量{agent.battery:g}，携带{agent.carrying_task_id or '无'}。"})
    for order in legacy._orders(env):
        result.append({"id": "order_" + order["id"],
            "en": f"Shared order {order['id']}: pickup {order['pickup']}, delivery {order['dropoff']}, status {order['status']}.",
            "zh": f"共享订单{order['id']}：取货位置{order['pickup']}，送货位置{order['dropoff']}，状态{order['status']}。"})
    result.extend({"id": f"last_event_{index}", "en": event["en"], "zh": event["zh"]}
                  for index, event in enumerate(state["events"]))
    result.extend({"id": f"public_rule_{index}", "en": en, "zh": zh}
                  for index, (en, zh) in enumerate(zip(rules(), rules("zh"))))
    if not state["terminal"]:
        actual = decide(state)
        if decision is not None and decision != actual:
            raise ValueError("decision does not match this pre-action state")
        result.append({"id": "actual_decision", "en": actual["reason_en"], "zh": actual["reason_zh"]})
        result.extend({"id": "alternative_" + option["action"],
                       "en": option["en"], "zh": option["zh"]}
                      for option in actual["alternatives"])
    return result


def audited_program_evidence(state):
    """Read-only local pattern evidence; the program never selects the AI action."""
    policy_id = state.get("policy_id")
    v6 = _v6_identity(policy_id)
    if v6 is not None and v6[0] == "regularized":
        return _audited_v6_program_evidence(state)
    if (policy_id not in ("warehouse-regularized-v5",
                          "warehouse-structured-regularized-v54") or
            state.get("terminal")):
        return None
    inventory = (structured_policy_artifacts()["regularized"]
                 if policy_id in _STRUCTURED_POLICY_IDS else
                 policy_artifacts()["regularized"])
    if not inventory["readiness"]:
        return None
    env = _restore(state)
    report = json.loads((ROOT / inventory["training_report_path"]).read_text())
    audit = (report.get("selected_program_audit") or {}).get("robot_2") or {}
    receipt = json.loads((ROOT / inventory["acceptance_path"]).read_text())
    depth, leaves, count = (audit.get("program_depth"),
                            audit.get("program_leaf_count"), audit.get("sample_count"))
    if (report.get("artifact", {}).get("sha256") != inventory["actor_sha256"]
            or report.get("selected_role_sha256", {}).get("robot_2") !=
            inventory["role_sha256"]["ai"]
            or receipt.get("final_program_audit", {}).get("robot_2") != audit
            or audit.get("audit_source") != "final_independent_development_rollouts"
            or audit.get("development_seeds") != list(range(240100, 240124))
            or audit.get("actor_role_sha256") != inventory["role_sha256"]["ai"]
            or audit.get("program_sha256") != inventory["program_sha256"].get("robot_2")
            or type(depth) is not int or not 0 <= depth <= 5
            or type(leaves) is not int or not 1 <= leaves <= 16
            or type(count) is not int or count < 1000
            or audit.get("explanation_eligible") is not True):
        return None
    measured = (audit.get("validation") or {}).get("explanation") or {}
    try:
        sample_count = int(audit["sample_count"])
        rows = int(measured["rows"])
        fidelity = float(measured["agreement"])
        coverage = float(measured["coverage"])
        kl = float(measured["kl"])
        if (not 0 < rows <= sample_count or abs(coverage - rows / sample_count) > 1e-6
                or not .20 <= coverage <= 1.0 or not .90 <= fidelity <= 1.0
                or not 0.0 <= kl <= .35
                or measured["illegal_count"] != 0):
            return None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    path = ROOT / inventory["program_paths"][1]
    program_sha = inventory["program_sha256"].get("robot_2")
    if not program_sha or sha256(path.read_bytes()).hexdigest() != program_sha:
        return None
    program_data = json.loads(path.read_text())
    if (not isinstance(program_data, dict) or
            (_config_for_policy(policy_id)["model_version"] == STRUCTURED_MODEL_VERSION and
             program_data.get("version") != SPARSE_PROGRAM_VERSION)):
        return None
    program_type = (SparseObliqueProgram if program_data.get("version") ==
                    SPARSE_PROGRAM_VERSION else AxisLeafProgram)
    program = program_type.load_json(
        path, expected_sha256=program_sha,
        actor_role_sha256=inventory["role_sha256"]["ai"])
    if program.role != "robot_2":
        return None
    obs = observation(env.state, "robot_2")
    mask = action_masks(env.state, env.config.map_layout_id)[1]
    selected, _, _ = _selection(env, state["scenario_seed"], state["policy_id"])
    evidence = program.covered({"obs": obs, "mask": mask}, actor_action=selected,
                               purpose="explanation")
    if evidence is None:
        return None
    return {
        "en": (f"The audited bounded program also selected {selected.lower()} here. "
               f"On eligible leaves, its independent development action agreement with the frozen AI "
               f"was {fidelity:.1%} across {coverage:.1%} of states. "
               "This describes an observed policy pattern, not a unique cause of the action."),
        "zh": (f"经审计的有界程序在此状态也选择了{selected.lower()}。"
               f"在合格叶节点中，它在独立开发样本上与冻结AI的动作一致率为{fidelity:.1%}，覆盖{coverage:.1%}的状态。"
               "这描述的是观测到的策略模式，不能证明该动作的唯一原因。"),
        "program_sha256": program_sha, "eligible": True, "path": evidence["path"],
        "leaf_id": evidence["leaf_id"],
        "nn_action": selected.lower(),
    }


def _audited_v6_program_evidence(state):
    """Explain only an independently audited, current-Actor-matching leaf."""
    if state.get("terminal"):
        return None
    from experiments.warehouse.protocol import final_audit_seeds
    from study_v3.release_catalog import ReleaseAssetUnavailable, warehouse_release_spec

    release_id = state.get("release_id")
    try:
        spec = warehouse_release_spec(release_id)
        episode_ids = [str(seed) for seed in final_audit_seeds(spec.release_id)]
        if state.get("policy_id") != spec.policy_ids["regularized"]:
            return None
        entry = policy_artifacts_v6(spec.release_id)["regularized"]
    except (KeyError, OSError, ValueError, ReleaseAssetUnavailable):
        return None
    if not entry["readiness"]:
        return None
    try:
        env = _restore(state)
        receipt = json.loads((ROOT / entry["acceptance_path"]).read_text())
        audit = (receipt.get("final_program_audit") or {}).get("robot_2") or {}
        report = json.loads((ROOT / entry["training_report_path"]).read_text())
        program_path = ROOT / entry["program_paths"][1]
        program_sha = entry["program_sha256"]["robot_2"]
        program = WarehouseV6Program.load(
            program_path, expected_sha256=program_sha,
            actor_role_sha256=entry["role_sha256"]["ai"])
        metric = audit["explanation"]
        if (receipt.get("passed") is not True
                or receipt.get("heldout_evaluated") is not True
                or receipt.get("release_id") != spec.release_id
                or receipt.get("artifact_sha256", {}).get("regularized") != entry["actor_sha256"]
                or receipt.get("role_sha256", {}).get("regularized", {}).get("robot_2") !=
                   entry["role_sha256"]["ai"]
                or receipt.get("program_sha256", {}).get("regularized", {}).get("robot_2") !=
                   program_sha
                or report.get("selected_role_sha256", {}).get("robot_2") !=
                   entry["role_sha256"]["ai"]
                or audit.get("actor_role_sha256") != entry["role_sha256"]["ai"]
                or audit.get("program_sha256") != program_sha
                or audit.get("feature_schema_sha256") != V6_FEATURE_SCHEMA_SHA256
                or audit.get("episode_ids") != episode_ids
                or audit.get("sample_count", 0) < 1000
                or audit.get("explanation_eligible") is not True
                or type(audit.get("max_depth")) is not int
                or not 0 <= audit["max_depth"] <= 5
                or type(audit.get("leaf_count")) is not int
                or not 1 <= audit["leaf_count"] <= 16
                or type(audit.get("max_features_per_node")) is not int
                or not 1 <= audit["max_features_per_node"] <= 3
                or metric.get("coverage", 0) < 0.20
                or metric.get("actionable_coverage", 0) < 0.20
                or metric.get("agreement", 0) < 0.90
                or metric.get("kl") is None or metric["kl"] > 0.35
                or metric.get("illegal_count") != 0):
            return None
        obs = v6_role_observation(env.state, "robot_2")
        mask = action_masks(env.state, env.config.map_layout_id)[1]
        selected, _, _ = _selection(env, state["scenario_seed"], state["policy_id"])
        actor = _actor(state["policy_id"])
        if (actor.artifact_sha256 != entry["actor_sha256"]
                or actor.role_sha256["robot_2"] != entry["role_sha256"]["ai"]):
            return None
        evidence = program.covered({"obs": obs, "mask": mask},
                                   actor_action=selected, purpose="explanation",
                                   actor_role=actor.roles["robot_2"])
        if evidence is None:
            return None
        agreement = float(metric["agreement"])
        actionable = float(metric["actionable_coverage"])
        return {
            "en": (f"The independently audited program also selected {selected.lower()} here. "
                   f"Across eligible audit states it agreed with this frozen AI {agreement:.1%} "
                   f"of the time; its action matched the sampled NN action on "
                   f"{actionable:.1%} of all audit states. "
                   "This pattern does not establish a unique motive."),
            "zh": (f"独立审计的程序在此状态也选择了{selected.lower()}。"
                   f"合格审计状态上的动作一致率为{agreement:.1%}；"
                   f"在全部审计状态中，程序动作与采样NN动作一致的覆盖率为{actionable:.1%}。"
                   "这种模式不能证明唯一动机。"),
            "program_sha256": program_sha, "eligible": True,
            "path": evidence["path"], "leaf_id": evidence["leaf_id"],
            "nn_action": selected.lower(),
        }
    except (KeyError, IndexError, OSError, TypeError, ValueError, RuntimeError):
        return None


def human_advisor(state):
    """Suggest the bound human-role Actor's action in a formal v5 task."""
    if state["terminal"]:
        return "wait"
    env = _restore(state)
    if (state.get("policy_id") in _V5_POLICY_IDS or
            _v6_identity(state.get("policy_id")) is not None or
            state.get("policy_id") in _STRUCTURED_POLICY_IDS):
        _, actions, _ = _selection(env, state["scenario_seed"], state["policy_id"])
        suggested = actions["robot_1"]
        if action_masks(env.state, env.config.map_layout_id)[0, ACTIONS.index(suggested)] <= 0.5:
            raise RuntimeError("Warehouse human-role NN selected a physically illegal action")
        return suggested.lower()
    # Historical v4 evidence retains its original advisory calculation.
    ai_action, _, _ = _selection(env, state["scenario_seed"], state.get("policy_id"))
    human = env.state.by_id("robot_1")
    task = next((item for item in env.state.tasks if item.task_id == human.carrying_task_id), None)
    destinations = ([task.delivery_position] if task else
                    [item.pickup_position for item in env.state.tasks if item.status == "available"])
    candidates = []
    for index, action in enumerate(ACTIONS):
        if action_masks(env.state, env.config.map_layout_id)[0, index] <= 0.5:
            continue
        branch = _restore(state)
        branch.step({"robot_1": action, "robot_2": ai_action})
        after = branch.state.by_id("robot_1")
        distance = min((shortest_path_distance(after.position, goal, branch.config.map_layout_id)
                        for goal in destinations), default=0)
        candidates.append((branch.state.robot_collision_events - env.state.robot_collision_events,
                           -(branch.state.total_deliveries - env.state.total_deliveries),
                           -branch.state.user_score, distance, index, action))
    return min(candidates)[-1].lower()


def demonstration(policy_id=None, release_id: str | None = None):
    """A real AI–AI rollout from the same frozen policy used in human tasks."""
    policy_id = _policy_id(policy_id)
    v6 = _v6_identity(policy_id)
    if v6 is not None and release_id not in (None, v6[1]):
        raise RuntimeError("Warehouse v6 demonstration release_id mismatch")
    if v6 is None and isinstance(release_id, str) and release_id.startswith(
            V6_RELEASE_ID.rsplit(".", 1)[0] + "."):
        raise RuntimeError("Warehouse v6 demonstration requires a v6 policy_id")
    _verify_v5_artifact(policy_id)
    actor = _actor(policy_id)
    return _demonstration_cached(policy_id, actor.artifact_sha256)


@lru_cache(maxsize=4)
def _demonstration_cached(policy_id, actor_sha256):
    actor = _actor(policy_id)
    if actor.artifact_sha256 != actor_sha256:
        raise RuntimeError("Warehouse demonstration Actor binding changed")
    seed = _config_for_policy(policy_id)["demo_seed"]
    env = WarehouseMultiAgentEnv(participant_study_config())
    if _v6_identity(policy_id) is not None:
        validate_study_action_physics(env.config)
    env.reset(seed=seed)
    if policy_id != LEGACY_POLICY_ID:
        marked = env.get_state()
        marked.participant_controlled_agent_id = "robot_1"
        env.set_state(marked)
    frames = [public_state(_state(env, 0, seed, policy_id=policy_id))]
    frames[0]["demo_actions"] = None
    charges = 0
    while not (env.state.terminated or env.state.truncated):
        before = env.get_state()
        actions, distributions = actor.act(env.state, base_seed=seed)
        _, _, _, _, info = env.step(actions, decision_metadata=distribution_decision_metadata(
            distributions, decision_source="two_pure_neural_actors",
            policy_actions=actions, selected_actions=actions,
            runtime_decision={"mode": "pure_neural", "selection_changed_policy": False}))
        if info["proposed_actions"] != actions or info["action_override_agents"]:
            raise RuntimeError("Warehouse AI–AI neural actions were rewritten before physics")
        events = legacy._events(before, env.state, info)
        charges += sum(event["type"] == "charge" for event in events)
        frame = public_state(_state(env, 0, seed, events,
                          {"human": actions["robot_1"].lower(), "ai": actions["robot_2"].lower()}, charges,
                          policy_id=policy_id))
        frame["demo_actions"] = {"human": actions["robot_1"].lower(),
                                 "ai": actions["robot_2"].lower()}
        frames.append(frame)
    end = len(frames) - 1
    captions = [
        {"index": 0, "en": "Both robots in this demonstration use frozen neural actors. In the tasks, you replace the blue robot.",
         "zh": "演示中的两个机器人使用冻结的神经网络；在任务中，你将接替蓝色机器人。"},
        {"index": end, "en": "The AI–AI demonstration has ended. Its actions and outcomes were recorded from the real environment.",
         "zh": "双机器人演示已结束；动作和结果均来自真实环境运行。"},
    ]
    return {"frames": frames, "captions": captions,
            "provenance": {"kind": "pure_neural_ai_ai_rollout", "seed": seed,
                           "policy_id": policy_id, "actor_sha256": actor.artifact_sha256,
                           **({"release_id": _v6_identity(policy_id)[1],
                               "feature_schema_sha256": V6_FEATURE_SCHEMA_SHA256,
                               "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256}
                              if _v6_identity(policy_id) is not None else {}),
                           "role_sha256": ({"human": actor.role_sha256["robot_1"], "ai": actor.role_sha256["robot_2"]}
                                           if policy_id != LEGACY_POLICY_ID else {}),
                           "frame_count": len(frames)}}


demonstration.cache_clear = _demonstration_cached.cache_clear


def comprehension(language="en"):
    zh = language.startswith("zh")
    return [
        {"id": "warehouse_prediction", "text": "机器人在充电格等待，电量40。没有碰撞时，本步后电量是多少？" if zh else "A robot waits on the charger with 40 battery. There is no collision. What is its battery after the step?", "options": ["50", "40", "38"], "answer": 0},
        {"id": "warehouse_controller", "text": "任务中的AI动作由什么决定？" if zh else "What selects the AI's action during the tasks?", "options": ["冻结的神经网络" if zh else "The frozen neural policy", "手写避让规则" if zh else "A hand-written avoidance rule", "解释程序" if zh else "The explanation program"], "answer": 0},
        {"id": "warehouse_change", "text": "订单送达后会怎样？" if zh else "What happens after an order is delivered?", "options": ["补充一个共享订单" if zh else "A replacement shared order appears", "不再有新订单" if zh else "No new order appears", "原包裹留在手上" if zh else "The old parcel remains carried"], "answer": 0},
    ]


def screening_report(policy_id=None):
    results = {}
    for task in (1, 2, 3):
        state = initial_state(100, task, policy_id=policy_id)
        while not state["terminal"]:
            state = step(state, human_advisor(state))
        results[str(task)] = score(state)
    config = _config_for_policy(_policy_id(policy_id))
    result = {"domain": DOMAIN, "version": config["version"], "not_human_results": True,
              "task_seeds": {str(task): _task_seed(100, task, _policy_id(policy_id))
                             for task in (1, 2, 3)}, "tasks": results}
    if policy_id is not None:
        result["policy_id"] = _policy_id(policy_id)
        result["task_seed_rule"] = "enrollment_seed_times_10_plus_task"
    else:
        result["fixed_task_seeds"] = config["task_seeds"]
    return result
