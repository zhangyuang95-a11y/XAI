"""Local, same-weight control ablations; never a production release gate."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .autonomy import AutonomyWarehouseEnv, static_mask_observations
from .decision_protocol import canonical_sha256, distribution_decision_metadata
from .domain import AgentState, DeliveryTask, WarehouseState, collaborative_study_config
from .environment import WarehouseMultiAgentEnv
from .navigation import ACTIONS
from .numpy_policy import NumpyWarehousePolicy
from .runtime_coordination import select_ai_ai_joint_actions, select_human_ai_action


AUTONOMY_VERSION = "warehouse-autonomy-ablation.v1"
DEFAULT_ACTOR = Path(__file__).resolve().parents[2] / "output/deployment/warehouse_mappo_v68_6x7_actor.npz"
MODE_DESCRIPTIONS = {
    "guarded": ("原协调执行", "Original coordination", "保留当前全部协调规则。", "All existing coordination rules remain."),
    "actor_direct": ("仅取消输出后接管", "No post-policy selection", "Actor 输出直接执行；仍有规则掩码和内部评分。", "Direct Actor output; original masks and rule scores remain."),
    "unmasked_actor": ("再取消规则动作掩码", "Geometry masks only", "只屏蔽墙体等静态非法动作；仍有内部规则评分。", "Only physical action masks; handcrafted logit scores remain."),
    "learned_only": ("自主神经头消融", "Learned heads ablation", "去除动作重选、规则掩码、人工评分和目标指令；同权重，未重新训练。", "No action re-selection, rule masks, rule scores or goal directives; same weights, no retraining."),
}
MODES = tuple(MODE_DESCRIPTIONS)


def runtime_signature(actor: NumpyWarehousePolicy) -> str:
    """Bind persisted runs to code as well as the unchanged Actor artifact."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    paths = sorted(here.glob("*.py")) + [
        root / "ui/warehouse_autonomy_server.py", root / "ui/warehouse_view.py",
        root / "core/policy_contracts.py",
    ]
    hashes = {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    return canonical_sha256({"version": AUTONOMY_VERSION, "actor": actor.artifact_sha256, "code": hashes})


def make_environment(mode: str, seed: int, *, human: bool = True) -> WarehouseMultiAgentEnv:
    if mode not in MODES:
        raise ValueError("Unknown autonomy mode")
    cls = AutonomyWarehouseEnv if mode == "learned_only" else WarehouseMultiAgentEnv
    environment = cls(collaborative_study_config())
    environment.reset(seed=int(seed))
    # Provenance is set before the first Actor call, never from this step's input.
    environment.state.participant_controlled_agent_id = "robot_1" if human else None
    return environment


def decide(environment, actor, mode: str, *, seed: int, deterministic: bool = True):
    """No human command argument: select Robot 2 using S_t alone."""
    if mode not in MODES:
        raise ValueError("Unknown autonomy mode")
    state = environment.get_state()
    if state.terminated or state.truncated:
        raise ValueError("Round has ended")
    observations = (
        static_mask_observations(state, environment.config)
        if mode == "unmasked_actor" else environment.observations()
    )
    proposed, distributions = actor.act(
        observations, deterministic=deterministic, base_seed=seed,
        decision_key=(state.episode_id, state.frame), learned_only=mode == "learned_only",
    )
    selected = dict(proposed)
    evidence = {"selection_changed_policy": False}
    if mode == "guarded":
        if state.participant_controlled_agent_id:
            selected["robot_2"], evidence = select_human_ai_action(environment, proposed["robot_2"])
        else:
            selected, evidence = select_ai_ai_joint_actions(environment, proposed)
    metadata = distribution_decision_metadata(
        distributions, decision_source=f"{AUTONOMY_VERSION}:{mode}",
        policy_actions=proposed, selected_actions=selected, runtime_decision=evidence,
    )
    metadata.update({
        "autonomy_mode": mode, "evaluation_only": True,
        "actor_artifact_sha256": actor.artifact_sha256,
        "policy_observation_sha256": {key: canonical_sha256(value) for key, value in observations.items()},
    })
    return selected, metadata


def advance(environment, actor, mode, *, seed, action=None, deterministic=True):
    selected, metadata = decide(environment, actor, mode, seed=seed, deterministic=deterministic)
    human = environment.state.participant_controlled_agent_id
    if human:
        if action not in ACTIONS:
            raise ValueError("Unknown player action")
        selected[human] = action
        metadata["participant_overrides"] = {human: action}
        metadata["selected_actions"] = dict(selected)
    _, rewards, terminated, truncated, info = environment.step(selected, decision_metadata=metadata)
    return {
        "actions": selected, "proposed_actions": metadata["policy_actions"],
        "executed_actions": dict(info["executed_actions"]),
        "metadata": metadata, "info": info, "rewards": rewards,
        "done": bool(terminated or truncated),
    }


def pack(value: Any) -> Any:
    """Explicit tagged JSON preserves tuple types, without pickle/code loading."""
    if is_dataclass(value):
        return {"$class": type(value).__name__, "fields": {f.name: pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple):
        return {"$tuple": [pack(x) for x in value]}
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value: Any) -> Any:
    if isinstance(value, list):
        return [unpack(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {"$tuple"}:
            return tuple(unpack(x) for x in value["$tuple"])
        if set(value) == {"$class", "fields"}:
            cls = {c.__name__: c for c in (AgentState, DeliveryTask, WarehouseState)}[value["$class"]]
            return cls(**{k: unpack(v) for k, v in value["fields"].items()})
        return {k: unpack(v) for k, v in value.items()}
    return value


def snapshot(environment) -> dict:
    return pack({"state": environment.get_state(), "rng": environment.get_rng_state(), "episode_counter": environment._episode_counter})


def restore(environment, payload) -> None:
    data = unpack(payload)
    errors = environment.validate_state(data["state"])
    if errors:
        raise ValueError("Invalid saved warehouse state: " + "; ".join(errors))
    # set_state() intentionally recomputes plans for interventions. Persistence
    # must restore the exact acknowledged state instead, including its RNG.
    environment.state = data["state"]
    environment._rng.setstate(data["rng"])
    environment._episode_counter = data["episode_counter"]
