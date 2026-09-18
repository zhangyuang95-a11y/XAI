"""Episode-stable external partners for Pong PPO training and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import random
from typing import Any, Mapping

import numpy as np
import torch

from ..config import PongConfig
from ..environment.engine import PongEnvironment
from ..policies.rule_demo import RuleDemoController
from ..adapters.core_adapter import feature_vector

ACTIONS = ("left", "right", "stay")


def _role(agent_index: int) -> str:
    return "player" if agent_index == 0 else "ai"


def _action_toward(current: float, target: float) -> str:
    return "right" if target - current > 0.025 else "left" if target - current < -0.025 else "stay"


class PerturbationPartner:
    """A deterministic-in-episode human-like disturbance, never iid noise."""

    def __init__(self, config: PongConfig, agent_index: int, *, seed: int,
                 min_decisions: int = 2, max_decisions: int = 8) -> None:
        self.config = config
        self.agent_index = agent_index
        self._rng = random.Random(seed)
        self.controller = RuleDemoController(config, controlled_agent=_role(agent_index))
        self.mode = self._rng.choice(("pause_then_recover", "reaction_delay", "small_overlap", "large_hesitate"))
        self.remaining = self._rng.randint(min_decisions, max(min_decisions, max_decisions))
        self.last_action = "stay"

    def choose(self, environment: PongEnvironment) -> str:
        frame = environment.frame()
        decision = self.controller.choose(frame)
        action = decision.action
        if self.remaining > 0:
            self.remaining -= 1
            if self.mode in {"pause_then_recover", "reaction_delay"}:
                action = "stay"
            elif self.mode == "large_hesitate" and decision.requires_partner:
                action = "stay"
            elif self.mode == "small_overlap":
                candidates = [item for item in self.controller._candidates(frame)
                              if item.prediction.ball_kind == "small" and item.ai_viable]
                if candidates:
                    target = min(candidates, key=lambda item: item.prediction.time_until_contact).target_x
                    own = frame.player_x if self.agent_index == 0 else frame.ai_x
                    action = _action_toward(own, target)
        self.last_action = action
        return action

    def state_dict(self) -> dict[str, Any]:
        return {"agent_index": self.agent_index, "rng": self._rng.getstate(), "mode": self.mode,
                "remaining": self.remaining, "last_action": self.last_action,
                "controller": self.controller.state_dict()}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        if int(payload["agent_index"]) != self.agent_index:
            raise ValueError("perturbation partner role mismatch")
        self._rng.setstate(_tupleize(payload["rng"]))
        self.mode = str(payload["mode"]); self.remaining = int(payload["remaining"])
        self.last_action = str(payload.get("last_action", "stay"))
        self.controller.restore_state(dict(payload["controller"]))


class FrozenActorPartner:
    """A fixed Actor snapshot; its actions are never current-policy samples."""

    def __init__(self, model: torch.nn.Module, agent_index: int, snapshot_index: int) -> None:
        self.model = model
        self.agent_index = agent_index
        self.snapshot_index = snapshot_index

    def choose(self, environment: PongEnvironment) -> str:
        row = np.asarray(feature_vector(environment, _role(self.agent_index)), dtype=np.float32)
        with torch.no_grad():
            logits = self.model.actor_logits(torch.as_tensor(row[None, :], dtype=torch.float32))
        return ACTIONS[int(torch.argmax(logits, dim=-1).item())]

    def state_dict(self) -> dict[str, Any]:
        return {"agent_index": self.agent_index, "snapshot_index": self.snapshot_index}


@dataclass
class EpisodePartner:
    kind: str
    external_agent: int | None
    controller: RuleDemoController | PerturbationPartner | FrozenActorPartner | None = None

    def action_for(self, environment: PongEnvironment, agent_index: int, sampled: int) -> tuple[int, bool]:
        if self.external_agent != agent_index:
            return sampled, True
        if self.controller is None:
            raise RuntimeError("external partner has no controller")
        return ACTIONS.index(self.controller.choose(environment.frame()).action if isinstance(self.controller, RuleDemoController)
                             else self.controller.choose(environment)), False

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "external_agent": self.external_agent,
                "controller": self.controller.state_dict() if self.controller else None}


class PartnerManager:
    """Draw a partner once per episode and preserve its state across decisions."""

    def __init__(self, config: PongConfig, settings: Mapping[str, object], count: int, *, seed: int,
                 model: torch.nn.Module | None = None) -> None:
        self.config = config
        self.settings = dict(settings)
        self._rng = random.Random(seed)
        self.frozen_models = ([deepcopy(model).cpu().eval()] if model is not None and
                              float(self.settings.get("frozen_partner_fraction", 0.0)) > 0 else [])
        self.plans: list[EpisodePartner] = [self._new_plan(index) for index in range(count)]
        self.episode_counts: dict[str, int] = {"self_play": 0, "frozen": 0, "rule": 0, "perturb": 0}
        self.joint_steps: dict[str, int] = {"self_play": 0, "frozen": 0, "rule": 0, "perturb": 0}
        self.neural_samples: dict[str, int] = {"self_play": 0, "frozen": 0, "rule": 0, "perturb": 0}
        for plan in self.plans:
            self.episode_counts[plan.kind] += 1

    def _new_plan(self, index: int) -> EpisodePartner:
        self_play = float(self.settings.get("self_play_fraction", .70))
        frozen = float(self.settings.get("frozen_partner_fraction", 0.0))
        rule = float(self.settings.get("rule_partner_fraction", .20))
        perturb = float(self.settings.get("perturb_partner_fraction", self.settings.get("hesitant_partner_fraction", .10)))
        if min(self_play, frozen, rule, perturb) < 0 or abs(self_play + frozen + rule + perturb - 1.0) > 1e-6:
            raise ValueError("partner fractions must be non-negative and sum to 1")
        roll = self._rng.random()
        if roll < self_play:
            return EpisodePartner("self_play", None)
        external_agent = self._rng.randrange(2) if frozen else index % 2
        if roll < self_play + frozen:
            if not self.frozen_models:
                raise ValueError("frozen partner requires an Actor snapshot")
            snapshot_index = self._rng.randrange(len(self.frozen_models))
            return EpisodePartner("frozen", external_agent,
                                  FrozenActorPartner(self.frozen_models[snapshot_index], external_agent, snapshot_index))
        if roll < self_play + frozen + rule:
            return EpisodePartner("rule", external_agent,
                                  RuleDemoController(self.config, controlled_agent=_role(external_agent)))
        return EpisodePartner("perturb", external_agent,
                              PerturbationPartner(self.config, external_agent,
                                                  seed=self._rng.randrange(2**31),
                                                  min_decisions=int(self.settings.get("perturb_min_decisions", 2)),
                                                  max_decisions=int(self.settings.get("perturb_max_decisions", 8))))

    def set_initial_actor(self, model: torch.nn.Module) -> None:
        if self.frozen_models:
            self.frozen_models[0].load_state_dict(deepcopy(model).cpu().state_dict())

    def add_actor_snapshot(self, model: torch.nn.Module, *, maximum: int = 4) -> None:
        self.frozen_models.append(deepcopy(model).cpu().eval())
        # Retain old snapshots while any episode may still refer to them.  A
        # checkpoint must resolve every recorded snapshot index exactly.

    def action_pair(self, environment: PongEnvironment, environment_index: int,
                    sampled_player: int, sampled_ai: int) -> tuple[int, int, bool, bool]:
        plan = self.plans[environment_index]
        player, player_neural = plan.action_for(environment, 0, sampled_player)
        ai, ai_neural = plan.action_for(environment, 1, sampled_ai)
        self.joint_steps[plan.kind] += 1
        self.neural_samples[plan.kind] += int(player_neural) + int(ai_neural)
        return player, ai, player_neural, ai_neural

    def reset_episode(self, environment_index: int) -> EpisodePartner:
        plan = self._new_plan(environment_index)
        self.plans[environment_index] = plan
        self.episode_counts[plan.kind] += 1
        return plan

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate(), "plans": [plan.state_dict() for plan in self.plans],
                "frozen_models": [model.state_dict() for model in self.frozen_models],
                "episode_counts": dict(self.episode_counts), "joint_steps": dict(self.joint_steps),
                "neural_samples": dict(self.neural_samples)}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        self._rng.setstate(_tupleize(payload["rng"]))
        stored_models = list(payload.get("frozen_models", ()))
        if stored_models and not self.frozen_models:
            raise ValueError("checkpoint has a frozen Actor but the configuration has no network")
        self.frozen_models = [deepcopy(self.frozen_models[0]).cpu().eval() for _ in stored_models]
        for model, parameters in zip(self.frozen_models, stored_models):
            model.load_state_dict(parameters)
        restored: list[EpisodePartner] = []
        for item in payload["plans"]:
            agent = item["external_agent"]
            kind = str(item["kind"])
            controller: RuleDemoController | PerturbationPartner | None = None
            if agent is not None and kind == "rule":
                controller = RuleDemoController(self.config, controlled_agent=_role(int(agent)))
                controller.restore_state(dict(item["controller"]))
            elif agent is not None and kind == "perturb":
                controller = PerturbationPartner(self.config, int(agent), seed=0)
                controller.restore_state(dict(item["controller"]))
            elif agent is not None and kind == "frozen":
                snapshot_index = int(item["controller"]["snapshot_index"])
                controller = FrozenActorPartner(self.frozen_models[snapshot_index], int(agent), snapshot_index)
            restored.append(EpisodePartner(kind, None if agent is None else int(agent), controller))
        if len(restored) != len(self.plans):
            raise ValueError("partner plan count mismatch")
        self.plans = restored
        self.episode_counts = {key: int(value) for key, value in payload["episode_counts"].items()}
        self.joint_steps = {key: int(value) for key, value in payload["joint_steps"].items()}
        self.neural_samples = {key: int(value) for key, value in payload["neural_samples"].items()}


def _tupleize(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value
