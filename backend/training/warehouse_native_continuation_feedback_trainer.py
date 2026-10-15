"""RCPD soft Actor feedback around an unchanged, genuine continuation trainer.

No budget or capability is granted here. Extraction is external, from real
197-column neural trajectories. Control and zero-lambda updates call the exact
native update; a program never supplies an action, label or action mask.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import warehouse_native_continuation_trainer as native
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import ppo_actor_objective
from .warehouse_native_partner_mix_trainer import _sha
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import COUNTERS, _cpu
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager

VERSION = "warehouse-native-continuation-feedback-trainer.v1"
PROTOCOL_VERSION = "warehouse-native-continuation-feedback-protocol.v1"
BRANCHES = ("control", "feedback")


def execution_sources():
    result = native.execution_sources()
    paths = [Path(__file__), ROOT / "env/warehouse_native/feedback.py"]
    paths += sorted((ROOT / "core").glob("*.py"))
    for path in paths: result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _config(value, fixture):
    baseline = FeedbackConfig()
    if value is None: return baseline
    if isinstance(value, FeedbackConfig): result = value
    elif isinstance(value, dict):
        values = copy.deepcopy(value)
        for key in ("depths", "leaves", "critical_groups"):
            if key in values: values[key] = tuple(values[key])
        result = FeedbackConfig(**values)
    else: raise ValueError("A complete declared FeedbackConfig is required")
    if not fixture and result != baseline:
        raise ValueError("Production soft-feedback thresholds and schedule must retain the declared defaults")
    if fixture:
        for key in ("lambda_max", "minimum_fidelity", "minimum_critical_fidelity", "maximum_mean_kl",
                    "maximum_performance_drop_fraction", "complexity_weight", "critical_groups"):
            if getattr(result, key) != getattr(baseline, key):
                raise ValueError("Fixture computation limits cannot weaken feedback gates")
    return result


def _retain_schedule(current, incoming):
    """A refitted tree must retain this training run's ramp and performance anchor."""
    for key in ("ramp_start", "performance_anchor_ratio", "reference_score"):
        setattr(incoming, key, copy.deepcopy(getattr(current, key)))
    return incoming


def _validate_refresh_state(manager, capability, pending, failure):
    if type(pending) is not bool or (failure is not None and (not pending or not isinstance(failure, dict))):
        raise ValueError("Invalid feedback refresh state")
    if failure is not None: digest(failure)
    if pending and (manager is None or manager.current_lambda != 0 or manager.reliable
            or manager.last_gate.get("reason") != "refresh_incomplete" or capability is None
            or capability.get("capability_eligible") is not False
            or capability.get("gate") != manager.last_gate or capability.get("step") != manager.last_step):
        raise ValueError("Pending or failed refresh must preserve a bound closed gate")


def make_protocol(source_trainer, *, source_checkpoint_sha256, source_state_sha256,
                  ppo_cap, cycle_id, evaluation_checkpoints=None, feedback_config=None, test_fixture=False):
    base = native.make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=evaluation_checkpoints, test_fixture=test_fixture)
    config = _config(feedback_config, test_fixture)
    return {**copy.deepcopy(base), "version": PROTOCOL_VERSION, "native_protocol": base,
        "feedback_training_enabled": True, "feedback_branches": list(BRANCHES),
        "feedback_config": asdict(config),
        "feedback": {"target": "detached_tree_distribution", "direction": "KL(nn||tree)",
            "complexity": "candidate_selection_only", "extract_interval": 50000,
            "schedule_step": "cumulative_joint_steps", "source_capability_required": True,
            "action_controller": "neural_actor_only"},
        "explanation_qualification_granted": False}


class ContinuationFeedbackTrainer:
    """Own envelope; its native checkpoint retains the unmodified native schema."""

    def __init__(self, protocol, source_trainer, *, feedback_branch="control",
                 expected_source_state_sha256, source_checkpoint_sha256, device="cpu", test_fixture=False):
        if type(feedback_branch) is not str or feedback_branch not in BRANCHES:
            raise ValueError("Unknown feedback condition")
        if not isinstance(protocol, dict): raise ValueError("A registered feedback protocol is required")
        expected = make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
            source_state_sha256=expected_source_state_sha256,
            ppo_cap=protocol.get("budget", {}).get("maximum_ppo_joint_steps_per_arm"),
            cycle_id=protocol.get("cycle_id"),
            evaluation_checkpoints=protocol.get("evaluation", {}).get("checkpoints_ppo_steps"),
            feedback_config=protocol.get("feedback_config"), test_fixture=test_fixture)
        # Tuples in the immutable dataclass become JSON lists in saved protocols.
        if digest(protocol) != digest(expected): raise ValueError("Feedback protocol or inherited learning contract differs")
        self.protocol = copy.deepcopy(expected)
        self.native = native.ContinuationTrainer(self.protocol["native_protocol"], source_trainer,
            expected_source_state_sha256=expected_source_state_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256, device=device, test_fixture=test_fixture)
        self.feedback_branch, self.feedback_enabled = feedback_branch, feedback_branch == "feedback"
        self.sources = execution_sources()
        self.feedback = FeedbackManager(self.native.envs[0].feature_names,
            _config(self.protocol["feedback_config"], test_fixture)) if self.feedback_enabled else None
        self.feedback_evidence = None
        self.capability_evidence = None
        self.refresh_pending, self.refresh_failure = False, None

    def __getattr__(self, name):
        inner = self.__dict__.get("native")
        if inner is not None: return getattr(inner, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in (*COUNTERS, "elapsed_seconds") and "native" in self.__dict__:
            setattr(self.native, name, value)
        else: object.__setattr__(self, name, value)

    def collect(self, time_steps):
        batch = self.native.collect(time_steps)
        for row in batch["transition_records"]:
            row.update(version=VERSION, feedback_branch=self.feedback_branch,
                feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.)
        return batch

    @property
    def feedback_clock(self):
        return self.native.source_counters["joint_steps"] + self.native.joint_steps

    def begin_feedback_refresh(self):
        """Close feedback before external work; retain its ramp and regression anchor."""
        if not self.feedback_enabled: raise ValueError("Control has no feedback refresh")
        old_report = (self.capability_evidence or {}).get("report_sha256")
        self.feedback.current_lambda, self.feedback.reliable = 0., False
        self.feedback.last_step = self.feedback_clock
        self.feedback.last_gate = {"step": self.feedback_clock, "active": False,
            "reason": "refresh_incomplete", "lambda": 0., "reliable_program": False}
        self.capability_evidence = {"report_sha256": old_report, "capability_eligible": False,
            "step": self.feedback_clock, "gate": copy.deepcopy(self.feedback.last_gate)}
        self.refresh_pending, self.refresh_failure = True, None
        return self.feedback.state_dict()

    def fail_feedback_refresh(self, report):
        """Keep the externally persisted failure record attached; never reuse an old gate."""
        if not self.feedback_enabled or not self.refresh_pending:
            raise ValueError("No pending feedback refresh")
        if not isinstance(report, dict): raise ValueError("A raw extraction failure report is required")
        digest(report)
        self.refresh_failure = copy.deepcopy(report)

    def _feedback_snapshot(self):
        return {"feedback_state": self.feedback.state_dict() if self.feedback_enabled else None,
            "feedback_evidence": copy.deepcopy(self.feedback_evidence),
            "capability_evidence": copy.deepcopy(self.capability_evidence),
            "refresh_pending": self.refresh_pending, "refresh_failure": copy.deepcopy(self.refresh_failure)}

    def attach_feedback_state(self, manager_state, *, source_actor_sha256,
                              source_actor_parameters_sha256, evidence_sha256):
        """Install externally extracted same-Actor evidence; this never opens its gate."""
        if not self.feedback_enabled: raise ValueError("Control cannot install feedback")
        for name, value in (("source Actor", source_actor_sha256), ("parameters", source_actor_parameters_sha256),
                            ("extraction evidence", evidence_sha256)): _sha(value, name)
        if initialization_sha256(self.native.model.actor.state_dict()) != source_actor_parameters_sha256:
            raise ValueError("Extraction labels belong to a different frozen neural Actor")
        manager = FeedbackManager(self.native.envs[0].feature_names, self.feedback.config)
        manager.load_state_dict(manager_state)
        fit = manager.last_fit_report
        if (fit.get("source_actor_sha256") != source_actor_sha256 or fit.get("step") != self.feedback_clock
                or manager.last_fit_step != self.feedback_clock or manager.current_lambda != 0
                or manager.last_step != self.feedback_clock or manager.program is None
                or tuple(manager.program.feature_names) != tuple(self.native.envs[0].feature_names)
                or manager.program.metadata.get("native_source_actor_sha256") != source_actor_sha256
                or fit.get("episode_overlap") != 0 or fit.get("exact_observation_overlap") != 0):
            raise ValueError("Extraction state lacks its current same-Actor and disjoint-data binding")
        manager = _retain_schedule(self.feedback, manager)
        self.feedback, self.feedback_evidence = manager, {"source_actor_sha256": source_actor_sha256,
            "source_actor_parameters_sha256": source_actor_parameters_sha256, "evidence_sha256": evidence_sha256,
            "fit_step": self.feedback_clock, "fit_additional_joint_steps": self.joint_steps, "manager_fit_sha256": digest(fit)}
        self.capability_evidence = None
        self.refresh_pending, self.refresh_failure = False, None

    def update_feedback_gate(self, validation_score, *, capability_eligible=False, reference_score=None,
                             capability_report_sha256=None):
        if not self.feedback_enabled: return {"active": False, "lambda": 0., "reason": "control"}
        if self.refresh_pending: raise ValueError("Feedback refresh is incomplete; its gate must remain closed")
        if type(capability_eligible) is not bool: raise ValueError("Capability result must be explicit")
        if capability_eligible: _sha(capability_report_sha256, "verified capability report")
        elif capability_report_sha256 is not None: _sha(capability_report_sha256, "capability report")
        result = self.feedback.update(self.feedback_clock, validation_score,
            capability_eligible=capability_eligible, reference_score=reference_score)
        self.capability_evidence = {"report_sha256": capability_report_sha256,
            "capability_eligible": capability_eligible, "step": self.feedback_clock, "gate": copy.deepcopy(result)}
        return result

    def update(self, batch):
        if not self.feedback_enabled or self.feedback.current_lambda <= 0:
            return {**self.native.update(batch), "feedback_loss": 0., "feedback_gradient_norm": 0.,
                "feedback_rows": 0., "feedback_kl": 0., "feedback_lambda": 0.}
        with self.native._rng_context(): result = self._feedback_update(batch)
        for role in ("actor", "critic"):
            current = int(getattr(self.native.optimizers, role).state_dict()["state"][0]["step"].item())
            setattr(self.native, role + "_optimizer_steps", current - self.native.source_counters[role + "_optimizer_steps"])
        return result

    def _feedback_update(self, batch):
        """The original PPO objective plus one already-weighted NN-only KL term."""
        obs = batch["observations"].reshape(-1, self.model.obs_dim)
        states = np.repeat(batch["states"][:, :, None, :], 2, axis=2).reshape(-1, self.model.state_dim)
        roles = np.tile([0, 1], len(obs) // 2); trainable = batch["trainable"].reshape(-1)
        advantages = batch["advantages"].reshape(-1).copy(); selected = trainable > 0
        if selected.any():
            advantages[selected] = (advantages[selected] - advantages[selected].mean()) / (advantages[selected].std() + 1e-8)
        data = {"obs": obs, "states": states, "roles": roles, "trainable": trainable,
            "actions": batch["actions"].reshape(-1), "old": batch["old_log_probs"].reshape(-1),
            "advantages": advantages, "returns": batch["returns"].reshape(-1)}
        data = {key: torch.as_tensor(value, device=self.device) for key, value in data.items()}; metrics = []
        for _ in range(self.cfg["epochs"]):
            permutation = self.rng.permutation(len(obs))
            for start in range(0, len(obs), self.cfg["minibatch"]):
                index = torch.as_tensor(permutation[start:start + self.cfg["minibatch"]], device=self.device)
                logits = self.model.actor_logits(data["obs"][index])
                actor_loss, part = ppo_actor_objective(logits, data["actions"][index], data["old"][index],
                    data["advantages"][index], data["trainable"][index], clip=self.cfg["clip"], entropy=self.cfg["entropy"])
                critic_loss = .5 * (self.model.values(data["states"][index], data["roles"][index]) - data["returns"][index]).square().mean()
                mask = data["trainable"][index].bool(); feedback_loss = logits.sum() * 0
                extra = {"feedback_loss": 0., "feedback_gradient_norm": 0., "feedback_rows": int(mask.sum()), "feedback_kl": 0.}
                if mask.any():
                    feedback_loss, evidence = self.feedback.loss(logits[mask], data["obs"][index][mask].detach().cpu().numpy())
                    gradients = torch.autograd.grad(feedback_loss, tuple(self.model.actor.parameters()), retain_graph=True, allow_unused=True)
                    norm = torch.sqrt(sum((g.square().sum() for g in gradients if g is not None), logits.new_zeros(())))
                    extra.update(feedback_loss=float(feedback_loss.detach()), feedback_gradient_norm=float(norm.detach()), feedback_kl=evidence.get("kl", 0.))
                result = self.optimizers.step(actor_loss + feedback_loss, critic_loss, train_actor=bool(mask.any()))
                metrics.append({**part, **result, **extra, "critic_loss": float(critic_loss.detach()),
                    "feedback_lambda": self.feedback.current_lambda})
                self.native.minibatch_updates += 1
        self.native.optimizer_updates += 1
        return {key: float(np.mean([row.get(key, 0.) for row in metrics])) for key in set().union(*(row.keys() for row in metrics))}

    def train_chunk(self, time_steps):
        batch = self.collect(time_steps)
        return batch, self.update(batch)

    def _bindings(self):
        return {"version": VERSION, "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "feedback_branch": self.feedback_branch, "feedback_enabled": self.feedback_enabled,
            "execution_sources": copy.deepcopy(self.sources), "source_sha256": digest(self.sources),
            "test_fixture": self.test_fixture}

    def state_dict(self):
        if execution_sources() != self.sources: raise ValueError("Feedback execution sources changed")
        return {**self._bindings(), "native_state": self.native.state_dict(),
            **self._feedback_snapshot()}

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid feedback checkpoint or changed execution sources")
        if any(digest(payload.get(key)) != digest(value) for key, value in self._bindings().items()):
            raise ValueError("Feedback condition/protocol/source binding differs")
        independent = copy.deepcopy(self.native)
        independent.load_state_dict(payload["native_state"])
        manager, evidence = None, copy.deepcopy(payload.get("feedback_evidence"))
        capability = copy.deepcopy(payload.get("capability_evidence"))
        pending, failure = payload.get("refresh_pending"), copy.deepcopy(payload.get("refresh_failure"))
        if self.feedback_enabled:
            manager = FeedbackManager(independent.envs[0].feature_names, self.feedback.config)
            manager.load_state_dict(payload["feedback_state"])
            if manager.last_step > independent.source_counters["joint_steps"] + independent.joint_steps:
                raise ValueError("Feedback schedule exceeds acknowledged PPO progress")
            if manager.program is not None and tuple(manager.program.feature_names) != tuple(independent.envs[0].feature_names):
                raise ValueError("Feedback program feature contract differs")
            if manager.program is not None and not self.test_fixture and evidence is None:
                raise ValueError("Production feedback tree requires extraction bindings")
            if evidence is not None:
                for key in ("source_actor_sha256", "source_actor_parameters_sha256", "evidence_sha256", "manager_fit_sha256"):
                    _sha(evidence.get(key), key)
                if (evidence.get("fit_step") != manager.last_fit_step or evidence["manager_fit_sha256"] != digest(manager.last_fit_report)
                        or evidence["source_actor_sha256"] != manager.last_fit_report.get("source_actor_sha256")):
                    raise ValueError("Saved feedback extraction binding differs")
            if capability is not None:
                if (type(capability.get("capability_eligible")) is not bool
                        or capability.get("step") != manager.last_gate.get("step")
                        or capability.get("gate") != manager.last_gate):
                    raise ValueError("Saved capability gate binding differs")
                if capability["capability_eligible"]: _sha(capability.get("report_sha256"), "capability report")
            if manager.current_lambda > 0 and not self.test_fixture and capability is None:
                raise ValueError("Active production feedback requires its capability report binding")
        elif payload.get("feedback_state") is not None or evidence is not None or capability is not None or pending or failure is not None:
            raise ValueError("Control cannot contain feedback state")
        _validate_refresh_state(manager, capability, pending, failure)
        self.native, self.feedback, self.feedback_evidence = independent, manager, evidence
        self.capability_evidence = capability
        self.refresh_pending, self.refresh_failure = pending, failure

    def save(self, path):
        payload = self.state_dict()
        with Path(path).open("xb") as stream: torch.save(payload, stream)

    def export(self, path):
        if execution_sources() != self.sources: raise ValueError("Feedback execution sources changed")
        metadata = self.native._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, protocol_sha256=digest(self.protocol),
            source_sha256=digest(self.sources), feedback_branch=self.feedback_branch,
            feedback_enabled=self.feedback_enabled, feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.,
            actor_parameters_sha256=initialization_sha256(self.model.actor.state_dict()),
            joint_steps=self.joint_steps, optimizer_updates=self.optimizer_updates, minibatch_updates=self.minibatch_updates,
            initialization="exact_genuine_source_with_inflight_episodes", candidate=True)
        return self.model.export_npz(path, metadata)
