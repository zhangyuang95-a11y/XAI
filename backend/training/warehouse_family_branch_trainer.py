"""Finite native MAPPO continuation with frozen ordinary/branch tree feedback.

Checkpoint decoding, physical branch collection and teacher fitting belong to
the caller. This module restores one complete genuine native state without
ancestry replay, then forks the original learner into a separately named cycle.
The exported policy remains the unmodified five-action neural Actor.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_native_continuation_feedback_trainer as ordinary
from . import warehouse_native_shutdown_feedback_trainer as frozen
from . import warehouse_native_expanded_rcpd as extraction
from . import warehouse_family_branch_ppo_update as branch_update
from . import warehouse_family_branch_teacher_fit as teacher
from .warehouse_family_feedback_cycle_source import gate_values
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_partner_mix_trainer import _sha
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_native_public_feedback_trainer import _integer

VERSION = "warehouse-family-branch-feedback-trainer.v1"
PROTOCOL_VERSION = "warehouse-family-branch-feedback-protocol.v1"
POOL_VERSION = "warehouse-family-branch-auxiliary-pool.v1"
TEACHER_VERSION = teacher.VERSION
USAGE_FIELDS = ("updates", "auxiliary_rng_draws", "auxiliary_pairs_used",
    "auxiliary_endpoints_used", "auxiliary_actor_forward_calls", "auxiliary_tree_rows",
    "ppo_actor_forward_calls", "ppo_nn_row_visits", "ordinary_tree_row_visits",
    "actor_adam_steps", "critic_adam_steps", "minibatch_updates")


def execution_sources():
    result = frozen.execution_sources()
    result.update(teacher.execution_sources())
    for module in (branch_update,):
        path = Path(module.__file__)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ("warehouse_family_branch_feedback_loss.py", "warehouse_family_feedback_cycle_source.py",
                 "warehouse_family_branch_teacher_fit.py", Path(__file__).name):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _feedback_config(value, fixture):
    baseline = teacher.feedback_config()
    if value is None: return baseline
    result = ordinary._config(value, True)
    if not fixture and result != baseline:
        raise ValueError("Production teacher capacity and unchanged reliability/schedule gates differ")
    return result


def restore_shutdown_source(payload, scenarios, *, expected_state_sha256,
                            device="cpu", test_fixture=False):
    """Restore the actual original class, using its unchanged complete loader.

    The caller must verify the containing checkpoint's bytes before decoding it
    and supply its exact native-state digest. No ancestor payload is needed.
    Construction/reset validates snapshots but performs no environment steps.
    """
    _sha(expected_state_sha256, "verified complete native state")
    device = torch.device(device)
    if (not isinstance(payload, dict) or semantic(payload) != expected_state_sha256
            or payload.get("version") != native.VERSION
            or payload.get("protocol", {}).get("version") != native.PROTOCOL_VERSION
            or type(test_fixture) is not bool or payload.get("test_fixture") is not test_fixture
            or payload.get("training_device") != device.type
            or payload.get("sources") != native.execution_sources()
            or payload.get("scenario_manifest_sha256") != digest(scenarios)
            or payload.get("public_feedback_mode") != "observed"
            or payload.get("branch") != "own_credit" or payload.get("delivery_credit_alpha") != .5
            or payload.get("feedback_enabled") is not False):
        raise ValueError("Verified native source identity, bytes digest, scenarios or device differs")
    original = native.ShutdownContinuationTrainer.__new__(native.ShutdownContinuationTrainer)
    original.device, original.mode, original.test_fixture = device, "observed", test_fixture
    original.protocol, original.scenarios = copy.deepcopy(payload["protocol"]), copy.deepcopy(scenarios)
    original.cfg, original.seed = original.protocol["training"], original.protocol["seed"]
    original.sources = native.execution_sources()
    for field in ("cycle_id", "branch", "shutdown_arm", "initialization_sha256",
            "source_checkpoint_sha256", "source_lineage", "source_counters", "source_parent_counters",
            "source_completed_episodes_sha256", "source_round_episode_prefixes_sha256", "source_frames"):
        setattr(original, field, copy.deepcopy(payload[field]))
    original.alpha, original.beta = payload["delivery_credit_alpha"], payload["own_shutdown_beta"]
    if original.shutdown_arm not in native.BETAS or original.beta != native.BETAS[original.shutdown_arm]:
        raise ValueError("Original shutdown reward arm differs")
    _integer(original.cfg["environments"], "native environments", 1)
    original.feedback_enabled, original._collecting = False, False
    original.envs = [original._environment() for _ in range(original.cfg["environments"])]
    original._validate_scenarios()
    native.ShutdownContinuationTrainer.load_state_dict(original, payload)
    if semantic(original.state_dict()) != expected_state_sha256:
        raise ValueError("Original loader did not preserve the complete supplied state")
    return original


def make_protocol(source_trainer, *, source_checkpoint_sha256, source_state_sha256,
                  ppo_cap, cycle_id, source_report, team_reference,
                  evaluation_checkpoints=None, feedback_config=None,
                  auxiliary_seed=260910, auxiliary_pairs_per_minibatch=64, test_fixture=False):
    base = native.make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=evaluation_checkpoints, test_fixture=test_fixture)
    initial = gate_values(source_report, source_report, team_reference=team_reference)
    if not initial["capability_eligible"]:
        raise ValueError("Initial full AND warmup AND per-partner NN retention guard must pass")
    _integer(auxiliary_seed, "independent auxiliary seed")
    _integer(auxiliary_pairs_per_minibatch, "auxiliary pairs per minibatch", 1)
    config = _feedback_config(feedback_config, test_fixture)
    return {**copy.deepcopy(base), "version": PROTOCOL_VERSION, "native_protocol": base,
        "feedback_training_enabled": True, "feedback_branches": ["control", "feedback"],
        "feedback_config": asdict(config), "feedback": {
            "direction": "KL(nn||tree)", "maximum_total_lambda": .01,
            "ordinary_fraction": .5, "branch_fraction": .5,
            "refresh_interval_joint_steps": 50000, "teacher_version": TEACHER_VERSION,
            "action_controller": "neural_actor_only", "runtime_action_override": False,
            "branch_rows_supply_ppo_targets": False},
        "auxiliary_seed": auxiliary_seed, "auxiliary_pairs_per_minibatch": auxiliary_pairs_per_minibatch,
        "source_report": copy.deepcopy(source_report), "source_report_sha256": digest(source_report),
        "team_reference": copy.deepcopy(team_reference), "team_reference_sha256": digest(team_reference),
        "guard": {"full_and_warmup": True, "per_partner_nn_retention": .9,
            "team_drop_guard": "unchanged_ExactProgramManager"},
        "explanation_qualification_granted": False}


def auxiliary_binding(observations, weights, *, source_actor_sha256,
                      source_actor_parameters_sha256, fit_step, evidence_sha256):
    """Bind externally verified real two-role WAIT/changed pairs to one Actor."""
    if not isinstance(weights, np.ndarray): raise ValueError("Auxiliary weights must be a NumPy array")
    branch_update._auxiliary(observations, weights)
    for key, value in (("Actor", source_actor_sha256), ("Actor parameters", source_actor_parameters_sha256),
                       ("physical collection evidence", evidence_sha256)): _sha(value, key)
    _integer(fit_step, "auxiliary fit step")
    return {"version": POOL_VERSION, "source_actor_sha256": source_actor_sha256,
        "source_actor_parameters_sha256": source_actor_parameters_sha256, "fit_step": fit_step,
        "evidence_sha256": evidence_sha256, "neural_roles": [0, 1],
        "pair_endpoints": ["WAIT", "changed_partner_action"],
        "pool_sha256": semantic({"observations": observations, "weights": weights})}


class BranchFeedbackTrainer:
    """Own checkpoint envelope; the native state keeps its true original schema."""
    __getattr__ = ordinary.ContinuationFeedbackTrainer.__getattr__
    __setattr__ = ordinary.ContinuationFeedbackTrainer.__setattr__
    feedback_clock = ordinary.ContinuationFeedbackTrainer.feedback_clock
    begin_feedback_refresh = ordinary.ContinuationFeedbackTrainer.begin_feedback_refresh
    fail_feedback_refresh = ordinary.ContinuationFeedbackTrainer.fail_feedback_refresh
    _feedback_snapshot = ordinary.ContinuationFeedbackTrainer._feedback_snapshot
    train_chunk = ordinary.ContinuationFeedbackTrainer.train_chunk

    def __init__(self, protocol, source_trainer, *, feedback_branch="feedback",
                 expected_source_state_sha256, source_checkpoint_sha256, device="cpu", test_fixture=False):
        if feedback_branch not in ("control", "feedback"):
            raise ValueError("Unknown feedback condition")
        expected = make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
            source_state_sha256=expected_source_state_sha256, ppo_cap=protocol["budget"]["maximum_ppo_joint_steps"],
            cycle_id=protocol["cycle_id"], source_report=protocol["source_report"],
            team_reference=protocol["team_reference"], evaluation_checkpoints=protocol["evaluation"]["checkpoints_ppo_steps"],
            feedback_config=protocol["feedback_config"], auxiliary_seed=protocol["auxiliary_seed"],
            auxiliary_pairs_per_minibatch=protocol["auxiliary_pairs_per_minibatch"], test_fixture=test_fixture)
        if digest(protocol) != digest(expected): raise ValueError("Branch trainer protocol differs")
        self.protocol = copy.deepcopy(expected)
        self.native = native.ShutdownContinuationTrainer(expected["native_protocol"], source_trainer,
            expected_source_state_sha256=expected_source_state_sha256, source_checkpoint_sha256=source_checkpoint_sha256,
            device=device, test_fixture=test_fixture)
        self.sources, self.feedback_branch = execution_sources(), feedback_branch
        self.feedback_enabled = feedback_branch == "feedback"
        self.feedback = extraction.ExactProgramManager(self.envs[0].feature_names,
            _feedback_config(expected["feedback_config"], test_fixture)) if self.feedback_enabled else None
        self.feedback_evidence = self.capability_evidence = None
        self.refresh_pending, self.refresh_failure = False, None
        self.auxiliary_observations = np.empty((0, 2, 197), np.float32)
        self.auxiliary_weights = np.empty(0, np.float32)
        self.auxiliary_evidence = None
        self.auxiliary_rng = np.random.default_rng(expected["auxiliary_seed"])
        self.usage = dict.fromkeys(USAGE_FIELDS, 0)

    def _expire_feedback(self):
        if (self.feedback_enabled and self.feedback_evidence is not None and not self.refresh_pending
                and self.feedback_clock - self.feedback_evidence["fit_step"] >=
                self.protocol["feedback"]["refresh_interval_joint_steps"]):
            self.begin_feedback_refresh()

    def collect(self, time_steps):
        self._expire_feedback()
        batch = self.native.collect(time_steps)
        for row in batch["transition_records"]:
            row.update(version=VERSION, feedback_branch=self.feedback_branch,
                feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.)
        return batch

    def attach_feedback_state(self, manager_state, *, source_actor_sha256,
                              source_actor_parameters_sha256, evidence_sha256):
        if not self.feedback_enabled: raise ValueError("Control cannot install a teacher")
        for name, value in (("Actor", source_actor_sha256), ("parameters", source_actor_parameters_sha256),
                            ("teacher evidence", evidence_sha256)): _sha(value, name)
        if frozen.actor_parameter_sha256(self.model.actor.state_dict()) != source_actor_parameters_sha256:
            raise ValueError("Teacher labels belong to a different current neural Actor")
        manager = extraction.ExactProgramManager(self.envs[0].feature_names, self.feedback.config)
        manager.load_state_dict(manager_state)
        self._validate_teacher(manager, expected_evidence_sha256=evidence_sha256)
        fit = manager.last_fit_report
        if (fit.get("source_actor_sha256") != source_actor_sha256
                or manager.last_fit_step != self.feedback_clock or manager.last_step != self.feedback_clock
                or manager.current_lambda != 0 or manager.program is None
                or (not self.test_fixture and fit.get("observed197_bindings", {}).get(
                    "actor_parameters_sha256") != source_actor_parameters_sha256)):
            raise ValueError("Teacher must be gate-closed and fitted from this current Actor/clock")
        manager = ordinary._retain_schedule(self.feedback, manager)
        self.feedback = manager
        self.feedback_evidence = {"source_actor_sha256": source_actor_sha256,
            "source_actor_parameters_sha256": source_actor_parameters_sha256,
            "evidence_sha256": evidence_sha256, "fit_step": self.feedback_clock,
            "manager_fit_sha256": digest(fit), "program_sha256": digest(manager.program.to_dict()),
            "frozen_manager_state": copy.deepcopy(manager_state)}
        self.auxiliary_observations = np.empty((0, 2, 197), np.float32)
        self.auxiliary_weights = np.empty(0, np.float32)
        self.auxiliary_evidence = self.capability_evidence = None
        self.refresh_pending, self.refresh_failure = False, None

    def _validate_teacher(self, manager, *, frozen_state=None, expected_evidence_sha256=None):
        if manager.program is None: return
        if tuple(manager.program.feature_names) != tuple(self.envs[0].feature_names):
            raise ValueError("Teacher observation feature contract differs")
        if self.test_fixture and manager.program.metadata.get("test_fixture") is True: return
        state = manager.state_dict() if frozen_state is None else frozen_state
        report = state["last_fit_report"]
        checked = teacher.validate_manager_state(state, feature_names=self.envs[0].feature_names,
            actor_bindings=report["actor_bindings"], step=report["step"],
            expected_evidence_sha256=expected_evidence_sha256)
        if (digest(checked.program.to_dict()) != digest(manager.program.to_dict())
                or digest(checked.last_fit_report) != digest(manager.last_fit_report)
                or (manager.reliable and not checked.reliable)):
            raise ValueError("Scheduled teacher differs from the original frozen fit")

    def attach_auxiliary_pool(self, observations, weights, *, binding):
        if not self.feedback_enabled or self.feedback_evidence is None or self.feedback.current_lambda != 0:
            raise ValueError("Install a gate-closed current teacher before its auxiliary pool")
        expected = auxiliary_binding(observations, weights,
            source_actor_sha256=self.feedback_evidence["source_actor_sha256"],
            source_actor_parameters_sha256=self.feedback_evidence["source_actor_parameters_sha256"],
            fit_step=self.feedback_evidence["fit_step"], evidence_sha256=binding.get("evidence_sha256"))
        if digest(binding) != digest(expected): raise ValueError("Auxiliary source/Actor/pool binding differs")
        if frozen.actor_parameter_sha256(self.model.actor.state_dict()) != binding["source_actor_parameters_sha256"]:
            raise ValueError("Auxiliary pool is not from the current frozen Actor")
        if not len(weights) or not np.any(weights > 0): raise ValueError("Branch training needs effective real pairs")
        self.auxiliary_observations, self.auxiliary_weights = observations.copy(), weights.copy()
        self.auxiliary_observations.flags.writeable = self.auxiliary_weights.flags.writeable = False
        self.auxiliary_evidence = copy.deepcopy(binding)

    def update_feedback_gate(self, report, *, capability_report_sha256):
        if not self.feedback_enabled: return {"active": False, "lambda": 0., "reason": "control"}
        if self.refresh_pending: raise ValueError("Feedback refresh is incomplete")
        if self.feedback_evidence is None or self.auxiliary_evidence is None:
            raise ValueError("Both current teacher and auxiliary pool must be installed")
        _sha(capability_report_sha256, "verified capability report")
        guard = gate_values(report, self.protocol["source_report"], team_reference=self.protocol["team_reference"])
        result = self.feedback.update(self.feedback_clock, **guard["update_kwargs"])
        self.capability_evidence = {"report_sha256": capability_report_sha256,
            "report_semantic_sha256": digest(report), "report": copy.deepcopy(report),
            "capability_eligible": guard["capability_eligible"], "strict_guard": guard["diagnostics"],
            "step": self.feedback_clock, "gate": copy.deepcopy(result)}
        return {**result, "strict_guard": copy.deepcopy(guard["diagnostics"])}

    def update(self, batch):
        self._expire_feedback()
        if self.feedback_enabled and self.feedback.current_lambda > 0:
            if (self.refresh_pending or self.auxiliary_evidence is None or self.capability_evidence is None
                    or self.capability_evidence.get("capability_eligible") is not True):
                raise ValueError("Active feedback requires complete current tree/pool and strict capability")
        result = branch_update.branch_ppo_update(self.native, batch, self.feedback,
            self.auxiliary_observations, self.auxiliary_weights, self.auxiliary_rng,
            branch_fraction=.5, auxiliary_pairs_per_minibatch=self.protocol["auxiliary_pairs_per_minibatch"])
        self.usage["updates"] += 1
        for name in USAGE_FIELDS[1:]: self.usage[name] += result["branch_update"][name]
        return result

    def _bindings(self):
        return {"version": VERSION, "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "feedback_branch": self.feedback_branch, "feedback_enabled": self.feedback_enabled,
            "execution_sources": copy.deepcopy(self.sources), "source_sha256": digest(self.sources),
            "test_fixture": self.test_fixture}

    def state_dict(self):
        if execution_sources() != self.sources: raise ValueError("Branch trainer execution sources changed")
        return {**self._bindings(), "native_state": self.native.state_dict(), **self._feedback_snapshot(),
            "auxiliary_observations": self.auxiliary_observations.copy(),
            "auxiliary_weights": self.auxiliary_weights.copy(), "auxiliary_evidence": copy.deepcopy(self.auxiliary_evidence),
            "auxiliary_rng": copy.deepcopy(self.auxiliary_rng.bit_generator.state), "usage": copy.deepcopy(self.usage)}

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid checkpoint or changed branch sources")
        if any(digest(payload.get(k)) != digest(v) for k, v in self._bindings().items()):
            raise ValueError("Saved branch protocol/identity/source differs")
        staged = copy.deepcopy(self)
        staged.native.load_state_dict(payload["native_state"])
        staged.feedback_evidence = copy.deepcopy(payload["feedback_evidence"])
        staged.capability_evidence = copy.deepcopy(payload["capability_evidence"])
        staged.refresh_pending, staged.refresh_failure = payload["refresh_pending"], copy.deepcopy(payload["refresh_failure"])
        obs, weights = payload["auxiliary_observations"], payload["auxiliary_weights"]
        branch_update._auxiliary(obs, weights)
        staged.auxiliary_evidence = copy.deepcopy(payload["auxiliary_evidence"])
        staged.auxiliary_observations, staged.auxiliary_weights = obs.copy(), weights.copy()
        staged.auxiliary_observations.flags.writeable = staged.auxiliary_weights.flags.writeable = False
        staged.auxiliary_rng = np.random.default_rng()
        staged.auxiliary_rng.bit_generator.state = copy.deepcopy(payload["auxiliary_rng"])
        staged.usage = copy.deepcopy(payload["usage"])
        if set(staged.usage) != set(USAGE_FIELDS): raise ValueError("Incomplete branch usage counters")
        for key, value in staged.usage.items(): _integer(value, key)
        if staged.usage["updates"] != staged.native.optimizer_updates:
            raise ValueError("Branch usage does not match applied native updates")
        for usage, native_counter in (("actor_adam_steps", "actor_optimizer_steps"),
                ("critic_adam_steps", "critic_optimizer_steps"), ("minibatch_updates", "minibatch_updates")):
            if staged.usage[usage] != getattr(staged.native, native_counter):
                raise ValueError("Branch usage does not match native " + native_counter)
        if (staged.usage["auxiliary_endpoints_used"] != 2*staged.usage["auxiliary_pairs_used"]
                or staged.usage["auxiliary_rng_draws"] != staged.usage["auxiliary_actor_forward_calls"]):
            raise ValueError("Auxiliary endpoint/RNG usage counters disagree")
        if staged.feedback_enabled:
            staged.feedback = extraction.ExactProgramManager(staged.envs[0].feature_names, self.feedback.config)
            staged.feedback.load_state_dict(payload["feedback_state"])
            evidence = staged.feedback_evidence
            staged._validate_teacher(staged.feedback,
                frozen_state=evidence.get("frozen_manager_state") if evidence is not None else None,
                expected_evidence_sha256=evidence.get("evidence_sha256") if evidence is not None else None)
            staged._validate_saved_feedback()
        elif (payload["feedback_state"] is not None or staged.feedback_evidence is not None
                or staged.capability_evidence is not None or len(weights) or staged.auxiliary_evidence is not None
                or staged.refresh_pending or staged.refresh_failure is not None):
            raise ValueError("Control checkpoint contains feedback state")
        ordinary._validate_refresh_state(staged.feedback, staged.capability_evidence,
            staged.refresh_pending, staged.refresh_failure)
        self.__dict__.update(staged.__dict__)

    def _validate_saved_feedback(self):
        manager, evidence, auxiliary, capability = (self.feedback, self.feedback_evidence,
            self.auxiliary_evidence, self.capability_evidence)
        if manager.last_step > self.feedback_clock: raise ValueError("Feedback clock exceeds native progress")
        if manager.program is not None and evidence is None: raise ValueError("Teacher lacks source bindings")
        if evidence is not None:
            for key in ("source_actor_sha256", "source_actor_parameters_sha256", "evidence_sha256",
                        "manager_fit_sha256", "program_sha256"): _sha(evidence.get(key), key)
            if (evidence["fit_step"] != manager.last_fit_step
                    or evidence["manager_fit_sha256"] != digest(manager.last_fit_report)
                    or evidence["program_sha256"] != digest(manager.program.to_dict())
                    or evidence["source_actor_sha256"] != manager.last_fit_report.get("source_actor_sha256")
                    or (not self.test_fixture and evidence["source_actor_parameters_sha256"] !=
                        manager.last_fit_report.get("observed197_bindings", {}).get("actor_parameters_sha256"))):
                raise ValueError("Saved teacher fit/program/source binding differs")
        if auxiliary is not None:
            if evidence is None: raise ValueError("Auxiliary pool lacks teacher source")
            expected = auxiliary_binding(self.auxiliary_observations, self.auxiliary_weights,
                source_actor_sha256=evidence["source_actor_sha256"],
                source_actor_parameters_sha256=evidence["source_actor_parameters_sha256"],
                fit_step=evidence["fit_step"], evidence_sha256=auxiliary.get("evidence_sha256"))
            if digest(expected) != digest(auxiliary): raise ValueError("Saved auxiliary pool binding differs")
        elif len(self.auxiliary_weights): raise ValueError("Saved auxiliary rows lack provenance")
        if capability is not None and not self.refresh_pending:
            _sha(capability.get("report_sha256"), "capability report")
            guard = gate_values(capability["report"], self.protocol["source_report"], team_reference=self.protocol["team_reference"])
            if (capability["report_semantic_sha256"] != digest(capability["report"])
                    or capability["strict_guard"] != guard["diagnostics"]
                    or capability["capability_eligible"] is not guard["capability_eligible"]
                    or capability["step"] != manager.last_gate.get("step") or capability["gate"] != manager.last_gate):
                raise ValueError("Saved strict capability evidence differs")
        if manager.current_lambda > 0 and (auxiliary is None or capability is None
                or capability.get("capability_eligible") is not True or self.refresh_pending):
            raise ValueError("Active feedback requires the strict capability and complete auxiliary source")

    def save(self, path):
        payload = self.state_dict()
        with Path(path).open("xb") as stream: torch.save(payload, stream)

    def export(self, path):
        if execution_sources() != self.sources: raise ValueError("Branch export sources changed")
        if Path(path).exists(): raise FileExistsError(path)
        metadata = self.native._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, protocol_sha256=digest(self.protocol),
            source_sha256=digest(self.sources), feedback_branch=self.feedback_branch,
            feedback_enabled=self.feedback_enabled, feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.,
            actor_parameters_sha256=frozen.actor_parameter_sha256(self.model.actor.state_dict()),
            joint_steps=self.joint_steps, optimizer_updates=self.optimizer_updates,
            minibatch_updates=self.minibatch_updates, candidate=True, explanation_qualified=False,
            initialization="exact_genuine_shutdown_source_with_inflight_episodes")
        return self.model.export_npz(path, metadata)
