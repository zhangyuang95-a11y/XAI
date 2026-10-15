"""RCPD soft feedback over a genuine retained-beta neural continuation.

The frozen PPO/KL implementation is reused as methods, not as a checkpoint
identity. The inner learner and outer feedback envelope each retain their own
explicit schema, source closure, complete learning state and training clock.
This component authorizes neither sampling nor participant explanations.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_partner_mix_trainer import _sha
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_native_continuation_feedback_trainer as algorithm
from . import warehouse_native_expanded_rcpd as extraction
from env.warehouse_native.feedback import FeedbackConfig

VERSION = "warehouse-native-own-shutdown-feedback-trainer.v2"
PROTOCOL_VERSION = "warehouse-native-own-shutdown-feedback-protocol.v2"
BRANCHES = ("control", "feedback")


def execution_sources():
    result = native.execution_sources()
    result.update(algorithm.execution_sources())
    for path in (Path(extraction.__file__),Path(extraction.program_batch.__file__)):
        result[str(path.relative_to(ROOT))]=file_hash(path)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def validated_feedback_config(value=None, fixture=False):
    """Explicit expanded capacity; the reliability/loss gates are unchanged."""
    if value is None:return extraction.feedback_config()
    if fixture:return algorithm._config(value,True)
    if isinstance(value,FeedbackConfig):config=value
    elif isinstance(value,dict):
        values=copy.deepcopy(value)
        for key in ('depths','leaves','critical_groups'):
            if key in values:values[key]=tuple(values[key])
        config=FeedbackConfig(**values)
    else:raise ValueError('A complete expanded feedback config is required')
    if config!=extraction.feedback_config():raise ValueError('The revised production feedback contract differs')
    return config


def feedback_contract():
    return {"target":"detached_tree_distribution","direction":"KL(nn||tree)",
        "complexity":"candidate_selection_only","extract_interval":50000,
        "schedule_step":"cumulative_joint_steps","source_capability_required":True,
        "action_controller":"neural_actor_only","extraction_version":extraction.VERSION,
        "prediction_semantics":extraction.program_batch.VERSION,
        "candidate_pairs":[list(x) for x in extraction.CANDIDATES]}


def actor_parameter_sha256(state):
    """Semantic byte hashing has a CPU contract, including on MPS training."""
    return initialization_sha256(algorithm._cpu(state))


def validate_program(manager,fixture):
    """Validate actual extraction identity; synthetic gradient targets stay explicit."""
    if manager.program is None:return
    if fixture and manager.program.metadata.get('test_fixture') is True:return
    fit=manager.last_fit_report;meta=manager.program.metadata;binding=fit.get('observed197_bindings',{})
    if (fit.get('version')!=extraction.VERSION or fit.get('prediction_semantics')!=extraction.program_batch.VERSION
            or fit.get('overlap')!=dict.fromkeys(('scene_fingerprints','episode_ids','public_states','exact_observations'),0)
            or fit.get('rows_removed')!=0 or fit.get('explanation_qualified') is not False
            or digest(fit.get('extraction_config'))!=digest(extraction.contract())
            or meta.get('native_feedback_version')!=extraction.VERSION
            or meta.get('prediction_semantics')!=extraction.program_batch.VERSION
            or digest(meta.get('native_feedback_config'))!=digest(asdict(manager.config))
            or meta.get('observed197_bindings')!=binding
            or meta.get('metrics',{}).get('explanation_eligible') is not False
            or meta.get('native_source_actor_sha256')!=fit.get('source_actor_sha256')
            or binding.get('cumulative_fit_step')!=manager.last_fit_step
            or binding.get('config_sha256')!=digest(extraction.contract())
            or fit.get('reliable') is not extraction.gates(fit['selection_metrics'])['reliable']):
        raise ValueError('Expanded feedback program contract or evidence differs')
    # During a pending refresh the installed old program is deliberately closed.
    if manager.reliable and fit.get('reliable') is not True:raise ValueError('An unreliable program cannot open feedback')


def make_protocol(source_trainer, *, source_checkpoint_sha256, source_state_sha256,
                  ppo_cap, cycle_id, evaluation_checkpoints=None, feedback_config=None, test_fixture=False):
    base = native.make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=evaluation_checkpoints, test_fixture=test_fixture)
    config = validated_feedback_config(feedback_config, test_fixture)
    return {**copy.deepcopy(base), "version": PROTOCOL_VERSION, "native_protocol": base,
        "feedback_training_enabled": True, "feedback_branches": list(BRANCHES),
        "feedback_config": asdict(config),
        "feedback": feedback_contract(),
        "explanation_qualification_granted": False}


class ShutdownFeedbackTrainer(algorithm.ContinuationFeedbackTrainer):
    """Reuse actual PPO/KL operations while preserving the new native identity.

Inherited update/feedback methods consume self.native, actual NN trajectories
and FeedbackManager. They neither construct nor relabel an old continuation.
Control and zero-lambda updates call the new native learner's exact update.
"""

    def __init__(self, protocol, source_trainer, *, feedback_branch="control",
                 expected_source_state_sha256, source_checkpoint_sha256, device="cpu", test_fixture=False):
        if type(feedback_branch) is not str or feedback_branch not in BRANCHES:
            raise ValueError("Unknown feedback condition")
        if not isinstance(protocol, dict):
            raise ValueError("An explicit retained-beta feedback protocol is required")
        expected = make_protocol(source_trainer, source_checkpoint_sha256=source_checkpoint_sha256,
            source_state_sha256=expected_source_state_sha256,
            ppo_cap=protocol.get("budget", {}).get("maximum_ppo_joint_steps_per_arm"),
            cycle_id=protocol.get("cycle_id"),
            evaluation_checkpoints=protocol.get("evaluation", {}).get("checkpoints_ppo_steps"),
            feedback_config=protocol.get("feedback_config"), test_fixture=test_fixture)
        if digest(protocol) != digest(expected):
            raise ValueError("Feedback must retain the selected beta and complete learning contract")
        self.protocol = copy.deepcopy(expected)
        self.native = native.ShutdownContinuationTrainer(self.protocol["native_protocol"], source_trainer,
            expected_source_state_sha256=expected_source_state_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256, device=device, test_fixture=test_fixture)
        self.feedback_branch = feedback_branch
        self.feedback_enabled = feedback_branch == "feedback"
        self.sources = execution_sources()
        self.feedback = extraction.ExactProgramManager(self.native.envs[0].feature_names,
            validated_feedback_config(self.protocol["feedback_config"], test_fixture)) if self.feedback_enabled else None
        self.feedback_evidence = None
        self.capability_evidence = None
        self.refresh_pending = False
        self.refresh_failure = None

    def collect(self, time_steps):
        batch = self.native.collect(time_steps)
        for row in batch["transition_records"]:
            row.update(version=VERSION, feedback_branch=self.feedback_branch,
                feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.)
        return batch

    def _bindings(self):
        return {"version": VERSION, "protocol": copy.deepcopy(self.protocol), "protocol_sha256": digest(self.protocol),
            "feedback_branch": self.feedback_branch, "feedback_enabled": self.feedback_enabled,
            "execution_sources": copy.deepcopy(self.sources), "source_sha256": digest(self.sources),
            "test_fixture": self.test_fixture}

    def attach_feedback_state(self, manager_state, *, source_actor_sha256,
                              source_actor_parameters_sha256, evidence_sha256):
        """Install a real revised extraction using its exact predicates, not old metadata."""
        if not self.feedback_enabled:raise ValueError('Control cannot install feedback')
        for name,value in [('source Actor',source_actor_sha256),('parameters',source_actor_parameters_sha256),
                ('extraction evidence',evidence_sha256)]:_sha(value,name)
        if actor_parameter_sha256(self.native.model.actor.state_dict())!=source_actor_parameters_sha256:
            raise ValueError('Extraction labels belong to a different frozen neural Actor')
        manager=extraction.ExactProgramManager(self.native.envs[0].feature_names,self.feedback.config)
        manager.load_state_dict(manager_state);validate_program(manager,self.test_fixture)
        fit=manager.last_fit_report;binding=fit.get('observed197_bindings',{})
        if (fit.get('source_actor_sha256')!=source_actor_sha256 or fit.get('step')!=self.feedback_clock
                or manager.last_fit_step!=self.feedback_clock or manager.last_step!=self.feedback_clock
                or manager.current_lambda!=0 or manager.program is None
                or tuple(manager.program.feature_names)!=tuple(self.native.envs[0].feature_names)
                or binding.get('actor_sha256')!=source_actor_sha256
                or binding.get('actor_parameters_sha256')!=source_actor_parameters_sha256
                or evidence_sha256!=digest({'binding':binding,'fit_report':fit})):
            raise ValueError('Extraction lacks current same-Actor and disjoint-data binding')
        manager=algorithm._retain_schedule(self.feedback,manager)
        self.feedback,self.feedback_evidence=manager,{'source_actor_sha256':source_actor_sha256,
            'source_actor_parameters_sha256':source_actor_parameters_sha256,'evidence_sha256':evidence_sha256,
            'fit_step':self.feedback_clock,'fit_additional_joint_steps':self.joint_steps,'manager_fit_sha256':digest(fit)}
        self.capability_evidence=None;self.refresh_pending=False;self.refresh_failure=None

    def state_dict(self):
        if execution_sources() != self.sources:
            raise ValueError("Retained-beta feedback execution sources changed")
        return {**self._bindings(), "native_state": self.native.state_dict(), **self._feedback_snapshot()}

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or execution_sources() != self.sources:
            raise ValueError("Invalid feedback checkpoint or changed sources")
        if any(digest(payload.get(key)) != digest(value) for key, value in self._bindings().items()):
            raise ValueError("Feedback condition, source or learning contract differs")
        independent = copy.deepcopy(self.native)
        independent.load_state_dict(payload["native_state"])
        manager = None
        evidence = copy.deepcopy(payload.get("feedback_evidence"))
        capability = copy.deepcopy(payload.get("capability_evidence"))
        pending = payload.get("refresh_pending")
        failure = copy.deepcopy(payload.get("refresh_failure"))
        if self.feedback_enabled:
            manager = extraction.ExactProgramManager(independent.envs[0].feature_names, self.feedback.config)
            manager.load_state_dict(payload["feedback_state"])
            validate_program(manager,self.test_fixture)
            clock = independent.source_counters["joint_steps"] + independent.joint_steps
            if manager.last_step > clock:
                raise ValueError("Feedback clock exceeds confirmed PPO progress")
            if manager.program is not None and tuple(manager.program.feature_names) != tuple(independent.envs[0].feature_names):
                raise ValueError("Feedback program feature contract differs")
            if manager.program is not None and not self.test_fixture and evidence is None:
                raise ValueError("Production feedback tree requires actual extraction bindings")
            if evidence is not None:
                for key in ("source_actor_sha256", "source_actor_parameters_sha256", "evidence_sha256", "manager_fit_sha256"):
                    _sha(evidence.get(key), key)
                if (evidence.get("fit_step") != manager.last_fit_step
                        or evidence["manager_fit_sha256"] != digest(manager.last_fit_report)
                        or evidence["source_actor_sha256"] != manager.last_fit_report.get("source_actor_sha256")
                        or evidence['source_actor_parameters_sha256']!=manager.last_fit_report.get('observed197_bindings',{}).get('actor_parameters_sha256')
                        or evidence['evidence_sha256']!=digest({'binding':manager.last_fit_report.get('observed197_bindings'),
                            'fit_report':manager.last_fit_report})
                        or evidence.get('fit_additional_joint_steps')!=manager.last_fit_step-independent.source_counters['joint_steps']):
                    raise ValueError("Saved feedback extraction binding differs")
                if (manager.program is None or manager.program.metadata.get("native_source_actor_sha256")
                        != evidence["source_actor_sha256"]):
                    raise ValueError("Saved feedback program source differs from extraction evidence")
            if capability is not None:
                if (type(capability.get("capability_eligible")) is not bool
                        or capability.get("step") != manager.last_gate.get("step")
                        or capability.get("gate") != manager.last_gate):
                    raise ValueError("Saved capability gate binding differs")
                if capability["capability_eligible"]:
                    _sha(capability.get("report_sha256"), "capability report")
            if manager.current_lambda > 0:
                if capability is None or capability.get("capability_eligible") is not True:
                    raise ValueError("Active feedback requires verified positive capability")
                _sha(capability.get("report_sha256"), "active capability report")
        elif payload.get("feedback_state") is not None or evidence is not None or capability is not None or pending or failure is not None:
            raise ValueError("Control cannot contain feedback state")
        algorithm._validate_refresh_state(manager, capability, pending, failure)
        self.native, self.feedback, self.feedback_evidence = independent, manager, evidence
        self.capability_evidence, self.refresh_pending, self.refresh_failure = capability, pending, failure

    def export(self, path):
        if execution_sources() != self.sources:
            raise ValueError("Retained-beta feedback export sources changed")
        metadata = self.native._bindings()
        for key in ("version", "sources", "protocol", "source_frames", "source_completed_episodes_sha256"):
            metadata.pop(key)
        metadata.update(experiment_version=VERSION, protocol_sha256=digest(self.protocol),
            source_sha256=digest(self.sources), feedback_branch=self.feedback_branch,
            feedback_enabled=self.feedback_enabled,
            feedback_lambda=self.feedback.current_lambda if self.feedback_enabled else 0.,
            actor_parameters_sha256=actor_parameter_sha256(self.model.actor.state_dict()),
            joint_steps=self.joint_steps, optimizer_updates=self.optimizer_updates,
            minibatch_updates=self.minibatch_updates,
            initialization="exact_genuine_source_with_retained_beta_and_inflight_episodes", candidate=True)
        return self.model.export_npz(path, metadata)
