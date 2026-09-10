"""Explicit lower-Actor-LR continuation, with a separately versioned tree loss.

The native collector, physics, observations, MLPs, Adam moments and RNG states
are retained. A new protocol records the one Actor optimizer rate change and
the stronger training-only KL schedule. It is never admitted as an old run.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np

from . import warehouse_family_branch_trainer as source_io
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_family_branch_grouped_teacher_fit as teacher
from . import warehouse_family_stable_ppo_update as updater
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_family_feedback_cycle_source import gate_values

VERSION = 'warehouse-family-stable-feedback-trainer.v1'
PROTOCOL_VERSION = 'warehouse-family-stable-feedback-protocol.v1'
MAX_LAMBDA = .2
RAMP_STEPS = 6000
ACTOR_LR = 3e-5
CRITIC_LR = 3e-4


def execution_sources():
    value = {**source_io.execution_sources(), **teacher.execution_sources()}
    for module in (updater,):
        value[str(Path(module.__file__).relative_to(ROOT))] = file_hash(module.__file__)
    value[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    return value


def make_protocol(source, *, source_checkpoint_sha256, source_state_sha256,
                  source_report, team_reference, ppo_cap=20000,
                  cycle_id='stable_20k_pair_20260910', test_fixture=False):
    checkpoints = [10000,20000] if not test_fixture else [ppo_cap]
    base = native.make_protocol(source, source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=checkpoints, test_fixture=test_fixture)
    if not test_fixture and ppo_cap != 20000:
        raise ValueError('This independent continuation fixes 20k per arm')
    if source.cfg['actor_learning_rate'] != 3e-4 or source.cfg['critic_learning_rate'] != CRITIC_LR:
        raise ValueError('Expected source rates differ')
    result = deepcopy(base)
    result.update(version=PROTOCOL_VERSION, engine_initialization_protocol=base,
        paired_branches=['control','feedback'], source_report=deepcopy(source_report),
        source_report_sha256=digest(source_report), team_reference=deepcopy(team_reference),
        team_reference_sha256=digest(team_reference), explanation_qualification_granted=False,
        feedback_training_enabled=True,
        feedback=dict(direction='KL(nn||tree)', maximum_total_lambda=MAX_LAMBDA,
            ramp_joint_steps=RAMP_STEPS, ordinary_fraction=.5, branch_fraction=.5,
            teacher_version=teacher.VERSION, refresh_interval_joint_steps=50000,
            action_controller='neural_actor_only', runtime_action_override=False,
            branch_rows_supply_ppo_targets=False),
        guard=dict(full_and_warmup=True, per_partner_nn_retention=.9,
                   maximum_team_drop_fraction=.1, team_anchor='fixed_source_NN',
                   failure_action='close_feedback_and_stop_pair_at_matched_boundary'))
    result['training']['actor_learning_rate'] = ACTOR_LR
    result['continuation']['learning_configuration'] = 'actor_learning_rate_3e-5_other_native_parameters_unchanged'
    return result


class StableFeedbackTrainer(native.ShutdownContinuationTrainer):
    """Native full-state loader with actual new bindings, not metadata aliases."""
    def __init__(self, protocol, source, *, source_checkpoint_sha256,
                 expected_source_state_sha256, branch, teacher_fit=None,
                 auxiliary_observations=None, device='mps', test_fixture=False):
        if branch not in ('control','feedback'):
            raise ValueError('Explicit paired branch required')
        expected = make_protocol(source, source_checkpoint_sha256=source_checkpoint_sha256,
            source_state_sha256=expected_source_state_sha256,
            source_report=protocol['source_report'], team_reference=protocol['team_reference'],
            ppo_cap=protocol['budget']['maximum_ppo_joint_steps'], cycle_id=protocol['cycle_id'],
            test_fixture=test_fixture)
        if semantic(expected) != semantic(protocol):
            raise ValueError('Frozen stable-continuation protocol differs')
        # Construct a genuine native fork first, then explicitly migrate this
        # independently versioned learner's sole changed optimizer parameter.
        engine = native.ShutdownContinuationTrainer(protocol['engine_initialization_protocol'], source,
            expected_source_state_sha256=expected_source_state_sha256,
            source_checkpoint_sha256=source_checkpoint_sha256, device=device, test_fixture=test_fixture)
        self.__dict__.update(engine.__dict__)
        self.protocol = deepcopy(protocol)
        self.cfg = deepcopy(protocol['training'])
        for group in self.optimizers.actor.param_groups:
            group['lr'] = ACTOR_LR
        self.feedback_branch = branch
        self.stable_sources = execution_sources()
        self.current_lambda = 0.
        self.feedback_allowed = branch == 'feedback'
        self.last_capability = None
        self.program = None
        self.teacher_fit = deepcopy(teacher_fit) if branch == 'feedback' else None
        self.auxiliary_observations = np.empty((0,2,197),np.float32)
        self.auxiliary_rng = np.random.default_rng(26091006)
        if branch == 'feedback':
            if not isinstance(teacher_fit,dict):
                raise ValueError('An actual frozen grouped teacher is required')
            report = teacher_fit['fit_report']
            manager = teacher.validate_manager_state(teacher_fit['manager_state'],
                feature_names=self.envs[0].feature_names, actor_bindings=report['actor_bindings'],
                step=self.source_counters['joint_steps'], expected_evidence_sha256=teacher_fit['evidence_sha256'])
            if not manager.reliable:
                raise ValueError('Teacher must pass the original ordinary reliability gate')
            if report['actor_bindings']['actor_parameters_sha256'] != source_io.frozen.actor_parameter_sha256(self.model.actor.state_dict()):
                raise ValueError('Teacher source differs from the actual current Actor')
            obs = np.asarray(auxiliary_observations)
            if obs.dtype != np.float32 or obs.ndim != 3 or obs.shape[1:] != (2,197) or not len(obs) or not np.isfinite(obs).all():
                raise ValueError('Original finite TRAIN-only paired observations required')
            self.auxiliary_observations = obs.copy()
            self.program = manager.program
        self.auxiliary_observations.setflags(write=False)
        self.initial_learning_state_sha256 = semantic({'model':native._cpu(self.model.state_dict()),
            'optimizers':native._cpu(self.optimizers.state_dict()), 'rng':self.rng.bit_generator.state,
            'owned_rng':self._owned_rng_state,'envs':[env.snapshot() for env in self.envs]})

    def _verify_stable(self):
        if self.stable_sources != execution_sources():
            raise ValueError('Stable-continuation source changed')
        for role,rate in (('actor',ACTOR_LR),('critic',CRITIC_LR)):
            if self.cfg[role+'_learning_rate'] != rate or any(g['lr']!=rate for g in getattr(self.optimizers,role).param_groups):
                raise ValueError('Actual optimizer rate differs from the new protocol')

    def _bindings(self):
        self._verify_stable()
        base = native.ShutdownContinuationTrainer._bindings(self)
        return {**base,'version':VERSION,'sources':deepcopy(self.stable_sources),
            'source_sha256':digest(self.stable_sources),'engine_sources':deepcopy(self.sources),
            'feedback_branch':self.feedback_branch,'feedback_enabled':self.feedback_branch=='feedback',
            'native_collector_feedback_enabled':False,
            'initial_learning_state_sha256':self.initial_learning_state_sha256,
            'teacher_fit_sha256':digest(self.teacher_fit) if self.teacher_fit is not None else None,
            'auxiliary_observations_sha256':semantic(self.auxiliary_observations)}

    def state_dict(self):
        value = native.ShutdownContinuationTrainer.state_dict(self)
        value['stable_feedback_state'] = dict(current_lambda=self.current_lambda,
            allowed=self.feedback_allowed,last_capability=deepcopy(self.last_capability),
            auxiliary_rng=deepcopy(self.auxiliary_rng.bit_generator.state))
        return value

    def load_state_dict(self,payload):
        runtime = deepcopy(payload['stable_feedback_state'])
        if (type(runtime.get('allowed')) is not bool or type(runtime.get('current_lambda')) not in (int,float)
                or not 0<=runtime['current_lambda']<=MAX_LAMBDA
                or self.feedback_branch=='control' and (runtime['allowed'] or runtime['current_lambda']!=0)):
            raise ValueError('Invalid persisted training-feedback state')
        rng = np.random.default_rng();rng.bit_generator.state=runtime['auxiliary_rng']
        native.ShutdownContinuationTrainer.load_state_dict(self,payload)
        self.current_lambda=runtime['current_lambda'];self.feedback_allowed=runtime['allowed']
        self.last_capability=runtime['last_capability'];self.auxiliary_rng=rng
        self._verify_stable()

    def collect(self,time_steps):
        self._verify_stable()
        self.current_lambda = (MAX_LAMBDA*min(1.,self.joint_steps/RAMP_STEPS)
            if self.feedback_allowed and self.feedback_branch=='feedback' else 0.)
        batch=native.ShutdownContinuationTrainer.collect(self,time_steps)
        for row in batch['transition_records']:
            row.update(version=VERSION,feedback_branch=self.feedback_branch,feedback_lambda=self.current_lambda)
        return batch

    def update(self,batch):
        self._verify_stable()
        return updater.stable_ppo_update(self,batch,self.program,
            lambda_value=self.current_lambda,auxiliary_observations=self.auxiliary_observations,
            auxiliary_rng=self.auxiliary_rng)

    def native_ppo_update(self,batch):
        return native.ShutdownContinuationTrainer.update(self,batch)

    def set_capability(self,report):
        value=gate_values(report,self.protocol['source_report'],team_reference=self.protocol['team_reference'])
        initial=gate_values(self.protocol['source_report'],self.protocol['source_report'],
            team_reference=self.protocol['team_reference'])
        team_retained=value['validation_score']>=.9*initial['validation_score']
        passed=value['capability_eligible'] and team_retained
        self.last_capability=dict(**value,team_retained=team_retained,passed=passed,
            report_sha256=digest(report))
        if not passed:
            self.feedback_allowed=False;self.current_lambda=0.
        return bool(passed)

    def export(self,path):
        self._verify_stable()
        metadata=self._bindings()
        for key in ('version','sources','protocol','engine_sources','source_frames','source_completed_episodes_sha256'):
            metadata.pop(key,None)
        metadata.update(experiment_version=VERSION,joint_steps=self.joint_steps,
            optimizer_updates=self.optimizer_updates,minibatch_updates=self.minibatch_updates,
            actor_parameters_sha256=source_io.frozen.actor_parameter_sha256(self.model.actor.state_dict()),
            feedback_lambda=self.current_lambda,initialization='actual_native_fork_then_recorded_actor_LR_change',
            candidate=True,explanation_qualified=False)
        return self.model.export_npz(path,metadata)
