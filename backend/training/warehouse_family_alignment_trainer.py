"""Explicit 50k continuation from a genuinely loaded stable 20k learner.

The frozen 388m teacher retains its origin. Current 390m TRAIN pairs replace
the auxiliary pool; ordinary development checks may close feedback permanently.
This module neither loads checkpoint files nor samples or fits a teacher.
"""
from copy import deepcopy
from pathlib import Path
import math
import re

import numpy as np
import torch

from core.program import ExecutableProgram
from . import warehouse_family_stable_trainer as stable
from . import warehouse_native_shutdown_continuation_trainer as native
from . import warehouse_family_branch_teacher_fit as ordinary
from . import warehouse_family_stable_ppo_update as updater
from . import warehouse_family_stable_evaluation as source_evaluation
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_public_feedback_initialization import initialization_sha256 as semantic
from .warehouse_family_feedback_cycle_source import gate_values

VERSION = 'warehouse-family-alignment-trainer.v1'
PROTOCOL_VERSION = 'warehouse-family-alignment-protocol.v1'
TEACHER_CHECK_VERSION = 'warehouse-family-alignment-teacher-check.v1'
AUXILIARY_VERSION = 'warehouse-current-teacher-auxiliary.v1'
MAX_LAMBDA = .2
ACTOR_LR = 3e-5
CRITIC_LR = 3e-4


def execution_sources():
    result = {**stable.execution_sources(), **ordinary.execution_sources()}
    result[str(Path(source_evaluation.__file__).relative_to(ROOT))] = file_hash(source_evaluation.__file__)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    return result


def _same(a, b, reason):
    if semantic(a) != semantic(b):
        raise ValueError(reason)


def _sha(value, name):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid SHA256: ' + name)
    return value


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer: ' + name)
    return value


def _actor_parameters(source):
    return stable.source_io.frozen.actor_parameter_sha256(source.model.actor.state_dict())


def _feedback_contract():
    return dict(direction='KL(nn||tree)', maximum_total_lambda=MAX_LAMBDA,
        initial_total_lambda=MAX_LAMBDA, ramp_joint_steps=0,
        ordinary_fraction=.5, branch_fraction=.5, teacher_version=stable.teacher.VERSION,
        teacher_origin='unchanged_frozen_388m_teacher', teacher_refresh=False,
        action_controller='neural_actor_only', runtime_action_override=False,
        branch_rows_supply_ppo_targets=False,
        reliability_scope='fixed_development_ordinary_rows', failure_latches_closed=True)


def _guard_contract():
    return dict(full_and_warmup=True, per_partner_nn_retention=.9,
        maximum_team_drop_fraction=.1, team_anchor='fixed_source_NN',
        failure_action='close_feedback_permanently_continue_both_arms')


def _capability(report, source_report, team_reference):
    value = gate_values(report, source_report, team_reference=team_reference)
    initial = gate_values(source_report, source_report, team_reference=team_reference)
    retained = value['validation_score'] >= .9 * initial['validation_score']
    return dict(**value, team_retained=retained,
        passed=bool(value['capability_eligible'] and retained), report_sha256=digest(report))


def _entry_binding(report, actor_sha, parameters, *, fixture):
    if not isinstance(report, dict):
        raise ValueError('Actual entry capability report required')
    if not fixture:
        binding = report.get('actor_bindings', {})
        if (report.get('status') != 'completed' or binding.get('actor_sha256') != actor_sha
                or binding.get('actor_parameters_sha256') != parameters):
            raise ValueError('Entry capability report is not bound to the actual source Actor')


def _auxiliary_binding(binding, names, actor_sha, parameters, *, fixture):
    fields = {'version','source_collection_manifest_sha256','source_train_data_sha256',
        'source_selection_data_sha256','actor_sha256','actor_parameters_sha256',
        'observations_sha256','pair_count','source_pool','pair_filter','selection_excluded',
        'row_index_pairs_sha256','feature_names'}
    if not isinstance(binding, dict) or set(binding) != fields:
        raise ValueError('Exact current-TRAIN auxiliary provenance required')
    for key in fields:
        if key.endswith('sha256'):
            _sha(binding[key], key)
    if (binding['version'] != AUXILIARY_VERSION or binding['actor_sha256'] != actor_sha
            or binding['actor_parameters_sha256'] != parameters
            or binding['source_pool'] != 'train'
            or binding['pair_filter'] != 'physical_effect_and_nn_changed'
            or binding['selection_excluded'] is not True
            or binding['feature_names'] != names):
        raise ValueError('Auxiliary pool must use current-source TRAIN public observations only')
    _int(binding['pair_count'], 'pair_count', 1)
    if not fixture and binding['pair_count'] != 824:
        raise ValueError('Production alignment uses the fixed 824 current TRAIN pairs')


def _alignment(report, actor_sha, parameters, program_sha, origin, *, fixture):
    if not isinstance(report, dict):
        raise ValueError('Completed fixed-teacher/current-Actor alignment evidence required')
    expected_version = 'warehouse-frozen-training-teacher-later-NN-alignment-diagnostic.v1'
    labels = report.get('evaluated_label_actor_bindings', {})
    if (report.get('version') != expected_version or report.get('status') != 'completed'
            or labels.get('actor_sha256') != actor_sha
            or labels.get('actor_parameters_sha256') != parameters
            or report.get('teacher_program_semantic_sha256') != program_sha
            or report.get('new_tree_extraction') is not False
            or report.get('metadata_unchanged') is not True
            or report.get('final_test_used') is not False
            or report.get('independent_explanation_holdout_used') is not False):
        raise ValueError('Fixed-teacher alignment identity or scope differs')
    _sha(report.get('teacher_fit_result_sha256'), 'teacher_fit_result_sha256')
    _same(report.get('fixed_teacher_original_bindings'), origin, 'Frozen teacher origin changed')
    selected = report.get('results', {}).get('selection', {})
    computed = ordinary.gates(selected['all_rows'], selected['ordinary_base'])
    _same(selected.get('gates_diagnostic_only'), computed, 'Alignment ordinary gates differ')
    if not computed['ordinary_reliable']:
        raise ValueError('Frozen teacher must pass current-Actor ordinary reliability at entry')


def _source(source, fixture):
    if (type(source) is not stable.StableFeedbackTrainer or type(fixture) is not bool
            or source.test_fixture is not fixture or source.protocol.get('version') != stable.PROTOCOL_VERSION
            or source.mode != 'observed' or source.branch != 'own_credit' or source.alpha != .5
            or source.feedback_enabled or source.feedback_branch not in ('control','feedback')
            or source.cfg['actor_learning_rate'] != ACTOR_LR
            or source.cfg['critic_learning_rate'] != CRITIC_LR):
        raise ValueError('A genuine unchanged loaded StableFeedbackTrainer source is required')
    if not fixture and (source.joint_steps != 20000
            or source.source_counters['joint_steps'] + source.joint_steps != 3900000):
        raise ValueError('Production alignment starts exactly at the completed 20k/390m source')
    if source.feedback_branch == 'feedback' and not fixture and (
            source.current_lambda != MAX_LAMBDA or source.feedback_allowed is not True):
        raise ValueError('Production feedback entry must retain its already-warmed allowed teacher')
    return source.state_dict()


def make_protocol(source, *, source_checkpoint_sha256, source_state_sha256,
                  entry_actor_sha256, entry_report, teacher_alignment, auxiliary_binding,
                  cycle_id='alignment_50k_20260910', ppo_cap=50000, test_fixture=False):
    saved = _source(source, test_fixture)
    _sha(entry_actor_sha256, 'entry_actor_sha256')
    endpoints = [ppo_cap] if test_fixture else list(range(10000, 50001, 10000))
    if not test_fixture and ppo_cap != 50000:
        raise ValueError('Production alignment fixes 50k per arm')
    # _protocol is a pure field builder. Unlike native.make_protocol it does
    # not claim the Stable source is one of the old native classes.
    result = native._protocol(source, saved, checkpoint_sha=source_checkpoint_sha256,
        state_sha=source_state_sha256, ppo_cap=ppo_cap, cycle_id=cycle_id,
        evaluation_checkpoints=endpoints, fixture=test_fixture)
    result['source']['source_sha256'] = saved['source_sha256']
    result['source_lineage'][-1] = deepcopy(result['source'])
    parameters = _actor_parameters(source)
    names = list(source.envs[0].feature_names)
    branch = source.feedback_branch
    program_sha = digest(source.program.to_dict()) if source.program is not None else None
    origin = deepcopy(source.program.metadata.get('observed197_bindings')) if source.program is not None else None
    _entry_binding(entry_report, entry_actor_sha256, parameters, fixture=test_fixture)
    source_report, team_reference = source.protocol['source_report'], source.protocol['team_reference']
    entry_gate = _capability(entry_report, source_report, team_reference)
    if branch == 'feedback':
        if not entry_gate['passed'] and not test_fixture:
            raise ValueError('Feedback entry must pass the original strict capability guard')
        _auxiliary_binding(auxiliary_binding, names, entry_actor_sha256, parameters, fixture=test_fixture)
        _alignment(teacher_alignment, entry_actor_sha256, parameters, program_sha, origin, fixture=test_fixture)
    elif any(v is not None for v in (teacher_alignment, auxiliary_binding, source.program, source.teacher_fit)):
        raise ValueError('Control entry has no teacher/alignment/auxiliary data')
    result.update(version=PROTOCOL_VERSION, paired_branches=['control','feedback'],
        feedback_branch=branch, feedback_training_enabled=True,
        source_report=deepcopy(source_report), source_report_sha256=digest(source_report),
        team_reference=deepcopy(team_reference), team_reference_sha256=digest(team_reference),
        entry_actor_sha256=entry_actor_sha256, entry_actor_parameters_sha256=parameters,
        entry_report=deepcopy(entry_report), entry_report_sha256=digest(entry_report),
        entry_capability=entry_gate, feature_names=names,
        teacher_alignment=deepcopy(teacher_alignment),
        teacher_alignment_sha256=digest(teacher_alignment) if teacher_alignment is not None else None,
        auxiliary_binding=deepcopy(auxiliary_binding),
        auxiliary_binding_sha256=digest(auxiliary_binding) if auxiliary_binding is not None else None,
        teacher_program_semantic_sha256=program_sha, fixed_teacher_original_bindings=origin,
        teacher_fit_semantic_sha256=digest(source.teacher_fit) if source.teacher_fit is not None else None,
        feedback=_feedback_contract(), guard=_guard_contract(), explanation_qualification_granted=False)
    validate_protocol(result, fixture=test_fixture)
    return result


def validate_protocol(protocol, fixture=False):
    """Pure admission of this independently versioned protocol, without a source object."""
    if (not isinstance(protocol, dict) or type(fixture) is not bool
            or protocol.get('version') != PROTOCOL_VERSION or protocol.get('test_fixture') is not fixture):
        raise ValueError('Explicit alignment protocol and fixture scope required')
    fields={'version','test_fixture','cycle_id','branch','shutdown_arm','own_shutdown_beta',
        'own_shutdown_reward_version','public_feedback_version','public_feedback_mode',
        'credit_reward_version','delivery_credit_alpha','training','reward','collision_training_cost',
        'seed','partners_by_branch','source','source_lineage','source_protocol','feedback_training_enabled',
        'budget','evaluation','continuation','paired_branches','feedback_branch','source_report',
        'source_report_sha256','team_reference','team_reference_sha256','entry_actor_sha256',
        'entry_actor_parameters_sha256','entry_report','entry_report_sha256','entry_capability',
        'feature_names','teacher_alignment','teacher_alignment_sha256','auxiliary_binding',
        'auxiliary_binding_sha256','teacher_program_semantic_sha256','fixed_teacher_original_bindings',
        'teacher_fit_semantic_sha256','feedback','guard','explanation_qualification_granted'}
    if set(protocol)!=fields:
        raise ValueError('Exact independent alignment protocol fields required')
    original, source = protocol.get('source_protocol', {}), protocol.get('source', {})
    if (original.get('version') != stable.PROTOCOL_VERSION or original.get('test_fixture') is not fixture
            or source.get('trainer_version') != stable.VERSION
            or source.get('protocol_sha256') != digest(original)
            or protocol.get('source_lineage', [])[-1:] != [source]):
        raise ValueError('Source must retain its actual Stable identity and protocol')
    source_evaluation._validate_protocol(original,fixture=fixture)
    if (protocol['source_lineage'][:-1]!=original.get('source_lineage')
            or original.get('source',{}).get('cumulative_joint_steps',-1)+source.get('joint_steps',-1)
                !=source.get('cumulative_joint_steps')):
        raise ValueError('Genuine Stable source closure or cumulative lineage differs')
    for key in ('checkpoint_sha256','state_sha256','protocol_sha256','source_sha256'):
        _sha(source.get(key), 'source.'+key)
    for key in ('joint_steps','cumulative_joint_steps'):
        _int(source.get(key), 'source.'+key)
    if not fixture and (source['joint_steps'] != 20000 or source['cumulative_joint_steps'] != 3900000):
        raise ValueError('Production source boundary differs')
    if (type(protocol.get('cycle_id')) is not str
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',protocol['cycle_id'])):
        raise ValueError('Invalid finite cycle identity')
    for key in ('training','reward','collision_training_cost','seed','partners_by_branch',
                'branch','shutdown_arm','own_shutdown_beta','own_shutdown_reward_version',
                'public_feedback_version','public_feedback_mode','credit_reward_version','delivery_credit_alpha'):
        _same(protocol.get(key), original.get(key), 'Unchanged physical/learning source contract differs: '+key)
    if (protocol['training'].get('actor_learning_rate') != ACTOR_LR
            or protocol['training'].get('critic_learning_rate') != CRITIC_LR
            or protocol.get('branch') != 'own_credit' or protocol.get('public_feedback_mode') != 'observed'
            or protocol.get('delivery_credit_alpha') != .5):
        raise ValueError('Original observed-input/rate/reward contract differs')
    cap = _int(protocol.get('budget',{}).get('maximum_ppo_joint_steps'),'PPO cap',1)
    nenv = _int(protocol['training'].get('environments'), 'environments', 1)
    if cap % nenv:
        raise ValueError('Finite budget must use complete environment batches')
    _same(protocol['budget'],dict(maximum_ppo_joint_steps_per_arm=cap,maximum_ppo_joint_steps=cap,
        curriculum_generation_steps=0),'Finite budget differs')
    endpoints=[cap] if fixture else list(range(10000,50001,10000))
    if not fixture and cap != 50000:
        raise ValueError('Production alignment fixes 50k per arm')
    expected_evaluation=dict(checkpoints_ppo_steps=endpoints,partners=['skilled','assertive','noisy'],
        scenarios_per_partner=50,deterministic=True,horizon=120,
        maximum_environment_steps=len(endpoints)*3*50*120,reuse_source_zero_step_validation=True,
        read_final_test=False,training_credit_in_evaluation=False,training_shutdown_in_evaluation=False)
    _same(protocol.get('evaluation'),expected_evaluation,'Original finite validation matrix differs')
    _same(protocol.get('continuation'),dict(inflight_episodes='preserve_exactly',rng='resume_without_reseeding',
        learning_configuration='inherit_unchanged',critic_structure='unchanged'),'Continuation preservation differs')
    _same(protocol.get('feedback'),_feedback_contract(),'Fixed teacher update contract differs')
    _same(protocol.get('guard'),_guard_contract(),'Capability/feedback latch contract differs')
    if (protocol.get('paired_branches') != ['control','feedback']
            or protocol.get('feedback_branch') not in ('control','feedback')
            or protocol.get('feedback_training_enabled') is not True
            or protocol.get('explanation_qualification_granted') is not False):
        raise ValueError('Paired branch or qualification scope differs')
    names=protocol.get('feature_names')
    if (type(names) is not list or len(names)!=197 or any(type(n) is not str or not n for n in names)
            or len(set(names))!=197):
        raise ValueError('Exact unique public observed197 names required')
    for key in ('source_report','team_reference','entry_report'):
        if digest(protocol.get(key)) != protocol.get(key+'_sha256'):
            raise ValueError('Bound report differs: '+key)
    for key in ('source_report','team_reference'):
        _same(protocol[key],original.get(key),'Common 388m capability anchor changed')
    actor_sha=_sha(protocol.get('entry_actor_sha256'),'entry_actor_sha256')
    parameters=_sha(protocol.get('entry_actor_parameters_sha256'),'entry_actor_parameters_sha256')
    _entry_binding(protocol['entry_report'],actor_sha,parameters,fixture=fixture)
    gate=_capability(protocol['entry_report'],protocol['source_report'],protocol['team_reference'])
    _same(protocol.get('entry_capability'),gate,'Entry capability guard differs')
    if protocol['feedback_branch']=='feedback':
        if not fixture and not gate['passed']:
            raise ValueError('Feedback source failed the original capability guard')
        for key in ('teacher_program_semantic_sha256','teacher_fit_semantic_sha256'):
            _sha(protocol.get(key),key)
        _auxiliary_binding(protocol.get('auxiliary_binding'),names,actor_sha,parameters,fixture=fixture)
        _alignment(protocol.get('teacher_alignment'),actor_sha,parameters,
            protocol['teacher_program_semantic_sha256'],protocol['fixed_teacher_original_bindings'],fixture=fixture)
        for key in ('teacher_alignment','auxiliary_binding'):
            if digest(protocol[key])!=protocol.get(key+'_sha256'):
                raise ValueError('Current teacher/auxiliary binding differs')
    elif any(protocol.get(k) is not None for k in ('teacher_alignment','teacher_alignment_sha256',
            'auxiliary_binding','auxiliary_binding_sha256','teacher_program_semantic_sha256',
            'fixed_teacher_original_bindings','teacher_fit_semantic_sha256')):
        raise ValueError('Control protocol must not claim teacher or auxiliary admission')
    return deepcopy(protocol)


class AlignmentTrainer(native.ShutdownContinuationTrainer):
    def __init__(self, protocol, source, *, source_checkpoint_sha256,
                 expected_source_state_sha256, auxiliary_observations=None,
                 device='mps', test_fixture=False):
        saved=_source(source,test_fixture)
        if torch.device(device)!=source.device:
            raise ValueError('Continuation requires the actual source training device')
        expected=make_protocol(source,source_checkpoint_sha256=source_checkpoint_sha256,
            source_state_sha256=expected_source_state_sha256,entry_actor_sha256=protocol['entry_actor_sha256'],
            entry_report=protocol['entry_report'],teacher_alignment=protocol['teacher_alignment'],
            auxiliary_binding=protocol['auxiliary_binding'],cycle_id=protocol['cycle_id'],
            ppo_cap=protocol['budget']['maximum_ppo_joint_steps'],test_fixture=test_fixture)
        _same(protocol,expected,'Alignment protocol differs from the actual source')
        independent=deepcopy(source)
        # Stages independent model/Adam tensors through the real unchanged old
        # Stable loader before installing the new protocol. No old-schema alias.
        stable.StableFeedbackTrainer.load_state_dict(independent,saved)
        self.__dict__.update(independent.__dict__)
        self.protocol,self.cfg=deepcopy(protocol),deepcopy(protocol['training'])
        self.sources=native.execution_sources()
        self.alignment_sources=execution_sources()
        self.initialization_sha256=expected_source_state_sha256
        self.source_checkpoint_sha256=source_checkpoint_sha256
        self.cycle_id,self.source_lineage=protocol['cycle_id'],deepcopy(protocol['source_lineage'])
        self.source_parent_counters={key:saved[key] for key in native.COUNTERS}
        self.source_counters={key:source.source_counters[key]+saved[key]
            for key in ('joint_steps','optimizer_updates','minibatch_updates','actor_optimizer_steps','critic_optimizer_steps')}
        self.source_frames=[env.state.frame for env in self.envs]
        self.source_completed_episodes_sha256=semantic(saved['completed_episodes'])
        self.source_round_episode_prefixes_sha256=semantic(saved['round_episode_prefixes'])
        self.round_episode_prefixes=[native._prefix(env,self.episode_returns[i],self.episode_reward_components[i])
            for i,env in enumerate(self.envs)]
        self.completed_episodes=[]
        for key in native.COUNTERS:
            setattr(self,key,0)
        self.elapsed_seconds,self.feedback_enabled,self._collecting=0.,False,False
        self.feedback_branch=protocol['feedback_branch']
        self.teacher_fit=deepcopy(source.teacher_fit)
        self.program=ExecutableProgram.from_dict(source.program.to_dict()) if source.program is not None else None
        self.auxiliary_observations=np.empty((0,2,197),np.float32)
        if self.feedback_branch=='feedback':
            if digest(self.teacher_fit)!=protocol['teacher_fit_semantic_sha256']:
                raise ValueError('Original frozen teacher fit differs')
            manager=stable.teacher.validate_manager_state(self.teacher_fit['manager_state'],
                feature_names=self.envs[0].feature_names,actor_bindings=self.teacher_fit['fit_report']['actor_bindings'],
                step=self.teacher_fit['fit_report']['step'],expected_evidence_sha256=self.teacher_fit['evidence_sha256'])
            _same(manager.program.to_dict(),self.program.to_dict(),'Original frozen program differs from original teacher state')
            aux=np.asarray(auxiliary_observations)
            if (aux.dtype!=np.float32 or aux.ndim!=3 or aux.shape[1:]!=(2,197)
                    or len(aux)!=protocol['auxiliary_binding']['pair_count'] or not np.isfinite(aux).all()
                    or semantic(aux)!=protocol['auxiliary_binding']['observations_sha256']):
                raise ValueError('Current actual TRAIN paired observation content differs')
            self.auxiliary_observations=aux.copy()
        elif auxiliary_observations is not None:
            aux=np.asarray(auxiliary_observations)
            if aux.dtype!=np.float32 or aux.shape!=(0,2,197):
                raise ValueError('Control auxiliary observations must be absent or exactly empty')
        self.auxiliary_observations.setflags(write=False)
        # auxiliary_rng was copied from the genuine source, never reseeded.
        self.feedback_allowed=self.feedback_branch=='feedback'
        self.current_lambda=MAX_LAMBDA if self.feedback_allowed else 0.
        self.last_capability=deepcopy(protocol['entry_capability'])
        self.capability_history=[]
        self.teacher_reliability_history=[]
        self.initial_learning_state_sha256=semantic(dict(model=native._cpu(self.model.state_dict()),
            optimizers=native._cpu(self.optimizers.state_dict()),rng=self.rng.bit_generator.state,
            owned_rng=self._owned_rng_state,envs=[env.snapshot() for env in self.envs],
            auxiliary_rng=self.auxiliary_rng.bit_generator.state))
        self._verify_alignment()

    def _verify_alignment(self):
        if self.alignment_sources!=execution_sources() or self.sources!=native.execution_sources():
            raise ValueError('Alignment execution sources changed')
        for role,rate in (('actor',ACTOR_LR),('critic',CRITIC_LR)):
            if self.cfg[role+'_learning_rate']!=rate or any(g['lr']!=rate for g in getattr(self.optimizers,role).param_groups):
                raise ValueError('Inherited actual optimizer rates changed')
        if self.feedback_enabled is not False:
            raise ValueError('Native collector feedback must remain disabled')
        if self.feedback_branch=='feedback':
            if (digest(self.program.to_dict())!=self.protocol['teacher_program_semantic_sha256']
                    or digest(self.teacher_fit)!=self.protocol['teacher_fit_semantic_sha256']
                    or semantic(self.auxiliary_observations)!=self.protocol['auxiliary_binding']['observations_sha256']):
                raise ValueError('Frozen original teacher or current auxiliary observations changed')
        elif self.program is not None or self.teacher_fit is not None or self.auxiliary_observations.shape!=(0,2,197):
            raise ValueError('Control must preserve its empty feedback state')
        if (type(self.feedback_allowed) is not bool
                or self.current_lambda!=(MAX_LAMBDA if self.feedback_allowed else 0.)
                or self.feedback_branch=='control' and self.feedback_allowed):
            raise ValueError('Feedback lambda/latched admission differs')

    def _bindings(self):
        self._verify_alignment()
        base=native.ShutdownContinuationTrainer._bindings(self)
        return {**base,'version':VERSION,'sources':deepcopy(self.alignment_sources),
            'source_sha256':digest(self.alignment_sources),'engine_sources':deepcopy(self.sources),
            'feedback_branch':self.feedback_branch,'feedback_enabled':self.feedback_branch=='feedback',
            'native_collector_feedback_enabled':False,
            'initial_learning_state_sha256':self.initial_learning_state_sha256,
            'entry_actor_sha256':self.protocol['entry_actor_sha256'],
            'entry_actor_parameters_sha256':self.protocol['entry_actor_parameters_sha256'],
            'entry_report_sha256':self.protocol['entry_report_sha256'],
            'teacher_alignment_sha256':self.protocol['teacher_alignment_sha256'],
            'auxiliary_binding_sha256':self.protocol['auxiliary_binding_sha256'],
            'teacher_fit_sha256':self.protocol['teacher_fit_semantic_sha256'],
            'program_sha256':self.protocol['teacher_program_semantic_sha256'],
            'auxiliary_observations_sha256':semantic(self.auxiliary_observations)}

    def state_dict(self):
        state=native.ShutdownContinuationTrainer.state_dict(self)
        state['alignment_material']=dict(program=self.program.to_dict() if self.program else None,
            teacher_fit=deepcopy(self.teacher_fit),auxiliary_observations=self.auxiliary_observations.copy(),
            auxiliary_binding=deepcopy(self.protocol['auxiliary_binding']),
            teacher_alignment=deepcopy(self.protocol['teacher_alignment']))
        state['alignment_feedback_state']=dict(current_lambda=self.current_lambda,allowed=self.feedback_allowed,
            last_capability=deepcopy(self.last_capability),capability_history=deepcopy(self.capability_history),
            teacher_reliability_history=deepcopy(self.teacher_reliability_history),
            auxiliary_rng=deepcopy(self.auxiliary_rng.bit_generator.state))
        return state

    def _teacher_check(self, report, *, expected_step=None, expected_parameters=None):
        required={'version','status','passed','actor_parameters_sha256','teacher_program_semantic_sha256',
            'protocol_sha256','joint_steps','selection_data_sha256','ordinary_metrics','ordinary_gate','evidence_sha256'}
        if (not isinstance(report,dict) or not required.issubset(report)
                or report['version']!=TEACHER_CHECK_VERSION or report['status']!='completed'
                or type(report['passed']) is not bool or self.feedback_branch!='feedback'):
            raise ValueError('Exact completed independent teacher-check report required')
        for key in ('actor_parameters_sha256','teacher_program_semantic_sha256','protocol_sha256',
                    'selection_data_sha256','evidence_sha256'):
            _sha(report[key],key)
        if (report['teacher_program_semantic_sha256']!=self.protocol['teacher_program_semantic_sha256']
                or report['protocol_sha256']!=digest(self.protocol)
                or report['selection_data_sha256']!=self.protocol['auxiliary_binding']['source_selection_data_sha256']):
            raise ValueError('Teacher check does not bind the frozen program/protocol/development selection')
        step=_int(report['joint_steps'],'teacher check step')
        if step not in self.protocol['evaluation']['checkpoints_ppo_steps']:
            raise ValueError('Teacher reliability check requires a registered boundary')
        if expected_step is not None and step!=expected_step:
            raise ValueError('Teacher check is stale')
        if expected_parameters is not None and report['actor_parameters_sha256']!=expected_parameters:
            raise ValueError('Teacher check used a different current Actor')
        gate=ordinary._fidelity_gate(report['ordinary_metrics'])
        _same(report['ordinary_gate'],gate,'Teacher ordinary threshold computation differs')
        if report['passed'] is not gate['passed']:
            raise ValueError('Teacher check passed flag differs from original ordinary gates')
        return gate

    def load_state_dict(self,payload):
        self._verify_alignment()
        expected_material=dict(program=self.program.to_dict() if self.program else None,
            teacher_fit=self.teacher_fit,auxiliary_observations=self.auxiliary_observations,
            auxiliary_binding=self.protocol['auxiliary_binding'],teacher_alignment=self.protocol['teacher_alignment'])
        _same(payload.get('alignment_material'),expected_material,'Persisted teacher/auxiliary material differs')
        runtime=deepcopy(payload.get('alignment_feedback_state'))
        keys={'current_lambda','allowed','last_capability','capability_history','teacher_reliability_history','auxiliary_rng'}
        if (not isinstance(runtime,dict) or set(runtime)!=keys or type(runtime['allowed']) is not bool
                or type(runtime['current_lambda']) not in (int,float)
                or runtime['current_lambda']!=(MAX_LAMBDA if runtime['allowed'] else 0.)
                or self.feedback_branch=='control' and runtime['allowed']):
            raise ValueError('Invalid saved alignment feedback latch')
        failed=False
        last=deepcopy(self.protocol['entry_capability'])
        for name in ('capability_history','teacher_reliability_history'):
            history=runtime[name]
            if not isinstance(history,list):
                raise ValueError('Saved reliability/capability journal must be a list')
            previous=-1
            for entry in history:
                if not isinstance(entry,dict) or set(entry)!={'joint_steps','report','report_sha256','result'}:
                    raise ValueError('Invalid saved admission journal entry')
                step=_int(entry['joint_steps'],'journal step')
                if step<=previous or step>payload['joint_steps'] or step not in self.protocol['evaluation']['checkpoints_ppo_steps']:
                    raise ValueError('Admission history boundary/order differs')
                previous=step
                if digest(entry['report'])!=entry['report_sha256']:
                    raise ValueError('Original admission report digest differs')
                if name=='teacher_reliability_history':
                    actual=self._teacher_check(entry['report'],expected_step=step)
                else:
                    actual=_capability(entry['report'],self.protocol['source_report'],self.protocol['team_reference'])
                    last=actual
                _same(entry['result'],actual,'Saved admission result differs')
                failed=failed or not actual['passed']
        expected_allowed=self.feedback_branch=='feedback' and not failed
        if runtime['allowed'] is not expected_allowed:
            raise ValueError('Persisted feedback state violates the irreversible failure latch')
        _same(runtime['last_capability'],last,'Saved last capability differs from report history')
        rng=np.random.default_rng();rng.bit_generator.state=runtime['auxiliary_rng']
        native.ShutdownContinuationTrainer.load_state_dict(self,payload)
        self.current_lambda=runtime['current_lambda'];self.feedback_allowed=runtime['allowed']
        self.last_capability=runtime['last_capability'];self.capability_history=runtime['capability_history']
        self.teacher_reliability_history=runtime['teacher_reliability_history'];self.auxiliary_rng=rng
        self._verify_alignment()

    def collect(self,time_steps):
        self._verify_alignment()
        batch=native.ShutdownContinuationTrainer.collect(self,time_steps)
        for row in batch['transition_records']:
            row.update(version=VERSION,feedback_branch=self.feedback_branch,feedback_lambda=self.current_lambda)
        return batch

    def update(self,batch):
        self._verify_alignment()
        return updater.stable_ppo_update(self,batch,self.program,lambda_value=self.current_lambda,
            auxiliary_observations=self.auxiliary_observations,auxiliary_rng=self.auxiliary_rng)

    def native_ppo_update(self,batch):
        return native.ShutdownContinuationTrainer.update(self,batch)

    def set_capability(self,report):
        self._verify_alignment()
        if self.joint_steps not in self.protocol['evaluation']['checkpoints_ppo_steps']:
            raise ValueError('Capability check requires a registered current boundary')
        if self.capability_history and self.capability_history[-1]['joint_steps']>=self.joint_steps:
            raise ValueError('A capability boundary is already recorded')
        if not self.test_fixture:
            binding=report.get('actor_bindings',{})
            if (report.get('status')!='completed' or binding.get('actor_parameters_sha256')!=_actor_parameters(self)
                    or binding.get('protocol_sha256')!=digest(self.protocol)
                    or binding.get('joint_steps')!=self.joint_steps):
                raise ValueError('Capability report does not bind the actual current checkpoint')
        value=_capability(report,self.protocol['source_report'],self.protocol['team_reference'])
        self.last_capability=value
        self.capability_history.append(dict(joint_steps=self.joint_steps,report=deepcopy(report),
            report_sha256=digest(report),result=deepcopy(value)))
        if not value['passed']:
            self.feedback_allowed=False;self.current_lambda=0.
        return bool(value['passed'])

    def set_teacher_reliability(self,report):
        self._verify_alignment()
        if self.teacher_reliability_history and self.teacher_reliability_history[-1]['joint_steps']>=self.joint_steps:
            raise ValueError('A teacher reliability boundary is already recorded')
        result=self._teacher_check(report,expected_step=self.joint_steps,expected_parameters=_actor_parameters(self))
        self.teacher_reliability_history.append(dict(joint_steps=self.joint_steps,report=deepcopy(report),
            report_sha256=digest(report),result=deepcopy(result)))
        if not result['passed']:
            self.feedback_allowed=False;self.current_lambda=0.
        return bool(result['passed'])

    def export(self,path):
        self._verify_alignment()
        metadata=self._bindings()
        for key in ('version','sources','protocol','engine_sources','source_frames','source_completed_episodes_sha256'):
            metadata.pop(key,None)
        metadata.update(experiment_version=VERSION,joint_steps=self.joint_steps,
            optimizer_updates=self.optimizer_updates,minibatch_updates=self.minibatch_updates,
            actor_parameters_sha256=_actor_parameters(self),feedback_lambda=self.current_lambda,
            feedback_allowed=self.feedback_allowed,initialization='genuine_stable_loader_then_independent_alignment_fork',
            candidate=True,explanation_qualified=False,release_ready=False,
            teacher_reliability_report_hashes=[r['report_sha256'] for r in self.teacher_reliability_history])
        return self.model.export_npz(path,metadata)
