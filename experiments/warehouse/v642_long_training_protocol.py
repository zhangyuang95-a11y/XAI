"""Long-budget protocol registration, not a seed or execution capability.

No model, environment, RNG, seed inventory or output file is constructed.
The real 20k calibration must independently replay before symbolic preparation
can be registered. Numerical declarations below cannot substitute for future
source-bound long-run producers and independent auditors.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import math

from backend.training.warehouse_v625_actor_training import PPOConfig
from . import v642_calibration_admission as calibration
from .v626_protocol import require, pointer, bound, digest, file_sha, ROOT

VERSION='warehouse-v642-long-training-protocol.v1'
REFERENCE_SHA256='1b6310f6d93309fb17e64516519a71680ad3c30e73f222be27df4dff10da81ec'
REFERENCE_ROLES=(('robot_1','7aee44859397b8cbe1bb9cd3b8541e82f2a080d884c83404fe1e4fd2a998eaf0'),
                 ('robot_2','17f2f75877421854c680fab07557f031389800a9503b14f060db5ff4f862fcf0'))
ROLES=tuple(role for role,_ in REFERENCE_ROLES)
VARIANTS=('raw','regularized')
BUDGETS=(200000,300000,400000)
DEVELOPMENT_STREAMS=('development-0','development-1','development-2')
FORMAL_STREAMS=('formal-0','formal-1','formal-2')
_TOKEN=object()


@dataclass(frozen=True)
class Protocol:
    version:str=field(default=VERSION,init=False)

    def payload(self):
        return dict(version=self.version,reference_actor_sha256=REFERENCE_SHA256,
            reference_role_sha256=dict(REFERENCE_ROLES),development_budgets=list(BUDGETS),
            development_streams=list(DEVELOPMENT_STREAMS),formal_streams=list(FORMAL_STREAMS),
            streams_are_continuations_from_one_complete_v620_BC_not_independent_BC_reproductions=True,
            every_budget_restarts_from_common_reference_at_joint_step_zero=True,
            paired_initialization=['complete_dual_Actor','critic','Actor_optimizer','critic_optimizer',
                                   'initialization_random_stream','optimizer_random_stream','scenario_sequence'],
            independent_streams_require_disjoint_initialization_optimizer_scenario_streams=True,
            config_by_budget={str(b):asdict(PPOConfig(joint_step_budget=b)) for b in BUDGETS},
            extraction_steps_by_budget={str(b):list(PPOConfig(joint_step_budget=b).extraction_steps) for b in BUDGETS},
            feedback_window=dict(start_fraction=.80,full_fraction=.85,maximum_coefficient=.01,
                                 normalization='all_physical_frames',raw_external_program_gradient=0),
            critic_warmup_joint_steps=4000,reference_KL_common_to_both_variants=True,
            true_complete_episode_shutdown_cost=True,
            variants_differ_only_in_external_program_feedback=True,
            program_source_episode_counts=dict(fit=900,calibration=150,eligibility=300,validation=300),
            evidence_scenes_excluded_from_NN_gradient_buffers=True,
            program=dict(method='warehouse-v642-fixed-language-protected-descendant-refit.v1',
                         maximum_depth=12,maximum_leaves=256,maximum_public_sources_per_node_or_leaf=3,
                         exact_confirmed_fold_zero_structure_and_support_lock_required=True,
                         program_never_selects_environment_actions=True),
            development_gates=dict(deliveries=5.5,reference_delivery_fraction=.95,collisions=6.,
                reference_collision_increase=1.,shutdown_rate=.05,feedback_coverage=.45,
                explanation_coverage=.45,expected_feedback_coverage=.40,feedback_agreement=.95,
                explanation_agreement=.97,KL=.20,illegal_actions=0,unproved_paths=0,
                regularized_positive_actual_updates_required_for_each_role=True),
            budget_selection='first_budget_where_all_three_complete_pairs_pass_every_gate',
            checkpoint_and_pair_selection='maximize_minimum_raw_regularized_quality_only',
            quality_score='deliveries_per_episode - .25*collisions_per_episode - 5*shutdown_rate',
            quality_tie_break='lexicographic_stream_id_then_checkpoint_id',
            formal_quality_checkpoint_pool=['extract-84','extract-93','final'],
            formal_quality_pool_locked_before_formal_training=True,
            critic_warmup_and_pre_feedback_checkpoints_excluded_from_formal_quality_pool=True,
            no_pair_reselection_after_program_prescreen=True,
            formal_gates=dict(deliveries=5.,collisions=8.,shutdown_rate=.10,feedback_coverage=.20,
                explanation_coverage=.20,feedback_agreement=.85,explanation_agreement=.90,KL=.35,
                illegal_actions=0,unproved_paths=0),
            sealed_order=['fixed_quality_selected_pair','program_prescreen','once_independent_program_audit',
                          'once_100_fresh_scenes_per_task_holdout','three_domain_binding_and_release_audit'],
            sealed_failure_or_interruption_closes_candidate=True,
            engineering_release_gate=dict(maximum_checkpoint_overhead_fraction=.10,blocks_model_experiments=False),
            training_authority=False,seed_allocation_authority=False,sealed_authority=False,
            model_acceptance_passed=False,pilot_opened=False,executor_implemented=False)


def source_files():
    files=calibration.collection.source_files()
    names=('experiments/warehouse/v642_long_training_protocol.py',
           'tests/test_warehouse_v642_long_training_protocol.py','docs/warehouse_v642_long_training_protocol.md',
           'experiments/warehouse/v642_calibration_admission.py')
    files.update({name:file_sha(ROOT/name) for name in names})
    return dict(sorted(files.items()))


@dataclass(frozen=True,init=False)
class PreparationRegistration:
    """A verified protocol declaration, explicitly insufficient to run training."""
    value:dict
    refs:tuple
    payload_sha256:str
    _token:object

    def __init__(self,*args,**kwargs):
        raise RuntimeError('register only after actual independent calibration replay')

    def verify(self):
        require(self._token is _TOKEN and digest(self.value)==self.payload_sha256
                and self.value['protocol']==Protocol().payload() and self.value['source_files']==source_files()
                and self.value['training_authority'] is self.value['seed_allocation_authority'] is False,
                'long-training registration, source or authority changed')
        for ref in self.refs:
            require(pointer(ref['path'])==ref,'long-training calibration prerequisite changed')
        return self


def register_preparation(calibration_precheck_ref):
    """Replay real calibration admission before any symbolic phase declaration.

    A supplied JSON `passed` flag cannot open this gate: invoke the existing
    independent NN/physics/program auditor and require its complete output.
    The caller may later persist this declaration via an explicit new command.
    This function itself writes nothing and grants no seed proposal capability.
    """
    require(pointer(calibration_precheck_ref['path'])==calibration_precheck_ref,
            'missing or changed independent 20k calibration receipt')
    recorded=bound(calibration_precheck_ref)
    require(recorded.get('version')==calibration.VERSION
            and recorded.get('calibration_precheck_passed') is True
            and recorded.get('model_acceptance_passed') is recorded.get('pilot_opened') is False,
            'real 20k task retention and both-role program precheck required before preparation')
    sources=source_files()
    replayed=calibration.audit(recorded['quality_result'],recorded['supervisor_exit'])
    require(replayed==recorded,'independent calibration admission did not exactly replay')
    refs=[deepcopy(calibration_precheck_ref),recorded['quality_result'],recorded['supervisor_exit'],
          *recorded['input_asset_refs']]
    unique={}
    for ref in refs:
        require(ref['path'] not in unique or unique[ref['path']]==ref,'conflicting calibration asset binding')
        unique[ref['path']]=deepcopy(ref)
    value=dict(version=VERSION,protocol=Protocol().payload(),calibration_precheck=deepcopy(calibration_precheck_ref),
               independent_calibration_replay_passed=True,source_files=sources,
               symbolic_preparation_registered=True,training_authority=False,seed_allocation_authority=False,
               sealed_authority=False,model_acceptance_passed=False,pilot_opened=False)
    out=object.__new__(PreparationRegistration)
    for key,item in dict(value=value,refs=tuple(unique[k] for k in sorted(unique)),
                         payload_sha256=digest(value),_token=_TOKEN).items():
        object.__setattr__(out,key,item)
    return out.verify()


def _sha(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def validate_pair_declaration(value,*,budget,phase):
    """Pure validation of future auditor output; not proof that a run occurred."""
    require(phase in ('development','formal') and budget in BUDGETS,'unregistered phase or budget')
    expected={'stream_id','checkpoint_id','joint_step_budget','config','start_joint_step','end_joint_steps',
              'initialization','quality_cases_sha256','quality','reference_quality','programs','program_gradient_updates'}
    require(type(value) is dict and set(value)==expected,'unknown or incomplete paired-run declaration')
    streams=DEVELOPMENT_STREAMS if phase=='development' else FORMAL_STREAMS
    require(value['stream_id'] in streams and type(value['checkpoint_id']) is str and value['checkpoint_id']
            and type(value['joint_step_budget']) is int and value['joint_step_budget']==budget
            and value['config']==asdict(PPOConfig(joint_step_budget=budget))
            and type(value['start_joint_step']) is int and value['start_joint_step']==0
            and value['end_joint_steps']=={variant:budget for variant in VARIANTS},
            'budget must restart at the common reference; no cross-budget continuation')
    init=value['initialization']; keys={'reference_actor_sha256','actor_role_sha256','critic_sha256',
        'Actor_optimizer_sha256','critic_optimizer_sha256','initialization_random_stream_sha256',
        'optimizer_random_stream_sha256','scenario_sequence_sha256'}
    require(set(init)==set(VARIANTS) and all(set(v)==keys for v in init.values())
            and init['raw']==init['regularized'],'paired Actor, critic, optimizers or random/scenario streams differ')
    for item in init.values():
        require(item['reference_actor_sha256']==REFERENCE_SHA256
                and item['actor_role_sha256']==dict(REFERENCE_ROLES)
                and all(_sha(v) for k,v in item.items() if k!='actor_role_sha256'),
                'each stream must begin at the exact complete v620 dual Actor')
    require(_sha(value['quality_cases_sha256']) and set(value['quality'])==set(VARIANTS)
            and set(value['programs'])==set(VARIANTS) and set(value['program_gradient_updates'])==set(VARIANTS),
            'missing common quality cohort, variants or program diagnostics')
    reference=_quality(value['reference_quality'])
    quality={variant:_quality(value['quality'][variant]) for variant in VARIANTS}
    require(all(q['episodes']==reference['episodes'] for q in quality.values()),'unpaired quality scene counts')
    quality_passed={variant:_quality_pass(q,reference,phase) for variant,q in quality.items()}
    program_passed={}; gradient_passed={}
    for variant in VARIANTS:
        require(set(value['programs'][variant])==set(ROLES)
                and set(value['program_gradient_updates'][variant])==set(ROLES),'missing role-specific evidence')
        program_passed[variant]={role:_program_pass(value['programs'][variant][role],phase) for role in ROLES}
        counts=value['program_gradient_updates'][variant]
        require(all(type(n) is int and n>=0 for n in counts.values()),'program gradient counts must be actual nonnegative integers')
        gradient_passed[variant]=all(n==0 for n in counts.values()) if variant=='raw' else all(n>0 for n in counts.values())
    return dict(version=VERSION,scope='declaration_validation_only_not_execution_evidence',
        quality_passed=quality_passed,program_passed=program_passed,actual_gradient_counts_passed=gradient_passed,
        all_declared_gates_passed=all(quality_passed.values()) and all(gradient_passed.values())
            and all(all(v.values()) for v in program_passed.values()),training_authority=False,model_acceptance_passed=False)


def _quality(value):
    require(type(value) is dict and set(value)=={'episodes','deliveries','collisions','shutdown_episodes'}
            and all(type(v) is int and v>=0 for v in value.values()) and value['episodes']>0
            and value['shutdown_episodes']<=value['episodes'],'quality requires complete-episode integer totals')
    return value


def _quality_pass(q,reference,phase):
    n=q['episodes']; r=reference['episodes']
    if phase=='formal':
        return q['deliveries']>=5*n and q['collisions']<=8*n and 10*q['shutdown_episodes']<=n
    return (2*q['deliveries']>=11*n and 100*q['deliveries']*r>=95*reference['deliveries']*n
            and q['collisions']<=6*n and q['collisions']*r<=(reference['collisions']+r)*n
            and 20*q['shutdown_episodes']<=n)


def _program_pass(value,phase):
    keys={'feedback_coverage','explanation_coverage','expected_feedback_coverage','feedback_agreement',
          'explanation_agreement','feedback_KL','explanation_KL','illegal_actions','unproved_paths'}
    require(type(value) is dict and set(value)==keys
            and all(type(v) in (int,float) and math.isfinite(v) for v in value.values()),'malformed program metrics')
    require(all(0<=value[k]<=1 for k in keys if k.endswith('coverage') or k.endswith('agreement'))
            and value['feedback_KL']>=0 and value['explanation_KL']>=0
            and all(type(value[k]) is int and value[k]>=0 for k in ('illegal_actions','unproved_paths')),
            'invalid program distribution, coverage or legality counts')
    if phase=='development':
        bounds=(.45,.45,.40,.95,.97,.20)
    else:
        bounds=(.20,.20,0.,.85,.90,.35)
    f,e,x,fa,ea,kl=bounds
    return (value['feedback_coverage']>=f and value['explanation_coverage']>=e
            and value['expected_feedback_coverage']>=x and value['feedback_agreement']>=fa
            and value['explanation_agreement']>=ea and value['feedback_KL']<=kl
            and value['explanation_KL']<=kl and value['illegal_actions']==value['unproved_paths']==0)


def development_budget_decision(completed_declarations):
    """First all-three pass, with failed lower budgets retained in order.

    Caller-supplied declarations never grant formal training or sealed access.
    Future actual budget-lock issuance still needs source-bound run replay.
    """
    require(type(completed_declarations) is list and len(completed_declarations)<=3,'invalid budget history')
    common_starts={}
    for index,group in enumerate(completed_declarations):
        budget=BUDGETS[index]
        require(type(group) is list and len(group)==3
                and {row['stream_id'] for row in group}==set(DEVELOPMENT_STREAMS),
                'each attempted budget requires all three complete development streams')
        results=[validate_pair_declaration(row,budget=budget,phase='development') for row in group]
        for row in group:
            initial={key:value for key,value in row['initialization']['raw'].items()
                     if key!='scenario_sequence_sha256'}
            require(row['stream_id'] not in common_starts or common_starts[row['stream_id']]==initial,
                    'each budget must restart the same stream-specific Actor, critic, optimizers and RNG state')
            common_starts[row['stream_id']]=deepcopy(initial)
        require(len({row['quality_cases_sha256'] for row in group})==1,'development streams need the same quality cohort')
        for key in ('initialization_random_stream_sha256','optimizer_random_stream_sha256','scenario_sequence_sha256'):
            require(len({row['initialization']['raw'][key] for row in group})==3,
                    'three continuation streams must be independently registered')
        if all(row['all_declared_gates_passed'] for row in results):
            require(index==len(completed_declarations)-1,'stop at the first all-three passing budget')
            return dict(version=VERSION,declared_budget=budget,next_budget=None,
                        requires_independent_long_run_replay=True,training_authority=False,
                        formal_preparation_authority=False,model_acceptance_passed=False)
    next_budget=BUDGETS[len(completed_declarations)] if len(completed_declarations)<3 else None
    return dict(version=VERSION,declared_budget=None,next_budget=next_budget,
                candidate_closed=next_budget is None,training_authority=False,formal_preparation_authority=False,
                model_acceptance_passed=False)


def select_quality_only(rows):
    """Rank fixed quality-only declarations; programs cannot enter the API."""
    require(type(rows) is list and rows,'quality-only selector requires a complete candidate table')
    keys={'stream_id','checkpoint_id','quality_cases_sha256','quality'}
    seen=set(); scores={}; cohorts=set(); streams=set()
    for row in rows:
        require(type(row) is dict and set(row)==keys and row['stream_id'] in FORMAL_STREAMS
                and type(row['checkpoint_id']) is str and row['checkpoint_id']
                and _sha(row['quality_cases_sha256']) and set(row['quality'])==set(VARIANTS),
                'selector accepts formal quality only, with no program/preaudit inputs')
        key=(row['stream_id'],row['checkpoint_id']); require(key not in seen,'duplicate formal quality checkpoint')
        seen.add(key); streams.add(key[0]); cohorts.add(row['quality_cases_sha256'])
        qualities=[_quality(row['quality'][variant]) for variant in VARIANTS]
        require(len({q['episodes'] for q in qualities})==1,'formal raw/reg quality scene counts differ')
        if all(_quality_pass(q,q,'formal') for q in qualities):
            scores[key]=min(Fraction(4*q['deliveries']-q['collisions']-20*q['shutdown_episodes'],4*q['episodes'])
                            for q in qualities)
    require(streams==set(FORMAL_STREAMS) and len(cohorts)==1,'all three formal pairs and one quality cohort required')
    require(scores,'no pair passes formal quality; program results cannot rescue or select a failed pair')
    winner=min(scores,key=lambda key:(-scores[key],*key))
    return dict(version=VERSION,stream_id=winner[0],checkpoint_id=winner[1],quality_only_table_sha256=digest(rows),
                score_numerator=scores[winner].numerator,score_denominator=scores[winner].denominator,
                program_prescreen_may_change_selection=False,scope='declaration_selection_not_run_audit',
                training_authority=False,sealed_authority=False,model_acceptance_passed=False)


def validate_formal_group_declaration(rows,*,budget):
    """Check three independent formal stream declarations, not their truth.

    Program diagnostics are returned for reporting; selection must separately
    consume only the fixed quality table through select_quality_only.
    """
    require(type(rows) is list and len(rows)==3
            and {row['stream_id'] for row in rows}==set(FORMAL_STREAMS),
            'all three independently declared formal pairs are required')
    results={row['stream_id']:validate_pair_declaration(row,budget=budget,phase='formal') for row in rows}
    for key in ('initialization_random_stream_sha256','optimizer_random_stream_sha256','scenario_sequence_sha256'):
        require(len({row['initialization']['raw'][key] for row in rows})==3,
                'formal initialization, optimizer and scene streams must be independent')
    require(len({row['quality_cases_sha256'] for row in rows})==1,'formal pairs require one fixed quality cohort')
    return dict(version=VERSION,scope='declaration_validation_only_not_run_replay',streams=results,
                program_diagnostics_may_not_reselect_pair=True,training_authority=False,
                formal_preparation_authority=False,sealed_authority=False,model_acceptance_passed=False)


def execution_authority(*args,**kwargs):
    raise RuntimeError('long-budget finite owner, exact seed clearance and run replay are not implemented; execution remains closed')
