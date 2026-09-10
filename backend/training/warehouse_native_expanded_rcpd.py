"""Versioned RCPD search for richer, same-Actor observed197 evidence.

Thirteen predeclared capacity pairs retain the original nine and add four
larger candidates. Reliability thresholds are unchanged. The batch predicate
comparison matches scalar program execution. This module never samples or
authorizes a participant experiment; a qualified source reader owns admission.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import warehouse_native_shutdown_rcpd as own
from . import warehouse_native_continuation_rcpd as data_api
from . import warehouse_native_program_batch as program_batch
from .warehouse_native_common import ROOT, digest, file_hash, jsonable
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager, _NativeRCPD
from env.warehouse_native.policy import ACTIONS
from core.rcpd import RCPDConfig
from core.program import ExecutableProgram
from core.policy_program_regularizer import program_complexity

VERSION = 'warehouse-native-expanded-evidence-rcpd.v1'
CANDIDATES = tuple((d,l) for d in (4,6,8) for l in (16,32,64)) + ((10,128),(12,128),(12,256),(16,256))


def feedback_config():
    return FeedbackConfig(depths=(4,6,8,10,12,16),leaves=(16,32,64,128,256))


def contract():
    return {'version':VERSION,'candidates':[list(x) for x in CANDIDATES],
        'feedback_config':asdict(feedback_config()),'action_structure_weight':0.,
        'minimum_critical_scenarios':10,'minimum_critical_non_wait_fidelity':.85,
        'minimum_critical_non_wait_rows':1,'prediction_semantics':program_batch.VERSION,
        'selection':'lowest_normalized_actual_complexity_among_all_reliable_candidates_then_KL_and_fidelity',
        'teacher_rows_allowed':0,'rows_removed':0,'final_explanation_qualification':False}


def validator(data):
    if data.get('collector_receipt',{}).get('version')==own.VERSION:return own
    from . import warehouse_native_shutdown_stage_rcpd as stage
    if data.get('collector_receipt',{}).get('version')==stage.VERSION:return stage
    raise ValueError('Unknown real neural collection producer')


def execution_sources(producer):
    value=producer.execution_sources()
    for path in (Path(__file__),Path(program_batch.__file__)):
        value[str(path.relative_to(ROOT))]=file_hash(path)
    return value


class ExactProgramManager(FeedbackManager):
    """Same soft KL and state fields; versioned exact predicate implementation."""
    def _predict(self,program,observations):
        return program_batch.predict(program,self._observations(observations),self.feature_names)


def _metrics(manager,data):
    m=data_api._fidelity(manager,data);p=manager._predict(manager.program,data['observations']);y=data['probabilities']
    m['mean_kl']=float(np.mean(np.sum(y*(np.log(y.clip(1e-8))-np.log(p.clip(1e-8))),axis=-1)))
    return m


def gates(metrics):
    checks={'overall':metrics['overall']['fidelity'] is not None and metrics['overall']['fidelity']>=.9,
        'mean_kl':metrics['mean_kl']<=.35}
    if set(metrics['critical'])!=set(data_api.GROUPS):raise ValueError('Critical collaboration groups differ')
    for name,g in metrics['critical'].items():
        checks[name]=(g['rows']>0 and g['scenarios']>=10 and g['fidelity'] is not None and g['fidelity']>=.85
            and g['non_wait']['rows']>0 and g['non_wait']['fidelity'] is not None and g['non_wait']['fidelity']>=.85)
    return {'checks':checks,'reliable':all(checks.values())}


def fit(train,selection,*,step,feature_names,prior_manager_state=None,allow_test_fixture=False):
    if type(step) is not int or type(allow_test_fixture) is not bool:raise ValueError('Explicit fit clock and scope required')
    producer=validator(train)
    if validator(selection) is not producer:raise ValueError('Collection producers differ')
    for data in (train,selection):producer._validate_data(data,allow_test_fixture)
    if (train['pool']!='train' or selection['pool']!='selection'
            or any(d['actor_training_clock']!=step or d['feature_names']!=list(feature_names) for d in (train,selection))
            or train['actor_bindings']!=selection['actor_bindings']
            or train['collector_receipt']['actor_metadata']!=selection['collector_receipt']['actor_metadata']):
        raise ValueError('Same frozen Actor, actual clock and separate source pools required')
    overlap={'scene_fingerprints':len(set(train['scene_fingerprints'])&set(selection['scene_fingerprints'])),
        'episode_ids':len(set(train['episode_ids'])&set(selection['episode_ids'])),
        'public_states':len({x['public_state_sha256'] for x in train['row_sources']}&{x['public_state_sha256'] for x in selection['row_sources']}),
        'exact_observations':len({x.tobytes() for x in train['observations']}&{x.tobytes() for x in selection['observations']})}
    if any(overlap.values()):raise ValueError('Overlap rejected without removing rows: '+str(overlap))
    cfg=feedback_config();manager=ExactProgramManager(feature_names,cfg)
    if prior_manager_state is not None:manager.load_state_dict(prior_manager_state)
    if step<manager.last_step:raise ValueError('Tree fit cannot move the feedback clock backward')
    manager.current_lambda=0.;manager.reliable=False
    x,y,ids=manager._dataset(train['observations'],train['probabilities'],train['episode_ids'],cfg.minimum_training_rows,'Training')
    vx,vy,vids=manager._dataset(selection['observations'],selection['probabilities'],selection['episode_ids'],cfg.minimum_validation_rows,'Selection')
    def samples(values,labels,episodes):
        return [{'obs':a,'probabilities':b,'episode':c} for a,b,c in zip(values,labels,episodes)]
    train_samples,selection_samples=samples(x,y,ids),samples(vx,vy,vids)
    encoder=lambda item:dict(zip(feature_names,map(float,item['obs'])))
    oracle=lambda item:dict(zip(ACTIONS,map(float,item['probabilities'])))
    source_actor=train['actor_bindings']['actor_sha256'];sources=execution_sources(producer);candidates=[]
    for depth,leaves in CANDIDATES:
        c=RCPDConfig(max_depth=depth,max_leaf_nodes=leaves,max_predicates=None,min_samples_leaf=8,
            complexity_penalty=.001,random_seed=260908,regularization_lambda=.01,action_structure_weight=0.)
        with data_api._isolated_host_rng():
            result=_NativeRCPD(c).fit(train_samples,oracle,encoder,validation_states=selection_samples,
                split_group_provider=lambda item:item['episode'],program_metadata={'native_source_actor_sha256':source_actor,
                    'native_feedback_version':VERSION,'prediction_semantics':program_batch.VERSION,
                    'runtime_controller':'native_neural_actor_only'})
        if set(result.program.feature_names)!=set(feature_names):raise ValueError('Extracted input schema changed')
        candidate_meta=deepcopy(result.program.metadata)
        candidate_meta['metrics']={**candidate_meta.get('metrics',{}),'feedback_eligible':False,
            'explanation_eligible':False,'feedback_weight':0.,
            'feedback_ineligibility_reasons':['native_candidate_selection_pending'],
            'explanation_ineligibility_reasons':['independent_intervention_audit_not_run']}
        program=ExecutableProgram(result.program.action_names,tuple(feature_names),result.program.root,candidate_meta)
        manager.program=program
        m=_metrics(manager,selection);gate=gates(m)
        complexity=program_complexity(program,max_depth=16,max_leaf_count=256,max_predicate_count=255).to_dict()
        report={'depth_cap':depth,'leaf_cap':leaves,'selection_metrics':m,'gates':gate,'complexity':complexity,
            'selection_objective':1-m['overall']['fidelity']+.2*m['mean_kl']+.001*complexity['loss']}
        candidates.append((report,program))
    available=[item for item in candidates if item[0]['gates']['reliable']]
    chosen=min(available,key=lambda item:(item[0]['complexity']['loss'],item[0]['selection_metrics']['mean_kl'],
        -item[0]['selection_metrics']['overall']['fidelity'],item[0]['depth_cap'],item[0]['leaf_cap'])) if available else min(candidates,key=lambda item:item[0]['selection_objective'])
    selected,program=chosen;reliable=selected['gates']['reliable']
    binding={'config_sha256':digest(contract()),'training_data_sha256':train['data_sha256'],
        'selection_data_sha256':selection['data_sha256'],'actor_sha256':source_actor,
        'actor_parameters_sha256':train['actor_bindings']['actor_parameters_sha256'],'cumulative_fit_step':step}
    report={'version':VERSION,'step':step,'source_actor_sha256':source_actor,'reliable':reliable,
        'source':'same_frozen_actor_soft_probabilities_on_neural_trajectory_observations','selected':selected,
        'candidates':[c[0] for c in candidates],'train_rows':len(x),'validation_rows':len(vx),
        'train_episodes':len(set(ids)),'validation_episodes':len(set(vids)),'overlap':overlap,'rows_removed':0,
        'selection_metrics':selected['selection_metrics'],'observed197_bindings':binding,
        'extraction_config':contract(),'prediction_semantics':program_batch.VERSION,
        'complexity_has_actor_gradient':False,'explanation_qualified':False,'intervention_direction_not_tested_here':True,
        'collector_producer':producer.VERSION,'execution_sources':sources,'test_fixture':allow_test_fixture}
    meta={**deepcopy(program.metadata),'native_feedback_config':asdict(cfg),'observed197_bindings':binding,
        'prediction_semantics':program_batch.VERSION,'metrics':{**deepcopy(program.metadata.get('metrics',{})),
            'reliable':reliable,'feedback_eligible':reliable,'explanation_eligible':False,'feedback_weight':0.,
            'feedback_ineligibility_reasons':[] if reliable else ['native_reliability_gate_failed'],
            'explanation_ineligibility_reasons':['independent_intervention_audit_not_run']}}
    manager.program=ExecutableProgram(program.action_names,program.feature_names,program.root,meta)
    manager.reliable=reliable;manager.last_step=manager.last_fit_step=step;manager.last_fit_report=jsonable(report)
    for data in (train,selection):producer._validate_data(data,allow_test_fixture)
    if sources!=execution_sources(producer):raise ValueError('Extraction sources changed during fitting')
    return jsonable({'version':VERSION,'reliable':reliable,'fit_report':manager.last_fit_report,
        'program':manager.program.to_dict(),'manager_state':manager.state_dict(),
        'candidate_programs':[p.to_dict() for _,p in candidates],
        'prediction_semantics':program_batch.VERSION,'evidence_sha256':digest({'binding':binding,'fit_report':manager.last_fit_report}),
        'test_fixture':allow_test_fixture,'explanation_qualified':False,'release_eligible':False,
        'actual_joint_steps':0,'neural_training_updates':0})
