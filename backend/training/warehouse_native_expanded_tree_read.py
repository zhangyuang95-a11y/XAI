"""Independent, zero-fit verification of all thirteen saved RCPD candidates."""
from pathlib import Path
import json

from . import warehouse_native_expanded_tree_run as runner
from .warehouse_native_common import digest,file_hash
from core.program import ExecutableProgram
from core.policy_program_regularizer import program_complexity

extraction=runner.extraction
VERSION='warehouse-native-expanded-tree-evidence-reader.v1'


def verify_result(result,train,selection,*,require_reliable=True,fixture=False):
    """Recompute candidate metrics/selection and the saved manager, never refit."""
    if (result.get('version')!=extraction.VERSION or result.get('test_fixture') is not fixture
            or result.get('prediction_semantics')!=extraction.program_batch.VERSION
            or result.get('explanation_qualified') is not False or result.get('release_eligible') is not False
            or result.get('actual_joint_steps')!=0 or result.get('neural_training_updates')!=0):
        raise ValueError('Revised extraction identity or scope differs')
    producer=extraction.validator(train)
    if extraction.validator(selection) is not producer:raise ValueError('Dataset producers differ')
    for data in (train,selection):producer._validate_data(data,fixture)
    if (train['actor_bindings']!=selection['actor_bindings'] or train['pool']!='train' or selection['pool']!='selection'
            or train['feature_names']!=selection['feature_names']
            or train['actor_training_clock']!=selection['actor_training_clock']
            or train['collector_receipt']['actor_metadata']!=selection['collector_receipt']['actor_metadata']):
        raise ValueError('Candidate source Actors, pools or clock differ')
    overlap={'scene_fingerprints':len(set(train['scene_fingerprints'])&set(selection['scene_fingerprints'])),
        'episode_ids':len(set(train['episode_ids'])&set(selection['episode_ids'])),
        'public_states':len({x['public_state_sha256'] for x in train['row_sources']}&{x['public_state_sha256'] for x in selection['row_sources']}),
        'exact_observations':len({x.tobytes() for x in train['observations']}&{x.tobytes() for x in selection['observations']})}
    if any(overlap.values()):raise ValueError('Actual extraction pools overlap')
    report=result['fit_report'];binding={'config_sha256':digest(extraction.contract()),
        'training_data_sha256':train['data_sha256'],'selection_data_sha256':selection['data_sha256'],
        'actor_sha256':train['actor_bindings']['actor_sha256'],
        'actor_parameters_sha256':train['actor_bindings']['actor_parameters_sha256'],
        'cumulative_fit_step':train['actor_training_clock']}
    if (report.get('observed197_bindings')!=binding or report.get('extraction_config')!=json.loads(json.dumps(extraction.contract()))
            or report.get('execution_sources')!=extraction.execution_sources(producer)
            or report.get('collector_producer')!=producer.VERSION
            or report.get('test_fixture') is not fixture
            or report.get('source_actor_sha256')!=binding['actor_sha256']
            or report.get('step')!=binding['cumulative_fit_step']
            or report.get('version')!=extraction.VERSION or report.get('overlap')!=overlap
            or report.get('rows_removed')!=0 or report.get('train_rows')!=len(train['observations'])
            or report.get('validation_rows')!=len(selection['observations'])
            or report.get('train_episodes')!=len(set(train['episode_ids']))
            or report.get('validation_episodes')!=len(set(selection['episode_ids']))
            or report.get('reliable') is not result.get('reliable')
            or report.get('prediction_semantics')!=extraction.program_batch.VERSION
            or report.get('complexity_has_actor_gradient') is not False
            or report.get('explanation_qualified') is not False
            or report.get('intervention_direction_not_tested_here') is not True
            or result['evidence_sha256']!=digest({'binding':binding,'fit_report':report})):
        raise ValueError('Actual revised fit provenance differs')
    if len(result['candidate_programs'])!=13 or len(report['candidates'])!=13:raise ValueError('The finite candidate search is incomplete')
    manager=extraction.ExactProgramManager(train['feature_names'],extraction.feedback_config());verified=[]
    for index,((depth,leaves),raw,saved) in enumerate(zip(extraction.CANDIDATES,result['candidate_programs'],report['candidates'])):
        program=ExecutableProgram.from_dict(raw)
        if (tuple(program.feature_names)!=tuple(train['feature_names'])
                or program.metadata.get('native_source_actor_sha256')!=binding['actor_sha256']
                or program.metadata.get('native_feedback_version')!=extraction.VERSION
                or program.metadata.get('prediction_semantics')!=extraction.program_batch.VERSION
                or program.metadata.get('metrics',{}).get('feedback_eligible') is not False
                or program.metadata.get('metrics',{}).get('explanation_eligible') is not False
                or program.metadata.get('metrics',{}).get('feedback_weight')!=0
                or program.root.depth()>depth or program.root.leaf_count()>leaves):raise ValueError('Candidate input/source/capacity differs')
        manager.program=program;metrics=extraction._metrics(manager,selection);gates=extraction.gates(metrics)
        complexity=program_complexity(program,max_depth=16,max_leaf_count=256,max_predicate_count=255).to_dict()
        actual={'depth_cap':depth,'leaf_cap':leaves,'selection_metrics':metrics,'gates':gates,'complexity':complexity,
            'selection_objective':1-metrics['overall']['fidelity']+.2*metrics['mean_kl']+.001*complexity['loss']}
        if actual!=saved:raise ValueError('Saved candidate metrics differ from actual NN evidence: '+str(index))
        verified.append((actual,program))
    allowed=[item for item in verified if item[0]['gates']['reliable']]
    chosen=min(allowed,key=lambda item:(item[0]['complexity']['loss'],item[0]['selection_metrics']['mean_kl'],
        -item[0]['selection_metrics']['overall']['fidelity'],item[0]['depth_cap'],item[0]['leaf_cap'])) if allowed else min(verified,key=lambda item:item[0]['selection_objective'])
    if report['selected']!=chosen[0] or result['reliable'] is not chosen[0]['gates']['reliable']:
        raise ValueError('The recorded tree is not the simplest reliable declared candidate')
    restored=extraction.ExactProgramManager(train['feature_names'],extraction.feedback_config())
    restored.load_state_dict(result['manager_state'])
    if (restored.program is None or restored.program.to_dict()!=result['program']
            or restored.program.root!=chosen[1].root or restored.last_fit_report!=report
            or restored.reliable is not result['reliable'] or restored.current_lambda!=0
            or restored.last_step!=binding['cumulative_fit_step'] or restored.last_fit_step!=restored.last_step
            or restored.program.metadata.get('native_source_actor_sha256')!=binding['actor_sha256']
            or restored.program.metadata.get('prediction_semantics')!=extraction.program_batch.VERSION
            or restored.program.metadata.get('native_feedback_version')!=extraction.VERSION
            or restored.program.metadata.get('observed197_bindings')!=binding
            or digest(restored.program.metadata.get('native_feedback_config'))!=digest(extraction.contract()['feedback_config'])
            or restored.program.metadata.get('metrics',{}).get('feedback_eligible') is not result['reliable']
            or restored.program.metadata.get('metrics',{}).get('reliable') is not result['reliable']
            or restored.program.metadata.get('metrics',{}).get('explanation_eligible') is not False
            or restored.program.metadata.get('metrics',{}).get('feedback_weight')!=0
            or extraction._metrics(restored,selection)!=report['selection_metrics']):
        raise ValueError('Selected program, exact-predicate manager or fit clock differs')
    if require_reliable and result['reliable'] is not True:raise ValueError('Expanded tree is not reliable')
    return {'version':VERSION,'reliable':result['reliable'],'verified_candidates':13,
        'selected':chosen[0],'tree_fits':0,'environment_steps':0,'neural_forwards':0,
        'model_capability_verified_here':False,'explanation_qualified':False,'release_ready':False}


def read_bundle(root,*,expected_manifest_sha256,require_reliable=True):
    root=Path(root).expanduser().resolve()
    if file_hash(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('External revised-tree byte anchor differs')
    plan,manifest=runner.initial.read_json(root/'plan.json'),runner.initial.read_json(root/'manifest.json')
    if (plan.get('version')!=runner.VERSION or plan.get('runtime_sources')!=runner.sources()
            or manifest.get('version')!=runner.VERSION or manifest.get('status')!='completed'
            or manifest.get('plan_sha256')!=digest(plan) or plan.get('test_fixture') is not False
            or plan.get('extraction_contract')!=json.loads(json.dumps(extraction.contract()))
            or plan.get('maximum_rcpd_calls')!=13 or plan.get('maximum_sklearn_fits')!=104
            or manifest.get('feedback_training_started') is not False
            or any(d.get(k) is not False for d in (plan,manifest) for k in ('explanation_qualified','release_ready'))
            or plan.get('new_environment_steps')!=0 or plan.get('new_ppo_steps')!=0
            or plan.get('automatic_fit_retry') is not False):raise ValueError('Revised completed fit contract differs')
    for name,sha in plan['runtime_sources'].items():
        if file_hash(root/'source_snapshot'/name)!=sha:raise ValueError('Revised fit source snapshot differs')
    corpus,train,selection=runner.read_corpus(plan['corpus'],plan['corpus_manifest_sha256'])
    if (plan['actor_bindings']!=corpus['actor_bindings'] or plan['training_data_sha256']!=train['data_sha256']
            or plan['selection_data_sha256']!=selection['data_sha256'] or plan['cumulative_fit_step']!=train['actor_training_clock']):
        raise ValueError('Revised fit source datasets differ')
    result=json.loads(runner.initial.bound_bytes(root,manifest['fit_result']))
    if json.loads(runner.initial.bound_bytes(root,manifest['program']))!=result['program']:raise ValueError('Standalone selected program differs')
    audit=verify_result(result,train,selection,require_reliable=require_reliable)
    if manifest.get('reliable') is not result['reliable']:raise ValueError('Manifest reliability differs')
    source_plan=runner.initial.read_json(Path(corpus['base'])/'plan.json')
    if file_hash(root/'manifest.json')!=expected_manifest_sha256:raise ValueError('Revised tree changed while reading')
    return {'extraction_plan':plan,'manifest':manifest,'fit_result':result,'audit':audit,
        'source_descriptor':source_plan['source_descriptor'],'source_report_sha256':source_plan['source_report_sha256'],
        'actor_bindings':corpus['actor_bindings'],'original_scene_pools':runner.initial.read_json(Path(corpus['base'])/'scene_pools.json'),
        'expanded_train_pools':runner.initial.read_json(Path(corpus['expansion'])/'scene_pools.json'),
        'manifest_sha256':expected_manifest_sha256,'explanation_qualified':False,'release_ready':False}
