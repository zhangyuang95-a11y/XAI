"""Freeze the registered final feedback endpoint for independent acceptance.

This is a candidate selection step, not a release. It never selects an earlier
boundary or the pure PPO control, and cannot invoke training, fitting or final
evaluation. The external completed-comparison SHA authenticates the already
inspected learning history. Actual saved validation and all 13 terminal tree
candidates are recomputed from their original recorded evidence.
"""
from pathlib import Path
import argparse
import json
import re

from backend.warehouse_shutdown_runtime import ShutdownRuntime
from backend import warehouse_family_explanation as explanation
from backend import warehouse_runtime_family as registry
from backend.training import warehouse_native_shutdown_feedback_result as paired
from backend.training import warehouse_native_shutdown_feedback_run as runner
from backend.training import warehouse_native_shutdown_terminal_tree as link
from backend.training.warehouse_native_common import ROOT, digest, file_hash
from backend.training.warehouse_native_shutdown_result import _Inputs, _json_bytes, _same

VERSION = 'warehouse-native-fixed-feedback-candidate-selection.v1'
ENERGY_VERSION = 'warehouse-native-energy-candidate-selection.v1'
NATIVE_CYCLE_VERSION = 'warehouse-native-finite-cycle-candidate-selection.v1'


def energy_sources():
    from . import warehouse_family_candidate_tree_run as tree
    return {**sources(), **tree.sources()}


def native_cycle_sources():
    from . import warehouse_family_candidate_tree_run as tree
    return {**sources(), **tree.sources('native_cycle')}


def _native_cycle_selection_gate(completion, validation, prepared, *, fixture=False):
    from . import warehouse_family_native_cycle_run as cycle
    endpoint=prepared.get('primary_endpoint')
    if (prepared.get('version') != cycle.VERSION or prepared.get('test_fixture') is not fixture
            or type(endpoint) is not int or endpoint <= 0
            or prepared.get('validation_endpoints') != [endpoint]
            or prepared.get('stop_gate') != 'full_and_warmup'
            or completion.get('version') != cycle.VERSION or completion.get('until') != endpoint
            or completion.get('status') != 'both_gates_ready'
            or completion.get('feedback_disabled_after_source') is not True
            or completion.get('feedback_lambda') != 0. or completion.get('used_final_test') is not False
            or any(completion.get(k) is not False for k in ('formal_ready','release_ready','explanation_qualified'))
            or completion.get('counts',{}).get('ppo_steps') != endpoint):
        raise ValueError('Only the completed declared native cycle endpoint may supply this candidate')
    for key in ('capability','warmup_capability'):
        if validation.get(key,{}).get('eligible') is not True:
            raise ValueError('Both original native cycle capability gates must pass')
        _same(completion[key],validation[key],'Completed cycle validation differs')
    binding=validation['actor_bindings']
    if (binding.get('joint_steps') != endpoint or binding.get('experiment_version') != cycle.native.VERSION
            or binding.get('shutdown_arm') != 'beta1' or binding.get('own_shutdown_beta') != 1.
            or binding.get('branch') != 'own_credit' or 'feedback_branch' in binding):
        raise ValueError('The actual native cycle Actor identity differs')
    return endpoint


def _energy_selection_gate(completion, validation, *, fixture=False):
    """The separately registered 50k diagnostic; never relabel the old pair."""
    from . import warehouse_family_energy_continuation_run as energy
    if (completion.get('version') != energy.VERSION or completion.get('until') != 50000
            or completion.get('status') != 'both_gates_ready'
            or completion.get('feedback_disabled_after_source') is not True
            or completion.get('feedback_lambda') != 0. or completion.get('used_final_test') is not False
            or any(completion.get(k) is not False for k in ('formal_ready','release_ready','explanation_qualified'))
            or completion.get('counts',{}).get('ppo_steps') != 50000):
        raise ValueError('Only the completed fixed 50k diagnostic may supply this candidate')
    for key in ('capability','warmup_capability'):
        if validation.get(key,{}).get('eligible') is not True:
            raise ValueError('Both original diagnostic capability gates must pass')
        _same(completion[key],validation[key],'Completed diagnostic validation differs')
    binding = validation['actor_bindings']
    if (binding.get('joint_steps') != 50000 or binding.get('experiment_version') != energy.native.VERSION
            or binding.get('shutdown_arm') != 'beta1' or binding.get('own_shutdown_beta') != 1.
            or binding.get('branch') != 'own_credit' or 'feedback_branch' in binding):
        raise ValueError('The actual native diagnostic Actor identity differs')
    if completion['source_descriptor'].get('test_fixture') is not fixture:
        raise ValueError('Diagnostic fixture scope differs')


def freeze_energy(source, *, expected_completion_sha256, tree_output,
                  expected_tree_manifest_sha256, output, allow_test_fixture=False):
    return _freeze_native(source,expected_completion_sha256=expected_completion_sha256,tree_output=tree_output,
        expected_tree_manifest_sha256=expected_tree_manifest_sha256,output=output,
        allow_test_fixture=allow_test_fixture,source_kind='energy')


def freeze_native_cycle(source, *, expected_completion_sha256, tree_output,
                        expected_tree_manifest_sha256, output, allow_test_fixture=False):
    return _freeze_native(source,expected_completion_sha256=expected_completion_sha256,tree_output=tree_output,
        expected_tree_manifest_sha256=expected_tree_manifest_sha256,output=output,
        allow_test_fixture=allow_test_fixture,source_kind='native_cycle')


def _freeze_native(source, *, expected_completion_sha256, tree_output,
                   expected_tree_manifest_sha256, output, allow_test_fixture=False, source_kind):
    """Freeze the real declared endpoint, without changing its producer identity."""
    from . import warehouse_family_candidate_tree_run as tree
    reader=tree.source_reader(source_kind)
    reader._sha(expected_completion_sha256); reader._sha(expected_tree_manifest_sha256)
    get_sources=energy_sources if source_kind=='energy' else native_cycle_sources
    version=ENERGY_VERSION if source_kind=='energy' else NATIVE_CYCLE_VERSION
    source, tree_output, output = (Path(v).resolve() for v in (source,tree_output,output))
    if output.exists(): raise ValueError('Candidate freeze requires a new output directory')
    runner._separate(output,source,tree_output)
    current = get_sources()
    saved = reader.read_completed(source,expected_completion_sha256=expected_completion_sha256,
                                  allow_test_fixture=allow_test_fixture)
    if source_kind=='energy':
        _energy_selection_gate(saved['completion'],saved['report'],fixture=allow_test_fixture)
        endpoint=50000;descriptor=saved['source_descriptor']
    else:
        endpoint=_native_cycle_selection_gate(saved['completion'],saved['report'],saved['prepared'],fixture=allow_test_fixture)
        descriptor=saved['original_feedback_descriptor']
    extracted = tree.read_completed(tree_output,expected_manifest_sha256=expected_tree_manifest_sha256,
                                   require_reliable=True,allow_test_fixture=allow_test_fixture)
    plan = extracted['plan']; binding = plan['actor_bindings']
    expected = {'completion_sha256':expected_completion_sha256,
        'actor_sha256':saved['actor_bindings']['actor_sha256'],
        'actor_parameters_sha256':saved['actor_parameter_receipt']['actor_parameters_sha256'],
        'validation_report_sha256':saved['completion']['report_sha256']}
    if (plan['source'] != str(source) or plan['endpoint'] != endpoint
            or plan.get('source_kind') != source_kind or plan.get('source_version') != reader.VERSION):
        raise ValueError('The new tree was extracted from another diagnostic')
    _same(plan['source_anchors'],expected,'The fresh tree does not bind the current endpoint')
    _same(binding['actor_parameters_sha256'],expected['actor_parameters_sha256'],'Tree parameter binding differs from the current tensor proof')
    _same(plan['actor_parameter_receipt'],saved['actor_parameter_receipt'],'Current tensor proof differs')
    _same({k:v for k,v in binding.items() if k != 'actor_parameters_sha256'},saved['actor_bindings'],
          'Tree and validated native Actor bindings differ')
    reads = _Inputs()
    for name,entry in saved['input_bindings'].items():
        path=Path(name); reads.raw(path.parent,path.name,entry['sha256'],entry['size'])
    for path in sorted(tree_output.rglob('*')):
        if path.is_file() and path.name != 'candidate_tree.lock':
            reads.raw(tree_output,str(path.relative_to(tree_output)))
    actor_path=Path(saved['actor_path'])
    raw=reads.raw(actor_path.parent,actor_path.name,binding['actor_sha256'])
    runtime=ShutdownRuntime(actor_path,protocol=saved['protocol'],expected_actor_sha256=binding['actor_sha256'],
        expected_protocol_sha256=digest(saved['protocol']),expected_bindings=saved['actor_bindings'],allow_test_fixture=allow_test_fixture,
        config=runner.collaborative_study_config(horizon=saved['scenarios']['configuration']['horizon']))
    identity=registry.verify(runtime,allow_test_fixture=allow_test_fixture,expected_family='retained_beta197')
    meta=runtime.actor.metadata
    rule=('fixed_50000_diagnostic_both_capability_gates_and_fresh_tree' if source_kind=='energy'
          else 'fixed_declared_native_cycle_both_capability_gates_and_fresh_tree')
    chosen={'version':version,'selection_rule':rule,
        'selection_split':'validation','selected_step':endpoint,
        'selected_cumulative_step':meta['source_counters']['joint_steps']+endpoint,
        'actor_sha256':binding['actor_sha256'],'actor_parameters_sha256':binding['actor_parameters_sha256'],
        'protocol_sha256':runtime.protocol_sha256,'scenario_manifest_sha256':digest(saved['scenarios']),
        'runtime_family':'retained_beta197','runtime_signature':runtime.signature,
        'cycle_id':meta['cycle_id'],'branch':meta['branch'],'shutdown_arm':meta['shutdown_arm'],
        'own_shutdown_beta':meta['own_shutdown_beta'],'source_checkpoint_sha256':meta['source_checkpoint_sha256'],
        'selected_effective_checkpoint':saved['checkpoint'],'feedback_disabled_after_source':True,
        'original_pair_extended':False,'original_comparison_sha256':descriptor['comparison_sha256'],
        'completion_sha256':expected_completion_sha256,'tree_manifest_sha256':expected_tree_manifest_sha256,
        'frozen':True,'final_test_used':False,'test_fixture':allow_test_fixture,
        'release_ready':False,'explanation_qualified':False}
    output.mkdir(parents=True,exist_ok=False)
    try:
        runner.write_bytes(output/'actor.npz',raw)
        documents={'protocol':saved['protocol'],'scenarios':saved['scenarios'],'program':extracted['program'],
            'actor_bindings':saved['actor_bindings'],'selection':chosen,'validation':saved['report'],
            'terminal_tree_verification':extracted['report']['verification']}
        for name,value in documents.items(): runner.write_json(output/(name+'.json'),value)
        component=explanation.FamilyExplainer(output/'program.json',expected_program_sha256=file_hash(output/'program.json'),
            runtime=runtime,allow_test_fixture=allow_test_fixture)
        if component.eligible or component.participant_enabled: raise ValueError('Component qualification changed')
        _same(get_sources(),current,'Candidate source changed during freezing'); reads.unchanged()
        manifest={'version':version,'status':'frozen_for_independent_acceptance','runtime':identity,
            'source_hashes':current,'comparison':{'path':descriptor['comparison_path'],'sha256':descriptor['comparison_sha256']},
            'diagnostic':{'directory':str(source),'completion_sha256':expected_completion_sha256,
                'source_kind':source_kind,'endpoint':endpoint},
            'tree':{'directory':str(tree_output),'manifest_sha256':expected_tree_manifest_sha256},
            'explanation_component':component.contract_report,'original_input_bindings':reads.records,
            'artifacts':{name:runner.binding(output,output/name) for name in ('actor.npz','protocol.json','scenarios.json',
                'program.json','actor_bindings.json','selection.json','validation.json','terminal_tree_verification.json')},
            'qualification_evaluated':False,'release_ready':False,'formal_ready':False,'explanation_qualified':False,
            'environment_steps':0,'neural_forwards':0,'new_ppo_steps':0,'tree_fits':0,'checkpoint_decodes':0,
            'test_fixture':allow_test_fixture,
            'scope':'Derived finite PPO endpoint from feedback-trained weights, with fresh same-Actor tree; independent acceptance pending'}
        runner.write_json(output/'manifest.json',manifest)
        return manifest
    except BaseException as error:
        runner.write_json(output/'failure.json',{'version':version,'reason':repr(error),'release_ready':False})
        raise


def sources():
    result = {**link.sources(), **registry.execution_sources(), **explanation.explanation_sources()}
    for module in (paired, runner, link):
        path = Path(module.__file__); result[str(path.relative_to(ROOT))] = file_hash(path)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(Path(__file__))
    return result


def _selection_gate(result, *, fixture=False):
    """Only the previously declared fixed-feedback rule; no best-boundary search."""
    if (result.get('version') != paired.VERSION or result.get('status') != 'fixed_endpoint_complete'
            or result.get('test_fixture') is not fixture or result.get('source_selection') is not None
            or result.get('result_source_sha256') != file_hash(Path(paired.__file__))
            or result.get('release_ready') is not False or result.get('formal_ready') is not False
            or result.get('explanation_qualified') is not False):
        raise ValueError('Externally anchored completed candidate comparison is required')
    endpoint = result['primary_endpoint']
    if type(endpoint) is not int or endpoint <= 0 or (not fixture and endpoint != 250000):
        raise ValueError('Only the registered fixed 250000 endpoint can be selected')
    primary = result['primary']['feedback']
    if any(primary[k].get('eligible') is not True for k in ('full_capability','warmup_capability')):
        raise ValueError('Fixed feedback endpoint failed capability; no alternative is selected')
    for arm in ('control','feedback'):
        if result['training'][arm]['joint_steps'] != endpoint or result['training'][arm]['raw_neural_overrides'] != 0:
            raise ValueError('Original paired PPO/NN submission evidence differs')
    if result['training']['control']['updates'] != result['training']['feedback']['updates']:
        raise ValueError('Paired PPO update counts differ')
    if result['training']['feedback']['positive_lambda_joint_steps'] <= 0:
        raise ValueError('No actual feedback update was recorded')
    audit = [v for v in result['checkpoint_audits'] if v.get('kind') == 'final_effective_boundary' and v.get('branch') == 'feedback']
    positive = [v for v in result['checkpoint_audits'] if v.get('kind') == 'first_positive_lambda_update']
    if (len(audit) != 1 or audit[0].get('actor_export_equal') is not True or audit[0].get('dual_adam_steps_verified') is not True
            or len(positive) != 1 or positive[0].get('nonzero_feedback_gradient_observed') is not True):
        raise ValueError('Actual final checkpoint export/Adam and committed feedback-gradient evidence are required')
    refresh = result['refresh_status'][f'step_{endpoint:07d}']
    if refresh.get('reliable') is not True:
        raise ValueError('Final feedback tree was not reliable; no alternative is selected')
    return endpoint, audit[0], refresh


def material(pair, comparison_path, expected_comparison_sha256, *, fixture=False):
    if type(expected_comparison_sha256) is not str or not re.fullmatch(r'[a-f0-9]{64}', expected_comparison_sha256):
        raise ValueError('Explicit completed comparison SHA256 required')
    pair, path = Path(pair).resolve(), Path(comparison_path).resolve(); reads = _Inputs()
    result = reads.json(path.parent, path.name, expected_comparison_sha256)
    endpoint, checkpoint_audit, refresh_summary = _selection_gate(result, fixture=fixture)
    def linked(name): return _json_bytes(link._linked(reads,result,pair/name))
    completed = linked(f'completion_{endpoint:07d}.json')
    if file_hash(pair/f'completion_{endpoint:07d}.json') != result['completion_sha256']:
        raise ValueError('Actual fixed completion differs')
    prepared, protocol, scenes = (linked(name+'.json') for name in ('prepared','protocol','scenarios'))
    _same(prepared['runtime_sources'], runner.sources(), 'Frozen paired runtime sources changed')
    for name, expected in prepared['runtime_sources'].items():
        reads.raw(pair, 'source_snapshot/'+name, expected)
    prefix = f'branches/feedback/validation/step_{endpoint:07d}'
    recorded = linked(prefix+'/report.json')
    validation_manifest = linked(prefix+'/manifest.json')
    for episode in validation_manifest['episodes']:
        for kind in ('row', 'trace'):
            linked_path = pair/prefix/episode[kind]['path']
            link._linked(reads,result,linked_path)
    verified = runner._report_record(pair,'feedback',endpoint,prepared,protocol,scenes,completed['ledger'],fixture)
    _same(recorded,verified,'Original terminal validation raw records differ')
    _same(paired._facts(verified),result['primary']['feedback'],'Completed comparison does not describe the terminal raw matrix')
    if verified['capability']['eligible'] is not True or verified['warmup_capability']['eligible'] is not True:
        raise ValueError('Recomputed terminal ability failed')
    binding = verified['actor_bindings']
    if binding['feedback_branch'] != 'feedback' or binding['joint_steps'] != endpoint:
        raise ValueError('Selected Actor is not the final feedback branch')
    if binding['actor_parameters_sha256'] != checkpoint_audit['actor_parameters_sha256']:
        raise ValueError('Actual checkpoint and Actor parameter binding differ')
    actor_path = pair/f'branches/feedback/actors/actor_{endpoint:07d}.npz'
    actor_raw = link._linked(reads,result,actor_path)
    folder = pair/f'branches/feedback/refresh/step_{endpoint:07d}'
    plan = _json_bytes(link._linked(reads,result,folder/'plan.json'))
    manifest = _json_bytes(link._linked(reads,result,folder/'manifest.json'))
    fit = _json_bytes(link._linked(reads,result,folder/'fit_result.json'))
    link._linked(reads,result,folder/'auxiliary_budget.json')
    _same(plan['actor_bindings'],binding,'Final program was sampled from another Actor')
    if file_hash(folder/'fit_result.json') != refresh_summary['fit_result_sha256']:
        raise ValueError('Final fit differs from the bound paired result')
    sampled = runner.check_manifest(folder,plan,manifest,fixture=fixture)
    if sampled['status'] != 'completed' or len(sampled['episodes']) != len(plan['contexts']):
        raise ValueError('Final program lacks all acknowledged original episodes')
    # Array/record bytes are anchored by the completed paired reader, before
    # the unchanged merger and exact-tree reader recompute selection metrics.
    for entry in sampled['episodes']:
        for kind in ('arrays','record'): link._linked(reads,result,folder/entry[kind]['path'])
    datasets = {pool:runner.merge_saved(folder,sampled['episodes'],pool,fixture=fixture) for pool in ('train','selection')}
    tree_check = runner.expanded_reader.verify_result(fit,datasets['train'],datasets['selection'],require_reliable=True,fixture=fixture)
    runtime = ShutdownRuntime(actor_path,protocol=protocol,expected_actor_sha256=binding['actor_sha256'],
        expected_protocol_sha256=digest(protocol),expected_bindings=binding,allow_test_fixture=fixture,
        config=runner.collaborative_study_config(horizon=scenes['configuration']['horizon']))
    identity = registry.verify(runtime,allow_test_fixture=fixture,expected_family='retained_beta197')
    reads.unchanged()
    return {'pair':pair,'comparison':result,'comparison_sha256':expected_comparison_sha256,
        'endpoint':endpoint,'runtime':runtime,'runtime_identity':identity,'actor_bytes':actor_raw,
        'protocol':protocol,'scenarios':scenes,'validation':verified,'actor_bindings':binding,
        'program':fit['program'],'tree_verification':tree_check,'input_bindings':reads.records,
        'checkpoint_audit':checkpoint_audit,'test_fixture':fixture}


def freeze(pair, comparison_path, expected_comparison_sha256, output, *, allow_test_fixture=False):
    output = Path(output).resolve()
    if output.exists(): raise ValueError('Candidate freeze requires a new output directory')
    runner._separate(output,Path(pair).resolve(),Path(comparison_path).resolve().parent)
    data = material(pair,comparison_path,expected_comparison_sha256,fixture=allow_test_fixture)
    current_sources = sources(); runtime = data['runtime']; meta = runtime.actor.metadata
    selection = {'version':VERSION,'selection_rule':'fixed_feedback_endpoint_after_registered_capability_and_fresh_tree_gates',
        'selection_split':'validation','selected_step':data['endpoint'],
        'selected_cumulative_step':meta['source_counters']['joint_steps']+data['endpoint'],
        'actor_sha256':runtime.actor_sha256,'actor_parameters_sha256':data['actor_bindings']['actor_parameters_sha256'],
        'protocol_sha256':runtime.protocol_sha256,'scenario_manifest_sha256':digest(data['scenarios']),
        'runtime_family':'retained_beta197','runtime_signature':runtime.signature,
        'cycle_id':meta['cycle_id'],'branch':meta['branch'],'feedback_branch':'feedback',
        'shutdown_arm':meta['shutdown_arm'],'own_shutdown_beta':meta['own_shutdown_beta'],
        'source_checkpoint_sha256':meta['source_checkpoint_sha256'],
        'selected_effective_checkpoint':data['checkpoint_audit']['checkpoint'],
        'comparison_sha256':expected_comparison_sha256,'frozen':True,'final_test_used':False,
        'test_fixture':allow_test_fixture,'release_ready':False,'explanation_qualified':False}
    output.mkdir(parents=True,exist_ok=False)
    try:
        runner.write_bytes(output/'actor.npz',data['actor_bytes'])
        for name, value in [('protocol',data['protocol']),('scenarios',data['scenarios']),('program',data['program']),
            ('actor_bindings',data['actor_bindings']),('selection',selection),('validation',data['validation']),
            ('terminal_tree_verification',data['tree_verification'])]: runner.write_json(output/(name+'.json'),value)
        # This admission checks actual NN/program identity but never changes
        # the renderer's independent qualification flags.
        component = explanation.FamilyExplainer(output/'program.json',expected_program_sha256=file_hash(output/'program.json'),
            runtime=runtime,allow_test_fixture=allow_test_fixture)
        if component.eligible or component.participant_enabled: raise ValueError('Component granted an undeclared qualification')
        registry.verify(runtime,allow_test_fixture=allow_test_fixture,expected_signature=selection['runtime_signature'])
        _same(sources(),current_sources,'Candidate selection sources changed')
        manifest = {'version':VERSION,'status':'frozen_for_independent_acceptance',
            'runtime':data['runtime_identity'],'source_hashes':current_sources,
            'comparison':{'path':str(Path(comparison_path).resolve()),'sha256':expected_comparison_sha256},
            'explanation_component':component.contract_report,
            'original_input_bindings':data['input_bindings'],
            'artifacts':{name:runner.binding(output,output/name) for name in ('actor.npz','protocol.json','scenarios.json',
                'program.json','actor_bindings.json','selection.json','validation.json','terminal_tree_verification.json')},
            'qualification_evaluated':False,'release_ready':False,'formal_ready':False,'explanation_qualified':False,
            'environment_steps':0,'neural_forwards':0,'new_ppo_steps':0,'tree_fits':0,'checkpoint_decodes':0,
            'test_fixture':allow_test_fixture,
            'scope':'Bound saved training history, independently recomputed terminal validation and tree selection; final/heldout/browser checks still required'}
        runner.write_json(output/'manifest.json',manifest)
        return manifest
    except BaseException as error:
        runner.write_json(output/'failure.json',{'version':VERSION,'reason':repr(error),'release_ready':False})
        raise


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pair',required=True); parser.add_argument('--comparison',required=True)
    parser.add_argument('--comparison-sha256',required=True); parser.add_argument('--output',required=True); args=parser.parse_args()
    result=freeze(args.pair,args.comparison,args.comparison_sha256,args.output)
    print(json.dumps({k:result[k] for k in ('version','status','environment_steps','neural_forwards','new_ppo_steps','release_ready')}))


if __name__=='__main__': main()
