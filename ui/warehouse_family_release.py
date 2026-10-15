"""Admission of a real retained-beta NN to the local pilot.

This module reads completed evidence only. It cannot train, resume an audit,
run the final test, or change the qualification flags on an explainer or bank.
The preflight is written before the one-use final evaluation; the local service
is admitted only after that evaluation also passes. All evidence is private.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import json
import re

from backend import warehouse_runtime_family as registry
from backend.warehouse_shutdown_runtime import ShutdownRuntime
from backend.warehouse_family_explanation import FamilyExplainer
from backend.training import warehouse_family_feedback_selection as selection_api
from backend.training import warehouse_family_explanation_audit as heldout
from backend.training import warehouse_family_explanation_run as heldout_driver
from backend.training import warehouse_family_final_evaluation as final_api
from backend.training import warehouse_family_bank_run as bank_driver
from backend.training.warehouse_native_shutdown_stage_evaluation import _gates
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import capability
from env.warehouse.domain import collaborative_study_config
from ui.warehouse_family_bank_view import FrozenFamilyBank
from ui import warehouse_family_saved_evidence as evidence
from ui import warehouse_native_release as original

VERSION = 'warehouse-retained-beta-local-pilot-release.v3'
PREFLIGHT_VERSION = 'warehouse-retained-beta-independent-preflight.v3'
REQUIRED = frozenset(('candidate', 'parity', 'heldout', 'answers', 'bank', 'calibration'))


def _same(a, b, reason):
    if canonical(a) != canonical(b): raise ValueError(reason)


def _sha(value):
    if type(value) is not str or not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('An explicit external SHA256 is required')
    return value


def _root(path):
    result = Path(path).expanduser().absolute()
    if result.resolve() != result or not result.is_dir(): raise ValueError('Existing canonical directory required')
    return result


def _bytes(path, expected):
    _sha(expected); path = Path(path).absolute()
    if path.resolve() != path or not path.is_file(): raise ValueError('Evidence is missing or symlinked')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected: raise ValueError('Externally bound bytes differ: ' + str(path))
    return raw


def _json(path, expected): return original.parse_json(_bytes(path, expected).decode())


def _sources(records):
    if type(records) is not dict or not records: raise ValueError('Source bindings are required')
    for name, expected in records.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Invalid source path')
        _bytes(ROOT / relative, expected)


def release_sources():
    result = {**selection_api.native_cycle_sources(), **evidence.sources(), **heldout_driver.sources(),
              **final_api.evaluation_sources()}
    for module in (bank_driver, bank_driver.bank_api, bank_driver.bank_api.original, bank_driver.old_io):
        path = Path(module.__file__); result[str(path.relative_to(ROOT))] = file_hash(path)
    for name in ('ui/warehouse_family_release.py', 'ui/warehouse_family_server.py',
                 'ui/warehouse_family_bank_view.py', 'ui/warehouse_native_release.py',
                 'ui/warehouse_native_bank.py',
                 'ui/warehouse_public_history_server.py', 'ui/warehouse_native_server.py',
                 'ui/warehouse_native_export.py', 'ui/warehouse_view.py'):
        result[name] = file_hash(ROOT / name)
    frontend = ROOT / 'ui/warehouse_public_history_research'
    for path in sorted(frontend.rglob('*')):
        if path.is_file(): result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


@dataclass
class _Context:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
    protocol: dict
    selection_raw: bytes
    candidate: dict
    manifest_sha256: str = ''
    signature: str = ''
    source_binding: dict = field(default_factory=dict)
    release: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    closed: bool = False

    def close(self):
        # Runtime/Actor contain memory only. The Store owns DB and worker lifetimes.
        self.closed = True


def _original_json(manifest, path, expected=None):
    """Only read bytes frozen by the prior, completed selection verification."""
    path=Path(path).absolute(); item=manifest['original_input_bindings'].get(str(path))
    if not item: raise ValueError('Completed evidence is missing from original selection bindings')
    if expected is not None: _same(item['sha256'],expected,'Original evidence anchor differs')
    raw=_bytes(path,item['sha256'])
    if len(raw) != item['size']: raise ValueError('Original evidence size differs')
    return original.parse_json(raw.decode())


def _bank_selection(root, expected_prepared_sha256, selection_raw):
    """Reconcile independently anchored JSON, preserving final-use raw identity.

    Candidate freeze writes canonical JSON plus a newline; bank preparation
    writes canonical JSON without it. Their byte anchors remain independent.
    This helper is called after the genuine FrozenFamilyBank saved replay read.
    """
    root = _root(root)
    prepared = _json(root/'prepared.json', expected_prepared_sha256)
    entry = prepared['identity']['inputs']['selection.json']
    raw = _bytes(root/'inputs/selection.json', entry['sha256'])
    if len(raw) != entry['size']: raise ValueError('Bank selection byte size differs')
    _same(original.parse_json(raw.decode()), original.parse_json(selection_raw.decode()),
          'Bank selection content differs from the frozen candidate')


def _candidate_selection(manifest, documents, comparison):
    """Reconcile the distinct registered endpoints without rerunning either."""
    chosen = documents['selection.json']; validation = documents['validation.json']
    if (chosen.get('version') != manifest['version'] or chosen.get('frozen') is not True
            or chosen.get('final_test_used') is not False or chosen.get('test_fixture') is not False):
        raise ValueError('Candidate selection identity or scope differs')
    if manifest['version'] == selection_api.VERSION:
        selection_api._selection_gate(comparison)
        if chosen['selected_step'] != 250000 or chosen['feedback_branch'] != 'feedback':
            raise ValueError('Only the registered frozen terminal feedback candidate is admitted')
        _same(selection_api.paired._facts(validation), comparison['primary']['feedback'], 'Validation differs from fixed comparison')
        return 'fixed_feedback_250000'
    if manifest['version'] not in (selection_api.ENERGY_VERSION,selection_api.NATIVE_CYCLE_VERSION):
        raise ValueError('Unknown candidate selection route')
    from backend.training import warehouse_family_energy_continuation_run as energy
    from backend.training import warehouse_family_candidate_tree_run as tree
    is_cycle=manifest['version']==selection_api.NATIVE_CYCLE_VERSION
    reader=tree.source_reader('native_cycle' if is_cycle else 'energy')
    current_sources=selection_api.native_cycle_sources() if is_cycle else selection_api.energy_sources()
    _same(manifest['source_hashes'],current_sources,'Native candidate source closure differs')
    endpoint=chosen.get('selected_step')
    expected_rule=('fixed_declared_native_cycle_both_capability_gates_and_fresh_tree' if is_cycle
                   else 'fixed_50000_diagnostic_both_capability_gates_and_fresh_tree')
    if (type(endpoint) is not int or endpoint<=0 or (not is_cycle and endpoint!=50000) or 'feedback_branch' in chosen
            or chosen.get('selection_rule') != expected_rule
            or chosen.get('selection_split') != 'validation'
            or chosen.get('feedback_disabled_after_source') is not True
            or chosen.get('original_pair_extended') is not False):
        raise ValueError('The separately declared fixed native endpoint is required')
    # Candidate freezing already recomputed every original episode/tree metric.
    # Startup preserves their complete external byte bindings; it does not
    # repeat those expensive readers or turn saved evidence into new execution.
    diagnostic=manifest['diagnostic']; source=_root(diagnostic['directory'])
    completion=_original_json(manifest,source/f'completion_{endpoint:07d}.json',diagnostic['completion_sha256'])
    if is_cycle:
        prepared=_original_json(manifest,source/'prepared.json')
        _same(diagnostic.get('source_kind'),'native_cycle','Native cycle source kind differs')
        _same(diagnostic.get('endpoint'),endpoint,'Declared cycle endpoint differs')
        _same(selection_api._native_cycle_selection_gate(completion,validation,prepared,fixture=False),endpoint,
            'Fixed native cycle endpoint differs')
    else:
        selection_api._energy_selection_gate(completion,validation,fixture=False)
    _same(_original_json(manifest,source/f'branches/beta1/validation/step_{endpoint:07d}/report.json',completion['report_sha256']),
        validation,'Frozen validation differs from completed diagnostic')
    tree_root=_root(manifest['tree']['directory'])
    tree_manifest=_original_json(manifest,tree_root/'manifest.json',manifest['tree']['manifest_sha256'])
    plan=_original_json(manifest,tree_root/'plan.json')
    if (tree_manifest.get('version') != tree.VERSION or tree_manifest.get('status') != 'completed'
            or tree_manifest.get('plan_sha256') != digest(plan)
            or plan.get('version') != tree.VERSION or plan.get('test_fixture') is not False):
        raise ValueError('Completed same-Actor tree identity differs')
    tree_report_binding=tree_manifest['report']
    if tree_report_binding['path'] != 'report.json': raise ValueError('Tree report path differs')
    tree_report=_original_json(manifest,tree_root/'report.json',tree_report_binding['sha256'])
    if (tree_report.get('version') != tree.VERSION or tree_report.get('status') != 'completed'
            or tree_report.get('reliable') is not True or tree_report.get('endpoint') != endpoint
            or tree_report.get('release_ready') is not False or tree_report.get('explanation_qualified') is not False):
        raise ValueError('Saved terminal tree did not pass its original extraction verification')
    _same(tree_report['verification'],documents['terminal_tree_verification.json'],'Saved tree verification differs')
    if tree_report['program']['path'] != 'program.json': raise ValueError('Tree program path differs')
    _same(_original_json(manifest,tree_root/'program.json',tree_report['program']['sha256']),documents['program.json'],
        'Frozen program differs from originally verified tree')
    descriptor=completion['original_feedback_descriptor'] if is_cycle else completion['source_descriptor']
    bindings=validation['actor_bindings']
    receipt=completion['actor_tensor_receipt']
    if receipt['path'] != 'actor_tensor_receipt.json': raise ValueError('Current tensor proof path differs')
    proof=_original_json(manifest,source/'actor_tensor_receipt.json',receipt['sha256'])
    if (proof.get('version') != reader.VERSION or proof.get('joint_steps') != endpoint
            or proof.get('all_six_arrays_equal') is not True or proof.get('actor_sha256') != bindings['actor_sha256']
            or proof.get('checkpoint_sha256') != completion['checkpoint']['sha256']):
        raise ValueError('Current diagnostic learner/export receipt differs')
    _same(manifest['comparison'],{'path':descriptor['comparison_path'],'sha256':descriptor['comparison_sha256']},
        'Original paired evidence was replaced by the later diagnostic')
    energy._comparison_gate(comparison,descriptor['completion_sha256'],False)
    _same(chosen['original_comparison_sha256'],descriptor['comparison_sha256'],'Original paired anchor differs')
    _same(chosen['completion_sha256'],diagnostic['completion_sha256'],'Diagnostic completion anchor differs')
    _same(chosen['tree_manifest_sha256'],manifest['tree']['manifest_sha256'],'Fresh tree anchor differs')
    _same(chosen['selected_effective_checkpoint'],completion['checkpoint'],'Actual native endpoint checkpoint differs')
    _same(plan['source'],str(_root(diagnostic['directory'])),'Fresh tree belongs to another native source')
    _same(plan['endpoint'],endpoint,'Fresh tree endpoint differs')
    _same(plan.get('source_kind'),'native_cycle' if is_cycle else 'energy','Tree source kind differs')
    _same(plan.get('source_version'),reader.VERSION,'Tree source producer differs')
    _same(plan['source_anchors'],{'completion_sha256':diagnostic['completion_sha256'],
        'actor_sha256':bindings['actor_sha256'],'actor_parameters_sha256':proof['actor_parameters_sha256'],
        'validation_report_sha256':completion['report_sha256']},'Fresh tree current-Actor anchors differ')
    _same(plan['actor_parameter_receipt'],proof,'Current learner/export parameter proof differs')
    _same(plan['actor_bindings'],{**bindings,'actor_parameters_sha256':proof['actor_parameters_sha256']},
        'Fresh tree and genuine native Actor bindings differ')
    _same(tree_report['actor_bindings'],plan['actor_bindings'],'Saved tree report Actor differs')
    for name,value in [('protocol.json',_original_json(manifest,source/'protocol.json')),
                       ('scenarios.json',_original_json(manifest,source/'scenarios.json')),('actor_bindings.json',bindings)]:
        _same(documents[name],value,'Frozen energy candidate differs from its completed evidence: '+name)
    _same(chosen['actor_sha256'],bindings['actor_sha256'],'Actual energy Actor bytes differ')
    _same(chosen['actor_parameters_sha256'],proof['actor_parameters_sha256'],'Actual energy parameter receipt differs')
    return 'native_cycle_fixed' if is_cycle else 'native_energy_50000'


def _candidate(item):
    root = _root(item['directory']); manifest = _json(root/'manifest.json', item['manifest_sha256'])
    if (manifest['version'] not in (selection_api.VERSION,selection_api.ENERGY_VERSION,selection_api.NATIVE_CYCLE_VERSION) or manifest['status'] != 'frozen_for_independent_acceptance'
            or manifest['test_fixture'] is not False or manifest['qualification_evaluated'] is not False):
        raise ValueError('A genuine registered frozen candidate is required')
    _sources(manifest['source_hashes'])
    required = {'actor.npz', 'protocol.json', 'scenarios.json', 'program.json', 'actor_bindings.json',
                'selection.json', 'validation.json', 'terminal_tree_verification.json'}
    if set(manifest['artifacts']) != required: raise ValueError('Incomplete candidate artifact set')
    documents = {}; raw = {}
    for name, entry in manifest['artifacts'].items():
        if entry['path'] != name: raise ValueError('Candidate artifact path differs')
        value = _bytes(root/name, entry['sha256'])
        if len(value) != entry['size']: raise ValueError('Candidate artifact size differs')
        raw[name] = value
        if name.endswith('.json'): documents[name] = original.parse_json(value.decode())
    for path, binding in manifest['original_input_bindings'].items():
        if len(_bytes(path, binding['sha256'])) != binding['size']: raise ValueError('Original candidate inputs changed')
    comparison = _json(manifest['comparison']['path'], manifest['comparison']['sha256'])
    chosen = documents['selection.json']; validation = documents['validation.json']
    route = _candidate_selection(manifest,documents,comparison)
    runtime = ShutdownRuntime(root/'actor.npz', protocol=documents['protocol.json'],
        expected_actor_sha256=chosen['actor_sha256'], expected_protocol_sha256=chosen['protocol_sha256'],
        expected_bindings=documents['actor_bindings.json'], allow_test_fixture=False,
        config=collaborative_study_config(horizon=documents['scenarios.json']['configuration']['horizon']))
    identity = registry.verify(runtime, expected_family='retained_beta197', expected_signature=chosen['runtime_signature'])
    _same(identity, manifest['runtime'], 'Candidate runtime identity differs')
    _same(digest(documents['scenarios.json']), chosen['scenario_manifest_sha256'], 'Candidate scenarios differ')
    if route in ('native_energy_50000','native_cycle_fixed'):
        meta=runtime.actor.metadata
        _same(evidence.actor_parameter_sha256(runtime.actor),chosen['actor_parameters_sha256'],'Current native six-array parameter hash differs')
        _same(meta['source_counters']['joint_steps']+chosen['selected_step'],chosen['selected_cumulative_step'],'Actual native cumulative clock differs')
        for key in ('cycle_id','branch','shutdown_arm','own_shutdown_beta','source_checkpoint_sha256'):
            _same(meta[key],chosen[key],'Actual native Actor selection metadata differs: '+key)
    explainer = FamilyExplainer(root/'program.json', expected_program_sha256=manifest['artifacts']['program.json']['sha256'], runtime=runtime)
    if explainer.eligible or explainer.participant_enabled: raise ValueError('Component qualification was altered')
    return _Context(runtime, documents['scenarios.json'], explainer, None, documents['protocol.json'], raw['selection.json'],
                    {'root': str(root), 'manifest': manifest, 'manifest_sha256': item['manifest_sha256'],'selection_route':route})


def _calibration(item, scenarios):
    root = _root(item['directory']); completed = _json(root/'completion.json', item['completion_sha256'])
    verified = _json(root/'independent_record_verification.json', item['record_verification_sha256'])
    declaration = _json(root/'execution_declaration.json', completed['declaration_sha256'])
    _sources(declaration['source_hashes'])
    _bytes(root/'read_records.py', verified['reader_sha256'])
    _same(verified['completion_sha256'], item['completion_sha256'], 'Calibration reader bound another execution')
    if verified['status'] != 'saved_records_verified' or verified['episodes'] != 77 or verified['matching_status'] != 'passed':
        raise ValueError('Actual fixed task calibration did not pass')
    for name, key in [('raw_steps.jsonl','raw_steps_sha256'), ('step_journal.jsonl','journal_sha256')]:
        _bytes(root/name, completed[key])
    report = _json(root/'original_calibration/report.json', completed['original_report_sha256'])
    for name, expected in report['artifacts'].items(): _bytes(root/'original_calibration'/name, expected)
    actual_scenes = _json(report['scenario_file'], declaration['scenario_file_sha256'])
    _same(actual_scenes, scenarios, 'Task calibration used a different scene manifest')
    rows = [original.parse_json(x) for x in (root/'original_calibration/episodes.jsonl').read_text().splitlines()]
    calculated = original.verify_task_calibration(report, rows, scenarios, declaration['scenario_file_sha256'])
    if (completed['execution']['pending'] is not None or verified['rows'] != completed['execution']['acknowledged']
            or verified['rows'] != sum(r['metrics']['steps'] for r in rows)):
        raise ValueError('Calibration physical record accounting differs')
    return {'passed': True, 'episodes': len(rows), 'recorded_steps': verified['rows'], 'matching': calculated}


def _evidence(bundle):
    if type(bundle) is not dict or set(bundle) != REQUIRED: raise ValueError('All six independent evidence components are required')
    context = _candidate(bundle['candidate']); runtime = context.runtime; scenarios = context.scenarios
    program = Path(context.candidate['root'])/'program.json'; checks = {}
    checks['registered_learning_and_validation'] = {'passed': True, 'selection_route':context.candidate['selection_route'],
        'candidate_manifest_sha256': bundle['candidate']['manifest_sha256']}
    item = bundle['parity']
    identity = registry.verify(runtime)
    parity_bindings = {'runtime_family':identity['family'], 'runtime_version':identity['runtime_version'],
        'runtime_signature':runtime.signature, 'actor_sha256':runtime.actor_sha256,
        'protocol_sha256':runtime.protocol_sha256, 'scenario_manifest_sha256':digest(scenarios),
        'actor_parameters_sha256':evidence.actor_parameter_sha256(runtime.actor),
        'feature_names_sha256':digest(runtime.actor.metadata['feature_names'])}
    parity_result = evidence.read_actor_parity(item['directory'], expected_report_sha256=item['report_sha256'],
        expected_artifact_sha256=item['artifacts'], runtime=runtime, scenarios=scenarios, expected_bindings=parity_bindings)
    checks['numeric_actor_parity'] = {'passed': parity_result['passed_saved_evidence'], 'evidence': parity_result}
    item = bundle['heldout']; root = _root(item['directory'])
    plan = _json(root/'plan.json', item['plan_sha256']); state = _json(root/'state.json', item['state_sha256'])
    bindings = heldout.input_bindings(runtime, program, scenarios)
    _same(plan, heldout_driver._plan(runtime,program,scenarios,bindings,plan['execution_id'],False), 'Heldout input/source plan differs')
    _same(state['manifest_sha256'], item['audit_manifest_sha256'], 'Heldout original manifest differs')
    result = heldout_driver._read_completed(root, plan)['audit_report']
    checks['heldout_policy_and_interventions'] = {'passed': result['statistics']['passed'], 'evidence': result}
    item = bundle['answers']
    answers = evidence.read_answer_evidence(item['directory'],
        expected_generation_report_sha256=item['generation_report_sha256'],
        expected_verification_report_sha256=item['verification_report_sha256'],runtime=runtime,
        program_path=program,scenarios=scenarios,expected_bindings=evidence.answer.input_bindings(runtime,program,scenarios))
    checks['bilingual_verified_answers'] = {'passed': answers['passed_saved_evidence'], 'evidence': answers}
    item = bundle['bank']
    context.question_bank = FrozenFamilyBank(item['directory'], expected_prepared_sha256=item['prepared_sha256'],
        expected_replay_receipt_sha256=item['replay_receipt_sha256'],expected_runtime_signature=runtime.signature,
        expected_actor_sha256=runtime.actor_sha256,expected_exclusions_sha256=item['exclusions_sha256'])
    _bank_selection(item['directory'], item['prepared_sha256'], context.selection_raw)
    # Require the entire genuine non-bank scene pools, not only nonempty labels.
    exclusions = json.loads((Path(item['directory'])/'inputs/exclusion_pools.json').read_bytes())
    for name, entries in scenarios['splits'].items():
        _same(exclusions[name], entries, 'Bank exclusion pool was truncated or changed: '+name)
    checks['frozen_prediction_bank'] = {'passed': context.question_bank.content_eligible, 'summary': context.question_bank.summary()}
    checks['fixed_task_matching'] = _calibration(bundle['calibration'], scenarios)
    if any(value['passed'] is not True for value in checks.values()):
        failed = ', '.join(k for k,v in checks.items() if v['passed'] is not True)
        raise ValueError('Independent acceptance did not pass: ' + failed)
    return context, checks


def _preflight(bundle, context, checks, source_binding):
    value = {'version': PREFLIGHT_VERSION, 'test_fixture': False,
        'bindings': {'actor_sha256': context.runtime.actor_sha256, 'protocol_sha256': context.runtime.protocol_sha256,
            'scenario_manifest_sha256': digest(context.scenarios), 'selection_sha256': sha256(context.selection_raw).hexdigest(),
            'runtime_signature': context.runtime.signature},
        'checks': checks, 'evidence_bundle_sha256': digest(bundle), 'source_binding': source_binding,
        'analysis': original.analysis_protocol(), 'formal_ready': False, 'final_test_used': False}
    value['signature'] = digest(value)
    return value


def prepare_preflight(bundle_path, *, expected_bundle_sha256, output):
    bundle = _json(bundle_path, expected_bundle_sha256); context, checks = _evidence(bundle)
    source_binding = release_sources(); result = _preflight(bundle,context,checks,source_binding)
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent: raise ValueError('Use a new canonical preflight directory')
    output.mkdir()
    for name, value in [('bundle.json',bundle), ('preflight.json',result)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    _sources(source_binding)
    return result


def _completed_preflight(root, expected_preflight_sha256, expected_bundle_sha256):
    root = _root(root); stored = _json(root/'preflight.json', expected_preflight_sha256)
    bundle = _json(root/'bundle.json', expected_bundle_sha256)
    context, checks = _evidence(bundle)
    _same(stored, _preflight(bundle,context,checks,release_sources()), 'Preflight evidence or code changed')
    return context, stored


def _final(context, preflight, item):
    result = final_api.read_completed(item['directory'], expected_receipt_sha256=item['receipt_sha256'], runtime=context.runtime,
        scenarios=context.scenarios,protocol=context.protocol,selection_raw=context.selection_raw,preflight=preflight,
        source_binding=final_api.evaluation_sources())
    reports = result['reports']
    passed = capability(reports['final_test'],reports['final_reference'],reports['final_random'],_gates(context.protocol))
    if passed['eligible'] is not True: raise ValueError('Independent final capability failed')
    return {'capability': passed, 'receipt': result['receipt'], 'verification': result['verification']}


def assemble_local_release(preflight_root, *, expected_preflight_sha256, expected_bundle_sha256, final, output):
    context, preflight = _completed_preflight(preflight_root, expected_preflight_sha256, expected_bundle_sha256)
    result = _final(context,preflight,final)
    manifest = {'version':VERSION,'status':'local_pilot_technically_verified','namespace':'local_pilot','formal_ready':False,
        'preflight':{'directory':str(_root(preflight_root)),'preflight_sha256':expected_preflight_sha256,'bundle_sha256':expected_bundle_sha256},
        'final':deepcopy(final),'final_capability':result['capability'],'sources':release_sources(),
        'analysis':original.analysis_protocol(),'test_fixture':False,'online_deployment':False,
        'human_explanation_effect_validated':False}
    output=Path(output).absolute()
    if output.exists() or output.parent.resolve()!=output.parent: raise ValueError('Use a new canonical release directory')
    output.mkdir(); (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return manifest


def load_family_release(root, *, expected_manifest_sha256):
    root = _root(root); manifest = _json(root/'manifest.json', expected_manifest_sha256)
    if (manifest['version']!=VERSION or manifest['status']!='local_pilot_technically_verified'
            or manifest['test_fixture'] is not False or manifest['formal_ready'] is not False
            or manifest['namespace']!='local_pilot' or manifest['online_deployment'] is not False):
        raise ValueError('A completed genuine local pilot release is required')
    _same(manifest['sources'],release_sources(),'Service or evidence reader sources changed')
    _sources(manifest['sources']); _same(manifest['analysis'],original.analysis_protocol(),'Frozen analysis changed')
    item = manifest['preflight']; context, preflight = _completed_preflight(item['directory'],item['preflight_sha256'],item['bundle_sha256'])
    result = _final(context,preflight,manifest['final']); _same(result['capability'],manifest['final_capability'],'Final ability changed')
    context.manifest_sha256 = expected_manifest_sha256; context.source_binding = manifest['sources']
    context.signature = digest({'version':VERSION,'manifest_sha256':expected_manifest_sha256,'runtime':context.runtime.signature,
        'bank':context.question_bank.signature,'explainer':context.explainer.signature})
    context.release = {'status':manifest['status'],'namespace':'local_pilot','model_ready':True,'explanation_ready':True,
        'study_ready':True,'formal_ready':False,'test_fixture':False,'qualification_evaluated':True,
        'message':{'zh':'本地预实验：技术验收已通过，解释效果仍需人类实验验证。',
                   'en':'Local pilot: technical acceptance passed; explanation effects require human evaluation.'}}
    context.provenance = {'version':VERSION,'namespace':'local_pilot','manifest_sha256':expected_manifest_sha256,
        'actor_sha256':context.runtime.actor_sha256,'runtime_signature':context.runtime.signature,
        'protocol_sha256':context.runtime.protocol_sha256,'scenario_manifest_sha256':digest(context.scenarios),
        'analysis':original.analysis_protocol(),'source_binding':context.source_binding,'release':deepcopy(context.release)}
    return context
