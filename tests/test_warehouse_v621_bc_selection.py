"""Synthetic receipts/checkpoints only; no reset, BC or scientific job."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.warehouse import v621_bc_selection as selection
from experiments.warehouse import v621_bc_protocol as protocol
from backend.training import warehouse_v621_actor_training as training
from backend.training.warehouse_v621_bc_pipeline import array_sha


ROOT = Path(__file__).resolve().parents[1]
CASES = protocol.diagnostics.source_cases()


def pointer(path):
    return {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return pointer(path)


def quality_fixture(recipe="shared_b035", deliveries=(6, 7, 8), collisions=(1, 1, 1)):
    results, collections = {}, {}
    for i, d, c in zip(protocol.INITIALIZATIONS, deliveries, collisions, strict=True):
        outcomes = [{"case": deepcopy(case), "deliveries": d, "robot_collisions": c,
            "shutdown": False} for case in CASES]
        passed = d >= 5.5 and c <= 6
        results[i] = {"initialization": i, "recipe": recipe, "status": "completed",
            "quality": {"mean_deliveries": float(d), "mean_collisions": float(c), "shutdown_episode_rate": 0.},
            "quality_passed": passed, "program_passed": True, "gradient_probe_passed": True,
            "passed": passed, "PPO_joint_steps": 0, "publishable": False}
        collections[i] = {"cases": deepcopy(CASES), "outcomes": outcomes}
    return results, collections


def test_exact_integer_score_and_only_quality_ranks():
    rows, collections = quality_fixture(deliveries=(8, 7, 7), collisions=(6, 0, 0))
    for i in rows: rows[i]["coverage"] = (i-6170)*.1
    result = selection.select_quality_only(rows, collections)
    assert result["selected_initialization"] == 6172
    assert result["scores"]["6171"]["score_integer_numerator"] == 1950
    assert result["scores"]["6172"]["score_integer_numerator"] == 2100
    assert result["scores"]["6172"]["common_denominator"] == 300


def test_main_priority_ignores_better_backup_quality():
    rows, collections = quality_fixture(deliveries=(6, 6, 6))
    backup, backup_collections = quality_fixture("shared_b050", deliveries=(15, 16, 17))
    result = selection.select_quality_only(rows, collections, backup_results=backup,
        backup_collections=backup_collections)
    assert result["selected_initialization"] == 6171 and result["candidate_recipe"] == "shared_b035"


def test_backup_needs_actual_quality_failure_and_all_three_pass():
    rows, collections = quality_fixture(deliveries=(4, 6, 6))
    backup, backup_collections = quality_fixture("shared_b050")
    result = selection.select_quality_only(rows, collections, backup_results=backup,
        backup_collections=backup_collections)
    assert result["candidate_recipe"] == "shared_b050" and result["selected_initialization"] == 6173
    backup[6173].update(program_passed=False, passed=False)
    with pytest.raises(RuntimeError, match="all three backup"):
        selection.select_quality_only(rows, collections, backup_results=backup, backup_collections=backup_collections)


@pytest.mark.parametrize("problem", ("missing", "mixed_recipe", "inconsistent_gate", "program_only", "gradient_only"))
def test_incomplete_or_wrong_recipe_cannot_select(problem):
    rows, collections = quality_fixture()
    if problem == "missing": del rows[6173]
    elif problem == "mixed_recipe": rows[6173]["recipe"] = "shared_b050"
    elif problem == "inconsistent_gate": rows[6173]["passed"] = 1
    else:
        rows[6173]["program_passed" if problem == "program_only" else "gradient_probe_passed"] = False
        rows[6173]["passed"] = False
    with pytest.raises(RuntimeError): selection.select_quality_only(rows, collections)


@pytest.mark.parametrize("problem", ("omitted", "reordered", "argmax", "float_counts", "bool_counts", "numeric_shutdown", "summary"))
def test_quality_score_requires_complete_same_natural_outcomes(problem):
    rows, collections = quality_fixture()
    result, collection = rows[6173], collections[6173]
    target = next(x for x in collection["outcomes"] if x["case"]["original_split"] == "validation")
    if problem == "omitted": collection["outcomes"].pop()
    if problem == "reordered": collection["outcomes"].reverse()
    if problem == "argmax": target["case"]["sampling_mode"] = "argmax"
    if problem == "float_counts": target["deliveries"] = 8.
    if problem == "bool_counts": target["deliveries"] = True
    if problem == "numeric_shutdown": target["shutdown"] = 0
    if problem == "summary": result["quality"]["mean_deliveries"] = 9.
    with pytest.raises(RuntimeError): selection.select_quality_only(rows, collections)


def one_leaf_model(bound=.35):
    # Synthetic initializer zero. The helper never acquires a capability.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        model = training.WarehouseV621Actor(core_residual_bound=bound)
    structure = {"root_ref": -1, "node_count": 0, "leaf_count": 1,
        "split_weights": np.zeros((0, training.FEATURE_DIM), np.float32),
        "split_bias": np.zeros(0, np.float32), "left_refs": np.zeros(0, np.int64),
        "right_refs": np.zeros(0, np.int64)}
    for role in selection.ROLES:
        model.install_structure(role, structure, [1], [4], np.zeros((1, 5), np.float32))
    return model


@pytest.fixture
def completed_fixture(tmp_path, monkeypatch):
    # Mock the native checker/prerequisite I/O boundaries, not the new strict
    # model-state restoration or actual versioned NPZ loader.
    manifest = json.loads((ROOT/'experiments/warehouse/evidence/warehouse-v621-bc-source-manifest-02.json').read_bytes())
    monkeypatch.setattr(selection, '_verify_source_closure', lambda _: deepcopy(manifest))
    monkeypatch.setattr(selection, '_verify_prerequisites', lambda *_: {})
    registration = tmp_path/'output/warehouse_v621/selector-registration-01.json'
    registration.parent.mkdir(parents=True)
    registration.write_bytes((ROOT/'output/warehouse_v621/selector-registration-01.json').read_bytes())
    directory = tmp_path/'output/warehouse_v621'/protocol.BATCH
    spec = {"schema": "warehouse-v621-bc-development-spec.v1", "candidate_id": protocol.diagnostics.CANDIDATE_ID,
        "batch_id": protocol.BATCH, "source_closure_sha256": selection.SOURCE02_SHA256,
        "source_files_sha256": manifest,
        "model_implementation_sha256": protocol.digest({p: manifest[p] for p in selection.MODEL_CODE_FILES}),
        "output_directory": str(directory), "PPO_joint_steps": 0,
        "fresh_precheck_authorized": False, "publishable": False,
        "recipe": write(tmp_path/'configs/warehouse_v621_bc_recipe.json', protocol.recipe()),
        "seed_plan": write(tmp_path/'configs/warehouse_v621_bc_seed_plan.json', protocol.seed_plan())}
    spec_path = tmp_path/'output/warehouse_v621/synthetic-bc.spec.json'
    spec_ref = write(spec_path, spec)
    rows, collections = quality_fixture()
    model = one_leaf_model()
    refs = {"additive_control": {}, "shared_b035": {}}
    for recipe in refs:
        for i in protocol.INITIALIZATIONS:
            folder = directory/f'init-{i}'/recipe
            artifact = training.export_actor(model, folder/'final-bc.actor.npz', pair_seed=i,
                variant='raw', candidate_id='synthetic-receipt-only')
            checkpoint = folder/'final-bc.checkpoint.pt'
            torch.save({"schema": "warehouse-v621-common-bc-checkpoint.v1", "model_version": model.model_version,
                "recipe": recipe, "initialization": i, "source_closure_sha256": selection.SOURCE02_SHA256,
                "seed_plan_sha256": spec['seed_plan']['sha256'], "model_state": model.state_dict(),
                "actor_artifact": artifact, "raw_and_regularized_must_clone_exactly": True,
                "PPO_joint_steps": 0}, checkpoint)
            report = {"checkpoint": pointer(checkpoint), "both_roles_changed": True,
                "external_program_feedback": 0, "external_program_actor_updates": 0,
                "role_maximum_bc_gradient_norm": {r: 1. for r in selection.ROLES},
                "post_install_bc_start_parameter_sha256": {r: '1'*64 for r in selection.ROLES},
                "trained_state_sha256": {r: array_sha(**{k: v.detach().numpy() for k, v in
                    model.actors[r].state_dict().items()}) for r in selection.ROLES},
                "trained_parameter_sha256": {r: array_sha(**{k: v.detach().numpy() for k, v in
                    model.actors[r].named_parameters() if v.requires_grad}) for r in selection.ROLES}}
            result = {**deepcopy(rows[i]), "recipe": recipe, "specification": spec_ref,
                "actor": artifact, "training_report": write(folder/'bc.training.json', report),
                "development_collection": write(folder/'development.collection.json', collections[i]),
                "success_journal": write(folder/'native-successful-resets.jsonl', []),
                "preauthorization_journal": write(folder/'reset-preauthorization.jsonl', []),
                "roles": {r: {"program": write(folder/(r+'.program.json'), {"synthetic": True})} for r in selection.ROLES}}
            result_ref = write(folder/'result.json', result)
            cap_ref = write(folder/'capability.completed.json', {
                "schema": "warehouse-v621-bc-capability-completed.v1", "initialization": i,
                "recipe": recipe, "preauthorized_resets": 1005})
            write(folder/'completed.json', {"result": result_ref, "capability_completed": cap_ref,
                "source_closure_sha256": selection.SOURCE02_SHA256})
            refs[recipe][str(i)] = result_ref
    summary = {"schema": "warehouse-v621-bc-development-summary.v1", "specification": spec_ref,
        "results": refs, "candidate_id": protocol.diagnostics.CANDIDATE_ID,
        "source_closure_sha256": selection.SOURCE02_SHA256, "candidate_recipe": "shared_b035",
        "main_quality_failure": False,
        "all_three_candidate_initializations_passed": True,
        "next_stage": "lock_common_bc_and_implement_fresh_precheck", "fresh_precheck_authorized": False,
        "PPO_joint_steps": 0, "publishable": False, "pilot_opened": False}
    write(directory/'main.summary.json', summary)
    calls = []
    def native(spec, i, recipe):
        calls.append((i, recipe))
        path = Path(spec['output_directory'])/f'init-{i}'/recipe/'result.json'
        return json.loads(path.read_bytes()), pointer(path)
    from scripts import run_warehouse_v621_bc_development as entry
    monkeypatch.setattr(entry, 'checked_result', native)
    return {'root': tmp_path, 'specification_path': spec_path, 'registration_path': registration,
        'directory': directory, 'calls': calls, 'model': model}


def build(f):
    return selection.build_model_lock(**{k: f[k] for k in ('root','specification_path','registration_path')})


def update_result(f, i, recipe, mutate):
    folder = f['directory']/f'init-{i}'/recipe
    value = json.loads((folder/'result.json').read_bytes()); mutate(value)
    ref = write(folder/'result.json', value)
    complete = json.loads((folder/'completed.json').read_bytes()); complete['result'] = ref
    write(folder/'completed.json', complete)
    path = f['directory']/'main.summary.json'
    summary = json.loads(path.read_bytes()); summary['results'][recipe][str(i)] = ref
    write(path, summary)


def update_checkpoint(f, i, mutate):
    folder = f['directory']/f'init-{i}'/'shared_b035'; path = folder/'final-bc.checkpoint.pt'
    state = torch.load(path, weights_only=True, map_location='cpu'); mutate(state); torch.save(state, path)
    report = json.loads((folder/'bc.training.json').read_bytes()); report['checkpoint'] = pointer(path)
    ref = write(folder/'bc.training.json', report)
    update_result(f, i, 'shared_b035', lambda row: row.update(training_report=ref))


def test_readonly_lock_checks_three_full_models_and_restores_rng(completed_fixture):
    before = torch.get_rng_state().clone(); result = build(completed_fixture)
    assert torch.equal(before, torch.get_rng_state())
    assert result['selected_initialization'] == 6173 and len(result['candidate_models']) == 3
    assert len(completed_fixture['calls']) == 6
    for model in result['candidate_models'].values():
        assert model['restored_role_sha256'] == model['actor']['role_sha256']
        assert not model['restoration']['optimizer_created']
        assert model['restoration']['external_cpu_torch_rng_restored']
    assert result['environment_resets'] == result['actor_updates'] == result['PPO_joint_steps'] == 0
    assert not result['fresh_precheck_authorized'] and not result['pilot_opened'] and not result['publishable']


@pytest.mark.parametrize('problem', ('actor_missing','checkpoint_missing','failed','registration','source','pointer_path','symlink','checkpoint_sha'))
def test_missing_changed_or_uncontained_evidence_rejected(completed_fixture, problem):
    f = completed_fixture; folder = f['directory']/'init-6171'/'shared_b035'
    if problem == 'actor_missing': (folder/'final-bc.actor.npz').unlink()
    if problem == 'checkpoint_missing': (folder/'final-bc.checkpoint.pt').unlink()
    if problem == 'failed': write(folder/'capability.failed.json', {})
    if problem == 'registration': f['registration_path'].write_text('{}')
    if problem == 'source':
        value = json.loads(f['specification_path'].read_bytes()); value['source_closure_sha256']='f'*64
        write(f['specification_path'], value)
    if problem == 'pointer_path':
        other = folder/'other.actor.npz'; other.write_bytes((folder/'final-bc.actor.npz').read_bytes())
        update_result(f,6171,'shared_b035',lambda row: row['actor'].update(path=str(other)))
    if problem == 'symlink':
        actual = folder/'actual.actor.npz'; (folder/'final-bc.actor.npz').rename(actual)
        (folder/'final-bc.actor.npz').symlink_to(actual)
    if problem == 'checkpoint_sha':
        with (folder/'final-bc.checkpoint.pt').open('ab') as handle: handle.write(b'changed')
    with pytest.raises((RuntimeError,FileNotFoundError)): build(f)


@pytest.mark.parametrize('problem', ('missing_critic','wrong_dtype','nonfinite','actor_drift','wrong_recipe','PPO','wrong_seed_plan'))
def test_checkpoint_complete_exact_and_export_equivalent(completed_fixture, problem):
    def mutate(saved):
        state=saved['model_state']
        if problem == 'missing_critic': del state['cost_critic.0.weight']
        if problem == 'wrong_dtype': state['cost_critic.0.weight']=state['cost_critic.0.weight'].double()
        if problem == 'nonfinite': state['cost_critic.0.weight'][0,0]=float('nan')
        if problem == 'actor_drift': state['actors.robot_1.action_residual.0.bias'][0]+=1
        if problem == 'wrong_recipe': saved['recipe']='shared_b050'
        if problem == 'PPO': saved['PPO_joint_steps']=1
        if problem == 'wrong_seed_plan': saved['seed_plan_sha256']='f'*64
    update_checkpoint(completed_fixture,6171,mutate); before=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError): build(completed_fixture)
    assert torch.equal(before,torch.get_rng_state())


def test_training_report_must_match_final_tensors(completed_fixture):
    f=completed_fixture; folder=f['directory']/'init-6171'/'shared_b035'
    report=json.loads((folder/'bc.training.json').read_bytes()); report['trained_parameter_sha256']['robot_2']='f'*64
    ref=write(folder/'bc.training.json',report)
    update_result(f,6171,'shared_b035',lambda row:row.update(training_report=ref))
    with pytest.raises(RuntimeError,match='final role tensors'): build(f)


def install_backup_receipts(f):
    """Build explicitly synthetic b=.50 receipts after an actual quality fail."""
    folder=f['directory']/'init-6171'/'shared_b035'
    collection=json.loads((folder/'development.collection.json').read_bytes())
    for row in collection['outcomes']: row['deliveries']=4
    ref=write(folder/'development.collection.json',collection)
    update_result(f,6171,'shared_b035',lambda row:row.update(development_collection=ref,
        quality={'mean_deliveries':4.,'mean_collisions':1.,'shutdown_episode_rate':0.},
        quality_passed=False,passed=False))
    spec=json.loads(f['specification_path'].read_bytes())
    old_summary=json.loads((f['directory']/'main.summary.json').read_bytes())
    backup_spec={**spec,'main_results':old_summary['results']['shared_b035']}
    backup_spec_ref=write(f['directory']/'backup.spec.json',backup_spec)
    rows,collections=quality_fixture('shared_b050')
    model=one_leaf_model(.50); refs={}
    for i in protocol.INITIALIZATIONS:
        main=f['directory']/f'init-{i}'/'shared_b035'
        folder=f['directory']/f'init-{i}'/'shared_b050'
        artifact=training.export_actor(model,folder/'final-bc.actor.npz',pair_seed=i,
            variant='raw',candidate_id='synthetic-receipt-only')
        old=torch.load(main/'final-bc.checkpoint.pt',weights_only=True,map_location='cpu')
        old.update(recipe='shared_b050',model_state=model.state_dict(),actor_artifact=artifact)
        path=folder/'final-bc.checkpoint.pt';torch.save(old,path)
        report=json.loads((main/'bc.training.json').read_bytes());report['checkpoint']=pointer(path)
        report['trained_state_sha256']={r:array_sha(**{k:v.detach().numpy() for k,v in
            model.actors[r].state_dict().items()}) for r in selection.ROLES}
        report['trained_parameter_sha256']={r:array_sha(**{k:v.detach().numpy() for k,v in
            model.actors[r].named_parameters() if v.requires_grad}) for r in selection.ROLES}
        result={**rows[i],'specification':backup_spec_ref,'actor':artifact,
            'training_report':write(folder/'bc.training.json',report),
            'development_collection':write(folder/'development.collection.json',collections[i]),
            'success_journal':write(folder/'native-successful-resets.jsonl',[]),
            'preauthorization_journal':write(folder/'reset-preauthorization.jsonl',[]),
            'roles':{r:{'program':write(folder/(r+'.program.json'),{'synthetic':True})} for r in selection.ROLES}}
        ref=write(folder/'result.json',result);refs[str(i)]=ref
        cap=write(folder/'capability.completed.json',{'schema':'warehouse-v621-bc-capability-completed.v1',
            'initialization':i,'recipe':'shared_b050','preauthorized_resets':1005})
        write(folder/'completed.json',{'result':ref,'capability_completed':cap,
            'source_closure_sha256':selection.SOURCE02_SHA256})
    write(f['directory']/'backup.summary.json',{**old_summary,'specification':backup_spec_ref,
        'candidate_recipe':'shared_b050','main_quality_failure':True,
        'results':{**old_summary['results'],'shared_b050':refs}})


def test_complete_backup_chain_restores_real_b050_models_only_after_main_quality_failure(completed_fixture):
    install_backup_receipts(completed_fixture)
    result=build(completed_fixture)
    assert result['candidate_recipe']=='shared_b050' and result['selected_initialization']==6173
    assert len(completed_fixture['calls'])==9
    assert result['native_bc_summary']['path'].endswith('backup.summary.json')
    assert all(m['restored_role_sha256']==m['actor']['role_sha256'] for m in result['candidate_models'].values())


def test_backup_spec_cannot_rebind_completed_main_identity(completed_fixture):
    install_backup_receipts(completed_fixture)
    path=completed_fixture['directory']/'backup.spec.json'
    value=json.loads(path.read_bytes());value['main_results']['6171']['sha256']='f'*64
    write(path,value)
    with pytest.raises(RuntimeError,match='backup specification'):build(completed_fixture)


def test_source02_manifest_and_registration_still_match_locked_bytes():
    assert len(selection._verify_source_closure(ROOT))==194
    assert pointer(ROOT/'output/warehouse_v621/selector-registration-01.json')['sha256']==selection.REGISTRATION_SHA256


def test_bound_evidence_mutation_during_restoration_is_rejected(completed_fixture, monkeypatch):
    original=selection._checkpoint
    def altered(root,result,spec,manifest,bound):
        verified=original(root,result,spec,manifest,bound)
        if result['initialization']==6173:
            with Path(result['roles']['robot_1']['program']['path']).open('a') as stream:stream.write(' ')
        return verified
    monkeypatch.setattr(selection,'_checkpoint',altered)
    with pytest.raises(RuntimeError,match='evidence changed'):build(completed_fixture)


def test_external_meta_default_device_does_not_move_restore_or_consume_backend_rng(completed_fixture):
    before=torch.get_rng_state().clone()
    with torch.device('meta'):
        assert torch.empty(0).device.type=='meta'
        result=build(completed_fixture)
        assert torch.empty(0).device.type=='meta'
    assert torch.equal(before,torch.get_rng_state())
    assert all(row['restoration']['constructor_device']=='cpu' for row in result['candidate_models'].values())


def test_changed_default_dtype_fails_closed_and_preserves_global_state(completed_fixture):
    before=torch.get_rng_state().clone();original=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with pytest.raises(RuntimeError,match='float32 constructor dtype'):build(completed_fixture)
        assert torch.get_default_dtype()==torch.float64
    finally:
        torch.set_default_dtype(original)
    assert torch.equal(before,torch.get_rng_state())


def test_json_sha_and_semantics_use_the_same_single_read(tmp_path, monkeypatch):
    path=tmp_path/'pointer.json';ref=write(path,{'bound':'A'})
    original=Path.read_bytes;reads=[]
    def changed(self):
        if self==path:
            reads.append(self)
            return original(self) if len(reads)==1 else b'{"bound":"unbound-B"}'
        return original(self)
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert selection._read_pointer(tmp_path,ref)=={'bound':'A'}
    assert len(reads)==1


def test_output_is_readonly_nonoverwriting_and_contained(completed_fixture):
    f=completed_fixture; args={k:f[k] for k in ('root','specification_path','registration_path')}
    with pytest.raises(RuntimeError,match='new JSON'):
        selection.write_model_lock(**args,output_path=f['root']/'outside.json')
    output=f['root']/'output/warehouse_v621/common-bc-locks/synthetic.json'
    ref=selection.write_model_lock(**args,output_path=output)
    assert ref==pointer(output) and output.stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError,match='new JSON'): selection.write_model_lock(**args,output_path=output)


def test_cli_has_no_reset_training_or_fresh_operation(monkeypatch,tmp_path,capsys):
    from scripts import lock_warehouse_v621_common_bc as cli
    seen=[]
    monkeypatch.setattr(selection,'write_model_lock',lambda **kw:seen.append(kw) or {'path':'/synthetic','sha256':'a'*64})
    assert cli.main(['--spec',str(tmp_path/'spec'),'--output',str(tmp_path/'out')])==0
    result=json.loads(capsys.readouterr().out)
    assert not result['fresh_precheck_authorized'] and result['PPO_joint_steps']==0 and len(seen)==1
    with pytest.raises(SystemExit): cli.main(['--spec','x','--output','y','--fresh'])
