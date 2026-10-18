"""BC-only capability tests with synthetic BC callbacks; no model training."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest
import torch

from backend.training import warehouse_v6_runner as runner
from backend.training.warehouse_v6_actor import WarehouseV6Actor, export_actor, role_hashes
from env.warehouse.reset_audit_v614 import record_successful_reset, stage1_reset_phase
from experiments.warehouse import protocol as p


@pytest.fixture
def capability(tmp_path, monkeypatch):
    # Test-only capability setup, confined to pytest's temporary filesystem.
    # No resulting receipt is a scientific run or model acceptance evidence.
    monkeypatch.setattr(runner, '_SOURCE_ROOT', tmp_path)
    monkeypatch.setattr(runner, '_LOCKED_EXECUTION_USED_IN_PROCESS', False)
    monkeypatch.setattr(runner, 'source_closure_sha256', lambda: 'f'*64)
    monkeypatch.setattr(runner, 'validate_v618_quality_selective_recipe',
                        lambda: p.REGISTERED_BC_SPEC['executable_recipe_file_sha256'])
    script = tmp_path/'scripts'/'run_warehouse_v618_bc_preflight.py'
    script.parent.mkdir()
    script.write_text('# Synthetic saved entry point for capability unit tests only.\n')
    monkeypatch.setattr(sys.modules['__main__'], '__file__', str(script))
    output = tmp_path/'output'/'warehouse_v618_preflight'
    output.mkdir(parents=True)
    spec = p.PairRunSpec(p.REGISTERED_CANDIDATE_ID+'-bc-preflight', 6150, 200_000,
        str(output), p.REGISTERED_PROGRAM_FIT_METHOD, checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    directory = output/spec.candidate_id/'pair_6150_steps_200000'
    artifacts = {
        'warmup_actor_path': str(directory/'warmup.actor.npz'),
        'final_bc_actor_path': str(directory/'final-bc.actor.npz'),
        'final_bc_precheck_receipt_path': str(directory/'final-bc-internal-precheck.json'),
    }
    raw = {**spec.to_mapping(), **artifacts, 'source_closure_sha256': 'f'*64}
    checked = runner._checked_spec(raw)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6150)
        initial = role_hashes(WarehouseV6Actor())
    required = {
        'authorization_kind': 'warehouse-v618-full-bc-preflight-only-v1',
        'candidate_id': spec.candidate_id, 'pair_seed': 6150,
        'future_pair_joint_steps': 200_000, 'PPO_joint_steps': 0, 'publishable': False,
        'source_closure_sha256': 'f'*64,
        'run_spec_sha256': runner._scientific_run_spec_sha256(checked),
        'training_config_sha256': runner._training_config_sha256(checked),
        'bc_recipe_sha256': p.REGISTERED_BC_SPEC_SHA256,
        'executable_recipe_file_sha256': runner.validate_v618_quality_selective_recipe(),
        'action_physics_config_sha256': runner.ACTION_PHYSICS_CONFIG_SHA256,
        'bc_data_seed_schedule_sha256': checked['bc_data_seed_schedule_sha256'],
        'bc_critic_reset_schedule_sha256': checked['bc_critic_reset_schedule_sha256'],
        'initial_role_sha256': initial, 'output_directory': str(directory),
        'artifact_paths_sha256': runner._digest_json(artifacts),
        'expected_reset_sequence_sha256': runner._digest_json(
            runner._formal_bc_reset_sequence(checked, include_quality_screen=True)),
        'lock_script_path': str(script),
        'lock_script_sha256': sha256(script.read_bytes()).hexdigest(),
    }
    authorization = tmp_path/'authorization.json'
    authorization.write_text(json.dumps(required))

    def forbidden(*_args, **_kwargs):
        raise AssertionError('BC preflight reached pair training or PPO')

    monkeypatch.setattr(runner, 'train_pair', forbidden)
    monkeypatch.setattr(runner, '_train_variant', forbidden)
    monkeypatch.setattr(runner, 'ppo_update', forbidden)
    return raw, checked, authorization, directory


def _record(row):
    physics = runner.ACTION_PHYSICS_CONFIG
    with stage1_reset_phase(row['phase']):
        record_successful_reset(row['seed'], 1, physics['map_layout_id'],
                                physics['move_battery_cost'], physics['charge_per_wait'])


def test_full_bc_capability_finishes_at_precheck_without_starting_ppo(capability, monkeypatch):
    raw, checked, authorization, directory = capability
    called = []

    def fake_bc(model, *, pair_seed, spec):
        called.append(pair_seed)
        for row in runner._formal_bc_reset_sequence(spec, include_quality_screen=True):
            _record(row)
        asset = export_actor(model, spec['final_bc_actor_path'], pair_seed=6150,
                             variant='raw', candidate_id=spec['candidate_id'])
        precheck = {'final_bc_actor': asset, 'passed': True, 'program_precheck_passed': True,
                    'quality_screen': {'summary': {'development_quality_passed': True}}}
        Path(spec['final_bc_precheck_receipt_path']).write_text(json.dumps(precheck))
        return {'initial_role_sha256': role_hashes(model), 'pretrained_role_sha256': role_hashes(model),
                'final_bc_internal_precheck': precheck, 'synthetic_unit_test_only': True}

    monkeypatch.setattr(runner, '_pretrain_pair_production', fake_bc)
    result = runner.run_locked_bc_preflight(raw, authorization_receipt_path=authorization)
    receipt = result['receipt']
    assert called == [6150]
    assert receipt['status'] == 'completed' and receipt['publishable'] is False
    assert receipt['PPO_joint_steps'] == 0 and receipt['future_pair_joint_steps'] == 200_000
    assert receipt['actual_reset_count'] == 1420 and receipt['exact_full_sequence_matches']
    assert receipt['program_precheck_passed'] and receipt['quality_screen_passed']
    assert receipt['final_actor']['role_sha256'] == receipt['current_role_sha256']
    assert runner._ACTIVE_LOCKED_EXECUTION.get() is None
    with pytest.raises(RuntimeError, match='capability unavailable'):
        runner.run_locked_bc_preflight(raw, authorization_receipt_path=authorization)


@pytest.mark.parametrize('failure_kind', ['gate', 'wrong_reset', 'interrupted'])
def test_bc_failure_always_preserves_real_reset_hook_journal_and_zero_ppo(
        capability, monkeypatch, failure_kind):
    raw, checked, authorization, directory = capability
    expected = runner._formal_bc_reset_sequence(checked, include_quality_screen=True)

    def fake_bc(model, *, pair_seed, spec):
        row = dict(expected[0])
        if failure_kind == 'wrong_reset':
            row['seed'] += 1
        _record(row)
        if failure_kind == 'interrupted':
            # A partially written JSON file must not suppress the final audit.
            Path(spec['final_bc_precheck_receipt_path']).write_text('{')
            raise KeyboardInterrupt('synthetic interruption')
        Path(spec['final_bc_precheck_receipt_path']).write_text(json.dumps({
            'program_precheck_passed': False,
            'quality_screen': {'summary': {'development_quality_passed': False}}}))
        raise RuntimeError('synthetic BC gate failure')

    monkeypatch.setattr(runner, '_pretrain_pair_production', fake_bc)
    expected_exception = KeyboardInterrupt if failure_kind == 'interrupted' else RuntimeError
    with pytest.raises(expected_exception):
        runner.run_locked_bc_preflight(raw, authorization_receipt_path=authorization)
    receipt = json.loads((directory/'bc-preflight.receipt.json').read_text())
    assert receipt['PPO_joint_steps'] == 0 and receipt['publishable'] is False
    assert receipt['actual_reset_count'] == 1 and not receipt['exact_full_sequence_matches']
    assert sha256((directory/'bc-preflight.resets.jsonl').read_bytes()).hexdigest() == receipt['reset_journal_sha256']
    category = {'gate': 'program_and_quality_precheck_failed',
                'wrong_reset': 'reset_sequence_violation', 'interrupted': 'interrupted'}[failure_kind]
    assert receipt['error_category'] == category
    assert receipt['declared_prefix_matches'] is (failure_kind != 'wrong_reset')
    assert runner._ACTIVE_LOCKED_EXECUTION.get() is None
    assert (directory/'final-bc-internal-precheck.json').is_file()
    if failure_kind == 'interrupted':
        assert receipt['precheck_read_error'].startswith('JSONDecodeError')


def test_bad_authorization_is_rejected_before_consuming_or_creating_output(capability):
    raw, _, authorization, directory = capability
    payload = json.loads(authorization.read_text())
    payload['lock_script_sha256'] = '0'*64
    authorization.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match='authorization mismatch'):
        runner.run_locked_bc_preflight(raw, authorization_receipt_path=authorization)
    assert not directory.exists() and runner._LOCKED_EXECUTION_USED_IN_PROCESS is False


def test_ordinary_bc_stays_closed_and_v618_config_rejects_legacy_recipe_hash(capability):
    _, checked, _, _ = capability
    with pytest.raises(RuntimeError, match='review-only'):
        runner._pretrain_pair_production(WarehouseV6Actor(), pair_seed=6150, spec=checked)
    runner._training_config_sha256({**checked, 'executable_recipe_file_sha256':
        p.REGISTERED_BC_SPEC['executable_recipe_file_sha256']})
    with pytest.raises(RuntimeError, match='drifted'):
        runner._training_config_sha256({**checked, 'executable_recipe_file_sha256': '0'*64})
