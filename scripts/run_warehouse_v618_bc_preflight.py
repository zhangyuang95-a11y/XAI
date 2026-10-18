#!/usr/bin/env python3
"""Prepare or run source-locked full BC only; this entry point cannot start PPO."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_new(path, payload):
    with path.open('x') as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write('\n')


def prepare():
    import torch
    from experiments.warehouse import protocol as p
    from backend.training import warehouse_v6_runner as runner
    from backend.training.warehouse_v6_actor import WarehouseV6Actor, role_hashes

    output = (ROOT / 'output' / 'warehouse_v618_preflight').resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate = p.REGISTERED_CANDIDATE_ID + '-bc-preflight'
    spec = p.PairRunSpec(candidate_id=candidate, pair_seed=6150, joint_steps=200_000,
        output_dir=str(output), program_fit_method=p.REGISTERED_PROGRAM_FIT_METHOD,
        checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)
    directory = output / candidate / 'pair_6150_steps_200000'
    if directory.exists():
        raise RuntimeError('BC preflight was already attempted; retain its evidence')
    artifacts = {
        'warmup_actor_path': str(directory / 'warmup.actor.npz'),
        'final_bc_actor_path': str(directory / 'final-bc.actor.npz'),
        'final_bc_precheck_receipt_path': str(directory / 'final-bc-internal-precheck.json'),
    }
    raw_spec = {**spec.to_mapping(), **artifacts,
                'source_closure_sha256': runner.source_closure_sha256()}
    checked = runner._checked_spec(raw_spec)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6150)
        initial = role_hashes(WarehouseV6Actor())
    metadata = output / '.authorizations' / (candidate + '-6150')
    metadata.mkdir(parents=True, exist_ok=False)
    spec_path, authorization_path = metadata/'spec.json', metadata/'authorization.json'
    _write_new(spec_path, raw_spec)
    main = Path(__file__).resolve(strict=True)
    authorization = {
        'authorization_kind': 'warehouse-v618-full-bc-preflight-only-v1',
        'candidate_id': candidate, 'pair_seed': 6150,
        'future_pair_joint_steps': 200_000, 'PPO_joint_steps': 0, 'publishable': False,
        'source_closure_sha256': checked['source_closure_sha256'],
        'run_spec_sha256': runner._scientific_run_spec_sha256(checked),
        'run_spec_file_sha256': sha256(spec_path.read_bytes()).hexdigest(),
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
        'lock_script_path': str(main),
        'lock_script_sha256': sha256(main.read_bytes()).hexdigest(),
    }
    _write_new(authorization_path, authorization)
    return {'spec': str(spec_path), 'authorization': str(authorization_path),
            'publishable': False, 'PPO_joint_steps': 0,
            'purpose': 'authorization_prepared_no_training_started'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--spec')
    parser.add_argument('--authorization')
    args = parser.parse_args()
    if args.prepare:
        if args.spec or args.authorization:
            parser.error('--prepare does not accept execution inputs')
        print(json.dumps(prepare(), sort_keys=True))
        return
    if not args.spec or not args.authorization:
        parser.error('provide --spec and --authorization, or use --prepare')
    spec_path = Path(args.spec).resolve(strict=True)
    authorization_path = Path(args.authorization).resolve(strict=True)
    authorization = json.loads(authorization_path.read_bytes())
    if (authorization.get('authorization_kind') != 'warehouse-v618-full-bc-preflight-only-v1'
            or authorization.get('run_spec_file_sha256') != sha256(spec_path.read_bytes()).hexdigest()):
        raise RuntimeError('Warehouse BC preflight spec file is not authorized')
    from backend.training.warehouse_v6_runner import run_locked_bc_preflight
    result = run_locked_bc_preflight(json.loads(spec_path.read_bytes()),
        authorization_receipt_path=authorization_path)
    receipt = result['receipt']
    print(json.dumps({'status': receipt['status'], 'receipt_path': result['receipt_path'],
                     'receipt_sha256': result['receipt_sha256'],
                     'program_precheck_passed': receipt['program_precheck_passed'],
                     'quality_screen_passed': receipt['quality_screen_passed'],
                     'publishable': False, 'PPO_joint_steps': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
