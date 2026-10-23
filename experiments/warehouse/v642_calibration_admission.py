"""Reconstruct the completed calibration gate; never issue training authority."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import gzip
import json
import numpy as np
from . import v642_quality_collection as collection
from .v626_protocol import require, pointer, bound, file_sha
from .v642_fresh_program_metrics import evaluate as score_programs
from env.warehouse.public_action_features_v617 import ACTION_PHYSICS_CONFIG

VERSION = 'warehouse-v642-replayed-calibration-precheck.v1'


def _asset(ref, inputs):
    path = Path(ref['path'])
    require(pointer(path) == ref and not path.is_symlink()
            and not path.stat().st_mode & 0o222,
            'calibration input changed, missing, symlinked or writable')
    inputs.append(deepcopy(ref))
    return path


def audit(result_ref, supervisor_exit_ref):
    """Read one completed, actually exited collection and replay its evidence.

    Original recorded action seeds are numerically replayed, not allocated as
    new random streams. This grants neither development resets nor release.
    """
    inputs = []
    result_path = _asset(result_ref, inputs)
    execution = bound(supervisor_exit_ref)
    _asset(supervisor_exit_ref, inputs)
    subprocess_refs=[]
    if execution.get('observed_by') == 'source_bound_subprocess_wait':
        from .v642_stage_supervisor import verify_execution
        subprocess_refs=verify_execution(supervisor_exit_ref,result_ref,stage='calibration-quality')
        for ref in subprocess_refs:
            _asset(ref,inputs)
    require(type(execution.get('exit_code')) is int and execution['exit_code'] == 0
            and (execution.get('observed_by') == 'exec_tool_terminal_result' or bool(subprocess_refs))
            and execution.get('result') == result_ref,
            'calibration collection terminal exit was not actually observed')
    result = bound(result_ref)
    require(result['version'] == collection.VERSION
            and result['model_acceptance_passed'] is result['pilot_opened'] is False,
            'unknown calibration result or release scope')
    plan, folder = collection.checked(result['plan'])
    output = folder / 'collection'
    require(result_path.resolve() == output / 'result.json'
            and Path(supervisor_exit_ref['path']).resolve() == output / 'supervisor-exit.json'
            and not (output / 'failed.json').exists(), 'failed or relocated calibration collection')
    seal = pointer(output / 'completed.json')
    _asset(seal, inputs)
    require(bound(seal) == dict(result=result_ref), 'calibration completion seal changed')
    _asset(result['plan'], inputs)
    clearance_ref, clearance = collection.clearance(result['plan'], execution=False)
    claim_ref = pointer(collection.claim_path())
    _asset(clearance_ref, inputs)
    _asset(claim_ref, inputs)
    claim = bound(claim_ref)
    issued = datetime.fromisoformat(clearance['issued_utc'])
    expires = datetime.fromisoformat(clearance['expires_utc'])
    started = datetime.fromisoformat(claim['started_utc'])
    require(issued.tzinfo and expires.tzinfo and started.tzinfo
            and issued <= started < expires and claim['version'] == collection.VERSION
            and claim['plan'] == result['plan'] and claim['clearance'] == clearance_ref
            and claim['single_use_even_if_interrupted'] is True
            and claim['pilot_opened'] is False
            and result['claim'] == claim_ref and result['clearance'] == clearance_ref,
            'calibration claim or start-time seed clearance differs')
    resets_path = _asset(result['actual_resets'], inputs)
    auth_path = _asset(result['reset_preauthorization'], inputs)
    require(resets_path == output / 'actual-resets.jsonl'
            and auth_path == output / 'reset-preauthorization.jsonl', 'relocated calibration reset journals')
    resets = [json.loads(line) for line in resets_path.read_text().splitlines()]
    auth = [json.loads(line) for line in auth_path.read_text().splitlines()]
    expected = [(variant, case) for case in plan['cases'] for variant in collection.quality.VARIANTS]
    require(len(resets) == len(auth) == len(result['entries']) == len(expected) == 225
            and type(result['actual_reset_count']) is int and result['actual_reset_count'] == 225,
            'missing or additional calibration episodes or resets')
    models = {variant: collection.FrozenWarehouseV6Actor.load(ref['path'], expected_sha256=ref['sha256'],
              expected_role_sha256=plan['fixed']['actor_role_sha256'][variant])
              for variant, ref in plan['fixed']['actors'].items()}
    outcomes = []
    action_checks = 0
    previous_time = started
    for index, (variant, case) in enumerate(expected):
        entry = result['entries'][index]
        actor_ref = plan['fixed']['actors'][variant]
        authorized = datetime.fromisoformat(auth[index]['authorized_utc'])
        require(entry['variant'] == variant and entry['case'] == case
                and auth[index] == dict(case=case, variant=variant, plan=result['plan'],
                    actor=actor_ref, authorized_utc=auth[index]['authorized_utc'])
                and previous_time <= authorized < expires
                and resets[index]['seed'] == case['scenario_seed']
                and resets[index]['episode_id'] == 1 and resets[index]['phase'] == 'quality'
                and all(resets[index][key] == ACTION_PHYSICS_CONFIG[key]
                    for key in ('map_layout_id', 'move_battery_cost', 'charge_per_wait')),
                'calibration actor, reset physics/order or authorization time differs')
        previous_time = authorized
        stem = output / (variant + '-' + case['case_id'])
        trace_path = _asset(entry['trace'], inputs)
        array_path = _asset(entry['arrays'], inputs)
        require(trace_path == stem.with_suffix('.jsonl.gz') and array_path == stem.with_suffix('.npz'),
                'calibration trajectory assets are outside their declared scene')
        with gzip.open(trace_path, 'rt') as handle:
            events = [json.loads(line) for line in handle]
        with np.load(array_path, allow_pickle=False) as archive:
            require(set(archive.files) == {'x', 'm', 'p', 'sampled'}, 'unknown calibration arrays')
            arrays = {key: archive[key].copy() for key in archive.files}
        require(all(len(value) == len(events) for value in arrays.values())
                and all(event['action_override_agents'] == [] for event in events),
                'calibration arrays, events or NN overrides differ')
        checks = sum(collection.verify_frame(models[variant], case, event, arrays, i)
                     for i, event in enumerate(events))
        outcome = collection.replay_public_episode(case, events, len(events), actor_ref['sha256'])
        outcome.update(variant=variant, actor=actor_ref, complete=True, NN_action_checks=checks, action_overrides=0)
        outcomes.append(outcome)
        action_checks += checks
    require(outcomes == result['outcomes'], 'calibration public counters differ from producer outcomes')
    metrics = collection.quality.evaluate(outcomes, plan['cases'], plan['fixed']['actors'])
    programs = score_programs(plan['final_programs'], plan['fixed']['actor_role_sha256'], result['entries'], plan['cases'])
    require(metrics == result['metrics'] and programs == result['program_metrics']
            and result['task_retention_passed'] is metrics['task_retention_passed']
            and result['calibration_precheck_passed'] is
                (metrics['task_retention_passed'] and programs['passed']),
            'calibration task/program metrics or combined verdict do not reproduce')
    require(metrics['task_retention_passed'] and programs['passed'],
            'calibration task retention or one of four program gates failed')
    require(collection.checked(result['plan'])[0] == plan, 'calibration prerequisites changed during replay')
    for ref in inputs:
        _asset(ref, [])
    return dict(version=VERSION, quality_result=result_ref, supervisor_exit=supervisor_exit_ref,
        plan=result['plan'], input_asset_refs=inputs, calibration_precheck_passed=True,
        metrics=metrics, program_metrics=programs, independent_scenes=75, complete_episodes=225,
        replayed_role_actions=action_checks, new_environment_resets=0, NN_updates=0,
        program_fit_calls=0, qualification_changes=0, training_authority=False,
        sealed_authority=False, model_acceptance_passed=False, pilot_opened=False,
        source_sha256=file_sha(__file__))
