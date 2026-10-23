#!/usr/bin/env python3
"""Explicit commands for the new method; old domain entrypoints remain legacy."""
import argparse
import json
import os
from pathlib import Path
import sys
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    if os.environ.get(_name, '1') != '1':
        raise RuntimeError('v7 requires single-thread numerical libraries: ' + _name)
    os.environ[_name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_v7 import workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register', 'amend-preflight', 'seed-audit', 'resume-seed-audit', 'finalize-seeds', 'bc', 'calibration', 'status'))
    parser.add_argument('--output', type=Path, default=workflow.OUTPUT)
    parser.add_argument('--domain', choices=('warehouse', 'pong', 'kitchen'))
    parser.add_argument('--previous-complete', type=Path,
        help='SHA-checked complete historical JSON validity evidence; never old candidate-match conclusions')
    args = parser.parse_args()
    if args.command == 'register':
        value = workflow.register(args.output)
        print(json.dumps({k: value[k] for k in ('candidate_id', 'release_id', 'method_sha256', 'pilot_opened')}))
    elif args.command in ('amend-preflight', 'finalize-seeds'):
        value = workflow.amend_preflight(args.output) if args.command == 'amend-preflight' else workflow.finalize_seeds(args.output)
        print(json.dumps({k: value[k] for k in ('passed', 'seed_guard_version', 'preflight_revision') if k in value}))
        return 1 if value.get('passed') is False else 0
    elif args.command == 'seed-audit':
        value = workflow.seed_audit(args.output, previous_complete=args.previous_complete)
        print(json.dumps(dict(passed=value['passed'], unresolved=len(value.get('unresolved_paths', [])), environment_resets=0)))
        return 0 if value['passed'] else 1
    elif args.command == 'resume-seed-audit':
        if args.previous_complete is None: parser.error('--previous-complete is required')
        from training_v7.seed_resume import resume
        value=resume(args.output,args.previous_complete)
        print(json.dumps(dict(unresolved=len(value['summary']['unresolved_paths']),authorization_granted=False,
            parsing=value['resume_adapter'])))
    elif args.command in ('bc', 'calibration'):
        if args.domain is None: parser.error('--domain is required')
        operation = workflow.run_bc if args.command == 'bc' else workflow.run_calibration
        print(json.dumps(operation(args.domain, args.output)))
    else:
        _, registration = workflow.registration_at(args.output)
        summary = dict(candidate=registration['candidate_id'], method_sha256=registration['method_sha256'],
                       domains={}, candidate_closed=(args.output/'CLOSED.json').exists(),
                       model_acceptance_passed=False, pilot_opened=False)
        if summary['candidate_closed']:
            summary['closure_reason']=json.loads((args.output/'CLOSED.json').read_text())['reason']
        for name in ('warehouse', 'pong', 'kitchen'):
            result = args.output/name/'calibration-0/bc/result.json'
            journal = args.output/name/'calibration-0/events.jsonl'
            events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            epochs = [e for e in events if e['event'] == 'bc_epoch_completed']
            episodes = [e for e in events if e['event'] == 'bc_episode_completed']
            summary['domains'][name] = {'bc_completed': result.is_file(), 'bc_complete_episodes': len(episodes),
                'bc_latest_epoch': epochs[-1] if epochs else None, 'new_model_acceptance_passed': False,
                'raw': {}, 'regularized': {}}
            for variant in ('raw', 'regularized'):
                ppo_journal = journal.parent/variant/'events.jsonl'
                ppo_events = [json.loads(line) for line in ppo_journal.read_text().splitlines()] if ppo_journal.exists() else []
                complete = [e for e in ppo_events if e['event'] == 'ppo_rollout_completed']
                summary['domains'][name][variant] = ({k:complete[-1][k] for k in
                    ('joint_steps','actor_updates','critic_updates','program_updates','program_gradient_norm_sum')}
                    if complete else {'joint_steps': 0, 'post_20k_quality': None})
            diagnosis=args.output/'closed-diagnosis-01'/name/'result.json'
            if diagnosis.is_file():
                from training_v7.contracts import file_sha
                actual=json.loads(diagnosis.read_text())
                if (actual['closed_receipt']['sha256']!=file_sha(args.output/'CLOSED.json')
                        or actual['reopened'] is not False or actual['model_acceptance_passed'] is not False):
                    raise ValueError('closed diagnosis identity differs')
                summary['domains'][name]['read_only_actual_20k_diagnosis']=dict(
                    path=str(diagnosis.resolve()),sha256=file_sha(diagnosis),
                    task_metrics=actual['task_metrics'],program_metrics=actual['program_metrics'],
                    raw_regularized_role_weights_distinct=actual['raw_regularized_role_weights_distinct'],
                    scientific_calibration_requirements_met=actual['scientific_calibration_requirements_met'])
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
