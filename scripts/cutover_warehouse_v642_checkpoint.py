#!/usr/bin/env python3
"""Prepare/assert by default. Only explicit --execute may stop the pinned v2 worker."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_checkpoint_cutover as c


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','assert','execute'):actions.add_argument('--'+name,action='store_true')
    for name in ('recovery-plan','supervisor-started','target-checkpoint','cutover-plan','successor-source-contract'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--worker-pid',type=int);parser.add_argument('--mode',choices=c.MODES)
    parser.add_argument('--cutover-id');parser.add_argument('--human-authorization')
    args=parser.parse_args(argv)
    if args.prepare:
        c.require(all(v is not None for v in (args.recovery_plan,args.supervisor_started,args.worker_pid,args.target_checkpoint,args.mode,args.cutover_id)),
            'prepare requires explicit pinned plan, launch, PID, target, mode and new ID')
        result=c.prepare(recovery_plan_ref=c.pointer(args.recovery_plan),supervisor_started_ref=c.pointer(args.supervisor_started),
            worker_pid=args.worker_pid,target_checkpoint_path=args.target_checkpoint,mode=args.mode,cutover_id=args.cutover_id)
    else:
        c.require(args.cutover_plan is not None,'assert/execute requires an explicit cutover plan')
        if getattr(args,'assert'):result=c.assert_safe(c.pointer(args.cutover_plan))
        else:
            c.require(all(v is not None for v in (args.successor_source_contract,args.worker_pid,args.target_checkpoint,args.supervisor_started,args.human_authorization)),
                'execute requires explicit ready source contract, PID, target, launch and authorization')
            result=c.execute(c.pointer(args.cutover_plan),authorization=args.human_authorization,
                successor_source_contract_ref=c.pointer(args.successor_source_contract),worker_pid=args.worker_pid,
                target_checkpoint_path=args.target_checkpoint,supervisor_started_ref=c.pointer(args.supervisor_started))
    print(json.dumps(result,sort_keys=True,allow_nan=False));return result


if __name__=='__main__':main()
