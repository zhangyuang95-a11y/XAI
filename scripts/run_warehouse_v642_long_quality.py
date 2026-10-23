#!/usr/bin/env python3
"""Explicit shared-cohort preparation, exact clearance or one finite stream."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_long_training_quality as q


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);stages=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-cohort','audit-seeds','collect'):stages.add_argument('--'+name,action='store_true')
    parser.add_argument('--stream',nargs=2,action='append',metavar=('COMPLETED_ASSETS','FINAL_PROGRAMS'))
    parser.add_argument('--run-id');parser.add_argument('--base-start',type=int);parser.add_argument('--action-start',type=int)
    parser.add_argument('--plan',type=Path);args=parser.parse_args(argv)
    if args.prepare_cohort:
        if args.stream is None or len(args.stream)!=3 or any(v is None for v in (args.run_id,args.base_start,args.action_start)):
            parser.error('prepare-cohort requires three --stream ASSETS PROGRAMS plus run-id/base-start/action-start')
        result=q.prepare_cohort(streams=[dict(completed_assets=q.pointer(a),final_programs=q.pointer(p)) for a,p in args.stream],
            run_id=args.run_id,base_start=args.base_start,action_start=args.action_start)
    else:
        if args.plan is None:parser.error('audit-seeds/collect requires --plan')
        result=(q.audit_seeds if args.audit_seeds else q.collect)(q.pointer(args.plan))
    print(json.dumps(result,allow_nan=False),flush=True);return result


if __name__=='__main__':main()
