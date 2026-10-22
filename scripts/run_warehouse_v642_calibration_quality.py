#!/usr/bin/env python3
"""Explicit prepare/audit/collect for the actual completed20k calibration pair."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_quality_collection as c


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--audit-seeds',action='store_true');group.add_argument('--collect',action='store_true')
    parser.add_argument('--completed-assets',type=Path);parser.add_argument('--final-programs',type=Path);parser.add_argument('--run-id')
    parser.add_argument('--base-start',type=int);parser.add_argument('--action-start',type=int);parser.add_argument('--plan',type=Path)
    args=parser.parse_args(argv)
    if args.prepare:
        if any(v is None for v in (args.completed_assets,args.final_programs,args.run_id,args.base_start,args.action_start)):
            parser.error('prepare requires completed-assets, final-programs, run-id, base-start and action-start')
        result=c.prepare(c.pointer(args.completed_assets),final_programs_ref=c.pointer(args.final_programs),
            run_id=args.run_id,base_start=args.base_start,action_start=args.action_start)
    else:
        if args.plan is None:parser.error('audit/collect requires the exact prepared --plan')
        result=(c.audit_seeds if args.audit_seeds else c.collect)(c.pointer(args.plan))
    print(json.dumps(result,allow_nan=False),flush=True);return result


if __name__=='__main__':main()
