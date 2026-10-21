#!/usr/bin/env python3
"""Conditional 20k paired calibration; no automatic retry, resume or release."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse import v629_calibration_pipeline as c


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','audit-seeds','run'):stages.add_argument('--'+name,action='store_true')
    parser.add_argument('--reference-directory',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path)
    args=parser.parse_args(argv)
    # The real A/B gate must precede even reading a caller-supplied seed plan.
    c.owner.seeds.partitions.admission.load_readiness(args.reference_directory).verify()
    plan_ref=c.pointer(args.plan)
    key='prepare' if args.prepare else 'audit_seeds' if args.audit_seeds else 'run'
    result=getattr(c,key)(reference_directory=args.reference_directory,plan_ref=plan_ref)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True);return result


if __name__=='__main__':main()
