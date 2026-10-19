#!/usr/bin/env python3
"""Explicit v6.22 fresh confirmation stages; absent authority fails closed."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main(argv=None):
    from experiments.warehouse import v622_confirmation as c
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract",action="store_true")
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--collect",action="store_true")
    mode.add_argument("--fit",action="store_true")
    mode.add_argument("--audit",action="store_true")
    parser.add_argument("--configuration",type=Path)
    parser.add_argument("--plan",type=Path)
    args=parser.parse_args(argv)
    if args.contract:
        if args.plan or args.configuration: parser.error("contract reads no plan/configuration")
        result={"schema":c.SCHEMA,"recipe":c.recipe(),"seed_allocation":False,"activation":"requires_complete_audited_recovery02_winner_and_own_history_clearance"}
    elif args.prepare:
        if not args.configuration or args.plan: parser.error("prepare requires configuration and no plan")
        result={"plan":str(c.prepare(ROOT,args.configuration))}
    else:
        if not args.plan or args.configuration: parser.error("collect/fit/audit requires immutable plan")
        value=(c.collect if args.collect else c.fit if args.fit else c.audit)(ROOT,args.plan)
        result={k:value[k] for k in ("schema","native_passed","confirmation_passed","actual_reset_count","PPO_joint_steps","pilot_opened") if k in value}
    print(json.dumps(result,sort_keys=True,allow_nan=False));return 0


if __name__=="__main__": raise SystemExit(main())
