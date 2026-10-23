#!/usr/bin/env python3
"""Independently audit one actually completed, budget-bound long pair.

Formal mode first reads the actual three-stream development budget lock.
No flag, source file, export or process receipt alone grants model acceptance.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.warehouse import v642_long_training_assets as assets
from experiments.warehouse.v626_protocol import pointer,require


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('development','formal'),default='development')
    parser.add_argument('--formal-budget-lock',type=Path)
    parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--calibration-precheck',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--supervisor-exit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    lock=None
    if args.phase=='formal':
        require(args.formal_budget_lock is not None,'formal audit needs an explicit real budget lock')
        lock=pointer(args.formal_budget_lock)
        assets._phase_admission('formal',lock)
    else:
        require(args.formal_budget_lock is None,'development audit cannot supply a formal budget lock')
    # Other caller files are read only after the formal lock has replayed.
    result=assets.run(reference_directory=args.reference_directory,
        calibration_precheck_ref=pointer(args.calibration_precheck),result_ref=pointer(args.result),
        supervisor_exit_ref=pointer(args.supervisor_exit),output=args.output,
        phase=args.phase,formal_budget_lock_ref=lock)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


if __name__=='__main__':main()
