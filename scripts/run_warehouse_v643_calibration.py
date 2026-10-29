#!/usr/bin/env python3
"""Explicit source-bound v64320k pair, or clean same-claim continuation.

No defaults select a plan, seed, prior run or recovery checkpoint. A successful
calibration precheck is not a development-budget or release/pilot admission.
"""
from pathlib import Path
import argparse
import json
import os
import re
import sys


ROOT=Path(__file__).resolve().parents[1]
THREAD_VARIABLES=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def _pipeline():
    sys.path.insert(0,str(ROOT))
    from experiments.warehouse import v643_calibration_pipeline
    return v643_calibration_pipeline


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-directory',required=True,type=Path)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-sha')
    args=parser.parse_args(argv)
    if not re.fullmatch(r'[0-9a-f]{64}',args.plan_sha):parser.error('--plan-sha must be the exact lowercase SHA256')
    if (args.resume is None)!=(args.resume_sha is None):parser.error('--resume and --resume-sha must be provided together')
    if args.resume_sha is not None and not re.fullmatch(r'[0-9a-f]{64}',args.resume_sha):
        parser.error('--resume-sha must be the exact lowercase SHA256')
    # Must precede the FIRST NumPy/Torch/pipeline import in this process.
    for name in THREAD_VARIABLES:os.environ[name]='1'
    try:
        pipeline=_pipeline()
        result=pipeline.run(reference_directory=args.reference_directory.resolve(),
            plan_ref=dict(path=str(args.plan.resolve()),sha256=args.plan_sha),
            resume_ref=None if args.resume is None else dict(path=str(args.resume.resolve()),sha256=args.resume_sha))
        report=pipeline.bound(result)
        passed=report.get('calibration_precheck_passed') is True
        # This exit is about this actual calibration only. Even its successful
        # result cannot authorize a larger budget, human enrollment or release.
        if report.get('model_acceptance_passed') is not False or report.get('pilot_opened') is not False:
            raise RuntimeError('calibration result has unsupported model/pilot authority')
        print(json.dumps(dict(result=result,calibration_precheck_passed=passed,
            candidate_closed_after_failed_calibration=report.get('candidate_closed_after_failed_calibration'),
            numeric_thread_environment={name:os.environ[name] for name in THREAD_VARIABLES},
            model_acceptance_passed=False,pilot_opened=False),sort_keys=True))
        return 0 if passed else 1
    except KeyboardInterrupt:
        print('Interrupted; the consumed claim is retained. Only an explicit verified clean checkpoint may continue.',file=sys.stderr)
        return 130
    except Exception as error:
        print('v643 calibration refused or failed: '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
