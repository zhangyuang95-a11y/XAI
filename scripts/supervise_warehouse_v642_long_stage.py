#!/usr/bin/env python3
"""Observe a fixed long final-program or quality worker's actual OS exit."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.warehouse.v642_long_stage_supervisor import supervise
from experiments.warehouse.v626_protocol import pointer


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=('long-final-programs','long-quality'))
    p.add_argument('--completed-assets',type=Path);p.add_argument('--reference-directory',type=Path)
    p.add_argument('--plan',type=Path);a=p.parse_args(argv)
    if a.stage=='long-final-programs':
        if a.completed_assets is None or a.reference_directory is None or a.plan is not None:
            p.error('long-final-programs requires completed-assets and reference-directory only')
        arguments=dict(completed_assets=pointer(a.completed_assets),reference_directory=str(a.reference_directory.resolve()))
    else:
        if a.plan is None or a.completed_assets is not None or a.reference_directory is not None:
            p.error('long-quality requires only plan')
        arguments=dict(plan=pointer(a.plan))
    result=supervise(a.stage,arguments);print(json.dumps(result,allow_nan=False),flush=True)
    return 0 if result['exit_code']==0 else 1


if __name__=='__main__':raise SystemExit(main())
