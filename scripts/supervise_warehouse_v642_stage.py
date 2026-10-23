#!/usr/bin/env python3
"""Durable, source-bound supervision of final program fitting or quality collection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse.v642_stage_supervisor import supervise
from experiments.warehouse.v626_protocol import pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('final-programs', 'calibration-quality'), required=True)
    parser.add_argument('--completed-assets', type=Path)
    parser.add_argument('--reference-directory', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args(argv)
    if args.stage == 'final-programs':
        if args.completed_assets is None or args.reference_directory is None or args.run_id is None or args.plan is not None:
            parser.error('final-programs requires completed-assets, reference-directory, run-id only')
        arguments = dict(completed_assets=pointer(args.completed_assets),
                         reference_directory=str(args.reference_directory.resolve()), run_id=args.run_id)
    else:
        if args.plan is None or any(v is not None for v in (args.completed_assets, args.reference_directory, args.run_id)):
            parser.error('calibration-quality requires only plan')
        arguments = dict(plan=pointer(args.plan))
    receipt = supervise(args.stage, arguments)
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
