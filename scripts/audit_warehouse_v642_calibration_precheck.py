#!/usr/bin/env python3
"""Replay completed calibration before considering any development training."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse.v642_calibration_admission import audit
from experiments.warehouse.v626_protocol import pointer
from scripts.run_warehouse_v625 import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality-result', type=Path, required=True)
    parser.add_argument('--supervisor-exit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(pointer(args.quality_result), pointer(args.supervisor_exit))
    write_new(args.output, result)
    args.output.chmod(0o444)
    print(json.dumps(pointer(args.output)), flush=True)


if __name__ == '__main__':
    main()
