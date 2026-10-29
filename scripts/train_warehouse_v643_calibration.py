#!/usr/bin/env python3
"""Inspect v643 numerical integration; actual finite training remains disabled."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import v643_calibration as calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--status', action='store_true')
    action.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if args.run:
        return calibration.run()  # Refuses before file access, RNG or claim.
    value = calibration.integration_status()
    print(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return value


if __name__ == '__main__':
    main()
