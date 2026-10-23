#!/usr/bin/env python3
"""Recount and export an explicitly recovered 20k pair after its actual exit0."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import v642_recovered_assets as audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-directory', required=True, type=Path)
    parser.add_argument('--recovery-completed', required=True, type=Path)
    parser.add_argument('--supervisor-exit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = audit.run(reference_directory=args.reference_directory,
        result_ref=audit.pointer(args.recovery_completed),
        supervisor_exit_ref=audit.pointer(args.supervisor_exit), output=args.output)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
