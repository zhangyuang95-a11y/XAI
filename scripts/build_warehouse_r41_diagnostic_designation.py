#!/usr/bin/env python3
"""Build the immutable designation for the terminal r4.1 diagnostic Actor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_designation as designation
from backend.training.warehouse_native_common import file_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--dual-evaluation", type=Path, required=True)
    parser.add_argument("--failure-closeout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    components = {
        "actor": args.actor,
        "protocol": args.protocol,
        "training_ledger": args.training_ledger,
        "dual_evaluation": args.dual_evaluation,
        "failure_closeout": args.failure_closeout,
    }
    result = designation.build_designation(components, output=args.output)
    print(json.dumps({
        "version": result["version"], "status": result["status"],
        "actor_sha256": result["bindings"]["actor_sha256"],
        "designation_sha256": file_hash(args.output.expanduser().resolve()),
        "behavior_performance_gate_waived": True,
        "formal_sample_eligible": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
