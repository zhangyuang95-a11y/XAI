#!/usr/bin/env python3
"""Build the immutable fail-closed record for an exhausted r4.1 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_failure_closeout as closeout
from backend.training.warehouse_native_common import file_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--expected-training-ledger-sha256", required=True)
    parser.add_argument("--partner-alias-diagnostic", type=Path, required=True)
    parser.add_argument("--expected-partner-alias-diagnostic-sha256", required=True)
    parser.add_argument("--energy-gate-diagnostic", type=Path, required=True)
    parser.add_argument("--expected-energy-gate-diagnostic-sha256", required=True)
    parser.add_argument("--trend-diagnostic", type=Path, required=True)
    parser.add_argument("--expected-trend-diagnostic-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = closeout.build_closeout(
        run_root=args.run_root,
        ledger_path=args.training_ledger,
        expected_ledger_sha256=args.expected_training_ledger_sha256,
        diagnostic_paths={
            "partner_alias": args.partner_alias_diagnostic,
            "energy_gate": args.energy_gate_diagnostic,
            "trend": args.trend_diagnostic,
        },
        diagnostic_sha256={
            "partner_alias": args.expected_partner_alias_diagnostic_sha256,
            "energy_gate": args.expected_energy_gate_diagnostic_sha256,
            "trend": args.expected_trend_diagnostic_sha256,
        },
        output_root=args.output_root,
    )
    path = args.output_root.expanduser().resolve() / "failure_closeout.json"
    print(json.dumps({
        "version": report["version"], "status": report["status"],
        "admission_eligible": report["admission_eligible"],
        "release_eligible": report["release_eligible"],
        "deployment_allowed": report["deployment_allowed"],
        "formal_ready": report["formal_ready"],
        "boundary_count": report["training"]["boundary_count"],
        "output": str(path), "output_sha256": file_hash(path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
