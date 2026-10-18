#!/usr/bin/env python3
"""Read-only Warehouse seed metadata evidence; never grants training authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inventory", "triage", "exact", "context", "delta"))
    args, remaining = parser.parse_known_args(argv)
    if args.operation == "inventory":
        from experiments.warehouse.seed_metadata_audit import main as run
        return run(remaining)
    if args.operation == "triage":
        from experiments.warehouse.seed_metadata_triage import main as run
        return run(remaining)
    if args.operation == "exact":
        from experiments.warehouse.exact_seed_audit import main as run
        return run(remaining)
    sub = argparse.ArgumentParser(prog=f"{parser.prog} {args.operation}")
    sub.add_argument("--inventory", type=Path, required=True)
    sub.add_argument("--output-dir", type=Path, required=True)
    if args.operation == "context":
        sub.add_argument("--source-root", type=Path, required=True)
        options = sub.parse_args(remaining)
        from experiments.warehouse.seed_context import annotations
        result = annotations(json.loads(options.inventory.read_bytes()), source_root=options.source_root,
                             output_directory=options.output_dir)
        print(json.dumps({"annotations": str(options.output_dir/"annotations.json"),
                          "classification_counts": result["classification_counts"], "authorization_granted": False}))
    else:
        sub.add_argument("--proposal", type=Path, required=True)
        options = sub.parse_args(remaining)
        from experiments.warehouse.exact_seed_audit import delta_inventory
        result = delta_inventory(json.loads(options.inventory.read_bytes()), options.proposal,
                                 output_directory=options.output_dir)
        print(json.dumps({"delta": str(options.output_dir/"delta.json"),
            "requires_new_content_audit": result["requires_new_content_audit"], "authorization_granted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
