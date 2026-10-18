#!/usr/bin/env python3
"""Development entry: prepare, compare spent scenes, or lock gamma; no admission authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # Intended repository scripts/ location.
sys.path.insert(0, str(ROOT))


def main():
    from backend.training import warehouse_v620_bc_pilot_eval as pilot
    from backend.training.warehouse_v620_bc_pilot_runtime import ProductionRuntime

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--gamma", choices=tuple(pilot.GAMMAS))
    mode.add_argument("--select", action="store_true")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    runtime = ProductionRuntime(ROOT, Path(__file__))
    if args.prepare:
        if not args.spec or args.plan:
            parser.error("prepare requires only --spec")
        result = {"plan": str(pilot.prepare(json.loads(args.spec.read_bytes()), runtime))}
    else:
        if not args.plan or args.spec:
            parser.error("gamma/select require only --plan")
        result = pilot.select(args.plan, runtime) if args.select else pilot.run_gamma(args.plan, args.gamma, runtime)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
