#!/usr/bin/env python3
"""Verify three completed BC candidates and create one non-executing model lock."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from experiments.warehouse.v621_bc_selection import write_model_lock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--registration", type=Path,
        default=ROOT/"output/warehouse_v621/selector-registration-01.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    lock = write_model_lock(root=ROOT, specification_path=args.spec,
        registration_path=args.registration, output_path=args.output)
    print(json.dumps({"model_lock": lock, "environment_resets": 0, "actor_updates": 0,
        "PPO_joint_steps": 0, "fresh_precheck_authorized": False, "pilot_opened": False,
        "publishable": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
