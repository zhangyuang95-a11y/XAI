#!/usr/bin/env python3
"""Prepare or execute a source-locked BC-only development pilot; no admission authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def prepare(spec_path: Path, authorization_path: Path):
    import torch
    from backend.training import warehouse_v6_runner as runner
    from backend.training.warehouse_v620_bc_capability import (
        authorization_fields, validate_spec,
    )
    spec = json.loads(spec_path.read_bytes())
    plan = validate_spec(spec, runner=runner, main_path=Path(__file__))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan.pair_seed)
        initial = runner.role_hashes(runner.WarehouseV6Actor())
    authorization = authorization_fields(plan, runner=runner, initial_roles=initial)
    authorization_path.parent.mkdir(parents=True, exist_ok=True)
    with authorization_path.open("x") as handle:
        json.dump(authorization, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"authorization": str(authorization_path), "PPO_joint_steps": 0,
            "publishable": False, "status": "prepared_no_training_started"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare:
        result = prepare(args.spec.resolve(strict=True), args.authorization.resolve())
    else:
        from backend.training import warehouse_v6_runner as runner
        from backend.training.warehouse_v620_bc_capability import execute
        result = execute(json.loads(args.spec.read_bytes()),
            authorization_receipt_path=args.authorization, runner=runner,
            main_path=Path(__file__))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
