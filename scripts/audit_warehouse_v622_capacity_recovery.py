#!/usr/bin/env python3
"""Independent audit adapter for a complete replacement batch02 only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_recovery_protocol as recovery
from scripts import audit_warehouse_v622_capacity as base
from scripts import run_warehouse_v622_capacity_recovery as runner


@contextmanager
def recovery_scope():
    previous_v = base.v
    base.v = recovery
    try:
        yield base
    finally:
        base.v = previous_v


def audit(plan_path):
    recovery.validate_recipe(ROOT)
    # Reuse only the immutable binding loader, never a production evaluator.
    # It also prevents auditing a capacity01 plan under the replacement scope.
    with runner.recovery_scope():
        path, _ = runner.base.load_plan(plan_path)
    with recovery_scope():
        return base.audit(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.plan)
    print(json.dumps({k: result[k] for k in ("passed", "job_count", "bound_NN_sampled_action_checks")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
