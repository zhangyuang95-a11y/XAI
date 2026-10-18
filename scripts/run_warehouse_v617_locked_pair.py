#!/usr/bin/env python3
"""Run one Warehouse v6.17 pair using an immutable admission authorization.

The runner verifies the signed-by-content scientific configuration again
before constructing any environment. Ordinary train_pair remains closed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training.warehouse_v6_runner import run_locked_formal_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec).resolve(strict=True)
    authorization_path = Path(args.authorization).resolve(strict=True)
    authorization = json.loads(authorization_path.read_bytes())
    if (authorization.get("authorization_kind") !=
            "warehouse-v617-registered-pair-locked-run-v1"
            or authorization.get("run_spec_file_sha256") !=
            sha256(spec_path.read_bytes()).hexdigest()):
        raise RuntimeError("Warehouse locked pair run spec is not authorized")
    spec = json.loads(spec_path.read_bytes())
    result = run_locked_formal_pair(
        spec, authorization_receipt_path=authorization_path)
    pair = result["pair"]
    print(json.dumps({"candidate_id": pair["candidate_id"],
                      "pair_seed": pair["pair_seed"],
                      "joint_steps": pair["joint_steps"],
                      "run_spec_sha256": pair["run_spec_sha256"],
                      "reset_audit_sha256": result["reset_audit_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
