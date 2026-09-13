#!/usr/bin/env python3
"""Run the irrevocable warehouse r4.1 v9 final audit once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_final_once_v9 as final_once


MATERIALIZER_CONFIG_ENV = "WAREHOUSE_R41_V9_FINAL_MATERIALIZER_CONFIG"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate-lock", type=Path, required=True)
    value.add_argument("--expected-candidate-lock-sha256", required=True)
    value.add_argument("--actor", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--runtime-manifest", type=Path, required=True)
    value.add_argument("--designation", type=Path, required=True)
    value.add_argument("--failed-outer-closeout", type=Path, required=True)
    value.add_argument("--fresh-outer-registry", type=Path, required=True)
    value.add_argument("--fresh-outer-registry-report", type=Path, required=True)
    value.add_argument("--outer-hash-projection", type=Path, required=True)
    value.add_argument("--outer-hash-projection-receipt", type=Path, required=True)
    value.add_argument("--development-rows", type=Path, required=True)
    value.add_argument("--program", type=Path, required=True)
    value.add_argument("--selector-report", type=Path, required=True)
    value.add_argument("--outer-result", type=Path, required=True)
    value.add_argument("--expected-outer-result-sha256", required=True)
    value.add_argument("--outer-permanent-registry", type=Path, required=True)
    value.add_argument("--permanent-final-registry", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--final-materializer-config",
        type=Path,
        required=True,
        help=("Path passed to the post-claim materializer through "
              + MATERIALIZER_CONFIG_ENV + "; the wrapper does not open it."),
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    previous_config = os.environ.get(MATERIALIZER_CONFIG_ENV)
    os.environ[MATERIALIZER_CONFIG_ENV] = str(
        args.final_materializer_config.expanduser().absolute())
    try:
        result = final_once.run_final_once(
            candidate_lock_path=args.candidate_lock,
            expected_candidate_lock_sha256=args.expected_candidate_lock_sha256,
            actor_path=args.actor,
            protocol_path=args.protocol,
            runtime_manifest_path=args.runtime_manifest,
            designation_path=args.designation,
            failed_outer_closeout_path=args.failed_outer_closeout,
            fresh_outer_registry_path=args.fresh_outer_registry,
            fresh_outer_registry_report_path=args.fresh_outer_registry_report,
            outer_hash_projection_path=args.outer_hash_projection,
            outer_hash_projection_receipt_path=(
                args.outer_hash_projection_receipt),
            development_rows_path=args.development_rows,
            program_path=args.program,
            selector_report_path=args.selector_report,
            outer_result_path=args.outer_result,
            expected_outer_result_sha256=args.expected_outer_result_sha256,
            outer_permanent_registry=args.outer_permanent_registry,
            permanent_final_registry=args.permanent_final_registry,
            output=args.output,
        )
    finally:
        if previous_config is None:
            os.environ.pop(MATERIALIZER_CONFIG_ENV, None)
        else:
            os.environ[MATERIALIZER_CONFIG_ENV] = previous_config
    print(json.dumps({
        "status": result["status"],
        "attempt_key": result["attempt_key"],
        "output": str(args.output.expanduser().absolute()),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == final_once.STATUS_PASSED else 1


if __name__ == "__main__":
    raise SystemExit(main())
