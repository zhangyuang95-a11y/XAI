#!/usr/bin/env python3
"""Create the immutable r4.1 diagnostic deployment receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_release_receipt as receipt_api
from backend.training.warehouse_native_common import file_hash


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    parser.add_argument("--designation", type=Path, required=True)
    parser.add_argument("--expected-designation-sha256", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base64", type=Path, required=True)
    parser.add_argument("--rollback-receipt", type=Path, required=True)
    parser.add_argument("--expected-rollback-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = receipt_api.build_receipt(
        admission_path=args.admission,
        expected_admission_sha256=args.expected_admission_sha256,
        designation_path=args.designation,
        expected_designation_sha256=args.expected_designation_sha256,
        package_path=args.package, base64_path=args.base64,
        rollback_receipt_path=args.rollback_receipt,
        expected_rollback_receipt_sha256=args.expected_rollback_receipt_sha256,
        output=args.output,
    )
    path = args.output.expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    print(json.dumps({
        "version": receipt_api.VERSION,
        "status": "diagnostic_release_receipt_created_and_reloaded",
        "receipt": str(path), "receipt_sha256": file_hash(path),
        "package_sha256": value["package_sha256"],
        "manifest_sha256": value["manifest_sha256"],
        "actor_sha256": value["actor_sha256"],
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
       allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
