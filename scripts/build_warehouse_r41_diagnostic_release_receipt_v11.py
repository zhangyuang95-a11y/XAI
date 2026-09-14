#!/usr/bin/env python3
"""Create the append-only r4.1 diagnostic v11 deployment receipt."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.training import warehouse_r41_diagnostic_admission_v11 as admission
from backend.training import warehouse_r41_diagnostic_release_receipt_v11 as receipt
from backend.training.warehouse_native_common import file_hash

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    for name in admission.ARTIFACT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--outer-permanent-registry", type=Path, required=True)
    parser.add_argument("--final-permanent-registry", type=Path, required=True)
    parser.add_argument(
        "--permanent-promotion-closeout-registry", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    components = {name: getattr(args, name) for name in admission.ARTIFACT_NAMES}
    value = receipt.build_receipt(
        admission_path=args.admission,
        expected_admission_sha256=args.expected_admission_sha256,
        components=components,
        outer_permanent_registry=args.outer_permanent_registry,
        final_permanent_registry=args.final_permanent_registry,
        permanent_promotion_closeout_registry=args.permanent_promotion_closeout_registry,
        package_path=args.package, base64_path=args.base64, output=args.output)
    print(json.dumps({"version": receipt.VERSION, "status": value["status"],
                      "receipt": str(args.output),
                      "receipt_sha256": file_hash(args.output),
                      "package_sha256": value["package_sha256"],
                      "manifest_sha256": value["manifest_sha256"],
                      "formal_ready": False}, ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
