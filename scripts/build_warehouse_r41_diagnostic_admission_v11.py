#!/usr/bin/env python3
"""Build the append-only r4.1 diagnostic v11 admission."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.training import warehouse_r41_diagnostic_admission_v11 as admission
from backend.training.warehouse_native_common import file_hash

def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in admission.ARTIFACT_NAMES:
        value.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    value.add_argument("--outer-permanent-registry", type=Path, required=True)
    value.add_argument("--final-permanent-registry", type=Path, required=True)
    value.add_argument(
        "--permanent-promotion-closeout-registry", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value

def main(argv=None) -> int:
    args = parser().parse_args(argv)
    components = {name: getattr(args, name) for name in admission.ARTIFACT_NAMES}
    result = admission.build_admission(
        components, outer_permanent_registry=args.outer_permanent_registry,
        final_permanent_registry=args.final_permanent_registry,
        permanent_promotion_closeout_registry=args.permanent_promotion_closeout_registry,
        output=args.output)
    print(json.dumps({"version": admission.VERSION, "status": result["status"],
                      "output": str(args.output),
                      "sha256": file_hash(args.output),
                      "formal_ready": False}, ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
