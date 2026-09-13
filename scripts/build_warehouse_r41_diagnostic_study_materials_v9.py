#!/usr/bin/env python3
"""Freeze the v9 diagnostic question bank and neutral AI-AI tutorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_study_materials_v9 as producer
from backend.training.warehouse_native_common import file_hash


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in (
        "actor", "protocol", "source_manifest", "candidate_lock", "program",
        "final_audit",
    ):
        value.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        value.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    values = {
        name: getattr(args, name)
        for name in (
            "actor", "protocol", "source_manifest", "candidate_lock",
            "program", "final_audit",
        )
    }
    values.update({
        "expected_" + name + "_sha256": getattr(
            args, "expected_" + name + "_sha256")
        for name in (
            "actor", "protocol", "source_manifest", "candidate_lock",
            "program", "final_audit",
        )
    })
    receipt = producer.build(output=args.output, **values)
    receipt_path = args.output.expanduser().absolute() / producer.RECEIPT_FILE
    print(json.dumps({
        "version": producer.VERSION,
        "status": receipt["status"],
        "output": str(args.output.expanduser().absolute()),
        "receipt_sha256": file_hash(receipt_path),
        "question_bank_sha256": receipt["bindings"]["question_bank_sha256"],
        "tutorial_sha256": receipt["bindings"]["tutorial_sha256"],
        "formal_ready": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
