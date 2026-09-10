#!/usr/bin/env python3
"""Build and independently reload the compact warehouse online release."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.warehouse_native_common import canonical
from ui.warehouse_alignment_online_release import (
    build_online_release,
    load_online_release,
)
from ui.warehouse_alignment_release import load_alignment_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-release-root", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--output-base64", type=Path)
    args = parser.parse_args(argv)

    parent = load_alignment_release(
        args.parent_release_root,
        expected_manifest_sha256=args.expected_parent_manifest_sha256,
    )
    try:
        result = build_online_release(
            parent,
            output_package=args.output_package,
            output_base64=args.output_base64,
        )
    finally:
        parent.close()

    reloaded = load_online_release(
        package_path=args.output_package,
        expected_package_sha256=result["package_sha256"],
        expected_manifest_sha256=result["manifest_sha256"],
    )
    try:
        if (reloaded.runtime.actor_sha256
                != reloaded.provenance["actor_sha256"]
                or reloaded.question_bank.signature
                != reloaded.provenance["question_bank_signature"]
                or len(reloaded.question_bank.public_items()) != 8):
            raise ValueError("Independent online-release reload differs")
        result["independent_reload_passed"] = True
        result["runtime_signature"] = reloaded.runtime.signature
        result["program_sha256"] = reloaded.explainer.program_sha256
        result["question_bank_signature"] = reloaded.question_bank.signature
    finally:
        reloaded.close()
    print(canonical(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
