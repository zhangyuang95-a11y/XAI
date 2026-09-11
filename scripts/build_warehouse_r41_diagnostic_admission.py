#!/usr/bin/env python3
"""Build or reverify the fail-closed r4.1 diagnostic admission."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_admission as admission
from backend.training.warehouse_native_common import file_hash


def _components(args) -> dict[str, Path]:
    return {name: getattr(args, name) for name in admission.ARTIFACT_NAMES}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in admission.ARTIFACT_NAMES:
        value.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-existing", type=Path)
    value.add_argument("--expected-admission-sha256")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    components = _components(args)
    if args.verify_existing is not None:
        if args.expected_admission_sha256 is None:
            raise ValueError("--expected-admission-sha256 is required for verification")
        result = admission.read_saved_admission(
            args.verify_existing,
            expected_sha256=args.expected_admission_sha256,
            components=components,
        )
        output = args.verify_existing.expanduser().absolute()
        status = "diagnostic_admission_reverified"
        sha = args.expected_admission_sha256
    else:
        if args.expected_admission_sha256 is not None:
            raise ValueError("--expected-admission-sha256 applies only to verification")
        result = admission.build_admission(components, output=args.output)
        output = args.output.expanduser()
        if not output.is_absolute():
            output = ROOT / output
        output = output.absolute()
        status = "diagnostic_admission_created_and_reloaded"
        sha = file_hash(output)
    print(json.dumps({
        "version": admission.VERSION, "status": status,
        "admission": str(output), "admission_sha256": sha,
        "actor_sha256": result["bindings"]["actor_sha256"],
        "program_sha256": result["bindings"]["program_sha256"],
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
       allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
