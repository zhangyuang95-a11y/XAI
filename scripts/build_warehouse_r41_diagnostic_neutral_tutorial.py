#!/usr/bin/env python3
"""Build the replay-verified diagnostic AI-AI tutorial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.warehouse_native_common import canonical, file_hash
from ui.warehouse_alignment_r41_diagnostic_tutorial import build_neutral_tutorial


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    if file_hash(manifest_path) != args.expected_manifest_sha256:
        parser.error("diagnostic manifest bytes differ from expected SHA-256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = build_neutral_tutorial(
        manifest, manifest_file_sha256=args.expected_manifest_sha256,
    )
    target = args.output.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(canonical(payload) + "\n")
    print(canonical({
        "status": "passed", "output": str(target),
        "tutorial_sha256": file_hash(target), "frames": len(payload["frames"]),
        "duration_ms": payload["duration_ms"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
