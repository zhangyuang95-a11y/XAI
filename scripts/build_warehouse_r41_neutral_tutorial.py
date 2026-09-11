#!/usr/bin/env python3
"""Produce and replay-verify the fixed neutral r4.1 AI-AI tutorial."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.warehouse_alignment_r41_tutorial import build_neutral_tutorial
from ui.warehouse_alignment_online_release import canonical, digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflict-manifest", type=Path, required=True)
    parser.add_argument("--expected-conflict-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.conflict_manifest.expanduser().resolve()
    output = args.output.expanduser().absolute()
    if (not source.is_file() or source.is_symlink()
            or sha256(source.read_bytes()).hexdigest()
                != args.expected_conflict_manifest_sha256):
        parser.error("conflict manifest bytes differ from the expected SHA-256")
    if (output.resolve() != output or output.exists() or output.parent.is_symlink()
            or not output.parent.is_dir()):
        parser.error("output must be a new canonical file")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    tutorial = build_neutral_tutorial(manifest)
    raw = (canonical(tutorial) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    print(json.dumps({
        "version": tutorial["version"], "status": "built_and_replay_verified",
        "output": str(output), "file_sha256": sha256(raw).hexdigest(),
        "tutorial_signature": digest(tutorial),
        "scene_id": tutorial["scene_id"],
        "frame_count": len(tutorial["frames"]),
        "uses_final_actor": False, "coverage": tutorial["coverage"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
