#!/usr/bin/env python3
"""Build the v9 transport tree from the passed v13 evidence rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import warehouse_r41_diagnostic_compact_public_tree_v9 as compact
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_once as rows_api
from backend.training.warehouse_native_common import canonical, file_hash


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--program", type=Path, required=True)
    value.add_argument("--expected-program-sha256", required=True)
    for name in ("development_rows", "combined_promoted_rows",
                 "outer_rows", "final_rows"):
        value.add_argument("--" + name.replace("_", "-"),
                           type=Path, required=True)
        value.add_argument("--expected-" + name.replace("_", "-")
                           + "-sha256", required=True)
    value.add_argument("--output-program", type=Path, required=True)
    value.add_argument("--output-report", type=Path, required=True)
    return value


def _observations(path: Path, expected_sha256: str, label: str) -> np.ndarray:
    return rows_api._safe_row_projection(
        path.expanduser().absolute(), expected_sha256=expected_sha256,
        fields=frozenset(("observations",)), label=label,
    )["observations"]


def _write_new(path: Path, raw: bytes) -> Path:
    target = path.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists() or target.is_symlink() or target.parent.is_symlink():
        raise FileExistsError(target)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    program = args.program.expanduser().absolute()
    if (not program.is_file() or program.is_symlink()
            or program.resolve() != program
            or file_hash(program) != args.expected_program_sha256):
        raise ValueError("Exact canonical v13-selected program bytes required")
    audited = {
        "development": np.concatenate((
            _observations(args.development_rows,
                          args.expected_development_rows_sha256,
                          "base development rows"),
            _observations(args.combined_promoted_rows,
                          args.expected_combined_promoted_rows_sha256,
                          "combined promoted development rows"),
        ), axis=0),
        "fresh_outer": _observations(
            args.outer_rows, args.expected_outer_rows_sha256,
            "passed v13 fresh outer rows"),
        "protected_final": _observations(
            args.final_rows, args.expected_final_rows_sha256,
            "passed v13 protected final rows"),
    }
    encoded, report = compact.encode_program(program.read_bytes(), audited)
    output_program = _write_new(args.output_program, encoded)
    output_report = None
    try:
        output_report = _write_new(
            args.output_report, (canonical(report) + "\n").encode("utf-8"))
    except BaseException:
        output_program.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "version": compact.VERSION,
        "status": report["status"],
        "program": str(output_program),
        "program_sha256": file_hash(output_program),
        "report": str(output_report),
        "report_sha256": file_hash(output_report),
        "audit_sets": report["audit"]["sets"],
        "formal_ready": False,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
