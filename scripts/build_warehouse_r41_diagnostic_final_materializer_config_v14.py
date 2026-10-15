#!/usr/bin/env python3
"""Build the public-path v14 final materializer configuration.

The builder canonicalizes path strings and creates one new mode-0600 JSON
file. It never opens, stats, or resolves any configured input, including the
private salt. The materializer authenticates those inputs only after the
controller has created its permanent final-attempt claim.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_final_materializer_v14 as materializer
from backend.training.warehouse_native_common import canonical, digest


PATH_FIELDS = tuple(sorted(materializer._CONFIG_PATH_FIELDS))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in PATH_FIELDS:
        value.add_argument("--" + name.replace("_", "-"), dest=name,
                           type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def create_config(paths: Mapping[str, str | Path]) -> dict:
    if set(paths) != set(PATH_FIELDS):
        raise ValueError("Exactly the 38 frozen v14 materializer paths are required")
    normalized = {
        name: str(Path(paths[name]).expanduser().absolute())
        for name in PATH_FIELDS
    }
    if Path(normalized["private_salt"]) != materializer.PRIVATE_SALT_PATH:
        raise ValueError("Only the frozen v14 private-salt path is permitted")
    value = {"version": materializer.CONFIG_VERSION, "paths": normalized}
    value["content_sha256"] = digest(value)
    return value


def write_exclusive(output: str | Path, value: Mapping) -> Path:
    destination = Path(output).expanduser().absolute()
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    paths = {name: getattr(args, name) for name in PATH_FIELDS}
    value = create_config(paths)
    destination = write_exclusive(args.output, value)
    print(canonical({
        "status": "created",
        "version": value["version"],
        "path_count": len(value["paths"]),
        "content_sha256": value["content_sha256"],
        "output": str(destination),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PATH_FIELDS", "parser", "create_config", "write_exclusive", "main"]
