#!/usr/bin/env python3
"""Explicit batch02 adapter; --write-recipe writes recovery metadata only."""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.warehouse import v622_recovery_protocol as recovery
from scripts import run_warehouse_v622_capacity as base


@contextmanager
def recovery_scope():
    """An explicit process-local adapter, restored even on failure."""
    previous_v, previous_output = base.v, base.output_directory
    base.v, base.output_directory = recovery, recovery.output_directory
    try:
        yield base
    finally:
        base.v, base.output_directory = previous_v, previous_output


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--write-recipe" in argv:
        if argv != ["--write-recipe"]:
            raise ValueError("--write-recipe is exclusive and cannot prepare or execute jobs")
        path = recovery.write_recipe(ROOT)
        print(json.dumps({"recipe": str(path), "recipe_sha256": sha256(path.read_bytes()).hexdigest(),
            "source_closure_sha256": recovery.source_closure_sha256(ROOT),
            "environment_resets": 0, "actor_updates": 0, "scientific_jobs_started": 0}))
        return 0
    recovery.validate_recipe(ROOT)
    with recovery_scope():
        return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
