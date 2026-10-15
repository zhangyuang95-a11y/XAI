#!/usr/bin/env python3
"""Stage a fresh v6.n release after the preceding candidate was closed.

Pong and Kitchen keep their byte-identical v5 assets and receipts. A new
Warehouse candidate begins pending with distinct policy IDs and a separate
content-addressed package. This command never activates a pilot.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_v3.release_catalog import (RELEASE_ID, UPSTREAM_RELEASE_ID,
                                      warehouse_release_spec)


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rewrite(value, previous: str, current: str):
    old = f"releases/{previous}/assets/"
    new = f"releases/{current}/assets/"
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old) else value
    if isinstance(value, list):
        return [_rewrite(item, previous, current) for item in value]
    if isinstance(value, dict):
        return {_rewrite(key, previous, current): _rewrite(item, previous, current)
                for key, item in value.items()}
    return value


def stage(release_id: str, closed_receipt: str | Path, *,
          repository_root: str | Path = ROOT) -> dict:
    root = Path(repository_root).resolve()
    spec = warehouse_release_spec(release_id)
    minor = int(release_id.rsplit(".", 1)[1])
    if minor < 1:
        raise ValueError("a successor release must increment the minor version")
    predecessor = f"policylens-three-domain-20261016.v6.{minor - 1}"
    closed_path = Path(closed_receipt).resolve()
    if closed_path.name != "candidate.closed.json":
        raise ValueError("candidate closure receipt required")
    closed_bytes = closed_path.read_bytes()
    closed = json.loads(closed_bytes)
    if (not isinstance(closed, dict)
            or closed.get("release_id") != predecessor
            or closed.get("passed") is not False
            or not isinstance(closed.get("candidate_id"), str)
            or not closed["candidate_id"]
            or not isinstance(closed.get("phase"), str)
            or not closed["phase"]):
        raise ValueError("predecessor candidate was not closed")
    source_release = RELEASE_ID
    source_catalog = root / "releases" / source_release / "catalog.json"
    original = json.loads(source_catalog.read_bytes())
    if (original.get("release_id") != source_release
            or original.get("upstream_release_id") != UPSTREAM_RELEASE_ID
            or original.get("warehouse", {}).get("status") == "accepted"):
        raise ValueError("v6.0 cannot be used as a pending release template")
    source_index = original.get("source_asset_index")
    if not isinstance(source_index, dict) or not source_index:
        raise ValueError("upstream source asset index missing")
    target = root / "releases" / release_id
    if target.exists():
        raise FileExistsError(f"candidate release already exists: {target}")
    staging = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        for item in source_index.values():
            path, expected = item["bundle_path"], item["sha256"]
            prefix = f"releases/{source_release}/assets/"
            if (not isinstance(path, str) or not path.startswith(prefix)
                    or not isinstance(expected, str) or len(expected) != 64
                    or Path(path).parent.name != expected):
                raise ValueError("source asset is not content addressed")
            source = (root / path).resolve()
            if (not source.is_relative_to(root / "releases" / source_release / "assets")
                    or _sha(source) != expected):
                raise ValueError("upstream source asset changed")
            destination = staging / "assets" / expected / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(source, destination)
            if _sha(destination) != expected:
                raise ValueError("copied source asset changed")
        closed_sha = sha256(closed_bytes).hexdigest()
        closure = staging / "assets" / closed_sha / closed_path.name
        closure.parent.mkdir(parents=True, exist_ok=True)
        closure.write_bytes(closed_bytes)
        catalog = _rewrite(deepcopy(original), source_release, release_id)
        catalog["release_id"] = release_id
        catalog["warehouse"] = {"status": "pending",
                                "policy_ids": spec.policy_ids,
                                "predecessor_closed_candidate": {
                                    "release_id": predecessor,
                                    "candidate_id": closed["candidate_id"],
                                    "phase": closed["phase"],
                                    "path": (f"releases/{release_id}/assets/"
                                             f"{closed_sha}/{closed_path.name}"),
                                    "sha256": closed_sha}}
        catalog["policies"].pop("warehouse", None)
        (staging / "catalog.json").write_text(
            json.dumps(catalog, sort_keys=True, indent=2) + "\n")
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"release_id": release_id, "warehouse_status": "pending",
            "catalog_sha256": _sha(target / "catalog.json"),
            "predecessor_closed_sha256": closed_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--closed-receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.release_id, args.closed_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
