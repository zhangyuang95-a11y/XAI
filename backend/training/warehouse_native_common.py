"""Versioned provenance and atomic writes for the native warehouse experiment."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("warehouse_native_protocol.json")
VERSION = "warehouse-native-rcpd.v1"


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def canonical(value):
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_protocol():
    return json.loads(PROTOCOL.read_text())


def source_hashes():
    # The runtime/UI can be developed while a foundation job runs. Only code
    # involved in this training/validation contract belongs in a resume hash.
    paths = [ROOT / "env/warehouse_native" / name for name in (
        "__init__.py", "environment.py", "observations.py", "partners.py", "scenarios.py", "policy.py")]
    paths += [Path(__file__), Path(__file__).with_name("warehouse_native.py"),
              Path(__file__).with_name("warehouse_native_evaluation.py"), PROTOCOL]
    # Native physics reuses base _resolve_motion/_sample_delivery_job; their
    # transitive warehouse helpers must also be bound, even if not neural inputs.
    paths += list((ROOT / "env/warehouse").glob("*.py"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths) if p.exists()}


def preserve_inventory(output):
    """Hash original sources and artifacts without reading secrets into logs."""
    protected = {}
    for directory in (ROOT / "env/warehouse", ROOT / "output", ROOT / "ui/web"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            if relative.parts[:2] == ("output", "warehouse_native"):
                continue
            if path.suffix in {".sqlite3", ".db", ".sqlite", ".env"} or "private" in path.parts:
                continue  # Live logs can legitimately change while the user plays.
            if path.name.endswith(("-wal", "-shm")):
                continue
            protected[str(relative)] = file_hash(path)
    atomic_json(output, {"sha256": protected, "live_database_excluded": True,
                         "secret_contents_recorded": False})
    return protected
