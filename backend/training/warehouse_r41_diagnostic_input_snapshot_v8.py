"""Immutable input snapshots for the r4.1 diagnostic evidence chain.

Every source is opened with ``O_NOFOLLOW`` exactly once, copied through that
descriptor into a private directory, and hashed while it is copied.  Semantic
readers consume only the private copies.  Callers retain the original paths
solely so they can fail closed if any pathname no longer contains the same
bytes before publication or return.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

from backend.training.warehouse_native_common import file_hash


DEFAULT_MAXIMUM_BYTES = 512 * 1024 * 1024
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")


def canonical_regular_file(
    value: str | Path, label: str, *, maximum: int = DEFAULT_MAXIMUM_BYTES,
) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def read_authenticated_bytes(
    value: str | Path, *, label: str, expected_sha256: str,
    maximum: int = DEFAULT_MAXIMUM_BYTES,
) -> bytes:
    """Return bytes hashed from the same no-follow descriptor before use."""
    path = canonical_regular_file(value, label, maximum=maximum)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError(label + " must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        hasher = sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ValueError(label + " exceeds its size limit")
            hasher.update(chunk)
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    if hasher.hexdigest() != expected_sha256:
        raise ValueError("Exact " + label + " bytes required")
    return b"".join(chunks)


@dataclass(frozen=True)
class SnapshotEntry:
    original: Path
    frozen: Path
    sha256: str


class ImmutableInputSnapshot(AbstractContextManager):
    """Private file set whose bytes are stable for one evidence transaction."""

    def __init__(
        self, paths: Mapping[str, str | Path], *,
        expected_sha256: Mapping[str, str],
        relative_names: Mapping[str, str] | None = None,
        maximum_bytes: Mapping[str, int] | None = None,
        prefix: str = "warehouse-r41-inputs-",
    ) -> None:
        if set(paths) != set(expected_sha256):
            raise ValueError("Every snapshot input requires one exact SHA-256")
        relative_names = dict(relative_names or {})
        maximum_bytes = dict(maximum_bytes or {})
        unknown = (set(relative_names) | set(maximum_bytes)) - set(paths)
        if unknown:
            raise ValueError("Snapshot options reference unknown inputs")
        self._temporary = tempfile.TemporaryDirectory(prefix=prefix)
        self.root = Path(self._temporary.name).resolve()
        os.chmod(self.root, 0o700)
        entries: dict[str, SnapshotEntry] = {}
        try:
            for index, name in enumerate(sorted(paths)):
                if not _SAFE_NAME.fullmatch(name):
                    raise ValueError("Snapshot input name is unsafe: " + name)
                maximum = maximum_bytes.get(name, DEFAULT_MAXIMUM_BYTES)
                original = canonical_regular_file(
                    paths[name], name, maximum=maximum)
                relative = relative_names.get(
                    name, f"files/{index:03d}-{name}{original.suffix}")
                relative_path = Path(relative)
                if (relative_path.is_absolute() or ".." in relative_path.parts
                        or not relative_path.parts):
                    raise ValueError("Snapshot relative path is unsafe: " + name)
                frozen = (self.root / relative_path).absolute()
                frozen.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                source_fd = os.open(
                    original, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                descriptor: int | None = None
                try:
                    descriptor = os.open(
                        frozen,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o400,
                    )
                    source_info = os.fstat(source_fd)
                    if (not stat.S_ISREG(source_info.st_mode)
                            or source_info.st_size > maximum):
                        raise ValueError(name + " must be a bounded regular file")
                    hasher = sha256()
                    total = 0
                    with os.fdopen(
                            source_fd, "rb", closefd=False) as incoming, os.fdopen(
                                descriptor, "wb", closefd=False) as outgoing:
                        while True:
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > maximum:
                                raise ValueError(name + " exceeds its size limit")
                            hasher.update(chunk)
                            outgoing.write(chunk)
                        outgoing.flush()
                        os.fsync(outgoing.fileno())
                    if hasher.hexdigest() != expected_sha256[name]:
                        raise ValueError("Exact " + name + " bytes required")
                finally:
                    os.close(source_fd)
                    if descriptor is not None:
                        os.close(descriptor)
                os.chmod(frozen, 0o400)
                entries[name] = SnapshotEntry(
                    original=original, frozen=frozen,
                    sha256=expected_sha256[name])
            for directory in sorted(
                    {self.root, *(entry.frozen.parent
                                  for entry in entries.values())},
                    key=lambda path: len(path.parts), reverse=True):
                descriptor = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        except BaseException:
            self._temporary.cleanup()
            raise
        self.entries = entries
        self.paths = {name: entry.frozen for name, entry in entries.items()}
        self.original_paths = {
            name: entry.original for name, entry in entries.items()}
        self.sha256 = {name: entry.sha256 for name, entry in entries.items()}

    def verify_frozen(self) -> None:
        for name, entry in self.entries.items():
            if (not entry.frozen.is_file() or entry.frozen.is_symlink()
                    or entry.frozen.resolve() != entry.frozen
                    or file_hash(entry.frozen) != entry.sha256):
                raise RuntimeError("Immutable input snapshot changed: " + name)

    def verify_originals(self) -> None:
        for name, entry in self.entries.items():
            if (not entry.original.is_file() or entry.original.is_symlink()
                    or entry.original.resolve() != entry.original
                    or file_hash(entry.original) != entry.sha256):
                raise RuntimeError(
                    "Evidence fixed inputs changed during transaction: " + name)

    def verify(self) -> None:
        self.verify_frozen()
        self.verify_originals()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._temporary.cleanup()
        return None


__all__ = [
    "DEFAULT_MAXIMUM_BYTES", "SnapshotEntry", "ImmutableInputSnapshot",
    "canonical_regular_file", "read_authenticated_bytes",
]
