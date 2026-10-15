"""Small, release-independent source-closure hasher for diagnostic evidence.

Evidence producers must not depend on the mutable release admission module just
to discover their Python import closure.  Keeping this utility separate means a
concurrent packaging edit cannot invalidate a long-running scientific audit.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import os
from pathlib import Path
import stat
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
MAXIMUM_SOURCE_BYTES = 16 * 1024 * 1024
MAXIMUM_ASSET_BYTES = 512 * 1024 * 1024


def _canonical_regular_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Resolve one regular file without accepting a symlink spelling."""
    supplied = Path(value).expanduser().absolute()
    try:
        resolved = supplied.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            "Diagnostic source closure contains a missing or external path"
        ) from error
    if supplied.is_symlink() or resolved != supplied:
        raise ValueError("Diagnostic source closure cannot contain a symlink")
    if root is not None and resolved != root and root not in resolved.parents:
        raise ValueError(
            "Diagnostic source closure contains a missing or external path")
    return resolved


def _read_regular_bytes(
    value: str | Path, *, maximum: int,
) -> tuple[Path, bytes, str]:
    """Read and hash one immutable view through one no-follow descriptor."""
    path = _canonical_regular_path(value)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(
                "Diagnostic source closure file is not a bounded regular file")
        chunks: list[bytes] = []
        hasher = sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ValueError("Diagnostic source closure file exceeds size limit")
            hasher.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_mode, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_mode, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_after != identity_before or total != before.st_size:
            raise RuntimeError(
                "Diagnostic source closure file changed while it was read")
    finally:
        os.close(descriptor)
    return path, b"".join(chunks), hasher.hexdigest()


def _module_files(parts: Sequence[str], *, root: Path) -> list[Path]:
    if not parts or any(not part or part in (".", "..") for part in parts):
        return []
    base = root.joinpath(*parts)
    result = []
    source = base.with_suffix(".py")
    package = base / "__init__.py"
    if source.is_file():
        result.append(source)
    if package.is_file():
        result.append(package)
    return result


def file_hash(path: str | Path) -> str:
    return _read_regular_bytes(path, maximum=MAXIMUM_ASSET_BYTES)[2]


def _discover_once(
    seed_paths: Sequence[Path], *, excluded: set[Path], root: Path,
) -> dict[str, str]:
    """Discover one complete closure from internally consistent file views."""
    queue = list(seed_paths)
    seen: set[Path] = set()
    captured_hashes: dict[Path, str] = {}
    while queue:
        path = queue.pop()
        if path in seen or path in excluded:
            continue
        path = _canonical_regular_path(path, root=root)
        maximum = (MAXIMUM_SOURCE_BYTES if path.suffix == ".py"
                   else MAXIMUM_ASSET_BYTES)
        _, raw, content_sha256 = _read_regular_bytes(path, maximum=maximum)
        seen.add(path)
        captured_hashes[path] = content_sha256
        if path.suffix != ".py":
            continue
        relative = path.relative_to(root)
        module_parts = list(relative.with_suffix("").parts)
        package_parts = module_parts[:-1]
        if module_parts[-1] == "__init__":
            package_parts = module_parts[:-1]
        try:
            source = raw.decode("utf-8")
            tree = ast.parse(source, filename=str(relative))
        except (SyntaxError, UnicodeError) as error:
            raise ValueError(
                "Cannot parse diagnostic source closure file: " + str(relative)
            ) from error
        imported: list[list[str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name.split(".") for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 0:
                        continue
                    base = package_parts[:keep]
                else:
                    base = []
                if node.module:
                    base = [*base, *node.module.split(".")]
                if base:
                    imported.append(base)
                for alias in node.names:
                    if alias.name != "*":
                        imported.append([*base, *alias.name.split(".")])
        for parts in imported:
            for imported_path in _module_files(parts, root=root):
                if imported_path.is_symlink():
                    raise ValueError("Diagnostic source closure cannot contain a symlink")
                if imported_path not in seen and imported_path not in excluded:
                    queue.append(imported_path)
            for index in range(1, len(parts)):
                initializer = root.joinpath(*parts[:index], "__init__.py")
                if initializer.is_symlink():
                    raise ValueError("Diagnostic source closure cannot contain a symlink")
                if (initializer.is_file() and initializer not in seen
                        and initializer not in excluded):
                    queue.append(initializer)
    return {
        str(path.relative_to(root)): captured_hashes[path]
        for path in sorted(seen)
    }


def local_source_hashes(seed_paths: Sequence[str | Path], *,
                        exclude_paths: Sequence[str | Path] = ()) -> dict[str, str]:
    """Hash a stable recursive local-Python import closure plus assets.

    Two independent complete discoveries bind both file contents and import
    target existence.  Rehashing only the first pass's paths would miss a
    dependency that was temporarily hidden while its importing source was
    inspected.
    """
    root = ROOT.resolve()
    excluded = {_canonical_regular_path(path, root=root)
                for path in exclude_paths}
    seeds = tuple(_canonical_regular_path(value, root=root)
                  for value in seed_paths)
    first = _discover_once(seeds, excluded=excluded, root=root)
    second = _discover_once(seeds, excluded=excluded, root=root)
    if second != first:
        raise RuntimeError(
            "Diagnostic source closure changed during discovery")
    return first


__all__ = [
    "ROOT", "MAXIMUM_SOURCE_BYTES", "MAXIMUM_ASSET_BYTES", "file_hash",
    "local_source_hashes",
]
