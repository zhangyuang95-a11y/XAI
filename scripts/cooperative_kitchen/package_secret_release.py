"""Prepare a private base64 ZIP and a public, content-free release descriptor.

Run only after the manifest and source are frozen. Upload the .b64 contents as
Render Secret File kitchen_release.b64; commit only the public descriptor.
Never publishes, pushes, reads credentials, or changes recruitment gates.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from package_release import ALLOWED, package, selected_files, sha256
from materialize_release import (DESCRIPTOR, MAX_SECRET_BYTES, SCHEMA, ReleaseError,
    load_descriptor, no_symlinks, require, validated_members)


def prepare(release, archive, secret_file, descriptor=DESCRIPTOR, *, replace_descriptor=False,
            allow_candidate=False, root=ROOT, runtime_sha256=None):
    root = no_symlinks(root)
    release, archive, secret_file, descriptor = [no_symlinks(path) for path in (release, archive, secret_file, descriptor)]
    require(release.is_relative_to(root), "release_outside_project")
    relative = release.relative_to(root).as_posix()
    require(relative.startswith("output/cooperative_kitchen/") and len(Path(relative).parts) == 3, "invalid_release_destination")
    # Private products stay under the ignored output tree, never deployment/.
    private_root = root / "output/cooperative_kitchen"
    require(archive.is_relative_to(private_root) and secret_file.is_relative_to(private_root), "private_output_location_required")
    require(archive != secret_file and not archive.exists() and not secret_file.exists(), "private_output_exists")
    require(secret_file.suffix == ".b64", "base64_output_suffix_required")
    require(not descriptor.exists() or replace_descriptor, "descriptor_exists_use_explicit_replace")
    require(descriptor.is_relative_to(root / "deployment/cooperative_kitchen"), "public_descriptor_location_required")
    manifest = json.loads((release / "manifest.json").read_bytes())
    if runtime_sha256 is None:
        from backend.cooperative_kitchen.artifacts import runtime_hash
        runtime_sha256 = runtime_hash()
    require(manifest.get("runtime_sha256") == runtime_sha256, "freeze_matching_runtime_first")
    files = selected_files(release, manifest)
    require(set(manifest.get("artifacts", {})) == set(ALLOWED)
            and {kind for kind, _, _ in files} == {*ALLOWED, "manifest"}, "all_twelve_artifacts_required")
    packaged = package(release, archive, allow_candidate=allow_candidate)
    raw = archive.read_bytes()
    encoded = base64.b64encode(raw) + b"\n"
    require(len(encoded) <= MAX_SECRET_BYTES, "render_secret_file_size_limit")
    document = {"schema": SCHEMA, "status": "prepared", "release_status": manifest.get("status"),
        "release_path": relative, "bundle_sha256": packaged["sha256"], "bundle_bytes": len(raw),
        "manifest_sha256": sha256(release / "manifest.json"), "runtime_sha256": runtime_sha256,
        "members": [{"kind": kind, "path": name, "sha256": sha256(path), "bytes": path.stat().st_size}
                    for kind, name, path in files]}
    # Validate ZIP-to-manifest bindings before emitting the private Secret File.
    validated_members(raw, document)
    descriptor.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=".release-descriptor-", dir=descriptor.parent)
    pending = Path(name)
    try:
        with os.fdopen(handle, "w") as target:
            json.dump(document, target, indent=2); target.write("\n"); target.flush(); os.fsync(target.fileno())
        load_descriptor(pending)
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(secret_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as target:
            target.write(encoded); target.flush(); os.fsync(target.fileno())
        if replace_descriptor:
            no_symlinks(descriptor)
            os.replace(pending, descriptor)
        else:
            os.link(pending, descriptor)
    finally:
        pending.unlink(missing_ok=True)
    return {"passed": True, "descriptor": str(descriptor), "private_archive": str(archive),
        "private_secret_file": str(secret_file), "secret_file_bytes": len(encoded),
        "render_combined_secret_limit_bytes": MAX_SECRET_BYTES, "runtime_sha256": runtime_sha256,
        "manifest_sha256": document["manifest_sha256"], "bundle_sha256": document["bundle_sha256"],
        "files": len(files), "release_status": document["release_status"],
        "private_contents_printed": False, "recruitment_gate_changed": False, "uploaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=ROOT / "output/cooperative_kitchen/v3-id-pilot")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR)
    parser.add_argument("--replace-descriptor", action="store_true")
    parser.add_argument("--allow-candidate", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.release, args.archive, args.secret_file, args.descriptor,
            replace_descriptor=args.replace_descriptor, allow_candidate=args.allow_candidate), indent=2))
    except (ReleaseError, OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Kitchen Secret File packaging failed validation; no private contents were logged.\n")
