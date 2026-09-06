"""Materialize a hash-pinned, private Render Secret File without publishing it.

Only the public descriptor belongs in Git. The base64 file, ZIP, extracted
questionnaire and program remain private. No archive names or contents are
included in error output.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
from package_release import ALLOWED

SCHEMA = "cooperative_kitchen_secret_release_v1"
DESCRIPTOR = ROOT / "deployment/cooperative_kitchen/release.json"
SECRET = Path("/etc/secrets/kitchen_release.b64")
MAX_SECRET_BYTES = 1_000_000  # Render's combined Secret Files limit is 1 MB.
MAX_EXPANDED_BYTES = 16 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 64 * 1024
TARGET_PREFIX = "output/cooperative_kitchen/"
SHA = re.compile(r"^[0-9a-f]{64}$")


class ReleaseError(ValueError):
    """Fixed diagnostic codes, never secrets or archive-provided filenames."""


def require(value, code):
    if not value:
        raise ReleaseError(code)


def sha(contents):
    return hashlib.sha256(contents).hexdigest()


def relative_path(value):
    require(isinstance(value, str) and bool(value), "invalid_relative_path")
    p = PurePosixPath(value)
    require(not p.is_absolute() and p.as_posix() == value and "\\" not in value and ":" not in value
            and all(part not in {"", ".", ".."} and not part.startswith(".") for part in value.split("/"))
            and not any(ord(c) < 32 or ord(c) == 127 for c in value), "unsafe_relative_path")
    return p


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_path_forbidden")
    return path


def read_limited(path, limit):
    path = no_symlinks(path)
    require(path.is_file(), "required_file_missing")
    with path.open("rb") as source:
        data = source.read(limit + 1)
    require(len(data) <= limit, "file_size_limit")
    return data


def strict_json(data):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(data, object_pairs_hook=pairs,
            parse_constant=lambda _: require(False, "nonfinite_json_value"))
    except (ValueError, UnicodeError):
        raise ReleaseError("invalid_json") from None


def load_descriptor(path=DESCRIPTOR):
    document = strict_json(read_limited(path, MAX_DESCRIPTOR_BYTES))
    require(isinstance(document, dict) and document.get("schema") == SCHEMA
            and document.get("status") == "prepared", "release_descriptor_not_prepared")
    required = {"schema", "status", "release_status", "release_path", "bundle_sha256", "bundle_bytes",
                "manifest_sha256", "runtime_sha256", "members"}
    require(set(document) == required, "descriptor_fields_mismatch")
    require(document['release_status'] in {'candidate', 'pilot_ready', 'formal_ready'}, 'invalid_release_status')
    target = relative_path(document["release_path"])
    require(document["release_path"].startswith(TARGET_PREFIX) and len(target.parts) == 3,
            "invalid_release_destination")
    for key in ("bundle_sha256", "manifest_sha256", "runtime_sha256"):
        require(isinstance(document[key], str) and SHA.fullmatch(document[key]), "invalid_descriptor_hash")
    require(type(document["bundle_bytes"]) is int and 0 < document["bundle_bytes"] <= MAX_SECRET_BYTES * 3 // 4,
            "invalid_bundle_size")
    members = document["members"]
    require(isinstance(members, list) and len(members) == len(ALLOWED) + 1, "member_count_mismatch")
    kinds, names, total = set(), set(), 0
    for item in members:
        require(isinstance(item, dict) and set(item) == {"kind", "path", "sha256", "bytes"}, "invalid_member_descriptor")
        kind, name = item["kind"], item["path"]
        require(isinstance(kind, str) and kind in {*ALLOWED, "manifest"} and kind not in kinds, "member_kind_mismatch")
        relative_path(name)
        require(name not in names and name.casefold() not in {p.casefold() for p in names}, "duplicate_member_path")
        require(name == "manifest.json" if kind == "manifest" else PurePosixPath(name).suffix == ALLOWED[kind], "member_type_mismatch")
        require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "invalid_member_hash")
        require(type(item["bytes"]) is int and 0 < item["bytes"] <= MAX_EXPANDED_BYTES, "invalid_member_size")
        kinds.add(kind); names.add(name); total += item["bytes"]
    require(total <= MAX_EXPANDED_BYTES, "expanded_size_limit")
    require(next(item["sha256"] for item in members if item["kind"] == "manifest") == document["manifest_sha256"],
            "manifest_descriptor_mismatch")
    return document


def validate_manifest(contents, document):
    manifest = strict_json(contents)
    require(isinstance(manifest, dict) and manifest.get("runtime_sha256") == document["runtime_sha256"]
            and manifest.get("status") == document["release_status"], "manifest_version_mismatch")
    entries = manifest.get("artifacts")
    require(isinstance(entries, dict) and set(entries) == set(ALLOWED), "manifest_artifact_whitelist_mismatch")
    for item in document["members"]:
        if item["kind"] == "manifest":
            continue
        entry = entries[item["kind"]]
        require(isinstance(entry, dict) and entry.get("path") == item["path"]
                and entry.get("sha256") == item["sha256"], "manifest_artifact_binding_mismatch")
    return manifest


def decode_bundle(secret_path, document):
    encoded = read_limited(secret_path, MAX_SECRET_BYTES)
    # Permit only line endings produced by text editors. Other whitespace,
    # excess padding, noncanonical encodings and appended data are rejected.
    compact = encoded.replace(b"\r\n", b"\n").replace(b"\n", b"")
    try:
        raw = base64.b64decode(compact, validate=True)
    except (ValueError, binascii.Error):
        raise ReleaseError("invalid_base64") from None
    require(base64.b64encode(raw) == compact, "noncanonical_base64")
    require(len(raw) == document["bundle_bytes"] and sha(raw) == document["bundle_sha256"], "bundle_checksum_mismatch")
    return raw


def validated_members(raw, document):
    expected = {item["path"]: item for item in document["members"]}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            require(not archive.comment and len(infos) == len(expected), "zip_member_count_mismatch")
            require(len({item.filename for item in infos}) == len(infos), "duplicate_zip_member")
            require({item.filename for item in infos} == set(expected), "zip_whitelist_mismatch")
            output = {}
            for info in infos:
                relative_path(info.filename)
                mode = info.external_attr >> 16
                require(not info.is_dir() and stat.S_IFMT(mode) in {0, stat.S_IFREG}
                        and not info.external_attr & 0x10 and not info.flag_bits & 1 and not info.comment and not info.extra,
                        "unsafe_zip_member")
                require(info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}, "unsupported_zip_compression")
                item = expected[info.filename]
                require(info.file_size == item["bytes"], "zip_member_size_mismatch")
                with archive.open(info) as source:
                    contents = source.read(item["bytes"] + 1)
                require(len(contents) == item["bytes"] and sha(contents) == item["sha256"], "zip_member_checksum_mismatch")
                output[info.filename] = contents
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError):
        raise ReleaseError("invalid_zip") from None
    validate_manifest(output["manifest.json"], document)
    return output


def verify_materialized(directory, document):
    directory = no_symlinks(directory)
    require(directory.is_dir(), "release_directory_missing")
    expected = {item["path"]: item for item in document["members"]}
    paths = list(directory.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlink_path_forbidden")
    actual = {p.relative_to(directory).as_posix() for p in paths if p.is_file()}
    require(actual == set(expected), "materialized_whitelist_mismatch")
    expected_dirs = {p.as_posix() for name in expected for p in PurePosixPath(name).parents if str(p) != "."}
    require({p.relative_to(directory).as_posix() for p in paths if p.is_dir()} == expected_dirs,
            "materialized_directory_mismatch")
    for name, item in expected.items():
        data = read_limited(directory / name, item["bytes"])
        require(len(data) == item["bytes"] and sha(data) == item["sha256"], "materialized_checksum_mismatch")
    validate_manifest(read_limited(directory / "manifest.json", MAX_EXPANDED_BYTES), document)


def materialize(descriptor_path=DESCRIPTOR, secret_path=SECRET, *, root=ROOT, runtime_sha256=None):
    document = load_descriptor(descriptor_path)
    root = no_symlinks(root)
    destination = no_symlinks(root / document["release_path"])
    require(destination.is_relative_to(root), "destination_outside_project")
    configured = os.environ.get("KITCHEN_OUTPUT")
    if configured:
        configured_path = Path(configured)
        if not configured_path.is_absolute(): configured_path = root / configured_path
        require(no_symlinks(configured_path) == destination, "configured_output_mismatch")
    if runtime_sha256 is None:
        sys.path.insert(0, str(root))
        from backend.cooperative_kitchen.artifacts import runtime_hash
        runtime_sha256 = runtime_hash()
    require(runtime_sha256 == document["runtime_sha256"], "runtime_checksum_mismatch")
    contents = validated_members(decode_bundle(secret_path, document), document)
    if destination.exists():
        verify_materialized(destination, document)
        return {"passed": True, "materialized": False, "already_present": True,
                "runtime_sha256": runtime_sha256, "files": len(contents), "recruitment_gate_changed": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".kitchen-release-", dir=destination.parent))
    stage.chmod(0o700)
    try:
        for name, data in contents.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data); target.chmod(0o600)
        verify_materialized(stage, document)
        # Never replace an existing release, including an empty directory.
        require(not destination.exists() and not destination.is_symlink(), "release_destination_exists")
        no_symlinks(destination.parent)
        stage.rename(destination)
    finally:
        if stage.exists(): shutil.rmtree(stage)
    return {"passed": True, "materialized": True, "already_present": False,
            "runtime_sha256": runtime_sha256, "files": len(contents), "recruitment_gate_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, default=DESCRIPTOR)
    parser.add_argument("--secret-file", type=Path, default=SECRET, help="Explicit local path for private packaging tests.")
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(args.descriptor, args.secret_file), indent=2))
    except ReleaseError as error:
        # ReleaseError messages are fixed diagnostic codes declared in this
        # module. They identify the failed check without printing archive
        # names, contents, credentials, or environment values.
        parser.exit(1, f"Kitchen release materialization failed validation ({error}); no private contents were logged.\n")
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(1, "Kitchen release materialization failed validation (runtime_error); no private contents were logged.\n")
