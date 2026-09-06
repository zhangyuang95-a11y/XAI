"""Explicitly package only manifest-selected kitchen runtime artifacts into a private ZIP.

This command does not publish, upload, commit, or deploy the archive. Keep the
archive private: questionnaire answer keys and extracted programs are included.
Candidates require --allow-candidate and remain locked by server release gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile


REPORTS = ("training", "extraction", "qa", "calibration", "questionnaire", "protocol", "remote_load", "recovery")
ALLOWED = {"actor": ".npz", "program": ".json", "scenarios": ".json", "questionnaire": ".json", **{key + "_report": ".json" for key in REPORTS}}
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "output/cooperative_kitchen/v3-id-pilot"


def sha256(path):
    with Path(path).open("rb") as source: return hashlib.file_digest(source, "sha256").hexdigest()


def selected_files(directory, manifest):
    entries = manifest.get("artifacts")
    if not isinstance(entries, dict): raise ValueError("A release manifest with an artifacts object is required.")
    selected = [("manifest", "manifest.json", directory / "manifest.json")]
    names = {"manifest.json"}
    for kind, suffix in ALLOWED.items():
        entry = entries.get(kind)
        if entry is None: continue
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str): raise ValueError(f"Invalid {kind} artifact entry.")
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts) or "\\" in entry["path"]:
            raise ValueError(f"Unsafe {kind} artifact path.")
        if relative.suffix.lower() != suffix or any(part.startswith(".") for part in relative.parts): raise ValueError(f"Unexpected {kind} artifact file type.")
        file = directory.joinpath(*relative.parts)
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(directory): raise ValueError(f"Missing or out-of-tree {kind} artifact.")
        if any(parent.is_symlink() for parent in file.parents if parent != directory and parent.is_relative_to(directory)):
            raise ValueError(f"Symlinked {kind} artifact path is not permitted.")
        if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64 or sha256(file) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {kind}.")
        archive_name = relative.as_posix()
        if archive_name in names: raise ValueError("Artifacts must have distinct archive paths.")
        names.add(archive_name); selected.append((kind, archive_name, file))
    if "actor" not in {kind for kind, _, _ in selected}: raise ValueError("A selected neural Actor is required, including for a candidate bundle.")
    return selected


def package(directory, destination, *, allow_candidate=False):
    directory = Path(directory).expanduser().resolve()
    manifest_file = directory / "manifest.json"
    if manifest_file.is_symlink(): raise ValueError("The manifest cannot be a symbolic link.")
    raw_manifest = manifest_file.read_bytes()
    manifest = json.loads(raw_manifest)
    status = manifest.get("status", "candidate")
    candidate = status not in {"pilot_ready", "formal_ready"}
    if candidate and not allow_candidate: raise ValueError("This release is a candidate. Use --allow-candidate only for private review; this does not open the study gate.")
    selected = selected_files(directory, manifest)
    included = {kind for kind, _, _ in selected}
    missing = sorted(set(ALLOWED) - included)
    if not candidate:
        if missing: raise ValueError("A ready release is missing required artifacts: " + ", ".join(missing))
        for kind, _, file in selected:
            if kind.endswith("_report") and json.loads(file.read_text()).get("passed") is not True:
                raise ValueError("A ready release contains a failing or unconfirmed gate: " + kind)
    destination = Path(destination).expanduser().resolve()
    if destination.suffix.lower() != ".zip": raise ValueError("The destination must be a .zip archive.")
    if destination.exists(): raise ValueError("Archive already exists. Choose a new filename; existing bundles are not overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".kitchen-release-", suffix=".zip", dir=destination.parent); os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for kind, name, file in selected:
                contents = raw_manifest if kind == "manifest" else file.read_bytes()
                # Recheck after reading: a concurrent training/export write must not
                # create a bundle whose contents differ from the frozen manifest.
                if kind != "manifest" and hashlib.sha256(contents).hexdigest() != manifest["artifacts"][kind]["sha256"]:
                    raise ValueError("An artifact changed while packaging: " + kind)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)); info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o600) << 16; info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, contents, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.chmod(temporary, 0o600)
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"schema": "cooperative_kitchen_private_bundle_v1", "archive": str(destination), "sha256": sha256(destination),
            "bytes": destination.stat().st_size, "release_status": status, "candidate": candidate,
            "files": [name for _, name, _ in selected], "missing_candidate_artifacts": missing,
            "ignored_manifest_keys": sorted(set(manifest["artifacts"]) - set(ALLOWED)),
            "distribution": "Private build artifact only. Contains questionnaire answer keys and extracted program."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default=str(DEFAULT_ROOT), help="Directory containing release manifest.json.")
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "release-bundle.zip"))
    parser.add_argument("--allow-candidate", action="store_true")
    args = parser.parse_args()
    try:
        result = package(args.release, args.output, allow_candidate=args.allow_candidate)
        if result["candidate"]: print("WARNING: Candidate bundle for private review. Research recruitment remains gated.", file=__import__("sys").stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as error: parser.exit(1, str(error) + "\n")


if __name__ == "__main__": main()
