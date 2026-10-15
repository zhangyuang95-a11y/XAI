#!/usr/bin/env python3
"""Fail-closed Render preflight for the ephemeral r4.1 diagnostic release."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_admission as admission_api
from backend.training import warehouse_r41_diagnostic_designation as designation_api
from backend.training import warehouse_r41_diagnostic_release_receipt as receipt_api
from scripts.preflight_warehouse_r41_render import _render_service
from ui import warehouse_alignment_r41_diagnostic_release as release


VERSION = "warehouse-r41-diagnostic-render-preflight.v1"
STATUS = "ready_for_manual_render_diagnostic_deploy"
RELEASE_MODULE = "ui.warehouse_alignment_r41_diagnostic_release"
SECRET_PATH = "/etc/secrets/warehouse_alignment_release.b64"
SERVICE_NAME = "policylens-warehouse-study"
PUBLIC_ORIGIN = "https://policylens-warehouse-study.onrender.com"
START_COMMAND = (
    "python -m ui.warehouse_alignment_online_server "
    "--release-module ui.warehouse_alignment_r41_diagnostic_release "
    "--base64 /etc/secrets/warehouse_alignment_release.b64"
)
HEX = re.compile(r"[0-9a-f]{64}\Z")
FILES = {
    "receipt": "release_receipt.json",
    "admission": "diagnostic_admission.json",
    "designation": "diagnostic_actor_designation.json",
    "package": "warehouse_r41_diagnostic_online_release.zip",
    "base64": "warehouse_r41_diagnostic_online_release.b64",
}


def _hash(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _regular(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(label + " must not be group/world accessible")
    return path


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            infos = [row for row in archive.infolist()
                     if row.filename == release.MANIFEST_NAME]
            if len(infos) != 1 or not 0 < infos[0].file_size <= release.MAX_MANIFEST_BYTES:
                raise ValueError("Diagnostic package must contain one bounded manifest")
            raw = archive.read(infos[0])
    except zipfile.BadZipFile as error:
        raise ValueError("Diagnostic package is not a ZIP") from error
    return sha256(raw).hexdigest()


def check_render_yaml(path: str | Path, *, package_sha256: str,
                      manifest_sha256: str) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("render.yaml must be a canonical regular file")
    service = _render_service(path.read_text(encoding="utf-8"))
    fields, disk, env = service["fields"], service["disk"], service["env"]
    if fields.get("name") != SERVICE_NAME or fields.get("runtime") != "python":
        raise ValueError("render.yaml targets another service/runtime")
    if fields.get("startCommand") != START_COMMAND:
        raise ValueError("render.yaml must explicitly start the diagnostic release")
    if fields.get("healthCheckPath") != "/health":
        raise ValueError("render.yaml must keep the warehouse health check")
    if fields.get("autoDeployTrigger") != "off":
        raise ValueError("Render automatic deployment must remain off")
    if fields.get("plan") != "free" or disk is not None:
        raise ValueError("Diagnostic deployment must use free ephemeral Render storage")
    if (fields.get("numInstances") not in (None, "1")
            or "scaling" in fields):
        raise ValueError("Diagnostic SQLite deployment must remain one instance")
    required = {
        "WAREHOUSE_RELEASE_PACKAGE_SHA256": package_sha256,
        "WAREHOUSE_RELEASE_MANIFEST_SHA256": manifest_sha256,
        "WAREHOUSE_PUBLIC_ORIGIN": PUBLIC_ORIGIN,
        "WAREHOUSE_STORAGE_MODE": "ephemeral",
    }
    for name, expected in required.items():
        if env.get(name) != expected:
            raise ValueError("render.yaml has a stale or missing " + name)
    database = env.get("WAREHOUSE_ONLINE_DATABASE", "")
    if (not re.fullmatch(r"/tmp/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.sqlite3",
                         database)
            or any(part in (".", "..") for part in Path(database).parts)):
        raise ValueError("Diagnostic SQLite database must be an explicit /tmp path")
    return {
        "service": SERVICE_NAME, "start_command": START_COMMAND,
        "release_module": RELEASE_MODULE, "secret_path": SECRET_PATH,
        "package_sha256": package_sha256, "manifest_sha256": manifest_sha256,
        "plan": "free", "auto_deploy": False, "single_instance": True,
        "database": database, "data_persistent": False,
    }


def preflight(*, release_root: str | Path, expected_receipt_sha256: str,
              render_yaml: str | Path) -> dict[str, Any]:
    if type(expected_receipt_sha256) is not str or HEX.fullmatch(
            expected_receipt_sha256) is None:
        raise ValueError("Exact external release receipt SHA-256 is required")
    root = Path(release_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("Diagnostic release root must be a canonical directory")
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("Diagnostic release root must stay inside repository") from None
    paths = {name: _regular(root, relative, "diagnostic " + name)
             for name, relative in FILES.items()}
    if _hash(paths["receipt"]) != expected_receipt_sha256:
        raise ValueError("External diagnostic release receipt SHA-256 differs")
    receipt = receipt_api.read_saved_receipt(
        paths["receipt"], expected_sha256=expected_receipt_sha256)
    if (receipt.get("admission_sha256") != _hash(paths["admission"])
            or receipt.get("designation_sha256") != _hash(paths["designation"])
            or receipt.get("package_sha256") != _hash(paths["package"])
            or receipt.get("base64_sha256") != _hash(paths["base64"])):
        raise ValueError("Diagnostic release-root files differ from receipt")
    admission = json.loads(paths["admission"].read_text(encoding="utf-8"))
    designation = json.loads(paths["designation"].read_text(encoding="utf-8"))
    if (admission.get("version") != admission_api.VERSION
            or admission.get("status") != admission_api.STATUS
            or admission.get("bindings", {}).get("diagnostic_designation_sha256")
                != receipt["designation_sha256"]
            or admission.get("behavior_performance_gate_passed") is not False
            or admission.get("behavior_performance_gate_waived") is not True
            or admission.get("waiver_scope") != ["behavior_performance"]
            or admission.get("formal_sample_eligible") is not False
            or admission.get("data_persistent") is not False
            or designation.get("version") != designation_api.VERSION
            or designation.get("status") != designation_api.STATUS
            or designation.get("bindings", {}).get("actor_sha256")
                != receipt["actor_sha256"]):
        raise ValueError("Diagnostic designation/admission boundary differs")
    decoded = release._package_bytes(base64_path=paths["base64"])
    package_bytes = release._package_bytes(package_path=paths["package"])
    if decoded != package_bytes:
        raise ValueError("Diagnostic Secret File does not encode the package")
    manifest_sha = _manifest_sha(paths["package"])
    if manifest_sha != receipt["manifest_sha256"]:
        raise ValueError("Diagnostic manifest bytes differ from receipt")
    manifest = release.inspect_online_release(
        package_path=paths["package"],
        expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=manifest_sha,
    )
    if (manifest.get("version") != release.VERSION
            or manifest.get("status") != release.STATUS
            or manifest.get("release") != release._release_projection()
            or manifest.get("parent", {}).get("diagnostic_admission_sha256")
                != receipt["admission_sha256"]
            or manifest.get("parent", {}).get("diagnostic_designation_sha256")
                != receipt["designation_sha256"]
            or manifest.get("identities", {}).get("actor_sha256")
                != receipt["actor_sha256"]):
        raise ValueError("Diagnostic package identity differs from receipt")
    loaded = release.load_online_release(
        base64_path=paths["base64"],
        expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=manifest_sha,
    )
    try:
        play = loaded.scenarios.get("splits", {}).get("play", [])
        formal = play[1:]
        fingerprints = [row.get("fingerprint") for row in play]
        families = [row.get("family_id") for row in formal]
        if (loaded.provenance.get("version") != release.VERSION
                or loaded.provenance.get("release_version")
                    != release.PUBLIC_RELEASE_VERSION
                or loaded.provenance.get("formal_sample_eligible") is not False
                or loaded.provenance.get("data_persistent") is not False
                or loaded.runtime.actor_sha256 != receipt["actor_sha256"]
                or loaded.release != release._release_projection()
                or len(play) != 7 or len(set(fingerprints)) != 7
                or len(formal) != 6 or len(set(families)) != 6
                or fingerprints != receipt["selected_scene_fingerprints"]):
            raise ValueError("Loaded diagnostic context differs from receipt")
    finally:
        loaded.close()
    render = check_render_yaml(
        render_yaml, package_sha256=receipt["package_sha256"],
        manifest_sha256=manifest_sha,
    )
    return {
        "version": VERSION, "status": STATUS,
        "release_module": RELEASE_MODULE, "release_version": release.PUBLIC_RELEASE_VERSION,
        "pilot_class": release.PILOT_CLASS, "secret_path": SECRET_PATH,
        "receipt_sha256": expected_receipt_sha256,
        "designation_sha256": receipt["designation_sha256"],
        "admission_sha256": receipt["admission_sha256"],
        "package_sha256": receipt["package_sha256"],
        "manifest_sha256": manifest_sha,
        "base64_sha256": receipt["base64_sha256"],
        "actor_sha256": receipt["actor_sha256"],
        "scene_fingerprints": {
            "tutorial": fingerprints[0], "X": fingerprints[1:4],
            "Y": fingerprints[4:7],
        },
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False, "render": render,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--expected-release-receipt-sha256", required=True)
    parser.add_argument("--render-yaml", type=Path, default=ROOT / "render.yaml")
    args = parser.parse_args(argv)
    result = preflight(
        release_root=args.release_root,
        expected_receipt_sha256=args.expected_release_receipt_sha256,
        render_yaml=args.render_yaml,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
