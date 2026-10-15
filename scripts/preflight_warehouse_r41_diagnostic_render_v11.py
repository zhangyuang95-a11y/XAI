#!/usr/bin/env python3
"""Fail-closed local Render preflight for r4.1 diagnostic v11.

This command validates an already admitted package.  It never calls Render or
reads protected evaluation material beyond the admission readers supplied by
the caller.
"""
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
from backend.training import warehouse_r41_diagnostic_admission_v11 as admission_api
from backend.training import warehouse_r41_diagnostic_release_receipt_v11 as receipt_api
from backend.training.warehouse_native_common import file_hash
from scripts.preflight_warehouse_r41_diagnostic_render_v6 import (
    check_clean_checkout_load, check_clean_checkout_source_closure,
)
from scripts.preflight_warehouse_r41_render import _render_service
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release

VERSION = "warehouse-r41-diagnostic-render-preflight.v11"
STATUS = "ready_for_manual_render_diagnostic_v11_deploy"
RELEASE_MODULE = "ui.warehouse_alignment_r41_diagnostic_release_v9"
SECRET_PATH = "/etc/secrets/warehouse_alignment_release.b64"
SERVICE_NAME = "policylens-warehouse-study"
PUBLIC_ORIGIN = "https://policylens-warehouse-study.onrender.com"
START_COMMAND = (
    "python -m ui.warehouse_alignment_online_server "
    "--release-module ui.warehouse_alignment_r41_diagnostic_release_v9 "
    "--base64 /etc/secrets/warehouse_alignment_release.b64")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _private_file(value: str | Path, label: str, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat().st_size <= maximum
            or stat.S_IMODE(path.stat().st_mode) & 0o077):
        raise ValueError(label + " must be a private bounded regular file")
    return path


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            items = [row for row in archive.infolist()
                     if row.filename == release.MANIFEST_NAME]
            if len(items) != 1:
                raise ValueError("V11 package must contain one manifest")
            return sha256(archive.read(items[0])).hexdigest()
    except zipfile.BadZipFile as error:
        raise ValueError("V11 package is not a ZIP") from error


def check_render_yaml(path: str | Path, *, package_sha256: str,
                      manifest_sha256: str) -> dict[str, Any]:
    yaml_path = Path(path).expanduser().absolute()
    if (not yaml_path.is_file() or yaml_path.is_symlink()
            or yaml_path.resolve() != yaml_path):
        raise ValueError("render.yaml must be a canonical regular file")
    service = _render_service(yaml_path.read_text(encoding="utf-8"))
    fields, disk, env = service["fields"], service["disk"], service["env"]
    if (fields.get("name") != SERVICE_NAME or fields.get("runtime") != "python"
            or fields.get("startCommand") != START_COMMAND
            or fields.get("healthCheckPath") != "/health"
            or fields.get("autoDeployTrigger") != "off"
            or fields.get("plan") != "free" or disk is not None
            or fields.get("numInstances") not in (None, "1")
            or "scaling" in fields):
        raise ValueError("render.yaml differs from the v9 single free service")
    required = {
        "WAREHOUSE_RELEASE_PACKAGE_SHA256": package_sha256,
        "WAREHOUSE_RELEASE_MANIFEST_SHA256": manifest_sha256,
        "WAREHOUSE_PUBLIC_ORIGIN": PUBLIC_ORIGIN,
        "WAREHOUSE_STORAGE_MODE": "ephemeral",
    }
    if any(env.get(name) != child for name, child in required.items()):
        raise ValueError("render.yaml has stale or missing v11 release variables")
    database = env.get("WAREHOUSE_ONLINE_DATABASE", "")
    if (not re.fullmatch(r"/tmp/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.sqlite3",
                         database)
            or any(part in (".", "..") for part in Path(database).parts)):
        raise ValueError("V9 diagnostic database must use explicit /tmp SQLite")
    return {"service": SERVICE_NAME, "release_module": RELEASE_MODULE,
            "secret_path": SECRET_PATH, "plan": "free",
            "auto_deploy": False, "single_instance": True,
            "data_persistent": False, "database": database}


def preflight(*, admission_path: str | Path,
              expected_admission_sha256: str,
              components, outer_permanent_registry: str | Path,
              final_permanent_registry: str | Path,
              permanent_promotion_closeout_registry: str | Path,
              package_path: str | Path, base64_path: str | Path,
              receipt_path: str | Path, expected_receipt_sha256: str,
              render_yaml: str | Path, check_clean_checkout: bool = True):
    if _HEX.fullmatch(str(expected_receipt_sha256)) is None:
        raise ValueError("Exact external v11 receipt SHA-256 required")
    package = _private_file(package_path, "v9 package", release.MAX_PACKAGE_BYTES)
    encoded = _private_file(base64_path, "v9 Base64 secret",
                            release.MAX_BASE64_BYTES)
    receipt_file = _private_file(receipt_path, "v9 receipt", 8 * 1024 * 1024)
    if file_hash(receipt_file) != expected_receipt_sha256:
        raise ValueError("V11 receipt bytes differ")
    inputs = dict(
        admission_path=admission_path,
        expected_admission_sha256=expected_admission_sha256,
        components=components,
        outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        permanent_promotion_closeout_registry=permanent_promotion_closeout_registry,
        package_path=package, base64_path=encoded)
    receipt = receipt_api.read_saved_receipt(
        receipt_file, expected_sha256=expected_receipt_sha256, **inputs)
    manifest_sha = _manifest_sha(package)
    manifest = release.inspect_online_release(
        package_path=package, expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=manifest_sha)
    if (manifest_sha != receipt["manifest_sha256"]
            or manifest["parent"]["admission_sha256"]
                != receipt["admission_sha256"]
            or len(manifest["play_scenes"]) != 7
            or len({row["family_id"] for row in manifest["play_scenes"][1:]}) != 6):
        raise ValueError("V11 release identity differs at preflight")
    sources = dict(manifest["sources"]["release"])
    clean = None
    if check_clean_checkout:
        check_clean_checkout_source_closure(ROOT, sources)
        clean = check_clean_checkout_load(
            ROOT, sources, base64_path=encoded,
            package_sha256=receipt["package_sha256"],
            manifest_sha256=manifest_sha, release_module=RELEASE_MODULE)
        if (clean.get("version") != release.VERSION
                or clean.get("release_version") != release.PUBLIC_RELEASE_VERSION
                or clean.get("actor_sha256") != receipt["actor_sha256"]
                or clean.get("formal_sample_eligible") is not False
                or clean.get("data_persistent") is not False):
            raise ValueError("Clean-checkout v11 release load differs")
    render = check_render_yaml(
        render_yaml, package_sha256=receipt["package_sha256"],
        manifest_sha256=manifest_sha)
    return {
        "version": VERSION, "status": STATUS,
        "release_module": RELEASE_MODULE,
        "release_version": release.PUBLIC_RELEASE_VERSION,
        "pilot_class": release.PILOT_CLASS,
        "secret_path": SECRET_PATH,
        "receipt_sha256": expected_receipt_sha256,
        "admission_sha256": receipt["admission_sha256"],
        "package_sha256": receipt["package_sha256"],
        "manifest_sha256": manifest_sha,
        "base64_sha256": receipt["base64_sha256"],
        "actor_sha256": receipt["actor_sha256"],
        "program_sha256": receipt["program_sha256"],
        "scene_fingerprints": {
            "tutorial": manifest["play_scenes"][0]["fingerprint"],
            "X": [row["fingerprint"] for row in manifest["play_scenes"][1:4]],
            "Y": [row["fingerprint"] for row in manifest["play_scenes"][4:7]],
        },
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "formal_ready": False, "formal_sample_eligible": False,
        "data_persistent": False, "render": render,
        "clean_checkout_load": clean,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    for name in admission_api.ARTIFACT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--outer-permanent-registry", type=Path, required=True)
    parser.add_argument("--final-permanent-registry", type=Path, required=True)
    parser.add_argument(
        "--permanent-promotion-closeout-registry", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--base64", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--render-yaml", type=Path, required=True)
    args = parser.parse_args(argv)
    components = {name: getattr(args, name)
                  for name in admission_api.ARTIFACT_NAMES}
    result = preflight(
        admission_path=args.admission,
        expected_admission_sha256=args.expected_admission_sha256,
        components=components,
        outer_permanent_registry=args.outer_permanent_registry,
        final_permanent_registry=args.final_permanent_registry,
        permanent_promotion_closeout_registry=args.permanent_promotion_closeout_registry,
        package_path=args.package, base64_path=args.base64,
        receipt_path=args.receipt,
        expected_receipt_sha256=args.expected_receipt_sha256,
        render_yaml=args.render_yaml)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
