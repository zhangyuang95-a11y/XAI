#!/usr/bin/env python3
"""Fail-closed preflight for the warehouse r4.1 Render deployment.

This command never changes Render.  It verifies an externally hashed release
receipt, the admission, ZIP and Base64 Secret File, performs a full NumPy-only
release load, and checks that ``render.yaml`` explicitly selects the r4.1
loader with the exact package and manifest hashes.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import warehouse_alignment_r41_online_release as release


VERSION = "warehouse-r41-render-preflight.v1"
RELEASE_MODULE = "ui.warehouse_alignment_r41_online_release"
SECRET_PATH = "/etc/secrets/warehouse_alignment_release.b64"
SERVICE_NAME = "policylens-warehouse-study"
START_COMMAND = (
    "python -m ui.warehouse_alignment_online_server "
    "--release-module ui.warehouse_alignment_r41_online_release "
    "--base64 /etc/secrets/warehouse_alignment_release.b64"
)
HEX = re.compile(r"[0-9a-f]{64}\Z")
PAID_WEB_PLANS = {
    "0.5c-512mb", "1c-2g", "2c-4g", "2c-8g", "2c-16g",
    "4c-8g", "4c-16g", "4c-32g", "8c-16g", "8c-32g",
    "8c-64g", "12c-24g", "12c-48g", "12c-96g",
}
FILES = {
    "receipt": "release_receipt.json",
    "admission": "production_admission.json",
    "package": "warehouse_r41_online_release.zip",
    "base64": "warehouse_r41_online_release.b64",
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
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(label + " must not be group/world accessible")
    return path


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _manifest_sha(package: Path) -> str:
    try:
        with zipfile.ZipFile(package, "r") as archive:
            infos = archive.infolist()
            if len([row for row in infos if row.filename == release.MANIFEST_NAME]) != 1:
                raise ValueError("r4.1 package must contain one manifest.json")
            info = archive.getinfo(release.MANIFEST_NAME)
            if info.file_size <= 0 or info.file_size > release.MAX_MANIFEST_BYTES:
                raise ValueError("r4.1 manifest size is invalid")
            raw = archive.read(info)
    except zipfile.BadZipFile as error:
        raise ValueError("r4.1 deployment package is not a ZIP") from error
    return sha256(raw).hexdigest()


def _yaml_scalar(raw: str, label: str) -> str:
    """Read the small scalar subset used by the checked-in Blueprint.

    The deployment preflight deliberately does not grow a production YAML
    dependency.  Instead it accepts the plain and fully quoted scalars used by
    this repository and rejects ambiguous YAML constructs.
    """
    value = raw.strip()
    if not value:
        raise ValueError(label + " must be a scalar")
    if value[0] in "\"'":
        if len(value) < 2 or value[-1] != value[0]:
            raise ValueError(label + " has an invalid quoted scalar")
        if value[0] == '"':
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError(label + " has an invalid quoted scalar") from error
            if not isinstance(decoded, str):
                raise ValueError(label + " must be a string scalar")
            return decoded
        return value[1:-1].replace("''", "'")
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if not value or value[0] in "[{&*!|>":
        raise ValueError(label + " uses an unsupported YAML scalar")
    return value


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _mapping_at(lines: list[str], indent: int, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or _indent(line) != indent:
            continue
        match = re.fullmatch(r"([^:#][^:]*):\s*(.*)", stripped)
        if not match:
            raise ValueError(label + " contains an unsupported field")
        key, raw = match.group(1).strip(), match.group(2)
        if key in result:
            raise ValueError(label + " repeats field " + key)
        result[key] = raw
    return result


def _render_service(text: str) -> dict[str, Any]:
    """Return the one target service and its nested disk/environment values.

    Earlier code searched the entire YAML with regular expressions.  That
    could combine a paid plan or disk on an unrelated service with the target
    service's environment variables.  This structural reader is intentionally
    strict: r4.1 deploys from one canonical web-service declaration.
    """
    if "\t" in text:
        raise ValueError("render.yaml must use spaces for indentation")
    lines = text.splitlines()
    headers = [index for index, line in enumerate(lines)
               if line.strip() == "services:" and _indent(line) == 0]
    if len(headers) != 1:
        raise ValueError("render.yaml must contain one services section")
    start = headers[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#") and _indent(lines[index]) == 0:
            end = index
            break
    section = lines[start:end]
    entries = []
    for index, line in enumerate(section):
        match = re.fullmatch(r"( *)-\s+(.+)", line)
        if match:
            entries.append((index, len(match.group(1)), match.group(2)))
    if not entries:
        raise ValueError("render.yaml must declare one web service")
    service_indent = min(row[1] for row in entries)
    service_entries = [row for row in entries if row[1] == service_indent]
    if len(service_entries) != 1:
        raise ValueError("render.yaml must declare only the warehouse service")
    entry_index, _, entry = service_entries[0]
    type_match = re.fullmatch(r"type:\s*(.+)", entry)
    if (type_match is None
            or _yaml_scalar(type_match.group(1), "service type") != "web"):
        raise ValueError("render.yaml warehouse service must be a web service")
    body = section[entry_index + 1:]
    field_indents = [_indent(line) for line in body
                     if line.strip() and not line.lstrip().startswith("#")
                     and _indent(line) > service_indent]
    if not field_indents:
        raise ValueError("render.yaml warehouse service is empty")
    field_indent = min(field_indents)
    raw_fields = _mapping_at(body, field_indent, "warehouse service")
    fields = {key: (_yaml_scalar(value, "service " + key) if value.strip()
                    else None) for key, value in raw_fields.items()}
    fields["type"] = "web"

    def nested(name: str, *, required: bool = True) -> tuple[list[str], int] | None:
        markers = [index for index, line in enumerate(body)
                   if _indent(line) == field_indent
                   and line.strip().split(":", 1)[0] == name]
        if not markers and not required:
            return None
        if len(markers) != 1 or fields.get(name) is not None:
            raise ValueError("render.yaml must declare one structured " + name)
        first = markers[0] + 1
        last = len(body)
        for index in range(first, len(body)):
            if (body[index].strip()
                    and not body[index].lstrip().startswith("#")
                    and _indent(body[index]) <= field_indent):
                last = index
                break
        rows = body[first:last]
        indents = [_indent(line) for line in rows
                   if line.strip() and not line.lstrip().startswith("#")]
        if not indents or min(indents) <= field_indent:
            raise ValueError("render.yaml " + name + " is empty")
        return rows, min(indents)

    disk_block = nested("disk", required=False)
    disk = None
    if disk_block is not None:
        disk_lines, disk_indent = disk_block
        raw_disk = _mapping_at(disk_lines, disk_indent, "warehouse disk")
        if set(raw_disk) != {"name", "mountPath", "sizeGB"}:
            raise ValueError("render.yaml disk must contain name, mountPath, and sizeGB")
        disk = {key: _yaml_scalar(value, "disk " + key)
                for key, value in raw_disk.items()}

    env_lines, env_indent = nested("envVars")
    env_entries = []
    for index, line in enumerate(env_lines):
        if _indent(line) == env_indent:
            match = re.fullmatch(r"\s*-\s+key:\s*(.+)", line)
            if match is None:
                raise ValueError("render.yaml envVars contains an unsupported entry")
            env_entries.append((index, _yaml_scalar(match.group(1), "environment key")))
    if not env_entries:
        raise ValueError("render.yaml envVars is empty")
    env: dict[str, str] = {}
    for position, (index, key) in enumerate(env_entries):
        if key in env:
            raise ValueError("render.yaml repeats environment key " + key)
        stop = env_entries[position + 1][0] if position + 1 < len(env_entries) else len(env_lines)
        value_rows = env_lines[index + 1:stop]
        value_indents = [_indent(line) for line in value_rows
                         if line.strip() and not line.lstrip().startswith("#")]
        if not value_indents:
            raise ValueError("render.yaml environment key has no value: " + key)
        raw_value = _mapping_at(value_rows, min(value_indents),
                                "environment " + key)
        if set(raw_value) != {"value"}:
            raise ValueError("render.yaml environment key needs one literal value: " + key)
        env[key] = _yaml_scalar(raw_value["value"], "environment " + key)
    return {"fields": fields, "disk": disk, "env": env}


def _render_env(text: str) -> dict[str, str]:
    return _render_service(text)["env"]


def check_render_yaml(path: str | Path, *, package_sha256: str,
                      manifest_sha256: str) -> dict[str, Any]:
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("render.yaml must be a canonical regular file")
    text = path.read_text(encoding="utf-8")
    service = _render_service(text)
    fields, disk, env = service["fields"], service["disk"], service["env"]
    if fields.get("startCommand") != START_COMMAND:
        raise ValueError("render.yaml must explicitly start the r4.1 release module")
    if fields.get("name") != SERVICE_NAME:
        raise ValueError("render.yaml targets a different service")
    if fields.get("runtime") != "python":
        raise ValueError("render.yaml warehouse service must use Python")
    if fields.get("healthCheckPath") != "/health":
        raise ValueError("render.yaml must keep the warehouse health check")
    if fields.get("autoDeployTrigger") != "off":
        raise ValueError("Render automatic deployment must remain off")
    required = {
        "WAREHOUSE_RELEASE_PACKAGE_SHA256": package_sha256,
        "WAREHOUSE_RELEASE_MANIFEST_SHA256": manifest_sha256,
        "WAREHOUSE_PUBLIC_ORIGIN":
            "https://policylens-warehouse-study.onrender.com",
    }
    for name, expected in required.items():
        if env.get(name) != expected:
            raise ValueError("render.yaml has a stale or missing " + name)
    # The participant protocol promises that confirmed records survive a
    # restart.  Do not accept an operator-set label on an ephemeral SQLite
    # path as evidence of that property.  This deployment path deliberately
    # requires the one low-risk backend already implemented by the online
    # server: a single-instance SQLite database on a Render persistent disk.
    if env.get("WAREHOUSE_STORAGE_MODE") != "persistent":
        raise ValueError("r4.1 deployment requires persistent storage mode")
    database = env.get("WAREHOUSE_ONLINE_DATABASE", "")
    if (not re.fullmatch(
            r"/var/data/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.sqlite3",
            database)
            or any(part in (".", "..") for part in Path(database).parts)):
        raise ValueError("r4.1 persistent database must be under /var/data")
    if fields.get("plan") not in PAID_WEB_PLANS:
        raise ValueError("r4.1 persistent disk requires one supported paid Render service plan")
    if fields.get("numInstances") != "1" or "scaling" in fields:
        raise ValueError("r4.1 persistent SQLite requires one non-autoscaled service instance")
    if disk is None:
        raise ValueError("render.yaml must declare one persistent disk")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", disk["name"]):
        raise ValueError("r4.1 persistent disk must have one stable name")
    if disk["mountPath"] != "/var/data":
        raise ValueError("r4.1 persistent disk must mount at /var/data")
    if not disk["sizeGB"].isdigit() or int(disk["sizeGB"]) < 1:
        raise ValueError("r4.1 persistent disk must declare a positive sizeGB")
    return {"service": SERVICE_NAME, "start_command": START_COMMAND,
            "package_sha256": package_sha256,
            "manifest_sha256": manifest_sha256,
            "auto_deploy": False, "data_persistent": True,
            "database": database, "disk_mount_path": "/var/data"}


def preflight(*, release_root: str | Path, expected_receipt_sha256: str,
              render_yaml: str | Path) -> dict[str, Any]:
    if not isinstance(expected_receipt_sha256, str) or not HEX.fullmatch(
            expected_receipt_sha256):
        raise ValueError("exact external release receipt SHA-256 is required")
    root = Path(release_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("release root must be a canonical directory")
    try:
        root.relative_to(ROOT.resolve())
    except ValueError:
        raise ValueError("release root must stay inside the repository") from None
    paths = {name: _regular(root, relative, "r4.1 " + name)
             for name, relative in FILES.items()}
    if _hash(paths["receipt"]) != expected_receipt_sha256:
        raise ValueError("external release receipt SHA-256 differs")
    receipt = _json(paths["receipt"], "r4.1 release receipt")
    required_receipt = {
        "version", "status", "formal_ready",
        "human_explanation_effect_validated", "actor_sha256",
        "admission_sha256", "package_sha256", "manifest_sha256",
        "base64_sha256",
    }
    if (not required_receipt <= set(receipt)
            or receipt.get("version")
                != "warehouse-r41-postfreeze-orchestration.v1"
            or receipt.get("status") != "completed_internal_pilot_release"
            or receipt.get("formal_ready") is not False
            or receipt.get("human_explanation_effect_validated") is not False
            or any(not isinstance(receipt.get(name), str)
                   or HEX.fullmatch(receipt[name]) is None
                   for name in required_receipt - {
                       "version", "status", "formal_ready",
                       "human_explanation_effect_validated"} )):
        raise ValueError("exact non-formal r4.1 release receipt is required")
    for name, key in (("admission", "admission_sha256"),
                      ("package", "package_sha256"),
                      ("base64", "base64_sha256")):
        if _hash(paths[name]) != receipt[key]:
            raise ValueError("r4.1 " + name + " bytes differ from receipt")
    encoded = b"".join(paths["base64"].read_bytes().split())
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("r4.1 Secret File is not strict Base64") from error
    package_bytes = paths["package"].read_bytes()
    if decoded != package_bytes:
        raise ValueError("r4.1 Secret File does not encode the admitted package")
    manifest_sha = _manifest_sha(paths["package"])
    if manifest_sha != receipt["manifest_sha256"]:
        raise ValueError("r4.1 manifest bytes differ from receipt")
    manifest = release.inspect_online_release(
        package_path=paths["package"],
        expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=manifest_sha,
    )
    if (manifest.get("version") != release.VERSION
            or manifest.get("status") != release.STATUS
            or manifest.get("formal_ready") is not False
            or manifest.get("parent", {}).get("production_admission_sha256")
                != receipt["admission_sha256"]
            or manifest.get("identities", {}).get("actor_sha256")
                != receipt["actor_sha256"]):
        raise ValueError("r4.1 package identity differs from release receipt")
    loaded = release.load_online_release(
        base64_path=paths["base64"],
        expected_package_sha256=receipt["package_sha256"],
        expected_manifest_sha256=manifest_sha,
    )
    try:
        if (loaded.package_sha256 != receipt["package_sha256"]
                or loaded.manifest_sha256 != manifest_sha
                or loaded.runtime.actor_sha256 != receipt["actor_sha256"]
                or loaded.release.get("formal_ready") is not False):
            raise ValueError("loaded r4.1 context differs from deployment receipt")
        play = loaded.scenarios.get("splits", {}).get("play", [])
        selected_fingerprints = [row.get("fingerprint") for row in play[1:7]]
        if (len(play) != 7 or len(selected_fingerprints) != 6
                or len(set(selected_fingerprints)) != 6
                or any(not isinstance(value, str) or HEX.fullmatch(value) is None
                       for value in selected_fingerprints)):
            raise ValueError("loaded r4.1 selected-scene identity differs")
    finally:
        loaded.close()
    render = check_render_yaml(
        render_yaml, package_sha256=receipt["package_sha256"],
        manifest_sha256=manifest_sha)
    return {
        "version": VERSION,
        "status": "ready_for_manual_render_deploy",
        "release_module": RELEASE_MODULE,
        "secret_path": SECRET_PATH,
        "receipt_sha256": expected_receipt_sha256,
        "admission_sha256": receipt["admission_sha256"],
        "package_sha256": receipt["package_sha256"],
        "manifest_sha256": manifest_sha,
        "base64_sha256": receipt["base64_sha256"],
        "actor_sha256": receipt["actor_sha256"],
        "scene_fingerprints": {
            "X": selected_fingerprints[:3],
            "Y": selected_fingerprints[3:],
        },
        "render": render,
        "data_persistent": True,
        "formal_ready": False,
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
