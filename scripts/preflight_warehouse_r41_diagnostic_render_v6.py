#!/usr/bin/env python3
"""Fail-closed Render preflight for the ephemeral r4.1 diagnostic release."""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_admission_v6 as admission_api
from backend.training import warehouse_r41_diagnostic_designation_v2 as designation_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as input_snapshot_api
from backend.training import warehouse_r41_diagnostic_release_receipt_v6 as receipt_api
from scripts.preflight_warehouse_r41_render import _render_service
from ui import warehouse_alignment_r41_diagnostic_release_v8 as release


VERSION = "warehouse-r41-diagnostic-render-preflight.v6"
STATUS = "ready_for_manual_render_diagnostic_deploy"
RELEASE_MODULE = "ui.warehouse_alignment_r41_diagnostic_release_v8"
SECRET_PATH = "/etc/secrets/warehouse_alignment_release.b64"
SERVICE_NAME = "policylens-warehouse-study"
PUBLIC_ORIGIN = "https://policylens-warehouse-study.onrender.com"
START_COMMAND = (
    "python -m ui.warehouse_alignment_online_server "
    "--release-module ui.warehouse_alignment_r41_diagnostic_release_v8 "
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

_CLEAN_LOAD_SCRIPT = r"""
import importlib
import json
from pathlib import Path
import sys

checkout = Path(sys.argv[1]).resolve()
module_name = sys.argv[2]
base64_path = Path(sys.argv[3]).resolve()
package_sha256 = sys.argv[4]
manifest_sha256 = sys.argv[5]
source_roots = set(json.loads(sys.argv[6]))
sys.path.insert(0, str(checkout))
module = importlib.import_module(module_name)
context = module.load_online_release(
    base64_path=base64_path,
    expected_package_sha256=package_sha256,
    expected_manifest_sha256=manifest_sha256,
)
try:
    play = context.scenarios.get("splits", {}).get("play", [])
    result = {
        "version": context.provenance.get("version"),
        "release_version": context.provenance.get("release_version"),
        "formal_sample_eligible": context.provenance.get(
            "formal_sample_eligible"),
        "data_persistent": context.provenance.get("data_persistent"),
        "actor_sha256": context.runtime.actor_sha256,
        "release": context.release,
        "play_scene_fingerprints": [row.get("fingerprint") for row in play],
        "formal_family_ids": [row.get("family_id") for row in play[1:]],
    }
    for name, loaded in tuple(sys.modules.items()):
        top = name.split(".", 1)[0]
        if top not in source_roots:
            continue
        source = getattr(loaded, "__file__", None)
        if source is not None:
            try:
                Path(source).resolve().relative_to(checkout)
            except ValueError:
                raise RuntimeError("Release module escaped clean checkout")
        paths = getattr(loaded, "__path__", None)
        if paths is not None:
            for value in paths:
                try:
                    Path(value).resolve().relative_to(checkout)
                except ValueError:
                    raise RuntimeError("Release package escaped clean checkout")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
finally:
    context.close()
"""


def check_clean_checkout_source_closure(
        root: str | Path, sources: dict[str, str]) -> dict[str, Any]:
    """Require every serving dependency to be tracked, clean and hash exact."""
    checkout = Path(root).expanduser().absolute()
    if (checkout.is_symlink() or not checkout.is_dir()
            or checkout.resolve() != checkout
            or not (checkout / ".git").exists()):
        raise ValueError("Release source closure requires a Git checkout")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Release source closure is empty")
    relatives = sorted(sources)
    for relative in relatives:
        expected = sources[relative]
        candidate = Path(relative)
        if (candidate.is_absolute() or ".." in candidate.parts
                or candidate.as_posix() != relative
                or type(expected) is not str or HEX.fullmatch(expected) is None):
            raise ValueError("Release source closure contains an unsafe path")
        path = checkout / candidate
        if (path.is_symlink() or not path.is_file() or path.resolve() != path
                or _hash(path) != expected):
            raise ValueError("Release source closure differs: " + relative)
    tracked = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "--", *relatives],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if set(tracked) != set(relatives):
        raise ValueError("Release source closure contains untracked files")
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain",
         "--untracked-files=no", "--", *relatives],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("Release source closure has uncommitted tracked changes")
    return {"tracked": len(relatives), "clean": True,
            "sources_sha256": sha256(json.dumps(
                sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _source_roots(relatives: list[str]) -> list[str]:
    roots: set[str] = set()
    for relative in relatives:
        parts = PurePosixPath(relative).parts
        if relative.endswith(".py"):
            roots.add(parts[0] if len(parts) > 1 else Path(parts[0]).stem)
    if not roots:
        raise ValueError("Release source closure has no Python source")
    return sorted(roots)


def _materialize_head_sources(checkout: Path, sources: dict[str, str],
                              destination: Path) -> None:
    """Materialize only the authenticated source closure from committed HEAD."""
    relatives = sorted(sources)
    try:
        archived = subprocess.run(
            ["git", "-C", str(checkout), "archive", "--format=tar", "HEAD",
             "--", *relatives],
            check=True, capture_output=True, timeout=120,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("Cannot materialize committed release source closure") from error
    if not archived:
        raise ValueError("Committed release source closure archive is empty")
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archived), mode="r:") as archive:
            for member in archive.getmembers():
                normalized = (member.name[:-1] if member.isdir()
                              and member.name.endswith("/") else member.name)
                pure = PurePosixPath(normalized)
                if (pure.is_absolute() or ".." in pure.parts
                        or pure.as_posix() != normalized
                        or member.issym() or member.islnk()
                        or not (member.isdir() or member.isfile())):
                    raise ValueError("Committed release source archive is unsafe")
                if member.isdir():
                    continue
                relative = pure.as_posix()
                if relative not in sources or relative in seen:
                    raise ValueError("Committed release source archive differs")
                stream = archive.extractfile(member)
                raw = stream.read() if stream is not None else b""
                if sha256(raw).hexdigest() != sources[relative]:
                    raise ValueError("Committed release source hash differs")
                target = destination.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = target.open("xb")
                try:
                    descriptor.write(raw)
                finally:
                    descriptor.close()
                seen.add(relative)
    except tarfile.TarError as error:
        raise ValueError("Committed release source archive is invalid") from error
    if seen != set(relatives):
        raise ValueError("Committed release source archive is incomplete")


def check_clean_checkout_load(
        root: str | Path, sources: dict[str, str], *,
        base64_path: str | Path, package_sha256: str, manifest_sha256: str,
        release_module: str = RELEASE_MODULE) -> dict[str, Any]:
    """Load the private package using only authenticated files from Git HEAD."""
    closure = check_clean_checkout_source_closure(root, sources)
    checkout = Path(root).expanduser().absolute()
    secret = Path(base64_path).expanduser().absolute()
    if (secret.is_symlink() or not secret.is_file() or secret.resolve() != secret
            or stat.S_IMODE(secret.stat().st_mode) & 0o077):
        raise ValueError(
            "diagnostic Base64 package must be a private canonical regular file")
    if (type(package_sha256) is not str or HEX.fullmatch(package_sha256) is None
            or type(manifest_sha256) is not str
            or HEX.fullmatch(manifest_sha256) is None):
        raise ValueError("Exact package and manifest SHA-256 are required")
    if (type(release_module) is not str
            or re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*",
                            release_module) is None):
        raise ValueError("Safe release module name required")
    relatives = sorted(sources)
    with tempfile.TemporaryDirectory(
            prefix="warehouse-r41-diagnostic-clean-checkout-") as temporary:
        material = Path(temporary)
        _materialize_head_sources(checkout, sources, material)
        command = [
            sys.executable, "-I", "-c", _CLEAN_LOAD_SCRIPT,
            str(material), release_module, str(secret), package_sha256,
            manifest_sha256, json.dumps(_source_roots(relatives),
                                        separators=(",", ":")),
        ]
        try:
            completed = subprocess.run(
                command, cwd=material, check=True, capture_output=True,
                text=True, timeout=120,
            )
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as error:
            raise ValueError("Clean-checkout diagnostic release load failed") from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or completed.stderr.strip():
        raise ValueError("Clean-checkout diagnostic release emitted unexpected output")
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("Clean-checkout diagnostic release output is invalid") from error
    expected_fields = {
        "version", "release_version", "formal_sample_eligible",
        "data_persistent", "actor_sha256", "release",
        "play_scene_fingerprints", "formal_family_ids",
    }
    if not isinstance(result, dict) or set(result) != expected_fields:
        raise ValueError("Clean-checkout diagnostic release output schema differs")
    result["source_closure"] = closure
    return result


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
            if (len(infos) != 1
                    or infos[0].compress_type != release.ARCHIVE_COMPRESSION
                    or not 0 < infos[0].file_size <= release.MAX_MANIFEST_BYTES):
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
    # Re-run the complete admission reader against every registered evidence
    # file.  Matching claims in admission, package and receipt are not enough:
    # the lineage must still agree with the fixed final ledger and live files.
    admission = receipt_api.read_strict_admission_anchor(
        paths["admission"], expected_sha256=receipt["admission_sha256"])
    with input_snapshot_api.ImmutableInputSnapshot(
            {
                "designation": paths["designation"],
                "package": paths["package"],
                "base64": paths["base64"],
            },
            expected_sha256={
                "designation": receipt["designation_sha256"],
                "package": receipt["package_sha256"],
                "base64": receipt["base64_sha256"],
            },
            relative_names={
                "designation": "release/designation.json",
                "package": "release/package.zip",
                "base64": "release/package.b64",
            },
            prefix="warehouse-r41-render-preflight-",
    ) as immutable:
        semantic_paths = immutable.paths
        if semantic_paths["package"].stat().st_size > 750_000:
            raise ValueError("Diagnostic ZIP exceeds the 750 KB deployment limit")
        if semantic_paths["base64"].stat().st_size > 1_000_000:
            raise ValueError("Diagnostic Base64 Secret File exceeds the 1 MB limit")
        designation = json.loads(semantic_paths["designation"].read_text(encoding="utf-8"))
        admission_bindings = admission.get("bindings", {})
        if (admission.get("version") != admission_api.VERSION
                or admission.get("status") != admission_api.STATUS
                or admission.get("bindings", {}).get("diagnostic_designation_sha256")
                    != receipt["designation_sha256"]
                or admission.get("bindings", {}).get("actor_sha256")
                    != admission_api.FIXED_ACTOR_SHA256
                or receipt.get("actor_sha256") != admission_api.FIXED_ACTOR_SHA256
                or receipt.get("designation_sha256")
                    != designation_binding.EXPECTED_DESIGNATION_SHA256
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
        for name in receipt_api.FIELDS & admission_api.BINDING_FIELDS:
            if receipt.get(name) != admission_bindings.get(name):
                raise ValueError(
                    "Diagnostic receipt differs from admitted binding: " + name)
        decoded = release._package_bytes(base64_path=semantic_paths["base64"])
        package_bytes = release._package_bytes(package_path=semantic_paths["package"])
        if (len(package_bytes) > release.MAX_PACKAGE_BYTES
                or semantic_paths["base64"].stat().st_size > release.MAX_BASE64_BYTES):
            raise ValueError("Diagnostic package exceeds the Render Secret File limits")
        if decoded != package_bytes:
            raise ValueError("Diagnostic Secret File does not encode the package")
        manifest_sha = _manifest_sha(semantic_paths["package"])
        if manifest_sha != receipt["manifest_sha256"]:
            raise ValueError("Diagnostic manifest bytes differ from receipt")
        manifest = release.inspect_online_release(
            package_path=semantic_paths["package"],
            expected_package_sha256=receipt["package_sha256"],
            expected_manifest_sha256=manifest_sha,
        )
        expected_parent = {
            "version": admission["version"],
            "status": admission["status"],
            "diagnostic_admission_sha256": receipt["admission_sha256"],
            **{
                name: admission_bindings[name]
                for name in release._PARENT_FIELDS
                - {"version", "status", "diagnostic_admission_sha256"}
            },
        }
        if manifest.get("parent") != expected_parent:
            raise ValueError("Diagnostic package parent differs from admission")
        expected_identity_bindings = {
            "actor_sha256": "actor_sha256",
            "protocol_file_sha256": "protocol_file_sha256",
            "protocol_content_sha256": "protocol_content_sha256",
            "program_sha256": "program_sha256",
            "program_content_sha256": "program_content_sha256",
            "program_identity_sha256": "program_identity_sha256",
            "public_feature_contract_sha256": "public_feature_contract_sha256",
            "public_feature_registry_sha256": "public_feature_registry_sha256",
            "program_complexity_sha256": "program_complexity_sha256",
            "parent_runtime_signature": "runtime_signature",
            "runtime_manifest_signature": "runtime_manifest_signature",
            "parent_explainer_signature": "explainer_signature",
            "parent_question_bank_signature": "question_bank_signature",
            "tutorial_signature": "tutorial_signature",
        }
        identities = manifest.get("identities", {})
        if any(
                identities.get(identity_name) != admission_bindings.get(binding_name)
                for identity_name, binding_name in expected_identity_bindings.items()):
            raise ValueError("Diagnostic package identity differs from admission")
        if (manifest.get("version") != release.VERSION
                or manifest.get("status") != release.STATUS
                or manifest.get("release") != release._release_projection()
                or manifest.get("parent", {}).get("diagnostic_admission_sha256")
                    != receipt["admission_sha256"]
                or manifest.get("parent", {}).get("diagnostic_designation_sha256")
                    != receipt["designation_sha256"]
                or manifest.get("parent", {}).get(
                    "diagnostic_publication_authentication_sha256")
                    != receipt["diagnostic_publication_authentication_sha256"]
                or manifest.get("parent", {}).get(
                    "portable_runtime_manifest_sha256")
                    != receipt["portable_runtime_manifest_sha256"]
                or manifest.get("parent", {}).get("question_bank_sha256")
                    != receipt["question_bank_sha256"]
                or manifest.get("parent", {}).get("tutorial_sha256")
                    != receipt["tutorial_sha256"]
                or manifest.get("parent", {}).get("release_sources_sha256")
                    != receipt["release_sources_sha256"]
                or manifest.get("parent", {}).get("package_contract_sha256")
                    != receipt["package_contract_sha256"]
                or manifest.get("parent", {}).get(
                    "dynamic_selection_protocol_sha256")
                    != receipt["dynamic_selection_protocol_sha256"]
                or manifest.get("parent", {}).get(
                    "dynamic_selection_prefilter_sha256")
                    != receipt["dynamic_selection_prefilter_sha256"]
                or manifest.get("parent", {}).get(
                    "dynamic_selection_episodes_sha256")
                    != receipt["dynamic_selection_episodes_sha256"]
                or any(
                    manifest.get("parent", {}).get(name) != receipt[name]
                    for name in receipt_api.CANDIDATE_LINEAGE_FIELDS
                )
                or manifest.get("identities", {}).get("actor_sha256")
                    != receipt["actor_sha256"]
                or manifest.get("identities", {}).get("program_sha256")
                    != receipt["program_sha256"]
                or manifest.get("identities", {}).get("program_content_sha256")
                    != receipt["program_content_sha256"]
                or manifest.get("identities", {}).get("program_identity_sha256")
                    != receipt["program_identity_sha256"]
                or manifest.get("identities", {}).get("public_feature_contract_sha256")
                    != receipt["public_feature_contract_sha256"]
                or manifest.get("identities", {}).get("public_feature_registry_sha256")
                    != receipt["public_feature_registry_sha256"]
                or manifest.get("identities", {}).get("program_complexity_sha256")
                    != receipt["program_complexity_sha256"]
                or manifest.get("parent", {}).get("final_once_identity_sha256")
                    != receipt["final_once_identity_sha256"]
                or manifest.get("parent", {}).get("final_once_campaign_key")
                    != receipt["final_once_campaign_key"]
                or manifest.get("parent", {}).get(
                    "final_once_permanent_anchor_sha256")
                    != receipt["final_once_permanent_anchor_sha256"]
                or manifest.get("parent", {}).get("final_once_attempt_started_sha256")
                    != receipt["final_once_attempt_started_sha256"]
                or manifest.get("parent", {}).get(
                    "final_once_candidate_authenticated_sha256")
                    != receipt["final_once_candidate_authenticated_sha256"]
                or manifest.get("parent", {}).get("final_once_holdout_started_sha256")
                    != receipt["final_once_holdout_started_sha256"]
                or manifest.get("parent", {}).get(
                    "final_once_historical_exclusion_started_sha256")
                    != receipt["final_once_historical_exclusion_started_sha256"]
                or manifest.get("parent", {}).get(
                    "final_once_historical_exclusion_completed_sha256")
                    != receipt["final_once_historical_exclusion_completed_sha256"]
                or manifest.get("parent", {}).get("final_once_holdout_completed_sha256")
                    != receipt["final_once_holdout_completed_sha256"]
                or manifest.get("parent", {}).get("final_once_audit_started_sha256")
                    != receipt["final_once_audit_started_sha256"]
                or manifest.get("parent", {}).get("final_once_audit_completed_sha256")
                    != receipt["final_once_audit_completed_sha256"]
                or manifest.get("parent", {}).get("final_once_attempt_completed_sha256")
                    != receipt["final_once_attempt_completed_sha256"]
                or manifest.get("parent", {}).get("explanation_audit_inputs_sha256")
                    != receipt["explanation_audit_inputs_sha256"]
                or manifest.get("parent", {}).get("explanation_audit_evidence_sha256")
                    != receipt["explanation_audit_evidence_sha256"]
                or manifest.get("parent", {}).get("explanation_audit_sha256")
                    != receipt["explanation_audit_sha256"]
                or manifest.get("parent", {}).get("physical_replay_sha256")
                    != receipt["physical_replay_sha256"]
                or manifest.get("parent", {}).get(
                    "fresh_final_v3_exclusion_sha256")
                    != receipt["fresh_final_v3_exclusion_sha256"]
                or manifest.get("parent", {}).get(
                    "fresh_final_v3_exclusion_content_sha256")
                    != receipt["fresh_final_v3_exclusion_content_sha256"]):
            raise ValueError("Diagnostic package identity differs from receipt")
        clean_load = check_clean_checkout_load(
            ROOT, dict(manifest["sources"]["release"]),
            base64_path=semantic_paths["base64"],
            package_sha256=receipt["package_sha256"],
            manifest_sha256=manifest_sha,
        )
        fingerprints = clean_load["play_scene_fingerprints"]
        families = clean_load["formal_family_ids"]
        if (clean_load.get("version") != release.VERSION
                    or clean_load.get("release_version")
                        != release.PUBLIC_RELEASE_VERSION
                    or clean_load.get("formal_sample_eligible") is not False
                    or clean_load.get("data_persistent") is not False
                    or clean_load.get("actor_sha256") != receipt["actor_sha256"]
                    or clean_load.get("release") != release._release_projection()
                    or len(fingerprints) != 7 or len(set(fingerprints)) != 7
                    or len(families) != 6 or len(set(families)) != 6
                    or fingerprints != receipt["selected_scene_fingerprints"]):
            raise ValueError("Loaded diagnostic context differs from receipt")
        render = check_render_yaml(
            render_yaml, package_sha256=receipt["package_sha256"],
            manifest_sha256=manifest_sha,
        )
        immutable.verify()
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
            "program_sha256": receipt["program_sha256"],
            "program_content_sha256": receipt["program_content_sha256"],
            "program_identity_sha256": receipt["program_identity_sha256"],
            "public_feature_contract_sha256": receipt[
                "public_feature_contract_sha256"],
            "public_feature_registry_sha256": receipt[
                "public_feature_registry_sha256"],
            "program_complexity_sha256": receipt["program_complexity_sha256"],
            "diagnostic_publication_authentication_sha256": receipt[
                "diagnostic_publication_authentication_sha256"],
            **{
                name: receipt[name]
                for name in receipt_api.CANDIDATE_LINEAGE_FIELDS
            },
            "final_once_identity_sha256": receipt["final_once_identity_sha256"],
            "final_once_campaign_key": receipt["final_once_campaign_key"],
            "final_once_permanent_anchor_sha256": receipt[
                "final_once_permanent_anchor_sha256"],
            "final_once_attempt_completed_sha256": receipt[
                "final_once_attempt_completed_sha256"],
            "final_once_phase_receipts": {
                "candidate_authenticated.json": receipt[
                    "final_once_candidate_authenticated_sha256"],
                "holdout_started.json": receipt["final_once_holdout_started_sha256"],
                "historical_exclusion_started.json": receipt[
                    "final_once_historical_exclusion_started_sha256"],
                "historical_exclusion_completed.json": receipt[
                    "final_once_historical_exclusion_completed_sha256"],
                "holdout_completed.json": receipt[
                    "final_once_holdout_completed_sha256"],
                "audit_started.json": receipt["final_once_audit_started_sha256"],
                "audit_completed.json": receipt["final_once_audit_completed_sha256"],
            },
            "fresh_final_v3_exclusion_sha256": receipt[
                "fresh_final_v3_exclusion_sha256"],
            "fresh_final_v3_exclusion_content_sha256": receipt[
                "fresh_final_v3_exclusion_content_sha256"],
            "explanation_audit_evidence_sha256": receipt[
                "explanation_audit_evidence_sha256"],
            "scene_fingerprints": {
                "tutorial": fingerprints[0], "X": fingerprints[1:4],
                "Y": fingerprints[4:7],
            },
            "behavior_performance_gate_passed": False,
            "behavior_performance_gate_waived": True,
            "waiver_scope": ["behavior_performance"],
            "formal_ready": False, "formal_sample_eligible": False,
            "data_persistent": False, "render": render,
            "clean_checkout_load": clean_load,
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
