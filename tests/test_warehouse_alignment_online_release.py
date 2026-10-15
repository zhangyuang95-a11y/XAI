from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

from backend.training.warehouse_native_common import canonical, digest
from ui import warehouse_alignment_online_release as online


def _artifacts():
    return {
        "actor": b"synthetic actor",
        "protocol": b'{"synthetic":true}\n',
        "play_scenarios": b'{"synthetic":true}\n',
        "program": b'{"synthetic":true}\n',
        "question_bank": b'{"synthetic":true}\n',
    }


def _manifest(artifacts):
    zero = "0" * 64
    sources = online._portable_sources()
    parent_sources = {next(iter(sources)): next(iter(sources.values()))}
    portable_sources = dict(list(sources.items())[1:])
    if not portable_sources:
        portable_sources = {"ui/warehouse_alignment_online_release.py":
                            sources["ui/warehouse_alignment_online_release.py"]}
        parent_sources = {"scripts/build_warehouse_alignment_online_release.py":
                          sources["scripts/build_warehouse_alignment_online_release.py"]}
    ids = [f"play_{index}" for index in range(12)]
    return {
        "version": online.VERSION,
        "status": online.STATUS,
        "test_fixture": False,
        "formal_ready": False,
        "parent": {
            "version": "parent.v1", "status": "local_pilot_technically_verified",
            "manifest_sha256": zero, "context_signature": zero,
            "source_binding_sha256": digest(parent_sources),
            "scenario_manifest_sha256": zero,
        },
        "artifacts": online._artifact_records(artifacts),
        "identities": {
            "actor_sha256": zero, "protocol_sha256": zero,
            "program_sha256": zero, "parent_runtime_signature": zero,
            "parent_explainer_signature": zero,
            "parent_question_bank_signature": zero,
            "question_bank_private_items_sha256": zero,
            "question_bank_public_items_sha256": zero,
            "portable_scenarios_sha256": zero, "play_scene_count": len(ids),
            "play_scene_ids": ids,
        },
        "sources": {"parent": parent_sources, "portable": portable_sources},
        "analysis": online._analysis_protocol(),
        "release": online._release_projection(),
    }


def _write_package(tmp_path, manifest, artifacts):
    raw = online._archive_bytes(manifest, artifacts)
    path = tmp_path / "release.zip"
    path.write_bytes(raw)
    manifest_raw = (canonical(manifest) + "\n").encode()
    return path, sha256(raw).hexdigest(), sha256(manifest_raw).hexdigest()


def test_archive_is_deterministic_and_accepts_package_or_base64(tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    first = online._archive_bytes(manifest, artifacts)
    second = online._archive_bytes(manifest, artifacts)
    assert first == second
    package, package_sha, manifest_sha = _write_package(tmp_path, manifest, artifacts)
    assert online.inspect_online_release(
        package_path=package, expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )["parent"]["manifest_sha256"] == "0" * 64
    secret = tmp_path / "release.b64"
    secret.write_bytes(base64.b64encode(first) + b"\n")
    assert online.inspect_online_release(
        base64_path=secret, expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )["identities"]["play_scene_count"] == 12


def test_render_style_secret_symlink_is_read_after_opened_target_validation(tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    package, package_sha, manifest_sha = _write_package(tmp_path, manifest, artifacts)
    target = tmp_path / "mounted-value.b64"
    target.write_bytes(base64.b64encode(package.read_bytes()) + b"\n")
    secret = tmp_path / "warehouse_alignment_release.b64"
    secret.symlink_to(target)

    inspected = online.inspect_online_release(
        base64_path=secret, expected_package_sha256=package_sha,
        expected_manifest_sha256=manifest_sha,
    )
    assert inspected["identities"]["play_scene_count"] == 12


def test_archive_rejects_wrong_package_manifest_and_artifact_hashes(tmp_path):
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    package, package_sha, manifest_sha = _write_package(tmp_path, manifest, artifacts)
    with pytest.raises(ValueError, match="package bytes differ"):
        online.inspect_online_release(
            package_path=package, expected_package_sha256="f" * 64,
            expected_manifest_sha256=manifest_sha,
        )
    with pytest.raises(ValueError, match="manifest bytes differ"):
        online.inspect_online_release(
            package_path=package, expected_package_sha256=package_sha,
            expected_manifest_sha256="f" * 64,
        )

    changed = deepcopy(manifest)
    changed["artifacts"]["protocol"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Manifest artifact records differ"):
        online._archive_bytes(changed, artifacts)


def _unsafe_zip(manifest, artifacts, extra_name, extra_data=b"unsafe"):
    members = {online.MANIFEST_NAME: (canonical(manifest) + "\n").encode(),
               **{online.ARTIFACT_PATHS[name]: raw for name, raw in artifacts.items()},
               extra_name: extra_data}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    return output.getvalue()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "artifacts/extra.json"])
def test_archive_rejects_non_whitelisted_and_escaping_members(tmp_path, name):
    artifacts = _artifacts()
    manifest = _manifest(artifacts)
    raw = _unsafe_zip(manifest, artifacts, name)
    path = tmp_path / "unsafe.zip"
    path.write_bytes(raw)
    manifest_sha = sha256((canonical(manifest) + "\n").encode()).hexdigest()
    with pytest.raises(ValueError, match="whitelist"):
        online.inspect_online_release(
            package_path=path, expected_package_sha256=sha256(raw).hexdigest(),
            expected_manifest_sha256=manifest_sha,
        )
    assert not (tmp_path.parent / "escape").exists()


def test_secret_selection_invalid_base64_and_size_are_fail_closed(tmp_path):
    secret = tmp_path / "bad.b64"
    secret.write_text("not base64!")
    with pytest.raises(ValueError, match="exactly one"):
        online.inspect_online_release(
            expected_package_sha256="0" * 64,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="Invalid Base64"):
        online.inspect_online_release(
            base64_path=secret, expected_package_sha256="0" * 64,
            expected_manifest_sha256="0" * 64,
        )
    secret.write_bytes(b"A" * (online.MAX_BASE64_BYTES + 1))
    with pytest.raises(ValueError, match="safe limit"):
        online.inspect_online_release(
            base64_path=secret, expected_package_sha256="0" * 64,
            expected_manifest_sha256="0" * 64,
        )


def test_manifest_cannot_claim_fixture_or_formal_readiness():
    artifacts = _artifacts()
    for field in ("formal_ready", "test_fixture"):
        manifest = _manifest(artifacts)
        manifest[field] = True
        with pytest.raises(ValueError, match="non-formal"):
            online._validate_manifest(manifest)
    manifest = _manifest(artifacts)
    manifest["release"]["formal_ready"] = True
    with pytest.raises(ValueError, match="release flags"):
        online._validate_manifest(manifest)


def test_build_command_help_is_importable():
    result = subprocess.run(
        [sys.executable, "scripts/build_warehouse_alignment_online_release.py", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert "--parent-release-root" in result.stdout
    assert "--expected-parent-manifest-sha256" in result.stdout


def test_online_loader_imports_no_torch_or_sklearn():
    code = """
import sys
import ui.warehouse_alignment_online_release
assert 'torch' not in sys.modules
assert not any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules)
assert not any(name == 'backend.training' or name.startswith('backend.training.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr


ROOT = Path(__file__).resolve().parents[1]
