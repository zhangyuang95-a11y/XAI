from __future__ import annotations

import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from scripts import preflight_warehouse_r41_render as preflight


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _material(tmp_path: Path):
    root = tmp_path / "release"
    root.mkdir()
    admission = root / preflight.FILES["admission"]
    admission.write_text('{"admitted":true}\n', encoding="utf-8")
    package = root / preflight.FILES["package"]
    manifest = {
        "version": preflight.release.VERSION,
        "status": preflight.release.STATUS,
        "formal_ready": False,
        "parent": {"production_admission_sha256": _sha(admission)},
        "identities": {"actor_sha256": "a" * 64},
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest) + "\n")
    secret = root / preflight.FILES["base64"]
    secret.write_bytes(base64.b64encode(package.read_bytes()) + b"\n")
    manifest_sha = preflight._manifest_sha(package)
    receipt = {
        "version": "warehouse-r41-postfreeze-orchestration.v1",
        "status": "completed_internal_pilot_release",
        "formal_ready": False,
        "human_explanation_effect_validated": False,
        "actor_sha256": "a" * 64,
        "admission_sha256": _sha(admission),
        "package_sha256": _sha(package),
        "manifest_sha256": manifest_sha,
        "base64_sha256": _sha(secret),
    }
    receipt_path = root / preflight.FILES["receipt"]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    for path in root.iterdir():
        path.chmod(0o600)
    render = tmp_path / "render.yaml"
    render.write_text(f"""services:
  - type: web
    name: {preflight.SERVICE_NAME}
    runtime: python
    plan: 0.5c-512mb
    numInstances: 1
    startCommand: {preflight.START_COMMAND}
    healthCheckPath: /health
    autoDeployTrigger: \"off\"
    disk:
      name: warehouse-study-data
      mountPath: /var/data
      sizeGB: 1
    envVars:
      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256
        value: {receipt['package_sha256']}
      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256
        value: {receipt['manifest_sha256']}
      - key: WAREHOUSE_ONLINE_DATABASE
        value: /var/data/warehouse_alignment_online.sqlite3
      - key: WAREHOUSE_STORAGE_MODE
        value: persistent
      - key: WAREHOUSE_PUBLIC_ORIGIN
        value: https://policylens-warehouse-study.onrender.com
""", encoding="utf-8")
    return root, render, receipt, manifest


def test_preflight_binds_receipt_secret_package_loader_and_render(
        monkeypatch, tmp_path):
    root, render, receipt, manifest = _material(tmp_path)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    closed = []
    context = SimpleNamespace(
        package_sha256=receipt["package_sha256"],
        manifest_sha256=receipt["manifest_sha256"],
        runtime=SimpleNamespace(actor_sha256=receipt["actor_sha256"]),
        scenarios={"splits": {"play": [
            {"fingerprint": format(index + 1, "064x")} for index in range(7)
        ]}},
        release={"formal_ready": False},
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        preflight.release, "inspect_online_release",
        lambda **kwargs: manifest,
    )
    monkeypatch.setattr(
        preflight.release, "load_online_release",
        lambda **kwargs: context,
    )
    result = preflight.preflight(
        release_root=root,
        expected_receipt_sha256=_sha(root / preflight.FILES["receipt"]),
        render_yaml=render,
    )
    assert result["status"] == "ready_for_manual_render_deploy"
    assert result["release_module"] == preflight.RELEASE_MODULE
    assert result["package_sha256"] == receipt["package_sha256"]
    assert result["manifest_sha256"] == receipt["manifest_sha256"]
    assert result["scene_fingerprints"] == {
        "X": [format(index, "064x") for index in range(2, 5)],
        "Y": [format(index, "064x") for index in range(5, 8)],
    }
    assert result["formal_ready"] is False
    assert result["data_persistent"] is True
    assert result["render"]["database"] == "/var/data/warehouse_alignment_online.sqlite3"
    assert closed == [True]


def test_preflight_rejects_tampered_secret_before_loading(monkeypatch, tmp_path):
    root, render, receipt, _ = _material(tmp_path)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    secret = root / preflight.FILES["base64"]
    secret.write_bytes(base64.b64encode(b"different package") + b"\n")
    receipt["base64_sha256"] = _sha(secret)
    receipt_path = root / preflight.FILES["receipt"]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    receipt_path.chmod(0o600)
    monkeypatch.setattr(
        preflight.release, "load_online_release",
        lambda **kwargs: pytest.fail("tampered package reached loader"),
    )
    with pytest.raises(ValueError, match="does not encode"):
        preflight.preflight(
            release_root=root,
            expected_receipt_sha256=_sha(receipt_path),
            render_yaml=render,
        )


def test_render_yaml_rejects_default_loader_stale_hash_and_ephemeral_root(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text("""services:
  - type: web
    name: policylens-warehouse-study
    runtime: python
    healthCheckPath: /health
    startCommand: python -m ui.warehouse_alignment_online_server --base64 /etc/secrets/warehouse_alignment_release.b64
    autoDeployTrigger: off
    envVars:
      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256
        value: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
""", encoding="utf-8")
    with pytest.raises(ValueError, match="explicitly start"):
        preflight.check_render_yaml(
            path, package_sha256="a" * 64, manifest_sha256="b" * 64)

    # The checked-in service must remain undeployable through the r4.1
    # admission command.  A successful release branch has already selected the
    # r4.1 loader and must still fail on persistence.  A failed training branch
    # deliberately restores the complete r3 blueprint and must fail even
    # earlier on the loader check.
    text = (preflight.ROOT / "render.yaml").read_text(encoding="utf-8")
    env = preflight._render_env(text)
    expected_error = ("persistent storage mode"
                      if preflight.RELEASE_MODULE in text else "explicitly start")
    if preflight.RELEASE_MODULE not in text:
        assert "plan: free" in text
        assert env["WAREHOUSE_STORAGE_MODE"] == "ephemeral"
        assert 'autoDeployTrigger: "off"' in text
    with pytest.raises(ValueError, match=expected_error):
        preflight.check_render_yaml(
            preflight.ROOT / "render.yaml",
            package_sha256=env["WAREHOUSE_RELEASE_PACKAGE_SHA256"],
            manifest_sha256=env["WAREHOUSE_RELEASE_MANIFEST_SHA256"],
        )

    root, persistent, receipt, _ = _material(tmp_path)
    del root
    with pytest.raises(ValueError, match="stale or missing"):
        preflight.check_render_yaml(
            persistent,
            package_sha256="f" * 64,
            manifest_sha256=receipt["manifest_sha256"],
        )


@pytest.mark.parametrize(("mutation", "message"), [
    ("plan: 0.5c-512mb", "paid Render service plan"),
    ("numInstances: 1", "one non-autoscaled service instance"),
    ("mountPath: /var/data", "mount at /var/data"),
    ("value: /var/data/warehouse_alignment_online.sqlite3", "under /var/data"),
])
def test_render_yaml_rejects_incomplete_persistent_disk_declaration(
        tmp_path, mutation, message):
    _, render, receipt, _ = _material(tmp_path)
    text = render.read_text(encoding="utf-8")
    replacement = {
        "plan: 0.5c-512mb": "plan: free",
        "numInstances: 1": "numInstances: 2",
        "mountPath: /var/data": "mountPath: /tmp",
        "value: /var/data/warehouse_alignment_online.sqlite3":
            "value: /tmp/warehouse_alignment_online.sqlite3",
    }[mutation]
    render.write_text(text.replace(mutation, replacement), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        preflight.check_render_yaml(
            render, package_sha256=receipt["package_sha256"],
            manifest_sha256=receipt["manifest_sha256"])


def test_render_yaml_rejects_persistent_path_traversal(tmp_path):
    _, render, receipt, _ = _material(tmp_path)
    text = render.read_text(encoding="utf-8").replace(
        "/var/data/warehouse_alignment_online.sqlite3",
        "/var/data/../tmp/warehouse_alignment_online.sqlite3")
    render.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="under /var/data"):
        preflight.check_render_yaml(
            render, package_sha256=receipt["package_sha256"],
            manifest_sha256=receipt["manifest_sha256"])


def test_render_yaml_cannot_borrow_plan_or_disk_from_another_service(tmp_path):
    _, render, receipt, _ = _material(tmp_path)
    text = render.read_text(encoding="utf-8")
    render.write_text(text + """  - type: web
    name: unrelated-paid-service
    runtime: python
    plan: 0.5c-512mb
""", encoding="utf-8")
    with pytest.raises(ValueError, match="only the warehouse service"):
        preflight.check_render_yaml(
            render, package_sha256=receipt["package_sha256"],
            manifest_sha256=receipt["manifest_sha256"])


def test_render_yaml_rejects_autoscaling_even_with_one_manual_instance(tmp_path):
    _, render, receipt, _ = _material(tmp_path)
    text = render.read_text(encoding="utf-8").replace(
        "    numInstances: 1\n", "    numInstances: 1\n    scaling:\n      minInstances: 1\n      maxInstances: 2\n")
    render.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="one non-autoscaled service instance"):
        preflight.check_render_yaml(
            render, package_sha256=receipt["package_sha256"],
            manifest_sha256=receipt["manifest_sha256"])


def test_render_yaml_and_release_root_symlinks_are_rejected(monkeypatch, tmp_path):
    root, render, receipt, _ = _material(tmp_path)
    render_link = tmp_path / "render-link.yaml"
    render_link.symlink_to(render)
    with pytest.raises(ValueError, match="canonical regular file"):
        preflight.check_render_yaml(
            render_link, package_sha256=receipt["package_sha256"],
            manifest_sha256=receipt["manifest_sha256"])

    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    root_link = tmp_path / "release-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical directory"):
        preflight.preflight(
            release_root=root_link,
            expected_receipt_sha256=_sha(root / preflight.FILES["receipt"]),
            render_yaml=render,
        )


def test_missing_or_world_readable_release_material_is_rejected(
        monkeypatch, tmp_path):
    root, render, _, _ = _material(tmp_path)
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    (root / preflight.FILES["base64"]).unlink()
    with pytest.raises(ValueError, match="canonical regular file"):
        preflight.preflight(
            release_root=root,
            expected_receipt_sha256="0" * 64,
            render_yaml=render,
        )


@pytest.mark.parametrize("name", [
    "future-release.b64", "private.env", "study.sqlite3",
    "participant-export.json", "render-secret.json", "events.jsonl",
    "participant-summary.csv",
])
def test_public_deployment_directory_keeps_private_artifacts_ignored(name):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q",
         "output/deployment/" + name],
        cwd=preflight.ROOT, check=False,
    )
    assert result.returncode == 0, name


def test_git_allowlist_is_complete_sorted_and_public_source_only():
    path = preflight.ROOT / "docs/warehouse_r41_git_allowlist.txt"
    names = path.read_text(encoding="utf-8").splitlines()
    required = {
        "backend/training/warehouse_r41_corrected_partner_audit.py",
        "backend/training/warehouse_r41_failure_closeout.py",
        "docs/archive/warehouse_r41_energy_gate_audit_20260911.json",
        "docs/archive/warehouse_r41_energy_gate_audit_20260911.md",
        "docs/archive/warehouse_r41_fixed_yield_alias_diagnostic_20260911.json",
        "docs/archive/warehouse_r41_fixed_yield_alias_diagnostic_20260911.md",
        "docs/archive/warehouse_r41_training_trend_audit_20260911.json",
        "docs/archive/warehouse_r41_training_trend_audit_20260911.md",
        "docs/archive/warehouse_r41_persistence_audit_20260911.json",
        "docs/warehouse_r41_failure_closeout.md",
        "docs/warehouse_r41_persistence_audit.md",
        "docs/warehouse_r41_render_preflight.md",
        "scripts/preflight_warehouse_r41_render.py",
        "scripts/build_warehouse_r41_failure_closeout.py",
        "tests/test_warehouse_r41_corrected_partner_audit.py",
        "tests/test_warehouse_r41_energy_gate_semantics.py",
        "tests/test_warehouse_r41_failure_closeout.py",
        "tests/test_warehouse_r41_participant_information_boundary.py",
        "tests/test_warehouse_r41_partner_alias_diagnostic.py",
        "tests/test_warehouse_r41_render_preflight.py",
    }
    assert names == sorted(set(names))
    assert required <= set(names)
    assert all((preflight.ROOT / name).is_file() for name in names)
    assert [name for name in names if name.startswith("output/")] == [
        "output/.gitignore"
    ]
    assert not any(Path(name).suffix.lower() in {
        ".b64", ".env", ".key", ".pem", ".sqlite", ".sqlite3", ".db",
        ".jsonl", ".csv",
    } for name in names)
    for name in names:
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", name],
            cwd=preflight.ROOT, check=False,
        )
        assert ignored.returncode == 1, name
