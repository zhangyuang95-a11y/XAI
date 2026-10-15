from __future__ import annotations

from scripts import preflight_warehouse_r41_diagnostic_render as preflight


def _render(package="a" * 64, manifest="b" * 64):
    return f"""services:
  - type: web
    name: {preflight.SERVICE_NAME}
    runtime: python
    plan: free
    numInstances: 1
    startCommand: {preflight.START_COMMAND}
    healthCheckPath: /health
    autoDeployTrigger: off
    envVars:
      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256
        value: {package}
      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256
        value: {manifest}
      - key: WAREHOUSE_ONLINE_DATABASE
        value: /tmp/warehouse_r41_diagnostic.sqlite3
      - key: WAREHOUSE_STORAGE_MODE
        value: ephemeral
      - key: WAREHOUSE_PUBLIC_ORIGIN
        value: https://policylens-warehouse-study.onrender.com
"""


def test_diagnostic_render_contract_is_free_ephemeral_and_manual(tmp_path):
    path = tmp_path / "render.yaml"; path.write_text(_render())
    result = preflight.check_render_yaml(
        path, package_sha256="a" * 64, manifest_sha256="b" * 64)
    assert result["release_module"] == "ui.warehouse_alignment_r41_diagnostic_release"
    assert result["plan"] == "free"
    assert result["auto_deploy"] is False
    assert result["single_instance"] is True
    assert result["data_persistent"] is False


def test_diagnostic_render_rejects_persistence_paid_plan_and_old_loader(tmp_path):
    path = tmp_path / "render.yaml"
    for changed, expected in (
        (_render().replace("value: ephemeral", "value: persistent"),
         "stale or missing WAREHOUSE_STORAGE_MODE"),
        (_render().replace("plan: free", "plan: 0.5c-512mb"),
         "free ephemeral"),
        (_render().replace("ui.warehouse_alignment_r41_diagnostic_release",
                           "ui.warehouse_alignment_r41_online_release"),
         "explicitly start"),
    ):
        path.write_text(changed)
        try:
            preflight.check_render_yaml(
                path, package_sha256="a" * 64, manifest_sha256="b" * 64)
        except ValueError as error:
            assert expected in str(error)
        else:
            raise AssertionError("unsafe diagnostic Render configuration accepted")
