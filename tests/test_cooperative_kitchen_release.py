"""Release hash-binding tests use fabricated local metadata, never publishable reports."""
import json
from pathlib import Path

from backend.cooperative_kitchen.artifacts import DEFAULT_OUTPUT, REQUIRED_GATES, file_hash, load_release, runtime_hash
from backend.cooperative_kitchen.llm import KitchenLLMClient


def metadata_fixture(tmp_path, monkeypatch,provider="qwen"):
    monkeypatch.setenv("DATABASE_URL", "postgresql://fixture-only")
    monkeypatch.setenv("KITCHEN_LLM_PROVIDER",provider)
    for name in ("KITCHEN_LLM_MODEL","KITCHEN_LLM_BASE_URL","KITCHEN_QWEN_MODEL","KITCHEN_QWEN_BASE_URL"):
        monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY" if provider=="deepseek" else "DASHSCOPE_API_KEY", "fixture-only")
    monkeypatch.delenv("DASHSCOPE_API_KEY" if provider=="deepseek" else "DEEPSEEK_API_KEY",raising=False)
    monkeypatch.setenv("KITCHEN_ADMIN_KEY", "fixture-only")
    objects = {"actor": {"test_only": True}, "program": {"tree": "fixture"}, "scenarios": {"pairs": []}, "questionnaire": {"items": []}}
    entries = {}
    def put(key, value):
        target = tmp_path / (key + ".json"); target.write_text(json.dumps(value))
        entries[key] = {"path": target.name, "sha256": file_hash(target)}
    for key, value in objects.items(): put(key, value)
    code = runtime_hash(); config = KitchenLLMClient().config
    for name in REQUIRED_GATES:
        put(name + "_report", {"passed": True, "mode": "real_remote", "actor_sha256": entries["actor"]["sha256"],
                              "program_sha256": entries["program"]["sha256"], "scenarios_sha256": entries["scenarios"]["sha256"],
                              "questionnaire_sha256": entries["questionnaire"]["sha256"], "runtime_sha256": code, "qa_configuration": config,
                              "fixture_only": True})
    manifest = {"status": "pilot_ready", "runtime_sha256": code, "qa_configuration": config, "artifacts": entries}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_same_actor_does_not_revalidate_swapped_program(tmp_path, monkeypatch):
    manifest = metadata_fixture(tmp_path, monkeypatch)
    assert load_release(tmp_path)["study_ready"]  # metadata contract only; no server constructed
    target = tmp_path / "program.json"; target.write_text('{"changed":true}')
    manifest["artifacts"]["program"]["sha256"] = file_hash(target)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = load_release(tmp_path)
    assert not result["study_ready"]
    assert "extraction_artifact_binding" in result["missing_configuration"]
    assert "qa_artifact_binding" in result["missing_configuration"]


def test_new_provider_model_config_cannot_reuse_previous_acceptance(tmp_path, monkeypatch):
    manifest = metadata_fixture(tmp_path, monkeypatch, provider="deepseek")
    monkeypatch.setenv("KITCHEN_LLM_MODEL", "deepseek-v4-pro")
    manifest["qa_configuration"] = KitchenLLMClient().config
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = load_release(tmp_path)
    assert not result["study_ready"]
    assert "qa_configuration_binding" in result["missing_configuration"]
    assert "remote_load_configuration_binding" in result["missing_configuration"]


def test_artifact_must_stay_inside_release_directory(tmp_path, monkeypatch):
    manifest = metadata_fixture(tmp_path, monkeypatch)
    manifest["artifacts"]["actor"]["path"] = "../outside.npz"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = load_release(tmp_path)
    assert not result["study_ready"] and "actor_hash" in result["missing_configuration"]


def test_candidate_and_missing_cloud_configuration_remain_closed(tmp_path, monkeypatch):
    manifest = metadata_fixture(tmp_path, monkeypatch)
    manifest["status"] = "candidate"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    assert not load_release(tmp_path)["study_ready"]


def test_deepseek_uses_its_own_key_and_records_alias_semantics(tmp_path,monkeypatch):
    metadata_fixture(tmp_path,monkeypatch,provider="deepseek")
    result=load_release(tmp_path)
    assert not result["study_ready"]  # a rolling alias cannot be a formal freeze
    assert result["qa_required_key_env"]=="DEEPSEEK_API_KEY" and result["qa_configured"]
    assert "DASHSCOPE_API_KEY" not in result["missing_configuration"]
    assert result["qa_configuration"]["provider"]=="deepseek"
    assert result["qa_configuration"]["model_version_pinned"] is False
    assert "qa_model_snapshot_unpinned" in result["missing_configuration"]
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    result=load_release(tmp_path)
    assert not result["study_ready"] and "DEEPSEEK_API_KEY" in result["missing_configuration"]
    assert "DASHSCOPE_API_KEY" not in result["missing_configuration"]


def test_switch_from_frozen_qwen_changes_version_without_rewriting_manifest(tmp_path,monkeypatch):
    metadata_fixture(tmp_path,monkeypatch,provider="qwen")
    original=(tmp_path/"manifest.json").read_bytes()
    previous=load_release(tmp_path)
    assert previous["study_ready"] and previous["qa_required_key_env"]=="DASHSCOPE_API_KEY"
    monkeypatch.setenv("KITCHEN_LLM_PROVIDER","deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY","fixture-only")
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    current=load_release(tmp_path)
    assert not current["study_ready"]
    assert "qa_configuration_version" in current["missing_configuration"]
    assert "qa_configuration_binding" in current["missing_configuration"]
    assert "remote_load_configuration_binding" in current["missing_configuration"]
    assert current["versions"]["manifest"]==previous["versions"]["manifest"]
    assert current["versions"]["qa_configuration_sha256"]!=previous["versions"]["qa_configuration_sha256"]
    assert (tmp_path/"manifest.json").read_bytes()==original


def test_provider_credentials_cannot_unlock_a_failed_extraction_gate(tmp_path,monkeypatch):
    manifest=metadata_fixture(tmp_path,monkeypatch)
    target=tmp_path/"extraction_report.json"
    report=json.loads(target.read_text());report["passed"]=False;target.write_text(json.dumps(report))
    manifest["artifacts"]["extraction_report"]["sha256"]=file_hash(target)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    current=load_release(tmp_path)
    assert current["qa_configured"] and not current["study_ready"]
    assert "extraction_gate" in current["missing_configuration"]


def test_user_id_pilot_has_explicit_v3_output_ui_and_protocol(tmp_path,monkeypatch):
    metadata_fixture(tmp_path,monkeypatch)
    current=load_release(tmp_path)
    assert DEFAULT_OUTPUT.name=="v3-id-pilot"
    assert current["versions"]["ui"]=="cooperative_kitchen_web_v3_id_pilot"
    assert current["versions"]["protocol"]=="cooperative_kitchen_user_id_pilot_v3"
    assert current["versions"]["runtime_sha256"]==runtime_hash()
