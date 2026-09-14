from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.training import warehouse_r41_diagnostic_study_materials_v10 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend import warehouse_r41_diagnostic_online_explanation_v9 as explanation
from scripts import build_warehouse_r41_diagnostic_study_materials_v10 as cli


def test_study_materials_authenticate_v13_candidate_lock():
    assert subject.outer_api.VERSION.startswith(
        "warehouse-r41-diagnostic-rcpd-v13-")


def _write(path: Path, value) -> Path:
    if isinstance(value, dict):
        path.write_text(canonical(value) + "\n", encoding="utf-8")
    else:
        path.write_bytes(value)
    return path


def _question_payload() -> dict:
    items = []
    for index in range(8):
        kind = "next_action" if index < 4 else "wait_three"
        snapshot = {"state": {"frame": index + 1}}
        evidence = ({"decision": "actor"} if kind == "next_action" else {
            "assumed_player_actions": ["WAIT", "WAIT", "WAIT"],
            "transitions": [{"step": step} for step in range(3)],
        })
        if kind == "wait_three":
            evidence["transitions_sha256"] = digest(evidence["transitions"])
        items.append({
            "id": f"item-{index}", "kind": kind,
            "scenario_id": f"scene-{index}", "frame": index + 1,
            "snapshot": snapshot, "snapshot_sha256": digest(snapshot),
            "prompt": {"zh": "问题", "en": "Question"},
            "answer": "WAIT",
            "options": [{"value": "WAIT", "label": {"zh": "等待", "en": "Wait"}}],
            "evidence": evidence,
        })
    return {"items": items}


def _synthetic_inputs(tmp_path: Path):
    paths = {
        "actor": _write(tmp_path / "actor.npz", b"synthetic actor"),
        "protocol": _write(tmp_path / "protocol.json", {"synthetic": True}),
        "source_manifest": _write(tmp_path / "manifest.json", {
            "splits": {"question_bank": [], "tutorial": [{"id": "tutorial"}]},
            "content_sha256": "0" * 64,
        }),
        "candidate_lock": _write(tmp_path / "lock.json", {"synthetic": True}),
        "combined_promoted_rows": _write(
            tmp_path / "combined-promoted-rows.npz", b"synthetic promoted rows"),
        "promotion_closeout": _write(
            tmp_path / "promotion-closeout.json", {"synthetic": True}),
        "program": _write(tmp_path / "program.json", {"synthetic": True}),
        "final_audit": _write(tmp_path / "audit.json", {"synthetic": True}),
    }
    return paths


def test_authentication_binds_actor_program_and_passed_final_audit(
        tmp_path, monkeypatch):
    paths = _synthetic_inputs(tmp_path)
    hashes = {name: file_hash(path) for name, path in paths.items()}
    public_contract = "7" * 64
    lock = {"content_sha256": "8" * 64}
    lock_bindings = {
        "actor_sha256": hashes["actor"],
        "protocol_sha256": hashes["protocol"],
        "runtime_manifest_sha256": hashes["source_manifest"],
        "program_sha256": hashes["program"],
        "combined_promoted_rows_sha256": hashes["combined_promoted_rows"],
        "promotion_closeout_sha256": hashes["promotion_closeout"],
        "public_feature_contract_sha256": public_contract,
    }
    final_bindings = {
        name: str(index) * 64
        for index, name in enumerate(sorted(subject._FINAL_BINDINGS), 1)
    }
    final_bindings.update({
        "candidate_lock_sha256": hashes["candidate_lock"],
        "actor_sha256": hashes["actor"],
        "program_sha256": hashes["program"],
        "public_feature_contract_sha256": public_contract,
    })
    audit = {"bindings": final_bindings, "content_sha256": "9" * 64}
    _write(paths["final_audit"], audit)
    hashes["final_audit"] = file_hash(paths["final_audit"])

    class Program:
        action_names = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
        base_feature_names = ("public",)
        relations = SimpleNamespace(contract=lambda: {"public": True})

        @classmethod
        def from_dict(cls, value):
            assert value == {"synthetic": True}
            return cls()

    runtime = SimpleNamespace(
        actor=SimpleNamespace(metadata={
            "actions": list(Program.action_names),
            "feature_names": list(Program.base_feature_names),
        }),
        verify_binding=lambda: "runtime",
    )
    lock_bindings["public_feature_contract_sha256"] = digest(
        Program.relations.contract())
    final_bindings["public_feature_contract_sha256"] = lock_bindings[
        "public_feature_contract_sha256"]
    audit["bindings"] = final_bindings
    _write(paths["final_audit"], audit)
    hashes["final_audit"] = file_hash(paths["final_audit"])

    monkeypatch.setattr(subject.outer_api, "_candidate_lock", lambda path, expected_sha256: (
        Path(path), lock, lock_bindings,
    ))
    closeout = {
        "content_sha256": "6" * 64,
        "combined_promoted_development": {
            "rows_sha256": hashes["combined_promoted_rows"],
            "rows_semantic_sha256": "5" * 64,
        },
    }
    closeout_calls = []
    monkeypatch.setattr(
        subject.promoted_closeout_api, "read_saved_closeout_public",
        lambda path, **kwargs: (
            closeout_calls.append((Path(path), kwargs)) or closeout))
    monkeypatch.setattr(subject, "R41DiagnosticPublicTreeProgramV9", Program)
    monkeypatch.setattr(subject, "_runtime", lambda *args: runtime)
    checked = []
    monkeypatch.setattr(subject.audit_api, "validate_report", lambda value, **kwargs: (
        checked.append((value, kwargs)) or value
    ))
    result = subject.authenticate_release_inputs(**{
        **paths,
        **{"expected_" + name + "_sha256": hashes[name] for name in paths},
        "permanent_promotion_closeout_registry": tmp_path,
    })
    assert result["runtime"] is runtime
    assert checked == [(audit, {
        "expected_bindings": final_bindings, "require_passed": True,
    })]
    assert closeout_calls == [(paths["promotion_closeout"], {
        "expected_closeout_sha256": hashes["promotion_closeout"],
        "permanent_closeout_registry": tmp_path,
    })]

    changed = deepcopy(audit)
    changed["bindings"]["program_sha256"] = "f" * 64
    _write(paths["final_audit"], changed)
    with pytest.raises(ValueError, match="Final audit does not bind"):
        subject.authenticate_release_inputs(**{
            **paths,
            **{"expected_" + name + "_sha256": file_hash(paths[name])
               for name in paths},
            "permanent_promotion_closeout_registry": tmp_path,
        })


def test_atomic_build_reuses_replayed_bilingual_bank_and_neutral_tutorial(
        tmp_path, monkeypatch):
    paths = _synthetic_inputs(tmp_path)
    hashes = {name + "_sha256": file_hash(path) for name, path in paths.items()}
    payload = _question_payload()
    report = {"replayed": True}
    tutorial = {
        "source": subject.tutorial_api.SOURCE,
        "uses_final_actor": False,
        "duration_ms": 380,
    }
    runtime = SimpleNamespace()
    manifest = {"splits": {"tutorial": [{"id": "tutorial"}]}}
    authenticated = {
        "paths": paths, "hashes": hashes, "runtime": runtime,
        "manifest": manifest, "lock": {"content_sha256": "a" * 64},
        "lock_bindings": {"public_feature_contract_sha256": "b" * 64},
        "program_payload": {"program": "synthetic"},
        "final_audit": {"content_sha256": "c" * 64},
        "promotion_closeout": {
            "content_sha256": "e" * 64,
            "combined_promoted_development": {"rows_semantic_sha256": "f" * 64},
        },
    }
    monkeypatch.setattr(subject, "authenticate_release_inputs",
                        lambda **kwargs: authenticated)

    def build_questions(_runtime, _manifest, *, output, manifest_file_sha256):
        output.mkdir()
        _write(output / subject.QUESTION_FILE, payload)
        _write(output / subject.QUESTION_REPORT, report)
        return report

    monkeypatch.setattr(subject.question_api, "build", build_questions)
    monkeypatch.setattr(subject.question_api, "validate_payload", lambda *args: {})
    monkeypatch.setattr(subject.question_api, "read_saved_report",
                        lambda *args, **kwargs: report)
    monkeypatch.setattr(subject.tutorial_api, "build_neutral_tutorial",
                        lambda *args, **kwargs: tutorial)
    monkeypatch.setattr(subject.tutorial_api, "bindings", lambda *args, **kwargs: {})
    monkeypatch.setattr(subject.tutorial_api, "validate_neutral_tutorial",
                        lambda *args, **kwargs: {"passed": True, "frame_count": 14})
    monkeypatch.setattr(subject, "producer_sources",
                        lambda: {"synthetic.py": "d" * 64})

    output = tmp_path / "frozen"
    receipt = subject.build(output=output, **{
        **paths,
        **{"expected_" + name + "_sha256": file_hash(path)
           for name, path in paths.items()},
        "permanent_promotion_closeout_registry": tmp_path,
    })
    assert receipt["status"] == subject.STATUS
    assert receipt["question_bank"] == {
        "bilingual_items": 8,
        "next_action_items_replayed": 4,
        "wait_three_items_replayed": 4,
        "independent_source_scenes": 8,
        "actual_actor_actions_independently_replayed": True,
        "counterfactuals_do_not_mutate_source_frames": True,
        "participant_projection_schema_unchanged": True,
    }
    assert receipt["tutorial"]["uses_final_actor"] is False
    assert receipt["information_boundary"]["final_rows_read"] is False
    assert receipt["information_boundary"]["combined_promoted_rows_read"] is False
    assert receipt["information_boundary"][
        "promotion_closeout_permanently_authenticated"] is True
    assert json.loads((output / subject.RECEIPT_FILE).read_text()) == receipt
    assert (output / subject.QUESTION_DIRECTORY / subject.QUESTION_FILE).is_file()
    assert (output / subject.TUTORIAL_FILE).is_file()


def test_question_contract_rejects_shortened_counterfactual():
    payload = _question_payload()
    payload["items"][4]["evidence"]["transitions"] = [{"step": 0}]
    payload["items"][4]["evidence"]["transitions_sha256"] = digest(
        payload["items"][4]["evidence"]["transitions"])
    with pytest.raises(ValueError, match="three-step"):
        subject._validate_question_contract(payload)


def test_existing_v9_access_boundary_keeps_answers_in_a_task1_only():
    base = {
        "condition": "A", "stage": "task1", "surface": "live",
        "request_kind": "new_question", "active_run_id": "run-1",
        "bound_run_id": "run-1", "selected_frame": 8,
        "available_frames": [0, 8], "round_closed": False,
    }
    assert explanation.explanation_access(base)["allowed"] is True
    for change in ({"condition": "B"}, {"stage": "task2"}):
        context = {**base, **change}
        assert explanation.explanation_access(context)["allowed"] is False


def test_cli_requires_hash_for_every_prerequisite():
    option_strings = {
        option for action in cli.parser()._actions for option in action.option_strings
    }
    for name in (
        "actor", "protocol", "source-manifest", "candidate-lock", "program",
        "final-audit", "combined-promoted-rows", "promotion-closeout",
    ):
        assert "--" + name in option_strings
        assert "--expected-" + name + "-sha256" in option_strings
    assert "--permanent-promotion-closeout-registry" in option_strings
