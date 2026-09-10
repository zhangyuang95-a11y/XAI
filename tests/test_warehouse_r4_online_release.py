from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui import warehouse_alignment_r4_online_release as release


HEX = "a" * 64


def selection():
    def scene(index):
        return {
            "id": f"scene_{index}", "seed": 900_000 + index,
            "fingerprint": f"{index + 1:064x}",
            "task_signature": f"{index + 11:064x}",
            "observed_state_signature": f"{index + 21:064x}",
            "snapshot": {"state": {"frame": 0, "index": index}},
        }

    rows = [scene(index) for index in range(7)]
    return {
        "version": release.production_admission.SCENE_VERSION,
        "actor_sha256": HEX,
        "source_scenario_manifest_sha256": "b" * 64,
        "practice": rows[0], "X": rows[1:4], "Y": rows[4:7],
        "pairs": [[rows[index]["id"], rows[index + 3]["id"]]
                  for index in range(1, 4)],
        "balance": {}, "selection_score": 0.0,
    }


def test_scene_selection_requires_exact_unique_physics_and_pairing():
    value = selection()
    assert len(release._play_scenes(value)) == 7

    duplicate = deepcopy(value)
    duplicate["Y"][2]["seed"] = duplicate["X"][0]["seed"]
    with pytest.raises(ValueError, match="distinct"):
        release._play_scenes(duplicate)

    wrong_pair = deepcopy(value)
    wrong_pair["pairs"][0].reverse()
    with pytest.raises(ValueError, match="pairing"):
        release._play_scenes(wrong_pair)

    extra = deepcopy(value)
    extra["unbound"] = True
    with pytest.raises(ValueError, match="Exact"):
        release._play_scenes(extra)


def test_json_reader_rejects_duplicate_fields_and_nonfinite_values(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"version":1,"version":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        release._read_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        release._read_json(nonfinite)


def test_release_rejects_credential_like_material():
    release._reject_secret_material({"ordinary": {"value": "safe"}}, "fixture")
    for value in ({"deepseek_api_key": "redacted"},
                  {"nested": ["postgresql://user:password@example/db"]},
                  {"authorization": "Bearer abcdefghijklmnop"}):
        with pytest.raises(ValueError, match="Credential-like"):
            release._reject_secret_material(value, "fixture")


def test_release_source_binding_includes_runtime_physics_closure():
    sources = release.release_sources()
    for path in ("backend/training/warehouse_r4_release_source_closure.txt",
                 "backend/training/warehouse_family_stable_run.py",
                 "backend/training/warehouse_family_alignment_trainer.py",
                 "core/__init__.py", "core/program.py", "env/__init__.py",
                 "env/warehouse_native/environment.py",
                 "env/warehouse/environment.py", "env/warehouse/navigation.py",
                 "env/warehouse/transition_outcome.py",
                 "backend/training/warehouse_r4_final_rcpd.py"):
        assert path in sources
    # These files are independently bound by the portable online source group.
    assert "backend/warehouse_alignment_online_runtime.py" not in sources


def test_release_source_manifest_matches_static_local_import_closure():
    root = release.ROOT
    closure = release.production_admission.local_source_hashes(tuple(
        root / name for name in (*release.SOURCE_CLOSURE_SEEDS,
                                  *release.SOURCE_CLOSURE_ASSETS)))
    expected = set(closure) - release.PORTABLE_DELEGATED_SOURCES
    actual = release.SOURCE_CLOSURE_MANIFEST.read_text(encoding="utf-8").splitlines()
    assert actual == sorted(expected)


def test_admission_paths_are_canonical_regular_files(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    monkeypatch.setattr(release, "ROOT", root)
    report = root / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    assert release._inside_root("report.json") == report
    for unsafe in ("../report.json", "./report.json", str(report)):
        with pytest.raises(ValueError):
            release._inside_root(unsafe)

    alias = root / "alias.json"
    alias.symlink_to(report)
    with pytest.raises(ValueError, match="unsafe"):
        release._inside_root("alias.json")


def test_release_component_path_rejects_links_and_external_files(monkeypatch, tmp_path):
    root = (tmp_path / "repo").resolve(); root.mkdir()
    monkeypatch.setattr(release, "ROOT", root)
    component = root / "component.json"; component.write_text("{}\n")
    assert release._component_path(component, "component") == component
    alias = root / "alias.json"; alias.symlink_to(component)
    with pytest.raises(ValueError, match="canonical regular"):
        release._component_path(alias, "component")
    outside = tmp_path / "outside.json"; outside.write_text("{}\n")
    with pytest.raises(ValueError, match="canonical regular"):
        release._component_path(outside, "component")


def test_failed_admission_cannot_leave_a_package(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    monkeypatch.setattr(release, "ROOT", root)
    paths = {}
    for name in ("actor", "protocol", "program", "selected", "question",
                 "scenarios", "admission"):
        path = root / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    selected = selection()
    protocol, question, admission = {}, {"source_bank_signature": "f" * 64}, {}
    payloads = {
        paths["protocol"]: protocol, paths["program"]: {},
        paths["selected"]: selected, paths["question"]: question,
        paths["admission"]: admission,
    }
    monkeypatch.setattr(release, "_read_json", lambda path: payloads[Path(path)])
    actor = SimpleNamespace(metadata={}, artifact_sha256=HEX)
    runtime = SimpleNamespace(
        actor=actor, actor_sha256=HEX, protocol_sha256="b" * 64,
        signature="c" * 64,
    )
    explainer = SimpleNamespace(
        program_sha256="d" * 64, signature="e" * 64,
        _assert_current=lambda current: None,
    )
    monkeypatch.setattr(release, "OnlineAlignmentRuntime",
                        lambda *args, **kwargs: runtime)
    monkeypatch.setattr(release, "OnlineAlignmentExplainer",
                        lambda *args, **kwargs: explainer)
    monkeypatch.setattr(release.warehouse_r4_question_bank, "validate_payload",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_validate_scene_runtime",
                        lambda *args, **kwargs: {})
    monkeypatch.setattr(release.portable, "_validate_question_projection",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(release, "_validate_admission",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Exact complete r4 PPO budget ledger required")))
    writes = []
    monkeypatch.setattr(release.portable, "_write_new",
                        lambda *args, **kwargs: writes.append(args))
    package, encoded = root / "release.zip", root / "release.b64"
    with pytest.raises(ValueError, match="complete r4 PPO budget"):
        release.build(
            actor_path=paths["actor"], protocol_path=paths["protocol"],
            program_path=paths["program"], selected_scenes_path=paths["selected"],
            question_bank_path=paths["question"], scenarios_path=paths["scenarios"],
            admission_path=paths["admission"],
            expected_admission_sha256=release.file_hash(paths["admission"]),
            output_package=package, output_base64=encoded,
        )
    assert writes == []
    assert not package.exists() and not encoded.exists()


def test_admission_is_exact_and_hash_bound(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    monkeypatch.setattr(release, "ROOT", root)
    source = root / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    sources = {"source.py": release.file_hash(source)}
    monkeypatch.setattr(release, "release_sources", lambda: sources)
    report_paths = {}
    for name in release.production_admission.REPORT_NAMES:
        report = root / f"{name}.json"
        report.write_text("{}\n", encoding="utf-8")
        report_paths[name] = report
    runtime = SimpleNamespace(actor_sha256=HEX, protocol_sha256="b" * 64,
                              signature="c" * 64)
    explainer = SimpleNamespace(program_sha256="d" * 64, signature="e" * 64)
    selected = selection()
    question = {"source_bank_signature": "f" * 64}
    admission_path = root / "admission.json"
    admission = {
        "version": release.ADMISSION_VERSION,
        "status": "local_pilot_technically_verified",
        "formal_ready": False,
        "human_explanation_effect_validated": False,
        "gates": {name: True for name in release.REQUIRED_GATES},
        "bindings": {
            "actor_sha256": runtime.actor_sha256,
            "protocol_sha256": runtime.protocol_sha256,
            "runtime_signature": runtime.signature,
            "program_sha256": explainer.program_sha256,
            "explainer_signature": explainer.signature,
            "selected_scenes_sha256": release.digest(selected),
            "question_bank_sha256": release.digest(question),
            "question_bank_signature": question["source_bank_signature"],
        },
        "reports": {name: {"path": path.name, "sha256": release.file_hash(path)}
                    for name, path in report_paths.items()},
        "sources": sources,
        "self_path": "admission.json",
    }
    admission_path.write_text(json.dumps(admission), encoding="utf-8")
    expected = release.file_hash(admission_path)
    monkeypatch.setattr(release.production_admission, "component_context",
                        lambda **kwargs: {"synthetic": "context"})
    calls = []
    def validate(**kwargs):
        calls.append(kwargs)
        return {name: True for name in release.REQUIRED_GATES}
    monkeypatch.setattr(release.production_admission, "validate_report_set", validate)
    assert release._validate_admission(
        admission, admission_path=admission_path,
        admission_sha256=expected, runtime=runtime, explainer=explainer,
        selection=selected, question=question,
        selected_scenes_path=root / "selected.json",
        scenarios_path=report_paths["validation_scenarios"],
    ) == sources
    assert len(calls) == 1

    changed = deepcopy(admission)
    changed["extra"] = True
    with pytest.raises(ValueError, match="Exact"):
        release._validate_admission(
            changed, admission_path=admission_path,
            admission_sha256=expected, runtime=runtime, explainer=explainer,
            selection=selected, question=question,
            selected_scenes_path=root / "selected.json",
            scenarios_path=report_paths["validation_scenarios"],
        )
