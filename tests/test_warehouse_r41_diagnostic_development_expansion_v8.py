from collections import Counter
import inspect
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as subject


def _candidate(family: str, index: int) -> dict:
    # A real SHA-shaped identity keeps this fixture inside the production schema.
    fingerprint = subject.digest({"family": family, "index": index})
    return {
        "id": f"candidate-{family}-{index}",
        "split": "play_candidates",
        "family_id": family,
        "fingerprint": fingerprint,
        "seed": 1_000_000 + subject.FAMILY_IDS.index(family) * 10_000 + index,
    }


def _manifest(per_family: int = 80) -> dict:
    return {
        "candidate_batches": [[
            _candidate(family, index)
            for family in subject.FAMILY_IDS
            for index in range(per_family)
        ]]
    }


def test_contract_freezes_balanced_program_blind_fit_and_validation_splits():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-development-expansion.v8"
    assert sum(contract["fit_family_quotas"].values()) == 128
    assert sum(contract["validation_family_quotas"].values()) == 64
    assert contract["selection_sequence"] == [
        "development_validation", "fit_supplement"
    ]
    assert contract["validation_frozen_before_fit_selection"] is True
    assert contract["program_access"] is False
    assert contract["program_predictions_access"] is False
    assert contract["actor_logits_access"] is False
    assert contract["actor_hidden_state_access"] is False
    assert contract["participant_data_access"] is False
    assert contract["final_audit_rows_access"] is False
    assert contract["final_labels_used_for_selection"] is False
    assert contract["runtime_action_override"] is False


def test_producer_source_closure_has_no_program_or_final_audit_source():
    sources = subject.producer_sources()
    assert sources
    assert all("explanation_audit" not in path for path in sources)
    assert all("model_tree" not in path for path in sources)
    assert all("online_explanation" not in path for path in sources)
    parameters = set(inspect.signature(subject.build).parameters)
    assert not any("program" in name or "audit" in name or "label" in name
                   for name in parameters)


def test_selection_is_balanced_disjoint_and_uses_frozen_workload_indexes(monkeypatch):
    calls = []

    def fake_screen(scene, *, split, scene_index, actor, train_count=128):
        calls.append((scene["fingerprint"], split, scene_index, train_count, actor))
        return {
            "passed": True,
            "split": split,
            "scene_index": scene_index,
            "receipt_sha256": subject.digest({
                "fingerprint": scene["fingerprint"],
                "split": split,
                "scene_index": scene_index,
                "train_count": train_count,
            }),
        }

    monkeypatch.setattr(subject, "screen_scene", fake_screen)
    manifest = _manifest()
    # Exclude several high-ranked rows without depending on their manifest order.
    first_family = subject.FAMILY_IDS[0]
    ordered = subject._ordered_candidates(
        manifest, first_family, salt=subject.VALIDATION_ORDER_SALT
    )
    excluded = {row["fingerprint"] for row in ordered[:3]}
    excluded_seeds = {row["seed"] for row in ordered[3:5]}

    fit, validation, trace = subject._select(
        manifest=manifest, actor="actor",
        excluded_fingerprints=set(excluded),
        excluded_seeds=set(excluded_seeds),
    )

    assert len(fit) == subject.FIT_SUPPLEMENT_SCENES == 128
    assert len(validation) == subject.VALIDATION_SCENES == 64
    assert Counter(row["family_id"] for row in fit) == Counter(
        subject.FIT_FAMILY_QUOTAS
    )
    assert Counter(row["family_id"] for row in validation) == Counter(
        subject.VALIDATION_FAMILY_QUOTAS
    )
    fit_fp = {row["fingerprint"] for row in fit}
    validation_fp = {row["fingerprint"] for row in validation}
    assert not fit_fp & validation_fp
    assert not (fit_fp | validation_fp) & excluded
    assert not {row["seed"] for row in (*fit, *validation)} & excluded_seeds
    assert len(trace["development_validation"]) >= len(validation)
    assert len(trace["fit_supplement"]) >= len(fit)

    validation_calls = [row for row in calls if row[1] == "conflict_validation"]
    fit_calls = [row for row in calls if row[1] == "train"]
    assert [row[2] for row in validation_calls] == list(range(64))
    assert {row[3] for row in validation_calls} == {
        subject.TOTAL_DEVELOPMENT_FIT_SCENES
    }
    assert [row[2] for row in fit_calls] == list(range(192, 320))
    assert all(row[4] == "actor" for row in calls)
    assert max(index for _, split, index, _, _ in calls
               if split == "conflict_validation") == 63


def test_selection_rejects_a_family_when_public_workload_cannot_fill_quota(monkeypatch):
    monkeypatch.setattr(
        subject,
        "screen_scene",
        lambda *args, **kwargs: {"passed": False},
    )
    with pytest.raises(RuntimeError, match="family quota unavailable"):
        subject._select(
            manifest=_manifest(per_family=40), actor=object(),
            excluded_fingerprints=set(), excluded_seeds=set(),
        )


def test_scene_identity_validation_rejects_duplicate_seed():
    rows = [_candidate(subject.FAMILY_IDS[0], 0), _candidate(subject.FAMILY_IDS[0], 1)]
    rows[1]["seed"] = rows[0]["seed"]
    with pytest.raises(ValueError, match="scene identity differs"):
        subject._scene_identities(rows, expected_count=2, label="fixture")


def test_atomic_output_is_exclusive_and_leaves_no_partial_destination(tmp_path: Path):
    output = tmp_path / "registry"
    registry = {"value": 1}
    report = {"value": 2}
    subject._atomic_output(output, registry, report)
    assert (output / "development_expansion.json").is_file()
    assert (output / "report.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        subject._atomic_output(output, registry, report)
    assert not list(tmp_path.glob(".registry.tmp-*"))
    assert not (tmp_path / ".registry.lock").exists()


def test_atomic_output_removes_temporary_directory_after_write_failure(
    tmp_path: Path, monkeypatch,
):
    original = subject._write_exclusive
    calls = 0

    def fail_second(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return original(path, value)

    monkeypatch.setattr(subject, "_write_exclusive", fail_second)
    with pytest.raises(RuntimeError, match="injected failure"):
        subject._atomic_output(
            tmp_path / "registry", {"value": 1}, {"value": 2}
        )
    assert not (tmp_path / "registry").exists()
    assert not list(tmp_path.glob(".registry.tmp-*"))
    assert not (tmp_path / ".registry.lock").exists()


def test_build_rejects_wrong_retired_registry_count_before_selection(tmp_path: Path):
    placeholder = tmp_path / "placeholder"
    placeholder.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Exactly two retired"):
        subject.build(
            actor_path=placeholder,
            manifest_path=placeholder,
            designation_path=placeholder,
            selected_scenes_path=placeholder,
            previous_development_path=placeholder,
            retired_holdout_paths=[],
            output=tmp_path / "out",
        )
