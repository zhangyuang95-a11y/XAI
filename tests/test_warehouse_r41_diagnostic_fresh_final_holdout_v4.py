import ast
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as subject


ROOT = Path(subject.__file__).resolve().parents[2]


def test_v4_contract_is_program_blind_and_claim_burns_v3_internally():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-fresh-final-holdout.v4"
    assert sum(contract["family_quotas"].values()) == subject.TOTAL_SCENES == 64
    assert contract["selection_salt_commitment"] == subject.HOLDOUT_SALT_COMMITMENT
    assert contract["program_access"] is False
    assert contract["program_predictions_access"] is False
    assert contract["actor_logits_access"] is False
    assert contract["final_labels_used_for_selection"] is False
    assert contract["external_retired_final_registries"] == list(
        subject.EXTERNAL_RETIRED_VERSIONS)
    assert contract["external_retired_final_sha256"] == dict(
        sorted(subject.EXPECTED_RETIRED_HOLDOUT_SHA256.items()))
    assert contract["internal_v3_exclusion_registry"] == subject.V3_TOMBSTONE_VERSION
    assert subject.RETIRED_VERSIONS == subject.EXTERNAL_RETIRED_VERSIONS


def test_holdout_has_no_program_import_cli_or_public_writer_export():
    tree = ast.parse(Path(subject.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("public_tree", "boosted_tree", "explanation_audit", "diagnostic_rcpd")
    assert not any(any(fragment in module for fragment in forbidden)
                   for module in imported)
    assert not hasattr(subject, "main")
    assert "build" not in subject.__all__


def test_holdout_source_closure_binds_transitive_selection_and_physics():
    sources = subject.producer_sources()
    assert {
        "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        "backend/training/warehouse_r41_diagnostic_fresh_final_holdout_v3.py",
        "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
        "backend/training/warehouse_native_common.py",
        "env/warehouse/transition_outcome.py",
        "env/warehouse_native/environment.py",
    }.issubset(sources)
    assert "core/__init__.py" in sources
    assert not any("admission" in path or "release" in path for path in sources)


def _rows(path, observations, fingerprints, *, corrupt=False):
    observations = np.asarray(observations, dtype=np.float32)
    hashes = [sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest()
              for row in observations]
    if corrupt:
        hashes[0] = "0" * 64
    np.savez_compressed(
        path,
        observations=observations,
        observation_hashes=np.asarray(hashes, dtype="S64"),
        scene_fingerprints=np.asarray(fingerprints, dtype="S64"),
    )
    return set(hashes)


def test_development_npz_hashes_are_recomputed_and_scene_coverage_is_required(tmp_path):
    fingerprint = "a" * 64
    path = tmp_path / "rows.npz"
    expected = _rows(path, [np.arange(197)], [fingerprint])
    hashes, scenes, bindings = subject._npz_development_observations(
        [path], required_fingerprints={fingerprint})
    assert hashes == expected
    assert scenes == {fingerprint}
    assert bindings["rows.npz"] == subject.file_hash(path)

    bad = tmp_path / "bad.npz"
    _rows(bad, [np.arange(197)], [fingerprint], corrupt=True)
    with pytest.raises(ValueError, match="hash differs"):
        subject._npz_development_observations(
            [bad], required_fingerprints={fingerprint})
    with pytest.raises(ValueError, match="cover every"):
        subject._npz_development_observations(
            [path], required_fingerprints={"b" * 64})


def test_selection_rejects_scene_seed_and_observation_overlap(monkeypatch):
    candidates = {}
    for family_index, family in enumerate(subject.FAMILY_IDS):
        rows = []
        for index in range(subject.FAMILY_QUOTAS[family] + 3):
            rows.append({
                "fingerprint": sha256(f"{family}:{index}".encode()).hexdigest(),
                "seed": family_index * 100 + index,
                "family_id": family,
            })
        candidates[family] = rows
    excluded_fp = {candidates[subject.FAMILY_IDS[0]][0]["fingerprint"]}
    excluded_seed = {candidates[subject.FAMILY_IDS[0]][1]["seed"]}
    forbidden = {"f" * 64}

    monkeypatch.setattr(
        subject, "_ordered_candidates",
        lambda _manifest, family, _salt: candidates[family])
    monkeypatch.setattr(subject, "screen_scene",
                        lambda *args, **kwargs: {"passed": True})

    def observations(_runtime, scene, _index):
        if scene is candidates[subject.FAMILY_IDS[0]][2]:
            return {"f" * 64}
        return {sha256((scene["fingerprint"] + ":obs").encode()).hexdigest()}

    monkeypatch.setattr(subject, "_exact_final_workload_observations", observations)
    scenes, trace, stats, accepted = subject._select(
        runtime=object(), actor=object(), manifest={}, selection_salt=b"secret",
        excluded_fingerprints=set(excluded_fp), excluded_seeds=set(excluded_seed),
        forbidden_observation_hashes=set(forbidden),
    )
    assert len(scenes) == 64
    assert not ({row["fingerprint"] for row in scenes} & excluded_fp)
    assert not ({row["seed"] for row in scenes} & excluded_seed)
    assert not (accepted & forbidden)
    assert stats["public_observation_overlap"] == 0
    assert {row["rejection_reason"] for row in trace[:3]} == {
        "excluded_scene_or_seed", "excluded_public_observation_overlap"
    }


def test_private_salt_uses_domain_separated_commitment(tmp_path, monkeypatch):
    raw = bytes(range(32))
    path = tmp_path / "salt.bin"
    path.write_bytes(raw)
    commitment = sha256(subject.HOLDOUT_SALT_DOMAIN + raw).hexdigest()
    monkeypatch.setattr(subject, "HOLDOUT_SALT_COMMITMENT", commitment)
    assert subject._read_committed_salt(path) == raw
    path.write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="commitment"):
        subject._read_committed_salt(path)


def test_claim_phase_partial_payload_never_gets_final_name(tmp_path, monkeypatch):
    marker = tmp_path / "holdout_started.json"

    def partial_then_fail(stream, raw):
        stream.write(raw[: max(1, len(raw) // 2)])
        stream.flush()
        raise OSError("synthetic phase payload failure")

    monkeypatch.setattr(subject, "_write_phase_payload", partial_then_fail)
    with pytest.raises(OSError, match="payload failure"):
        subject._claim_marker(tmp_path, marker.name, {
            "status": "started_no_retry", "campaign_key": "a" * 64,
        })
    assert not marker.exists()
    assert not (tmp_path / ".holdout_started.json.partial").exists()


def test_direct_writer_rejects_before_salt_or_final_input_access(monkeypatch):
    calls = []

    def reject(*args, **kwargs):
        calls.append("claim")
        raise ValueError("claim rejected")

    monkeypatch.setattr(subject, "_claim_receipt", reject)
    monkeypatch.setattr(
        subject, "_read_committed_salt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("salt was read before claim")))
    with pytest.raises(ValueError, match="claim rejected"):
        subject.build(
            actor_path="missing-actor", protocol_path="missing-protocol",
            manifest_path="missing-manifest", designation_path="missing-designation",
            selected_scenes_path="missing-selection",
            development_registry_paths=[], development_rows_paths=[],
            legacy_v3_rows_path="missing-v3-rows", retired_holdout_paths=[],
            output="missing-output", claim_receipt_path="missing-claim",
            expected_claim_sha256="a" * 64, expected_campaign_key="b" * 64,
            expected_candidate_identity_sha256="c" * 64,
            selection_salt_path="missing-salt",
        )
    assert calls == ["claim"]


def test_forged_ledger_claim_without_permanent_anchor_is_rejected(
        tmp_path, monkeypatch):
    monkeypatch.setattr(subject, "HOLDOUT_SALT_COMMITMENT", "9" * 64)
    ledger = tmp_path / "ledger"
    anchor = tmp_path / "outside" / "permanent.anchor"
    monkeypatch.setattr(subject, "DEFAULT_LEDGER_ROOT", ledger)
    monkeypatch.setattr(subject, "DEFAULT_PERMANENT_ANCHOR", anchor)
    identity = subject._campaign_identity()
    key = subject.digest(identity)
    campaign = ledger / key
    campaign.mkdir(parents=True)
    receipt = {
        "version": subject.FINAL_ONCE_VERSION, "key": key,
        "campaign_key": key, "candidate_identity_sha256": subject.digest(identity),
        "status": "started_irrevocable_no_retry", "identity": identity,
        "permanent_anchor_path": str(anchor),
        "permanent_anchor_sha256": "a" * 64,
        "program_evaluation_started": False,
    }
    claim = campaign / "attempt_started.json"
    claim.write_text(subject.canonical(receipt) + "\n", encoding="utf-8")
    (campaign / "candidate_authenticated.json").write_text(subject.canonical({
        "status": "passed_strict_reader_and_refit", "campaign_key": key,
        "candidate_identity_sha256": subject.digest(identity),
        "attempt_started_sha256": subject.file_hash(claim),
        "require_passed": True, "refit": True,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="permanent final-once anchor"):
        subject._claim_receipt(
            claim, expected_claim_sha256=subject.file_hash(claim),
            expected_campaign_key=key,
            expected_candidate_identity_sha256=subject.digest(identity))


def test_retired_registry_bytes_are_fixed_even_if_replacement_is_self_consistent(
        tmp_path, monkeypatch):
    retired_paths = [
        ROOT / "output/warehouse_native/r41_diagnostic_fresh_final_holdout_v1_20260912/holdout.json",
        ROOT / "output/warehouse_native/r41_diagnostic_fresh_final_holdout_v2_20260912/holdout.json",
    ]
    if not all(path.is_file() for path in retired_paths):
        pytest.skip("retired local evidence is not present")
    registries, bindings = subject._retired_registries(retired_paths)
    assert [row["version"] for row in registries] == list(
        subject.EXTERNAL_RETIRED_VERSIONS)
    assert {version: value["file_sha256"] for version, value in bindings.items()} == (
        subject.EXPECTED_RETIRED_HOLDOUT_SHA256)

    replacement = json.loads(retired_paths[0].read_text(encoding="utf-8"))
    replacement["attacker_self_consistent_note"] = "replacement"
    replacement["content_sha256"] = subject.digest({
        key: value for key, value in replacement.items() if key != "content_sha256"
    })
    replaced_path = tmp_path / "holdout-v1-replaced.json"
    replaced_path.write_text(
        subject.canonical(replacement) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        subject._retired_registries([replaced_path, retired_paths[1]])


def test_expansion_must_bind_the_same_fixed_retired_entities():
    bindings = {
        version: {"file_sha256": file_sha, "content_sha256": str(index) * 64}
        for index, (version, file_sha) in enumerate(
            subject.EXPECTED_RETIRED_HOLDOUT_SHA256.items(), start=1)
    }
    subject._validate_retired_expansion_bindings(
        bindings, {"bindings": {"retired_holdouts": bindings}})
    replacement = {version: dict(value) for version, value in bindings.items()}
    replacement[subject.EXTERNAL_RETIRED_VERSIONS[0]]["file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fixed development expansion"):
        subject._validate_retired_expansion_bindings(
            replacement, {"bindings": {"retired_holdouts": replacement}})


def test_retired_trace_closure_excludes_rejected_and_accepted_rows(monkeypatch):
    rejected = {"fingerprint": "a" * 64, "seed": 1, "family_id": "family"}
    accepted = {"fingerprint": "b" * 64, "seed": 2, "family_id": "family"}
    manifest = {"candidate_batches": [[rejected, accepted]]}
    accepted_hashes = {"c" * 64}
    failed_receipt = {"passed": False, "reason": "screen"}
    passed_receipt = {"passed": True, "reason": None}
    registry = {
        "version": "retired",
        "scenes": [accepted],
        "selection_trace": [
            {"fingerprint": rejected["fingerprint"], "family_id": "family",
             "scene_index": 0, "accepted": False,
             "workload_receipt": failed_receipt},
            {"fingerprint": accepted["fingerprint"], "family_id": "family",
             "scene_index": 7, "accepted": True,
             "workload_receipt": passed_receipt,
             "public_observation_count": 1,
             "public_observations_sha256": subject.digest(sorted(accepted_hashes))},
        ],
    }
    seen_indexes = []

    def screen(scene, *, split, scene_index, actor):
        assert split == "final_test"
        return (failed_receipt if scene["fingerprint"] == rejected["fingerprint"]
                else passed_receipt)

    def observations(_runtime, scene, index):
        assert scene["fingerprint"] == accepted["fingerprint"]
        seen_indexes.append(index)
        return set(accepted_hashes)

    monkeypatch.setattr(subject, "screen_scene", screen)
    monkeypatch.setattr(subject, "_exact_final_workload_observations", observations)
    fingerprints, seeds, hashes, stats = subject._retired_exposure_closure(
        runtime=object(), actor=object(), manifest=manifest,
        registries=[registry])
    assert fingerprints == {"a" * 64, "b" * 64}
    assert seeds == {1, 2}
    assert hashes == accepted_hashes
    assert seen_indexes == [7]
    assert stats["retired"]["touched_trace_scenes"] == 2


def test_v3_tombstone_is_claim_bound_and_program_blind(monkeypatch):
    scene = {"fingerprint": "a" * 64, "seed": 7}
    monkeypatch.setattr(
        subject.legacy_v3, "_select",
        lambda **kwargs: ([scene], [{"fingerprint": scene["fingerprint"]}],
                          {"accepted": 1}))
    binding = {"campaign_key": "b" * 64, "attempt_started_sha256": "c" * 64}
    tombstone = subject._build_v3_tombstone(
        runtime=object(), actor=object(), manifest={}, selected={},
        legacy_development_hashes=set(), retired_registries=[],
        claim_binding=binding)
    assert tombstone["version"] == subject.V3_TOMBSTONE_VERSION
    assert tombstone["status"] == "burned_program_blind_exclusion"
    assert tombstone["claim_binding"] == binding
    assert tombstone["program_access"] is False
    assert tombstone["program_predictions_access"] is False
    assert tombstone["content_sha256"] == subject.digest({
        key: value for key, value in tombstone.items() if key != "content_sha256"
    })


def test_retired_v1_v2_trace_observation_receipts_replay_with_wait_three_branch():
    retired_paths = [
        ROOT / "output/warehouse_native/r41_diagnostic_fresh_final_holdout_v1_20260912/holdout.json",
        ROOT / "output/warehouse_native/r41_diagnostic_fresh_final_holdout_v2_20260912/holdout.json",
    ]
    required = [
        *retired_paths,
        ROOT / "output/warehouse_native/r41_active_2m_20260911/boundaries/step_2000000/actor.npz",
        ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json",
        ROOT / "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/manifest.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("retired local evidence is not present")
    actor_path, protocol_path, manifest_path = required[2:]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = dict(manifest)
    manifest_content = content.pop("content_sha256")
    runtime = subject.R41DiagnosticOnlineAlignmentRuntime(
        actor_path, training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=subject.file_hash(actor_path),
        expected_training_protocol_file_sha256=subject.file_hash(protocol_path),
        expected_training_protocol_content_sha256=subject.digest(protocol),
        expected_manifest_file_sha256=subject.file_hash(manifest_path),
        expected_manifest_content_sha256=manifest_content,
        expected_manifest_semantic_sha256=subject.digest(manifest),
    )
    candidates = subject._candidate_index(manifest)
    for path in retired_paths:
        retired = json.loads(path.read_text(encoding="utf-8"))
        row = next(item for item in retired["selection_trace"]
                   if item.get("workload_receipt", {}).get("passed") is True)
        hashes = subject._exact_final_workload_observations(
            runtime, candidates[row["fingerprint"]], row["scene_index"])
        expected_count = row.get(
            "public_observation_hash_count", row.get("public_observation_count"))
        expected_digest = row.get(
            "public_observation_hashes_sha256", row.get("public_observations_sha256"))
        assert len(hashes) == expected_count
        assert subject.digest(sorted(hashes)) == expected_digest
