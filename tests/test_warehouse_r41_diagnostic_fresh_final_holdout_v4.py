import ast
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as subject
from backend.training import warehouse_r41_diagnostic_workload_screen as workload_screen
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from ui import warehouse_alignment_r41_tutorial as tutorial_base


ROOT = Path(subject.__file__).resolve().parents[2]


def _fresh_outer_registry_fixture():
    fit = [
        {
            "id": f"fit-{index}", "split": "fit_supplement",
            "seed": 10_000 + index,
            "fingerprint": sha256(f"fit:{index}".encode()).hexdigest(),
        }
        for index in range(subject.outer_split_api.FIT_SUPPLEMENT_SCENES)
    ]
    outer = []
    index = 0
    for family in subject.outer_split_api.FAMILY_IDS:
        for _ in range(subject.outer_split_api.FAMILY_QUOTAS[family]):
            outer.append({
                "id": f"outer-{index}", "split": "development_validation",
                "batch_index": index % 3, "family_id": family,
                "seed": 20_000 + index,
                "fingerprint": sha256(f"outer:{index}".encode()).hexdigest(),
            })
            index += 1
    identities = [
        {key: row[key] for key in (
            "batch_index", "family_id", "seed", "fingerprint")}
        for row in outer
    ]
    exposed = [sha256(f"old-outer:{index}".encode()).hexdigest()
               for index in range(subject.outer_split_api.OLD_OUTER_SCENE_COUNT)]
    sources = {"frozen/source.py": "a" * 64}
    registry = {
        "version": subject.outer_split_api.VERSION,
        "status": subject.outer_split_api.STATUS,
        "contract": subject.outer_split_api.contract(),
        "bindings": subject._expected_fresh_outer_bindings(),
        "fit_supplement": fit,
        "development_validation": outer,
        "selected_outer_identities": identities,
        "previously_exposed_validation_scene_fingerprints": exposed,
        "statistics": {
            "fresh_outer_scenes": len(outer),
            "fresh_outer_family_counts": dict(sorted(
                subject.outer_split_api.FAMILY_QUOTAS.items())),
        },
        "information_boundary": (
            subject._expected_fresh_outer_information_boundary()),
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "runtime_action_override": False,
        "producer_sources": sources,
        "producer_sources_sha256": subject.digest(sources),
        "formal_ready": False,
    }
    registry["content_sha256"] = subject.digest(registry)
    return registry


def _fresh_outer_report_fixture(registry, *, registry_file_sha256):
    selection = {
        "salt": subject.outer_split_api.SELECTION_SALT,
        "family_quotas": dict(sorted(
            subject.outer_split_api.FAMILY_QUOTAS.items())),
        "selected_identity_sha256": subject.digest(
            registry["selected_outer_identities"]),
        "source_rows_scene_fingerprints_sha256": "b" * 64,
        "previously_exposed_outer_fingerprints_sha256": subject.digest(
            registry["previously_exposed_validation_scene_fingerprints"]),
        "prior_expansion_trace_fingerprints_sha256": "c" * 64,
    }
    report = {
        "version": subject.outer_split_api.REPORT_VERSION,
        "status": subject.outer_split_api.STATUS,
        "registry_content_sha256": registry["content_sha256"],
        "registry_file_sha256": registry_file_sha256,
        "bindings": deepcopy(registry["bindings"]),
        "selection": selection,
        "statistics": deepcopy(registry["statistics"]),
        "information_boundary": deepcopy(registry["information_boundary"]),
        "producer_sources": deepcopy(registry["producer_sources"]),
        "producer_sources_sha256": registry["producer_sources_sha256"],
        "formal_ready": False,
    }
    report["content_sha256"] = subject.digest(report)
    return report


def _exception_module_material(error, module_names):
    """Collect scalar material reachable through protected-module tracebacks."""
    material = []
    seen = set()

    def visit(value):
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, (BaseException, dict, list, tuple, set, frozenset)):
            seen.add(identity)
        if isinstance(value, bytes):
            material.extend((repr(value), value.hex()))
        elif isinstance(value, (str, int)):
            material.append(str(value))
        elif isinstance(value, BaseException):
            material.append(str(value))
            visit(value.args)
            visit(value.__dict__)
            traceback = value.__traceback__
            while traceback is not None:
                frame = traceback.tb_frame
                if frame.f_globals.get("__name__") in module_names:
                    visit(dict(frame.f_locals))
                traceback = traceback.tb_next
            if value.__cause__ is not None:
                visit(value.__cause__)
            if value.__context__ is not None:
                visit(value.__context__)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(error)
    return "\n".join(material)


def _assert_private_material_absent(error, fragments, *module_names):
    rendered = _exception_module_material(error, set(module_names))
    assert all(str(fragment) not in rendered for fragment in fragments), rendered


def test_v4_contract_is_program_blind_and_claim_burns_v3_internally():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-fresh-final-holdout.v4"
    assert sum(contract["family_quotas"].values()) == subject.TOTAL_SCENES == 64
    assert contract["selection_salt_commitment"] == subject.HOLDOUT_SALT_COMMITMENT
    assert contract["program_access"] is False
    assert contract["program_predictions_access"] is False
    assert contract["actor_logits_access"] is False
    assert contract["final_labels_used_for_selection"] is False
    assert contract["retired_identity_projection_version"] == (
        subject.retired_identity_api.VERSION)
    assert contract["retired_identity_projection_file_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256)
    assert contract["retired_identity_projection_report_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256)
    assert contract["retired_identity_count"] == 139
    assert contract["full_retired_holdout_access"] is False
    assert contract["internal_v3_exclusion_registry"] == subject.V3_TOMBSTONE_VERSION


def test_fresh_outer_registry_contract_is_accepted_and_old_registry_is_rejected():
    registry = _fresh_outer_registry_fixture()
    subject._validate_fresh_outer_registry(registry)
    old = deepcopy(registry)
    old["version"] = "warehouse-r41-diagnostic-development-expansion.v8"
    old["status"] = "passed_program_blind_registry"
    old["content_sha256"] = subject.digest({
        key: value for key, value in old.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="registry contract differs"):
        subject._validate_fresh_outer_registry(old)


def test_fresh_outer_registry_rejects_nested_retired_binding():
    registry = _fresh_outer_registry_fixture()
    bindings = registry["bindings"]
    projection = {
        "file_sha256": bindings.pop("retired_projection_file_sha256"),
        "audit_file_sha256": bindings.pop(
            "retired_projection_report_file_sha256"),
    }
    bindings["retired_identity_projection"] = projection
    registry["content_sha256"] = subject.digest({
        key: value for key, value in registry.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="registry contract differs"):
        subject._validate_fresh_outer_registry(registry)


def test_fresh_outer_registry_rejects_wrong_formal_selection_binding():
    registry = _fresh_outer_registry_fixture()
    registry["bindings"]["formal_selection_file_sha256"] = "f" * 64
    registry["content_sha256"] = subject.digest({
        key: value for key, value in registry.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="registry contract differs"):
        subject._validate_fresh_outer_registry(registry)


def test_fresh_outer_registry_rejects_wrong_family_quota():
    registry = _fresh_outer_registry_fixture()
    outer = registry["development_validation"]
    moved = next(row for row in outer
                 if row["family_id"] == subject.outer_split_api.FAMILY_IDS[0])
    moved["family_id"] = subject.outer_split_api.FAMILY_IDS[1]
    selected = next(row for row in registry["selected_outer_identities"]
                    if row["fingerprint"] == moved["fingerprint"])
    selected["family_id"] = moved["family_id"]
    registry["statistics"]["fresh_outer_family_counts"] = dict(sorted(
        Counter(row["family_id"] for row in outer).items()))
    registry["content_sha256"] = subject.digest({
        key: value for key, value in registry.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="family quota differs"):
        subject._validate_fresh_outer_registry(registry)


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
        "scripts/build_warehouse_r41_diagnostic_designation_v2.py",
        "backend/training/warehouse_native_common.py",
        "env/warehouse/transition_outcome.py",
        "env/warehouse_native/environment.py",
    }.issubset(sources)
    assert "core/__init__.py" in sources
    assert not any("admission" in path or "release" in path for path in sources)


def test_staged_publication_failure_leaves_no_public_output(tmp_path):
    output = tmp_path / "holdout"
    artifacts = {
        "v3_exclusion.json": {"artifact": "v3"},
        "holdout.json": {"artifact": "holdout"},
        "report.json": {"artifact": "report"},
    }
    hashes = {
        name: subject._json_file_sha256(value)
        for name, value in artifacts.items()
    }

    def drift():
        raise RuntimeError("synthetic frozen-input drift")

    with pytest.raises(RuntimeError, match="frozen-input drift"):
        subject._publish_staged_holdout(
            output, artifacts=artifacts, expected_hashes=hashes,
            before_publish=drift)
    assert not output.exists()
    assert not (tmp_path / ".holdout.partial").exists()


def test_post_claim_input_drift_burns_without_output_or_completion(
        tmp_path, monkeypatch):
    claim_dir = tmp_path / "claim"
    claim_dir.mkdir()
    candidate = claim_dir / "candidate_authenticated.json"
    candidate.write_text("{}\n", encoding="utf-8")
    inputs = {}
    for name in (
        "actor", "protocol", "manifest", "designation", "selected",
        "development", "rows", "legacy_rows", "expansion_rows", "retired",
        "implicit",
    ):
        path = tmp_path / (name + (".npz" if "rows" in name else ".json"))
        path.write_bytes(b"input\n")
        inputs[name] = path
    (tmp_path / "report.json").write_bytes(b"input\n")
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REGISTRY_SHA256",
        subject.file_hash(inputs["development"]))
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REPORT_SHA256",
        subject.file_hash(tmp_path / "report.json"))
    output = tmp_path / "public-holdout"
    monkeypatch.setattr(
        subject, "_claim_receipt", lambda *args, **kwargs: (claim_dir, {}))
    monkeypatch.setattr(subject, "producer_sources", lambda: {"source.py": "a" * 64})
    monkeypatch.setattr(
        subject, "_implicit_input_paths",
        lambda **kwargs: {
            "manifest_validation": inputs["implicit"],
            **{
                "designation_component:" + name: inputs["implicit"]
                for name in subject.designation_api.ARTIFACT_NAMES
            },
        })
    class Snapshot:
        def __init__(self, *args, **kwargs):
            self.paths = dict(args[0])
            self.original_paths = dict(args[0])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def verify(self):
            return None
    monkeypatch.setattr(
        subject.input_snapshot_api, "ImmutableInputSnapshot", Snapshot)
    phases = []

    def guard(*args, phase, **kwargs):
        phases.append(phase)
        if phase == "post-holdout phase start":
            raise RuntimeError("synthetic transaction drift")

    monkeypatch.setattr(subject, "_guard_frozen_holdout_inputs", guard)
    monkeypatch.setattr(
        subject, "_read_committed_salt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("private salt read after detected drift")))
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject.build(
            actor_path=inputs["actor"], protocol_path=inputs["protocol"],
            manifest_path=inputs["manifest"],
            designation_path=inputs["designation"],
            selected_scenes_path=inputs["selected"],
            development_registry_paths=[inputs["development"]],
            development_rows_paths=[inputs["rows"]],
            legacy_v3_rows_path=inputs["legacy_rows"],
            expansion_rows_path=inputs["expansion_rows"],
            retired_identity_projection_path=inputs["retired"], output=output,
            claim_receipt_path=candidate,
            expected_claim_sha256="1" * 64,
            expected_campaign_key="2" * 64,
            expected_candidate_identity_sha256="3" * 64,
            selection_salt_path=tmp_path / "private-salt",
        )
    assert phases == [
        "pre-holdout authentication", "claim-chain reauthentication",
        "holdout phase start",
        "post-holdout phase start",
    ]
    assert (claim_dir / "holdout_started.json").is_file()
    assert not (claim_dir / "holdout_completed.json").exists()
    assert not output.exists()


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


def test_auxiliary_observation_collectors_match_frozen_consumer_workloads(
        monkeypatch):
    actor = ROOT / (
        "output/warehouse_native/r41_active_2m_20260911/boundaries/"
        "step_2000000/actor.npz")
    protocol = ROOT / "output/warehouse_native/r41_active_2m_20260911/protocol.json"
    manifest_path = ROOT / (
        "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/"
        "manifest.json")
    if not all(path.is_file() for path in (actor, protocol, manifest_path)):
        pytest.skip("frozen diagnostic runtime inputs are not present")
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    public_splits, _ = subject.manifest_binding.regenerate_development_splits(
        actor, splits=("tutorial", "question_bank"))
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor,
        training_protocol_path=protocol,
        manifest_path=manifest_path,
        expected_actor_sha256=subject.EXPECTED_ACTOR_SHA256,
        expected_training_protocol_file_sha256=subject.EXPECTED_PROTOCOL_SHA256,
        expected_training_protocol_content_sha256=subject.digest(
            protocol_value),
        expected_manifest_file_sha256=subject.EXPECTED_MANIFEST_SHA256,
        expected_manifest_content_sha256=(
            subject.manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256),
        expected_manifest_semantic_sha256=(
            subject.manifest_binding.EXPECTED_MANIFEST_SEMANTIC_SHA256),
    )

    question_scene = public_splits["question_bank"][0]
    actual_question = subject._question_bank_workload_observations(
        runtime, question_scene, 0)
    canonical_question: set[str] = set()
    original_policy_action = workload_screen._policy_action

    def capture_policy_action(actor_value, env):
        canonical_question.add(subject._observation_hash(
            env.observations()["robot_2"]))
        return original_policy_action(actor_value, env)

    monkeypatch.setattr(
        workload_screen, "_policy_action", capture_policy_action)
    workload_screen._screen_question_bank(
        runtime.actor, question_scene, scene_index=0)
    assert actual_question == canonical_question

    tutorial_scene = public_splits["tutorial"][0]
    actual_tutorial = subject._tutorial_workload_observations(
        runtime, tutorial_scene, 0)
    canonical_tutorial: set[str] = set()
    original_public_frame = tutorial_base._public_frame

    def capture_public_frame(env, *args, **kwargs):
        canonical_tutorial.add(subject._observation_hash(
            env.observations()["robot_2"]))
        return original_public_frame(env, *args, **kwargs)

    monkeypatch.setattr(
        tutorial_base, "_public_frame", capture_public_frame)
    workload_screen._screen_tutorial(tutorial_scene)
    assert actual_tutorial == canonical_tutorial


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
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject.build(
            actor_path="missing-actor", protocol_path="missing-protocol",
            manifest_path="missing-manifest", designation_path="missing-designation",
            selected_scenes_path="missing-selection",
            development_registry_paths=[], development_rows_paths=[],
            legacy_v3_rows_path="missing-v3-rows",
            expansion_rows_path="missing-expansion-rows",
            retired_identity_projection_path="missing-retired-projection",
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


def test_exact_identity_projection_is_the_only_retired_input():
    projection = (
        ROOT / "output/warehouse_native/"
            "r41_diagnostic_retired_identity_projection_v6_sourceclosure4_20260913/"
        "retired_identity_projection.json"
    )
    if not projection.is_file():
        pytest.skip("frozen retired identity projection is not present")
    identities, binding = subject._retired_identity_projection(projection)
    assert len(identities) == 139
    assert set(identities[0]) == {"seed", "fingerprint"}
    assert binding["file_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256)
    assert binding["audit_file_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256)
    assert binding["identity_count"] == 139


def test_expansion_must_bind_the_same_identity_only_projection():
    binding = {
        "version": subject.retired_identity_api.VERSION,
        "file_sha256": subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256,
        "content_sha256": "1" * 64,
        "audit_file_sha256": (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        "source_file_sha256": {"v1": "2" * 64, "v2": "3" * 64},
        "identity_count": 139,
    }
    subject._validate_retired_expansion_binding(
        binding, {"bindings": {
            "retired_projection_file_sha256": binding["file_sha256"],
            "retired_projection_report_file_sha256": binding["audit_file_sha256"],
        }})
    with pytest.raises(ValueError, match="identity projection differs"):
        subject._validate_retired_expansion_binding(
            binding, {"bindings": {"retired_identity_projection": binding}})
    replacement = dict(binding)
    replacement["file_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity projection differs"):
        subject._validate_retired_expansion_binding(
            replacement,
            {"bindings": {
                "retired_projection_file_sha256": replacement["file_sha256"],
                "retired_projection_report_file_sha256": replacement[
                    "audit_file_sha256"],
            }},
        )


def test_old_and_fresh_outer_scenes_are_both_in_identity_exclusion_closure(
        monkeypatch):
    registry = _fresh_outer_registry_fixture()
    old = []
    for index, fingerprint in enumerate(
            registry["previously_exposed_validation_scene_fingerprints"]):
        family = subject.outer_split_api.FAMILY_IDS[
            min(index // 11, len(subject.outer_split_api.FAMILY_IDS) - 1)]
        # The final two families have ten rows; derive the exact frozen order.
        old.append({
            "seed": 30_000 + index, "fingerprint": fingerprint,
            "family_id": family,
        })
    # Reassign by the authoritative quotas rather than relying on division.
    cursor = 0
    for family in subject.outer_split_api.FAMILY_IDS:
        for row in old[cursor:cursor + subject.outer_split_api.FAMILY_QUOTAS[family]]:
            row["family_id"] = family
        cursor += subject.outer_split_api.FAMILY_QUOTAS[family]
    manifest = {
        "content_sha256": subject.manifest_binding.EXPECTED_MANIFEST_CONTENT_SHA256,
        "authentication": {
            "candidate_scenes_disjoint_from_all_base_splits_authenticated": True,
        },
        "candidate_batches": [old],
    }
    ordered = []
    for family in subject.outer_split_api.FAMILY_IDS:
        candidates = [row for row in old if row["family_id"] == family]
        candidates.sort(key=lambda row: subject.digest({
            "salt": subject.outer_split_api.expansion_api.VALIDATION_ORDER_SALT,
            "family_id": family,
            "fingerprint": row["fingerprint"],
        }))
        ordered.extend(candidates)
    monkeypatch.setattr(
        subject, "EXPECTED_OLD_OUTER_ORDERED_FINGERPRINTS_SHA256",
        subject.digest([row["fingerprint"] for row in ordered]))
    monkeypatch.setattr(
        subject, "EXPECTED_OLD_OUTER_ORDERED_SEEDS_SHA256",
        subject.digest([row["seed"] for row in ordered]))
    old_result, fresh_result = subject._outer_split_exclusion_scenes(
        manifest, registry)
    assert len(old_result) == len(fresh_result) == 64
    assert {row["seed"] for row in old_result} == {
        30_000 + index for index in range(64)}
    assert {row["fingerprint"] for row in fresh_result} == {
        row["fingerprint"] for row in registry["development_validation"]}
    assert [row["fingerprint"] for row in old_result] == [
        row["fingerprint"] for row in ordered]


def test_old_outer_observation_replay_uses_retired_collection_schedule(
        monkeypatch):
    class State:
        frame = 0

    class Environment:
        def __init__(self, token):
            self.token = token
            self.state = State()
            self.done = False

        def snapshot(self):
            return {"token": self.token, "frame": self.state.frame}

        def observations(self):
            value = np.zeros(197, dtype=np.float32)
            value[0] = self.token
            return {"robot_2": value}

    class Runtime:
        def __init__(self):
            self.calls = 0

        def environment(self, scene):
            self.calls += 1
            return Environment(scene["seed"] * 10 + self.calls % 3)

        def step(self, env, action):
            env.state.frame += 1
            env.done = True
            return {
                "submitted_actions": {"robot_2": "WAIT"},
                "policy_actions": {"robot_2": "WAIT"},
            }

    monkeypatch.setattr(subject, "critical_groups", lambda *args: ())
    monkeypatch.setattr(subject, "partner_action", lambda *args: "WAIT")
    scenes = [{"seed": index + 1} for index in range(
        subject.outer_split_api.OLD_OUTER_SCENE_COUNT)]
    runtime = Runtime()
    observations = subject._old_outer_collection_observations(runtime, scenes)
    assert runtime.calls == len(scenes) * len(subject.PARTNERS)
    assert len(observations) == runtime.calls


def test_development_registry_authenticates_and_binds_sibling_report(
        tmp_path, monkeypatch):
    supplement = {
        "version": subject.DEVELOPMENT_SUPPLEMENT_VERSION,
        "status": "passed", "program_access": False,
        "final_audit_rows_access": False,
        "scenes": [
            {"seed": 40_000 + index,
             "fingerprint": sha256(f"supplement:{index}".encode()).hexdigest()}
            for index in range(64)
        ],
    }
    supplement["content_sha256"] = subject.digest(supplement)
    supplement_path = tmp_path / "supplement.json"
    supplement_path.write_text(subject.canonical(supplement) + "\n", encoding="utf-8")

    outer_dir = tmp_path / "outer"; outer_dir.mkdir()
    registry = _fresh_outer_registry_fixture()
    registry_path = outer_dir / "development_expansion.json"
    registry_path.write_text(subject.canonical(registry) + "\n", encoding="utf-8")
    registry_sha = subject.file_hash(registry_path)
    monkeypatch.setattr(subject, "EXPECTED_EXPANSION_REGISTRY_SHA256", registry_sha)
    report = _fresh_outer_report_fixture(
        registry, registry_file_sha256=registry_sha)
    report_path = outer_dir / "report.json"
    report_path.write_text(subject.canonical(report) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REPORT_SHA256", subject.file_hash(report_path))

    _, bindings, _ = subject._development_registry_evidence(
        [supplement_path, registry_path])
    frozen = bindings[subject.DEVELOPMENT_EXPANSION_VERSION]
    assert frozen["report_file_sha256"] == subject.file_hash(report_path)
    assert frozen["report_content_sha256"] == report["content_sha256"]

    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report"):
        subject._development_registry_evidence([supplement_path, registry_path])


def test_retired_projection_maps_only_identity_and_never_executes_actor():
    identities = [
        {"fingerprint": sha256(f"retired:{index}".encode()).hexdigest(),
         "seed": 1000 + index}
        for index in range(139)
    ]
    manifest = {"candidate_batches": [[
        {**identity, "family_id": "public", "geometry": {"x": index}}
        for index, identity in enumerate(identities)
    ]]}
    fingerprints, seeds, stats = subject._projected_identity_exclusions(
        manifest=manifest, identities=identities)
    assert fingerprints == {row["fingerprint"] for row in identities}
    assert seeds == {row["seed"] for row in identities}
    assert stats["projected_identity_count"] == 139
    assert stats["retired_actor_executed"] is False
    assert stats["retired_observations_derived"] is False
    assert "public_observations_sha256" not in stats


def test_v3_tombstone_is_claim_bound_and_program_blind(monkeypatch):
    candidates = {}
    for family_index, family in enumerate(subject.FAMILY_IDS):
        candidates[family] = [
            {"fingerprint": sha256(f"{family}:{index}".encode()).hexdigest(),
             "seed": 1000 * family_index + index, "family_id": family}
            for index in range(subject.FAMILY_QUOTAS[family])
        ]
    monkeypatch.setattr(
        subject.legacy_v3, "_ordered_candidates",
        lambda manifest, family: candidates[family])
    monkeypatch.setattr(subject, "screen_scene",
                        lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(
        subject, "_exact_final_workload_observations",
        lambda runtime, scene, index: {
            sha256((scene["fingerprint"] + ":obs").encode()).hexdigest()})
    binding = {"campaign_key": "b" * 64, "attempt_started_sha256": "c" * 64}
    tombstone = subject._build_v3_tombstone(
        runtime=object(), actor=object(),
        manifest={"splits": {"train": []}, "candidate_batches": []},
        selected={"X": [], "Y": []}, legacy_development_hashes=set(),
        projected_retired_fingerprints=set(), projected_retired_seeds=set(),
        claim_binding=binding)
    assert tombstone["version"] == subject.V3_TOMBSTONE_VERSION
    assert tombstone["status"] == "burned_program_blind_exclusion"
    assert tombstone["claim_binding"] == binding
    assert tombstone["program_access"] is False
    assert tombstone["program_predictions_access"] is False
    assert tombstone["statistics"]["retired_actor_executed"] is False
    assert tombstone["statistics"]["retired_public_observation_count"] == 0
    assert tombstone["content_sha256"] == subject.digest({
        key: value for key, value in tombstone.items() if key != "content_sha256"
    })


def _historical_phase_fixture(tmp_path, monkeypatch):
    claim_dir = tmp_path / "claim"
    claim_dir.mkdir(parents=True)
    expected_claim = "1" * 64
    expected_campaign = "2" * 64
    expected_candidate = "3" * 64
    salt = b"s" * 32
    monkeypatch.setattr(
        subject, "HOLDOUT_SALT_COMMITMENT",
        sha256(subject.HOLDOUT_SALT_DOMAIN + salt).hexdigest())

    class Runtime:
        actor_sha256 = subject.EXPECTED_ACTOR_SHA256
        signature = "bound"
        actor = object()
        _actor_path = tmp_path / "actor.npz"

        def verify_binding(self):
            return self.signature

    Runtime._actor_path.write_bytes(b"actor")
    monkeypatch.setattr(subject, "R41DiagnosticOnlineAlignmentRuntime", Runtime)
    identities = [
        {
            "id": f"diagnostic_final_test_{index:04d}",
            "split": "final_test",
            "fingerprint": sha256(
                f"historical:{index}".encode()).hexdigest(),
            "seed": 1000 + index,
            "diagnostic_contract_sha256": "a" * 64,
            "diagnostic_conflict_graph_sha256": "b" * 64,
            "conflict_families_sha256": "c" * 64,
            "family_id": "opposing_pickup_corridor",
            "initial_edge_id": "edge",
            "task_geometry_signature": "d" * 64,
            "initial_conflict": {},
            "initial_public_joint_work_steps": 1,
            "initial_robot_positions": [[1, 1], [2, 2]],
            "successor_stream_seed": 5000 + index,
            "snapshot": {},
            # This malformed saved result must remain completely opaque to the
            # historical observation-only exclusion path.
            "workload_screen": {
                "metrics": {"poisoned_saved_metric": index},
                "actions": ["FORBIDDEN_SAVED_ACTION"],
            },
        }
        for index in range(subject.TOTAL_SCENES)
    ]
    payload = {"splits": {"final_test": [dict(row) for row in identities]}}
    payload["content_sha256"] = subject.digest(payload)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(subject.canonical(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(subject, "EXPECTED_MANIFEST_SHA256", subject.file_hash(manifest))
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_MANIFEST_CONTENT_SHA256",
        payload["content_sha256"])
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_MANIFEST_SEMANTIC_SHA256",
        subject.digest(payload))
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_FINAL_IDENTITY_SHA256",
        subject.digest([
            {"id": row["id"], "seed": row["seed"],
             "fingerprint": row["fingerprint"]}
            for row in identities
        ]))
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_FINAL_FINGERPRINTS_SHA256",
        subject.digest(sorted(row["fingerprint"] for row in identities)))
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_FINAL_SEEDS_SHA256",
        subject.digest(sorted(row["seed"] for row in identities)))
    monkeypatch.setattr(subject.manifest_binding, "_validate_contract", lambda value: None)
    monkeypatch.setattr(
        subject.manifest_binding, "_validate_source_bindings", lambda value: None)
    replay_calls = []
    monkeypatch.setattr(
        subject.manifest_binding, "_validate_rows",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("saved workload comparator must not run")))

    public_manifest = {
        "version": subject.MANIFEST_VERSION,
        "content_sha256": payload["content_sha256"],
        "splits": {
            "train": [], "conflict_validation": [],
            "tutorial": [], "question_bank": [],
        },
        "candidate_batches": [],
    }
    monkeypatch.setattr(
        subject.manifest_binding, "read_saved_manifest",
        lambda *args, **kwargs: public_manifest)

    selected = {"X": [], "Y": []}
    for index in range(6):
        selected["X" if index < 3 else "Y"].append({
            "fingerprint": sha256(f"xy:{index}".encode()).hexdigest(),
            "seed": 2000 + index,
        })
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(subject.canonical(selected) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        subject, "EXPECTED_SELECTED_SCENES_SHA256",
        subject.file_hash(selected_path))

    development_paths = []
    development_bindings = {}
    for version in (
            subject.DEVELOPMENT_SUPPLEMENT_VERSION,
            subject.DEVELOPMENT_EXPANSION_VERSION):
        path = tmp_path / (version.rsplit(".", 1)[-1] + ".json")
        path.write_text(subject.canonical({"version": version}) + "\n",
                        encoding="utf-8")
        development_paths.append(path)
        development_bindings[version] = {
            "file_sha256": subject.file_hash(path), "content_sha256": "d" * 64}
    monkeypatch.setattr(
        subject, "_development_registry_evidence",
        lambda paths: (
            [], development_bindings,
            {
                version: {
                    "version": version,
                    "content_sha256": binding["content_sha256"],
                }
                for version, binding in development_bindings.items()
            },
        ))

    projection_path = tmp_path / "projection.json"
    projection_path.write_text("{}\n", encoding="utf-8")
    (tmp_path / "report.json").write_text("{}\n", encoding="utf-8")
    projected_fp = sha256(b"retired-projected").hexdigest()
    projected_seed = 3001
    projected_binding = {
        "version": subject.retired_identity_api.VERSION,
        "file_sha256": subject.file_hash(projection_path),
        "content_sha256": "e" * 64,
        "audit_file_sha256": subject.file_hash(tmp_path / "report.json"),
        "source_file_sha256": {"v1": "f" * 64},
        "identity_count": 1,
    }
    monkeypatch.setattr(
        subject, "_retired_identity_projection",
        lambda path: ([{"fingerprint": projected_fp, "seed": projected_seed}],
                      projected_binding))
    monkeypatch.setattr(
        subject, "_validate_retired_expansion_binding",
        lambda *args, **kwargs: None)
    old_outer_scene = {
        "fingerprint": sha256(b"old-outer-scene").hexdigest(), "seed": 3501,
    }
    old_outer_observation = sha256(b"old-outer-observation").hexdigest()
    monkeypatch.setattr(
        subject, "_outer_split_exclusion_scenes",
        lambda *args, **kwargs: ([old_outer_scene], []))
    monkeypatch.setattr(
        subject, "_old_outer_collection_observations",
        lambda runtime, scenes: (
            {old_outer_observation}
            if scenes == [old_outer_scene]
            else (_ for _ in ()).throw(AssertionError("old outer scenes differ"))))
    monkeypatch.setattr(
        subject, "_projected_identity_exclusions",
        lambda **kwargs: ({projected_fp}, {projected_seed}, {}))

    v3_scene = {"fingerprint": sha256(b"v3-scene").hexdigest(), "seed": 4001}
    v3_observation = sha256(b"v3-observation").hexdigest()
    v3_tombstone = {
        "version": subject.V3_TOMBSTONE_VERSION,
        "scenes": [v3_scene], "selection_trace": [],
        "content_sha256": sha256(b"v3-content").hexdigest(),
    }
    monkeypatch.setattr(
        subject, "_build_v3_tombstone", lambda **kwargs: v3_tombstone)
    monkeypatch.setattr(
        subject, "_selection_exposure_closure",
        lambda **kwargs: ({v3_scene["fingerprint"]}, {v3_scene["seed"]},
                          {v3_observation}, {}))
    final_hashes = {
        index: sha256(f"historical-observation:{index}".encode()).hexdigest()
        for index in range(subject.TOTAL_SCENES)
    }
    historical_by_fingerprint = {
        row["fingerprint"]: final_hashes[index]
        for index, row in enumerate(identities)
    }
    xy_observations = {
        row["fingerprint"]: sha256(
            ("xy-observation:" + row["fingerprint"]).encode()).hexdigest()
        for rows in selected.values() for row in rows
    }

    def workload_observations(runtime, scene, index):
        fingerprint = scene["fingerprint"]
        if fingerprint in historical_by_fingerprint:
            return {historical_by_fingerprint[fingerprint]}
        if fingerprint in xy_observations:
            return {xy_observations[fingerprint]}
        raise AssertionError("unexpected workload scene")

    monkeypatch.setattr(
        subject, "_exact_final_workload_observations", workload_observations)
    fresh_scene = {
        "fingerprint": sha256(b"fresh-selected").hexdigest(),
        "seed": 9001,
    }

    def select_fresh(**kwargs):
        assert kwargs["selection_salt"] == salt
        assert not (set(row["fingerprint"] for row in identities)
                    & kwargs["excluded_fingerprints"])
        assert not (set(row["seed"] for row in identities)
                    & kwargs["excluded_seeds"])
        assert not (set(final_hashes.values())
                    & kwargs["forbidden_observation_hashes"])
        return ([fresh_scene], [{"accepted": True}], {"accepted": 1},
                {sha256(b"fresh-observation").hexdigest()})

    monkeypatch.setattr(subject, "_select", select_fresh)

    evidence = {}
    for index, name in enumerate(
            ("merged_rows.npz", "prior_rows.npz", "expansion_rows.npz")):
        path = tmp_path / name
        fingerprint = sha256(f"row-scene:{index}".encode()).hexdigest()
        observation_hashes = _rows(
            path, [np.arange(197, dtype=np.float32) + 10000 * (index + 1)],
            [fingerprint])
        evidence[name] = (path, {fingerprint}, observation_hashes)
    candidate = claim_dir / "candidate_authenticated.json"
    candidate.write_text(subject.canonical({
        "development_registries": {
            version: binding["file_sha256"]
            for version, binding in development_bindings.items()
        },
        "rows_file_sha256": subject.file_hash(evidence["merged_rows.npz"][0]),
        "prior_v7_rows_file_sha256": subject.file_hash(
            evidence["prior_rows.npz"][0]),
        "expansion_rows_file_sha256": subject.file_hash(
            evidence["expansion_rows.npz"][0]),
    }) + "\n", encoding="utf-8")
    (claim_dir / "holdout_started.json").write_text(subject.canonical({
        "version": subject.VERSION,
        "status": "started_no_retry",
        "campaign_key": expected_campaign,
        "candidate_identity_sha256": expected_candidate,
        "attempt_started_sha256": expected_claim,
        "candidate_authenticated_sha256": subject.file_hash(candidate),
        "selection_salt_commitment": subject.HOLDOUT_SALT_COMMITMENT,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        subject, "_claim_receipt", lambda *args, **kwargs: (claim_dir, {}))

    replayed = {
        v3_observation, old_outer_observation, *xy_observations.values()}
    row_observations = set().union(*(value[2] for value in evidence.values()))
    excluded_fingerprints = {
        projected_fp, v3_scene["fingerprint"], old_outer_scene["fingerprint"],
        *xy_observations.keys()}
    excluded_seeds = {
        projected_seed, v3_scene["seed"], old_outer_scene["seed"],
        *(row["seed"] for rows in selected.values() for row in rows),
    }
    args = {
        "manifest_path": manifest,
        "runtime": Runtime(),
        "selection_manifest": public_manifest,
        "selection_salt": salt,
        "selected_scenes_path": selected_path,
        "development_registry_paths": development_paths,
        "retired_identity_projection_path": projection_path,
        "v3_tombstone": v3_tombstone,
        "claim_receipt_path": claim_dir / "attempt_started.json",
        "expected_claim_sha256": expected_claim,
        "expected_campaign_key": expected_campaign,
        "expected_candidate_identity_sha256": expected_candidate,
        "excluded_fingerprints": excluded_fingerprints,
        "excluded_seeds": excluded_seeds,
        "preexisting_observation_hashes": row_observations | replayed,
        "replayed_excluded_observation_hashes": replayed,
        "row_artifact_evidence": evidence,
    }
    return claim_dir, manifest, identities, set(final_hashes.values()), replay_calls, args


def test_historical_final_access_is_marked_before_parse_and_returns_only_hashes(
        tmp_path, monkeypatch):
    claim_dir, manifest, identities, expected_hashes, replay_calls, args = (
        _historical_phase_fixture(tmp_path, monkeypatch))
    public_fingerprints = set(args["excluded_fingerprints"])
    public_seeds = set(args["excluded_seeds"])
    original_read = subject._read_exact_json
    reads = []

    def guarded_read(path, label, **kwargs):
        if Path(path) == manifest:
            assert (claim_dir / "historical_exclusion_started.json").is_file()
            reads.append("manifest_after_marker")
        return original_read(path, label, **kwargs)

    monkeypatch.setattr(subject, "_read_exact_json", guarded_read)
    scenes, trace, statistics, accepted_hashes, receipt = (
        subject._build_claimed_historical_final_observation_exclusion(**args))
    assert len(scenes) == 1 and trace == [{"accepted": True}]
    assert statistics == {"accepted": 1}
    assert accepted_hashes == {sha256(b"fresh-observation").hexdigest()}
    assert reads == ["manifest_after_marker"]
    assert replay_calls == []
    assert receipt["historical_final_scenes_returned"] is False
    assert receipt["historical_final_actor_outputs_exposed"] is False
    assert receipt["historical_final_actor_outputs_persisted"] is False
    assert receipt["historical_final_metrics_used_for_fit_or_program_selection"] is False
    assert receipt[
        "historical_final_saved_metrics_replayed_for_authentication"] is False
    assert receipt["fresh_selection_conditioned_on_historical_final"] is False
    assert receipt["fresh_selection_private_overlap_fallback"] is False
    assert receipt["fresh_selection_historical_scene_fingerprint_overlap"] == 0
    assert receipt["fresh_selection_historical_seed_overlap"] == 0
    assert receipt["fresh_selection_historical_public_observation_overlap"] == 0
    assert "scenes" not in receipt and "actions" not in receipt
    assert receipt["historical_final_scene_count"] == subject.TOTAL_SCENES
    assert args["excluded_fingerprints"] == public_fingerprints
    assert args["excluded_seeds"] == public_seeds
    assert receipt["combined_excluded_scene_fingerprint_count"] == (
        subject.TOTAL_SCENES + len(public_fingerprints))
    assert receipt["combined_excluded_seed_count"] == (
        subject.TOTAL_SCENES + len(public_seeds))
    assert (claim_dir / "historical_exclusion_completed.json").is_file()
    # The durable receipt may contain aggregate counts and commitments, but it
    # must never serialize any raw historical identity/observation value,
    # whether as a scalar, mapping key, or member of a nested collection.
    private_values = {
        *(row["fingerprint"] for row in identities),
        *(row["seed"] for row in identities),
        *expected_hashes,
    }

    def assert_no_private_values(value, path="receipt"):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in private_values, f"{path}.<key>"
                assert_no_private_values(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple, set)):
            for index, item in enumerate(value):
                assert_no_private_values(item, f"{path}[{index}]")
        else:
            assert value not in private_values, path

    assert_no_private_values(receipt)
    for evidence in receipt["row_artifact_evidence"].values():
        assert evidence["zero_historical_overlap"] is True
        assert evidence["historical_scene_fingerprint_overlap"] == 0
        assert evidence["historical_public_observation_overlap"] == 0


def test_historical_final_crash_after_started_burns_phase_and_cannot_reenter(
        tmp_path, monkeypatch):
    claim_dir, manifest, _, _, _, args = _historical_phase_fixture(
        tmp_path, monkeypatch)
    original_read = subject._read_exact_json

    def crash_after_marker(path, label, **kwargs):
        if Path(path) == manifest:
            assert (claim_dir / "historical_exclusion_started.json").is_file()
            raise RuntimeError("synthetic full-manifest crash")
        return original_read(path, label, **kwargs)

    monkeypatch.setattr(subject, "_read_exact_json", crash_after_marker)
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$") as raised:
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert raised.value.__cause__ is None
    assert "full-manifest crash" not in str(raised.value)
    assert (claim_dir / "historical_exclusion_started.json").is_file()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)


def test_historical_final_overlap_and_forged_row_sets_fail_closed(
        tmp_path, monkeypatch):
    claim_dir, _, _, expected_hashes, _, args = _historical_phase_fixture(
        tmp_path, monkeypatch)
    forged = dict(args["row_artifact_evidence"])
    path, fingerprints, observations = forged["merged_rows.npz"]
    forged["merged_rows.npz"] = (path, set(fingerprints), {"e" * 64})
    args["row_artifact_evidence"] = forged
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert not (claim_dir / "historical_exclusion_started.json").exists()

    claim_dir, _, _, expected_hashes, _, args = _historical_phase_fixture(
        tmp_path / "overlap", monkeypatch)
    args["preexisting_observation_hashes"].add(next(iter(expected_hashes)))
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert not (claim_dir / "historical_exclusion_started.json").exists()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()


@pytest.mark.parametrize("forgery", ["salt", "manifest", "exclusions", "rows"])
def test_historical_helper_rejects_caller_chosen_selection_inputs_before_marker(
        tmp_path, monkeypatch, forgery):
    claim_dir, _, _, _, _, args = _historical_phase_fixture(
        tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(
        subject, "_select",
        lambda **kwargs: calls.append(kwargs) or (_ for _ in ()).throw(
            AssertionError("selection ran before public inputs authenticated")))
    if forgery == "salt":
        args["selection_salt"] = b"x" * 32
    elif forgery == "manifest":
        args["selection_manifest"] = {
            **args["selection_manifest"], "candidate_batches": [[{"forged": True}]]}
    elif forgery == "exclusions":
        args["excluded_fingerprints"] = set(args["excluded_fingerprints"])
        args["excluded_fingerprints"].add("f" * 64)
    else:
        replacement = tmp_path / "replacement.npz"
        fingerprint = sha256(b"replacement-row").hexdigest()
        observations = _rows(
            replacement, [np.arange(197, dtype=np.float32) + 90000],
            [fingerprint])
        args["row_artifact_evidence"] = dict(args["row_artifact_evidence"])
        args["row_artifact_evidence"]["merged_rows.npz"] = (
            replacement, {fingerprint}, observations)
        row_union = set().union(*(
            value[2] for value in args["row_artifact_evidence"].values()))
        args["preexisting_observation_hashes"] = (
            row_union | set(args["replayed_excluded_observation_hashes"]))
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert calls == []
    assert not (claim_dir / "historical_exclusion_started.json").exists()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()


@pytest.mark.parametrize("overlap_kind", ["fingerprint", "seed", "observation"])
def test_fresh_selection_private_overlap_burns_without_fallback(
        tmp_path, monkeypatch, overlap_kind):
    claim_dir, _, identities, historical_hashes, _, args = (
        _historical_phase_fixture(tmp_path, monkeypatch))
    calls = []
    fresh_scene = {
        "fingerprint": sha256(b"otherwise-fresh-scene").hexdigest(),
        "seed": 9917,
    }
    accepted = {sha256(b"otherwise-fresh-observation").hexdigest()}
    if overlap_kind == "fingerprint":
        fresh_scene["fingerprint"] = identities[0]["fingerprint"]
    elif overlap_kind == "seed":
        fresh_scene["seed"] = identities[0]["seed"]
    else:
        accepted = {next(iter(historical_hashes))}

    def select_once(**kwargs):
        calls.append(kwargs)
        # Private historical membership must not be supplied to selection.
        assert identities[0]["fingerprint"] not in kwargs["excluded_fingerprints"]
        assert identities[0]["seed"] not in kwargs["excluded_seeds"]
        assert not (historical_hashes & kwargs["forbidden_observation_hashes"])
        return ([dict(fresh_scene)], [{"accepted": True}], {"accepted": 1},
                set(accepted))

    monkeypatch.setattr(subject, "_select", select_once)
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert len(calls) == 1
    assert (claim_dir / "historical_exclusion_started.json").is_file()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$"):
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert len(calls) == 1


def test_private_replay_exception_payload_is_redacted(
        tmp_path, monkeypatch):
    claim_dir, _, identities, historical_hashes, _, args = (
        _historical_phase_fixture(tmp_path, monkeypatch))
    private_fragments = (
        identities[0]["fingerprint"], str(identities[0]["seed"]),
        next(iter(historical_hashes)), "actor_action=RIGHT",
        args["selection_salt"].hex(),
    )
    payload = " | ".join(private_fragments)
    original_observations = subject._exact_final_workload_observations
    historical_fingerprints = {row["fingerprint"] for row in identities}

    def fail_for_historical(runtime, scene, index):
        if scene["fingerprint"] in historical_fingerprints:
            raise RuntimeError(payload)
        return original_observations(runtime, scene, index)

    monkeypatch.setattr(
        subject, "_exact_final_workload_observations",
        fail_for_historical)

    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseError,
            match="^historical_final_private_phase_failed$") as raised:
        subject._build_claimed_historical_final_observation_exclusion(**args)
    rendered = repr(raised.value) + " " + str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.__suppress_context__ is True
    assert all(fragment not in rendered for fragment in private_fragments)
    _assert_private_material_absent(
        raised.value, private_fragments, subject.__name__)
    assert (claim_dir / "historical_exclusion_started.json").is_file()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()


def test_private_replay_interrupt_traceback_has_no_private_frame_locals(
        tmp_path, monkeypatch):
    claim_dir, _, identities, historical_hashes, _, args = (
        _historical_phase_fixture(tmp_path, monkeypatch))
    private_fragments = (
        identities[0]["fingerprint"], str(identities[0]["seed"]),
        next(iter(historical_hashes)), args["selection_salt"].hex(),
        "private-interrupt-payload",
    )
    payload = " | ".join(private_fragments)
    historical_fingerprints = {row["fingerprint"] for row in identities}

    class PrivateInterrupt(BaseException):
        pass

    original_observations = subject._exact_final_workload_observations

    def interrupt_historical(runtime, scene, index):
        if scene["fingerprint"] in historical_fingerprints:
            raise PrivateInterrupt(payload)
        return original_observations(runtime, scene, index)

    monkeypatch.setattr(
        subject, "_exact_final_workload_observations",
        interrupt_historical)
    with pytest.raises(
            subject._HistoricalFinalPrivatePhaseInterrupt,
            match="^historical_final_private_phase_interrupted$") as raised:
        subject._build_claimed_historical_final_observation_exclusion(**args)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    _assert_private_material_absent(
        raised.value, private_fragments, subject.__name__)
    assert (claim_dir / "historical_exclusion_started.json").is_file()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_post_selection_failure_does_not_expose_fresh_identity_or_salt(
        tmp_path, monkeypatch, interrupt):
    claim_dir, _, _, _, _, args = _historical_phase_fixture(
        tmp_path, monkeypatch)
    private_fragments = (
        sha256(b"fresh-selected").hexdigest(), "9001",
        sha256(b"fresh-observation").hexdigest(),
        args["selection_salt"].hex(),
    )
    payload = " | ".join(private_fragments)
    original_marker = subject._claim_marker

    class PrivateInterrupt(BaseException):
        pass

    def fail_before_historical_completion(directory, name, value, **kwargs):
        if name == "historical_exclusion_completed.json":
            if interrupt:
                raise PrivateInterrupt(payload)
            raise RuntimeError(payload)
        return original_marker(directory, name, value, **kwargs)

    monkeypatch.setattr(subject, "_claim_marker",
                        fail_before_historical_completion)
    expected = (
        subject._HistoricalFinalPrivatePhaseInterrupt
        if interrupt else subject._HistoricalFinalPrivatePhaseError)
    with pytest.raises(expected) as raised:
        subject._build_claimed_historical_final_observation_exclusion(**args)
    _assert_private_material_absent(
        raised.value, private_fragments, subject.__name__)
    assert (claim_dir / "historical_exclusion_started.json").is_file()
    assert not (claim_dir / "historical_exclusion_completed.json").exists()


@pytest.mark.parametrize("interrupt", [False, True])
def test_public_build_redacts_post_salt_exception_and_worker_is_not_callable(
        tmp_path, monkeypatch, interrupt):
    claim_dir = tmp_path / "claim"
    claim_dir.mkdir()
    candidate = claim_dir / "candidate_authenticated.json"
    candidate.write_text("{}\n", encoding="utf-8")
    inputs = {}
    for name in (
        "actor", "protocol", "manifest", "designation", "selected",
        "development", "rows", "legacy_rows", "expansion_rows", "retired",
        "implicit",
    ):
        path = tmp_path / (name + (".npz" if "rows" in name else ".json"))
        path.write_bytes(b"input\n")
        inputs[name] = path
    (tmp_path / "report.json").write_bytes(b"input\n")
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REGISTRY_SHA256",
        subject.file_hash(inputs["development"]))
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REPORT_SHA256",
        subject.file_hash(tmp_path / "report.json"))
    monkeypatch.setattr(
        subject, "_claim_receipt", lambda *args, **kwargs: (claim_dir, {}))
    monkeypatch.setattr(subject, "producer_sources", lambda: {"source.py": "a" * 64})
    monkeypatch.setattr(
        subject, "_implicit_input_paths",
        lambda **kwargs: {
            "manifest_validation": inputs["implicit"],
            **{
                "designation_component:" + name: inputs["implicit"]
                for name in subject.designation_api.ARTIFACT_NAMES
            },
        })

    class Snapshot:
        def __init__(self, *args, **kwargs):
            self.paths = dict(args[0])
            self.original_paths = dict(args[0])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return None
        def verify(self):
            return None

    class PrivateInterrupt(BaseException):
        pass

    raw_salt = b"private-salt-material-0123456789"
    private_fragments = (
        raw_salt.hex(), "private-fresh-fingerprint",
        "private-fresh-seed-918273", "private-observation-deadbeef",
    )
    payload = " | ".join(private_fragments)
    monkeypatch.setattr(
        subject.input_snapshot_api, "ImmutableInputSnapshot", Snapshot)
    monkeypatch.setattr(subject, "_read_committed_salt", lambda _path: raw_salt)

    def guard(*args, phase, **kwargs):
        if phase == "private salt authentication":
            if interrupt:
                raise PrivateInterrupt(payload)
            raise RuntimeError(payload)

    monkeypatch.setattr(subject, "_guard_frozen_holdout_inputs", guard)
    expected = (
        subject._HistoricalFinalPrivatePhaseInterrupt
        if interrupt else subject._HistoricalFinalPrivatePhaseError)
    with pytest.raises(expected) as raised:
        subject.build(
            actor_path=inputs["actor"], protocol_path=inputs["protocol"],
            manifest_path=inputs["manifest"],
            designation_path=inputs["designation"],
            selected_scenes_path=inputs["selected"],
            development_registry_paths=[inputs["development"]],
            development_rows_paths=[inputs["rows"]],
            legacy_v3_rows_path=inputs["legacy_rows"],
            expansion_rows_path=inputs["expansion_rows"],
            retired_identity_projection_path=inputs["retired"],
            output=tmp_path / "holdout", claim_receipt_path=candidate,
            expected_claim_sha256="1" * 64,
            expected_campaign_key="2" * 64,
            expected_candidate_identity_sha256="3" * 64,
            selection_salt_path=tmp_path / "private-salt",
        )
    _assert_private_material_absent(
        raised.value, private_fragments, subject.__name__)
    assert not hasattr(subject, "_build_sensitive")
    assert not hasattr(
        subject,
        "_build_claimed_historical_final_observation_exclusion_sensitive")


def test_active_holdout_source_has_no_raw_retired_reader_or_artifact_path():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "_retired_registries" not in source
    assert "retired_holdout_paths" not in source
    assert "r41_diagnostic_fresh_final_holdout_v1_20260912" not in source
    assert "r41_diagnostic_fresh_final_holdout_v2_20260912" not in source
    assert "_projected_exposure_closure" not in source
    assert "manifest_binding._validate_rows(" not in source
    historical_source = inspect.getsource(
        subject._build_claimed_historical_final_observation_exclusion)
    historical_source += inspect.getsource(subject._historical_public_scene)
    assert 'scene["workload_screen"]' not in historical_source
