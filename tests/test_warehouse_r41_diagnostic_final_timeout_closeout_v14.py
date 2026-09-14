from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import inspect

import pytest

from backend.training import warehouse_r41_diagnostic_final_timeout_closeout_public_v14 as public
from backend.training import warehouse_r41_diagnostic_final_timeout_closeout_v14 as subject
from backend.training.warehouse_native_common import digest
from scripts import build_warehouse_r41_diagnostic_final_timeout_closeout_v14 as cli


ROOT = Path(__file__).resolve().parents[1]


def _identity(index: int) -> dict:
    return {
        "batch_index": index // 720,
        "family_offset": index % 120,
        "family_id": f"conflict_family_{index % 6 + 1:02d}",
        "seed": 50_000_000 + index,
        "fingerprint": sha256(f"identity:{index}".encode()).hexdigest(),
    }


def _universe() -> dict:
    ranking = [_identity(index) for index in range(2160)]
    outer = ranking[:64]
    eligible = ranking[64:]
    evaluated = []
    for index, identity in enumerate(eligible[:64]):
        observation_hash = sha256(f"observation:{index}".encode()).hexdigest()
        evaluated.append({
            **identity,
            "evaluation_index": index,
            "projector_scene_index": 900_000 + index,
            "accepted": True,
            "accepted_scene_index": index,
            "row_count": 1,
            "ordered_observation_hashes_sha256": digest([observation_hash]),
            "unique_observation_count": 1,
            "unique_observation_hashes_sha256": digest([observation_hash]),
            "environment_steps": 1,
        })
    value = {
        "version": subject.VERSION + ".burned-candidate-universe.v1",
        "status": "all_v13_ranking_inputs_and_completed_prefix_retired",
        "ranking_input_identities": ranking,
        "ranking_input_count": len(ranking),
        "ranking_input_identities_sha256": digest(ranking),
        "historically_excluded_seeds": sorted(row["seed"] for row in outer),
        "historically_excluded_seed_count": len(outer),
        "historically_excluded_seeds_sha256": digest(sorted(
            row["seed"] for row in outer)),
        "historically_excluded_fingerprints": sorted(
            row["fingerprint"] for row in outer),
        "historically_excluded_fingerprint_count": len(outer),
        "historically_excluded_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in outer)),
        "projector_eligible_identities": eligible,
        "projector_eligible_identity_count": len(eligible),
        "projector_eligible_identities_sha256": digest(eligible),
        "v13_outer_identities": outer,
        "v13_outer_identity_count": len(outer),
        "v13_outer_identities_sha256": digest(outer),
        "completed_evaluated_prefix": evaluated,
        "completed_evaluated_prefix_count": len(evaluated),
        "completed_evaluated_prefix_sha256": digest(evaluated),
        "reconstructed_accepted_identities": eligible[:64],
        "reconstructed_accepted_identity_count": 64,
        "reconstructed_accepted_identities_sha256": digest(eligible[:64]),
        "all_ranking_input_identities_retired": True,
        "all_completed_prefix_identities_retired": True,
        "same_v13_candidate_universe_reuse_permitted": False,
        "scene_snapshots_included": False,
        "rng_state_or_salt_included": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _projection(universe: dict) -> dict:
    ordered = [sha256(f"observation:{index}".encode()).hexdigest()
               for index in range(64)]
    projector_sources = subject.materializer.final_projection_api.producer_sources()
    accepted_summaries = [{
        "local_scene_index": 0,
        "scene_index": 900_000 + index,
        "fingerprint": universe["reconstructed_accepted_identities"][index][
            "fingerprint"],
        "row_count": 1,
        "ordered_observation_hashes_sha256": digest([observation_hash]),
        "unique_observation_count": 1,
        "unique_observation_hashes_sha256": digest([observation_hash]),
        "accepted_scene_index": index,
    } for index, observation_hash in enumerate(ordered)]
    accepted_projection = {
        "version": public.EXPECTED_PROJECTOR_VERSION,
        "contract_sha256": public.EXPECTED_PROJECTOR_CONTRACT_SHA256,
        "producer_sources_sha256": public.EXPECTED_PROJECTOR_SOURCES_SHA256,
        "scene_offset": public.EXPECTED_FINAL_SCENE_OFFSET,
        "scene_count": 64,
        "partners": ["skilled", "assertive", "noisy"],
        "critical_anchor_period": 5,
        "dense_critical": False,
        "row_count": len(ordered),
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_count": len(set(ordered)),
        "unique_observation_hashes_sha256": digest(sorted(set(ordered))),
        "scene_summaries": accepted_summaries,
        "scene_summaries_sha256": digest(accepted_summaries),
        "environment_steps": 64,
        "prior_observation_overlap": 0,
        "within_final_observation_overlap": 0,
        "actions_read": False,
        "probabilities_read": False,
        "program_access": False,
        "labels_read": False,
    }
    accepted_projection["content_sha256"] = digest(accepted_projection)
    value = {
        "version": subject.VERSION + ".burned-observation-projection.v1",
        "status": "completed_v13_evaluated_prefix_hashes_permanently_excluded",
        "selection_algorithm_version": subject.materializer.VERSION,
        "selection_algorithm_source_closure_sha256": (
            public.EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256),
        "projector_version": subject.materializer.final_projection_api.VERSION,
        "projector_contract": subject.materializer.final_projection_api.contract(),
        "projector_contract_sha256": digest(
            subject.materializer.final_projection_api.contract()),
        "projector_sources": projector_sources,
        "projector_sources_sha256": digest(projector_sources),
        "completed_evaluated_prefix_count": 64,
        "accepted_scene_count": 64,
        "ordered_observation_hashes": ordered,
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_observation_hashes": sorted(set(ordered)),
        "unique_observation_hashes_sha256": digest(sorted(set(ordered))),
        "row_count": len(ordered),
        "unique_observation_count": len(set(ordered)),
        "environment_steps": 64,
        "frozen_accepted_projection": accepted_projection,
        "timing": {
            "elapsed_seconds": 1.25,
            "clock": "time.perf_counter",
            "process_model": "single_process_in_process_frozen_v13_selector",
            "parallel_workers": 1,
            "python_implementation": "CPython",
            "python_version": "3.13.0",
            "platform": "test",
            "logical_cpu_count": 1,
        },
        "information_boundary": {
            "closeout_claim_preceded_config_identity_and_salt_access": True,
            "burned_salt_used_only_for_exact_ranking_reconstruction": True,
            "salt_or_rng_state_included": False,
            "raw_observations_included": False,
            "actor_actions_included": False,
            "actor_probabilities_included": False,
            "program_file_or_predictions_accessed": False,
            "labels_or_rows_accessed_or_generated": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def test_exact_burned_attempt_identity_and_frozen_source_closures():
    assert len(subject.EXPECTED_V13_ATTEMPT_KEY) == 64
    assert public._HEX.fullmatch(subject.EXPECTED_V13_ATTEMPT_KEY)
    assert subject.EXPECTED_V13_ATTEMPT_KEY == public.EXPECTED_V13_ATTEMPT_KEY
    assert subject.contract() == public.contract()
    assert digest(subject.final_api.producer_sources()) == (
        subject.EXPECTED_V13_CONTROLLER_SOURCE_CLOSURE_SHA256)
    assert digest(subject.final_api._official_materializer_binding()[1]) == (
        subject.EXPECTED_V13_MATERIALIZER_SOURCE_CLOSURE_SHA256)


def test_real_two_file_failure_campaign_authenticates_without_salt():
    registry = ROOT / (
        "output/warehouse_native/"
        "r41_diagnostic_final_once_v13_permanent_registry_20260914")
    campaign = registry / subject.EXPECTED_V13_ATTEMPT_KEY
    if not campaign.is_dir():
        pytest.skip("local append-only v13 failure campaign is unavailable")
    result = subject._authenticate_timeout_failure(
        final_anchor_path=campaign / subject.final_api.ANCHOR_NAME,
        final_completion_path=campaign / subject.final_api.COMPLETION_NAME,
        permanent_final_registry=registry,
        failed_public_output=ROOT / (
            "output/warehouse_native/r41_diagnostic_final_once_v13_20260914"))
    assert result["anchor"]["attempt_key"] == subject.EXPECTED_V13_ATTEMPT_KEY
    assert {path.name for path in campaign.iterdir()} == {
        subject.final_api.ANCHOR_NAME, subject.final_api.COMPLETION_NAME}


def test_irrevocable_claim_precedes_every_postclaim_access(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry, output_parent = tmp_path / "registry", tmp_path / "output"
    registry.mkdir(); output_parent.mkdir()
    evidence = {
        "anchor": {"attempt_key": subject.EXPECTED_V13_ATTEMPT_KEY,
                   "content_sha256": "a" * 64,
                   "attempt_key_inputs": {
                       "candidate_lock_sha256": "1" * 64,
                       "actor_sha256": "2" * 64,
                       "protocol_sha256": "3" * 64,
                       "runtime_manifest_sha256": "4" * 64,
                       "program_sha256": "5" * 64,
                       "outer_result_sha256": "6" * 64,
                   }},
        "completion": {"content_sha256": "b" * 64},
    }
    monkeypatch.setattr(subject, "_authenticate_timeout_failure",
                        lambda **kwargs: evidence)
    observed = {}

    def reconstruct(*, claim_path, **kwargs):
        observed["claim_path"] = claim_path
        assert claim_path.is_file()
        assert {child.name for child in claim_path.parent.iterdir()} == {
            subject.CLAIM_NAME}
        return {"content_sha256": "c" * 64}, {
            "content_sha256": "d" * 64}, {}

    monkeypatch.setattr(subject, "_postclaim_reconstruct", reconstruct)
    def publish(**kwargs):
        destination = kwargs["destination"]
        destination.mkdir()
        (destination / subject.RECEIPT_NAME).write_text("fixture")
        return {"ok": True}

    monkeypatch.setattr(subject, "_publish", publish)
    monkeypatch.setattr(public, "read_saved_closeout_public",
                        lambda *args, **kwargs: {"ok": True})
    result = subject.build(
        final_anchor_path=tmp_path / "anchor",
        final_completion_path=tmp_path / "completion",
        permanent_final_registry=tmp_path,
        failed_public_output=tmp_path / "absent",
        materializer_config_path=tmp_path / "config",
        permanent_closeout_registry=registry,
        output=output_parent / "closeout")
    assert result == {"ok": True}
    assert observed["claim_path"].is_file()
    with pytest.raises(FileExistsError, match="already claimed"):
        subject._create_permanent_claim(
            evidence=evidence, permanent_closeout_registry=registry)


def test_tracer_executes_unmodified_selector_and_restores_projector(
        monkeypatch: pytest.MonkeyPatch):
    projection_api = subject.materializer.final_projection_api
    original = projection_api.project_observation_hashes
    identities = [_identity(100), _identity(101)]
    scenes = [{**row, "id": f"scene-{index}"}
              for index, row in enumerate(identities)]
    hashes = {
        row["fingerprint"]: sha256(row["fingerprint"].encode()).hexdigest()
        for row in identities}

    def projector(runtime, values, *, scene_offset, dense_critical=False):
        value = hashes[values[0]["fingerprint"]]
        return {
            "ordered_observation_hashes": [value],
            "ordered_observation_hashes_sha256": digest([value]),
            "unique_observation_count": 1,
            "unique_observation_hashes_sha256": digest([value]),
            "row_count": 1,
            "environment_steps": 1,
            "scenes": [{}],
        }

    monkeypatch.setattr(projection_api, "project_observation_hashes", projector)
    monkeypatch.setattr(projection_api, "validate_projection", lambda value: value)
    monkeypatch.setattr(projection_api, "producer_sources", lambda: {"x": "a" * 64})
    monkeypatch.setattr(subject, "ACCEPTED_SCENE_COUNT", 1)

    def selector(**kwargs):
        api = subject.materializer.final_projection_api
        first = api.project_observation_hashes(
            None, [scenes[0]], scene_offset=900_000)
        api.project_observation_hashes(None, [scenes[1]], scene_offset=900_000)
        return [scenes[0]], {
            "evaluated": 2,
            "projection": {
                "row_count": 1, "scene_count": 64,
                "ordered_observation_hashes_sha256": first[
                    "ordered_observation_hashes_sha256"],
                "unique_observation_hashes_sha256": first[
                    "unique_observation_hashes_sha256"],
            },
        }

    monkeypatch.setattr(subject.materializer, "_select_and_replay", selector)
    accepted, statistics, trace, elapsed = subject._traced_selection(
        prepared={
            "runtime": object(), "identities": identities,
            "scene_by_fingerprint": {}, "excluded_seeds": set(),
            "excluded_fingerprints": set(),
            "forbidden_observation_hashes": set(),
        }, salt=b"x" * 32)
    assert accepted == [scenes[0]]
    assert statistics["evaluated"] == len(trace) == 2
    assert elapsed >= 0
    assert projection_api.project_observation_hashes is projector
    monkeypatch.setattr(projection_api, "project_observation_hashes", original)


def test_producer_public_reader_roundtrip(tmp_path: Path):
    registry, output_parent = tmp_path / "registry", tmp_path / "output"
    registry.mkdir(); output_parent.mkdir()
    attempt_inputs = {
        "candidate_lock_sha256": "1" * 64,
        "actor_sha256": "2" * 64,
        "protocol_sha256": "3" * 64,
        "runtime_manifest_sha256": "4" * 64,
        "program_sha256": "5" * 64,
        "outer_result_sha256": "6" * 64,
    }
    evidence = {
        "anchor": {
            "attempt_key": subject.EXPECTED_V13_ATTEMPT_KEY,
            "content_sha256": subject.EXPECTED_V13_ANCHOR_CONTENT_SHA256,
            "attempt_key_inputs": attempt_inputs,
        },
        "completion": {
            "content_sha256": subject.EXPECTED_V13_COMPLETION_CONTENT_SHA256,
        },
    }
    campaign, claim, claim_raw = subject._create_permanent_claim(
        evidence=evidence, permanent_closeout_registry=registry)
    output = output_parent / "closeout"
    receipt = subject._publish(
        destination=output, campaign=campaign, claim=claim,
        claim_raw=claim_raw, evidence=evidence,
        universe=_universe(), projection=_projection(_universe()))
    saved = public.read_saved_closeout_public(
        output / subject.RECEIPT_NAME,
        expected_closeout_sha256=subject.file_hash(
            output / subject.RECEIPT_NAME),
        permanent_closeout_registry=registry)
    assert saved == receipt
    assert saved["bindings"]["actor_sha256"] == attempt_inputs["actor_sha256"]


def test_public_companion_schemas_reject_secret_or_incomplete_exposure():
    universe = _universe()
    checked = public._validate_universe(universe)
    projection = _projection(checked)
    assert public._validate_projection(projection, checked) == projection

    leaked = deepcopy(universe)
    leaked["rng_state_or_salt_included"] = True
    leaked["content_sha256"] = digest({
        key: value for key, value in leaked.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="universe semantics"):
        public._validate_universe(leaked)

    incomplete = deepcopy(projection)
    incomplete["accepted_scene_count"] = 63
    incomplete["content_sha256"] = digest({
        key: value for key, value in incomplete.items()
        if key != "content_sha256"})
    with pytest.raises(ValueError, match="projection differs"):
        public._validate_projection(incomplete, checked)

    nested_extra = deepcopy(projection)
    nested_extra["frozen_accepted_projection"]["actor_actions"] = ["WAIT"]
    nested_extra["frozen_accepted_projection"]["content_sha256"] = digest({
        key: value
        for key, value in nested_extra["frozen_accepted_projection"].items()
        if key != "content_sha256"
    })
    nested_extra["content_sha256"] = digest({
        key: value for key, value in nested_extra.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="projection differs"):
        public._validate_projection(nested_extra, checked)

    wrong_trace = deepcopy(projection)
    wrong_trace["ordered_observation_hashes"][0] = "f" * 64
    wrong_trace["ordered_observation_hashes_sha256"] = digest(
        wrong_trace["ordered_observation_hashes"])
    wrong_trace["unique_observation_hashes"] = sorted(set(
        wrong_trace["ordered_observation_hashes"]))
    wrong_trace["unique_observation_hashes_sha256"] = digest(
        wrong_trace["unique_observation_hashes"])
    wrong_trace["content_sha256"] = digest({
        key: value for key, value in wrong_trace.items()
        if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="projection differs"):
        public._validate_projection(wrong_trace, checked)


def test_cli_has_no_salt_program_or_rows_argument():
    options = {option for action in cli.parser()._actions
               for option in action.option_strings}
    assert {"--final-anchor", "--final-completion", "--materializer-config",
            "--permanent-final-registry", "--failed-public-output",
            "--permanent-closeout-registry", "--output"} <= options
    assert not any(any(token in option for token in ("salt", "program", "rows"))
                   for option in options)
    source = inspect.getsource(subject.build)
    assert source.index("_create_permanent_claim") < source.index(
        "_postclaim_reconstruct")
