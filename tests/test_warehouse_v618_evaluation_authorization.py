"""Frozen-only development capabilities and once-only sealed batch guards."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from backend.training.warehouse_v6_actor import (
    WarehouseV6Actor, FrozenWarehouseV6Actor, export_actor,
)
from backend.training.warehouse_v6_evaluation_authorization import (
    development_evaluation_authorization, is_sealed_scenario_seed,
    sealed_evaluation_authorization, validate_evaluation_authorization,
)
from experiments.warehouse import protocol as p


@pytest.fixture
def frozen_actor(tmp_path):
    artifact = export_actor(WarehouseV6Actor(), tmp_path / "actor.npz",
                            pair_seed=6150, variant="raw", candidate_id="unit-test")
    return FrozenWarehouseV6Actor.load(
        artifact["path"], expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])


def test_development_descriptor_is_bound_to_source_artifact_and_both_roles(frozen_actor):
    spec = development_evaluation_authorization(frozen_actor)
    result = validate_evaluation_authorization(actor=frozen_actor, seeds=[263100], authorization=spec)
    assert result["scenario_count"] == 1
    assert result["publishable"] is False
    for key, value in (("expected_source_closure_sha256", "0"*64),
                       ("authorization_module_sha256", "0"*64),
                       ("actor_sha256", "0"*64),
                       ("actor_role_sha256", {role: "0"*64 for role in p.ROLES})):
        bad = deepcopy(spec)
        bad[key] = value
        with pytest.raises(RuntimeError, match="binding mismatch"):
            validate_evaluation_authorization(actor=frozen_actor, seeds=[263100], authorization=bad)


@pytest.mark.parametrize("minor", [0, 1, 27])
def test_default_development_cannot_touch_any_versioned_sealed_seeds(frozen_actor, minor):
    release = f"policylens-three-domain-20261016.v6.{minor}"
    blocks = p.sealed_seed_blocks(release)
    for base in (blocks["final_program_audit"][0], blocks["heldout"][0]):
        for seed in (base, *(p.task_scenario_seed(base, task) for task in p.TASKS)):
            assert is_sealed_scenario_seed(seed)
            with pytest.raises(RuntimeError, match="sealed seeds"):
                validate_evaluation_authorization(actor=frozen_actor, seeds=[seed])


@pytest.mark.parametrize("seeds", [[], [1, 1], [True], [-1], [1.5]])
def test_empty_duplicate_or_noninteger_scenes_rejected(frozen_actor, seeds):
    with pytest.raises(ValueError, match="scenario seeds"):
        validate_evaluation_authorization(actor=frozen_actor, seeds=seeds)


def test_trainable_actor_and_in_memory_identity_cannot_be_authorized(frozen_actor):
    with pytest.raises(TypeError, match="frozen Actor"):
        validate_evaluation_authorization(actor=WarehouseV6Actor(), seeds=[263100])
    frozen_actor.artifact_sha256 = "in-memory"
    with pytest.raises(RuntimeError, match="artifact and dual-role hashes"):
        validate_evaluation_authorization(actor=frozen_actor, seeds=[263100])


def _claim(tmp_path, actor, phase):
    path = tmp_path / f"{phase}.claim.json"
    p.claim_once(path, phase=phase, candidate_id="nonpublishing-permission-test",
                 actor_hashes={"raw": actor.artifact_sha256, "regularized": "a"*64},
                 seed_rows=p.sealed_seed_blocks()[phase])
    spec = sealed_evaluation_authorization(
        actor, claim_path=path, expected_claim_sha256=sha256(path.read_bytes()).hexdigest())
    return path, spec


def test_sealed_descriptor_requires_complete_claimed_batch_and_is_spent_once(tmp_path, frozen_actor):
    path, spec = _claim(tmp_path, frozen_actor, "heldout")
    seeds = p.heldout_schedule()[1]
    with pytest.raises(RuntimeError, match="complete claimed batch"):
        validate_evaluation_authorization(actor=frozen_actor, seeds=seeds[:-1], authorization=spec)
    report = validate_evaluation_authorization(
        actor=frozen_actor, seeds=seeds, authorization=spec, consume_sealed=True)
    assert report["sealed_execution_claim_sha256"]
    with pytest.raises(FileExistsError):
        validate_evaluation_authorization(
            actor=frozen_actor, seeds=seeds, authorization=spec, consume_sealed=True)
    # Validation is read-only; consuming another complete task remains possible.
    validate_evaluation_authorization(actor=frozen_actor, seeds=seeds, authorization=spec)
    validate_evaluation_authorization(
        actor=frozen_actor, seeds=p.heldout_schedule()[2], authorization=spec, consume_sealed=True)
    path.write_text(json.dumps({"tampered": True}))
    with pytest.raises(RuntimeError, match="claim changed"):
        validate_evaluation_authorization(actor=frozen_actor, seeds=seeds, authorization=spec)


def test_program_audit_requires_exact_claim_actor_and_restores_backend_context(tmp_path, frozen_actor):
    from experiments.warehouse.admission import NativeBackend
    path, spec = _claim(tmp_path, frozen_actor, "final_program_audit")
    validate_evaluation_authorization(
        actor=frozen_actor, seeds=p.final_audit_seeds(), authorization=spec)
    backend = NativeBackend()
    with pytest.raises(RuntimeError, match="sentinel"):
        with backend.sealed_evaluation(
                claim_path=path, expected_claim_sha256=sha256(path.read_bytes()).hexdigest()):
            descriptor = backend._evaluation_authorization(frozen_actor)
            assert descriptor["purpose"] == "final_program_audit"
            raise RuntimeError("sentinel")
    assert backend._evaluation_authorization(frozen_actor)["purpose"] == "unsealed_development_diagnostic"


def test_real_native_development_rollout_and_independent_physics_replay(frozen_actor):
    from backend.training import warehouse_v6_runner as runner
    from experiments.warehouse.admission import NativeBackend
    from experiments.warehouse.quality_replay import replay_episode
    assert runner._ACTIVE_LOCKED_EXECUTION.get() is None
    result = NativeBackend().evaluate(frozen_actor, [263100], trace=True)
    assert runner._ACTIVE_LOCKED_EXECUTION.get() is None
    episode = result["episodes"][0]
    replayed = replay_episode(frozen_actor, episode, scenario_seed=263100, action_seed=263100)
    assert replayed["frame_count"] == episode["quality_trace"]["frame_count"]
    assert replayed["frame_count"] > 0
    assert result["evaluation_authorization"]["publishable"] is False
    with pytest.raises(RuntimeError, match="review-only"):
        runner.evaluate(frozen_actor, [263100])


def test_baseline_schedule_is_matched_and_never_sealed():
    from scripts.diagnose_warehouse_v618_baseline import diagnostic_schedule
    schedule = diagnostic_schedule(range(263100, 263124))
    assert len(schedule) == 216
    assert len({(row["task"], row["base_seed"], row["stream"]) for row in schedule}) == 216
    assert diagnostic_schedule(range(263100, 263124)) == schedule
    with pytest.raises(ValueError, match="sealed seeds"):
        diagnostic_schedule([p.HELDOUT_BASE_SEEDS[0]])
