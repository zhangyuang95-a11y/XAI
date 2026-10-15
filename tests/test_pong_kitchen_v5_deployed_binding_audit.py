"""Exercise the shipped Pong/Kitchen v5 pairs, not replacement test Actors."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from domains.kitchen import neural_domain as kitchen
from domains.kitchen import neural_policy as kitchen_policy
from domains.pong import nn_policy as pong_policy
from domains.pong import turnbased as pong
from study_v3.policies import policy_manifest
from study_v3.release_catalog import policy_artifacts as release_policy_artifacts


ROOT = Path(__file__).resolve().parents[1]


def _digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("domain", ("pong", "kitchen"))
@pytest.mark.parametrize("variant", ("raw", "regularized"))
def test_shipped_v5_pair_matches_accepted_receipt(domain, variant):
    """Enrollment's accepted pair and its files have the same immutable identity."""
    # The active v6 study binds content-addressed copies of the accepted v5
    # assets. Their hashes must still match the original v5 acceptance receipt.
    catalog = release_policy_artifacts(domain)
    entry = catalog[variant]
    receipt = json.loads((ROOT / entry["acceptance_path"]).read_text())
    binding = policy_manifest()[domain][variant]
    assert receipt["passed"] is True
    assert receipt["heldout_evaluated"] is True
    assert binding["accepted"] is True
    assert binding["policy_id"] == entry["policy_id"]
    assert _digest(entry["actor_path"]) == entry["actor_sha256"] == receipt["artifact_sha256"][variant]
    assert (_digest(entry["training_report_path"])
            == entry["training_report_sha256"]
            == receipt["training_report_sha256"][variant])
    assert binding["human_actor_sha256"] == entry["role_sha256"]["human"]
    assert binding["ai_actor_sha256"] == entry["role_sha256"]["ai"]
    assert entry["role_sha256"]["human"] != entry["role_sha256"]["ai"]
    if domain == "pong":
        payload = json.loads((ROOT / entry["actor_path"]).read_text())
        for role in ("human", "ai"):
            assert pong_policy._role_digest(payload["actors"][role]) == entry["role_sha256"][role]
    else:
        actor = kitchen_policy.FrozenActors(ROOT / entry["actor_path"])
        assert actor.role_sha256 == entry["role_sha256"]
    expected_paths = 2 if variant == "regularized" else 0
    assert len(entry["program_paths"]) == expected_paths
    for role, path in zip(("human", "ai"), entry["program_paths"]):
        pinned = (entry["program_sha256"][path] if domain == "pong"
                  else entry["program_sha256"][role])
        assert _digest(path) == pinned == receipt["program_sha256"][variant][role]
        assert binding["program_hashes"][path] == pinned


@pytest.mark.parametrize("variant", ("raw", "regularized"))
def test_shipped_pong_actions_and_ai_ai_demo_use_the_bound_pair(variant):
    policy_id = f"pong-{variant}-v5"
    state = pong.initial_state(880101, 1, policy_id=policy_id)
    decision = pong.decide(state)
    assert decision["action"] == pong_policy.choose(state, "ai", policy_id)
    assert decision["human_action"] == pong_policy.choose(state, "human", policy_id)
    assert decision["action"] in pong.legal_actions(state, "ai")
    next_state = pong.step(state, "wait", decision)
    assert next_state["ai"]["x"] - state["ai"]["x"] == pong._DELTAS[decision["action"]]
    assert next_state["actor_sha256"] == state["actor_sha256"]
    forged = dict(decision, action="left" if decision["action"] != "left" else "right")
    with pytest.raises(ValueError, match="does not match"):
        pong.step(state, "wait", forged)
    frames = pong.demonstration(policy_id)["frames"]
    assert len(frames) > 2
    for before, after in zip(frames, frames[1:]):
        assert before["policy_id"] == after["policy_id"] == policy_id
        assert before["actor_sha256"] == after["actor_sha256"] == state["actor_sha256"]
        for role in ("human", "ai"):
            action = pong_policy.choose(before, role, policy_id)
            assert after[role]["x"] - before[role]["x"] == pong._DELTAS[action]


@pytest.mark.parametrize("variant", ("raw", "regularized"))
def test_shipped_kitchen_actions_and_ai_ai_demo_use_the_bound_pair(variant):
    policy_id = f"kitchen-{variant}-v5"
    state = kitchen.initial_state(1000, 1, policy_id=policy_id)
    actor = kitchen._actor(policy_id)
    decision = kitchen.decide(state)
    assert decision["action_id"] == actor.choose(state, "ai")[0]
    assert decision["action_id"] in kitchen.legal_actions(state, "ai")
    assert kitchen.human_advisor(state) == actor.choose(state, "human")[0]
    assert kitchen.step(state, "wait", decision)["policy_sha256"] == actor.sha256
    forged = dict(decision, action_id="wait" if decision["action_id"] != "wait" else "left")
    with pytest.raises(ValueError, match="does not match"):
        kitchen.step(state, "wait", forged)
    frames = kitchen.demonstration(policy_id)["frames"]
    assert len(frames) > 2
    assert kitchen.public_state(state) == frames[0]
    for expected in frames[1:]:
        human_action = actor.choose(state, "human")[0]
        ai_decision = kitchen.decide(state)
        assert ai_decision["action_id"] == actor.choose(state, "ai")[0]
        state = kitchen.step(state, human_action, ai_decision)
        assert kitchen.public_state(state) == expected


@pytest.mark.parametrize("failure", ("missing", "hash_mismatch"))
def test_pong_v5_rejects_missing_or_hash_mismatched_actor_before_state(tmp_path, monkeypatch, failure):
    entry = deepcopy(pong_policy.policy_artifacts()["raw"])
    actor_path = tmp_path / "actor.json"
    if failure == "hash_mismatch":
        actor_path.write_bytes((ROOT / entry["actor_path"]).read_bytes())
    entry["actor_path"] = actor_path.name
    entry["actor_sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({
        "format": "pong-paired-policies-v5", "variants": {"raw": entry}}))
    monkeypatch.setattr(pong_policy, "ROOT", tmp_path)
    monkeypatch.setattr(pong_policy, "V5_MANIFEST_PATH", tmp_path / "manifest.json")
    pong_policy._cached_bound_payload.cache_clear()
    with pytest.raises(RuntimeError, match=("actor is missing" if failure == "missing"
                                            else "actor hash mismatch")):
        pong.initial_state(880101, 1, policy_id="pong-raw-v5")
    pong_policy._cached_bound_payload.cache_clear()


@pytest.mark.parametrize("failure", ("missing", "hash_mismatch"))
def test_kitchen_v5_rejects_missing_or_hash_mismatched_actor_before_state(tmp_path, monkeypatch, failure):
    entry = deepcopy(kitchen.policy_artifacts()["raw"])
    actor_path = tmp_path / "actor.npz"
    if failure == "hash_mismatch":
        actor_path.write_bytes((ROOT / entry["actor_path"]).read_bytes())
    entry["actor_path"] = str(actor_path)
    entry["actor_sha256"] = "0" * 64
    monkeypatch.setattr(kitchen, "_v5_deployment", lambda _: entry)
    with pytest.raises(RuntimeError, match=("frozen Actor is unavailable" if failure == "missing"
                                            else "frozen Actor checksum mismatch")):
        kitchen.initial_state(1000, 1, policy_id="kitchen-raw-v5")


def _temporary_raw_binding(domain, tmp_path, monkeypatch):
    if domain == "pong":
        entry = deepcopy(pong_policy.policy_artifacts()["raw"])
        release_key, release = "release", pong_policy.V5_RELEASE_ID
    else:
        entry = deepcopy(kitchen.policy_artifacts()["raw"])
        release_key, release = "release_id", kitchen.V5_RELEASE_ID
    for field in ("actor_path", "training_report_path"):
        source = ROOT / entry[field]
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes())
        entry[field] = target.name
    entry["acceptance_path"] = "acceptance.json"
    receipt = {
        release_key: release, "passed": True, "heldout_evaluated": True,
        "artifact_sha256": {"raw": entry["actor_sha256"]},
        "training_report_sha256": {"raw": entry["training_report_sha256"]},
    }
    receipt_path = tmp_path / entry["acceptance_path"]
    receipt_path.write_text(json.dumps(receipt))
    if domain == "pong":
        manifest = {"format": "pong-paired-policies-v5", "variants": {"raw": entry}}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr(pong_policy, "ROOT", tmp_path)
        monkeypatch.setattr(pong_policy, "V5_MANIFEST_PATH", manifest_path)
        pong_policy._cached_bound_payload.cache_clear()
    else:
        monkeypatch.setattr(kitchen, "ROOT", tmp_path)
        monkeypatch.setattr(kitchen, "policy_artifacts", lambda: {"raw": entry})
        kitchen_policy.load_frozen.cache_clear()
    return receipt_path, receipt


@pytest.mark.parametrize("domain", ("pong", "kitchen"))
@pytest.mark.parametrize("failure", ("missing", "rebound", "not_approved"))
def test_direct_v5_calls_fail_closed_when_acceptance_is_missing_or_changed(
        domain, failure, tmp_path, monkeypatch):
    receipt_path, receipt = _temporary_raw_binding(domain, tmp_path, monkeypatch)
    module = pong if domain == "pong" else kitchen
    policy_id = f"{domain}-raw-v5"
    state = module.initial_state(880101 if domain == "pong" else 1000, 1, policy_id)
    if failure == "missing":
        receipt_path.unlink()
        error = "acceptance receipt is missing"
    else:
        if failure == "rebound":
            receipt["artifact_sha256"]["raw"] = "0" * 64
            error = "acceptance and policy manifest disagree"
        else:
            receipt["passed"] = False
            error = "acceptance receipt is not approved"
        receipt_path.write_text(json.dumps(receipt))
    for operation in (
        lambda: module.initial_state(880101 if domain == "pong" else 1000, 1, policy_id),
        lambda: module.decide(state),
        lambda: module.step(state, "wait"),
        lambda: module.demonstration(policy_id),
    ):
        with pytest.raises(RuntimeError, match=error):
            operation()
    pong_policy._cached_bound_payload.cache_clear()
    kitchen_policy.load_frozen.cache_clear()


def test_legacy_unbound_domain_entrypoints_remain_available():
    pong_state = pong.initial_state(880101, 1)
    kitchen_state = kitchen.initial_state(1000, 1)
    assert pong_state.get("policy_id") is None
    assert kitchen_state.get("policy_id") is None
    assert pong.decide(pong_state)["action"] in pong.legal_actions(pong_state, "ai")
    assert kitchen.decide(kitchen_state)["action_id"] in kitchen.legal_actions(kitchen_state, "ai")
