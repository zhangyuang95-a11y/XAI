"""Synthetic tensor/authorization tests; no Warehouse environment is created."""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from backend.training import warehouse_v625_actor_training as p


def test_registered_configuration_and_exact_boundaries():
    config = p.PPOConfig()
    assert config.extraction_steps == (15000, 16800, 18600)
    assert config.next_collection_length(3992) == 1
    assert config.next_collection_length(14992) == 1
    assert config.next_collection_length(20000) == 0
    assert config.program_coefficient(16000) == 0
    assert config.program_coefficient(17000) == pytest.approx(.01)
    for budget in (200000, 300000, 400000):
        other = p.PPOConfig(joint_step_budget=budget)
        assert all(step % 8 == 0 for step in other.extraction_steps)
    with pytest.raises(ValueError, match="hyperparameter"):
        p.PPOConfig(actor_learning_rate=3.e-4)
    with pytest.raises(ValueError, match="budget"):
        p.PPOConfig(joint_step_budget=4000)


def test_reward_gae_bootstraps_horizon_but_stops_recursion_at_reset():
    rewards = np.ones((2, 1, 2))
    values = np.zeros_like(rewards)
    next_values = np.zeros_like(rewards)
    next_values[0] = 2
    terminated, truncated = np.array([[False], [False]]), np.array([[True], [False]])
    advantage, returns = p.reward_gae(rewards, values, next_values, terminated, truncated)
    assert advantage[0, 0, 0] == pytest.approx(1 + .99 * 2)
    assert advantage[1, 0, 0] == pytest.approx(1)
    np.testing.assert_array_equal(advantage, returns)
    terminated[0, 0], truncated[0, 0] = True, False
    assert p.reward_gae(rewards, values, next_values, terminated, truncated)[0][0, 0, 0] == 1


def test_complete_true_shutdown_targets_use_whole_episode_and_no_discount():
    rows = [{"shutdown": False, "cost_value": .3}, {"shutdown": True, "cost_value": .4}]
    result, outcome = p.complete_episode_cost(rows, terminated=True, truncated=False)
    assert outcome is True
    assert [r["cost_return"] for r in result] == [1, 1]
    assert [r["cost_advantage"] for r in result] == pytest.approx([.7, .6])
    assert all(r["episode_length"] == 2 for r in result)
    with pytest.raises(ValueError, match="completed"):
        p.complete_episode_cost(rows[:1], terminated=False, truncated=False)
    with pytest.raises(ValueError, match="physical horizon"):
        p.complete_episode_cost(rows[:1], terminated=False, truncated=True)


def test_full_safe_horizon_is_zero_but_partial_safe_fragment_is_not():
    rows = [{"shutdown": False, "cost_value": .1}] * 120
    complete, outcome = p.complete_episode_cost(rows, terminated=False, truncated=True)
    assert outcome is False and all(r["cost_return"] == 0 for r in complete)
    with pytest.raises(ValueError):
        p.complete_episode_cost(rows[:119], terminated=False, truncated=True)


def test_missing_or_wrong_capability_never_reaches_a_constructor(tmp_path):
    factory_calls = []
    with pytest.raises(ValueError, match="verified"):
        p.PPOContinuationDriver(None, None, None, make_environment=lambda i: factory_calls.append(i),
            reset_environment=None, observe=None, stream_seed=1, variant="raw")
    file = tmp_path / "capability.json"
    file.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        p.HashBoundTrainingCapability(file, "0" * 64, root=tmp_path, state=None,
            variant="raw", config=p.PPOConfig(), journal_path=tmp_path / "journal")
    assert factory_calls == []
    assert not (tmp_path / "journal").exists()


def test_authority_predebits_constructors_resets_and_rollout_before_use(tmp_path):
    path = tmp_path / "capability.json"
    path.write_text("{}")
    capability = object.__new__(p.HashBoundTrainingCapability)
    capability.path, capability.sha256 = path, sha256(path.read_bytes()).hexdigest()
    capability.config = p.PPOConfig()
    capability.constructors, capability.resets, capability.reserved_steps = set(), [0] * 8, 0
    capability.cases = {0: [{"id": "train-0", "seed": 12, "task": 1, "slot": 0, "ordinal": 0, "partition": "train"}]}
    capability.journal_path = tmp_path / "journal.jsonl"
    capability.journal = capability.journal_path.open("x")
    capability.before_constructor(0)
    assert json.loads(capability.journal_path.read_text().splitlines()[-1])["kind"] == "construct"
    assert capability.before_reset(0)["id"] == "train-0"
    capability.reserve_rollout(8)
    with pytest.raises(RuntimeError, match="constructor"):
        capability.before_constructor(0)
    with pytest.raises(RuntimeError, match="schedule"):
        capability.before_reset(0)
    with pytest.raises(RuntimeError, match="joint-step"):
        capability.reserve_rollout(20000)
    path.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="changed"):
        capability.reserve_rollout(8)
    capability.close()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.actors = nn.ModuleDict({r: nn.Linear(p.FEATURE_DIM, 5) for r in p.ROLES})
        self.reward_critic = nn.Linear(2 * p.OBSERVATION_DIM, 2)
        self.cost_critic = nn.Linear(2 * p.OBSERVATION_DIM, 1)
    def outputs_encoded(self, encoded, mask):
        return torch.stack([self.actors[r](encoded[:, i]) for i, r in enumerate(p.ROLES)], dim=1).masked_fill(mask <= .5, -1.e9)


def tensor_hashes(model):
    return {r: sha256(b"".join(value.detach().numpy().tobytes() for value in model.actors[r].parameters())).hexdigest() for r in p.ROLES}


def synthetic_driver(monkeypatch, *, variant="raw", warmup=False):
    torch.manual_seed(72)
    model = TinyModel()
    frozen = deepcopy(model)
    roles = {}
    for role in p.ROLES:
        linear = frozen.actors[role]
        roles[role] = SimpleNamespace(logits_encoded=lambda x, linear=linear: linear(torch.as_tensor(x)).detach().numpy())
    reference = SimpleNamespace(roles=roles, role_sha256=tensor_hashes(model))
    actor_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3.e-5)
    critic_optimizer = torch.optim.Adam(list(model.reward_critic.parameters()) + list(model.cost_critic.parameters()), lr=3.e-4)
    end = 1024 if warmup else (17000 if variant == "regularized" else 4032)
    state = p.TrainingState(model, reference, "not-a-production-model", "a" * 64, 12,
        actor_optimizer, critic_optimizer, {}, joint_steps=end)
    driver = object.__new__(p.PPOContinuationDriver)
    driver.state, driver.config, driver.variant = state, p.PPOConfig(), variant
    driver.rng = np.random.default_rng(54)
    driver.risk, driver.updates = p.ShutdownRiskDual(), []
    driver.capability = SimpleNamespace(sha256="b" * 64, event=lambda value: None)
    driver.program_feedback_disabled, driver.program_feedback_disabled_since, driver.last_fidelity_failure = False, None, None
    driver.extractions, driver.extraction_attempts = [], []
    driver.program_lock = {role: {"structure_sha256": "f" * 64, "supports": [[]], "witnesses": [4]} for role in p.ROLES}
    encoded = np.random.default_rng(87).normal(0, .1, size=(32, 2, p.FEATURE_DIM)).astype(np.float32)
    masks = np.ones((32, 2, 5), np.float32)
    driver.reference_buffer = p.ReferenceTrainingBuffer(encoded.copy(), masks.copy(), tuple(f"teacher-{i}" for i in range(32)), "c" * 64)
    rows = []
    with torch.no_grad():
        logits = model.outputs_encoded(torch.as_tensor(encoded), torch.as_tensor(masks))
        logp = torch.log_softmax(logits, -1)
        actions = logits.argmax(-1)
        chosen = logp.gather(-1, actions[..., None]).squeeze(-1).numpy()
    for i in range(32):
        rows.append({"obs": np.random.default_rng(i).normal(0, .1, size=(2, p.OBSERVATION_DIM)).astype(np.float32),
            "encoded": encoded[i], "mask": masks[i], "actions": actions[i].numpy(), "old_logp": chosen[i],
            "reward_advantage": np.array([i / 32, -i / 32], np.float32),
            "reward_return": np.array([.2, -.3], np.float32), "episode_id": f"current-{i}"})
    monkeypatch.setattr(p, "role_hashes", tensor_hashes)
    batch = {"start_joint_step": 0 if warmup else end - 32, "end_joint_step": end,
        "rows": rows, "complete_cost_rows": []}
    if variant == "regularized":
        class Program:
            feedback_leaves = frozenset({0})
            def __init__(self, role):
                self.role = role
            def sha256(self):
                return ("d" if self.role == p.ROLES[0] else "e") * 64
            def distributions(self, x, m):
                with torch.no_grad():
                    raw = frozen.actors[self.role](torch.as_tensor(x)) / .8
                    q = torch.softmax(raw, -1)
                return np.zeros(len(x), int), raw.numpy(), q.numpy()
        driver.programs = {r: Program(r) for r in p.ROLES}
        driver.validator = lambda programs, current, epoch: {"actor_role_sha256": tensor_hashes(current.model),
            "program_sha256": {r: programs[r].sha256() for r in p.ROLES}, "independent_validation": True,
            "no_train_scene_overlap": True, "passed": {r: True for r in p.ROLES}}
    return driver, batch


def test_warmup_updates_only_critics_and_does_not_change_beta(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, warmup=True)
    before, beta = tensor_hashes(driver.state.model), dict(driver.state.beta)
    report = driver.update(batch)
    assert driver.state.actor_updates == 0
    assert driver.state.critic_updates == 4
    assert tensor_hashes(driver.state.model) == before
    assert driver.state.beta == beta
    assert report["partial_episodes_labeled_negative"] == 0


def test_raw_actual_ppo_updates_two_actors_with_zero_program_gradients(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch)
    before = tensor_hashes(driver.state.model)
    report = driver.update(batch)
    assert driver.state.actor_updates == 4
    assert all(tensor_hashes(driver.state.model)[r] != before[r] for r in p.ROLES)
    assert report["program_gradient_updates"] == {r: 0 for r in p.ROLES}
    assert all(update["program_loss"] == 0 for epoch in report["epochs"] for update in epoch["updates"])
    assert all(update["program_gradient"][r]["l2"] == 0 for epoch in report["epochs"] for update in epoch["updates"] for r in p.ROLES)


def test_regularized_actual_program_gradients_reach_both_actors(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    report = driver.update(batch)
    assert all(report["program_gradient_updates"][r] > 0 for r in p.ROLES)
    assert driver.state.actor_updates == 4
    assert all(update["program_gradient"][r]["coefficient"] == pytest.approx(.01)
        for epoch in report["epochs"] for update in epoch["updates"] for r in p.ROLES)
    assert all(update["program_gradient"][r]["accepted_optimizer_step"]
        and update["program_gradient"][r]["accepted_positive_program_gradient"]
        and update["program_gradient"][r]["role_weight_delta_l2"] > 0
        for epoch in report["epochs"] for update in epoch["updates"] for r in p.ROLES)


def test_program_validation_rejects_missing_current_evidence(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    driver.validator = lambda *args: {"passed": {r: True for r in p.ROLES}}
    with pytest.raises(RuntimeError, match="per-epoch"):
        driver.update(batch)
    assert driver.state.actor_updates == 0


def test_program_validation_cannot_mutate_the_actor(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    def bad(*args):
        with torch.no_grad():
            next(driver.state.model.actors.parameters()).add_(1)
        return {}
    driver.validator = bad
    with pytest.raises(RuntimeError, match="changed the current NN"):
        driver.update(batch)


def test_failed_fidelity_disables_feedback_but_ppo_reference_continue(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    calls = []
    original = driver.validator
    def failed(programs, state, epoch):
        calls.append(epoch)
        result = original(programs, state, epoch)
        result["passed"][p.ROLES[0]] = False
        return result
    driver.validator = failed
    before = tensor_hashes(driver.state.model)
    report = driver.update(batch)
    assert driver.state.actor_updates == driver.state.critic_updates == 4
    assert all(tensor_hashes(driver.state.model)[r] != before[r] for r in p.ROLES)
    assert report["program_gradient_updates"] == {r: 0 for r in p.ROLES}
    assert all(update["reference_loss"] >= 0 and update["program_loss"] == 0
        for epoch in report["epochs"] for update in epoch["updates"])
    assert calls == [0]
    assert driver.program_feedback_disabled
    assert driver.last_fidelity_failure["passed"][p.ROLES[0]] is False
    # A later epoch cannot rescue the same programs by returning a better gate.
    driver.validator = original
    assert driver._validated_programs(9) is None
    assert driver.extraction_attempts == []


def test_positive_gradient_probe_is_not_counted_when_no_weights_move(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    monkeypatch.setattr(driver.state.actor_optimizer, "step", lambda *args, **kwargs: None)
    report = driver.update(batch)
    assert driver.state.actor_updates == 4
    assert report["program_gradient_updates"] == {r: 0 for r in p.ROLES}
    for epoch in report["epochs"]:
        for update in epoch["updates"]:
            for role in p.ROLES:
                audit = update["program_gradient"][role]
                assert audit["l2"] > 0
                assert audit["accepted_optimizer_step"]
                assert not audit["role_weights_changed"]
                assert not audit["accepted_positive_program_gradient"]


def locked_program_fixture(driver):
    structure = {"root_ref": -1, "split_weights": np.zeros((0, p.FEATURE_DIM), np.float32),
        "split_bias": [], "left_refs": [], "right_refs": [], "leaf_count": 1, "feature_dim": p.FEATURE_DIM}
    class Program:
        def __init__(self, role):
            self.role = role
            self.actor_role_sha256 = tensor_hashes(driver.state.model)[role]
            self.structure = deepcopy(structure)
            self.leaves = [{"support": [p.FEATURE_NAMES.index("self_row")], "witness": 4}]
        def validate(self):
            return self
        def sha256(self):
            return ("d" if self.role == p.ROLES[0] else "e") * 64
    programs = {role: Program(role) for role in p.ROLES}
    lock = {role: {"structure_sha256": p.program_structure_sha256(programs[role].structure),
        "supports": [deepcopy(leaf["support"]) for leaf in programs[role].leaves],
        "witnesses": [leaf["witness"] for leaf in programs[role].leaves]} for role in p.ROLES}
    return programs, lock


def test_next_scheduled_extraction_reopens_feedback_only_after_lock_check(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    programs, driver.program_lock = locked_program_fixture(driver)
    driver.state.joint_steps = 18600
    driver.program_feedback_disabled = True
    driver.program_feedback_disabled_since = 17000
    calls = []
    def extractor(step, state, capability):
        calls.append(step)
        return programs
    driver.extractor = extractor
    driver._extract_if_due()
    assert calls == [18600]
    assert driver.extractions == driver.extraction_attempts == [18600]
    assert not driver.program_feedback_disabled
    assert driver.program_feedback_disabled_since is None
    driver._extract_if_due()
    assert calls == [18600]


@pytest.mark.parametrize("changed", ["topology", "support", "witness"])
def test_extraction_cannot_change_b_locked_language(monkeypatch, changed):
    driver, batch = synthetic_driver(monkeypatch, variant="regularized")
    programs, driver.program_lock = locked_program_fixture(driver)
    program = programs[p.ROLES[0]]
    if changed == "support":
        program.leaves[0]["support"] = [p.FEATURE_NAMES.index("self_column")]
    elif changed == "witness":
        program.leaves[0]["witness"] = 0
    else:
        weights = np.zeros((1, p.FEATURE_DIM), np.float32)
        weights[0, p.FEATURE_NAMES.index("self_row")] = 1
        program.structure.update(root_ref=0, split_weights=weights, split_bias=[-.5],
            left_refs=[-1], right_refs=[-2], leaf_count=2)
    driver.state.joint_steps = 18600
    driver.program_feedback_disabled = True
    calls = []
    def extractor(*args):
        calls.append(1)
        return programs
    driver.extractor = extractor
    with pytest.raises(RuntimeError, match="B-locked"):
        driver._extract_if_due()
    assert driver.program_feedback_disabled
    assert driver.extractions == []
    # Failed extraction consumes its attempt and cannot be retried for rescue.
    with pytest.raises(RuntimeError, match="retried"):
        driver._extract_if_due()
    assert calls == [1]


def test_reference_mix_is_exact_half_and_train_only(monkeypatch):
    driver, batch = synthetic_driver(monkeypatch)
    encoded, masks, provenance = driver._reference_mix(batch["rows"])
    assert len(provenance) == 32
    assert sum(r.source == "reference_train" for r in provenance) == 16
    assert sum(r.source == "current_train" for r in provenance) == 16
    assert all(r.split == "train" for r in provenance)


def test_snapshot_converts_numpy_to_safe_weights_only_primitives():
    value = {"array": np.array([1, 2], np.int64), "nested": [{"x": np.float32(2)}]}
    converted = p._checkpoint_primitives(value)
    stream = io.BytesIO()
    torch.save(converted, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    assert torch.equal(loaded["array"], torch.tensor([1, 2]))
    with pytest.raises(TypeError, match="safe primitives"):
        p._checkpoint_primitives(object())
