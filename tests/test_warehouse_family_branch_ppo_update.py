"""Small synthetic CPU models/batches with real PPO, tree KL and dual Adam.

No environments, source checkpoint reads, production model or fitted tree.
The native transport harness is not a qualified or persistent trainer.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from backend.training import warehouse_family_branch_ppo_update as module
from backend.training.warehouse_native_continuation_feedback_trainer import ContinuationFeedbackTrainer
from backend.training.warehouse_native_partner_mix_trainer import PartnerMixTrainer
from backend.training.warehouse_native_public_feedback_trainer import PublicFeedbackTrainer, _global_rng
from backend.training.warehouse_native_v2 import IndependentOptimizers
from backend.training.warehouse_native_expanded_rcpd import ExactProgramManager
from core.program import ExecutableProgram, ProgramNode
from env.warehouse_native.feedback import FeedbackConfig
from env.warehouse_native.policy import NativeActorCritic, ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv

COUNTS = dict(synthetic_models=0, actor_forwards=0, critic_forwards=0,
    backward_calls=0, diagnostic_autograd_calls=0, adam_steps=0, component_calls=0,
    tree_targets_calls=0, environment_constructions=0, environment_steps=0,
    PT_loads=0, production_model_loads=0, MPS_calls=0, tree_fits=0)
LIMITS = dict(synthetic_models=60, actor_forwards=400, critic_forwards=300,
    backward_calls=600, diagnostic_autograd_calls=450, adam_steps=600,
    component_calls=60, tree_targets_calls=250)


@pytest.fixture(scope="module", autouse=True)
def counted_cpu_scope():
    patch = pytest.MonkeyPatch(); previous = torch.get_num_threads(); torch.set_num_threads(1)
    def wrap(owner, name, counter):
        original = getattr(owner, name)
        def call(*args, **kwargs):
            COUNTS[counter] += 1
            assert COUNTS[counter] <= LIMITS[counter], "Declared CPU fixture ceiling exceeded"
            return original(*args, **kwargs)
        patch.setattr(owner, name, call)
    for owner, name, counter in ((NativeActorCritic, "__init__", "synthetic_models"),
            (NativeActorCritic, "actor_logits", "actor_forwards"),
            (NativeActorCritic, "values", "critic_forwards"),
            (torch.Tensor, "backward", "backward_calls"),
            (torch.autograd, "grad", "diagnostic_autograd_calls"),
            (torch.optim.Adam, "step", "adam_steps"),
            (ExactProgramManager, "targets", "tree_targets_calls")):
        wrap(owner, name, counter)
    def forbidden(*args, **kwargs):
        raise AssertionError("Environment/PT operations are forbidden in this fixture scope")
    patch.setattr(NativeWarehouseEnv, "__init__", forbidden)
    patch.setattr(NativeWarehouseEnv, "step", forbidden)
    patch.setattr(torch, "load", forbidden)
    yield
    patch.undo(); torch.set_num_threads(previous)
    print("BRANCH_PPO_CPU_SCOPE=" + json.dumps(COUNTS, sort_keys=True))


class SyntheticNative:
    _rng_context = PublicFeedbackTrainer._rng_context
    update = PartnerMixTrainer.update

    def __init__(self, *, epochs=2, minibatch=4):
        self.device = torch.device("cpu")
        # CPU generator only: do not initialize MPS or CUDA RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(260910)
            self.model = NativeActorCritic(197, 354, hidden=7)
            self._owned_rng_state = _global_rng(self.device)
        self.cfg = dict(actor_learning_rate=3e-4, critic_learning_rate=7e-4,
            actor_gradient_norm=.5, critic_gradient_norm=1., epochs=epochs,
            minibatch=minibatch, clip=.2, entropy=.01)
        self.optimizers = IndependentOptimizers(self.model, self.cfg)
        self.rng = np.random.default_rng(411)
        self.source_counters = dict(actor_optimizer_steps=0, critic_optimizer_steps=0)
        self.joint_steps = 0
        self.minibatch_updates = self.optimizer_updates = 0
        self.actor_optimizer_steps = self.critic_optimizer_steps = 0

    def snapshot(self):
        return deepcopy(dict(model=self.model.state_dict(), optimizers=self.optimizers.state_dict(),
            rng=self.rng.bit_generator.state, owned_rng=self._owned_rng_state,
            counters=[self.joint_steps, self.minibatch_updates, self.optimizer_updates,
                      self.actor_optimizer_steps, self.critic_optimizer_steps]))

    def restore(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizers.load_state_dict(deepcopy(state["optimizers"]))
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self._owned_rng_state = deepcopy(state["owned_rng"])
        (self.joint_steps, self.minibatch_updates, self.optimizer_updates,
         self.actor_optimizer_steps, self.critic_optimizer_steps) = state["counters"]


def equal(a, b):
    if torch.is_tensor(a): assert torch.equal(a, b)
    elif isinstance(a, np.ndarray): assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b


def batch():
    rng = np.random.default_rng(55)
    return dict(observations=rng.normal(0, .2, (2,2,2,197)).astype(np.float32),
        states=rng.normal(0, .2, (2,2,354)).astype(np.float32),
        trainable=np.array([[[1,0],[1,1]],[[0,1],[1,1]]], np.float32),
        actions=np.array([[[0,4],[1,2]],[[4,3],[2,0]]], np.int64),
        old_log_probs=np.full((2,2,2), -np.log(5), np.float32),
        advantages=rng.normal(size=(2,2,2)).astype(np.float32),
        returns=rng.normal(size=(2,2,2)).astype(np.float32),
        transition_records=[{"scope":"synthetic transport only", "requested_actions": "unchanged"}])


def aux():
    return np.random.default_rng(91).normal(0, .3, (5,2,197)).astype(np.float32), np.array([1,2,0,4,1.], np.float32)


def manager(lam=.01):
    names = tuple("fixture_feature_"+str(i) for i in range(197))
    result = ExactProgramManager(names, FeedbackConfig())
    result.program = ExecutableProgram(ACTIONS, names,
        ProgramNode(probabilities=(.72,.07,.07,.07,.07)), {"test_fixture":True})
    result.reliable = True; result.current_lambda = lam
    return result


class OrdinaryView:
    def __init__(self, native, feedback): self.native, self.feedback = native, feedback
    def __getattr__(self, key): return getattr(self.native, key)


def ordinary(native, data, feedback):
    # Direct frozen ordinary algorithm as an independent reference transport.
    with native._rng_context():
        result = ContinuationFeedbackTrainer._feedback_update(OrdinaryView(native, feedback), data)
    for role in ("actor", "critic"):
        state = getattr(native.optimizers, role).state_dict()["state"]
        step = int(next(iter(state.values()))["step"].item()) if state else 0
        setattr(native, role+"_optimizer_steps", step-native.source_counters[role+"_optimizer_steps"])
    return result


def run(native, data, feedback, observations=None, weights=None, rng=None, **kwargs):
    COUNTS["component_calls"] += 1
    assert COUNTS["component_calls"] <= LIMITS["component_calls"]
    if observations is None: observations, weights = aux()
    return module.branch_ppo_update(native, data, feedback, observations, weights,
        np.random.default_rng(812) if rng is None else rng, **kwargs)


@pytest.mark.parametrize("kind", ["none", "zero_lambda"])
def test_disabled_is_exact_native_including_warm_adam_and_rng(kind):
    left = SyntheticNative(); data = batch(); left.update(data)
    right = SyntheticNative(); right.restore(left.snapshot())
    feedback = None if kind == "none" else manager(0.)
    rng = np.random.default_rng(99); saved_rng = deepcopy(rng.bit_generator.state)
    before = deepcopy(data); counts = deepcopy(COUNTS)
    got = run(left, data, feedback, rng=rng)
    native_actor_calls = COUNTS["actor_forwards"]-counts["actor_forwards"]
    expected = right.update(data)
    equal(left.snapshot(), right.snapshot()); equal(data, before); equal(rng.bit_generator.state, saved_rng)
    equal({k:v for k,v in got.items() if k != "branch_update"}, expected)
    assert native_actor_calls == 4 and got["branch_update"]["auxiliary_actor_forward_calls"] == 0


@pytest.mark.parametrize("kind", ["empty", "zero_weight", "zero_fraction"])
def test_empty_aux_is_exact_ordinary_feedback_not_plain_ppo(kind):
    left = SyntheticNative(); data = batch(); left.update(data)
    right = SyntheticNative(); right.restore(left.snapshot())
    obs, weights = aux(); fraction=.5
    if kind == "empty": obs, weights = np.empty((0,2,197),np.float32), np.empty(0,np.float32)
    if kind == "zero_weight": weights *= 0
    if kind == "zero_fraction": fraction=0.
    rng = np.random.default_rng(90); previous = deepcopy(rng.bit_generator.state)
    got = run(left, data, manager(), obs, weights, rng, branch_fraction=fraction)
    expected = ordinary(right, data, manager())
    equal(left.snapshot(), right.snapshot()); equal(rng.bit_generator.state, previous)
    equal({k:v for k,v in got.items() if k != "branch_update"}, expected)
    assert got["feedback_loss"] > 0 and got["branch_update"]["path"] == "ordinary_feedback_exact"
    assert got["branch_update"]["auxiliary_actor_forward_calls"] == 0


def test_mixed_gradients_share_single_adam_and_preserve_ppo_batch_and_critic(monkeypatch):
    native = SyntheticNative(); reference = SyntheticNative(); data = batch(); before = deepcopy(data)
    feedback = manager(); saved_manager = deepcopy(feedback.state_dict())
    ordinary_observations=[]; old=feedback.loss
    def observed(logits, obs):
        ordinary_observations.append(obs.copy()); return old(logits, obs)
    monkeypatch.setattr(feedback, "loss", observed)
    obs, weights = aux(); aux_before = obs.copy(); weights_before = weights.copy()
    result=run(native, data, feedback, obs, weights, auxiliary_pairs_per_minibatch=3)
    reference.update(data)
    audit=result["branch_update"]
    assert audit["actor_adam_steps"] == audit["critic_adam_steps"] == audit["minibatch_updates"] == 4
    assert audit["optimizer_updates"] == 1 and native.joint_steps == 0
    assert audit["ppo_nn_row_visits"] == audit["ordinary_tree_row_visits"] == 12
    assert audit["auxiliary_pairs_used"] == 12 and audit["auxiliary_endpoints_used"] == 24
    assert audit["auxiliary_actor_forward_calls"] == 4
    assert result["branch_feedback_gradient_norm"] > 0 and result["ordinary_feedback_gradient_norm"] > 0
    for row in audit["minibatches"]:
        assert all(x > 0 for x in row["branch_endpoint_logit_gradient_norms"])
        assert 2 not in row["auxiliary_pair_indices"]  # zero-weight row is not sampled
        assert len(set(row["auxiliary_pair_indices"])) == 3
        assert row["feedback_loss"] == pytest.approx(.005*(row["ordinary_feedback_kl"]+row["branch_feedback_kl"]), rel=1e-5)
    expected = data["observations"].reshape(-1,197)[data["trainable"].reshape(-1).astype(bool)]
    assert sorted(x.tobytes() for x in np.concatenate(ordinary_observations)) == sorted(x.tobytes() for x in np.tile(expected,(2,1)))
    equal(data,before); equal(obs,aux_before); equal(weights,weights_before)
    equal(feedback.state_dict(),saved_manager)
    equal(native.model.critic.state_dict(),reference.model.critic.state_dict())
    equal(native.optimizers.critic.state_dict(),reference.optimizers.critic.state_dict())
    equal(native.rng.bit_generator.state,reference.rng.bit_generator.state)
    equal(native._owned_rng_state,reference._owned_rng_state)
    assert any(not torch.equal(a,b) for a,b in zip(native.model.actor.parameters(),reference.model.actor.parameters()))


def test_all_program_rows_skip_aux_and_actor_momentum_but_update_critic():
    native = SyntheticNative(); native.update(batch())  # Real existing Adam momentum.
    actor_before=deepcopy(native.model.actor.state_dict()); adam_before=deepcopy(native.optimizers.actor.state_dict())
    data=batch(); data["trainable"][:]=0
    rng=np.random.default_rng(3); before=deepcopy(rng.bit_generator.state)
    result=run(native,data,manager(),rng=rng)
    equal(native.model.actor.state_dict(),actor_before); equal(native.optimizers.actor.state_dict(),adam_before)
    equal(rng.bit_generator.state,before)
    assert result["branch_update"]["actor_adam_steps"]==0 and result["branch_update"]["critic_adam_steps"]==4
    assert result["branch_update"]["auxiliary_actor_forward_calls"]==0
    assert result["branch_update"]["auxiliary_tree_rows"]==0


def test_caller_saved_aux_rng_and_native_state_resume_exact_next_update():
    first=SyntheticNative(); rng=np.random.default_rng(130); data=batch()
    run(first,data,manager(),rng=rng,auxiliary_pairs_per_minibatch=2)
    saved=first.snapshot(); saved_rng=deepcopy(rng.bit_generator.state)
    expected=run(first,data,manager(),rng=rng,auxiliary_pairs_per_minibatch=2)
    restored=SyntheticNative(); restored.restore(saved)
    resumed_rng=np.random.default_rng(); resumed_rng.bit_generator.state=saved_rng
    got=run(restored,data,manager(),rng=resumed_rng,auxiliary_pairs_per_minibatch=2)
    equal(first.snapshot(),restored.snapshot()); equal(expected,got)
    equal(rng.bit_generator.state,resumed_rng.bit_generator.state)


@pytest.mark.parametrize("bad", ["shape", "dtype", "weight_shape", "negative", "nan_obs",
    "infinite_weight", "unreliable", "missing_tree", "invalid_targets", "shared_rng", "shared_bit_generator",
    "lambda_high", "lambda_nan", "fraction", "fraction_bool", "minibatch"])
def test_invalid_contract_rejects_before_forward_rng_or_optimizer(bad, monkeypatch):
    native=SyntheticNative(); feedback=manager(); obs,weights=aux(); rng=np.random.default_rng(1); kwargs={}
    if bad=="shape": obs=obs[:,0]
    elif bad=="dtype": obs=obs.astype(np.float64)
    elif bad=="weight_shape": weights=weights[:,None]
    elif bad=="negative": weights[0]=-1
    elif bad=="nan_obs": obs[0,0,0]=np.nan
    elif bad=="infinite_weight": weights[0]=np.inf
    elif bad=="unreliable": feedback.reliable=False
    elif bad=="missing_tree": feedback.program=None
    elif bad=="invalid_targets": monkeypatch.setattr(feedback,"targets",lambda x: np.zeros((len(x),5),np.float32))
    elif bad=="shared_rng": rng=native.rng
    elif bad=="shared_bit_generator": rng=np.random.Generator(native.rng.bit_generator)
    elif bad=="lambda_high": feedback.current_lambda=.011
    elif bad=="lambda_nan": feedback.current_lambda=np.nan
    elif bad=="fraction": kwargs["branch_fraction"]=1.01
    elif bad=="fraction_bool": kwargs["branch_fraction"]=True
    elif bad=="minibatch": kwargs["auxiliary_pairs_per_minibatch"]=0
    saved=native.snapshot(); saved_rng=deepcopy(rng.bit_generator.state); counts=deepcopy(COUNTS)
    with pytest.raises(ValueError):run(native,batch(),feedback,obs,weights,rng,**kwargs)
    equal(native.snapshot(),saved); equal(rng.bit_generator.state,saved_rng)
    assert COUNTS["actor_forwards"]==counts["actor_forwards"] and COUNTS["adam_steps"]==counts["adam_steps"]
