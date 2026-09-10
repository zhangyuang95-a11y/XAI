"""Small CPU synthetic Actor/batch, genuine PPO/dual Adam; no env or checkpoint.

Reuses only the earlier synthetic transport helpers, not its test executions.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.training import warehouse_family_stable_ppo_update as module
from backend.training.warehouse_native_v2 import IndependentOptimizers
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor, ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv
from core.program import ExecutableProgram, ProgramNode
from tests.test_warehouse_family_branch_ppo_update import SyntheticNative, equal, batch

NAMES = tuple(f"fixture_feature_{i}" for i in range(197))
COUNTS = dict(synthetic_models=0, actor_forwards=0, critic_forwards=0, backward_calls=0,
    diagnostic_autograd_calls=0, adam_steps=0, tree_batch_predictions=0, component_calls=0,
    environment_constructions=0, environment_steps=0, PT_loads=0, production_model_loads=0,
    MPS_calls=0, tree_fits=0)
LIMITS = dict(synthetic_models=30, actor_forwards=150, critic_forwards=110,
    backward_calls=220, diagnostic_autograd_calls=180, adam_steps=220,
    tree_batch_predictions=50, component_calls=35)


@pytest.fixture(scope="module", autouse=True)
def bounded_scope():
    patch = pytest.MonkeyPatch(); threads = torch.get_num_threads(); torch.set_num_threads(1)
    def wrap(owner, name, counter):
        old = getattr(owner, name)
        def call(*args, **kwargs):
            COUNTS[counter] += 1
            assert COUNTS[counter] <= LIMITS[counter], "Synthetic test ceiling exceeded"
            return old(*args, **kwargs)
        patch.setattr(owner, name, call)
    for owner, name, count in ((NativeActorCritic, "__init__", "synthetic_models"),
            (NativeActorCritic, "actor_logits", "actor_forwards"), (NativeActorCritic, "values", "critic_forwards"),
            (torch.Tensor, "backward", "backward_calls"), (torch.autograd, "grad", "diagnostic_autograd_calls"),
            (torch.optim.Adam, "step", "adam_steps"), (module.prediction, "predict", "tree_batch_predictions"),
            (module, "stable_ppo_update", "component_calls")):
        wrap(owner, name, count)
    def forbidden(*args, **kwargs): raise AssertionError("No environment, production Actor, or PT")
    for owner, name in ((NativeWarehouseEnv, "__init__"), (NativeWarehouseEnv, "step"),
            (NumPyNativeActor, "__init__"), (torch, "load")):
        patch.setattr(owner, name, forbidden)
    yield
    patch.undo(); torch.set_num_threads(threads)
    print("STABLE_PPO_CPU_SCOPE="+json.dumps(COUNTS, sort_keys=True))


class Native(SyntheticNative):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cfg.update(actor_learning_rate=3e-5, critic_learning_rate=3e-4)
        self.optimizers = IndependentOptimizers(self.model, self.cfg)
        self.envs = [SimpleNamespace(feature_names=NAMES)]  # No physical environment.


def program():
    return ExecutableProgram(ACTIONS, NAMES, ProgramNode(feature=NAMES[0], threshold=0.,
        left=ProgramNode(probabilities=(.72,.07,.07,.07,.07)),
        right=ProgramNode(probabilities=(.07,.07,.07,.07,.72))), {"test_fixture": True})


def auxiliary():
    return np.random.default_rng(431).normal(0,.3,(5,2,197)).astype(np.float32)


def call(native, data, *, lam=.2, obs=None, rng=None, p=None):
    return module.stable_ppo_update(native,data,program() if p is None else p, lambda_value=lam,
        auxiliary_observations=auxiliary() if obs is None else obs,
        auxiliary_rng=np.random.default_rng(12) if rng is None else rng)


def test_zero_lambda_exact_native_warm_adam_rng_and_explicit_base_fallback():
    class Subclass(Native):
        def update(self, data): return module.stable_ppo_update(self,data,lambda_value=0)
        def native_ppo_update(self, data): return SyntheticNative.update(self,data)
    a=Subclass(); d=batch(); a.native_ppo_update(d)
    b=Native(); b.restore(a.snapshot())
    rng=np.random.default_rng(3); before=deepcopy(rng.bit_generator.state)
    got=module.stable_ppo_update(a,d,object(),lambda_value=0,
        auxiliary_observations="deliberately uninspected",auxiliary_rng=rng)
    expected=b.update(d)
    equal(a.snapshot(),b.snapshot());equal(rng.bit_generator.state,before)
    equal({k:v for k,v in got.items() if k!="stable_update"},expected)
    assert got["stable_update"]["path"]=="native_exact"
    assert got["stable_update"]["actor_adam_steps"]==4
    assert got["stable_update"]["auxiliary_actor_forward_calls"]==0


def test_positive_shared_adam_gradients_roles_critic_rng_and_immutable_inputs():
    a=Native();b=Native();d=batch();old=deepcopy(d);aux=auxiliary();before_aux=aux.copy()
    p=program();before_program=deepcopy(p.to_dict());result=call(a,d,obs=aux,p=p)
    b.update(d);audit=result["stable_update"]
    assert audit["actor_adam_steps"]==audit["critic_adam_steps"]==4
    assert audit["ppo_nn_row_visits"]==audit["ordinary_kl_row_visits"]==12
    assert audit["ordinary_tree_rows"]==6 and audit["auxiliary_tree_rows"]==10
    assert audit["auxiliary_endpoints_used"]==40 and audit["auxiliary_actor_forward_calls"]==4
    for row in audit["minibatches"]:
        assert row["ppo_gradient_norm"]>0 and row["ordinary_feedback_gradient_norm"]>0
        assert row["auxiliary_feedback_gradient_norm"]>0
        assert all(x>0 for x in row["auxiliary_endpoint_logit_gradient_norms"])
        assert row["feedback_loss"]==pytest.approx(.1*(row["ordinary_feedback_kl"]+row["auxiliary_feedback_kl"]),rel=1e-5)
    equal(a.model.critic.state_dict(),b.model.critic.state_dict())
    equal(a.optimizers.critic.state_dict(),b.optimizers.critic.state_dict())
    equal(a.rng.bit_generator.state,b.rng.bit_generator.state);equal(a._owned_rng_state,b._owned_rng_state)
    equal(d,old);equal(aux,before_aux);equal(p.to_dict(),before_program)
    assert any(not torch.equal(x,y) for x,y in zip(a.model.actor.parameters(),b.model.actor.parameters()))


def test_lambda_above_old_cap_scales_actual_preclip_gradient_tenfold():
    a=Native(epochs=1,minibatch=8);b=Native(epochs=1,minibatch=8)
    small=call(a,batch(),lam=.02);large=call(b,batch(),lam=.2)
    for key in ("feedback_gradient_norm","ordinary_feedback_gradient_norm","auxiliary_feedback_gradient_norm","feedback_loss"):
        assert large[key]==pytest.approx(10*small[key],rel=1e-5)
    assert large["stable_update"]["actor_adam_steps"]==1
    assert large["ppo_gradient_norm"]==small["ppo_gradient_norm"]


def test_kl_hand_calculation_and_no_target_gradient():
    logits=torch.tensor([[[.3,-.2,.1,.7,-.4],[.2,.4,-.1,0.,.8]]],requires_grad=True)
    targets=torch.tensor([[[.5,.1,.1,.2,.1],[.1,.2,.3,.3,.1]]],requires_grad=True)
    got=module._kl(logits,targets)
    pi=np.exp(logits.detach().numpy());pi/=pi.sum(-1,keepdims=True)
    expected=(pi*(np.log(pi)-np.log(targets.detach().numpy()))).sum(-1).mean()
    assert float(got.detach())==pytest.approx(float(expected),rel=1e-6)
    got.backward()
    assert logits.grad is not None and torch.count_nonzero(logits.grad)==10
    assert targets.grad is None


def test_no_auxiliary_full_ordinary_lambda_and_no_auxiliary_rng_use():
    a=Native(epochs=1,minibatch=8);rng=np.random.default_rng(6);before=deepcopy(rng.bit_generator.state)
    result=module.stable_ppo_update(a,batch(),program(),lambda_value=.2,auxiliary_rng=rng)
    assert result["feedback_loss"]==pytest.approx(.2*result["ordinary_feedback_kl"],rel=1e-6)
    assert result["auxiliary_feedback_loss"]==0 and result["stable_update"]["auxiliary_rng_draws"]==0
    equal(rng.bit_generator.state,before)


def test_no_nn_rows_preserves_actor_and_momentum_and_has_no_auxiliary_update():
    a=Native();a.update(batch());d=batch();d["trainable"][:]=0
    actor=deepcopy(a.model.actor.state_dict());adam=deepcopy(a.optimizers.actor.state_dict())
    rng=np.random.default_rng(6);before=deepcopy(rng.bit_generator.state)
    result=call(a,d,rng=rng);audit=result["stable_update"]
    equal(actor,a.model.actor.state_dict());equal(adam,a.optimizers.actor.state_dict())
    equal(rng.bit_generator.state,before)
    assert audit["actor_adam_steps"]==0 and audit["critic_adam_steps"]==4
    assert audit["auxiliary_tree_rows"]==audit["auxiliary_actor_forward_calls"]==0
    assert result["feedback_loss"]==0


def test_aux_rng_and_full_native_state_resume_exact_and_source_relative_counters():
    a=Native();rng=np.random.default_rng(42);call(a,batch(),rng=rng)
    a.source_counters=dict(actor_optimizer_steps=4,critic_optimizer_steps=4)
    a.actor_optimizer_steps=a.critic_optimizer_steps=0
    saved=a.snapshot();saved_rng=deepcopy(rng.bit_generator.state)
    expected=call(a,batch(),rng=rng)
    b=Native();b.restore(saved);b.source_counters=deepcopy(a.source_counters)
    resumed=np.random.default_rng(0);resumed.bit_generator.state=saved_rng
    got=call(b,batch(),rng=resumed)
    equal(a.snapshot(),b.snapshot());equal(expected,got);equal(rng.bit_generator.state,resumed.bit_generator.state)
    assert a.actor_optimizer_steps==a.critic_optimizer_steps==4


@pytest.mark.parametrize("bad",["lambda_high","lambda_nan","missing_program","aux_shape","shared_rng","shared_bit_generator","program_order"])
def test_invalid_positive_inputs_reject_before_nn_rng_or_adam(bad):
    a=Native();d=batch();p=program();obs=auxiliary();rng=np.random.default_rng(5);lam=.2
    if bad=="lambda_high":lam=.201
    elif bad=="lambda_nan":lam=np.nan
    elif bad=="missing_program":p=None
    elif bad=="aux_shape":obs=obs[:,0]
    elif bad=="shared_rng":rng=a.rng
    elif bad=="shared_bit_generator":rng=np.random.Generator(a.rng.bit_generator)
    elif bad=="program_order":p=ExecutableProgram(ACTIONS,NAMES[::-1],p.root,p.metadata)
    before=a.snapshot();r=deepcopy(rng.bit_generator.state);counts=deepcopy(COUNTS)
    with pytest.raises(ValueError):
        module.stable_ppo_update(a,d,p,lambda_value=lam,auxiliary_observations=obs,auxiliary_rng=rng)
    equal(a.snapshot(),before);equal(rng.bit_generator.state,r)
    assert COUNTS["actor_forwards"]==counts["actor_forwards"] and COUNTS["adam_steps"]==counts["adam_steps"]
