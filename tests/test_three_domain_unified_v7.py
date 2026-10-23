"""Software checks on spent development seeds; no v7 admission claims."""
from copy import deepcopy
from dataclasses import asdict, replace
import numpy as np
import pytest
import torch

from training_v7.contracts import MethodSpec, BC_STAGES, program_gates
from training_v7.domains import domain
from training_v7.model import DualActorCritic, FrozenRole, mask_logits
from training_v7.losses import bc_loss, masked_kl, next_beta, program_loss, reference_loss, risk_config
from training_v7.program import (FitBudget, Program, certify, verify_certificate, scene_partition,
                                checked_data, legal_facts)

torch.set_num_threads(1)


@pytest.fixture(params=[('warehouse', 240024), ('pong', 880100), ('kitchen', 1000)])
def native(request):
    name, seed = request.param
    d = domain(name); s = d.initial_state(seed, 1)
    return d, s


def test_method_is_one_immutable_recipe():
    assert sum(s[1] for s in BC_STAGES) == 1270
    assert MethodSpec().sha256 == MethodSpec().sha256
    with pytest.raises(ValueError): replace(MethodSpec(), actor_lr=1e-3)
    assert MethodSpec().coefficient(160000, 200000) == 0
    assert MethodSpec().coefficient(170000, 200000) == pytest.approx(.01)
    assert risk_config().target == .02
    assert risk_config().release_upper == .01
    assert risk_config().minimum_episodes == 128


def test_public_labels_mask_and_physics_without_runtime_teacher(native, monkeypatch):
    d, s = native
    for _ in range(3):
        public = d.public(s); before = deepcopy(public)
        obs = d.observe_public(public); labels = d.label_public(public)
        assert public == before
        for role in d.spec.roles:
            assert obs[role.name]['mask'][labels[role.name]]
        # The physical step cannot invoke the adapter's teacher.
        with monkeypatch.context() as m:
            m.setattr(d, 'label_public', lambda *a: pytest.fail('runtime teacher call'))
            s, reward = d.step(s, labels)
            assert np.isfinite(reward).all()
    restored = d.restore(d.snapshot(s))
    for r in d.spec.roles:
        np.testing.assert_array_equal(d.observe_public(d.public(restored))[r.name]['x'],
                                      d.observe_public(d.public(s))[r.name]['x'])
    with pytest.raises(ValueError): d.outcome(s, restored)


def test_shared_architecture_and_independent_frozen_forward(native):
    d, s = native; obs = d.observe_public(d.public(s)); model = DualActorCritic(d.spec)
    assert not set(map(id, model.actors[d.spec.roles[0].name].parameters())) & set(map(id, model.actors[d.spec.roles[1].name].parameters()))
    for r in d.spec.roles:
        actor = model.actors[r.name]; o = obs[r.name]
        x = torch.tensor(np.repeat(o['x'][None], 3, axis=0)); af = torch.tensor(np.repeat(o['action_features'][None], 3, axis=0))
        masks = torch.tensor(np.repeat(o['mask'][None], 3, axis=0))
        # Nonzero branch verifies the tree and gate, not just the main NN.
        with torch.no_grad():
            actor.tree.leaf_logits.normal_(); actor.gate[-1].bias.fill_(.7)
        actual = actor(x, af, masks).detach().numpy()
        expected = FrozenRole.from_actor(actor).logits(x.numpy(), af.numpy(), masks.numpy())
        np.testing.assert_allclose(actual[masks.numpy()], expected[masks.numpy()], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))
        assert actor.primary[0].out_features == actor.primary[2].out_features == 256
        assert actor.gate[0].out_features == 64
        assert actor.tree.leaf_logits.shape == (16, len(r.actions))
        full, tree, primary = actor.components(x, af)
        labels = masks.long().argmax(-1)
        loss = bc_loss(full, tree, primary, masks, labels)
        loss.backward()
        assert actor.primary[0].weight.grad.abs().sum() > 0
        assert actor.tree.leaf_logits.grad.abs().sum() > 0
        # No silent clipping of physical observation violations.
        bad = o['x'].copy(); bad[0] = 1e20
        with pytest.raises(ValueError):
            from training_v7.domains import validate_observation
            validate_observation(r, bad, o['mask'], o['action_features'])


@pytest.mark.parametrize('actions', [1, 3, 5, 6, 13])
def test_all_action_dimensions_and_losses(actions):
    logits = torch.randn(8, actions, requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    q = torch.softmax(torch.randn_like(logits), -1)
    sampled = logits.detach().argmax(-1); top = sampled.clone()
    qualified = torch.ones(8, dtype=torch.bool)
    raw, selected = program_loss(logits, None, mask, sampled, None, None, variant='raw', coefficient=.01)
    assert raw.item() == 0 and not selected.any()
    raw.backward(); assert torch.count_nonzero(logits.grad) == 0
    logits.grad = None
    reg, selected = program_loss(logits, q, mask, sampled, qualified, top, variant='regularized', coefficient=.01)
    reg.backward(); assert selected.all()
    if actions > 1: assert logits.grad.abs().sum() > 0
    assert masked_kl(logits.detach(), logits, mask).max() < 1e-10
    with pytest.raises(ValueError): reference_loss(logits, logits, mask, ['validation'] * 8, 1.)
    reference_loss(logits, logits, mask, ['reference_train'] * 4 + ['current_train'] * 4, 1.)
    assert next_beta(1., .02) > 1
    if actions > 1:
        single = torch.zeros_like(mask); single[:, 0] = True
        assert torch.softmax(mask_logits(logits, single), -1)[:, 0].eq(1).all()
    with pytest.raises(ValueError): mask_logits(logits, torch.zeros_like(mask))


def test_kitchen_does_not_observe_private_future_or_commitment():
    d = domain('kitchen'); s = d.initial_state(1000, 1); t = deepcopy(s)
    t['_future_orders'] = [{'hidden': 'changed'}]; t['seed'] += 1
    t['policy_memory'] = {'goal': 'hidden route'}; t['next_item_id'] = 100000
    assert d.public(s) == d.public(t)
    assert d.label_public(d.public(s)) == d.label_public(d.public(t))
    assert len(d.spec.roles[1].actions) == 13


def test_pong_excludes_hidden_schedule_and_memory():
    d = domain('pong'); s = d.initial_state(880100, 1); t = deepcopy(s)
    t['_schedule'] = []; t['policy_memory'] = {'assignments': [1, 2]}; t['seed'] += 1
    assert d.public(s) == d.public(t)
    assert d.label_public(d.public(s)) == d.label_public(d.public(t))


def test_certificates_recheck_stored_outputs_and_boundaries(native):
    d, s = native; r = d.spec.roles[0]; a = len(r.actions)
    leaf = dict(support=[], bias=[0. if i == r.wait_index else -20. for i in range(a)], weights=[[] for _ in range(a)], witness=r.wait_index)
    proof = certify(r, leaf, [], FitBudget())
    verify_certificate(r, leaf, [], proof)
    program = Program(r, dict(leaf=leaf, proof=proof), 'a' * 64)
    obs = d.observe_public(d.public(s))[r.name]
    q, top, ids = program.distributions(obs['x'][None], obs['mask'][None])
    assert q.shape == (1, a) and top[0] == r.wait_index
    changed = deepcopy(leaf); changed['bias'][(r.wait_index + 1) % a] = 30
    with pytest.raises(ValueError): verify_certificate(r, changed, [], proof)
    bad = deepcopy(proof); bad['checks'][0]['upper'][0] += 1
    with pytest.raises(ValueError): verify_certificate(r, leaf, [], bad)
    i = r.legal_indices[0]
    path = [dict(indices=[i], weights=[1.], threshold=.5, right=True)]
    assert 0 in legal_facts(r, path)


def test_gates_do_not_accept_old_or_partial_receipts():
    metrics = dict(feedback_coverage=.5, explanation_coverage=.5, expected_feedback_coverage=.45,
                   feedback_agreement=.98, explanation_agreement=.99, feedback_kl=.09, explanation_kl=.09,
                   illegal_actions=0, unproved_paths=0, support_passed=True)
    assert program_gates(metrics)
    assert not program_gates({**metrics, 'feedback_kl': .10001})
    assert not program_gates({**metrics, 'unproved_paths': 1})
    assert not program_gates({**metrics, 'support_passed': False})


def test_complete_scene_partition():
    data = {k: np.array([i // 4 for i in range(40)]) for k in ('x', 'mask', 'p', 'scene', 'sampled')}
    a, b = scene_partition(data)
    assert not set(a['scene']) & set(b['scene'])
    assert len(a['scene']) + len(b['scene']) == 40


def test_runtime_hash_binding_takeover_replay_and_counterfactual(native, tmp_path, monkeypatch):
    from training_v7.storage import export_actor
    from training_v7.runtime import Binding, Runtime
    d, s = native; model = DualActorCritic(d.spec)
    asset = export_actor(model, tmp_path/'actor.npz')
    binding = Binding(d.spec.sha256, d.spec.name+'-regularized-v7', 'regularized', asset['path'], asset['sha256'], tuple(asset['role_sha256'].items()))
    runtime = Runtime(d, binding)
    state = runtime.wrap(s, seed=1000, task=1)
    state = __import__('json').loads(__import__('json').dumps(state))
    with monkeypatch.context() as m:
        m.setattr(d, 'label_public', lambda *a: pytest.fail('runtime used BC teacher'))
        demo = runtime.demonstration(binding, state=state, steps=3)
        assert runtime.replay(demo, binding) == demo['records'][-1]['after']
        human_wait = d.spec.roles[0].wait_index
        counter = runtime.counterfactual(state, [human_wait] * 3, binding)
        assert runtime.replay(counter, binding) == counter['records'][-1]['after']
        assert runtime.explanation(state, binding)['available'] is False
    changed = deepcopy(state); changed['task'] = 3
    with pytest.raises(ValueError): runtime.restore(changed, binding)
    with pytest.raises(ValueError): Runtime(d, binding, mode='pilot')
    raw = bytearray((tmp_path/'actor.npz').read_bytes()); raw[-1] ^= 1; (tmp_path/'actor.npz').write_bytes(raw)
    with pytest.raises(ValueError): runtime.decide(state, binding)


@pytest.mark.parametrize('variant', ['raw', 'regularized'])
def test_real_common_ppo_gradients_warmup_and_complete_episode_cost(native, variant, tmp_path):
    from training_v7.ppo import PPO
    from training_v7.bc import choose
    d, s = native; model = DualActorCritic(d.spec); observations = d.observe_public(d.public(s))
    generator = torch.Generator().manual_seed(123)
    indices, probabilities = choose(model, observations, generator, deterministic=True)
    reference_rows = {r.name: {k: np.repeat(observations[r.name][k][None], 8, axis=0)
                             for k in ('x', 'mask', 'action_features')} for r in d.spec.roles}
    engine = PPO(model, optimizer_seed=321, reference_rows=reference_rows)
    rows = [dict(observations=deepcopy(observations), indices=indices,
        old_logp={n: float(np.log(probabilities[n][indices[n]])) for n in indices},
        advantage=np.array([1., -1.], np.float32)*(1 if i%2 else -1), returns=np.array([2., 3.], np.float32),
        cost_value=.25) for i in range(8)]
    completed = [dict(rows=rows[:4], failed=True), dict(rows=rows[4:], failed=False)]
    old = model.role_hashes(); engine.joint_steps = 4000
    engine.update(rows, completed, variant=variant, budget=20000,
        program_provider=lambda *a: pytest.fail('program evaluated before feedback window'))
    assert model.role_hashes() == old and engine.actor_updates == 0 and engine.critic_updates == 4
    class Envelope:
        def targets(self, name, x, mask, sampled, current_actor):
            # Synthetic software evidence, explicitly not scientific eligibility.
            logits = np.zeros(mask.shape)
            for i, a in enumerate(sampled): logits[i, a] = 1
            from training_v7.program import probabilities
            return probabilities(logits, mask), np.ones(len(x), bool), sampled.copy()
        def invalidate_after_update(self, hashes): pass
    engine.joint_steps = 18000
    engine.update(rows, completed, variant=variant, budget=20000,
                  program_provider=lambda *a: Envelope())
    assert engine.actor_updates == 4
    assert all(model.role_hashes()[n] != old[n] for n in old)
    if variant == 'raw': assert not any(engine.program_updates.values())
    else: assert all(engine.program_updates.values()) and all(engine.program_norm.values())
    checkpoint = engine.checkpoint(tmp_path/'checkpoint.pt', collector_state={'public': 'software_fixture'},
                                  programs={}, plan_sha256='a'*64, sources={})
    assert checkpoint['sha256']


def test_quality_gates_raw_shutdown_and_no_program_pair_reselection():
    from training_v7.evaluation import quality, select_quality_only, common_budget_lock
    def rows(shutdowns):
        return [dict(seed=i, task=1+i%3, complete=True, deliveries=15, collisions=3, failed=i<shutdowns) for i in range(75)]
    raw = quality('warehouse', rows(5), rows(2))
    reg = quality('warehouse', rows(2), rows(2))
    assert not raw['passed'] and reg['passed']
    with pytest.raises(ValueError): select_quality_only([dict(pair_id=0, checkpoint_id='final', raw=raw, regularized=reg)])
    with pytest.raises(ValueError): select_quality_only([dict(pair_id=0, checkpoint_id='final', raw=reg, regularized=reg, program_score=.99)])
    complete = {n: {i: dict(passed=True) for i in range(3)} for n in ('warehouse', 'pong', 'kitchen')}
    assert common_budget_lock({200000: complete}) == 200000
    with pytest.raises(ValueError): common_budget_lock({200000: complete, 300000: complete})


def test_program_fit_has_disjoint_partitions_and_independent_new_proofs(native):
    from training_v7.program import fit_program, metrics
    d, s = native; role = d.spec.roles[0]; obs = d.observe_public(d.public(s))[role.name]
    n = 192
    x = np.repeat(obs['x'][None], n, axis=0); mask = np.repeat(obs['mask'][None], n, axis=0)
    p = np.zeros(mask.shape); p[:, role.wait_index] = 1.
    partitions = {}
    for j, key in enumerate(('fit', 'calibration', 'eligibility', 'validation')):
        partitions[key] = dict(x=x, mask=mask, p=p, scene=np.repeat(np.arange(j*100, j*100+32), 6),
                               sampled=np.full(n, role.wait_index))
    program, report = fit_program(role, partitions, 'a'*64, FitBudget())
    assert report['passed'] and report['illegal_actions'] == 0
    broken = dict(partitions); broken['validation'] = partitions['fit']
    with pytest.raises(ValueError): fit_program(role, broken, 'a'*64, FitBudget())
    assert program.as_dict()['runtime_control_allowed'] is False


def test_collector_cuts_are_not_fake_safe_episodes(tmp_path):
    from training_v7.collector import Collector
    d = domain('pong'); model = DualActorCritic(d.spec)
    cases = [dict(case_id=f'old-spent-{slot}-{block}', purpose='ppo', slot=slot, block=block,
                  seed=880100+slot, task=1) for slot in range(8) for block in range(2)]
    class SpentFixtureOwner:
        folder = tmp_path
        journal = tmp_path/'events.jsonl'
        plan = dict(cases=cases)
        used = set()
        def reset(self, adapter, case, purpose):
            assert case in cases and purpose == 'ppo'
            return adapter.initial_state(case['seed'], case['task'])
    collector = Collector(d, model, SpentFixtureOwner(), action_seed=9, budget=20000)
    rows, episodes = collector.collect()
    assert len(rows) == 1024
    assert any(r['training_cut'] for r in rows)
    assert len(episodes) == 8
    assert all(len(e['rows']) == 60 for e in episodes)
    assert all(e['outcome']['frames'] == 60 for e in episodes)
    assert all(r['physical_action_overrides'] == 0 for r in rows)


def test_program_gradient_uses_all_physical_frames():
    z = torch.tensor([[1., 0.]] * 8, requires_grad=True)
    q = torch.tensor([[.6, .4]] * 8)
    mask = torch.ones_like(z, dtype=torch.bool)
    sampled = torch.zeros(8, dtype=torch.long)
    qualified = torch.tensor([True]*4 + [False]*4)
    mixed, _ = program_loss(z, q, mask, sampled, qualified, sampled, variant='regularized', coefficient=.01)
    full, _ = program_loss(z, q, mask, sampled, torch.ones(8, dtype=torch.bool), sampled, variant='regularized', coefficient=.01)
    assert float(mixed.detach()) == pytest.approx(float(full.detach())/2)


def test_moving_leaf_requires_explicit_path_proof_and_new_outputs(native):
    d, s = native; role = d.spec.roles[0]
    action = next(i for i in range(len(role.actions)) if i != role.wait_index)
    leaf = dict(support=[], bias=[2. if i == action else 0. if i == role.wait_index else -20. for i in range(len(role.actions))],
                weights=[[] for _ in role.actions], witness=action)
    with pytest.raises(ValueError): certify(role, leaf, [], FitBudget())
    path = [dict(indices=[role.legal_indices[action]], weights=[1.], threshold=.5, right=True)]
    proof = certify(role, leaf, path, FitBudget())
    verify_certificate(role, leaf, path, proof)
    other = next(i for i in range(len(role.actions)) if i not in (action, role.wait_index))
    changed = deepcopy(leaf); changed['bias'][other] = 4.
    assert certify(role, changed, path, FitBudget()) is None


def test_evidence_is_invalidated_immediately_after_actor_change(native):
    from training_v7.bridge import Envelope
    from training_v7.model import tensor_hash
    d, s = native; role = d.spec.roles[0]; model = DualActorCritic(d.spec)
    leaf = dict(support=[], bias=[0. if i == role.wait_index else -20. for i in range(len(role.actions))],
                weights=[[] for _ in role.actions], witness=role.wait_index)
    p = Program(role, dict(leaf=leaf, proof=certify(role, leaf, [], FitBudget())), model.role_hashes()[role.name], feedback=(0,), explanation=(0,))
    envelope = Envelope({role.name: p}, model.role_hashes(), {role.name: dict(passed=True)})
    obs = d.observe_public(d.public(s))[role.name]
    args = (role.name, obs['x'][None], obs['mask'][None], np.array([role.wait_index]))
    assert envelope.targets(*args, current_actor=model.actors[role.name])[1].all()
    with torch.no_grad(): model.actors[role.name].primary[-1].bias.add_(.001)
    assert not envelope.targets(*args, current_actor=model.actors[role.name])[1].any()


def test_checkpoint_resume_produces_identical_next_update(tmp_path):
    from pathlib import Path
    from training_v7.ppo import PPO
    from training_v7.storage import source_files
    from training_v7.bc import choose
    d = domain('pong'); s = d.initial_state(880100, 1); observations = d.observe_public(d.public(s))
    model = DualActorCritic(d.spec)
    ref = {r.name: {k: np.repeat(observations[r.name][k][None], 8, axis=0)
                   for k in ('x', 'mask', 'action_features')} for r in d.spec.roles}
    engine = PPO(model, optimizer_seed=48, reference_rows=ref)
    indices, p = choose(model, observations, torch.Generator().manual_seed(98), deterministic=True)
    rows = [dict(observations=deepcopy(observations), indices=indices,
        old_logp={n: float(np.log(p[n][indices[n]])) for n in indices},
        advantage=np.array([i-4, 4-i], np.float32), returns=np.array([2., 3.], np.float32), cost_value=.1)
        for i in range(8)]
    engine.joint_steps = 15000
    engine.update(rows, [], variant='raw', budget=20000, program_provider=None)
    root = Path(__file__).resolve().parents[1]; sources = source_files(root)
    checkpoint = engine.checkpoint(tmp_path/'state.pt', collector_state={}, programs={}, plan_sha256='b'*64, sources=sources)
    restored = PPO(deepcopy(engine.reference), optimizer_seed=999, reference_rows=ref)
    restored.restore_checkpoint(checkpoint['path'], checkpoint['sha256'], root=root, sources=sources, plan_sha256='b'*64)
    for e in (engine, restored): e.update(deepcopy(rows), [], variant='raw', budget=20000, program_provider=None)
    assert engine.model.role_hashes() == restored.model.role_hashes()
    assert engine.beta == restored.beta
    assert torch.equal(engine.generator.get_state(), restored.generator.get_state())
    for key, tensor in engine.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[key])
    with pytest.raises(ValueError):
        restored.restore_checkpoint(checkpoint['path'], checkpoint['sha256'], root=root, sources=sources, plan_sha256='c'*64)


def test_real_path_lp_and_independent_rational_dual_check(native):
    d, s = native; role = d.spec.roles[0]
    action = next(i for i in range(len(role.actions)) if i != role.wait_index)
    index = role.legal_indices[action]
    leaf = dict(support=[index], bias=[-2. if i == action else 0. if i == role.wait_index else -20. for i in range(len(role.actions))],
        weights=[[20. if i == action else 0.] for i in range(len(role.actions))], witness=role.wait_index)
    path = [dict(indices=[index], weights=[1.], threshold=.5, right=False)]
    budget = FitBudget(); proof = certify(role, leaf, path, budget)
    assert proof is not None
    check = next(c for c in proof['checks'] if c['action'] == action)
    assert any(y > 0 for y in check['dual'])
    verify_certificate(role, leaf, path, proof)
    broken = deepcopy(proof)
    next(c for c in broken['checks'] if c['action'] == action)['dual'][0] *= .5
    with pytest.raises(ValueError): verify_certificate(role, leaf, path, broken)
    from training_v7.program import BudgetExceeded
    exhausted = FitBudget(); exhausted.counts['proof'] = 100000
    with pytest.raises(BudgetExceeded): certify(role, leaf, path, exhausted)


def test_real_episode_risk_dual_ignores_partial_trajectories():
    from backend.training.warehouse_v6_shutdown_cost import ShutdownRiskDual
    dual = ShutdownRiskDual(risk_config())
    ignored = dual.update([(False, False)] * 200)
    assert ignored['window_episodes'] == 0
    first = dual.update([(True, False)] * 128)
    assert first['window_episodes'] == 128 and first['one_sided_upper'] < .02
    second = dual.update([(True, False)] * 128)
    assert second['window_episodes'] == 256 and second['one_sided_upper'] < .01
    assert second['next_multiplier'] < first['next_multiplier']
    with pytest.raises(ValueError): dual.update([(False, True)])


def test_external_threshold_midpoint_uses_same_float64_route():
    from training_v7.program import right_branch
    a = np.float32(1.); b = np.nextafter(a, np.float32(2.))
    midpoint = (float(a)+float(b))/2
    split = dict(indices=[0], weights=[1.], threshold=midpoint)
    np.testing.assert_array_equal(right_branch(np.array([[a], [b]], np.float32), split), [False, True])


def test_search_cannot_spend_final_refit_reserve():
    from training_v7.program import BudgetExceeded
    budget = FitBudget(); budget.counts['leaf'] = MethodSpec().leaf_calls-MethodSpec().final_leaf_call_reserve
    with pytest.raises(BudgetExceeded): budget.debit('leaf')
    budget.begin_final(); budget.debit('leaf')
    assert budget.counts['leaf'] == MethodSpec().leaf_calls-MethodSpec().final_leaf_call_reserve+1


def test_seed_scope_includes_pong_kitchen_and_start_guards_without_patching_legacy(tmp_path):
    import json
    from training_v7.seed_audit import audit, SCOPE
    from experiments.warehouse import seed_metadata_audit_indexed_v642 as original
    selector = original.candidates
    root = tmp_path/'history'; root.mkdir()
    for relative in ('pong/rolling_actor_v5.acceptance.json', 'kitchen/training_report.json',
                     'mixed/registration.json', 'mixed/started.json'):
        p = root/relative; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'scenario_seed': 1234567}))
    proposal = tmp_path/'proposal/seed-proposal.json'; proposal.parent.mkdir()
    proposal.write_text(json.dumps(dict(schema='warehouse-seed-proposal.v1', candidate_id='software-scope-fixture',
        families=[dict(name='cases', kind='scenario', values=[1234567])], registered_trace_paths=[])))
    result = audit(proposal, [root], tmp_path/'audit')
    assert original.candidates is selector
    assert result['selection_scope'] == SCOPE
    assert result['selection_adapter']['original_module_monkeypatch'] is False
    found = result['summary']['families']['cases']['references_or_reservations']
    assert len(found) == 4
    assert not result['summary']['unresolved_paths']


def test_final_metadata_delta_detects_same_stat_content_change_and_new_files(tmp_path):
    import json, os
    from training_v7.seed_audit import audit
    from training_v7.seed_delta import verify
    from experiments.warehouse.seed_triage_indexed_v642 import triage
    root = tmp_path/'history'; root.mkdir()
    metadata = root/'training_report.json'; metadata.write_text(json.dumps({'scenario_seed': 1000}))
    proposal = tmp_path/'proposal/seed-proposal.json'; proposal.parent.mkdir()
    proposal.write_text(json.dumps(dict(schema='warehouse-seed-proposal.v1', candidate_id='software-delta-fixture',
        families=[dict(name='cases', kind='scenario', values=[9999999])], registered_trace_paths=[])))
    audit(proposal, [root], tmp_path/'inventory')
    triage(tmp_path/'inventory/inventory.json', proposal, tmp_path/'triage')
    first = verify(proposal, tmp_path/'triage/result.json', tmp_path/'delta1.json')
    assert first['passed']
    old_stat = metadata.stat(); metadata.write_text(json.dumps({'scenario_seed': 2000}))
    os.utime(metadata, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
    changed = verify(proposal, tmp_path/'triage/result.json', tmp_path/'delta2.json')
    assert not changed['passed'] and changed['changed'] == [str(metadata)]
    (root/'new_registration.json').write_text(json.dumps({'scenario_seed': 9999999}))
    added = verify(proposal, tmp_path/'triage/result.json', tmp_path/'delta3.json')
    assert not added['passed'] and not added['selection_unchanged']


def test_preliminary_clearance_never_grants_a_reset(tmp_path):
    import json
    from training_v7.storage import Run
    from training_v7.contracts import digest
    plan = dict(cases=[])
    clearance = tmp_path/'clearance.json'
    clearance.write_text(json.dumps(dict(version='three-domain-unified-seed-clearance.v1',
        plan_sha256=digest(plan), passed=True, evidence=[])))
    with pytest.raises(ValueError, match='clearance'):
        Run(tmp_path/'run', plan, clearance)
    assert not (tmp_path/'run/claim.json').exists()


def test_preflight_revision_forbidden_after_experiment_claim(tmp_path):
    from training_v7.workflow import amend_preflight
    claim = tmp_path/'warehouse/calibration-0/claim.json'; claim.parent.mkdir(parents=True)
    claim.write_text('{}')
    with pytest.raises(ValueError, match='experiment claim'): amend_preflight(tmp_path)


def test_legacy_csv_complete_rows_and_integer_values(tmp_path):
    import csv
    from training_v7.seed_context import legacy_csv
    p = tmp_path/'training_metrics.jsonl'
    header = ['episode', 'scenario_seed'] + ['metric_'+str(i) for i in range(50)]
    with p.open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(header); writer.writerow([1, '31000000000.0']+[0]*50)
    rows, evidence = legacy_csv(p)
    assert rows[0]['scenario_seed'] == 31000000000 and type(rows[0]['scenario_seed']) is int
    assert evidence['complete_rows'] == 1
    with p.open('a') as f: f.write('2,3\n')
    with pytest.raises(ValueError, match='ragged'): legacy_csv(p)


def test_legacy_pickle_unknown_bytes_cannot_be_cleared(tmp_path):
    import pickle, gzip
    from training_v7.seed_context import legacy_trajectory
    p = tmp_path/'training_trajectory.json'
    p.write_bytes(gzip.compress(pickle.dumps(dict(format='warehouse_collaborative_trajectory_v1',
        metadata=dict(source='current_two_robot_training', seed=7), episodes=dict(opaque=b'unknown')))))
    with pytest.raises(ValueError, match='opaque'): legacy_trajectory(p)


def test_inert_reader_does_not_call_original_global(tmp_path):
    import io, pickle
    from training_v7.seed_context import LabeledReader, InertRecord
    class Payload:
        def __reduce__(self): return (eval, ('1/0',))
    value = LabeledReader(io.BytesIO(pickle.dumps(Payload()))).load()
    assert isinstance(value, InertRecord) and value.constructor_args == ('1/0',)
    assert value.pickle_global == ('builtins', 'eval')


def test_first_use_context_forbidden_after_claim(tmp_path):
    from training_v7.seed_context import reconcile
    p = tmp_path/'seed-proposal.json'; p.write_text('{}')
    claim = tmp_path/'warehouse'/'claim.json'; claim.parent.mkdir(); claim.write_text('{}')
    with pytest.raises(ValueError, match='experiment claim'):
        reconcile(p, tmp_path/'not_read.json', tmp_path/'result.json')
