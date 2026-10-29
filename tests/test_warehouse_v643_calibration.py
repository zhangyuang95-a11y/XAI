"""Actual Torch PPO integration on synthetic arrays; no Warehouse scenes."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.warehouse import v643_calibration as v


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def prepared(tmp_path):
    spec = importlib.util.spec_from_file_location('v643_toy_PPO',
        Path(__file__).with_name('test_warehouse_v629_ppo_update.py'))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers.prepared.__wrapped__(tmp_path)


def core_for(prepared, *, variant='raw', start=0):
    state, buffer, rows, _ = prepared
    state.joint_steps = start
    core = v.V643PPOUpdateCore(state, buffer, variant=variant,
        rng=np.random.default_rng(12), processed_joint_steps=start)
    state.advance_joint_steps(len(rows))
    return core, dict(start_joint_step=start, end_joint_step=state.joint_steps,
                     rows=deepcopy(rows), complete_cost_rows=[])


def complete_failure(batch, *, history=False):
    rows = batch['rows']
    for index, row in enumerate(rows):
        row.update(episode_id='failed' if index < (1 if history else 2)
            else 'partial-' + str((index-1)//10), frame=(index-1) % 10)
    if history:
        rows[0].update(frame=1, shutdown=True, terminated=True)
        first = deepcopy(rows[0]);first.update(frame=0, shutdown=False, terminated=False)
        # A distinct historical observation makes accidental historical Actor
        # gradients observable without an environment or simulated rollout.
        first['encoded'] = first['encoded'].copy();first['encoded'][..., 0] += .314
        batch['complete_cost_rows'] = [first, deepcopy(rows[0])]
    else:
        rows[0]['frame'] = 0;rows[1]['frame'] = 1
        rows[1].update(shutdown=True, terminated=True)
        for index, row in enumerate(rows[2:]):
            row.update(episode_id='partial-' + str(index//10), frame=index % 10)
        batch['complete_cost_rows'] = deepcopy(rows[:2])


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):equal(x, y)
    else:
        assert a == b


def test_real_warmup_PPO_updates_separate_critics_and_keeps_failure_examples(prepared):
    core, batch = core_for(prepared)
    complete_failure(batch)
    roles = v.previous.role_hashes(core.state.model)
    cost_before = deepcopy(core.state.model.cost_critic.state_dict())
    result = core.update_batch(batch, batch_sha256='a'*64)
    assert core.state.actor_updates == 0 and core.state.critic_updates == 4
    assert core.risk_training.critic_updates == 4
    assert core.risk_training.risk.config.target == .02
    assert core.risk_training.risk.config.window_episodes == 256
    assert core.risk.multiplier == 2
    assert v.previous.role_hashes(core.state.model) == roles
    assert core.risk_training.episodes['failed']['shutdown'] is True
    assert len(core.risk_training.risk.completed) == 1
    assert any(not torch.equal(cost_before[k], p) for k, p in core.state.model.cost_critic.state_dict().items())
    reward_owned = {id(p) for g in core.state.critic_optimizer.param_groups for p in g['params']}
    cost_owned = {id(p) for g in core.risk_training.optimizer.param_groups for p in g['params']}
    assert not reward_owned & cost_owned
    assert reward_owned == {id(p) for p in core.state.model.reward_critic.parameters()}
    assert cost_owned == {id(p) for p in core.state.model.cost_critic.parameters()}
    assert result['accepted_positive_program_gradient_updates'] == dict(robot_1=0, robot_2=0)
    assert result['training_authority'] is False


def test_historical_rows_train_cost_critic_but_never_Actor_surrogate(prepared):
    core, batch = core_for(prepared, start=4000)
    complete_failure(batch, history=True)
    before = v.previous.role_hashes(core.state.model)
    report = core.update_batch(batch, batch_sha256='a'*64)
    assert core.state.actor_updates == 4
    assert all(v.previous.role_hashes(core.state.model)[r] != before[r] for r in before)
    for epoch in report['epochs']:
        for update in epoch['updates']:
            assert update['current_rollout_Actor_cost_frames'] == 1
            assert update['historical_Actor_cost_frames'] == 0
            assert update['historical_cost_frames_excluded_from_Actor'] == 1
            assert update['cost_critic_replay']['physical_observations'] == 2
    assert len(core.risk_training.episodes['failed']['observations']) == 2


def test_historical_encoding_changes_cannot_change_actual_Actor_cost_gradient(prepared):
    core, batch = core_for(prepared, start=4000)
    complete_failure(batch, history=True)
    costs, outcomes = v.previous.complete_cost_rows(deepcopy(batch['complete_cost_rows']))
    core._current_frames = {(r['episode_id'], r['frame']) for r in batch['rows']}
    dummy = torch.zeros((1, 2, 5), requires_grad=True)
    loss, critic_loss = core._cost_loss(costs, outcomes, dummy)
    parameters = tuple(p for p in core.state.model.actors.parameters() if p.requires_grad)
    first = torch.autograd.grad(loss, parameters, allow_unused=True)
    costs[0]['encoded'] = np.full_like(costs[0]['encoded'], 100.)
    again, _ = core._cost_loss(costs, outcomes, dummy)
    second = torch.autograd.grad(again, parameters, allow_unused=True)
    assert critic_loss is None and torch.equal(loss, again)
    assert any(x is not None and bool((x != 0).any()) for x in first)
    for x, y in zip(first, second):
        if x is None:assert y is None
        else:assert torch.equal(x, y)
    core._current_frames = None


def test_invalid_cost_history_is_rejected_before_risk_or_optimizer_update(prepared):
    core, batch = core_for(prepared)
    complete_failure(batch)
    batch['complete_cost_rows'][0]['cost_return'] = 0.0
    before = v.training_fingerprint(core.state)
    with pytest.raises(RuntimeError, match='supplied cost labels'):
        core.update_batch(batch, batch_sha256='a'*64)
    assert core.failed and not core.risk_training.episodes
    assert core.risk_training.critic_updates == core.state.critic_updates == core.state.actor_updates == 0
    assert v.training_fingerprint(core.state) == before


def test_paired_variants_before_feedback_remain_numerically_identical(prepared):
    core, batch = core_for(prepared, variant='raw', start=4000)
    complete_failure(batch)
    from backend.training.warehouse_v622_actor_training import initialize_from_reference
    paired = initialize_from_reference(core.state.reference_path, core.state.reference_sha256,
        core.state.reference.role_sha256, initialization_seed=core.state.initialization_seed, actor_lr=3e-5)
    paired.joint_steps = 4000
    regularized = v.V643PPOUpdateCore(paired, core.buffer, variant='regularized',
        rng=np.random.default_rng(12), processed_joint_steps=4000)
    paired.advance_joint_steps(32)
    for current in (core, regularized):current.update_batch(deepcopy(batch),batch_sha256='a'*64)
    equal(core.state.model.state_dict(), regularized.state.model.state_dict())
    equal(core.state.actor_optimizer.state_dict(), regularized.state.actor_optimizer.state_dict())
    equal(core.state.critic_optimizer.state_dict(), regularized.state.critic_optimizer.state_dict())
    equal(core.risk_training.snapshot(), regularized.risk_training.snapshot())
    assert core.state.beta == regularized.state.beta


def test_raw_after_feedback_window_never_consults_program_provider(prepared):
    core, batch = core_for(prepared, start=17968)
    complete_failure(batch)
    def forbidden(*_):raise AssertionError('raw consulted external program')
    result = core.update_batch(batch,batch_sha256='a'*64,evidence_provider=forbidden)
    assert result['accepted_positive_program_gradient_updates'] == dict(robot_1=0,robot_2=0)
    assert all(u['program_loss'] == 0 for e in result['epochs'] for u in e['updates'])
    assert core.state.actor_updates == 4 and core.risk_training.critic_updates == 4


def test_regularized_actual_positive_feedback_with_fresh_proof_after_each_update(prepared,tmp_path):
    core, batch = core_for(prepared,variant='regularized',start=17968)
    complete_failure(batch)
    initial=prepared[3];calls=[]
    from backend.training import warehouse_v629_online_evidence as o
    from backend.training.warehouse_v622_actor_training import export_current_actor
    import json
    def provider(state,epoch_id):
        folder=tmp_path/f'proof-{len(calls)}';folder.mkdir();calls.append(epoch_id)
        exported=export_current_actor(state,folder/'current.npz',pair_seed=0,
            variant='regularized',candidate_id='synthetic-v643-PPO-only')
        ref=o.pointer(exported['path']);actor=o.FrozenWarehouseV6Actor.load(ref['path'],expected_sha256=ref['sha256'])
        result={}
        for role,original in initial.items():
            manifest=deepcopy(o.bound(original.pool_ref));manifest['target_actor']=ref
            for split,source in manifest['parts'].items():
                with np.load(source['arrays']['path'],allow_pickle=False) as z:
                    arrays={key:z[key].copy() for key in z.files}
                rows=o.bound(source['rows']);arrays['p']=o.runtime_probabilities(actor,role,arrays['x'],arrays['m'])
                arrays['sampled']=np.array([np.random.default_rng(o.independent_agent_seed(
                    base_seed=r['action_seed'],episode_id=r['episode_id'],frame=r['frame'],agent_id=role)).choice(5,p=q)
                    for r,q in zip(rows,arrays['p'],strict=True)],np.int64)
                file=folder/f'{role}-{split}.npz';np.savez_compressed(file,**arrays);source['arrays']=o.pointer(file)
            file=folder/(role+'-manifest.json');file.write_text(json.dumps(manifest))
            result[role]=o.build_evidence(original.program_ref,ref,o.pointer(file),epoch_id=epoch_id)
        return result
    result=core.update_batch(batch,batch_sha256='a'*64,evidence_provider=provider)
    assert len(calls)==len(set(calls))==4
    assert all(result['accepted_positive_program_gradient_updates'][r]==4 for r in v.previous.ROLES)
    assert core.risk_training.critic_updates==4
    assert all(u['program_loss']>0 for e in result['epochs'] for u in e['updates'])


def test_full_numerical_boundary_restore_repeats_next_PPO_update(prepared, tmp_path):
    core, batch = core_for(prepared, start=4000)
    complete_failure(batch)
    core.update_batch(batch, batch_sha256='a'*64)
    ref = v.save_boundary(core, tmp_path/'boundary.pt')
    value = v.read_boundary(ref['path'], ref['sha256'])
    from backend.training.warehouse_v622_actor_training import initialize_from_reference
    state = initialize_from_reference(core.state.reference_path, core.state.reference_sha256,
        core.state.reference.role_sha256, initialization_seed=core.state.initialization_seed, actor_lr=3e-5)
    restored = v.V643PPOUpdateCore(state, core.buffer, variant='raw', rng=np.random.default_rng(999))
    v.restore_numerical_boundary(restored, value)
    equal(core.snapshot(), restored.snapshot())
    next_rows = deepcopy(batch['rows'])
    for row in next_rows:
        row['episode_id'] = 'second-' + row['episode_id']
        row['shutdown'] = row['terminated'] = row['truncated'] = False
    for current in (core, restored):
        start = current.state.joint_steps
        current.state.advance_joint_steps(32)
        current.update_batch(dict(start_joint_step=start,end_joint_step=current.state.joint_steps,
            rows=deepcopy(next_rows),complete_cost_rows=[]), batch_sha256='b'*64)
    equal(core.snapshot(), restored.snapshot())
    assert v.training_fingerprint(core.state) == v.training_fingerprint(restored.state)


@pytest.mark.parametrize('fault', ['risk_step', 'old_candidate', 'missing_risk', 'optimizer', 'critic', 'risk_config'])
def test_incomplete_old_or_incoherent_boundary_cannot_restore(prepared, tmp_path, fault):
    core, batch = core_for(prepared)
    complete_failure(batch);core.update_batch(batch,batch_sha256='a'*64)
    ref = v.save_boundary(core, tmp_path/'boundary.pt');value = v.read_boundary(ref['path'],ref['sha256'])
    if fault == 'risk_step':value['core']['risk_training']['joint_step'] += 1
    elif fault == 'old_candidate':value['candidate_id'] = 'warehouse-v642-depth9-depth12-01'
    elif fault == 'missing_risk':value['core'].pop('risk_training')
    elif fault == 'optimizer':value['reward_optimizer']['param_groups'][0]['lr'] = .5
    elif fault == 'critic':next(iter(value['model'].values())).view(-1)[0] = float('nan')
    else:value['core']['risk_training']['config']['risk']['target'] = .04
    from backend.training.warehouse_v622_actor_training import initialize_from_reference
    state = initialize_from_reference(core.state.reference_path,core.state.reference_sha256,
        core.state.reference.role_sha256,initialization_seed=core.state.initialization_seed,actor_lr=3e-5)
    restored = v.V643PPOUpdateCore(state,core.buffer,variant='raw',rng=np.random.default_rng(99))
    before = v.training_fingerprint(state)
    risk_before = restored.risk_training.snapshot()
    with pytest.raises((ValueError,RuntimeError,KeyError)):
        v.restore_numerical_boundary(restored,value)
    assert v.training_fingerprint(state) == before
    equal(risk_before, restored.risk_training.snapshot())


def test_run_refuses_before_any_claim_or_path_access(monkeypatch):
    def forbidden(*_args,**_kwargs):raise AssertionError('disabled training accessed files')
    monkeypatch.setattr(Path,'open',forbidden)
    with pytest.raises(RuntimeError,match='numerical component is not an execution capability'):
        v.run(old_20k_passed=True,old_claim='ignored',plan='must-not-read.json')


def test_start_gate_does_not_accept_old_pass_markers_or_arbitrary_reference(prepared):
    state, buffer, _, _ = prepared
    with pytest.raises(RuntimeError, match='original complete v620'):
        v.validate_common_start(dict(calibration_passed=True, old_claim='v642'), buffer)
    with pytest.raises(RuntimeError, match='original complete v620'):
        v.validate_common_start(state, buffer)
