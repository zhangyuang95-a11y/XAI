"""Environment-free v6.43 PPO integration; finite calibration owner is absent.

This is an actual numerical bridge and complete-boundary checkpoint helper,
not a run capability. It constructs no environment or RNG, claims no seeds,
and refuses the CLI training entrypoint. Old v6.42 pass markers and claims do
not authorize this changed method. Its finite collector, program namespace,
source-bound history audit and single-use owner still require implementation.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import io
import os

import numpy as np
import torch

from backend.training import warehouse_v629_ppo_update as previous
from backend.training.warehouse_v643_risk_training import RiskTrainingConfig, make_risk_training_core
from backend.training.warehouse_v622_actor_training import verify_initial_equivalence
from backend.training.warehouse_v629_rollout_replay import training_fingerprint
from backend.training.warehouse_v6_shutdown_cost import joint_shutdown_surrogate
from experiments.warehouse.v626_protocol import require, file_sha, digest
from .v643_risk_recipe import CANDIDATE, REFERENCE_SHA256

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'warehouse-v643-risk-aware-PPO-integration.v1'
BOUNDARY_VERSION = 'warehouse-v643-complete-numerical-boundary.v1'
MISSING_FINITE_OWNER = (
    'new_candidate_finite_training_program_and_quality_inventories',
    'new_source_closed_partition_owner_and_program_output_namespaces',
    'new_exact_full_history_seed_clearance_and_claim_at_start',
    'finite_collector_and_reset_journal_bound_to_v643_PPO_and_risk_state',
    'complete_training_reader_environment_program_provider_checkpoint_and_explicit_recovery',
    'new_20k_quality_and_program_auditors_without_old_20k_pass_requirement',
)


def source_files():
    """Current numerical dependency closure; not a scientific source permit."""
    from .v629_training_admission import source_files as inherited
    files = inherited()
    for name in ('backend/training/warehouse_v629_ppo_update.py',
                 'backend/training/warehouse_v643_risk_training.py',
                 'experiments/warehouse/v643_risk_recipe.py',
                 'experiments/warehouse/v643_shutdown_diagnostic.py',
                 'experiments/warehouse/v643_calibration.py',
                 'scripts/train_warehouse_v643_calibration.py',
                 'tests/test_warehouse_v643_calibration.py',
                 'docs/warehouse_v643_calibration_bridge.md'):
        files[name] = file_sha(ROOT / name)
    return dict(sorted(files.items()))


def integration_status():
    return dict(version=VERSION, candidate_id=CANDIDATE,
        numerical_PPO_integration_implemented=True, finite_owner_implemented=False,
        training_entrypoint_enabled=False, missing_implementation=list(MISSING_FINITE_OWNER),
        reference_actor_sha256=REFERENCE_SHA256,
        source_files=source_files(), source_sha256=digest(source_files()),
        previous_20k_pass_can_authorize_this_candidate=False,
        old_claim_reuse_allowed=False, automatic_start=False,
        new_environment_resets=0, new_seed_claims=0, training_authority=False,
        model_acceptance_passed=False, pilot_opened=False)


def run(*_args, **_kwargs):
    # Deliberately before reading plan paths, opening output, constructing RNGs
    # or burning a single-use claim. A CLI named train is not a capability.
    raise RuntimeError('v643 finite calibration owner is not implemented; no training or claim: '
                       + '; '.join(MISSING_FINITE_OWNER))


def validate_common_start(state, reference_buffer):
    """Actual zero-update v6.20 import, not an old trained v6.42 checkpoint.

    This function initializes nothing. The future finite owner must authorize
    the caller's Actor/critic construction before calling it, then compare the
    returned complete starting identities between raw and regularized.
    """
    require(isinstance(state, previous.TrainingState) and state.reference_sha256 == REFERENCE_SHA256
        and file_sha(state.reference_path) == REFERENCE_SHA256,
        'v643 must start from the original complete v620 BC Actor artifact')
    require(state.joint_steps == state.actor_updates == state.critic_updates == 0
        and not state.resumed_training_state and not state.actor_optimizer.state
        and not state.critic_optimizer.state and not state.extra_rng_states,
        'v643 calibration requires a fresh full paired start, not old trained weights or risk state')
    require(previous.role_hashes(state.model) == state.reference.role_sha256,
        'complete original reference Actor weights differ')
    reference_buffer.validate()
    require(reference_buffer.source_path is not None
        and file_sha(reference_buffer.source_path) == reference_buffer.source_sha256,
        'common starting reference buffer is not bound')
    verified = previous.ReferenceTrainingBuffer.load(reference_buffer.source_path,
        reference_buffer.source_sha256, state.reference_sha256)
    require(np.array_equal(verified.encoded, reference_buffer.encoded)
        and np.array_equal(verified.masks, reference_buffer.masks)
        and verified.episode_ids == reference_buffer.episode_ids,
        'common starting reference buffer differs from its original file')
    equivalent = verify_initial_equivalence(state,
        previous.tensor(reference_buffer.encoded[:32]), previous.tensor(reference_buffer.masks[:32]))
    return dict(reference_sha256=state.reference_sha256,
        actor_role_sha256=previous.role_hashes(state.model),
        critic_initialization_sha256=previous.critic_initialization_sha256(state.model),
        initialization_seed=state.initialization_seed,
        initial_training_state_sha256=training_fingerprint(state),
        equivalence=equivalent, Actor_updates=0, critic_updates=0, training_authority=False)


class _ObservedRisk:
    """Hook the inherited core only after its full batch validation succeeds."""
    def __init__(self, core):
        self.core = core

    @property
    def multiplier(self):
        return self.core.risk_training.risk.multiplier

    def snapshot(self):
        return self.core.risk_training.risk.snapshot()

    def update(self, outcomes):
        pending = self.core._observing
        require(pending is not None, 'risk observation requires the current validated batch')
        rows, expected = previous.complete_cost_rows(deepcopy(pending['complete_cost_rows']))
        require(list(outcomes) == [(True, outcome) for outcome in expected.values()],
            'risk events differ from complete physical episodes')
        result = self.core.risk_training.observe_batch(rows,
            batch_sha256=pending['sha256'], end_joint_step=pending['end_joint_step'], partition='train')
        return result['risk']


class V643PPOUpdateCore(previous.PPOUpdateCore):
    """Reuse PPO/reference/program algebra, replace cost ownership explicitly.

    The reward optimizer owns only reward-critic parameters. The separately
    checkpointed risk optimizer owns only cost-critic parameters. Actor costs
    use only frames from the current rollout, even when a true complete label
    needs a history extending into previous rollouts. Historical observations
    may train the cost critic, but cannot enter an Actor gradient here.
    """
    def __init__(self, state, reference_buffer, *, variant, rng,
                 config=previous.PPOConfig(), risk_config=None, processed_joint_steps=0):
        settings = risk_config or RiskTrainingConfig(
            learning_rate=config.critic_learning_rate,
            low_battery_columns=(47, previous.OBSERVATION_DIM + 47))
        require(settings.episode_horizon == config.episode_horizon
            and settings.learning_rate == config.critic_learning_rate,
            'cost replay and PPO numerical settings disagree')
        require(not state.critic_optimizer.state,
            'construct the new risk bridge before loading any trained reward optimizer')
        super().__init__(state, reference_buffer, variant=variant, rng=rng,
                         config=config, processed_joint_steps=processed_joint_steps)
        # A separate owner avoids two Adam moment sets mutating cost weights.
        # Construction is numerical only; initialization seed authority is the
        # future finite owner's responsibility.
        state.critic_optimizer = torch.optim.Adam(state.model.reward_critic.parameters(),
            lr=config.critic_learning_rate, eps=1e-5)
        self.risk_training = make_risk_training_core(state.model,
            observation_dim=2 * previous.OBSERVATION_DIM, rng=rng, config=settings)
        self.risk_training.joint_step = processed_joint_steps
        self.risk = _ObservedRisk(self)
        self._observing = None
        self._current_frames = None

    def _cost_loss(self, rows, outcomes, logits):
        require(self._current_frames is not None, 'Actor cost needs the current rollout identity set')
        current = [row for row in rows
                   if (row['episode_id'], row['frame']) in self._current_frames]
        if not current:
            return logits[..., 0].mul(0).sum(), None
        x = previous.tensor([r['encoded'] for r in current])
        mask = previous.tensor([r['mask'] for r in current])
        logp = previous.masked_log_probabilities(self.state.model.outputs_encoded(x, mask), mask)
        actions = torch.as_tensor(np.asarray([r['actions'] for r in current]), dtype=torch.long)
        chosen = logp.gather(-1, actions[..., None]).squeeze(-1)
        ids = {key: i for i, key in enumerate(outcomes)}
        labels = torch.tensor([ids[r['episode_id']] for r in current])
        loss, _ = joint_shutdown_surrogate(chosen,
            previous.tensor([r['old_logp'] for r in current]),
            previous.tensor([r['cost_advantage'] for r in current]), labels,
            sorted(set(int(x) for x in labels)), clip=self.config.clip,
            reference_horizon=self.config.episode_horizon)
        # The inherited reward-critic optimizer must receive no cost objective.
        return loss, None

    def _minibatch(self, rows, costs, outcomes, **kwargs):
        # TrainingState's old critic_step clips both modules. Clearing detached
        # replay gradients makes that identical to reward-only clipping here.
        self.risk_training.optimizer.zero_grad(set_to_none=True)
        result = super()._minibatch(rows, costs, outcomes, **kwargs)
        current_count = sum((r['episode_id'], r['frame']) in self._current_frames for r in costs)
        result['current_rollout_Actor_cost_frames'] = current_count
        result['historical_Actor_cost_frames'] = 0
        result['historical_cost_frames_excluded_from_Actor'] = len(costs) - current_count
        if self.risk_training.episodes:
            result['cost_critic_replay'] = self.risk_training.update_critic()
            self.risk_training.optimizer.zero_grad(set_to_none=True)
        else:
            result['cost_critic_replay'] = dict(skipped=True,
                reason='no_complete_training_episode', partial_episodes_labeled_negative=0)
        return result

    def update_batch(self, batch, *, batch_sha256, evidence_provider=None):
        require(self._observing is None, 'risk PPO updates cannot nest')
        self._observing = dict(complete_cost_rows=batch['complete_cost_rows'],
            sha256=batch_sha256, end_joint_step=batch['end_joint_step'])
        self._current_frames = {(r['episode_id'], r['frame']) for r in batch['rows']}
        try:
            result = super().update_batch(batch, batch_sha256=batch_sha256,
                                          evidence_provider=evidence_provider)
            require(self.risk_training.joint_step == self.processed_joint_steps
                and self.risk_training.seen_batches == self.seen_batch_receipts
                and set(self.risk_training.seen_episodes) == self.seen_complete_episodes,
                'cost replay and PPO episode/update ledgers differ')
            result.update(version=VERSION, risk_training_config=asdict(self.risk_training.config),
                separate_reward_and_cost_optimizers=True, training_authority=False)
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self._observing = None
            self._current_frames = None

    def snapshot(self):
        require(self._observing is None and self.processed_joint_steps == self.state.joint_steps
            and self.risk_training.joint_step == self.processed_joint_steps and not self.failed,
            'only a completed v643 PPO/risk boundary can be checkpointed')
        value = super().snapshot()
        # Include the core's actual inherited source and our changed numerical
        # implementation separately. This is not the old core restore schema.
        return dict(version=VERSION, PPO=value, risk_training=self.risk_training.snapshot(),
            current_source_files=source_files(), training_authority=False,
            resume_authority=False, finite_owner_implemented=False)


def save_boundary(core, path):
    """Save complete numerical weights/optimizers/risk/RNG; no run capability."""
    require(type(core) is V643PPOUpdateCore, 'explicit v643 numerical core required')
    state = core.state
    value = dict(version=BOUNDARY_VERSION, candidate_id=CANDIDATE,
        source_files=source_files(), reference_sha256=state.reference_sha256,
        reference_buffer_sha256=core.buffer.source_sha256,
        initialization_seed=state.initialization_seed, training_state_sha256=training_fingerprint(state),
        model=deepcopy(state.model.state_dict()), actor_optimizer=deepcopy(state.actor_optimizer.state_dict()),
        reward_optimizer=deepcopy(state.critic_optimizer.state_dict()), beta=dict(state.beta),
        trainability={n: p.requires_grad for n, p in state.model.named_parameters()},
        joint_steps=state.joint_steps, actor_updates=state.actor_updates, reward_critic_updates=state.critic_updates,
        core=core.snapshot(), training_authority=False, resume_authority=False,
        reader_environment_provider_and_claim_state_included=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    return dict(path=str(path.resolve()), sha256=file_sha(path))


def read_boundary(path, expected_sha256):
    """Read-only checkpoint inspection; does not instantiate or restore a model."""
    path = Path(path)
    require(not path.is_symlink() and not path.stat().st_mode & 0o222,
        'v643 boundary must be an immutable regular asset')
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected_sha256, 'v643 numerical boundary hash differs')
    value = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    require(value['version'] == BOUNDARY_VERSION and value['candidate_id'] == CANDIDATE
        and value['source_files'] == source_files() and value['training_authority'] is False
        and value['resume_authority'] is False
        and value['reader_environment_provider_and_claim_state_included'] is False,
        'v643 checkpoint identity/source differs; an old 20k pass cannot replace it')
    return value


def restore_numerical_boundary(core, value):
    """Restore into explicit caller-created modules, without Actor/RNG creation.

    This restores only the numerical component. No finite experiment can resume
    until its missing reader/environment/provider/claim state is implemented.
    """
    require(type(core) is V643PPOUpdateCore and not core.failed
        and core.processed_joint_steps == core.state.joint_steps == 0
        and not core.state.actor_optimizer.state and not core.state.critic_optimizer.state
        and not core.risk_training.optimizer.state,
        'restore needs a new explicit zero-update numerical bridge')
    require(value['version'] == BOUNDARY_VERSION and value['candidate_id'] == CANDIDATE
        and value['source_files'] == source_files() and value['training_authority'] is False
        and value['resume_authority'] is False
        and value['reader_environment_provider_and_claim_state_included'] is False,
        'v643 numerical checkpoint candidate/source/authority differs')
    state = core.state
    require(value['reference_sha256'] == state.reference_sha256
        and value['reference_buffer_sha256'] == core.buffer.source_sha256
        and value['initialization_seed'] == state.initialization_seed,
        'v643 full checkpoint reference, buffer or initialization differs')
    saved = value['core'];ppo = saved['PPO'];risk = saved['risk_training']
    require(saved['version'] == VERSION and saved['current_source_files'] == source_files()
        and saved['training_authority'] is saved['resume_authority'] is saved['finite_owner_implemented'] is False
        and ppo['version'] == previous.VERSION and ppo['source_sha256'] == file_sha(previous.__file__)
        and ppo['config'] == asdict(core.config) and ppo['variant'] == core.variant
        and ppo['safe_batch_boundary'] is True and ppo['failed'] is False
        and ppo['processed_joint_steps'] == value['joint_steps'] == risk['joint_step']
        and ppo['risk'] == risk['risk'] and ppo['rng_state'] == risk['rng_state'],
        'v643 saved PPO and complete-episode risk boundary differ')
    receipts = ppo['seen_batch_receipts'];episodes = ppo['seen_complete_episodes']
    require(type(receipts) is list and receipts == sorted(set(receipts))
        and receipts == risk['seen_batches'] and type(ppo['update_sequence']) is int
        and len(receipts) == ppo['update_sequence'] >= 0
        and episodes == sorted(set(episodes)) == sorted(risk['seen_episodes'])
        and set(ppo['accepted_program_updates']) == set(previous.ROLES)
        and all(type(n) is int and 0 <= n <= value['actor_updates']
                for n in ppo['accepted_program_updates'].values())
        and (core.variant != 'raw' or not any(ppo['accepted_program_updates'].values())),
        'v643 saved complete episode, batch or program-gradient ledger differs')
    require(type(value['joint_steps']) is int and 0 <= value['joint_steps'] <= core.config.joint_step_budget
        and all(type(value[k]) is int and value[k] >= 0 for k in ('actor_updates','reward_critic_updates'))
        and value['actor_updates'] == ppo['actor_updates']
        and value['reward_critic_updates'] == ppo['critic_updates']
        and value['beta'] == ppo['beta'] and set(value['beta']) == set(previous.ROLES)
        and all(type(b) in (int,float) and np.isfinite(b) and .001 <= b <= 100 for b in value['beta'].values()),
        'v643 saved numerical counters or reference KL multipliers differ')

    def finite(item):
        if isinstance(item,torch.Tensor):
            require(bool(torch.isfinite(item).all()), 'nonfinite saved numerical tensor')
        elif isinstance(item,dict):
            for v in item.values():finite(v)
        elif isinstance(item,(tuple,list)):
            for v in item:finite(v)
        elif isinstance(item,float):
            require(np.isfinite(item), 'nonfinite saved numerical value')
    finite(value)
    parameters = dict(state.model.named_parameters())
    require(set(parameters) == set(value['trainability'])
        and all(type(x) is bool for x in value['trainability'].values()),
        'v643 trainable parameter map differs')
    old = dict(model=deepcopy(state.model.state_dict()),
        actor=deepcopy(state.actor_optimizer.state_dict()),reward=deepcopy(state.critic_optimizer.state_dict()),
        beta=dict(state.beta),trainability={k:p.requires_grad for k,p in parameters.items()})
    try:
        state.model.load_state_dict(value['model'],strict=True)
        state.actor_optimizer.load_state_dict(value['actor_optimizer'])
        state.critic_optimizer.load_state_dict(value['reward_optimizer'])
        for name,enabled in value['trainability'].items():parameters[name].requires_grad_(enabled)
        state.beta = dict(value['beta']);state.joint_steps = value['joint_steps']
        state.actor_updates = value['actor_updates'];state.critic_updates = value['reward_critic_updates']
        require(training_fingerprint(state) == value['training_state_sha256']
            and ppo['reference_sha256'] == state.reference_sha256
            and ppo['reference_buffer_sha256'] == core.buffer.source_sha256
            and ppo['actor_role_sha256'] == previous.role_hashes(state.model)
            and ppo['critic_sha256'] == previous.critic_initialization_sha256(state.model),
            'v643 full model and optimizer state do not match their boundary fingerprint')
        require(all(torch.equal(t,risk['cost_critic_state'][k])
            for k,t in state.model.cost_critic.state_dict().items()),
            'cost critic differs between full model and replay checkpoint')
        core.risk_training.restore(risk)
    except BaseException:
        state.model.load_state_dict(old['model'],strict=True)
        state.actor_optimizer.load_state_dict(old['actor']);state.critic_optimizer.load_state_dict(old['reward'])
        state.beta = old['beta'];state.joint_steps = state.actor_updates = state.critic_updates = 0
        for name,enabled in old['trainability'].items():parameters[name].requires_grad_(enabled)
        raise
    core.processed_joint_steps = ppo['processed_joint_steps']
    core.seen_complete_episodes = set(episodes);core.seen_batch_receipts = set(receipts)
    core.update_sequence = ppo['update_sequence']
    core.accepted_program_updates = dict(ppo['accepted_program_updates'])
    state.resumed_training_state = True
    return core
