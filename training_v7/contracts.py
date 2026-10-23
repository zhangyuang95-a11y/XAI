"""One immutable scientific method; domain names never select a training loss."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

CANDIDATE = 'three-domain-unified-02'
RELEASE = 'policylens-three-domain-20261024.v7.0'
DOMAINS = ('warehouse', 'pong', 'kitchen')
BC_STAGES = (
    ('initial_public_teacher', 100, 20, 256, .0003, 'teacher'),
    ('initial_nn_dagger_1', 40, 10, 256, .0003, 'sampled'),
    ('initial_nn_dagger_2', 40, 10, 256, .0003, 'sampled'),
    ('joint_augmentation_sampled', 25, 8, 128, .0001, 'sampled'),
    ('joint_augmentation_argmax', 25, 8, 128, .0001, 'argmax'),
    ('risk_public_teacher', 40, 8, 128, .0001, 'teacher'),
    ('risk_nn_visitation', 80, 8, 128, .0001, 'sampled'),
    ('large_public_teacher', 200, 8, 128, .0001, 'teacher'),
    ('large_nn_visitation', 120, 8, 128, .0001, 'sampled'),
    ('deep_public_teacher', 200, 6, 128, .0001, 'teacher'),
    *((f'deep_{i}_{kind}', 50, 6, 128, .0001, kind)
      for i in range(1, 5) for kind in ('sampled', 'argmax')),
)
assert sum(s[1] for s in BC_STAGES) == 1270
BC_FITS = (
    ('initial_public_bc', ('initial_public_teacher',), 20, 256, .0003),
    ('dagger_1', ('initial_public_teacher', 'initial_nn_dagger_1'), 10, 256, .0003),
    ('dagger_2', ('initial_public_teacher', 'initial_nn_dagger_1', 'initial_nn_dagger_2'), 10, 256, .0003),
    ('joint', ('joint_augmentation_sampled', 'joint_augmentation_argmax'), 8, 256, .0001),
    ('risk', ('risk_public_teacher', 'risk_nn_visitation', 'joint_augmentation_sampled', 'joint_augmentation_argmax'), 8, 128, .0001),
    ('large', ('large_public_teacher', 'large_nn_visitation', 'joint_augmentation_sampled', 'joint_augmentation_argmax'), 8, 128, .0001),
    *((f'deep_{i}', ('deep_public_teacher', 'joint_augmentation_sampled', 'joint_augmentation_argmax',
        *(f'deep_{j}_{kind}' for j in range(1, i+1) for kind in ('sampled', 'argmax'))), 6, 128, .0001) for i in range(1, 5)),
    ('final', ('deep_public_teacher', 'joint_augmentation_sampled', 'joint_augmentation_argmax',
        *(f'deep_{j}_{kind}' for j in range(1, 5) for kind in ('sampled', 'argmax'))), 6, 128, .0001),
)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False, indent=2)
        f.write('\n')


@dataclass(frozen=True)
class MethodSpec:
    version: str = 'three-domain-unified-method.v2'
    end_semantics: str = 'native_time_limit_truncated_true_terminal_and_training_cut_separate.v1'
    actor_width: int = 256
    gate_width: int = 64
    internal_depth: int = 5
    internal_leaves: int = 16
    source_limit: int = 3
    bc_stages: tuple = BC_STAGES
    bc_fits: tuple = BC_FITS
    bc_tree_ce: float = .20
    bc_distillation: float = .20
    bc_alignment: float = .03
    bc_final_epochs: int = 6
    sample_weight: float = 1.
    execution_selection: str = 'bound_seed_PCG64_two_uniforms_per_physical_frame_categorical.v1'
    parallel_environments: int = 8
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch: int = 256
    actor_lr: float = 3e-5
    critic_lr: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    entropy: float = .005
    critic_weight: float = .5
    gradient_clip: float = .5
    critic_warmup: int = 4000
    reference_kl_target: float = .01
    beta_initial: float = 1.
    beta_min: float = .001
    beta_max: float = 100.
    beta_rate: float = .05
    extraction_percentages: tuple = (75, 84, 93)
    program_start: float = .80
    program_ramp: float = .05
    program_lambda: float = .01
    program_depth: int = 12
    program_leaves: int = 256
    program_fit_seconds: int = 3600
    program_final_seconds: int = 600
    leaf_calls: int = 20000
    proof_calls: int = 100000
    final_leaf_call_reserve: int = 2048
    final_proof_call_reserve: int = 32768
    evidence_counts: tuple = (900, 150, 300, 300)
    risk_target: float = .02
    risk_release: float = .01
    risk_window: int = 256
    risk_minimum: int = 128
    risk_confidence: float = .90
    cost_sampling: str = 'uniform_complete_episodes_then_uniform_frames_unbiased_episode_sum.v1'
    budgets: tuple = (200000, 300000, 400000)
    calibration_steps: int = 20000
    development_pairs: int = 3
    formal_pairs: int = 3
    calibration_scenes: int = 75
    pressure_scenes: int = 72
    pressure_trajectories: int = 3
    heldout_per_task: int = 100

    def __post_init__(self):
        # This version has one method, not a permissive tuning config.
        expected = {f.name: f.default for f in self.__dataclass_fields__.values()}
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError('method changes require a new registered version')

    @property
    def sha256(self):
        return digest(asdict(self))

    def coefficient(self, step, budget):
        return self.program_lambda * min(1., max(0., (step / budget - self.program_start) / self.program_ramp))


@dataclass(frozen=True)
class Feature:
    name: str
    lower: float
    upper: float
    sources: tuple[str, ...]
    reason: str

    def __post_init__(self):
        import math
        if (not self.name or not self.sources or not self.reason or
                not math.isfinite(self.lower) or not math.isfinite(self.upper) or
                self.lower > self.upper or len(set(self.sources)) != len(self.sources)):
            raise ValueError('feature needs finite physical bounds and explicit public sources')


@dataclass(frozen=True)
class RoleSpec:
    name: str
    actions: tuple[str, ...]
    features: tuple[Feature, ...]
    legal_indices: tuple[int, ...]
    position_indices: tuple[int, int, int, int]
    action_deltas: tuple[tuple[float, float], ...]
    action_feature_dim: int
    wait_index: int

    def __post_init__(self):
        if (not self.name or len(set(self.actions)) != len(self.actions) or not self.actions or
                len(self.legal_indices) != len(self.actions) or
                len(self.action_deltas) != len(self.actions) or
                not 0 <= self.wait_index < len(self.actions) or
                len({f.name for f in self.features}) != len(self.features) or
                any(not 0 <= i < len(self.features) for i in (*self.legal_indices, *self.position_indices))):
            raise ValueError('invalid role dimensions or physical action contract')

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class DomainSpec:
    name: str
    roles: tuple[RoleSpec, RoleSpec]
    physics_sha256: str
    failure_definition: str
    reward_definition: str
    horizon: int

    def __post_init__(self):
        if (self.name not in DOMAINS or len(self.roles) != 2 or self.roles[0].name == self.roles[1].name
                or len(self.physics_sha256) != 64 or self.horizon < 1):
            raise ValueError('invalid domain specification')

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ExperimentSpec:
    domain_sha256: str
    method_sha256: str
    phase: str
    pair_id: int
    initialization_seed: int
    optimizer_seed: int
    scenario_sha256: str
    candidate: str = CANDIDATE
    release: str = RELEASE

    def __post_init__(self):
        if (self.method_sha256 != MethodSpec().sha256 or self.phase not in ('calibration', 'development', 'formal')
                or any(len(v) != 64 for v in (self.domain_sha256, self.scenario_sha256))
                or self.candidate != CANDIDATE or self.release != RELEASE
                or any(type(v) is not int or v < 0 for v in (self.pair_id, self.initialization_seed, self.optimizer_seed))):
            raise ValueError('invalid immutable experiment')


def source_support(features, indices):
    return set().union(*(set(features[i].sources) for i in indices))


def program_gates(metrics, *, formal=False):
    required = dict(feedback_coverage=.20 if formal else .45,
                    explanation_coverage=.20 if formal else .45,
                    expected_feedback_coverage=.20 if formal else .40,
                    feedback_agreement=.85 if formal else .95,
                    explanation_agreement=.95 if formal else .97)
    return (all(metrics[k] >= v for k, v in required.items())
            and all(0 <= metrics[k] <= .10 for k in ('feedback_kl', 'explanation_kl'))
            and metrics['illegal_actions'] == metrics['unproved_paths'] == 0
            and metrics['support_passed'] is True)
