"""External adapters own the environment and semantic features; no domain code here."""
from dataclasses import dataclass, asdict
from typing import Protocol, Mapping, Any
import numpy as np

@dataclass(frozen=True)
class EnvSpec:
    observation_size: int
    state_size: int
    action_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    state_names: tuple[str, ...]
    agents: int = 1
    version: str = "1"

    def signature(self):
        if self.agents < 1 or len(self.action_names) < 2:
            raise ValueError("Need fixed positive agents and at least two discrete actions")
        for names, size in [(self.observation_names, self.observation_size),
                            (self.state_names, self.state_size),
                            (self.feature_names, len(self.feature_names)),
                            (self.action_names, len(self.action_names))]:
            if not size or len(names) != size or len(set(names)) != size:
                raise ValueError("Names must be unique, ordered, and match dimensions")
        return asdict(self)

@dataclass
class Frame:
    observations: np.ndarray  # [agents, observation_size]
    state: np.ndarray         # [state_size], centralized critic input
    public: Any = None        # already observed information only

@dataclass
class Transition:
    frame: Frame              # terminal frame BEFORE reset, never auto-reset
    rewards: np.ndarray      # [agents]; team rewards can be identical
    terminated: np.ndarray   # [agents], true terminal: no bootstrap
    truncated: np.ndarray    # [agents], time limit: bootstrap terminal frame
    actor_mask: np.ndarray   # [agents], false if an external controller chose action
    submitted_actions: np.ndarray  # actual actions submitted to environment

class Environment(Protocol):
    spec: EnvSpec
    def reset(self, seed: int | None = None) -> Frame: ...
    def step(self, actions: np.ndarray) -> Transition: ...
    def features(self, frame: Frame, agent: int) -> Mapping[str, float]: ...
    # Optional: snapshot() -> serializable value, restore(value) -> None.
    # Capture ALL environment/controller RNG, episode and history state.
    # Fixed agents terminate/reset synchronously; asynchronous reset is unsupported.

class NeuralPolicy(Protocol):
    def probabilities(self, observations, roles): ...


def validate_frame(frame, spec):
    for value, shape in [(frame.observations, (spec.agents, spec.observation_size)),
                         (frame.state, (spec.state_size,))]:
        if np.asarray(value).shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid frame; expected finite {shape}")


def encode_features(env, frame, agent):
    row = env.features(frame, agent)
    if tuple(row) != tuple(env.spec.feature_names) or not np.isfinite(list(row.values())).all():
        raise ValueError("Feature order/names or finite values do not match specification")
    return dict(row)
