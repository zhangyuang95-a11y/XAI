"""Training-time regularization toward a bounded executable policy program.

This module is deliberately environment-independent.  Environment adapters
collect semantic states and RCPD performs tree fitting; this class owns only
the research objective that connects the current neural policy to the frozen
program between extraction events.

The deployed controller remains the neural policy.  The program is a training
regularizer and an explanation/audit artifact, never the runtime controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np


PROGRAM_REGULARIZATION_PRESETS: Mapping[
    str,
    Mapping[str, float | int],
] = {
    # Canonical paper modes requested by the experiment protocol.
    "baseline": {
        "lambda_extract": 0.0,
        "lambda_complexity": 0.0,
        "extraction_interval": 500,
    },
    "RCPD": {
        "lambda_extract": 0.02,
        "lambda_complexity": 0.001,
        "extraction_interval": 500,
    },
    "strong_RCPD": {
        "lambda_extract": 0.05,
        "lambda_complexity": 0.005,
        "extraction_interval": 500,
    },
    # Backwards-compatible names keep old commands and saved manifests
    # readable.  The canonical modes above are used by new experiments.
    "fidelity_regularized": {
        "lambda_extract": 0.01,
        "lambda_complexity": 0.001,
        "extraction_interval": 500,
    },
    "strong_regularized": {
        "lambda_extract": 0.05,
        "lambda_complexity": 0.001,
        "extraction_interval": 500,
    },
}

PROGRAM_REGULARIZATION_MODES: Mapping[str, float] = {
    name: float(config["lambda_extract"])
    for name, config in PROGRAM_REGULARIZATION_PRESETS.items()
}


@dataclass(frozen=True)
class RegularizationStateBatch:
    """Precomputed program targets and optional absolute sample weights.

    MAPPO recomputes neural logits inside each minibatch so gradients reach the
    current Actor.  Program probabilities are frozen extraction targets.  This
    small carrier lets the generic three-argument ``regularization_loss`` API
    be used without placing extraction logic in the RL optimizer.
    """

    program_probabilities: Any
    sample_weights: Any | None = None


@dataclass(frozen=True)
class ProgramComplexity:
    depth: int
    leaves: int
    predicates: int
    normalized_depth: float
    normalized_leaves: float
    normalized_predicates: float
    loss: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "depth": self.depth,
            "leaves": self.leaves,
            "predicates": self.predicates,
            "normalized_depth": self.normalized_depth,
            "normalized_leaves": self.normalized_leaves,
            "normalized_predicates": self.normalized_predicates,
            "loss": self.loss,
        }


def program_complexity(
    program: Any,
    *,
    max_depth: int = 5,
    max_leaf_count: int = 16,
    max_predicate_count: int = 64,
) -> ProgramComplexity:
    """Return the exact normalized complexity objective used in experiments."""

    if program is None:
        return ProgramComplexity(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    if hasattr(program, "complexity"):
        raw = dict(program.complexity())
        depth = int(raw.get("depth", 0))
        leaves = int(raw.get("leaf_nodes", raw.get("leaves", 0)))
        predicates = int(raw.get("predicates", 0))
    elif hasattr(program, "root"):
        root = program.root
        depth = int(root.depth())
        leaves = int(root.leaf_count())
        predicates = int(len(root.used_predicates()))
    elif isinstance(program, Mapping):
        depth = int(program.get("depth", 0))
        leaves = int(program.get("leaf_nodes", program.get("leaves", 0)))
        predicates = int(program.get("predicates", 0))
    else:
        raise TypeError("Program must expose complexity(), root, or complexity fields.")

    normalized_depth = min(1.0, max(0.0, depth / max(1, int(max_depth))))
    normalized_leaves = min(
        1.0,
        max(0.0, leaves / max(1, int(max_leaf_count))),
    )
    normalized_predicates = min(
        1.0,
        max(0.0, predicates / max(1, int(max_predicate_count))),
    )
    loss = (
        0.5 * normalized_depth
        + 0.3 * normalized_leaves
        + 0.2 * normalized_predicates
    )
    return ProgramComplexity(
        depth=depth,
        leaves=leaves,
        predicates=predicates,
        normalized_depth=normalized_depth,
        normalized_leaves=normalized_leaves,
        normalized_predicates=normalized_predicates,
        loss=float(loss),
    )


class PolicyProgramRegularizer:
    """Periodic program extraction and ``KL(pi_nn || pi_program)`` objective."""

    def __init__(
        self,
        lambda_extract: float = 0.01,
        lambda_complexity: float = 0.001,
        extraction_interval: int = 500,
        *,
        extractor: Callable[..., Any] | None = None,
        max_depth: int = 5,
        max_leaf_count: int = 16,
        max_predicate_count: int = 64,
        epsilon: float = 1e-8,
    ) -> None:
        if not np.isfinite(lambda_extract) or lambda_extract < 0.0:
            raise ValueError("lambda_extract must be finite and non-negative.")
        if not np.isfinite(lambda_complexity) or lambda_complexity < 0.0:
            raise ValueError("lambda_complexity must be finite and non-negative.")
        if int(extraction_interval) <= 0:
            raise ValueError("extraction_interval must be positive.")
        if min(max_depth, max_leaf_count, max_predicate_count) <= 0:
            raise ValueError("Program complexity bounds must be positive.")
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be finite and in (0, 1).")
        self.lambda_extract = float(lambda_extract)
        self.lambda_complexity = float(lambda_complexity)
        self.extraction_interval = int(extraction_interval)
        self.extractor = extractor
        self.max_depth = int(max_depth)
        self.max_leaf_count = int(max_leaf_count)
        self.max_predicate_count = int(max_predicate_count)
        self.epsilon = float(epsilon)
        self.program: Any | None = None
        self.last_extraction_step: int | None = None
        self.last_fidelity_loss: float | None = None
        self.last_complexity: ProgramComplexity | None = None
        self.last_total_loss: float | None = None

    def with_lambda_extract(self, value: float) -> "PolicyProgramRegularizer":
        """Return an update-local view with the scheduled fidelity weight."""

        clone = PolicyProgramRegularizer(
            lambda_extract=value,
            lambda_complexity=self.lambda_complexity,
            extraction_interval=self.extraction_interval,
            extractor=self.extractor,
            max_depth=self.max_depth,
            max_leaf_count=self.max_leaf_count,
            max_predicate_count=self.max_predicate_count,
            epsilon=self.epsilon,
        )
        clone.program = self.program
        clone.last_extraction_step = self.last_extraction_step
        return clone

    def should_extract(self, step: int, *, force: bool = False) -> bool:
        return bool(
            force
            or (
                self.last_extraction_step is None
                and int(step) >= self.extraction_interval
            )
            or (
                self.last_extraction_step is not None
                and int(step) - self.last_extraction_step
                >= self.extraction_interval
            )
        )

    def extract_program(self, policy: Any, **kwargs: Any) -> Any:
        """Extract and retain the current policy using the configured adapter."""

        if self.extractor is not None:
            result = self.extractor(policy, **kwargs)
        elif hasattr(policy, "extract_program"):
            result = policy.extract_program(**kwargs)
        else:
            raise RuntimeError(
                "No program extractor is configured for PolicyProgramRegularizer."
            )
        extracted = getattr(result, "program", result)
        if extracted is not None:
            self.program = extracted
        step = kwargs.get("step")
        if step is not None:
            self.last_extraction_step = int(step)
        return result

    def compute_fidelity_loss(
        self,
        policy: Any,
        program: Any,
        states: Any,
    ) -> Any:
        """Return mean ``KL(pi_nn || pi_program)`` with optional row weights."""

        actor_output = self._policy_output(policy, states)
        program_output, weights = self._program_output(program, states)
        loss = self._forward_kl(actor_output, program_output, weights)
        self.last_fidelity_loss = _as_float(loss)
        return loss

    def compute_complexity_loss(self, program: Any) -> float:
        complexity = program_complexity(
            program,
            max_depth=self.max_depth,
            max_leaf_count=self.max_leaf_count,
            max_predicate_count=self.max_predicate_count,
        )
        self.last_complexity = complexity
        return complexity.loss

    def regularization_loss(
        self,
        policy: Any,
        program: Any,
        states: Any,
    ) -> Any:
        """Return ``lambda_extract*KL + lambda_complexity*complexity``.

        Tree structure is discrete, so the complexity term is intentionally a
        detached scalar for Actor backpropagation.  It directly affects tree
        candidate selection in RCPD; the Actor receives simplicity pressure
        through KL toward that bounded, complexity-selected program.
        """

        fidelity = self.compute_fidelity_loss(policy, program, states)
        structural_program = (
            self.program
            if isinstance(states, RegularizationStateBatch)
            and not _looks_like_executable_program(program)
            else program
        )
        complexity = self.compute_complexity_loss(structural_program)
        total = self.lambda_extract * fidelity
        try:
            import torch

            if isinstance(fidelity, torch.Tensor):
                total = total + fidelity.new_tensor(
                    self.lambda_complexity * complexity
                )
            else:
                total = total + self.lambda_complexity * complexity
        except ImportError:
            total = total + self.lambda_complexity * complexity
        self.last_total_loss = _as_float(total)
        return total

    def _policy_output(self, policy: Any, states: Any) -> Any:
        if _is_array_like(policy):
            return policy
        raw_states = states
        if hasattr(policy, "distribution_for_states"):
            return policy.distribution_for_states(raw_states)
        if callable(policy):
            try:
                return policy(raw_states)
            except (TypeError, ValueError):
                return [policy(state) for state in raw_states]
        raise TypeError("Policy must be logits/probabilities or a callable policy.")

    def _program_output(self, program: Any, states: Any) -> tuple[Any, Any | None]:
        if isinstance(states, RegularizationStateBatch):
            return states.program_probabilities, states.sample_weights
        if _is_array_like(program):
            return program, None
        if not _looks_like_executable_program(program):
            raise TypeError(
                "Program must expose predict_proba() or provide probability rows."
            )
        rows = []
        for state in states:
            mapping = program.predict_proba(state)
            action_names = tuple(program.action_names)
            rows.append([float(mapping[action]) for action in action_names])
        return np.asarray(rows, dtype=np.float64), None

    def _forward_kl(
        self,
        actor_output: Any,
        program_output: Any,
        weights: Any | None,
    ) -> Any:
        try:
            import torch

            if isinstance(actor_output, torch.Tensor):
                actor_probabilities = _torch_probabilities(actor_output)
                target = torch.as_tensor(
                    program_output,
                    dtype=actor_probabilities.dtype,
                    device=actor_probabilities.device,
                )
                if target.ndim == 1:
                    target = target.unsqueeze(0)
                target = target.clamp_min(self.epsilon)
                target = target / target.sum(dim=-1, keepdim=True)
                actor_safe = actor_probabilities.clamp_min(self.epsilon)
                losses = torch.sum(
                    actor_probabilities
                    * (torch.log(actor_safe) - torch.log(target)),
                    dim=-1,
                )
                if weights is not None:
                    weight_tensor = torch.as_tensor(
                        weights,
                        dtype=losses.dtype,
                        device=losses.device,
                    )
                    losses = losses * weight_tensor
                return losses.mean()
        except ImportError:
            pass

        actor = _numpy_probabilities(actor_output)
        target = np.asarray(program_output, dtype=np.float64)
        if target.ndim == 1:
            target = target[None, :]
        target = np.clip(target, self.epsilon, None)
        target = target / target.sum(axis=-1, keepdims=True)
        actor_safe = np.clip(actor, self.epsilon, None)
        losses = np.sum(actor * (np.log(actor_safe) - np.log(target)), axis=-1)
        if weights is not None:
            losses = losses * np.asarray(weights, dtype=np.float64)
        return float(np.mean(losses))


def _looks_like_executable_program(value: Any) -> bool:
    return hasattr(value, "predict_proba") and hasattr(value, "complexity")


def _is_array_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except ImportError:
        return False


def _numpy_probabilities(output: Any) -> np.ndarray:
    values = np.asarray(output, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    is_probability = bool(
        np.all(values >= 0.0)
        and np.allclose(values.sum(axis=-1), 1.0, atol=1e-5)
    )
    if not is_probability:
        values = values - np.max(values, axis=-1, keepdims=True)
        values = np.exp(values)
    return values / np.clip(values.sum(axis=-1, keepdims=True), 1e-12, None)


def _torch_probabilities(output: Any) -> Any:
    import torch

    values = output
    if values.ndim == 1:
        values = values.unsqueeze(0)
    detached = values.detach()
    is_probability = bool(
        torch.all(detached >= 0.0).item()
        and torch.allclose(
            detached.sum(dim=-1),
            torch.ones_like(detached.sum(dim=-1)),
            atol=1e-5,
        )
    )
    if is_probability:
        return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.softmax(values, dim=-1)


def _as_float(value: Any) -> float:
    try:
        return float(value.detach().cpu())
    except AttributeError:
        return float(value)


__all__ = [
    "PROGRAM_REGULARIZATION_MODES",
    "PROGRAM_REGULARIZATION_PRESETS",
    "PolicyProgramRegularizer",
    "ProgramComplexity",
    "RegularizationStateBatch",
    "program_complexity",
]
