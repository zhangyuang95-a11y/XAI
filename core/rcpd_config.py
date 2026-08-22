"""Configuration and neural-oracle contracts for RCPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


State = Any


@dataclass(frozen=True)
class OracleOutput:
    """Action distribution and optional action values from the neural oracle."""

    probabilities: Mapping[str, float]
    q_values: Mapping[str, float] | None = None

    def normalized(self, action_names: Sequence[str]) -> np.ndarray:
        values = np.asarray([max(0.0, float(self.probabilities.get(action, 0.0))) for action in action_names])
        total = float(values.sum())
        if total <= 0.0:
            raise ValueError("Oracle probabilities must have positive mass.")
        return values / total


class OraclePolicy(Protocol):
    def __call__(self, state: State) -> OracleOutput | Mapping[str, float]: ...


@dataclass(frozen=True)
class RCPDConfig:
    enabled: bool = True
    extraction_interval: int = 500
    minimum_extraction_samples: int = 32
    # The program is intentionally small.  It acts as a regularity constraint,
    # rather than trying to reproduce every irregular neural boundary.
    regularization_lambda: float = 0.01
    # Extraction starts immediately, but feedback can be delayed until the
    # Actor has learned a useful task policy.  This prevents a tiny program
    # fitted to an early, nearly random Actor from becoming a self-reinforcing
    # regularity reference.  The configured lambda is reached after the optional linear
    # ramp and is then held for the rest of training.
    regularization_start_fraction: float = 0.0
    regularization_ramp_fraction: float = 0.0
    # Explicit safety ceiling for experiments.  The training entry point
    # rejects values above the configured ceiling instead of silently
    # truncating them, because a reported lambda must equal the lambda that
    # actually reached the Actor loss.
    maximum_regularization_lambda: float = 1.0
    minimum_target_weight: float = 0.20
    # A bounded program is deliberately imperfect.  When it disagrees with a
    # confident Actor, blindly imitating it can destroy a task-critical
    # decision.  Disagreement feedback is therefore restricted to Actor
    # decisions whose top-two probability margin is below this threshold.
    # Agreement states retain the ordinary program-confidence weight.
    maximum_disagreement_actor_margin: float = 0.10
    # A bounded program must never become a replacement controller.  When this
    # guard is enabled it can regularise only states where its selected action
    # already matches the current NN action.  The KL term may then smooth an
    # existing neural decision boundary, but it cannot introduce a different
    # action class into the Actor.  The domain-neutral default remains false
    # for backwards compatibility; the Warehouse training entry point enables
    # the stricter research protocol by default.
    require_action_agreement_for_feedback: bool = False
    # ``program_distribution`` preserves the soft distribution represented by
    # each simple tree region.  In the Warehouse protocol it is applied only
    # after action-agreement, uncertainty, fidelity, and gradient guards, so it
    # regularises a decision class the NN already selected rather than teaching
    # the program to act as a replacement controller.
    # ``program_blend`` keeps the NN's detached distribution as the proximal
    # reference and moves only a bounded fraction toward the extracted program.
    # ``action_anchor`` is retained as an explicit ablation that moves toward a
    # one-hot program action.  The two latter modes are retained as ablations.
    feedback_target_mode: str = "program_distribution"
    feedback_target_strength: float = 0.10
    # Apply the program only near the NN's uncertain decision boundary.  The
    # domain-neutral core keeps a permissive default for compatibility; the
    # RL entry point uses a much smaller explicit value.
    maximum_feedback_actor_margin: float = 1.0
    # Tree leaves contain averages of many Actor distributions.  The default
    # preserves those calibrated soft targets; temperatures below one are
    # explicit sharpening ablations, and zero is a hard one-hot target.
    program_target_temperature: float = 1.0
    max_depth: int = 5
    max_leaf_nodes: int | None = 16
    max_predicates: int | None = 64
    min_samples_leaf: int = 8
    validation_fraction: float = 0.20
    complexity_penalty: float = 0.001
    distribution_penalty: float = 0.20
    interaction_loss_weight: float = 0.50
    counterfactual_loss_weight: float = 0.20
    # Optional paired-causal bonus for feature preselection. It is disabled
    # by default because even a small bonus can evict ordinary task features
    # from a tight predicate budget; experiments may enable it explicitly.
    counterfactual_feature_selection_weight: float = 0.0
    # Scikit-learn's regression tree otherwise optimises every state
    # independently.  Upweighting both endpoints of an NN-changing legal pair
    # lets the bounded structure spend scarce splits on observable factors that
    # really alter the NN decision.  The label remains the current NN output;
    # no counterfactual action is invented. One preserves the neutral baseline.
    counterfactual_changed_pair_weight: float = 1.0
    # Optional one-hot auxiliary outputs affect split selection only.  The
    # exported leaf probabilities are still the mean soft NN distributions in
    # that leaf, so this can improve action-boundary fidelity without turning
    # the program into a hard classifier or a replacement controller.
    action_structure_weight: float = 0.0
    # Domain-neutral defaults remain permissive. Training entry points
    # that request relational feedback set explicit non-zero guards.
    minimum_overall_fidelity_for_feedback: float = 0.0
    minimum_interaction_fidelity_for_feedback: float = 0.0
    minimum_interaction_validation_samples: int = 0
    # Argmax fidelity alone is insufficient for a soft KL regularity target.  A tree
    # whose leaf distribution is far from the Actor can strongly reshape the
    # shared network even when both choose the same action.
    maximum_mean_kl_for_feedback: float | None = None
    # Explanation reliability is intentionally stricter and independent from
    # the training-feedback gate.  A locally safe program can provide a small
    # regularity signal without automatically becoming evidence for a user-
    # facing claim about why the NN acted.  Domain-neutral defaults remain
    # permissive; environment entry points set explicit audit thresholds.
    minimum_overall_fidelity_for_explanation: float = 0.0
    minimum_interaction_fidelity_for_explanation: float = 0.0
    maximum_mean_kl_for_explanation: float | None = None
    minimum_counterfactual_direction_fidelity_for_explanation: float = 0.0
    minimum_counterfactual_changed_pairs_for_explanation: int = 0
    # Minimum number of complete baseline/intervention pairs in the complete
    # extraction dataset.  This is separate from the held-out changed-pair
    # threshold: unchanged pairs are still important coverage evidence.
    minimum_counterfactual_pairs: int = 0
    # Relational training may require the same paired-causal audit used by
    # explanations.  The core default remains permissive for non-causal
    # domains; environment entry points can enable the stricter contract.
    require_explanation_eligibility_for_feedback: bool = False
    # A more targeted safeguard keeps ordinary low-margin regularisation
    # active while withholding counterfactual probe rows from an explanation-
    # ineligible program.  This avoids both causal leakage and an all-or-
    # no zero-weight training stall.
    require_explanation_eligibility_for_counterfactual_feedback: bool = False
    importance_weight_scale: float = 8.0
    random_seed: int = 2026
