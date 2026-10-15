"""Native RCPD extraction and optional, gated neural-policy regularization.

Callers supply observations from real neural trajectories, labelled again by
one frozen current Actor. Trees are fitted on training episodes and selected
on separate validation episodes. The tree never supplies executed actions or
PPO action labels. Complexity selects discrete trees; only KL has an Actor
gradient. Foundation training may keep this entire optional module disabled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core.policy_program_regularizer import program_complexity
from core.program import ExecutableProgram, ProgramNode
from core.rcpd import RCPD, RCPDConfig

from .policy import ACTIONS


FEEDBACK_VERSION = "warehouse_native_rcpd_feedback_v1"


@dataclass(frozen=True)
class FeedbackConfig:
    warmup_steps: int = 100_000
    ramp_steps: int = 100_000
    lambda_max: float = .01
    minimum_fidelity: float = .90
    minimum_critical_fidelity: float = .85
    maximum_mean_kl: float = .35
    maximum_performance_drop_fraction: float = .10
    complexity_weight: float = .001
    depths: tuple[int, ...] = (4, 6, 8)
    leaves: tuple[int, ...] = (16, 32, 64)
    min_samples_leaf: int = 8
    minimum_training_rows: int = 64
    minimum_validation_rows: int = 32
    critical_groups: tuple[str, ...] = ("narrow_passage", "shared_pickup", "shared_charger")
    minimum_critical_rows: int = 1
    seed: int = 260908

    def __post_init__(self):
        if min(self.warmup_steps, self.ramp_steps) < 0:
            raise ValueError("Feedback schedule cannot be negative")
        if not self.depths or not self.leaves or min(*self.depths, *self.leaves) <= 0:
            raise ValueError("Extraction bounds must be positive")
        if min(self.min_samples_leaf, self.minimum_training_rows,
               self.minimum_validation_rows, self.minimum_critical_rows) <= 0:
            raise ValueError("Feedback sample requirements must be positive")
        for name in ("lambda_max", "maximum_mean_kl", "complexity_weight"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("Invalid feedback coefficient: " + name)
        for name in ("minimum_fidelity", "minimum_critical_fidelity", "maximum_performance_drop_fraction"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError("Invalid feedback fraction: " + name)
        if not self.critical_groups or len(set(self.critical_groups)) != len(self.critical_groups):
            raise ValueError("Critical categories must be explicit and unique")


class _NativeRCPD(RCPD):
    def _limit_feature_names(self, samples, feature_names, *, action_names, required):
        # Native role bits are public inputs to the Actor. The generic core
        # excludes concrete robot-name features for older adapters; applying
        # that exclusion here would silently give the tree different inputs.
        return tuple(feature_names)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _fingerprint(row: np.ndarray) -> str:
    return sha256(np.asarray(row, dtype="<f4").tobytes()).hexdigest()


def _json_safe(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


class FeedbackManager:
    def __init__(self, feature_names: Sequence[str], config: FeedbackConfig | None = None):
        self.feature_names = tuple(str(name) for name in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Native tree feature names must be nonempty and unique")
        self.config = config or FeedbackConfig()
        self.program: ExecutableProgram | None = None
        self.reliable = False
        self.current_lambda = 0.0
        self.last_step = 0
        self.last_fit_step: int | None = None
        self.ramp_start: int | None = None
        self.performance_anchor_ratio: float | None = None
        self.reference_score: float | None = None
        self.last_fit_report: dict[str, Any] = {}
        self.last_gate: dict[str, Any] = {"reason": "not_extracted", "active": False}

    def _observations(self, value) -> np.ndarray:
        result = _array(value)
        if result.ndim < 1 or result.shape[-1] != len(self.feature_names) or not np.isfinite(result).all():
            raise ValueError("Invalid native feedback observations")
        return result

    def _dataset(self, obs, probs, episode_ids, minimum, name):
        obs = self._observations(obs)
        probs = _array(probs)
        if obs.ndim != 2 or len(obs) < minimum or probs.shape != (len(obs), len(ACTIONS)):
            raise ValueError(name + " extraction data have insufficient rows or incompatible dimensions")
        if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(-1), 1, atol=1e-5):
            raise ValueError("Extraction labels must be actual normalized neural soft probabilities")
        if episode_ids is None or len(episode_ids) != len(obs):
            raise ValueError("Every extraction row requires its source episode ID")
        episodes = tuple(str(value) for value in episode_ids)
        if any(not value for value in episodes) or len(set(episodes)) < 2:
            raise ValueError(name + " requires at least two explicitly identified episodes")
        return obs, probs / probs.sum(-1, keepdims=True), episodes

    @staticmethod
    def _groups(values, length):
        if values is None:
            return tuple(() for _ in range(length))
        if len(values) != length:
            raise ValueError("Critical labels must correspond to observation rows")
        return tuple((str(value),) if isinstance(value, str)
                     else tuple(str(group) for group in value) for value in values)

    def _predict(self, program: ExecutableProgram, observations) -> np.ndarray:
        values = self._observations(observations)
        flat = values.reshape(-1, len(self.feature_names))
        output = np.empty((len(flat), len(ACTIONS)), dtype=np.float32)
        columns = {name: index for index, name in enumerate(self.feature_names)}

        def visit(node: ProgramNode, indices):
            if not len(indices):
                return
            if node.is_leaf:
                output[indices] = node.probabilities
                return
            left = flat[indices, columns[node.feature]] <= node.threshold
            visit(node.left, indices[left])
            visit(node.right, indices[~left])

        visit(program.root, np.arange(len(flat)))
        return output.reshape(*values.shape[:-1], len(ACTIONS))

    def fit(self, train_obs, train_probs, val_obs, val_probs, *, step: int,
            source_actor_sha256: str, train_episode_ids, val_episode_ids,
            train_groups=None, val_groups=None) -> dict[str, Any]:
        """Fit only training labels; use the independent validation set to select.

        ``*_probs`` must be re-evaluated with the SAME frozen current Actor,
        never teacher actions, selected tree actions, or stale rollout logits.
        Episode IDs are mandatory; exact shared observations are rejected too.
        """
        self.current_lambda = 0.0
        self.reliable = False
        self.last_gate = {"reason": "extracting_or_failed", "active": False}
        if step < self.last_step:
            raise ValueError("Extraction step cannot move backwards")
        if len(source_actor_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_actor_sha256):
            raise ValueError("Frozen source Actor SHA-256 is required")
        x, y, train_ids = self._dataset(train_obs, train_probs, train_episode_ids,
                                       self.config.minimum_training_rows, "Training")
        vx, vy, val_ids = self._dataset(val_obs, val_probs, val_episode_ids,
                                       self.config.minimum_validation_rows, "Validation")
        if set(train_ids) & set(val_ids):
            raise ValueError("Training and validation episodes must be disjoint")
        train_fingerprints = {_fingerprint(row) for row in x}
        val_fingerprints = {_fingerprint(row) for row in vx}
        if train_fingerprints & val_fingerprints:
            raise ValueError("Exact observations overlap between extraction and validation")
        groups = self._groups(train_groups, len(x))
        validation_groups = self._groups(val_groups, len(vx))
        samples = [{"obs": row, "probabilities": probabilities, "episode": episode, "groups": labels}
                   for row, probabilities, episode, labels in zip(x, y, train_ids, groups)]
        validation = [{"obs": row, "probabilities": probabilities, "episode": episode, "groups": labels}
                      for row, probabilities, episode, labels in zip(vx, vy, val_ids, validation_groups)]
        encoder = lambda item: dict(zip(self.feature_names, map(float, item["obs"])))
        oracle = lambda item: dict(zip(ACTIONS, map(float, item["probabilities"])))
        candidates = []
        for depth in self.config.depths:
            for leaves in self.config.leaves:
                extractor = _NativeRCPD(RCPDConfig(
                    max_depth=depth, max_leaf_nodes=leaves, max_predicates=None,
                    min_samples_leaf=self.config.min_samples_leaf,
                    complexity_penalty=self.config.complexity_weight,
                    random_seed=self.config.seed, regularization_lambda=self.config.lambda_max,
                ))
                result = extractor.fit(
                    samples, oracle, encoder, validation_states=validation,
                    split_group_provider=lambda item: item["episode"],
                    program_metadata={"native_source_actor_sha256": source_actor_sha256,
                                      "native_feedback_version": FEEDBACK_VERSION,
                                      "runtime_controller": "native_neural_actor_only"},
                )
                prediction = self._predict(result.program, vx)
                agreement = prediction.argmax(-1) == vy.argmax(-1)
                mean_kl = float(np.mean(np.sum(
                    vy * (np.log(vy.clip(1e-8)) - np.log(prediction.clip(1e-8))), axis=-1,
                )))
                critical = {}
                for group in self.config.critical_groups:
                    mask = np.asarray([group in labels for labels in validation_groups])
                    critical[group] = {
                        "rows": int(mask.sum()),
                        "episodes": len({episode for episode, present in zip(val_ids, mask) if present}),
                        "fidelity": float(agreement[mask].mean()) if mask.any() else None,
                    }
                complexity = program_complexity(
                    result.program, max_depth=max(self.config.depths),
                    max_leaf_count=max(self.config.leaves),
                    max_predicate_count=max(self.config.leaves) - 1,
                )
                reliable = bool(
                    agreement.mean() >= self.config.minimum_fidelity
                    and mean_kl <= self.config.maximum_mean_kl
                    and all(item["rows"] >= self.config.minimum_critical_rows
                            and item["fidelity"] is not None
                            and item["fidelity"] >= self.config.minimum_critical_fidelity
                            for item in critical.values())
                )
                report = {
                    "depth_cap": depth, "leaf_cap": leaves,
                    "fidelity": float(agreement.mean()), "mean_kl": mean_kl,
                    "critical": critical, "complexity": complexity.to_dict(), "reliable": reliable,
                    "selection_objective": float(1 - agreement.mean() + .2 * mean_kl
                                                 + self.config.complexity_weight * complexity.loss),
                }
                # Generic core defaults do not certify the native protocol.
                # The serialized program must carry the same fail-closed
                # verdict as this manager, including missing critical groups.
                metadata = {
                    **dict(result.program.metadata),
                    "native_feedback_config": asdict(self.config),
                    "metrics": {**report, "feedback_eligible": reliable,
                                "explanation_eligible": False,
                                "explanation_ineligibility_reasons": ["independent_intervention_audit_not_run"]},
                    "program_roles": ["optional_training_regularity_signal", "explanation_evidence_pending_audit"],
                }
                program = ExecutableProgram(result.program.action_names, result.program.feature_names,
                                            result.program.root, metadata)
                candidates.append((report, program))
        reliable_candidates = [candidate for candidate in candidates if candidate[0]["reliable"]]
        if reliable_candidates:
            chosen = min(reliable_candidates, key=lambda item: (
                item[0]["complexity"]["loss"], item[0]["mean_kl"], -item[0]["fidelity"],
                item[0]["depth_cap"], item[0]["leaf_cap"],
            ))
        else:
            chosen = min(candidates, key=lambda item: item[0]["selection_objective"])
        selected, self.program = chosen
        self.reliable = selected["reliable"]
        self.last_fit_step = int(step)
        self.last_step = int(step)
        self.last_fit_report = {
            "version": FEEDBACK_VERSION, "step": int(step), "source_actor_sha256": source_actor_sha256,
            "source": "same_frozen_actor_soft_probabilities_on_neural_trajectory_observations",
            "train_rows": len(x), "validation_rows": len(vx),
            "train_episodes": len(set(train_ids)), "validation_episodes": len(set(val_ids)),
            "episode_overlap": 0, "exact_observation_overlap": 0,
            "training_observation_sha256": sha256(x.tobytes()).hexdigest(),
            "validation_observation_sha256": sha256(vx.tobytes()).hexdigest(),
            "training_probabilities_sha256": sha256(y.tobytes()).hexdigest(),
            "validation_probabilities_sha256": sha256(vy.tobytes()).hexdigest(),
            "selected": selected, "candidates": [item[0] for item in candidates],
            "reliable": self.reliable, "complexity_has_actor_gradient": False,
            "explanation_qualified": False, "intervention_direction_not_tested_here": True,
        }
        return _json_safe(self.last_fit_report)

    def update(self, step: int, validation_score: float | None, *,
               capability_eligible: bool = False, reference_score: float | None = None) -> dict[str, Any]:
        """Gate on complete caller capability checks and ramp the feedback weight.

        Scores may be deliveries or another frozen metric. A positive reference
        is mandatory; regression is relative to the Actor/reference ratio at
        first feedback activation, not a claim about literal success rates.
        """
        if step < self.last_step:
            raise ValueError("Feedback schedule cannot move backwards")
        self.last_step = int(step)
        reason = None
        ratio = None
        if step < self.config.warmup_steps:
            reason = "warmup"
        elif capability_eligible is not True:
            reason = "capability_gate"
        elif reference_score is None or not np.isfinite(reference_score) or reference_score <= 0:
            reason = "missing_positive_reference"
        elif validation_score is None or not np.isfinite(validation_score) or validation_score < 0:
            reason = "invalid_validation_score"
        elif not self.reliable or self.program is None:
            reason = "unreliable_or_missing_program"
        else:
            ratio = float(validation_score / reference_score)
            if ratio <= 0:
                reason = "non_positive_performance_ratio"
            elif self.reference_score is not None and float(reference_score) != self.reference_score:
                reason = "reference_changed"
            elif (self.performance_anchor_ratio is not None
                  and ratio < self.performance_anchor_ratio * (1 - self.config.maximum_performance_drop_fraction)):
                reason = "performance_regression"
        if reason is not None:
            self.current_lambda = 0.0
            self.ramp_start = None
        else:
            if self.performance_anchor_ratio is None:
                self.performance_anchor_ratio = ratio
                self.reference_score = float(reference_score)
            if self.ramp_start is None:
                self.ramp_start = int(step)
            multiplier = (1.0 if self.config.ramp_steps == 0
                          else min(1.0, max(0.0, (step - self.ramp_start) / self.config.ramp_steps)))
            self.current_lambda = self.config.lambda_max * multiplier
        self.last_gate = {
            "step": int(step), "active": bool(self.current_lambda > 0),
            "reason": reason or ("active" if self.current_lambda > 0 else "ramp_start"),
            "lambda": self.current_lambda,
            "validation_score": float(validation_score) if validation_score is not None and np.isfinite(validation_score) else None,
            "reference_score": float(reference_score) if reference_score is not None and np.isfinite(reference_score) else None,
            "performance_ratio": ratio,
            "performance_anchor_ratio": self.performance_anchor_ratio,
            "reliable_program": self.reliable,
        }
        return dict(self.last_gate)

    def targets(self, observations) -> np.ndarray | None:
        if self.program is None or not self.reliable:
            return None
        return self._predict(self.program, observations)

    def loss(self, actor_logits: torch.Tensor, observations) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return already-weighted KL(nn||tree); gradients flow ONLY into logits."""
        if actor_logits.shape[-1] != len(ACTIONS):
            raise ValueError("Native Actor must expose all five actions")
        targets = self.targets(observations) if self.current_lambda > 0 else None
        if targets is None:
            return actor_logits.sum() * 0, {"active": False, "lambda": 0.0, "kl": 0.0,
                                           "complexity_has_actor_gradient": False}
        target = torch.as_tensor(targets, dtype=actor_logits.dtype, device=actor_logits.device).detach()
        if target.shape != actor_logits.shape:
            raise ValueError("Feedback observations and neural logit rows must correspond")
        target = target.clamp_min(1e-8)
        target = target / target.sum(-1, keepdim=True)
        log_probabilities = torch.log_softmax(actor_logits, dim=-1)
        kl = (log_probabilities.exp() * (log_probabilities - target.log())).sum(-1).mean()
        loss = self.current_lambda * kl
        return loss, {"active": True, "lambda": self.current_lambda,
                      "kl": float(kl.detach().cpu()), "loss": float(loss.detach().cpu()),
                      "complexity_has_actor_gradient": False}

    def state_dict(self) -> dict[str, Any]:
        return _json_safe({
            "version": FEEDBACK_VERSION, "feature_names": self.feature_names,
            "config": asdict(self.config), "program": self.program.to_dict() if self.program else None,
            "reliable": self.reliable, "current_lambda": self.current_lambda,
            "last_step": self.last_step, "last_fit_step": self.last_fit_step,
            "ramp_start": self.ramp_start, "performance_anchor_ratio": self.performance_anchor_ratio,
            "reference_score": self.reference_score, "last_fit_report": self.last_fit_report,
            "last_gate": self.last_gate,
        })

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if (payload.get("version") != FEEDBACK_VERSION
                or tuple(payload.get("feature_names", ())) != self.feature_names
                or payload.get("config") != _json_safe(asdict(self.config))):
            raise ValueError("Feedback state has a different protocol or feature contract")
        program = ExecutableProgram.from_dict(payload["program"]) if payload.get("program") else None
        if program is not None:
            if tuple(program.action_names) != ACTIONS or program.metadata.get("action_legality_features"):
                raise ValueError("Native feedback tree cannot contain an action mask")
            stack = [(program.root, 0)]
            leaves = 0
            while stack:
                node, depth = stack.pop()
                if depth > max(self.config.depths):
                    raise ValueError("Restored feedback tree exceeds its depth bound")
                if node.is_leaf:
                    p = np.asarray(node.probabilities)
                    if p.shape != (5,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
                        raise ValueError("Invalid restored neural-target leaf")
                    leaves += 1
                else:
                    if (node.feature not in self.feature_names or node.threshold is None
                            or not np.isfinite(node.threshold) or node.left is None or node.right is None):
                        raise ValueError("Invalid restored native tree predicate")
                    stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
            if leaves > max(self.config.leaves):
                raise ValueError("Restored feedback tree exceeds its leaf bound")
        current_lambda = float(payload["current_lambda"])
        if not np.isfinite(current_lambda) or not 0 <= current_lambda <= self.config.lambda_max:
            raise ValueError("Restored feedback strength is invalid")
        if bool(payload["reliable"]) and program is None:
            raise ValueError("Reliable feedback state requires a program")
        if current_lambda > 0 and (not payload["reliable"] or not payload["last_gate"].get("active")):
            raise ValueError("Active feedback state requires a reliable, enabled gate")
        last_step = payload["last_step"]
        if type(last_step) is not int or last_step < 0:
            raise ValueError("Invalid restored feedback step")
        for name in ("last_fit_step", "ramp_start"):
            value = payload[name]
            if value is not None and (type(value) is not int or not 0 <= value <= last_step):
                raise ValueError("Invalid restored feedback schedule: " + name)
        for name in ("performance_anchor_ratio", "reference_score"):
            value = payload[name]
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Invalid restored feedback reference: " + name)
        if current_lambda > 0 and (payload["ramp_start"] is None or payload["performance_anchor_ratio"] is None
                                   or payload["reference_score"] is None):
            raise ValueError("Active feedback requires its saved ramp and performance reference")
        self.program = program
        for name in ("reliable", "current_lambda", "last_step", "last_fit_step", "ramp_start",
                     "performance_anchor_ratio", "reference_score", "last_fit_report", "last_gate"):
            setattr(self, name, _json_safe(payload[name]))

    def save_program(self, path: str | Path) -> Path:
        if self.program is None:
            raise ValueError("No native RCPD program has been extracted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": FEEDBACK_VERSION, "reliable": self.reliable,
                   "program": self.program.to_dict(), "fit_report": self.last_fit_report}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, allow_nan=False))
        temporary.chmod(0o600)
        temporary.replace(destination)
        return destination
