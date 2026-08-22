"""Executable symbolic-program model and persistence contracts for RCPD."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FeatureVector = Mapping[str, float]


@dataclass(frozen=True)
class ProgramTraceStep:
    feature: str
    operator: str
    threshold: float
    observed_value: float
    result: bool


ProgramTrace = tuple[ProgramTraceStep, ...]


@dataclass(frozen=True)
class ActionExclusion:
    """One action removed by observable execution constraints."""

    action: str
    legality_feature: str
    active_reason_features: tuple[str, ...] = ()
    bound_entities: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramExecutionTrace:
    """Complete trace including the legality layer after the tree leaf."""

    tree_steps: ProgramTrace
    pre_mask_distribution: Mapping[str, float]
    excluded_actions: tuple[ActionExclusion, ...]
    post_mask_distribution: Mapping[str, float]
    selected_action: str
    regularization_version: bool = True
    program_complexity: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgramExecution:
    action: str
    probabilities: Mapping[str, float]
    trace: ProgramExecutionTrace


@dataclass(frozen=True)
class ProgramNode:
    """One node in an executable relational policy program."""

    probabilities: tuple[float, ...] | None = None
    feature: str | None = None
    threshold: float | None = None
    left: "ProgramNode | None" = None
    right: "ProgramNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.probabilities is not None

    def size(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def depth(self) -> int:
        if self.is_leaf:
            # Match the conventional decision-tree definition used by
            # ``max_depth``: a leaf-only tree has depth 0 and the root split is
            # depth 1.  This keeps reported complexity directly comparable to
            # the configured extraction bound.
            return 0
        return 1 + max(self.left.depth() if self.left else 0, self.right.depth() if self.right else 0)

    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        return (self.left.leaf_count() if self.left else 0) + (
            self.right.leaf_count() if self.right else 0
        )

    def used_predicates(self) -> frozenset[str]:
        if self.is_leaf:
            return frozenset()
        own = frozenset((self.feature,)) if self.feature is not None else frozenset()
        return (
            own
            | (self.left.used_predicates() if self.left else frozenset())
            | (self.right.used_predicates() if self.right else frozenset())
        )


@dataclass(frozen=True)
class ExecutableProgram:
    """Relational program with direct execution, traces, and Python export."""

    action_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    root: ProgramNode
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def execute(
        self,
        features: FeatureVector,
        context: Any | None = None,
    ) -> ProgramExecution:
        """Execute the tree and expose every observable post-tree constraint."""

        leaf, tree_steps = self._execute(features)
        pre_mask = {
            action: float(leaf.probabilities[index])
            for index, action in enumerate(self.action_names)
        }
        probabilities = dict(pre_mask)
        legality_features = self.metadata.get("action_legality_features", {})
        reason_features = self.metadata.get(
            "action_constraint_reason_features",
            {},
        )
        context_bindings = getattr(context, "entity_bindings", {})
        if isinstance(context, Mapping):
            context_bindings = context.get("entity_bindings", context_bindings)
        if not isinstance(context_bindings, Mapping):
            context_bindings = {}
        exclusions: list[ActionExclusion] = []
        if isinstance(legality_features, Mapping):
            for action, feature in legality_features.items():
                if action in probabilities and float(features.get(str(feature), 0.0)) <= 0.5:
                    probabilities[action] = 0.0
                    action_reason_mapping = (
                        reason_features.get(action, {})
                        if isinstance(reason_features, Mapping)
                        else {}
                    )
                    active_reason_features = tuple(
                        str(reason_feature)
                        for reason_feature in (
                            action_reason_mapping.values()
                            if isinstance(action_reason_mapping, Mapping)
                            else ()
                        )
                        if float(features.get(str(reason_feature), 0.0)) > 0.5
                    )
                    bound_entities = {
                        str(role): str(entity)
                        for role, entity in context_bindings.items()
                        if str(role).startswith(f"candidate.{action}.")
                    }
                    exclusions.append(
                        ActionExclusion(
                            action=str(action),
                            legality_feature=str(feature),
                            active_reason_features=active_reason_features,
                            bound_entities=bound_entities,
                        )
                    )
            total = sum(probabilities.values())
            if total > 0.0:
                probabilities = {action: value / total for action, value in probabilities.items()}
            else:
                legal_actions = [
                    action
                    for action in self.action_names
                    if float(
                        features.get(
                            str(legality_features.get(action, "")),
                            1.0,
                        )
                    )
                    > 0.5
                ]
                fallback = legal_actions or list(self.action_names)
                probabilities = {
                    action: (1.0 / len(fallback) if action in fallback else 0.0)
                    for action in self.action_names
                }
        selected_action = max(
            self.action_names,
            key=probabilities.__getitem__,
        )
        execution_trace = ProgramExecutionTrace(
            tree_steps=tree_steps,
            pre_mask_distribution=pre_mask,
            excluded_actions=tuple(exclusions),
            post_mask_distribution=probabilities,
            selected_action=selected_action,
            regularization_version=bool(
                self.metadata.get("regularization_version", True)
            ),
            program_complexity={
                "depth": self.root.depth(),
                "leaves": self.root.leaf_count(),
                "predicates": len(self.root.used_predicates()),
            },
        )
        return ProgramExecution(
            action=selected_action,
            probabilities=probabilities,
            trace=execution_trace,
        )

    def predict_proba(self, features: FeatureVector) -> dict[str, float]:
        return dict(self.execute(features).probabilities)

    def predict(self, features: FeatureVector) -> str:
        return self.execute(features).action

    def trace(self, features: FeatureVector) -> tuple[ProgramTraceStep, ...]:
        return self.execute(features).trace.tree_steps

    def complexity(self) -> dict[str, int]:
        """Return stable, publication-ready structural complexity metrics."""

        return {
            "nodes": self.root.size(),
            "depth": self.root.depth(),
            "leaf_nodes": self.root.leaf_count(),
            "predicates": len(self.root.used_predicates()),
            "available_features": len(self.feature_names),
            "actions": len(self.action_names),
        }

    def _execute(self, features: FeatureVector) -> tuple[ProgramNode, tuple[ProgramTraceStep, ...]]:
        node = self.root
        trace: list[ProgramTraceStep] = []
        while not node.is_leaf:
            if node.feature is None or node.threshold is None or node.left is None or node.right is None:
                raise ValueError("Malformed program node.")
            value = float(features.get(node.feature, 0.0))
            result = value <= node.threshold
            trace.append(ProgramTraceStep(node.feature, "<=", node.threshold, value, result))
            node = node.left if result else node.right
        return node, tuple(trace)

    def to_python(self, *, function_name: str = "distilled_policy") -> str:
        """Compile the program to standalone Python using named relational facts."""

        lines = [
            (
                '"""Read-only policy-audit export generated by Relational '
                'Counterfactual Policy Distillation.\n\nThe deployed controller '
                'remains the neural policy.\n"""'
            ),
            "",
            f"ACTIONS = {self.action_names!r}",
            f"ACTION_LEGALITY_FEATURES = {dict(self.metadata.get('action_legality_features', {}))!r}",
            f"ACTION_CONSTRAINT_REASON_FEATURES = {dict(self.metadata.get('action_constraint_reason_features', {}))!r}",
            f"REGULARIZATION_VERSION = {bool(self.metadata.get('regularization_version', True))!r}",
            (
                "PROGRAM_COMPLEXITY = "
                f"{{'depth': {self.root.depth()}, "
                f"'leaves': {self.root.leaf_count()}, "
                f"'predicates': {len(self.root.used_predicates())}}}"
            ),
            "",
            f"def {function_name}(features):",
            (
                "    \"\"\"Return the local audit action and probabilities; "
                "this function does not control the environment.\"\"\""
            ),
        ]

        def emit(node: ProgramNode, indent: int) -> None:
            prefix = " " * indent
            if node.is_leaf:
                probabilities = tuple(round(float(value), 10) for value in node.probabilities)
                lines.append(f"{prefix}probabilities = list({probabilities!r})")
                lines.append(f"{prefix}for action, feature in ACTION_LEGALITY_FEATURES.items():")
                lines.append(f"{prefix}    if action in ACTIONS and float(features.get(feature, 0.0)) <= 0.5:")
                lines.append(f"{prefix}        probabilities[ACTIONS.index(action)] = 0.0")
                lines.append(f"{prefix}total = sum(probabilities)")
                lines.append(f"{prefix}if total > 0.0:")
                lines.append(f"{prefix}    probabilities = [value / total for value in probabilities]")
                lines.append(f"{prefix}else:")
                lines.append(f"{prefix}    legal = [action for action in ACTIONS if float(features.get(ACTION_LEGALITY_FEATURES.get(action, ''), 1.0)) > 0.5] or list(ACTIONS)")
                lines.append(f"{prefix}    probabilities = [1.0 / len(legal) if action in legal else 0.0 for action in ACTIONS]")
                lines.append(f"{prefix}index = max(range(len(ACTIONS)), key=probabilities.__getitem__)")
                lines.append(f"{prefix}return ACTIONS[index], dict(zip(ACTIONS, probabilities))")
                return
            lines.append(f"{prefix}if float(features.get({node.feature!r}, 0.0)) <= {float(node.threshold)!r}:")
            emit(node.left, indent + 4)
            lines.append(f"{prefix}else:")
            emit(node.right, indent + 4)

        emit(self.root, 4)
        encoded_root = self.to_dict()["root"]
        lines.extend(
            (
                "",
                f"PROGRAM_TREE = {encoded_root!r}",
                "",
                "def _execute_tree_with_trace(features):",
                "    node = PROGRAM_TREE",
                "    steps = []",
                "    for _ in range(64):",
                "        if 'probabilities' in node:",
                "            return dict(zip(ACTIONS, [float(value) for value in node['probabilities']])), steps",
                "        feature = node['feature']",
                "        threshold = float(node['threshold'])",
                "        value = float(features.get(feature, 0.0))",
                "        result = value <= threshold",
                "        steps.append({'feature': feature, 'operator': '<=', 'threshold': threshold, 'observed_value': value, 'result': result})",
                "        node = node['left'] if result else node['right']",
                "    raise RuntimeError('Program tree exceeded the bounded execution depth')",
                "",
                "def distilled_policy_with_trace(features):",
                "    pre_mask, tree_steps = _execute_tree_with_trace(features)",
                "    post_mask = dict(pre_mask)",
                "    excluded = []",
                "    for action, legality_feature in ACTION_LEGALITY_FEATURES.items():",
                "        if action in post_mask and float(features.get(legality_feature, 0.0)) <= 0.5:",
                "            post_mask[action] = 0.0",
                "            reason_mapping = ACTION_CONSTRAINT_REASON_FEATURES.get(action, {})",
                "            active_reasons = [feature for feature in reason_mapping.values() if float(features.get(feature, 0.0)) > 0.5]",
                "            excluded.append({'action': action, 'legality_feature': legality_feature, 'active_reason_features': active_reasons})",
                "    total = sum(post_mask.values())",
                "    if total > 0.0:",
                "        post_mask = {action: value / total for action, value in post_mask.items()}",
                "    else:",
                "        legal = [action for action in ACTIONS if float(features.get(ACTION_LEGALITY_FEATURES.get(action, ''), 1.0)) > 0.5] or list(ACTIONS)",
                "        post_mask = {action: (1.0 / len(legal) if action in legal else 0.0) for action in ACTIONS}",
                "    action = max(ACTIONS, key=post_mask.__getitem__)",
                "    trace = {'tree_steps': tree_steps, 'pre_mask_distribution': pre_mask, 'excluded_actions': excluded, 'post_mask_distribution': post_mask, 'selected_action': action, 'regularization_version': REGULARIZATION_VERSION, 'program_complexity': dict(PROGRAM_COMPLEXITY)}",
                "    return action, post_mask, trace",
            )
        )
        serializable_metadata = json.loads(
            json.dumps(dict(self.metadata), ensure_ascii=False, sort_keys=True)
        )
        lines.extend(("", f"PROGRAM_METADATA = {serializable_metadata!r}"))
        return "\n".join(lines) + "\n"

    def export_python(self, path: str | Path, *, function_name: str = "distilled_policy") -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_python(function_name=function_name), encoding="utf-8")
        return target

    def to_dict(self) -> dict[str, Any]:
        """Serialize the executable tree without environment-specific objects."""

        def encode(node: ProgramNode) -> dict[str, Any]:
            if node.is_leaf:
                return {"probabilities": list(node.probabilities or ())}
            if node.feature is None or node.threshold is None or node.left is None or node.right is None:
                raise ValueError("Malformed program node.")
            return {
                "feature": node.feature,
                "threshold": float(node.threshold),
                "left": encode(node.left),
                "right": encode(node.right),
            }

        return {
            "format": "rcpd_executable_program_v2",
            "action_names": list(self.action_names),
            "feature_names": list(self.feature_names),
            "root": encode(self.root),
            "metadata": json.loads(json.dumps(dict(self.metadata), ensure_ascii=False)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutableProgram":
        source_format = str(payload.get("format", ""))
        if source_format not in {
            "rcpd_executable_program_v1",
            "rcpd_executable_program_v2",
        }:
            raise ValueError("Unsupported RCPD executable program format.")

        def decode(node: Mapping[str, Any]) -> ProgramNode:
            if "probabilities" in node:
                probabilities = tuple(float(value) for value in node["probabilities"])
                if not probabilities:
                    raise ValueError("RCPD leaf probabilities cannot be empty.")
                return ProgramNode(probabilities=probabilities)
            required = ("feature", "threshold", "left", "right")
            if any(name not in node for name in required):
                raise ValueError("Malformed serialized RCPD decision node.")
            return ProgramNode(
                feature=str(node["feature"]),
                threshold=float(node["threshold"]),
                left=decode(node["left"]),
                right=decode(node["right"]),
            )

        metadata = dict(payload.get("metadata", {}))
        if source_format == "rcpd_executable_program_v1":
            metadata.setdefault("regularization_version", False)
        program = cls(
            action_names=tuple(str(value) for value in payload["action_names"]),
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            root=decode(payload["root"]),
            metadata=metadata,
        )
        for node in _walk_program_nodes(program.root):
            if node.is_leaf and len(node.probabilities or ()) != len(program.action_names):
                raise ValueError("RCPD leaf width does not match the action schema.")
        return program

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "ExecutableProgram":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# The research text uses "extracted program"; keep the executable
# implementation as the single source of truth while exposing that name.
ExtractedProgram = ExecutableProgram


@dataclass(frozen=True)
class DistillationMetrics:
    """Diagnostics for whether the current Actor is simply extractable.

    These values train and audit the regularity loop.  They are not evidence
    shown to a user and are not the source of natural-language explanations.
    """

    action_fidelity: float
    mean_kl_divergence: float
    action_regret: float | None
    program_size: int
    program_depth: int
    program_leaf_count: int | None = None
    program_predicate_count: int | None = None
    extractability_loss: float | None = None
    extractability_score: float | None = None
    feedback_weight: float = 0.0
    safety_property_violation_rate: float | None = None
    extraction_time_seconds: float | None = None
    sample_count: int = 0
    group_action_fidelity: Mapping[str, float] = field(default_factory=dict)
    group_validation_samples: Mapping[str, int] = field(default_factory=dict)
    interaction_macro_fidelity: float | None = None
    counterfactual_delta_error: float | None = None
    counterfactual_direction_fidelity: float | None = None
    counterfactual_validation_pairs: int = 0
    counterfactual_changed_pairs: int = 0
    counterfactual_dataset_pairs: int = 0
    relational_predicate_count: int = 0
    required_predicate_group_coverage: Mapping[
        str,
        tuple[str, ...],
    ] = field(default_factory=dict)
    semantic_predicate_coverage_complete: bool = True
    interaction_validation_samples: int = 0
    feedback_eligible: bool = True
    feedback_ineligibility_reasons: tuple[str, ...] = ()
    explanation_eligible: bool = True
    explanation_ineligibility_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_fidelity": self.action_fidelity,
            "mean_kl_divergence": self.mean_kl_divergence,
            "action_regret": self.action_regret,
            "program_size": self.program_size,
            "program_depth": self.program_depth,
            "program_leaf_count": self.program_leaf_count,
            "program_predicate_count": self.program_predicate_count,
            "extractability_loss": self.extractability_loss,
            "extractability_score": self.extractability_score,
            "feedback_weight": self.feedback_weight,
            "safety_property_violation_rate": self.safety_property_violation_rate,
            "extraction_time_seconds": self.extraction_time_seconds,
            "sample_count": self.sample_count,
            "group_action_fidelity": dict(self.group_action_fidelity),
            "group_validation_samples": dict(
                self.group_validation_samples
            ),
            "interaction_macro_fidelity": self.interaction_macro_fidelity,
            "counterfactual_delta_error": self.counterfactual_delta_error,
            "counterfactual_direction_fidelity": (
                self.counterfactual_direction_fidelity
            ),
            "counterfactual_validation_pairs": (
                self.counterfactual_validation_pairs
            ),
            "counterfactual_changed_pairs": (
                self.counterfactual_changed_pairs
            ),
            "counterfactual_dataset_pairs": (
                self.counterfactual_dataset_pairs
            ),
            "relational_predicate_count": self.relational_predicate_count,
            "required_predicate_group_coverage": {
                str(group): list(features)
                for group, features in (
                    self.required_predicate_group_coverage.items()
                )
            },
            "semantic_predicate_coverage_complete": (
                self.semantic_predicate_coverage_complete
            ),
            "interaction_validation_samples": (
                self.interaction_validation_samples
            ),
            "feedback_eligible": self.feedback_eligible,
            "feedback_ineligibility_reasons": list(
                self.feedback_ineligibility_reasons
            ),
            "explanation_eligible": self.explanation_eligible,
            "explanation_ineligibility_reasons": list(
                self.explanation_ineligibility_reasons
            ),
        }

def _walk_program_nodes(root: ProgramNode) -> Iterable[ProgramNode]:
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        if node.right is not None:
            pending.append(node.right)
        if node.left is not None:
            pending.append(node.left)
