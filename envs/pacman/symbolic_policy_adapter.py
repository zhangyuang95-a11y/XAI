"""
symbolic_policy.py -- Distill the fixed RL policy into a symbolic decision-tree surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

from core.symbolic_policy import SymbolicAnalysis

try:
    import joblib
except ModuleNotFoundError:  # joblib is optional for plain pickle-compatible artifacts.
    joblib = None

from .agent import RLAgent
from .environment import (
    ACTION_NAMES,
    MazeEnvironment,
    estimate_action_risks,
    get_relative_direction,
    manhattan_distance,
    nearest_monster_distance,
    projected_player_position,
    shielded_action_mask,
    shortest_path_distances,
    target_position_from_state,
)

SYMBOLIC_MAX_DEPTH = 5
SYMBOLIC_MIN_SAMPLES_LEAF = 24
SYMBOLIC_ROLLOUT_EPISODES = 48
SYMBOLIC_HOLDOUT_FRACTION = 0.20


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: tuple[str, ...]
    label_en: str
    label_zh: str


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("action_is_up", ("chosen_action",), "candidate action is UP", "候选动作是上"),
    FeatureSpec("action_is_down", ("chosen_action",), "candidate action is DOWN", "候选动作是下"),
    FeatureSpec("action_is_left", ("chosen_action",), "candidate action is LEFT", "候选动作是左"),
    FeatureSpec("action_is_right", ("chosen_action",), "candidate action is RIGHT", "候选动作是右"),
    FeatureSpec("action_is_stay", ("chosen_action",), "candidate action is STAY", "候选动作是停留"),
    FeatureSpec("immediate_risk", ("collision_risks",), "immediate collision risk", "即时碰撞风险"),
    FeatureSpec("risk_gap_to_best", ("collision_risks",), "risk gap to the safest valid move", "相对最安全动作的风险差"),
    FeatureSpec("risk_close_to_best", ("collision_risks",), "risk is close to the safest valid move", "风险接近最安全动作"),
    FeatureSpec("risk_rank", ("collision_risks",), "risk rank among valid moves", "候选动作的风险排序"),
    FeatureSpec("is_safest_move", ("collision_risks",), "candidate is among the safest valid moves", "候选动作属于最安全动作"),
    FeatureSpec("target_distance_after", ("nearest_dot_distance", "exit_distance"), "distance to the active target after the move", "动作后到当前目标的距离"),
    FeatureSpec("target_distance_delta", ("nearest_dot_distance", "exit_distance"), "how much the move reduces active target distance", "动作让当前目标距离缩短多少"),
    FeatureSpec("aligns_with_target", ("nearest_dot_direction", "exit_direction"), "move points toward the active target", "动作朝向当前目标"),
    FeatureSpec("target_progress_rank", ("nearest_dot_distance", "exit_distance"), "target progress rank among valid moves", "候选动作的目标推进排序"),
    FeatureSpec("best_target_progress", ("nearest_dot_distance", "exit_distance"), "candidate gives the strongest target progress", "候选动作提供最强目标推进"),
    FeatureSpec("immediate_dot", ("nearest_dot_distance",), "move collects a dot immediately", "动作会立刻吃到豆子"),
    FeatureSpec("immediate_exit", ("exit_distance", "exit_open"), "move reaches the exit immediately", "动作会立刻到达出口"),
    FeatureSpec("nearest_monster_after", ("monster_positions",), "nearest monster distance after the move", "动作后最近怪物距离"),
    FeatureSpec("monster_distance_delta", ("monster_positions",), "change in distance to the nearest monster", "动作带来的怪物距离变化"),
    FeatureSpec("moves_toward_nearest_monster", ("monster_positions",), "move points toward the nearest monster", "动作朝向最近怪物"),
    FeatureSpec("monster_clearance_rank", ("monster_positions",), "monster-clearance rank among valid moves", "候选动作的怪物拉开距离排序"),
    FeatureSpec("best_monster_clearance", ("monster_positions",), "candidate gives the best monster clearance", "候选动作提供最佳怪物拉开距离"),
    FeatureSpec("target_is_dot", ("dots_remaining",), "the active target is a dot", "当前目标是豆子"),
    FeatureSpec("dots_remaining_ratio", ("dots_remaining", "total_dots"), "remaining dot ratio", "剩余豆子比例"),
    FeatureSpec("exit_open", ("exit_open",), "exit is already open", "出口已经打开"),
)

FEATURE_NAMES = [spec.name for spec in FEATURE_SPECS]

ACTION_FLAG_FEATURES = {
    "UP": "action_is_up",
    "DOWN": "action_is_down",
    "LEFT": "action_is_left",
    "RIGHT": "action_is_right",
    "STAY": "action_is_stay",
}

ACTION_LABELS = {
    "UP": {"en": "up", "zh": "上"},
    "DOWN": {"en": "down", "zh": "下"},
    "LEFT": {"en": "left", "zh": "左"},
    "RIGHT": {"en": "right", "zh": "右"},
    "STAY": {"en": "stay", "zh": "停留"},
}


def _pick(zh: str, en: str, lang: str) -> str:
    return zh if lang == "zh" else en


def _action_label(action: str, lang: str) -> str:
    return ACTION_LABELS.get(action, {"en": action, "zh": action})[lang]


def _direction_to_actions(direction: str) -> set[str]:
    mapping = {
        "north": {"UP"},
        "south": {"DOWN"},
        "west": {"LEFT"},
        "east": {"RIGHT"},
    }
    if direction == "same":
        return {"STAY"}
    result: set[str] = set()
    for token in direction.split("-"):
        result.update(mapping.get(token, set()))
    return result


def _action_aligns(action: str, direction: str) -> bool:
    desired = _direction_to_actions(direction)
    return action in desired if desired else False


def _valid_actions(state: dict) -> list[str]:
    mask = shielded_action_mask(state)
    actions = [name for name, allowed in zip(ACTION_NAMES, mask) if allowed]
    return actions or ["STAY"]


def _feature_source_names(feature_name: str) -> tuple[str, ...]:
    for spec in FEATURE_SPECS:
        if spec.name == feature_name:
            return spec.source
    return ("state",)


def _dense_rank(value: float, all_values: Iterable[float], *, reverse: bool) -> int:
    ordered = sorted({round(float(item), 6) for item in all_values}, reverse=reverse)
    return ordered.index(round(float(value), 6)) + 1


def _next_state_features(state: dict, action: str) -> tuple[tuple[int, int], dict, int, int]:
    next_pos = projected_player_position(state, action)
    if next_pos is None:
        next_pos = state["player_pos"]

    next_dots = state["dots"]
    collected_dots = int(state.get("collected_dots", 0))
    if next_pos in next_dots:
        next_dots = frozenset(dot for dot in next_dots if dot != next_pos)
        collected_dots += 1

    next_state = dict(state)
    next_state["player_pos"] = next_pos
    next_state["dots"] = next_dots
    next_state["collected_dots"] = collected_dots
    next_state["exit_open"] = not next_dots

    distances_after = shortest_path_distances(next_state["grid"], next_pos)
    target_after = target_position_from_state(next_state, distances=distances_after)
    target_distance_after = int(
        distances_after.get(target_after, next_state["grid_size"] * next_state["grid_size"])
    )
    nearest_monster_after = min(
        (manhattan_distance(next_pos, (row, col)) for _, row, col in state["monsters"]),
        default=999,
    )
    return next_pos, next_state, target_distance_after, nearest_monster_after


def extract_symbolic_feature_values(state: dict, action: str) -> dict[str, float]:
    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown action: {action}")

    player = state["player_pos"]
    dots = state.get("dots", frozenset())
    total_dots = max(1, int(state.get("total_dots", 1)))
    risk_map = estimate_action_risks(state)
    valid_actions = [candidate for candidate in _valid_actions(state) if candidate in risk_map]
    best_risk = min((risk_map[candidate] for candidate in valid_actions), default=1.0)

    distances_before = shortest_path_distances(state["grid"], player)
    target_before = target_position_from_state(state, distances=distances_before)
    target_distance_before = int(
        distances_before.get(target_before, state["grid_size"] * state["grid_size"])
    )
    target_direction = get_relative_direction(player, target_before)
    nearest_monster_before = nearest_monster_distance(state)

    nearest_monster_pos = player
    best_monster_distance = float("inf")
    for _, row, col in state["monsters"]:
        distance = manhattan_distance(player, (row, col))
        if distance < best_monster_distance:
            best_monster_distance = distance
            nearest_monster_pos = (row, col)
    nearest_monster_direction = get_relative_direction(player, nearest_monster_pos)

    candidate_metrics: dict[str, dict[str, float | tuple[int, int]]] = {}
    for candidate in valid_actions:
        next_pos_candidate, _, target_after_candidate, monster_after_candidate = _next_state_features(state, candidate)
        candidate_metrics[candidate] = {
            "next_pos": next_pos_candidate,
            "risk": float(risk_map.get(candidate, 1.0)),
            "target_distance_after": float(target_after_candidate),
            "target_distance_delta": float(target_distance_before - target_after_candidate),
            "monster_distance_delta": float(monster_after_candidate - nearest_monster_before),
            "nearest_monster_after": float(monster_after_candidate),
        }

    action_metrics = candidate_metrics[action]
    next_pos = action_metrics["next_pos"]
    target_distance_after = int(action_metrics["target_distance_after"])
    nearest_monster_after = int(action_metrics["nearest_monster_after"])
    best_target_delta = max(float(metrics["target_distance_delta"]) for metrics in candidate_metrics.values())
    best_monster_delta = max(float(metrics["monster_distance_delta"]) for metrics in candidate_metrics.values())

    values = {name: 0.0 for name in FEATURE_NAMES}
    for action_name, feature_name in ACTION_FLAG_FEATURES.items():
        values[feature_name] = 1.0 if action == action_name else 0.0

    values["immediate_risk"] = float(action_metrics["risk"])
    values["risk_gap_to_best"] = float(values["immediate_risk"] - best_risk)
    values["risk_close_to_best"] = 1.0 if abs(values["risk_gap_to_best"]) <= 0.05 else 0.0
    values["risk_rank"] = float(_dense_rank(values["immediate_risk"], (metrics["risk"] for metrics in candidate_metrics.values()), reverse=False))
    values["is_safest_move"] = 1.0 if values["risk_rank"] == 1.0 else 0.0
    values["target_distance_after"] = float(target_distance_after)
    values["target_distance_delta"] = float(action_metrics["target_distance_delta"])
    values["aligns_with_target"] = 1.0 if _action_aligns(action, target_direction) else 0.0
    values["target_progress_rank"] = float(
        _dense_rank(
            values["target_distance_delta"],
            (metrics["target_distance_delta"] for metrics in candidate_metrics.values()),
            reverse=True,
        )
    )
    values["best_target_progress"] = 1.0 if values["target_distance_delta"] >= best_target_delta else 0.0
    values["immediate_dot"] = 1.0 if next_pos in dots else 0.0
    values["immediate_exit"] = 1.0 if next_pos == state["exit_pos"] and bool(state.get("exit_open", False)) else 0.0
    values["nearest_monster_after"] = float(nearest_monster_after)
    values["monster_distance_delta"] = float(action_metrics["monster_distance_delta"])
    values["moves_toward_nearest_monster"] = 1.0 if _action_aligns(action, nearest_monster_direction) else 0.0
    values["monster_clearance_rank"] = float(
        _dense_rank(
            values["monster_distance_delta"],
            (metrics["monster_distance_delta"] for metrics in candidate_metrics.values()),
            reverse=True,
        )
    )
    values["best_monster_clearance"] = 1.0 if values["monster_distance_delta"] >= best_monster_delta else 0.0
    values["target_is_dot"] = 1.0 if bool(dots) else 0.0
    values["dots_remaining_ratio"] = float(len(dots) / total_dots)
    values["exit_open"] = 1.0 if bool(state.get("exit_open", False)) else 0.0
    return values


def feature_vector_from_values(values: dict[str, float]) -> np.ndarray:
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)


@dataclass
class RolloutState:
    group_id: int
    state: dict
    teacher_action: str
    valid_actions: list[str]
    teacher_scores: dict[str, float]


def describe_predicate(
    feature_name: str,
    operator: str,
    threshold: float,
    value: float,
    lang: str,
) -> str:
    positive = operator == ">" and value > threshold
    if feature_name.startswith("action_is_"):
        action = feature_name.removeprefix("action_is_").upper()
        return _pick(
            f"候选动作是{_action_label(action, lang)}",
            f"the candidate action is {_action_label(action, lang)}",
            lang,
        )
    if feature_name == "target_distance_delta" and positive:
        return _pick("它会缩短当前目标距离", "it reduces the active target distance", lang)
    if feature_name == "target_distance_delta":
        return _pick("它不会缩短当前目标距离", "it does not reduce the active target distance", lang)
    if feature_name == "risk_gap_to_best" and operator == "<=":
        return _pick("它的风险接近最安全动作", "its risk is close to the safest move", lang)
    if feature_name == "risk_gap_to_best":
        return _pick("它明显比最安全动作更冒险", "it is clearly riskier than the safest move", lang)
    if feature_name == "risk_rank":
        return _pick(
            "它的风险排序靠前" if positive else "它的风险排序不靠前",
            "it ranks near the safest moves" if positive else "it does not rank near the safest moves",
            lang,
        )
    if feature_name == "is_safest_move":
        return _pick(
            "它属于最安全动作" if positive else "它不属于最安全动作",
            "it is among the safest moves" if positive else "it is not among the safest moves",
            lang,
        )
    if feature_name == "aligns_with_target":
        return _pick(
            "它朝向当前目标" if positive else "它没有朝向当前目标",
            "it points toward the active target" if positive else "it does not point toward the active target",
            lang,
        )
    if feature_name == "target_progress_rank":
        return _pick(
            "它的目标推进排序靠前" if positive else "它的目标推进排序不靠前",
            "it ranks near the best target-progress moves" if positive else "it does not rank near the best target-progress moves",
            lang,
        )
    if feature_name == "best_target_progress":
        return _pick(
            "它提供最强目标推进" if positive else "它不是最佳目标推进动作",
            "it gives the strongest target progress" if positive else "it is not the best target-progress move",
            lang,
        )
    if feature_name == "immediate_dot":
        return _pick(
            "它会立刻吃到豆子" if positive else "它不会立刻吃到豆子",
            "it collects a dot immediately" if positive else "it does not collect a dot immediately",
            lang,
        )
    if feature_name == "immediate_exit":
        return _pick(
            "它会立刻到达出口" if positive else "它不会立刻到达出口",
            "it reaches the exit immediately" if positive else "it does not reach the exit immediately",
            lang,
        )
    if feature_name == "monster_distance_delta":
        return _pick(
            "它会拉开和最近怪物的距离" if positive else "它不会拉开和最近怪物的距离",
            "it increases distance from the nearest monster" if positive else "it does not increase distance from the nearest monster",
            lang,
        )
    if feature_name == "moves_toward_nearest_monster":
        return _pick(
            "它朝向最近怪物" if positive else "它没有朝向最近怪物",
            "it points toward the nearest monster" if positive else "it does not point toward the nearest monster",
            lang,
        )
    if feature_name == "monster_clearance_rank":
        return _pick(
            "它的怪物拉开距离排序靠前" if positive else "它的怪物拉开距离排序不靠前",
            "it ranks near the best monster-clearance moves" if positive else "it does not rank near the best monster-clearance moves",
            lang,
        )
    if feature_name == "best_monster_clearance":
        return _pick(
            "它提供最佳怪物拉开距离" if positive else "它不是最佳怪物拉开距离动作",
            "it gives the best monster clearance" if positive else "it is not the best monster-clearance move",
            lang,
        )
    label = next(
        (
            spec.label_zh if lang == "zh" else spec.label_en
            for spec in FEATURE_SPECS
            if spec.name == feature_name
        ),
        feature_name,
    )
    return f"{label} {operator} {threshold:.2f} (actual={value:.2f})"


def path_condition_to_python(item: dict) -> str:
    feature = item["feature"]
    operator = item["operator"]
    threshold = float(item["threshold"])
    return f"features['{feature}'] {operator} {threshold:.6f}"


def localize_summary_bullet(text: str, lang: str) -> str:
    if lang == "en":
        return text
    replacements = {
        "When risk stays close to the safest move, prefer actions that reduce target distance.": "当风险接近最安全动作时，优先选择能缩短当前目标距离的动作。",
        "Immediate dot collection is usually preferred when it does not add extra risk.": "如果不会明显增加风险，策略通常优先立即吃到豆子。",
        "When exit is open, the policy shifts toward finishing quickly.": "当出口打开后，策略会转向尽快冲向终点。",
        "Moves that increase clearance from the nearest monster are favored when progress is similar.": "当推进效果相近时，策略更偏好能拉开与最近怪物距离的动作。",
        "Stay is rarely preferred unless movement offers no safe progress.": "除非移动没有安全收益，否则策略很少选择停留。",
    }
    return replacements.get(text, text)


def build_symbolic_comparison(
    chosen_action: str,
    chosen_values: dict[str, float],
    alternative_action: Optional[str],
    alternative_values: Optional[dict[str, float]],
    lang: str,
) -> dict[str, object]:
    if alternative_action is not None and alternative_values is None:
        return {
            "supports_chosen": True,
            "target_clause": _pick(
                f"{_action_label(alternative_action, lang)}在当前状态下不可用，因此它不是可执行替代动作。",
                f"{_action_label(alternative_action, lang)} is unavailable in the current state, so it is not an executable alternative.",
                lang,
            ),
            "risk_clause": "",
            "monster_clause": "",
            "decision_clause": _pick(
                f"因此符号策略继续支持{_action_label(chosen_action, lang)}。",
                f"So the symbolic policy continues to support {_action_label(chosen_action, lang)}.",
                lang,
            ),
            "sources": ["available_actions", "collision_risks"],
        }

    if alternative_action is None or alternative_values is None:
        return {
            "supports_chosen": True,
            "target_clause": _pick("当前没有可比较的替代动作。", "There is no comparable alternative action here.", lang),
            "risk_clause": "",
            "monster_clause": "",
            "decision_clause": _pick(
                f"因此符号策略继续支持{_action_label(chosen_action, lang)}。",
                f"So the symbolic policy continues to support {_action_label(chosen_action, lang)}.",
                lang,
            ),
            "sources": ["collision_risks", "nearest_dot_distance", "exit_distance", "monster_positions"],
        }

    chosen_risk = float(chosen_values["immediate_risk"])
    alternative_risk = float(alternative_values["immediate_risk"])
    chosen_target_delta = float(chosen_values["target_distance_delta"])
    alternative_target_delta = float(alternative_values["target_distance_delta"])
    chosen_monster_delta = float(chosen_values["monster_distance_delta"])
    alternative_monster_delta = float(alternative_values["monster_distance_delta"])

    if abs(chosen_risk - alternative_risk) <= 0.05:
        risk_clause = _pick(
            f"{_action_label(chosen_action, lang)}和{_action_label(alternative_action, lang)}的即时风险接近，约分别为 {chosen_risk:.0%} 和 {alternative_risk:.0%}。",
            f"{_action_label(chosen_action, lang)} and {_action_label(alternative_action, lang)} have comparable immediate risk, about {chosen_risk:.0%} and {alternative_risk:.0%}.",
            lang,
        )
    elif chosen_risk < alternative_risk:
        risk_clause = _pick(
            f"{_action_label(chosen_action, lang)}的即时风险更低，约为 {chosen_risk:.0%}，而{_action_label(alternative_action, lang)}约为 {alternative_risk:.0%}。",
            f"{_action_label(chosen_action, lang)} has the lower immediate risk at about {chosen_risk:.0%}, while {_action_label(alternative_action, lang)} is about {alternative_risk:.0%}.",
            lang,
        )
    else:
        risk_clause = _pick(
            f"{_action_label(chosen_action, lang)}更冒险，约为 {chosen_risk:.0%}，而{_action_label(alternative_action, lang)}约为 {alternative_risk:.0%}。",
            f"{_action_label(chosen_action, lang)} is riskier at about {chosen_risk:.0%}, while {_action_label(alternative_action, lang)} is about {alternative_risk:.0%}.",
            lang,
        )

    if chosen_target_delta > alternative_target_delta:
        target_clause = _pick(
            f"{_action_label(chosen_action, lang)}更能推进当前目标，目标距离缩短 {chosen_target_delta:.0f}，而{_action_label(alternative_action, lang)}只缩短 {alternative_target_delta:.0f}。",
            f"{_action_label(chosen_action, lang)} makes better progress on the active target: it reduces target distance by {chosen_target_delta:.0f}, while {_action_label(alternative_action, lang)} reduces it by {alternative_target_delta:.0f}.",
            lang,
        )
    elif chosen_target_delta < alternative_target_delta:
        target_clause = _pick(
            f"{_action_label(alternative_action, lang)}更能推进当前目标，目标距离缩短 {alternative_target_delta:.0f}，而{_action_label(chosen_action, lang)}只缩短 {chosen_target_delta:.0f}。",
            f"{_action_label(alternative_action, lang)} makes better progress on the active target: it reduces target distance by {alternative_target_delta:.0f}, while {_action_label(chosen_action, lang)} reduces it by {chosen_target_delta:.0f}.",
            lang,
        )
    else:
        target_clause = _pick(
            f"{_action_label(chosen_action, lang)}和{_action_label(alternative_action, lang)}对当前目标的推进接近。",
            f"{_action_label(chosen_action, lang)} and {_action_label(alternative_action, lang)} make similar progress on the active target.",
            lang,
        )

    if chosen_monster_delta > alternative_monster_delta:
        monster_clause = _pick(
            f"{_action_label(chosen_action, lang)}还会更好地拉开和最近怪物的距离。",
            f"{_action_label(chosen_action, lang)} also creates more clearance from the nearest monster.",
            lang,
        )
    elif chosen_monster_delta < alternative_monster_delta:
        monster_clause = _pick(
            f"{_action_label(alternative_action, lang)}在怪物距离上更有利。",
            f"{_action_label(alternative_action, lang)} is better for monster clearance.",
            lang,
        )
    else:
        monster_clause = _pick("两者在怪物压力上的变化接近。", "Both moves change monster pressure similarly.", lang)

    supports_chosen = False
    if chosen_risk <= alternative_risk + 0.05 and chosen_target_delta >= alternative_target_delta:
        supports_chosen = True
    elif abs(chosen_risk - alternative_risk) <= 0.05 and chosen_target_delta > alternative_target_delta:
        supports_chosen = True
    elif chosen_risk < alternative_risk and chosen_target_delta >= alternative_target_delta - 0.5:
        supports_chosen = True
    elif chosen_target_delta > alternative_target_delta and chosen_monster_delta >= alternative_monster_delta:
        supports_chosen = True

    if supports_chosen:
        decision_clause = _pick(
            f"因此，符号策略认为{_action_label(chosen_action, lang)}相对{_action_label(alternative_action, lang)}更合适。",
            f"So the symbolic policy judges {_action_label(chosen_action, lang)} to be the better choice than {_action_label(alternative_action, lang)} here.",
            lang,
        )
    else:
        decision_clause = _pick(
            f"因此，符号策略并没有明确支持{_action_label(chosen_action, lang)}优于{_action_label(alternative_action, lang)}。",
            f"So the symbolic policy does not clearly support {_action_label(chosen_action, lang)} over {_action_label(alternative_action, lang)} here.",
            lang,
        )

    return {
        "supports_chosen": supports_chosen,
        "risk_clause": risk_clause,
        "target_clause": target_clause,
        "monster_clause": monster_clause,
        "decision_clause": decision_clause,
        "sources": ["collision_risks", "nearest_dot_distance", "exit_distance", "monster_positions"],
    }


class DistilledSymbolicPolicy:
    def __init__(
        self,
        classifier: DecisionTreeClassifier,
        metrics: dict[str, float],
        metadata: dict[str, object],
        summary_bullets: list[str],
        policy_code: str,
        leaf_support: dict[int, int],
    ):
        self.classifier = classifier
        self.metrics = metrics
        self.metadata = metadata
        self.summary_bullets = summary_bullets
        self.policy_code = policy_code
        self.leaf_support = leaf_support

    def to_payload(self) -> dict[str, object]:
        return {
            "classifier": self.classifier,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "summary_bullets": self.summary_bullets,
            "policy_code": self.policy_code,
            "leaf_support": self.leaf_support,
            "feature_names": FEATURE_NAMES,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "DistilledSymbolicPolicy":
        feature_names = payload.get("feature_names", FEATURE_NAMES)
        if list(feature_names) != FEATURE_NAMES:
            raise ValueError("Symbolic policy feature set is incompatible with this runtime")
        return cls(
            classifier=payload["classifier"],
            metrics=dict(payload.get("metrics", {})),
            metadata=dict(payload.get("metadata", {})),
            summary_bullets=list(payload.get("summary_bullets", [])),
            policy_code=str(payload.get("policy_code", "")),
            leaf_support={int(key): int(value) for key, value in dict(payload.get("leaf_support", {})).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if joblib is not None:
            joblib.dump(self.to_payload(), path)
            return
        with path.open("wb") as handle:
            pickle.dump(self.to_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def validate_compatibility(self, checkpoint_metadata: dict[str, object]) -> None:
        for key in ("grid_size", "num_monsters", "reward_preset", "max_steps"):
            symbolic_value = self.metadata.get(key)
            checkpoint_value = checkpoint_metadata.get(key)
            if symbolic_value is None or checkpoint_value is None:
                continue
            if str(symbolic_value) != str(checkpoint_value):
                raise ValueError(
                    f"Symbolic policy mismatch for {key}: symbolic={symbolic_value} checkpoint={checkpoint_value}"
                )

    def _positive_probability(self, sample: np.ndarray) -> float:
        probabilities = self.classifier.predict_proba(sample.reshape(1, -1))[0]
        positive_index = list(self.classifier.classes_).index(1)
        return float(probabilities[positive_index])

    def score_actions(self, state: dict) -> dict[str, dict[str, object]]:
        scored: dict[str, dict[str, object]] = {}
        for action in _valid_actions(state):
            values = extract_symbolic_feature_values(state, action)
            vector = feature_vector_from_values(values)
            probability = self._positive_probability(vector)
            leaf_id = int(self.classifier.apply(vector.reshape(1, -1))[0])
            scored[action] = {
                "probability": probability,
                "values": values,
                "vector": vector,
                "leaf_id": leaf_id,
                "support": self.leaf_support.get(leaf_id, 0),
            }
        return scored

    def predict_action(self, state: dict) -> tuple[str, dict[str, dict[str, object]]]:
        scored = self.score_actions(state)
        ranked = sorted(scored.items(), key=lambda item: (-float(item[1]["probability"]), item[0]))
        return ranked[0][0], scored

    def _path_conditions(self, vector: np.ndarray, lang: str) -> list[dict]:
        tree = self.classifier.tree_
        node_indicator = self.classifier.decision_path(vector.reshape(1, -1))
        leaf_id = int(self.classifier.apply(vector.reshape(1, -1))[0])
        nodes = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

        trace: list[dict] = []
        for node_id in nodes:
            if node_id == leaf_id:
                continue
            feature_index = int(tree.feature[node_id])
            if feature_index == _tree.TREE_UNDEFINED:
                continue
            feature_name = FEATURE_NAMES[feature_index]
            threshold = float(tree.threshold[node_id])
            value = float(vector[feature_index])
            operator = "<=" if value <= threshold else ">"
            trace.append(
                {
                    "feature": feature_name,
                    "condition": f"{feature_name} {operator} {threshold:.2f}",
                    "value": value,
                    "threshold": threshold,
                    "operator": operator,
                    "source": list(_feature_source_names(feature_name)),
                    "description": describe_predicate(feature_name, operator, threshold, value, lang),
                }
            )
        return trace

    def render_rule_for_action(self, action: str, values: dict[str, float], lang: str) -> tuple[str, str]:
        vector = feature_vector_from_values(values)
        trace = self._path_conditions(vector, lang)
        predicate_text = [item["description"] for item in trace if item["description"]]
        probability = self._positive_probability(vector)
        if predicate_text:
            text = _pick(
                f"如果候选动作是{_action_label(action, lang)}，并且 " + "，且 ".join(predicate_text) + f"，那么这条规则会给它较高偏好（score={probability:.2f}）。",
                f"If the candidate action is {_action_label(action, lang)} and " + " and ".join(predicate_text) + f", then this rule gives it a high preference score ({probability:.2f}).",
                lang,
            )
        else:
            text = _pick(
                f"符号策略对{_action_label(action, lang)}给出了 score={probability:.2f}。",
                f"The symbolic policy assigns {_action_label(action, lang)} a score of {probability:.2f}.",
                lang,
            )

        python_lines = [
            "if " + " and ".join(path_condition_to_python(item) for item in trace) + ":" if trace else "if True:",
            f"    return {probability:.6f}",
        ]
        return text, "\n".join(python_lines)

    def get_policy_summary(self, lang: str) -> dict[str, object]:
        lines = self.policy_code.splitlines()
        return {
            "bullets": [localize_summary_bullet(text, lang) for text in self.summary_bullets[:5]],
            "python_snippet": "\n".join(lines[: min(18, len(lines))]),
        }

    def analyze_state(
        self,
        state: dict,
        chosen_action: str,
        lang: str,
        requested_alternative: Optional[str] = None,
    ) -> SymbolicAnalysis:
        predicted_action, scored = self.predict_action(state)
        chosen_payload = scored[chosen_action]

        alternatives = [(action, payload) for action, payload in scored.items() if action != chosen_action]
        alternative_action: str | None = None
        alternative_payload: dict[str, object] | None = None
        if requested_alternative in scored and requested_alternative != chosen_action:
            alternative_action = requested_alternative
            alternative_payload = scored[requested_alternative]
        elif requested_alternative and requested_alternative != chosen_action:
            alternative_action = requested_alternative
            alternative_payload = None
        elif alternatives:
            alternative_action, alternative_payload = sorted(
                alternatives,
                key=lambda item: (-float(item[1]["probability"]), item[0]),
            )[0]

        chosen_trace = self._path_conditions(feature_vector_from_values(chosen_payload["values"]), lang)
        predicted_trace = self._path_conditions(feature_vector_from_values(scored[predicted_action]["values"]), lang)
        rule_text, rule_python = self.render_rule_for_action(chosen_action, chosen_payload["values"], lang)
        comparison = build_symbolic_comparison(
            chosen_action=chosen_action,
            chosen_values=chosen_payload["values"],
            alternative_action=alternative_action,
            alternative_values=alternative_payload["values"] if alternative_payload else None,
            lang=lang,
        )
        return SymbolicAnalysis(
            chosen_action=chosen_action,
            predicted_action=predicted_action,
            alternative_action=alternative_action,
            symbolic_match=predicted_action == chosen_action,
            symbolic_support=bool(comparison["supports_chosen"]),
            chosen_score=float(chosen_payload["probability"]),
            alternative_score=float(alternative_payload["probability"]) if alternative_payload else None,
            scores={action: float(payload["probability"]) for action, payload in scored.items()},
            chosen_trace=chosen_trace,
            predicted_trace=predicted_trace,
            chosen_rule=rule_text,
            chosen_rule_python=rule_python,
            comparison=comparison,
        )


def _leaf_support_counts(classifier: DecisionTreeClassifier, features: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for leaf_id in classifier.apply(features):
        counts[int(leaf_id)] = counts.get(int(leaf_id), 0) + 1
    return counts


def _leaf_path(classifier: DecisionTreeClassifier, leaf_id: int) -> list[tuple[str, str, float]]:
    tree = classifier.tree_

    def recurse(node_id: int, current: list[tuple[str, str, float]]) -> Optional[list[tuple[str, str, float]]]:
        if node_id == leaf_id:
            return list(current)
        feature_index = int(tree.feature[node_id])
        if feature_index == _tree.TREE_UNDEFINED:
            return None
        threshold = float(tree.threshold[node_id])
        left = recurse(int(tree.children_left[node_id]), current + [(FEATURE_NAMES[feature_index], "<=", threshold)])
        if left is not None:
            return left
        return recurse(int(tree.children_right[node_id]), current + [(FEATURE_NAMES[feature_index], ">", threshold)])

    return recurse(0, []) or []


def _positive_leaf_probability(classifier: DecisionTreeClassifier, leaf_id: int) -> float:
    tree = classifier.tree_
    counts = tree.value[leaf_id][0]
    positive_index = list(classifier.classes_).index(1)
    total = float(np.sum(counts))
    return float(counts[positive_index] / max(1.0, total))


def build_summary_bullets(classifier: DecisionTreeClassifier, leaf_support: dict[int, int]) -> list[str]:
    bullets: list[str] = [
        "When risk stays close to the safest move, prefer actions that reduce target distance.",
        "Immediate dot collection is usually preferred when it does not add extra risk.",
        "When exit is open, the policy shifts toward finishing quickly.",
        "Moves that increase clearance from the nearest monster are favored when progress is similar.",
        "Stay is rarely preferred unless movement offers no safe progress.",
    ]
    positive_leaves = [
        leaf_id
        for leaf_id in leaf_support
        if _positive_leaf_probability(classifier, leaf_id) >= 0.5
    ]
    for leaf_id in sorted(positive_leaves, key=lambda item: (-leaf_support.get(item, 0), item))[:3]:
        conditions = _leaf_path(classifier, leaf_id)
        if not conditions:
            continue
        bullets.append(
            "High-coverage rule: " + "; ".join(
                f"{feature} {operator} {threshold:.2f}" for feature, operator, threshold in conditions[:3]
            )
        )
    deduped: list[str] = []
    for bullet in bullets:
        if bullet not in deduped:
            deduped.append(bullet)
    return deduped[:5]


def render_tree_as_python(classifier: DecisionTreeClassifier) -> str:
    positive_index = list(classifier.classes_).index(1)
    tree = classifier.tree_
    lines = ["def symbolic_action_score(features):"]

    def recurse(node_id: int, indent: str) -> None:
        feature_index = int(tree.feature[node_id])
        if feature_index == _tree.TREE_UNDEFINED:
            counts = tree.value[node_id][0]
            probability = float(counts[positive_index] / max(1.0, float(np.sum(counts))))
            lines.append(f"{indent}return {probability:.6f}")
            return
        feature_name = FEATURE_NAMES[feature_index]
        threshold = float(tree.threshold[node_id])
        lines.append(f"{indent}if features['{feature_name}'] <= {threshold:.6f}:")
        recurse(int(tree.children_left[node_id]), indent + "    ")
        lines.append(f"{indent}else:")
        recurse(int(tree.children_right[node_id]), indent + "    ")

    recurse(0, "    ")
    lines.extend(
        [
            "",
            "def choose_action(candidate_features):",
            "    best_action = None",
            "    best_score = float('-inf')",
            "    for action, features in candidate_features.items():",
            "        score = symbolic_action_score(features)",
            "        if score > best_score:",
            "            best_action = action",
            "            best_score = score",
            "    return best_action",
        ]
    )
    return "\n".join(lines)


def _state_level_predictions(
    policy: DistilledSymbolicPolicy,
    states: Iterable[RolloutState],
) -> tuple[float, float]:
    total = 0
    correct_actions = 0
    pairwise_total = 0
    pairwise_correct = 0
    for item in states:
        predicted_action, scored = policy.predict_action(item.state)
        total += 1
        correct_actions += int(predicted_action == item.teacher_action)
        teacher_score = float(scored[item.teacher_action]["probability"])
        for action, payload in scored.items():
            if action == item.teacher_action:
                continue
            pairwise_total += 1
            pairwise_correct += int(teacher_score >= float(payload["probability"]))
    return correct_actions / max(1, total), pairwise_correct / max(1, pairwise_total)


def collect_symbolic_rollouts(
    teacher: RLAgent,
    *,
    episodes: int = SYMBOLIC_ROLLOUT_EPISODES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RolloutState]]:
    grid_size = int(teacher.metadata.get("grid_size", 11))
    num_monsters = int(teacher.metadata.get("num_monsters", 2))
    max_steps = int(teacher.metadata.get("max_steps", grid_size * grid_size * 2))
    reward_preset = str(teacher.metadata.get("reward_preset", "stable"))
    env = MazeEnvironment(
        grid_size=grid_size,
        num_monsters=num_monsters,
        seed=seed + 120_000,
        max_steps=max_steps,
        reward_preset=reward_preset,
    )

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    sample_weights: list[float] = []
    states: list[RolloutState] = []
    group_id = 0

    for episode in range(episodes):
        env.reset(seed=seed + 120_000 + episode)
        state = env.get_state()
        done = False
        while not done:
            teacher_action = teacher.choose_action(state)
            valid_actions = _valid_actions(state)
            teacher_scores = teacher.score_action_values(state)
            shifted = np.array(
                [teacher_scores.get(action, float("-inf")) for action in valid_actions],
                dtype=np.float64,
            )
            shifted = shifted - float(np.max(shifted))
            teacher_probs = np.exp(shifted)
            teacher_probs = teacher_probs / max(1e-9, float(np.sum(teacher_probs)))
            probability_by_action = {
                action: float(prob)
                for action, prob in zip(valid_actions, teacher_probs)
            }
            best_prob = max(probability_by_action.values(), default=1.0)
            second_prob = sorted(probability_by_action.values(), reverse=True)[1] if len(probability_by_action) > 1 else 0.0
            clarity = best_prob - second_prob
            state_weight = 0.5 + 3.0 * clarity
            states.append(
                RolloutState(
                    group_id=group_id,
                    state=dict(state),
                    teacher_action=teacher_action,
                    valid_actions=list(valid_actions),
                    teacher_scores={action: float(teacher_scores[action]) for action in valid_actions},
                )
            )
            for action in valid_actions:
                values = extract_symbolic_feature_values(state, action)
                feature_rows.append(feature_vector_from_values(values))
                labels.append(1 if action == teacher_action else 0)
                preference_gap = max(0.0, probability_by_action.get(teacher_action, best_prob) - probability_by_action.get(action, 0.0))
                row_weight = state_weight * (1.0 + preference_gap)
                sample_weights.append(row_weight)

            _, _, done, info = env.step_rl(teacher_action)
            state = info["state"]
            group_id += 1

    return (
        np.vstack(feature_rows).astype(np.float32, copy=False),
        np.array(labels, dtype=np.int64),
        np.array(sample_weights, dtype=np.float32),
        states,
    )


def distill_symbolic_policy(
    checkpoint_path: Path,
    *,
    output_path: Path,
    policy_code_path: Path,
    summary_path: Path,
    rollout_episodes: int = SYMBOLIC_ROLLOUT_EPISODES,
    seed: int = 42,
) -> DistilledSymbolicPolicy:
    teacher = RLAgent(model_path=checkpoint_path)
    features, targets, sample_weights, states = collect_symbolic_rollouts(
        teacher,
        episodes=rollout_episodes,
        seed=seed,
    )

    state_indices = np.arange(len(states))
    rng = np.random.default_rng(seed)
    rng.shuffle(state_indices)
    holdout_size = max(1, int(round(len(states) * SYMBOLIC_HOLDOUT_FRACTION)))
    holdout_ids = set(int(idx) for idx in state_indices[:holdout_size])

    rows_per_state = [len(item.valid_actions) for item in states]
    row_offsets = [0]
    for count in rows_per_state:
        row_offsets.append(row_offsets[-1] + count)

    train_rows: list[int] = []
    holdout_rows: list[int] = []
    train_states: list[RolloutState] = []
    holdout_states: list[RolloutState] = []
    for state_idx, item in enumerate(states):
        rows = list(range(row_offsets[state_idx], row_offsets[state_idx + 1]))
        if state_idx in holdout_ids:
            holdout_rows.extend(rows)
            holdout_states.append(item)
        else:
            train_rows.extend(rows)
            train_states.append(item)

    classifier = DecisionTreeClassifier(
        max_depth=SYMBOLIC_MAX_DEPTH,
        min_samples_leaf=SYMBOLIC_MIN_SAMPLES_LEAF,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(features[train_rows], targets[train_rows], sample_weight=sample_weights[train_rows])

    leaf_support = _leaf_support_counts(classifier, features[train_rows])
    policy_code = render_tree_as_python(classifier)
    summary_bullets = build_summary_bullets(classifier, leaf_support)

    policy = DistilledSymbolicPolicy(
        classifier=classifier,
        metrics={},
        metadata={
            "teacher_checkpoint": str(checkpoint_path),
            "grid_size": teacher.metadata.get("grid_size", 11),
            "num_monsters": teacher.metadata.get("num_monsters", 2),
            "reward_preset": teacher.metadata.get("reward_preset", "stable"),
            "max_steps": teacher.metadata.get("max_steps"),
            "teacher_model_version": teacher.metadata.get("model_version", 0),
            "rollout_episodes": rollout_episodes,
            "max_depth": SYMBOLIC_MAX_DEPTH,
            "min_samples_leaf": SYMBOLIC_MIN_SAMPLES_LEAF,
        },
        summary_bullets=summary_bullets,
        policy_code=policy_code,
        leaf_support=leaf_support,
    )

    holdout_action_agreement, holdout_pairwise_agreement = _state_level_predictions(policy, holdout_states)
    train_action_agreement, _ = _state_level_predictions(policy, train_states)

    total_supported = sum(leaf_support.values())
    weighted_path_length = 0.0
    for leaf_id, support in leaf_support.items():
        weighted_path_length += len(_leaf_path(classifier, leaf_id)) * support
    avg_rule_length = weighted_path_length / max(1, total_supported)
    top_leaf_coverage = sum(sorted(leaf_support.values(), reverse=True)[:3]) / max(1, total_supported)

    policy.metrics = {
        "train_action_agreement": round(train_action_agreement, 4),
        "holdout_action_agreement": round(holdout_action_agreement, 4),
        "holdout_pairwise_agreement": round(holdout_pairwise_agreement, 4),
        "top_leaf_coverage": round(top_leaf_coverage, 4),
        "average_rule_length": round(avg_rule_length, 4),
        "rollout_states": len(states),
        "rollout_rows": int(features.shape[0]),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_code_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    policy.save(output_path)
    policy_code_path.write_text(policy.policy_code, encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "metrics": policy.metrics,
                "metadata": policy.metadata,
                "bullets": policy.summary_bullets,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return policy


def load_symbolic_policy(path: Path) -> DistilledSymbolicPolicy:
    if joblib is not None:
        payload = joblib.load(path)
    else:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    return DistilledSymbolicPolicy.from_payload(payload)
