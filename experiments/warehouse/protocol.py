"""Immutable Warehouse v6 seed blocks, acceptance gates, and one-use claims.

This module contains no policy or environment action logic.  A program can
provide training feedback and audited evidence, never an executed action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence


INITIAL_RELEASE_ID = "policylens-three-domain-20261016.v6.0"
BUDGETS = (200_000, 300_000, 400_000)
DEVELOPMENT_PAIR_SEEDS = (6120, 6121, 6122)
FORMAL_PAIR_SEEDS = (6101, 6102, 6103)
DEVELOPMENT_SELECTION_SEEDS = tuple(range(261_100, 261_124))
DEVELOPMENT_STRESS_BASE_SEEDS = tuple(range(261_200, 261_456))
DEVELOPMENT_PROGRAM_AUDIT_SEEDS = tuple(range(261_500, 261_524))
DEVELOPMENT_ACTION_STREAMS = (0, 1, 2)
FORMAL_SELECTION_SEEDS = tuple(range(240_048, 240_072))
FORMAL_PROGRAM_PREAUDIT_SEEDS = tuple(range(240_072, 240_096))
FINAL_PROGRAM_AUDIT_SEEDS = tuple(range(240_100, 240_124))
HELDOUT_BASE_SEEDS = tuple(range(250_000, 250_100))
_RELEASE_PATTERN = re.compile(
    r"policylens-three-domain-20261016\.v6\.(0|[1-9][0-9]*)\Z")
TASKS = (1, 2, 3)
ROLES = ("robot_1", "robot_2")
VARIANTS = ("raw", "regularized")
PROGRAM_LAMBDA = 0.01
PROGRAM_START_FRACTION = 0.80
PROGRAM_RAMP_FRACTION = 0.05


def _hash_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _require_disjoint() -> None:
    blocks = {
        "development_selection": set(DEVELOPMENT_SELECTION_SEEDS),
        "stress": set(DEVELOPMENT_STRESS_BASE_SEEDS),
        "development_program_audit": set(DEVELOPMENT_PROGRAM_AUDIT_SEEDS),
        "selection": set(FORMAL_SELECTION_SEEDS),
        "preaudit": set(FORMAL_PROGRAM_PREAUDIT_SEEDS),
        "audit": set(FINAL_PROGRAM_AUDIT_SEEDS),
        "heldout": set(HELDOUT_BASE_SEEDS),
    }
    for first, a in blocks.items():
        for second, b in blocks.items():
            if first < second and a & b:
                raise RuntimeError(f"Warehouse v6 seed blocks overlap: {first}, {second}")
    if set(DEVELOPMENT_PAIR_SEEDS) & set(FORMAL_PAIR_SEEDS):
        raise RuntimeError("Warehouse v6 development and formal pair seeds overlap")


_require_disjoint()


def task_scenario_seed(base_seed: int, task: int) -> int:
    if task not in TASKS:
        raise ValueError("Warehouse task must be 1, 2, or 3")
    return int(base_seed) * 10 + task


def stress_action_seed(base_seed: int, task: int, stream: int) -> int:
    """Independent, reproducible action RNG for the same public scenario."""
    if base_seed not in DEVELOPMENT_STRESS_BASE_SEEDS or stream not in DEVELOPMENT_ACTION_STREAMS:
        raise ValueError("unregistered Warehouse v6 stress scenario or stream")
    task_scenario_seed(base_seed, task)
    data = f"warehouse-v6-stress|{base_seed}|{task}|{stream}".encode()
    return int.from_bytes(sha256(data).digest()[:8], "big") & 0x7fff_ffff


def stress_schedule() -> tuple[tuple[int, int, int], ...]:
    """(task, environment seed, action RNG seed), 2304 disjoint trajectories."""
    return tuple((task, task_scenario_seed(seed, task),
                  stress_action_seed(seed, task, stream))
                 for task in TASKS for seed in DEVELOPMENT_STRESS_BASE_SEEDS
                 for stream in DEVELOPMENT_ACTION_STREAMS)


def sealed_seed_blocks(release_id: str = INITIAL_RELEASE_ID) -> dict[str, tuple[int, ...]]:
    """Reserve fresh final-audit and heldout blocks for every v6 candidate.

    The original v6.0 blocks are fixed. A failed candidate can only advance to
    a new release ID, whose once-only seed blocks cannot alias earlier ones.
    """
    match = _RELEASE_PATTERN.fullmatch(release_id)
    if match is None:
        raise ValueError("unregistered Warehouse v6 release ID; canonical minor required")
    offset = 100_000 * int(match.group(1))
    return {
        "final_program_audit": tuple(seed + offset for seed in FINAL_PROGRAM_AUDIT_SEEDS),
        "heldout": tuple(seed + offset for seed in HELDOUT_BASE_SEEDS),
    }


def final_audit_seeds(release_id: str = INITIAL_RELEASE_ID) -> tuple[int, ...]:
    return sealed_seed_blocks(release_id)["final_program_audit"]


def heldout_base_seeds(release_id: str = INITIAL_RELEASE_ID) -> tuple[int, ...]:
    return sealed_seed_blocks(release_id)["heldout"]


def heldout_schedule(release_id: str = INITIAL_RELEASE_ID) -> dict[int, tuple[int, ...]]:
    base_seeds = sealed_seed_blocks(release_id)["heldout"]
    return {task: tuple(task_scenario_seed(seed, task)
                        for seed in base_seeds) for task in TASKS}


@dataclass(frozen=True)
class PairRunSpec:
    candidate_id: str
    pair_seed: int
    joint_steps: int
    output_dir: str
    scenario_namespace: str = "warehouse-v6-matched-120-step-blocks"
    checkpoint_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or "/" in self.candidate_id or ".." in self.candidate_id:
            raise ValueError("candidate_id must be a simple, nonempty identifier")
        if self.pair_seed not in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
            raise ValueError("unregistered Warehouse v6 pair seed")
        if self.joint_steps not in BUDGETS:
            raise ValueError("Warehouse v6 training budget must be preregistered")
        if not self.output_dir:
            raise ValueError("output_dir required")
        required = (DEVELOPMENT_SELECTION_SEEDS if
                    self.pair_seed in DEVELOPMENT_PAIR_SEEDS else
                    FORMAL_SELECTION_SEEDS)
        if self.checkpoint_seeds != required:
            raise ValueError("Warehouse v6 checkpoint seeds do not match phase")

    def to_mapping(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        # A moved artifact directory does not change the scientific run.
        return _hash_json({key: value for key, value in self.to_mapping().items()
                           if key != "output_dir"})


def quality_pass(summary: Mapping, *, development: bool) -> bool:
    """Development has safety margin; release quality limits are unchanged."""
    limits = (5.5, 6.0, 0.05) if development else (5.0, 8.0, 0.10)
    try:
        deliveries = float(summary["mean_deliveries"])
        collisions = float(summary["mean_collisions"])
        shutdown = float(summary["shutdown_episode_rate"])
    except (KeyError, TypeError, ValueError):
        return False
    return (deliveries >= limits[0] and 0 <= collisions <= limits[1]
            and 0 <= shutdown <= limits[2]
            and all(map(_finite, (deliveries, collisions, shutdown))))


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def program_pass(metrics: Mapping, *, purpose: str, development: bool) -> bool:
    if purpose not in ("feedback", "explanation"):
        raise ValueError("unknown Warehouse program audit purpose")
    minimum_coverage = 0.30 if development else 0.20
    minimum_agreement = ((0.93 if purpose == "feedback" else 0.95)
                         if development else
                         (0.85 if purpose == "feedback" else 0.90))
    maximum_kl = 0.25 if development else 0.35
    try:
        coverage = float(metrics["coverage"])
        agreement = float(metrics["agreement"])
        kl = float(metrics["kl"])
        rows = int(metrics["rows"])
        illegal = int(metrics["illegal_count"])
    except (KeyError, TypeError, ValueError):
        return False
    return (rows > 0 and illegal == 0 and
            all(map(_finite, (coverage, agreement, kl))) and
            minimum_coverage <= coverage <= 1 and
            minimum_agreement <= agreement <= 1 and 0 <= kl <= maximum_kl)


def paired_training_pass(variants: Mapping, *, joint_steps: int) -> bool:
    """Check identical starts/schedules and actual gradients for both roles."""
    if joint_steps not in BUDGETS or set(variants) != set(VARIANTS):
        return False
    raw, regularized = (variants[name] for name in VARIANTS)
    if not isinstance(raw, Mapping) or not isinstance(regularized, Mapping):
        return False
    for report in (raw, regularized):
        if (report.get("joint_steps") != joint_steps or
                report.get("actor_decisions") != joint_steps * len(ROLES) or
                report.get("post_policy_interventions") != 0 or
                report.get("physical_no_legal_action_rows") != 0):
            return False
        initial = report.get("initial_role_sha256")
        selected = report.get("selected_role_sha256")
        changes = report.get("role_weight_change_norm")
        if (not isinstance(initial, Mapping) or not isinstance(selected, Mapping)
                or not isinstance(changes, Mapping)
                or set(initial) != set(ROLES) or set(selected) != set(ROLES)
                or set(changes) != set(ROLES) or
                selected[ROLES[0]] == selected[ROLES[1]]):
            return False
        for role in ROLES:
            try:
                change = float(changes[role])
            except (TypeError, ValueError):
                return False
            if (not isinstance(initial[role], str) or
                    not isinstance(selected[role], str) or
                    initial[role] == selected[role] or
                    not _finite(change) or change <= 0):
                return False
    if (raw.get("initial_role_sha256") != regularized.get("initial_role_sha256")
            or not raw.get("scenario_schedule_sha256")
            or raw.get("scenario_schedule_sha256") !=
            regularized.get("scenario_schedule_sha256")):
        return False
    if any(raw["selected_role_sha256"][role] ==
           regularized["selected_role_sha256"][role] for role in ROLES):
        return False
    for report, should_be_positive in ((raw, False), (regularized, True)):
        rows = report.get("feedback_rows")
        gradients = report.get("feedback_gradient_norm")
        if (not isinstance(rows, Mapping) or not isinstance(gradients, Mapping)
                or set(rows) != set(ROLES) or set(gradients) != set(ROLES)):
            return False
        for role in ROLES:
            try:
                count, norm = int(rows[role]), float(gradients[role])
            except (TypeError, ValueError):
                return False
            if (not _finite(norm) or
                    (should_be_positive and (count <= 0 or norm <= 0)) or
                    (not should_be_positive and (count != 0 or norm != 0))):
                return False
    return True


def pair_quality_score(variants: Mapping) -> float:
    """Predeclared maximin selection uses quality alone, never program audit."""
    scores = []
    for variant in VARIANTS:
        summary = variants[variant]["quality"]
        if not quality_pass(summary, development=False):
            raise ValueError("quality selection requires both variants to pass")
        scores.append(float(summary["mean_deliveries"])
                      - 0.25 * float(summary["mean_collisions"])
                      - 5.0 * float(summary["shutdown_episode_rate"]))
    return min(scores)


def development_pair_pass(report: Mapping) -> bool:
    """Both variants and both roles need margin on independent stress scenes."""
    try:
        pair_seed = report["pair_seed"]
        budget = report["budget"]
        training = report["training"]
        evaluation = report["evaluation"]
        audits = report["program_audit"]
        regularized_hashes = training["regularized"]["selected_role_sha256"]
    except (KeyError, TypeError):
        return False
    if (pair_seed not in DEVELOPMENT_PAIR_SEEDS or budget not in BUDGETS or
            report.get("stress_episodes") != len(stress_schedule()) or
            report.get("stress_schedule_sha256") != _hash_json(stress_schedule()) or
            not paired_training_pass(training, joint_steps=budget) or
            set(evaluation) != set(VARIANTS) or set(audits) != set(ROLES)):
        return False
    if not all(quality_pass(evaluation[variant], development=True)
               for variant in VARIANTS):
        return False
    for role in ROLES:
        audit = audits[role]
        if (audit.get("actor_role_sha256") != regularized_hashes[role]
                or audit.get("sample_count", 0) < 1_000
                or not audit.get("program_sha256")):
            return False
        if not all(program_pass(audit.get(purpose, {}),
                                purpose=purpose, development=True)
                   for purpose in ("feedback", "explanation")):
            return False
    return True


def choose_development_budget(reports_by_budget: Mapping[int, Sequence[Mapping]]) -> int | None:
    """Lock the smallest passing budget before any formal paired run starts."""
    for budget in BUDGETS:
        reports = reports_by_budget.get(budget, ())
        if (len(reports) == len(DEVELOPMENT_PAIR_SEEDS)
                and {row.get("pair_seed") for row in reports} ==
                set(DEVELOPMENT_PAIR_SEEDS)
                and all(row.get("budget") == budget and development_pair_pass(row)
                        for row in reports)):
            return budget
    return None


def select_formal_pair(reports: Sequence[Mapping]) -> Mapping:
    """Only the fixed 24-seed quality rows select one of the three pairs."""
    if (len(reports) != len(FORMAL_PAIR_SEEDS) or
            {row.get("pair_seed") for row in reports} != set(FORMAL_PAIR_SEEDS)):
        raise ValueError("formal selection requires the three preregistered pairs")
    passing = []
    for row in reports:
        if row.get("selection_seeds") != list(FORMAL_SELECTION_SEEDS):
            raise ValueError("formal checkpoint selection seed mismatch")
        try:
            score = pair_quality_score(row["variants"])
        except (KeyError, TypeError, ValueError):
            continue
        passing.append((score, -int(row["pair_seed"]), row))
    if not passing:
        raise ValueError("no formal pair passed both Warehouse quality gates")
    return max(passing, key=lambda item: (item[0], item[1]))[2]


def claim_once(path: str | Path, *, phase: str, candidate_id: str,
               actor_hashes: Mapping, seed_rows: Sequence[int],
               release_id: str = INITIAL_RELEASE_ID) -> dict:
    """Durably claim final audit or heldout seeds before the first rollout.

    A started claim is never reusable after a crash or a failed audit.  A new
    candidate must use a new version and fresh sealed seed blocks.
    """
    if phase not in ("final_program_audit", "heldout"):
        raise ValueError("only sealed phases require a one-use claim")
    expected = sealed_seed_blocks(release_id)[phase]
    if tuple(seed_rows) != expected:
        raise ValueError("one-use claim has wrong seeds")
    if not candidate_id or not actor_hashes:
        raise ValueError("candidate and actor hashes required")
    receipt = {"phase": phase, "candidate_id": candidate_id,
               "release_id": release_id,
               "actor_hashes": actor_hashes, "seeds": list(seed_rows),
               "seed_sha256": _hash_json(list(seed_rows))}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    return receipt
