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
SCENARIO_BLOCK_STEPS = 120
ONLINE_PROGRAM_EXTRACTION_PERCENTAGES = (75, 84, 93)
FINAL_PROGRAM_SPLIT_COUNTS = {"fit": 50, "calibration": 25,
                              "eligibility": 25, "validation": 24}
REGISTERED_ARCHITECTURE_MODE = "intrinsic_certified"
REGISTERED_BC_RECIPE_VERSION = "warehouse-public-dual-bc.v6.5-intrinsic-certified"
REGISTERED_BC_SPEC = {
    "bc_episodes": 200,
    "bc_epochs": 30,
    "dagger_rounds": 4,
    "dagger_episodes": 80,
    "dagger_epochs": 10,
}


def _hash_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def scenario_seed(namespace: str, pair_seed: int, joint_step: int) -> int:
    """Assigned public scenario for a fixed joint-step block."""
    if joint_step < 0:
        raise ValueError("joint_step must be nonnegative")
    message = f"{namespace}|{pair_seed}|{joint_step // SCENARIO_BLOCK_STEPS}".encode()
    return int.from_bytes(sha256(message).digest()[:8], "big") & 0x7fff_ffff


def scenario_schedule_sha256(namespace: str, pair_seed: int, joint_steps: int) -> str:
    """Hash the exact assigned scenario for every joint training step."""
    if joint_steps < 0:
        raise ValueError("joint_steps must be nonnegative")
    digest = sha256()
    for step in range(joint_steps):
        digest.update(scenario_seed(namespace, pair_seed, step).to_bytes(4, "little"))
    return digest.hexdigest()


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


def final_program_fit_seed_schedule(release_id: str, pair_seed: int
                                    ) -> dict[str, list[int]]:
    """Fresh four-part program data fixed by release and paired run identity.

    Candidate directory names cannot change the samples used by a formal
    Actor's post-selection program. This schedule is distinct from every
    quality, stress, program-audit and sealed test block.
    """
    if pair_seed not in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        raise ValueError("unregistered Warehouse v6 pair seed")
    sealed = sealed_seed_blocks(release_id)
    minor = int(release_id.rsplit(".v6.", 1)[1])
    beginning = 50_000_000 + minor * 1_000_000 + pair_seed * 1_000
    schedule, cursor = {}, beginning
    for name, count in FINAL_PROGRAM_SPLIT_COUNTS.items():
        schedule[name] = list(range(cursor, cursor + count))
        cursor += count
    chosen = {seed for block in schedule.values() for seed in block}
    reserved = (
        set(DEVELOPMENT_SELECTION_SEEDS)
        | set(DEVELOPMENT_STRESS_BASE_SEEDS)
        | {scenario for _, scenario, _ in stress_schedule()}
        | set(DEVELOPMENT_PROGRAM_AUDIT_SEEDS)
        | set(FORMAL_SELECTION_SEEDS)
        | set(FORMAL_PROGRAM_PREAUDIT_SEEDS)
        | set(sealed["final_program_audit"])
        | set(sealed["heldout"])
        | {task_scenario_seed(seed, task)
           for seed in sealed["heldout"] for task in TASKS}
    )
    if chosen & reserved:
        raise RuntimeError("Warehouse final program seeds overlap protocol blocks")
    return schedule


def online_program_extraction_steps(joint_steps: int) -> tuple[int, ...]:
    """Lock complete training-block boundaries before a paired run starts."""
    if joint_steps not in BUDGETS:
        raise ValueError("Warehouse online program budget must be preregistered")
    numerators = tuple(joint_steps * percent
                       for percent in ONLINE_PROGRAM_EXTRACTION_PERCENTAGES)
    if any(numerator % 100 for numerator in numerators):
        raise RuntimeError("Warehouse online extraction percentage is not integral")
    steps = tuple(numerator // 100 for numerator in numerators)
    if (any(step % SCENARIO_BLOCK_STEPS for step in steps)
            or any(later - earlier <
                   SCENARIO_BLOCK_STEPS * sum(FINAL_PROGRAM_SPLIT_COUNTS.values())
                   for earlier, later in zip(steps, steps[1:]))
            or not all(0 < step < joint_steps for step in steps)):
        raise RuntimeError("Warehouse online program cohorts overlap or split a block")
    return steps


@dataclass(frozen=True)
class PairRunSpec:
    candidate_id: str
    pair_seed: int
    joint_steps: int
    output_dir: str
    release_id: str = INITIAL_RELEASE_ID
    scenario_namespace: str = "warehouse-v6-matched-120-step-blocks"
    checkpoint_seeds: tuple[int, ...] = ()
    online_program_extraction_steps: tuple[int, ...] = ()
    architecture_mode: str = REGISTERED_ARCHITECTURE_MODE
    bc_recipe_version: str = REGISTERED_BC_RECIPE_VERSION

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
        if (self.architecture_mode != REGISTERED_ARCHITECTURE_MODE
                or self.bc_recipe_version != REGISTERED_BC_RECIPE_VERSION):
            raise ValueError("Warehouse registered Actor or BC recipe mismatch")
        scheduled = online_program_extraction_steps(self.joint_steps)
        if self.online_program_extraction_steps:
            if self.online_program_extraction_steps != scheduled:
                raise ValueError("Warehouse online program extraction schedule mismatch")
        else:
            object.__setattr__(self, "online_program_extraction_steps", scheduled)
        sealed_seed_blocks(self.release_id)
        final_program_fit_seed_schedule(self.release_id, self.pair_seed)

    def to_mapping(self) -> dict:
        row = asdict(self)
        schedule = final_program_fit_seed_schedule(self.release_id, self.pair_seed)
        row["final_program_fit_seed_schedule"] = schedule
        row["final_program_fit_seed_schedule_sha256"] = _hash_json(schedule)
        return row

    def sha256(self) -> str:
        # A moved artifact directory does not change the scientific run.
        return _hash_json({key: value for key, value in self.to_mapping().items()
                           if key != "output_dir"})

    def training_config_sha256(self, source_closure_sha256: str) -> str:
        """Independently derive the registered BC/checkpoint config identity."""
        if (not isinstance(source_closure_sha256, str)
                or len(source_closure_sha256) != 64
                or any(char not in "0123456789abcdef"
                       for char in source_closure_sha256)):
            raise ValueError("invalid Warehouse training source closure")
        config = {**self.to_mapping(), **REGISTERED_BC_SPEC,
                  "source_closure_sha256": source_closure_sha256}
        return _hash_json({key: value for key, value in config.items()
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
        actionable_coverage = float(metrics["actionable_coverage"])
        agreement = float(metrics["agreement"])
        kl = float(metrics["kl"])
        rows = int(metrics["rows"])
        actionable_rows = int(metrics["actionable_rows"])
        illegal = int(metrics["illegal_count"])
    except (KeyError, TypeError, ValueError):
        return False
    return (rows > 0 and actionable_rows > 0 and
            actionable_rows <= rows and illegal == 0 and
            all(map(_finite, (coverage, actionable_coverage, agreement, kl))) and
            minimum_coverage <= coverage <= 1 and
            minimum_coverage <= actionable_coverage <= coverage and
            minimum_agreement <= agreement <= 1 and 0 <= kl <= maximum_kl)


def online_program_cohorts_valid(extractions: object, *,
                                 joint_steps: int,
                                 training_episodes: object) -> bool:
    """Recompute the four purpose cohorts instead of trusting a report flag."""
    if not isinstance(extractions, Mapping) or set(extractions) != set(ROLES):
        return False
    if not isinstance(training_episodes, list) or not training_episodes:
        return False
    episodes = {}
    for episode in training_episodes:
        if not isinstance(episode, Mapping):
            return False
        key = episode.get("episode_key")
        seed = episode.get("seed")
        start = episode.get("start_joint_step")
        length = episode.get("joint_steps")
        if (not isinstance(key, str) or key in episodes
                or type(seed) is not int or type(start) is not int
                or type(length) is not int or start < 0 or length <= 0
                or start + length > joint_steps):
            return False
        parts = key.split(":")
        if (len(parts) != 2 or parts[0] != str(seed)
                or not parts[1].isdigit() or parts[1] != str(int(parts[1]))):
            return False
        episodes[key] = (seed, start)
    steps = online_program_extraction_steps(joint_steps)
    all_scenes: set[int] = set()
    all_episode_ids: set[str] = set()
    for index, step in enumerate(steps):
        reference = None
        for role in ROLES:
            rows = extractions[role]
            if not isinstance(rows, list) or len(rows) != len(steps):
                return False
            row = rows[index]
            if not isinstance(row, Mapping) or (
                    row.get("joint_steps"), row.get("window_start_joint_step"),
                    row.get("window_end_joint_step")) != (
                    step, 0 if index == 0 else steps[index - 1], step):
                return False
            scenes = row.get("training_split_scenario_seeds")
            episode_ids = row.get("training_split_episode_ids")
            counts = row.get("training_split_episode_counts")
            if (not isinstance(scenes, Mapping)
                    or not isinstance(episode_ids, Mapping)
                    or not isinstance(counts, Mapping)
                    or any(set(part) != set(FINAL_PROGRAM_SPLIT_COUNTS)
                           for part in (scenes, episode_ids, counts))):
                return False
            local_scenes: set[int] = set()
            local_ids: set[str] = set()
            for purpose, required in FINAL_PROGRAM_SPLIT_COUNTS.items():
                seeds, ids = scenes[purpose], episode_ids[purpose]
                if (not isinstance(seeds, list) or len(seeds) != required
                        or any(type(seed) is not int for seed in seeds)
                        or not isinstance(ids, list) or not ids
                        or any(not isinstance(item, str) for item in ids)
                        or counts[purpose] != len(ids)):
                    return False
                seed_set, id_set = set(seeds), set(ids)
                if (len(seed_set) != required or len(id_set) != len(ids)
                        or seed_set & local_scenes or id_set & local_ids):
                    return False
                try:
                    id_scenes = {int(item.split(":", 1)[0]) for item in ids}
                except ValueError:
                    return False
                if id_scenes != seed_set:
                    return False
                if any(item not in episodes
                       or episodes[item][0] not in seed_set
                       or not (row["window_start_joint_step"] <=
                               episodes[item][1] < step)
                       for item in ids):
                    return False
                local_scenes.update(seed_set)
                local_ids.update(id_set)
            if reference is None:
                reference = (scenes, episode_ids)
                if local_scenes & all_scenes or local_ids & all_episode_ids:
                    return False
                all_scenes.update(local_scenes)
                all_episode_ids.update(local_ids)
            elif reference != (scenes, episode_ids):
                return False
    return True


def paired_training_pass(variants: Mapping, *, joint_steps: int) -> bool:
    """Check identical starts/schedules and actual gradients for both roles."""
    if joint_steps not in BUDGETS or set(variants) != set(VARIANTS):
        return False
    raw, regularized = (variants[name] for name in VARIANTS)
    if not isinstance(raw, Mapping) or not isinstance(regularized, Mapping):
        return False
    extraction_steps = list(online_program_extraction_steps(joint_steps))
    for report in (raw, regularized):
        if (report.get("joint_steps") != joint_steps or
                report.get("actor_decisions") != joint_steps * len(ROLES) or
                report.get("architecture_mode") != REGISTERED_ARCHITECTURE_MODE or
                report.get("bc_recipe_version") != REGISTERED_BC_RECIPE_VERSION or
                report.get("online_program_extraction_steps") != extraction_steps or
                report.get("online_program_split_block_cycle") !=
                FINAL_PROGRAM_SPLIT_COUNTS or
                report.get("online_program_cohorts_disjoint") is not True or
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
    if not online_program_cohorts_valid(
            regularized.get("program_extractions"), joint_steps=joint_steps,
            training_episodes=regularized.get("training_episodes")):
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
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt
