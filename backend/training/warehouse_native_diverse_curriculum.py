"""Train-only legal starts selected across independently shuffled source episodes.

This is a new selection experiment, not the completed v2 generator. The v2 bank
version is retained only as its snapshot serialization contract. No NN is created
or trained here. Only ``advance`` and an explicitly requested replay execute
environment steps; constructors, bank inspection and resume are read-only.
"""
from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path
import re

import numpy as np

from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.partners import partner_actions
from env.warehouse_native.scenarios import reset_scenario, scenario_fingerprint
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native_v2 import (
    CURRICULUM_VERSION, curriculum_categories, validate_curriculum_bank,
    v2_source_hashes,
)

SELECTION_VERSION = "warehouse-native-diverse-curriculum.v1"
DEFAULT_SELECTION_CONFIG = {
    "source_rollout_steps": 60,
    "minimum_remaining_steps": 60,
    "sources_per_category": 32,
    "entries_per_source_per_category": 4,
    "minimum_unique_sources": 32,
    "minimum_sources_per_category": 16,
}
CATEGORIES = (
    "near_pickup", "carrying", "low_battery_near_charger", "narrow_encounter",
)
ENTRY_FIELDS = {
    "id", "source_id", "source_initial_fingerprint", "source_split", "category",
    "frame", "physical_fingerprint", "initial_content_fingerprint", "snapshot",
}
BANK_FIELDS = {
    "version", "selection_version", "experiment_revision", "protocol_sha256",
    "selection_config", "sources", "scenario_manifest_sha256", "generation_steps",
    "maximum_generation_steps", "entries_per_category", "generator_partner",
    "generator_source_sha256", "counts", "entries", "teacher_action_labels",
    "manual_state_edits", "source_split", "heldout_initial_exclusions",
    "generalization_scope", "selection_method", "source_visits", "coverage",
}
_HASH = re.compile(r"^[0-9a-f]{64}$")


def diverse_curriculum_source_hashes():
    """Bind the new selector and all frozen v2/physical dependencies."""
    result = v2_source_hashes()
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    return result


def _integer(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {label}")
    return value


def _selection_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_SELECTION_CONFIG):
        raise ValueError("Unknown diverse curriculum configuration")
    cfg = {**DEFAULT_SELECTION_CONFIG, **copy.deepcopy(value)}
    for key, number in cfg.items():
        _integer(number, key, 1)
    for key, limit in (("source_rollout_steps", 60), ("sources_per_category", 32),
                       ("entries_per_source_per_category", 4)):
        _integer(cfg[key], key, 1, limit)
    if cfg["minimum_sources_per_category"] > cfg["sources_per_category"]:
        raise ValueError("Per-category source requirement exceeds reservoir capacity")
    return cfg


def _configuration(protocol):
    if not isinstance(protocol.get("experiment_revision"), (dict, str)) or not protocol["experiment_revision"]:
        raise ValueError("A distinct experiment_revision is required")
    cfg = protocol["curriculum"]
    if (cfg.get("version") != CURRICULUM_VERSION or cfg.get("source_split") != "train"
            or cfg.get("generator_partner") != "skilled"
            or cfg.get("teacher_action_labels") is not False
            or cfg.get("manual_state_edits") is not False
            or cfg.get("categories") != list(CATEGORIES)):
        raise ValueError("Curriculum requires deterministic skilled train-only public-state generation")
    _integer(protocol.get("seed"), "protocol seed")
    _integer(cfg.get("maximum_generation_steps"), "generation cap", 1, 10000)
    _integer(cfg.get("entries_per_category"), "category size", 1)
    selection = _selection_config(protocol.get("diverse_curriculum", {}))
    if cfg["entries_per_category"] != selection["sources_per_category"] * selection["entries_per_source_per_category"]:
        raise ValueError("Category size must equal source buckets times entries per source")
    return copy.deepcopy(cfg), selection


def _source_index(scenarios):
    """Check initial content without simulating any episode, including held-out."""
    sources = scenarios["splits"]["train"]
    if not sources:
        raise ValueError("Diverse curriculum needs at least one train source")
    heldout = {e["fingerprint"] for key, entries in scenarios["splits"].items()
               if key != "train" for e in entries}
    ids, fingerprints = set(), set()
    env = NativeWarehouseEnv()
    for source in sources:
        if (source["id"] in ids or source["fingerprint"] in fingerprints
                or source["fingerprint"] in heldout):
            raise ValueError("Duplicate or held-out train initial source")
        reset_scenario(env, source)
        if env.done or env.state.frame != 0:
            raise ValueError("Curriculum source must be a live initial frame zero")
        ids.add(source["id"])
        fingerprints.add(source["fingerprint"])
    return {entry["id"]: entry for entry in sources}, heldout


def _coverage(bank, requirements=None):
    cfg = bank["selection_config"]
    requirement = {
        "entries_per_category": bank["entries_per_category"],
        "minimum_unique_sources": cfg["minimum_unique_sources"],
        "minimum_sources_per_category": cfg["minimum_sources_per_category"],
    }
    if requirements is not None:
        if not isinstance(requirements, dict) or set(requirements) - set(requirement):
            raise ValueError("Unknown curriculum coverage requirement")
        for key, value in requirements.items():
            _integer(value, key, 1)
        requirement.update(requirements)
    by_category = {category: Counter() for category in CATEGORIES}
    remaining = []
    for entry in bank["entries"]:
        by_category[entry["category"]][entry["source_id"]] += 1
        remaining.append(entry["snapshot"]["configuration"]["horizon"] - entry["frame"])
    unique = {entry["source_id"] for entry in bank["entries"]}
    failures = []
    if len(unique) < requirement["minimum_unique_sources"]:
        failures.append("insufficient_unique_sources")
    for category, counts in by_category.items():
        if sum(counts.values()) < requirement["entries_per_category"]:
            failures.append(f"{category}:insufficient_entries")
        if len(counts) < requirement["minimum_sources_per_category"]:
            failures.append(f"{category}:insufficient_sources")
    return {
        "unique_sources": len(unique),
        "source_ids": sorted(unique),
        "entries_per_category": {key: sum(value.values()) for key, value in by_category.items()},
        "sources_per_category": {key: len(value) for key, value in by_category.items()},
        "source_counts_per_category": {key: dict(sorted(value.items())) for key, value in by_category.items()},
        "minimum_remaining_steps_observed": min(remaining) if remaining else None,
        "maximum_remaining_steps_observed": max(remaining) if remaining else None,
        "requirements": requirement,
        "quality_passed": not failures,
        "quality_failures": failures,
        "scope": "legal_train_starts_initial_state_isolation_not_capability_or_unseen_layouts",
    }


class CurriculumQualityError(ValueError):
    def __init__(self, coverage):
        self.coverage = copy.deepcopy(coverage)
        super().__init__("Diverse curriculum quality failed: " + ", ".join(coverage["quality_failures"]))


def _validate_bank(bank, scenarios, *, replay=False):
    if (set(bank) != BANK_FIELDS or bank.get("selection_version") != SELECTION_VERSION
            or not isinstance(bank.get("experiment_revision"), (str, dict))
            or not bank["experiment_revision"]
            or not _HASH.fullmatch(str(bank.get("protocol_sha256", "")))
            or bank.get("sources") != diverse_curriculum_source_hashes()):
        raise ValueError("Diverse curriculum version/provenance mismatch")
    cfg = _selection_config(bank.get("selection_config"))
    if cfg != bank["selection_config"]:
        raise ValueError("Bank must record complete selection configuration")
    _integer(bank.get("entries_per_category"), "category size", 1)
    if bank["entries_per_category"] != cfg["sources_per_category"] * cfg["entries_per_source_per_category"]:
        raise ValueError("Invalid category reservoir capacity")
    cap = _integer(bank.get("maximum_generation_steps"), "generation cap", 1, 10000)
    _integer(bank.get("generation_steps"), "generation steps", 0, cap)
    sources, _ = _source_index(scenarios)
    if bank.get("generalization_scope") != "initial_state_split_not_full_trajectory_isolation":
        raise ValueError("Curriculum scope mismatch")
    if bank.get("selection_method") != "unique_source_reservoir_then_unique_state_reservoir":
        raise ValueError("Curriculum selection method mismatch")
    _integer(bank.get("heldout_initial_exclusions"), "held-out exclusion count", 0, bank["generation_steps"])
    if not isinstance(bank.get("entries"), list):
        raise ValueError("Curriculum entries must be a list")
    # Check category/shape before asking the frozen validator to restore states.
    for entry in bank["entries"]:
        if set(entry) != ENTRY_FIELDS or entry["category"] not in CATEGORIES:
            raise ValueError("Unexpected curriculum field/category, including possible teacher labels")
        _integer(entry["frame"], "entry frame", 1, cfg["source_rollout_steps"])
        if entry["snapshot"]["configuration"]["horizon"] - entry["frame"] < cfg["minimum_remaining_steps"]:
            raise ValueError("Curriculum entry has insufficient remaining steps")
    # Includes duplicate physical states, source membership, exact frame/category,
    # physical fingerprints and optional deterministic real-transition witnesses.
    validate_curriculum_bank(bank, scenarios, replay=replay)
    visits = bank.get("source_visits")
    if not isinstance(visits, dict) or set(visits) - set(sources):
        raise ValueError("Invalid curriculum source visitation accounting")
    for value in visits.values():
        _integer(value, "source visit count", 1)
    if sum(visits.values()) > bank["generation_steps"]:
        raise ValueError("Too many source visits for generated steps")
    if {entry["source_id"] for entry in bank["entries"]} - set(visits):
        raise ValueError("Unvisited source in curriculum entries")
    coverage = _coverage(bank)
    if bank.get("coverage") != coverage:
        raise ValueError("Curriculum coverage report does not match entries")
    for category, counts in coverage["source_counts_per_category"].items():
        if len(counts) > cfg["sources_per_category"] or any(n > cfg["entries_per_source_per_category"] for n in counts.values()):
            raise ValueError(f"Per-source reservoir capacity exceeded: {category}")
    return coverage


def validate_diverse_curriculum_bank(bank, scenarios, requirements=None, replay=False):
    """Return recomputed coverage; reject insufficient quality without filling it.

    Explicit smaller requirements are useful for bounded correctness fixtures.
    Production callers omit them to enforce the recorded frozen requirements.
    ``replay=False`` validates snapshot facts but does not claim path reachability.
    """
    _validate_bank(bank, scenarios, replay=replay)
    coverage = _coverage(bank, requirements)
    if not coverage["quality_passed"]:
        raise CurriculumQualityError(coverage)
    return coverage


class DiverseCurriculumBuilder:
    """Resumable source-stratified reservoirs with independent ordering RNG.

    Each source enters a category's source reservoir only once, regardless of
    later revisits. Within a selected source, each unique assigned physical state
    enters its state reservoir once. Evicted sources/states are not reintroduced
    as new candidates. Thus long trajectories and early initial states cannot
    fill an entire category. Each snapshot is assigned to only one category.
    """
    def __init__(self, protocol, scenarios):
        self.protocol = copy.deepcopy(protocol)
        self.scenarios = copy.deepcopy(scenarios)
        self.cfg, self.selection_cfg = _configuration(self.protocol)
        self.sources, self.heldout = _source_index(self.scenarios)
        self.order_rng = np.random.default_rng(np.random.SeedSequence([protocol["seed"], 1301]))
        self.reservoir_rng = np.random.default_rng(np.random.SeedSequence([protocol["seed"], 1302]))
        self.source_order = list(self.sources)
        self.order_rng.shuffle(self.source_order)
        self.source_cursor = 0
        self.source_cycle = 0
        self.source_visits = Counter()
        self.env = NativeWarehouseEnv()
        self.current_source_id = None
        self.source_steps = 0
        self.steps = 0
        self.reservoirs = {category: [] for category in CATEGORIES}
        self.category_sources_seen = {category: [] for category in CATEGORIES}
        self._category_source_sets = {category: set() for category in CATEGORIES}
        self.seen_physical = set()
        self.category_candidates = Counter()
        self.heldout_exclusions = 0
        self.remaining_exclusions = 0
        self.duplicate_exclusions = 0
        self.next_entry_id = 0

    @property
    def entries(self):
        return [entry for category in CATEGORIES for bucket in self.reservoirs[category]
                for entry in bucket["entries"]]

    @property
    def counts(self):
        return Counter(entry["category"] for entry in self.entries)

    def _start_source(self):
        if self.source_cursor == len(self.source_order):
            self.source_order = list(self.sources)
            self.order_rng.shuffle(self.source_order)
            self.source_cursor = 0
            self.source_cycle += 1
        self.current_source_id = self.source_order[self.source_cursor]
        self.source_cursor += 1
        self.source_visits[self.current_source_id] += 1
        reset_scenario(self.env, self.sources[self.current_source_id])
        self.source_steps = 0

    def _bucket(self, category, source_id):
        buckets = self.reservoirs[category]
        if source_id in self._category_source_sets[category]:
            return next((bucket for bucket in buckets if bucket["source_id"] == source_id), None)
        self._category_source_sets[category].add(source_id)
        self.category_sources_seen[category].append(source_id)
        bucket = {"source_id": source_id, "seen_entries": 0, "entries": []}
        capacity = self.selection_cfg["sources_per_category"]
        if len(buckets) < capacity:
            buckets.append(bucket)
            return bucket
        slot = int(self.reservoir_rng.integers(len(self.category_sources_seen[category])))
        if slot < capacity:
            buckets[slot] = bucket
            return bucket
        return None

    def _consider(self):
        if self.env.done:
            return
        if self.env.config.horizon - self.env.state.frame < self.selection_cfg["minimum_remaining_steps"]:
            self.remaining_exclusions += 1
            return
        initial_content = scenario_fingerprint(self.env)
        if initial_content in self.heldout:
            self.heldout_exclusions += 1
            return
        categories = curriculum_categories(self.env)
        fingerprint = self.env.fingerprint()
        if fingerprint in self.seen_physical:
            self.duplicate_exclusions += 1
            return
        if not categories:
            return
        self.seen_physical.add(fingerprint)
        counts = self.counts
        category = min(categories, key=lambda key: (counts[key], self.category_candidates[key], CATEGORIES.index(key)))
        self.category_candidates[category] += 1
        bucket = self._bucket(category, self.current_source_id)
        if bucket is None:
            return
        bucket["seen_entries"] += 1
        capacity = self.selection_cfg["entries_per_source_per_category"]
        slot = len(bucket["entries"])
        if slot >= capacity:
            slot = int(self.reservoir_rng.integers(bucket["seen_entries"]))
            if slot >= capacity:
                return
        source = self.sources[self.current_source_id]
        entry = {
            "id": f"curriculum_{self.next_entry_id:05d}",
            "source_id": source["id"],
            "source_initial_fingerprint": source["fingerprint"],
            "source_split": "train", "category": category,
            "frame": self.env.state.frame, "physical_fingerprint": fingerprint,
            "initial_content_fingerprint": initial_content,
            "snapshot": self.env.snapshot(),
        }
        self.next_entry_id += 1
        if slot == len(bucket["entries"]):
            bucket["entries"].append(entry)
        else:
            bucket["entries"][slot] = entry

    def advance(self, n):
        _integer(n, "advance step count")
        if self.steps + n > self.cfg["maximum_generation_steps"]:
            raise ValueError("Diverse curriculum generation budget exceeded")
        for _ in range(n):
            if (self.current_source_id is None or self.env.done
                    or self.source_steps >= self.selection_cfg["source_rollout_steps"]):
                self._start_source()
            # The deterministic reference program only generates legal starts.
            # No actions, targets or routes are stored as supervision or inputs.
            self.env.step(partner_actions(self.env, "skilled"))
            self.steps += 1
            self.source_steps += 1
            self._consider()

    def bank(self):
        bank = {
            "version": CURRICULUM_VERSION,
            "selection_version": SELECTION_VERSION,
            "experiment_revision": copy.deepcopy(self.protocol["experiment_revision"]),
            "protocol_sha256": digest(self.protocol),
            "selection_config": copy.deepcopy(self.selection_cfg),
            "sources": diverse_curriculum_source_hashes(),
            "scenario_manifest_sha256": digest(self.scenarios),
            "generation_steps": self.steps,
            "maximum_generation_steps": self.cfg["maximum_generation_steps"],
            "entries_per_category": self.cfg["entries_per_category"],
            "generator_partner": "skilled",
            "generator_source_sha256": file_hash(ROOT / "env/warehouse_native/partners.py"),
            "counts": dict(self.counts), "entries": copy.deepcopy(self.entries),
            "teacher_action_labels": False, "manual_state_edits": False,
            "source_split": "train", "heldout_initial_exclusions": self.heldout_exclusions,
            "generalization_scope": "initial_state_split_not_full_trajectory_isolation",
            "selection_method": "unique_source_reservoir_then_unique_state_reservoir",
            "source_visits": dict(sorted(self.source_visits.items())),
        }
        bank["coverage"] = _coverage(bank)
        return bank

    def state_dict(self):
        payload = {
            "version": CURRICULUM_VERSION, "selection_version": SELECTION_VERSION,
            "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios),
            "sources": diverse_curriculum_source_hashes(),
            "steps": self.steps,
            "order_rng": copy.deepcopy(self.order_rng.bit_generator.state),
            "reservoir_rng": copy.deepcopy(self.reservoir_rng.bit_generator.state),
            "source_order": list(self.source_order), "source_cursor": self.source_cursor,
            "source_cycle": self.source_cycle, "source_visits": dict(self.source_visits),
            "current_source_id": self.current_source_id, "source_steps": self.source_steps,
            "environment": self.env.snapshot() if self.current_source_id is not None else None,
            "reservoirs": copy.deepcopy(self.reservoirs),
            "category_sources_seen": copy.deepcopy(self.category_sources_seen),
            "category_candidates": dict(self.category_candidates),
            "seen_physical": sorted(self.seen_physical),
            "heldout_exclusions": self.heldout_exclusions,
            "remaining_exclusions": self.remaining_exclusions,
            "duplicate_exclusions": self.duplicate_exclusions,
            "next_entry_id": self.next_entry_id,
        }
        payload["content_sha256"] = digest(payload)
        return payload

    def load_state_dict(self, payload):
        """Validate in a temporary builder so malformed state cannot partly apply."""
        if (set(payload) != set(self.state_dict())
                or payload["content_sha256"] != digest({key: value for key, value in payload.items() if key != "content_sha256"})
                or payload["version"] != CURRICULUM_VERSION
                or payload["selection_version"] != SELECTION_VERSION
                or payload["protocol_sha256"] != digest(self.protocol)
                or payload["scenario_manifest_sha256"] != digest(self.scenarios)
                or payload["sources"] != diverse_curriculum_source_hashes()):
            raise ValueError("Diverse curriculum resume version/provenance/integrity mismatch")
        candidate = type(self)(self.protocol, self.scenarios)
        candidate._restore(payload)
        self.__dict__.update(candidate.__dict__)

    def _restore(self, payload):
        self.steps = _integer(payload["steps"], "resume generation steps", 0, self.cfg["maximum_generation_steps"])
        order = payload["source_order"]
        if not isinstance(order, list) or len(order) != len(self.sources) or set(order) != set(self.sources):
            raise ValueError("Invalid source shuffle permutation")
        self.source_order = list(order)
        self.source_cursor = _integer(payload["source_cursor"], "source cursor", 0, len(order))
        self.source_cycle = _integer(payload["source_cycle"], "source cycle", 0, self.steps)
        self.source_visits = Counter(payload["source_visits"])
        if sum(self.source_visits.values()) != self.source_cycle * len(order) + self.source_cursor:
            raise ValueError("Source cycle/visit accounting mismatch")
        for source_id in self.sources:
            expected = self.source_cycle + int(source_id in order[:self.source_cursor])
            if self.source_visits[source_id] != expected:
                raise ValueError("Source visitation is not shuffled round-robin")
        self.current_source_id = payload["current_source_id"]
        self.source_steps = _integer(payload["source_steps"], "current source steps", 0, self.selection_cfg["source_rollout_steps"])
        if self.current_source_id is None:
            if self.steps != 0 or self.source_cursor != 0 or self.source_steps != 0 or payload["environment"] is not None:
                raise ValueError("Missing in-progress source state")
        else:
            if (self.current_source_id not in self.sources or self.source_cursor == 0
                    or order[self.source_cursor - 1] != self.current_source_id or self.source_steps == 0):
                raise ValueError("Current source/cursor mismatch")
            self.env.restore(payload["environment"])
            if self.env.state.frame != self.source_steps:
                raise ValueError("Current source frame mismatch; frame edits are forbidden")
        self.order_rng.bit_generator.state = copy.deepcopy(payload["order_rng"])
        self.reservoir_rng.bit_generator.state = copy.deepcopy(payload["reservoir_rng"])
        for field in ("heldout_exclusions", "remaining_exclusions", "duplicate_exclusions", "next_entry_id"):
            setattr(self, field, _integer(payload[field], field, 0, self.steps))
        if set(payload["category_candidates"]) - set(CATEGORIES):
            raise ValueError("Unknown category candidate count")
        self.category_candidates = Counter(payload["category_candidates"])
        for n in self.category_candidates.values():
            _integer(n, "category candidate count", 1, self.steps)
        seen = payload["seen_physical"]
        if (not isinstance(seen, list) or len(set(seen)) != len(seen)
                or any(not isinstance(value, str) or not _HASH.fullmatch(value) for value in seen)
                or len(seen) != sum(self.category_candidates.values()) or len(seen) > self.steps):
            raise ValueError("Invalid unique physical candidate accounting")
        self.seen_physical = set(seen)
        if set(payload["reservoirs"]) != set(CATEGORIES) or set(payload["category_sources_seen"]) != set(CATEGORIES):
            raise ValueError("Invalid category reservoirs")
        self.reservoirs = copy.deepcopy(payload["reservoirs"])
        self.category_sources_seen = copy.deepcopy(payload["category_sources_seen"])
        for category in CATEGORIES:
            seen_sources = self.category_sources_seen[category]
            if (not isinstance(seen_sources, list) or len(set(seen_sources)) != len(seen_sources)
                    or set(seen_sources) - set(self.source_visits)
                    or len(seen_sources) > self.category_candidates[category]):
                raise ValueError("Invalid unique category source accounting")
            self._category_source_sets[category] = set(seen_sources)
            buckets = self.reservoirs[category]
            if not isinstance(buckets, list) or len(buckets) != min(len(seen_sources), self.selection_cfg["sources_per_category"]):
                raise ValueError("Invalid source reservoir size")
            bucket_ids = set()
            total_candidates = 0
            for bucket in buckets:
                if (set(bucket) != {"source_id", "seen_entries", "entries"}
                        or bucket["source_id"] not in seen_sources or bucket["source_id"] in bucket_ids):
                    raise ValueError("Invalid or duplicate source reservoir bucket")
                bucket_ids.add(bucket["source_id"])
                count = _integer(bucket["seen_entries"], "bucket seen count", 1, self.steps)
                total_candidates += count
                entries = bucket["entries"]
                if not isinstance(entries, list) or len(entries) != min(count, self.selection_cfg["entries_per_source_per_category"]):
                    raise ValueError("Invalid inner state reservoir size")
                for entry in entries:
                    if (entry.get("source_id") != bucket["source_id"] or entry.get("category") != category
                            or entry.get("physical_fingerprint") not in self.seen_physical):
                        raise ValueError("Reservoir entry source/category mismatch")
                    try:
                        entry_number = int(entry["id"].removeprefix("curriculum_"))
                    except (KeyError, ValueError, AttributeError) as error:
                        raise ValueError("Invalid curriculum entry ID") from error
                    if not entry["id"].startswith("curriculum_") or not 0 <= entry_number < self.next_entry_id:
                        raise ValueError("Invalid curriculum entry serial")
            if total_candidates > self.category_candidates[category]:
                raise ValueError("Reservoir count exceeds category candidates")
        # Quality need not pass mid-generation. All structural/physical checks do.
        _validate_bank(self.bank(), self.scenarios, replay=False)
