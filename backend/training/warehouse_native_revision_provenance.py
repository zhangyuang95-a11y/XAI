"""Portable, byte-preserving r1 provenance; deliberately not a release verifier.

The writer's expected contracts must come from a trusted caller, never from the
run being inspected. The reader requires the writer's manifest SHA-256 through
an independent trusted channel. A bundle cannot authenticate itself. This only
checks identities, recorded budget/review bindings and file preservation; it
does not verify capability, selection, NN tensors, curriculum physics or grant
training/publication permission. Failed completed candidates may be archived.

No imports from the trainer/receipt are intentional: those import torch and the
environment. This module uses only the standard library and never unpickles a
checkpoint. Returned file access is immutable verified bytes, never a path that
could be replaced after verification. Unknown files in original run directories
are not copied; only the explicit evidence whitelist below is transported.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType

VERSION = "warehouse-native-revision-provenance.v1"
R1 = "warehouse-native-foundation.r1"
V2 = "warehouse-native-foundation.v2"
BUDGET = "warehouse-native-revision-budget.v1"
PROTOCOL_SOURCE = "backend/training/warehouse_native_revision_protocol.json"
PARENT_PROTOCOL_SOURCE = "backend/training/warehouse_native_v2_protocol.json"
PARENT_FILES = frozenset({"run.lock", "run.json", "protocol.json", "scenarios.json",
    "progress.json", "sampling_budget.json", "validation_reference.json", "validation_random.json"})
FOUNDATION_REQUIRED = frozenset({"run.lock", "run.json", "protocol.json", "scenarios.json",
    "progress.json", "curriculum_bank.json", "curriculum_binding.json", "curriculum_quality.json",
    "initial_actor.npz", "latest_checkpoint.pt", "training.jsonl", "episodes.jsonl",
    "validation_reference.json", "validation_random.json"})
FOUNDATION_OPTIONAL = frozenset({"curriculum_generation_latest.json",
    "diagnostic_review_required.json", "diagnostic_review_acknowledged.json"})
PATTERNS = {
    "checkpoints": re.compile(r"(?:actor_[0-9]{7}\.npz|checkpoint_[0-9]{7}\.pt)\Z"),
    "validation": re.compile(r"step_[0-9]{7}\.json\Z"),
    "diagnostics": re.compile(r"(?:step_[0-9]{7}\.json|probe_actor_[0-9]{7}\.npz)\Z"),
    "diagnostics/acknowledgments": re.compile(r"step_[0-9]{7}\.json\Z"),
}
ROLES = frozenset({"foundation", "parent", "registry"})
HEX = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _hash(_canonical(value))


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key in provenance evidence")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Non-finite JSON in provenance evidence")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    # Also reject float overflow (e.g. 1e999), which parse_constant does not see.
    _canonical(value)
    return value


def _integer(value, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid provenance integer")
    return value


def _relative(value):
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or value.startswith("/") or any(x in ("", ".", "..") for x in value.split("/"))
            or str(PurePosixPath(value)) != value):
        raise ValueError("Non-canonical relative evidence path")
    return value


def _original(value):
    if (not isinstance(value, str) or not value.startswith("/") or value == "/"
            or "\\" in value or "\0" in value or str(PurePosixPath(value)) != value
            or any(x in ("", ".", "..") for x in value.split("/")[1:])):
        raise ValueError("Non-canonical original absolute identity")
    return value


def _absolute(value):
    return Path(_original(os.path.abspath(os.fspath(value))))


def _open_dir(path):
    """Walk from / using no-follow directory descriptors, including ancestors."""
    path = _absolute(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = following
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_file(path):
    with ExitStack() as stack:
        parent = _open_dir(path.parent)
        stack.callback(os.close, parent)
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise ValueError("Evidence must be a regular file without symbolic or hard links")
    return descriptor


def _bytes(path):
    descriptor = _open_file(path)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(descriptor)
        raw = stream.read()
        after = os.fstat(descriptor)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        if any(getattr(before, name) != getattr(after, name) for name in fields) or len(raw) != after.st_size:
            raise ValueError("Evidence changed while being read")
        return raw


def _entries(directory):
    descriptor = _open_dir(directory)
    try:
        return os.listdir(descriptor)
    finally:
        os.close(descriptor)


def _allowed(role, path, sources):
    _relative(path)
    if role == "parent": return path in PARENT_FILES
    if role == "registry": return path == "registry.json"
    if role != "foundation": return False
    if path in FOUNDATION_REQUIRED | FOUNDATION_OPTIONAL: return True
    if path.startswith("source_bundle/"): return path[14:] in sources
    folder, _, name = path.rpartition("/")
    return folder in PATTERNS and bool(PATTERNS[folder].fullmatch(name))


def _source_contract(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("A nonempty trusted source contract is required")
    for name, sha in value.items():
        _relative(name)
        if (not name.startswith(("backend/training/", "env/warehouse/", "env/warehouse_native/"))
                or PurePosixPath(name).suffix not in (".py", ".json")
                or not isinstance(sha, str) or not HEX.fullmatch(sha)):
            raise ValueError("Invalid trusted source contract")


def _contracts(protocol, sources, parent_sources, fixture):
    if type(fixture) is not bool:
        raise ValueError("Explicit fixture mode must be boolean")
    _source_contract(sources); _source_contract(parent_sources)
    revision = protocol.get("experiment_revision", {})
    if (protocol.get("version") != V2 or revision.get("version") != R1
            or PROTOCOL_SOURCE not in sources or PARENT_PROTOCOL_SOURCE not in parent_sources
            or any(sources.get(k) != v for k, v in parent_sources.items())):
        raise ValueError("Trusted contracts must bind r1 and its preserved v2 source subset")
    _integer(revision.get("maximum_remaining_training_related_steps"), 1, 290000)
    if fixture and protocol.get("test_fixture") is not True:
        raise ValueError("Fixture protocol must be explicitly marked")
    if not fixture and protocol.get("test_fixture") is not None:
        raise ValueError("Fixture protocol cannot enter production provenance")


def _fixture_check(value, fixture):
    if isinstance(value, dict):
        for key, item in value.items():
            if not fixture and key in ("fixture", "test_fixture", "synthetic_fixture"):
                raise ValueError("Fixture evidence cannot enter production provenance")
            _fixture_check(item, fixture)
    elif isinstance(value, list):
        for item in value: _fixture_check(item, fixture)


def _budget_snapshot(old, current, *, ppo, curriculum):
    # Same recorded lineage constraints as the frozen receipt, without importing
    # its environment/torch dependencies or inferring any capability verdict.
    for key in ("version", "parent_run", "parent_sampling_budget_sha256", "parent_reserved",
                "parent_baseline", "authorized_total", "child_id", "registry_path"):
        if old.get(key) != current.get(key): raise ValueError("Recorded budget lineage differs")
    if old.get("child_registered") is not True or old.get("read_only_preview") is not False:
        raise ValueError("Recorded budget is an unregistered preview")
    _integer(old.get("revision"), 1, current["revision"])
    children = old.get("children")
    if not isinstance(children, dict) or current["child_id"] not in children:
        raise ValueError("Recorded budget omits its registered child")
    for child_id, child in children.items():
        if (child_id not in current["children"] or child.get("cap") != current["children"][child_id]["cap"]
                or set(child) != {"cap", "reserved"} or set(child["reserved"]) != {"ppo", "curriculum"}):
            raise ValueError("Recorded budget child differs")
        for kind, amount in child["reserved"].items():
            _integer(amount, 0, current["children"][child_id]["reserved"][kind])
        if sum(child["reserved"].values()) > child["cap"]: raise ValueError("Recorded child exceeds cap")
    child = children[current["child_id"]]
    if (old.get("child_reserved") != child["reserved"] or ppo > child["reserved"]["ppo"]
            or curriculum > child["reserved"]["curriculum"]):
        raise ValueError("Recorded transitions exceed their own reservations")
    remaining = old["authorized_total"] - old["parent_baseline"] - sum(sum(c["reserved"].values()) for c in children.values())
    if (remaining < 0 or old.get("global_remaining") != remaining
            or old.get("remaining") != min(remaining, child["cap"] - sum(child["reserved"].values()))):
        raise ValueError("Recorded remaining budget differs")


def _validate(raws, identity, contracts, fixture):
    """Validate raw-byte bindings using original identities, never local paths."""
    protocol, sources, parent_sources = (contracts[k] for k in ("protocol", "sources", "parent_sources"))
    _contracts(protocol, sources, parent_sources, fixture)
    for role, name in raws:
        if not _allowed(role, name, sources): raise ValueError("Unexpected evidence file")
    required = {("foundation", name) for name in FOUNDATION_REQUIRED}
    required |= {("parent", name) for name in PARENT_FILES} | {("registry", "registry.json")}
    required |= {("foundation", "source_bundle/" + name) for name in sources}
    if not required.issubset(raws): raise ValueError("Missing required provenance evidence")
    def get(role, name):
        value = _json(raws[role, name]); _fixture_check(value, fixture); return value
    for role, name in raws:
        if name.startswith("source_bundle/"): continue
        if name.endswith(".json"): get(role, name)
        elif name.endswith(".jsonl"):
            for line in raws[role, name].splitlines():
                if line.strip(): _fixture_check(_json(line), fixture)
    run, actual = get("foundation", "run.json"), get("foundation", "protocol.json")
    parent, parent_protocol = get("parent", "run.json"), get("parent", "protocol.json")
    foundation_root = _original(identity["foundation_root"])
    parent_root = _original(identity["parent_root"])
    if (foundation_root == parent_root or PurePosixPath(parent_root) in PurePosixPath(foundation_root).parents
            or PurePosixPath(foundation_root) in PurePosixPath(parent_root).parents):
        raise ValueError("Original foundation and parent must be independent roots")
    registry_path = str(PurePosixPath(parent_root).parent / ".warehouse_native_revision_budget" / PurePosixPath(parent_root).name / "registry.json")
    scenes = get("foundation", "scenarios.json")
    parent_budget, parent_progress = get("parent", "sampling_budget.json"), get("parent", "progress.json")
    if (actual != protocol or parent.get("version") != V2 or parent.get("sources") != parent_sources
            or parent.get("protocol_sha256") != _digest(parent_protocol)
            or parent.get("scenario_sha256") != _digest(scenes) or scenes != get("parent", "scenarios.json")
            or parent_progress.get("status") != "v2_paused_for_diagnostic_review"
            or parent_budget != {"version": V2, "cap": 500000, "reserved": {"curriculum": 10000, "ppo": 200000}}):
        raise ValueError("Preserved parent or foundation protocol binding differs")
    baselines = {name: _hash(raws["parent", name]) for name in ("validation_reference.json", "validation_random.json")}
    parent_binding = {"parent_run": parent_root, "reserved_before_revision": 210000,
        "global_cap": 500000, "remaining_before_revision": 290000,
        **{key: _hash(raws["parent", name]) for key, name in (
            ("parent_record_sha256", "run.json"), ("parent_protocol_sha256", "protocol.json"),
            ("parent_budget_sha256", "sampling_budget.json"), ("parent_progress_sha256", "progress.json"))},
        "validation_baseline_sha256": baselines}
    child_id = "r1_" + _digest({"output": foundation_root, "protocol": _digest(protocol)})[:24]
    expected_identity = {"foundation_root": foundation_root, "parent_root": parent_root,
        "registry_path": registry_path, "child_id": child_id, "run_record_sha256": _digest(run),
        "run_file_sha256": _hash(raws["foundation", "run.json"]), "protocol_sha256": _digest(protocol),
        "source_sha256": _digest(sources), "parent_source_sha256": _digest(parent_sources)}
    if (identity != expected_identity or run.get("version") != R1 or run.get("parent") != parent_binding
            or run.get("sources") != sources or run.get("protocol_sha256") != _digest(protocol)
            or run.get("child_id") != child_id or run.get("explicit_revision_flag_recorded") is not True
            or run.get("approved_child_cap") != protocol["experiment_revision"]["maximum_remaining_training_related_steps"]
            or run.get("feedback_enabled") is not False or run.get("parent_model_loaded") is not False
            or run.get("device") not in ("cpu", "mps")):
        raise ValueError("Original foundation identity or source binding differs")
    if fixture and run.get("test_fixture") is not True: raise ValueError("Fixture run must be marked")
    for name, sha in sources.items():
        if _hash(raws["foundation", "source_bundle/" + name]) != sha:
            raise ValueError("Frozen source bytes differ from trusted contract")
    if get("foundation", "source_bundle/" + PROTOCOL_SOURCE) != protocol:
        raise ValueError("Archived revision protocol differs")
    if get("foundation", "source_bundle/" + PARENT_PROTOCOL_SOURCE) != parent_protocol:
        raise ValueError("Archived parent protocol differs")
    for name, sha in baselines.items():
        get("parent", name); get("foundation", name)
        if _hash(raws["foundation", name]) != sha: raise ValueError("Parent baseline bytes changed")
    registry = get("registry", "registry.json")
    base = {"version": BUDGET, "parent_run": parent_root,
        "parent_sampling_budget_sha256": _hash(raws["parent", "sampling_budget.json"]),
        "parent_reserved": {"curriculum": 10000, "ppo": 200000}, "parent_baseline": 210000, "authorized_total": 500000}
    if set(registry) != set(base) | {"revision", "children"} or any(registry[k] != v for k, v in base.items()):
        raise ValueError("Registry parent binding differs")
    _integer(registry["revision"], 1)
    children = registry["children"]
    if not isinstance(children, dict) or child_id not in children: raise ValueError("Unregistered foundation child")
    for key, child in children.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", key) or set(child) != {"cap", "reserved"}:
            raise ValueError("Invalid registry child")
        _integer(child["cap"], 1, 290000)
        if set(child["reserved"]) != {"ppo", "curriculum"}: raise ValueError("Invalid reservation kinds")
        for amount in child["reserved"].values(): _integer(amount)
        if sum(child["reserved"].values()) > child["cap"]: raise ValueError("Child cap exceeded")
    child = children[child_id]
    remaining = 290000 - sum(sum(c["reserved"].values()) for c in children.values())
    if remaining < 0 or child["cap"] != run["approved_child_cap"]: raise ValueError("Registry allowance exceeded or cap changed")
    budget = {**registry, "registry_path": registry_path, "child_id": child_id,
        "child_registered": True, "read_only_preview": False, "child_reserved": child["reserved"],
        "global_remaining": remaining, "remaining": min(remaining, child["cap"] - sum(child["reserved"].values()))}
    binding = {"run_record_sha256": _digest(run), "child_id": child_id,
        "parent": parent_binding, "registry_path": registry_path}
    progress = get("foundation", "progress.json")
    if progress.get("status") != "revision_budget_complete_candidate" or progress.get("formal_ready") is not False:
        raise ValueError("Only completed candidate provenance is supported")
    total = _integer(progress["joint_steps"], 1)
    generated = _integer(progress["curriculum_generation_steps"])
    _budget_snapshot(progress["budget"], budget, ppo=total, curriculum=generated)
    if child["reserved"] != {"ppo": total, "curriculum": generated} or budget["remaining"] >= protocol["training"]["environments"]:
        raise ValueError("Completed counters/reservations or endpoint differ")
    bank = get("foundation", "curriculum_bank.json")
    bank_binding = get("foundation", "curriculum_binding.json")
    if (bank_binding.get("run_binding") != binding or bank_binding.get("bank_sha256") != _digest(bank)
            or bank.get("generation_steps") != generated or bank.get("protocol_sha256") != _digest(protocol)
            or bank.get("experiment_revision") != protocol["experiment_revision"]):
        raise ValueError("Curriculum recorded binding differs")
    _budget_snapshot(bank_binding["budget_at_checkpoint"], budget, ppo=0, curriculum=generated)
    logs = [_json(line) for line in raws["foundation", "training.jsonl"].splitlines() if line.strip()]
    by_step = {}; previous = 0
    for row in logs:
        _fixture_check(row, fixture)
        step = _integer(row["joint_steps"], previous + 1, total)
        if row.get("version") != R1 or row.get("curriculum_generation_steps") != generated:
            raise ValueError("Training journal lineage differs")
        _budget_snapshot(row["budget"], budget, ppo=step, curriculum=generated)
        previous = step; by_step[step] = row
    if previous != total: raise ValueError("Training journal does not reach the endpoint")
    interval = _integer(protocol["training"]["checkpoint_interval"], 1)
    due = sorted(set([*range(interval, total + 1, interval), total]))
    for directory, prefix, extension in (("checkpoints", "actor", "npz"), ("checkpoints", "checkpoint", "pt"),
                                          ("validation", "step", "json"), ("diagnostics", "step", "json")):
        expected = {f"{directory}/{prefix}_{step:07d}.{extension}" for step in due}
        actual_set = {name for role, name in raws if role == "foundation" and name.startswith(f"{directory}/{prefix}_")}
        if actual_set != expected: raise ValueError("Registered checkpoint/report inventory differs")
    reviews = {}
    for step in due:
        if step not in by_step: raise ValueError("Diagnostic boundary absent from training journal")
        report = get("foundation", f"validation/step_{step:07d}.json")
        if (report.get("run_binding") != binding or report.get("joint_steps") != step
                or report.get("experiment_revision") != R1
                or report.get("actor_sha256") != _hash(raws["foundation", f"checkpoints/actor_{step:07d}.npz"])):
            raise ValueError("Validation Actor/run binding differs")
        review = get("foundation", f"diagnostics/step_{step:07d}.json")
        cfg = protocol["diagnostic_checkpoints"]; summary = report["summary"]["skilled"]
        numbers = [summary["mean_ai_deliveries"], summary["neural_static_wall_rate"]]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in numbers):
            raise ValueError("Invalid recorded diagnostic summary")
        reasons = []
        if numbers[0] < cfg["mean_ai_delivery_warning"]: reasons.append("low_neural_delivery_contribution")
        if numbers[1] >= cfg["static_wall_rate_warning"]: reasons.append("high_static_wall_command_rate")
        expected = {"joint_steps": step, "source": "fixed_validation_skilled_episodes", "reasons": reasons,
            "pause_required": step >= cfg["stop_for_review_ppo_steps"] and bool(reasons),
            "candidate_gate_unchanged": True, "not_a_checkpoint_selection_metric": True}
        if review != expected: raise ValueError("Diagnostic does not match its recorded summary")
        reviews[step] = review
    acknowledgments = {}
    for role, name in raws:
        if role != "foundation": continue
        if name == "diagnostic_review_acknowledged.json" or name.startswith("diagnostics/acknowledgments/"):
            ack = get(role, name)
            matched = [step for step, review in reviews.items() if _digest(review) == ack.get("review_sha256")]
            if len(matched) != 1: raise ValueError("Acknowledgment has no unique original review")
            step = matched[0]
            if (name != "diagnostic_review_acknowledged.json" and name != f"diagnostics/acknowledgments/step_{step:07d}.json"):
                raise ValueError("Acknowledgment archive name differs from review")
            if reviews[step]["pause_required"] is not True or ack.get("explicit_review_flag") is not True:
                raise ValueError("Acknowledgment does not record an actual pause review")
            _budget_snapshot(ack["budget"], budget, ppo=step, curriculum=generated)
            if ack["budget"]["child_reserved"] != by_step[step]["budget"]["child_reserved"]:
                raise ValueError("Acknowledgment reservation differs from its own boundary")
            if step in acknowledgments and acknowledgments[step] != raws[role, name]:
                raise ValueError("Acknowledgment copies have different original bytes")
            acknowledgments[step] = raws[role, name]
    for step, review in reviews.items():
        if step < total and review["pause_required"] and step not in acknowledgments:
            raise ValueError("Continued training lacks its original historical review acknowledgment")
    required_pauses = [step for step, review in reviews.items() if review["pause_required"]]
    if required_pauses and ("foundation", "diagnostic_review_required.json") not in raws:
        raise ValueError("Recorded diagnostic pause file is missing")
    if ("foundation", "diagnostic_review_required.json") in raws:
        current = get("foundation", "diagnostic_review_required.json")
        if (current != reviews.get(current.get("joint_steps")) or current.get("pause_required") is not True
                or current.get("joint_steps") != max(required_pauses)):
            raise ValueError("Current required review differs from the archived pause")
    best = progress.get("best", {})
    step = best.get("joint_steps")
    if (step not in due or best.get("actor") != f"checkpoints/actor_{step:07d}.npz"
            or best.get("actor_sha256") != _hash(raws["foundation", best["actor"]])):
        raise ValueError("Recorded selected Actor binding differs")
    # Selection optimality, capability values, pickle/NPZ payload and all physical
    # scene/curriculum semantics intentionally remain unverified at this layer.
    return {"original_identity": copy.deepcopy(identity), "run_binding": binding,
        "budget": budget, "recorded_selected_step": step, "completed_ppo_steps": total,
        "curriculum_generation_steps": generated, "provenance_only": True,
        "qualification_not_evaluated": True, "environment_steps": 0, "neural_updates": 0,
        "checkpoint_payload_verified": False, "test_fixture": fixture}


class ProvenanceSnapshot:
    """Detached verified bytes. No resolver returns a mutable filesystem path."""
    def __init__(self, manifest, manifest_sha256, raws, report):
        self._manifest = copy.deepcopy(manifest)
        self.manifest_sha256 = manifest_sha256
        self._raws = MappingProxyType(dict(raws))
        self._report = copy.deepcopy(report)

    @property
    def manifest(self): return copy.deepcopy(self._manifest)

    @property
    def report(self): return copy.deepcopy(self._report)

    def read_bytes(self, role, path):
        if role not in ROLES: raise ValueError("Unknown evidence role")
        return self._raws[role, _relative(path)]

    def read_json(self, role, path): return _json(self.read_bytes(role, path))


def _capture_paths(foundation, parent, registry, sources):
    names = set(FOUNDATION_REQUIRED)
    top = _entries(foundation)
    names.update(name for name in FOUNDATION_OPTIONAL if name in top)
    for folder, pattern in PATTERNS.items():
        # Do not recursively inspect unknown/private directories.
        parts = folder.split("/")
        containing = foundation
        present = True
        for part in parts:
            if part not in _entries(containing): present = False; break
            containing = containing / part
        if present:
            names.update(folder + "/" + name for name in _entries(containing) if pattern.fullmatch(name))
    names.update("source_bundle/" + name for name in sources)
    return {**{("foundation", name): foundation / name for name in names},
        **{("parent", name): parent / name for name in PARENT_FILES}, ("registry", "registry.json"): registry}


def _bundle_files(root):
    """Reject all aliases and unlisted bundle files without following links."""
    found = set()
    def visit(path, relative):
        descriptor = _open_dir(path)
        try:
            for name in os.listdir(descriptor):
                rel = _relative(relative + name)
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode): visit(path / name, rel + "/")
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1: found.add(rel)
                else: raise ValueError("Bundle contains a symbolic/hard link or nonregular file")
        finally:
            os.close(descriptor)
    visit(root, "")
    return found


def read_revision_provenance(bundle, *, expected_manifest_sha256):
    """Require an external trusted digest; package contents are not a trust root."""
    if not isinstance(expected_manifest_sha256, str) or not HEX.fullmatch(expected_manifest_sha256):
        raise ValueError("An independently trusted manifest SHA-256 is required")
    root = _absolute(bundle)
    manifest_bytes = _bytes(root / "manifest.json")
    if _hash(manifest_bytes) != expected_manifest_sha256: raise ValueError("Trusted manifest digest differs")
    manifest = _json(manifest_bytes)
    if (set(manifest) != {"version", "scope", "test_fixture", "original_identity", "contracts", "inventory"}
            or manifest["version"] != VERSION or manifest["scope"] != "provenance_only"
            or set(manifest["contracts"]) != {"protocol", "sources", "parent_sources"}
            or not isinstance(manifest["inventory"], list)):
        raise ValueError("Invalid provenance manifest schema")
    seen = set(); raws = {}; expected_files = {"manifest.json"}
    for row in manifest["inventory"]:
        if set(row) != {"role", "path", "sha256", "size"}: raise ValueError("Invalid inventory entry")
        role, name = row["role"], _relative(row["path"])
        if not _allowed(role, name, manifest["contracts"]["sources"]): raise ValueError("Unlisted evidence role/path")
        key = (role, name)
        if key in seen: raise ValueError("Duplicate evidence path")
        seen.add(key)
        _integer(row["size"])
        if not isinstance(row["sha256"], str) or not HEX.fullmatch(row["sha256"]): raise ValueError("Invalid file digest")
        storage = "files/" + role + "/" + name
        raw = _bytes(root / storage)
        if len(raw) != row["size"] or _hash(raw) != row["sha256"]: raise ValueError("Transported evidence bytes changed")
        raws[key] = raw; expected_files.add(storage)
    if _bundle_files(root) != expected_files: raise ValueError("Bundle file inventory differs")
    report = _validate(raws, manifest["original_identity"], manifest["contracts"], manifest["test_fixture"])
    return ProvenanceSnapshot(manifest, expected_manifest_sha256, raws, report)


def write_revision_provenance(foundation, output, *, expected_protocol, expected_sources,
                              expected_parent_sources, allow_test_fixture=False):
    """Snapshot a completed run under read-only leases, preserving every byte.

    Required expected_* arguments are caller-trusted frozen contracts. Output
    must be new and its parent must exist. Manifest is written before payloads;
    an interrupted copy remains invalid and is never reported as complete.
    Persist the returned manifest_sha256 outside this bundle for later reads.
    """
    foundation, output = _absolute(foundation), _absolute(output)
    contracts = copy.deepcopy({"protocol": expected_protocol, "sources": expected_sources,
                              "parent_sources": expected_parent_sources})
    _contracts(**contracts, fixture=allow_test_fixture)
    first = _bytes(foundation / "run.json")
    original_run = _json(first)
    parent = Path(_original(original_run["parent"]["parent_run"]))
    registry = parent.parent / ".warehouse_native_revision_budget" / parent.name / "registry.json"
    for source in (foundation, parent, registry.parent):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from original provenance roots")
    with ExitStack() as stack:
        leases = []
        for root in (parent, foundation):
            descriptor = _open_file(root / "run.lock")
            stack.callback(os.close, descriptor)
            try: fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc: raise ValueError("Original run is active") from exc
            leases.append((root / "run.lock", descriptor))
        def unchanged_leases():
            for path, held in leases:
                current = _open_file(path)
                try:
                    old_info, new_info = os.fstat(held), os.fstat(current)
                    if (old_info.st_dev, old_info.st_ino) != (new_info.st_dev, new_info.st_ino):
                        raise ValueError("Original run lock was replaced during snapshotting")
                finally: os.close(current)
        unchanged_leases()
        paths = _capture_paths(foundation, parent, registry, contracts["sources"])
        raws = {key: _bytes(path) for key, path in paths.items()}
        if raws["foundation", "run.json"] != first: raise ValueError("Run identity changed before lock")
        identity = {"foundation_root": str(foundation), "parent_root": str(parent),
            "registry_path": str(registry), "child_id": original_run["child_id"],
            "run_record_sha256": _digest(original_run), "run_file_sha256": _hash(first),
            "protocol_sha256": _digest(contracts["protocol"]), "source_sha256": _digest(contracts["sources"]),
            "parent_source_sha256": _digest(contracts["parent_sources"])}
        report = _validate(raws, identity, contracts, allow_test_fixture)
        if (_capture_paths(foundation, parent, registry, contracts["sources"]) != paths
                or any(_bytes(path) != raws[key] for key, path in paths.items())):
            raise ValueError("Original evidence changed while snapshotting")
        unchanged_leases()
        manifest = {"version": VERSION, "scope": "provenance_only", "test_fixture": allow_test_fixture,
            "original_identity": identity, "contracts": contracts,
            "inventory": [{"role": role, "path": name, "sha256": _hash(raws[role, name]), "size": len(raws[role, name])}
                          for role, name in sorted(raws)]}
        manifest_bytes = _canonical(manifest) + b"\n"
        # Parent directory is walked without aliases before exclusive creation.
        parent_fd = _open_dir(output.parent)
        try:
            os.mkdir(output.name, mode=0o700, dir_fd=parent_fd)
            output_fd = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
        finally: os.close(parent_fd)
        stack.callback(os.close, output_fd)
        def write(path, raw):
            # Keep every mkdir/open anchored to the already opened output root.
            # A symlink at any component is rejected before creating children.
            descriptor = os.dup(output_fd)
            try:
                for component in path.relative_to(output).parts[:-1]:
                    try: os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    except FileExistsError: pass
                    following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                    os.close(descriptor); descriptor = following
            except BaseException:
                os.close(descriptor); raise
            try: fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
            finally: os.close(descriptor)
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        write(output / "manifest.json", manifest_bytes)
        for (role, name), raw in sorted(raws.items()): write(output / "files" / role / name, raw)
        result = read_revision_provenance(output, expected_manifest_sha256=_hash(manifest_bytes))
        if result.report != report: raise ValueError("Written provenance verification differs")
        return result
