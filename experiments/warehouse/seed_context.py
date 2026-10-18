"""Source-backed annotations for numeric matches, without executing environments.

An annotation never grants seed authority. In particular a key called rng does
not establish that its descendants are generator state: the complete Python
Random state structure and the Warehouse snapshot/getstate producer are checked.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess

from .seed_metadata_audit import (MAX_BYTES, OPAQUE_WEIGHT_DIRECTORIES,
    OPAQUE_WEIGHT_SUFFIXES, WEIGHT_METADATA, WEIGHT_NAME, INTERNAL_STATE_KEYS,
    digest, file_sha, write_new)

SOURCE_LOCATIONS = {
    "snapshot": ("env/warehouse/autonomy_runtime.py", ("\"rng\": environment.get_rng_state()",)),
    "environment_random": ("env/warehouse/environment.py", ("random.Random(self.config.seed)", "return self._rng.getstate()")),
    "checkpoint_steps": ("backend/training/warehouse_native_cycle_budget.py", ("operation[\"old_head\"][\"step\"] + actual_steps",)),
    "optimizer_counts": ("backend/training/warehouse_nn/trainer.py", ("self.actor_updates += 1", "self.critic_updates += 1")),
    "row_count_v4": ("backend/training/warehouse_r41_diagnostic_rcpd_v4.py", ("\"train_anchors_affected_by_duplicate_row_removal\": len(affected_anchors)",)),
    "row_count_v5": ("backend/training/warehouse_r41_diagnostic_rcpd_v5.py", ("\"train_anchors_affected_by_duplicate_row_removal\": len(affected_anchors)",)),
    "row_count_v6": ("backend/training/warehouse_r41_diagnostic_rcpd_v6.py", ("\"train_anchors_affected_by_duplicate_row_removal\": len(affected_anchors)",)),
}


def source_evidence(root):
    result = {}
    for name, (relative, needles) in SOURCE_LOCATIONS.items():
        path = Path(root) / relative
        if not path.is_file():
            continue
        raw = path.read_bytes()
        text = raw.decode()
        if not all(needle in text for needle in needles):
            continue
        result[name] = {"path": str(path.resolve()), "sha256": file_sha(path),
            "lines": sorted({text[:text.index(needle)].count("\n") + 1 for needle in needles})}
    return result


def get(value, path):
    for key in path:
        if type(key) is int and isinstance(value, dict) and set(value) == {"$tuple"}:
            value = value["$tuple"]
        value = value[key]
    return value


def python_mt_shape(value):
    if isinstance(value, dict) and set(value) == {"$tuple"}:
        value = value["$tuple"]
    if not isinstance(value, (list, tuple)) or len(value) != 3 or value[0] != 3:
        return None
    state = value[1]
    if isinstance(state, dict) and set(state) == {"$tuple"}:
        state = state["$tuple"]
    if (not isinstance(state, (list, tuple)) or len(state) != 625
            or any(type(v) is not int or not 0 <= v < 2**32 for v in state[:624])
            or type(state[624]) is not int or not 0 <= state[624] <= 624
            or (value[2] is not None and type(value[2]) not in (int, float))):
        return None
    return {"version": 3, "internal_tuple_length": 625, "mt_word_count": 624,
        "index": state[624], "gauss_cache_kind": "none" if value[2] is None else type(value[2]).__name__}


def context_prefix(path):
    for index, key in enumerate(path):
        if key == "rng" or key in INTERNAL_STATE_KEYS:
            return path[:index + 1]
    if path[-1:] == ["step"] and "operations" in path:
        return path[:path.index("operations") + 2]
    return path[:-1]


def classify_context(path, scalar, lookup, evidence):
    state_keys = [i for i, key in enumerate(path) if key == "rng" or key in INTERNAL_STATE_KEYS]
    if state_keys:
        index = state_keys[0]
        tail = [x for x in path[index + 1:] if x != "$tuple"]
        if len(tail) == 2 and tail[0] == 1 and type(tail[1]) is int and 0 <= tail[1] < 624:
            shape = python_mt_shape(lookup(path[:index + 1]))
            if shape and lookup(path) == scalar and {"snapshot", "environment_random"} <= evidence.keys():
                return "confirmed_python_mt19937_word_not_initialization", {
                    "rng_path": path[:index + 1], **shape, "source_keys": ["snapshot", "environment_random"]}
    if path[-1:] == ["step"] and "operations" in path and "checkpoint_steps" in evidence:
        index = path.index("operations")
        operation = lookup(path[:index + 2])
        tail = path[index + 2:]
        if tail in (["old_head", "step"], ["completion", "checkpoint", "step"]):
            before = operation.get("old_head")
            completion = operation.get("completion") or {}
            after, amount = completion.get("checkpoint"), completion.get("actual_steps")
            if (isinstance(before, dict) and isinstance(after, dict)
                    and set(before) == set(after) == {"path", "sha256", "step"}
                    and type(amount) is int and type(before["step"]) is type(after["step"]) is int
                    and after["step"] == before["step"] + amount
                    and operation.get("status") == "acknowledged"
                    and type(operation.get("request", {}).get("steps")) is int
                    and lookup(path) == scalar):
                return "confirmed_checkpoint_interaction_counter_not_seed", {
                    "old_step": before["step"], "actual_steps": amount, "new_step": after["step"],
                    "request_kind": operation["request"].get("kind"), "source_keys": ["checkpoint_steps"]}
    if path[-1:] in (["actor_updates"], ["critic_updates"]) and "optimizer_counts" in evidence:
        row = lookup(path[:-1])
        if isinstance(row, dict) and {"actor_updates", "critic_updates", "joint_steps"} <= row.keys() and lookup(path) == scalar:
            return "confirmed_optimizer_update_counter_not_seed", {
                "counter": path[-1], "joint_steps": row["joint_steps"], "source_keys": ["optimizer_counts"]}
    if path == ["row_accounting", "train_anchors_affected_by_duplicate_row_removal"]:
        keys = [key for key in ("row_count_v4", "row_count_v5", "row_count_v6") if key in evidence]
        if keys and lookup(path) == scalar:
            return "confirmed_affected_anchor_count_not_seed", {"counter": path[-1], "source_keys": keys}
    return "requires_context_review", {}


def _put(root, keys, value):
    key = keys[0]
    if type(key) is int:
        if not 0 <= key < 100000:
            raise ValueError("unbounded context index")
        while len(root) <= key:
            root.append(None)
    if len(keys) == 1:
        root[key] = value
        return
    if (isinstance(root, list) and root[key] is None) or (isinstance(root, dict) and key not in root):
        root[key] = [] if type(keys[1]) is int else {}
    _put(root[key], keys[1:], value)


def stream_context(path, prefixes, error_path, *, max_events=100000, max_bytes=32*1024*1024):
    """Read complete JSON with jq streaming; materialize only bounded parents."""
    prefixes = [list(x) for x in dict.fromkeys(tuple(p) for p in prefixes)]
    if not prefixes or len(prefixes) > 1024:
        raise ValueError("invalid number of context parents")
    is_jsonl = Path(path).suffix == ".jsonl"
    query = 'select(length == 2)'
    if is_jsonl:
        query += ' | [([input_line_number - 1] + .[0]), .[1]]'
    query += ' | . as $e | $prefixes | to_entries[] | .key as $i | .value as $p | select($e[0][0:($p|length)] == $p) | [$i, $e[0][($p|length):], $e[1]]'
    values = {}
    event_count = byte_count = 0
    with Path(error_path).open("x") as errors:
        process = subprocess.Popen(["/usr/bin/jq", "--stream", "-c", "--argjson", "prefixes",
            json.dumps(prefixes, separators=(",", ":")), query, str(path)],
            stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                event_count += 1
                byte_count += len(line.encode())
                if event_count > max_events or byte_count > max_bytes:
                    raise ValueError("complete context projection exceeded declared materialization bound")
                index, tail, value = json.loads(line)
                if not tail:
                    values[index] = value
                else:
                    values.setdefault(index, [] if type(tail[0]) is int else {})
                    _put(values[index], tail, value)
            returncode = process.wait()
            if returncode:
                raise ValueError("complete context JSON parse failed")
        except BaseException:
            process.kill()
            process.wait()
            raise
    if set(values) != set(range(len(prefixes))):
        raise ValueError("requested context parent absent")
    def lookup(keys):
        for index, prefix in enumerate(prefixes):
            if list(keys[:len(prefix)]) == prefix:
                return get(values[index], keys[len(prefix):])
        raise KeyError(tuple(keys))
    return lookup, {"parser": "jq_complete_bounded_parent_projection", "events": event_count,
                    "materialized_bytes": byte_count, "parent_count": len(prefixes)}


def annotations(inventory, *, source_root, output_directory):
    """Annotate possible MT words and initialization numeric coincidences."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    evidence = source_evidence(source_root)
    records, cache = [], {}
    for index, row in enumerate(inventory["inventory"]):
        selected = [m for m in row.get("detail", {}).get("candidate_matches", [])
                    if any(key == "rng" or key in INTERNAL_STATE_KEYS for key in m["path"])
                    or any("init" in f for f in m.get("families", []))]
        if not selected:
            continue
        path, expected = Path(row["path"]), row.get("sha256")
        if (any(p.lower() in OPAQUE_WEIGHT_DIRECTORIES for p in path.parts)
                or path.suffix.lower() in OPAQUE_WEIGHT_SUFFIXES
                or path.suffix.lower() == ".json" and WEIGHT_NAME.search(path.name) and not WEIGHT_METADATA.search(path.name)):
            raise RuntimeError("refusing model content in a contextual metadata inventory")
        if expected is None or file_sha(path) != expected:
            raise RuntimeError("metadata identity changed before context annotation")
        parser = {"parser": "bounded_json_complete"}
        try:
            if row["bytes"] <= MAX_BYTES and path.suffix in (".json", ".jsonl"):
                if expected not in cache:
                    raw = path.read_bytes()
                    cache[expected] = json.loads(raw) if path.suffix == ".json" else [json.loads(line) for line in raw.splitlines() if line.strip()]
                value = cache[expected]
                lookup = lambda keys: get(value, keys)
            else:
                lookup, parser = stream_context(path, [context_prefix(m["path"]) for m in selected],
                                                 output / f"context-{index:05d}.stderr")
            for match in selected:
                try:
                    status, proof = classify_context(match["path"], match["value"], lookup, evidence)
                except (KeyError, IndexError, TypeError, ValueError):
                    status, proof = "requires_context_review", {}
                records.append({"file": str(path), "file_sha256": expected, **match,
                    "context_status": status, "context_proof": proof, **parser})
        except (ValueError, UnicodeError) as error:
            records.extend({"file": str(path), "file_sha256": expected, **m,
                "context_status": "context_parse_unresolved", "error": str(error)} for m in selected)
        if file_sha(path) != expected:
            raise RuntimeError("metadata changed during context annotation")
    for item in evidence.values():
        if file_sha(item["path"]) != item["sha256"]:
            raise RuntimeError("producer evidence source changed during context annotation")
    result = {"schema": "warehouse-seed-contextual-annotations.v1", "preliminary_only": True,
        "input_inventory_digest": digest(inventory), "context_producer_sha256": file_sha(__file__),
        "authorization_granted": False, "environment_resets": 0, "fit_calls": 0,
        "model_weight_content_reads": 0, "source_evidence": evidence, "records": records,
        "classification_counts": dict(Counter(row["context_status"] for row in records)),
        "limitations": ["Annotations do not modify the raw inventory.",
            "Only structurally verified state words and source-backed counters are excluded.",
            "All remaining matches and reservations require exact purpose review.",
            "No global freshness or execution authority is inferred."]}
    write_new(output / "annotations.json", result)
    return result
