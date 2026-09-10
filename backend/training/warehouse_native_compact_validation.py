"""Durable fixed-episode I/O for new experiment adapters.

No admission contract or experiment identity is invented here. The calling
adapter validates its own Actor/protocol and supplies ordered contexts. Uses
the frozen observed public-physics episode, including original score/reward.
"""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

from backend.training.warehouse_native_common import canonical
from backend.training import warehouse_native_public_feedback_evaluation as physical
from backend.training.warehouse_native_partner_mix_evaluation import _locked, _read_entry, _same, _put, _file
from env.warehouse.layouts import get_map_layout


def execute_episodes(actor, scenes, *, config, contexts, identity, output, step_budget,
                     dynamics, validate_unchanged, before_episode=None, on_episode=None,
                     confirmed_operation_ids=None):
    """Reservations are permanent; complete unacknowledged rows cannot resume."""
    output = Path(output).expanduser().absolute()
    if output.is_symlink(): raise ValueError("Evaluation output cannot be a symlink")
    output = output.resolve()
    with _locked(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if not _same(manifest.get("identity"), identity):
                raise ValueError("Existing evaluation bindings or reserved budget differ")
        else:
            if set(p.name for p in output.iterdir()) != {".lock"}:
                raise ValueError("Existing unbound evaluation output cannot be adopted")
            manifest = {"version": identity["version"], "identity": identity, "episodes": [], "reserved_environment_steps": 0,
                "actual_environment_steps": 0, "status": "running"}
            _put(manifest_path, manifest)
        entries = manifest.get("episodes")
        if not isinstance(entries, list) or len(entries) > len(contexts):
            raise ValueError("Invalid saved episode matrix")
        if manifest.get("reserved_environment_steps") != len(entries) * config.horizon:
            raise ValueError("Reserved step accounting differs")
        rows = []
        for entry, context in zip(entries, contexts):
            rows.append(_read_entry(output, entry, context))
            if confirmed_operation_ids is None or context["operation_id"] not in confirmed_operation_ids:
                raise ValueError("Saved complete episode requires external ledger acknowledgment before continuing")
        if manifest.get("actual_environment_steps") != sum(r["steps"] for r in rows):
            raise ValueError("Confirmed actual step accounting differs")
        for context in contexts[len(rows):]:
            if manifest["reserved_environment_steps"] + config.horizon > step_budget:
                break
            validate_unchanged()
            entry = {"context": context, "status": "reserved", "actual_steps": None}
            manifest["episodes"].append(entry)
            manifest["reserved_environment_steps"] += config.horizon
            _put(manifest_path, manifest)
            admitted = 0; completed = 0; full = [0, 0]; charger_full = [0, 0]
            double = joint_wait = 0; last_after = None
            name = f"{context['episode_index']:04d}_{context['partner']}_{context['scenario_id']}"
            trace_path = output / (name + ".jsonl.gz")
            temporary = output / (name + ".jsonl.gz.incomplete")
            row_path = output / (name + ".json")
            try:
                physical._callback(before_episode, context)
                with temporary.open("xb") as raw:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as compressed:
                        def before_step(value):
                            nonlocal admitted
                            if admitted >= config.horizon: raise ValueError("Episode would exceed its permanent reservation")
                            admitted += 1
                        def on_step(record):
                            nonlocal completed, double, joint_wait, last_after
                            completed += 1
                            if record["submitted_actions"]["robot_2"] != record["policy_actions"]["robot_2"]:
                                raise ValueError("The NN command was overwritten")
                            agents = record["before"]["agents"]
                            charger = tuple(get_map_layout(config.map_layout_id).charger_position)
                            for i, key in enumerate(("robot_1", "robot_2")):
                                is_full_wait = (agents[i]["active"] and agents[i]["battery"] >= 100. - 1e-8
                                                and record["submitted_actions"][key] == "WAIT")
                                full[i] += int(is_full_wait)
                                charger_full[i] += int(is_full_wait and tuple(agents[i]["position"]) == charger)
                            both = all(record["submitted_actions"][key] == "WAIT" for key in ("robot_1", "robot_2"))
                            joint_wait += int(both); double += int(both and all(a["active"] for a in agents))
                            last_after = record["after"]
                            compressed.write(canonical(record).encode() + b"\n")
                        row = physical._episode(actor, scenes[context["scenario_index"]], context["partner"],
                            "observed", dynamics, config, context, before_step, on_step, False)
                    raw.flush(); os.fsync(raw.fileno())
                if admitted != completed or row["steps"] != completed:
                    raise ValueError("Physical steps and written trace rows differ")
                os.replace(temporary, trace_path)
                descriptor = os.open(output, os.O_RDONLY)
                try: os.fsync(descriptor)
                finally: os.close(descriptor)
                row.update(full_battery_waits_by_role=full, full_battery_charger_waits_by_role=charger_full,
                    double_wait_steps=double, joint_submitted_wait_steps=joint_wait,
                    active_end_by_role=[bool(a["active"]) for a in last_after["agents"]],
                    wait_metric_definition="Full battery>=100 before an active role submits WAIT; double_wait requires both roles active.")
                row_item = _put(row_path, row)
                committed = deepcopy(manifest)
                committed["episodes"][-1].update(status="completed", actual_steps=completed,
                    row=row_item, trace=_file(trace_path))
                committed["actual_environment_steps"] += completed
                _put(manifest_path, committed)
                manifest = committed
                entry = manifest["episodes"][-1]
                rows.append(row)
                physical._callback(on_episode, {**context, "actual_steps": completed,
                    "row_path": str(row_path), "row_sha256": row_item["sha256"],
                    "trace_path": str(trace_path), "trace_sha256": entry["trace"]["sha256"]})
            except BaseException as error:
                # A completed row remains durable even if the caller's ack
                # fails. It cannot be skipped later without an external ack.
                if entry["status"] != "completed":
                    entry.update(status="incomplete", actual_steps=completed,
                        admitted_step_calls=admitted, completed_trace_rows=completed)
                    manifest["actual_environment_steps"] += completed
                manifest.update(status="interrupted", error=f"{type(error).__name__}: {error}")
                _put(manifest_path, manifest)
                raise
        manifest["status"] = "completed" if len(rows) == len(contexts) else "budget_exhausted"
        _put(manifest_path, manifest)
        return rows, manifest
