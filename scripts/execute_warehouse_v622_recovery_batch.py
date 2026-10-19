#!/usr/bin/env python3
"""Persistently execute one complete, source-bound engineering recovery batch.

The scheduler does not retry jobs, splice batches, fit trees, select models, or
collect environment trajectories. Scientific selection stays in the frozen
capacity runner and independent auditor. Run this scheduler in a detached
process so an interrupted chat does not interrupt its scientific workers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=3, choices=(1, 2, 3))
    args = parser.parse_args()
    from experiments.warehouse import v622_recovery_protocol as v

    control = ROOT / v.OUTPUT_ROOT / "recovery-02-execution"
    control.mkdir(parents=True, exist_ok=False)
    lock = (control / "scheduler.lock").open("x")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = (control / "events.jsonl").open("x", buffering=1)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    runner = ROOT / "scripts/run_warehouse_v622_capacity_recovery.py"
    auditor = ROOT / "scripts/audit_warehouse_v622_capacity_recovery.py"

    def event(kind, **fields):
        row = {"event": kind, "at_utc": datetime.now(timezone.utc).isoformat(), **fields}
        journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        journal.flush()
        print(json.dumps(row, sort_keys=True), flush=True)

    def command(label, argv, timeout=900):
        with (control / (label + ".log")).open("x") as log:
            event("command_started", label=label, argv=[str(x) for x in argv])
            result = subprocess.run([str(x) for x in argv], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
        event("command_finished", label=label, returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f"{label} failed with return code {result.returncode}; no retry")

    def stage(name):
        event("stage_started", stage=name)
        command(name + "-prepare", [sys.executable, runner, "--prepare", "--stage", name,
                                   "--expected-source-sha256", v.source_closure_sha256(ROOT)])
        folder = v.output_directory(name)
        plan_path = folder / "plan.json"
        plan = json.loads(plan_path.read_bytes())
        logs = folder / "logs"
        logs.mkdir(exist_ok=False)
        # Fixed structures complete first. Every registered job still runs,
        # regardless of any preceding outcome.
        pending = sorted(plan["jobs"], key=lambda j: (j["arm"] != "B32", j["job_id"]))
        active = {}
        completed = []
        last_heartbeat = time.monotonic()
        while pending or active:
            while pending and len(active) < args.max_workers:
                job = pending.pop(0)
                log = (logs / (job["job_id"] + ".log")).open("x")
                argv = [sys.executable, str(runner), "--plan", str(plan_path),
                        "--job-id", job["job_id"]]
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[job["job_id"]] = (proc, log, time.monotonic())
                event("job_started", stage=name, job_id=job["job_id"], pid=proc.pid, argv=argv)
            for job_id, (proc, log, begin) in list(active.items()):
                rc = proc.poll()
                if rc is None:
                    # The fitter enforces its registered scientific deadline.
                    # This outer timeout catches a hung process; it does not
                    # fabricate a scientific timeout receipt.
                    if time.monotonic() - begin > 6300:
                        event("worker_hard_timeout", stage=name, job_id=job_id, pid=proc.pid)
                        proc.kill()
                        proc.wait()
                        log.close()
                        raise RuntimeError(f"{job_id} exceeded process timeout; no retry")
                    continue
                log.close()
                del active[job_id]
                event("job_finished", stage=name, job_id=job_id, returncode=rc,
                      elapsed_seconds=time.monotonic() - begin)
                if rc:
                    raise RuntimeError(f"{job_id} failed with return code {rc}; no retry")
                if not (folder / "jobs" / job_id / "completed.json").is_file():
                    raise RuntimeError(f"{job_id} exited without a completion receipt")
                completed.append(job_id)
            if time.monotonic() - last_heartbeat >= 45:
                event("heartbeat", stage=name, completed=len(completed), pending=len(pending),
                      active={j: p.pid for j, (p, _, _) in active.items()})
                last_heartbeat = time.monotonic()
            if active:
                time.sleep(5)
        command(name + "-summarize", [sys.executable, runner, "--plan", plan_path, "--summarize"])
        command(name + "-audit", [sys.executable, auditor, "--plan", plan_path], timeout=1800)
        summary = json.loads((folder / "summary.json").read_bytes())
        audit = json.loads((folder / "independent-audit.json").read_bytes())
        if audit.get("passed") is not True:
            raise RuntimeError("complete stage lacks a passing independent arithmetic audit")
        event("stage_finished", stage=name, selected_arm=summary["selected_arm"],
              next_stage=summary["next_stage"], independent_audit_passed=True)
        return summary

    try:
        event("scheduler_started", pid=os.getpid(), max_workers=args.max_workers,
              source_closure_sha256=v.source_closure_sha256(ROOT), scientific_retry=False)
        primary = stage("primary")
        last = stage("fallback") if primary["next_stage"] == "fallback" else primary
        outcome = {"status": "completed", "next_stage": last["next_stage"],
                   "selected_arm": last["selected_arm"], "environment_resets": 0,
                   "actor_updates": 0, "PPO_joint_steps": 0, "pilot_opened": False}
        with (control / "completed.json").open("x") as handle:
            json.dump(outcome, handle, indent=2, sort_keys=True)
            handle.write("\n")
        event("scheduler_finished", **outcome)
    except BaseException as exc:
        event("scheduler_failed", error_type=type(exc).__name__, error=str(exc))
        with (control / "failed.json").open("x") as handle:
            json.dump({"error_type": type(exc).__name__, "error": str(exc),
                       "automatic_retry": False, "pilot_opened": False}, handle, indent=2)
            handle.write("\n")
        raise
    finally:
        journal.close()
        lock.close()


if __name__ == "__main__":
    main()
