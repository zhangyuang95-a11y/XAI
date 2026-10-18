"""Development-only frozen-NN adapter; no operation runs on import.

Requires the persistent-journal collector and source-bound shared-BC module.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import inspect


class ProductionRuntime:
    def __init__(self, root, script_path):
        from backend.training import warehouse_v6_runner as runner
        from backend.training import warehouse_v620_bc_capability as contract
        from experiments.warehouse import v620_protocol as protocol

        if "journal_path" not in inspect.signature(runner.collect_authorized_program_rows).parameters:
            raise RuntimeError("pilot comparison requires the reviewed persistent-journal collector")
        self.root = Path(root).resolve()
        self.script_path = Path(script_path)
        self.runner = runner
        self.bc_contract = contract
        self.methods = dict(protocol.METHODS)
        self.physics_sha256 = runner.ACTION_PHYSICS_CONFIG_SHA256

    def source_sha256(self):
        return self.runner.source_closure_sha256()

    def native_cv_proof(self, summary_pointer):
        import json
        bound = self.bc_contract._checked_native_cv(self.root, summary_pointer)
        path = (self.root / summary_pointer["path"]).resolve(strict=True)
        return {"bound_files": list(bound), "summary": json.loads(path.read_bytes())}

    def initial_role_hashes(self, seed):
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return self.runner.role_hashes(self.runner.WarehouseV6Actor())

    def spent_cases(self):
        from experiments.warehouse.v619_protocol import seed_plan
        return deepcopy(seed_plan()["splits"])

    def load_actor(self, artifact):
        from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
        return FrozenWarehouseV6Actor.load(artifact["path"], expected_sha256=artifact["sha256"],
                                          expected_role_sha256=artifact["role_sha256"])

    def actor_identity(self, actor):
        from scripts.run_warehouse_v619_cached_comparison import actor_identity
        return actor_identity(actor)

    def collect(self, actor, case, *, journal_path):
        # Native actions use exactly the declared old sampling mode. The base
        # action argument equals the environment scene seed, with no new RNG.
        return self.runner.collect_authorized_program_rows(actor, [case["scenario_seed"]],
            deterministic=case["sampling_mode"] == "argmax", action_seed_offset=0,
            include_episode_outcomes=True, journal_path=journal_path)

    def write_trace(self, rows, split, seeds, directory, artifact):
        return self.runner._write_final_fit_trace(rows, split=split, seeds=seeds,
            trace_root=directory, actor_sha256=artifact["sha256"])

    def quality_row(self, rows, outcome, actor):
        from scripts.run_warehouse_v619_fixed_validation import quality_row_from_recorded_nn
        return quality_row_from_recorded_nn(rows, outcome, actor)

    def replay(self, artifact, episodes, seeds, *, journal_path):
        from env.warehouse.reset_audit_v614 import capture_stage1_resets, stage1_reset_phase
        from experiments.warehouse.quality_replay import replay_series
        from backend.training.warehouse_v620_bc_pilot_eval import file_sha

        resets = []
        try:
            # The collector operation has already exited its own reset audit.
            with capture_stage1_resets(journal_path) as resets:
                with stage1_reset_phase("quality"):
                    receipt = replay_series(actor_path=artifact["path"], actor_sha256=artifact["sha256"],
                        role_sha256=artifact["role_sha256"], rows=episodes, seeds=seeds)
            return {"receipt": receipt, "reset_audit": {"resets": list(resets),
                "actual_reset_count": len(resets), "journal_path": str(Path(journal_path).resolve()),
                "journal_sha256": file_sha(journal_path)}}
        except BaseException as exc:
            exc.warehouse_reset_audit = {"resets": list(resets), "actual_reset_count": len(resets),
                "journal_path": str(Path(journal_path).resolve()),
                "journal_sha256": file_sha(journal_path) if Path(journal_path).is_file() else None}
            raise

    def validate_replay_claim(self, artifact, episodes, seeds, receipt):
        from experiments.warehouse.quality_replay import claimed_series
        claimed = claimed_series(actor_sha256=artifact["sha256"], role_sha256=artifact["role_sha256"],
                                 rows=episodes, seeds=seeds)
        if claimed != receipt:
            raise RuntimeError("saved replay receipt differs from its original24 trajectory claims")

    def fit_program(self, role, actor, parts, method, arm, target):
        from backend.training.warehouse_v6_program import WarehouseV6Program, fit_program
        from scripts.run_warehouse_v619_cached_comparison import program_metrics
        from scripts.run_warehouse_development_cv import search_budget_audit
        from backend.training.warehouse_v620_bc_pilot_eval import pointer

        program = fit_program(role, actor, *(parts[name] for name in
            ("fit", "calibration", "eligibility", "validation")), method=method)
        budget = search_budget_audit(program.report, arm)
        if not budget["passed"]:
            raise RuntimeError("pilot external tree exceeded its source-locked search budget")
        target.parent.mkdir(parents=True, exist_ok=True)
        program.save(target)
        program = WarehouseV6Program.load(target, expected_sha256=pointer(target)["sha256"],
                                          actor_role_sha256=actor.role_sha256[role])
        metrics = {name: program_metrics(program, actor, rows) for name, rows in parts.items()}
        return {"program": pointer(target), "metrics": metrics["validation"],
                "metrics_by_split": metrics, "search_budget_audit": budget,
                "actor_role_sha256": actor.role_sha256[role], "method": method}

    def validate_program_record(self, artifact, role, fitted, method, arm):
        """Recheck the saved program itself; no fit, rollout or Actor gradients."""
        from backend.training.warehouse_v6_program import WarehouseV6Program
        from scripts.run_warehouse_development_cv import search_budget_audit
        program = WarehouseV6Program.load(fitted["program"]["path"],
            expected_sha256=fitted["program"]["sha256"], actor_role_sha256=artifact["role_sha256"][role])
        budget = search_budget_audit(program.report, arm)
        if (program.role != role or program.report.get("fit_method") != method
                or fitted.get("method") != method
                or fitted.get("actor_role_sha256") != artifact["role_sha256"][role]
                or budget != fitted.get("search_budget_audit") or budget.get("passed") is not True
                or fitted.get("metrics_by_split", {}).get("validation") != fitted.get("metrics")
                or any(program.report["validation"][purpose] != fitted["metrics"][purpose]
                       for purpose in ("feedback", "explanation"))):
            raise RuntimeError("saved pilot program/budget/validation evidence does not recompute")
