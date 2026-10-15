#!/usr/bin/env python3
"""Build and independently reload the selected r4.1 online Secret File."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.warehouse_alignment_r41_online_release import assemble_from_admitted_components


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-ledger", type=Path, required=True)
    parser.add_argument("--dual-evaluation", type=Path, required=True)
    parser.add_argument("--corrected-six-partner-audit-report", type=Path,
                        required=True)
    parser.add_argument("--conflict-manifest", type=Path, required=True)
    parser.add_argument("--conflict-validation", type=Path, required=True)
    parser.add_argument("--task-conflict-graph", type=Path, required=True)
    parser.add_argument("--dynamic-selection-report", type=Path, required=True)
    parser.add_argument("--selected-scenes", type=Path, required=True)
    parser.add_argument("--final-rcpd-report", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--explanation-audit-report", type=Path, required=True)
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--question-bank-report", type=Path, required=True)
    parser.add_argument("--tutorial", type=Path, required=True)
    parser.add_argument("--production-admission", type=Path, required=True)
    parser.add_argument("--expected-production-admission-sha256", required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--output-base64", type=Path)
    args = parser.parse_args(argv)
    result = assemble_from_admitted_components(
        production_admission_path=args.production_admission,
        expected_production_admission_sha256=args.expected_production_admission_sha256,
        components={
            "actor": args.actor,
            "protocol": args.protocol,
            "training_ledger": args.training_ledger,
            "dual_evaluation": args.dual_evaluation,
            "corrected_six_partner_audit_report":
                args.corrected_six_partner_audit_report,
            "conflict_manifest": args.conflict_manifest,
            "conflict_validation": args.conflict_validation,
            "task_conflict_graph": args.task_conflict_graph,
            "dynamic_selection_report": args.dynamic_selection_report,
            "selected_scenes": args.selected_scenes,
            "final_rcpd_report": args.final_rcpd_report,
            # This is the independent post-freeze final RCPD artifact.  It is
            # deliberately not the training ledger's boundary program.
            "final_rcpd_program": args.program,
            "explanation_audit_report": args.explanation_audit_report,
            "question_bank": args.question_bank,
            "question_bank_report": args.question_bank_report,
            "tutorial": args.tutorial,
        },
        output_package=args.output_package, output_base64=args.output_base64,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
