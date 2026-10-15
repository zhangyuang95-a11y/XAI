#!/usr/bin/env python3
"""Build and independently reload the admitted r4 Render Secret File."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.warehouse_alignment_r4_online_release import build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--selected-scenes", type=Path, required=True)
    parser.add_argument("--question-bank", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True,
                        help="Exact registered manifest used by the paired and explanation audits")
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--output-base64", type=Path)
    args = parser.parse_args(argv)
    result = build(
        actor_path=args.actor, protocol_path=args.protocol,
        program_path=args.program, selected_scenes_path=args.selected_scenes,
        question_bank_path=args.question_bank, scenarios_path=args.scenarios,
        admission_path=args.admission,
        expected_admission_sha256=args.expected_admission_sha256,
        output_package=args.output_package, output_base64=args.output_base64,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
