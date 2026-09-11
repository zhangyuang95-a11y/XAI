#!/usr/bin/env python3
"""Build the admitted r4.1 diagnostic ZIP and Base64 Secret File."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_diagnostic_admission as admission
from ui import warehouse_alignment_r41_diagnostic_release as release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--expected-admission-sha256", required=True)
    for name in admission.ARTIFACT_NAMES:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-package", type=Path, required=True)
    parser.add_argument("--output-base64", type=Path, required=True)
    args = parser.parse_args(argv)
    components = {
        name: getattr(args, name) for name in admission.ARTIFACT_NAMES
    }
    result = release.assemble_from_admitted_components(
        diagnostic_admission_path=args.admission,
        expected_diagnostic_admission_sha256=args.expected_admission_sha256,
        components=components,
        output_package=args.output_package,
        output_base64=args.output_base64,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
