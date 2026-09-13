#!/usr/bin/env python3
"""Build the frozen fit-only RCPD v8 config selection evidence."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.warehouse_r41_diagnostic_rcpd_v8_fit_selector import main


if __name__ == "__main__":
    raise SystemExit(main())
