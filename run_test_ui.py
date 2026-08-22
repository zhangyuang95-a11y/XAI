"""Launch the Warehouse UI with development-only condition controls."""

from __future__ import annotations

import sys

from run import main


if __name__ == "__main__":
    if "--test-condition-selector" not in sys.argv:
        sys.argv.append("--test-condition-selector")
    main()
