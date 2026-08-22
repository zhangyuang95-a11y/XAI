"""Launch the formal pilot/confirmatory Warehouse experiment UI."""

from __future__ import annotations

import sys

from run import main


if __name__ == "__main__":
    if "--formal-study" not in sys.argv:
        sys.argv.append("--formal-study")
    main()
