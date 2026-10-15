"""Launch the historical single-domain Warehouse formal UI.

This compatibility entrypoint does not enroll the three-domain v6 study.
"""

from __future__ import annotations

import sys

from run import main


if __name__ == "__main__":
    if "--formal-study" not in sys.argv:
        sys.argv.append("--formal-study")
    main()
