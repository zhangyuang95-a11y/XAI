"""Extract the r4.4 post-hoc program under the immediate charger rule."""
from __future__ import annotations

from backend.training import warehouse_r42_program_extract as base
from backend.training.warehouse_r44_adaptation import actor_environment


def main(argv=None):
    base.actor_environment = actor_environment
    base.EXTRACT_VERSION = "warehouse-r44-program-extraction.v1"
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
