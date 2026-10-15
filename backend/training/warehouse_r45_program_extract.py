"""Extract the r4.5 post-hoc program under unified energy-cycle observations."""
from __future__ import annotations

from backend.training import warehouse_r42_program_extract as base
from backend.training.warehouse_r45_adaptation import actor_environment


def main(argv=None):
    base.actor_environment = actor_environment
    base.EXTRACT_VERSION = "warehouse-r45-program-extraction.v1"
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
