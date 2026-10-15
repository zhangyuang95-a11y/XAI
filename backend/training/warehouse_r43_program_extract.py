"""Extract the r4.3 post-hoc program under the deployed 3% dynamics."""
from __future__ import annotations

from backend.training import warehouse_r42_program_extract as base
from backend.training.warehouse_r43_adaptation import actor_environment


def main(argv=None):
    base.actor_environment = actor_environment
    base.EXTRACT_VERSION = "warehouse-r43-program-extraction.v2"
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
