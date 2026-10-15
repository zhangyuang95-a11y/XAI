"""Compatibility import and CLI for the archived v5.4 variant runner."""
from importlib import import_module
import sys

_legacy = import_module(
    "backend.training.legacy.warehouse_v5.run_warehouse_structured_v54_variant")
if __name__ == "__main__":
    _legacy.main()
else:
    sys.modules[__name__] = _legacy
