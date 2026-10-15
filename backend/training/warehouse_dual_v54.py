"""Compatibility import and CLI for the archived Warehouse v5.4 trainer."""
from importlib import import_module
import sys

_legacy = import_module("backend.training.legacy.warehouse_v5.warehouse_dual_v54")
if __name__ == "__main__":
    _legacy.main()
else:
    sys.modules[__name__] = _legacy
