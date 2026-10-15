"""Compatibility import and CLI for the archived v5.4 development gate."""
from importlib import import_module
import sys

_legacy = import_module("backend.training.legacy.warehouse_v5.warehouse_v54_gate")
if __name__ == "__main__":
    _legacy.main()
else:
    sys.modules[__name__] = _legacy
