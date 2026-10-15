"""Compatibility import and CLI for archived Warehouse NN pretraining."""
from importlib import import_module
import sys

_legacy = import_module("backend.training.legacy.warehouse_v5.warehouse_pure_nn")
if __name__ == "__main__":
    _legacy.main()
else:
    sys.modules[__name__] = _legacy
