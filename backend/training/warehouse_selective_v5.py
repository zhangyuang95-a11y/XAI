"""Compatibility import for the archived v5 selective program."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module(
    "backend.training.legacy.warehouse_v5.warehouse_selective_v5")
