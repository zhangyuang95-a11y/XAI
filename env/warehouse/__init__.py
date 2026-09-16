"""Strict multi-agent Warehouse Robot environment and MAPPO policy.

Keep the package initializer lightweight.  Runtime modules import pure domain
and navigation helpers through ``env.warehouse.*``; eagerly importing the full
training environment here also imports the offline scikit-learn RCPD fitter.
The participant service does not need that dependency.  Public attributes are
loaded lazily so existing ``from env.warehouse import ...`` callers retain the
same API.
"""

from typing import Any


__all__ = ["ACTIONS", "WarehouseConfig", "WarehouseMultiAgentEnv"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from .environment import ACTIONS, WarehouseConfig, WarehouseMultiAgentEnv

    values = {
        "ACTIONS": ACTIONS,
        "WarehouseConfig": WarehouseConfig,
        "WarehouseMultiAgentEnv": WarehouseMultiAgentEnv,
    }
    return values[name]
