"""Independent PPO-first warehouse training stack.

This package is deliberately separate from the released r4.x runtime.  Importing
it never starts training and none of its classes are used by the live service.
"""

VERSION = "warehouse-nn-ppo-rcpd.v7-safety-targeted-final"

__all__ = ["VERSION"]
