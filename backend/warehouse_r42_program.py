"""Pure-NumPy reader for the r4.2 post-hoc decision-tree program."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping

import numpy as np


# Keep the serving reader dependency-light.  This order is the immutable
# five-action environment contract and avoids importing offline training code.
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")


VERSION = "warehouse-r42-decision-program.v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return sha256(canonical(value)).hexdigest()


class R42DecisionProgram:
    """Immutable tree approximation used for evidence, never action control."""

    def __init__(self, payload: Mapping[str, Any]):
        value = deepcopy(dict(payload))
        claimed = value.pop("content_sha256", None)
        if claimed != digest(value) or value.get("version") != VERSION:
            raise ValueError("r4.2 program content differs")
        tree = value.get("tree")
        audit = value.get("audit")
        if (not isinstance(tree, dict) or not isinstance(audit, dict)
                or value.get("actions") != list(ACTIONS)
                or any(float(audit.get(key, -1)) < threshold for key, threshold in (
                    ("overall_fidelity", .90),
                    ("non_wait_fidelity", .90),
                    ("critical_fidelity", .85),
                    ("effective_intervention_direction_accuracy", .85),
                ))):
            raise ValueError("r4.2 program audit did not pass")
        fields = ("children_left", "children_right", "feature", "threshold", "value")
        if any(field not in tree for field in fields):
            raise ValueError("r4.2 program tree is incomplete")
        size = len(tree["feature"])
        if size <= 1 or any(len(tree[field]) != size for field in fields):
            raise ValueError("r4.2 program tree arrays differ")
        self._payload = deepcopy(dict(payload))
        self._tree = tree
        self.actor_sha256 = str(value["actor_sha256"])
        self.audit = deepcopy(audit)
        self.signature = str(claimed)

    def action(self, observation) -> str:
        vector = np.asarray(observation, dtype=np.float32).reshape(-1)
        node = 0
        while int(self._tree["children_left"][node]) != -1:
            feature = int(self._tree["feature"][node])
            node = (int(self._tree["children_left"][node])
                    if float(vector[feature]) <= float(self._tree["threshold"][node])
                    else int(self._tree["children_right"][node]))
        values = np.asarray(self._tree["value"][node], dtype=np.float64)
        return ACTIONS[int(values.argmax())]

    @property
    def payload(self):
        return deepcopy(self._payload)


__all__ = ["VERSION", "R42DecisionProgram", "canonical", "digest"]
