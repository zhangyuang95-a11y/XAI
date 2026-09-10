"""Unmasked tree distributions with the scalar program's predicate semantics.

This helper is not a controller or a release gate. An owning training/extraction
revision must explicitly adopt and bind it; frozen workflows remain unchanged.
"""
import numpy as np

from core.program import ExecutableProgram
from env.warehouse_native.policy import ACTIONS

VERSION = 'warehouse-native-float64-program-predicates.v1'


def predict(program, observations, feature_names):
    if (type(program) is not ExecutableProgram or tuple(program.feature_names) != tuple(feature_names)
            or tuple(program.action_names) != ACTIONS or program.metadata.get('action_legality_features')
            or program.metadata.get('action_constraint_reason_features')):
        raise ValueError('Exact feature order, five actions and an unmasked program are required')
    values = np.asarray(observations)
    if (values.dtype != np.float32 or values.ndim < 1 or values.shape[-1] != len(feature_names)
            or not np.isfinite(values).all()):
        raise ValueError('Actual finite float32 Actor observations are required')
    flat = values.reshape(-1, len(feature_names))
    result = np.empty((len(flat), len(ACTIONS)), np.float32)
    columns = {name: i for i, name in enumerate(feature_names)}
    stack = [(program.root, np.arange(len(flat)))]
    while stack:
        node, rows = stack.pop()
        if not len(rows): continue
        if node.is_leaf:
            p = np.asarray(node.probabilities, np.float64)
            if p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                raise ValueError('Invalid program leaf distribution')
            result[rows] = p / p.sum()
        else:
            if (node.feature not in columns or node.threshold is None or not np.isfinite(node.threshold)
                    or node.left is None or node.right is None):
                raise ValueError('Invalid program predicate')
            # NumPy's float32 array/scalar comparison can round a midpoint
            # threshold onto a row. Scalar Program.execute compares Python
            # floats; promote only the selected column to preserve that test.
            go_left = flat[rows, columns[node.feature]].astype(np.float64) <= float(node.threshold)
            stack.extend(((node.left, rows[go_left]), (node.right, rows[~go_left])))
    return result.reshape(*values.shape[:-1], len(ACTIONS))
