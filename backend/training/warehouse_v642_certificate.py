"""New capacity contract; v626 float32 leaves and exact proof semantics retained.

No global constant or historical reader is patched. Rebinding is allowed only
for byte-identical paths and outputs, followed by exact rational verification.
"""
from copy import deepcopy
from .warehouse_v622_program import normalize_structure, digest
from .warehouse_v626_certificate import binding, verify, certify

VERSION = 'warehouse-explicit-capacity-depth9-or12.v6.42'
CAPACITIES = (64, 128, 256)


def structure_info(structure, maximum_leaves=256, maximum_depth=12):
    if type(maximum_depth) is not int or maximum_depth not in (9,12):
        raise ValueError("unknown v642 depth capacity")
    if type(maximum_leaves) is not int or maximum_leaves not in CAPACITIES:
        raise ValueError('unknown v642 capacity')
    t, info = normalize_structure(structure, enforce_limits=False)
    if (info['leaf_count'] > maximum_leaves or info['depth'] > maximum_depth
            or info['maximum_source_features_per_node'] > 3):
        raise ValueError('v642 depth/leaf/source limit exceeded')
    return t, info


def region(tree, leaf, maximum_leaves=256, maximum_depth=12):
    tree, info = structure_info(tree, maximum_leaves, maximum_depth)
    return dict(path=[dict(branch=p['branch'], weight=tree['split_weights'][p['node']].tolist(),
        bias=float(tree['split_bias'][p['node']])) for p in info['leaf_paths'][leaf]],
        tree_sha256=digest(tree), leaf=leaf)


def rebind(leaf, old_path, new_path, old_legal, new_legal):
    if old_path['path'] != new_path['path'] or set(old_legal) != set(new_legal):
        raise ValueError('proof reuse requires identical numeric path and legal set')
    verify(leaf, old_path, old_legal, leaf['certificate'])
    out = deepcopy(leaf)
    out['certificate']['binding_sha256'] = binding(out, new_path, new_legal, True)
    verify(out, new_path, new_legal, out['certificate'])
    return out
