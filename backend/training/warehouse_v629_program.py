"""v6.29 surrogate envelope; execution still uses the complete frozen NN.

The normalized sparse leaves and exact numeric Farkas checker are inherited
unchanged. Only the explicitly versioned logical mask proof gains canonical
two/three-bit conjunctions. All protected v6.26 payloads remain byte-identical.
"""
from copy import deepcopy
from .warehouse_v628_program import (Program as Previous, from_parent as previous_from_parent,
    qualify, measure, complete_program, protected_indices, path_key)
from .warehouse_v629_guards import leaf_actions, tree_evidence

VERSION = 'warehouse-role-program.v6.29-cumulative-guard'


class Program(Previous):
    def certificates(self):
        return leaf_actions(self.structure, self.maximum_leaves)

    def to_dict(self):
        result = super().to_dict()
        result['version'] = VERSION
        result['guard_evidence'] = tree_evidence(self.structure, self.maximum_leaves)
        return result

    @classmethod
    def from_dict(cls, value, actor_role_sha256):
        if value.get('version') != VERSION or 'guard_evidence' not in value:
            raise ValueError('unknown v629 physical guard program')
        value = deepcopy(value)
        evidence = value.pop('guard_evidence')
        if evidence != tree_evidence(value['structure'], value['maximum_leaves']):
            raise ValueError('physical guard evidence does not replay')
        # Previous's parser checks its entire binding and rejects unknown keys;
        # call its classmethod implementation with cls so virtual proofs apply.
        from .warehouse_v628_program import VERSION as previous_version
        value['version'] = previous_version
        return Previous.from_dict.__func__(cls, value, actor_role_sha256)


def from_parent(parent, reference, maximum=64):
    old = previous_from_parent(parent, reference, maximum)
    result = Program(**old.__dict__)
    return result.validate()
