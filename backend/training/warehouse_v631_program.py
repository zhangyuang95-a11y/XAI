"""Opt-in surrogate envelope for canonical fused guards; NN control unchanged."""
from copy import deepcopy
import numpy as np
from .warehouse_v629_program import Program as Previous,VERSION as PREVIOUS_VERSION
from . import warehouse_v629_guards as old_guards,warehouse_v631_guards as guards
from .warehouse_v622_program import MASK_INDICES

VERSION='warehouse-role-program.v6.31-fused-guard'

class Program(Previous):
    def certificates(self):return guards.leaf_actions(self.structure,self.maximum_leaves)
    def leaf_indices(self,x):
        guards.validate_guard_rows(x)
        return super().leaf_indices(x)
    def distributions(self,x,m):
        rows=guards.validate_guard_rows(x);m=np.asarray(m,np.float32)
        if m.shape!=(len(rows),5) or not np.array_equal(rows[:,MASK_INDICES],m):
            raise ValueError('physical masks and fused guard features differ')
        return super().distributions(rows,m)
    def to_dict(self):
        result=super().to_dict();result['version']=VERSION
        result['guard_evidence']=guards.tree_evidence(self.structure,self.maximum_leaves)
        return result
    @classmethod
    def from_dict(cls,value,actor_role_sha256):
        if value.get('version')!=VERSION or 'guard_evidence' not in value:
            raise ValueError('unknown fused guard program')
        value=deepcopy(value);evidence=value.pop('guard_evidence')
        if evidence!=guards.tree_evidence(value['structure'],value['maximum_leaves']):
            raise ValueError('fused guard evidence does not replay')
        value['version']=PREVIOUS_VERSION
        value['guard_evidence']=old_guards.tree_evidence(value['structure'],value['maximum_leaves'])
        return Previous.from_dict.__func__(cls,value,actor_role_sha256)
