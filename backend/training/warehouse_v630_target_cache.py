"""One in-memory certificate for already replayed immutable NN targets.

This stores no optimizer epoch, program qualification or training authority.
The pool validator alone seals it after a complete scalar-NN/data replay.
Every reuse hashes the complete target arrays and every bound source file.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from experiments.warehouse.v626_protocol import require,pointer,digest

VERSION='warehouse-v630-replayed-target-memory-certificate.v1'
_TOKEN=object()
_ROLES=('robot_1','robot_2')
_SPLITS=('fit','calibration','eligibility','validation')


def parts_sha(parts):
    h=sha256()
    for role in _ROLES:
        h.update(role.encode())
        for split in _SPLITS:
            a=parts[role][split];h.update(split.encode());h.update(digest(a['rows']).encode())
            for key in ('x','m','p','sampled'):
                v=a[key];h.update(key.encode());h.update(v.dtype.str.encode())
                h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


@dataclass(frozen=True,init=False)
class ReplayedTargets:
    target_ref:dict
    payload:dict
    parts:dict
    role_hashes:dict
    asset_refs:tuple
    observed_audit_sha256:str
    sources_sha256:str
    parts_sha256:str
    binding_sha256:str
    _token:object

    def __init__(self,*args,**kwargs):
        raise RuntimeError('target cache can only follow complete target replay')

    def binding(self):
        return dict(version=VERSION,target=self.target_ref,payload=self.payload,
            roles=self.role_hashes,assets=self.asset_refs,observed_audit=self.observed_audit_sha256,
            sources=self.sources_sha256,parts_sha256=self.parts_sha256)

    def verify(self,observed):
        observed.verify()
        require(self._token is _TOKEN and self.observed_audit_sha256==observed.audit_sha256
                and self.sources_sha256==digest(observed.sources)
                and digest(self.binding())==self.binding_sha256,
                'target memory certificate or source binding changed')
        require(parts_sha(self.parts)==self.parts_sha256
                and all(not a[key].flags.writeable for role in _ROLES
                    for a in self.parts[role].values() for key in ('x','m','p','sampled')),
                'replayed target arrays or row provenance changed in memory')
        for ref in self.asset_refs:
            require(pointer(ref['path'])==ref,'replayed target asset changed after validation')
        return self


def _seal_after_replay(*,observed,ref,payload,parts,role_hashes,asset_refs):
    """Internal pool-validator boundary, not a public proof issuer."""
    require(set(parts)==set(role_hashes)==set(_ROLES)
            and all(set(parts[r])==set(_SPLITS) for r in _ROLES),'incomplete replayed target parts')
    for role in _ROLES:
        for a in parts[role].values():
            for key in ('x','m','p','sampled'):a[key].flags.writeable=False
    out=object.__new__(ReplayedTargets)
    values=dict(target_ref=deepcopy(ref),payload=deepcopy(payload),parts=parts,
        role_hashes=deepcopy(role_hashes),asset_refs=tuple(deepcopy(asset_refs)),
        observed_audit_sha256=observed.audit_sha256,sources_sha256=digest(observed.sources),
        parts_sha256=parts_sha(parts),binding_sha256='',_token=_TOKEN)
    for key,value in values.items():object.__setattr__(out,key,value)
    object.__setattr__(out,'binding_sha256',digest(out.binding()))
    return out.verify(observed)
