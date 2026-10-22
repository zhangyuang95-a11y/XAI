"""Synthetic frame arithmetic exercises all four fixed qualification sets.

The fake reader is only a unit fixture; no scientific NN/scene is executed.
The actual legacy metric/gate arithmetic is used unchanged.
"""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.warehouse import v642_fresh_program_metrics as m
from experiments.warehouse.v642_calibration_quality import cases


class FakeProgram:
    def __init__(self,value):
        self.value=deepcopy(value);self.role=value['role'];self.target_actor_role_sha256=value['target_actor_role_sha256']
        self.feedback_leaves=self.explanation_leaves=frozenset({0 if value['qualified_current_route'] else 1})
    @classmethod
    def from_dict(cls,value,expected):
        if value['target_actor_role_sha256']!=expected:raise ValueError('stale final-NN binding')
        return cls(value)
    def validate(self):return self
    def to_dict(self):return deepcopy(self.value)
    def certificates(self):return {0:frozenset(range(5)),1:frozenset(range(5))}
    def distributions(self,x,mask):
        q=np.broadcast_to(np.array([.98,.005,.005,.005,.005],np.float32),(len(x),5)).copy()
        return np.zeros(len(x),np.int64),np.log(q),q


@pytest.fixture
def sample(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'Program',FakeProgram)
    monkeypatch.setattr(m,'forward',lambda p,x,mask:p.distributions(x,mask))
    rows=cases(90100000,9010000000)
    mask=np.ones((2,2,5),np.float32);x=np.zeros((2,2,m.FEATURE_DIM),np.float32);x[...,m.MASK_INDICES]=mask
    p=np.broadcast_to(np.array([.98,.005,.005,.005,.005],np.float32),(2,2,5)).copy()
    a=tmp_path/'synthetic-frames.npz';np.savez_compressed(a,x=x,m=mask,p=p,sampled=np.zeros((2,2),np.int64))
    manifest={'programs':{}};roles={}
    for variant in ('raw','regularized'):
        manifest['programs'][variant]={};roles[variant]={}
        for role in m.ROLES:
            sha=variant+'-'+role;path=tmp_path/(sha+'.json')
            path.write_text(json.dumps(dict(role=role,target_actor_role_sha256=sha,qualified_current_route=True)))
            manifest['programs'][variant][role]=m.pointer(path);roles[variant][role]=sha
    entries=[dict(variant=v,case=c,arrays=m.pointer(a)) for v in ('raw','regularized') for c in rows]
    return manifest,roles,entries,rows


def test_all_four_fixed_programs_score_without_requalification(sample):
    result=m.evaluate(*sample)
    assert result['passed'] and result['qualification_changes']==result['program_fit_calls']==0
    assert not result['model_acceptance_passed'] and not result['pilot_opened']
    for variant in ('raw','regularized'):
        for role in m.ROLES:
            r=result['roles'][variant][role];assert r['gate']['passed'] and r['independent_scenes']==75
            assert r['metrics']['explanation']['coverage']==1. and r['metrics']['feedback']['agreement']==1.


def test_one_roles_missing_coverage_cannot_be_pooled_away(sample):
    manifest,roles,entries,rows=sample;ref=manifest['programs']['regularized']['robot_2']
    p=ref['path'];value=m.bound(ref);value['qualified_current_route']=False
    with open(p,'w') as h:json.dump(value,h)
    manifest['programs']['regularized']['robot_2']=m.pointer(p)
    result=m.evaluate(manifest,roles,entries,rows)
    assert not result['passed'] and not result['roles']['regularized']['robot_2']['gate']['passed']
    assert result['roles']['raw']['robot_2']['gate']['passed']


@pytest.mark.parametrize('change',['missing_scene','duplicate_scene','stale_role','changed_arrays','illegal_NN_mass'])
def test_invalid_fresh_data_or_NN_binding_is_rejected(sample,change):
    manifest,roles,entries,rows=sample
    if change=='missing_scene':entries.pop()
    elif change=='duplicate_scene':entries.append(deepcopy(entries[-1]))
    elif change=='stale_role':roles['regularized']['robot_2']='newer-NN'
    elif change=='changed_arrays':
        with open(entries[0]['arrays']['path'],'ab') as h:h.write(b'changed')
    else:
        path=entries[0]['arrays']['path']
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        a['m'][0,0,1]=0.;a['x'][0,0,m.MASK_INDICES[1]]=0.
        np.savez_compressed(path,**a)
        for entry in entries:entry['arrays']=m.pointer(path)
    with pytest.raises((RuntimeError,ValueError)):m.evaluate(manifest,roles,entries,rows)
