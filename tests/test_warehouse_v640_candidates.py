"""Candidate retention and fit/qualification separation in alternating pass."""
from types import SimpleNamespace
import numpy as np
from backend.training import warehouse_v640_alternation as s


def test_zero_whole_leaf_score_does_not_discard_fitted_candidate(monkeypatch):
    class Source:
        leaves=[{'tag':'old'}];structure={'tag':'fixed'}
        def validate(self):return self
        def certificates(self):return [{4}]
        def leaf_indices(self,x):return np.zeros(len(x),int)
    class Budget:
        def check(self):pass
        def checkpoint(self,d):pass
    a=dict(x=np.zeros((4,173),np.float32),m=np.ones((4,5),np.float32),p=np.ones((4,5),np.float32)/5,
        sampled=np.zeros(4,int),rows=[dict(episode=str(i),base_seed=i,task=1) for i in range(4)])
    parts={key:a for key in ('fit','calibration','eligibility','validation')}
    score=dict(score=[0,0.,0],children=[dict(reliable_selected={'explanation':False})])
    monkeypatch.setattr(s.language,'protection',lambda p:dict(protected_descendants=[]))
    monkeypatch.setattr(s.fitting,'internal_split',lambda rows:(np.array([1,1,0,0],bool),np.array([0,0,1,1],bool)))
    monkeypatch.setattr(s.choices.metrics,'risk_subsets',lambda rows:[np.ones(4,bool)])
    monkeypatch.setattr(s.choices,'decode',lambda x:None);monkeypatch.setattr(s.choices,'encode',lambda x:(None,[()]))
    monkeypatch.setattr(s.choices,'root_parts',lambda source,root,parts:(parts,{k:np.ones(4,bool) for k in parts}))
    monkeypatch.setattr(s.choices,'option_score',lambda *args:score)
    monkeypatch.setattr(s.fitting,'leaf_option',lambda *args:None)
    monkeypatch.setattr(s,'region',lambda *args:dict(path=[]))
    monkeypatch.setattr(s.fitting,'select_leaf',lambda *args:({'tag':'selected'},dict(selected_original=False)))
    monkeypatch.setattr(s.fitting,'fit_leaf',lambda *args,**kwargs:{'tag':'fitted'})
    monkeypatch.setattr(s.fitting,'calibrate',lambda model,*args,**kwargs:dict(model,calibrated=True))
    out=s.fit_candidates(Source(),parts,parts,Budget());r=out['records'][0]
    assert r['candidate_score']['score']==[0,0.,0] and r['candidate_model']==dict(tag='fitted',calibrated=True)
    assert r['retained_even_if_whole_leaf_zero_score'] and out['outer_qualification_not_consulted']
