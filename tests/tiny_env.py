"""Test-only simulator; not shipped as an application environment."""
from copy import deepcopy
import numpy as np
from core.interfaces import EnvSpec, Frame, Transition

class TinyEnv:
    def __init__(self, agents=2, external=False, snapshots=True):
        self.spec=EnvSpec(2,2,('choice_a','choice_b'),('signal','context'),('signal','context'),('signal','context'),agents)
        self.rng=np.random.default_rng(1);self.t=0;self.external=external
        if not snapshots: self.snapshot=None;self.restore=None
    def frame(self):
        return Frame(np.repeat(self.x[None,:],self.spec.agents,axis=0),self.x.copy())
    def reset(self,seed=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        self.t=0;self.x=self.rng.normal(size=2).astype(np.float32)
        return self.frame()
    def step(self,actions):
        actions=np.asarray(actions).copy();mask=np.ones(self.spec.agents,dtype=bool)
        if self.external: mask[:]=False;actions[:]=0
        rewards=(actions==(self.x[0]>0)).astype(np.float32)
        self.t+=1;self.x=self.rng.normal(size=2).astype(np.float32)
        return Transition(self.frame(),rewards,np.zeros(self.spec.agents,bool),
                          np.full(self.spec.agents,self.t==7),mask,actions)
    def features(self,frame,agent):return dict(zip(self.spec.feature_names,map(float,frame.observations[agent])))
    def snapshot(self):return deepcopy(dict(t=self.t,x=self.x,rng=self.rng.bit_generator.state))
    def restore(self,s):self.t=s['t'];self.x=s['x'].copy();self.rng.bit_generator.state=s['rng']

def make_env(): return TinyEnv()
