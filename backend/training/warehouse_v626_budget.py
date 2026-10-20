"""Separate nonrefundable solver budgets on a shared durable checkpoint."""
from __future__ import annotations
import time,math
from .warehouse_v625_checkpoint import CheckpointSession,BudgetExhausted


class Budget:
    def __init__(self,path,identity,limits,seconds,*,resume=False,started_at=None):
        self.limits=dict(limits)
        if set(limits)!={'leaf','node','proof'} or any(type(n) is not int or n<0 for n in limits.values()):
            raise ValueError('explicit solver categories required')
        self.session=CheckpointSession(path,{**identity,'solver_limits':self.limits},max_calls=sum(limits.values()),
            wall_seconds=seconds,resume=resume,started_at=started_at)
        self.counts={k:0 for k in limits}
        self.pending={};self.phase='fit';self.final_leaf_reserve=0
        if resume:
            for row in self.session.events():
                if row['event']=='call_started':self.counts[row['data']['metadata']['solver_kind']]+=1

    def check(self):self.session.check()

    def consume(self,kind,metadata=None):
        self.check()
        maximum=self.limits[kind]-(self.final_leaf_reserve if kind=='leaf' and self.phase!='final' else 0)
        if self.counts[kind]>=maximum:raise BudgetExhausted('v626 '+kind+' budget exhausted')
        token=self.session.consume({'solver_kind':kind,'phase':self.phase,**(metadata or {})})
        self.counts[kind]+=1;self.pending[token]=kind
        return token

    def finish(self,token,*,success,detail):
        if token not in self.pending:raise ValueError('unknown or repeated solver settlement')
        self.session.finish(token,success=success,detail=detail);self.pending.pop(token)

    def solve(self,kind,metadata,callback):
        token=self.consume(kind,metadata);started=time.monotonic()
        try:
            result=callback()
        except BaseException as error:
            self.finish(token,success=False,detail={'exception':type(error).__name__,'elapsed_seconds':time.monotonic()-started})
            raise
        detail={'success':bool(result.success),'status':int(result.status),'elapsed_seconds':time.monotonic()-started}
        if getattr(result,'nit',None) is not None:detail['iterations']=int(result.nit)
        if getattr(result,'fun',None) is not None:detail['objective']=float(result.fun) if math.isfinite(float(result.fun)) else None
        self.finish(token,success=bool(result.success),detail=detail)
        return result

    def checkpoint(self,state,*,status='running'):
        self.session.commit({'solver_counts':self.counts,'phase':self.phase,**state},status=status)

    def summary(self):return {'counts':dict(self.counts),'limits':self.limits,'checkpoint':self.session.summary()}
