"""Exact prefix-extension of already verified sparse-leaf certificates.

No solver is called. Old path inequalities retain their nonnegative duals;
new inequalities get multiplier zero. New source coordinates have zero old
residual. The full resulting rational inequality is independently replayed.
"""
from copy import deepcopy
from .warehouse_v628_program import payload
from .warehouse_v629_guards import branch_proof
from .warehouse_v626_certificate import verify,binding,public_polytope,polynomial,dual_upper
from .warehouse_v622_program import digest
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-exact-prefix-extension-zero-dual.v1'


def path_actions(path):
    actions={4}
    for item in path['path']:
        require(item['branch'] in ('left','right'),'unknown refined path branch')
        proof=branch_proof(item['weight'],item['bias']);actions.update(proof[item['branch']+'_proven'])
    return frozenset(actions)


def extend(leaf,old_path,new_path,old_legal,new_legal):
    prefix=len(old_path['path'])
    require(len(new_path['path'])>=prefix and digest(new_path['path'][:prefix])==digest(old_path['path']),
        'certificate extension needs the exact original route prefix')
    require(set(old_legal)==set(path_actions(old_path)) and set(new_legal)==set(path_actions(new_path))
        and set(old_legal)<=set(new_legal),'certificate extension overclaimed a physical legality fact')
    verify(leaf,old_path,old_legal,leaf['certificate'])
    require(leaf['certificate']['use_path'] is True,'refined C3 leaves need the original path certificate language')
    out=deepcopy(leaf);proof=deepcopy(leaf['certificate']);checks=[]
    for old in proof['checks']:
        if old['action'] in new_legal:continue
        check=deepcopy(old)
        if check['kind']=='path':
            _,old_A,_,_=public_polytope(leaf,old_path,True)
            _,new_A,_,_=public_polytope(leaf,new_path,True)
            require(len(check['dual'])==len(old_A) and len(new_A)>=len(old_A),'path dual dimension differs')
            check['dual']+= [0.]*(len(new_A)-len(old_A))
        ids,A,rhs,bounds=public_polytope(out,new_path,check['kind']=='path')
        c,k=polynomial(out,check['action'],out['witness'],ids)
        upper=dual_upper(c,k,A,rhs,bounds,check['dual'])
        require([upper.numerator,upper.denominator]==check['upper'],'exact inherited upper bound changed')
        checks.append(check)
    proof.update(checks=checks,binding_sha256=binding(out,new_path,new_legal,True))
    out['certificate']=proof
    verify(out,new_path,new_legal,proof)
    require(digest(payload(out))==digest(payload(leaf)),'refined proof changed original leaf arithmetic')
    return out,dict(version=VERSION,old_certificate_sha256=digest(leaf['certificate']),
        new_certificate_sha256=digest(proof),old_path=old_path,new_path=new_path,
        old_legal=sorted(old_legal),new_legal=sorted(new_legal),new_inequalities_have_zero_dual=True,
        exact_upper_bounds_unchanged=True,solver_calls=0,leaf_coefficients_changed=False)
