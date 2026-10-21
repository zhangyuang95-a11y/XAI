"""Protected-region v628 envelope. Surrogate only; no environment actions."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from .warehouse_v626_program import (legacy, evaluate_leaf, fit_leaf, calibrate, metrics,
    kl, _eligible, _route, gate_metrics, digest, plain, MASK_INDICES,
    FEATURE_SCHEMA_SHA256, ACTION_PHYSICS_CONFIG_SHA256, certified_leaf_actions,
    domain_sha, validate_rows, Program as ParentProgram)
from .warehouse_v628_certificate import structure_info, region, verify, rebind
VERSION='warehouse-role-program.v6.28-protected-capacity'
ARMS=('C3',)


def payload(leaf):
    return {k:v for k,v in leaf.items() if k!='certificate'}


def path_key(path):
    return digest(path['path'])


def make_protection(parent, ref):
    parent.validate()
    ids=sorted(parent.feedback_leaves | parent.explanation_leaves)
    legal=parent.certificates()
    return dict(parent=ref, parent_version=parent.to_dict()['version'], entries=[dict(
        parent_leaf=i, path=region(parent.structure,i,64)['path'],
        leaf_payload=payload(parent.leaves[i]), legal=sorted(legal[i])) for i in ids])


def verify_protection(program):
    p=program.protection
    if not isinstance(p,dict) or set(p)!={'parent','parent_version','entries'}:
        raise ValueError('missing protected parent contract')
    ref=p['parent']; raw=Path(ref['path']).read_bytes()
    if sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('protected parent asset changed')
    parent=ParentProgram.from_dict(json.loads(raw),program.actor_role_sha256)
    if p!=make_protection(parent,ref):raise ValueError('protection contract differs from bound parent')
    paths={path_key(region(program.structure,i,program.maximum_leaves)):i for i in range(len(program.leaves))}
    legal=program.certificates()
    for e in p['entries']:
        i=paths.get(digest(e['path']))
        if (i is None or payload(program.leaves[i])!=e['leaf_payload']
                or sorted(legal[i])!=e['legal']):raise ValueError('protected path or float32 output modified')
    return True


def protected_indices(program):
    keys={digest(e['path']) for e in program.protection['entries']}
    return frozenset(i for i in range(len(program.leaves))
        if path_key(region(program.structure,i,program.maximum_leaves)) in keys)


def from_parent(parent, ref, maximum):
    out=Program(parent.role,parent.actor_role_sha256,deepcopy(parent.structure),
        deepcopy(parent.leaves),'C3',maximum,protection=make_protection(parent,ref))
    # The same numeric path is bound to the new envelope; no qualification inherited.
    for i in range(len(out.leaves)):
        out.leaves[i]=rebind(out.leaves[i],region(parent.structure,i,64),
            region(out.structure,i,maximum),parent.certificates()[i],out.certificates()[i])
    return out.validate()

@dataclass
class Program:
    role:str
    actor_role_sha256:str
    structure:dict
    leaves:list
    arm:str
    maximum_leaves:int=64
    feedback_leaves:frozenset=frozenset()
    explanation_leaves:frozenset=frozenset()
    report:dict=None
    protection:dict=None

    def leaf_indices(self,x):
        tree,info=structure_info(self.structure,self.maximum_leaves)
        return _route(tree,np.asarray(x,np.float32),info['depth'])

    def certificates(self):
        t,_=structure_info(self.structure,self.maximum_leaves)
        return certified_leaf_actions(t['split_weights'],t['split_bias'],t['left_refs'],t['right_refs'],
            len(t['split_bias']),t['leaf_count'],MASK_INDICES,explicit=True,wait_action=4)

    def distributions(self,x,m):
        x=validate_rows(x);m=np.asarray(m,np.float32)
        if (m.shape!=(len(x),5) or np.any((m!=0)&(m!=1)) or np.any(m[:,4]!=1)
                or not np.array_equal(x[:,MASK_INDICES],m)):
            raise ValueError('encoded legal bits and physical masks disagree')
        routes=self.leaf_indices(x)
        z,q=np.empty((len(x),5),np.float32),np.empty((len(x),5),np.float32)
        for i,leaf in enumerate(self.leaves):
            take=routes==i;z[take],q[take]=evaluate_leaf(leaf,x[take],m[take])
        return routes,z,q

    def validate(self):
        t,info=structure_info(self.structure,self.maximum_leaves)
        if self.arm not in ARMS or self.role not in ('robot_1','robot_2') or len(self.leaves)!=t['leaf_count']:
            raise ValueError('program identity/shape mismatch')
        proofs=self.certificates()
        for i,leaf in enumerate(self.leaves):
            if self.arm in ('C0','C1'):
                if legacy.certify_leaf(leaf,FEATURE_NAMES,proofs[i])!=leaf['certificate'] or not leaf['certificate']['passed']:
                    raise ValueError('invalid legacy control certificate')
                if self.arm=='C0' and leaf['support']:raise ValueError('variable constant control')
            else:
                verify(leaf,region(t,i,self.maximum_leaves),proofs[i],leaf['certificate'])
                if leaf['certificate']['use_path'] != (self.arm=='C3'):raise ValueError('wrong certificate language')
        for selected in (self.feedback_leaves,self.explanation_leaves):
            if not isinstance(selected,frozenset) or not selected<=set(range(len(self.leaves))):raise ValueError('invalid eligibility set')
        if self.report is not None:
            r=self.report
            if (r['domain_sha256']!=domain_sha() or r['actor_role_sha256']!=self.actor_role_sha256
                    or r['structure_sha256']!=digest(t) or r['external_program_selects_execution_action'] is not False):
                raise ValueError('lost model/domain/source evidence')
            for key in ('split_episode_ids','split_scenario_ids'):
                groups=r[key]
                if set(groups)!=set(legacy.SPLITS) or any(set(groups[a])&set(groups[b]) for a in groups for b in groups if a<b):
                    raise ValueError('evidence partitions overlap')
            for purpose,selected,threshold in (('feedback',self.feedback_leaves,.93),('explanation',self.explanation_leaves,.95)):
                actual=frozenset(i for i in range(len(self.leaves)) if r['fit_leaf_counts'][i] and _eligible(r['eligibility'][str(i)][purpose],threshold))
                if actual!=selected:raise ValueError('qualification differs from eligibility evidence')
        verify_protection(self)
        return self

    def to_dict(self):
        self.validate()
        return plain(dict(version=VERSION,protection=self.protection,role=self.role,actor_role_sha256=self.actor_role_sha256,
            structure=self.structure,leaves=self.leaves,arm=self.arm,maximum_leaves=self.maximum_leaves,
            feedback_leaves=sorted(self.feedback_leaves),explanation_leaves=sorted(self.explanation_leaves),
            domain_sha256=domain_sha(),feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256,report=self.report))

    @classmethod
    def from_dict(cls,a,actor_role_sha256):
        keys={'version','role','actor_role_sha256','structure','leaves','arm','maximum_leaves',
            'protection','feedback_leaves','explanation_leaves','domain_sha256','feature_schema_sha256','action_physics_config_sha256','report'}
        if (set(a)!=keys or a.get('version')!=VERSION or a.get('actor_role_sha256')!=actor_role_sha256
                or a.get('domain_sha256')!=domain_sha() or a.get('feature_schema_sha256')!=FEATURE_SCHEMA_SHA256
                or a.get('action_physics_config_sha256')!=ACTION_PHYSICS_CONFIG_SHA256):raise ValueError('unknown or mismatched program binding')
        for key in ('feedback_leaves','explanation_leaves'):
            if a[key]!=sorted(set(a[key])) or any(type(i) is not int for i in a[key]):raise ValueError('invalid serialized qualified IDs')
        return cls(a['role'],actor_role_sha256,a['structure'],a['leaves'],a['arm'],a['maximum_leaves'],
            frozenset(a['feedback_leaves']),frozenset(a['explanation_leaves']),a['report'],a['protection']).validate()


def metrics(program,rows,arrays,p,selected,purpose,predictions=None):
    return legacy._metrics(program,rows,arrays,p,selected,purpose,predictions=predictions)


def qualify(program, parts, before=None):
    """Every leaf, including protected leaves, needs current eligibility evidence."""
    counts=np.bincount(program.leaf_indices(parts['fit']['x']),minlength=len(program.leaves)).tolist()
    e=parts['eligibility']; pred=program.distributions(e['x'],e['m']); arr=(e['x'],e['m'],e['sampled'])
    eligibility={str(i):{purpose:metrics(program,e['rows'],arr,e['p'],{i},purpose,pred)
        for purpose in ('feedback','explanation')} for i in range(len(program.leaves))}
    program.feedback_leaves=frozenset(i for i in range(len(counts)) if counts[i] and _eligible(eligibility[str(i)]['feedback'],.93))
    program.explanation_leaves=frozenset(i for i in range(len(counts)) if counts[i] and _eligible(eligibility[str(i)]['explanation'],.95))
    program.report=dict(domain_sha256=domain_sha(),actor_role_sha256=program.actor_role_sha256,
        structure_sha256=digest(program.structure),external_program_selects_execution_action=False,
        full_NN_distribution_targets=True,fit_leaf_counts=counts,eligibility=eligibility,
        split_episode_ids={s:sorted({str(r['episode']) for r in a['rows']}) for s,a in parts.items()},
        split_scenario_ids={s:sorted({str(r['base_seed']) for r in a['rows']}) for s,a in parts.items()},
        calibration_before=before or [],protected_refit_and_calibration_skipped=sorted(protected_indices(program)))
    return program.validate()


def measure(program,a):
    pred=program.distributions(a['x'],a['m']);arr=(a['x'],a['m'],a['sampled'])
    return {p:metrics(program,a['rows'],arr,a['p'],sel,p,pred) for p,sel in (
        ('feedback',program.feedback_leaves),('explanation',program.explanation_leaves))}


def complete_program(program,parts,budget,*,refit=True):
    t,_=structure_info(program.structure,program.maximum_leaves);legal=program.certificates()
    routes={s:program.leaf_indices(a['x']) for s,a in parts.items()}
    protected=protected_indices(program);before=[]
    for i,leaf in enumerate(program.leaves):
        budget.check()
        if refit and i not in protected:
            a=parts['fit'];take=routes['fit']==i
            leaf=fit_leaf(a['x'][take],a['m'][take],a['p'][take],arm='C3',
                path=region(t,i,program.maximum_leaves),legal=legal[i],budget=budget,initial=leaf,final=True)
            before.append(deepcopy(leaf))
            a=parts['calibration'];take=routes['calibration']==i
            program.leaves[i]=calibrate(leaf,a['x'][take],a['m'][take],a['p'][take],arm='C3',
                path=region(t,i,program.maximum_leaves),legal=legal[i],budget=budget)
        else:before.append(deepcopy(leaf))
    qualify(program,parts,before)
    results={s:measure(program,a) for s,a in parts.items()}
    for value in results.values():value['gate']=gate_metrics(value,phase='development')
    return program,results

