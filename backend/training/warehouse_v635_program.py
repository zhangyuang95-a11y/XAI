"""Versioned non-executing program with bounded learned legality predicates."""
from copy import deepcopy
from dataclasses import dataclass
from . import warehouse_v635_guards as guards
from .warehouse_v628_program import (Program as Base,ParentProgram,make_protection,payload,legacy,
    domain_sha,FEATURE_SCHEMA_SHA256,ACTION_PHYSICS_CONFIG_SHA256,_eligible,metrics,measure)
from .warehouse_v628_certificate import structure_info,region
from .warehouse_v626_certificate import verify as verify_leaf,binding
from .warehouse_v622_program import digest,plain
from experiments.warehouse.v626_protocol import require,bound
import numpy as np

VERSION='warehouse-role-program.v6.35-bounded-guard-subtrees'


def protection(program):
    contract=program.protection
    require(isinstance(contract,dict) and set(contract)=={'parent','parent_version','entries'},'missing original protection contract')
    parent=ParentProgram.from_dict(bound(contract['parent']),program.actor_role_sha256)
    require(contract==make_protection(parent,contract['parent']) and parent.role==program.role,'original protected regions changed')
    tree,info=structure_info(program.structure,program.maximum_leaves)
    legal=program.certificates();require(legal==guards.leaf_actions(tree,program.maximum_leaves),'guard fact map differs')
    paths=[region(tree,i,program.maximum_leaves)['path'] for i in range(len(program.leaves))];all_ids=set();entries=[]
    for e in contract['entries']:
        prefix=e['path'];ids=[i for i,path in enumerate(paths) if len(path)>=len(prefix) and digest(path[:len(prefix)])==digest(prefix)]
        require(ids and not all_ids.intersection(ids),'missing or overlapping original prefix')
        for i in ids:require(payload(program.leaves[i])==e['leaf_payload'] and set(e['legal'])<=set(legal[i]),'original protected output or physical fact changed')
        all_ids.update(ids);entries.append(dict(parent_leaf=e['parent_leaf'],descendant_leaves=ids))
    return dict(version='warehouse-original-prefix-protection-with-bounded-guards.v1',parent=contract['parent'],
                protected_descendants=sorted(all_ids),entries=entries,structure_sha256=digest(tree),depth=info['depth'])


@dataclass
class Program(Base):
    target_actor_role_sha256:str=None
    def __post_init__(self):
        if self.target_actor_role_sha256 is None:self.target_actor_role_sha256=self.actor_role_sha256
    def certificates(self):return guards.leaf_actions(self.structure,self.maximum_leaves)
    def leaf_indices(self,x):return super().leaf_indices(guards.validate_rows_with_masks(x))
    def validate(self):
        tree,_=structure_info(self.structure,self.maximum_leaves)
        require(self.arm=='C3' and self.role in ('robot_1','robot_2') and len(self.leaves)==tree['leaf_count'],'invalid bounded program identity')
        require(isinstance(self.target_actor_role_sha256,str) and len(self.target_actor_role_sha256)==64,'missing target NN hash')
        legal=self.certificates()
        for i,leaf in enumerate(self.leaves):
            verify_leaf(leaf,region(tree,i,self.maximum_leaves),legal[i],leaf['certificate'])
            require(leaf['certificate']['use_path'] is True,'path-conditioned proof required')
        for ids in (self.feedback_leaves,self.explanation_leaves):require(isinstance(ids,frozenset) and all(type(i) is int and 0<=i<len(self.leaves) for i in ids),'invalid eligibility IDs')
        if self.report is None:require(not self.feedback_leaves and not self.explanation_leaves,'qualification cannot be inherited')
        else:
            r=self.report;require(r['domain_sha256']==domain_sha() and r['actor_role_sha256']==self.actor_role_sha256 and
                r['target_actor_role_sha256']==self.target_actor_role_sha256 and r['structure_sha256']==digest(tree) and
                r['external_program_selects_execution_action'] is False,'qualification target, tree or schema differs')
            for key in ('split_episode_ids','split_scenario_ids'):
                sets=r[key];require(set(sets)==set(legacy.SPLITS) and all(not set(sets[a])&set(sets[b]) for a in sets for b in sets if a<b),'evidence partitions overlap')
            require(len(r['fit_leaf_counts'])==len(self.leaves) and set(r['eligibility'])==set(map(str,range(len(self.leaves)))),'incomplete eligibility evidence')
            for purpose,ids,threshold in (('feedback',self.feedback_leaves,.93),('explanation',self.explanation_leaves,.95)):
                expected=frozenset(i for i in range(len(self.leaves)) if r['fit_leaf_counts'][i] and _eligible(r['eligibility'][str(i)][purpose],threshold))
                require(expected==ids,'qualified leaf set differs from evidence')
        protection(self);return self
    def to_dict(self):
        self.validate()
        return plain(dict(version=VERSION,role=self.role,actor_role_sha256=self.actor_role_sha256,
            target_actor_role_sha256=self.target_actor_role_sha256,structure=self.structure,leaves=self.leaves,arm=self.arm,
            maximum_leaves=self.maximum_leaves,protection=self.protection,report=self.report,
            feedback_leaves=sorted(self.feedback_leaves),explanation_leaves=sorted(self.explanation_leaves),
            domain_sha256=domain_sha(),feature_schema_sha256=FEATURE_SCHEMA_SHA256,action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256,
            guard_evidence=guards.tree_evidence(self.structure,self.maximum_leaves),semantic_protection=protection(self),
            external_program_selects_execution_action=False))
    @classmethod
    def from_dict(cls,a,target_actor_role_sha256):
        keys={'version','role','actor_role_sha256','target_actor_role_sha256','structure','leaves','arm','maximum_leaves','protection','report',
              'feedback_leaves','explanation_leaves','domain_sha256','feature_schema_sha256','action_physics_config_sha256',
              'guard_evidence','semantic_protection','external_program_selects_execution_action'}
        require(set(a)==keys and a['version']==VERSION and a['target_actor_role_sha256']==target_actor_role_sha256 and
            a['domain_sha256']==domain_sha() and a['feature_schema_sha256']==FEATURE_SCHEMA_SHA256 and
            a['action_physics_config_sha256']==ACTION_PHYSICS_CONFIG_SHA256 and a['external_program_selects_execution_action'] is False,'unknown or mismatched bounded program')
        for key in ('feedback_leaves','explanation_leaves'):require(a[key]==sorted(set(a[key])) and all(type(i) is int for i in a[key]),'invalid serialized eligibility')
        out=cls(a['role'],a['actor_role_sha256'],a['structure'],a['leaves'],a['arm'],a['maximum_leaves'],
            frozenset(a['feedback_leaves']),frozenset(a['explanation_leaves']),a['report'],a['protection'],a['target_actor_role_sha256']).validate()
        require(a['guard_evidence']==guards.tree_evidence(out.structure,out.maximum_leaves) and a['semantic_protection']==protection(out),'physical/protection evidence differs')
        return out


def import_previous(source):
    source.validate();out=Program(source.role,source.actor_role_sha256,deepcopy(source.structure),deepcopy(source.leaves),
        'C3',source.maximum_leaves,protection=deepcopy(source.protection),
        target_actor_role_sha256=getattr(source,'target_actor_role_sha256',source.actor_role_sha256))
    old_legal=source.certificates();new_legal=out.certificates()
    for i,leaf in enumerate(source.leaves):
        out.leaves[i]=rebind(leaf,region(source.structure,i,source.maximum_leaves),region(out.structure,i,out.maximum_leaves),old_legal[i],new_legal[i])
    return out.validate()


def rebind(leaf,old_path,new_path,old_legal,new_legal):
    require(old_path['path']==new_path['path'] and set(old_legal)<=set(new_legal),'proof reuse changed a numeric path or lost a physical fact')
    verify_leaf(leaf,old_path,old_legal,leaf['certificate']);out=deepcopy(leaf)
    out['certificate']['checks']=[c for c in out['certificate']['checks'] if c['action'] not in new_legal]
    out['certificate']['binding_sha256']=binding(out,new_path,new_legal,True)
    verify_leaf(out,new_path,new_legal,out['certificate']);return out


def qualify(program,parts):
    counts=np.bincount(program.leaf_indices(parts['fit']['x']),minlength=len(program.leaves)).tolist()
    e=parts['eligibility'];pred=program.distributions(e['x'],e['m']);arr=(e['x'],e['m'],e['sampled'])
    eligibility={str(i):{p:metrics(program,e['rows'],arr,e['p'],{i},p,pred) for p in ('feedback','explanation')} for i in range(len(program.leaves))}
    program.feedback_leaves=frozenset(i for i in range(len(counts)) if counts[i] and _eligible(eligibility[str(i)]['feedback'],.93))
    program.explanation_leaves=frozenset(i for i in range(len(counts)) if counts[i] and _eligible(eligibility[str(i)]['explanation'],.95))
    program.report=dict(domain_sha256=domain_sha(),actor_role_sha256=program.actor_role_sha256,target_actor_role_sha256=program.target_actor_role_sha256,
        structure_sha256=digest(program.structure),external_program_selects_execution_action=False,full_NN_distribution_targets=True,
        fit_leaf_counts=counts,eligibility=eligibility,split_episode_ids={s:sorted({str(r['episode']) for r in a['rows']}) for s,a in parts.items()},
        split_scenario_ids={s:sorted({str(r['base_seed']) for r in a['rows']}) for s,a in parts.items()},
        protected_refit_and_calibration_skipped=protection(program)['protected_descendants'])
    return program.validate()
