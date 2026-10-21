"""Provisional surrogate asset language with exact-output region refinement.

This parser never controls environment actions or grants candidate admission.
Every new path receives a complete numerical proof; no eligibility is copied.
No fitting or selection method is registered by this module.
"""
from copy import deepcopy
import numpy as np
from .warehouse_v629_program import Program as Previous
from .warehouse_v629_guards import tree_evidence
from .warehouse_v628_program import legacy,domain_sha,_eligible,FEATURE_SCHEMA_SHA256,ACTION_PHYSICS_CONFIG_SHA256
from .warehouse_v628_certificate import structure_info,region
from .warehouse_v626_certificate import verify as verify_certificate
from .warehouse_v626_search import Node,decode,encode,replace
from .warehouse_v622_program import digest,plain
from .warehouse_v630_certificate import extend
from experiments.warehouse import v630_semantic_protection as protection
from experiments.warehouse.v626_protocol import require

VERSION='warehouse-role-program.v6.30-exact-output-refinement-prototype'


class Program(Previous):
    def validate(self):
        t,_=structure_info(self.structure,self.maximum_leaves)
        require(self.arm=='C3' and self.role in ('robot_1','robot_2') and len(self.leaves)==t['leaf_count'],
            'invalid refined program identity/shape')
        proofs=self.certificates()
        for i,leaf in enumerate(self.leaves):
            verify_certificate(leaf,region(t,i,self.maximum_leaves),proofs[i],leaf['certificate'])
            require(leaf['certificate']['use_path'] is True,'refined program needs path-conditioned numeric proofs')
        for selected in (self.feedback_leaves,self.explanation_leaves):
            require(isinstance(selected,frozenset) and all(type(i) is int and 0<=i<len(self.leaves) for i in selected),
                'invalid refined qualified leaf indices')
        if self.report is None:
            require(not self.feedback_leaves and not self.explanation_leaves,'refined qualification cannot be inherited without new evidence')
        else:
            r=self.report
            require(r['domain_sha256']==domain_sha() and r['actor_role_sha256']==self.actor_role_sha256
                and r['structure_sha256']==digest(t) and r['external_program_selects_execution_action'] is False,
                'refined qualification source/tree binding differs')
            for key in ('split_episode_ids','split_scenario_ids'):
                groups=r[key]
                require(set(groups)==set(legacy.SPLITS) and all(not set(groups[a])&set(groups[b])
                    for a in groups for b in groups if a<b),'refined evidence partitions overlap')
            require(len(r['fit_leaf_counts'])==len(self.leaves) and set(r['eligibility'])==set(map(str,range(len(self.leaves)))),
                'refined eligibility omitted a counted child')
            for purpose,selected,threshold in [('feedback',self.feedback_leaves,.93),('explanation',self.explanation_leaves,.95)]:
                actual=frozenset(i for i in range(len(self.leaves)) if r['fit_leaf_counts'][i]
                    and _eligible(r['eligibility'][str(i)][purpose],threshold))
                require(actual==selected,'refined qualification differs from new per-child evidence')
        protection.verify(self)
        return self

    def to_dict(self):
        self.validate()
        return plain(dict(version=VERSION,role=self.role,actor_role_sha256=self.actor_role_sha256,structure=self.structure,
            leaves=self.leaves,arm=self.arm,maximum_leaves=self.maximum_leaves,protection=self.protection,
            feedback_leaves=sorted(self.feedback_leaves),explanation_leaves=sorted(self.explanation_leaves),report=self.report,
            domain_sha256=domain_sha(),feature_schema_sha256=FEATURE_SCHEMA_SHA256,
            action_physics_config_sha256=ACTION_PHYSICS_CONFIG_SHA256,guard_evidence=tree_evidence(self.structure,self.maximum_leaves),
            semantic_protection=protection.verify(self),external_program_selects_execution_action=False))

    @classmethod
    def from_dict(cls,a,actor_role_sha256):
        keys={'version','role','actor_role_sha256','structure','leaves','arm','maximum_leaves','protection','feedback_leaves',
            'explanation_leaves','report','domain_sha256','feature_schema_sha256','action_physics_config_sha256','guard_evidence',
            'semantic_protection','external_program_selects_execution_action'}
        require(set(a)==keys and a['version']==VERSION and a['actor_role_sha256']==actor_role_sha256
            and a['domain_sha256']==domain_sha() and a['feature_schema_sha256']==FEATURE_SCHEMA_SHA256
            and a['action_physics_config_sha256']==ACTION_PHYSICS_CONFIG_SHA256
            and a['external_program_selects_execution_action'] is False,'unknown or mismatched refined program asset')
        for key in ('feedback_leaves','explanation_leaves'):
            require(a[key]==sorted(set(a[key])) and all(type(i) is int for i in a[key]),'invalid refined serialized qualification')
        out=cls(a['role'],actor_role_sha256,a['structure'],a['leaves'],a['arm'],a['maximum_leaves'],
            frozenset(a['feedback_leaves']),frozenset(a['explanation_leaves']),a['report'],a['protection']).validate()
        require(a['guard_evidence']==tree_evidence(out.structure,out.maximum_leaves)
            and a['semantic_protection']==protection.verify(out),'refined logical proof envelope differs')
        return out


def from_previous(program):
    program.validate();args=deepcopy(program.__dict__)
    args.update(feedback_leaves=frozenset(),explanation_leaves=frozenset(),report=None)
    return Program(**args).validate()


def split_identical(program,leaf_index,weights,bias):
    """One supplied split, no search; all old distributions remain identical.

    The predicate is a caller-supplied public condition. This function neither
    fits nor chooses it, and clears all qualification until re-evaluated.
    """
    program.validate();require(type(leaf_index) is int and 0<=leaf_index<len(program.leaves),'invalid refined source leaf')
    tree=decode(program.structure);_,old_paths=encode(tree)
    replacement=Node(tuple(map(float,np.asarray(weights,np.float32))),float(np.float32(bias)))
    structure,new_paths=encode(replace(tree,old_paths[leaf_index],replacement));structure=plain(structure)
    structure_info(structure,program.maximum_leaves)  # Limits precede any proof construction.
    out=Program(program.role,program.actor_role_sha256,structure,[],'C3',program.maximum_leaves,
        protection=deepcopy(program.protection));new_legal=out.certificates();old_legal=program.certificates();derivations=[]
    for i,path in enumerate(new_paths):
        old_index=next(j for j,prefix in enumerate(old_paths) if path[:len(prefix)]==prefix)
        leaf,proof=extend(program.leaves[old_index],region(program.structure,old_index,program.maximum_leaves),
            region(structure,i,program.maximum_leaves),old_legal[old_index],new_legal[i])
        out.leaves.append(leaf);derivations.append(dict(source_leaf=old_index,child_leaf=i,proof=proof))
    out.validate()
    return out,dict(version=VERSION,source_tree_sha256=digest(program.structure),target_tree_sha256=digest(out.structure),
        derivations=derivations,every_old_leaf_output_preserved=True,qualification_inherited=False,
        solver_calls=0,NN_updates=0,environment_resets=0,model_acceptance_passed=False,pilot_opened=False)
