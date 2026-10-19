"""Explicit depth-six/32-leaf surrogate programs for the unchanged v6.20 NN.

This module never constructs an environment, changes an Actor, selects an
execution action, or grants an experiment capability. Only the legacy pure
leaf optimizer, feature encoder and numerical routing/certification functions
are reused. The historical 16-leaf classes and constants remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import time

import numpy as np

from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS
from env.warehouse.public_observation_v6 import OBSERVATION_DIM
from env.warehouse.public_action_features_v617 import encode_actor_tree_features
from .warehouse_v6_actor import (FEATURE_DIM, FEATURE_NAMES, FEATURE_SCHEMA_SHA256,
    ACTION_PHYSICS_CONFIG_SHA256, MODEL_VERSION)
from .warehouse_v6_tree_routing import (HARD_SPLIT_ROUTING_VERSION,
    certified_feature_bounds, certified_leaf_actions, leaf_indices,
    validate_certification_domain)
from .warehouse_v619_leaf_fit import (fit_constrained_leaf_distribution,
    calibrate_constrained_leaf_distribution, float32_preference_proof,
    LEAF_EPSILON, LEAF_REGULARIZATION, LEAF_MAXITER, LEAF_TOLERANCE,
    CALIBRATION_TEMPERATURES)


VERSION = "warehouse-role-program.v6.22-depth6-leaves32"
COMPILER_VERSION = "warehouse-v622-internal-topology-legal-guards.v1"
FIT_METHODS = frozenset(("v620_m2_expanded_distribution_refit_v622",
    "actor_topology_guarded_distribution_refit_v622", "fixed_template_leaf_budget_dp_v622"))
DEFAULT_METHOD = "actor_topology_guarded_distribution_refit_v622"
D32_METHOD = "fixed_template_leaf_budget_dp_v622"
TARGET_ARITHMETIC = "legacy_fp32_where_mask_minus1e9_exp_sum.v1"
SPLITS = ("fit", "calibration", "eligibility", "validation")
MASK_INDICES = tuple(FEATURE_NAMES.index("legal_"+a.lower()) for a in ACTIONS)
_STRUCTURE_KEYS = ("root_ref", "split_weights", "split_bias", "left_refs", "right_refs", "leaf_count", "feature_dim")


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value


def digest(value):
    return sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _is_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class ProgramLimits:
    max_depth: int = 6
    max_leaves: int = 32
    max_source_features: int = 3

    def __post_init__(self):
        for value,maximum in ((self.max_depth,6),(self.max_leaves,32),(self.max_source_features,3)):
            if type(value) is not int or not 1<=value<=maximum:
                raise ValueError("v6.22 program limits must be explicit positive integers within 6/32/3")

    def to_dict(self):
        return {"maximum_depth":self.max_depth,"maximum_leaves":self.max_leaves,
            "maximum_source_features_per_node":self.max_source_features,"guards_and_empty_branches_count":True}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value,Mapping) or set(value)!={"maximum_depth","maximum_leaves",
                "maximum_source_features_per_node","guards_and_empty_branches_count"} or value["guards_and_empty_branches_count"] is not True:
            raise ValueError("unknown v6.22 program limit schema")
        return cls(value["maximum_depth"],value["maximum_leaves"],value["maximum_source_features_per_node"])


DEFAULT_LIMITS = ProgramLimits()


class ProgramFitError(RuntimeError):
    """Failure carries actual attempted leaf and solver work, never a retry."""
    def __init__(self, message, *, report):
        super().__init__(message)
        self.report = plain(deepcopy(report))


class ProgramBudgetExhausted(ProgramFitError):
    """A scientific wall-budget failure, carrying the attempted work receipt."""


def _deadline(report, stage):
    deadline=report["fit_budget"].get("deadline_monotonic")
    if deadline is not None and time.monotonic()>=deadline:
        report["status"]="budget_exhausted";report["failure_stage"]=stage
        report["scientific_failure"]=True
        report["fit_budget"].update({"actual_final_leaf_optimization_calls":report["leaf_optimization_calls"],
            "actual_total_leaf_optimization_calls":report["fit_budget"]["prior_leaf_optimization_calls"]+report["leaf_optimization_calls"]})
        raise ProgramBudgetExhausted("v6.22 final-program wall budget exhausted at "+stage,report=report)


def source_feature_cost(weights, feature_names=FEATURE_NAMES):
    # Exact existing semantics, including the two-source public risk feature.
    return sum(2 if feature_names[i]=="public_hard_risk_context" else 1
               for i in np.flatnonzero(weights))


def normalize_structure(structure, feature_names=FEATURE_NAMES, *, limits=DEFAULT_LIMITS, enforce_limits=True):
    if not isinstance(limits,ProgramLimits) or not isinstance(structure,Mapping):
        raise ValueError("explicit ProgramLimits and a structure mapping are required")
    names=tuple(feature_names)
    if not names or len(set(names))!=len(names) or any(not isinstance(n,str) for n in names):
        raise ValueError("invalid feature schema")
    w=np.asarray(structure["split_weights"],np.float32)
    if w.shape==(0,): w=np.zeros((0,len(names)),np.float32)
    b=np.asarray(structure["split_bias"],np.float32)
    l,r=np.asarray(structure["left_refs"]),np.asarray(structure["right_refs"])
    root,count=structure["root_ref"],structure["leaf_count"]
    integer=lambda x:isinstance(x,(int,np.integer)) and not isinstance(x,(bool,np.bool_))
    if (not integer(root) or not integer(count) or count<1 or w.ndim!=2 or w.shape[1]!=len(names)
            or b.shape!=(len(w),) or l.shape!=b.shape or r.shape!=b.shape
            or any(not np.isfinite(x).all() for x in (w,b,l,r))
            or np.any(l!=np.floor(l)) or np.any(r!=np.floor(r))
            or root!=(0 if len(w) else -1) or len(w)!=count-1
            or structure.get("feature_dim",len(names))!=len(names)):
        raise ValueError("invalid v6.22 full rooted binary tree")
    l,r=l.astype(np.int64),r.astype(np.int64)
    seen,paths=set(),{}
    def walk(ref,path):
        if ref<0:
            leaf=-1-ref
            if not 0<=leaf<count or leaf in paths: raise ValueError("invalid or repeated leaf")
            paths[leaf]=list(path);return
        if ref>=len(w) or ref in seen: raise ValueError("invalid, repeated or cyclic node")
        seen.add(ref)
        walk(int(l[ref]),(*path,{"node":ref,"branch":"left"}))
        walk(int(r[ref]),(*path,{"node":ref,"branch":"right"}))
    walk(int(root),())
    if seen!=set(range(len(w))) or set(paths)!=set(range(count)): raise ValueError("unreachable tree tensors")
    costs=[source_feature_cost(row,names) for row in w]
    capacity={"node_count":len(w),"leaf_count":int(count),"depth":max(map(len,paths.values())),
        "maximum_source_features_per_node":max(costs,default=0),"source_features_per_node":costs,
        "leaf_paths":paths}
    if enforce_limits and (capacity["depth"]>limits.max_depth or count>limits.max_leaves
            or capacity["maximum_source_features_per_node"]>limits.max_source_features):
        raise ValueError("tree exceeds explicit v6.22 depth/leaf/source-feature limits")
    tree={"root_ref":int(root),"split_weights":w.copy(),"split_bias":b.copy(),
        "left_refs":l.copy(),"right_refs":r.copy(),"leaf_count":int(count),"feature_dim":len(names)}
    return tree,capacity


def program_structure_sha256(structure):
    tree,_=normalize_structure(structure)
    return digest(tree)


def _certificates(tree, feature_names=FEATURE_NAMES):
    masks=tuple(feature_names.index("legal_"+a.lower()) for a in ACTIONS)
    return certified_leaf_actions(tree["split_weights"],tree["split_bias"],tree["left_refs"],tree["right_refs"],
        len(tree["split_weights"]),tree["leaf_count"],masks,explicit=True,wait_action=4)


def _route(tree,x,depth):
    return leaf_indices(x,tree["split_weights"],tree["split_bias"],tree["left_refs"],tree["right_refs"],len(tree["split_weights"]),depth)


def compile_actor_structure(role_actor, *, limits=DEFAULT_LIMITS):
    """Guard frozen raw internal leaf preferences; never claim policy parity.

    No observations enter guard selection. Every empty false child is counted.
    The returned structure discards old leaf outputs and needs full-NN refit.
    """
    names=tuple(role_actor.feature_names)
    masks=tuple(names.index("legal_"+a.lower()) for a in ACTIONS)
    values=role_actor.weights
    raw_n,raw_c=np.asarray(values["tree.node_count"]),np.asarray(values["tree.leaf_count"])
    if (raw_n.shape!=(1,) or raw_c.shape!=(1,) or not np.isfinite(raw_n).all() or not np.isfinite(raw_c).all()
            or raw_n[0]!=np.floor(raw_n[0]) or raw_c[0]!=np.floor(raw_c[0])
            or raw_n[0]<0 or raw_c[0]<1):raise ValueError("invalid frozen internal tree counts")
    n,c=int(raw_n[0]),int(raw_c[0])
    original,old_capacity=normalize_structure({"root_ref":0 if n else -1,
        "split_weights":values["tree.split.weight"][:n],"split_bias":values["tree.split.bias"][:n],
        "left_refs":values["tree.left_refs"][:n],"right_refs":values["tree.right_refs"][:n],
        "leaf_count":c,"feature_dim":len(names)},names,limits=limits,enforce_limits=False)
    if any(np.count_nonzero(row)>3 for row in original["split_weights"]):
        raise ValueError("frozen internal tree has unsparsified weights; exact hard-route parity is not established")
    z=np.asarray(values["tree.leaf_logits"],np.float32)
    if z.ndim!=2 or z.shape[1]!=5 or len(z)<c or not np.isfinite(z).all(): raise ValueError("invalid internal leaf logits")
    previous=_certificates(original,names);preferred=z[:c].argmax(-1)
    w,b,l,r,origins,decisions=[],[],[],[],[],[]
    def leaf(source,action,branch):
        i=len(origins);origins.append({"compiled_leaf":i,"original_leaf":source,
            "guard_branch":branch,"frozen_preferred_action":ACTIONS[action]});return -1-i
    def copy(ref):
        if ref<0:
            source=-1-ref;action=int(preferred[source]);guard=action not in previous[source]
            detail={"original_leaf":source,"frozen_preferred_action":ACTIONS[action],"guard_added":guard,
                "original_path":old_capacity["leaf_paths"][source],"original_path_legal_actions":sorted(previous[source])}
            decisions.append(detail)
            if not guard:return leaf(source,action,"unchanged")
            node=len(w);row=np.zeros(len(names),np.float32);row[masks[action]]=1
            w.append(row);b.append(np.float32(-.5));l.append(leaf(source,4,"illegal_false_wait"));r.append(leaf(source,action,"legal_true"))
            detail.update({"guard_node":node,"guard_feature":names[masks[action]],"guard_bias":-.5,"guard_coefficient":1.})
            return node
        node=len(w);w.append(original["split_weights"][ref].copy());b.append(original["split_bias"][ref]);l.append(0);r.append(0)
        l[node]=copy(int(original["left_refs"][ref]));r[node]=copy(int(original["right_refs"][ref]));return node
    root=copy(original["root_ref"])
    tree,capacity=normalize_structure({"root_ref":root,"split_weights":np.asarray(w,np.float32).reshape(-1,len(names)),
        "split_bias":np.asarray(b,np.float32),"left_refs":np.asarray(l,np.int64),"right_refs":np.asarray(r,np.int64),
        "leaf_count":len(origins),"feature_dim":len(names)},names,limits=limits,enforce_limits=False)
    proofs=_certificates(tree,names)
    for i,origin in enumerate(origins):
        if ACTIONS.index(origin["frozen_preferred_action"]) not in proofs[i]:raise RuntimeError("compiled preferred action lacks new path proof")
        origin.update({"path_legal_actions":sorted(proofs[i]),"compiled_path":capacity["leaf_paths"][i]})
    reasons=[]
    if capacity["depth"]>limits.max_depth:reasons.append("maximum_depth_exceeded")
    if capacity["leaf_count"]>limits.max_leaves:reasons.append("maximum_leaf_count_exceeded")
    if capacity["maximum_source_features_per_node"]>limits.max_source_features:reasons.append("maximum_source_feature_cost_exceeded")
    report={"schema":COMPILER_VERSION,"status":"structurally_infeasible" if reasons else "feasible","feasible":not reasons,
        "infeasibility_reasons":reasons,"limits":limits.to_dict(),"actor_role_sha256":role_actor.role_sha256,
        "original_structure":plain(original),"required_guarded_structure":plain(tree),
        "original_structure_sha256":digest(original),"required_guarded_structure_sha256":digest(tree),
        "original_capacity":plain(old_capacity),"required_capacity":plain(capacity),
        "guard_count":sum(d["guard_added"] for d in decisions),"leaf_guard_decisions":decisions,"compiled_leaf_origins":origins,
        "guard_policy":"frozen_raw_float32_leaf_argmax_then_explicit_legal_guard_else_WAIT",
        "guard_policy_uses_observations":False,"all_guards_and_empty_children_counted":True,
        "original_unguarded_hard_route_parity":True,"input_weights_modified":False,
        "full_actor_policy_equivalence_claimed":False,"masked_fallback_ranking_preserved":False,
        "leaf_outputs_reused_as_fitted_program":False,"new_leaf_preferred_action_requires_new_path_proof":True,
        "routing_contract":HARD_SPLIT_ROUTING_VERSION}
    return (None if reasons else tree),report


def masked_probabilities(logits,masks):
    """Reproduce the source-locked old fp32 target arithmetic, or reject.

    The finite sentinel is deliberately unchanged for the capacity-only
    comparison. If it gives an illegal probability or overflows, fail closed;
    no normalization repair or execution-action substitution occurs here.
    """
    z,m=np.asarray(logits,np.float32),np.asarray(masks,np.float32)
    if (z.shape!=m.shape or z.ndim<1 or z.shape[-1]!=5 or not np.isfinite(z).all()
            or not np.isfinite(m).all() or np.any((m!=0)&(m!=1)) or np.any(m[...,4]!=1)):
        raise ValueError("invalid full five-action logits/physical mask")
    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):
        selected=np.where(m>.5,z,-1.e9)
        exp=np.exp(selected-selected.max(-1,keepdims=True));p=exp/exp.sum(-1,keepdims=True)
    if not np.isfinite(p).all() or np.any(p[m==0]!=0):raise RuntimeError("legacy target arithmetic produces illegal/nonfinite probability")
    return p


def _arrays(rows):
    rows=list(rows)
    if not rows:raise ValueError("program evidence needs nonempty whole-episode rows")
    obs=np.asarray([r["obs"] for r in rows],np.float32);m=np.asarray([r["mask"] for r in rows],np.float32)
    if obs.shape!=(len(rows),OBSERVATION_DIM) or m.shape!=(len(rows),5) or not np.isfinite(obs).all():raise ValueError("invalid public observations/masks")
    if np.any((m!=0)&(m!=1)) or np.any(m[:,4]!=1):raise ValueError("invalid physical mask")
    x=encode_actor_tree_features(obs,m)
    validate_certification_domain(x,MASK_INDICES,certified_feature_bounds(tuple(FEATURE_NAMES)))
    if any("encoded" in r and not np.array_equal(np.asarray(r["encoded"],np.float32),x[i]) for i,r in enumerate(rows)):
        raise ValueError("encoded features differ from public observations")
    if any(r.get("sampled_action") not in ACTIONS or "episode" not in r or "base_seed" not in r for r in rows):
        raise ValueError("actual sampled NN actions and whole episode/scenario IDs required")
    actions=np.asarray([ACTIONS.index(r["sampled_action"]) for r in rows],np.int64)
    if np.any(m[np.arange(len(m)),actions]!=1):raise ValueError("recorded sampled NN action is physically illegal")
    return x,m,actions


def _role_actor(actor,role):
    if role not in AGENTS:raise ValueError("invalid role")
    value=actor.roles[role]
    if (getattr(value,"model_version",None)!=MODEL_VERSION or value.feature_dim!=FEATURE_DIM
            or tuple(value.feature_names)!=tuple(FEATURE_NAMES) or not _is_sha(value.role_sha256)):
        raise RuntimeError("v6.22 program requires the unchanged explicit v6.18 full-NN schema/role hash")
    return value


def _wilson(p,n):
    if not n:return 0.
    z=1.645
    return max(0.,(p+z*z/(2*n)-z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/(1+z*z/n))


def _metrics(program,rows,arrays,p,selected,purpose):
    x,m,sampled=arrays;route=program.leaf_indices(x);q=masked_probabilities(program.leaf_logits[route],m)
    raw=program.leaf_logits[route].argmax(-1);top=q.argmax(-1);nn=p.argmax(-1)
    cert=program.certificates();legal=m[np.arange(len(m)),raw]>0
    path=np.asarray([int(a) in cert[int(l)] for a,l in zip(raw,route,strict=True)])
    chosen=np.isin(route,tuple(selected));n=int(chosen.sum());valid=legal&path
    matched=valid&(top==sampled);expected=valid*p[np.arange(len(p)),top]
    if purpose=="feedback":matched&=sampled==nn;expected*=top==nn
    # Keep the old floating arithmetic and floor for an isolated capacity change.
    with np.errstate(under="ignore"):
        kl=np.sum(p*(np.log(np.maximum(p,np.float32(1e-8)))-np.log(np.maximum(q,np.float32(1e-8)))),axis=-1)
    episodes=[str(r["episode"]) for r,keep in zip(rows,chosen,strict=True) if keep]
    scenes={str(r["base_seed"]) for r,keep in zip(rows,chosen,strict=True) if keep}
    frequencies=Counter(episodes);agreement=float((top[chosen]==nn[chosen]).mean()) if n else 0.
    metric={"sample_count":len(rows),"physical_frame_count":len(rows),"excluded_frames":0,
        "rows":n,"coverage":n/len(rows),"actionable_rows":int((chosen&matched).sum()),
        "actionable_coverage":float((chosen&matched).sum()/len(rows)),
        "expected_actionable_rows":float(expected[chosen].sum()),"expected_actionable_coverage":float(expected[chosen].sum()/len(rows)),
        "agreement":agreement,"agreement_wilson_lower_90":_wilson(agreement,n),"kl":float(kl[chosen].mean()) if n else None,
        "illegal_count":int((chosen&~legal).sum()),"masked_illegal_count":int((chosen&(m[np.arange(len(m)),top]==0)).sum()),
        "path_uncertified_count":int((chosen&~path).sum()),"whole_tree_raw_illegal_count":int((~legal).sum()),
        "whole_tree_path_uncertified_count":int((~path).sum()),"certified_rows":int((chosen&valid).sum()),
        "distinct_episode_count":len(frequencies),"distinct_scenario_seed_count":len(scenes),
        "max_episode_fraction":max(frequencies.values())/n if n else 0.,
        "whole_tree_agreement":float((top==nn).mean()),"whole_tree_kl":float(kl.mean()),
        "per_nn_action":{a:{"physical_frames":int((nn==i).sum()),"rows":int((chosen&(nn==i)).sum()),
            "qualified_coverage":float((chosen&(nn==i)).sum()/(nn==i).sum()) if (nn==i).any() else None,
            "agreement":float((top[chosen&(nn==i)]==nn[chosen&(nn==i)]).mean()) if (chosen&(nn==i)).any() else None,
            "actual_actionable_rows":int((chosen&matched&(nn==i)).sum()),
            "expected_actionable_rows":float(expected[chosen&(nn==i)].sum())} for i,a in enumerate(ACTIONS)}}
    conflict=x[:,FEATURE_NAMES.index("same_target_collision_streak_capped")]>0
    near=x[:,FEATURE_NAMES.index("robots_shortest_distance")]*np.float32(42)<=2
    low=x[:,FEATURE_NAMES.index("self_charger_return_margin")]*np.float32(50)<=4
    metric["public_strata"]={}
    for label,take in {"ordinary":~(conflict|near|low),"conflict":conflict,"near_robots":near,"low_return_margin":low}.items():
        selected=take&chosen;denominator=int(take.sum());support=int(selected.sum())
        metric["public_strata"][label]={"rows":support,"physical_frames":denominator,
            "qualified_coverage":support/denominator if denominator else None,
            "agreement":float((top[selected]==nn[selected]).mean()) if support else None,
            "kl":float(kl[selected].mean()) if support else None,"illegal_count":int((selected&~legal).sum()),
            "actual_actionable_rows":int((selected&matched).sum()),"expected_actionable_rows":float(expected[selected].sum())}
    return metric


def _eligible(m,threshold):
    return bool(m["rows"]>=20 and m["distinct_episode_count"]>=3 and m["distinct_scenario_seed_count"]>=3
        and m["max_episode_fraction"]<=.5 and m["agreement"]>=threshold
        and m["agreement_wilson_lower_90"]>=threshold-.08 and m["kl"] is not None and m["kl"]<=.25
        and m["illegal_count"]==0 and m["path_uncertified_count"]==0)


def gate_metrics(metrics, *, phase="development"):
    phases={"development":(.45,.40,.95,.97,.20),"precheck":(.45,.40,.95,.97,.20),
        "pressure":(.30,.30,.93,.95,.25),"formal":(.20,None,.85,.90,.35)}
    if phase not in phases:raise ValueError("unknown v6.22 program evidence phase")
    coverage,expected,fa,ea,kl=phases[phase];f,e=metrics["feedback"],metrics["explanation"]
    checks={"feedback_coverage":f["coverage"]>=coverage,"explanation_coverage":e["coverage"]>=coverage,
        "feedback_agreement":f["agreement"]>=fa,"explanation_agreement":e["agreement"]>=ea,
        "feedback_kl":f["kl"] is not None and math.isfinite(f["kl"]) and f["kl"]<=kl,
        "explanation_kl":e["kl"] is not None and math.isfinite(e["kl"]) and e["kl"]<=kl}
    if expected is not None:checks["expected_feedback_coverage"]=f["expected_actionable_coverage"]>=expected
    if phase=="pressure":
        checks.update({"feedback_actionable_coverage":f["actionable_coverage"]>=coverage,
            "explanation_actionable_coverage":e["actionable_coverage"]>=coverage})
    for purpose,m in (("feedback",f),("explanation",e)):
        for key in ("whole_tree_raw_illegal_count","whole_tree_path_uncertified_count","illegal_count","path_uncertified_count"):
            checks[purpose+"_"+key]=type(m[key]) is int and m[key]==0
    return {"passed":all(checks.values()),"checks":checks}


def _validate_frozen_selection(certificate, fits, initial, tree, proofs, counts):
    """Bind selected DP terminals to this exported tree, not just a SHA label.

    The whole frozen table and its optimality proof remain the search report's
    responsibility. This program retains and checks the selected table rows;
    it never upgrades their limited certificate to a continuous optimum.
    """
    _,capacity=normalize_structure(tree)
    n=tree["leaf_count"]
    paths=[[0 if step["branch"]=="left" else 1 for step in capacity["leaf_paths"][i]] for i in range(n)]
    node_ids=certificate.get("selected_table_node_ids")
    if (not isinstance(fits,list) or len(fits)!=n or certificate.get("selected_terminal_paths")!=paths
            or not isinstance(node_ids,list) or len(node_ids)!=n or len(set(node_ids))!=n
            or any(type(v) is not int or v<0 for v in node_ids)
            or certificate.get("selected_leaf_logits_sha256")!=sha256(np.ascontiguousarray(initial,dtype="<f4").tobytes()).hexdigest()
            or not _is_sha(certificate.get("selected_table_sha256"))
            or type(certificate.get("restricted_table_gap")) not in (int,float) or certificate["restricted_table_gap"]!=0
            or certificate.get("global_function_class_optimality_gap","missing") is not None):
        raise RuntimeError("D32 selected frozen-table certificate does not match exported paths/logits")
    for i,item in enumerate(fits):
        detail=item.get("optimizer",{})
        if (item.get("leaf")!=i or item.get("rows")!=counts[i] or detail.get("row_count")!=counts[i]
                or item.get("table_node_id")!=node_ids[i] or item.get("selected_terminal_path")!=paths[i]
                or set(item.get("path_legal_actions",()))!=set(proofs[i])
                or item.get("raw_action")!=int(initial[i].argmax()) or detail.get("chosen_action")!=int(initial[i].argmax())
                or not float32_preference_proof(initial[i],int(initial[i].argmax()),proofs[i])["passed"]):
            raise RuntimeError("D32 selected frozen table leaf identity/route/path/count differs")


@dataclass
class WarehouseV622Program:
    role: str
    actor_role_sha256: str
    structure: dict
    leaf_logits: np.ndarray
    feedback_leaves: frozenset
    explanation_leaves: frozenset
    report: dict
    fit_method: str = DEFAULT_METHOD
    limits: ProgramLimits = DEFAULT_LIMITS

    @property
    def feature_names(self):return FEATURE_NAMES
    @property
    def feature_dim(self):return FEATURE_DIM

    def certificates(self):
        tree,_=normalize_structure(self.structure,limits=self.limits)
        return _certificates(tree)

    def leaf_indices(self,encoded):
        tree,capacity=normalize_structure(self.structure,limits=self.limits)
        x=np.asarray(encoded,np.float32)
        if x.ndim!=2 or x.shape[1]!=FEATURE_DIM:raise ValueError("program features need [physical frame,feature]")
        validate_certification_domain(x,MASK_INDICES,certified_feature_bounds(tuple(FEATURE_NAMES)))
        return _route(tree,x,capacity["depth"])

    def validate(self):
        tree,capacity=normalize_structure(self.structure,limits=self.limits);n=tree["leaf_count"]
        if (self.role not in AGENTS or not _is_sha(self.actor_role_sha256) or self.fit_method not in FIT_METHODS
                or self.leaf_logits.dtype!=np.float32 or self.leaf_logits.shape!=(n,5) or not np.isfinite(self.leaf_logits).all()
                or not isinstance(self.feedback_leaves,frozenset) or not isinstance(self.explanation_leaves,frozenset)
                or any(type(i) is not int for i in self.feedback_leaves|self.explanation_leaves)
                or not (self.feedback_leaves|self.explanation_leaves)<=set(range(n))):raise RuntimeError("invalid typed v6.22 program tensors/method/binding")
        r=self.report;proof=self.certificates();keys={str(i) for i in range(n)}
        if (r.get("version")!=VERSION or r.get("fit_method")!=self.fit_method or r.get("actor_role_sha256")!=self.actor_role_sha256
                or r.get("feature_schema_sha256")!=FEATURE_SCHEMA_SHA256 or r.get("action_physics_config_sha256")!=ACTION_PHYSICS_CONFIG_SHA256
                or r.get("program_structure_sha256")!=program_structure_sha256(tree) or r.get("limits")!=self.limits.to_dict()
                or r.get("routing_contract")!=HARD_SPLIT_ROUTING_VERSION or r.get("target_arithmetic")!=TARGET_ARITHMETIC
                or r.get("full_NN_distribution_targets") is not True or r.get("external_program_selects_execution_action") is not False
                or r.get("excluded_frames")!=0 or r.get("leaf_count")!=n or r.get("max_depth")!=capacity["depth"]):
            raise RuntimeError("v6.22 program report lost version, schema, structure or full-policy binding")
        if (set(r.get("leaf_fits",{}))!=keys or set(r.get("leaf_calibrations",{}))!=keys
                or set(r.get("leaf_eligibility",{}))!=keys or len(r.get("fit_leaf_row_counts",[]))!=n):raise RuntimeError("incomplete all-leaf fitting evidence")
        for i in range(n):
            f,c=r["leaf_fits"][str(i)],r["leaf_calibrations"][str(i)];a=int(self.leaf_logits[i].argmax());count=r["fit_leaf_row_counts"][i]
            if (not float32_preference_proof(self.leaf_logits[i],a,proof[i])["passed"]
                    or f.get("success") is not True or f.get("float32_feasible") is not True
                    or f.get("chosen_action")!=a or c.get("chosen_action")!=a or c.get("temperature") not in CALIBRATION_TEMPERATURES
                    or type(count) is not int or count<0 or f.get("row_count")!=count
                    or f.get("solver")!="SLSQP" or f.get("solver_status")!=0 or f.get("epsilon")!=LEAF_EPSILON
                    or f.get("regularization")!=LEAF_REGULARIZATION or f.get("maxiter")!=LEAF_MAXITER or f.get("tolerance")!=LEAF_TOLERANCE
                    or type(f.get("enumerated_action_solves")) is not int or f["enumerated_action_solves"]!=len(proof[i])):
                raise RuntimeError("final refitted/calibrated raw action lacks its actual path/float32 certificate")
        frozen=self.fit_method==D32_METHOD
        actual_calls=0 if frozen else n
        if (r.get("leaf_optimization_calls")!=actual_calls or r.get("leaf_optimization_calls_outside_search")!=actual_calls
                or r.get("convex_solver_calls")!=(0 if frozen else sum(f["enumerated_action_solves"] for f in r["leaf_fits"].values()))):
            raise RuntimeError("actual final leaf API/solver accounting differs")
        if frozen:
            certificate=r.get("frozen_table_certificate",{})
            initial=np.asarray(r.get("precalibration_leaf_logits"),np.float32)
            if (initial.shape!=(n,5) or not np.isfinite(initial).all()
                    or certificate.get("selected_leaf_logits_sha256")!=sha256(np.ascontiguousarray(initial,dtype="<f4").tobytes()).hexdigest()
                    or not _is_sha(certificate.get("selected_table_sha256"))
                    or certificate.get("restricted_table_gap")!=0
                    or certificate.get("global_function_class_optimality_gap","missing") is not None
                    or r.get("DP_certificate_scope")!="frozen_precalibration_table_only_not_final_continuous_optimum"):
                raise RuntimeError("D32 lacks its bound frozen-table-only certificate")
            fits=r.get("frozen_fit_leaf_reports")
            if r.get("frozen_fit_leaf_reports_sha256")!=digest(fits):raise RuntimeError("D32 selected frozen leaf evidence SHA changed")
            _validate_frozen_selection(certificate,fits,initial,tree,proof,r["fit_leaf_row_counts"])
            for i in range(n):
                if r["leaf_fits"][str(i)]!=fits[i]["optimizer"]:raise RuntimeError("D32 optimizer evidence differs from selected frozen table")
                c=r["leaf_calibrations"][str(i)];trial=np.asarray(initial[i]/c["temperature"],np.float32)
                if not np.array_equal(trial,self.leaf_logits[i]):raise RuntimeError("D32 final logits are not its frozen outputs plus recorded calibration")
        budget=r.get("fit_budget",{})
        if (type(budget.get("prior_leaf_optimization_calls")) is not int or budget["prior_leaf_optimization_calls"]<0
                or budget.get("actual_final_leaf_optimization_calls")!=actual_calls
                or budget.get("actual_total_leaf_optimization_calls")!=budget["prior_leaf_optimization_calls"]+actual_calls
                or actual_calls>budget.get("max_leaf_optimization_calls",-1)
                or budget["actual_total_leaf_optimization_calls"]>budget.get("max_total_leaf_optimization_calls",-1)):
            raise RuntimeError("program final API budget mismatch")
        for purpose,chosen,threshold in (("feedback",self.feedback_leaves,.93),("explanation",self.explanation_leaves,.95)):
            expected=frozenset(i for i in range(n) if r["fit_leaf_row_counts"][i] and _eligible(r["leaf_eligibility"][str(i)][purpose],threshold))
            if chosen!=expected:raise RuntimeError("qualification must use independent eligibility only")
        for key in ("split_episode_ids","split_scenario_seed_ids"):
            groups=r.get(key,{})
            if set(groups)!=set(SPLITS) or any(not isinstance(v,list) or not v or v!=sorted(set(v)) for v in groups.values()):
                raise RuntimeError("four complete canonical episode/scenario partitions required")
            if any(set(groups[a])&set(groups[b]) for a in SPLITS for b in SPLITS if a<b):raise RuntimeError("program episode/scenario partitions overlap")
        validation=r.get("validation",{})
        if validation.get("gate")!=gate_metrics(validation,phase=r.get("validation_phase")):
            raise RuntimeError("stored validation gate does not match complete evidence")
        for purpose in ("feedback","explanation"):
            if any(type(validation[purpose].get(k)) is not int or validation[purpose][k]!=0
                   for k in ("whole_tree_raw_illegal_count","whole_tree_path_uncertified_count")):
                raise RuntimeError("explicit whole-tree raw legality evidence is required")
        return self

    def to_dict(self):
        self.validate()
        return {"version":VERSION,"method":self.fit_method,"role":self.role,"actor_role_sha256":self.actor_role_sha256,
            "feature_schema_sha256":FEATURE_SCHEMA_SHA256,"action_physics_config_sha256":ACTION_PHYSICS_CONFIG_SHA256,
            "limits":self.limits.to_dict(),"structure":plain(self.structure),"leaf_logits":self.leaf_logits.tolist(),
            "feedback_leaves":sorted(self.feedback_leaves),"explanation_leaves":sorted(self.explanation_leaves),"report":plain(self.report)}

    def sha256(self):return digest(self.to_dict())

    def save(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open("x") as f:json.dump(self.to_dict(),f,sort_keys=True,separators=(",",":"),allow_nan=False)
        return sha256(p.read_bytes()).hexdigest()

    @classmethod
    def from_dict(cls,data,*,actor_role_sha256,role=None,expected_limits=DEFAULT_LIMITS,expected_fit_method=None):
        required={"version","method","role","actor_role_sha256","feature_schema_sha256","action_physics_config_sha256",
            "limits","structure","leaf_logits","feedback_leaves","explanation_leaves","report"}
        if (not isinstance(data,Mapping) or set(data)!=required or data.get("version")!=VERSION or data.get("method") not in FIT_METHODS
                or data.get("actor_role_sha256")!=actor_role_sha256 or data.get("feature_schema_sha256")!=FEATURE_SCHEMA_SHA256
                or data.get("action_physics_config_sha256")!=ACTION_PHYSICS_CONFIG_SHA256
                or role is not None and data.get("role")!=role
                or expected_fit_method is not None and data.get("method")!=expected_fit_method):raise RuntimeError("unknown program version/method/schema/role/hash")
        limits=ProgramLimits.from_dict(data["limits"])
        if limits!=expected_limits:raise RuntimeError("program limits differ from explicitly expected v6.22 limits")
        tree,_=normalize_structure(data["structure"],limits=limits)
        sets=[]
        for key in ("feedback_leaves","explanation_leaves"):
            z=data[key]
            if not isinstance(z,list) or any(type(i) is not int for i in z) or z!=sorted(set(z)):raise RuntimeError("invalid qualification list")
            sets.append(frozenset(z))
        logits=np.asarray(data["leaf_logits"],np.float32);logits.flags.writeable=False
        for value in tree.values():
            if isinstance(value,np.ndarray):value.flags.writeable=False
        return cls(data["role"],actor_role_sha256,tree,logits,*sets,deepcopy(data["report"]),data["method"],limits).validate()

    @classmethod
    def load(cls,path,*,expected_sha256,actor_role_sha256,role=None,expected_limits=DEFAULT_LIMITS,expected_fit_method=None):
        raw=Path(path).read_bytes()
        if not _is_sha(expected_sha256) or sha256(raw).hexdigest()!=expected_sha256:raise RuntimeError("program file SHA mismatch")
        return cls.from_dict(json.loads(raw),actor_role_sha256=actor_role_sha256,role=role,expected_limits=expected_limits,expected_fit_method=expected_fit_method)


def _fit_budget(value,n,*,frozen=False):
    value={} if value is None else dict(value)
    allowed={"max_leaf_optimization_calls","maximum_leaf_api_calls","prior_leaf_optimization_calls","max_total_leaf_optimization_calls","provenance_sha256","deadline_monotonic"}
    if set(value)-allowed:raise ValueError("unknown final-fit budget field")
    if "max_leaf_optimization_calls" in value and "maximum_leaf_api_calls" in value and value["max_leaf_optimization_calls"]!=value["maximum_leaf_api_calls"]:
        raise ValueError("conflicting final leaf API budget aliases")
    calls=0 if frozen else n
    result={"max_leaf_optimization_calls":value.get("max_leaf_optimization_calls",value.get("maximum_leaf_api_calls",0 if frozen else 32)),
        "prior_leaf_optimization_calls":value.get("prior_leaf_optimization_calls",0),
        "max_total_leaf_optimization_calls":value.get("max_total_leaf_optimization_calls",32),
        "provenance_sha256":value.get("provenance_sha256"),"deadline_monotonic":value.get("deadline_monotonic")}
    if (any(type(result[k]) is not int or result[k]<0 for k in ("max_leaf_optimization_calls","prior_leaf_optimization_calls","max_total_leaf_optimization_calls"))
            or result["max_leaf_optimization_calls"]>32 or result["max_total_leaf_optimization_calls"]>20540
            or result["max_leaf_optimization_calls"]<calls or result["prior_leaf_optimization_calls"]+calls>result["max_total_leaf_optimization_calls"]
            or result["deadline_monotonic"] is not None and (type(result["deadline_monotonic"]) not in (int,float) or not math.isfinite(result["deadline_monotonic"]))
            or result["provenance_sha256"] is not None and not _is_sha(result["provenance_sha256"])):
        raise ValueError("insufficient/unregistered final-fit API budget before any leaf solve")
    return result


def fit_program(role,actor,fit,calibration,eligibility,validation,*,structure,structure_sha256,
                limits=DEFAULT_LIMITS,fit_method=DEFAULT_METHOD,fit_budget=None,validation_phase="development",frozen_leaf_outputs=None):
    if fit_method not in FIT_METHODS:raise ValueError("unknown v6.22 fitting method")
    role_actor=_role_actor(actor,role);actor_sha=role_actor.role_sha256
    tree,capacity=normalize_structure(structure,limits=limits);n=tree["leaf_count"]
    if program_structure_sha256(tree)!=structure_sha256:raise RuntimeError("fixed structure hash mismatch")
    frozen=fit_method==D32_METHOD
    if frozen_leaf_outputs is not None and (type(frozen_leaf_outputs) is not bool or frozen_leaf_outputs!=frozen):raise ValueError("only D32 uses frozen table outputs; D32 cannot silently refit them")
    budget=_fit_budget(fit_budget,n,frozen=frozen)
    report={"leaf_optimization_calls":0,"leaf_optimization_calls_outside_search":0,"convex_solver_calls":0,"fit_budget":budget,
        "version":VERSION,"fit_method":fit_method,"actor_role_sha256":actor_sha}
    _deadline(report,"before_partition_arrays")
    parts=dict(zip(SPLITS,map(list,(fit,calibration,eligibility,validation)),strict=True))
    arrays={name:_arrays(rows) for name,rows in parts.items()}
    _deadline(report,"after_partition_arrays")
    ids={name:sorted({str(r["episode"]) for r in rows}) for name,rows in parts.items()}
    scenes={name:sorted({str(r["base_seed"]) for r in rows}) for name,rows in parts.items()}
    if any(set(groups[a])&set(groups[b]) for groups in (ids,scenes) for a in SPLITS for b in SPLITS if a<b):
        raise ValueError("fit/calibration/eligibility/validation must use independent whole episodes and scenes")
    proofs=_certificates(tree);routes={s:_route(tree,a[0],capacity["depth"]) for s,a in arrays.items()}
    targets={s:masked_probabilities(role_actor.logits_encoded(a[0]),a[1]) for s,a in arrays.items()}
    _deadline(report,"after_full_NN_targets")
    counts=np.bincount(routes["fit"],minlength=n)
    report.update({"version":VERSION,"fit_method":fit_method,"actor_role_sha256":actor_sha,
        "feature_schema_sha256":FEATURE_SCHEMA_SHA256,"action_physics_config_sha256":ACTION_PHYSICS_CONFIG_SHA256,
        "program_structure_sha256":structure_sha256,"limits":limits.to_dict(),"routing_contract":HARD_SPLIT_ROUTING_VERSION,
        "target_arithmetic":TARGET_ARITHMETIC,"full_NN_distribution_targets":True,
        "external_program_selects_execution_action":False,"excluded_frames":0,"leaf_count":n,"max_depth":capacity["depth"],
        "source_features_per_node":capacity["source_features_per_node"],"split_episode_ids":ids,"split_scenario_seed_ids":scenes,
        "fit_target_sha256":sha256(np.ascontiguousarray(targets["fit"],dtype="<f4").tobytes()).hexdigest(),
        "physical_frames_per_split":{s:len(parts[s]) for s in SPLITS},"fit_leaf_row_counts":counts.astype(int).tolist(),
        "leaf_fits":{},"leaf_calibrations":{},"leaf_eligibility":{},"validation":{},"validation_phase":validation_phase,
        "leaf_optimization_calls":0,"leaf_optimization_calls_outside_search":0,"convex_solver_calls":0,"fit_budget":budget})
    # Validate the phase before any solver call.
    if validation_phase not in ("development","precheck","pressure","formal"):raise ValueError("unknown evidence phase")
    z=[]
    if frozen:
        frozen_logits=np.asarray(structure.get("initial_leaf_logits"),np.float32)
        fits=structure.get("fit_leaf_reports");certificate=deepcopy(structure.get("frozen_table_certificate",{}))
        if (frozen_logits.shape!=(n,5) or not np.isfinite(frozen_logits).all() or not isinstance(fits,list) or len(fits)!=n
                or structure.get("fit_target_sha256")!=report["fit_target_sha256"]
                or certificate.get("selected_leaf_logits_sha256")!=sha256(np.ascontiguousarray(frozen_logits,dtype="<f4").tobytes()).hexdigest()
                or not _is_sha(certificate.get("selected_table_sha256")) or certificate.get("restricted_table_gap")!=0
                or certificate.get("global_function_class_optimality_gap","missing") is not None):
            raise ValueError("D32 requires complete selected frozen-table outputs, target and restricted certificate")
        try:_validate_frozen_selection(certificate,fits,frozen_logits,tree,proofs,counts.astype(int).tolist())
        except RuntimeError as error:raise ValueError("D32 frozen-table selection invalid: "+str(error)) from error
        report.update({"precalibration_leaf_logits":frozen_logits.tolist(),"frozen_table_certificate":certificate,
            "frozen_fit_leaf_reports":deepcopy(fits),"frozen_fit_leaf_reports_sha256":digest(fits),"frozen_leaf_outputs":True,
            "DP_certificate_scope":"frozen_precalibration_table_only_not_final_continuous_optimum"})
    try:
        for i in range(n):
            _deadline(report,"before_leaf_"+str(i))
            take=routes["fit"]==i
            if frozen:
                item=fits[i];detail=deepcopy(item.get("optimizer",{}));logits=frozen_logits[i].copy()
                if (item.get("leaf")!=i or item.get("rows")!=int(counts[i]) or detail.get("row_count")!=int(counts[i])
                        or set(item.get("path_legal_actions",()))!=set(proofs[i])
                        or item.get("raw_action")!=int(logits.argmax()) or detail.get("chosen_action")!=int(logits.argmax())):
                    raise RuntimeError("D32 frozen table leaf identity/route/path/count differs")
            else:
                report["leaf_optimization_calls"]+=1;report["leaf_optimization_calls_outside_search"]+=1
                try:
                    logits,detail=fit_constrained_leaf_distribution(targets["fit"][take],arrays["fit"][1][take],proofs[i])
                except Exception as error:
                    report["convex_solver_calls"]+=int(getattr(error,"enumerated_action_solves",0));raise
                report["convex_solver_calls"]+=detail["enumerated_action_solves"]
            report["leaf_fits"][str(i)]=detail
            _deadline(report,"after_leaf_fit_"+str(i))
            if not float32_preference_proof(logits,int(logits.argmax()),proofs[i])["passed"]:raise RuntimeError("new fitted top lacks path proof")
            take=routes["calibration"]==i
            logits,detail=calibrate_constrained_leaf_distribution(logits,targets["calibration"][take],arrays["calibration"][1][take],proofs[i],chosen_action=int(logits.argmax()))
            _deadline(report,"after_leaf_calibration_"+str(i))
            if not float32_preference_proof(logits,int(logits.argmax()),proofs[i])["passed"]:raise RuntimeError("new calibrated top lacks path proof")
            report["leaf_calibrations"][str(i)]=detail;z.append(logits)
    except ProgramBudgetExhausted:
        raise
    except Exception as error:
        report["status"]="failed";budget.update({"actual_final_leaf_optimization_calls":report["leaf_optimization_calls"],
            "actual_total_leaf_optimization_calls":budget["prior_leaf_optimization_calls"]+report["leaf_optimization_calls"]})
        raise ProgramFitError(f"v6.22 final leaf fit failed: {type(error).__name__}: {error}",report=report) from error
    actual=report["leaf_optimization_calls"]
    budget.update({"actual_final_leaf_optimization_calls":actual,"actual_total_leaf_optimization_calls":budget["prior_leaf_optimization_calls"]+actual})
    program=WarehouseV622Program(role,actor_sha,tree,np.stack(z).astype(np.float32),frozenset(),frozenset(),report,fit_method,limits)
    chosen={"feedback":set(),"explanation":set()}
    for i in range(n):
        _deadline(report,"before_leaf_qualification_"+str(i))
        metrics={purpose:_metrics(program,parts["eligibility"],arrays["eligibility"],targets["eligibility"],{i},purpose) for purpose in chosen}
        report["leaf_eligibility"][str(i)]=metrics
        for purpose,threshold in (("feedback",.93),("explanation",.95)):
            if counts[i] and _eligible(metrics[purpose],threshold):chosen[purpose].add(i)
        _deadline(report,"after_leaf_qualification_"+str(i))
    program.feedback_leaves=frozenset(chosen["feedback"]);program.explanation_leaves=frozenset(chosen["explanation"])
    _deadline(report,"before_validation")
    report["validation"]={purpose:_metrics(program,parts["validation"],arrays["validation"],targets["validation"],selected,purpose)
        for purpose,selected in (("feedback",program.feedback_leaves),("explanation",program.explanation_leaves))}
    report["validation"]["gate"]=gate_metrics(report["validation"],phase=validation_phase)
    _deadline(report,"after_validation")
    report["status"]="completed"
    if _role_actor(actor,role).role_sha256!=actor_sha:raise RuntimeError("Actor identity changed during surrogate fitting")
    program.validate()
    _deadline(report,"after_complete_program_validation")
    return program


def program_metrics(program,actor,rows):
    program.validate();role_actor=_role_actor(actor,program.role)
    if role_actor.role_sha256!=program.actor_role_sha256:raise RuntimeError("program is bound to another complete NN")
    rows=list(rows);a=_arrays(rows);p=masked_probabilities(role_actor.logits_encoded(a[0]),a[1])
    result={purpose:_metrics(program,rows,a,p,chosen,purpose) for purpose,chosen in (("feedback",program.feedback_leaves),("explanation",program.explanation_leaves))}
    result["per_leaf"]={str(i):{purpose:_metrics(program,rows,a,p,{i},purpose) for purpose in ("feedback","explanation")} for i in range(program.structure["leaf_count"])}
    return result


def current_actor_fidelity(program,actor,rows,*,purpose="feedback",phase=None):
    program.validate()
    if purpose not in ("feedback","explanation"):raise ValueError("unknown program evidence purpose")
    role_actor=_role_actor(actor,program.role);rows=list(rows);a=_arrays(rows)
    ids={str(r["episode"]) for r in rows};scenes={str(r["base_seed"]) for r in rows}
    if any(ids&set(program.report["split_episode_ids"][s]) or scenes&set(program.report["split_scenario_seed_ids"][s]) for s in ("fit","calibration","eligibility")):
        raise ValueError("current fidelity cannot reuse fit/calibration/eligibility episodes")
    p=masked_probabilities(role_actor.logits_encoded(a[0]),a[1]);metrics={which:_metrics(program,rows,a,p,selected,which)
        for which,selected in (("feedback",program.feedback_leaves),("explanation",program.explanation_leaves))}
    phase=program.report["validation_phase"] if phase is None else phase
    gate=gate_metrics(metrics,phase=phase)
    return {**metrics[purpose],"passed":bool(program.report["validation"]["gate"]["passed"] and gate["passed"]),
        "purpose":purpose,"phase":phase,"independent_validation_required":True,
        "current_actor_role_sha256":role_actor.role_sha256,"program_sha256":program.sha256(),
        "current_validation_gate":gate,"current_actor_topology_matches_program":None,
        "topology_relation":"external_surrogate_not_required_to_equal_internal_NN_tree"}


def feedback_evidence(program,actor,rows,*,current_fidelity=None):
    """Program distributions and predicates for a caller's NN-only loss.

    No action is selected here. The caller additionally matches actual sampled
    NN actions and current NN top, preserving the all-physical-frame denominator.
    """
    program.validate();role_actor=_role_actor(actor,program.role);rows=list(rows);a=_arrays(rows)
    if current_fidelity is None:
        if role_actor.role_sha256!=program.actor_role_sha256:raise RuntimeError("changed NN requires independent current fidelity")
        passed=program.report["validation"]["gate"]["passed"]
    else:
        if (current_fidelity.get("program_sha256")!=program.sha256()
                or current_fidelity.get("current_actor_role_sha256")!=role_actor.role_sha256
                or current_fidelity.get("purpose")!="feedback" or current_fidelity.get("independent_validation_required") is not True):
            raise RuntimeError("current program/NN fidelity receipt binding mismatch")
        passed=current_fidelity.get("passed") is True
    route=program.leaf_indices(a[0]);raw=program.leaf_logits[route].argmax(-1)
    legal=a[1][np.arange(len(rows)),raw]>0;cert=program.certificates()
    path=np.asarray([int(action) in cert[int(leaf)] for action,leaf in zip(raw,route,strict=True)])
    p=masked_probabilities(role_actor.logits_encoded(a[0]),a[1])
    return {"probabilities":masked_probabilities(program.leaf_logits[route],a[1]),
        "qualified":np.isin(route,tuple(program.feedback_leaves))&legal&path&bool(passed),
        "raw_program_top":raw,"nn_top":p.argmax(-1),"raw_legal":legal,"path_certified":path,
        "leaf_indices":route,"current_actor_role_sha256":role_actor.role_sha256,
        "program_sha256":program.sha256(),"independent_gate_passed":bool(passed)}
