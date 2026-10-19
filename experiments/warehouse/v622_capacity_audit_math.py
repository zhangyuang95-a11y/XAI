"""Independent receipt/compiler/frozen-table arithmetic; no search or fitting.

Only NumPy, the standard library, and the earlier independent auditor's tree
arithmetic are used.  A passed receipt checks recorded evidence, not admission.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math

import numpy as np

from scripts import audit_warehouse_v621_diagnostics as old

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
RECIPES = (
    ("probability", (.25,.50,.75), 0), ("log_probability", (.20,.40,.60,.80), 0),
    ("probability", (.15,.35,.65,.85), 1), ("log_probability", (.10,.30,.50,.70,.90), 1),
    ("probability", (.20,.50,.80), 2), ("log_probability", (.15,.40,.60,.85), 2),
    ("probability", (.10,.35,.65,.90), 3), ("log_probability", (.25,.50,.75), 3),
)


def _plain(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k):_plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_plain(v) for v in x]
    return x


def digest(x):
    return sha256(json.dumps(_plain(x), sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()


def _require(ok, message):
    if not ok: raise RuntimeError("independent v622 audit: "+message)


def _integer(x, name, maximum=None):
    _require(type(x) is int and x >= 0 and (maximum is None or x <= maximum), "invalid counter "+name)
    return x


def _sha(x):
    return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x)


def _equal(actual, expected, label):
    old.assert_equal(_plain(actual), _plain(expected), label)


def _arrays(value):
    if isinstance(value,dict):
        value = tuple(value[k] for k in ("encoded","masks","targets","episode_ids","feature_names","actor_role_sha256"))
    _require(isinstance(value,(tuple,list)) and len(value)==6, "fit_arrays requires x,m,p,episodes,names,rolehash")
    x,m,p = (np.asarray(a,np.float32) for a in value[:3])
    episodes,names,role = np.asarray([str(a) for a in value[3]]),tuple(value[4]),value[5]
    masks = [names.index("legal_"+a.lower()) for a in ACTIONS]
    _require(x.ndim==2 and x.shape[1]==len(names) and m.shape==p.shape==(len(x),5)
        and len(x)>0 and episodes.shape==(len(x),) and _sha(role)
        and len(set(names))==len(names) and all(np.isfinite(a).all() for a in (x,m,p))
        and np.all((m==0)|(m==1)) and np.all(m[:,4]==1)
        and np.array_equal(x[:,masks],m) and np.all(p>=0) and np.all(p[m==0]==0)
        and np.allclose(p.sum(-1),1,atol=1e-6,rtol=0) and all(episodes), "invalid bound fit arrays")
    return x,m,p,episodes,names,role


def _identity(arrays):
    x,m,p,e,names,role = arrays
    return {"actor_role_sha256":role,"feature_names":list(names),
        "fit_encoded_sha256":sha256(np.ascontiguousarray(x,dtype="<f4").tobytes()).hexdigest(),
        "fit_masks_sha256":sha256(np.ascontiguousarray(m,dtype="<f4").tobytes()).hexdigest(),
        "fit_target_sha256":sha256(np.ascontiguousarray(p,dtype="<f4").tobytes()).hexdigest(),
        "fit_episode_ids_sha256":digest(e.tolist())}


def _attempts(report, label):
    calls = _integer(report.get("leaf_optimization_calls"),label+".calls")
    solves = _integer(report.get("actual_convex_solver_calls"),label+".solves",5*calls)
    events = report.get("leaf_api_attempts")
    _require(isinstance(events,list) and len(events)==calls, label+" missing complete leaf API ledger")
    total=0
    for i,e in enumerate(events):
        _require(e.get("call")==i+1 and type(e.get("call")) is int and e.get("cache_miss") is True
            and type(e.get("success")) is bool and _sha(e.get("rows_sha256")), label+" malformed attempt")
        _integer(e.get("rows"),label+".attempt.rows")
        legal=e.get("path_legal_actions")
        _require(isinstance(legal,list) and legal==sorted(set(legal)) and 4 in legal
            and all(type(a) is int and 0<=a<5 for a in legal), label+" invalid path action ledger")
        n=_integer(e.get("actual_convex_solver_calls"),label+".attempt.solves",len(legal))
        _require(not e["success"] or n==len(legal), label+" successful API omitted action solves")
        _require(e["success"] or isinstance(e.get("error"),str), label+" failed API lacks error")
        total+=n
    _require(total==solves, label+" solver sum mismatch")
    return calls,solves


def audit_search_budget(search, arm, final_report, fit_arrays=None):
    """Check complete search ledgers; incomplete jobs need a separate failure audit."""
    _require(arm in ("A32","B32","D32"),"unknown arm")
    final=_integer(final_report.get("leaf_optimization_calls_outside_search"),"final calls",32)
    leaves=_integer(final_report.get("leaf_count"),"leaf count",32)
    _require(leaves>=1 and final==(0 if arm=="D32" else leaves),"final leaf calls differ")
    _require(_integer(final_report.get("leaf_optimization_calls"),"final API alias")==final,"final API aliases disagree")
    fits=final_report.get("leaf_fits")
    _require(isinstance(fits,dict) and set(fits)=={str(i) for i in range(leaves)},"incomplete final leaf fit evidence")
    final_solves=sum(_integer(f.get("enumerated_action_solves"),"final solves",5) for f in fits.values()) if arm!="D32" else 0
    _require(_integer(final_report.get("convex_solver_calls"),"final solver total")==final_solves,"final solver ledger differs")
    if arm=="B32":
        _require(type(search.get("leaf_optimization_calls")) is int and search["leaf_optimization_calls"]==0
            and type(search.get("global_search_rounds")) is int and search["global_search_rounds"]==0,
            "fixed B32 topology unexpectedly searched")
        return {"passed":True,"search_leaf_api_calls":0,"final_leaf_api_calls":final,"total_leaf_api_calls":final}
    saved=search.get("report_sha256")
    _require(_sha(saved) and saved==digest({k:v for k,v in search.items() if k!="report_sha256"}),"search report digest differs")
    _require(search.get("search_uses_fit_only") is True and search.get("calibration_eligibility_validation_used") is False
        and search.get("complete_incumbent") is True and search.get("execution_action_override") is False,
        "search fit-only/complete-incumbent authority differs")
    if fit_arrays is not None:
        identity=_identity(_arrays(fit_arrays))
        _equal(search.get("input_identity"),identity,"search.input_identity")
        _require(search.get("fit_rows")==len(_arrays(fit_arrays)[0]) and search.get("actor_role_sha256")==identity["actor_role_sha256"],"search role/fit rows differ")
    cfg=search.get("search_budget",{})
    for key,value in {"initializations":8,"max_rounds":60,"max_leaf_optimization_calls":19968,
                      "max_depth":6,"max_leaves":32,"template_count":4,
                      "template_max_leaf_optimization_calls":20000,"table_max_leaf_optimization_calls":508}.items():
        _require(type(cfg.get(key)) is int and cfg[key]==value,"unregistered search budget "+key)
    if arm=="A32":
        calls,solves=_attempts(search,"A32")
        _require(calls<=19968 and calls+final<=20000,"A32 end-to-end budget exceeded")
        runs=search.get("search_runs")
        _require(isinstance(runs,list) and len(runs)==8,"A32 must account for all eight starts")
        prefix=_integer(search.get("bootstrap_leaf_optimization_calls"),"bootstrap.calls",calls)
        solve_prefix=_integer(search.get("bootstrap_actual_convex_solver_calls"),"bootstrap.solves",solves)
        _require(prefix==1,"shared safe root must be one API call")
        _require(solve_prefix==search['leaf_api_attempts'][0]['actual_convex_solver_calls'],"bootstrap solver attribution differs")
        cumulative_rounds=0
        for i,run in enumerate(runs):
            response,quantiles,contrast=RECIPES[i]
            _equal(run.get("initialization"),{"id":i,"response":response,"quantiles":list(quantiles),"contrast":contrast},"initialization")
            count=_integer(run.get("leaf_optimization_calls"),"run.calls",calls)
            nsolves=_integer(run.get("actual_convex_solver_calls"),"run.solves",5*count)
            _require(nsolves==sum(e['actual_convex_solver_calls'] for e in search['leaf_api_attempts'][prefix:prefix+count]),"run solver attribution differs")
            history=run.get("rounds")
            _require(isinstance(history,list),"run lacks round log")
            if run.get("initialization_started") is False:
                _require(count==nsolves==len(history)==0 and run.get("stop_reason")=="wall_time_budget"
                    and search.get("wall_time_budget_exhausted") is True,"unstarted A32 run without actual wall exhaustion")
            else:
                for key in ('global_rounds_before_start','reserved_cumulative_round_limit','reserved_cumulative_leaf_call_limit',
                            'round_budget_available_at_start','global_rounds_after_start'):_integer(run.get(key),'run.'+key)
                _require(run.get("global_rounds_before_start")==cumulative_rounds
                    and run.get("reserved_cumulative_round_limit")==((i+1)*60)//8
                    and run.get("reserved_cumulative_leaf_call_limit")==((i+1)*19968)//8
                    and run.get("round_budget_available_at_start")==((i+1)*60)//8-cumulative_rounds,
                    "A32 cumulative reservation log differs")
                last=prefix
                for j,row in enumerate(history):
                    _require(type(row.get("round")) is int and row["round"]==j+1
                        and type(row.get("global_round")) is int and row["global_round"]==cumulative_rounds+j+1,
                        "A32 global/within-run round log differs")
                    current=_integer(row.get("leaf_optimization_calls_cumulative"),"round.calls",prefix+count)
                    _require(current>=last,"A32 cumulative calls decreased");last=current
                cumulative_rounds+=len(history)
                _require(run.get("global_rounds_after_start")==cumulative_rounds
                    and cumulative_rounds<=((i+1)*60)//8 and prefix+count<=((i+1)*19968)//8,
                    "A32 run exceeded its cumulative share")
            prefix+=count;solve_prefix+=nsolves
        _require(prefix==calls and solve_prefix==solves,"A32 bootstrap/run API or solver sum differs")
        _require(_integer(search.get("global_search_rounds"),"rounds",60)==cumulative_rounds,"A32 global rounds differ")
    else:
        generation,gsolves=_attempts(search.get("generation",{}),"D32.generation")
        table,tsolves=_attempts(search.get("terminal_table",{}),"D32.table")
        calls=_integer(search.get("leaf_optimization_calls"),"D32.total",20540)
        for key in ('template_generation_leaf_optimization_calls','frozen_table_leaf_optimization_calls','actual_convex_solver_calls',
                    'final_refit_reserved_calls','unused_final_refit_allowance'):_integer(search.get(key),'D32.'+key)
        _require(generation<=20000 and table<=508 and table+final<=540
            and search.get("template_generation_leaf_optimization_calls")==generation
            and search.get("frozen_table_leaf_optimization_calls")==table and calls==generation+table
            and search.get("actual_convex_solver_calls")==gsolves+tsolves,
            "D32 phase/API/solver accounting differs")
        _require(search.get("final_leaf_outputs_frozen_from_selected_table") is True
            and search.get("final_refit_reserved_calls")==0 and search.get("unused_final_refit_allowance")==32,
            "D32 unexpectedly refitted selected outputs")
        _require(type(search.get('global_search_rounds')) is int and search['global_search_rounds']==0
            and search.get('search_runs')==[],'D32 unexpectedly ran M2 rounds')
        histories=search.get('template_generation_reports')
        _require(isinstance(histories,list) and 1<=len(histories)<=4,'D32 generation history missing')
        spent=0
        for i,row in enumerate(histories):
            _require(type(row.get('template')) is int and row['template']==i
                and type(row.get('completed_structure')) is bool,'D32 incomplete template history')
            spent+=_integer(row.get('generation_calls'),'D32.template.calls')
        _require(spent==generation,'D32 generation history API sum differs')
    return {"passed":True,"search_leaf_api_calls":calls,"final_leaf_api_calls":final,
            "total_leaf_api_calls":calls+final,"input_identity_checked":fit_arrays is not None}


def audit_compiler(report, role_actor, structure):
    """Rebuild old hard topology plus necessary guards, independently."""
    names=tuple(role_actor.feature_names);values=role_actor.weights
    n,c=int(values["tree.node_count"][0]),int(values["tree.leaf_count"][0])
    source={"root_ref":0 if n else -1,"split_weights":values["tree.split.weight"][:n],
        "split_bias":values["tree.split.bias"][:n],"left_refs":values["tree.left_refs"][:n],
        "right_refs":values["tree.right_refs"][:n],"leaf_count":c,"feature_dim":len(names)}
    tree,capacity=old.structure(source,names,enforce_limits=False)
    proof=old.certificates(tree,names,explicit_crossing=True)
    z=np.asarray(values["tree.leaf_logits"],np.float32)[:c]
    _require(z.shape==(c,5) and np.isfinite(z).all(),"old Actor leaf outputs invalid")
    w,b,left,right,origins,decisions=[],[],[],[],[],[]
    def leaf(index,action,branch):
        result=-1-len(origins)
        origins.append({"compiled_leaf":len(origins),"original_leaf":index,"guard_branch":branch,"frozen_preferred_action":ACTIONS[action]})
        return result
    def copy(ref):
        if ref<0:
            index=-1-ref;a=int(z[index].argmax());guard=a not in proof[index]
            entry={"original_leaf":index,"frozen_preferred_action":ACTIONS[a],"guard_added":guard,
                "original_path":capacity["leaf_paths"][index],"original_path_legal_actions":sorted(proof[index])}
            decisions.append(entry)
            if not guard:return leaf(index,a,"unchanged")
            node=len(w);weight=np.zeros(len(names),np.float32);feature=names.index("legal_"+ACTIONS[a].lower());weight[feature]=1.
            w.append(weight);b.append(np.float32(-.5));left.append(leaf(index,4,"illegal_false_wait"));right.append(leaf(index,a,"legal_true"))
            entry.update({"guard_node":node,"guard_feature":names[feature],"guard_bias":-.5,"guard_coefficient":1.})
            return node
        node=len(w);w.append(tree["split_weights"][ref]);b.append(tree["split_bias"][ref]);left.append(0);right.append(0)
        left[node]=copy(int(tree["left_refs"][ref]));right[node]=copy(int(tree["right_refs"][ref]));return node
    root=copy(tree["root_ref"])
    expected,cap=old.structure({"root_ref":root,"split_weights":np.asarray(w,np.float32).reshape(-1,len(names)),
        "split_bias":np.asarray(b,np.float32),"left_refs":left,"right_refs":right,"leaf_count":len(origins)},names,enforce_limits=False)
    _require(cap["depth"]<=6 and cap["leaf_count"]<=32 and cap["maximum_source_features_per_node"]<=3,"guarded B32 topology is infeasible")
    derived=old.certificates(expected,names,explicit_crossing=True)
    for i,origin in enumerate(origins):
        origin.update(path_legal_actions=sorted(derived[i]),compiled_path=cap["leaf_paths"][i])
    actual,_=old.structure(structure,names,enforce_limits=False)
    _equal(actual,expected,"compiled topology")
    for key,value in {"actor_role_sha256":role_actor.role_sha256,"guard_count":sum(d["guard_added"] for d in decisions),
        "leaf_guard_decisions":decisions,"compiled_leaf_origins":origins,
        "original_structure":tree,"required_guarded_structure":expected,
        "original_structure_sha256":digest(tree),"required_guarded_structure_sha256":digest(expected),
        "original_capacity":capacity,"required_capacity":cap,"feasible":True,"status":"feasible", "infeasibility_reasons":[],
        "schema":"warehouse-v622-internal-topology-legal-guards.v1",
        "limits":{"maximum_depth":6,"maximum_leaves":32,"maximum_source_features_per_node":3,"guards_and_empty_branches_count":True},
        "guard_policy":"frozen_raw_float32_leaf_argmax_then_explicit_legal_guard_else_WAIT",
        "guard_policy_uses_observations":False,"all_guards_and_empty_children_counted":True,
        "input_weights_modified":False,"full_actor_policy_equivalence_claimed":False,
        "leaf_outputs_reused_as_fitted_program":False,"new_leaf_preferred_action_requires_new_path_proof":True}.items():
        _equal(report.get(key),value,"compiler."+key)
    return {"passed":True,"guard_count":sum(d["guard_added"] for d in decisions),"capacity":_plain(cap),
            "independent_structure_sha256":digest(expected)}


def _nodes(tree,x,names):
    """All internal/terminal path regions, including empty guarded children."""
    result={};mask=[names.index("legal_"+a.lower()) for a in ACTIONS]
    def visit(ref,path,rows,legal):
        result[path]=(ref,rows,frozenset(legal))
        if ref<0:return
        w=tree["split_weights"][ref];b=float(tree["split_bias"][ref]);features=np.flatnonzero(w)
        score=np.zeros(len(rows),np.float64)
        for j in features:score+=x[rows,j].astype(np.float64)*float(w[j])
        score+=b
        right=score>old.TIE_EPSILON;ll,rr=set(),set()
        if len(features)==1 and int(features[0]) in mask:
            a=mask.index(int(features[0]));one=b+float(w[features[0]])
            if b<=old.TIE_EPSILON<one:rr.add(a)
            if one<=old.TIE_EPSILON<b:ll.add(a)
        visit(int(tree["left_refs"][ref]),path+(0,),rows[~right],legal|ll)
        visit(int(tree["right_refs"][ref]),path+(1,),rows[right],legal|rr)
    visit(tree["root_ref"],(),np.arange(len(x)),{4})
    return result


def _leaf_numbers(record,rows,legal,arrays):
    x,m,p,episodes,names,role=arrays;z=np.asarray(record.get("logits"),np.float32)
    _require(z.shape==(5,) and np.isfinite(z).all(),"invalid frozen terminal logits")
    action=int(z.argmax());others=np.delete(z.astype(np.float64),action)
    _require(action in legal and float(z[action])-float(others.max())>=.001,"frozen table leaf lacks float32 preference/path proof")
    optimizer=record.get('optimizer',{})
    _require(optimizer.get('success') is True and optimizer.get('float32_feasible') is True
        and type(optimizer.get('chosen_action')) is int and optimizer['chosen_action']==action
        and type(optimizer.get('row_count')) is int and optimizer['row_count']==len(rows)
        and type(optimizer.get('enumerated_action_solves')) is int and optimizer['enumerated_action_solves']==len(legal)
        and optimizer.get('solver')=='SLSQP' and type(optimizer.get('solver_status')) is int and optimizer['solver_status']==0,
        'frozen optimizer evidence does not bind the actual path/rows/action')
    _require(not len(rows) or np.all(m[rows,action]==1),"frozen table raw action illegal on fit rows")
    p=p[rows].astype(np.float64);masked=np.where(m[rows]>.5,z.astype(np.float64),-np.inf)
    shifted=masked-masked.max(-1,keepdims=True)
    logq=shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))
    kl=np.sum(p*(np.log(np.maximum(p,1e-300))-np.where(m[rows]>.5,logq,0.)),axis=-1)
    matches=p.argmax(-1)==action;counts=Counter(episodes[rows]);agreement=float(matches.mean()) if len(rows) else 0.
    qualified=bool(len(rows)>=80 and len(counts)>=8 and max(counts.values())/len(rows)<=.20
        and agreement>=.95 and float(kl.mean())<=.25)
    order=sorted(set(episodes),key=lambda e:sha256(e.encode()).digest());groups={e:i%3 for i,e in enumerate(order)}
    memberships=np.asarray([groups[e] for e in episodes[rows]]);stability=[]
    for i in range(3):
        take=memberships==i
        stability.append({"group":i,"rows":int(take.sum()),"episodes":len(set(episodes[rows[take]])),
            "agreement":float(matches[take].mean()) if take.any() else None,"masked_kl":float(kl[take].mean()) if take.any() else None})
    robust=bool(qualified and agreement>=.97 and float(kl.mean())<=.20
        and all(s["episodes"]>=2 and s["agreement"]>=.97 and s["masked_kl"]<=.20 for s in stability))
    expected=float(np.sum(p[:,action]*matches)) if qualified else 0.
    score=[float(len(rows)) if robust else 0.,expected,-float(kl.sum()),-1.]
    for key,value in {"rows":len(rows),"rows_sha256":sha256(np.ascontiguousarray(rows,dtype="<i8").tobytes()).hexdigest(),
        "path_legal_actions":sorted(legal),"agreement":agreement,"fit_qualified":qualified,
        "fit_stable_explanation_qualified":robust,"episode_stability_groups":stability,"score":score}.items():
        _equal(record.get(key),value,"table."+key)
    return tuple(record["score"])


def _key(node,tree):
    h=sha256()
    def visit(item):
        if item is None:h.update(b"L");return
        ref,l,r=item;h.update(b"N");h.update(np.asarray(tree["split_weights"][ref],dtype="<f4").tobytes())
        h.update(np.asarray([tree["split_bias"][ref]],dtype="<f4").tobytes());visit(l);visit(r)
    visit(node);return h.digest()


def _dp(tree,by_path):
    choices=0
    def solve(ref,path):
        nonlocal choices
        states={1:(tuple(by_path[path]["score"]),None,(path,))}
        if ref>=0:
            left=solve(int(tree["left_refs"][ref]),path+(0,));right=solve(int(tree["right_refs"][ref]),path+(1,))
            for lk,lv in left.items():
                for rk,rv in right.items():
                    if lk+rk>32:continue
                    choices+=1;score=tuple(a+b for a,b in zip(lv[0],rv[0],strict=True));node=(ref,lv[1],rv[1]);value=(score,node,lv[2]+rv[2])
                    previous=states.get(lk+rk)
                    if previous is None or score>previous[0] or score==previous[0] and _key(node,tree)<_key(previous[1],tree):states[lk+rk]=value
        return states
    states=solve(tree["root_ref"],())
    winner=min(states.values(),key=lambda a:(tuple(-v for v in a[0]),_key(a[1],tree)))
    return winner,{"objective_vector":list(winner[0]),"reachable_leaf_budgets":sorted(states),"complete_DP":True,
        "combinations_evaluated":choices,"selected_terminal_paths":[list(p) for p in winner[2]],
        "allowed_moves":"terminal_at_node_or_keep_both_children_no_child_promotion"}


def _export_pruned(node,tree):
    w,b,l,r=[],[],[],[];leaves=0
    def visit(item):
        nonlocal leaves
        if item is None:leaves+=1;return -leaves
        ref,ll,rr=item;index=len(w);w.append(tree["split_weights"][ref]);b.append(tree["split_bias"][ref]);l.append(0);r.append(0)
        l[index]=visit(ll);r[index]=visit(rr);return index
    root=visit(node)
    return {"root_ref":root,"split_weights":np.asarray(w,np.float32).reshape(-1,tree["feature_dim"]),
        "split_bias":np.asarray(b,np.float32),"left_refs":np.asarray(l,np.int64),"right_refs":np.asarray(r,np.int64),
        "leaf_count":leaves,"feature_dim":tree["feature_dim"]}


def audit_frozen_table(search,raw_program,structure,fit_arrays):
    """Recompute each node's fit score and the complete frozen-table DP."""
    arrays=_arrays(fit_arrays);x,m,p,episodes,names,role=arrays
    tables=search.get("frozen_terminal_tables");generations=search.get("template_generation_reports")
    _require(isinstance(tables,list) and 1<=len(tables)<=4 and isinstance(generations,list)
        and 1<=len(generations)<=4 and search.get("templates_requested")==4,"missing declared template family")
    _require(all(type(g.get('template')) is int and g['template']==i for i,g in enumerate(generations)),"generation template IDs incomplete")
    scored={g['template'] for g in generations if g.get('dp_completed') is True}
    _require(scored=={t.get('template') for t in tables},"a completed DP template was omitted from the compared family")
    candidates=[];seen=set()
    for table in tables:
        i=_integer(table.get("template"),"template ID",3);_require(i not in seen,"duplicate frozen template");seen.add(i)
        generated=next((a for a in generations if a.get("template")==i),None)
        _require(generated is not None and generated.get("completed_structure") is True
            and generated.get("frozen_table_completed") is True and generated.get("dp_completed") is True,
            "table has no completed template/DP producer")
        raw=generated["structure"];tree,cap=old.structure(raw,names,enforce_limits=False)
        _require(cap["depth"]<=6 and cap["leaf_count"]<=64 and cap["maximum_source_features_per_node"]<=3,"invalid maximal template capacity")
        _require(generated.get("structure_sha256")==digest(raw)==table.get("template_structure_sha256"),"template structure hashes differ")
        records=table.get("records");regions=_nodes(tree,x,names)
        _require(isinstance(records,list) and len(records)==len(regions) and digest(records)==table.get("frozen_terminal_table_sha256"),"frozen node-table hash/size differs")
        by_path={}
        for j,record in enumerate(records):
            path=tuple(record.get("path",()))
            _require(type(record.get("node_id")) is int and record["node_id"]==j and path in regions
                and path not in by_path and all(type(b) is int and b in (0,1) for b in path),"table node path/ID differs")
            ref,rows,legal=regions[path];_leaf_numbers(record,rows,legal,arrays);by_path[path]=record
        _require(set(by_path)==set(regions),"table omits a node or empty branch")
        winner,certificate=_dp(tree,by_path);_equal(table.get("certificate"),certificate,"independent.DP")
        candidates.append((winner[0],_key(winner[1],tree),i,tree,winner,table,by_path))
    selected=min(candidates,key=lambda a:(tuple(-x for x in a[0]),a[1],a[2]))
    score,_,index,tree,winner,table,by_path=selected
    _require(type(search.get("selected_template")) is int and search["selected_template"]==index
        and type(search.get("complete_frozen_tables")) is int and search["complete_frozen_tables"]==len(tables),"wrong winning template or complete count")
    _equal(search.get("selected_fit_score"),list(score),"selected frozen score")
    expected=_export_pruned(winner[1],tree);actual,_=old.structure(structure,names,enforce_limits=False)
    _equal(actual,expected,"selected DP topology")
    paths=winner[2];logits=np.asarray([by_path[path]["logits"] for path in paths],np.float32)
    _require(np.array_equal(np.asarray(structure.get("initial_leaf_logits"),np.float32),logits),"selected logits not preserved from frozen table")
    logits_sha=sha256(np.ascontiguousarray(logits,dtype="<f4").tobytes()).hexdigest()
    binding={"selected_table_sha256":table["frozen_terminal_table_sha256"],"selected_terminal_paths":[list(a) for a in paths],
        "selected_table_node_ids":[by_path[a]["node_id"] for a in paths],"selected_leaf_logits_sha256":logits_sha,
        "restricted_table_gap":0.,"global_function_class_optimality_gap":None}
    _equal(structure.get("frozen_table_certificate"),binding,"selected table binding")
    certificate=search.get("certificate",{})
    for key,value in {"selected_table_sha256":binding["selected_table_sha256"],"selected_terminal_paths":binding["selected_terminal_paths"],
        "selected_leaf_logits_sha256":logits_sha,"restricted_objective_lower_bound_vector":list(score),
        "restricted_objective_upper_bound_vector":list(score),"restricted_table_gap":0.,"completed_template_family_combination_gap":0.,
        "global_function_class_optimality_gap":None,"continuous_leaf_output_optimality_claimed":False,
        "final_refit_and_calibrated_program_covered_by_DP_certificate":False,
        "all_requested_templates_completed_and_enumerated":len(tables)==4}.items():_equal(certificate.get(key),value,"family certificate."+key)
    selected_reports=[]
    for i,path in enumerate(paths):
        record=by_path[path];n=record['rows']
        selected_reports.append({'leaf':i,'rows':n,'depth':len(path),'path_legal_actions':record['path_legal_actions'],
            'raw_action':int(logits[i].argmax()),'fit_qualified':record['fit_qualified'],
            'fit_stable_explanation_qualified':record['fit_stable_explanation_qualified'],
            'table_node_id':record['node_id'],'selected_terminal_path':list(path),'agreement':record['agreement'],
            'expected_feedback_mass':record['score'][1],'masked_kl':-record['score'][2]/n if n else None,
            'optimizer':record['optimizer']})
    _equal(structure.get('fit_leaf_reports'),selected_reports,'selected structure leaf evidence')
    _equal(search.get('fit_leaf_reports'),selected_reports,'selected search leaf evidence')
    report=raw_program["report"]
    raw_tree,_=old.structure(raw_program['structure'],names,enforce_limits=False)
    _equal(raw_tree,expected,'final program frozen topology')
    _require(raw_program.get('method')=='fixed_template_leaf_budget_dp_v622'
        and raw_program.get('actor_role_sha256')==role,'wrong D32 full-NN/method binding')
    _require(np.array_equal(np.asarray(report.get("precalibration_leaf_logits"),np.float32),logits),"program changed precalibration frozen outputs")
    _equal(report.get("frozen_table_certificate"),binding,"program frozen binding")
    _equal(report.get('frozen_fit_leaf_reports'),selected_reports,'program selected frozen leaf evidence')
    _require(report.get('frozen_fit_leaf_reports_sha256')==digest(selected_reports),'selected frozen fit reports SHA differs')
    _equal(report.get('fit_leaf_row_counts'),[by_path[p]['rows'] for p in paths],'program fit occupancy')
    _require(structure.get("fit_target_sha256")==_identity(arrays)["fit_target_sha256"]==report.get("fit_target_sha256"),"frozen table target changed")
    final=np.asarray(raw_program["leaf_logits"],np.float32)
    _require(final.shape==logits.shape,"program changed frozen leaf count")
    for i,z in enumerate(logits):
        cal=report["leaf_calibrations"][str(i)];temperature=cal.get("temperature")
        _require(type(temperature) in (int,float) and temperature in (.8,.9,1.,1.1,1.2)
            and np.array_equal(final[i],np.asarray(z/temperature,np.float32)),"final output is not frozen logits plus registered calibration")
        action=int(final[i].argmax())
        _require(action==cal.get('chosen_action') and type(cal.get('chosen_action')) is int
            and action in by_path[paths[i]]['path_legal_actions']
            and float(final[i,action])-float(np.delete(final[i].astype(np.float64),action).max())>=.001,
            'calibrated frozen leaf lost raw action/path/margin certificate')
        _equal(report['leaf_fits'][str(i)],by_path[paths[i]]['optimizer'],'program frozen optimizer evidence')
    return {"passed":True,"completed_templates_audited":len(tables),"selected_template":index,
        "selected_table_sha256":binding["selected_table_sha256"],"restricted_table_gap":0.,
        "global_function_class_optimality_gap":None,"DP_nodes_checked":sum(len(t["records"]) for t in tables),
        "final_outputs_only_recorded_temperature_calibration":True}
