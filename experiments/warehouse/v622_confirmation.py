"""Source-bound, inactive-by-default v6.22 fresh development confirmation.

No seed is generated here. A separately reviewed exact schedule and history
clearance are required before prepare. Collection uses a new finite capability,
never the closed candidates' ContextVar markers. Nothing authorizes PPO,
sealed evaluation, publication, or a pilot.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from types import MappingProxyType

from experiments.warehouse import v622_protocol as capacity
from experiments.warehouse import v622_recovery_protocol as recovery
from experiments.warehouse.v622_completeness import require_complete_d32_search

SCHEMA = "warehouse-v622-confirmation-config.v1"
PLAN_SCHEMA = "warehouse-v622-confirmation-plan.v1"
CLEARANCE_SCHEMA = "warehouse-v622-confirmation-seed-clearance.v1"
COUNTS = {"fit": 100, "calibration": 25, "eligibility": 75, "validation": 75, "pressure": 216}
NEW_SOURCES = ("experiments/warehouse/v622_confirmation.py",
    "scripts/run_warehouse_v622_confirmation.py", "tests/test_warehouse_v622_confirmation.py",
    "experiments/warehouse/v622_completeness.py")
HISTORY_ROOT_NAMES = ("XAI", "XAI-study-v3", "XAI-v6-integration", "XAI-v6-recovery-20261015T165508Z")
HISTORICAL_TRACE_INVENTORY = "output/warehouse_v621/seed-ledger-03/rebound-01/inventory.json"
HISTORICAL_TRACE_INVENTORY_SHA256 = "98ea8b4d0aed59978d8537e45b8677fa84af848416a64ccc6aa1b0d9edf62b19"
_SEAL = object()
_ACTIVE = ContextVar("warehouse_v622_confirmation", default=None)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8*1024*1024): h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())


def pointer(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path)}


def _regular(root, value):
    root = Path(root).resolve(); value = Path(value)
    path = value if value.is_absolute() else root/value
    if not path.resolve().is_relative_to(root): raise RuntimeError("confirmation evidence outside checkout")
    if any(p.is_symlink() for p in (path, *path.parents)): raise RuntimeError("confirmation symlink evidence rejected")
    if not path.is_file(): raise RuntimeError("confirmation evidence must be an existing regular file")
    return path.resolve()


def read_bound(root, ref, bound):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}: raise RuntimeError("malformed confirmation pointer")
    path = _regular(root, ref["path"]); raw = path.read_bytes()
    if sha256(raw).hexdigest() != ref["sha256"]: raise RuntimeError("confirmation evidence SHA mismatch")
    if str(path) in bound and bound[str(path)] != ref["sha256"]: raise RuntimeError("conflicting confirmation binding")
    bound[str(path)] = ref["sha256"]
    return path, json.loads(raw)


def _closed(value):
    return (all(type(value.get(k)) is int and value[k] == 0 for k in ("environment_resets", "actor_updates", "PPO_joint_steps"))
        and value.get("publishable") is False and value.get("pilot_opened") is False)


def _capacity_bundle(root, refs, stage, bound):
    """Rebuild selection from all independently audited recovery-02 children."""
    if not isinstance(refs, dict) or set(refs) != {"plan", "summary", "audit"}: raise RuntimeError("complete winner bundle required")
    directory = root/"output/warehouse_v622"/("capacity-02" if stage == "primary" else "fallback-02")
    values = {}
    for key, name in (("plan", "plan.json"), ("summary", "summary.json"), ("audit", "independent-audit.json")):
        path, values[key] = read_bound(root, refs[key], bound)
        if path != directory/name: raise RuntimeError("winner must come from canonical complete recovery-02 batch")
    plan, summary, audit = (values[k] for k in ("plan", "summary", "audit"))
    plan_sha = refs["plan"]["sha256"]; jobs = capacity.jobs(stage)
    _, claim = read_bound(root, pointer(directory/"plan.claim.json"), bound)
    if (plan.get("schema") != "warehouse-v622-capacity-plan.v1" or plan.get("candidate_id") != capacity.CANDIDATE_ID
            or plan.get("stage") != stage or plan.get("jobs") != jobs or plan.get("folds") != capacity.grouped_folds()
            or plan.get("gates") != capacity.GATES or plan.get("limits") != capacity.LIMITS
            or plan.get("budgets") != capacity.round_budget_contract()
            or claim != {"sha256":plan_sha,"single_use":True}
            or not _closed(plan) or plan.get("scientific_retry_permitted") is not False
            or summary.get("schema") != "warehouse-v622-capacity-summary.v1" or summary.get("candidate_id") != capacity.CANDIDATE_ID
            or summary.get("stage") != stage or summary.get("plan_sha256") != plan_sha or not _closed(summary)
            or audit.get("schema") != "warehouse-v622-independent-capacity-audit.v1" or audit.get("passed") is not True
            or audit.get("candidate_id") != capacity.CANDIDATE_ID or audit.get("stage") != stage
            or audit.get("plan_sha256") != plan_sha or audit.get("summary_sha256") != refs["summary"]["sha256"]
            or audit.get("job_count") != len(jobs) or not _closed(audit)):
        raise RuntimeError("winner lacks complete independent recovery audit")
    identity = plan["identity"]; sources = identity["source_files_sha256"]
    if audit.get("source_closure_sha256") != identity["source_closure_sha256"] or digest(sources) != identity["source_closure_sha256"]:
        raise RuntimeError("winner source closure differs")
    if (sources != recovery.source_closure_files(root) or identity.get("recipe_sha256") != recovery.validate_recipe(root)):
        raise RuntimeError("winner is not bound to the exact recovery source and recipe")
    for name, h in sources.items():
        path = _regular(root, name)
        if file_sha(path) != h: raise RuntimeError("winner source changed")
    if identity.get("actor_role_sha256") != capacity.ROLE_SHA256 or identity.get("actor", {}).get("sha256") != capacity.ACTOR_SHA256:
        raise RuntimeError("winner is not bound to the unchanged dual NN")
    if set(summary.get("jobs", {})) != {j["job_id"] for j in jobs}: raise RuntimeError("winner summary is incomplete")
    audited = {r["job"]["job_id"]: r for r in audit.get("jobs", [])}
    if len(audited) != len(jobs) or set(audited) != set(summary["jobs"]): raise RuntimeError("audit omits a winner unit")
    results = []
    for job in jobs:
        folder = directory/"jobs"/job["job_id"]
        result_path, result = read_bound(root, summary["jobs"][job["job_id"]], bound)
        if result_path != folder/"result.json" or (folder/"failed.json").exists(): raise RuntimeError("failed/noncanonical winner child")
        _, completion = read_bound(root, pointer(folder/"completed.json"), bound)
        a = audited[job["job_id"]]
        if (result.get("job") != job or result.get("plan_sha256") != plan_sha or not _closed(result)
                or result.get("actor_weights_unchanged") is not True or result.get("excluded_frames") != 0
                or a.get("job") != job or a.get("passed") is not True
                or completion.get("plan_sha256") != plan_sha or completion.get("result_sha256") != summary["jobs"][job["job_id"]]["sha256"]):
            raise RuntimeError("winner child completion/audit binding differs")
        if job['arm']=='D32':
            if result.get('search_report') is None:
                require_complete_d32_search(result.get('budget_exhaustion'))
            else:
                search_path,search=read_bound(root,result.get('search_report'),bound)
                if search_path!=folder/'search-report.json': raise RuntimeError('noncanonical D32 winner search evidence')
                require_complete_d32_search(search)
        if result.get("status") == "completed":
            gate = capacity.gate_metrics(result["metrics"]["validation"])
            _, raw = read_bound(root, result["program"], bound)
            if (result.get("development_gate") != gate or completion.get("passed") is not gate["passed"]
                    or a.get("program_sha256") != result["program"]["sha256"]
                    or raw.get("method") != capacity.METHODS[job["arm"]]
                    or raw.get("actor_role_sha256") != capacity.ROLE_SHA256[job["role"]]):
                raise RuntimeError("winner child program/gate differs")
        elif result.get("status") != "scientific_budget_exhausted" or completion.get("passed") is not False or a.get("status") != result["status"]:
            raise RuntimeError("unknown winner child disposition")
        results.append(result)
    recomputed = capacity.summarize_units(results, stage=stage)
    if any(summary.get(k) != value for k, value in recomputed.items()): raise RuntimeError("winner does not recompute")
    return summary, plan


def checked_winner(root, refs, bound):
    if not isinstance(refs, dict) or set(refs) != {"primary", "fallback"}: raise RuntimeError("explicit primary/fallback winner binding required")
    primary, plan = _capacity_bundle(root, refs["primary"], "primary", bound)
    if primary["selected_arm"] is not None:
        if refs["fallback"] is not None: raise RuntimeError("fallback cannot override a passing primary method")
        return primary["selected_arm"], plan
    if primary["next_stage"] != "fallback" or refs["fallback"] is None: raise RuntimeError("no audited winner; confirmation remains closed")
    fallback, fallback_plan = _capacity_bundle(root, refs["fallback"], "fallback", bound)
    if (fallback_plan.get("fallback_parent") != {"summary": refs["primary"]["summary"], "independent_audit": refs["primary"]["audit"]}
            or fallback_plan['identity']!=plan['identity']):
        raise RuntimeError("fallback is not bound to the same failed primary")
    if fallback["selected_arm"] != "D32": raise RuntimeError("no audited winner; confirmation remains closed")
    return "D32", fallback_plan


def checked_cases(cases):
    from backend.training.warehouse_v6_evaluation_authorization import is_sealed_scenario_seed
    if not isinstance(cases, list) or len(cases) != 491: raise ValueError("confirmation requires all275+216 explicit cases")
    required = {"case_id", "split", "base", "scenario_seed", "task", "action_seed", "sampling_mode", "draw"}
    if any(not isinstance(c, dict) or set(c) != required for c in cases): raise ValueError("unknown confirmation case schema")
    if len({c["case_id"] for c in cases}) != len(cases): raise ValueError("case IDs must be unique")
    if Counter(c["split"] for c in cases) != COUNTS: raise ValueError("exact split counts differ")
    if [c["split"] for c in cases] != [s for s, n in COUNTS.items() for _ in range(n)]: raise ValueError("reset schedule must follow the registered split order")
    for c in cases:
        if (not isinstance(c["case_id"], str) or not 1<=len(c["case_id"])<=100 or any(x not in "abcdefghijklmnopqrstuvwxyz0123456789-" for x in c["case_id"])
                or any(type(c[k]) is not int or not 0 <= c[k] < 2**63 for k in ("base", "scenario_seed", "action_seed", "draw"))
                or type(c["task"]) is not int or c["task"] not in (1, 2, 3) or c["scenario_seed"] != 10*c["base"]+c["task"]
                or c["sampling_mode"] not in ("sampled", "argmax") or any(is_sealed_scenario_seed(c[k]) for k in ("base", "scenario_seed", "action_seed"))):
            raise ValueError("invalid/reused sealed seed argument or task expansion")
    fresh = [c for c in cases if c["split"] != "pressure"]
    pressure = [c for c in cases if c["split"] == "pressure"]
    if (len({c["base"] for c in fresh}) != 275 or any(c["draw"] != 0 for c in fresh)
            or Counter(c["sampling_mode"] for c in fresh[:100]) != {"sampled": 50, "argmax": 50}
            or any(c["sampling_mode"] != "sampled" for c in fresh[100:])
            or any(Counter(c["task"] for c in fresh if c["split"] == s) != {1:25, 2:25, 3:25} for s in ("eligibility", "validation"))):
        raise ValueError("fresh purpose partitions/modes/task balance differ")
    groups = {}
    for c in pressure: groups.setdefault(c["base"], []).append(c)
    if (len(groups) != 72 or set(groups) & {c["base"] for c in fresh}
            or Counter(g[0]["task"] for g in groups.values()) != {1:24, 2:24, 3:24}
            or any(len(g) != 3 or {c["draw"] for c in g} != {0,1,2} or len({c["scenario_seed"] for c in g}) != 1
                or any(c["sampling_mode"] != "sampled" for c in g) for g in groups.values())):
        raise ValueError("pressure requires72 independent bases, three matched NN draws each")
    bases, scenes, actions = ({c[k] for c in cases} for k in ("base", "scenario_seed", "action_seed"))
    if len(actions) != 491 or bases & scenes or bases & actions or scenes & actions: raise ValueError("fresh RNG namespaces overlap")
    return cases


def exact_seed_plan(cases):
    checked_cases(cases); uses = []
    for c in cases:
        purpose = "v622_confirmation_"+c["split"]; group = c["case_id"]
        for kind, argument, value, tag in (
            ("scenario", "task_expansion_base", c["base"], "base:"+str(c["base"])),
            ("env", "random.Random(config.seed)", c["scenario_seed"], "scene:"+str(c["scenario_seed"])),
            ("env", "reset(seed)", c["scenario_seed"], "scene:"+str(c["scenario_seed"])),
            ("action_rng", "FrozenActor.act(base_seed)", c["action_seed"], "action:"+str(c["action_seed"]))):
            use={"kind":kind,"purpose":purpose,"argument":argument,"value":value,"group":group,"reuse_tag":tag}
            if kind=='action_rng':use['consumed_as_random_draw']=c['sampling_mode']=='sampled'
            uses.append(use)
    return {"schema":"warehouse-exact-seed-plan.v1","candidate_id":capacity.CANDIDATE_ID,
        "uses":uses,"confirmation_cases_sha256":digest(cases),"authorization_granted":False}


def configuration_identity(config):
    return digest({k:v for k,v in config.items() if k != "seed_clearance"})


def recipe():
    return {"counts":COUNTS,"fit_modes":{"sampled":50,"argmax":50},"limits":capacity.LIMITS,
        "program_gates":capacity.GATES,"pressure_program_gate":"same_strict_v622_precheck_as_validation",
        "quality_gates_by_split":{s:{"minimum_deliveries":5.5,"maximum_collisions":6.,"maximum_shutdown_episode_rate":.05}
            for s in ("validation","pressure")},
        "reference_retention":"same_frozen_NN_self_reference_descriptive_only","training_authorized":False,
        "D32_requires_all_four_templates_tables_and_DP":True,
        "programs_never_select_actions":True,"no_validation_or_pressure_gradients":True,"pilot_opened":False}


def source_files(root, winner_plan):
    files = dict(winner_plan["identity"]["source_files_sha256"])
    files.update({name:file_sha(_regular(root, name)) for name in NEW_SOURCES})
    if files[NEW_SOURCES[0]]!=file_sha(__file__): raise RuntimeError("executing confirmation module is not the source-guarded module")
    return dict(sorted(files.items()))


def required_history_traces(root,bound):
    _,authority=read_bound(root,{"path":HISTORICAL_TRACE_INVENTORY,"sha256":HISTORICAL_TRACE_INVENTORY_SHA256},bound)
    registered=authority.get("registered_trace_paths")
    if not isinstance(registered,list) or len(registered)!=8 or len(set(registered))!=8:
        raise RuntimeError("pinned historical trace inventory differs")
    return set(registered)


def _seed_clearance(root, config, bound, output, *, require_current_history=True):
    from experiments.warehouse import exact_seed_audit as exact
    from experiments.warehouse import seed_metadata_audit as metadata
    _, clearance = read_bound(root, config["seed_clearance"], bound)
    checks = {"schema":CLEARANCE_SCHEMA,"candidate_id":capacity.CANDIDATE_ID,
        "configuration_identity_sha256":configuration_identity(config),"cases_sha256":digest(config["cases"]),
        "source_closure_sha256":config["source_closure_sha256"],"approved_for_confirmation":True,
        "unresolved":[],"remaining_conflicts":[],"PPO_joint_steps":0,"actor_updates":0,
        "pilot_opened":False,"publishable":False,"sealed_evaluation_authorized":False}
    if any(type(clearance.get(k)) is not type(v) or clearance[k] != v for k,v in checks.items()): raise RuntimeError("fresh confirmation requires its own exact source/case clearance")
    expires = datetime.fromisoformat(clearance["valid_until_utc"])
    if expires.tzinfo is None or require_current_history and datetime.now(timezone.utc) >= expires: raise RuntimeError("confirmation seed clearance expired")
    plan_path, seed_plan = read_bound(root, config["exact_seed_plan"], bound)
    if seed_plan != exact_seed_plan(config["cases"]): raise RuntimeError("exact seed calls differ from actual planned collector")
    exact.read_exact_plan(plan_path)
    _, evidence = read_bound(root, clearance["exact_evidence"], bound)
    if (evidence.get("schema") != "warehouse-exact-seed-evidence.v1" or evidence.get("authorization_granted") is not False
            or evidence.get("exact_plan") != config["exact_seed_plan"] or evidence.get("exact_auditor_sha256") != file_sha(exact.__file__)
            or digest(evidence.get("families")) != digest(exact.compile_families(seed_plan)) or evidence.get("read_model_weight_files") is not False
            or evidence.get("selection_scope") != metadata.SELECTION_SCOPE or evidence.get("summary",{}).get("unresolved_paths") != []):
        raise RuntimeError("fresh exact history evidence is incomplete or for different arguments")
    _, inventory = read_bound(root,evidence["bounded_or_triaged_inventory"],bound)
    projected=exact.project_inventory(inventory,exact.compile_families(seed_plan))
    if evidence.get("context_annotations"):
        _,annotation=read_bound(root,evidence["context_annotations"],bound)
        projected=exact.apply_context(projected,annotation)
    if (digest(evidence.get("inventory"))!=digest(projected["inventory"])
            or evidence.get("summary")!=projected["summary"] or evidence.get("selection_scope")!=projected.get("selection_scope")):
        raise RuntimeError("exact evidence does not replay its original inventory/context")
    expected_roots = sorted(str((root.parent/name).resolve()) for name in HISTORY_ROOT_NAMES)
    if sorted(evidence.get("scan_roots",[])) != expected_roots or any(not Path(p).is_dir() for p in expected_roots): raise RuntimeError("all four historical roots are required")
    if any(any(section.get(k) for k in ("references_or_reservations","reservation_overlaps","audit_references_not_original_execution","excluded_weight_path_seed_hints"))
           for section in evidence["summary"]["families"].values()): raise RuntimeError("unresolved historical seed use/reservation/context remains")
    _, delta = read_bound(root, clearance["final_delta"], bound)
    if (delta.get("schema") != "warehouse-seed-metadata-delta.v1" or delta.get("authorization_granted") is not False
            or delta.get("requires_new_content_audit") is not False or delta.get("requires_weight_path_receipt_review") is not False
            or any(delta.get(k) for k in ("new_paths","removed_paths","changed_paths","new_opaque_weight_paths_name_stat_only",
                "removed_opaque_weight_paths_name_stat_only","changed_opaque_weight_paths_name_stat_only"))):
        raise RuntimeError("final history delta needs content/context review")
    _, selection = read_bound(root, delta["selection"], bound)
    if sorted(selection.get("scan_roots",[])) != expected_roots: raise RuntimeError("delta omitted a history root")
    registered=inventory.get("registered_trace_paths")
    if (not isinstance(registered,list) or any(not isinstance(p,str) for p in registered) or len(set(registered))!=len(registered)
            or evidence.get("registered_trace_paths")!=registered or selection.get("registered_trace_paths")!=registered
            or not required_history_traces(root,bound).issubset(set(registered))):
        raise RuntimeError("history trace registration omits or changes the pinned historical set")
    if any(not Path(p).is_file() or str(Path(p).resolve())!=p or not any(Path(p).is_relative_to(r) for r in expected_roots) for p in registered):
        raise RuntimeError("registered history trace is missing or outside the audited roots")
    ledger = root/"output/warehouse_v622"/("confirmation-"+config["run_id"]+"-seed-ledger")
    allowed = {str(ledger.resolve()), str(output.resolve())}
    exclusions = [Path(p).resolve() for p in selection.get("scope_exclusions",[])]
    if any(str(p) not in allowed and not p.is_relative_to(ledger) for p in exclusions): raise RuntimeError("history exclusions are not restricted to registered review controls")
    controls = {Path(p).resolve() for p in bound if Path(p).resolve().is_relative_to(ledger)}
    previous = {r["path"]:(r["bytes"],r.get("mtime_ns")) for r in selection["candidates"]
        if not Path(r["path"]).resolve().is_relative_to(ledger) and not Path(r["path"]).resolve().is_relative_to(output)}
    audited={r["path"]:(r["bytes"],r.get("mtime_ns")) for r in inventory["inventory"]
        if r.get("classification")!="source_template_environment_reservation"
        and not Path(r["path"]).resolve().is_relative_to(ledger) and not Path(r["path"]).resolve().is_relative_to(output)}
    if previous!=audited: raise RuntimeError("final delta does not cover exactly the scanned historical metadata")
    if require_current_history:
        current={};stats={}
        for p in metadata.candidates([Path(p) for p in expected_roots], [*exclusions, output, *controls],
                registered_trace_paths=selection.get("registered_trace_paths",()), selection_stats=stats):
            s=p.stat(); current[str(p.resolve())]=(s.st_size,s.st_mtime_ns)
        old_weights={r['path']:(r['bytes'],r['mtime_ns']) for r in selection.get('selection_stats',{}).get('excluded_weights_name_stat_only',[])}
        now_weights={r['path']:(r['bytes'],r['mtime_ns']) for r in stats.get('excluded_weights_name_stat_only',[])}
        if current != previous or old_weights != now_weights: raise RuntimeError("history metadata changed after final delta; new scan/context review required")
    return clearance


def validate_configuration(root, path, *, require_current_history=True):
    root=Path(root).resolve(); path=_regular(root,path); bound={}; _, config=read_bound(root,pointer(path),bound)
    keys={"schema","candidate_id","run_id","winner","method","actor","cases","recipe","source_closure_sha256","exact_seed_plan","seed_clearance"}
    if (set(config)!=keys or config.get("schema")!=SCHEMA or config.get("candidate_id")!=capacity.CANDIDATE_ID
            or not isinstance(config.get("run_id"),str) or not config["run_id"] or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in config["run_id"])
            or config.get("recipe")!=recipe()): raise RuntimeError("explicit registered confirmation configuration required")
    arm,winner_plan=checked_winner(root,config["winner"],bound)
    if config["method"]!=capacity.METHODS[arm]: raise RuntimeError("confirmation cannot change the locked winner method")
    checked_cases(config["cases"])
    actor_path=_regular(root,config["actor"]["path"])
    if (config["actor"]!={"path":str(actor_path),"sha256":capacity.ACTOR_SHA256,"role_sha256":capacity.ROLE_SHA256}
            or file_sha(actor_path)!=capacity.ACTOR_SHA256): raise RuntimeError("confirmation full NN binding differs")
    bound[str(actor_path)]=capacity.ACTOR_SHA256
    files=source_files(root,winner_plan)
    if config["source_closure_sha256"]!=digest(files): raise RuntimeError("confirmation source freeze differs")
    output=root/"output/warehouse_v622"/("confirmation-"+config["run_id"])
    _seed_clearance(root,config,bound,output,require_current_history=require_current_history)
    for name,h in bound.items():
        if file_sha(name)!=h: raise RuntimeError("confirmation prerequisite changed during validation")
    return config,arm,files,bound,output


def _plan_value(root,config_path,config,arm,files,bound):
    return {"schema":PLAN_SCHEMA,"configuration":pointer(_regular(root,config_path)),"configuration_identity_sha256":configuration_identity(config),
        "candidate_id":capacity.CANDIDATE_ID,"run_id":config["run_id"],"arm":arm,"method":config["method"],
        "cases":config["cases"],"actor":config["actor"],"source_files_sha256":files,"source_closure_sha256":digest(files),
        "bound_files_sha256":bound,"recipe":recipe(),"PPO_joint_steps":0,"actor_updates":0,"publishable":False,"pilot_opened":False,
        "single_use_even_if_interrupted":True,"independent_final_admission":False}


def prepare(root, config_path):
    config,arm,files,bound,output=validate_configuration(root,config_path)
    output.mkdir(parents=True,exist_ok=False)
    plan=_plan_value(root,config_path,config,arm,files,bound)
    write_new(output/"plan.json",plan);write_new(output/"plan.claim.json",{"plan_sha256":file_sha(output/"plan.json"),"single_use":True})
    return output/"plan.json"


def load_plan(root, path, *, start_collection=False):
    root=Path(root).resolve(); path=_regular(root,path); plan=json.loads(path.read_bytes())
    if plan.get("schema")!=PLAN_SCHEMA: raise RuntimeError("unknown confirmation plan")
    config,arm,files,bound,output=validate_configuration(root,plan["configuration"]["path"],require_current_history=start_collection)
    if (path!=output/"plan.json" or plan!=_plan_value(root,plan['configuration']['path'],config,arm,files,bound)
            or json.loads((output/"plan.claim.json").read_bytes())!={"plan_sha256":file_sha(path),"single_use":True}):
        raise RuntimeError("immutable confirmation plan changed")
    return path,plan


@dataclass(frozen=True)
class ConfirmationPlan:
    root:Path
    path:Path
    value:dict
    output:Path
    _seal:object=field(repr=False)
    cursor:list=field(default_factory=lambda:[0],repr=False)

    def __post_init__(self):
        if self._seal is not _SEAL: raise RuntimeError("ordinary callers cannot construct confirmation capability")

    def authorize_reset(self,case):
        if _ACTIVE.get() is not self or self._seal is not _SEAL: raise RuntimeError("ordinary rollout has no confirmation authority")
        if self.cursor[0]>=len(self.value["cases"]) or case!=self.value["cases"][self.cursor[0]]: raise RuntimeError("reset outside exact confirmation schedule")
        for name,h in self.value["source_files_sha256"].items():
            if file_sha(self.root/name)!=h: raise RuntimeError("confirmation source changed before reset")
        row={"index":self.cursor[0],"case":case,"plan_sha256":file_sha(self.path),"consumed_before_reset":True}
        with (self.output/"reset-preauthorization.jsonl").open("a") as handle:
            handle.write(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n");handle.flush();os.fsync(handle.fileno())
        self.cursor[0]+=1


@contextmanager
def collection_capability(root,path):
    if _ACTIVE.get() is not None: raise RuntimeError("confirmation capability cannot nest")
    path,value=load_plan(root,path,start_collection=True); output=path.parent/"collection";output.mkdir(exist_ok=False)
    write_new(output/"started.json",{"plan_sha256":file_sha(path),"scheduled_resets":491,"single_use_even_if_interrupted":True,
        "history_clearance_checked_before_first_constructor":True,"started_at_utc":datetime.now(timezone.utc).isoformat()})
    (output/"reset-preauthorization.jsonl").touch(exist_ok=False)
    cap=ConfirmationPlan(Path(root).resolve(),path,value,output,_SEAL);token=_ACTIVE.set(cap)
    try:
        yield cap
        if cap.cursor[0]!=491: raise RuntimeError("not every confirmation reset was consumed")
        for name,h in value["bound_files_sha256"].items():
            if file_sha(name)!=h: raise RuntimeError("confirmation prerequisite changed during collection")
        for name,h in value["source_files_sha256"].items():
            if file_sha(Path(root)/name)!=h: raise RuntimeError("confirmation source changed during collection")
    except BaseException as error:
        write_new(output/"failed.json",{"error_type":type(error).__name__,"error":str(error),"preauthorized_resets":cap.cursor[0],"automatic_retry":False})
        raise
    finally: _ACTIVE.reset(token)


def collect(root,path):
    """The only environment construction path, guarded before constructor RNG."""
    import numpy as np
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v622_program import masked_probabilities
    from backend.training.warehouse_v6_runner import _set_dual_nn_mode
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv
    from env.warehouse.public_action_features_v617 import validate_study_action_physics
    from env.warehouse.public_observation_v6 import encode_public_observation,public_pre_action_snapshot
    from env.warehouse.pure_nn import AGENTS,action_masks
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.decision_protocol import independent_agent_seed
    from env.warehouse.reset_audit_v614 import capture_stage1_resets,stage1_reset_phase
    with collection_capability(root,path) as cap:
        ref=cap.value["actor"]; actor=FrozenWarehouseV6Actor.load(ref["path"],expected_sha256=ref["sha256"],expected_role_sha256=ref["role_sha256"])
        outcomes=[];traces={}; checks=0
        with capture_stage1_resets(cap.output/"actual-resets.jsonl") as actual:
            for case in cap.value["cases"]:
                cap.authorize_reset(case)
                config=participant_study_config(seed=case["scenario_seed"]);validate_study_action_physics(config)
                with stage1_reset_phase("pressure" if case["split"]=="pressure" else "program"):
                    env=WarehouseMultiAgentEnv(config);env.reset(seed=case["scenario_seed"])
                _set_dual_nn_mode(env)
                trace=cap.output/(case["case_id"]+".jsonl"); previous=None;frames=0
                with trace.open("x") as handle:
                    while not (env.state.terminated or env.state.truncated):
                        obs,masks=encode_public_observation(env.state),action_masks(env.state)
                        before=public_pre_action_snapshot(env.state);before_sha=digest(before)
                        if previous is not None and previous!=before_sha: raise RuntimeError("disconnected public state trace")
                        frame,episode_id=int(env.state.frame),int(env.state.episode_id)
                        actions,distributions=actor.act(env.state,base_seed=case["action_seed"],deterministic=case["sampling_mode"]=="argmax")
                        probs=masked_probabilities(actor.logits(obs,masks),masks); roles={}
                        for i,role in enumerate(AGENTS):
                            rng=independent_agent_seed(base_seed=case["action_seed"],episode_id=episode_id,frame=frame,agent_id=role)
                            action=int(probs[i].argmax()) if case["sampling_mode"]=="argmax" else int(np.random.default_rng(rng).choice(5,p=probs[i]))
                            if actions[role]!=ACTIONS[action] or masks[i,action]!=1 or not np.array_equal(np.asarray(distributions[role].probabilities,np.float32),probs[i]): raise RuntimeError("execution differs from complete bound NN proposal")
                            roles[role]={"obs":obs[i].tolist(),"mask":masks[i].tolist(),"nn_probabilities":probs[i].tolist(),"rng_seed":rng,
                                "sampled_action":actions[role],"actor_role_sha256":ref["role_sha256"][role]}; checks+=1
                        _,_,_,_,info=env.step(actions)
                        if info["proposed_actions"]!=actions or info["action_override_agents"]: raise RuntimeError("post-NN action controller detected")
                        after=public_pre_action_snapshot(env.state);previous=digest(after)
                        if {a['agent_id']:a['last_action'] for a in after['agents']}!=actions:
                            raise RuntimeError("physical environment recorded a different NN proposal")
                        handle.write(json.dumps({"case":case,"frame":frame,"episode_id":episode_id,"actor_sha256":ref["sha256"],"roles":roles,
                            "pre_state_public":before,"pre_state_public_sha256":before_sha,"post_state_public":after,"post_state_public_sha256":previous,
                            "proposed_actions":actions,"post_policy_interventions":0},sort_keys=True,separators=(",",":"))+"\n");frames+=1
                traces[case["case_id"]]=pointer(trace)
                outcomes.append({"case":case,"deliveries":int(env.state.total_deliveries),"robot_collisions":int(env.state.robot_collision_events),
                    "shutdown":bool(env.state.shutdown_count),"physical_frames":frames})
        if len(actual)!=491 or [r["seed"] for r in actual]!=[c["scenario_seed"] for c in cap.value["cases"]]: raise RuntimeError("native reset journal differs from full491 schedule")
        result={"schema":"warehouse-v622-confirmation-collection.v1","plan_sha256":file_sha(cap.path),"actor":ref,"traces":traces,"outcomes":outcomes,
            "actual_reset_count":491,"actual_resets":pointer(cap.output/"actual-resets.jsonl"),"preauthorization":pointer(cap.output/"reset-preauthorization.jsonl"),
            "NN_action_checks":checks,"post_policy_interventions":0,"excluded_frames":0,"actor_updates":0,"PPO_joint_steps":0,
            "publishable":False,"pilot_opened":False,"evaluation_scope":"fresh_development_not_sealed_or_human_evaluation"}
        write_new(cap.output/"result.json",result)
    write_new(cap.output/"completed.json",{"result_sha256":file_sha(cap.output/"result.json"),"plan_sha256":file_sha(cap.path)})
    return result


def collection_rows(root,plan_path):
    """Read and replay recorded draws on public states; never reset or refit."""
    import numpy as np
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v622_program import masked_probabilities
    from env.warehouse.public_observation_v6 import rebuild_public_pre_action_state,encode_public_observation
    from env.warehouse.pure_nn import AGENTS,action_masks
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.decision_protocol import independent_agent_seed
    path,plan=load_plan(root,plan_path);folder=path.parent/"collection";bound={}
    _,done=read_bound(root,pointer(folder/"completed.json"),bound)
    _,result=read_bound(root,{"path":str(folder/"result.json"),"sha256":done["result_sha256"]},bound)
    ref=plan["actor"]
    if ((folder/"failed.json").exists() or done.get("plan_sha256")!=file_sha(path) or result.get("plan_sha256")!=file_sha(path)
            or result.get("actor")!=ref or result.get("actual_reset_count")!=491
            or result.get("post_policy_interventions")!=0 or result.get("excluded_frames")!=0
            or result.get("actor_updates")!=0 or result.get("PPO_joint_steps")!=0
            or [r["case"] for r in result["outcomes"]]!=plan["cases"] or set(result["traces"])!={c["case_id"] for c in plan["cases"]}):
        raise RuntimeError("confirmation collection is incomplete or changed")
    actual=Path(result["actual_resets"]["path"]);pre=Path(result["preauthorization"]["path"])
    for file,reference in ((actual,result["actual_resets"]),(pre,result["preauthorization"])):
        if file_sha(_regular(root,file))!=reference["sha256"]: raise RuntimeError("reset journal changed")
    actual=[json.loads(s) for s in actual.read_bytes().splitlines()];pre=[json.loads(s) for s in pre.read_bytes().splitlines()]
    if len(actual)!=491 or len(pre)!=491: raise RuntimeError("reset journals incomplete")
    for i,case in enumerate(plan["cases"]):
        if (pre[i].get("index")!=i or pre[i].get("case")!=case or pre[i].get("plan_sha256")!=file_sha(path)
                or actual[i].get("seed")!=case["scenario_seed"] or actual[i].get("phase")!=("pressure" if case["split"]=="pressure" else "program")):
            raise RuntimeError("native/preauthorization journal schedule differs")
    actor=FrozenWarehouseV6Actor.load(ref["path"],expected_sha256=ref["sha256"],expected_role_sha256=ref["role_sha256"])
    parts={split:{role:[] for role in AGENTS} for split in COUNTS};checks=0
    for case,outcome in zip(plan["cases"],result["outcomes"],strict=True):
        trace_ref=result["traces"][case["case_id"]];trace=_regular(root,trace_ref["path"])
        if trace!=folder/(case["case_id"]+".jsonl") or file_sha(trace)!=trace_ref["sha256"]: raise RuntimeError("trace identity differs")
        events=[json.loads(line) for line in trace.read_bytes().splitlines()]
        if not 1<=len(events)<=120 or len(events)!=outcome["physical_frames"]: raise RuntimeError("whole episode frames missing")
        previous=None;deliveries=0;collisions=0;shutdown=False
        for frame,event in enumerate(events):
            before,after=event["pre_state_public"],event["post_state_public"]
            if (event["case"]!=case or event["frame"]!=frame or before["frame"]!=frame or after["frame"]!=frame+1
                    or event["episode_id"]!=before["episode_id"] or after["episode_id"]!=before["episode_id"]
                    or event["actor_sha256"]!=ref["sha256"] or event["post_policy_interventions"]!=0
                    or digest(before)!=event["pre_state_public_sha256"] or digest(after)!=event["post_state_public_sha256"]
                    or previous is not None and previous!=digest(before)): raise RuntimeError("public trace/frame chain differs")
            previous=digest(after);state=rebuild_public_pre_action_state(before);obs,masks=encode_public_observation(state),action_masks(state)
            removed={t['task_id'] for t in before['tasks']}-{t['task_id'] for t in after['tasks']}
            if len(removed)!=len({t['task_id'] for t in after['tasks']}-{t['task_id'] for t in before['tasks']}): raise RuntimeError("delivery task replacement differs")
            for task in before['tasks']:
                if task['task_id'] in removed:
                    carrier=next((a['agent_id'] for a in before['agents'] if a['carrying_task_id']==task['task_id']),None)
                    if carrier is None or next(a for a in after['agents'] if a['agent_id']==carrier)['position']!=task['delivery_position']: raise RuntimeError("removed task lacks physical delivery")
            deliveries+=len(removed);collisions+=bool(after['last_robot_collision_event'])
            terminal_shutdown=any(a['active'] and not b['active'] for a,b in zip(before['agents'],after['agents'],strict=True))
            if terminal_shutdown and frame!=len(events)-1: raise RuntimeError("trajectory continues after shutdown")
            shutdown|=terminal_shutdown
            probs=masked_probabilities(actor.logits(obs,masks),masks)
            for i,role in enumerate(AGENTS):
                row=event["roles"][role];rng=independent_agent_seed(base_seed=case["action_seed"],episode_id=event["episode_id"],frame=frame,agent_id=role)
                a=int(probs[i].argmax()) if case["sampling_mode"]=="argmax" else int(np.random.default_rng(rng).choice(5,p=probs[i]))
                if (not np.array_equal(np.asarray(row["obs"],np.float32),obs[i]) or not np.array_equal(np.asarray(row["mask"],np.float32),masks[i])
                        or not np.array_equal(np.asarray(row["nn_probabilities"],np.float32),probs[i]) or row["rng_seed"]!=rng
                        or row["actor_role_sha256"]!=ref["role_sha256"][role] or row["sampled_action"]!=ACTIONS[a]
                        or event["proposed_actions"][role]!=ACTIONS[a]
                        or next(agent for agent in after['agents'] if agent['agent_id']==role)['last_action']!=ACTIONS[a]):
                    raise RuntimeError("recorded action differs from full bound NN draw")
                parts[case["split"]][role].append({**row,"episode":case["case_id"],"base_seed":case["scenario_seed"],"task":case["task"],
                    "frame":frame,"episode_id":event["episode_id"]});checks+=1
        if file_sha(trace)!=trace_ref["sha256"]: raise RuntimeError("trace changed during read")
        computed={'deliveries':deliveries,'robot_collisions':collisions,'shutdown':shutdown,'physical_frames':len(events)}
        if len(events)<120 and not shutdown or any(type(outcome.get(k)) is not type(v) or outcome[k]!=v for k,v in computed.items()): raise RuntimeError("episode is truncated or outcome differs from public transitions")
    if checks!=result["NN_action_checks"]: raise RuntimeError("physical action count differs")
    return actor,parts,result


def fit(root,plan_path):
    """Only the already locked method is searched on fit episodes, once."""
    from time import monotonic
    import numpy as np
    from backend.training.warehouse_v622_program import (fit_program,program_metrics,program_structure_sha256,
        compile_actor_structure,masked_probabilities,gate_metrics,_arrays,ProgramBudgetExhausted)
    from backend.training.warehouse_v622_search import fit_structure,SearchConfig
    path,plan=load_plan(root,plan_path);actor,parts,collection=collection_rows(root,path)
    folder=path.parent/"programs";folder.mkdir(exist_ok=False)
    reports={}
    try:
        for role in capacity.ROLES:
            target=folder/role;target.mkdir();arm=plan["arm"];limits=capacity.round_budget_contract()[arm]
            begin=monotonic();deadline=begin+limits["wall_seconds"]
            if arm=="B32":
                structure,compiler=compile_actor_structure(actor.roles[role])
                if structure is None: raise RuntimeError("locked B32 became infeasible")
                search={"leaf_optimization_calls":0,"global_search_rounds":0};search["report_sha256"]=digest(search)
                write_new(target/"compiler.json",compiler)
            else:
                x,m,_=_arrays(parts["fit"][role]);p=masked_probabilities(actor.roles[role].logits_encoded(x),m)
                structure,search=fit_structure(arm,x,m,p,[r["episode"] for r in parts["fit"][role]],
                    feature_names=actor.roles[role].feature_names,actor_role_sha256=actor.role_sha256[role],
                    config=SearchConfig(wall_time_seconds=limits["wall_seconds"]-limits["reserved_final_wall_seconds"]))
            write_new(target/"structure.json",capacity.plain(structure));write_new(target/"search.json",search)
            if arm=='D32': require_complete_d32_search(search)
            program=fit_program(role,actor,*(parts[s][role] for s in capacity.SPLITS),structure=structure,
                structure_sha256=program_structure_sha256(structure),fit_method=plan["method"],validation_phase="precheck",
                fit_budget={"deadline_monotonic":deadline,"maximum_leaf_api_calls":0 if arm=="D32" else 32,
                    "prior_leaf_optimization_calls":search["leaf_optimization_calls"],"max_total_leaf_optimization_calls":20540 if arm=="D32" else 20000 if arm=="A32" else 32,
                    "provenance_sha256":search["report_sha256"]})
            metrics={s:program_metrics(program,actor,rows[role]) for s,rows in parts.items()}
            if monotonic()>=deadline: raise ProgramBudgetExhausted("confirmation native metrics exhausted wall budget",report=program.report)
            program.save(target/"program.json")
            reports[role]={"program":pointer(target/"program.json"),"metrics":metrics,"precheck_gate":gate_metrics(metrics["validation"],phase="precheck"),
                "pressure_gate":gate_metrics(metrics["pressure"],phase="precheck"),"search":pointer(target/"search.json"),"structure":pointer(target/"structure.json"),
                "compiler":pointer(target/"compiler.json") if arm=="B32" else None,
                "fitting_wall_seconds":monotonic()-begin,"fitting_wall_limit_seconds":limits["wall_seconds"]}
        quality_by_split,quality_gates_by_split=development_quality(collection["outcomes"])
        passed=all(g["passed"] for g in quality_gates_by_split.values()) and all(r["precheck_gate"]["passed"] and r["pressure_gate"]["passed"] for r in reports.values())
        result={"schema":"warehouse-v622-confirmation-fit.v1","plan_sha256":file_sha(path),"collection":pointer(path.parent/"collection/result.json"),
            "method":plan["method"],"actor":plan["actor"],"roles":reports,"quality_by_split":quality_by_split,"quality_gates_by_split":quality_gates_by_split,"native_passed":passed,
            "reference_quality_on_same_splits":quality_by_split,"reference_is_the_unchanged_NN":True,"reference_retention_comparison_is_self_comparison":True,
            "environment_resets":0,"actor_updates":0,"PPO_joint_steps":0,"publishable":False,"pilot_opened":False,
            "independent_reverification_required":True,"human_collaboration_claimed":False}
        load_plan(root,path);write_new(folder/"result.json",result);write_new(folder/"completed.json",{"result_sha256":file_sha(folder/"result.json"),"plan_sha256":file_sha(path)})
        return result
    except BaseException as error:
        write_new(folder/"failed.json",{"error_type":type(error).__name__,"error":str(error),"attempted_fit_work":capacity.plain(getattr(error,"report",None)),"retry_permitted":False})
        raise


def audit(root,plan_path):
    """Independent scalar metric/path/eligibility replay, not a sealed audit.

    This does not call the program fitter, production program metrics, or any
    environment. The earlier independent scalar arithmetic is reused.
    """
    import numpy as np
    from scripts import audit_warehouse_v622_capacity as independent
    path,plan=load_plan(root,plan_path);bound={}
    folder=path.parent/"programs";_,done=read_bound(root,pointer(folder/"completed.json"),bound)
    _,result=read_bound(root,{"path":str(folder/"result.json"),"sha256":done["result_sha256"]},bound)
    if done.get("plan_sha256")!=file_sha(path) or result.get("plan_sha256")!=file_sha(path) or (folder/"failed.json").exists() or set(result.get("roles",{}))!=set(capacity.ROLES):
        raise RuntimeError("cannot audit incomplete confirmation fit")
    if plan['arm']=='D32':
        for role in capacity.ROLES:
            search_path,search=read_bound(root,result['roles'][role].get('search'),bound)
            if search_path!=folder/role/'search.json': raise RuntimeError('noncanonical fresh D32 search evidence')
            require_complete_d32_search(search)
    actor,parts,collection=collection_rows(root,path)
    quality_by_split,quality_gates_by_split=development_quality(collection["outcomes"])
    if result.get("quality_by_split")!=quality_by_split or result.get("quality_gates_by_split")!=quality_gates_by_split:
        raise RuntimeError("quality flag/counters differ from separate public validation and pressure episodes")
    role_reports={}
    for role in capacity.ROLES:
        stored=result["roles"][role];program_path,raw=read_bound(root,stored["program"],bound)
        if program_path!=folder/role/"program.json": raise RuntimeError("confirmation program must have its own canonical role path")
        if raw.get("method")!=plan["method"] or raw.get("actor_role_sha256")!=actor.role_sha256[role] or raw.get("limits")!=capacity.LIMITS: raise RuntimeError("program method/NN/capacity differs")
        contexts={}
        for split in COUNTS:
            c=independent.context(raw,independent.encoded_data(actor,role,parts[split][role]),actor.roles[role].feature_names);contexts[split]=c
            for purpose in ("feedback","explanation"):
                independent.math.assert_equal(independent.section(c,raw[purpose+"_leaves"],purpose),stored["metrics"][split][purpose],role+"."+split+"."+purpose)
        for purpose,threshold in (("feedback",.93),("explanation",.95)):
            expected=[i for i in range(len(raw["leaf_logits"])) if (contexts["fit"]["leaves"]==i).any()
                and independent.math.eligible(independent.section(contexts["eligibility"],[i],purpose),threshold)]
            if raw[purpose+"_leaves"]!=expected: raise RuntimeError("fresh eligibility selection differs")
        for i,z in enumerate(np.asarray(raw["leaf_logits"],np.float32)):
            top=int(z.argmax())
            if top not in contexts["fit"]["proof"][i] or float(z[top])-float(np.delete(z.astype(np.float64),top).max())<.001: raise RuntimeError("final raw action path/margin proof failed")
        from experiments.warehouse.v622_capacity_audit_math import audit_search_budget,audit_frozen_table,audit_compiler
        _,search=read_bound(root,stored["search"],bound);_,structure=read_bound(root,stored["structure"],bound)
        fit_context=contexts['fit'];fit_arrays=(fit_context['x'],fit_context['m'],fit_context['p'],fit_context['episode'],actor.roles[role].feature_names,actor.role_sha256[role])
        audit_search_budget(search,plan['arm'],raw['report'],fit_arrays=fit_arrays)
        if plan['arm']=='D32':audit_frozen_table(search,raw,structure,fit_arrays)
        if plan['arm']=='B32':
            _,compiler=read_bound(root,stored['compiler'],bound);audit_compiler(compiler,actor.roles[role],structure)
        tree,_=independent.math.structure(structure,actor.roles[role].feature_names,enforce_limits=False)
        independent.math.assert_equal(tree,contexts['fit']['tree'],role+'.fixed_structure')
        if independent.independent_gate(stored["metrics"]["validation"]) is not stored["precheck_gate"]["passed"]: raise RuntimeError("fresh development gate differs")
        pressure_pass=independent.independent_gate(stored["metrics"]["pressure"])
        if stored["pressure_gate"]["passed"] is not pressure_pass: raise RuntimeError("pressure gate differs")
        role_reports[role]={"precheck_passed":stored["precheck_gate"]["passed"],"pressure_passed":pressure_pass,"physical_frames":{s:len(parts[s][role]) for s in COUNTS}}
    passed=all(g["passed"] for g in quality_gates_by_split.values()) and all(v["precheck_passed"] and v["pressure_passed"] for v in role_reports.values())
    if result["native_passed"] is not passed: raise RuntimeError("confirmation decision differs")
    load_plan(root,path)
    receipt={"schema":"warehouse-v622-confirmation-arithmetic-audit.v1","passed":True,"confirmation_passed":passed,"plan_sha256":file_sha(path),
        "fit_result_sha256":done["result_sha256"],"collection_sha256":file_sha(path.parent/"collection/result.json"),"roles":role_reports,
        "quality_by_split":quality_by_split,"quality_gates_by_split":quality_gates_by_split,
        "source_closure_sha256":plan["source_closure_sha256"],"method":plan["method"],"environment_resets":0,"fit_calls":0,"actor_updates":0,"PPO_joint_steps":0,
        "publishable":False,"pilot_opened":False,"new_training_authority":False,
        "scope":"fresh_development_scalar_arithmetic_not_one_time_final_program_audit_or_human_evaluation"}
    write_new(path.parent/"arithmetic-audit.json",receipt);return receipt


def development_quality(outcomes):
    """Separate absolute quality gates; no pooled pressure or self-retention claim."""
    quality={};gates={}
    for split in ('validation','pressure'):
        natural=[r for r in outcomes if r['case']['split']==split];n=COUNTS[split]
        if len(natural)!=n or any(r['case']['sampling_mode']!='sampled' for r in natural):
            raise RuntimeError("quality requires every registered natural episode in each split")
        if any(any(type(r[k]) is not int or r[k]<0 for k in ('deliveries','robot_collisions')) or type(r['shutdown']) is not bool for r in natural):
            raise RuntimeError("quality requires finite physical integer outcomes")
        q={'mean_deliveries':sum(r['deliveries'] for r in natural)/n,'mean_collisions':sum(r['robot_collisions'] for r in natural)/n,
            'shutdown_episode_rate':sum(r['shutdown'] for r in natural)/n,'episodes':n}
        checks={'deliveries_at_least_5_5':q['mean_deliveries']>=5.5,'collisions_at_most_6':q['mean_collisions']<=6.,
            'shutdown_at_most_five_percent':q['shutdown_episode_rate']<=.05}
        quality[split]=q;gates[split]={'passed':all(checks.values()),'checks':checks}
    return quality,gates
