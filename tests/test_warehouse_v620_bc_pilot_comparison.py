"""Engineering-only: no production imports, fitted models or environment resets."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
try:
    import warehouse_v620_bc_pilot_eval as p
except ModuleNotFoundError:
    from backend.training import warehouse_v620_bc_pilot_eval as p

INIT = {"robot_1": "1"*64, "robot_2": "2"*64}
SOURCE, BCSOURCE = "e"*64, "b"*64


def save(path, value):
    p.write_new(path, value)
    return p.pointer(path)


def metrics(coverage=.4, expected=.35, fa=.98, ea=.98, kl=.1, illegal=0, path=0):
    return {"feedback": {"coverage": coverage, "expected_actionable_coverage": expected,
            "agreement": fa, "kl": kl, "whole_tree_raw_illegal_count": illegal,
            "whole_tree_path_uncertified_count": path},
        "explanation": {"coverage": coverage, "agreement": ea, "kl": kl,
            "whole_tree_raw_illegal_count": illegal, "whole_tree_path_uncertified_count": path}}


def profile(**kwargs):
    return p.program_profile({role: metrics(**kwargs) for role in p.ROLES})


def episodes(deliveries=6, collisions=2, shutdowns=0):
    return [{"deliveries": deliveries, "collisions": collisions,
             "shutdown_episode": i < shutdowns} for i in range(24)]


def failed_summary(m1=None, m2=None):
    return {"chosen_method_arm": None,
        "next_stage": "common_bc_structure_learning_before_fresh_validation",
        "arms": {"M1": {"selection_score": m1 or [.1, .1, -.2, -8]},
                 "M2": {"selection_score": m2 or [.2, .1, -.2, -8]}}}


class Contract:
    SCHEMA = "warehouse-v620-shared-bc-pilot-spec.v1"
    CANDIDATE_ID = "warehouse-v620-01-shared-bc-pilot"
    STAGE_COUNTS = p.BC_STAGE_COUNTS

    @staticmethod
    def recipe(gamma):
        return {"schema": "synthetic-bc-recipe", "alignment": {"gamma": gamma}}

    @staticmethod
    def seed_binding(schedule, critic):
        return {"initialization_pair_seed": 6160, "schedule": schedule,
                "critic_seed": critic, "critic_actual_reset_seeds": [critic, critic],
                "training_rng_uses": "synthetic complete declared RNG fixture"}


class FakeRuntime:
    def __init__(self, root):
        self.root = root
        self.script_path = root / "scripts/run_warehouse_v620_bc_pilot_comparison.py"
        self.script_path.parent.mkdir()
        self.script_path.write_text("# synthetic saved entry only\n")
        self.bc_contract = Contract
        self.physics_sha256 = "f"*64
        self.methods = {"M1": "queued_path_constrained_tree_v1", "M2": "episode_robust_explanation_tree_v1"}
        self.current_source = SOURCE
        self.collect_calls, self.replay_calls, self.fit_calls = [], [], []
        self.metric_by_gamma = {name: metrics() for name in p.GAMMAS}
        self.outcome_by_gamma = {name: (6, 2, False) for name in p.GAMMAS}
        self.interrupt_at = self.drift_at = self.mutate_at = None
        self.native_complete = True
        self.summary = failed_summary()
        self.cv_pointer = save(root / "native-cv02-summary.json", self.summary)
        self.cv_bound = [(self.cv_pointer["path"], self.cv_pointer["sha256"])]
        self._cases = {}
        seed = 900000
        for split, count in (("fit", 50), ("calibration", 25), ("eligibility", 25), ("validation", 24)):
            self._cases[split] = [{"base": seed+i, "task": 1, "scenario_seed": seed+i,
                "sampling_mode": "argmax" if split == "fit" and i >= 25 else "sampled"}
                for i in range(count)]
            seed += count

    def source_sha256(self): return self.current_source
    def initial_role_hashes(self, seed):
        assert seed == 6160
        return dict(INIT)
    def spent_cases(self): return deepcopy(self._cases)
    def native_cv_proof(self, pointer):
        assert pointer == self.cv_pointer
        if not self.native_complete:
            raise RuntimeError("native 18-job completed chain incomplete")
        return {"bound_files": self.cv_bound, "summary": deepcopy(self.summary)}
    def load_actor(self, artifact):
        return SimpleNamespace(artifact=deepcopy(artifact), role_sha256=deepcopy(artifact["role_sha256"]))
    def actor_identity(self, actor): return dict(actor.role_sha256)
    def gamma_name(self, actor):
        return "gamma-001" if "gamma-001" in actor.artifact["path"] else "gamma-003"

    def audit(self, journal_path, seeds, phase):
        rows = [{"phase": phase, "seed": seed} for seed in seeds]
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        with journal_path.open("x") as stream:
            for row in rows: stream.write(json.dumps(row)+"\n")
        return {"resets": rows, "actual_reset_count": len(rows),
            "journal_path": str(journal_path.resolve()), "journal_sha256": p.file_sha(journal_path)}

    def collect(self, actor, case, *, journal_path):
        assert (self.root / "output/warehouse_v620_bc_pilot/comparison-01/method.lock.json").is_file()
        self.collect_calls.append(deepcopy(case))
        audit = self.audit(journal_path, [case["scenario_seed"]], "program")
        if self.interrupt_at == len(self.collect_calls):
            exc = KeyboardInterrupt("synthetic interruption after recorded reset")
            exc.warehouse_reset_audit = audit
            raise exc
        if self.drift_at == len(self.collect_calls): self.current_source = "d"*64
        if self.mutate_at == len(self.collect_calls): actor.role_sha256["robot_1"] = "0"*64
        rows = {role: [{"frame": 0, "base_seed": case["scenario_seed"],
            "actor_sha256": actor.artifact["sha256"], "actor_role_sha256": actor.artifact["role_sha256"][role],
            "sampled_action": "wait", "nn_sampled_action": "wait"}] for role in p.ROLES}
        d,c,s = self.outcome_by_gamma[self.gamma_name(actor)]
        outcome = {"seed": case["scenario_seed"], "action_seed": case["scenario_seed"],
            "sampling_mode": case["sampling_mode"], "frame_count": 1,
            "deliveries": d, "collisions": c, "shutdown_episode": s}
        return rows, {"reset_audit": audit, "episode_outcomes": [outcome]}

    def write_trace(self, rows, split, seeds, directory, artifact):
        path = directory / "joint-fullNN-trace.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"rows": rows})+"\n")
        return {**p.pointer(path), "row_count": 1, "seeds": seeds}
    def quality_row(self, rows, outcome, actor):
        return {k: outcome[k] for k in ("seed", "deliveries", "collisions", "shutdown_episode")}
    def replay_receipt(self, artifact, rows, seeds):
        q = p.quality_summary(rows)
        return {"passed": True, "episode_count": 24, "actor_sha256": artifact["sha256"],
            "actor_role_sha256": artifact["role_sha256"], "rows_sha256": p.digest(rows),
            "schedule_sha256": p.digest(seeds), "measured": {k: q[k] for k in
                ("mean_deliveries", "mean_collisions", "shutdown_episode_rate")}}
    def replay(self, artifact, rows, seeds, *, journal_path):
        assert len(self.collect_calls) % 124 == 0
        self.replay_calls.append(deepcopy(seeds))
        return {"receipt": self.replay_receipt(artifact, rows, seeds),
            "reset_audit": self.audit(journal_path, [s[0] for s in seeds], "quality")}
    def validate_replay_claim(self, artifact, rows, seeds, receipt):
        assert receipt == self.replay_receipt(artifact, rows, seeds)
    def fit_program(self, role, actor, parts, method, arm, target):
        assert len(self.collect_calls) % 124 == 0
        assert {k: len(v) for k,v in parts.items()} == {"fit": 50, "calibration":25,"eligibility":25,"validation":24}
        self.fit_calls.append((role, method))
        metric = deepcopy(self.metric_by_gamma[self.gamma_name(actor)])
        budget = {"passed": True, "global_total_rounds": 30}
        program = save(target, {"role": role, "method": method, "actor_role_sha256": actor.role_sha256[role],
            "validation": metric, "budget": budget})
        return {"program": program, "metrics": metric, "metrics_by_split": {s: metric for s in p.SPLITS},
            "search_budget_audit": budget, "actor_role_sha256": actor.role_sha256[role], "method": method}
    def validate_program_record(self, artifact, role, fitted, method, arm):
        value = json.loads(Path(fitted["program"]["path"]).read_bytes())
        if (value["validation"] != fitted["metrics"] or value["method"] != method
                or value["actor_role_sha256"] != artifact["role_sha256"][role]
                or value["budget"] != fitted["search_budget_audit"]):
            raise RuntimeError("saved program consistency/budget differs")


def fixture(root):
    rt = FakeRuntime(root)
    save(root / p.RECIPE_PATH, p.recipe())
    bc_main = root / "scripts/run_warehouse_v620_shared_bc_pilot.py"
    bc_main.write_text("# synthetic BC entry only\n")
    schedule, start = {}, 100000
    for name, count in p.BC_STAGE_COUNTS:
        schedule[name] = [{"scenario_seed": start+i, "action_seed": 200000+start+i} for i in range(count)]
        start += count
    critic = 500000
    binding = Contract.seed_binding(schedule, critic)
    paired = save(root / "paired-plan.json", {"schema":"warehouse-v620-paired-bc-seeds.v1",
        "candidate_id": Contract.CANDIDATE_ID, "pair_seed": 6160,
        "gamma_candidates": [.01,.03], "seed_binding": binding})
    clearance = save(root / "clearance.json", {"approved_for_shared_bc": True,
        "seed_binding_sha256":p.digest(binding),"conflict_count":0})
    spec = {"schema":p.SCHEMA,"candidate_id":p.CANDIDATE_ID,"source_closure_sha256":SOURCE,
        "bc_source_closure_sha256":BCSOURCE,"cv_summary":rt.cv_pointer,"bc_runs":{},
        "output_directory":str(root / "output/warehouse_v620_bc_pilot/comparison-01")}
    for index,(name,gamma) in enumerate(p.GAMMAS.items()):
        run_id = "pair-6160-"+name
        directory = root / "output/warehouse_v620_bc_pilot" / Contract.CANDIDATE_ID / run_id
        recipe = save(root / (name+".recipe.json"), Contract.recipe(gamma))
        bc_spec = {"schema":Contract.SCHEMA,"candidate_id":Contract.CANDIDATE_ID,"pair_seed":6160,
            "gamma":gamma,"run_id":run_id,"bc_seed_schedule":schedule,"critic_seed":critic,
            "source_closure_sha256":BCSOURCE,"cv_summary":rt.cv_pointer,"recipe":recipe,
            "paired_seed_plan":paired,"seed_clearance":clearance,"output_directory":str(directory)}
        spec_ptr = save(root / (name+".spec.json"),bc_spec)
        expected = []
        for stage,count in p.BC_STAGE_COUNTS:
            expected.extend({"phase":"formal_bc","seed":row["scenario_seed"]} for row in schedule[stage])
            if stage == "joint_augmentation_argmax": expected.extend({"phase":"formal_bc","seed":critic} for _ in range(2))
        bound = [(str(path),p.file_sha(path)) for path in
            (bc_main,Path(recipe["path"]),Path(paired["path"]),Path(clearance["path"]))]+rt.cv_bound
        auth = {"authorization_kind":"warehouse-v620-shared-bc-pilot-only-v1",
            "candidate_id":Contract.CANDIDATE_ID,"pair_seed":6160,"gamma":gamma,"run_id":run_id,
            "original_spec_sha256":p.digest(bc_spec),"source_closure_sha256":BCSOURCE,
            "recipe_file_sha256":recipe["sha256"],"bound_files_sha256":p.digest(bound),
            "initial_role_sha256":INIT,"expected_reset_sequence_sha256":p.digest(expected),
            "output_directory":str(directory),"action_physics_config_sha256":rt.physics_sha256,
            "PPO_joint_steps":0,"publishable":False,"fresh_validation_episodes":0}
        auth_ptr = save(root / (name+".authorization.json"),auth)
        directory.mkdir(parents=True)
        actor = directory / "final-bc.actor.npz"
        actor.write_bytes(("synthetic model "+name).encode())
        final_roles = {role:str(3+index*2+r)*64 for r,role in enumerate(p.ROLES)}
        data = {stage:[r["scenario_seed"] for r in schedule[stage]] for stage,_ in p.BC_STAGE_COUNTS}
        report = {"status":"bc_completed_not_admitted","pretrained_role_sha256":final_roles,
            "initial_role_sha256":INIT,"final_bc_actor":{**p.pointer(actor),"role_sha256":final_roles},
            "source_closure_sha256":BCSOURCE,"bc_recipe":Contract.recipe(gamma)["alignment"],
            "executable_recipe_file_sha256":recipe["sha256"],"PPO_joint_steps":0,
            "publishable":False,"fresh_validation_episodes":0,"mid_bc_extracted_tree_installed":False,
            "external_program_feedback":0,"bc_data_seed_blocks":data,
            "bc_data_seed_schedule_sha256":p.digest(data),"expected_bc_training_episodes":1270,
            "expected_critic_environment_resets":2,"collision_critic":{"validation_passed":True}}
        report_ptr = save(directory / "bc-pilot.training.json",report)
        journal = directory / "bc-pilot.resets.jsonl"
        journal.write_text("".join(json.dumps(r)+"\n" for r in expected))
        receipt = {**auth,"status":"bc_completed_not_admitted","error":None,
            "authorization_unchanged":True,"authorization_receipt_sha256":auth_ptr["sha256"],
            "bc_report_sha256":report_ptr["sha256"],"actual_reset_count":1272,"expected_reset_count":1272,
            "reset_full_sequence_matches":True,"reset_prefix_matches":True,"journal_path":str(journal),
            "journal_sha256":p.file_sha(journal),"actual_resets":expected}
        receipt_ptr=save(directory / "bc-pilot.receipt.json",receipt)
        spec["bc_runs"][name]={"spec":spec_ptr,"authorization":auth_ptr,"receipt":receipt_ptr,"report":report_ptr}
    return rt,spec


def run_both(tmp_path):
    rt,spec=fixture(tmp_path)
    plan=p.prepare(spec,rt)
    for name in p.GAMMAS:p.run_gamma(plan,name,rt)
    return rt,plan


def test_method_maximin_and_tie():
    assert p.choose_method(failed_summary()) == "M2"
    assert p.choose_method(failed_summary([.1,.2,-.3,-8],[.1,.2,-.3,-8])) == "M1"
    assert p.choose_method(failed_summary([None]*4,[None]*4)) == "M1"
    value=failed_summary();value["chosen_method_arm"]="M1"
    with pytest.raises(RuntimeError):p.choose_method(value)


def test_quality_boundaries_and_same_24():
    assert p.quality_summary(episodes(shutdowns=1))["quality_passed"]
    assert not p.quality_summary(episodes(shutdowns=2))["quality_passed"]
    assert not p.quality_summary(episodes(deliveries=5))["quality_passed"]
    assert not p.quality_summary(episodes(collisions=7))["quality_passed"]
    with pytest.raises(RuntimeError):p.quality_summary(episodes()[:-1])


def test_consistency_beats_coverage_and_no_coverage_gate():
    q=p.quality_summary(episodes())
    results={"gamma-001":{"quality":q,"program_profile":profile(coverage=.001,expected=.001)},
             "gamma-003":{"quality":q,"program_profile":profile(coverage=.99,expected=.99,ea=.94)}}
    assert p.choose_gamma(results)==("gamma-001","consistency_then_coverage")
    results["gamma-003"]=deepcopy(results["gamma-001"])
    assert p.choose_gamma(results)[0]=="gamma-001"


def test_all_inconsistent_margin_before_coverage():
    q=p.quality_summary(episodes())
    results={"gamma-001":{"quality":q,"program_profile":profile(coverage=.1,ea=.94)},
             "gamma-003":{"quality":q,"program_profile":profile(coverage=.99,ea=.7)}}
    assert p.choose_gamma(results)==("gamma-001","consistency_margin_fallback")


def test_missing_qualified_metrics_are_zero_margin_not_success():
    value=profile(fa=None,ea=None,kl=None,coverage=0,expected=0)
    assert value["minimum_normalized_consistency_margin"]==0
    assert value["maximum_qualified_kl"] is None
    assert not value["program_consistency_pass"]
    assert not profile(illegal=1)["program_consistency_pass"]
    assert not profile(path=1)["program_consistency_pass"]


def test_hard_quality_only_eligibility():
    results={"gamma-001":{"quality":p.quality_summary(episodes(deliveries=5)),"program_profile":profile()},
        "gamma-003":{"quality":p.quality_summary(episodes()),"program_profile":profile(coverage=0,fa=None,ea=None,kl=None)}}
    assert p.choose_gamma(results)[0]=="gamma-003"
    results["gamma-003"]["quality"]=results["gamma-001"]["quality"]
    assert p.choose_gamma(results)==(None,"closed_both_quality_failed")


def test_full_stub_pipeline_no_new_quality_samples(tmp_path):
    rt,plan=run_both(tmp_path)
    selected=json.loads(Path(p.select(plan,rt)["path"]).read_bytes())
    assert len(rt.collect_calls)==248 and len(rt.replay_calls)==2 and len(rt.fit_calls)==4
    assert [len(x) for x in rt.replay_calls]==[24,24]
    assert rt.collect_calls[:124]==rt.collect_calls[124:]
    assert selected["selected_gamma"]==.01
    assert selected["fresh_precheck_requires_new_275_and_refitted_programs"]
    assert not selected["fresh_precheck_failure_may_switch_gamma"]
    assert not selected["fresh_collection_authority"] and not selected["admission_evidence"]
    with pytest.raises(RuntimeError):p.select(plan,rt)
    with pytest.raises(FileExistsError):p.run_gamma(plan,"gamma-001",rt)


@pytest.mark.parametrize("fault",["cv","source","bc_source","asset","reset","shared_init","paired_plan"])
def test_invalid_precondition_before_rollout(tmp_path,fault):
    rt,spec=fixture(tmp_path)
    if fault=="cv":rt.native_complete=False
    if fault=="source":rt.current_source="0"*64
    if fault=="bc_source":spec["bc_source_closure_sha256"]="0"*64
    if fault=="asset":
        report=json.loads(Path(spec["bc_runs"]["gamma-001"]["report"]["path"]).read_bytes())
        Path(report["final_bc_actor"]["path"]).write_bytes(b"mutated")
    if fault=="reset":
        receipt=json.loads(Path(spec["bc_runs"]["gamma-001"]["receipt"]["path"]).read_bytes())
        Path(receipt["journal_path"]).write_text('{"phase":"formal_bc","seed":0}\n')
    if fault=="shared_init":rt.initial_role_hashes=lambda seed:{"robot_1":"0"*64,"robot_2":"2"*64}
    if fault=="paired_plan":
        bcspec=json.loads(Path(spec["bc_runs"]["gamma-003"]["spec"]["path"]).read_bytes())
        Path(bcspec["paired_seed_plan"]["path"]).write_text('{}')
    with pytest.raises((RuntimeError,ValueError)):p.prepare(spec,rt)
    assert not rt.collect_calls and not rt.fit_calls
    assert not Path(spec["output_directory"]).exists()


@pytest.mark.parametrize("kind",["interrupt","source","actor"])
def test_partial_journal_failure_and_no_retry(tmp_path,kind):
    rt,spec=fixture(tmp_path)
    plan=p.prepare(spec,rt)
    setattr(rt,{"interrupt":"interrupt_at","source":"drift_at","actor":"mutate_at"}[kind],1)
    with pytest.raises((KeyboardInterrupt,RuntimeError)):p.run_gamma(plan,"gamma-001",rt)
    failure=json.loads((Path(spec["output_directory"])/"gamma-001/failure.json").read_bytes())
    assert failure["status"]=="failed_or_interrupted"
    assert failure["attempted_persistent_journals"][0]["sha256"]
    assert not failure["scientific_retry_permitted"]
    assert not rt.fit_calls
    if kind=="interrupt":assert failure["collector_failure_reset_audit"]["actual_reset_count"]==1
    rt.current_source=SOURCE
    with pytest.raises(FileExistsError):p.run_gamma(plan,"gamma-001",rt)


def test_missing_gamma_cannot_select(tmp_path):
    rt,spec=fixture(tmp_path);plan=p.prepare(spec,rt)
    p.run_gamma(plan,"gamma-001",rt)
    with pytest.raises(FileNotFoundError):p.select(plan,rt)
    assert not (Path(spec["output_directory"])/"gamma.lock.json").exists()


def test_both_quality_fail_close_without_fresh(tmp_path):
    rt,spec=fixture(tmp_path)
    rt.outcome_by_gamma={name:(5,2,False) for name in p.GAMMAS}
    plan=p.prepare(spec,rt)
    for name in p.GAMMAS:p.run_gamma(plan,name,rt)
    result=p.select(plan,rt)
    assert Path(result["path"]).name=="closed.json"
    closed=json.loads(Path(result["path"]).read_bytes())
    assert closed["selected_gamma"] is None and not closed["fresh_collection_authority"]


def test_saved_result_hash_tamper_rejects(tmp_path):
    rt,plan=run_both(tmp_path)
    directory=Path(plan).parent
    (directory/"gamma-003/result.json").write_text('{}')
    with pytest.raises(RuntimeError):p.select(plan,rt)


def test_method_lock_tamper_before_first_rollout(tmp_path):
    rt,spec=fixture(tmp_path);plan=p.prepare(spec,rt)
    (plan.parent/"method.lock.json").write_text('{}')
    with pytest.raises(RuntimeError):p.run_gamma(plan,"gamma-001",rt)
    assert not rt.collect_calls


def rewrite(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')


def test_program_record_rechecked_even_if_result_pointer_rewritten(tmp_path):
    rt,plan=run_both(tmp_path)
    directory=plan.parent/'gamma-001'
    result=json.loads((directory/'result.json').read_bytes())
    # Invent a high score in both report copies without changing the real program.
    for role in p.ROLES:
        result['programs'][role]['metrics']['explanation']['coverage']=.99
        result['role_metrics'][role]['explanation']['coverage']=.99
    result['program_profile']=p.program_profile(result['role_metrics'])
    rewrite(directory/'result.json',result)
    completed=json.loads((directory/'completed.json').read_bytes())
    completed['result']=p.pointer(directory/'result.json')
    rewrite(directory/'completed.json',completed)
    with pytest.raises(RuntimeError,match='program consistency'):p.select(plan,rt)


def test_failed_model_load_gets_zero_reset_failure_receipt(tmp_path):
    rt,spec=fixture(tmp_path);plan=p.prepare(spec,rt)
    def fail(artifact):raise RuntimeError('synthetic malformed actor')
    rt.load_actor=fail
    with pytest.raises(RuntimeError):p.run_gamma(plan,'gamma-001',rt)
    failure=json.loads((plan.parent/'gamma-001/failure.json').read_bytes())
    assert failure['completed_collection_episodes']==0
    assert failure['attempted_persistent_journals']==[]
    assert not rt.collect_calls


def test_quality_only_from_same_validation_outcomes(tmp_path):
    rt,plan=run_both(tmp_path)
    directory=plan.parent/'gamma-001'
    result=json.loads((directory/'result.json').read_bytes())
    # A fully rehashed self-reported score still fails its physical replay claim.
    quality_path=Path(result['quality_episodes']['path'])
    rows=json.loads(quality_path.read_bytes());rows[0]['deliveries']+=1
    rewrite(quality_path,rows)
    result['quality_episodes']=p.pointer(quality_path)
    result['quality']=p.quality_summary(rows)
    rewrite(directory/'result.json',result)
    completed=json.loads((directory/'completed.json').read_bytes())
    completed['result']=p.pointer(directory/'result.json');rewrite(directory/'completed.json',completed)
    with pytest.raises(AssertionError):p.select(plan,rt)
    assert len(rt.collect_calls)==248 and len(rt.replay_calls)==2


def test_plan_has_no_fresh_or_gradient_authority(tmp_path):
    rt,spec=fixture(tmp_path);plan=p.prepare(spec,rt)
    value=json.loads(plan.read_bytes())
    assert value['recipe']['actor_updates']==value['recipe']['program_gradient_updates']==0
    assert value['recipe']['pilot_coverage_hard_gate'] is None
    assert not value['recipe']['fresh_collection_authority']
    assert not value['recipe']['fresh_precheck_failure_may_switch_gamma']
    with pytest.raises(ValueError):p.run_gamma(plan,'gamma-002',rt)
    assert not rt.collect_calls


def test_expected_then_kl_then_quality_then_smaller_gamma():
    q=p.quality_summary(episodes())
    def result(prof,quality=q):return {'quality':quality,'program_profile':prof}
    a,b=profile(expected=.2),profile(expected=.3)
    assert p.choose_gamma({'gamma-001':result(a),'gamma-003':result(b)})[0]=='gamma-003'
    a,b=profile(kl=.2),profile(kl=.1)
    assert p.choose_gamma({'gamma-001':result(a),'gamma-003':result(b)})[0]=='gamma-003'
    a,b=profile(),profile()
    higher=p.quality_summary(episodes(deliveries=7))
    assert p.choose_gamma({'gamma-001':result(a),'gamma-003':result(b,higher)})[0]=='gamma-003'
    assert p.choose_gamma({'gamma-001':result(a),'gamma-003':result(b)})[0]=='gamma-001'


def frozen_synthetic_seed_plan():
    parts, offset = {}, 0
    for split, count in (("fit",100),("calibration",25),("eligibility",75),("validation",75)):
        parts[split]=[{"base":10000+offset+i,"task":(offset+i)%3+1,
            "scenario_seed":(10000+offset+i)*10+(offset+i)%3+1,
            "action_seed":2000000000+offset+i,
            "sampling_mode":"argmax" if split=="fit" and i>=50 else "sampled"}
            for i in range(count)]
        offset+=count
    independent=[{"base":base,"task":task,"stream":stream,"scenario_seed":base*10+task,
        "action_seed":2000010000+(base-20000)*9+(task-1)*3+stream,"sampling_mode":"sampled"}
        for base in range(20000,20024) for task in (1,2,3) for stream in (0,1,2)]
    return {"splits":parts,"independent":independent}


def test_native_selected_lock_is_frozen_entry_compatible_without_collection(tmp_path):
    from scripts import run_warehouse_v620_frozen_gate as gate
    rt,plan=run_both(tmp_path)
    p.select(plan,rt)
    model_pointer=p.pointer(plan.parent/'model-method.lock.json')
    model=json.loads(Path(model_pointer['path']).read_bytes())
    receipt=gate.read_bound(model['method_selection_receipt'])
    assert receipt['gamma_lock']==model['gamma_lock']
    assert set(model['recipe'])==set(model['method_selection_receipt'])=={'path','sha256'}
    assert model['bc_source_closure_sha256']==BCSOURCE!=model['source_closure_sha256']
    assert not model['fresh_collection_authority'] and not receipt['fresh_collection_authority']
    seeds=frozen_synthetic_seed_plan()
    seeds_ptr=save(tmp_path/'synthetic-frozen-seeds.json',seeds)
    output=tmp_path/'synthetic-frozen-output-not-created'
    ledger={'schema':'warehouse-frozen-seed-clearance.v1','approved_for_frozen_validation':True,
        'purpose':gate.PURPOSE,'candidate_id':p.CANDIDATE_ID,'seed_plan_sha256':gate.digest(seeds),
        'authorized_reset_schedule_sha256':gate.digest(gate.reset_contract(seeds)),
        'model_method_lock_sha256':model_pointer['sha256'],'source_closure_sha256':SOURCE,
        'recipe_sha256':model['recipe']['sha256'],'output_directory':str(output),
        'unresolved':[],'remaining_conflicts':[],'PPO_joint_steps':0,'publishable':False}
    ledger_ptr=save(tmp_path/'synthetic-clearance-never-executed.json',ledger)
    frozen_rt=SimpleNamespace(source=lambda:SOURCE,load=rt.load_actor,identity=rt.actor_identity)
    checked,checked_seeds,actor=gate.validate_bindings(model_pointer,seeds_ptr,ledger_ptr,frozen_rt,output)
    assert checked==model and checked_seeds==seeds
    assert len(rt.collect_calls)==248 and not output.exists()
    # Editing even the native receipt invalidates the exact pointer immediately.
    Path(model['method_selection_receipt']['path']).write_text('{}')
    with pytest.raises(RuntimeError,match='bound artifact'):
        gate.validate_bindings(model_pointer,seeds_ptr,ledger_ptr,frozen_rt,output)
    assert len(rt.collect_calls)==248


def test_quality_failure_emits_no_downstream_model_method_lock(tmp_path):
    rt,spec=fixture(tmp_path)
    rt.outcome_by_gamma={name:(5,2,False) for name in p.GAMMAS}
    plan=p.prepare(spec,rt)
    for name in p.GAMMAS:p.run_gamma(plan,name,rt)
    p.select(plan,rt)
    assert not (plan.parent/'model-method.lock.json').exists()
    assert not (plan.parent/'selected-method.receipt.json').exists()
