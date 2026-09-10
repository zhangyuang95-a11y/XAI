"""One finite, genuinely restored OwnShutdown continuation stage.

Only the completed, registered fixed 250k source is admitted in production.
Capability is a stopping condition, not fabricated source provenance. This stage
preserves beta, PPO, partners, public history, Adam and RNG; it never reads final
test trajectories, grants release qualification, or overwrites the parent ledger.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
import torch
from .warehouse_native_common import ROOT, digest, file_hash
from .warehouse_native import atomic_torch_save
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_partner_mix_run import (
    write_bytes, write_json, decode, save_batch, export_actor, AUTHORIZATION, AUTHORIZATION_SHA)
from .warehouse_native_cycle_budget import CycleBudget
from env.warehouse.domain import collaborative_study_config
from . import warehouse_native_shutdown_run as original
from . import warehouse_native_shutdown_result as source_records
from . import warehouse_native_shutdown_continuation_trainer as native

VERSION = "warehouse-native-own-shutdown-continuation-run.v1"
ARMS = ("beta0", "beta1")
PRODUCTION_SOURCE = ROOT / "output/warehouse_native/own_shutdown_20260909_r1"
SOURCE_STEP = 250000


def _json(path): return json.loads(Path(path).read_bytes())
def _same(a, b, reason):
    if digest(a) != digest(b): raise ValueError(reason)


def sources():
    from .warehouse_native_shutdown_stage_evaluation import execution_sources
    result = original.sources(); result.update(native.execution_sources()); result.update(execution_sources())
    for path in (Path(__file__), Path(source_records.__file__)):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def _descriptor(value): return {k:v for k,v in value.items() if k != "inspection_counts"}


def _registered(source_run, shutdown_arm, source_step, fixture, expected=None):
    run = Path(source_run).expanduser().resolve()
    if shutdown_arm not in ARMS or type(source_step) is not int or source_step <= 0:
        raise ValueError("Explicit beta and a positive registered source step required")
    if not fixture and (run != PRODUCTION_SOURCE.resolve() or source_step != SOURCE_STEP):
        raise ValueError("Only the fixed own-shutdown 250000 source is supported")
    descriptor, prepared, protocol, scenes = source_records.registered_source(run, shutdown_arm,
        source_step, allow_test_fixture=fixture)
    descriptor = _descriptor(descriptor)
    if prepared["version"] != original.VERSION or descriptor["shutdown_arm"] != shutdown_arm:
        raise ValueError("Source must retain its genuine OwnShutdown run and beta")
    if expected is not None: _same(descriptor, _descriptor(expected), "Registered source records changed")
    return descriptor, prepared, protocol, scenes


def load_source(source_run, shutdown_arm, source_step, *, device, allow_test_fixture=False, expected=None):
    # All actual source registration/record gates precede any checkpoint decoding or NN construction.
    execution = sources()
    descriptor, prepared, protocol, scenes = _registered(source_run, shutdown_arm, source_step, allow_test_fixture, expected)
    if prepared["device"] != device: raise ValueError("The inherited training device cannot change")
    run = Path(source_run).expanduser().resolve()
    saved, actual_protocol, actual_scenes, ancestor = original.read_prepared(run, allow_test_fixture=allow_test_fixture)
    _same(saved, prepared, "Source prepared record changed during full restoration")
    _same(actual_protocol, protocol, "Source protocol differs")
    _same(actual_scenes, scenes, "Source scenarios differ")
    learner = original._trainer(saved, protocol, ancestor, shutdown_arm)
    checkpoint = original.original._check_file(run, descriptor["checkpoint"])
    payload = decode(checkpoint, descriptor["checkpoint"]["sha256"])
    original._envelope(payload, learner, shutdown_arm)
    if (payload.get("operation_id") != descriptor["operation_id"]
            or payload.get("audit", {}).get("neural_overrides") != 0
            or set(payload.get("evidence", {})) != {"arrays", "trace"}):
        raise ValueError("Source checkpoint is not the actual acknowledged neural output")
    for binding in payload["evidence"].values(): original.original._check_file(run, binding)
    learner.load_state_dict(payload["trainer"])
    if learner.joint_steps != source_step: raise ValueError("Restored source clock differs")
    actor = run / "branches" / shutdown_arm / "actors" / f"actor_{source_step:07d}.npz"
    if file_hash(actor) != descriptor["actor_sha256"]: raise ValueError("Source Actor bytes changed")
    # Compare the actual restored six arrays, without any neural forward or action selection.
    weights = {k.removeprefix("actor."): v.detach().cpu().numpy()
        for k,v in learner.model.state_dict().items() if k.startswith("actor.")}
    with np.load(io.BytesIO(actor.read_bytes()), allow_pickle=False) as data:
        if set(data.files) != {*weights, "metadata_json"}: raise ValueError("Source Actor inventory differs")
        if any(not np.array_equal(value, data[key]) for key,value in weights.items()):
            raise ValueError("Source NPZ and genuine checkpoint Actor parameters differ")
    if sources() != execution: raise ValueError("Sources changed during source loading")
    return learner, descriptor, scenes


def _trainer(prepared, protocol, source):
    return native.ShutdownContinuationTrainer(protocol, source,
        expected_source_state_sha256=prepared["source_state_sha256"],
        source_checkpoint_sha256=prepared["source_checkpoint_sha256"],
        device=prepared["device"], test_fixture=prepared["test_fixture"])


def _material(output, fixture):
    prepared = _json(output / "prepared.json")
    if (prepared.get("version") != VERSION or prepared.get("test_fixture") is not fixture
            or prepared.get("runtime_sources") != sources() or prepared.get("branch") != "own_credit"
            or prepared.get("shutdown_arm") not in ARMS or prepared.get("stop_when_ready") is not True):
        raise ValueError("Prepared stage identity, scope or sources changed")
    for key,value in prepared["identity"].items(): _same(prepared.get(key), value, "Stage identity differs: " + key)
    if prepared["authorization_record_sha256"] != AUTHORIZATION_SHA or file_hash(AUTHORIZATION) != AUTHORIZATION_SHA:
        raise ValueError("Autonomous training authorization changed")
    for name,key in (("authorization.json", "authorization_record_sha256"), ("baselines.json", "baselines_sha256")):
        if file_hash(output/name) != prepared[key]: raise ValueError("Bound stage file changed: " + name)
    for path,sha in prepared["runtime_sources"].items():
        if file_hash(output/"source_snapshot"/path) != sha: raise ValueError("Archived execution source changed")
    protocol, scenes = _json(output/"protocol.json"), _json(output/"scenarios.json")
    _same(digest(protocol), prepared["protocol_sha256"], "Stage protocol changed")
    _same(digest(scenes), prepared["scenario_manifest_sha256"], "Stage scenarios changed")
    cap=protocol["budget"]["maximum_ppo_joint_steps"]
    endpoints=protocol["evaluation"]["checkpoints_ppo_steps"]
    maximum=len(endpoints)*3*len(scenes["splits"]["validation"])*scenes["configuration"]["horizon"]
    if (prepared["primary_endpoint"] != cap or prepared["validation_endpoints"] != endpoints
            or prepared["cycle_id"] != protocol["cycle_id"] or prepared["shutdown_arm"] != protocol["shutdown_arm"]
            or prepared["own_shutdown_beta"] != protocol["own_shutdown_beta"]):
        raise ValueError("Registered stage boundaries or beta differ")
    _same(prepared["budget_caps"], {prepared["shutdown_arm"]:{"ppo":cap,"evaluation":maximum}}, "Finite caps differ")
    initial=_json(output/"initialization_check.json")
    if initial.get("identical_learning_state") is not True or initial["source_state_sha256"] != prepared["source_state_sha256"]:
        raise ValueError("Original learning state preservation was not confirmed")
    return prepared,protocol,scenes


def read_prepared(output, *, allow_test_fixture=False):
    output=Path(output).expanduser().resolve()
    prepared,protocol,scenes=_material(output,allow_test_fixture)
    descriptor=prepared["source_descriptor"]
    source,actual,source_scenes=load_source(descriptor["run"],descriptor["shutdown_arm"],descriptor["step"],
        device=prepared["device"],allow_test_fixture=allow_test_fixture,expected=descriptor)
    _same(initialization_sha256(source.state_dict()),prepared["source_state_sha256"],"Full source state differs")
    _same(scenes,source_scenes,"Continuation cannot change scenarios")
    return prepared,protocol,scenes,source


def prepare(output, *, source_run=PRODUCTION_SOURCE, shutdown_arm, source_step=SOURCE_STEP,
            ppo_cap=500000, validation_interval=100000, cycle_id=None, device="mps", allow_test_fixture=False):
    output=Path(output).expanduser().resolve(); source_run=Path(source_run).expanduser().resolve()
    if output.exists(): raise FileExistsError("Use a new finite continuation stage directory")
    if output==source_run or output in source_run.parents or source_run in output.parents:
        raise ValueError("New stage must be separate from its original source")
    if type(allow_test_fixture) is not bool or device not in ("cpu","mps"):
        raise ValueError("Explicit fixture scope and device required")
    if file_hash(AUTHORIZATION)!=AUTHORIZATION_SHA: raise ValueError("Autonomous authorization changed")
    if type(ppo_cap) is not int or ppo_cap<16 or type(validation_interval) is not int or validation_interval<1:
        raise ValueError("Finite positive cap and interval required")
    if not allow_test_fixture and (ppo_cap!=500000 or validation_interval!=100000):
        raise ValueError("This stage freezes the 500000 cap and 100000 validation interval")
    source,descriptor,scenes=load_source(source_run,shutdown_arm,source_step,device=device,allow_test_fixture=allow_test_fixture)
    n=source.cfg["environments"]
    if ppo_cap%n or validation_interval%n: raise ValueError("Cap and interval must align with environment batch")
    endpoints=list(range(validation_interval,ppo_cap,validation_interval))+[ppo_cap]
    probe=min(4096,ppo_cap);probe-=probe%n
    state=source.state_dict();state_sha=initialization_sha256(state);cycle_id=cycle_id or output.name
    protocol=native.make_protocol(source,source_checkpoint_sha256=descriptor["checkpoint"]["sha256"],
        source_state_sha256=state_sha,ppo_cap=ppo_cap,cycle_id=cycle_id,evaluation_checkpoints=endpoints,test_fixture=allow_test_fixture)
    maximum=len(endpoints)*3*len(scenes["splits"]["validation"])*scenes["configuration"]["horizon"]
    prepared={"version":VERSION,"cycle_id":cycle_id,"device":device,"branch":"own_credit",
        "shutdown_arm":shutdown_arm,"own_shutdown_beta":source.beta,"source_descriptor":descriptor,
        "source_checkpoint_sha256":descriptor["checkpoint"]["sha256"],"source_state_sha256":state_sha,
        "protocol_sha256":digest(protocol),"scenario_manifest_sha256":digest(scenes),
        "baselines_sha256":file_hash(source_run/"baselines.json"),"runtime_sources":sources(),
        "authorization_record_sha256":AUTHORIZATION_SHA,"budget_caps":{shutdown_arm:{"ppo":ppo_cap,"evaluation":maximum}},
        "primary_endpoint":ppo_cap,"validation_endpoints":endpoints,"probe_endpoint":probe,
        "test_fixture":allow_test_fixture,"stop_when_ready":True,"formal_ready":False,"created_unix":time.time()}
    prepared["identity"]={k:copy.deepcopy(v) for k,v in prepared.items() if k not in ("created_unix","formal_ready")}
    learner=_trainer(prepared,protocol,source);saved=learner.state_dict();matched={}
    for key in ("model","optimizers","envs","rng","python_rng","numpy_rng","torch_rng","partner_kinds",
                "program_roles","scenario_ids","episode_context","episode_returns","episode_reward_components"):
        matched[key]=initialization_sha256(saved[key]);_same(matched[key],initialization_sha256(state[key]),"Fork changed "+key)
    if device=="mps":
        matched["mps_rng"]=initialization_sha256(saved["mps_rng"])
        _same(matched["mps_rng"],initialization_sha256(state["mps_rng"]),"Fork changed MPS RNG")
    if sources()!=prepared["runtime_sources"]:raise ValueError("Sources changed during preparation")
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (("prepared.json",prepared),("protocol.json",protocol),("scenarios.json",scenes)):
        write_json(output/name,value)
    write_bytes(output/"baselines.json",(source_run/"baselines.json").read_bytes())
    write_bytes(output/"authorization.json",AUTHORIZATION.read_bytes())
    for name in prepared["runtime_sources"]:write_bytes(output/"source_snapshot"/name,(ROOT/name).read_bytes())
    ledger=CycleBudget.create(output,prepared["identity"])
    with ledger.lease():
        checkpoint=output/"branches"/shutdown_arm/"checkpoints/initial.pt"
        atomic_torch_save(checkpoint,{"version":VERSION,"cycle_id":cycle_id,"branch":"own_credit","shutdown_arm":shutdown_arm,
            "operation_id":None,"trainer":saved,"new_environment_steps":0})
        ledger.initialize_head("ppo",shutdown_arm,str(checkpoint.relative_to(output)),file_hash(checkpoint))
        reference=output/"branches"/shutdown_arm/"initial_evaluation_reference.json"
        write_json(reference,{"source_descriptor":descriptor,"new_environment_steps":0})
        ledger.initialize_head("evaluation",shutdown_arm,str(reference.relative_to(output)),file_hash(reference))
    write_json(output/"initialization_check.json",{"identical_learning_state":True,"matched":matched,
        "source_state_sha256":state_sha,"new_environment_steps":0})
    return {"status":"prepared","output":str(output),"cycle_id":cycle_id,"shutdown_arm":shutdown_arm,
        "ppo_cap":ppo_cap,"evaluation_cap":maximum,"probe_endpoint":probe,"validation_endpoints":endpoints,"formal_ready":False}


def _envelope(payload,trainer,arm):
    if (payload.get("version")!=VERSION or payload.get("branch")!="own_credit" or payload.get("shutdown_arm")!=arm
            or payload.get("cycle_id")!=trainer.protocol["cycle_id"]):raise ValueError("Wrong continuation checkpoint identity")


def train_to(output, trainer, ledger, arm, target):
    if arm not in ARMS or trainer.shutdown_arm != arm:
        raise ValueError("Learner and ledger arm differ")
    n = trainer.cfg["environments"]
    while trainer.joint_steps < target:
        if any(ledger.read()["branches"][arm][kind]["pending"] for kind in ("ppo", "evaluation")):
            raise ValueError("Unconfirmed operation requires diagnosis; never resample")
        amount = min(n * trainer.cfg["rollout_steps"], target - trainer.joint_steps, ledger.remaining("ppo", arm))
        amount -= amount % n
        if amount <= 0:
            raise ValueError("Finite budget cannot reach this endpoint")
        head = ledger.head("ppo", arm)
        if trainer.joint_steps != head["step"]:
            raise ValueError("Learner clock differs from acknowledged head")
        before = initialization_sha256(trainer.state_dict())
        old = decode(output / head["path"], head["sha256"])
        _envelope(old, trainer, arm)
        _same(initialization_sha256(old["trainer"]), before, "Learner differs from confirmed predecessor")
        operation_id = "ppo_" + uuid.uuid4().hex
        reservation = ledger.reserve("ppo", arm, amount, operation_id, expected_step=trainer.joint_steps)
        if not reservation["execution_permitted"]:
            raise ValueError("Repeated reservation is not execution permission")
        started = time.monotonic()
        batch, metrics = trainer.train_chunk(amount // n)
        if (trainer.joint_steps != head["step"] + amount or len(batch["transition_records"]) != amount
                or batch["audit"]["neural_overrides"] != 0
                or any(row.get("shutdown_arm") != arm for row in batch["transition_records"])
                or any(not np.isfinite(v) for v in metrics.values())):
            raise ValueError("Actual reward training differs from its reservation or NN contract")
        evidence = save_batch(output, arm, operation_id, batch)
        episodes = copy.deepcopy(trainer.completed_episodes)
        trainer.completed_episodes.clear()
        checkpoint = output / "branches" / arm / "checkpoints" / f"step_{trainer.joint_steps:07d}_{operation_id}.pt"
        payload = {"version": VERSION, "cycle_id": trainer.protocol["cycle_id"], "branch": "own_credit",
            "shutdown_arm": arm, "operation_id": operation_id, "trainer": trainer.state_dict(),
            "audit": batch["audit"], "metrics": metrics, "evidence": evidence, "actual_steps": amount,
            "completed_episodes": episodes, "before_state_sha256": before,
            "elapsed_seconds": time.monotonic() - started}
        atomic_torch_save(checkpoint, payload)
        ledger.ack(operation_id, str(checkpoint.relative_to(output)), file_hash(checkpoint), amount)
        write_json(output / "progress.json", {"status": "training", "shutdown_arm": arm,
            "branch_ppo_steps": trainer.joint_steps, "target": target, "last_metrics": metrics,
            "last_chunk_seconds": payload["elapsed_seconds"], "ledger": ledger.read()}, replace=True)
        print(json.dumps({"event": "ppo_ack", "shutdown_arm": arm, "steps": trainer.joint_steps,
            "seconds": round(payload["elapsed_seconds"], 3), "neural_overrides": 0}), flush=True)


def _evaluation_material(output,trainer,ledger,arm,scenes):
    from .warehouse_native_shutdown_stage_evaluation import REQUIRED_BINDINGS
    actor=export_actor(output,trainer,arm,ledger.head("ppo",arm))
    bindings={key:actor.metadata[key] for key in REQUIRED_BINDINGS if key!="actor_sha256"}
    bindings["actor_sha256"]=actor.artifact_sha256
    return actor,bindings


def evaluate_boundary(output,trainer,ledger,arm,scenes):
    from .warehouse_native_shutdown_stage_evaluation import evaluate
    actor,bindings=_evaluation_material(output,trainer,ledger,arm,scenes)
    baselines=_json(output/"baselines.json")
    confirmed={key for key,op in ledger.read()["operations"].items() if op["status"]=="acknowledged"
        and op["request"]["kind"]=="evaluation" and op["request"]["branch"]==arm}
    def reserve(context):
        operation=ledger.reserve("evaluation",arm,context["horizon"],context["operation_id"])
        if not operation["execution_permitted"]:raise ValueError("Evaluation reservation is not repeat permission")
    def ack(result):
        ledger.ack(result["operation_id"],str(Path(result["row_path"]).relative_to(output)),result["row_sha256"],result["actual_steps"])
    maximum=3*len(scenes["splits"]["validation"])*scenes["configuration"]["horizon"]
    return evaluate(actor,scenes["splits"]["validation"],trainer.protocol,
        output/"branches"/arm/"validation"/f"step_{trainer.joint_steps:07d}",maximum,
        expected_bindings=bindings,reference_report=baselines["reference"],random_report=baselines["random"],
        before_episode=reserve,on_episode=ack,confirmed_operation_ids=confirmed,allow_test_fixture=trainer.test_fixture,
        config=collaborative_study_config(horizon=scenes["configuration"]["horizon"]))


def _report_record(output,arm,step,prepared,protocol,scenes,account,fixture):
    """Original saved matrix + actual external acknowledgments, never fresh evaluation."""
    from . import warehouse_native_shutdown_stage_evaluation as evaluation
    folder=output/"branches"/arm/"validation"/f"step_{step:07d}"
    report=_json(folder/"report.json");manifest=_json(folder/"manifest.json")
    binding=report["actor_bindings"]
    if (binding["joint_steps"]!=step or binding["shutdown_arm"]!=arm
            or binding["protocol_sha256"]!=prepared["protocol_sha256"]
            or binding["source_checkpoint_sha256"]!=prepared["source_checkpoint_sha256"]
            or binding["initialization_sha256"]!=prepared["source_state_sha256"]):
        raise ValueError("Stored validation belongs to another learner boundary")
    actor=output/"branches"/arm/"actors"/f"actor_{step:07d}.npz"
    parity=_json(actor.with_suffix(".json"))
    matches=[op for op in account["operations"].values() if op["status"]=="acknowledged"
        and op["request"]["kind"]=="ppo" and op["request"]["branch"]==arm
        and op["completion"]["checkpoint"]["step"]==step]
    if (len(matches)!=1 or parity["checkpoint_sha256"]!=matches[0]["completion"]["checkpoint"]["sha256"]
            or parity["actor"]["sha256"]!=binding["actor_sha256"] or parity.get("argmax_equal") is not True
            or not 0<=parity["maximum_absolute_error"]<=1e-4):raise ValueError("Bound endpoint parity is incomplete")
    original.original._check_file(output,matches[0]["completion"]["checkpoint"])
    confirmed=set()
    for entry in manifest["episodes"]:
        context=entry["context"];op=account["operations"].get(context["operation_id"])
        if (not op or op["status"]!="acknowledged" or op["request"]["branch"]!=arm
                or op["request"]["kind"]!="evaluation" or op["request"]["steps"]!=context["horizon"]):
            raise ValueError("Validation episode lacks its external reservation")
        rowpath=original.original._check_file(folder,entry["row"])
        row=_json(rowpath);completion=op["completion"]
        if (completion["checkpoint"]["path"]!=str(rowpath.relative_to(output))
                or completion["checkpoint"]["sha256"]!=entry["row"]["sha256"]
                or completion["actual_steps"]!=row["steps"]):raise ValueError("Validation row was not acknowledged")
        confirmed.add(context["operation_id"])
    baseline=_json(output/"baselines.json")
    maximum=3*len(scenes["splits"]["validation"])*scenes["configuration"]["horizon"]
    return evaluation.read_completed(actor,scenes["splits"]["validation"],protocol,folder,maximum,
        expected_bindings=binding,reference_report=baseline["reference"],random_report=baseline["random"],
        confirmed_operation_ids=confirmed,expected_report_sha256=file_hash(folder/"report.json"),allow_test_fixture=fixture,
        config=collaborative_study_config(horizon=scenes["configuration"]["horizon"]))


def _ready(report): return report["capability"]["eligible"] is True or report["warmup_capability"]["eligible"] is True


def _freeze(output,prepared,ledger,report):
    arm=prepared["shutdown_arm"];step=report["additional_joint_steps"]
    if not _ready(report) or ledger.head("ppo",arm)["step"]!=step:raise ValueError("Cannot freeze an unqualified or superseded boundary")
    folder=output/"branches"/arm
    record={"version":VERSION,"cycle_id":prepared["cycle_id"],"shutdown_arm":arm,"selected_step":step,
        "checkpoint":ledger.head("ppo",arm),"report_sha256":file_hash(folder/"validation"/f"step_{step:07d}"/"report.json"),
        "manifest_sha256":file_hash(folder/"validation"/f"step_{step:07d}"/"manifest.json"),
        "actor_sha256":file_hash(folder/"actors"/f"actor_{step:07d}.npz"),
        "protocol_sha256":prepared["protocol_sha256"],"capability":report["capability"],"warmup_capability":report["warmup_capability"],
        "status":"capability_ready" if report["capability"]["eligible"] else "warmup_ready",
        "used_final_test":False,"formal_ready":False,"release_ready":False}
    path=output/"selected_boundary.json"
    if path.exists():_same(_json(path),record,"Frozen boundary record changed")
    else:write_json(path,record)
    return record


def advance(output,*,until=None,allow_test_fixture=False):
    output=Path(output).expanduser().resolve();raw=_json(output/"prepared.json")
    until=raw["primary_endpoint"] if until is None else until
    if type(until) is not int or until not in (raw["probe_endpoint"],*raw["validation_endpoints"]):
        raise ValueError("Use only registered probe or validation boundaries")
    ledger=CycleBudget(output,raw["identity"])
    with ledger.lease():
        arm=raw["shutdown_arm"];account=ledger.read()
        if any(op["status"] in ("pending","abandoned") for op in account["operations"].values()):
            raise ValueError("Unconfirmed or abandoned operation requires diagnosis; never resample")
        prepared,protocol,scenes,source=read_prepared(output,allow_test_fixture=allow_test_fixture)
        head=ledger.head("ppo",arm)
        reports={};frozen=None
        # A saved eligible validation is a permanent stop, even if interruption occurred before publishing its marker.
        for step in prepared["validation_endpoints"]:
            if step>head["step"]:break
            folder=output/"branches"/arm/"validation"/f"step_{step:07d}"
            if not (folder/"report.json").exists():
                if step<head["step"]:raise ValueError("Missing completed validation before further training")
                continue
            report=_report_record(output,arm,step,prepared,protocol,scenes,account,allow_test_fixture)
            reports[str(step)]=report
            if _ready(report):frozen=_freeze(output,prepared,ledger,report);break
        if (output/"selected_boundary.json").exists() and frozen is None:
            raise ValueError("Frozen selection has no qualifying original validation")
        if frozen is None:
            learner=_trainer(prepared,protocol,source)
            payload=decode(output/head["path"],head["sha256"]);_envelope(payload,learner,arm)
            if head["step"]:
                op=account["operations"].get(payload.get("operation_id"))
                if (not op or op["status"]!="acknowledged" or op["completion"]["checkpoint"]!=head
                        or payload.get("audit",{}).get("neural_overrides")!=0):raise ValueError("Checkpoint lacks its actual acknowledgment")
                for binding in payload["evidence"].values():original.original._check_file(output,binding)
            learner.load_state_dict(payload["trainer"])
            if learner.joint_steps!=head["step"] or learner.joint_steps>until:raise ValueError("Invalid requested stage boundary")
            for target in sorted(set((prepared["probe_endpoint"],*prepared["validation_endpoints"]))):
                if target>until or target<learner.joint_steps:continue
                train_to(output,learner,ledger,arm,target)
                if target in prepared["validation_endpoints"] and str(target) not in reports:
                    evaluate_boundary(output,learner,ledger,arm,scenes)
                    report=_report_record(output,arm,target,prepared,protocol,scenes,ledger.read(),allow_test_fixture)
                    reports[str(target)]=report
                    print(json.dumps({"event":"validation_complete","steps":target,"primary_value":report["primary_value"],
                        "capability":report["capability"],"warmup_capability":report["warmup_capability"]}),flush=True)
                    if _ready(report):frozen=_freeze(output,prepared,ledger,report);break
        actual=ledger.head("ppo",arm)["step"]
        status=frozen["status"] if frozen else ("fixed_endpoint_completed" if actual==prepared["primary_endpoint"] else "boundary_completed")
        result={"status":status,"cycle_id":prepared["cycle_id"],"shutdown_arm":arm,"branch":"own_credit","until":actual,
            "ledger":ledger.read(),"selected_boundary":frozen,"validation_reports":{k:{n:v for n,v in r.items() if n not in ("rows","artifacts")}
                for k,r in reports.items()},"formal_ready":False,"website_model_changed":False}
        write_json(output/f"completion_{actual:07d}.json",result,replace=True)
        write_json(output/"progress.json",result,replace=True)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument("--prepare",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--source-run",default=str(PRODUCTION_SOURCE));parser.add_argument("--shutdown-arm",choices=ARMS)
    parser.add_argument("--source-step",type=int,default=SOURCE_STEP);parser.add_argument("--ppo-cap",type=int,default=500000)
    parser.add_argument("--validation-interval",type=int,default=100000);parser.add_argument("--cycle-id")
    parser.add_argument("--device",choices=("cpu","mps"),default="mps");parser.add_argument("--until",type=int)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.prepare:
        if args.shutdown_arm is None:parser.error("Preparation requires an explicit shutdown arm")
        result=prepare(args.output,source_run=args.source_run,shutdown_arm=args.shutdown_arm,source_step=args.source_step,
            ppo_cap=args.ppo_cap,validation_interval=args.validation_interval,cycle_id=args.cycle_id,device=args.device)
    else:
        class Tee:
            def __init__(self,*streams):self.streams=streams
            def write(self,value):
                for stream in self.streams:stream.write(value);stream.flush()
                return len(value)
            def flush(self):
                for stream in self.streams:stream.flush()
        with (Path(args.output).expanduser().resolve()/"stdout.log").open("a") as log,contextlib.redirect_stdout(Tee(sys.stdout,log)):
            result=advance(args.output,until=args.until)
    print(json.dumps({k:v for k,v in result.items() if k not in ("ledger","validation_reports")}),flush=True)


if __name__=="__main__":main()
