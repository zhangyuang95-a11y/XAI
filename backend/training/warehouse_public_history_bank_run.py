"""Finite private197 question-bank preparation/generation/reverification.

This is a candidate-content workflow, never model or study authorization. No
participant-derived scene sampling, PT loading, training, retry or refund exists.
"""
from copy import deepcopy
from dataclasses import asdict
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile

from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.warehouse_public_history_runtime import PublicHistoryRuntime
from env.warehouse.domain import WarehouseConfig
from env.warehouse.rewards import RewardConfig
from ui.warehouse_public_history_bank import PublicHistoryQuestionBank, generate_bank, bank_sources

VERSION = "warehouse-observed197-bank-workflow.v1"
LEDGER_VERSION = "warehouse-observed197-bank-finite-ledger.v1"
INPUT_NAMES = ("actor.npz", "protocol.json", "selection.json", "pool_scenes.json", "exclusion_pools.json")
_SHA = re.compile(r"[a-f0-9]{64}\Z")


def put(path, value, *, replace=False):
    path = Path(path)
    raw = value if isinstance(value, bytes) else canonical(value).encode()
    if path.exists() and not replace: raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".writing_", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        if path.exists() and not replace: raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream: return stream.read()


def _json(path): return json.loads(_read(path))


def _binding(raw): return {"sha256": sha256(raw).hexdigest(), "size": len(raw)}


def _selection(value, runtime):
    required = {"selected_step", "actor_sha256", "protocol_sha256", "scenario_manifest_sha256",
                "selection_split", "final_test_used", "frozen", "test_fixture", "source_checkpoint_sha256"}
    metadata = runtime.actor.metadata
    if (not isinstance(value, dict) or not required <= set(value)
            or value["frozen"] is not True or value["final_test_used"] is not False
            or value["selection_split"] != "validation" or value["test_fixture"] is not runtime.test_fixture
            or type(value["selected_step"]) is not int or value["selected_step"] != metadata["joint_steps"]
            or value["actor_sha256"] != runtime.actor_sha256 or value["protocol_sha256"] != runtime.protocol_sha256
            or value["scenario_manifest_sha256"] != metadata["scenario_manifest_sha256"]
            or value["source_checkpoint_sha256"] != metadata["source_checkpoint_sha256"]):
        raise ValueError("An explicitly frozen same-Actor validation selection is required; it is not qualification")
    for optional in ("cycle_id", "branch", "feedback_branch"):
        if optional in value and value[optional] != metadata.get(optional):
            raise ValueError("Frozen selection source identity differs: " + optional)


def _sources(runtime):
    result = bank_sources(runtime)
    result[str(Path(__file__).relative_to(ROOT))] = file_hash(__file__)
    return result


def _configuration(raw):
    values = deepcopy(raw)
    if isinstance(values.get("reward"), dict): values["reward"] = RewardConfig(**values["reward"])
    return WarehouseConfig(**values)


def _runtime(directory, protocol, *, fixture, config):
    return PublicHistoryRuntime(directory / "actor.npz", protocol=protocol,
        expected_actor_sha256=file_hash(directory / "actor.npz"), expected_protocol_sha256=digest(protocol),
        allow_test_fixture=fixture, config=_configuration(config) if config else None)


def prepare(output, *, actor, protocol, frozen_selection, pool_scenes, exclusion_pools,
            trajectory_steps, generation_step_cap, verification_step_cap, minimum_frame=1,
            allow_test_fixture=False, config=None):
    """Freeze raw inputs and finite caps. No Actor forward, physics step or PT load."""
    if type(allow_test_fixture) is not bool: raise ValueError("Explicit fixture mode must be boolean")
    if config is not None and not allow_test_fixture: raise ValueError("Custom configuration is fixture-only")
    paths = dict(zip(INPUT_NAMES, map(Path, (actor, protocol, frozen_selection, pool_scenes, exclusion_pools))))
    output = Path(output).expanduser().absolute()
    if output.exists(): raise FileExistsError(output)
    if any(output == path.absolute() or output in path.absolute().parents for path in paths.values()):
        raise ValueError("Output cannot contain a supplied input")
    raws = {name: _read(path) for name, path in paths.items()}
    p, selection, pool, excluded = (json.loads(raws[name]) for name in INPUT_NAMES[1:])
    if (not isinstance(pool, list) or not 4 <= len(pool) <= 100 or not isinstance(excluded, dict)
            or not {"train", "validation", "extraction", "final_test", "play"} <= set(excluded)
            or any(not isinstance(rows, list) for rows in excluded.values())):
        raise ValueError("Explicit independent scenario/exclusion arrays are required")
    if (type(trajectory_steps) is not int or not 1 <= trajectory_steps <= 120
            or type(minimum_frame) is not int or not 0 <= minimum_frame < trajectory_steps):
        raise ValueError("Invalid bounded trajectory interval")
    maximum = len(pool) * (trajectory_steps + 3 * (trajectory_steps - minimum_frame))
    zero_queries = len(pool) * (2 * trajectory_steps - minimum_frame)
    for cap in (generation_step_cap, verification_step_cap):
        if type(cap) is not int or not 0 <= cap <= maximum: raise ValueError("Finite step cap exceeds the declared pool/trajectory upper bound")
    # Validate the actual NumPy identity against the exact source bytes before
    # creating output. The temporary directory is not a sampler or a publication.
    with tempfile.TemporaryDirectory(prefix="bank_prepare_") as temp:
        private = Path(temp).resolve()
        put(private / "actor.npz", raws["actor.npz"])
        runtime = _runtime(private, p, fixture=allow_test_fixture, config=asdict(config) if config else None)
        _selection(selection, runtime)
        if trajectory_steps > runtime.config.horizon: raise ValueError("Trajectory exceeds the runtime horizon")
        sources = _sources(runtime)
        identity = {"version": VERSION, "test_fixture": allow_test_fixture,
            "inputs": {name: _binding(raw) for name, raw in raws.items()},
            "actor_sha256": runtime.actor_sha256, "protocol_sha256": runtime.protocol_sha256,
            "runtime_signature": runtime.signature, "runtime_configuration": asdict(runtime.config),
            "sources": sources, "sources_sha256": digest(sources), "trajectory_steps": trajectory_steps,
            "minimum_frame": minimum_frame, "caps": {phase: {"environment_steps": cap,
                "zero_step_nn_queries": zero_queries, "nn_queries": cap + zero_queries}
                for phase, cap in (("generation", generation_step_cap), ("verification", verification_step_cap))}}
    for name, path in paths.items():
        if _read(path) != raws[name]: raise ValueError("Source bytes changed during preparation")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "inputs").mkdir(mode=0o700)
    for name, raw in raws.items(): put(output / "inputs" / name, raw)
    prepared = {**identity, "identity_sha256": digest(identity),
        "original_input_paths": {name: str(path.absolute()) for name, path in paths.items()},
        "status": "prepared_candidate", "formal_ready": False, "release_ready": False,
        "qualification_evaluated": False, "prepare_environment_steps": 0, "prepare_nn_queries": 0}
    put(output / "prepared.json", prepared)
    ledger = {"version": LEDGER_VERSION, "identity_sha256": prepared["identity_sha256"], "revision": 0,
        "automatic_resume": False, "reservations_refunded": False, "operations": {},
        "phases": {phase: {"status": "not_started", "caps": deepcopy(caps),
            "reserved": {key: 0 for key in caps}, "confirmed": {key: 0 for key in caps},
            "artifact": None, "error": None} for phase, caps in identity["caps"].items()}}
    put(output / "ledger.json", ledger)
    put(output / "run.lock", b"")
    return deepcopy(prepared)


def _load(output, allow_test_fixture):
    output = Path(output).expanduser().resolve()
    prepared = _json(output / "prepared.json")
    if (prepared.get("version") != VERSION or type(allow_test_fixture) is not bool
            or prepared.get("test_fixture") is not allow_test_fixture):
        raise ValueError("Wrong workflow/fixture identity")
    identity = {key: prepared[key] for key in ("version", "test_fixture", "inputs", "actor_sha256", "protocol_sha256",
        "runtime_signature", "runtime_configuration", "sources", "sources_sha256", "trajectory_steps", "minimum_frame", "caps")}
    if digest(identity) != prepared["identity_sha256"]: raise ValueError("Prepared identity changed")
    directory = output / "inputs"
    if set(prepared["inputs"]) != set(INPUT_NAMES): raise ValueError("Frozen input family differs")
    raw = {name: _read(directory / name) for name in INPUT_NAMES}
    if any(_binding(raw[name]) != prepared["inputs"][name] for name in INPUT_NAMES):
        raise ValueError("Frozen input bytes changed")
    p = json.loads(raw["protocol.json"])
    runtime = _runtime(directory, p, fixture=allow_test_fixture,
                       config=prepared["runtime_configuration"] if allow_test_fixture else None)
    _selection(json.loads(raw["selection.json"]), runtime)
    if (_sources(runtime) != prepared["sources"] or digest(_sources(runtime)) != prepared["sources_sha256"]
            or runtime.signature != prepared["runtime_signature"] or runtime.actor_sha256 != prepared["actor_sha256"]
            or runtime.protocol_sha256 != prepared["protocol_sha256"]):
        raise ValueError("Frozen runtime/execution source differs")
    return output, prepared, runtime, json.loads(raw["pool_scenes.json"]), json.loads(raw["exclusion_pools.json"])


class _Budget:
    def __init__(self, output, prepared, phase):
        self.output, self.prepared, self.phase = output, prepared, phase
        self.ledger = _json(output / "ledger.json")
        if (self.ledger.get("version") != LEDGER_VERSION or self.ledger.get("identity_sha256") != prepared["identity_sha256"]
                or self.ledger.get("automatic_resume") is not False or self.ledger.get("reservations_refunded") is not False):
            raise ValueError("Ledger identity or no-retry contract differs")
        if any(op["status"] != "confirmed" for op in self.ledger["operations"].values()):
            raise ValueError("Unconfirmed operation requires diagnosis; automatic replay/refund forbidden")
        for name, value in self.ledger["phases"].items():
            if value["caps"] != prepared["caps"][name]: raise ValueError("Budget caps changed")
            reserved = {key: 0 for key in value["caps"]}; confirmed = dict(reserved)
            for op in self.ledger["operations"].values():
                if op["phase"] != name: continue
                for key in reserved:
                    reserved[key] += op["reserved"][key]; confirmed[key] += op["confirmed"][key]
            if value["reserved"] != reserved or value["confirmed"] != confirmed:
                raise ValueError("Ledger totals differ from confirmed records")
        if self.ledger["phases"][phase]["status"] != "not_started":
            raise ValueError("This phase was already attempted; no automatic regeneration or re-verification")

    def _commit(self, candidate):
        candidate["revision"] = self.ledger["revision"] + 1
        put(self.output / "ledger.json", candidate, replace=True)
        self.ledger = candidate

    def start(self):
        candidate = deepcopy(self.ledger); candidate["phases"][self.phase]["status"] = "running"; self._commit(candidate)

    def before(self, context):
        expected_phase = "generation" if self.phase == "generation" else "load_reverification"
        kind, maximum = context.get("kind"), context.get("maximum_steps")
        expected = {"next_action": 0, "selfplay_decision": 0, "trajectory": 1, "wait_three": 3}
        if (context.get("phase") != expected_phase or kind not in expected or type(maximum) is not int
                or maximum != expected[kind] or context.get("operation_id") in self.ledger["operations"]):
            raise ValueError("Unregistered question-bank operation shape")
        reserved = {"environment_steps": maximum, "zero_step_nn_queries": int(maximum == 0),
                    "nn_queries": max(1, maximum)}
        candidate = deepcopy(self.ledger); phase = candidate["phases"][self.phase]
        if any(phase["reserved"][key] + amount > phase["caps"][key] for key, amount in reserved.items()):
            raise ValueError("Finite question-bank budget exhausted")
        for key, amount in reserved.items(): phase["reserved"][key] += amount
        candidate["operations"][context["operation_id"]] = {"phase": self.phase, "status": "reserved",
            "context": deepcopy(context), "context_sha256": digest(context), "reserved": reserved,
            "confirmed": {key: 0 for key in reserved}, "completion": None}
        self._commit(candidate)
        return {"execution_permitted": True}

    def after(self, result):
        opid = result["operation_id"]; op = self.ledger["operations"].get(opid)
        if (not op or op["status"] != "reserved" or any(result.get(key) != value for key, value in op["context"].items())
                or not isinstance(result.get("result_sha256"), str) or not _SHA.fullmatch(result["result_sha256"])):
            raise ValueError("Completion does not match its actual reserved operation")
        actual = result.get("actual_steps")
        if type(actual) is not int or not 0 <= actual <= op["reserved"]["environment_steps"]:
            raise ValueError("Confirmed steps exceed their reservation")
        confirmed = {"environment_steps": actual, "zero_step_nn_queries": op["reserved"]["zero_step_nn_queries"],
                     "nn_queries": actual + op["reserved"]["zero_step_nn_queries"]}
        candidate = deepcopy(self.ledger)
        candidate["operations"][opid].update(status="confirmed", confirmed=confirmed, completion=deepcopy(result))
        for key, amount in confirmed.items(): candidate["phases"][self.phase]["confirmed"][key] += amount
        self._commit(candidate)

    def finish(self, status, artifact=None, error=None):
        candidate = deepcopy(self.ledger)
        candidate["phases"][self.phase].update(status=status, artifact=artifact, error=error)
        self._commit(candidate)


def _work(output, phase, allow_test_fixture):
    output = Path(output).expanduser().resolve()
    with (output / "run.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output, prepared, runtime, pool, excluded = _load(output, allow_test_fixture)
        budget = _Budget(output, prepared, phase)
        if phase == "verification":
            generated = budget.ledger["phases"]["generation"]
            if generated["status"] != "completed" or not generated["artifact"]:
                raise ValueError("Durably generated candidate bank is required before replay")
            raw = _read(output / "bank.private.json")
            if _binding(raw) != generated["artifact"]: raise ValueError("Generated bank bytes changed")
        budget.start()
        try:
            if phase == "generation":
                bank = generate_bank(runtime, pool, excluded, trajectory_steps=prepared["trajectory_steps"],
                    minimum_frame=prepared["minimum_frame"], before_operation=budget.before, after_operation=budget.after)
                raw = canonical(bank).encode(); put(output / "bank.private.json", raw)
                report = {"status": "candidate_generated", "content_checks": bank["checks"], "item_count": len(bank["items"])}
                binding = _binding(raw)
            else:
                bank = PublicHistoryQuestionBank(output / "bank.private.json", runtime, excluded,
                    expected_bank_sha256=generated["artifact"]["sha256"], before_operation=budget.before,
                    after_operation=budget.after, allow_test_fixture=allow_test_fixture)
                report = {"status": "candidate_content_verified", "summary": bank.summary(), "content_checks": bank.checks,
                    "bank_signature": bank.signature, "replay_audit": bank.audit, "public_items": bank.public_items()}
                binding = generated["artifact"]
            runtime.verify_binding()
            if _sources(runtime) != prepared["sources"]: raise ValueError("Execution sources changed")
            report.update(version=VERSION, test_fixture=allow_test_fixture, formal_ready=False, release_ready=False,
                qualification_evaluated=False, actor_sha256=runtime.actor_sha256, protocol_sha256=runtime.protocol_sha256,
                selection_sha256=prepared["inputs"]["selection.json"]["sha256"], identity_sha256=prepared["identity_sha256"],
                bank_binding=binding, confirmed_work=deepcopy(budget.ledger["phases"][phase]["confirmed"]))
            put(output / f"{phase}_report.json", report)
            budget.finish("completed", binding)
            return report
        except BaseException as exc:
            # Sampled work is not retried, and the durable ledger is never rolled
            # back to the in-memory last success after an uncertain I/O failure.
            try:
                disk = _json(output / "ledger.json")
                if disk == budget.ledger:
                    budget.finish("failed", error=repr(exc))
            except BaseException:
                pass
            raise


def generate(output, *, allow_test_fixture=False): return _work(output, "generation", allow_test_fixture)


def verify(output, *, allow_test_fixture=False): return _work(output, "verification", allow_test_fixture)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "generate", "verify"): mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    for name in ("actor", "protocol", "frozen-selection", "pool-scenes", "exclusion-pools"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--trajectory-steps", type=int); parser.add_argument("--minimum-frame", type=int, default=1)
    parser.add_argument("--generation-step-cap", type=int); parser.add_argument("--verification-step-cap", type=int)
    parser.add_argument("--allow-test-fixture", action="store_true")
    parser.add_argument("--fixture-configuration", type=Path)
    args = parser.parse_args(argv)
    if args.prepare:
        required = ("actor", "protocol", "frozen_selection", "pool_scenes", "exclusion_pools", "trajectory_steps", "generation_step_cap", "verification_step_cap")
        if any(getattr(args, key) is None for key in required): parser.error("Prepare requires all frozen inputs and both explicit finite step caps")
        if args.fixture_configuration and not args.allow_test_fixture:
            parser.error("Custom configuration is allowed only for explicitly marked fixtures")
        result = prepare(args.output, **{key: getattr(args, key) for key in required},
            minimum_frame=args.minimum_frame, allow_test_fixture=args.allow_test_fixture,
            config=_configuration(_json(args.fixture_configuration)) if args.fixture_configuration else None)
    else:
        result = (generate if args.generate else verify)(args.output, allow_test_fixture=args.allow_test_fixture)
    print(canonical({key: value for key, value in result.items() if key not in ("sources", "public_items", "original_input_paths")}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
