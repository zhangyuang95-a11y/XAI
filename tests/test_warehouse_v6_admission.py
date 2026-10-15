"""Admission uses immutable raw evidence and never spends sealed seeds early."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from experiments.warehouse import admission as a
from experiments.warehouse import protocol as p


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return sha256(path.read_bytes()).hexdigest()


class FakeActor:
    def __init__(self, digest, roles, pair_seed, variant):
        self.artifact_sha256 = digest
        self.role_sha256 = roles
        self.pair_seed = pair_seed
        self.variant = variant


class FakeProgram:
    def __init__(self, role, digest):
        self.role, self.digest = role, digest
        self.report = {"split_episode_ids": {
            "fit": [f"fit-{i}" for i in range(50)],
            "calibration": [f"cal-{i}" for i in range(25)],
            "eligibility": [f"elig-{i}" for i in range(25)],
            "validation": [f"val-{i}" for i in range(24)],
        }}

    def sha256(self):
        return self.digest


class FakeBackend:
    source = "7" * 64

    def __init__(self, root):
        self.root = root
        self.audit_calls = []
        self.evaluate_calls = []
        self.fail_final = False

    def source_closure_sha256(self):
        return self.source

    def train_pair(self, spec):
        seed, budget = spec["pair_seed"], spec["joint_steps"]
        directory = (Path(spec["output_dir"]) / spec["candidate_id"] /
                     f"pair_{seed}_steps_{budget}")
        initial = {"robot_1": "1" * 64, "robot_2": "2" * 64}
        variants = {}
        for variant in p.VARIANTS:
            chosen = {"robot_1": ("3" if variant == "raw" else "5") * 64,
                      "robot_2": ("4" if variant == "raw" else "6") * 64}
            actor_path = directory / f"{variant}.actor.npz"
            actor_sha = _write(actor_path, {"seed": seed, "variant": variant})
            report_path = directory / f"{variant}.training.json"
            report_sha = _write(report_path, {"seed": seed, "variant": variant,
                                              "joint_steps": budget})
            programs = {}
            if variant == "regularized":
                for role in p.ROLES:
                    program_path = directory / f"{role}.program.json"
                    digest = _write(program_path, {"seed": seed, "role": role})
                    programs[role] = {"path": str(program_path), "sha256": digest,
                                      "actor_role_sha256": chosen[role]}
            variants[variant] = {
                "joint_steps": budget, "actor_decisions": budget * 2,
                "online_program_extraction_steps": list(
                    spec["online_program_extraction_steps"]),
                "online_program_split_block_cycle": p.FINAL_PROGRAM_SPLIT_COUNTS,
                "online_program_cohorts_disjoint": True,
                "post_policy_interventions": 0,
                "physical_no_legal_action_rows": 0,
                "scenario_schedule_sha256": "8" * 64,
                "source_closure_sha256": self.source,
                "initial_role_sha256": initial,
                "selected_role_sha256": chosen,
                "role_weight_change_norm": dict.fromkeys(p.ROLES, 1.0),
                "feedback_rows": dict.fromkeys(p.ROLES, 0 if variant == "raw" else 10),
                "feedback_gradient_norm": dict.fromkeys(
                    p.ROLES, 0.0 if variant == "raw" else .5),
                "actor_artifact": {"path": str(actor_path), "sha256": actor_sha,
                                   "role_sha256": chosen,
                                   "model_version": "warehouse-dual-actor.v6.1",
                                   "feature_schema_sha256": "9" * 64},
                "program_artifacts": programs,
                "training_report_path": str(report_path),
                "training_report_sha256": report_sha,
            }
        pair = {"candidate_id": spec["candidate_id"],
                "release_id": spec["release_id"],
                "final_program_fit_seed_schedule":
                spec["final_program_fit_seed_schedule"],
                "final_program_fit_seed_schedule_sha256":
                spec["final_program_fit_seed_schedule_sha256"],
                "pair_seed": seed,
                "joint_steps": budget,
                "online_program_extraction_steps": list(
                    spec["online_program_extraction_steps"]),
                "checkpoint_seeds": list(spec["checkpoint_seeds"]),
                "source_closure_sha256": self.source,
                "initial_role_sha256": initial, "variants": variants}
        _write(directory / "pair.training.json", pair)
        return pair

    def load_actor(self, artifact):
        payload = json.loads(Path(artifact["path"]).read_text())
        return FakeActor(artifact["sha256"], artifact["role_sha256"],
                         payload["seed"], payload["variant"])

    def load_program(self, artifact, actor_role_sha256):
        payload = json.loads(Path(artifact["path"]).read_text())
        assert artifact["actor_role_sha256"] == actor_role_sha256
        return FakeProgram(payload["role"], artifact["sha256"])

    def evaluate(self, actor, seeds, *, action_seed_offset=0, trace=False):
        self.evaluate_calls.append((actor.pair_seed, actor.variant,
                                    list(seeds), action_seed_offset, trace))
        deliveries = 7 if actor.pair_seed == 6102 else 6
        episodes = []
        for seed in seeds:
            row = {"seed": seed, "deliveries": deliveries, "collisions": 1,
                   "shutdown_episode": False}
            if trace:
                row["quality_trace"] = {
                    "schema": "warehouse-quality-trace.v1",
                    "actor_sha256": actor.artifact_sha256,
                    "actor_role_sha256": actor.role_sha256,
                    "scenario_seed": seed,
                    "action_seed": seed + action_seed_offset,
                    "episode_id": 0,
                    "frame_count": 1,
                    "action_encoding": "base64-u8-a1-times-5-plus-a2",
                    "actions_b64": "AA==",
                    "initial_pre_state_sha256": "0" * 64,
                    "final_post_state_sha256": "0" * 64,
                    "chain_sha256": "0" * 64,
                }
            episodes.append(row)
        return {"episodes": episodes, "summary": a._summary(episodes)}

    def audit_programs(self, actor, programs, seeds, *, trace_dir=None):
        self.audit_calls.append(list(seeds))
        if self.fail_final and tuple(seeds) == p.final_audit_seeds():
            claim = (self.root / "sealed_claims" / p.INITIAL_RELEASE_ID /
                     "final_program_audit.claim.json")
            assert claim.is_file(), "claim must exist before final rollout"
            raise RuntimeError("synthetic independent audit failure")
        metrics = {"coverage": .50, "actionable_coverage": .40,
                   "agreement": .98, "kl": .04,
                   "illegal_count": 0, "rows": 1000, "actionable_rows": 800,
                   "sample_count": 2000,
                   "certified_rows": 900}
        directory = Path(trace_dir) if trace_dir is not None else self.root / "fake_traces"
        directory.mkdir(parents=True, exist_ok=True)
        traces = {}
        for role in p.ROLES:
            target = directory / f"{role}.jsonl"
            target.write_bytes(b"{}\n" * 2000)
            traces[role] = (str(target), sha256(target.read_bytes()).hexdigest())
        return {role: {"role": role, "actor_role_sha256": actor.role_sha256[role],
                       "program_sha256": programs[role].sha256(),
                       "trace_path": traces[role][0],
                       "trace_sha256": traces[role][1],
                       "trace_row_count": 2000,
                       "episode_ids": sorted(map(str, seeds)),
                       "episode_count": len(seeds), "sample_count": 2000,
                       "max_depth": 4, "leaf_count": 16,
                       "max_features_per_node": 3,
                       "feedback": metrics, "explanation": metrics,
                       "feedback_eligible": True,
                       "explanation_eligible": True,
                       "passed": True} for role in p.ROLES}


@pytest.fixture
def ready(tmp_path, monkeypatch):
    # The gate still sees every registered stage; only pressure trajectory
    # count is reduced for this pure orchestration test.
    monkeypatch.setattr(p, "stress_schedule", lambda: ((1, 2600001, 123),
                                                        (2, 2600002, 456)))
    backend = FakeBackend(tmp_path)
    admission = a.Admission(tmp_path, "warehouse-v6-test", backend=backend,
                            claim_root=tmp_path / "sealed_claims")
    assert admission.develop()["locked_budget"] == 200_000
    selection = admission.formal_train()
    assert selection["selected_pair_seed"] == 6102
    assert backend.audit_calls == [list(p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS)] * 3
    assert admission.preaudit()["passed"] is True
    return admission, backend


def test_formal_pair_is_chosen_before_program_preaudit_and_from_quality_only(ready):
    admission, backend = ready
    selection = admission._required("formal.selection.json")
    assert selection["selected_pair_seed"] == 6102
    assert selection["selection_seeds"] == list(p.FORMAL_SELECTION_SEEDS)
    assert backend.audit_calls[-1] == list(p.FORMAL_PROGRAM_PREAUDIT_SEEDS)
    assert not admission.claim_root.exists()


def test_budget_lock_rejects_online_program_schedule_change(ready):
    admission, _ = ready
    path = admission._path("budget.lock.json")
    lock = json.loads(path.read_text())
    assert lock["online_program_extraction_steps"] == [150_000, 168_000, 186_000]
    lock["online_program_extraction_steps"][1] += 120
    path.write_text(json.dumps(lock))
    with pytest.raises(RuntimeError, match="seed lock changed"):
        admission.develop()


def test_final_audit_claim_is_one_use_and_failure_closes_candidate(ready):
    admission, backend = ready
    backend.fail_final = True
    with pytest.raises(RuntimeError, match="synthetic independent audit failure"):
        admission.final_audit()
    claim = admission._claim_path("final_program_audit")
    assert claim.is_file()
    assert admission._path("candidate.closed.json").is_file()
    assert not (claim.parent / "heldout.claim.json").exists()
    with pytest.raises(RuntimeError, match="closed"):
        admission.final_audit()


def test_heldout_is_100_per_task_and_acceptance_rechecks_raw_rows(ready):
    admission, backend = ready
    assert admission.final_audit()["passed"] is True
    heldout = admission.heldout()
    assert heldout["passed"] is True
    for variant in p.VARIANTS:
        for task in p.TASKS:
            episodes = heldout["outcomes"][variant]["tasks"][str(task)]["episodes"]
            assert len(episodes) == 100
            assert [row["seed"] for row in episodes] == list(
                p.heldout_schedule()[task])
    accepted = admission.acceptance()
    assert accepted["passed"] is True
    assert accepted["selected_pair_seed"] == 6102
    assert set(accepted["phase_receipt_sha256"]) == {
        "budget.lock.json", "formal.selection.json", "program.preaudit.json",
        "final_program_audit.json", "heldout.json",
        *(f"formal.training.{seed}.json" for seed in p.FORMAL_PAIR_SEEDS)}
    changed = json.loads(json.dumps(heldout["outcomes"]))
    changed["raw"]["tasks"]["1"]["episodes"][0]["deliveries"] += 1
    assert a._verify_heldout(changed, admission.release_id) is False
    changed = json.loads(json.dumps(heldout["outcomes"]))
    changed["raw"]["summary"]["mean_deliveries"] = 999
    assert a._verify_heldout(changed, admission.release_id) is False


def test_nonfinite_or_missing_evaluator_summary_is_rejected():
    episodes = [{"seed": 1, "deliveries": 6, "collisions": 1,
                 "shutdown_episode": False}]
    for summary in ({}, {"mean_deliveries": float("nan"),
                         "mean_collisions": 1, "shutdown_episode_rate": 0}):
        with pytest.raises((RuntimeError, ValueError)):
            a._validated_episodes({"episodes": episodes, "summary": summary}, [1])


def test_formal_audit_requires_ai_explanation_and_both_feedback_roles(ready):
    admission, backend = ready
    _, pair = admission._selected_pair()
    actor = admission._load_variant(pair, "regularized")
    programs = admission._load_programs(pair, actor)
    audit = backend.audit_programs(actor, programs, p.FORMAL_PROGRAM_PREAUDIT_SEEDS)
    audit["robot_1"]["explanation"] = {
        "coverage": 0.0, "agreement": 0.0, "kl": 1.0,
        "illegal_count": 1, "rows": 0}
    audit["robot_1"]["explanation_eligible"] = False
    assert a._audit_valid(audit, actor, programs,
                          p.FORMAL_PROGRAM_PREAUDIT_SEEDS, development=False)
    assert not a._audit_valid(audit, actor, programs,
                              p.FORMAL_PROGRAM_PREAUDIT_SEEDS, development=True)
    audit["robot_1"]["feedback"]["illegal_count"] = 1
    assert not a._audit_valid(audit, actor, programs,
                              p.FORMAL_PROGRAM_PREAUDIT_SEEDS, development=False)


def test_audit_recomputes_metric_sample_count_coverage_and_certification(ready):
    admission, backend = ready
    _, pair = admission._selected_pair()
    actor = admission._load_variant(pair, "regularized")
    programs = admission._load_programs(pair, actor)
    seeds = p.FORMAL_PROGRAM_PREAUDIT_SEEDS
    original = backend.audit_programs(actor, programs, seeds)
    assert a._audit_valid(original, actor, programs, seeds, development=False)
    for role, purpose, field, value in (
            ("robot_1", "feedback", "sample_count", 1999),
            ("robot_1", "feedback", "coverage", .9),
            ("robot_1", "feedback", "actionable_coverage", .9),
            ("robot_1", "feedback", "actionable_rows", 1999),
            ("robot_2", "explanation", "rows", 999),
            ("robot_2", "explanation", "certified_rows", 1001)):
        changed = deepcopy(original)
        changed[role][purpose][field] = value
        assert not a._audit_valid(changed, actor, programs, seeds,
                                  development=False), (role, purpose, field)


def test_packaged_acceptance_recomputes_actionable_coverage():
    from experiments.warehouse.acceptance import _audit_metric

    metric = {"sample_count": 2000, "rows": 1000, "coverage": .5,
              "actionable_rows": 800, "actionable_coverage": .4,
              "agreement": .98, "kl": .04, "illegal_count": 0}
    assert _audit_metric(metric, 2000, purpose="feedback", development=True)
    for change in ({"actionable_rows": 900},
                   {"actionable_coverage": .49},
                   {"actionable_rows": True},
                   {"actionable_rows": 1001}):
        assert not _audit_metric({**metric, **change}, 2000,
                                 purpose="feedback", development=True)


def test_source_change_before_sealed_phase_rejects_without_claim(ready):
    admission, backend = ready
    backend.source = "a" * 64
    with pytest.raises(RuntimeError, match="source closure"):
        admission.final_audit()
    assert not admission.claim_root.exists()


def test_acceptance_rejects_tampered_heldout_episode(ready):
    admission, _ = ready
    admission.final_audit()
    admission.heldout()
    path = admission._path("heldout.json")
    receipt = json.loads(path.read_text())
    receipt["outcomes"]["raw"]["tasks"]["1"]["episodes"][0]["deliveries"] = 999
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="raw rows"):
        admission.acceptance()
    assert not admission._path("warehouse_v6_acceptance.json").exists()


def test_development_advances_to_next_budget_when_programs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "stress_schedule", lambda: ((1, 2600001, 123),))

    class MissingAt200k(FakeBackend):
        def train_pair(self, spec):
            report = super().train_pair(spec)
            if spec["joint_steps"] == 200_000:
                report["variants"]["regularized"]["program_artifacts"] = {}
                path = (Path(spec["output_dir"]) / spec["candidate_id"] /
                        f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}" /
                        "pair.training.json")
                _write(path, report)
            return report

    admission = a.Admission(tmp_path, "warehouse-v6-budget-test",
                            backend=MissingAt200k(tmp_path))
    assert admission.develop()["locked_budget"] == 300_000
    for seed in p.DEVELOPMENT_PAIR_SEEDS:
        failed = admission._required(f"development.200000.{seed}.json")
        assert failed["passed"] is False
        assert "lacks both role programs" in failed["program_error"]
    assert not admission._path(f"development.400000.{p.DEVELOPMENT_PAIR_SEEDS[0]}.json").exists()


def test_packaged_development_gate_rechecks_audit_identity_and_eligibility(ready):
    from experiments.warehouse.acceptance import _development_pass

    admission, _ = ready
    original = admission._required(f"development.200000.{p.DEVELOPMENT_PAIR_SEEDS[0]}.json")
    assert _development_pass(original)
    for mutation in ("episode_ids", "feedback_eligible", "explanation_eligible"):
        changed = deepcopy(original)
        audit = changed["program_audit"]["robot_1"]
        if mutation == "episode_ids":
            audit[mutation] = ["unregistered-episode"] * len(audit[mutation])
        else:
            audit[mutation] = False
        changed["passed"] = True
        assert not _development_pass(changed), mutation


def test_next_release_gets_distinct_policy_ids_and_sealed_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "stress_schedule", lambda: ((1, 2600001, 123),))
    release_id = "policylens-three-domain-20261016.v6.1"
    admission = a.Admission(tmp_path, "warehouse-v6-next-release",
                            release_id=release_id, backend=FakeBackend(tmp_path),
                            claim_root=tmp_path / "sealed_claims")
    assert admission.policy_ids == {
        "raw": "warehouse-raw-v6.1", "regularized": "warehouse-regularized-v6.1"}
    assert admission.develop()["locked_budget"] == 200_000
    admission.formal_train()
    admission.preaudit()
    assert admission.final_audit()["passed"] is True
    assert admission.heldout()["passed"] is True
    assert admission.acceptance()["policy_id"] == admission.policy_ids
    claim = json.loads(admission._claim_path("final_program_audit").read_text())
    assert claim["seeds"] == list(p.final_audit_seeds(release_id))
    with pytest.raises(ValueError, match="canonical minor"):
        a.Admission(tmp_path, "alias", release_id=
                    "policylens-three-domain-20261016.v6.01")


def test_sealed_seed_blocks_reject_noncanonical_release_aliases():
    for release_id in (
            "policylens-three-domain-20261016.v6.00",
            "policylens-three-domain-20261016.v6.01",
            "policylens-three-domain-20261016.v6.1suffix"):
        with pytest.raises(ValueError, match="unregistered"):
            p.sealed_seed_blocks(release_id)
    assert p.final_audit_seeds("policylens-three-domain-20261016.v6.1") != \
        p.final_audit_seeds("policylens-three-domain-20261016.v6.0")


def _package_asset(repository: Path, release_id: str, source: Path) -> dict:
    digest = sha256(source.read_bytes()).hexdigest()
    relative = f"releases/{release_id}/assets/{digest}/{source.name}"
    target = repository / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return {"path": relative, "sha256": digest}


def _package_pair_assets(repository: Path, release_id: str,
                         pair: dict) -> dict:
    return {variant: {
        "actor": _package_asset(
            repository, release_id,
            Path(pair["variants"][variant]["actor_artifact"]["path"])),
        "training_report": _package_asset(
            repository, release_id,
            Path(pair["variants"][variant]["training_report_path"])),
        "programs": {role: _package_asset(
            repository, release_id,
            Path(item["path"])) for role, item in
            pair["variants"][variant]["program_artifacts"].items()},
    } for variant in p.VARIANTS}


def test_full_packaged_acceptance_chain_and_tampered_evidence(ready, tmp_path,
                                                            monkeypatch):
    from experiments.warehouse import acceptance as packaged
    from experiments.warehouse.acceptance import validate_acceptance

    # This orchestration fixture writes synthetic JSON instead of NN artifacts
    # and actual reset rows. Their arithmetic is covered by dedicated tests.
    monkeypatch.setattr(packaged, "_verify_program_trace", lambda **_: True)
    monkeypatch.setattr(packaged, "_paired_schedule_evidence", lambda _: True)
    monkeypatch.setattr(packaged, "_training_assets", lambda *_, **__: True)
    monkeypatch.setattr(packaged, "_verify_final_fit_pair", lambda **_: True)
    monkeypatch.setattr(packaged, "_final_fit_index_claim", lambda **_: True)
    monkeypatch.setattr(packaged, "_final_fit_replay_receipt", lambda **_: True)
    monkeypatch.setattr(packaged, "_quality_replay_receipt", lambda **_: True)

    admission, _ = ready
    admission.final_audit()
    admission.heldout()
    receipt = admission.acceptance()
    repository = tmp_path / "clean-repository"
    release_id = admission.release_id
    evidence = {
        "phase_receipts": {name: _package_asset(repository, release_id, Path(path))
                           for name, path in receipt["phase_receipt_path"].items()},
        "one_use_claims": {name: _package_asset(repository, release_id, Path(path))
                            for name, path in receipt["one_use_claim_path"].items()},
        "pair_reports": {str(row["pair_seed"]): _package_asset(
            repository, release_id, Path(row["pair_report_path"]))
            for row in receipt["candidate_training"]},
        "development_receipts": {budget: {
            seed: _package_asset(repository, release_id, Path(path))
            for seed, path in rows.items()}
            for budget, rows in receipt["development_receipt_path"].items()},
        "development_pair_reports": {budget: {seed: _package_asset(
            repository, release_id,
            admission.directory / f"pair_{seed}_steps_{budget}" /
            "pair.training.json") for seed in rows}
            for budget, rows in receipt["development_receipt_path"].items()},
        "training_assets": {str(row["pair_seed"]): _package_pair_assets(
            repository, release_id,
            json.loads(Path(row["pair_report_path"]).read_text()))
            for row in receipt["candidate_training"]},
        "development_training_assets": {budget: {seed: _package_pair_assets(
            repository, release_id,
            json.loads((admission.directory /
                        f"pair_{seed}_steps_{budget}" /
                        "pair.training.json").read_text()))
            for seed in rows}
            for budget, rows in receipt["development_receipt_path"].items()},
        "program_traces": {
            "development": {budget: {seed: {role: _package_asset(
                repository, release_id, Path(audit["trace_path"]))
                for role, audit in json.loads(Path(path).read_text())[
                    "program_audit"].items()}
                for seed, path in rows.items()}
                for budget, rows in receipt["development_receipt_path"].items()},
            "preaudit": {role: _package_asset(
                repository, release_id, Path(audit["trace_path"]))
                for role, audit in admission._required("program.preaudit.json")[
                    "audit"].items()},
            "final_audit": {role: _package_asset(
                repository, release_id, Path(audit["trace_path"]))
                for role, audit in admission._required("final_program_audit.json")[
                    "audit"].items()},
        },
        "final_fit_traces": {
            "formal": {str(seed): {} for seed in p.FORMAL_PAIR_SEEDS},
            "development": {budget: {seed: {} for seed in rows}
                            for budget, rows in receipt[
                                "development_receipt_path"].items()},
        },
        "final_fit_replays": {
            "formal": {str(seed): {} for seed in p.FORMAL_PAIR_SEEDS},
            "development": {budget: {seed: {} for seed in rows}
                            for budget, rows in receipt[
                                "development_receipt_path"].items()},
        },
        "quality_replays": {
            "formal": {str(seed): {variant: {} for variant in p.VARIANTS}
                       for seed in p.FORMAL_PAIR_SEEDS},
            "development": {budget: {seed: {
                variant: {} for variant in p.VARIANTS}
                if json.loads(Path(path).read_text()).get("stress") else {}
                for seed, path in rows.items()}
                for budget, rows in receipt[
                    "development_receipt_path"].items()},
            "heldout": {variant: {str(task): {} for task in p.TASKS}
                        for variant in p.VARIANTS},
        },
    }
    selected = next(row for row in receipt["candidate_training"]
                    if row["pair_seed"] == receipt["selected_pair_seed"])
    selected_pair = json.loads(Path(selected["pair_report_path"]).read_text())
    entries = {}
    for variant in p.VARIANTS:
        trained = selected_pair["variants"][variant]
        actor = _package_asset(repository, release_id,
                               Path(trained["actor_artifact"]["path"]))
        report = _package_asset(repository, release_id,
                                Path(trained["training_report_path"]))
        assert actor["sha256"] == receipt["artifact_sha256"][variant]
        assert report["sha256"] == receipt["training_report_sha256"][variant]
        entry = {"policy_id": receipt["policy_id"][variant],
                 "actor_path": actor["path"], "actor_sha256": actor["sha256"],
                 "training_report_path": report["path"],
                 "training_report_sha256": report["sha256"],
                 "role_sha256": {
                     "human": receipt["role_sha256"][variant]["robot_1"],
                     "ai": receipt["role_sha256"][variant]["robot_2"]},
                 "program_sha256": {}}
        if variant == "regularized":
            entry["program_sha256"] = receipt["program_sha256"][variant]
            entry["program_paths"] = [
                _package_asset(repository, release_id,
                               Path(trained["program_artifacts"][role]["path"]))["path"]
                for role in p.ROLES]
        else:
            entry["program_paths"] = []
        entries[variant] = entry
    acceptance_asset = _package_asset(
        repository, release_id,
        admission._path("warehouse_v6_acceptance.json"))
    for entry in entries.values():
        entry["acceptance_path"] = acceptance_asset["path"]
        entry["acceptance_sha256"] = acceptance_asset["sha256"]
    assert validate_acceptance(receipt, release_id=release_id, entries=entries,
                               evidence=evidence, repository_root=repository)
    heldout_asset = evidence["phase_receipts"]["heldout.json"]
    (repository / heldout_asset["path"]).write_text("{}")
    assert not validate_acceptance(receipt, release_id=release_id, entries=entries,
                                   evidence=evidence, repository_root=repository)
    # Restore heldout and show that the separately sealed claim is also bound.
    shutil.copyfile(admission._path("heldout.json"),
                    repository / heldout_asset["path"])
    claim_asset = evidence["one_use_claims"]["final_program_audit"]
    (repository / claim_asset["path"]).write_text("{}")
    assert not validate_acceptance(receipt, release_id=release_id, entries=entries,
                                   evidence=evidence, repository_root=repository)
