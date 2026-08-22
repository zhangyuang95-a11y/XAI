from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import evaluate_trace_ablation as trace_cli

from backend.nlp.explanation_generator import ExecutionGroundedExplanationGenerator
from backend.nlp.schemas import EvidenceBundle, QueryPlan
from backend.nlp.tokenizer import CallableTransformerBackend
from backend.simulation.query_engine import (
    EXPLANATION_MODE_NO_TRACE,
    EXPLANATION_MODE_RCPD_TRACE,
)
from evaluate_trace_ablation import (
    _semantic_plan_diff,
    _validate_semantic_diff,
)
from evaluation.trace_ablation import (
    analyze_study_log,
    load_manifest,
    paired_summary,
)
from backend.artifacts import file_sha256


def test_manifest_requires_unique_heldout_cases_and_matching_artifacts(
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    record = {
        "case_id": "case_1",
        "episode_id": "episode_1",
        "frame": 3,
        "question": "Why did robot 1 wait?",
        "language": "en",
        "seed": 17,
        "split": "heldout",
        "checkpoint_sha256": "checkpoint",
        "program_sha256": "program",
    }
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    cases = load_manifest(
        manifest,
        checkpoint_sha256="checkpoint",
        program_sha256="program",
        available_frames={"episode_1": {3}},
    )
    assert cases[0].case_id == "case_1"

    manifest.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_manifest(
            manifest,
            checkpoint_sha256="checkpoint",
            program_sha256="program",
            available_frames={"episode_1": {3}},
        )

    invalid = {**record, "split": "training"}
    manifest.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split=heldout"):
        load_manifest(
            manifest,
            checkpoint_sha256="checkpoint",
            program_sha256="program",
            available_frames={"episode_1": {3}},
        )


def test_generate_parses_each_case_once_and_resumes_in_stable_order(
    tmp_path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    program = tmp_path / "program.json"
    checkpoint.write_bytes(b"checkpoint")
    program.write_text("{}", encoding="utf-8")
    checkpoint_hash = file_sha256(checkpoint)
    program_hash = file_sha256(program)
    manifest = tmp_path / "manifest.jsonl"
    records = [
        {
            "case_id": case_id,
            "episode_id": "episode",
            "frame": frame,
            "question": f"question {case_id}",
            "language": "en",
            "seed": 7,
            "split": "heldout",
            "checkpoint_sha256": checkpoint_hash,
            "program_sha256": program_hash,
        }
        for case_id, frame in (("case_b", 2), ("case_a", 1))
    ]
    manifest.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )
    class FakeStore:
        def episode_ids(self):
            return ("episode",)

        def frames(self, _episode):
            return (SimpleNamespace(frame=1), SimpleNamespace(frame=2))

        def decision_snapshot(self, episode, frame):
            return (episode, frame)

    class FakePlan:
        response_language = "en"

        def __init__(self, question):
            self.question = question

        def to_dict(self):
            return {"question": self.question}

    class FakePlanner:
        def __init__(self):
            self.calls = 0

        def parse(self, question, **_kwargs):
            self.calls += 1
            return FakePlan(question)

    class FakeAdapter:
        def observation_schema(self):
            return {}

        def action_schema(self):
            return ()

        def entity_schema(self):
            return {}

        def restore(self, _snapshot, _policy):
            return None

    class FakeAnswer:
        def __init__(self, condition):
            self.condition = condition

        def to_dict(self):
            return {
                "query_plan": {"requires_scene_edit": False},
                "evidence": {
                    "query_plan": {"intent": "factual"},
                    "direct_result": {"action": "WAIT"},
                    "program_trace": [],
                    "disagreement": {},
                },
                "explanation": "Displayed.",
                "raw_explanation": "Raw.",
                "claims": [],
                "verdicts": [],
                "raw_claims": [],
                "raw_verdicts": [],
                "explanation_mode": self.condition,
                "generation_grounding": {
                    "semantic_plan": {
                        "program_facts": [],
                        "mandatory_requirement_keys": [],
                    },
                    "covered_requirement_keys": [],
                },
                "trace_audit": {
                    "eligible": False,
                    "exposed": False,
                    "fallback_status": "all_claims_supported",
                },
            }

    class FakeEngine:
        def __init__(self):
            self.planner = FakePlanner()
            self.adapter = FakeAdapter()
            self.policy = object()
            self.executions = []

        def execute_plan(self, plan, snapshot, *, explanation_mode, **_kwargs):
            self.executions.append((plan, snapshot, explanation_mode))
            return FakeAnswer(explanation_mode)

    store = FakeStore()
    engine = FakeEngine()
    monkeypatch.setattr(trace_cli.TrajectoryStore, "load", lambda _path: store)
    monkeypatch.setattr(trace_cli, "_build_engine", lambda _args, _artifacts: engine)
    output = tmp_path / "results.jsonl"
    args = SimpleNamespace(
        trajectory=str(tmp_path / "trajectory.pkl.gz"),
        manifest=str(manifest),
        checkpoint=str(checkpoint),
        program=str(program),
        transformer_model="unused",
        device="cpu",
        local_files_only=True,
        output=str(output),
    )
    Path(args.trajectory).write_bytes(b"trajectory")

    trace_cli.generate_records(args)
    assert engine.planner.calls == 2
    assert len(engine.executions) == 4
    assert engine.executions[0][0] is engine.executions[1][0]
    assert engine.executions[2][0] is engine.executions[3][0]
    result_rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in result_rows] == ["case_a", "case_b"]
    trace_cli.generate_records(args)
    assert engine.planner.calls == 2
    assert len(engine.executions) == 4

def _answer(*, trace: bool, unsupported: bool, latency: float) -> dict:
    verdict = {
        "claim": {"claim_type": "action"},
        "status": "UNVERIFIABLE" if unsupported else "SUPPORTED",
    }
    return {
        "raw_verdicts": [verdict],
        "generation_grounding": {
            "semantic_plan": {"mandatory_requirement_keys": ["action.final"]},
            "covered_requirement_keys": ["action.final"],
        },
        "query_plan": {"requires_scene_edit": False},
        "raw_explanation": "raw",
        "explanation": "shown",
        "trace_audit": {
            "eligible": trace,
            "exposed": trace,
            "fallback_status": "all_claims_supported",
        },
        "latency_ms": latency,
    }


def test_paired_summary_uses_case_paired_differences() -> None:
    records = [
        {
            "case_id": "a",
            "conditions": {
                EXPLANATION_MODE_NO_TRACE: _answer(
                    trace=False, unsupported=True, latency=10.0
                ),
                EXPLANATION_MODE_RCPD_TRACE: _answer(
                    trace=True, unsupported=False, latency=12.0
                ),
            },
        },
        {
            "case_id": "b",
            "conditions": {
                EXPLANATION_MODE_NO_TRACE: _answer(
                    trace=False, unsupported=True, latency=20.0
                ),
                EXPLANATION_MODE_RCPD_TRACE: _answer(
                    trace=True, unsupported=False, latency=24.0
                ),
            },
        },
    ]
    summary = paired_summary(
        records,
        bootstrap_rounds=100,
        permutation_rounds=100,
        seed=4,
    )
    assert summary["trace_coverage"] == 1.0
    assert summary["trace_eligible_cases"]["unsupported_claim_rate"][
        "mean_difference"
    ] == -1.0
    assert summary["all_cases"]["latency_ms"]["mean_difference"] == 3.0


def test_semantic_plan_acceptance_allows_only_trace_derived_fields() -> None:
    common = {
        "action_facts": [{"action": "WAIT"}],
        "program_facts": [],
        "explanation_method": "neural_policy_and_joint_execution",
        "available_evidence_ids": ["action.final"],
        "mandatory_evidence_ids": ["action.final"],
        "mandatory_requirement_keys": ["action.final"],
    }
    trace = {
        **common,
        "program_facts": [{"feature": "self.battery_percent"}],
        "explanation_method": "rcpd_execution_trace",
        "available_evidence_ids": ["action.final", "program.0"],
        "mandatory_evidence_ids": ["action.final", "program.0"],
        "mandatory_requirement_keys": ["action.final", "program.0"],
    }
    diff = _semantic_plan_diff(common, trace)
    _validate_semantic_diff("eligible", diff, eligible=True)
    with pytest.raises(RuntimeError, match="Trace-ineligible"):
        _validate_semantic_diff("ineligible", diff, eligible=False)
    with pytest.raises(RuntimeError, match="non-trace"):
        _validate_semantic_diff(
            "bad",
            _semantic_plan_diff(common, {**trace, "action_facts": []}),
            eligible=True,
        )


def test_ineligible_trace_produces_identical_generation_and_fallback_inputs() -> None:
    prompts: list[str] = []

    def generate_json(prompt: str, schema_name: str) -> dict[str, object]:
        assert schema_name == "EvidenceBoundNaturalLanguageExplanation"
        prompts.append(prompt)
        plan = json.loads(prompt.split("Semantic plan:\n", 1)[1])
        return {
            "answer": 'Robot 1\'s final action was "wait".',
            "used_evidence_ids": list(plan["mandatory_evidence_ids"]),
            "covered_requirement_keys": list(
                plan["mandatory_requirement_keys"]
            ),
        }

    evidence = EvidenceBundle(
        query_plan=QueryPlan.from_dict(
            {
                "intent": "factual",
                "subjects": ["robot_1"],
                "requires_policy_query": True,
                "requires_program_trace": True,
                "target_variables": ["robot_1.observed_action"],
                "response_language": "en",
                "confidence": 1.0,
            },
            raw_text="What did robot 1 do?",
        ),
        direct_result={
            "target": "robot_1",
            "argmax_action": "WAIT",
            "executed_action": "WAIT",
            "action_descriptions": {
                "WAIT": {"zh": "等待", "en": "wait"},
            },
        },
        policy_results={
            "target": "robot_1",
            "argmax_action": "WAIT",
            "executed_action": "WAIT",
            "action_descriptions": {
                "WAIT": {"zh": "等待", "en": "wait"},
            },
        },
        program_trace=(),
        disagreement={},
    )
    generator = ExecutionGroundedExplanationGenerator(
        CallableTransformerBackend(generate_json)
    )
    no_trace = generator.generate(
        evidence,
        include_program_trace=False,
        language="en",
    )
    split = len(prompts)
    trace_condition = generator.generate(
        evidence,
        include_program_trace=True,
        language="en",
    )
    assert prompts[:split] == prompts[split:]
    assert no_trace == trace_condition
    assert generator.render_verified(
        evidence,
        language="en",
        include_program_trace=False,
    ) == generator.render_verified(
        evidence,
        language="en",
        include_program_trace=True,
    )
