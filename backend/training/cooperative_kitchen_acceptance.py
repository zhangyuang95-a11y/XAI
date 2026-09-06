"""Reproducible release evidence. Missing cloud credentials produce a failed gate."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from backend.cooperative_kitchen.artifacts import file_hash, runtime_hash
from backend.cooperative_kitchen.explanations import ExplanationEngine, isolated_branch, snapshot_hash
from backend.cooperative_kitchen.llm import KitchenLLMClient
from backend.cooperative_kitchen.policy import NumpyKitchenPolicy
from env.cooperative_kitchen import CooperativeKitchen, program_decision


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"); temp.replace(path)


def numpy_parity(policy, checkpoint):
    import torch
    from backend.cooperative_kitchen.torch_policy import SharedActor
    actor = SharedActor(len(policy.feature_names))
    actor.load_state_dict(torch.load(checkpoint, weights_only=False, map_location="cpu")["actor"])
    observations = []
    for seed in range(99000, 99020):
        env = CooperativeKitchen(seed=seed, scenario_id="generated")
        for i in range(30):
            observations.extend(env.observations().values())
            env.step({k: program_decision(env, k)["action"] for k in ("human", "ai")})
            if env.state["done"]: break
    values = np.asarray(observations, np.float32)
    with torch.no_grad(): logits = actor(torch.from_numpy(values)).numpy()
    numpy_logits = policy.logits(values)
    error = float(np.max(np.abs(logits - numpy_logits)))
    agreement = float(np.mean(logits.argmax(-1) == numpy_logits.argmax(-1)))
    return {"schema": "kitchen_numpy_parity_v1", "actor_sha256": policy.artifact_sha256,
            "checkpoint_sha256": file_hash(checkpoint), "observations": len(values),
            "maximum_logit_error": error, "argmax_agreement": agreement,
            "passed": error <= 1e-5 and agreement == 1.0}


def question_bank_audit(policy, bank):
    records = []
    for item in bank["items"]:
        env = CooperativeKitchen(seed=item["source_seed"], scenario_id="generated")
        while env.state["turn"] < item["frame"] and not env.state["done"]:
            action = policy.act({"ai": env.observations()["ai"]})[0]["ai"]
            env.step({"human": program_decision(env, "human")["action"], "ai": action})
        same_state = env.public_view() == item["state"]
        if item["type"] == "prediction":
            expected = policy.act({"ai": env.observations()["ai"]})[0]["ai"]
        else:
            branch = isolated_branch(policy, env.snapshot(), ["WAIT"] * 3)
            expected = branch["frames"][-1]["ai_action"] if len(branch["frames"]) == 3 else None
        records.append({"id": item["id"], "type": item["type"], "frame_matches": same_state,
                        "answer_matches": expected == item["correct_answer"], "actor_matches": item["actor_sha256"] == policy.artifact_sha256})
    valid_counts = all(sum(r["type"] == kind for r in records) == 4 for kind in ("prediction", "counterfactual"))
    return {"schema": "kitchen_questionnaire_audit_v1", "actor_sha256": policy.artifact_sha256,
            "items": records, "passed": valid_counts and len(records) == 8 and all(r["frame_matches"] and r["answer_matches"] and r["actor_matches"] for r in records)}


QA_CASES = [
    ("why", "base_empty", "为什么你选择这个动作？", "Why do you choose this action?"),
    ("waiting", "base_congestion", "你在等什么？", "What are you waiting for?"),
    ("alternative", "base_inprogress", "为什么队友不向上走？", "Why does the teammate not move up?"),
    ("counterfactual", "base_inprogress", "如果我先向右，然后等待两步，会怎样？", "What if I move right, then wait for two steps?"),
    ("rules", "base_empty", "一锅汤要几份洋葱，煮多久？", "How many onions does one soup need, and how long does it cook?"),
    ("rules", "base_congestion", "共享工作台已经满了，还能放汤吗？", "Can soup be placed on a shared counter that is already full?"),
    ("failure", "base_empty", "为什么回合失败了？", "Why did the round fail?"),
    ("clarify", "base_empty", "预测接下来的二十步。", "Predict the next 20 steps."),
    ("clarify", "base_empty", "你是不是因为害怕我才左转？", "Did you turn left because you are afraid of me?"),
]


def qa_audit(policy, program, *, run_cloud=False):
    client = KitchenLLMClient(); engine = ExplanationEngine(policy, program, client=client)
    report = {"schema": "kitchen_qa_audit_v1", "mode": "real_remote" if run_cloud and client.configured else "not_run",
              "actor_sha256": policy.artifact_sha256, "program_sha256": file_hash(program), "runtime_sha256": runtime_hash(),
              "qa_configuration": client.config, "passed": False, "cases": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0},
              "blocker": None if run_cloud and client.configured else "Cloud API configuration and real bilingual acceptance required"}
    if not run_cloud or not client.configured: return report
    for expected, scenario, zh, en in QA_CASES:
        for language, question in (("zh", zh), ("en", en)):
            env = CooperativeKitchen(scenario_id=scenario)
            if expected == "failure":
                while not env.state["done"]: env.step({"human": "WAIT", "ai": "WAIT"})
            snapshot = env.snapshot(); before = snapshot_hash(snapshot); started = time.monotonic()
            answer = engine.generate(snapshot, question, kind="free", language=language)
            is_clarification = expected == "clarify" and answer["kind"] == "clarify"
            calls = answer["diagnostics"]["calls"]
            for call in calls:
                for key in report["usage"]: report["usage"][key] += int(call.get("usage", {}).get(key, 0))
            verified = (answer["kind"] == expected and answer["verified"] and snapshot_hash(snapshot) == before
                        and (answer["diagnostics"]["llm_success"] or is_clarification))
            report["cases"].append({"language": language, "intent": expected, "question": question, "scenario": scenario,
                                     "elapsed_seconds": time.monotonic() - started, "passed": verified, "answer": answer})
    report["passed"] = bool(report["cases"]) and all(row["passed"] for row in report["cases"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/cooperative_kitchen/v3-id-pilot")
    parser.add_argument("--run-cloud", action="store_true")
    args = parser.parse_args(); root = Path(args.output); reports = root / "acceptance"
    policy = NumpyKitchenPolicy(root / "selected/actor.npz")
    parity = numpy_parity(policy, root / "selected/checkpoint.pt"); write(reports / "numpy_parity.json", parity)
    bank = json.loads((root / "artifacts/questionnaire.json").read_text())
    questionnaire = question_bank_audit(policy, bank)
    questionnaire["questionnaire_sha256"] = file_hash(root / "artifacts/questionnaire.json")
    write(reports / "questionnaire_report.json", questionnaire)
    qa = qa_audit(policy, root / "artifacts/program_ai.json", run_cloud=args.run_cloud); write(reports / "qa_report.json", qa)
    print(json.dumps({"numpy_parity": parity["passed"], "questionnaire": questionnaire["passed"], "real_cloud_qa": qa["passed"], "qa_blocker": qa["blocker"]}))


if __name__ == "__main__": main()
