"""Development-only preflight for the Warehouse sparse-program method.

This gate never reads the final program-audit or held-out seeds.  It checks
whether a method has enough development margin to justify a formal three-pair
run; the release gate remains the independent, weaker preregistered threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _metric_ok(value: dict, *, agreement: float) -> bool:
    return (
        isinstance(value, dict)
        and value.get("coverage", 0) >= 0.25
        and value.get("agreement", 0) >= agreement
        and value.get("kl") is not None
        and 0 <= value["kl"] <= 0.30
        and value.get("illegal_count") == 0
        and value.get("rows", 0) > 0
    )


def development_preflight(acceptance: dict) -> dict:
    if acceptance.get("pair_seeds") != [6099]:
        raise ValueError("preflight requires the separate 6099 development pilot")
    if acceptance.get("heldout_evaluated") or acceptance.get("final_program_audit"):
        raise ValueError("development preflight cannot use final evidence")
    selected = acceptance.get("selected_pair_seed")
    rows = [item for item in acceptance.get("development", [])
            if item.get("pair_seed") == selected]
    if len(rows) != 1:
        raise ValueError("development pilot must have one selected pair")
    row = rows[0]
    audit = row.get("preliminary_program_audit") or {}
    feedback = acceptance.get("selected_regularized_feedback_rows") or {}
    selected_role_hashes = (acceptance.get("role_sha256") or {}).get("regularized") or {}
    selected_program_hashes = ((acceptance.get("program_sha256") or {})
                               .get("regularized") or {})
    candidates = acceptance.get("candidate_training") or []
    candidate = candidates[0] if len(candidates) == 1 else {}
    variants = candidate.get("variants") or {}
    raw, regularized = (variants.get(name) or {} for name in
                        ("raw", "regularized"))
    paired_training = bool(
        candidate.get("pair_seed") == 6099
        and raw.get("joint_steps") == 200000
        and regularized.get("joint_steps") == 200000
        and raw.get("actor_decisions") == 400000
        and regularized.get("actor_decisions") == 400000
        and raw.get("scenario_schedule_sha256") ==
            regularized.get("scenario_schedule_sha256")
        and raw.get("initial_role_sha256") ==
            regularized.get("initial_role_sha256") ==
            candidate.get("initial_role_sha256")
        and raw.get("both_roles_changed") is True
        and regularized.get("both_roles_changed") is True
        and raw.get("post_policy_interventions") == 0
        and regularized.get("post_policy_interventions") == 0
        and raw.get("feedback_rows") == {"robot_1": 0, "robot_2": 0}
    )
    model = all(row.get(variant, {}).get("passed") is True
                for variant in ("raw", "regularized"))
    actual_gradient = set(feedback) == {"robot_1", "robot_2"} and all(
        type(feedback[role]) is int and feedback[role] > 0 for role in feedback)
    role_feedback = {role: ((audit.get(role) or {}).get("audit_source") ==
                            "preliminary_development_rollouts"
                            and (audit.get(role) or {}).get("development_seeds") ==
                            list(range(240024, 240048))
                            and (audit.get(role) or {}).get("sample_count", 0) >= 1000
                            and (audit.get(role) or {}).get("actor_role_sha256") ==
                            selected_role_hashes.get(role)
                            and (audit.get(role) or {}).get("program_sha256") ==
                            selected_program_hashes.get(role)
                            and (audit.get(role) or {}).get("max_features_per_node") == 3
                            and (audit.get(role) or {}).get("feedback_eligible") is True
                            and _metric_ok((audit.get(role) or {}).get("feedback"),
                                           agreement=0.90))
                     for role in ("robot_1", "robot_2")}
    ai_explanation = ((audit.get("robot_2") or {}).get("explanation_eligible") is True
                      and _metric_ok((audit.get("robot_2") or {}).get("explanation"),
                                     agreement=0.93))
    return {"pilot_pair_seed": selected, "paired_training": paired_training,
            "model_quality": model,
            "actual_regularized_gradient": actual_gradient,
            "role_feedback": role_feedback, "ai_explanation": ai_explanation,
            "passed": bool(paired_training and model and actual_gradient
                           and all(role_feedback.values())
                           and ai_explanation)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("acceptance", type=Path)
    args = parser.parse_args()
    print(json.dumps(development_preflight(json.loads(args.acceptance.read_text())),
                     sort_keys=True))


if __name__ == "__main__":
    main()
