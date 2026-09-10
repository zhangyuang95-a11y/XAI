"""Real held-out explanation acceptance for the frozen r4 online runtime.

The audit evaluates robot_2 on the registered 100-scene explanation split with
three public-state partners.  It records ordinary Actor/program agreement and
one-step player-action interventions from isolated snapshots.  It also renders
the six participant shortcuts in both languages plus the bounded wait-three
counterfactual, checking that the main answer remains short, contains no audit
terminology, and leaves its source record unchanged.

This command never trains, fits, or changes an action.  It writes a failed
report when a numerical or language gate is missed; only the separate r4
production-admission builder may consume a passing report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_r4_production_admission import (
    EXPLANATION_VERSION, local_source_hashes,
)
from backend.warehouse_alignment_online_explanation import (
    OnlineAlignmentExplainer, _MAIN_ANSWER_FORBIDDEN,
)
from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action


VERSION = EXPLANATION_VERSION
PARTNERS = ("skilled", "assertive", "noisy")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
LANGUAGE_SCENES = 10
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSONL_BYTES = 128 * 1024 * 1024
AGENT_PHYSICS = ("agent_id", "position", "battery", "active", "carrying_task_id",
                 "deliveries_completed", "last_battery_delta", "steps_since_charging",
                 "charger_wait_streak")
TASK_PHYSICS = ("task_id", "pickup_position", "delivery_position", "status",
                "carrier_agent_id", "created_frame", "claimed_frame", "delivered_frame")
JOINT_PHYSICS = ("next_task_index", "total_deliveries", "terminated", "truncated",
                 "terminal_reason")
QUESTIONS = (
    ("reason", "executed", "机器人2刚才为什么这样行动？", "Why did Robot 2 choose that action?"),
    ("wait", "executed", "机器人2刚才为什么等待？", "Why did Robot 2 wait?"),
    ("collision", "executed", "我们刚才为什么发生碰撞？", "Why did we just collide?"),
    ("influence", "executed", "我的上一步动作影响了机器人2吗？", "Did my last action affect Robot 2?"),
    ("goal", "next", "机器人2当前在朝哪个任务前进？", "Which task is Robot 2 moving toward?"),
    ("energy", "next", "机器人2现在需要充电吗？", "Does Robot 2 need to charge now?"),
    ("counterfactual", "next", "如果我等待三步会怎样？", "What happens if I wait for three steps?"),
)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "split": "explanation_test", "scenes": 100,
        "partners": list(PARTNERS), "horizon": 120,
        "evaluated_role": "robot_2",
        "ordinary_fidelity_min": .90, "ordinary_nonwait_min": .85,
        "critical_fidelity_min": .85, "critical_nonwait_min": .85,
        "critical_minimum_scenes": 10,
        "effective_direction_min": .85, "effective_direction_minimum_scenes": 10,
        "intervention_anchor": "pre-action frame divisible by 10 with a public critical group",
        "intervention_population": "different physical projection and different next Actor action versus WAIT",
        "intervention_correct": "program agrees with Actor at both nonterminal endpoints",
        "language_scenes": LANGUAGE_SCENES,
        "language_matrix": "six shortcuts plus wait-three, Chinese and English",
        "main_answer_sentences": [1, 2],
        "main_answer_audit_terms_forbidden": True,
        "tree_controls_runtime": False, "test_fixture_allowed": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return local_source_hashes((Path(__file__),))


def _read_json(path: Path) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError("Explanation JSON is missing, linked, noncanonical, or oversized")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field in explanation evidence")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in explanation evidence")))
    if not isinstance(value, dict):
        raise ValueError("Explanation evidence must be a JSON object")
    return value


def _write_new(path: Path, raw: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Explanation evidence output path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_new(path, canonical(value) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new(path, "".join(canonical(row) + "\n" for row in rows))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > MAX_JSONL_BYTES):
        raise ValueError("Explanation JSONL is missing, linked, noncanonical, or oversized")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        rows.append(_read_json_line(line))
    return rows


def _read_json_line(raw: str) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field in explanation row")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in explanation row")))
    if not isinstance(value, dict):
        raise ValueError("Explanation row must be a JSON object")
    return value


def _regular_input(value: str | Path, label: str) -> Path:
    supplied = Path(value).expanduser().absolute()
    if (supplied.is_symlink() or supplied.resolve() != supplied
            or not supplied.is_file()):
        raise ValueError(label + " must be a canonical regular file")
    return supplied


def _physical(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    state = snapshot["state"]
    return {
        "agents": [{key: agent[key] for key in AGENT_PHYSICS} for agent in state["agents"]],
        "tasks": [{key: task[key] for key in TASK_PHYSICS} for task in state["tasks"]],
        "completed_tasks": [{key: task[key] for key in TASK_PHYSICS}
                            for task in state["completed_tasks"]],
        **{key: state[key] for key in JOINT_PHYSICS},
    }


def _program_action(explainer: OnlineAlignmentExplainer, env) -> str:
    observation = env.observations()["robot_2"].astype(np.float32)
    features = dict(zip(env.feature_names, map(float, observation)))
    return explainer.program.predict(features)


def _stat(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return {"rows": len(rows), "scenes": len({row["fingerprint"] for row in rows}),
            "fidelity": sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0}


def _sentence_count(text: str, language: str) -> int:
    marks = re.findall(r"[。！？]" if language == "zh" else r"[.!?](?:\s|$)", text)
    return len(marks) if marks else 1


def _validated_episode_count(base: Sequence[Mapping[str, Any]],
                             expected_holdout: set[str]) -> int:
    """Verify that the ordinary rows are complete, terminal trajectories.

    Counting a caller-provided ``episodes`` field would let a short or partial
    row file masquerade as the registered 100 x 3 matrix.  Episode identity
    and frame continuity are therefore derived from the immutable rows.
    """
    episode_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in base:
        if (not isinstance(row, dict) or row.get("partner") not in PARTNERS
                or row.get("fingerprint") not in expected_holdout
                or type(row.get("frame")) is not int or row["frame"] < 0
                or row.get("after_frame") != row["frame"] + 1
                or type(row.get("done")) is not bool
                or row.get("action") not in ACTIONS or row.get("tree_action") not in ACTIONS
                or row.get("correct") is not (row["action"] == row["tree_action"])
                or type(row.get("submitted_equal")) is not bool
                or type(row.get("decision_source_unchanged")) is not bool
                or row.get("policy_controller") != "frozen_actor"
                or not isinstance(row.get("groups"), list)
                or len(row["groups"]) != len(set(row["groups"]))
                or not set(row["groups"]).issubset(GROUPS)):
            raise ValueError("Malformed ordinary explanation evidence row")
        episode_rows.setdefault((row["fingerprint"], row["partner"]), []).append(row)
    expected_keys = {(fingerprint, partner) for fingerprint in expected_holdout
                     for partner in PARTNERS}
    if set(episode_rows) != expected_keys:
        raise ValueError("Explanation episode matrix is incomplete")
    for rows in episode_rows.values():
        ordered = sorted(rows, key=lambda row: row["frame"])
        if ([row["frame"] for row in ordered] != list(range(len(ordered)))
                or any(row["done"] for row in ordered[:-1])
                or not ordered[-1]["done"]):
            raise ValueError("Explanation episode is partial or has a frame gap")
    return len(episode_rows)


def _validate_pair_rows(pairs: Sequence[Mapping[str, Any]],
                        expected_holdout: set[str]) -> int:
    anchors: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    directions = set(ACTIONS[:-1])
    for row in pairs:
        if (not isinstance(row, dict) or row.get("partner") not in PARTNERS
                or row.get("fingerprint") not in expected_holdout
                or type(row.get("frame")) is not int or row["frame"] < 0
                or row["frame"] % 10 != 0 or row.get("player_action") not in directions
                or not isinstance(row.get("groups"), list) or not row["groups"]
                or len(row["groups"]) != len(set(row["groups"]))
                or not set(row["groups"]).issubset(GROUPS)
                or type(row.get("active")) is not bool
                or type(row.get("physical_effect")) is not bool
                or type(row.get("nn_changed")) is not bool
                or type(row.get("correct")) is not bool
                or type(row.get("wait_submitted_equal")) is not bool
                or type(row.get("changed_submitted_equal")) is not bool
                or type(row.get("submitted_equal")) is not bool
                or type(row.get("source_unchanged")) is not bool
                or row.get("policy_controller") != "frozen_actor"):
            raise ValueError("Malformed intervention explanation evidence row")
        active = (row.get("wait_next_action") in ACTIONS
                  and row.get("changed_next_action") in ACTIONS)
        physical_effect = bool(active and row.get("wait_physical") != row.get("changed_physical"))
        nn_changed = bool(active and row.get("wait_next_action") != row.get("changed_next_action"))
        correct = bool(active and row.get("wait_next_tree") == row.get("wait_next_action")
                       and row.get("changed_next_tree") == row.get("changed_next_action"))
        if (row["active"] is not active or row["physical_effect"] is not physical_effect
                or row["nn_changed"] is not nn_changed or row["correct"] is not correct
                or row["submitted_equal"] is not (
                    row["wait_submitted_equal"] and row["changed_submitted_equal"])):
            raise ValueError("Intervention derivation differs from recorded endpoints")
        anchors.setdefault((row["fingerprint"], row["partner"], row["frame"]), []).append(row)
    for rows in anchors.values():
        if (len(rows) != len(directions)
                or {row["player_action"] for row in rows} != directions
                or len({canonical({"physical": row["wait_physical"],
                                   "action": row["wait_next_action"],
                                   "tree": row["wait_next_tree"],
                                   "equal": row["wait_submitted_equal"]}) for row in rows}) != 1):
            raise ValueError("Intervention anchor does not contain the exact four directions")
    return len(anchors)


def _validate_language_rows(language: Sequence[Mapping[str, Any]],
                            expected_holdout: set[str]) -> None:
    allowed_intents = {intent for intent, *_ in QUESTIONS}
    for row in language:
        if (not isinstance(row, dict) or row.get("fingerprint") not in expected_holdout
                or row.get("intent") not in allowed_intents
                or row.get("language") not in ("zh", "en")
                or row.get("action") not in ACTIONS or row.get("tree_action") not in ACTIONS
                or not isinstance(row.get("answer"), str)
                or not isinstance(row.get("evidence_detail"), str)
                or type(row.get("source_unchanged")) is not bool
                or type(row.get("runtime_step_count")) is not int
                or row["runtime_step_count"] < 1):
            raise ValueError("Malformed language explanation evidence row")
        main = row["answer"]
        mismatch = row["action"] != row["tree_action"]
        mismatch_hidden = (not mismatch or
            ("决策树" not in main and "近似程序" not in main and
             "decision tree" not in main.casefold()
             and "approximate program" not in main.casefold()))
        derived = {
            "main_terms_clean": _MAIN_ANSWER_FORBIDDEN.search(main) is None,
            "sentence_count": _sentence_count(main, row["language"]),
            "sentence_count_ok": 1 <= _sentence_count(main, row["language"]) <= 2,
            "evidence_present": bool(row["evidence_detail"].strip()),
            "tree_mismatch_case": mismatch,
            "tree_mismatch_hidden": mismatch_hidden,
        }
        if any(row.get(key) != value for key, value in derived.items()):
            raise ValueError("Language explanation checks differ from raw answer text")


def _summarize(base: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
               language: Sequence[Mapping[str, Any]], expected_holdout: set[str]) -> dict[str, Any]:
    episodes = _validated_episode_count(base, expected_holdout)
    _validate_pair_rows(pairs, expected_holdout)
    _validate_language_rows(language, expected_holdout)
    nonwait = [row for row in base if row["action"] != "WAIT"]
    fidelity = {
        "overall": _stat(base, "correct"),
        "nonwait": _stat(nonwait, "correct"),
        "by_group": {group: _stat([row for row in base if group in row["groups"]], "correct")
                     for group in GROUPS},
        "by_group_nonwait": {group: _stat([row for row in nonwait if group in row["groups"]], "correct")
                             for group in GROUPS},
    }
    effective = [row for row in pairs if row["physical_effect"] and row["nn_changed"]]
    direction = {
        "eligible_pairs": len(effective), "all_pairs": len(pairs),
        "no_physical_effect_pairs": sum(not row["physical_effect"] for row in pairs),
        "unchanged_actor_pairs": sum(not row["nn_changed"] for row in pairs),
        "overall": _stat(effective, "correct"),
        "by_group": {group: _stat([row for row in effective if group in row["groups"]], "correct")
                     for group in GROUPS},
    }
    checks = {
        "complete_100_scene_three_partner_matrix": episodes == 300,
        "ordinary_fidelity": bool(base) and fidelity["overall"]["fidelity"] >= .90,
        "ordinary_nonwait_fidelity": bool(nonwait) and fidelity["nonwait"]["fidelity"] >= .85,
        "effective_direction_fidelity": bool(effective) and direction["overall"]["scenes"] >= 10
                                        and direction["overall"]["fidelity"] >= .85,
        "language_matrix": len(language) == LANGUAGE_SCENES * len(QUESTIONS) * 2
                           and len({row["fingerprint"] for row in language}) == LANGUAGE_SCENES
                           and all(Counter((row["intent"], row["language"])
                                           for row in language
                                           if row["fingerprint"] == fingerprint)
                                   == Counter({(intent, lang): 1
                                      for intent, *_ in QUESTIONS for lang in ("zh", "en")})
                                   for fingerprint in {row["fingerprint"] for row in language}),
        "main_answer_terms": bool(language) and all(row["main_terms_clean"] for row in language),
        "main_answer_length": bool(language) and all(row["sentence_count_ok"] for row in language),
        "language_source_unchanged": bool(language) and all(row["source_unchanged"] for row in language),
        "language_evidence_present": bool(language) and all(row["evidence_present"] for row in language),
        "tree_mismatch_boundary": all(row["tree_mismatch_hidden"] for row in language
                                      if row["tree_mismatch_case"]),
        "action_authority": bool(base) and all(row["submitted_equal"] for row in base),
        "decision_source_unchanged": bool(base) and all(row["decision_source_unchanged"] for row in base),
        "counterfactual_source_unchanged": bool(pairs) and all(row["source_unchanged"] for row in pairs),
    }
    for group in GROUPS:
        checks[f"ordinary_{group}"] = (fidelity["by_group"][group]["scenes"] >= 10
                                             and fidelity["by_group"][group]["fidelity"] >= .85)
        checks[f"ordinary_nonwait_{group}"] = (fidelity["by_group_nonwait"][group]["scenes"] >= 10
                                                     and fidelity["by_group_nonwait"][group]["fidelity"] >= .85)
        checks[f"direction_{group}"] = (direction["by_group"][group]["scenes"] >= 10
                                              and direction["by_group"][group]["fidelity"] >= .85)
    return {"fidelity": fidelity, "intervention_direction": direction,
            "checks": checks, "passed": all(checks.values()),
            "evaluated_role": "robot_2", "no_effect_pairs_counted_as_success": False}


def _language_rows(runtime: OnlineAlignmentRuntime, explainer: OnlineAlignmentExplainer,
                   transition: Mapping[str, Any], fingerprint: str) -> list[dict[str, Any]]:
    rows = []
    action = transition["policy_actions"]["robot_2"]
    tree_action = _program_action(explainer, runtime.from_snapshot(transition["before"]))
    for intent, focus, zh, en in QUESTIONS:
        for language, question in (("zh", zh), ("en", en)):
            record = deepcopy(transition)
            before = digest(record)
            step_count = 0
            original_step = runtime.step

            def counted_step(env, player_action):
                nonlocal step_count
                result = original_step(env, player_action)
                step_count += 1
                return result

            # ``answer`` verifies an executed frame with one isolated replay;
            # a counterfactual answer then advances its isolated branch.  Count
            # those real environment transitions without changing the answer
            # API or the live/source environment.
            runtime.step = counted_step
            try:
                answer = explainer.answer({"question": question, "focus": focus,
                    "language": language, "frame": transition["after"]["state"]["frame"]},
                    record, runtime)
            finally:
                runtime.step = original_step
            main = answer["answer"]
            source_unchanged = digest(record) == before
            tree_mismatch = action != tree_action
            mismatch_hidden = (not tree_mismatch or
                ("决策树" not in main and "近似程序" not in main and
                 "decision tree" not in main.casefold() and "approximate program" not in main.casefold()))
            rows.append({"fingerprint": fingerprint, "intent": intent, "language": language,
                "answer": main, "evidence_detail": answer["evidence_detail"],
                "action": action, "tree_action": tree_action,
                "main_terms_clean": _MAIN_ANSWER_FORBIDDEN.search(main) is None,
                "sentence_count": _sentence_count(main, language),
                "sentence_count_ok": 1 <= _sentence_count(main, language) <= 2,
                "evidence_present": bool(answer["evidence_detail"].strip()),
                "source_unchanged": source_unchanged,
                "runtime_step_count": step_count,
                "tree_mismatch_case": tree_mismatch,
                "tree_mismatch_hidden": mismatch_hidden})
    return rows


def _branch(runtime: OnlineAlignmentRuntime, explainer: OnlineAlignmentExplainer,
            snapshot: Mapping[str, Any], player_action: str) -> dict[str, Any]:
    env = runtime.from_snapshot(snapshot)
    transition = runtime.step(env, player_action)
    next_action = next_tree = None
    if not transition["done"]:
        next_action = runtime.decision(env)[0]["robot_2"]
        next_tree = _program_action(explainer, env)
    return {"physical": digest(_physical(transition["after"])),
            "next_action": next_action, "next_tree": next_tree,
            "submitted_equal": transition["submitted_actions"]["robot_2"]
                               == transition["policy_actions"]["robot_2"]
                               and transition["decision"]["post_policy_overrides"] == 0,
            "policy_controller": "frozen_actor"}


def audit(*, actor_path: str | Path, protocol_path: str | Path, program_path: str | Path,
          scenarios_path: str | Path, output: str | Path) -> dict[str, Any]:
    actor_path = _regular_input(actor_path, "Explanation Actor")
    protocol_path = _regular_input(protocol_path, "Explanation protocol")
    program_path = _regular_input(program_path, "Explanation program")
    scenarios_path = _regular_input(scenarios_path, "Explanation scenarios")
    supplied_output = Path(output).expanduser().absolute()
    if (supplied_output.exists() or supplied_output.is_symlink()
            or supplied_output.resolve() != supplied_output):
        raise FileExistsError(output)
    output = supplied_output
    protocol = _read_json(protocol_path)
    scenarios = _read_json(scenarios_path)
    runtime = OnlineAlignmentRuntime(actor_path, protocol=protocol,
        expected_actor_sha256=file_hash(actor_path), expected_protocol_sha256=digest(protocol))
    explainer = OnlineAlignmentExplainer(program_path,
        expected_program_sha256=file_hash(program_path), runtime=runtime)
    split = scenarios.get("splits", {}).get("explanation_test")
    if (scenarios.get("version") != "warehouse-native-physical-splits-v1"
            or not isinstance(split, list) or len(split) != 100
            or runtime.config.horizon != 120
            or digest(scenarios) != runtime.actor.metadata.get("scenario_manifest_sha256")):
        raise ValueError("Exact registered 100-scene explanation split required")
    heldout = {scene.get("fingerprint") for scene in split}
    other = {scene.get("fingerprint") for name, rows in scenarios["splits"].items()
             if name != "explanation_test" for scene in rows}
    if len(heldout) != 100 or heldout & other:
        raise ValueError("Explanation trajectories overlap another registered split")
    initial = {"actor": file_hash(actor_path), "protocol": file_hash(protocol_path),
               "program": file_hash(program_path), "scenarios": file_hash(scenarios_path),
               "sources": producer_sources()}
    bindings = {"actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256, "runtime_signature": runtime.signature,
        "program_sha256": explainer.program_sha256, "explainer_signature": explainer.signature,
        "scenario_manifest_sha256": digest(scenarios), "test_fixture": False,
        "contract_sha256": digest(contract()), "producer_sources_sha256": digest(initial["sources"]),
        "holdout_fingerprints_sha256": digest(sorted(heldout))}
    output.mkdir(parents=True, mode=0o700)
    output.chmod(0o700)
    inputs = {"version": VERSION, "bindings": bindings, "contract": contract(),
              "sources": initial["sources"], "input_file_sha256": {key: value for key, value in initial.items() if key != "sources"},
              "holdout_fingerprints": sorted(heldout), "formal_ready": False}
    _write_json(output / "inputs.json", inputs)
    base_rows, pair_rows, language_rows = [], [], []
    episodes = base_steps = counterfactual_steps = anchors = 0
    for partner_index, partner in enumerate(PARTNERS):
        for scene_index, scene in enumerate(split):
            env = runtime.environment(scene)
            rng = np.random.default_rng(17000 + partner_index * 1000 + scene_index)
            first_transition = None
            while not env.done:
                source = env.snapshot(); source_sha = digest(source)
                actions, decision = runtime.decision(env)
                tree_action = _program_action(explainer, env)
                unchanged = digest(env.snapshot()) == source_sha
                groups = list(critical_groups(env, "robot_2"))
                if env.state.frame % 10 == 0 and groups:
                    branches = {action: _branch(runtime, explainer, source, action) for action in ACTIONS}
                    counterfactual_steps += len(branches); anchors += 1
                    wait = branches["WAIT"]
                    for action in ACTIONS[:-1]:
                        changed = branches[action]
                        active = wait["next_action"] is not None and changed["next_action"] is not None
                        pair_rows.append({"fingerprint": scene["fingerprint"], "partner": partner,
                            "frame": env.state.frame, "groups": groups, "player_action": action,
                            "active": active,
                            "wait_physical": wait["physical"],
                            "changed_physical": changed["physical"],
                            "wait_next_action": wait["next_action"],
                            "changed_next_action": changed["next_action"],
                            "wait_next_tree": wait["next_tree"],
                            "changed_next_tree": changed["next_tree"],
                            "physical_effect": active and wait["physical"] != changed["physical"],
                            "nn_changed": active and wait["next_action"] != changed["next_action"],
                            "correct": active and wait["next_tree"] == wait["next_action"]
                                       and changed["next_tree"] == changed["next_action"],
                            "wait_submitted_equal": wait["submitted_equal"],
                            "changed_submitted_equal": changed["submitted_equal"],
                            "submitted_equal": wait["submitted_equal"] and changed["submitted_equal"],
                            "source_unchanged": digest(env.snapshot()) == source_sha,
                            "policy_controller": "frozen_actor"})
                player = partner_action(env, "robot_1", partner, rng)
                if digest(env.snapshot()) != source_sha:
                    raise RuntimeError("Explanation partner mutated the source state")
                transition = runtime.step(env, player)
                submitted_equal = (transition["submitted_actions"]["robot_2"] == actions["robot_2"]
                    == transition["policy_actions"]["robot_2"]
                    and decision["post_policy_overrides"] == transition["decision"]["post_policy_overrides"] == 0)
                base_rows.append({"fingerprint": scene["fingerprint"], "partner": partner,
                    "frame": source["state"]["frame"], "groups": groups,
                    "after_frame": transition["after"]["state"]["frame"],
                    "done": transition["done"],
                    "action": actions["robot_2"], "tree_action": tree_action,
                    "correct": tree_action == actions["robot_2"],
                    "submitted_equal": submitted_equal, "decision_source_unchanged": unchanged,
                    "policy_controller": "frozen_actor"})
                base_steps += 1
                if first_transition is None:
                    first_transition = transition
            episodes += 1
            if partner_index == 0 and scene_index < LANGUAGE_SCENES:
                if first_transition is None:
                    raise RuntimeError("Language audit scene produced no executed transition")
                language_rows.extend(_language_rows(runtime, explainer, first_transition,
                                                     scene["fingerprint"]))
    statistics = _summarize(base_rows, pair_rows, language_rows, heldout)
    if episodes != 300:
        raise RuntimeError("Explanation episode execution count differs")
    _write_jsonl(output / "ordinary_rows.jsonl", base_rows)
    _write_jsonl(output / "intervention_rows.jsonl", pair_rows)
    _write_jsonl(output / "language_rows.jsonl", language_rows)
    evidence = {name: file_hash(output / name) for name in
        ("inputs.json", "ordinary_rows.jsonl", "intervention_rows.jsonl", "language_rows.jsonl")}
    language_runtime_steps = sum(row["runtime_step_count"] for row in language_rows)
    zero_overrides = (all(row["submitted_equal"] for row in base_rows)
                      and all(row["submitted_equal"] for row in pair_rows))
    report = {"version": VERSION, "status": "passed" if statistics["passed"] else "failed",
        "test_fixture": False, "formal_ready": False, "bindings": bindings,
        "statistics": statistics, "base_rows": len(base_rows), "branch_anchor_count": anchors,
        "episodes": episodes, "zero_nn_overrides": zero_overrides,
        "source_state_unchanged": all(row["decision_source_unchanged"] for row in base_rows),
        "counterfactual_isolated": all(row["source_unchanged"] for row in pair_rows)
            and all(row["source_unchanged"] for row in language_rows),
        "program_never_controls_action": zero_overrides and all(
            row["policy_controller"] == "frozen_actor" for row in (*base_rows, *pair_rows)),
        "execution": {"accounting_complete": True, "pending_operation": None,
            "counts": {"base_steps": base_steps, "counterfactual_steps": counterfactual_steps,
                "language_runtime_steps": language_runtime_steps,
                "acknowledged_steps": base_steps + counterfactual_steps + language_runtime_steps,
                "neural_updates": 0, "tree_fits": 0, "torch_loads": 0}},
        "evidence_artifacts": evidence}
    if (producer_sources() != initial["sources"] or file_hash(actor_path) != initial["actor"]
            or file_hash(protocol_path) != initial["protocol"] or file_hash(program_path) != initial["program"]
            or file_hash(scenarios_path) != initial["scenarios"]):
        raise RuntimeError("Frozen explanation input changed during audit")
    _write_json(output / "report.json", report)
    return report


def read_saved_report(output: str | Path, *, expected_report_sha256: str,
                      expected_bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute metrics from immutable row files without Actor/environment calls."""
    output = Path(output).expanduser().absolute()
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output):
        raise ValueError("Explanation evidence directory is missing, linked, or noncanonical")
    report_path = output / "report.json"
    if (report_path.is_symlink() or not report_path.is_file()
            or file_hash(report_path) != expected_report_sha256):
        raise ValueError("Explanation report hash differs")
    report = _read_json(report_path)
    if report.get("bindings") != expected_bindings:
        raise ValueError("Explanation report bindings differ")
    artifacts = report.get("evidence_artifacts")
    expected_artifacts = {"inputs.json", "ordinary_rows.jsonl",
                          "intervention_rows.jsonl", "language_rows.jsonl"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise ValueError("Explanation evidence artifact set differs")
    for name, expected in artifacts.items():
        path = output / name
        if path.parent != output or not path.is_file() or path.is_symlink() or file_hash(path) != expected:
            raise ValueError("Explanation evidence artifact changed")
    ordinary = _read_jsonl(output / "ordinary_rows.jsonl")
    interventions = _read_jsonl(output / "intervention_rows.jsonl")
    language = _read_jsonl(output / "language_rows.jsonl")
    inputs = _read_json(output / "inputs.json")
    if (not isinstance(inputs, dict) or inputs.get("version") != VERSION
            or inputs.get("bindings") != expected_bindings
            or inputs.get("contract") != contract()
            or inputs.get("sources") != producer_sources()
            or digest(inputs.get("contract")) != expected_bindings.get("contract_sha256")
            or digest(inputs.get("sources")) != expected_bindings.get("producer_sources_sha256")
            or not isinstance(inputs.get("holdout_fingerprints"), list)
            or len(inputs["holdout_fingerprints"]) != 100
            or len(set(inputs["holdout_fingerprints"])) != 100
            or digest(sorted(inputs["holdout_fingerprints"]))
               != expected_bindings.get("holdout_fingerprints_sha256")):
        raise ValueError("Explanation audit inputs differ")
    expected_holdout = set(inputs["holdout_fingerprints"])
    statistics = _summarize(ordinary, interventions, language, expected_holdout)
    if digest(statistics) != digest(report["statistics"]):
        raise ValueError("Explanation statistics differ from immutable rows")
    anchor_count = len({(row["fingerprint"], row["partner"], row["frame"])
                        for row in interventions})
    expected_execution = {"base_steps": len(ordinary),
        "counterfactual_steps": anchor_count * len(ACTIONS),
        "language_runtime_steps": sum(row["runtime_step_count"] for row in language),
        "neural_updates": 0, "tree_fits": 0, "torch_loads": 0}
    expected_execution["acknowledged_steps"] = (
        expected_execution["base_steps"] + expected_execution["counterfactual_steps"]
        + expected_execution["language_runtime_steps"])
    zero_overrides = (all(row["submitted_equal"] for row in ordinary)
                      and all(row["submitted_equal"] for row in interventions))
    controller = zero_overrides and all(row["policy_controller"] == "frozen_actor"
                                        for row in (*ordinary, *interventions))
    if (report["base_rows"] != len(ordinary)
            or report["episodes"] != 300
            or report["branch_anchor_count"] != anchor_count
            or report["zero_nn_overrides"] != zero_overrides
            or report["source_state_unchanged"] != all(row["decision_source_unchanged"] for row in ordinary)
            or report["counterfactual_isolated"] != (all(row["source_unchanged"] for row in interventions)
                and all(row["source_unchanged"] for row in language))
            or report["program_never_controls_action"] is not controller
            or report.get("execution") != {"accounting_complete": True,
                "pending_operation": None, "counts": expected_execution}):
        raise ValueError("Explanation invariants differ from immutable rows")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(actor_path=args.actor, protocol_path=args.protocol,
        program_path=args.program, scenarios_path=args.scenarios, output=args.output)
    print(canonical({"status": report["status"], "statistics": report["statistics"],
                     "report": str((args.output / "report.json").resolve())}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "PARTNERS", "GROUPS", "QUESTIONS", "contract",
           "producer_sources", "audit", "read_saved_report", "main"]
