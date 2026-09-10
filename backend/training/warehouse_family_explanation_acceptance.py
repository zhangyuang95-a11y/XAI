"""Prospective explanation-system contract and hash-bound record aggregation.

No Actor, environment, tree or evidence file is loaded here. A trusted caller
must anchor the preregistered plan and records produced by the real renderer and
an independent raw-NN/physics replay. Matching JSON cannot prove that execution
occurred: this module reports record-level checks, never release eligibility.

This is an explicit revision of the acceptance object, not an equivalent pass
under the old tree-direction contract. Numerical 90/85 thresholds remain, but
tree direction remains a diagnostic and the actual NN branch engine carries
the counterfactual correctness obligation. Historical failures remain failures.
"""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import math
import re

VERSION = "warehouse-family-explanation-system-record-acceptance.v2"
PLAN_VERSION = "warehouse-family-explanation-system-case-plan.v2"
EVIDENCE_VERSION = "warehouse-family-explanation-system-records.v2"
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
ROLES = ("robot_1", "robot_2")
GROUPS = ("narrow_passage", "shared_pickup", "shared_charger")
BINDINGS = ("actor_sha256", "program_sha256", "protocol_sha256", "runtime_signature",
    "runtime_sources_sha256", "independent_replay_sources_sha256", "answer_verifier_sources_sha256",
    "plan_sha256", "contract_sha256", "acceptance_sources_sha256")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def execution_sources():
    return {"backend/training/warehouse_family_explanation_acceptance.py": sha256(Path(__file__).read_bytes()).hexdigest()}


def contract():
    return {"version": VERSION, "plan_version": PLAN_VERSION, "evidence_version": EVIDENCE_VERSION,
        "preregistration": "freeze plan, Actor, program, sources and all cases before new acceptance outcomes",
        "revision": "counterfactual responsibility assigned to actual isolated NN engine; not equivalent to old tree-direction acceptance",
        "historical_tree_failures_reclassified_as_passes": False,
        "roles": list(ROLES), "actions": list(ACTIONS), "critical_groups": list(GROUPS),
        "minimum_holdout_scenes": 64, "minimum_stratum_scenes": 10,
        "tree_base_overall": .90, "tree_base_nonwait": .85, "tree_critical": .85,
        "tree_critical_nonwait": .85, "tree_each_role": .85, "tree_each_role_nonwait": .85,
        "engine_action_replay_agreement": .85, "engine_effective_direction_agreement": .85,
        "engine_each_role_and_critical_agreement": .85,
        "direction_population": "independent replay has nonterminal endpoints, different physical projections and different next NN argmax",
        "direction_correct": "engine agrees with the independent NN at both endpoints",
        "tree_pair_diagnostic": "unchanged: tree agrees with NN at both physically effective, NN-changing endpoints",
        "tree_pair_diagnostic_is_gate": False,
        "engine_physical_trace_agreement": 1., "live_state_and_rng_preservation": 1.,
        "neural_submission_preservation": 1., "answer_oracle_match_and_disclosure": 1.,
        "maximum_counterfactual_steps": 3, "unspecified_followups": "WAIT",
        "empty_evidence_passes": False, "fit_or_selection_data_allowed": False,
        "new_holdout_result_read_by_this_module_development": False,
        "release_eligibility_granted": False}


def _sha(value):
    if type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected exact SHA256")
    return value


def _probabilities(value):
    if (type(value) is not list or len(value) != 5 or any(type(x) not in (int, float)
            or not math.isfinite(x) or not 0 <= x <= 1 for x in value)
            or not math.isclose(sum(value), 1., abs_tol=1e-6, rel_tol=0)):
        raise ValueError("Five finite genuine probability labels required")
    return ACTIONS[max(range(5), key=value.__getitem__)]


def _cases(plan, kind):
    rows = plan[kind]
    if type(rows) is not list: raise ValueError("Explicit preregistered case list required")
    result = {}
    for row in rows:
        required = {"id", "fingerprint", "role", "groups"}
        if kind == "counterfactual_cases": required |= {"horizon", "intervention_action"}
        if type(row) is not dict or set(row) != required or type(row["id"]) is not str or not row["id"]:
            raise ValueError("Case schema differs")
        if (row["id"] in result or row["role"] not in ROLES
                or type(row["groups"]) is not list or len(set(row["groups"])) != len(row["groups"])
                or not set(row["groups"]) <= set(GROUPS)):
            raise ValueError("Duplicate or invalid case stratum")
        _sha(row["fingerprint"])
        if kind == "counterfactual_cases" and (type(row["horizon"]) is not int
                or not 1 <= row["horizon"] <= 3 or row["intervention_action"] not in ACTIONS[:-1]):
            raise ValueError("Only direction-versus-WAIT cases of one to three steps are supported")
        result[row["id"]] = row
    return result


def _match_rows(rows, cases):
    if type(rows) is not list: raise ValueError("Raw case records required")
    out = {}
    for row in rows:
        if type(row) is not dict or row.get("id") not in cases or row["id"] in out:
            raise ValueError("Duplicate, missing or unregistered evidence case")
        out[row["id"]] = row
    if set(out) != set(cases): raise ValueError("Complete preregistered case matrix required")
    return out


def _rate(rows, key):
    count = len(rows); correct = sum(bool(row[key]) for row in rows)
    return {"rows": count, "correct": correct, "scenes": len({r["fingerprint"] for r in rows}),
        "rate": correct / count if count else None}


def _strata(rows, key):
    return {"overall": _rate(rows, key),
        **{"role:"+role: _rate([r for r in rows if r["role"] == role], key) for role in ROLES},
        **{"critical:"+group: _rate([r for r in rows if group in r["groups"]], key) for group in GROUPS}}


def _meets(stat, threshold, scenes=10):
    return stat["rows"] > 0 and stat["scenes"] >= scenes and stat["rate"] >= threshold


def _run(value, case, bindings, *, independent):
    """Validate saved branch structure; never pretend this is physical replay."""
    required = {"implementation", "execution_id", "actor_sha256", "runtime_signature", "sources_sha256",
        "live_before_sha256", "live_after_sha256", "rng_before_sha256", "rng_after_sha256", "branches"}
    if type(value) is not dict or set(value) != required: raise ValueError("Independent run evidence is incomplete")
    method = "independent_raw_nn_physics" if independent else "runtime_isolated_counterfactual"
    source = bindings["independent_replay_sources_sha256" if independent else "runtime_sources_sha256"]
    if (value["implementation"] != method or value["actor_sha256"] != bindings["actor_sha256"]
            or value["runtime_signature"] != bindings["runtime_signature"] or value["sources_sha256"] != source
            or type(value["execution_id"]) is not str or not value["execution_id"]):
        raise ValueError("Independent and engine evidence require the same frozen Actor and pinned real producers")
    for key in ("live_before_sha256", "live_after_sha256", "rng_before_sha256", "rng_after_sha256"): _sha(value[key])
    if type(value["branches"]) is not list or len(value["branches"]) != 2: raise ValueError("Two actual branch endpoints required")
    result = []; submission_ok = True
    other = next(role for role in ROLES if role != case["role"])
    for index, branch in enumerate(value["branches"]):
        if set(branch) != {"assumed_actions", "steps", "next_probabilities"}: raise ValueError("Branch schema differs")
        assumed = ["WAIT"] * case["horizon"]
        if index: assumed[0] = case["intervention_action"]
        if branch["assumed_actions"] != assumed: raise ValueError("Player assumptions differ; unspecified suffix must WAIT")
        steps = branch["steps"]
        if type(steps) is not list or not 1 <= len(steps) <= case["horizon"]: raise ValueError("Invalid actual branch step count")
        previous = None; actions = []; physical = []; terminal = False
        for offset, step in enumerate(steps):
            fields = {"before_sha256", "after_sha256", "physical_after_sha256", "after_rng_sha256",
                "frame", "nn_probabilities", "submitted_actions", "executed_actions", "terminal"}
            if set(step) != fields or type(step["frame"]) is not int or step["frame"] < 0 or type(step["terminal"]) is not bool:
                raise ValueError("Actual trace fields differ")
            for field in ("before_sha256", "after_sha256", "physical_after_sha256", "after_rng_sha256"): _sha(step[field])
            if previous is not None and (step["before_sha256"] != previous["after_sha256"] or step["frame"] != previous["frame"]+1 or terminal):
                raise ValueError("Saved branch state/frame chain is broken")
            nn_action = _probabilities(step["nn_probabilities"])
            if any(type(step[k]) is not dict or set(step[k]) != set(ROLES)
                   or any(a not in ACTIONS for a in step[k].values()) for k in ("submitted_actions", "executed_actions")):
                raise ValueError("Both submitted and physical role actions required")
            submission_ok &= step["submitted_actions"][case["role"]] == nn_action
            if step["submitted_actions"][other] != assumed[offset]: raise ValueError("Counterfactual command differs from its declared intervention")
            actions.append((nn_action, step["submitted_actions"], step["executed_actions"]))
            physical.append((step["before_sha256"], step["after_sha256"], step["physical_after_sha256"], step["after_rng_sha256"], step["terminal"]))
            previous = step; terminal = step["terminal"]
        if len(steps) != case["horizon"] and not terminal: raise ValueError("Truncated nonterminal evidence cannot count as a complete branch")
        if terminal != (branch["next_probabilities"] is None): raise ValueError("Terminal endpoint must not invent a next NN decision")
        result.append({"actions": actions, "physical": physical, "terminal": terminal,
            "origin": (steps[0]["before_sha256"], steps[0]["frame"]),
            "physical_end": steps[-1]["physical_after_sha256"],
            "next": None if terminal else _probabilities(branch["next_probabilities"])})
    if result[0]["origin"] != result[1]["origin"]: raise ValueError("Both interventions must start from one identical bound frame")
    return result, submission_ok, (value["live_before_sha256"] == value["live_after_sha256"]
        and value["rng_before_sha256"] == value["rng_after_sha256"])


def evaluate_records(evidence, *, expected_evidence_sha256, expected_bindings, allow_test_fixture=False):
    """Pure record aggregation. External hashes must come from trusted receipts.

    The expected evidence hash is the canonical JSON semantic hash defined by
    digest(), not a file-byte hash. No report eligible/passed booleans are used.
    """
    _sha(expected_evidence_sha256)
    if digest(evidence) != expected_evidence_sha256: raise ValueError("External evidence hash differs")
    fields = {"version", "test_fixture", "bindings", "plan", "base_rows", "counterfactual_rows", "answer_rows"}
    if (type(evidence) is not dict or set(evidence) != fields or evidence["version"] != EVIDENCE_VERSION
            or type(allow_test_fixture) is not bool or evidence["test_fixture"] is not allow_test_fixture):
        raise ValueError("New evidence schema and explicit matching fixture scope required")
    if type(expected_bindings) is not dict or set(expected_bindings) != set(BINDINGS): raise ValueError("All external evidence/source bindings required")
    for value in expected_bindings.values(): _sha(value)
    if evidence["bindings"] != expected_bindings: raise ValueError("Evidence belongs to another frozen input")
    if (expected_bindings["contract_sha256"] != digest(contract())
            or expected_bindings["acceptance_sources_sha256"] != digest(execution_sources())):
        raise ValueError("Acceptance contract or source changed")
    plan = evidence["plan"]
    if (type(plan) is not dict or set(plan) != {"version", "contract_sha256", "holdout_fingerprints", "excluded_fingerprints", "base_cases", "counterfactual_cases"}
            or plan["version"] != PLAN_VERSION or plan["contract_sha256"] != expected_bindings["contract_sha256"]
            or digest(plan) != expected_bindings["plan_sha256"]): raise ValueError("Externally preregistered plan differs")
    pools = []
    for key in ("holdout_fingerprints", "excluded_fingerprints"):
        rows = plan[key]
        if type(rows) is not list or len(set(rows)) != len(rows): raise ValueError("Unique actual physical initial fingerprints required")
        for value in rows: _sha(value)
        pools.append(set(rows))
    if pools[0] & pools[1]: raise ValueError("Holdout overlaps exposed train/selection/other pools")
    base_cases, cf_cases = _cases(plan, "base_cases"), _cases(plan, "counterfactual_cases")
    if set(base_cases) & set(cf_cases): raise ValueError("Case identities must be globally unique")
    if any(c["fingerprint"] not in pools[0] for c in (*base_cases.values(), *cf_cases.values())): raise ValueError("Case not in frozen holdout pool")
    base, cf = _match_rows(evidence["base_rows"], base_cases), _match_rows(evidence["counterfactual_rows"], cf_cases)
    base_stats = []; answers_expected = {}; cf_stats = []; live = []; submissions = []
    for key, row in base.items():
        if set(row) != {"id", "nn_probabilities", "tree_probabilities"}: raise ValueError("Only ordinary base action rows belong in the ordinary gate")
        nn, tree = _probabilities(row["nn_probabilities"]), _probabilities(row["tree_probabilities"])
        base_stats.append({**base_cases[key], "nonwait": nn != "WAIT", "correct": nn == tree})
        answers_expected[key] = ("nn_and_approximate_tree", nn != tree)
    for key, row in cf.items():
        if set(row) != {"id", "engine", "independent_replay", "tree_endpoint_probabilities"}: raise ValueError("Original and independent branch evidence both required")
        case = cf_cases[key]
        engine, es, el = _run(row["engine"], case, expected_bindings, independent=False)
        replay, rs, rl = _run(row["independent_replay"], case, expected_bindings, independent=True)
        if (row["engine"]["execution_id"] == row["independent_replay"]["execution_id"]
                or engine[0]["origin"] != replay[0]["origin"]): raise ValueError("A collector response is not its independent replay")
        live.extend((el, rl)); submissions.extend((es, rs))
        active = not any(b["terminal"] for b in replay)
        effective = active and replay[0]["physical_end"] != replay[1]["physical_end"] and replay[0]["next"] != replay[1]["next"]
        probabilities = row["tree_endpoint_probabilities"]
        if type(probabilities) is not list or len(probabilities) != 2: raise ValueError("Preserve both original tree diagnostic labels")
        tree = []
        for p, end in zip(probabilities, replay):
            if end["terminal"]:
                if p is not None: raise ValueError("No fabricated terminal tree label")
                tree.append(None)
            else: tree.append(_probabilities(p))
        cf_stats.append({**case, "effective": effective,
            "action_correct": all(a["actions"] == b["actions"] for a, b in zip(engine, replay)),
            "physical_correct": all(a["physical"] == b["physical"] for a, b in zip(engine, replay)),
            "direction_correct": all(a["next"] == b["next"] and a["terminal"] == b["terminal"] for a, b in zip(engine, replay)),
            "tree_correct": all(a == b["next"] for a, b in zip(tree, replay))})
        answers_expected[key] = ("isolated_nn_branch", False)
    answers = evidence["answer_rows"]; seen = set(); answer_checks = []
    if type(answers) is not list: raise ValueError("Actual bilingual rendered answers and independent oracle text required")
    for row in answers:
        fields = {"case_id", "language", "evidence_method", "text", "independent_expected_text", "mismatch_disclosed", "claims_tree_as_internal_cause"}
        if type(row) is not dict or set(row) != fields or row["case_id"] not in answers_expected or row["language"] not in ("zh", "en"):
            raise ValueError("Bilingual answer identity differs")
        identity = (row["case_id"], row["language"])
        if identity in seen: raise ValueError("Duplicate answer cannot inflate evidence")
        seen.add(identity); method, mismatch = answers_expected[row["case_id"]]
        if type(row["mismatch_disclosed"]) is not bool or type(row["claims_tree_as_internal_cause"]) is not bool: raise ValueError("Explicit answer provenance flags required")
        phrase = "不能用这个树分支解释本次选择" if row["language"] == "zh" else "its branch cannot explain this choice"
        text_ok = type(row["text"]) is str and bool(row["text"].strip()) and row["text"] == row["independent_expected_text"]
        answer_checks.append(text_ok and row["evidence_method"] == method and not row["claims_tree_as_internal_cause"]
            and (not mismatch or (row["mismatch_disclosed"] and phrase in row["text"])))
    if seen != {(key, language) for key in answers_expected for language in ("zh", "en")}:
        raise ValueError("Complete bilingual answer matrix required")
    tree_stats = _strata(base_stats, "correct")
    nonwait = _strata([r for r in base_stats if r["nonwait"]], "correct")
    engine_actions = _strata(cf_stats, "action_correct")
    effective_rows = [r for r in cf_stats if r["effective"]]
    engine_direction = _strata(effective_rows, "direction_correct")
    checks = {"holdout_matrix": len(pools[0]) >= contract()["minimum_holdout_scenes"] and {r["fingerprint"] for r in base_stats} == pools[0],
        "tree_ordinary_overall": _meets(tree_stats["overall"], .90),
        "tree_ordinary_nonwait": _meets(nonwait["overall"], .85),
        "engine_live_state_rng_unchanged": bool(live) and all(live),
        "engine_nn_submissions_unchanged": bool(submissions) and all(submissions),
        "engine_physical_trace_identical": bool(cf_stats) and all(r["physical_correct"] for r in cf_stats),
        "bilingual_answers_and_disclosure": bool(answer_checks) and all(answer_checks)}
    for name in tree_stats:
        if name != "overall":
            checks["tree_ordinary_"+name] = _meets(tree_stats[name], .85)
            checks["tree_nonwait_"+name] = _meets(nonwait[name], .85)
        checks["engine_actions_"+name] = _meets(engine_actions[name], .85)
        checks["engine_direction_"+name] = _meets(engine_direction[name], .85)
    return {"version": VERSION, "test_fixture": allow_test_fixture, "bindings": deepcopy(expected_bindings),
        "evidence_sha256": expected_evidence_sha256, "checks": checks, "records_passed": all(checks.values()),
        "ordinary_tree": tree_stats, "ordinary_tree_nonwait": nonwait,
        "engine_actions": engine_actions, "engine_effective_direction": engine_direction,
        "tree_pair_diagnostic": _strata(effective_rows, "tree_correct"),
        "all_counterfactual_cases": len(cf_stats), "effective_direction_cases": len(effective_rows),
        "tree_pair_diagnostic_is_gate": False, "historical_failures_reclassified": False,
        "scope": "externally anchored record aggregation; independent producer execution/authenticity requires upstream verification",
        "actual_environment_steps": 0, "actual_nn_queries": 0, "actual_tree_fits": 0,
        "physical_replay_performed_by_this_call": False, "qualification_evaluated": False,
        "explanation_eligible": False, "participant_enabled": False, "release_ready": False}
