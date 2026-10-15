"""Participant-facing Kitchen adapter: a frozen NN chooses every AI action."""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from core.program import ExecutableProgram

from . import engine_v5_frozen as physics
from . import neural_policy as policy

DOMAIN = physics.DOMAIN
VERSION = "kitchen-v7.0.0-neural"
RELEASE_VERSION = "kitchen-v8.0.0-dual-actor-paired"
V5_RELEASE_ID = "policylens-three-domain-20261014.v5.0"
SCENARIO_VERSION = physics.SCENARIO_VERSION
MENU_VERSION = physics.MENU_VERSION
ROOT = Path(__file__).resolve().parents[2]


def __getattr__(name):
    return getattr(physics, name)


@lru_cache(maxsize=1)
def _deployment():
    from study_v3.release_catalog import runtime_config

    config = json.loads(runtime_config("kitchen").read_text())
    deployment = config.get("deployment_policy") or {}
    if deployment.get("model_version") != policy.MODEL_VERSION or not deployment.get("sha256"):
        raise RuntimeError("Kitchen frozen NN is not configured")
    return deployment


@lru_cache(maxsize=2)
def policy_artifacts(release_id=None):
    """Pinned v5 variants; an absent digest is an explicitly unready policy."""
    if release_id is not None:
        from study_v3.release_catalog import RELEASE_ID, policy_artifacts as bundled
        if release_id != RELEASE_ID:
            raise RuntimeError("Unknown Kitchen release catalog")
        return bundled("kitchen")
    payload = json.loads((ROOT / "configs/study_v3_kitchen_v5.json").read_text())
    result = payload["policies"]
    if set(result) != {"regularized", "raw"}:
        raise RuntimeError("Kitchen v5 requires raw and regularized policy manifests")
    return result


def _v5_deployment(policy_id, release_id=None):
    artifacts = policy_artifacts() if release_id is None else policy_artifacts(release_id)
    for variant, entry in artifacts.items():
        if entry.get("policy_id") == policy_id:
            digest = entry.get("actor_sha256")
            roles = entry.get("role_sha256") or {}
            if (not isinstance(digest, str) or len(digest) != 64
                    or not all(isinstance(roles.get(role), str) and len(roles[role]) == 64
                               for role in ("human", "ai"))):
                raise RuntimeError(f"Kitchen {variant} v5 frozen Actor is not ready")
            _verify_v5_acceptance(entry, variant)
            return entry
    raise RuntimeError(f"Unknown Kitchen v5 policy_id: {policy_id}")


def _v5_pinned_digest(path, expected, label):
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"Kitchen v5 {label} is not pinned")
    target = (ROOT / path).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"Kitchen v5 {label} path escapes the release root")
    try:
        actual = sha256(target.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError(f"Kitchen v5 {label} is missing") from exc
    if actual != expected:
        raise RuntimeError(f"Kitchen v5 {label} checksum mismatch")
    return actual


def _verify_v5_acceptance(entry, variant):
    """Direct v5 calls require the same accepted asset identity as enrollment."""
    receipt_path = (ROOT / entry["acceptance_path"]).resolve()
    if not receipt_path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("Kitchen v5 acceptance path escapes the release root")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Kitchen v5 acceptance receipt is missing or malformed") from exc
    if not isinstance(receipt, dict):
        raise RuntimeError("Kitchen v5 acceptance receipt is malformed")
    _v5_pinned_digest(entry["training_report_path"], entry["training_report_sha256"],
                      "training report")
    if ((receipt.get("artifact_sha256") or {}).get(variant) != entry["actor_sha256"]
            or (receipt.get("training_report_sha256") or {}).get(variant)
            != entry["training_report_sha256"]):
        raise RuntimeError("Kitchen v5 acceptance and policy manifest disagree")
    if variant == "regularized":
        paths = entry.get("program_paths") or []
        if len(paths) != 2:
            raise RuntimeError("Kitchen v5 regularized pair needs two role programs")
        expected = (receipt.get("program_sha256") or {}).get("regularized") or {}
        for role, path in zip(("human", "ai"), paths):
            digest = (entry.get("program_sha256") or {}).get(role)
            _v5_pinned_digest(path, digest, "program")
            if expected.get(role) != digest:
                raise RuntimeError("Kitchen v5 program acceptance hash mismatch")
    if (receipt.get("release_id", receipt.get("release")) != V5_RELEASE_ID
            or receipt.get("passed") is not True
            or receipt.get("heldout_evaluated") is not True):
        raise RuntimeError("Kitchen v5 acceptance receipt is not approved for execution")


def _actor(policy_id=None, release_id=None):
    if policy_id is None:
        deployment = _deployment()
        return policy.load_frozen(ROOT / deployment["path"], deployment["sha256"])
    deployment = (_v5_deployment(policy_id) if release_id is None
                  else _v5_deployment(policy_id, release_id))
    path = ROOT / deployment["actor_path"]
    try:
        actual_sha256 = sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("Kitchen v5 frozen Actor is unavailable") from exc
    if actual_sha256 != deployment["actor_sha256"]:
        raise RuntimeError("Kitchen v5 frozen Actor checksum mismatch")
    actor = policy.load_frozen(path, deployment["actor_sha256"])
    if actor.role_sha256 != deployment["role_sha256"]:
        raise RuntimeError("Kitchen v5 role Actor checksum mismatch")
    return actor


def _bound_actor(state):
    policy_id = state.get("policy_id")
    if state.get("release_id") is not None:
        from study_v3.release_catalog import runtime_config

        runtime_config("kitchen", state["release_id"])
    actor = _actor(policy_id, state.get("release_id")) if policy_id is not None else _actor()
    if policy_id is not None:
        if state.get("policy_sha256") != actor.sha256 or state.get("role_sha256") != actor.role_sha256:
            raise RuntimeError("Kitchen saved policy binding does not match the pinned Actor")
    return actor


def rule_metadata(policy_id=None):
    metadata = physics.rule_metadata()
    # The former two-turn disposal commitment was a controller preference,
    # not a physical rule. The frozen Actor may choose another legal action.
    metadata.pop("handoff_grace_turns", None)
    metadata.update(engine_version=RELEASE_VERSION if policy_id else VERSION,
                    scenario_version=SCENARIO_VERSION,
                    policy_version=policy.MODEL_VERSION,
                    policy_execution="frozen_actor_masked_argmax")
    return metadata


def initial_state(seed: int, task: int, policy_id=None, release_id=None):
    if release_id is not None:
        from study_v3.release_catalog import runtime_config

        runtime_config("kitchen", release_id)
    actor = _actor(policy_id, release_id) if policy_id is not None else _actor()
    state = physics.initial_state(seed, task)
    state["version"] = RELEASE_VERSION if policy_id is not None else VERSION
    state["rule_metadata"] = rule_metadata(policy_id)
    state["policy_memory"] = {}
    if policy_id is not None:
        state["policy_id"] = policy_id
        state["policy_sha256"] = actor.sha256
        state["role_sha256"] = dict(actor.role_sha256)
        if release_id is not None:
            state["release_id"] = release_id
    return state


def decide(state):
    expected_version = RELEASE_VERSION if state.get("policy_id") else VERSION
    if state["version"] != expected_version:
        raise ValueError("Kitchen NN state version mismatch")
    model = _bound_actor(state)
    if state["terminal"]:
        return {"action": "wait", "action_id": "terminal", "policy_origin": "kitchen_nn",
                "model_version": policy.MODEL_VERSION, "model_sha256": model.sha256,
                "role_sha256": dict(model.role_sha256) if hasattr(model, "role_sha256") else {},
                "policy_id": state.get("policy_id"), "reason_code": "terminal",
                "reason_en": "The task has ended; there is no next AI action.",
                "reason_zh": "任务已结束，AI 没有下一步动作。", "goal": None, "memory": {},
                "alternatives": [], "facts": [], "action_label_en": "No action", "action_label_zh": "无动作"}
    action_id, scores, mask = model.choose(state, "ai")
    choice = policy.action_spec(action_id)
    label_en = physics.action_label(choice["action"])
    label_zh = physics.action_label(choice["action"], "zh")
    if choice["action"] == "interact":
        station = physics._front(state["ai"])
        target = station["label_en"] if station else "station"
        target_zh = station["label_zh"] if station else "工位"
        label_en = f"Interact with {target}"
        label_zh = f"操作{target_zh}"
        if "slot" in choice:
            label_en += f" slot {choice['slot'] + 1}"
            label_zh += f"第 {choice['slot'] + 1} 槽"
        if "order_id" in choice:
            label_en += f" for {choice['order_id']}"
            label_zh += f"并绑定{choice['order_id']}"
    legal_scores = [{"action_id": name, "score": float(scores[i]), "legal": bool(mask[i])}
                    for i, name in enumerate(policy.AI_ACTIONS)]
    return {**choice, "action_id": action_id, "policy_origin": "kitchen_nn",
            "model_version": policy.MODEL_VERSION, "model_sha256": model.sha256,
            "policy_id": state.get("policy_id"),
            "role_sha256": dict(model.role_sha256) if hasattr(model, "role_sha256") else {},
            "legal_mask": [bool(x) for x in mask], "action_scores": legal_scores,
            "reason_code": "neural_selection", "goal": None, "memory": {},
            "reason_en": f"The frozen Kitchen neural network assigned its highest legal score to: {label_en.lower()}. This output alone does not reveal its internal motive.",
            "reason_zh": f"冻结的 Kitchen 神经网络在合法动作中给“{label_zh}”最高分。仅凭这一输出无法推断其内部动机。",
            "alternatives": [], "facts": [], "action_label_en": label_en, "action_label_zh": label_zh}


def step(state, human_action, decision=None):
    expected = decide(state)
    if decision is not None and decision != expected:
        raise ValueError("AI decision does not match the frozen NN on this state")
    return physics.step(state, human_action, expected)


def legal_actions(state, actor="human"):
    if actor == "human":
        return physics.legal_actions(state, actor)
    if actor == "ai":
        mask = policy.legal_mask(state, role="ai")
        return [name for name, legal in zip(policy.AI_ACTIONS, mask) if legal]
    raise ValueError("Unknown actor")


def human_advisor(state):
    """Suggest the frozen human-role Actor's legal action in a bound v5 task."""
    if state.get("policy_id") is None:
        return physics.human_advisor(state)
    if state["terminal"]:
        raise ValueError("There is no participant action after the task ends")
    action = _bound_actor(state).choose(state, "human")[0]
    if action not in physics.legal_actions(state, "human"):
        raise RuntimeError("Frozen human Actor selected an unavailable participant action")
    return action


def public_state(state):
    view = physics.public_state(state)
    view["rule_metadata"] = rule_metadata(state.get("policy_id"))
    view.pop("handoff_grace", None)
    view["ai"]["current_cooking"] = [row for row in view["ai"]["current_cooking"] if row.get("location") != "plan"]
    view["ai"]["policy"] = {"kind": "frozen_neural_actor", "version": policy.MODEL_VERSION,
                            "policy_id": state.get("policy_id"),
                            "actor_sha256": state.get("policy_sha256")}
    return view


def demonstration(policy_id=None, release_id=None):
    """AI–AI trace from the same pinned pair used for human takeover."""
    state = initial_state(1000, 1, policy_id=policy_id, release_id=release_id)
    actor = _bound_actor(state)
    frames = [public_state(state)]
    captions = [{"index": 0,
                 "en": "Watch two frozen neural Actors cooperate. You will take over the human-side role.",
                 "zh": "观察两个冻结神经网络智能体协作；随后你将接管人类侧角色。"}]
    served = 0
    while not state["terminal"]:
        human_action, _, _ = actor.choose(state, "human")
        decision = decide(state)
        state = step(state, human_action, decision)
        frames.append(public_state(state))
        if state["metrics"]["completed_orders"] > served:
            served = state["metrics"]["completed_orders"]
            captions.append({"index": len(frames) - 1,
                             "en": f"The neural pair completed order {served} together.",
                             "zh": f"两个神经网络智能体合作完成了第 {served} 单。"})
            if served >= 1:
                break
    if served == 0:
        captions.append({"index": len(frames) - 1,
                         "en": "The neural pair did not complete an order in this demonstration.",
                         "zh": "这次演示中，两个神经网络智能体未完成订单。"})
    return {"frames": frames, "captions": captions,
            "policy_id": state.get("policy_id"), "actor_sha256": actor.sha256}


def _audited_ai_program(policy_id, release_id=None):
    entry = (_v5_deployment(policy_id) if release_id is None
             else _v5_deployment(policy_id, release_id))
    if policy_id != "kitchen-regularized-v5":
        return None
    if (len(entry.get("program_paths", ())) < 2 or not entry.get("acceptance_path")
            or not entry.get("training_report_path")):
        return None
    acceptance_path = ROOT / entry["acceptance_path"]
    program_paths = {role: ROOT / destination for role, destination in zip(
        ("human", "ai"), entry["program_paths"])}
    program_path = program_paths["ai"]
    report_path = ROOT / entry["training_report_path"]
    try:
        acceptance = json.loads(acceptance_path.read_text())
        report_bytes = report_path.read_bytes()
        program_bytes = {role: path.read_bytes() for role, path in program_paths.items()}
    except (OSError, ValueError, IndexError):
        return None
    program_digests = {role: sha256(payload).hexdigest() for role, payload in program_bytes.items()}
    audits = acceptance.get("final_role_program_development_audit") or {}
    if not isinstance(audits, dict) or set(audits) != {"human", "ai"}:
        return None
    if (not acceptance.get("passed")
            or acceptance.get("verification_errors") != []
            or acceptance.get("artifact_sha256", {}).get("regularized") != entry["actor_sha256"]
            or sha256(report_bytes).hexdigest() != entry.get("training_report_sha256")
            or sha256(report_bytes).hexdigest() != acceptance.get("training_report_sha256", {}).get("regularized")
            or acceptance.get("final_ai_program_development_audit") != audits["ai"]):
        return None
    for role in ("human", "ai"):
        audit = audits[role]
        try:
            fidelity = float(audit.get("action_fidelity", 0.0))
            mean_kl = float(audit.get("mean_kl_divergence", float("inf")))
        except (AttributeError, TypeError, ValueError):
            return None
        if (program_digests[role] != entry.get("program_sha256", {}).get(role)
                or program_digests[role] != acceptance.get("program_sha256", {}).get("regularized", {}).get(role)
                or audit.get("role") != role
                or audit.get("actor_sha256") != entry["actor_sha256"]
                or audit.get("role_sha256") != entry.get("role_sha256", {}).get(role)
                or audit.get("program_sha256") != program_digests[role]
                or audit.get("development_seeds") != list(range(41_000, 41_020))
                or audit.get("samples", 0) < 1
                or not audit.get("passed")
                or not isfinite(fidelity) or fidelity < .80
                or not isfinite(mean_kl) or mean_kl > .35):
            return None
    try:
        program = ExecutableProgram.load_json(program_path)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if (program.metadata.get("role") != "ai"
            or program.metadata.get("program_phase") != "final_frozen_actor_training_only_reextraction"
            or program.metadata.get("actor_sha256") != entry["actor_sha256"]
            or program.metadata.get("role_sha256") != entry.get("role_sha256", {}).get("ai")
            or program.metadata.get("runtime_control_allowed") is not False
            or not (program.metadata.get("metrics") or {}).get("feedback_eligible")):
        return None
    return program, audits["ai"], program_digests["ai"]


def audited_program_evidence(state):
    """Optional factual explanation evidence; never a controller action."""
    if state.get("policy_id") != "kitchen-regularized-v5" or state.get("terminal"):
        return None
    _bound_actor(state)
    audited = (_audited_ai_program(state["policy_id"])
               if state.get("release_id") is None else
               _audited_ai_program(state["policy_id"], state["release_id"]))
    if audited is None:
        return None
    program, metrics, digest = audited
    execution = program.execute(policy.program_features(state, "ai"))
    neural_action = decide(state)["action_id"]
    if execution.action != neural_action:
        return None
    fidelity = float(metrics["action_fidelity"])
    path = [f"{step.feature} {step.operator} {step.threshold:.3g} (observed {step.observed_value:.3g})"
            for step in execution.trace.tree_steps]
    path_en = "; ".join(path) if path else "a single leaf with no predicate"
    en = (f"The audited bounded program and frozen AI Actor both selected {neural_action} on this state. "
          f"Its independent final-Actor development action fidelity was {fidelity:.1%}. "
          f"Its observed predicate path was: {path_en}.")
    zh = (f"经审计的有界程序与冻结 AI Actor 在当前状态都选择了{neural_action}。"
          f"程序相对于最终冻结 Actor 在独立开发集上的动作一致率为{fidelity:.1%}。"
          f"此次经过的可观察条件为：{path_en}。")
    return {"en": en, "zh": zh, "program_sha256": digest, "eligible": True,
            "policy_id": state["policy_id"], "model_sha256": state["policy_sha256"],
            "action_id": neural_action}


def rules(language="en"):
    # Physical rules stay intact. Remove only the former fixed controller's
    # promised blocked-output route from participant-facing help.
    rows = physics.rules(language)
    return [row for index, row in enumerate(rows) if index != 10]


def facts(state, decision=None):
    selected = decide(state)
    if decision is not None and decision != selected:
        raise ValueError("AI evidence decision does not match the frozen NN")
    if state.get("policy_id") is None:
        rows = physics.facts(state, selected)
        rows = [row for row in rows if row["id"] not in (
            "next_input_ingredient", "human_available_option")]
        if not state["terminal"]:
            rows.append({"id": "ai_policy_source",
                         "en": "The AI action comes from the frozen neural Actor and the current legal-action mask. Its score does not establish a human-readable intention.",
                         "zh": "AI 动作来自冻结神经网络及当前合法动作掩码。动作分数不等于可直接解释的人类意图。"})
        return rows
    rows = physics.facts(state, selected, include_human_advice=False)
    rows = [row for row in rows if row["id"] not in (
        "next_input_ingredient", "human_available_option", "handoff_recovery",
        "ai_current_dishes") and not row["id"].startswith(("public_rule", "recipe_sequence_"))]
    actual_jobs = public_state(state)["ai"]["current_cooking"]
    descriptions_en, descriptions_zh = [], []
    for job in actual_jobs:
        location = job["location"]
        if location == "pan":
            en_location = f"in stove {job['pot_id'][-1]} ({job['status']})"
            zh_location = f"在炉灶 {job['pot_id'][-1]}（{job['status']}）"
        elif location == "held":
            en_location, zh_location = "held by the AI", "由 AI 手持"
        elif location == "stored":
            en_location, zh_location = "on a physical temporary counter", "在实体临时台上"
        else:
            raise RuntimeError("Kitchen v5 public view contains a nonphysical dish plan")
        descriptions_en.append(f"{job['recipe_label_en']} {en_location}")
        descriptions_zh.append(f"{job['recipe_label_zh']}{zh_location}")
    rows.append({"id": "ai_current_dishes",
                 "en": "Visible AI-side cooking or finished dishes: "
                       + ("; ".join(descriptions_en) if descriptions_en else "none") + ".",
                 "zh": "AI 一侧可见的烹饪中或已完成菜品："
                       + ("；".join(descriptions_zh) if descriptions_zh else "无") + "。"})
    for recipe, spec in physics.RECIPES.items():
        protein = physics.LABELS[spec["protein"]]
        vegetable = physics.LABELS[spec["vegetable"]]
        rows.append({"id": f"recipe_sequence_{recipe}",
                     "en": f"Kitchen interactions for {spec['en']} require cooking {protein[0]} first, "
                           f"moving it to its pan's temporary counter, cooking {vegetable[0]}, "
                           "then returning the cooked first component and mixing. "
                           "This describes the environment sequence, not the Actor's future choices.",
                     "zh": f"制作{spec['zh']}的环境交互顺序是先炒{protein[1]}并放到对应临时台，"
                           f"再炒{vegetable[1]}，然后倒回已熟主料合炒。"
                           "这是环境操作顺序，不代表 Actor 未来一定这样行动。"})
    for index, (en, zh) in enumerate(zip(physics.rules("en"), physics.rules("zh"))):
        if index != 10:
            rows.append({"id": f"public_rule{index}", "en": en, "zh": zh})
    if not state["terminal"]:
        rows.append({"id": "ai_policy_source",
                     "en": "The AI action comes from the frozen neural Actor and the current legal-action mask. Its score does not establish a human-readable intention.",
                     "zh": "AI 动作来自冻结神经网络及当前合法动作掩码。动作分数不等于可直接解释的人类意图。"})
    return rows
