"""Paired Kitchen self-play: two separate Actors, raw PPO versus PPO+RCPD.

The command is deliberately release-gated.  A short smoke run is useful for
checking mechanics, but only three completed 200,000-step pairs followed by
one held-out evaluation can write the pinned v5 policy manifest.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from core.program import ExecutableProgram
from core.rcpd import RCPD
from core.rcpd_config import RCPDConfig
from . import engine_v5_frozen as e
from . import neural_policy as p
from .training import DualKitchenNetwork, _transition, bc_update, evaluation, export, teacher_data

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "policylens-three-domain-20261014.v5.0"
PAIR_SEEDS = (7, 11, 19)
JOINT_STEPS = 200_000
EXTRACTION_INTERVAL = 10_000
EXTRACTION_ROWS = 4_096
DEVELOPMENT_START = 40_000
FINAL_PROGRAM_VALIDATION_START = 41_000
FINAL_PROGRAM_VALIDATION_SEEDS = 20
HELDOUT_START = 50_000
SCENARIO_BLOCK_STEPS = 360  # The Kitchen engine's full episode horizon.


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def _role_l2(initial, final, role):
    return float(np.sqrt(sum(torch.sum((after.detach() - before.detach()) ** 2).item()
                             for before, after in zip(getattr(initial, role).parameters(),
                                                      getattr(final, role).parameters()))))


def initial_actor_digest(model):
    """Canonical SHA-256 of the two BC Actor weight sets, excluding Critic."""
    tensors = {}
    for role in ("human", "ai"):
        actor = getattr(model, role)
        for layer, module_index in enumerate((0, 2, 4)):
            tensors[f"{role}.weight{layer}"] = actor[module_index].weight.detach().cpu().numpy()
            tensors[f"{role}.bias{layer}"] = actor[module_index].bias.detach().cpu().numpy()
    role_hashes = {role: p.role_weight_sha256(tensors, role) for role in ("human", "ai")}
    return sha256(f"{role_hashes['human']}:{role_hashes['ai']}".encode("ascii")).hexdigest()


def scenario_for_step(pair_seed, joint_step):
    """A policy-independent scenario for each global joint environment step."""
    block = joint_step // SCENARIO_BLOCK_STEPS
    return 17_000 + pair_seed * 1_000_000 + block, 1 + block % 3


def scenario_schedule_digest(pair_seed, budget=JOINT_STEPS):
    """Digest the actual expanded per-step seed/task schedule for one pair."""
    digest = sha256()
    for joint_step in range(budget):
        seed, task = scenario_for_step(pair_seed, joint_step)
        digest.update(f"{seed}:{task}\n".encode("ascii"))
    return digest.hexdigest()


def _collect(model, pair_seed, start_step, max_blocks, remaining, history, seed_digest=None):
    """Collect fixed global-step blocks, restarting a terminal scene within its block.

    Both variants therefore see the same seed/task on every one of their
    200,000 joint steps, even when their policies finish episodes at different
    times.  A nonterminal trajectory is bootstrapped at a block boundary.
    """
    if start_step % SCENARIO_BLOCK_STEPS:
        raise ValueError("Kitchen collection must begin at a scenario block boundary")
    trajectories = []
    steps = 0
    for offset in range(max_blocks):
        if steps >= remaining:
            break
        block_index = start_step // SCENARIO_BLOCK_STEPS + offset
        scenario_seed, task = scenario_for_step(pair_seed, block_index * SCENARIO_BLOCK_STEPS)
        block_budget = min(SCENARIO_BLOCK_STEPS, remaining - steps)
        within_block = 0
        restart = 0
        while within_block < block_budget:
            episode_id = f"{pair_seed}:{block_index}:{restart}"
            state = e.initial_state(scenario_seed, task)
            rows = []
            while not state["terminal"] and within_block < block_budget:
                vector = p.features(state)
                masks = {role: p.legal_mask(state, role) for role in ("human", "ai")}
                named = {role: p.program_features(state, role) for role in ("human", "ai")}
                x = torch.as_tensor(vector[None, :])
                with torch.no_grad():
                    hdist = Categorical(logits=model.logits(x, "human", torch.as_tensor(masks["human"][None, :])))
                    adist = Categorical(logits=model.logits(x, "ai", torch.as_tensor(masks["ai"][None, :])))
                    hsample, asample = hdist.sample(), adist.sample()
                    logp = float((hdist.log_prob(hsample) + adist.log_prob(asample)).item())
                    value = float(model.critic(x).item())
                human_id = p.HUMAN_ACTIONS[int(hsample)]
                ai_id = p.AI_ACTIONS[int(asample)]
                before = state["raw_score"]
                state = _transition(state, human_id, ai_id)
                rows.append({"x": vector, "human_mask": masks["human"], "ai_mask": masks["ai"],
                             "human_action": int(hsample), "ai_action": int(asample),
                             "old_logp": logp, "value": value,
                             "reward": float(state["raw_score"] - before),
                             "human_features": named["human"], "ai_features": named["ai"]})
                for role in ("human", "ai"):
                    history[role].append({"x": vector, "mask": masks[role], "features": named[role],
                                          "episode_id": episode_id})
                if seed_digest is not None:
                    seed_digest.update(f"{scenario_seed}:{task}\n".encode("ascii"))
                steps += 1
                within_block += 1
            with torch.no_grad():
                bootstrap = 0.0 if state["terminal"] else float(model.critic(torch.as_tensor(p.features(state)[None, :])).item())
            trajectories.append({"rows": rows, "bootstrap": bootstrap})
            restart += 1
    return trajectories, steps, len(trajectories)


def _extraction_records(model, history, role):
    rows = list(history)
    if not rows:
        return []
    x = torch.as_tensor(np.stack([row["x"] for row in rows]))
    mask = torch.as_tensor(np.stack([row["mask"] for row in rows]))
    with torch.no_grad():
        probs = torch.softmax(model.logits(x, role, mask), dim=-1).numpy()
    actions = p.HUMAN_ACTIONS if role == "human" else p.AI_ACTIONS
    return [{"features": row["features"], "episode_id": row["episode_id"],
             "probabilities": {action: float(probability) for action, probability in zip(actions, distribution)}}
            for row, distribution in zip(rows, probs)]


def _regularizers():
    config = RCPDConfig(extraction_interval=EXTRACTION_INTERVAL,
                        minimum_extraction_samples=64,
                        max_depth=5, max_leaf_nodes=16, max_predicates=64,
                        regularization_lambda=.01,
                        regularization_start_fraction=.80,
                        regularization_ramp_fraction=.05,
                        require_action_agreement_for_feedback=True,
                        minimum_overall_fidelity_for_feedback=.70,
                        minimum_overall_fidelity_for_explanation=.80,
                        maximum_mean_kl_for_feedback=.50,
                        maximum_mean_kl_for_explanation=.35)
    return {role: RCPD(replace(config, random_seed=2026 + index))
            for index, role in enumerate(("human", "ai"))}


def _extract(model, programs, history, steps):
    events = {}
    for role, rcpd in programs.items():
        records = _extraction_records(model, history[role], role)
        actions = p.HUMAN_ACTIONS if role == "human" else p.AI_ACTIONS
        result = rcpd.maybe_extract(
            steps, records, lambda record: record["probabilities"],
            lambda record: record["features"],
            split_group_provider=lambda record: record["episode_id"],
            action_legality_features={action: f"legal_{action}" for action in actions},
            program_metadata={"domain": "kitchen", "role": role,
                              "training_source": "current_dual_actor_on_policy_states",
                              "runtime_control_allowed": False})
        events[role] = {"extracted": result is not None, "error": rcpd.last_error,
                        "feedback_eligible": bool(result and result.metrics.feedback_eligible),
                        "explanation_eligible": bool(result and result.metrics.explanation_eligible)}
    return events


def _program_loss(rcpd, logits, named):
    if rcpd.program is None or rcpd.regularization_weight <= 0:
        return None, 0
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    targets = rcpd.program_feedback_targets(named, actor_probabilities=probabilities)
    weights = rcpd.program_target_weights(named, actor_probabilities=probabilities)
    if targets is None or weights is None:
        return None, 0
    positive = int(np.count_nonzero(weights > 0))
    if positive == 0:
        return None, 0
    return rcpd.regularization_loss(logits, targets, weights), positive


def _update(model, trajectories, optimizer, programs, fraction, epochs=3, batch_size=1024):
    flat = []
    for trajectory in trajectories:
        following = trajectory["bootstrap"]
        gae = 0.0
        processed = []
        for row in reversed(trajectory["rows"]):
            delta = row["reward"] + .99 * following - row["value"]
            gae = delta + .99 * .95 * gae
            processed.append((row, gae, gae + row["value"]))
            following = row["value"]
        flat.extend(reversed(processed))
    if not flat:
        return {"steps": 0, "feedback_rows": {"human": 0, "ai": 0}}
    rows = [item[0] for item in flat]
    x = torch.as_tensor(np.stack([row["x"] for row in rows]))
    hm = torch.as_tensor(np.stack([row["human_mask"] for row in rows]))
    am = torch.as_tensor(np.stack([row["ai_mask"] for row in rows]))
    hi = torch.as_tensor([row["human_action"] for row in rows])
    ai = torch.as_tensor([row["ai_action"] for row in rows])
    old = torch.as_tensor([row["old_logp"] for row in rows], dtype=torch.float32)
    advantage = torch.as_tensor([item[1] for item in flat], dtype=torch.float32)
    returns = torch.as_tensor([item[2] for item in flat], dtype=torch.float32)
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    feedback_rows = {"human": 0, "ai": 0}
    for _ in range(epochs):
        for ids in torch.randperm(len(rows)).split(batch_size):
            hl = model.logits(x[ids], "human", hm[ids])
            al = model.logits(x[ids], "ai", am[ids])
            hd, ad = Categorical(logits=hl), Categorical(logits=al)
            logp = hd.log_prob(hi[ids]) + ad.log_prob(ai[ids])
            ratio = torch.exp(logp - old[ids])
            objective = -torch.minimum(ratio * advantage[ids],
                                       torch.clamp(ratio, .8, 1.2) * advantage[ids]).mean()
            value = .5 * nn.functional.mse_loss(model.critic(x[ids]).squeeze(-1), returns[ids])
            loss = objective + .5 * value - .002 * (hd.entropy().mean() + ad.entropy().mean())
            if programs:
                for role, logits in (("human", hl), ("ai", al)):
                    rcpd = programs[role]
                    lam = rcpd.scheduled_regularization_weight(fraction)
                    if lam:
                        named = [rows[int(i)][f"{role}_features"] for i in ids]
                        auxiliary, count = _program_loss(rcpd, logits, named)
                        if auxiliary is not None:
                            loss = loss + lam * auxiliary
                            feedback_rows[role] += count
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {"steps": len(rows), "feedback_rows": feedback_rows}


def _train_variant(initial, pair_seed, variant, args):
    model = deepcopy(initial)
    random.seed(pair_seed); np.random.seed(pair_seed); torch.manual_seed(pair_seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ppo_lr)
    programs = _regularizers() if variant == "regularized" else None
    history = {role: deque(maxlen=EXTRACTION_ROWS) for role in ("human", "ai")}
    steps = episode = blocks = 0
    actual_schedule = sha256()
    feedback = {"human": 0, "ai": 0}
    extraction_events = []
    while steps < args.steps:
        trajectories, collected, episodes = _collect(model, pair_seed, steps, args.episodes_per_update,
                                                      args.steps - steps, history, actual_schedule)
        if collected == 0:
            raise RuntimeError("Kitchen collector made no progress")
        steps += collected
        episode += episodes
        blocks += (collected + SCENARIO_BLOCK_STEPS - 1) // SCENARIO_BLOCK_STEPS
        if programs:
            event = _extract(model, programs, history, steps)
            if any(value["extracted"] or value["error"] for value in event.values()):
                extraction_events.append({"joint_steps": steps, "roles": event})
        update = _update(model, trajectories, optimizer, programs, steps / args.steps,
                         epochs=args.ppo_epochs, batch_size=args.batch_size)
        for role in feedback:
            feedback[role] += update["feedback_rows"][role]
        if blocks % args.progress_episodes < args.episodes_per_update or steps == args.steps:
            print(json.dumps({"domain": "kitchen", "pair_seed": pair_seed,
                              "variant": variant, "joint_steps": steps,
                              "feedback_rows": feedback}), flush=True)
    return model, programs, {"pair_seed": pair_seed, "variant": variant, "joint_steps": steps,
                             "joint_episodes": episode, "action_overrides": 0,
                             "scenario_blocks": blocks,
                             "scenario_block_steps": SCENARIO_BLOCK_STEPS,
                             "expanded_scenario_seed_sha256": actual_schedule.hexdigest(),
                             "post_policy_action_overrides": 0,
                             "post_policy_override_count": 0,
                             "actor_l2_change": {role: _role_l2(initial, model, role) for role in ("human", "ai")},
                             "feedback_rows": feedback, "extractions": extraction_events,
                             "program_state": {role: item.training_state() for role, item in (programs or {}).items()}}


def _quality(model, start, count):
    return {partner: evaluation(model, start, count, partner)
            for partner in ("nn", "proxy")}


def final_role_program_development_audit(actor, programs, *,
                                         seed_start=DEVELOPMENT_START, seed_count=20):
    """Audit both final role Actors against their programs before held-out use.

    ``programs`` maps role to ``(ExecutableProgram, SHA-256)``.  Both roles are
    checked on the same NN–NN development trajectories and each result binds
    the whole frozen artifact, that role's canonical weights, and its program.
    """
    if set(programs) != {"human", "ai"}:
        raise ValueError("Both Kitchen role programs are required for final audit")
    agreements = {role: [] for role in programs}
    divergences = {role: [] for role in programs}
    for seed in range(seed_start, seed_start + seed_count):
        for task in (1, 2, 3):
            state = e.initial_state(seed, task)
            while not state["terminal"]:
                choices = {role: actor.choose(state, role) for role in programs}
                for role, (program, _) in programs.items():
                    action, scores, mask = choices[role]
                    neural_logits = np.where(mask, scores, -1e9)
                    neural = np.exp(neural_logits - np.max(neural_logits))
                    neural = neural / neural.sum()
                    execution = program.execute(p.program_features(state, role))
                    names = p.HUMAN_ACTIONS if role == "human" else p.AI_ACTIONS
                    symbolic = np.asarray([execution.probabilities[name] for name in names])
                    agreements[role].append(execution.action == action)
                    divergences[role].append(float(np.sum(neural * (
                        np.log(np.clip(neural, 1e-12, None))
                        - np.log(np.clip(symbolic, 1e-12, None))))))
                human_action, ai_action = choices["human"][0], choices["ai"][0]
                state = _transition(state, human_action, ai_action)
    result = {}
    for role, (_, program_digest) in programs.items():
        fidelity = float(np.mean(agreements[role])) if agreements[role] else 0.0
        mean_kl = float(np.mean(divergences[role])) if divergences[role] else float("inf")
        result[role] = {
            "role": role,
            "development_seeds": list(range(seed_start, seed_start + seed_count)),
            "samples": len(agreements[role]), "action_fidelity": fidelity,
            "mean_kl_divergence": mean_kl,
            "actor_sha256": actor.sha256,
            "role_sha256": actor.role_sha256[role],
            "program_sha256": program_digest,
            "passed": fidelity >= .80 and mean_kl <= .35}
    return result


def _passes(rows, seed_count):
    return all(item["successes"] >= int(np.ceil(.8 * 3 * seed_count))
               for item in rows.values())


def _heldout_already_used(root):
    """At-most-once receipt also catches an interrupted held-out evaluation."""
    receipt_path = root / "kitchen_dual_nn_v5_heldout_receipt.json"
    if receipt_path.exists():
        return True
    prior_report = root / "kitchen_dual_nn_v5_pair_report.json"
    if prior_report.exists():
        try:
            if json.loads(prior_report.read_text()).get("heldout"):
                return True
        except (OSError, ValueError):
            return True
    manifest = json.loads((ROOT / "configs/study_v3_kitchen_v5.json").read_text())
    prior_acceptance = ROOT / manifest["policies"]["raw"]["acceptance_path"]
    if prior_acceptance.exists():
        try:
            return bool(json.loads(prior_acceptance.read_text()).get("heldout_evaluated"))
        except (OSError, ValueError):
            return True
    return False


def train_pairs(args):
    torch.set_num_threads(args.threads)
    if len(args.pair_seeds) != 3 or len(set(args.pair_seeds)) != 3:
        raise ValueError("Kitchen v5 requires exactly three distinct paired seeds")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    pairs = []
    for pair_seed in args.pair_seeds:
        random.seed(pair_seed); np.random.seed(pair_seed); torch.manual_seed(pair_seed)
        initial = DualKitchenNetwork(args.hidden)
        teacher, teacher_completed = teacher_data(seed_count=args.teacher_seeds)
        if args.bc_epochs:
            bc_update(initial, teacher, args.bc_epochs, seed=pair_seed)
        pair = {"seed": pair_seed, "bc_teacher_completed": teacher_completed,
                "bc_rows": len(teacher[0]), "variants": {}}
        pair["initial_bc_actor_sha256"] = initial_actor_digest(initial)
        pair["scenario_seed_schedule_sha256"] = scenario_schedule_digest(pair_seed)
        pair["scenario_block_steps"] = SCENARIO_BLOCK_STEPS
        for variant in ("raw", "regularized"):
            model, programs, report = _train_variant(initial, pair_seed, variant, args)
            expected_schedule = scenario_schedule_digest(pair_seed, args.steps)
            if report["expanded_scenario_seed_sha256"] != expected_schedule:
                raise RuntimeError(f"{pair_seed} {variant} did not collect the fixed per-step scenario schedule")
            path = root / f"kitchen_dual_nn_v5_pair{pair_seed}_{variant}.npz"
            report["actor_path"] = str(path)
            report["actor_sha256"] = export(model, path)
            report["role_sha256"] = p.FrozenActors(path).role_sha256
            report["development"] = _quality(model, DEVELOPMENT_START, args.development_seeds)
            report["development_passed"] = _passes(report["development"], args.development_seeds)
            if programs:
                report["programs"] = {}
                for role, rcpd in programs.items():
                    if rcpd.program is not None:
                        program_path = root / f"kitchen_dual_nn_v5_pair{pair_seed}_{variant}_{role}_program.json"
                        rcpd.program.save_json(program_path)
                        report["programs"][role] = {
                            "path": str(program_path), "sha256": sha256(program_path.read_bytes()).hexdigest(),
                            "feedback_eligible": bool(rcpd.last_result and rcpd.last_result.metrics.feedback_eligible),
                            "explanation_eligible": bool(rcpd.last_result and rcpd.last_result.metrics.explanation_eligible)}
                if all(role in report["programs"] for role in ("human", "ai")):
                    final_actor = p.FrozenActors(path, report["actor_sha256"])
                    executable = {role: (ExecutableProgram.load_json(
                        ROOT / report["programs"][role]["path"]), report["programs"][role]["sha256"])
                        for role in ("human", "ai")}
                    report["final_role_program_development_audit"] = final_role_program_development_audit(
                        final_actor, executable)
            pair["variants"][variant] = report
        pair["development_passed"] = all(pair["variants"][v]["development_passed"] for v in ("raw", "regularized"))
        regularized = pair["variants"]["regularized"]
        pair["programs_eligible"] = bool(
            all(regularized.get("programs", {}).get(role, {}).get("feedback_eligible")
                for role in ("human", "ai"))
            and all(regularized.get("final_role_program_development_audit", {}).get(role, {}).get("passed")
                    for role in ("human", "ai")))
        pair["worst_development_successes"] = min(
            result["successes"] for variant in ("raw", "regularized")
            for result in pair["variants"][variant]["development"].values())
        pairs.append(pair)
    eligible = [pair for pair in pairs if pair["development_passed"] and pair["programs_eligible"]]
    selected = max(eligible, key=lambda pair: (pair["worst_development_successes"], -pair["seed"])) if eligible else None
    # Held-out seeds are touched once, only after selection. An incomplete or
    # failed development run writes a report but cannot pin a release model.
    complete = args.steps == JOINT_STEPS and tuple(args.pair_seeds) == PAIR_SEEDS and args.development_seeds == 20 and args.heldout_seeds == 100
    heldout = {}
    heldout_receipt_sha256 = None
    if selected and complete:
        if _heldout_already_used(root):
            raise RuntimeError("Kitchen v5 held-out test has already been used; refusing a second evaluation")
        receipt_path = root / "kitchen_dual_nn_v5_heldout_receipt.json"
        started_receipt = {"status": "started", "selected_pair_seed": selected["seed"],
                           "seed_start": HELDOUT_START, "seed_count": args.heldout_seeds,
                           "actor_sha256": {variant: selected["variants"][variant]["actor_sha256"]
                                            for variant in ("raw", "regularized")}}
        # Exclusive creation is intentional: an interrupted evaluation still
        # burns this one-use test set until a researcher audits its receipt.
        with receipt_path.open("x") as handle:
            json.dump(started_receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        for variant in ("raw", "regularized"):
            artifact = p.FrozenActors(selected["variants"][variant]["actor_path"])
            heldout[variant] = _quality(artifact, HELDOUT_START, args.heldout_seeds)
        heldout_receipt_sha256 = _write_json(receipt_path, {**started_receipt,
                                                            "status": "completed", "heldout": heldout})
    programs_valid = bool(selected and all(
        selected["variants"]["regularized"].get("programs", {}).get(role, {}).get(
            "feedback_eligible") for role in ("human", "ai"))
        and all(selected["variants"]["regularized"].get(
            "final_role_program_development_audit", {}).get(role, {}).get("passed")
                for role in ("human", "ai")))
    feedback_valid = bool(selected and
                          sum(selected["variants"]["raw"]["feedback_rows"].values()) == 0 and
                          all(selected["variants"]["regularized"]["feedback_rows"][role] > 0
                              for role in ("human", "ai")))
    passed = bool(complete and selected and programs_valid and feedback_valid and all(
        _passes(heldout[variant], args.heldout_seeds) for variant in ("raw", "regularized")))
    _write_json(root / "kitchen_dual_nn_v5_pair_report.json",
                {"release_id": RELEASE_ID, "pair_seeds": args.pair_seeds,
                 "joint_steps_per_variant": args.steps,
                 "scenario_block_steps": SCENARIO_BLOCK_STEPS,
                 "scenario_schedule_contract": "fixed_global_joint_step_blocks_with_same_seed_restart_after_early_terminal",
                 "teacher_only_pretraining": True, "ppo_teacher_anchor": 0.0,
                 "regularization_start_fraction": .80, "regularization_ramp_fraction": .05,
                 "regularization_lambda": .01, "pairs": pairs,
                 "selected_pair_seed": selected["seed"] if selected else None,
                 "heldout": heldout, "heldout_receipt_sha256": heldout_receipt_sha256,
                 "passed": passed})
    if not passed:
        failure = {"release_id": RELEASE_ID, "passed": False,
                   "heldout_evaluated": bool(heldout),
                   "selected_pair_seed": selected["seed"] if selected else None,
                   "pair_seeds": args.pair_seeds, "joint_steps_per_variant": args.steps,
                   "reason": "paired development, explanation, feedback, or held-out gate failed",
                   "pair_report_path": str(root / "kitchen_dual_nn_v5_pair_report.json")}
        _write_json(root / "kitchen_dual_nn_v5_failed_acceptance.json", failure)
        return failure
    manifest_path = ROOT / "configs/study_v3_kitchen_v5.json"
    manifest = json.loads(manifest_path.read_text())
    existing_acceptance = ROOT / manifest["policies"]["raw"]["acceptance_path"]
    if existing_acceptance.exists():
        try:
            prior_accepted = bool(json.loads(existing_acceptance.read_text()).get("passed"))
        except (OSError, ValueError):
            prior_accepted = False
        if prior_accepted:
            raise RuntimeError("Refusing to overwrite an already accepted Kitchen v5 release")
    artifact_hashes = {}
    report_hashes = {}
    for variant in ("raw", "regularized"):
        source = selected["variants"][variant]
        entry = manifest["policies"][variant]
        target = ROOT / entry["actor_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source["actor_path"]).read_bytes())
        entry["actor_sha256"] = sha256(target.read_bytes()).hexdigest()
        entry["role_sha256"] = source["role_sha256"]
        artifact_hashes[variant] = entry["actor_sha256"]
        if variant == "regularized":
            for role, destination in zip(("human", "ai"), entry["program_paths"]):
                program = source.get("programs", {}).get(role)
                if program:
                    program_target = ROOT / destination
                    program_target.parent.mkdir(parents=True, exist_ok=True)
                    program_target.write_bytes(Path(program["path"]).read_bytes())
                    entry["program_sha256"][role] = sha256(program_target.read_bytes()).hexdigest()
        training_report = {key: value for key, value in source.items() if key != "program_state"}
        report_hash = _write_json(ROOT / entry["training_report_path"], training_report)
        entry["training_report_sha256"] = report_hash
        report_hashes[variant] = report_hash
    # The independent audit is the only step allowed to open the pilot.
    acceptance = {"release_id": RELEASE_ID, "passed": False,
                  "pre_audit_quality_passed": True,
                  "heldout_evaluated": bool(heldout),
                  "selected_pair_seed": selected["seed"], "pair_seeds": args.pair_seeds,
                  "scenario_block_steps": SCENARIO_BLOCK_STEPS,
                  "expanded_scenario_seed_sha256": selected["scenario_seed_schedule_sha256"],
                  "joint_steps": {variant: selected["variants"][variant]["joint_steps"]
                                  for variant in ("raw", "regularized")},
                  "development": {variant: selected["variants"][variant]["development"]
                                  for variant in ("raw", "regularized")},
                  "heldout": heldout, "artifact_sha256": artifact_hashes,
                  "heldout_receipt_sha256": heldout_receipt_sha256,
                  "training_report_sha256": report_hashes,
                  "program_sha256": {"regularized": manifest["policies"]["regularized"]["program_sha256"]},
                  "final_role_program_development_audit": selected["variants"]["regularized"]["final_role_program_development_audit"],
                  "final_ai_program_development_audit": selected["variants"]["regularized"]["final_role_program_development_audit"]["ai"],
                  "action_overrides": 0, "post_policy_action_overrides": 0,
                  "post_policy_override_count": 0,
                  "program_feedback_rows": selected["variants"]["regularized"]["feedback_rows"],
                  "regularized_program_feedback_rows": selected["variants"]["regularized"]["feedback_rows"],
                  "raw_program_feedback_rows": selected["variants"]["raw"]["feedback_rows"],
                  "regularized_program_feedback_total": sum(selected["variants"]["regularized"]["feedback_rows"].values()),
                  "raw_program_feedback_total": sum(selected["variants"]["raw"]["feedback_rows"].values())}
    _write_json(ROOT / manifest["policies"]["raw"]["acceptance_path"], acceptance)
    _write_json(manifest_path, manifest)
    return acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-seeds", nargs="+", type=int, default=list(PAIR_SEEDS))
    parser.add_argument("--steps", type=int, default=JOINT_STEPS)
    parser.add_argument("--teacher-seeds", type=int, default=300)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ppo-lr", type=float, default=1e-6)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--episodes-per-update", type=int, default=12)
    parser.add_argument("--progress-episodes", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--development-seeds", type=int, default=20)
    parser.add_argument("--heldout-seeds", type=int, default=100)
    parser.add_argument("--output-dir", default="output/kitchen_v5_matched_candidates")
    args = parser.parse_args()
    print(json.dumps(train_pairs(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
