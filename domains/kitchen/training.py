"""Two-Actor Kitchen BC warm start followed by genuine joint MAPPO updates.

No rule controller supplies actions during self-play. Teacher functions are used
only to create the warm-start data (and the separate evaluation proxy).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from . import engine_v5_frozen as e
from . import neural_policy as p


class DualKitchenNetwork(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        dimension = len(p.features(e.initial_state(1000, 1)))
        self.human = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, len(p.HUMAN_ACTIONS)))
        self.ai = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, len(p.AI_ACTIONS)))
        self.critic = nn.Sequential(nn.Linear(dimension, hidden * 2), nn.Tanh(),
                                    nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def logits(self, x, role, mask):
        values = (self.human if role == "human" else self.ai)(x)
        return values.masked_fill(~mask, -1e9)


def _teacher_ai_action(state, decision):
    if decision["action"] != "interact":
        return decision["action"]
    station = e._front(state["ai"])
    if station and station["id"] == "ai_raw":
        slot = decision.get("slot")
        if slot is None:
            slot = e._descriptor(state, "ai")["slot"]
        return f"interact_slot{slot}"
    held = state["ai"]["holding"]
    if station and station["kind"] == "pot" and held and held["stage"] == "prepared":
        order = e._order(state, held.get("order_id")) if held.get("order_id") else e._choose_order(state, held)
        if order is None:
            raise ValueError("Teacher loaded food without a current order")
        return "interact_" + order["id"]
    return "interact"


def _transition(state, human_id, ai_id):
    decision = {**p.action_spec(ai_id), "policy_origin": "kitchen_nn", "memory": {},
                "reason_code": "neural_training"}
    return e.step(state, human_id, decision)


def teacher_data(seed_start=10000, seed_count=80):
    states, human_masks, ai_masks, human_labels, ai_labels = [], [], [], [], []
    completed = 0
    for seed in range(seed_start, seed_start + seed_count):
        for task in (1, 2, 3):
            state = e.initial_state(seed, task)
            while not state["terminal"]:
                human = e.simulation_partner(state)
                decision = e.decide(state)
                ai = _teacher_ai_action(state, decision)
                human_mask, ai_mask = p.legal_mask(state, "human"), p.legal_mask(state, "ai")
                hi, ai_index = p.HUMAN_ACTIONS.index(human), p.AI_ACTIONS.index(ai)
                if not human_mask[hi] or not ai_mask[ai_index]:
                    raise AssertionError(f"Teacher emitted an action outside the NN action mask: {human}, {ai}")
                states.append(p.features(state))
                human_masks.append(human_mask)
                ai_masks.append(ai_mask)
                human_labels.append(hi)
                ai_labels.append(ai_index)
                # Expert labels are collected on the *neural* transition
                # semantics: pickup never silently binds an order.
                state = _transition(state, human, ai)
            completed += state["metrics"]["completed_orders"] == 5
    return (np.stack(states), np.stack(human_masks), np.stack(ai_masks),
            np.asarray(human_labels), np.asarray(ai_labels)), completed


def bc_update(model, data, epochs=12, batch_size=1024, lr=1e-3, seed=5, role_weights=(1.0, 1.0)):
    torch.manual_seed(seed)
    parameters = (tuple(model.human.parameters()) if role_weights[0] else ()) + (tuple(model.ai.parameters()) if role_weights[1] else ())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    x, hm, am, hy, ay = [torch.as_tensor(part) for part in data]
    n = len(x)
    for epoch in range(epochs):
        order = torch.randperm(n)
        losses = []
        for ids in order.split(batch_size):
            logits_h = model.logits(x[ids], "human", hm[ids])
            logits_a = model.logits(x[ids], "ai", am[ids])
            loss = (role_weights[0] * nn.functional.cross_entropy(logits_h, hy[ids])
                    + role_weights[1] * nn.functional.cross_entropy(logits_a, ay[ids]))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        print(json.dumps({"stage": "bc", "epoch": epoch + 1, "loss": float(np.mean(losses))}), flush=True)


def select(model, state, role, stochastic=False):
    if isinstance(model, p.FrozenActors):
        if stochastic:
            raise ValueError("Frozen evaluation uses deterministic Actor inference")
        return model.choose(state, role)[0]
    mask = p.legal_mask(state, role)
    x = torch.as_tensor(p.features(state)[None, :])
    mt = torch.as_tensor(mask[None, :])
    with torch.no_grad():
        logits = model.logits(x, role, mt)[0]
        index = int(Categorical(logits=logits).sample().item()) if stochastic else int(torch.argmax(logits).item())
    actions = p.HUMAN_ACTIONS if role == "human" else p.AI_ACTIONS
    return actions[index]


def rollout(model, seed, task, partner="nn", stochastic=False, capture=False):
    state = e.initial_state(seed, task)
    trace = []
    while not state["terminal"]:
        human = select(model, state, "human", stochastic) if partner == "nn" else e.simulation_partner(state)
        ai = select(model, state, "ai", stochastic)
        if capture:
            teacher_human = e.simulation_partner(state)
            teacher_ai = _teacher_ai_action(state, e.decide(state))
            # Offline recovery examples teach the actors to yield at the
            # shared counter. No such rule runs in participant execution.
            if (teacher_human == "interact" and teacher_ai == "interact"
                    and (e._front(state["human"]) or {}).get("id") == "handoff"
                    and (e._front(state["ai"]) or {}).get("id") == "handoff"):
                teacher_ai = "wait"
            trace.append((p.features(state), p.legal_mask(state, "human"), p.legal_mask(state, "ai"),
                          p.HUMAN_ACTIONS.index(teacher_human), p.AI_ACTIONS.index(teacher_ai)))
        state = _transition(state, human, ai)
    return state, trace


def partner_data(model, seed_start, episodes):
    """Label the human role on successful frozen-AI + offline-proxy states."""
    rows, completed = [], 0
    for episode in range(episodes):
        state = e.initial_state(seed_start + episode, 1 + episode % 3)
        while not state["terminal"]:
            human = e.simulation_partner(state)
            ai = select(model, state, "ai")
            rows.append((p.features(state), p.legal_mask(state, "human"), p.legal_mask(state, "ai"),
                         p.HUMAN_ACTIONS.index(human), p.AI_ACTIONS.index(ai)))
            state = _transition(state, human, ai)
        completed += state["metrics"]["completed_orders"] == 5
    columns = list(zip(*rows))
    result = [np.stack(columns[i]) if i < 3 else np.asarray(columns[i]) for i in range(5)]
    return result, completed


def evaluation(model, seed_start, seed_count, partner):
    success = 0
    scores = []
    failures = []
    for seed in range(seed_start, seed_start + seed_count):
        for task in (1, 2, 3):
            state, _ = rollout(model, seed, task, partner)
            ok = state["metrics"]["completed_orders"] == 5
            success += ok
            scores.append(state["raw_score"])
            if not ok:
                failures.append({"seed": seed, "task": task, "completed": state["metrics"]["completed_orders"],
                                 "turn": state["turn"], "metrics": state["metrics"]})
    return {"partner": partner, "seeds": list(range(seed_start, seed_start + seed_count)),
            "episodes": seed_count * 3, "successes": success, "completion_rate": success / (seed_count * 3),
            "mean_score": float(np.mean(scores)), "failures": failures}


def collect_joint(model, start_seed, episodes=12):
    """On-policy trajectories. Both actions come from the current Actors."""
    trajectories = []
    for episode in range(episodes):
        state = e.initial_state(start_seed + episode, 1 + episode % 3)
        rows = []
        while not state["terminal"]:
            vector = p.features(state)
            hm, am = p.legal_mask(state, "human"), p.legal_mask(state, "ai")
            x = torch.as_tensor(vector[None, :])
            with torch.no_grad():
                hd = Categorical(logits=model.logits(x, "human", torch.as_tensor(hm[None, :])))
                ad = Categorical(logits=model.logits(x, "ai", torch.as_tensor(am[None, :])))
                hi, ai = hd.sample(), ad.sample()
                old_logp = float((hd.log_prob(hi) + ad.log_prob(ai)).item())
                value = float(model.critic(x).item())
            before_score = state["raw_score"]
            state = _transition(state, p.HUMAN_ACTIONS[int(hi)], p.AI_ACTIONS[int(ai)])
            reward = float(state["raw_score"] - before_score)
            rows.append((vector, hm, am, int(hi), int(ai), old_logp, value, reward))
        trajectories.append(rows)
    return trajectories


def ppo_update(model, trajectories, optimizer, epochs=3, batch_size=1024,
               anchor_data=None, anchor_weight=0.0):
    flattened = []
    for episode in trajectories:
        gae, following = 0.0, 0.0
        processed = []
        for row in reversed(episode):
            delta = row[7] + 0.99 * following - row[6]
            gae = delta + 0.99 * 0.95 * gae
            processed.append((*row[:7], gae, gae + row[6]))
            following = row[6]
        flattened.extend(reversed(processed))
    if not flattened:
        return 0
    arrays = list(zip(*flattened))
    x = torch.as_tensor(np.stack(arrays[0]))
    hm, am = torch.as_tensor(np.stack(arrays[1])), torch.as_tensor(np.stack(arrays[2]))
    hi, ai = torch.as_tensor(arrays[3]), torch.as_tensor(arrays[4])
    old = torch.as_tensor(arrays[5], dtype=torch.float32)
    advantage = torch.as_tensor(arrays[7], dtype=torch.float32)
    returns = torch.as_tensor(arrays[8], dtype=torch.float32)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    for _ in range(epochs):
        for ids in torch.randperm(len(x)).split(batch_size):
            hd = Categorical(logits=model.logits(x[ids], "human", hm[ids]))
            ad = Categorical(logits=model.logits(x[ids], "ai", am[ids]))
            logp = hd.log_prob(hi[ids]) + ad.log_prob(ai[ids])
            ratio = torch.exp(logp - old[ids])
            policy_loss = -torch.minimum(ratio * advantage[ids], torch.clamp(ratio, .8, 1.2) * advantage[ids]).mean()
            value_loss = .5 * nn.functional.mse_loss(model.critic(x[ids]).squeeze(-1), returns[ids])
            entropy = hd.entropy().mean() + ad.entropy().mean()
            loss = policy_loss + .5 * value_loss - .002 * entropy
            if anchor_data is not None and anchor_weight:
                anchor_ids = torch.randint(len(anchor_data[0]), (min(len(ids), 512),))
                anchor_x = torch.as_tensor(anchor_data[0][anchor_ids])
                anchor_hm = torch.as_tensor(anchor_data[1][anchor_ids])
                anchor_am = torch.as_tensor(anchor_data[2][anchor_ids])
                anchor_hy = torch.as_tensor(anchor_data[3][anchor_ids])
                anchor_ay = torch.as_tensor(anchor_data[4][anchor_ids])
                anchor_loss = (nn.functional.cross_entropy(model.logits(anchor_x, "human", anchor_hm), anchor_hy)
                               + nn.functional.cross_entropy(model.logits(anchor_x, "ai", anchor_am), anchor_ay))
                loss = loss + anchor_weight * anchor_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return len(x)


def export(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata_json": np.asarray(json.dumps({"model_version": p.MODEL_VERSION,
               "feature_version": p.FEATURE_VERSION, "observation_dim": len(p.features(e.initial_state(1000, 1))),
               "human_actions": p.HUMAN_ACTIONS, "ai_actions": p.AI_ACTIONS, "hidden": model.hidden}))}
    for role in ("human", "ai"):
        actor = getattr(model, role)
        for layer, module_index in enumerate((0, 2, 4)):
            module = actor[module_index]
            payload[f"{role}.weight{layer}"] = module.weight.detach().cpu().numpy()
            payload[f"{role}.bias{layer}"] = module.bias.detach().cpu().numpy()
    np.savez_compressed(path, **payload)
    return sha256(path.read_bytes()).hexdigest()


def import_actors(model, path):
    artifact = p.FrozenActors(path)
    if artifact.metadata["hidden"] != model.hidden:
        raise ValueError("Kitchen checkpoint hidden width mismatch")
    with torch.no_grad():
        for role in ("human", "ai"):
            actor = getattr(model, role)
            for layer, module_index in enumerate((0, 2, 4)):
                actor[module_index].weight.copy_(torch.as_tensor(artifact.weights[f"{role}.weight{layer}"].copy()))
                actor[module_index].bias.copy_(torch.as_tensor(artifact.weights[f"{role}.bias{layer}"].copy()))


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model = DualKitchenNetwork(args.hidden)
    if args.resume_model:
        import_actors(model, args.resume_model)
    data, teacher_completed = teacher_data(seed_count=args.teacher_seeds)
    print(json.dumps({"stage": "teacher", "samples": len(data[0]), "completed": teacher_completed}), flush=True)
    if args.bc_epochs:
        bc_update(model, data, args.bc_epochs, seed=args.seed)
    for round_index in range(args.dagger_rounds):
        extra = []
        for episode in range(args.dagger_episodes):
            _, rows = rollout(model, args.dagger_seed_start + round_index * args.dagger_episodes + episode,
                              1 + episode % 3, capture=True)
            extra.extend(rows)
        if extra:
            columns = list(zip(*extra))
            more = [np.stack(columns[i]) if i < 3 else np.asarray(columns[i]) for i in range(5)]
            data = tuple(np.concatenate((old, new)) for old, new in zip(data, more))
            print(json.dumps({"stage": "dagger", "round": round_index + 1, "total_samples": len(data[0])}), flush=True)
            bc_update(model, data, args.dagger_epochs, lr=3e-4, seed=args.seed + round_index + 1)
    if args.partner_episodes:
        partner, completed = partner_data(model, 16000, args.partner_episodes)
        data = tuple(np.concatenate((old, new)) for old, new in zip(data, partner))
        print(json.dumps({"stage": "partner_pretraining", "episodes": args.partner_episodes,
                          "completed": completed, "total_samples": len(data[0])}), flush=True)
        bc_update(model, data, args.partner_epochs, lr=2e-4, seed=args.seed + 100, role_weights=(1.0, 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.ppo_lr)
    joint_steps = 0
    iteration = 0
    while joint_steps < args.ppo_steps:
        trajectories = collect_joint(model, 17000 + iteration * args.ppo_episodes, args.ppo_episodes)
        joint_steps += ppo_update(model, trajectories, optimizer, anchor_data=data,
                                  anchor_weight=args.ppo_anchor)
        iteration += 1
        print(json.dumps({"stage": "joint_ppo", "iteration": iteration, "joint_steps": joint_steps,
                          "episodes": iteration * args.ppo_episodes}), flush=True)
    digest = export(model, args.output)
    print(json.dumps({"stage": "export", "path": str(args.output), "sha256": digest}), flush=True)
    for partner in ("nn", "proxy"):
        print(json.dumps({"stage": "tuning", **evaluation(model, 20000, args.tuning_seeds, partner)}), flush=True)


def evaluate_file(args):
    artifact = p.FrozenActors(args.model)
    for partner in ("nn", "proxy"):
        result = evaluation(artifact, args.seed_start, args.seed_count, partner)
        result.update(model_sha256=artifact.sha256, passed=result["completion_rate"] >= .80)
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser("train")
    trainer.add_argument("--output", default="output/kitchen_candidate.npz")
    trainer.add_argument("--seed", type=int, default=7)
    trainer.add_argument("--threads", type=int, default=4)
    trainer.add_argument("--hidden", type=int, default=256)
    trainer.add_argument("--resume-model")
    trainer.add_argument("--teacher-seeds", type=int, default=300)
    trainer.add_argument("--bc-epochs", type=int, default=15)
    trainer.add_argument("--dagger-episodes", type=int, default=24)
    trainer.add_argument("--dagger-rounds", type=int, default=0)
    trainer.add_argument("--dagger-seed-start", type=int, default=15000)
    trainer.add_argument("--dagger-epochs", type=int, default=5)
    trainer.add_argument("--partner-episodes", type=int, default=0)
    trainer.add_argument("--partner-epochs", type=int, default=5)
    trainer.add_argument("--ppo-episodes", type=int, default=12)
    trainer.add_argument("--ppo-steps", type=int, default=5000)
    trainer.add_argument("--ppo-lr", type=float, default=1e-6)
    trainer.add_argument("--ppo-anchor", type=float, default=0.1)
    trainer.add_argument("--tuning-seeds", type=int, default=20)
    tester = sub.add_parser("evaluate")
    tester.add_argument("--model", default="output/deployment/kitchen_dual_nn_v1.npz")
    tester.add_argument("--seed-start", type=int, default=30000)
    tester.add_argument("--seed-count", type=int, default=100)
    args = parser.parse_args()
    train(args) if args.command == "train" else evaluate_file(args)


if __name__ == "__main__":
    main()
