# Pong neural-policy release

## v5 paired-policy protocol

The v5 study binds each instance to `pong-regularized-v5` (A/B) or
`pong-raw-v5` (C). Both variants contain separate human-role and AI-role Actor
weights. The demonstration uses both bound Actors; during formal play the
participant supplies the human-role action and the AI-role Actor supplies the
other action. `step_pair` remains the physical training transition. The only
runtime action mask excludes moves beyond the nine-lane court. No extracted
program or planner supplies an action.

Run `python3 scripts/train_rolling_pong_v5.py` to reproduce three paired
random seeds, 200,000 joint self-play steps per variant, and selection on the
fresh 24-seed-per-task development split. Each pair starts from the same
dual-Actor BC checkpoint, uses the same scenario seed schedule, and differs
only in late-training RCPD feedback. Role-specific programs use named visible
pre-action features, depth at most five, and at most 16 leaves. The feedback
weight starts at 80% of training, ramps to 0.01 by 85%, and applies only when
validation permits feedback and the program agrees with the current NN action.
Raw has zero program feedback.

The first development-only pilot used feedback fidelity ≥0.80 and mean KL
≤0.20. Its bounded programs reached 0.764/0.510 (human role) and 0.918/0.232
(AI role), so the feedback gate yielded zero rows. Before looking at the new
held-out set, the feedback gate was revised to fidelity ≥0.75 and KL ≤0.60;
the extraction-time explanation gate is fidelity ≥0.90 and KL ≤0.25. The
independent explanation audit against the selected frozen Actor uses fidelity
≥0.95 and KL ≤0.10 on development states. Extraction-time fidelity is
also reported; the selected-Actor audit determines whether the program can
support a local explanation, which additionally requires action agreement in
that state. Both final role programs must pass a frozen-Actor development
audit before held-out evaluation: the human-role training-only program uses
fidelity ≥0.75 and KL ≤0.60, while the AI-role explanation program uses the
stricter threshold above. If the human-role training-time program misses this
final audit, it may be re-extracted from the selected frozen human Actor on
separate training seeds 870100–870131 and validated on seeds 870200–870207.
This later tree never supplies a training gradient or runtime action; the
report distinguishes its provenance from the earlier audited trees that
actually supplied RCPD feedback. Only the AI-role program can support A's
explanation. These choices are
development tuning, recorded in the v5 training reports and acceptance
receipt, not a pre-registered threshold. All three paired seeds are rerun
from their common initial checkpoint after the change.

Only a pair whose two variants pass development gates and whose two final
programs pass the frozen-Actor audits is eligible for the
single 100-seed-per-task held-out evaluation. The domain-level v5 acceptance
receipt records the three pair seeds, selected pair, exact step budgets,
Actor/program/report hashes, runtime parity, feedback, and both held-out
results. A failed or incomplete receipt keeps v5 pilot enrollment closed.
Preview and tests can inspect a hash-pinned candidate without claiming
acceptance. The old v4 actor and evaluation below remain historical evidence.

The completed v5 receipt selected pair seed `251016`. Both role programs passed
the frozen-Actor development audit: human action fidelity 0.849 and mean KL
0.370, AI fidelity 0.981 and mean KL 0.094. The selected programs both came
from training-time extraction. On the once-evaluated 100-seed-per-task held-out
set, raw and regularized both passed: their task-wise mean contact capacity was
91.91–92.18% and 91.70–92.01%, respectively; every episode was at least
83.75%. The largest AI wait rate was 11.11% for either variant. These are
NN–NN measurements, not effects measured with human participants.

The current three-domain Pong uses `pong-concurrent.v7.0` physics and the
unchanged `pong-concurrent-schedule.v5.0` scenarios. Each paddle moves at most
one lane per submitted turn. The same visible balls, contact rules, score,
Task 1/2/3 lengths, and fixed development/held-out seeds remain in place.

## Controller boundary

`step_pair(state, human_action, ai_action)` is the training transition. Both
actions come from separate neural actors during reinforcement learning. It does
not call the production controller. `decide(state)` is the production path: it
loads the frozen AI actor, encodes only visible balls and paddle positions,
masks boundary moves, and chooses the highest-scoring legal action. The human
actor is used for offline evaluation, the demonstration, and optional advice;
participant input always supplies the actual human action.

The actor file and acceptance receipt in `domains/pong/` are immutable release
assets. The receipt must match the model SHA-256 and state that both fixed
development and held-out gates passed. Missing or mismatched assets stop the
Pong service before a new run can begin. The release manifest includes both
assets in its source hash and publishes the actor hash.

The model has no access to the seed, unseen schedule, experimental group,
question text, or the human's unsubmitted action. Its inputs include normalized
positions and a fixed-length, sorted representation of at most ten visible
balls, with contact geometry and remaining turns. The only runtime action
constraint is staying inside the nine-lane court. The old planner is never a
runtime fallback.

## Reproduce training and evaluation

From this repository root, install `requirements-pong-training.txt` and run:

```bash
python3 scripts/train_rolling_pong_nn.py --clone-epochs 120 --dagger-rounds 2 --self-play-rounds 8 --train-seeds 120
```

Training first clones labels from the old visible-state planner on 120 training
seeds. It then runs neural-versus-neural episodes in the real engine on training
seeds and updates both actors with team-point PPO. Only checkpoints after a
dual-NN self-play update are eligible for release. Model selection uses only the
24 fixed development seeds per task; none enter the gradient updates. The
selected checkpoint is evaluated once on the 24 fixed held-out seeds per task.
The full per-run results are written
to `output/pong_rolling_nn/evaluation.json` and a matching release evidence file
in `domains/pong/`; the PyTorch checkpoint remains in `output/` and is not
needed by the production server. The exported actor uses
pure Python inference, so the server does not need PyTorch.

For each task and split, the acceptance gates are mean earned points divided
by the offline contact-capacity upper bound at least 90%, every run at least
80%, mean AI wait rate at most 10%, and every run at most 20%. The capacity
bound ignores travel and is not a human-achievable score guarantee. Human
replacement proxies are separate simulations and do not establish an effect
of explanations on real participants.

An exported model that misses any gate has no matching acceptance receipt and
cannot start a new Pong study run. Previous Pong records remain exportable
under their original release IDs; they cannot continue into the new release.

The accepted self-play-updated actor has SHA-256
`c9e1d27ded3a8ec735133f6299171d836553ee2b8c90eb01fd24cfd38287c8f8`.
Across the 24 held-out seeds per task, its mean/minimum capacity percentages
are 90.83/87.50 (Task 1), 91.67/86.67 (Task 2), and 91.46/87.50 (Task 3).
Mean AI wait rates are below 1% in all three tasks. The exported pure-Python
actor matches the evaluated PyTorch checkpoint on every fixed development and
held-out rollout. Full per-run scores and simulated human-proxy results are in
the checked-in evaluation evidence.

Mean held-out task scores (0–100 scale) with the same frozen AI Actor are:

| Other paddle | Task 1 | Task 2 | Task 3 |
| --- | ---: | ---: | ---: |
| Trained human-role NN | 60.56 | 61.11 | 60.97 |
| Always wait proxy | 38.92 | 37.92 | 36.50 |
| Earliest-ball greedy proxy | 48.47 | 48.98 | 41.30 |
| History-based proxy | 41.77 | 41.64 | 40.21 |
| Random legal-action proxy | 29.97 | 31.32 | 32.29 |

The proxy rows measure simulated behavior, not participant outcomes.

## Explanation boundary

The Task 2 evidence reports the frozen actor's saved action, visible contact
geometry and actual observed outcomes. It does not state that the neural model
committed to a particular ball or infer an internal reason from a planner.
Counterfactuals rerun that same frozen actor from copied visible states and
stop at the first newly visible ball boundary.
