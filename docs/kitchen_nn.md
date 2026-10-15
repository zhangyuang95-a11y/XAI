# Kitchen dual neural policy

The physical Kitchen engine remains the v6.3 map, recipes, timers, scoring and
simultaneous-turn transition. The participant-facing Kitchen adapter is
`kitchen-v7.0.0-neural`: it loads two trained Actors, runs only the frozen AI
Actor during human play, and rejects a missing or mismatched artifact. The
human continues to use the six existing keyboard actions.

## Action contract

The AI Actor scores 13 fixed action IDs: four moves, wait, generic interact,
two ingredient-slot interactions and five order-specific pan interactions.
The legal mask comes solely from the physical state. When a prepared portion
is loaded into a pan, the selected order ID is the Actor's choice; the engine
checks recipe, freshness, pan phase and order status. A raw-counter action
likewise names its slot. The legacy controller remains available only for
offline pretraining and historical engine regression tests.
The former promise to wait exactly two turns before disposing of a blocked
output is omitted from the new public rules; the Actor chooses how to respond.

`neural_domain.decide` records the action ID, legal mask, full Actor scores,
model version and SHA-256. `neural_domain.step` recomputes the decision from
the saved pre-action state and rejects a replacement decision. Saved frames
therefore replay the same frozen action; Q&A branches call the same adapter.
Explanations report the model's selected legal action and physical facts, not
a hand-authored claim about the network's motives.

## Training and acceptance

The training command uses two separate Actors and a centralized Critic.
Behavior cloning starts both Actors from legacy cooperative traces generated
with the *new explicit-order transition*. DAgger can add NN-visited states;
joint PPO samples both actions from the current Actors. Training-only teacher
labels and collision recovery do not run in participant execution.

The accepted candidate was trained in two stages. The first stage learns from
300 teacher seeds; the second stage starts from that snapshot and updates both
Actors through 7,271 joint PPO steps. The `--output` paths below are candidate
files, not deployment paths.

```sh
python3 -m domains.kitchen.training train --output output/kitchen_bc_candidate.npz --hidden 256 --teacher-seeds 300 --bc-epochs 15 --dagger-rounds 0 --ppo-steps 0 --tuning-seeds 20
python3 -m domains.kitchen.training train --output output/kitchen_joint_candidate.npz --resume-model output/kitchen_bc_candidate.npz --hidden 256 --teacher-seeds 20 --bc-epochs 0 --dagger-rounds 0 --ppo-episodes 12 --ppo-steps 5000 --ppo-lr 0.000001 --ppo-anchor 0.1 --tuning-seeds 20
python3 -m domains.kitchen.training evaluate --model output/kitchen_joint_candidate.npz --seed-start 30000 --seed-count 100
```

The seed ranges are disjoint: teacher 10000+, DAgger 15000+, optional partner
pretraining 16000+, joint PPO 17000+, tuning 20000–20019, and final test
30000–30099. Each final seed runs all three tasks with NN–NN and frozen NN
plus the offline human proxy. Each pairing must finish all five dishes in at
least 240 of 300 episodes before the artifact is configured for participant
tasks. Proxy results are software tests and make no claim about real people.

The pinned `kitchen-dual-nn-v1` artifact passed: 60/60 on each tuning pairing
and 300/300 on each final pairing. The deployment SHA-256 and exact counts are
recorded in `output/deployment/kitchen_dual_nn_v1_training.json`. These are
synthetic software results; performance with real participants remains to be
measured.

The new Kitchen version starts a separate enrollment. Earlier Kitchen runs
retain their recorded release and public frames, which remain available as
read-only replay. The current Actor cannot continue an earlier run.
