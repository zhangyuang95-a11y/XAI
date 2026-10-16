# Warehouse v6.11 design review after the 12069 BC screen

Status: design only. No v6.11 seed has been reset, no PPO result exists for the
12069 Actor, and this note is not a model admission receipt. The immutable
v6.10 source and outputs are under
`output/warehouse_v6_repro_snapshots/warehouse-v610-core-12069-20261016-v2/`.
The v6.10 first-quality result is tied to source-closure SHA
`89c19e10fcfd0ca67552e49942b13bef359efd9edafc408b0519b01d4f962b7b`
and frozen Actor SHA
`4c04d33b6907cd04d596e29dd7fa39c857c6b5851a5762d0580f3376783a2843`.

## What failed, and what the evidence does not show

The 96 consumed BC-screen episodes delivered 11.573 items on average and had
one shutdown, but had 739 collisions (7.698 per episode, above the exploratory
screen's six). Task 1 had 333 collisions in 32 episodes (10.406 per episode).
Removing the worst 100-collision episode still leaves 639/95 = 6.726.
Seventeen episodes with at least ten collisions account for 430/739 (58.2%) of
all collisions; 15 episodes with a same-pair collision streak of at least four
account for 371/739 (50.2%). The worst episode, scenario seed 12001181, made
one delivery and collided for a 100-frame same-pair/action streak. These
figures come from `first-quality-result.json` and its unchanged 96 episode
rows; they are not post-PPO acceptance estimates.

An independent frame replay communicated by the root investigation found
that during frames 5–104 of that episode, robot 1 and robot 2 stayed in core
leaves 4 and 10, selecting RIGHT and UP toward the same cell with mean action
probabilities approximately 0.9975 and 0.9913. Their tree margins were about
6.16 and 4.76, while residual bounds were about 0.086 and 0.083. The public
collision-streak feature changed, then saturated, without changing the route.
The root's counterfactual calculation raised the residual bound to 8.05 and
still found 94/100 same-target decisions. The independent
[frame and counterfactual diagnosis](warehouse-v610-core-12069-root-cause.md)
is now linked in the v6.10 failure receipt. Its raw JSON and deterministic
reproducer are preserved with separate SHA manifests in the ignored run
snapshot and external recovery archive. The original quality receipt alone
proves the 100-collision streak, not these logit details.

The public lookahead teacher's four BC collection stages had about 17.7
deliveries, zero collisions and zero shutdowns, while the final NN assigned
only 0.699 mean joint probability to its public teacher actions over the
67,106 final BC rows. Sampled deep visits had 5.98, 10.12, 10.98 and 6.24
mean collisions across four rounds; argmax deep visits had 21.26, 25.74,
29.90 and 14.46. The final deep training mix recorded only 3.39% rows with a
collision increment. Repeated collision states are intentionally thinned
(`BC_DUPLICATE_COLLISION_CAP=8`), and the topology was fitted before any NN
visits, from collision-free public teacher trajectories. This supports a
specific failure hypothesis: a leaf that appears stable on teacher data
aliases post-collision corridor states, while BC margin training makes its
action nearly deterministic. At probability 0.9975, the selected-action
log-probability gradient for its own logit is only 0.0025. A 200k-step PPO
trial from this failed BC screen might learn, but there is no evidence that it
will escape this attractor; consuming new training seeds to test it first is
less informative than rebuilding the topology and BC distribution.

## v6.11 candidate method

1. Keep two independent role Actors, each with a sparse trainable-logit tree,
   learned gate, bounded 256-wide residual, and no execution-time program or
   controller. Raw and regularized must start from the exact same dual Actor
   BC tensor/hash and receive the same scenario schedule. Physical legality,
   collisions and scoring stay in the environment. The PPO shutdown cost
   critic must still train from complete, real 0/1 shutdown episodes and only
   change NN weights.
2. Move *topology fit* after an initial public-teacher warm-up and NN
   visitation phase. During warm-up the same Actor architecture runs with no
   core leaves, so its residual can explore. Collect both sampled and argmax
   dual-NN visits and label their **pre-action public observations** with the
   same public lookahead teacher. Record real collision outcomes separately;
   they may define training strata and sample weights, never execution
   actions or hidden teacher inputs. Fit candidate sparse predicates on a
   train-only mix of teacher and visited episodes, explicitly including
   before/after-collision, same-target corridor, low-return-margin and
   ordinary traffic states. Use episode-balanced strata so 100 identical
   frames cannot dominate, but do not discard nearly all crash contexts.
3. Split the topology data by whole scenario episode before fitting. Use
   disjoint topology fit, leaf-logit calibration, core-leaf eligibility, and
   internal-validation episodes. A core leaf is allowed only if its raw top
   action is provably legal from its public mask path and its masked top
   agreement, KL and action margin pass in **every represented interaction
   stratum** on eligibility episodes. A leaf that mixes ordinary and
   post-collision decisions becomes complex, with full residual capacity;
   the model must learn its action. Validate the frozen topology once on the
   internal episode split, then freeze predicates. Preserve at most three
   public features per predicate, depth five and 16 leaves. If safe core
   occupancy becomes too small, refit only on a new candidate's unspent
   development training split, without changing the evaluation gate.
4. Train the final BC Actor on fit-only teacher and visitation states, with
   explicit episode/stratum balancing of collision and normal contexts.
   Replace the unbounded one-hot margin growth in core leaves with a
   predeclared masked soft-target or confidence ceiling. Preserve a core
   top-two margin above twice the residual cap (0.24 at cap 0.12) for the
   legality/top-action proof, while checking that the resulting sampled
   policy has meaningful exploration in collision-prone states. Calibrate
   this with development training data only. Do not infer safety from a
   high tree/NN agreement on the same training rows. The warm-up, topology,
   later BC, critic, and PPO source/recipe/seeds must be locked and hashed.
5. Run paired raw/regularized PPO only after the new BC source and topology
   checks, at the smallest of 200k/300k/400k steps that passes the full
   development precheck. PPO's collision reward and real shutdown cost
   critic provide outcome learning; the sparse program supplies only
   independently validated regularized gradients after 80% progress with
   lambda 0.01 over a 5% ramp. Raw program feedback remains exactly zero.

The 12069 v6.10 snapshot's `_fit_training_programs` and
`_fresh_final_programs` called `fit_program` without `method`, so the snapshot
used the default `axis_tree_v1`. Live commit `70daad4` already fixes the
production path: the immutable run spec selects
`actor_sparse_topology_v1`, online and fresh final fits pass it explicitly,
and checkpoints/reports bind it. v6.11 must inherit that fix and test both
call sites; the historical v6.10 snapshot must remain unchanged. The program
must still fit the **complete masked NN distribution** on separate fit,
calibration, eligibility and validation episodes, with illegal raw actions
excluded and no runtime action authority.

## Required diagnostics and decision gates

Before any v6.11 reset, allocate and hash a new candidate ID, source closure,
BC recipe, scenario namespace, exploratory seeds and whole-episode split
schedule. Do not reuse 12069's 96 quality episodes, reserved pressure/program
blocks, registered formal seeds, independent audit, or heldout seeds as new
method-selection evidence. `first-quality-result.json` and the BC receipt
may be used solely for the failure analysis above.

For each new frozen BC and later PPO checkpoint, log per task and per episode:
deliveries, collisions, shutdown, repeated same-pair/action collision length,
collision kind, interaction stratum, and late collisions. For each role and
leaf, log on-policy occupancy, raw legality, masked NN/tree top agreement,
KL, top margin, selected-action probability, gate, residual bound, teacher
top accuracy, and collision vs non-collision occupancy. Aggregate by episode
as well as by frame; show the highest collision tails rather than only the
mean. Confirm every action equals that Actor's masked NN sample and record
zero post-policy overrides. The old 12001181 state pattern must route to a
complex leaf or have an independently verified changed NN action; this is a
diagnostic on an already-consumed seed, not an acceptance test.

The independent development stress set must contain multiple trajectories
per task/action stream and pass **both** raw and regularized quality:
deliveries >=5.5, collisions <=6, shutdown episode rate <=5%. The fresh
four-way program check must pass **each role**: feedback and actionable AI
explanation coverage >=30%, agreements >=93% and >=95%, masked KL <=0.25,
illegal program actions zero, and nonzero program-gradient norm in both
regularized Actors. Count *actionable sampled-NN rows*, not merely leaf
occupancy. If these checks fail, retain the receipts and revise a newly
versioned development candidate. The formal 6101–6103 pairs, one-time
independent audit, per-task 100-seed heldout test, and v6 publication remain
unopened until the locked development method passes. Formal criteria remain
deliveries >=5, collisions <=8, shutdown <=10%, program coverage >=20%,
feedback agreement >=85%, explanation agreement >=90%, KL <=0.35, illegal
actions zero. No threshold or runtime action rule changes are proposed.
