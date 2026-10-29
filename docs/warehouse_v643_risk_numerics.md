# v6.43 training-only risk numerical kernel

This implements one component of the proposed `warehouse-v643-risk-calibration-01` experiment. It does not authorize a run, create an environment or Actor, claim a seed, perform a new Warehouse evaluation, or change the website or release status. The archived v6.42 calibration still fails because raw shutdown was 5/75; no old result or receipt is amended.

## Implemented behavior

`backend/training/warehouse_v643_risk_training.py` exports `RiskTrainingConfig`, `RiskTrainingCore`, and `make_risk_training_core`. A future finite experiment owner passes existing cost-critic, Actor and reward-critic modules, a NumPy generator that it owns, and the dimension of the joint public observation. The factory creates only the dedicated cost-critic Adam optimizer. Its parameters must be disjoint from both Actors and the reward critic.

The kernel accepts hash-bound TRAIN batches containing whole, contiguous episodes, starting at frame zero. In Warehouse, the only accepted completed outcomes are an actual shutdown termination or the full physical time limit. An incomplete fragment cannot be assigned a safe label. Validation, program fitting, quality and holdout partitions are rejected. The future owner must separately verify scenario membership and the physical event trace; the string `partition='train'` is not a capability or independent evidence of provenance.

Only detached public observations and the real complete outcome survive ingestion. Actions, log probabilities and historical Actor autograd graphs are discarded. The latest 256 complete episodes are retained, including warmup failure examples, until normal FIFO eviction. Critic updates sample episodes uniformly without replacement and average one mean BCE per episode. Short shutdown episodes therefore have equal total weight to long safe episodes. Historical failures are not oversampled, duplicated or converted into surrogate labels.

The proposed default dual is explicitly configurable: 2% target, 256-episode window, at least 64 complete episodes, 90% one-sided upper bound, 1% release line and initial multiplier 2. The other dual settings reuse the unchanged mathematical implementation. Each newly completed episode is consumed exactly once. Critic replay cannot increase the dual's episode count or risk multiplier; batches without a completed episode do not update the multiplier.

Only cost-critic weights are optimized by this component. It does not compute any historical Actor cost surrogate. Future PPO integration must retain on-policy cost updates on the current rollout and preserve the paired initialization, scenario stream, reference KL, program feedback and admission gates. Better calibration of a historical outcome predictor does not itself demonstrate lower shutdown risk for the current policy.

The checkpoint contains the complete buffer, consumed episode and batch identities, cost-critic and Adam state, multiplier state, caller-owned sampling RNG state and numerical counters. It excludes Actor and reward-critic weights. Restoration validates hashes, shapes, finite values, parameter ownership, optimizer settings and risk config; rejected checkpoints leave the existing numerical state unchanged. It does not construct a new RNG. The eventual experiment owner still needs its full checkpoint and hash-bound restore contract.

## Actual numerical checks

Executed with Torch 2.13.0 on synthetic arrays only. No new or sealed Warehouse scenario was sampled.

`python3 -m pytest tests/test_warehouse_v643_risk_training.py tests/test_warehouse_v643_shutdown_diagnostic.py -q`

Result: **24 passed** (18 numerical risk tests and 6 archived diagnostic tests).

The numerical cases verify completed TRAIN episodes only, preservation of warmup positives within the declared FIFO capacity, duplicate prevention after eviction, no repeated dual observations during replay, and equal episode weighting. They verify actual Torch parameter updates, intact Actor/reward weights and pre-existing gradients, detached historical graphs, exact next-update restoration including Adam moments and RNG, and atomic rejection of corrupted observations, risk settings, optimizer settings, moments, RNG and critic weights.

One reproducible two-class synthetic check (seed 31, 80 critic updates, test-only learning rate 0.01) produced:

| Synthetic unit quantity | Before | After |
|---|---:|---:|
| Episode-balanced BCE | 0.74433529 | 0.06023817 |
| Episode-balanced Brier | 0.27513582 | 0.00433058 |
| Distinct complete risk observations | 2 | 2 |
| Actor updates | 0 | 0 |

The last cost-critic gradient norm before clipping was 0.13051833. The multiplier remained 2 because only two complete episodes were observed, below the default minimum of 64. These numbers establish that the numerical component learns and isolates gradients; they are **not task-performance or model-acceptance results**.

## Remaining execution boundary

The full v6.43 finite 20k owner is not implemented by this component. Before any scientific rollout it must register the new candidate and exact source hashes, establish seed-history clearance for finite training and evaluation inventories, create a fresh single-use claim, bind the common raw/regularized BC starting point and critic initialization, integrate this numerical component with current-rollout PPO, and implement resumable evidence generation and fail-closed admission. The closed v6.42 claim cannot be reused. The earlier proposal's `executor_implemented=False` remains accurate for the full experiment, even though this isolated numerical component is now implemented.
