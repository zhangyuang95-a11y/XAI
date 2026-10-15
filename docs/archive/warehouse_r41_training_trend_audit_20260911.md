# Warehouse r4.1 training trend audit through 650k

Audit date: 2026-09-11. Scope: r3 baseline and every atomically committed r4.1 boundary from 50k through 650k additional PPO joint steps. This was a read-only audit. It did not modify the frozen trainer, evaluator, runner, checkpoints, Actors, programs, scenes, or the live PID 86030.

Machine-readable evidence: `warehouse_r41_training_trend_audit_20260911.json`, SHA-256 `65edf0381d7553d4c6ab72b9ec2886de68e1d1a897ba1c5d0da3fbcb8792dc6c`.

## Decision

**Do not stop PID 86030 for integrity reasons.** I found no corrupted sampling, overwritten neural command, shaping arithmetic error, reversed KL, missing optimizer step, or discontinuity between immutable boundaries. Continuing within the already approved 2M budget remains useful because the conflict suite has shown real, if insufficient, gains and every 50k boundary preserves a separately auditable candidate.

The chance of satisfying the current dual-suite gate is low at 650k. The original suite is forgetting, conflict delivery gains have never reached either delivery threshold, and Actor shutdowns are nonzero and highly unstable. Any boundary that the frozen aggregate later labels selected must remain quarantined until the corrected sixth-partner audit and a source-complete admission receipt pass.

## Confirmed implementation defects

1. **The six-partner validation is actually five behaviors.** `warehouse_r41_active_evaluation.py:30` inherits `PARTNERS` from `warehouse_r4_active_evaluation.py:33`, where both `skilled` and `fixed_yield` are listed. Evaluation always asks the program partner to control `robot_1`, but the only `fixed_yield` special case in `env/warehouse_native/partners.py:142` runs for `robot_2`. After removing only `partner` and `seed`, every `skilled`/`fixed_yield` episode pair is identical for the baseline and every 50k–650k boundary: 50/50 original scenes and 64/64 conflict scenes at each checkpoint. This is a release-blocking evaluation bug. It does not contaminate PPO because `fixed_yield` is absent from the training mix.

2. **The implementation source closure omits the code that selects candidates.** `_implementation_sources()` in `warehouse_r41_active_trainer.py:171–183` binds the trainer, inherited trainer, KL updater, scenario module, runtime, and environment, but omits `warehouse_r41_active_run.py` and `warehouse_r41_active_evaluation.py`. Those files commit each boundary and calculate selection (`warehouse_r41_active_run.py:140–198`). The existing receipts therefore cannot cryptographically prove the runner/evaluator bytes behind every decision. Current hashes were stable during this audit, so this is a provenance gap rather than evidence that current PPO data were altered.

3. **`run.json` is stale during a live run.** It still reports zero new steps and 3.95M cumulative steps while the immutable 650k boundary records 4.60M. The runner writes progress at start and final completion (`warehouse_r41_active_run.py:280–345`), not after each boundary. Resume and evidence use immutable boundaries, so training integrity is unaffected; operational monitoring must use the boundary ledger.

## Verified training mechanics

- The registered episode mix is self-play 20%, skilled 20%, assertive 50%, noisy 10% (`warehouse_r41_active_trainer.py:44`). At 650k, robot_2 transition counts are 146,265 / 120,115 / 320,670 / 62,950 respectively. The small percentage shift follows from episode-length variation. All 796,265 trainable actions equal the submitted NN actions; override count is zero.
- Active shaping is applied to trainable roles at collection (`warehouse_r41_active_trainer.py:591–627`). At every boundary, the saved sum reconciles with `0.02×productive − 0.03×avoidable_wait − 0.02×distance_regression`; the largest residual is 1.26e-07. At 650k its absolute magnitude is 13.37% of absolute base reward. The code and counters agree.
- `_kl` in `warehouse_r4_role_feedback_update.py:20–36` computes `KL(pi_nn || program)` from detached, normalized tree targets. All 325 update audits use `ordinary_feedback`, role `robot_2`, and lambda 0.001. Every boundary program is training-reliable; fidelity ranges from 86.48% to 91.41%. There is no KL direction, scope, or schedule error.
- At 650k the trainer has 325 PPO updates and 10,192 matching Actor/Critic Adam steps. The checkpoint counter is exactly 650,000 joint steps.

## Boundary trend

| Actor | Original productive | Original AI deliveries | Original no-progress p95 | Original AI shutdowns | Conflict productive | Conflict AI deliveries | Conflict no-progress p95 | Conflict AI shutdowns | RCPD fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r3 baseline | 72.20% | 8.623 | 35.10 | 19 | 71.26% | 9.094 | 53.00 | 49 | — |
| +50k | 69.84% | 8.063 | 62.00 | 53 | 74.47% | 9.458 | 40.00 | 43 | 87.83% |
| +100k | 71.28% | 8.257 | 39.05 | 35 | 75.57% | 9.521 | 25.00 | 26 | 89.56% |
| +150k | 69.29% | 7.593 | 76.00 | 63 | 74.74% | 9.419 | 24.70 | 25 | 89.08% |
| +200k | 70.33% | 7.850 | 61.60 | 54 | 75.14% | 9.471 | 26.00 | 40 | 90.04% |
| +250k | 70.37% | 7.863 | 57.05 | 61 | 76.22% | 9.766 | 23.00 | 40 | 89.00% |
| +300k | 68.31% | 7.747 | 76.00 | 44 | 77.36% | 9.638 | 20.00 | 50 | 91.23% |
| +350k | 68.52% | 7.473 | 62.05 | 60 | 76.50% | 9.487 | 21.00 | 49 | 88.78% |
| +400k | 67.52% | 7.403 | 80.10 | 48 | 75.85% | 9.544 | 43.00 | 56 | 86.79% |
| +450k | 67.37% | 7.320 | 83.10 | 49 | 76.46% | 9.372 | 31.00 | 73 | 89.48% |
| +500k | 66.65% | 7.357 | 77.10 | 36 | 76.11% | 9.591 | 24.85 | 38 | 89.49% |
| +550k | 65.10% | 7.063 | 77.00 | 37 | 76.17% | 9.695 | 21.85 | 34 | 86.48% |
| +600k | 66.04% | 7.400 | 83.05 | 25 | 74.61% | 9.617 | 34.00 | 16 | 89.01% |
| +650k | 69.15% | 7.353 | 63.30 | 99 | 76.77% | 9.625 | 24.00 | 79 | 91.41% |

At 650k versus r3, the original suite has lost 3.05 percentage points of productive action and 1.270 AI deliveries per episode. Its no-progress p95 rose from 35.1 to 63.3, and Actor shutdowns rose from 19 to 99. The conflict suite gained 5.51 points of productive action and 0.531 deliveries, while no-progress p95 fell from 53 to 24 and collision cancellation fell from 8.62% to 4.99%.

The best conflict checkpoints still miss the registered improvements: productive action peaks at 77.36% at 300k, only +6.10 points versus the required +8; AI deliveries peak at 9.766 at 250k, only +0.672 / +7.39% versus the required +1 and +15%. Shutdowns briefly fall to 16 at 600k and then jump to 79 at 650k. The same 600k→650k jump is 25→99 on the original suite. That volatility rules out interpreting the recent low count as convergence.

The opposite per-domain behavior is also visible within the assertive partner. At 550k, conflict productive action is 78.45%, AI deliveries 10.016, no-progress p95 15, collision cancellation 3.11%, and zero Actor shutdowns. On the original suite at the same checkpoint, the values are 58.31%, 6.38, 98.25, 22.16%, and four shutdowns. This is consistent with conflict-family specialization, not a globally broken optimizer.

## Why the original suite degrades

All ordinary starts come from `r41_conflict_train`, and all energy starts are snapshots collected from the same conflict family (`warehouse_r41_active_trainer.py:545–587`). The original suite is mandatory for admission but never supplies PPO states. The RCPD reservoir is likewise filled from those conflict trajectories. Training therefore optimizes one state distribution while selection requires two, and the KL regularizer preserves a conflict-only approximation rather than anchoring original behavior.

This is an explicit protocol choice, so it is a train/admission distribution mismatch rather than a hidden data-loader bug. The observed split is exactly what that mismatch predicts: better conflict movement and collision recovery alongside sustained original-suite forgetting.

## Why shutdowns do not approach zero

The reset sampler is numerically correct: 1,245 of 6,388 resets are energy curriculum starts, or 19.49%, close to the registered 20%. The semantic exposure is much smaller. Those episodes begin late and terminate sooner, so they contribute only 51,431 of 650,000 PPO transitions, or 7.91%. The fixed bank has 192 snapshots drawn from only 18 of 128 training scenes.

Energy segments do help locally: completed energy segments contain 72 team shutdown events across 1,245 episodes, versus 728 across 5,127 ordinary episodes. They are too short and narrow to satisfy a zero-Actor-shutdown requirement across 684 dual-suite evaluation episodes. Active shaping also gives no distance-regression penalty when `charge_needed` is true (`warehouse_r4_active_trainer.py:929–934`); it supplies a positive approach signal and an avoidable-WAIT penalty, but no direct Actor-specific survival objective. The environment base shutdown score is shared through `score_delta`, so partner and Actor terminal credit are not cleanly separated.

This is not a reset-probability or battery-reachability implementation error. It is a curriculum-unit and objective mismatch: 20% of resets was treated as though it meant 20% of learning transitions, and a narrow fixed bank was asked to support a strict zero-event gate.

## Why conflict delivery improves too little

The base reward uses team `score_delta` for both roles. Role-specific progress shaping exists, but a delivery by the program partner still raises robot_2's return. The Actor can therefore improve team throughput and conflict recovery without satisfying the registered own-delivery gates. The additional active shaping is only 0.02 per productive step and does not distinguish the Actor's own completed delivery from team progress strongly enough to overcome that credit ambiguity.

This explains why conflict collision and no-progress metrics improve more than Actor delivery contribution. It is an objective-alignment limitation, not evidence that the delivery counter is wrong.

## Minimum next round if 2M exhausts

Keep r3 online and close r4.1 as failed if no boundary passes the corrected admission. The smallest defensible revision is:

1. Replace the duplicate sixth validation behavior and bind the exact runner/evaluator hashes before selection.
2. Stratify PPO starts 50/50 across original and conflict training families, matching the two mandatory validation suites, and extract RCPD from both strata.
3. Allocate 20% of PPO **transitions** to reachable-energy states, regenerate a diverse bank across all training scenes and energy/distance bins, and monitor transition exposure at each boundary.
4. Add a registered robot_2-specific shutdown/survival term and centered own-delivery credit so the training objective directly matches zero Actor shutdowns and the +1/+15% delivery gates.

These changes preserve the central constraint: NN actions remain authoritative, with no program takeover, action repair, or post-policy override. The thresholds and both validation suites should remain unchanged.

## Source hashes at cutoff

- `warehouse_r41_active_trainer.py`: `9b4caec47e49d2ca6ec83c5b4174852e2b25b1fd9f745ddc61a9cc664b1239aa`
- `warehouse_r41_active_evaluation.py`: `c75194254e46e5aa224cbde4979a844797bc2a9f8f03ab318b1e199f60fa87e5`
- `warehouse_r41_active_run.py`: `61b9e8b627f9fd3005938359dd98a0516cefbb48b8754f7e9169c44d7d945172`
- `warehouse_r4_role_feedback_update.py`: `0dbba0df53c8ef6652a90f81426cca6085d7c94d31ba95ad2ad40f9b6eb12fcf`
- `partners.py`: `8f70f3234f898e66ae3b790d8b4f6e869d0d7f60ae3086cc30a15f7d1dbceb95`
