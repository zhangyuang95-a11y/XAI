# Warehouse dual-Actor v5 release gate

The v5 Warehouse adapter binds `warehouse-raw-v5` or
`warehouse-regularized-v5` in each state. Each policy artifact holds distinct
`robot_1` and `robot_2` Actor tensors. During human tasks, the participant
replaces `robot_1`; `robot_2` still acts through its frozen Actor. AI–AI
demonstrations use both Actors from the same bound artifact. The static
physical action mask and environment collision rules remain active; no
task-assignment or avoidance controller changes an Actor action.

The training entry point keeps the fresh validation and held-out sets locked:

```sh
python3 -m backend.training.warehouse_dual_v5 \
  --pair-seeds 6001 6002 6003 --joint-steps 200000 \
  --bc-episodes 200 --bc-epochs 30 \
  --output-dir output/deployment
```

Each pair shares the same dual-Actor BC initialization and scenario seed
sequence. Both variants run PPO with two neural actions and per-role advantages
from a centralized two-value critic at each joint step. The training reward
uses collision score −100 and shutdown score −200; development, final testing,
and participant scoring retain the ordinary environment coefficients. Only
the regularized variant uses role-specific selective RCPD. It extracts an
observable feature tree every 20,000 joint steps from NN-visited training
states, with episode-disjoint tree fitting, leaf calibration, leaf eligibility,
and validation. Each extraction uses the most recent 124 distinct training
scenario seeds, split chronologically into 50 fitting, 25 calibration,
25 leaf eligibility, and 24 internal validation scenarios. The fixed role
presets use the same observable target-priority feature transform, omitting
previous-action features for the AI role. These presets were tuned on an
already-used development block and can fail at a later Actor checkpoint;
the selected program still needs the untouched final independent audit.
Each tree has depth at most 5 and at most 16 leaves.

A program reports **global fidelity** even when it is used selectively.
Online feedback uses only leaves that separately satisfy at least 20 rows,
action agreement of at least 85%, KL at most .35, and zero illegal actions
on both the eligibility and disjoint internal validation splits. AI
explanation leaves separately require 90% agreement on both splits. There must
be at least one covered row on validation, and the covered aggregate must
also meet the conditional agreement, KL, and legality gates. The online
feedback gate has no aggregate coverage floor; the initial 20k and 40k
development probes showed that a 20% online floor could prevent one role
from receiving any gradient. The frozen program must still independently
pass at least 20% coverage, 85% covered agreement, KL at most .35, and zero
illegal actions for both roles before release. AI explanations require at
least 20% coverage and 90% covered agreement under the same KL and legality
gates.
For a feedback row, the program action must equal both the current masked NN
argmax and the NN action sampled and executed on that row. The feedback weight
is `λ=0.01`; its schedule starts at 80% training progress and reaches full
strength at 85%. Raw feedback is exactly zero. Neither program ever chooses
an environment action. Training reports the number of PPO role-row
presentations during the positive feedback window, the number with no
validated legal program action agreeing with the current NN argmax and its
executed sample, and the separate number with no physically legal action.
Each optimization epoch counts as a presentation.

Teacher demonstrations are used only before PPO. An offline diagnostic on
2,400 NN-visited training states found that resetting environment-owned goal
and route memory changed the teacher's joint label on 209 states (8.7%). That
memory is present before the teacher labels an offline state but absent from
the NN observation, so BC supervision has a partial-observability limitation.
It supplies no teacher action, task rule, or override during formal self-play
or frozen execution.

The command writes resumable checkpoints under
`output/deployment/_warehouse_dual_v5_work/` and a final acceptance receipt.
Development uses seeds 240000–240023 to select the best matched pair among
pairs whose raw and regularized models both pass all quality gates. Only then
is the selected frozen program audited; program failure cannot switch the
selection to a lower-quality pair. A separate finalization run with
`--enable-heldout` is allowed only after **all three** regularized candidates
have positive audited program-feedback counts for both roles and the selected
pair passes its preliminary development program audit. It claims the untouched
program validation seeds
240100–240123 once before any fresh held-out rollout. It audits the **fixed**
selected leaf IDs and distributions, without refitting or changing coverage.
Fresh held-out
enrollment seeds 250000–250099 map to scenario seed `enrollment_seed * 10 +
task` for each of Tasks 1, 2, and 3. The 300 scenarios per variant are
evaluated once, only after both model and final independent program audits
pass and every pair completes the full training budget. Both variants must average at least
5 deliveries, at most 8 collisions, and at most 10% episodes with shutdown.
Both role programs must pass the selected frozen Actor's independent
development audit; the AI-role program must also pass the stricter explanation
audit. An audit failure leaves the held-out set untouched. The final program
audit and held-out evaluation each have an atomic started receipt, so a crash
cannot silently repeat a fresh evaluation.
`--install` can be run with the finalization or after a completed acceptance;
the resume path rechecks both receipts and the exact frozen asset hashes.
The study pilot stays closed until the receipt passes and all actor, program,
and report hashes verify. Simulated partner outcomes are not human results.

## 2026-10-15 full-budget development result

All six matched runs finished 200,000 joint training steps. The development
scores below use only seeds 240000–240023, with the ordinary evaluation reward.
Each row gives deliveries, robot collisions, and the fraction of episodes with
shutdown. Every selected raw and regularized Actor meets the model-quality
thresholds; all three regularized pairs received actual selective KL feedback
for both roles. Raw feedback was zero in every pair, and no post-policy action
was overridden.

| Pair seed | Raw selected step | Raw dev | Regularized selected step | Regularized dev | Regularized feedback rows (human/AI) |
| --- | ---: | --- | ---: | --- | ---: |
| 6001 | 160,320 | 7.792 / 2.917 / 4.17% | 200,000 | 8.542 / 3.042 / 4.17% | 8,432 / 6,898 |
| 6002 | 200,000 | 8.417 / 2.792 / 4.17% | 200,000 | 8.167 / 4.875 / 4.17% | 10,253 / 12,574 |
| 6003 | 200,000 | 9.042 / 2.292 / 0% | 200,000 | 7.583 / 2.625 / 8.33% | 17,767 / 11,438 |

The predetermined maximum of the weaker raw/regularized quality selects
pair 6001. Its **preliminary**, already-used development-state audit of the
frozen program fails the required 20% selective coverage: human-role feedback
coverage is 0.85% (100% agreement, KL .00002), AI-role feedback coverage is
12.69% (87.5% agreement, KL .322), and AI explanation coverage is 2.79%
(97.47% agreement, KL .151). Global program action agreements are 69.75% and
69.43% for the human and AI roles; those global figures are reported openly
but are not substituted for the selective gates. The [dev-only re-extraction
probe](study_v3_warehouse_reextract_dev_probe.md) did not find a bounded
program clearing the fixed coverage thresholds for both roles and AI
explanation.

The [development acceptance receipt](../output/deployment/warehouse_dual_v5_acceptance.json)
records `passed=false`, `heldout_locked=true`, and `heldout_evaluated=false`.
No final program audit on seeds 240100–240123 or held-out evaluation on
enrollment seeds 250000–250099 has run; neither once-only receipt exists.
The v5 Warehouse policy config still has null Actor hashes, so this domain
does not open the pilot. A separate **development-only** pair-6001
width-32 Actor probe used the same 200 initial teacher episodes, 30 initial
BC epochs, and six rounds of 80 NN-visited offline relabel scenarios. Its
pretraining took 501.8 seconds on one CPU core, and its already-used
240000–240023 development result was 3.75 deliveries, 14.25 collisions,
and 16.67% shutdown. It failed all three model gates before any PPO probe,
so no formal retraining or final-set evaluation used that compact model.

The earlier shared-parameter v4 artifact and QA bank remain available for
historical records. Calls without a v5 `policy_id` still use that historical
artifact; the v5 study store must always pass an explicit policy ID.
