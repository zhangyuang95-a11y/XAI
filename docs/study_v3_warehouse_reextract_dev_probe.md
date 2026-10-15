# Warehouse bounded-program development probe (2026-10-15)

This is a **development-only** diagnosis of the frozen pair-6001 regularized
Actor selected after the three matched 200,000-step runs. It does not change
the trained NN, its PPO feedback history, the published program schema, or
the model-quality selection rule. The evaluation block was the already-used
development seeds 240000–240023. Final program audit seeds 240100–240123 and
held-out enrollment seeds 250000–250099 were not used.

The selected v5.3 program, fitted from NN-visited training states with
episode-disjoint fit/calibration/leaf-eligibility/internal-validation splits,
has the following separate preliminary development results:

| Role and purpose | Coverage | Covered action agreement | Covered KL | Result |
| --- | ---: | ---: | ---: | --- |
| Human-role feedback | 0.85% | 100.00% | .00002 | Fails 20% coverage |
| AI-role feedback | 12.69% | 87.50% | .322 | Fails 20% coverage |
| AI-role explanation | 2.79% | 97.47% | .151 | Fails 20% coverage |

The frozen NN's masked top-action probability exceeds the second action by
an average of .703 (human role) and .714 (AI role) on those same development
states. WAIT is its top action on 40.94% and 36.42% of states, respectively.
The shortage of auditable coverage is therefore tied to the shallow tree's
state partition, rather than an absence of a frequent or confident action.

Two bounded post-freeze extraction diagnostics kept depth at most 5 and at
most 16 leaves, used only pre-action observable features, and used independent
training-episode splits for fitting, soft leaf calibration, leaf eligibility,
and internal validation:

| Training-state source / search | Best human feedback coverage | Best AI feedback coverage | Best AI explanation coverage | Finding |
| --- | ---: | ---: | ---: | --- |
| Existing 124-scenario split; action-targeted tree objective including NN WAIT probability | 19.46% for a WAIT-targeted human tree | — | — | Human covered agreement 95.29%, KL .172, still below 20% |
| Expanded 400 already-used training-schedule scenarios; 200/75/75/50 split; targeted action scan | 14.88% | 22.32% | 7.33% | Per-role best previews; no two-role feedback plus AI-explanation pass |

The best values are search maxima on **reused** development seeds, not an
independent final audit or a coherent passing program trio. Additional
observable order fields did not improve the bounded search. This probe does
not justify publishing a new program, lowering a gate, or using the fresh
final seeds. Any successful later extractor would need a new explicit program
version, disjoint training-only calibration and validation, and its own frozen
program audit. The v5.3 in-loop feedback program and any post-freeze
explanation program must remain separately identified by SHA and provenance.

Source of record for the formal selection and preliminary metrics:
[Warehouse v5 development acceptance](../output/deployment/warehouse_dual_v5_acceptance.json).
The current acceptance has `passed=false`, `heldout_locked=true`, and
`heldout_evaluated=false`.
