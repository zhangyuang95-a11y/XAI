# Warehouse v6.14 exploratory candidate 12469: closed Stage 1

Candidate `warehouse-v614-certificate-core-unregistered-12469` is closed. Its
locked Stage 1 performed 1,396 environment resets and failed during the final
BC Actor's Torch versus frozen NumPy parity check, before quality, pressure,
program preflight, PPO, formal training, audit, or holdout evaluation.

The immutable evidence is under
`output/warehouse_v6_repro_snapshots/warehouse-v614-certificate-core-12469-20261016-r2/`:

| File | SHA-256 |
| --- | --- |
| `stage1-failure.json` | `12fd54b4e973f468a3f665dafef69b0a99e8cc932c9cbbaa7b800a8622f2a951` |
| `reset-audit.json` | `c49a0658fb70650764ee885b21a28abe003ddfba251b16c4980d5f90e7d0dac0` |
| `reset-audit.jsonl` | `7d68ce8b4c8c0df676a4f2a5a89e2cc64069fcf62789cd1380325024856b9d54` |
| `final-bc-dual-actor.npz` | `7a1f99cbd8ea6fc71d3ae05b51f0bdb0e3a683093b8271d3584aae765456fabe` |

The reset audit reports `status=failed`, `failed_phase=bc`, and
`exact_prefix_matches=true`. The 1,396 actual resets are 1,394 BC data
scenarios plus two critic resets. This candidate's seeds are spent and are
excluded from later exploratory schedules.

Root cause: sparse tree split scores near zero changed sign between batched
float32 Torch inference and single-frame frozen NumPy inference. Diagnostic
replay of the saved trained weights on independently constructed public states
found different tree leaves and actions. The successor source uses float64 hard
split routing with an explicit left tie band in both implementations, while
retaining the float32 surrogate gradient. Exported Actors now carry a routing
contract version so an older artifact cannot be interpreted under the new
semantics. A fresh candidate and fresh seed schedule are required to test this
fix; the old candidate provides no acceptance evidence.
