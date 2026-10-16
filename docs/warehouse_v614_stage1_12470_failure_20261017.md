# Warehouse v6.14 exploratory candidate 12470: closed Stage 1

Candidate `warehouse-v614-certificate-core-unregistered-12470` is closed. Its
locked Stage 1 completed 230 BC environment resets, then rejected the newly
fitted sparse core because the frozen Actor and the `CoreTreeResult` refit
interpreter routed a threshold row to different leaves. It did not reach full
BC, quality, pressure, program preflight, PPO, formal training, audit, or
holdout evaluation. Its seeds will not be reused.

Immutable local evidence is under
`output/warehouse_v6_repro_snapshots/warehouse-v614-certificate-core-12470-20261017/`:

| File | SHA-256 |
| --- | --- |
| `stage1-failure.json` | `133a5d1d6465a4e828882d7d49ebcb81879cb53039c3b6d097586b375d1aa221` |
| `reset-audit.json` | `6989e86d458521d827e3ed44361e65fc3d890359b356742ec10cf272d4130fff` |
| `reset-audit.jsonl` | `618525f9b242a1cbaf403cab2541216fe80f808e3b19b8cf20fb7166f832c707` |
| `warmup-dual-actor.npz` | `0ef95a0cce060fa85fa6dbc709ab177711f29ed12dc071a3cc53d099aa92e68d` |

The reset audit reports `status=failed`, `failed_phase=bc`, and
`exact_prefix_matches=true`. The previous fix unified Torch and frozen Actor
routing, but the converted core result and the explanatory program still had
separate float32 route implementations. The successor source uses one shared
float64 split and left tie contract across Torch hard decisions, frozen Actor,
certificate fitting, core refit, and program explanation. This changes the
method and therefore requires a fresh, independently audited seed schedule.
