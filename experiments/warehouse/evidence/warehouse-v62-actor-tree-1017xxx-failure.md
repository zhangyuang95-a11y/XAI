# Warehouse v6.2 exploratory Actor-tree failure

This is an **unregistered exploration**, not a v6 development, formal,
independent-audit, or held-out result. The fixed Actor is the public-observation
BC pair from the separate 10077 snapshot (`actor_sha256`
`9ae8aa8a501422f5c2aea602040ab261a20f4cda9b675838a60fca9324c36c6f`).
No registered or sealed v6 seed was used.

The method was locked on fit-only cross-validation (`method_lock_sha256`
`9e1cc632e11d932b993cd66861c6066fb93be0efcb000f6732b5e306c563f6f6`)
before independent 1017xxx validation. Both roles used the Actor's own sparse
tree structure as an extracted program. The independent validation results were:

| Role | Feedback actionable coverage | Explanation actionable coverage | Agreement | KL | Illegal actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| robot_1 | 26.63% | 26.63% | 98.86% | 0.061 | 0 |
| robot_2 | 25.24% | 25.28% | 99.59% | 0.018 | 0 |

The required exploratory development coverage was 30% for **each** role; the
method failed. A separate 24-episode holdback also stayed below 30% (robot_1
feedback 27.60%, robot_2 feedback 24.48%). It cannot be promoted by citing
its high agreement or low KL. Fit-only five-fold comparison did not support
switching robot_2 back to the old extracted tree: mean actionable coverage was
21.41% for the Actor tree and 18.20% for the old tree.

The root-cause diagnostic on 5,760 public pre-action frames found masked
top-1 agreement between each Actor's tree and its **complete NN** of only
67.34% (robot_1) and 65.80% (robot_2). High-traffic leaves 9 and 13 held
roughly 35–37% of frames yet had only about 46–49% agreement. Their learned
residual contribution dominated the tree margin. This is why fitting or
extracting the same old tree cannot by itself solve the coverage gap; simply
clamping the residual on these mixed leaves risks task quality.

The necessary raw JSONL, fit/calibration/eligibility/validation receipts,
locked method, holdback, five-fold comparison, and leaf diagnostic are preserved
in the ignored snapshot
`output/warehouse_v6_repro_snapshots/warehouse-v62-actor-tree-1017xxx-20261016/`.
Its verified file inventory is `archive_manifest.json`, SHA-256
`b184aeeb56be4f3ab56ded991b6a3eb16b3b2fa0dc00857a339e563049ed5e0c`.
The independent report SHA-256 is
`d6053b4be4f75e1a5d817dbc3be75f16ce03ba96557acaa945c1edd568fb1e5b`,
the five-fold summary SHA-256 is
`68f4df01c07dcec420c0a042c5d73d9f953de732ca0b32e75aa962a1bae13756`,
and the leaf diagnostic SHA-256 is
`30ea95393b75c2bc2510358df52dc94fadfcb321f339502c30341736e46f3f78`.
The matching frozen Actor and source closure are in the earlier
`warehouse-v62-bc-10077-20261016` snapshot. These outputs are intentionally
not added to Git wholesale.
