# Warehouse v6.6 static topology: fit-only CV below the coverage target

This is an **unregistered exploratory topology screen** using the frozen v6.5
BC Actor (artifact SHA-256
`bd2f1f63502589ec8261aeed2927955a51c64e759513b3c5fda23477f747e8b9`).
It is neither a new from-scratch BC pair nor paired raw/regularized PPO, an
independent four-part program validation, a final audit, or a held-out test.
The v6 pilot remains closed.

The [prelocked exploratory seeds](warehouse-v66-unregistered-seed-lock.json)
assigned `1110000–1110049` to topology fitting and `1110050–1110074` to
fit-only CV. The fresh dual-NN self-play collection contains 8,941 public
pre-action rows; compressed row file SHA-256 is
`aa917241df161aa08a05abaa270ec7c4d64b7fd5e44d6d2fe932f207342e0f55`.
The collection script checked that those scenario seeds were disjoint from
the source Actor's BC data. Source-closure SHA-256 is
`e2b35bab3f589bfabb410214c1af2089d9f596746d48bb15419988c5967f8686`.

The frozen sparse topology prototype (SHA-256
`3b70326535a187c276e4a2dffde8f54ed72e41d7914c545929112e2215912dce`)
uses the complete frozen NN masked action distribution. It adds learned
two/three-feature public directions, row-weighted split thresholds, and a
bounded two-step beam search. Its internal fit and CV scenario seed sets are
explicit and disjoint. Both roles used all 16 leaves and 15 internal nodes;
the maximum depth remains five and each split uses at most three public
pre-action features. Six synthetic tests passed, but those tests do not
establish real-domain coverage.

| Internal fit-only CV measure | robot_1 | robot_2 |
| --- | ---: | ---: |
| Qualified-leaf row mass | 30.13% | 31.60% |
| Expected sampled-action actionable coverage | **28.86%** | **29.68%** |
| Full-NN/program top-1 agreement on qualified rows | 98.23% | 98.73% |
| Masked KL on qualified rows | 0.0667 | 0.0635 |
| Raw illegal actions on qualified rows | 0 | 0 |

These figures are **fit-only CV estimates used in structure search**, not an
unbiased four-part program result. Both expected actionable values are below
the development threshold of 30% and well below the desired safety margin
of 40–50%. Large rejected leaves remain: robot_1 has leaves containing
22.1% and 19.3% of CV rows; robot_2 has leaves containing 20.8% and 20.1%.
The reserved v6.6 program fit/calibration/eligibility/validation, descriptive
audit, and pressure blocks were **not used**, and no new BC/PPO run was
started from this static topology.

The exact collection/evaluation scripts, source module, rows, tensors and
report are preserved in ignored
`output/warehouse_v6_repro_snapshots/warehouse-v66-static-topology-11166-20261016/`.
Its 10-file, 1,714,203-byte checked inventory has SHA-256
`a8675e60557a9a777714f3eae2695cb4d44a582a9257ca76f12e45cba3d8d249`.
An independent copy outside the ignored worktree is at
`/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v66-static-topology-11166-20261016/`;
its manifest and all ten file hashes and byte counts were verified again.
The fit/CV evaluation script SHA-256 is
`b02fd72475de6b1083f8b9a97d8af33dafc378edddb0ac1c32c64115f3c67e53`;
its report SHA-256 is
`c2dfac11d1dc6ed2bef60e4b972976784722ea0f78913dec37ee34e444424726`.

The next hypothesis is to train Actor and tree jointly so common public
states have a stable tree-dominant *complete NN distribution*, while the
256-wide residual remains available in complex states. That change must
preserve sampled NN quality and be evaluated on a newly locked, disjoint
fit/CV and independent program/pressure schedule. The real CV result here
provides no basis to claim v6.6 acceptance.
