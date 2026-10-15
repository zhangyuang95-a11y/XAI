# Warehouse v6.3 irregular-tree BC probe: development failure

This is an **unregistered exploratory comparison**, not a formal paired PPO
run, independent audit, or held-out result. It used pair seed `10377` and a
single source closure (`4a28b1a526abc4e9fd68810719879a9eddef7e4568b018ccb9c46d08aafd9c04`).
The three BC arms used identical initial role weights and BC scenario schedule
(`bc_data_seed_schedule_sha256` is recorded in each report). Their frozen
weights diverged through the specified training objectives. No registered
`6120`–`6122`, formal `6101`–`6103`, or sealed v6 seed was used.

The probe replaced the fixed-depth initialization with an irregular sparse
tree of depth at most 5 and at most 16 leaves. It refit the tree from the
complete Actor's masked action distribution, then compared three BC
objectives: residual only (`none`), tree supervision (`tree`), and tree
supervision plus learned gate (`tree_gate`). All execution actions came from
the frozen complete NN with the physical action mask. The tree-derived
programs were audited separately on episode-disjoint fit, calibration,
eligibility, and validation partitions. The evaluation used 288 fresh
pressure episodes per arm across three tasks, plus 24 program validation
episodes. The separate 24-episode diagnostic audit is not a selection gate.

| BC arm | Pressure deliveries | Collisions | Shutdown episodes | robot_1 feedback / explanation coverage | robot_2 feedback / explanation coverage | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `none` | 13.8125 | 5.2361 | 1.74% | 31.70% / 31.77% | 29.24% / 26.77% | robot_2 coverage below 30% |
| `tree` | 13.4375 | 6.2535 | 1.04% | 29.44% / 29.69% | 31.73% / 31.90% | Collisions above 6 and robot_1 coverage below 30% |
| `tree_gate` | 13.9097 | 4.8229 | 0.35% | 28.37% / 28.44% | 28.02% / 28.06% | Both roles' coverage below 30% |

For `tree_gate`, feedback agreements were 97.99% and 99.39%, explanation
agreements were the same, KL values were 0.0656 and 0.0184, and both illegal
program-action counts were zero. These passing fidelity components do not
offset the failed coverage. Leaf diagnostics found several high-traffic mixed
leaves where the residual changes the Actor's preferred action; tightening
residual bounds on an entire such leaf would risk task quality. The next
development variant must change the public observation/action representation
before a registered run can be considered.

The complete probe receipts, all three frozen Actor pairs, extracted programs,
probe scripts, and three modified source files are preserved in the ignored
snapshot `output/warehouse_v6_repro_snapshots/warehouse-v63-irregular-bc-10377-20261016/`.
Its `archive_manifest.json` lists 33 copied files (3,428,442 bytes) and has
SHA-256 `dd764f883e4cbb3c30b9e49c27c45dada0978ad9fbae5f95fb789da3367589ec`.
The original source snapshot manifest has SHA-256
`3b019a72c6571eb9d0cc90af30a1cd98220f5ee9a79c2152ed34afe3ee152a64`.
The three frozen pair asset hashes are `975098f265a7c854fac85ce0fdab29e2258114d55d38243a3702c1706a2e4a37`
(`none`), `c1edce8e3fb105a3c8935aa39a0b58b91771b4856d8cdf3ff39f190868aed949`
(`tree`), and `d9c7eb315283e4c6d1240b0fc101354e02c540cd99fa16c19dbd30714f31449d`
(`tree_gate`). Raw exploratory outputs are intentionally not added to Git.
