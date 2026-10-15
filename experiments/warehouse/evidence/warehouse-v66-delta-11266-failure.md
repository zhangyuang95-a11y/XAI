# Warehouse v6.6 public distance progress: fit-only CV failure

This is an **unregistered exploratory feature comparison**, not a new BC/PPO
pair, four-part program validation, independent audit, or held-out result. The
v6 pilot remains closed. The [prelocked 11266 schedule](warehouse-v66-joint-actor-seed-lock.json)
assigned scenario seeds `1120000–1120049` to topology fit and
`1120050–1120074` to internal fit-only CV. The 8,983 public pre-action rows
were produced by the unchanged frozen v6.5 BC dual Actor (artifact SHA-256
`bd2f1f63502589ec8261aeed2927955a51c64e759513b3c5fda23477f747e8b9`,
source closure `e2b35bab3f589bfabb410214c1af2089d9f596746d48bb15419988c5967f8686`).
The row archive SHA-256 is
`8331e91dedd1d736c9761020bcbe9e40cf5771a22195c4d666d7ddd3626157c9`.

The comparison used the same sparse topology fitter (SHA-256
`3b70326535a187c276e4a2dffde8f54ed72e41d7914c545929112e2215912dce`)
and full masked frozen NN distributions in both arms. The new 30 public
features give each candidate action's shortest-distance change from WAIT to
the charger, both orders' pickup and delivery endpoints, and the partner.
They select no target or action. In the new arm, redundant non-WAIT absolute
distances were removed **only from the split search**: each is exactly its
WAIT distance plus the added change. The frozen source Actor and its actions
were not changed.

| Internal fit-only CV measure | Absolute r1 | Absolute r2 | Delta substitution r1 | Delta substitution r2 |
| --- | ---: | ---: | ---: | ---: |
| Qualified-leaf row mass | 26.99% | 29.87% | 20.48% | 25.68% |
| Expected sampled-action actionable coverage | **25.62%** | **27.20%** | **19.95%** | **23.22%** |
| Top-1 agreement on qualified rows | 97.39% | 96.75% | 99.84% | 96.21% |
| Masked KL on qualified rows | 0.0763 | 0.0761 | 0.0254 | 0.0647 |
| Raw illegal actions on qualified rows | 0 | 0 | 0 | 0 |

The result is below the 30% development floor for both roles and far below
the desired 40% exploratory margin. It is an internal CV screen used for
method development, not an unbiased program audit. The explicit progress
features alone do not solve the mixed-policy leaf problem; no BC/PPO or
quality claim follows from this comparison.

The first script stopped before the new arm because it exceeded the fitter's
256-direction cap. The second substituted information-equivalent split
features and then stopped at a 200-versus-230-dimensional temporary Torch
route-parity check. The third widened only that temporary tree; the fit
objective, source Actor, source rows, and seed split stayed fixed. Thus the
two errors were not counted as negative feature results. Script SHA-256 values
are `22e69e1d29dcd5d2c55636cfb849436c5747304ac646f2e9be169bc2b6f891f3`,
`d8895d30444fac18441dd5d936bdf28b6aab5a5e0dd4ab4b1eb890a219611de7`,
and `10fe694feaaf0eec31bd3189785bc223698ed63ca47dc5be775bafe5f2bac48c`.
The final report SHA-256 is
`b0d2f0d276831446d169f6a2cb0ae85a1f7347419ebaa435aab0bbb6f2a0cdbc`.

The exact scripts, error logs, frozen fitter, source Actor, public rows,
topology tensors, and report are preserved in ignored
`output/warehouse_v6_repro_snapshots/warehouse-v66-delta-11266-20261016/`.
All 18 files (2,947,503 bytes) were copied and verified against the manifest,
whose SHA-256 is
`376086e674fa1f684b11e3bbc2ee4a44b58753dd838abd514a01fb5133d3e917`.
The same snapshot was also copied outside the integration worktree to
`/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v66-delta-11266-20261016/`;
all 18 copied files, their byte counts and SHA-256 hashes, and the manifest
bytes were independently checked against the in-worktree snapshot.
The reserved `1120100–1120223` four-part program, `1120300–1120323` audit,
`1120400–1120431` pressure-base, and `1121100–1121431` post-PPO blocks were
**not used**. The next Actor/tree joint-training method needs a new
unregistered seed lock before any evaluation, and the formal registered and
sealed blocks remain untouched.
