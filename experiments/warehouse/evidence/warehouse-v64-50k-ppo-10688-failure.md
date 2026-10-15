# Warehouse v6.4 50k PPO probe: dual-role program precheck failure

This is **unregistered exploratory evidence**, not a formal 200k/300k/400k
paired run, final program audit, or held-out test. No registered or sealed
Warehouse seed was used. The probe reused the v6.4 `none` BC pair as its
starting point (frozen BC pair SHA-256
`95946ea51e05c1ed29c02eaf201dfdcc7ab36e015d7d6a065aa5d8aeb63a42bc`),
then trained separate raw and regularized dual-Actor PPO arms for **50,000
joint steps each**. PPO pair seed `10688`, identical scenario schedule SHA-256
`dca9cd687ac3a4919a4f3afd9fb238b676d94d43a1eb4edae26b1f17b5a367c8`.
The BC and PPO source closures were respectively
`248b11fe3800344ef3ebdaabda38b6dc8f54675b917faa636a533d9cc057c2a7`
and `3f468561a1098d63bac8d877dfe6e74be5eccfea3bd649c032914aab9c8fd0a8`.
The second closure remained unchanged to the end of training. Both arms used
the same structured-tree Actor architecture and true-shutdown cost critic;
raw external program feedback was zero. This probe cannot establish the
required formal same-source-closure, fresh-BC-from-scratch comparison.

The selected checkpoints were chosen from the 24-episode development quality
series before program evaluation. On an independent 288-episode, three-task
pressure evaluation, both passed the development quality thresholds. Their
programs then failed the **30% feedback and explanation coverage threshold for
each role** on the prelocked, episode-disjoint `1071xxx` fit/calibration/leaf
eligibility/validation schedule:

| Arm | Selected step | Actor SHA-256 | Pressure deliveries | Collisions | Shutdown episodes | robot_1 feedback / explanation | robot_2 feedback / explanation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| raw | 21,120 | `c034be67af49ee7758d18cdd886ac61926dd588f50374b93aee56559ee66363b` | 15.913 | 3.476 | 1/288 | 29.19% / 29.27% | 31.18% / 31.32% |
| regularized | 45,840 | `8100d6bbf883c60f3571a4a324bc350147cede7cf383202244bed44cb05f5686` | 15.993 | 3.156 | 6/288 | **28.36% / 28.39%** | 34.86% / 32.94% |

The separate `1072xxx` unregistered diagnostic audit agreed that this was
not a fit-split accident: regularized robot_1 had 27.88% actionable feedback
and explanation coverage; robot_2 had 31.72% / 29.59%. Covered-action
agreement was at least 98.5% on the main regularized validation, KL was at
most 0.050, and illegal program actions were zero. Those passing quantities
do not compensate for low coverage. The `1071xxx` and `1072xxx` seeds are now
consumed for this exploratory method and cannot become independent method-lock
or sealed audit evidence.

The selected regularized checkpoint had **zero external program feedback
rows and zero external program gradient for robot_1**, versus 7,734 rows and
gradient norm 0.20075 for robot_2. Robot_1 began receiving a nonzero
external gradient only after that selected checkpoint (from step 47,640).
The online split provided robot_1 too few leaf-eligibility episodes at the
first feedback window. A later checkpoint cannot be substituted after seeing
program evaluation. Future candidates need predeclared per-role split quotas
and checkpoint admissibility requiring nonzero verified external program
gradient for both regularized roles, while raw remains zero.

A descriptive diagnostic on the earlier `1051xxx` **fit partition only**
showed the representation issue. For the `none` BC pair, the full Actor's mean
probability for the public BC label was 0.923 / 0.909 across roles, but its
internal tree branch's was 0.594 / 0.613. Tree/full top-action agreement was
only 66.4% / 70.2% on those fit rows. This is diagnostic, not an independent
validation claim. The next method must make the program-compatible path
intrinsic to the trained Actor, while still fitting the extracted program to
the **full masked NN distribution**, and must not turn the program into a
runtime action selector.

The ignored, byte-verified snapshot
`output/warehouse_v6_repro_snapshots/warehouse-v64-ppo-10688-20261016/`
contains the BC starting Actor, all 73 PPO source-closure files, scripts,
training receipts, selected raw/regularized Actor and checkpoints, four
programs, pressure/program evaluations, and the descriptive fit diagnostic.
Its 95-file, 10,823,263-byte `snapshot_manifest.json` was verified file by
file; SHA-256
`e667fc68e8465f15827b5805b6b37055f8f14e2e9c897531074a05f5b9dde649`.
The PPO pair report SHA-256 is
`cd59061da254afc7b6dc325ef96ff72a6e4959f45f2b4b1ee91adb4391ef0147`.
Raw exploratory artifacts remain out of Git. Warehouse v6.0 and the pilot
remain closed.
