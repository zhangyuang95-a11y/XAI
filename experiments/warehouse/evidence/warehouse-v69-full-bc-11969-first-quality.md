# Warehouse v6.9 unregistered full BC: first quality screen

Status: **exploratory first quality screen passed; model admission and pilot remain closed.** This is a from-scratch, shared dual-Actor BC start. It is not a raw/regularized PPO pair, a program fidelity result, a registered development preflight, an independent audit, or a sealed test.

## Frozen identity and scope

- Pair seed: `11969`; public observation schema: `warehouse-public-observation.v6.1`, feature schema SHA-256 `f1cb243f1903f4c3462f9de78ed6df5e6d0b5ade7ebfc2a56d9966fa337f99c7`.
- Source closure SHA-256: `8569cdf9dcca314d5ec738a62c962bbbc51ea5c3125b3f37677ebb0b6b330be9`, 121 files. Git commit at snapshot: `539fe89107b89b8d2004cfc375dd2d406b3d26d3`.
- Seed-lock SHA-256: `01f5600cbd1457aa9836e3a8e1e41df25a280e35244f2ed56eb10ac468dde82e`; source snapshot manifest SHA-256: `88925fc414d2cd8aa9c9acd89ec8587558863b9f62b57eb70725eba819dc0fb4`; predeclared run SHA-256: `b6c7146bdbdf1113329ebe276149b116cdbb6ae09fd80b9aa4bf185b50233c0d`.
- Execution script SHA-256: `b5055affcbfa7a5a38822244e824dfe4fcb47888c708086a8668cc3020d5d6bb`; separate pre-execution script-lock SHA-256: `5fbcd3e9cc0365e5f2343d60665ba76db3fc7ee5abb763f343f42bf0e2f5dd1f`. It ran under `python3 -I` from a neutral working directory, with all 79 imported project modules in the isolated source copy.
- The full public BC recipe used 16 scenario seed blocks and 1,220 training episodes. BC and quality seeds were disjoint. The 119 pressure set, 118 confirmation, registered development pairs, one-time program audit, and sealed heldout set were not used.

## First quality screen

The frozen dual NN alone chose both actions in each of 96 episodes (32 base seeds × three tasks). The predeclared overall screen was deliveries ≥5.5, collisions ≤6, and shutdown episode rate ≤5%.

| Set | Episodes | Mean deliveries | Mean robot collisions | Shutdown episodes |
| --- | ---: | ---: | ---: | ---: |
| Overall | 96 | 13.8229 | 3.2083 | 2 (2.083%) |
| Task 1 | 32 | 14.0625 | 2.3750 | 0 |
| Task 2 | 32 | 13.6875 | 3.6563 | 2 (6.25%) |
| Task 3 | 32 | 13.7188 | 3.5938 | 0 |

All six predeclared overall gates passed. Across the 96 episodes, executed illegal actions and post-policy action overrides were both zero. Torch and exported NumPy Actor logits agreed within `9.78e-6` on legal actions, with identical masked argmax on all checked rows. The longest repeated collision with the same proposed joint action lasted four frames. Task 2's 2/32 shutdown count requires the reserved multi-stream pressure evaluation before a quality claim with margin.

On 70,257 BC training rows, the final dual Actor assigned mean probability `0.9064` and `0.9070` to the two public teacher actions, respectively; their mean joint probability was `0.8271`. These are training-distribution diagnostics, not independent human-partner results.

## Preserved evidence

The original ignored run directory is `output/warehouse_v6_repro_snapshots/warehouse-v69-full-bc-11969-20261016-v3/`. Its frozen Actor NPZ SHA-256 is `c4080edeaff2b633fbb274db709a0685ce70692978e5fc710c22c8553792584f`; BC receipt SHA-256 is `9ce9273623270df41b1ff1e3578b98d411f9e153a728195aa6b76abfb9b21181`; the 96-episode raw result SHA-256 is `99f1600e00e7324ba5d56c0ee120b40c262ebf71ba33d3de1f7e5f0add702ee5`. The run log SHA-256 is `8d40873c0593a52685673d4685f995675de44ca5f54e6e1a9aef2ad14b126257`.

The selected output manifest SHA-256 is `8f87906efcaa32a7e3197ea5e230f3492cbd21fff4a0e4d3c16ab74a55fbccbb`. An external recovery copy at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-full-bc-11969-20261016/` contains 14 selected artifacts, all 121 source files, and the manifest. All 136 copied files (5,902,920 bytes) were verified against source and destination SHA-256. Its verification receipt SHA-256 is `a5e2da61f34872016349814e6804e73381db539d3129902c12507945a65b06a2`.
