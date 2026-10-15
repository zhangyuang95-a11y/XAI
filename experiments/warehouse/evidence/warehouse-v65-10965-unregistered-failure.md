# Warehouse v6.5 intrinsic-tree probe: unregistered failure

This is exploratory evidence from a from-scratch BC pair, **not** a registered
raw/regularized PPO comparison, final program audit, held-out test, or release
asset. The v6 pilot remains closed. The frozen source closure is SHA-256
`e2b35bab3f589bfabb410214c1af2089d9f596746d48bb15419988c5967f8686`;
the BC pair seed is `10965`. The exploratory recipe SHA-256 is
`23df6d7c5d2dd90ef518dbf133d800bedebe112ba01d93029a93344f4f325314`.
That recipe incorrectly claimed a gate reset, whereas the run reset only leaf
bounds. This claim was corrected in the working recipe **after** the frozen
probe; the probe was never used as a formal BC starting point.

The frozen dual-Actor artifact is SHA-256
`bd2f1f63502589ec8261aeed2927955a51c64e759513b3c5fda23477f747e8b9`
(role hashes `c92a12158872e00fd78890de60320907f7b15329b325bac60945ff1d00afbdae`
and `28f60f7f80f0b32601ee7573408b753faa6223fb368d96de92ad8be9a1300bea`).
The from-scratch BC report SHA-256 is
`65fc283457423ad2fdd519968ee5ab7670d4a0a08529bec9b82149e72c8d8799`.
The predeclared evaluation script SHA-256 is
`f3d03278775ffd18e2ab9ab3cf00754132760f356c43a69127197a54b2673ee0`;
its result SHA-256 is
`f7896a18a20e5604fdc91a960e6e5ab25fd6ab91621748cb3b805004163aef5b`.

Program fitting used disjoint unregistered scenario seeds: fit `108300–108349`,
calibration `108350–108374`, eligibility `108375–108399`, and internal validation
`108400–108423`. An additional exploratory audit used `108500–108523`.
The pressure evaluation used base seeds `108600–108631`, each expanded to
three tasks and three action streams (288 episodes). The script checked all
these scene seeds against the BC data seeds. None is a registered or sealed
Warehouse seed block.

| Check | robot_1 | robot_2 | Required development value |
| --- | ---: | ---: | ---: |
| Four-part validation feedback actionable coverage | 26.77% | 24.48% | ≥30% each |
| Four-part validation explanation actionable coverage | 26.77% | 24.76% | ≥30% each |
| Exploratory audit feedback actionable coverage | 24.24% | 22.19% | ≥30% precheck each |
| Four-part validation feedback agreement | 100% | 98.02% | ≥93% |
| Four-part validation explanation agreement | 100% | 98.02% | ≥95% |
| Four-part validation KL | 0.0099 | 0.0086 | ≤0.25 |
| Illegal actions in selected program rows | 0 | 0 | 0 |

The independent 288-episode BC pressure mean was 14.747 deliveries, 5.910
collisions, and 0% shutdown episodes. Task 3 alone averaged 6.5 collisions,
so the aggregate collision margin is small. These are **BC-only** results;
they say nothing about quality after the required PPO self-play.

Eligibility diagnosis on the already used eligibility episodes found that
coverage failure is concentrated in high-traffic, mixed leaves at depth 5:

| Role and leaf | Eligibility mass | NN/program top-1 agreement | KL | Raw illegal program actions | Nearby robots | Tree/NN top-1 agreement | Tree margin / gate×bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| robot_1 leaf 9 | 49.4% | 44.3% | 0.850 | 807/1482 | 49.9% | 40.8% | 0.248 / 6.767 |
| robot_2 leaf 13 | 30.0% | 41.8% | 0.869 | 153/900 | 57.1% | 42.1% | 0.133 / 6.645 |
| robot_2 leaf 14 | 19.9% | 59.5% | 0.573 | 416/597 | 35.2% | 59.5% | 0.569 / 6.116 |

The `raw illegal` counts describe **rejected** leaves; selected leaves had
zero illegal actions. Once a leaf qualified, sampled-action mismatch reduced
coverage only from 27.67% to 26.77% for robot_1 and from 26.25% to 24.48%
for robot_2. Thus the main bottleneck is routing and Actor/tree disagreement,
not action sampling. The diagnostic reused eligibility episodes and is
descriptive, not another independent audit. Its JSON SHA-256 is
`14e3e052f2ba0eeb57067ce4cdd7512c15b4ba45b4bd7e0c4ccc720097d0c4a0`.

The necessary 79 source-closure files, frozen Actor/state, raw reports,
programs, scripts, and logs are preserved in the ignored snapshot
`output/warehouse_v6_repro_snapshots/warehouse-v65-10965-unregistered-20261016/`.
The 95-file, 5,801,257-byte inventory was checked file by file; its
`archive_manifest.json` SHA-256 is
`f41f385bf9dc4fbecfedb99634e87bd4381bcc8cc9ae1b8d4718863d061c0ce4`.
No old failure checkpoint or sealed data was moved into release assets.

Next exploratory hypothesis: expand legal sparse directions to three public
features, propose thresholds by row-weighted quantiles, and use bounded
lookahead to allocate the 16-leaf budget to these high-mass mixed regions.
Maintain the full NN masked distribution as the fitting target. Separately
train the intrinsic Actor margin on the resulting stable regions without
suppressing its residual on complex states. This could exceed 30% for both
roles because the three rejected leaves alone contain about half of all
eligibility frames per role; it is a testable hypothesis, not an acceptance
claim. A new seed lock will precede v6.6 evaluation.
