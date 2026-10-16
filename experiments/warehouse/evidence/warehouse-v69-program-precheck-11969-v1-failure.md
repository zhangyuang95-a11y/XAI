# Warehouse v6.9 frozen Actor program precheck: v1 failure

Status: **exploratory script failure; no program metric was obtained.** The v6 pilot remains closed. The frozen dual Actor was the 11969 BC artifact (SHA-256 `c4080edeaff2b633fbb274db709a0685ce70692978e5fc710c22c8553792584f`), already assessed in the [first quality](warehouse-v69-full-bc-11969-first-quality.md) and [pressure](warehouse-v69-frozen-bc-11969-pressure.md) screens. This precheck did not train or execute a program, run PPO, or use registered/formal/sealed seeds.

The preregistered exploratory method was `actor_sparse_topology_v1`: copy each frozen Actor's public sparse tree, fit leaf distributions to the complete masked NN distribution, then use separate calibration, eligibility, and validation episodes. The v1 schedule locked 124 base seeds `1190200–1190323`, three tasks per base, as fit 50 / calibration 25 / eligibility 25 / validation 24. Schedule SHA-256: `025980d1bcc31cc1c4354874ad89b82a37811fd0d515fb7a2bf83c1174d5efb4`.

The one authorized run collected **all 150 fit scenes** (50 base seeds × three tasks). Before writing the first trace, its extra RNG provenance check converted the Actor's float32 probabilities to float64 and passed them to NumPy `Generator.choice`. The stricter float64 sum tolerance raised `ValueError: Probabilities do not sum to 1`. The Actor itself had sampled with the original float32 probabilities. The run exited 1; no trace, program artifact, result, validation, or coverage number exists. Every v1 fit scene is considered consumed, and all four v1 splits are retired. The correction is a newly locked v2 script and entirely fresh four-part seeds, with no selective rerun.

Identity and recovery:

| Artifact | SHA-256 |
| --- | --- |
| BC training source closure | `8569cdf9dcca314d5ec738a62c962bbbc51ea5c3125b3f37677ebb0b6b330be9` |
| Program precheck source closure | `1430c59278e28a672c636682225bb94d9ecfae655ccb937c26f6cee0df14642e` |
| Program source snapshot manifest | `cca8bc9542fc76eeb5613ec033273872674bad672ba739da350cd97f563dfc2d` |
| Program predeclared JSON | `3f35356e1b0a2b560504400f14a85c2142b70da0d80e29ab71306e8f83756cd1` |
| Program script | `8c8b028b55c34a11e8cf74e87b021c52f08e61b9bd3664ef772d190ed06a810d` |
| Script lock | `96a06047979ddd16d00970d513a14b0375c9b7f9f8c7ad1805abad58d1c72aaf` |
| Isolated `--validate-lock-only` output | `557f46ce83df1d23b9f97957e23c5578c48044b46b0a7c74de4ed2984328831f` |
| Failed run log | `91e7f09308fbd4f8a21dee050d411aed77bdf8923d4e45dce93e55afbc990a96` |
| Failure receipt | `876834ebf5617b7f150846a04c49ec26d088299b74cd2e8e882f7051f90e82a3` |

The original ignored directory is `output/warehouse_v6_repro_snapshots/warehouse-v69-program-precheck-11969-20261016-v1/`. A separate recovery copy is at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-program-precheck-11969-v1-failed-20261016/`: nine selected run artifacts plus all 121 source files (130 files, 2,706,868 bytes) were copied and individually checked against both locations. Its `archive-manifest.json` SHA-256 is `4fa1259a194123701d45cc78bd73c15b45ca7b764ae6ac7d4b44b7c3fad5aced`. The v1 source, script, locks, log, and failure receipt were left unchanged.
