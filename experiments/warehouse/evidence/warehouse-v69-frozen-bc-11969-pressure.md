# Warehouse v6.9 frozen BC: unregistered pressure evaluation

Status: **the predeclared exploratory pressure screen passed; v6 model admission and pilot remain closed.** This evaluates the single frozen dual Actor produced by the [11969 full BC first screen](warehouse-v69-full-bc-11969-first-quality.md). Raw/regularized self-play PPO, role programs, registered development pairs, independent program audit, and sealed heldout testing have not run for this model.

## Locked run

- Source closure SHA-256 `8569cdf9dcca314d5ec738a62c962bbbc51ea5c3125b3f37677ebb0b6b330be9`; frozen dual Actor NPZ SHA-256 `c4080edeaff2b633fbb274db709a0685ce70692978e5fc710c22c8553792584f`.
- Pressure schedule: 32 new base seeds `1190100–1190131` × three tasks × three independent NN action streams (`+0`, `+1000000`, `+2000000`) = 288 trajectories over 96 distinct environment scenario seeds. Schedule SHA-256 `0fc54063f3d94873765c6d02d3eef4f8d29199bf7ffdd65d2777c6583af01c72`. These scenario seeds do not overlap the BC data or earlier 96-episode screen.
- Predeclared configuration SHA-256 `e4002221f6a76a0ac2dc9cdfb99c46fc7e0bee609095f658dcbbcd6384e2dfd7`; frozen-only script SHA-256 `31e30c52cfafc8240ee397dfc1300e0fb0187b7a50be0745a85d46d3e8879dfc`; script-lock SHA-256 `bc0077e0073ea77581bf109ea21c9f18abdfa6c00bef09ed3647a56e051ea1d2`.
- The run used `python3 -I` from a neutral working directory and the pinned isolated source. Both actions at every frame came from `FrozenWarehouseV6Actor.act`; the environment then applied its physical rules. No program or teacher selected an executed action.

## Outcome

The predeclared overall screen required mean deliveries ≥5.5, mean robot collisions ≤6, shutdown episode rate ≤5%, zero illegal executed actions, zero post-policy action overrides, and complete independent replay of saved quality traces. All six gates passed.

| Set | Trajectories | Mean deliveries | Mean collisions | Shutdown episodes |
| --- | ---: | ---: | ---: | ---: |
| Overall | 288 | 13.5417 | 4.6285 | 6 (2.083%) |
| Task 1 | 96 | 13.6250 | 4.4792 | 3 |
| Task 2 | 96 | 13.5417 | 5.0000 | 2 |
| Task 3 | 96 | 13.4583 | 4.4063 | 1 |

The three action streams had 3, 1, and 2 shutdown episodes. Every one of the 288 saved trajectories passed deterministic frame-by-frame replay against the frozen NN and physical environment. Illegal actions and action overrides were zero. Tail risk remains visible: one trajectory repeated a colliding joint action for 17 frames; across all runs, 94 frames belonged to such streaks of at least four, and 782 collisions occurred at frame 60 or later. The three streams share each environment scenario seed, so 288 trajectories should not be described as 288 distinct scenarios.

## Receipts and recovery

The original ignored pressure directory is `output/warehouse_v6_repro_snapshots/warehouse-v69-pressure-11969-20261016-v1/`. The final result SHA-256 is `dc0dc7ab929d0f1438e60159de24cf8e125022cd616436c90c24df13327db43d`; raw 288-row JSONL SHA-256 `c85e53e9c4947f3301d38f6d0ec30e44f52ec203b58004c36fdb50ee4e32241e`; independent replay JSONL SHA-256 `a126dc4ae2d2726deb827055e27b4ba7dba58dcb14002f2f2d665ec7c5b30968`; run log SHA-256 `5b15b6f33df33824246c1bd8a65765ac737e6634ee30e4498bfe74515560e984`.

The selected pressure manifest SHA-256 is `a94489f4fb1faf8a36deeb2eaf2741e5cb69c875a09c5eed7b5ca1292476d826`. An external recovery copy at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-pressure-11969-20261016/` contains the ten selected pressure artifacts plus the manifest; all 11 files (525,437 bytes) were checked against source and destination SHA-256. Its verification receipt SHA-256 is `c512efc46d2488e66bd601ed478d79840a17406001f6677000902d29072dcd36`. The linked BC recovery directory contains the frozen Actor and all 121 source files.
