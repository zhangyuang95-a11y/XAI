# Warehouse v6.13 exploratory core refit failure

The unregistered v6.13 candidate corrected the batched role-logit error from v6.12, then ran one locked Stage 1 from pair seed 12369. Both roles passed frozen-warmup target provenance checks, but the episode-separated sparse core refit failed its internal validation. The process stopped before the collision critic, final BC, frozen first-quality evaluation, pressure, program fit, PPO, or any registered or sealed run. This is a model-method failure, not an execution-path acceptance result.

## Locked identity and preservation

| Item | SHA-256 |
| --- | --- |
| 126-file source closure | `2e9242cb636ce18cb35745f482427b2c8d2532ac65fce297fe903248814d2cba` |
| Training recipe | `8dd593309d88182cc36e8745ca118efdd452b08101a1d8228dc96882a0b02ee0` |
| Source snapshot manifest | `d36959f5e7e02b972b56575983a030113ac7286a0d453a7ca5020c1b53cf3573` |
| Predeclared run | `f36001b6ee017a7f42f2d562a9d79dd7db4e60f2d262df7d5c268df8e25c892b` |
| Seed lock | `6d8ad5e00ea86325fd9f1d81c1e63c0ce96d9a3bc2b26325446b211013fdbcf0` |
| Script lock | `38c60dd34e6e20a1574f3af2a86773719a0e62d51a437302a469388fa83fd9b1` |
| Stage 1 script | `9f4ef80352e8a57d52055e85937edefee7492c16f5ef715ba3975c28e191fba5` |
| Zero-reset independent lock audit | `0d87ac45d8c0f2293be842808f6bc17a4a5b4e50531a9ff8f77fe2492bf2f556` |
| Frozen warmup dual Actor | `714a2ab82db3dcd59ab779d89c79d55e76f9fb8a4db4d53e3d1ab0cccc262c3d` |
| Topology failure detail | `d40db397e936ea9f3c6f3ba542d8c130a61e436f7d2c2fdcfddb8dbd29476cde` |
| Process failure receipt | `9e3c5f758c6b2e28b869def6a98b965335734aa44dbf4ad0cc6268c9f43f9af3` |
| External archive manifest | `9d185ba7450b8b77caeb4aa066077a3e645b421c02fc20427e3edcbf9eab8f13` |

The original package is at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration/output/warehouse_v6_repro_snapshots/warehouse-v613-role-parity-12369-20261016/`. A byte-identical external copy is at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v613-role-parity-12369-failed-20261016/`; all 144 source-package files, totaling 4,613,044 bytes, were compared by SHA-256. The external `archive-manifest.json` records each file.

## Exact seed use and stop point

The only environment resets were 306 BC episodes: initial public-teacher 100 (`1236910000–1236910099`), two NN DAgger blocks of 40 (`1236911000–1236911039`, `1236912000–1236912039`), and eight distinct sampled/argmax topology blocks totaling 126 (`1236960000–1236960024`, `1236961000–1236961024`, `1236962000–1236962012`, `1236963000–1236963012`, `1236964000–1236964012`, `1236965000–1236965012`, `1236966000–1236966011`, `1236967000–1236967011`). The ordered consumed-stage schedule has SHA-256 `86a318b62e00f7a1c819e39e5efafbca0665c083227014d8ac92d9aced5ae965`. The warmup Actor was exported, but no final BC Actor or first-quality result exists. The critic seed 18569 and every other predeclared 12369 block are retired without use.

The four topology purposes were episode and scenario separated. Both roles' fit, calibration, eligibility, and validation targets were recomputed from the same frozen warmup Actor; recorded target parity error was zero. This establishes the target's provenance for this failed experiment only.

## Failure measurements

| Role | Fit rows / leaves | Eligibility locked core | Independent validation core mass | Expected feedback actionable mass |
| --- | ---: | ---: | ---: | ---: |
| robot_1 | 4,579 / 14 | 0 leaves | 0% | 0% |
| robot_2 | 4,579 / 13 | 2 leaves | 11.97% | 11.84% |

The frozen-warmup masked-distribution tree produced only two fit core candidates for robot_1 and three for robot_2. Robot_1's candidates both failed eligibility's predeclared low-return or conflict stratum support/fidelity checks; its final core mask was empty. Robot_2 retained only two leaves, with independent validation coverage far below both the 35% internal target and 30% actionable development requirement. Large near-robot and ordinary leaves remained mixed. The fit-only best core mass was 16.77% for robot_1 and 14.30% for robot_2, so merely relaxing sparse-stratum treatment would not provide the required coverage margin.

The v6.13 route is closed. Subsequent development requires a new method, new source and recipe hashes, and fresh unregistered seeds. The program, execution, formal audit, heldout, and pilot gates remain closed.
