# Warehouse v6.9 exploratory short BC screen: stopped

This is an unregistered method screen, not a v6 admission result. The predeclared continuation gate failed, so no 117 topology fit or CV, 118 confirmation, registered pair, independent audit, or heldout test was run. The pilot remains closed.

## Reproducible identity and seed use

- Isolated source snapshot: Git commit `db7f156670dd16687d749037a213fdd6dd132bda` (includes `1dc4c20` and public history `7b76c57`); 73-file source closure SHA-256 `7efba527d09cb595d67bf400369b194af92f16e8b5dd9ede57fa84c8f325e602`.
- Pair initialization seed: `11769`. Public teacher BC used six episodes `1170000–1170005`; sampled dual-NN DAgger used four episodes `1170006–1170009`. The Actor started from random initialization, with a public-label sparse tree initialization, five BC epochs and three DAgger epochs. These ten seeds were assigned to the 117 exploratory fit block and **cannot now serve as independent topology-fit data**.
- Frozen quality used bases `1170100–1170103` with scenario/action seed `base × 10 + task`, for tasks 1, 2, and 3: 12 episodes total. The first episode (`11701001`) was executed twice because the original script failed while serializing its diagnostic log; the second run used the identical exported Actor bytes. No other scenario was repeated.
- Predeclared configuration SHA-256: `44001e76c78515c42795e248d53ccc69b03d14d462a10ad343d23c1a5f08ba73`. Original training script SHA-256: `5da67bc23f0ab7d9a1d30fbbe71bc7f4c752c00b5ac0a4b8c53119ef71f98d0f`. Frozen dual-Actor NPZ SHA-256: `4caebbe638af9274b7eb3a217983c62ddf61870f6808a8638d78b6a152584c62` (role hashes `robot_1`: `a61661c31339c07459792cefe3635a70b55bc463316b6b0eaaebcb460b90b2bd`; `robot_2`: `d6bc52aa44dd80de8d0b06ea5b9d2b770912f676209a82397d1b5488d81a3ee2`).
- Evidence directory: `output/warehouse_v6_repro_snapshots/warehouse-v69-screen-11769-20261016/` (ignored, retained locally). Its selective artifact and complete source-file SHA manifest is `selective-manifest.json`, SHA-256 `a1543695dca82c3cda628d86a4798a9145c19483a97f48dff6ffc6e46f993872`. The result receipt `result-v2.json` SHA-256 is `1eda64c7db4b9a329b738bcb1e877a60152559640543f16c8acf109bc66a6502`.
- Independent recovery copy: `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-screen-11769-20261016/`. All 10 selected artifacts (926,783 bytes) and the same manifest were copied and rehashed against that manifest; original source files remain recoverable from the pinned Git commit and the manifest's 73 per-file digests.

## Predeclared screen result

Both role-specific sparse tree splits received real gradients on the first BC batch (`0.08145`, `0.10514`) and changed weights (`L2 0.02938`, `0.03332`). The exported frozen NumPy Actor matched Torch on 64 sampled public rows (maximum valid-logit absolute difference `4.77 × 10⁻⁷`, matching masked argmax). No diagnostic frame used an illegal action or post-policy action override.

The principal learning gate failed: mean teacher-action probability on the same BC rows moved from `0.65474` to `0.65633` for robot 1 and from `0.63915` to `0.64206` for robot 2. The gains (`0.00159`, `0.00291`) were below the predeclared `0.03` requirement. NN-visited teacher-action probabilities after DAgger were `0.37063`, `0.38664`; those cleared the screen's loose `0.25` diagnostic threshold but did not rescue the failed fit gain.

The frozen NN–NN diagnostic quality also failed the short screen's shutdown gate: 12 episodes averaged `1.083` deliveries and `12.25` robot collisions, with shutdown in `11/12` episodes (`91.7%`, versus the predeclared maximum `50%`). Quality by task was:

| Task | Episodes | Mean deliveries | Mean collisions | Shutdown episodes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 1.00 | 12.75 | 3 |
| 2 | 4 | 0.75 | 9.00 | 4 |
| 3 | 4 | 1.50 | 15.00 | 4 |

The Actor still encountered the previously identified collision tail: one episode had 20 consecutive frames of the same position and joint action causing collisions; across the 12 episodes, 32 collision frames belonged to such streaks of length at least four and 33 collision frames occurred from frame 60 onward. These are descriptive diagnostics, not independent quality estimates.

## Recovery and decision

The original script reached frozen Actor export and Torch/NumPy parity, then failed on the first quality-log event because `json.dumps` could not serialize `numpy.int64`. Its unchanged script, log, failure receipt, and progress receipt are in `v1-failed-serializer/` (original log SHA-256 `fd46b2909fc2251a344009c190232ac500cb745980554bc2f3995d8fa242f4f6`; failure SHA-256 `b20cf41521b4f337f19f576813fb74425bfdf26a723f2125477c73c8ab292c8e`). A separate quality-only script, SHA-256 `cc86cc85202ef1581a9b79b22f9b627676156f95945edfd62e09639ef4a684b1`, converted NumPy scalar log values and loaded the exact exported Actor; it did not rerun BC. Its log SHA-256 is `0b75d1f331b899c217e34cf889b189ac9cd721874f813073d7118c9f8c70b17d`. The 12-episode resumed quality pass took `3.74` CPU seconds; the initial script did not record its CPU seconds before the serializer error (export was reached in `5.68` wall seconds).

**Decision:** `continue_to_method_design = false` for this screen configuration. The 117 CV seeds `1170050–1170074`, all 118-series seeds, registered development/formal pair seeds, independent audit seeds, and heldout seeds remain untouched. The source snapshot and failed receipt remain available for method redesign; these results do not qualify any Warehouse model or open v6 pilot.
