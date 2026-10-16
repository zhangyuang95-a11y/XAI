# Warehouse v6.9 frozen Actor program precheck: v2 coverage failure

Status: **exploratory four-part program precheck failed; pilot remains closed.** This is the same frozen dual Actor from the 11969 BC quality and pressure screens, not a PPO-trained raw/regularized pair. No registered, formal, or sealed evaluation seeds were used. The program never selected an environment action.

The v1 script failed after consuming its 150 fit scenes; its [failure receipt](warehouse-v69-program-precheck-11969-v1-failure.md) and all four v1 seed blocks remain retired. V2 used an independently locked script and four entirely fresh blocks: fit `1190400–1190449`, calibration `1190450–1190474`, eligibility `1190475–1190499`, validation `1190500–1190523`, each with tasks 1–3. The original frozen Actor and 121-file program source snapshot remained unchanged. The `actor_sparse_topology_v1` extractor copied each role's public sparse Actor topology and fit/calibrated leaves against its **complete masked NN action distribution**. Only eligibility episodes locked the leaf sets; validation did not remove or select leaves.

| Role | Validation frames | Feedback eligible leaves | Leaf coverage | Feedback actionable | Explanation actionable | Agreement in eligible leaves | KL in eligible leaves | Raw illegal in eligible leaves |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| robot_1 | 8,547 | 4, 8 | 11.548% | 11.326% | 11.337% | 99.189% | 0.02194 | 0 |
| robot_2 | 8,547 | 1, 9 | 11.899% | 11.677% | 11.688% | 98.820% | 0.03301 | 0 |

Both roles fail the locked 30% development actionable and leaf-coverage thresholds; they also fall below the 20% formal coverage threshold. Agreement, KL, and physical legality are good **inside the selected leaves**, but those leaves cover too few frames. All four development feedback/explanation pass flags are false. The run produced 17,922 fit, 8,982 calibration, 8,918 eligibility, and 8,547 validation joint pre-action frames. Both roles' sampled NN actions, saved probabilities, RNG reconstruction, action legality, and role hashes had zero mismatches. The raw program action legality rule correctly rejected otherwise promising leaves.

## Leaf diagnosis

The table reports frozen leaf visits in the order fit/calibration/eligibility/validation, followed by the **eligibility** agreement with the complete masked NN, mean KL, and number of illegal *raw* program top actions. `selected` means the leaf passed the fixed eligibility checks. All other leaves failed at least one check; the complete, potentially overlapping rejection reasons, per-action confusion counts, and per-leaf metrics are in the archived `leaf-diagnostic.json`. No diagnostic output changed the fixed leaf set or validation result.

| Role | Leaf | Four-part visits | Agreement | KL | Raw illegal | Outcome |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| robot_1 | 0 | 4107 / 2086 / 2136 / 2073 | 47.7% | 1.048 | 926 | top-1, KL, legality fail |
| robot_1 | 1 | 2274 / 1080 / 1099 / 1090 | 57.6% | 0.910 | 0 | top-1, KL fail |
| robot_1 | 2 | 889 / 451 / 432 / 398 | 88.9% | 0.328 | 19 | top-1, KL, legality fail |
| robot_1 | 3 | 768 / 375 / 377 / 342 | 85.1% | 0.295 | 118 | top-1, KL, legality fail |
| robot_1 | 4 | 1119 / 524 / 570 / 509 | 99.6% | 0.014 | 0 | selected |
| robot_1 | 5 | 1818 / 924 / 838 / 818 | 55.1% | 0.689 | 21 | top-1, KL, legality fail |
| robot_1 | 6 | 1547 / 797 / 750 / 707 | 69.1% | 0.604 | 33 | top-1, KL, legality fail |
| robot_1 | 7 | 598 / 292 / 253 / 256 | 43.5% | 0.654 | 0 | top-1, KL fail |
| robot_1 | 8 | 1083 / 540 / 492 / 478 | 99.8% | 0.028 | 0 | selected |
| robot_1 | 9 | 765 / 402 / 443 / 385 | 66.1% | 0.625 | 0 | top-1, KL fail |
| robot_1 | 10 | 0 / 0 / 0 / 0 | — | — | 0 | no fit support |
| robot_1 | 11 | 1264 / 671 / 638 / 672 | 66.8% | 0.686 | 0 | top-1, KL fail |
| robot_1 | 12 | 208 / 121 / 94 / 119 | 57.4% | 0.858 | 24 | top-1, KL, legality fail |
| robot_1 | 13 | 1350 / 652 / 709 / 631 | 45.0% | 0.827 | 353 | top-1, KL, legality fail |
| robot_1 | 14 | 132 / 67 / 87 / 69 | 57.5% | 0.590 | 0 | top-1, KL fail |
| robot_1 | 15 | 0 / 0 / 0 / 0 | — | — | 0 | no fit support |
| robot_2 | 0 | 3722 / 1787 / 1745 / 1667 | 31.5% | 1.036 | 0 | top-1, KL fail |
| robot_2 | 1 | 1653 / 882 / 829 / 745 | 97.5% | 0.074 | 0 | selected |
| robot_2 | 2 | 33 / 18 / 19 / 22 | 100.0% | 0.003 | 0 | under 20-row support |
| robot_2 | 3 | 1126 / 559 / 593 / 562 | 87.2% | 0.218 | 246 | top-1, legality fail |
| robot_2 | 4 | 472 / 225 / 232 / 213 | 91.8% | 0.270 | 0 | top-1, KL fail |
| robot_2 | 5 | 1295 / 610 / 656 / 570 | 98.5% | 0.064 | 2 | legality fail |
| robot_2 | 6 | 1820 / 899 / 888 / 906 | 56.5% | 0.688 | 26 | top-1, KL, legality fail |
| robot_2 | 7 | 1521 / 780 / 758 / 756 | 66.1% | 0.645 | 14 | top-1, KL, legality fail |
| robot_2 | 8 | 937 / 477 / 466 / 449 | 71.5% | 0.481 | 0 | top-1, KL fail |
| robot_2 | 9 | 498 / 279 / 261 / 272 | 99.6% | 0.016 | 0 | selected |
| robot_2 | 10 | 1672 / 817 / 839 / 821 | 61.5% | 0.694 | 0 | top-1, KL fail |
| robot_2 | 11 | 271 / 158 / 178 / 161 | 74.7% | 0.636 | 0 | top-1, KL fail |
| robot_2 | 12 | 1597 / 826 / 802 / 749 | 48.4% | 0.845 | 0 | top-1, KL fail |
| robot_2 | 13 | 622 / 335 / 318 / 328 | 48.7% | 0.806 | 65 | top-1, KL, legality fail |
| robot_2 | 14 | 0 / 0 / 0 / 0 | — | — | 0 | no fit support |
| robot_2 | 15 | 683 / 330 / 334 / 326 | 51.8% | 0.719 | 0 | top-1, KL fail |

The largest unqualified leaves contain several different NN top actions. For robot_1 leaf 0, the NN's eligibility top actions were WAIT 716, RIGHT 482, DOWN 428, UP 382, LEFT 128, while the masked program top action was DOWN 1,210 or WAIT 926; agreement was 47.7%. For robot_2 leaf 0, the NN top actions were WAIT 550, UP 475, DOWN 324, LEFT 205, RIGHT 191, while the program chose WAIT for all 1,745 frames; agreement was 31.5%. This explains why copying the existing Actor topology and recalibrating only leaf logits cannot make those leaves eligible. Robot_2 leaf 5 would add 7.36 eligibility percentage points if its raw action were always legal, but its two raw illegal actions require rejection; even that hypothetical addition would leave coverage far below 30%.

Across the 8,918 eligibility rows, the rejected-leaf mass tagged for top-1 disagreement is 7,856 rows for robot_1 and 7,153 for robot_2; KL failure tags 7,856 and 6,560. These categories overlap and must not be added. Low support is negligible (robot_2 leaf 2 has 19 rows); sampled-action mismatch is also small in the selected leaves, as shown by the less than 0.3 percentage-point drop from leaf to actionable coverage. The structural problem is high-traffic leaves whose full NN distribution is too mixed for one sparse leaf distribution, sometimes with changing physical masks.

## Next development implication

The next exploratory method should change the **trainable Actor/tree relationship** on public pre-action states: keep the complete dual NN as the only action source; train its high-traffic, learnable tree regions and residual gate/bound jointly so the complete masked distribution is stable within those regions, while retaining the 256-wide residual for complex states. Continue quality optimization from public BC labels and real environment PPO/cost events, with raw and regularized variants sharing the same architecture and BC origin. Extract a separate role-specific program from the final complete NN distribution using fresh, disjoint fit/calibration/eligibility/validation episodes. Check leaf and actionable coverage for both roles together with multi-trajectory deliveries, collisions, and shutdowns; never use this spent v2 validation block to tune or requalify leaves. This is a design hypothesis, not a passing result.

## Identity and recovery

| Artifact | SHA-256 |
| --- | --- |
| Frozen dual Actor NPZ | `c4080edeaff2b633fbb274db709a0685ce70692978e5fc710c22c8553792584f` |
| BC training source closure | `8569cdf9dcca314d5ec738a62c962bbbc51ea5c3125b3f37677ebb0b6b330be9` |
| Program precheck source closure | `1430c59278e28a672c636682225bb94d9ecfae655ccb937c26f6cee0df14642e` |
| Program source snapshot manifest | `cca8bc9542fc76eeb5613ec033273872674bad672ba739da350cd97f563dfc2d` |
| V2 scenario schedule | `be00ac6b07518c1fdc568194e34a9092cbb0704accded9b5b5a9e05d5e14620e` |
| V2 predeclared JSON | `f32c297b94cc3fb407bff7da825d3cddbdeb184f7df4e5a88f96b9445ffae64b` |
| V2 script | `104d31cf6aac2019e9bbf3efabe82b8f6cb63f14fc0f60967e9296940cd3e012` |
| V2 script lock | `02aed0a09985883404c55ae22a7c48007438c2510f473ba42729d28f5e274999` |
| Raw result JSON | `2dffdfc602ebf4903e2df20c6d4e1cfc920a5717023eb206cffce25ca490ce19` |
| V2 run log | `ecfc4fb9d5cf86ae52dd14e14a89870e80c8be3a766d7bb2fa7082ec1dfbee5e` |
| Robot 1 / robot 2 program | `6aca2ecc95256a9cb6711bf66981acc81a79b7643b0fed97259dca8774a24bb4` / `c83c6cb824c0a94d8ec5a6689318f85afd774aa06919df91f577079ee4a7c147` |
| Derived leaf diagnostic script / JSON | `49902378846ef379bd009e23e1ae268120e985233b9ab7a202724befd5304835` / `f5c4da138065d443919808ce2b7bb0cff55740b985548937b971bdc05a604a2c` |

The original ignored V2 directory is `output/warehouse_v6_repro_snapshots/warehouse-v69-program-precheck-11969-20261016-v2/`. The independently copied recovery package is `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-program-precheck-11969-v2-failed-20261016/`: 141 files, 241,312,444 bytes, including four raw joint JSONL traces and all 121 source files, each bytewise SHA checked. Its `archive-manifest.json` SHA-256 is `f5113979222c80552ed30cf3cf5df4f258ed5ca06b7889702cd1e8fe8133b310`. The derived diagnostic was copied separately, leaving that archive immutable, to `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v69-program-precheck-11969-v2-leaf-diagnostic-20261016/`; its two files were SHA checked and `diagnostic-manifest.json` has SHA-256 `0309c5072deed13c0fc81c0fc315c7d31d9412c88f0ac040e17f366921eb15cc`. Original program artifacts, traces, locks, and receipts were not changed.
