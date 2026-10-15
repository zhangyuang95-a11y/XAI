# Warehouse v6.7 trainable soft route: hard-tree CV failure

This is an **unregistered exploratory screen**, not a new paired BC/PPO
model, independent four-part program validation, final audit, or held-out
result. The v6 pilot remains closed. The [prelocked schedule](warehouse-v67-joint-actor-seed-lock.json)
used `1130000–1130049` only to fit and `1130050–1130074` for fit-only
internal CV. It also reserved `1130100–1130123 × 3 tasks` as a diagnostic
quality block; the fresh pair `11368` and its `114xxxx` confirmation blocks
were **not used**. No registered or sealed seed was used.

The source is the unchanged v6.5 frozen BC dual Actor (artifact SHA-256
`bd2f1f63502589ec8261aeed2927955a51c64e759513b3c5fda23477f747e8b9`,
source closure `e2b35bab3f589bfabb410214c1af2089d9f596746d48bb15419988c5967f8686`).
The 75 source episodes yielded 9,000 public pre-action rows; row NPZ SHA-256
`a8a25cd6b2e27781011656e006f03991b58df4bec68022a23704bfa8d66e1643`.
Restoring this Actor into Torch reproduced its saved masked action
probabilities on all rows within `3.04×10⁻⁶`; frozen NumPy differed by at
most `1.19×10⁻⁷` from the recorded probabilities.

The prototype replaced each role's frozen split with a trainable shadow
split. Its **hard forward and export** projected every used node onto at most
three public pre-action features, with depth at most five and 16 leaves.
Only the complete frozen NN masked distribution trained the split; public
teacher labels, environment outcomes and program outputs did not. Separate
synthetic tests showed teacher/quality gradients into the split were absent,
and the 128-fit-row, two-step real smoke had nonzero split gradients. Thirteen
synthetic tests passed. These are mechanism checks, not quality evidence.

The one-shot structure screen fixed 500 optimizer steps, batch size 512,
Adam split/leaf rates `0.001/0.003`, route temperature `0.5`, and equal hard
straight-through/soft-mixture masked KL weights **before** running. Its
script SHA-256 is
`1103abf15ba57454a44f761f58e7ead9cd4b72aa87d7931778aca5d5f024395c`.
It never inspected the CV results to retune these values.

| Fit-only internal CV measure | robot_1 | robot_2 |
| --- | ---: | ---: |
| Qualified-leaf row mass | 5.27% | 0% |
| Expected sampled-action actionable coverage | **5.18%** | **0%** |
| Complete NN/tree top-1 agreement | 54.67% | 55.90% |
| Masked KL | 0.963 | 1.077 |
| Raw illegal program actions | 1,164 | 1,276 |
| Frozen Torch/NumPy leaf or masked-action mismatches | 0 | 0 |

The soft-route fit KL fell (`1.085→0.956` and `1.093→0.889`), while the
**actual hard-route** fit KL rose (`0.543→0.945` and `0.530→1.084`). The
surrogate moved rows into poorly calibrated/illegal hard leaves. This is a
failure of the method's hard execution objective, not a Torch/NumPy export
bug. The result is too far below both the 30% development gate and desired
40% margin to justify the 17–25 minute from-scratch BC probe. Receipt SHA-256
is `7d8881a1fd413833768e226fccfe6e6d5d55f185a88ed9698c524ac6772d7076`.

On the diagnostic source Actor alone, 72 episodes averaged 14.625 deliveries,
7.042 collisions and zero shutdown; task 3 averaged 12.417 collisions. Two
seeds exposed a tail failure: `11301113` repeated the same NN-proposed
`UP+LEFT` same-target collision at `(5,2)/(4,3)` for 103 frames, and
`11301073` repeated it for 70 frames. The relevant Actor action probabilities
were about `0.986/0.997`, with `action_override_agents=[]`. This is a
high-confidence learned joint-action loop, not an environment controller
choosing AI actions. The compact public-frame trace SHA-256 is
`38dd704ff97c142595226ee73ac5274c547a3a7795fd00f4c53fe1017e1f4bc6`.
No post-update quality result is claimed because the topology screen failed.

The exact 23 files (3,492,793 bytes) containing the source Actor, rows,
scripts, losses, tests, receipts, quality diagnostic and public traces are in
ignored `output/warehouse_v6_repro_snapshots/warehouse-v67-soft-hard-11367-20261016/`.
Every file was verified against manifest SHA-256
`b4ddf4ca8755cdeff59b4b8cdefe49f8c1f89e694ef71c46772670abd77083ef`.
The next candidate must use fresh unregistered fit/CV and pressure data. A
possible route is to fit an exact hard topology to the complete NN masked
distribution, train the full Actor in selected public states while preserving
quality, then recollect and **refit hard topology**. It requires its own
evidence; this failed soft-route experiment does not establish feasibility.
