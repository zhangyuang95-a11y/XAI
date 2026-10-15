# Warehouse r4.1 fixed-yield evaluation alias diagnostic

Status: **confirmed validation-partner alias**. This diagnosis was produced after the fact and did not modify the running trainer, any checkpoint, Actor, extracted program, scene, or frozen training source. The machine-readable evidence is in `warehouse_r41_fixed_yield_alias_diagnostic_20260911.json` (semantic SHA-256 `32aef2abd78d4cb127c20dd06b1513b26743280beffc69d06dbaa25a7746c871`; file SHA-256 `340cd5c2affcb6a0dcf0076c294dbf33e3f215c89ba26001ae5e86fd6912848d`).

## Finding

The recorded dual evaluation names six partners, but `skilled` and `fixed_yield` are the same behavior in this evaluation role. Both evaluators always request a program action for `robot_1` and deploy the neural candidate as `robot_2`. The only `fixed_yield`-specific guard in `env/warehouse_native/partners.py` applies when `agent_id == "robot_2"`. For `robot_1`, both names therefore use the same symmetric task assignment, goal construction, and joint route.

The following eight registered comparisons remove only the labels `partner` and `seed`, pair rows by scenario, and compare every remaining field.

| Actor boundary | Suite | Paired scenes | Exact core-row matches | Differing fields | Core-row SHA-256 |
|---|---:|---:|---:|---|---|
| r3 baseline | original | 50 | 50 | none | `037f02b7dc96b1c0c90b53e01b04cd4374febd62a918afd3c49ade27a3962680` |
| r3 baseline | conflict | 64 | 64 | none | `c2793e9891f557ce1273545a0b059966e7e27e1a346912463aab6b8b1d558a04` |
| 50k | original | 50 | 50 | none | `ab0c5058a295ffc58107d9d98bbfaac48422d54d8f4d3563e3d284689262d01c` |
| 50k | conflict | 64 | 64 | none | `180cf99bdb8cfc0527072eb742d72e73fe3f918085576d660e0c6434b8501830` |
| 100k | original | 50 | 50 | none | `22755ce12ba94ca914b21e6df670f5d87662f87da4010ed4343b50d667780c11` |
| 100k | conflict | 64 | 64 | none | `e284d59c1e7848c5a0d1f6128711acf767079d943ecd8915853c1470f02d1612` |
| 150k | original | 50 | 50 | none | `7db9a7e9436e0e92683f0d953367e1a58ec1db96ef2c4d73d1ee0bdc04be4fde` |
| 150k | conflict | 64 | 64 | none | `fa6d1f171aea17b23725942619a4b41d55c7ae360722831ac0c9743177c98f8c` |

The 200k boundary independently repeats the result: 50/50 original rows and 64/64 conflict rows are identical after removing the same two labels.

## Effect on completed boundaries

The alias double-weights `skilled`, so the recorded aggregate is not a six-behavior audit. It does not invalidate PPO samples or the neural action-authority receipts because `fixed_yield` is not a distinct member of the training mixture.

The 50k, 100k, 150k, and 200k candidates remain failed even if the duplicate is replaced. In both suites at every one of those boundaries, at least one of the five nonduplicated partner types already contains an AI shutdown. Replacing only `fixed_yield` therefore cannot satisfy the absolute zero-AI-shutdown gate. Other recorded failures include collision streak, no-progress streak, relative improvement, and, on later original-suite boundaries, static-wall commands.

There is no valid earliest passing boundary through 200k. If a later boundary passes the current recorded audit, it still requires an independent corrected audit before selection; the first passing step must be determined from that corrected sequence.

## Corrected sixth behavior

Use a separately versioned `fixed_yield_r1` evaluator after training freezes. It reads only the public pre-action state. It first computes the normal skilled action for `robot_1`; when the two robots are within path distance two and `robot_1` is not in a public must-charge state, it returns `WAIT`, otherwise it returns the skilled action. It must not read neural logits, probabilities, the current `robot_2` command, or future state.

Re-evaluate the frozen r3 baseline and every committed candidate on the same original and conflict scenes with matched seeds, then apply the unchanged gates. The running evaluator and its historical artifacts remain intact.
