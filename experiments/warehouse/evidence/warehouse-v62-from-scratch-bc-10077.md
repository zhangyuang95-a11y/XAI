# Warehouse v6.2 unregistered from-scratch diagnostic (2026-10-16)

This is exploratory evidence, not admission or a release asset. The run rebuilt two independent Actors from random initialization using only public pre-action teacher labels, dual-NN visits, and an offline critic trained on real `env.step` collision events. It did not run paired PPO or spend registered development, final-audit, or heldout seeds.

## Identity and disjoint seeds

- Pair seed: `10077`; source closure SHA-256: `86c6fa4db8e7a6be960fe9d335d0ba383517613e8c0bfaca2a1e08c61b175bc1` at start and end.
- Frozen dual-Actor SHA-256: `9ae8aa8a501422f5c2aea602040ab261a20f4cda9b675838a60fca9324c36c6f`.
- The 16 BC stages used 1,220 distinct seeds generated from pair seed 10077; the offline collision critic used seed 16277. Initial frozen evaluation used 96 fresh episodes, 32 per task. A second pressure evaluation used 288 new episodes, 96 per task (`1014000 + 10i + task`).
- Program fit/calibration/eligibility/validation used respectively `1011000–1011049`, `1011050–1011074`, `1011075–1011099`, and `1011100–1011123`; exploratory holdback used `1012000–1012023`. A separate seed-intersection check found no overlap with registered development, formal selection, program audit, heldout, or v6.0–v6.9 sealed blocks.

## Frozen NN quality

| Evaluation | Episodes | Mean deliveries | Mean collisions | Shutdown episodes |
| --- | ---: | ---: | ---: | ---: |
| Initial fresh set | 96 | 13.344 | 5.760 | 2 |
| New pressure set | 288 | 13.847 | 4.097 | 2 |

The second set has task-level mean collisions of 3.448, 3.510, and 5.333. These are sampled actions from the frozen dual NN, with no teacher or program action override. They are exploratory quality measurements before PPO.

## Sparse program finding

The independent 16-leaf, depth-5 programs had zero raw illegal actions on validation and holdback. Their masked-distribution KL stayed below 0.06. Robot 2 had feedback actionable coverage 33.44% on validation and 31.21% on holdback. Robot 1 had 28.51% and 28.57%, below the fixed 30% development requirement. Four high-traffic, mixed robot-1 leaves accounted for roughly 62% of eligibility rows; program split capacity is the remaining bottleneck. This method is **not eligible** for a registered candidate or pilot.

## Recoverable files

Necessary exploratory artifacts are in ignored `output/warehouse_v6_repro_snapshots/warehouse-v62-bc-10077-20261016/`. Its hash inventory is `archive_manifest.json`, SHA-256 `b48fe87e1cfb02a617a4b8092e9811066438ed09f15f51910cd1f4e995fd6dd2`. The directory contains the BC, pressure, and program reports; frozen Actor and two program files; a 712-file snapshot manifest; and the training source-closure hash list. It does not contain the full source copy or a release package.

| Artifact | SHA-256 |
| --- | --- |
| `bc_probe.json` | `990e19914c4b5326c8a9afac3637f0d95dced0de21a880da48aead469c815071` |
| `pressure_probe.json` | `b9e83a73f1ec512879c7a5de6cdcc26a99e54bdb4942e8c332439ae553f982d8` |
| `program_probe.json` | `74d5f9df7ada06969f0f2002b304c9af42bfa68b89c16771388ac66def2ef3fa` |
| `actor.npz` | `9ae8aa8a501422f5c2aea602040ab261a20f4cda9b675838a60fca9324c36c6f` |
| `robot_1.program.json` | `5317ee1008e327fa758c956dc651b358b064249f74d052b4400d95d0d2b4b270` |
| `robot_2.program.json` | `957abc850b92bcd121e913c3993b338c3ebaea7a707d34e466a39dd6af843892` |
| `snapshot_manifest.json` | `3c375f510715e83b402325e1e49229d939a018e1b36d75cf8d70f4ee67c8c131` |
| `source_closure.json` | `144abbffd6b03babb11f7d5c2e72b4d5cc9f7602c45ee7e472edccddf9d30eab` |
