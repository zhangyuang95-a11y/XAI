# Warehouse r4.3 internal pilot release

Release date: 2026-09-15

This release adds the shared-charger occupancy rule, changes movement energy use to 3%, adapts the frozen neural policy to the new public rule state, and replaces the action-focused question UI with current-frame system questions. It remains an internal pilot: `formal_ready=false`, `formal_sample_eligible=false`, and the Render SQLite database uses ephemeral storage.

## Runtime behavior

- Movement costs 3 percentage points of battery and a charging wait restores 10 percentage points.
- A robot occupying the charger at 60% battery or more while an active teammate at 20% or less is within two path steps receives two qualifying grace waits. A third consecutive qualifying wait costs the team 50 points. The rule is symmetric, collision and passive-wait frames are exempt, each continuous occupation is penalized once, and later independent occupations can be penalized again without a score cap.
- The Actor receives ten public rule-state features. Runtime actions are submitted exactly as selected by the neural policy; no script replaces, retries, or filters an action.
- System questions bind to the selected frame and route by meaning. Physical facts about charging, entering the charger, collisions, penalties, and both robots' charging need do not depend on decision-tree agreement. Technical evidence is folded below the one- or two-sentence answer.
- Group A can ask during and after Task 1. Group B has replay only. Neither group can retrieve current or cached answers in Task 2.

## Frozen artifacts

- Actor SHA-256: `94334881f65367f9726726e8f0b456b494924faef1f1e078b501e112d7ae4b84`
- Program SHA-256: `44eb68810e51f5d72f9f097ca3fe2c4c2e7aa9ac9c61e00351f7a673531d0322`
- Program content SHA-256: `226d28025bae54c7002549479e602176680be1c602f8427740e268930a7992e1`
- Release ZIP SHA-256: `71e7ffbbb87fae66cb6b3993b3929d8f81cd77859145366e65a3c2f13d44f585`
- Release manifest SHA-256: `9dc714c2e4a6fa3565fa672dd547ab3e994fef85875eb77d95866eb43b0d879f`
- Base64 secret file SHA-256: `958fed62e0c089910d1a172ae34b0b358f4baf50f573775d4da03ec4d71a2440`
- Runtime signature: `7a0104404f3e8f09f0595087de44ce690590468382e18b6f71db72061f867ba2`

## Training and validation

The 197-input parent Actor was expanded to 207 inputs without changing its original columns. Adaptation used 22,800 public-state correction rows, 9,180 retention rows, four DAgger epochs, and 1,024 genuine on-policy PPO joint steps on MPS. The final model was then frozen.

The extracted program passed the independent audit: overall agreement 90.37%, non-wait agreement 90.31%, critical-state agreement 90.71%, and intervention-direction agreement 88.23%. The rule-state probe covered 48 states and released the charger in all of them.

The focused behavior evaluation covered all six study scenes with wait, skilled, and assertive partners. Actor action equaled submitted action in every frame, action overrides were zero, and loaded deliveries were completed within shortest-path-plus-two steps in 64/64 probes. With the skilled partner, robot 2 delivered 4, 2, 2, 2, 2, and 3 items across the six scenes.

The 120-step tutorial contains 121 frames including the initial state. Both robots move concurrently, it demonstrates pickup, delivery, collision, wait, and charging, and it records 13 deliveries.

Automated checks: 26 selected Python tests passed. Browser checks passed at 1365x900 and 1280x800, including stable scroll position, 380 ms playback, Group A/B visibility, current-frame questions, Task 2 denial, and the red-edge/`-50` penalty feedback.

## Known pilot limitations

This build is intended to test whether explanations reduce coordination errors, not to claim a production-quality partner. Some compatible-partner evaluations ended in battery shutdown, and scenes 62 and 21 produced long collision deadlocks. Those results remain in the behavior report. The free Render service can restart and discard its `/tmp` SQLite records, so this release is not eligible for formal data collection.

## Reproduction

```bash
PYTHONPATH=. pytest -q tests/test_warehouse_r43.py tests/test_warehouse_alignment_r42_tutorial.py tests/test_warehouse_alignment_r42_source.py tests/test_warehouse_r42_manual_preview.py
PYTHONPATH=. python -m backend.training.warehouse_r43_behavior_evaluation --manifest output/warehouse_native/r43_observed_rule_20260915/evaluation_manifest.json --actor output/warehouse_native/r43_observed_rule_20260915/adaptation/actor.npz --output output/warehouse_native/r43_observed_rule_20260915/behavior_report.json
PYTHONPATH=. python scripts/build_warehouse_r42_delivery_release.py --help
```

Deployment target: <https://policylens-warehouse-study.onrender.com>
