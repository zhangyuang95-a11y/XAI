# Warehouse r4.2 internal pilot release

## Release identity

- Public release: `r4.2-internal-pilot`
- Server: `warehouse-alignment-online-study-server.r4.2`
- Actor SHA-256: `4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b`
- Program SHA-256: `79da2cd273d2a5e3a9a1af0ec1c4c8a42f4c887c8971be9a72642e292b42aff1`
- Release package SHA-256: `1c372b2294a3739aadee2d346a043c2cbdbe8d7c35c69d78a66ae8acd24837dd`
- Manifest SHA-256: `56790275e22a017c015a4c47cc86109864e712d822ee521c05814960c6c715b5`
- Render Secret File SHA-256: `1e43dda1a0446464d0271b65308845a46b19db575dfe4f7798404c3e9c515d2c`

This release is an internal pilot. It uses ephemeral Render storage, is not a formal-ready release, and its records are not eligible as formal samples.

## Participant protocol

The sequence is fixed for every participant: consent, a separate 120-step AI–AI tutorial, X1–X3, Y1–Y3, questionnaire, completion. Every formal round first shows a clean frame-0 preview with two active tasks, full batteries, empty hands, and zero score, deliveries, collisions, shutdowns, and steps. A round advances for at most 120 joint steps; delivered tasks are replaced, so the round is not capped at three deliveries.

Participants are assigned in balanced pairs to A or B. A can ask explanations during Task 1 and its round review. B cannot request explanations. Task 2 blocks new and cached answers for both groups. The participant interface does not reveal the assigned condition.

The primary analysis compares the two groups' mean Task 2 score. Standardized change from Task 1 to Task 2 remains a secondary descriptive measure.

## Tutorial and explanation checks

The tutorial contains 121 frames: one initial frame plus exactly 120 physical joint steps. Both robots move, wait, charge, encounter a collision, recover, pick up, and complete at least one delivery each. Its maximum consecutive waits are 16 for robot 1 and 8 for robot 2, and it is stored separately from formal task scores.

Six fixed intent IDs cover action reason, wait reason, collision reason, player influence, task direction, and charging need in Chinese and English. The charging answer begins with a direct yes/no conclusion and then states the verified battery and estimated task-plus-return requirement. Only the latest answer is shown in the interface; every question remains in SQLite for audit.

The admitted explanation audit reports overall fidelity 94.03%, non-WAIT fidelity 94.13%, narrow-state fidelity 95.64%, charging fidelity 91.61%, pickup fidelity 94.58%, and intervention direction consistency 88.18%.

## Actor and scene evidence

The deployed Actor's actions are submitted unchanged. Dynamic scene replay contains 5,118,522 Actor submission frames and zero action overrides. It generated 286,633 successor tasks with zero successor-sampling failures.

Compared with r3 on the high-conflict evaluation, active non-WAIT increased from 85.04% to 87.08%, productive action from 71.26% to 76.16%, and mean AI deliveries from 9.09 to 9.62. Some earlier behavioral targets, including shutdowns and long no-progress tails, remain unmet; the behavior-performance gate is therefore explicitly waived for this internal pilot rather than reported as passed.

The six frozen scenes cover six conflict families. X has a conflict rate of 27.69% and workload 18.67; Y has a conflict rate of 28.24% and workload 18.33. Relative X/Y differences are 1.95% for conflict and 1.80% for workload. More than 286,000 validated successor tasks demonstrate that conflict is maintained after deliveries, not only at frame 0.

## Validation and deployment

Run the complete local admission check with:

```bash
python scripts/check_warehouse_r42_internal_pilot.py \
  --output output/warehouse_native/r42_internal_pilot_20260914/acceptance_report.json
```

The check validates the package and manifest hashes, Secret File size, tutorial, clean round starts, fixed X/Y sequence, balanced A/B allocation, Task 1 and Task 2 explanation permissions, latest-answer presentation, preserved question history, Actor action authority, dynamic successor tasks, and X/Y balance. The browser check additionally verifies the old PolicyLens layout, continuous 380 ms tutorial playback, and the explicit frame-0 round start.

Render must start `ui.warehouse_alignment_r42_server` with `ui.warehouse_alignment_r42_release`, the pinned hashes above, and `/etc/secrets/warehouse_alignment_release.b64`.
