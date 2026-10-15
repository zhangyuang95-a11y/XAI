# Warehouse r4 scene selection

The selector in `backend/training/warehouse_r4_conflict_scene_selection.py`
uses development seeds only. It does not read participant records and never
selects, changes, or overrides an Actor.

The v2 audit keeps the five pre-registered simple conventions and every gate
unchanged. It replaces the original underpowered compatible reference after a
retained v8/120 diagnostic showed that the old reference lost to the best simple
convention in all 36 geometry-pass scenes. The old reference merely followed
the nearest task until an immediate collision. The revised offline reference
uses public positions, tasks, energy and the Actor's actual submitted direction
to choose the complementary task, breaking shared-corridor ambiguity by minimum
joint pickup work. It then chooses a collision-free shortest public move. It
does not read Actor probabilities and cannot change or reselect the Actor action.

Collision recovery accounting was audited separately. Every collision is
counted once, recovery requires public task progress no later than the inclusive
tenth subsequent frame, repeated progress cannot double-count it, and an
unresolved collision at termination remains a failure. Version 3 also corrects
the meaning of persistent deadlock: the earlier diagnostic rejected the maximum
collision streak above ten or maximum no-progress streak above forty even when
the pair subsequently recovered. Those transient limits were not part of the
requested gates. A persistent deadlock now requires reaching the horizon with
more than ten consecutive terminal frames of collisions or no public task
progress. Maximum transient streaks remain reported but cannot reject a scene.

Geometry-only audit:

```bash
python -m backend.training.warehouse_r4_conflict_scene_selection \
  --scenarios output/warehouse_native/native_cycle_500k_candidate_20260909/scenarios.json \
  --output output/warehouse_native/r4_conflict_geometry \
  --distinct-count 120
```

Final frozen-Actor audit:

```bash
python -m backend.training.warehouse_r4_conflict_scene_selection \
  --scenarios output/warehouse_native/native_cycle_500k_candidate_20260909/scenarios.json \
  --actor /absolute/path/to/release_bundle/actor.npz \
  --actor-protocol /absolute/path/to/release_bundle/runtime_protocol.json \
  --output output/warehouse_native/r4_conflict_final \
  --distinct-count 120 \
  --workers 4
```

If fewer than six scenes pass, retain that failed report and run a new output
directory over 300 distinct public starting states without changing a gate:

```bash
python -m backend.training.warehouse_r4_conflict_scene_selection \
  --scenarios output/warehouse_native/native_cycle_500k_candidate_20260909/scenarios.json \
  --actor /absolute/path/to/release_bundle/actor.npz \
  --actor-protocol /absolute/path/to/release_bundle/runtime_protocol.json \
  --output output/warehouse_native/r4_conflict_final_300 \
  --distinct-count 300 \
  --allow-repeated-task-geometry \
  --workers 4
```

The expansion permits the same two-task geometry under a different public
initial heading, but the seven deployment scenes must still have seven distinct
physical fingerprints. The generated `deployment_play_scenes.json` orders the
practice scene first, followed by X1–X3 and Y1–Y3. Its final validation creates
all seven scenes through `OnlineAlignmentRuntime`, checks frame zero and unknown
initial public history, and runs mutation-free deterministic inference.
Use the runtime protocol and re-bound Actor emitted by the r4 release builder;
the internal training protocol intentionally is not an online admission object.

The retained r3 v1 diagnostic evaluated 36 strict geometry candidates, 4,320 episodes, and
501,207 Actor frames. It recorded zero Actor-action overrides. Only seed 500021
passed every dynamic gate, so r3 correctly produced no six-scene release. That
report is retained at
`output/warehouse_native/r4_conflict_dynamic_r3_20260910/report.json`.

The final available r4 diagnostic Actor (`bc9c2697…`) was audited with v3 over
300 unseen public starting states. Ninety-nine passed geometry, but none passed
all dynamic gates; only four distinct physical states came within one gate.
Consequently the diagnostic directory intentionally contains no
`selected_scenes.json` or deployment package. The full report and derived gate
summary are retained under
`output/warehouse_native/r4_conflict_diagnostic_v8_selector_v3_300_20260911/`.
