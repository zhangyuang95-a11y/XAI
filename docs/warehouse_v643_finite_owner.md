# v6.43 finite-run ownership components

This is implementation evidence for the new candidate `warehouse-v643-risk-calibration-01`, not a training result, seed freshness conclusion, model acceptance or pilot authorization. The old v6.42 failure and consumed claims remain closed. No real Warehouse environment or new scientific scenario was used to implement these components.

`v643_training_partitions` provides a new, strict finite partition contract. TRAIN belongs to the eight-slot 20k PPO stream. Program-only evidence has 900/150/300/300 complete episodes, each split balanced across three tasks; these observations cannot enter a reward/cost gradient buffer. Quality has 75 whole scenes, 25/task, reserved for the frozen original reference and both trained variants. Original base/scenario/action arguments, episode identities, initialization and optimizer streams must be disjoint across partitions. Inventory entries are reservations, not executed episodes.

The historical read-only readiness reader authenticates the complete original v6.20 BC Actor and reference buffer. Its earlier frozen-NN/program confirmation is historical startup evidence only. The failed old 20k calibration is not a prerequisite pass, an old claim is never accepted, and a caller-written `passed` field has no authority. Raw and regularized must later be initialized from this same zero-update state and optimizer/RNG identity through the numerical bridge's `validate_common_start`.

`v643_training_seed_plan` records the original constructor/reset and hash-derived action-stream arguments for both variants, plus reference/raw/regularized quality evaluation. Shared scenes are explicitly marked paired reuse, never independent replication. Source snapshots bind the numerical bridge, the new program namespace/provider and owner/recovery components. The metadata → triage → exact/context auditor pipeline must reproduce the new plan against all four historical roots and the registered trace inventory. Its finite clearance expires after 24 hours; all selected metadata and excluded weight-path stats must match again at execution. Neither an inventory nor a clearance grants reset or optimizer authority.

The actual owner/collector is implemented in `v643_calibration_pipeline`. The explicit finite CLI accepts only a newly frozen exact plan/hash with the original reference, complete source/inventory/history checks and one candidate claim; it has no enable/pass switch and does not automatically start. Every initializer/reset is durably predebited before it occurs, raw has zero external program feedback, program output uses the new candidate namespace, and a full checkpoint includes the numerical bridge's separate cost optimizer/risk buffer plus all eight private environments, rollout-reader pending histories, source/provider/refit state, cursors, claim and journal boundaries. A numerical-only `.pt` file is insufficient. Interrupted work cannot automatically restart or recollect a scene.

Tests use synthetic inventories and temporary empty historical roots. Physical integration uses explicit manually built states, a control RNG and real NN/physics replay; the actual environment constructor/reset is forbidden in those fixtures. No actual candidate stream/global claim is consumed. Independent final-program/quality auditors are connected to the workflow. Actual execution is still gated by the as-yet unprepared real plan/clearance, and no delivery, collision, shutdown or coverage improvement is claimed.

The implemented `ResetLedger` writes and synchronizes each initialization/reset predebit before advancing its finite cursor. A successful reset must match the reserved original seed and fresh physical episode ID 1; a pending, missing, duplicated or reordered binding cannot become a safe boundary. Its checkpoint contains the exact byte length/hash of a complete journal prefix, so later appends preserve the earlier boundary while changes/truncation of that prefix fail validation. A supplied matching hook row is not independent proof that the real hook produced it; the collector must still bind the actual hook journal and physical replay.

`v643_calibration_recovery.save_components` accepts actual `V643PPOUpdateCore`, `EnvironmentRolloutReplay` and `ResetLedger` instances, saving the complete numerical `.pt` plus a JSON owner envelope. The envelope checks source/claim/reference identities, all eight private and pending histories, exact TRAIN membership, physical configuration and complete private RNG snapshots, finite cursors, refit schedule, program source/provider and update references. Explicit `--resume/--resume-sha` accepts only the tested clean-pair continuation under the original consumed claim.

Executed engineering regression: the three new inventory/seed/owner test modules and the existing v6.43 numerical bridge tests produced **71 passed in 75.94 seconds**. The source closure contains 303 repository files with no absolute/traversing source names. The actual candidate's global single-use claim is absent. These checks used temporary fixtures; the historical-root auditor was tested against four empty temporary directories, not the actual scientific history, and no new history clearance is claimed.

The actual private workflow now connects the eight-environment collector, physical/NN replay reader, `V643PPOUpdateCore`, true complete-episode cost replay, scheduled program source/refit/provider, independent final-program qualification and 225 matched quality episodes. Both zero-update starts must pass `validate_common_start`; initializer/optimizer allocation and resets are predebited; the global claim is exclusively written only after the final real-history stat check. Quality scenes cannot run before final programs are independently validated. Actual failed quality/program results propagate into both pipeline/completion receipts and an immutable closed-calibration marker. Numerical completion alone is not a calibration pass or release admission.

Clean continuation retains the original consumed claim and holds an OS lock on its existing execution journal. It rejects changed owned artifacts, any unaccounted initialization/reset/rollout/update, a previous restoration attempt, external-history changes, and scientifically closed/completed candidates. Its attempt and RNG-allocation predebits are appended to the existing journal before reconstruction; no second claim/lease is created. Torch modules allocate on the meta device and load all saved tensors without sampling. PCG64 uses the original registered optimizer argument for allocation only, overwrites the entire saved state before exposing the Generator, and retains one shared RNG object for PPO/risk. These bounded non-sampling restoration calls are declared in the seed plan for each possible completed optimizer boundary, with at most one attempt per checkpoint. Private environments and reader histories restore without constructor/reset. Nonclean recovery remains refused, and automatic retry/resume remains forbidden.

New executed integration checks: **9 pipeline tests passed in 78.01 seconds**, including a genuine 1024-frame NN/physics batch across eight slots, 16 controlled fixture-reset bindings, 2048 role-action checks, 960 complete-cost rows, a real critic-only PPO/risk update and a full saved-boundary readback. A separate **clean-pair recovery test passed in 73.36 seconds**: both variants first collect/update 1024 fixture frames, then reconstruction preserves global Torch/NumPy state and the original claim hash. The next 16 physical frames/actions/states and subsequent NN/reward-Adam/cost-Adam/risk/RNG numerical states exactly match an uninterrupted shadow control. Only distinct trace-path receipt identities are compared as provenance rather than numerical equality. No real scientific scene or acceptance metric is supplied by these software checks.

Remaining before scientific execution: a newly frozen source/finite inventory plan, exact clearance against the actual four historical roots, and the candidate's first single-use claim. No real v6.43 training, final independent audit or holdout has been run by this implementation task.

## Explicit preparation and launch

Current original reference bytes were read and SHA checked during implementation (this is not a new task-quality result):

- Actor: `output/warehouse_v620_bc_pilot/warehouse-v620-01-shared-bc-pilot/pair-6160-gamma-003/final-bc.actor.npz`, SHA `1b6310f6d93309fb17e64516519a71680ad3c30e73f222be27df4dff10da81ec`.
- Shared buffer: `output/warehouse_v629/training-preparation/reference-buffer-01/reference-training.npz`, SHA `2d4b2e593ed3d16a60e90564f8ee9c58a8026a722c4786f1e9ccff574b125dfa`.
- Its preparation result: `output/warehouse_v629/training-preparation/reference-buffer-01/result.json`, SHA `2afc4d502b4fff65c4558a135d498150e99aee1f601203e3726acef907bb3c16`.

Use that `reference-buffer-01` directory. `load_readiness` additionally rechecks the full historical reference/extraction provenance; these three hashes alone do not authorize training.

The operator supplies immutable JSON inventories and explicit fresh initialization/optimizer arguments. No real argument values are selected here. TRAIN uses the canonical finite TRAIN schema with `episode_id,scenario_seed,task,action_seed,slot,partition`; each of eight slots needs at least 21 reserved cases and may reserve up to 2500. Reserving more protects against earlier terminations; the owner refuses inventory exhaustion instead of inventing seeds. Program inventory uses `PROGRAM_VERSION/PROGRAM_SCOPE`, exact candidate ID, 900/150/300/300 cases, and `episode,base_seed,scenario_seed,task,action_seed,split`. Quality uses `QUALITY_VERSION`, exact candidate ID and 75 `case_id,base,scenario_seed,task,action_seed` records. Every ID starts with the candidate ID plus `-`. All original numeric arguments are globally disjoint across these partitions and initialization/optimizer streams; scenario arguments equal `10*base+task`. Program splits and quality tasks are balanced. The same inventories bind both variants; quality also binds the original reference.

This parameterized preparation recipe performs no reset, model initialization or training:

```python
from pathlib import Path
from experiments.warehouse import v643_training_partitions as p
from experiments.warehouse import v643_training_seed_plan as s
from experiments.warehouse.v626_protocol import pointer

def prepare_candidate(training_json, program_json, quality_json,
                      initialization_seed, optimizer_seed, run_id):
    reference = Path('output/warehouse_v629/training-preparation/reference-buffer-01').resolve()
    partition = p.prepare(
        reference_directory=reference,
        training_inventory_ref=pointer(training_json),
        program_inventory_ref=pointer(program_json),
        quality_inventory_ref=pointer(quality_json),
        initialization_seed=initialization_seed, optimizer_seed=optimizer_seed,
        output=p.OUTPUT_ROOT/run_id)
    plan = s.prepare(reference_directory=reference, partition_ref=partition.ref, run_id=run_id)
    clearance = s.audit_seeds(plan)
    return plan.ref, clearance['clearance']
```

Keep preparation, exact audit and launch in the **same Python process** so the historical mathematical reference admission is replayed once. All new v643 reference entrypoints normalize the directory to an absolute resolved Path, ensuring the existing historical cache has one consistent key. After `prepare_candidate` returns, explicitly call `v643_calibration_pipeline.run(reference_directory=reference, plan_ref=plan_ref)` in that process (or invoke the new CLI's `main` in the same process). The cache still rehashes its bound read set; no current-NN evidence is reused across updates.

The standalone equivalent is `python scripts/run_warehouse_v643_calibration.py --reference-directory <reference-buffer-01> --plan <new-plan.json> --plan-sha <its-exact-SHA256>`, which incurs a fresh-process historical replay. An explicit continuation additionally supplies `--resume <clean-pair-boundary.json> --resume-sha <its-exact-SHA256>` with the same original plan. Missing/mismatched refs and old claims fail before RNG/reset. The CLI returns 0 only for the actual independently measured calibration precheck; failed or unmeasured prechecks return nonzero. Neither status opens model admission or pilot.
