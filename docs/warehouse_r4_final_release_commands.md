# Warehouse r4 final acceptance and release commands

The 2026-09-11 training round is closed and must not proceed through this
release sequence. Its authenticated outcome is recorded in
[`warehouse_r4_training_closeout.md`](warehouse_r4_training_closeout.md).

This is the production order for the r4 **internal pilot**. It never marks the
system as formal-study ready and it does not treat a human explanation effect
as validated. Run it only after the trainer has frozen one selected Actor,
written the complete PPO budget ledger, and stopped all processes that can
change the source or artifacts.

The admission and package builders hash the current source tree. Do not edit a
bound source, Actor, program, protocol, scenario file, question file, report,
or row file after starting this sequence. If one changes, discard the new
admission/package and repeat every affected producer. Never copy a passing
boolean into a report or use a fixture report.

## Source and private-artifact prerequisites

A fresh Git checkout supplies the source only. The training lineage and final
release evidence live under the ignored `output/` tree and are deliberately
not committed. Before running this sequence, restore the private warehouse
artifact bundle at its original repository-relative paths and verify its
recorded SHA-256 values. At minimum this includes the frozen 3.95M alignment
parent, every explicitly admitted r4 stage checkpoint used by the selected
continuation, the registered foundation scenario manifest, and the registered
question pool. The final Render Secret File is sufficient to serve the frozen
online study; it is not a substitute for these training and audit inputs.

Offline training and RCPD extraction require PyTorch and scikit-learn in
addition to the lightweight Render dependencies. Browser acceptance requires
Node.js plus Playwright. These tools are not installed by
`requirements-render.txt`, which intentionally contains only online runtime
dependencies.

## 0. Set the frozen paths

Run from the XAI repository root and replace only the angle-bracket values.

```bash
cd /path/to/XAI

export R4_FINAL="output/warehouse_native/<selected-r4-run>"
export R4_TRAINING_ACTOR="$R4_FINAL/<selected-boundary>/actor.npz"
export R4_SCENARIOS="output/warehouse_native/v1-foundation-seed260908/scenarios.json"
export R3_ACTOR="output/warehouse_native/alignment_metadata_bank_395m_20260910/inputs/actor.npz"
export R3_CHECKPOINT="output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/boundaries/step_0050000.pt"
export R4_ATTEMPT_INVENTORY="output/warehouse_native/r4_training_budget_ledger_inputs_20260911/attempt_inventory.json"
export R4_RELEASE="$R4_FINAL/release"
export R4_TRAINING_LEDGER="$R4_RELEASE/training_budget_ledger.json"
```

The training ledger is built only after the final paired audit, final RCPD,
and explanation audit exist. Its v2 semantic reader rediscovers every r4 run
directory, reopens its run, protocol, source receipt, training log, every 50k
boundary Actor and opaque checkpoint, authenticates parent edges, and counts
only each edge's fresh delta. It must remain at or below 1,000,000 and select
the earliest Actor with a complete passing paired audit. A cumulative child
counter is never added as if it were all-new PPO work.

The source-r3 pre-extraction command used at the start of training is recorded
here so the run is reproducible. It is read-only evidence collection and must
report zero PPO steps and an unchanged r3 Actor:

```bash
python -m backend.training.warehouse_r4_r3_preextract \
  --output "$R4_FINAL/r3_preextraction" \
  --environment-steps 50000
```

## 1. Re-run the final paired Actor audit

This is the selection audit on all 50 registered validation scenes and all six
partners, 300 episodes for each Actor. Continue only if `paired_report.json`
has `status=passed`, `selected=true`, all absolute checks, at least four of the
five relative checks, and zero action overrides.

```bash
python -m backend.training.warehouse_r4_active_evaluation \
  --baseline-actor "$R3_ACTOR" \
  --candidate-actor "$R4_TRAINING_ACTOR" \
  --scenarios "$R4_SCENARIOS" \
  --output "$R4_RELEASE/paired_active_audit"
```

If no Actor passes, stop the release sequence here. The ordinary ledger command
in step 7 will fail without creating a file. For a machine-verifiable closeout,
and only for a closeout, add `--diagnostic-failed-ledger`. That explicit mode
reopens the same complete training DAG and writes
`status=failed_no_eligible_actor`, `admission_eligible=false`, and
`selection=null`. Production admission rejects this diagnostic schema.

The 2026-09-11 r4 run exhausted the approved 1,000,000 fresh PPO joint steps in
18 attempts and 20 authenticated 50k segments. No Actor passed the frozen
paired gate. Its reproducible closeout command is:

```bash
python - <<'PY'
import json, os, subprocess, sys

inventory = json.load(open(os.environ["R4_ATTEMPT_INVENTORY"], encoding="utf-8"))
attempt_args = []
for row in inventory["attempts"]:
    attempt_args.extend(("--attempt", row["output_path"]))
command = [
    sys.executable, "-m", "backend.training.warehouse_r4_training_ledger",
    "--inventory-root", "output/warehouse_native",
    *attempt_args,
    "--r3-actor", os.environ["R3_ACTOR"],
    "--r3-checkpoint", os.environ["R3_CHECKPOINT"],
    "--scenario-manifest", os.environ["R4_SCENARIOS"],
    "--active-policy-report",
        "output/warehouse_native/r4_active_lowentropy_v8_1m_20260911/"
        "full_paired_audit_300/paired_report.json",
    "--zero-step-record",
        "output/warehouse_native/r4_r3_preextract_v7_20260911/report.json",
    "--diagnostic-failed-ledger",
    "--output",
        "output/warehouse_native/r4_release_closeout_20260911/"
        "training_budget_failed.json",
]
subprocess.run(command, check=True)
PY
```

The resulting file must report exactly 18 attempts, 20 segments, 1,000,000
actual additional steps, zero remaining steps, and no selection. Do not run
steps 2–10 with it and do not update Render from this failed r4 closeout.

## 2. Extract the post-freeze final RCPD

Do not reuse a boundary program that regularized an earlier training cycle.
Collect a fresh, read-only evidence set from the selected Actor. This command
uses all 100 registered `extraction` scenes, partitions them 70/30 by scene,
runs three partners, fits the complete portable 5×5 capacity grid (including
depths 4/6/8 and 16/32/64 leaves), and performs zero PPO or optimizer updates.
It writes `program.json` only when the simplest eligible tree reaches at least
90% overall fidelity and 85% fidelity in each critical group.

```bash
python -m backend.training.warehouse_r4_final_rcpd \
  --actor "$R4_TRAINING_ACTOR" \
  --scenarios "$R4_SCENARIOS" \
  --output "$R4_RELEASE/final_rcpd"

export R4_TRAINING_PROGRAM="$R4_RELEASE/final_rcpd/program.json"
export R4_FINAL_RCPD_REPORT="$R4_RELEASE/final_rcpd/report.json"
```

The runtime builder reopens the NPZ evidence and every candidate program,
physically replays each registered episode from its saved player actions,
recomputes Actor labels, split separation, fidelity, critical metrics and the
selection order, and rejects a stale or edited report.

## 3. Bind the final runtime components

Use the final-Actor RCPD program. The builder re-exports the same weights with
the online protocol binding and independently checks NumPy logits, actions,
and one real warehouse transition.

```bash
python -m backend.training.warehouse_r4_active_release \
  --actor "$R4_TRAINING_ACTOR" \
  --program "$R4_TRAINING_PROGRAM" \
  --extraction-report "$R4_FINAL_RCPD_REPORT" \
  --evaluation "$R4_RELEASE/paired_active_audit/paired_report.json" \
  --scenarios "$R4_SCENARIOS" \
  --output "$R4_RELEASE/runtime_components"
```

The remaining commands use these bound artifacts:

```bash
export R4_ACTOR="$R4_RELEASE/runtime_components/actor.npz"
export R4_PROTOCOL="$R4_RELEASE/runtime_components/protocol.json"
export R4_PROGRAM="$R4_RELEASE/runtime_components/program.json"
```

## 4. Run independent explanation acceptance

This producer uses the episode-disjoint 100-scene `explanation_test` split and
three partners (300 complete episodes). It records ordinary and critical-state
Actor/program fidelity, effective isolated intervention pairs, exact Actor to
submitted-action equality, source-state immutability, and the Chinese/English
participant answer boundary. Ineffective intervention pairs are never counted
as correct. Continue only if `report.json` has `status=passed`.

```bash
python -m backend.training.warehouse_r4_explanation_audit \
  --actor "$R4_ACTOR" \
  --protocol "$R4_PROTOCOL" \
  --program "$R4_PROGRAM" \
  --scenarios "$R4_SCENARIOS" \
  --output "$R4_RELEASE/explanation_audit"
```

## 5. Select six high-conflict play scenes

Start with 120 new development states. If and only if this run cannot produce
six scenes that pass every frozen geometry, dynamic, recovery, balance, and
online-runtime check, rerun in a new output directory with 300. Do not loosen
the thresholds. A failed output stays as evidence and is never overwritten.

```bash
python -m backend.training.warehouse_r4_conflict_scene_selection \
  --scenarios "$R4_SCENARIOS" \
  --actor "$R4_ACTOR" \
  --actor-protocol "$R4_PROTOCOL" \
  --output "$R4_RELEASE/conflict_scenes_120" \
  --seed-start 500000 \
  --distinct-count 120 \
  --maximum-draws 20000 \
  --workers 4
```

The bounded expansion, when required, is:

```bash
python -m backend.training.warehouse_r4_conflict_scene_selection \
  --scenarios "$R4_SCENARIOS" \
  --actor "$R4_ACTOR" \
  --actor-protocol "$R4_PROTOCOL" \
  --output "$R4_RELEASE/conflict_scenes_300" \
  --seed-start 500000 \
  --distinct-count 300 \
  --maximum-draws 60000 \
  --allow-repeated-task-geometry \
  --workers 4
```

Point `R4_CONFLICT` at the one accepted directory. Its report must say
`accepted_six_scene_selection` and its online-runtime section must validate all
seven practice/task scenes with zero overrides.

```bash
export R4_CONFLICT="$R4_RELEASE/<conflict_scenes_120-or-conflict_scenes_300>"
export R4_SELECTED_SCENES="$R4_CONFLICT/selected_scenes.json"
```

## 6. Regenerate and replay the frozen questionnaire

The existing independent development pool is not a play, validation,
explanation-test, or final-test split. The generator selects eight distinct
scenes, four distinct next actions and four distinct three-step displacements,
then independently replays every item.

```bash
python -m backend.training.warehouse_r4_question_bank \
  --actor "$R4_ACTOR" \
  --protocol "$R4_PROTOCOL" \
  --pool output/warehouse_native/family_feedback_question_pool_20260910/pool.json \
  --pool-manifest output/warehouse_native/family_feedback_question_pool_20260910/manifest.json \
  --output "$R4_RELEASE/questionnaire"

export R4_QUESTION_BANK="$R4_RELEASE/questionnaire/question_bank.json"
```

## 7. Build and independently replay the PPO budget ledger

The machine inventory is only the explicit attempt-path input. The builder
does not trust its step totals: it requires that its path list exactly equal
all discovered `warehouse-r4-active-trainer.*` run records and recalculates
the DAG and segment total from the original files. The final-RCPD and
explanation reports are listed separately as zero-update work; neither is
charged to the PPO budget. Add other diagnostic records only when the record
itself explicitly reports zero `ppo_joint_steps`, `optimizer_updates`, or
`neural_updates`.

```bash
python - <<'PY'
import json, os, subprocess, sys

inventory = json.load(open(os.environ["R4_ATTEMPT_INVENTORY"], encoding="utf-8"))
attempt_args = []
for row in inventory["attempts"]:
    attempt_args.extend(("--attempt", row["output_path"]))
command = [
    sys.executable, "-m", "backend.training.warehouse_r4_training_ledger",
    "--inventory-root", "output/warehouse_native",
    *attempt_args,
    "--r3-actor", os.environ["R3_ACTOR"],
    "--r3-checkpoint", os.environ["R3_CHECKPOINT"],
    "--scenario-manifest", os.environ["R4_SCENARIOS"],
    "--active-policy-report",
        os.environ["R4_RELEASE"] + "/paired_active_audit/paired_report.json",
    "--zero-step-record",
        os.environ["R4_FINAL"] + "/r3_preextraction/report.json",
    "--zero-step-record",
        os.environ["R4_RELEASE"] + "/final_rcpd/report.json",
    "--zero-step-record",
        os.environ["R4_RELEASE"] + "/explanation_audit/report.json",
    "--output", os.environ["R4_TRAINING_LEDGER"],
]
subprocess.run(command, check=True)
PY
```

This command must fail when no Actor passed the full paired audit, an attempt
is still running or omitted, a parent hash does not resolve, a boundary/log is
changed, or the unique segment total exceeds the cap. Do not create a manual
replacement ledger in those cases.

## 8. Create and independently re-verify the fail-closed admission

The builder reads exactly seven report inputs. It recomputes the gates from
their metrics and immutable explanation rows, binds the loaded runtime and
files, writes a new admission exclusively, and immediately exercises the
release-side validator. Missing, stale, fixture, partial, hand-edited, or
threshold-failing evidence aborts the command and removes the incomplete new
admission.

```bash
python scripts/build_warehouse_r4_admission.py \
  --actor "$R4_ACTOR" \
  --protocol "$R4_PROTOCOL" \
  --program "$R4_PROGRAM" \
  --selected-scenes "$R4_SELECTED_SCENES" \
  --question-bank "$R4_QUESTION_BANK" \
  --validation-scenarios "$R4_SCENARIOS" \
  --training-budget "$R4_TRAINING_LEDGER" \
  --active-policy-report "$R4_RELEASE/paired_active_audit/paired_report.json" \
  --runtime-components-report "$R4_RELEASE/runtime_components/verification.json" \
  --high-conflict-scenes-report "$R4_CONFLICT/report.json" \
  --explanation-program-report "$R4_RELEASE/explanation_audit/report.json" \
  --questionnaire-report "$R4_RELEASE/questionnaire/report.json" \
  --output "$R4_RELEASE/admission.json"

shasum -a 256 "$R4_RELEASE/admission.json"
```

Copy the printed lowercase SHA-256 into `R4_ADMISSION_SHA`. It must come from
outside the admission JSON. Then run the validator-only path:

```bash
export R4_ADMISSION_SHA="<64-character admission SHA-256>"

python scripts/build_warehouse_r4_admission.py \
  --actor "$R4_ACTOR" \
  --protocol "$R4_PROTOCOL" \
  --program "$R4_PROGRAM" \
  --selected-scenes "$R4_SELECTED_SCENES" \
  --question-bank "$R4_QUESTION_BANK" \
  --validation-scenarios "$R4_SCENARIOS" \
  --verify-existing "$R4_RELEASE/admission.json" \
  --expected-admission-sha256 "$R4_ADMISSION_SHA"
```

## 9. Build and independently reload the Render Secret File

```bash
python scripts/build_warehouse_r4_online_release.py \
  --actor "$R4_ACTOR" \
  --protocol "$R4_PROTOCOL" \
  --program "$R4_PROGRAM" \
  --selected-scenes "$R4_SELECTED_SCENES" \
  --question-bank "$R4_QUESTION_BANK" \
  --scenarios "$R4_SCENARIOS" \
  --admission "$R4_RELEASE/admission.json" \
  --expected-admission-sha256 "$R4_ADMISSION_SHA" \
  --output-package "$R4_RELEASE/warehouse_alignment_r4_online.zip" \
  --output-base64 "$R4_RELEASE/warehouse_alignment_release.b64"
```

Record the package and manifest SHA-256 values printed by this command. The
builder independently reloads the archive, enforces the Secret File size cap,
and repeats admission and seven-scene runtime checks before returning success.

## 10. Post-package checks before any Render change

Start a local service against a new QA database, using the package and manifest
hashes printed in step 9:

```bash
python -m ui.warehouse_alignment_online_server \
  --base64 "$R4_RELEASE/warehouse_alignment_release.b64" \
  --expected-package-sha256 "<package SHA-256>" \
  --expected-manifest-sha256 "<manifest SHA-256>" \
  --database "$R4_RELEASE/r4_http_qa.sqlite3" \
  --storage-mode persistent \
  --public-origin http://127.0.0.1:8013 \
  --host 127.0.0.1 \
  --port 8013
```

In another terminal run the real HTTP and browser acceptance checks into new
output directories:

```bash
python scripts/check_warehouse_alignment_http.py \
  --base http://127.0.0.1:8013 \
  --database "$R4_RELEASE/r4_http_qa.sqlite3" \
  --expected-manifest-sha256 "<manifest SHA-256>" \
  --output "$R4_RELEASE/http_acceptance" \
  --execute

node scripts/check_warehouse_family_browser.cjs \
  --base http://127.0.0.1:8013 \
  --output "$R4_RELEASE/browser_acceptance" \
  --execute
```

Stop and restart the same server command, then use the HTTP check's documented
`--verify-restart` or the dedicated resume command if its injected lost-response
case produced a failed-run receipt. A human operator must also inspect the
1365×900 and 1280×800 Chinese/English screenshots, all six shortcuts, long
evidence details, stable scroll/focus, and 380 ms continuous movement.

Only after all post-package checks pass should the recorded package and
manifest hashes and the Base64 bytes replace the current Render configuration.
Deployment itself is outside this command sequence. The released status remains
`formal_ready=false`; a technical pass does not establish a human explanation
treatment effect or current free-instance concurrency capacity.
