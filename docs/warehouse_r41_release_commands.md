# Warehouse r4.1 frozen-Actor admission and release

This pipeline admits an internal pilot only. Every command is fail closed;
none may be skipped, and no artifact from the failed r4 run is a substitute.
The final explanation program is extracted after Actor selection and is never
fed back into PPO.

If the full 2,000,000-step budget ends without a selected Actor, do not run
this pipeline. Use the non-release procedure in
`docs/warehouse_r41_failure_closeout.md`; it preserves all forty boundary
receipts and leaves r3 online without creating an admission or package.

Frozen conflict publication:

- manifest: `output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json`
- manifest file SHA-256: `4e640e19ea37405c3613d8f6ea1ace036ecff5da50cfb75c7e3f07d825f5b98e`
- manifest content SHA-256: `6afa2efe7c7dde8e4d7297a4e5855348948817b196acbf64159867f8d3fe5c12`
- contract SHA-256: `cf45637317fc4311fb1ac160125356a02e8b30203ffc72c7d3be7fe4630344c5`
- conflict graph SHA-256: `408894f802fb80bd0b65799b971ae5c0d3953f3d37c3fd3dc51715cffd9c202c`
- conflict graph file SHA-256: `57a0c14cfef7b6659685f85bf0503997294e1a0d7641907996219fdf6bdbf8a4`
- validation file SHA-256: `114ca16caeef2ec3dda729201af328c1b67b1766bfa7f9cd9a528820892b64a4`

Before a selected Actor is available, verify the frozen scene replay and every
post-freeze producer/package contract:

```bash
python scripts/run_warehouse_r41_postfreeze_release.py --check-contracts
```

After the training ledger has selected an Actor, the preferred command is the
single fail-closed orchestration below. The Actor, protocol and dual evaluation
must be the exact files named by the externally hashed ledger. The output root
must not already exist.

```bash
python scripts/run_warehouse_r41_postfreeze_release.py \
  --training-ledger "$R41_LEDGER" \
  --expected-training-ledger-sha256 "$R41_LEDGER_SHA" \
  --actor "$R41_ACTOR" \
  --protocol "$R41_PROTOCOL" \
  --dual-evaluation "$R41_DUAL_EVALUATION" \
  --output-root "output/warehouse_native/r41_postfreeze_$(date +%Y%m%d_%H%M%S)" \
  --workers 4
```

The command runs dynamic scene selection, independent final RCPD extraction,
the explanation audit, question-bank generation, neutral-tutorial generation
and replay, production admission, and portable packaging in that order. It
stores an fsynced `orchestration.jsonl` with each output SHA-256. Any failed
producer stops the chain and removes `production_admission.json`, the ZIP,
the Base64 Secret File and `release_receipt.json`; failed diagnostic evidence
is retained for inspection. On an M4 Pro, reserve roughly 1–3 hours for this
post-freeze chain, excluding training. Physical replay inside admission and
package verification intentionally repeats the expensive evidence checks.

The output tree is:

```text
$R41_OUT/
  orchestration.jsonl
  dynamic_selection/{report.json,selected_scenes.json,...}
  final_rcpd/{report.json,program.json,...}
  explanation_audit/{report.json,...}
  question_bank/{question_bank.json,report.json}
  tutorial.json
  production_admission.json
  warehouse_r41_online_release.zip
  warehouse_r41_online_release.b64
  release_receipt.json
```

The individual commands below are retained for diagnosis. They are not a
substitute for the single orchestration command when creating a release.

The exact paths below are placeholders for the selected training run and new
immutable output directories. Never overwrite an earlier output.

```bash
python -m backend.training.warehouse_r41_training_ledger \
  --run-root "$R41_RUN" \
  --output "$R41_OUT/training_ledger.json"

R41_LEDGER_SHA=$(shasum -a 256 "$R41_OUT/training_ledger.json" | awk '{print $1}')
```

The ledger must select the earliest 50,000-step boundary that passes both the
original validation suite and the frozen conflict validation suite. Its
selected Actor, protocol, and dual-evaluation report become the inputs below.
Because the frozen evaluator inherited a role bug that makes its `skilled` and
`fixed_yield` robot-1 programs identical, the additive audit below must replay
every committed boundary with six behaviorally distinct partners. It fails
closed unless the ledger Actor is also the corrected earliest passing Actor.

```bash
python -m backend.training.warehouse_r41_corrected_partner_audit \
  --training-ledger "$R41_OUT/training_ledger.json" \
  --expected-training-ledger-sha256 "$R41_LEDGER_SHA" \
  --dual-evaluation "$R41_DUAL_EVALUATION" \
  --conflict-manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --workers 4 \
  --output "$R41_OUT/corrected_six_partner_audit/report.json"

python -m backend.training.warehouse_r41_conflict_play_selection \
  --manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --actor "$R41_ACTOR" \
  --output "$R41_OUT/dynamic_selection" \
  --workers 4

python -m backend.training.warehouse_r41_final_rcpd \
  --actor "$R41_ACTOR" \
  --scenarios output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --output "$R41_OUT/final_rcpd"

python -m backend.training.warehouse_r41_explanation_audit \
  --actor "$R41_ACTOR" \
  --protocol "$R41_PROTOCOL" \
  --program "$R41_OUT/final_rcpd/program.json" \
  --scenarios output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --output "$R41_OUT/explanation_audit"

python -m backend.training.warehouse_r41_question_bank \
  --actor "$R41_ACTOR" \
  --protocol "$R41_PROTOCOL" \
  --conflict-manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --expected-conflict-manifest-sha256 4e640e19ea37405c3613d8f6ea1ace036ecff5da50cfb75c7e3f07d825f5b98e \
  --output "$R41_OUT/question_bank"

python scripts/build_warehouse_r41_neutral_tutorial.py \
  --conflict-manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --expected-conflict-manifest-sha256 4e640e19ea37405c3613d8f6ea1ace036ecff5da50cfb75c7e3f07d825f5b98e \
  --output "$R41_OUT/tutorial.json"
```

Create the admission only after every producer above has completed. The
`dual_evaluation` path must be the selected boundary's exact report. The
conflict validation and graph paths are the frozen publication files.

```bash
python scripts/build_warehouse_r41_admission.py \
  --actor "$R41_ACTOR" \
  --protocol "$R41_PROTOCOL" \
  --training-ledger "$R41_OUT/training_ledger.json" \
  --dual-evaluation "$R41_DUAL_EVALUATION" \
  --corrected-six-partner-audit-report "$R41_OUT/corrected_six_partner_audit/report.json" \
  --conflict-manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --conflict-validation output/warehouse_native/r41_conflict_scenes_v1_20260911/validation.json \
  --task-conflict-graph output/warehouse_native/r41_conflict_scenes_v1_20260911/task_conflict_graph.json \
  --dynamic-selection-report "$R41_OUT/dynamic_selection/report.json" \
  --selected-scenes "$R41_OUT/dynamic_selection/selected_scenes.json" \
  --final-rcpd-report "$R41_OUT/final_rcpd/report.json" \
  --final-rcpd-program "$R41_OUT/final_rcpd/program.json" \
  --explanation-audit-report "$R41_OUT/explanation_audit/report.json" \
  --question-bank "$R41_OUT/question_bank/question_bank.json" \
  --question-bank-report "$R41_OUT/question_bank/report.json" \
  --tutorial "$R41_OUT/tutorial.json" \
  --output "$R41_OUT/production_admission.json"

R41_ADMISSION_SHA=$(shasum -a 256 "$R41_OUT/production_admission.json" | awk '{print $1}')
```

Reverification requires the same sixteen component flags, plus:

```bash
python scripts/build_warehouse_r41_admission.py \
  ...same component flags... \
  --verify-existing "$R41_OUT/production_admission.json" \
  --expected-admission-sha256 "$R41_ADMISSION_SHA"
```

Only then build the archive from the same exact component set:

```bash
python scripts/build_warehouse_r41_online_release.py \
  --actor "$R41_ACTOR" \
  --protocol "$R41_PROTOCOL" \
  --training-ledger "$R41_OUT/training_ledger.json" \
  --dual-evaluation "$R41_DUAL_EVALUATION" \
  --corrected-six-partner-audit-report "$R41_OUT/corrected_six_partner_audit/report.json" \
  --conflict-manifest output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json \
  --conflict-validation output/warehouse_native/r41_conflict_scenes_v1_20260911/validation.json \
  --task-conflict-graph output/warehouse_native/r41_conflict_scenes_v1_20260911/task_conflict_graph.json \
  --dynamic-selection-report "$R41_OUT/dynamic_selection/report.json" \
  --selected-scenes "$R41_OUT/dynamic_selection/selected_scenes.json" \
  --final-rcpd-report "$R41_OUT/final_rcpd/report.json" \
  --program "$R41_OUT/final_rcpd/program.json" \
  --explanation-audit-report "$R41_OUT/explanation_audit/report.json" \
  --question-bank "$R41_OUT/question_bank/question_bank.json" \
  --question-bank-report "$R41_OUT/question_bank/report.json" \
  --tutorial "$R41_OUT/tutorial.json" \
  --production-admission "$R41_OUT/production_admission.json" \
  --expected-production-admission-sha256 "$R41_ADMISSION_SHA" \
  --output-package "$R41_OUT/warehouse_r41_online_release.zip" \
  --output-base64 "$R41_OUT/warehouse_r41_online_release.b64"
```

The release builder calls `read_saved_admission`, packages the independent
`final_rcpd_program`, and takes the tutorial plus X/Y scenes from
`selected_scenes.json`. A missing or failed producer leaves no admission and
therefore no deployable Secret File.

The admitted and packaged records always retain `formal_ready=false` and
`human_explanation_effect_validated=false`. Human-study evidence is required
before either statement can change.

## Render deployment preflight

The post-freeze receipt now records the portable `manifest_sha256` as well as
the admission, package, Base64 and Actor hashes. The preflight also requires
the disk-backed SQLite declaration in
`docs/warehouse_r41_persistence_audit.md`; the current Free `/tmp` setup is a
deliberate failure. Before changing the existing Render service, configure the
persistent storage declaration, update the two hash values in `render.yaml`
from that externally hashed receipt, and run:

```bash
R41_RECEIPT_SHA=$(shasum -a 256 "$R41_OUT/release_receipt.json" | awk '{print $1}')
python scripts/preflight_warehouse_r41_render.py \
  --release-root "$R41_OUT" \
  --expected-release-receipt-sha256 "$R41_RECEIPT_SHA" \
  --render-yaml render.yaml
```

Proceed only when the command returns `ready_for_manual_render_deploy`. The
exact Secret File bytes, staging allowlist, manual-deploy sequence and online
identity checks are documented in `docs/warehouse_r41_render_preflight.md`.
