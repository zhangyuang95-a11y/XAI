# Warehouse r4.1 diagnostic: v13 post-final release commands

Run this sequence only after the v13 one-shot outer and protected final both
complete with passed status. The commands do not read the holdout salt and do
not call Render. Replace the two angle-bracket placeholders with the successful
v13 final directories. All remaining inputs are frozen repository artifacts.

```bash
cd /Users/zhangyuang/Desktop/ICLR/XAI
umask 077

sha256_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

ACTOR=output/warehouse_native/r41_active_2m_20260911/boundaries/step_2000000/actor.npz
PROTOCOL=output/warehouse_native/r41_active_2m_20260911/protocol.json
SOURCE_MANIFEST=output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/manifest.json
SOURCE_MANIFEST_VALIDATION=output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/validation.json
RUNTIME_MANIFEST=output/warehouse_native/r41_diagnostic_runtime_manifest_v9_20260914/runtime_manifest.json
DESIGNATION=output/warehouse_native/r41_diagnostic_designation_v2_sourceclosure2_20260913/diagnostic_actor_designation.json
DEVELOPMENT_ROWS=output/warehouse_native/r41_outer_v6/rows.npz
SELECTED_SCENES=output/warehouse_native/r41_diagnostic_dynamic_selection_v3r1_20260912/selected_scenes.json

PROMOTION_DIR=output/warehouse_native/r41_diagnostic_final_attempt_closeout_v13_20260914
PROMOTION_REGISTRY=output/warehouse_native/r41_diagnostic_final_attempt_closeout_v13_permanent_registry_20260914
SELECTOR_DIR=output/warehouse_native/r41_diagnostic_rcpd_v13_selector_20260914
OUTER_COLLECTION_DIR=output/warehouse_native/r41_diagnostic_rcpd_v13_fresh_outer_collection_20260914
OUTER_RESULT_DIR=output/warehouse_native/r41_diagnostic_rcpd_v13_outer_once_20260914
OUTER_REGISTRY=output/warehouse_native/r41_diagnostic_rcpd_v13_outer_attempt_registry_20260914
FINAL_DIR='<v13-passed-final-output-directory>'
FINAL_REGISTRY='<v13-permanent-final-registry-directory>'

CANDIDATE_LOCK="$SELECTOR_DIR/candidate_lock.json"
PROGRAM="$SELECTOR_DIR/program.json"
OUTER_COLLECTION_REPORT="$OUTER_COLLECTION_DIR/collection_report.json"
OUTER_ROWS="$OUTER_COLLECTION_DIR/rows.npz"
OUTER_RESULT="$OUTER_RESULT_DIR/outer_result.json"
FINAL_ANCHOR="$FINAL_DIR/attempt_anchor.json"
FINAL_COMPLETION="$FINAL_DIR/attempt_completed.json"
FINAL_MATERIAL="$FINAL_DIR/final_material.json"
FINAL_ROWS="$FINAL_DIR/final_rows.npz"
FINAL_PARITY="$FINAL_DIR/observation_projection_parity.json"
FINAL_AUDIT="$FINAL_DIR/explanation_audit.json"

COMPACT_DIR=output/warehouse_native/r41_diagnostic_compact_program_v10_20260914
STUDY_DIR=output/warehouse_native/r41_diagnostic_study_materials_v10_20260914
RELEASE_DIR=output/warehouse_native/r41_diagnostic_online_release_v10_20260914
mkdir -m 700 "$COMPACT_DIR" "$RELEASE_DIR"

python scripts/build_warehouse_r41_diagnostic_compact_program_v10.py \
  --program "$PROGRAM" \
  --expected-program-sha256 "$(sha256_file "$PROGRAM")" \
  --development-rows "$DEVELOPMENT_ROWS" \
  --expected-development-rows-sha256 "$(sha256_file "$DEVELOPMENT_ROWS")" \
  --combined-promoted-rows "$PROMOTION_DIR/combined_promoted_rows.npz" \
  --expected-combined-promoted-rows-sha256 "$(sha256_file "$PROMOTION_DIR/combined_promoted_rows.npz")" \
  --outer-rows "$OUTER_ROWS" \
  --expected-outer-rows-sha256 "$(sha256_file "$OUTER_ROWS")" \
  --final-rows "$FINAL_ROWS" \
  --expected-final-rows-sha256 "$(sha256_file "$FINAL_ROWS")" \
  --output-program "$COMPACT_DIR/program.ctree.xz" \
  --output-report "$COMPACT_DIR/report.json"

python scripts/build_warehouse_r41_diagnostic_study_materials_v10.py \
  --actor "$ACTOR" --expected-actor-sha256 "$(sha256_file "$ACTOR")" \
  --protocol "$PROTOCOL" --expected-protocol-sha256 "$(sha256_file "$PROTOCOL")" \
  --source-manifest "$SOURCE_MANIFEST" \
  --expected-source-manifest-sha256 "$(sha256_file "$SOURCE_MANIFEST")" \
  --candidate-lock "$CANDIDATE_LOCK" \
  --expected-candidate-lock-sha256 "$(sha256_file "$CANDIDATE_LOCK")" \
  --program "$PROGRAM" --expected-program-sha256 "$(sha256_file "$PROGRAM")" \
  --final-audit "$FINAL_AUDIT" \
  --expected-final-audit-sha256 "$(sha256_file "$FINAL_AUDIT")" \
  --combined-promoted-rows "$PROMOTION_DIR/combined_promoted_rows.npz" \
  --expected-combined-promoted-rows-sha256 "$(sha256_file "$PROMOTION_DIR/combined_promoted_rows.npz")" \
  --promotion-closeout "$PROMOTION_DIR/closeout_receipt.json" \
  --expected-promotion-closeout-sha256 "$(sha256_file "$PROMOTION_DIR/closeout_receipt.json")" \
  --permanent-promotion-closeout-registry "$PROMOTION_REGISTRY" \
  --output "$STUDY_DIR"

QUESTION_BANK="$STUDY_DIR/question_bank/question_bank.json"
QUESTION_REPORT="$STUDY_DIR/question_bank/report.json"
TUTORIAL="$STUDY_DIR/tutorial.json"
COMPACT_PROGRAM="$COMPACT_DIR/program.ctree.xz"
COMPACT_REPORT="$COMPACT_DIR/report.json"
ADMISSION="$RELEASE_DIR/admission.json"
PACKAGE="$RELEASE_DIR/warehouse_r41_diagnostic_online_release.zip"
BASE64="$RELEASE_DIR/warehouse_r41_diagnostic_online_release.b64"
RECEIPT="$RELEASE_DIR/release_receipt.json"
PREFLIGHT="$RELEASE_DIR/render_preflight.json"
RENDER_CANDIDATE="$RELEASE_DIR/render.v10.yaml"
```

The admission, package, receipt, and preflight commands share the exact
component map below. This function is only shell text reuse; each Python reader
still independently authenticates the bytes and append-only registries.

```bash
components=(
  --designation "$DESIGNATION"
  --actor "$ACTOR"
  --protocol "$PROTOCOL"
  --source-manifest "$SOURCE_MANIFEST"
  --source-manifest-validation "$SOURCE_MANIFEST_VALIDATION"
  --runtime-manifest "$RUNTIME_MANIFEST"
  --candidate-lock "$CANDIDATE_LOCK"
  --development-rows "$DEVELOPMENT_ROWS"
  --promotion-closeout "$PROMOTION_DIR/closeout_receipt.json"
  --promotion-identity-registry "$PROMOTION_DIR/burned_identity_registry.json"
  --promotion-observation-projection "$PROMOTION_DIR/ordered_promoted_observation_hashes.json"
  --combined-promoted-projection "$PROMOTION_DIR/combined_promoted_observation_hashes.json"
  --promoted-burned-final-rows "$PROMOTION_DIR/promoted_rows.npz"
  --combined-promoted-rows "$PROMOTION_DIR/combined_promoted_rows.npz"
  --program "$PROGRAM"
  --compact-program "$COMPACT_PROGRAM"
  --compact-program-report "$COMPACT_REPORT"
  --outer-result "$OUTER_RESULT"
  --outer-collection-report "$OUTER_COLLECTION_REPORT"
  --outer-rows "$OUTER_ROWS"
  --final-anchor "$FINAL_ANCHOR"
  --final-completion "$FINAL_COMPLETION"
  --final-material "$FINAL_MATERIAL"
  --final-rows "$FINAL_ROWS"
  --final-projection-parity "$FINAL_PARITY"
  --final-audit "$FINAL_AUDIT"
  --question-bank "$QUESTION_BANK"
  --question-bank-report "$QUESTION_REPORT"
  --selected-scenes "$SELECTED_SCENES"
  --tutorial "$TUTORIAL"
)

python scripts/build_warehouse_r41_diagnostic_admission_v10.py \
  "${components[@]}" \
  --outer-permanent-registry "$OUTER_REGISTRY" \
  --final-permanent-registry "$FINAL_REGISTRY" \
  --permanent-promotion-closeout-registry "$PROMOTION_REGISTRY" \
  --output "$ADMISSION"

ADMISSION_SHA="$(sha256_file "$ADMISSION")"
python scripts/build_warehouse_r41_diagnostic_online_release_v10.py \
  --admission "$ADMISSION" --expected-admission-sha256 "$ADMISSION_SHA" \
  "${components[@]}" \
  --outer-permanent-registry "$OUTER_REGISTRY" \
  --final-permanent-registry "$FINAL_REGISTRY" \
  --permanent-promotion-closeout-registry "$PROMOTION_REGISTRY" \
  --output-package "$PACKAGE" --output-base64 "$BASE64"

python scripts/build_warehouse_r41_diagnostic_release_receipt_v10.py \
  --admission "$ADMISSION" --expected-admission-sha256 "$ADMISSION_SHA" \
  "${components[@]}" \
  --outer-permanent-registry "$OUTER_REGISTRY" \
  --final-permanent-registry "$FINAL_REGISTRY" \
  --permanent-promotion-closeout-registry "$PROMOTION_REGISTRY" \
  --package "$PACKAGE" --base64 "$BASE64" --output "$RECEIPT"

PACKAGE_SHA="$(sha256_file "$PACKAGE")"
MANIFEST_SHA="$(python - "$PACKAGE" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
import zipfile
with zipfile.ZipFile(Path(sys.argv[1])) as archive:
    print(sha256(archive.read("manifest.json")).hexdigest())
PY
)"

python - "$PACKAGE_SHA" "$MANIFEST_SHA" "$RENDER_CANDIDATE" <<'PY'
from pathlib import Path
import re
import sys
package_sha, manifest_sha, output = sys.argv[1:]
text = Path("render.yaml").read_text(encoding="utf-8")
text = re.sub(
    r"(?m)^    startCommand:.*$",
    "    startCommand: python -m ui.warehouse_alignment_online_server "
    "--release-module ui.warehouse_alignment_r41_diagnostic_release_v9 "
    "--base64 /etc/secrets/warehouse_alignment_release.b64",
    text,
)
text = re.sub(
    r"(?m)(^      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256\n"
    r"        value: )[0-9a-f]{64}$", r"\g<1>" + package_sha, text)
text = re.sub(
    r"(?m)(^      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256\n"
    r"        value: )[0-9a-f]{64}$", r"\g<1>" + manifest_sha, text)
Path(output).write_text(text, encoding="utf-8")
Path(output).chmod(0o600)
PY

python scripts/preflight_warehouse_r41_diagnostic_render_v10.py \
  --admission "$ADMISSION" --expected-admission-sha256 "$ADMISSION_SHA" \
  "${components[@]}" \
  --outer-permanent-registry "$OUTER_REGISTRY" \
  --final-permanent-registry "$FINAL_REGISTRY" \
  --permanent-promotion-closeout-registry "$PROMOTION_REGISTRY" \
  --package "$PACKAGE" --base64 "$BASE64" \
  --receipt "$RECEIPT" \
  --expected-receipt-sha256 "$(sha256_file "$RECEIPT")" \
  --render-yaml "$RENDER_CANDIDATE" | tee "$PREFLIGHT"
```

Expected binding chain:

1. `candidate_lock.json` binds the fixed Actor, protocol, source manifest,
   v13 selector program, base development rows, combined promoted rows, and
   promotion closeout.
2. `outer_result.json` binds that lock/program and the exact outer rows and
   collection report in the permanent outer registry.
3. `attempt_completed.json` binds the passed nine-gate audit, exact physical
   replay, cross-split isolation, and materializer/collector projection parity
   in the permanent final registry.
4. The compact report must match the exact `development`, `fresh_outer`, and
   `protected_final` observation arrays used above. It may not alter Actor
   actions or control runtime actions.
5. The admission binds all offline evidence, the unchanged participant UI,
   six scenes, questionnaire, tutorial, and release source closure. The ZIP
   packages only Actor, protocol, portable runtime manifest, compact program,
   question bank, and tutorial.
6. The receipt binds admission, ZIP, Base64, and manifest bytes. The preflight
   reopens all evidence, tests a clean-checkout load, and requires the already
   whitelisted v9 runtime module while preserving the public
   `r4.1-diagnostic` identity.
