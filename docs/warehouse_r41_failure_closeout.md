# Warehouse r4.1 budget-exhaustion closeout

Use this path only when the r4.1 runner reaches exactly 2,000,000 additional
PPO joint steps and no 50,000-step boundary passes both frozen validation
suites. It creates a non-release record. It does not select a diagnostic Actor,
run the post-freeze release chain, build a ZIP or Base64 Secret File, or call a
Render mutation API.

The training ledger already fails closed in this case:

- its status is `failed_no_eligible_actor`;
- `selected` is null and `admission_eligible` is false;
- all forty 50k boundaries remain available with their Actor, checkpoint,
  RCPD, dual-evaluation and training-segment hashes;
- `run_warehouse_r41_postfreeze_release.py` reads the ledger with
  `require_selected=True` before creating its output directory, so the release
  pipeline cannot start from this ledger.

The additional closeout record fills the remaining provenance gap. It rebuilds
the ledger from the untouched run, hashes every regular file under every
boundary, requires exact NN action submission and zero overrides, and binds the
three read-only diagnostics. In particular, it preserves the confirmed
`skilled`/`fixed_yield` validation alias and records that the frozen protocol
omitted the runner and evaluator from `implementation_sources`. The closeout
binds those two omitted source files, while leaving
`protocol_source_closure_complete=false`; it does not rewrite history.

The corrected six-partner replay is a release gate for a selected Actor. When
all forty Actors have already failed the original dual-suite gate, a corrected
partner result cannot make any of them release eligible. The closeout therefore
records that this gate was not run, binds its protocol and producer-source
closure, and keeps `admission_eligible=false`. This avoids spending hours on a
post-freeze audit that cannot change the failure decision.

After the runner has written terminal `run.json`, create a new ledger and hash
all four external inputs. Never overwrite an earlier ledger or closeout:

```bash
R41_RUN=output/warehouse_native/r41_active_2m_20260911
R41_FAILURE=output/warehouse_native/r41_failure_closeout_$(date +%Y%m%d_%H%M%S)
mkdir -m 700 "$R41_FAILURE-ledger"

python -m backend.training.warehouse_r41_training_ledger \
  --run-root "$R41_RUN" \
  --output "$R41_FAILURE-ledger/training_ledger.json"

R41_LEDGER_SHA=$(shasum -a 256 "$R41_FAILURE-ledger/training_ledger.json" | awk '{print $1}')
R41_ALIAS_SHA=$(shasum -a 256 docs/archive/warehouse_r41_fixed_yield_alias_diagnostic_20260911.json | awk '{print $1}')
R41_ENERGY_SHA=$(shasum -a 256 docs/archive/warehouse_r41_energy_gate_audit_20260911.json | awk '{print $1}')
R41_TREND_SHA=$(shasum -a 256 docs/archive/warehouse_r41_training_trend_audit_20260911.json | awk '{print $1}')

python scripts/build_warehouse_r41_failure_closeout.py \
  --run-root "$R41_RUN" \
  --training-ledger "$R41_FAILURE-ledger/training_ledger.json" \
  --expected-training-ledger-sha256 "$R41_LEDGER_SHA" \
  --partner-alias-diagnostic docs/archive/warehouse_r41_fixed_yield_alias_diagnostic_20260911.json \
  --expected-partner-alias-diagnostic-sha256 "$R41_ALIAS_SHA" \
  --energy-gate-diagnostic docs/archive/warehouse_r41_energy_gate_audit_20260911.json \
  --expected-energy-gate-diagnostic-sha256 "$R41_ENERGY_SHA" \
  --trend-diagnostic docs/archive/warehouse_r41_training_trend_audit_20260911.json \
  --expected-trend-diagnostic-sha256 "$R41_TREND_SHA" \
  --output-root "$R41_FAILURE"
```

The command performs two read-only deployment-retention checks. It inspects
`origin/main:render.yaml` and makes GET requests to the existing service's
`/health` and `/api/view` endpoints. The origin configuration must still bind
the registered r3 package and manifest, automatic deployment must remain off,
and the live public shape must remain consistent with the twelve-scene r3
release. The participant-safe endpoint intentionally omits Actor hashes, so the
record describes the Actor identity as an inference from the reverified r3
package and the configured package hash. It does not claim access to Render's
private deployment state.

The new output directory contains exactly one file:

```text
failure_closeout.json
```

Hash it outside the record and re-read it when needed:

```bash
R41_CLOSEOUT_SHA=$(shasum -a 256 "$R41_FAILURE/failure_closeout.json" | awk '{print $1}')
python - <<'PY' "$R41_FAILURE/failure_closeout.json" "$R41_CLOSEOUT_SHA"
import sys
from backend.training.warehouse_r41_failure_closeout import read_saved_closeout
value = read_saved_closeout(sys.argv[1], expected_sha256=sys.argv[2])
print(value["status"], value["release_eligible"], value["formal_ready"])
PY
```

The expected output is
`closed_budget_exhausted_no_eligible_actor False False`. Do not update the
package or manifest hashes in `render.yaml`, upload a Secret File, or trigger a
manual deployment after this closeout. The existing r3 service remains the
active internal pilot; the r4.1 boundary Actors remain diagnostic evidence.
