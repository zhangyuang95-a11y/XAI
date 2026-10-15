# Warehouse r4 training closeout

Closeout date: 2026-09-11. Status: **failed frozen gates; not released**.

This round continued from the frozen r3 Actor
`309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b`.
The authorized r4 budget was 1,000,000 fresh PPO joint steps. The authenticated
failed-ledger record accounts for exactly 1,000,000 steps across 18 attempts
and 20 unique training segments, leaving zero unused steps. The parent's
3,950,000 cumulative steps are lineage context and are not counted again as
fresh r4 work.

All 18 attempts kept the neural policy authoritative at runtime. The selection
record reports zero action overrides for every attempt and no script-based
runtime replacement. The final v18 fixed screen likewise records an action
equality rate of 1.0 across 7,019 submitted decision frames. The environment
continued to resolve physical collisions after submission; it did not select a
different policy action.

## Selection result

No checkpoint passed every frozen behavior gate, so `selected_actor_sha256` is
null. The v8 checkpoint
`bc9c269738d31565460065747d855dfb026fa2a841432cd612149de24a7cce5c`
was retained only as the best diagnostic candidate. Its full paired audit used
50 validation scenes and six partner types, for 300 episodes:

| Metric | r3 baseline | v8 diagnostic | Frozen requirement | Result |
| --- | ---: | ---: | ---: | --- |
| Active non-WAIT rate | 84.81% | 85.93% | at least 85% | pass |
| Productive action rate | 72.20% | 73.72% | at least 65%; improve by 8 points | absolute pass, relative fail |
| Mean AI deliveries | 8.623 | 8.793 | improve by at least 1 and 15% | fail |
| AI delivery share | 63.91% | 65.20% | at least 35% | pass |
| AI shutdowns | 19 | 45 | zero | fail |
| Mean longest collision streak | 5.00 | 4.44 | at most 3 | fail |
| No-progress streak p95 | 35.10 | 38.05 | at most 18 | fail |
| Action overrides | 0 | 0 | zero | pass |

Only one of the five required relative checks passed; at least four were
required. The final v18 continuation was screened on 60 fixed episodes before
a full audit. Relative to its r3 screen, productive action rose by 2.385
percentage points and mean AI deliveries by 0.367, while it recorded six AI
shutdowns, a no-progress p95 of 27 and a mean longest collision streak of
3.017. It therefore was not promoted to the 300-episode audit.

## Conflict-scene diagnostic

The v8 Actor was also used for a development-only selector diagnostic. Of 300
new states, 99 passed the geometric filter and none passed every dynamic gate.
The diagnostic ran 11,880 episodes and submitted 1,377,725 Actor decisions
with zero overrides. It produced neither a balanced six-scene selection nor a
`selected_scenes.json` file. This diagnostic does not make v8 release eligible.

## Release consequence

The fail-closed pipeline produced no r4 release Actor, final RCPD program,
runtime component set, frozen r4 questionnaire, admission record, Render Secret
File, or deployable archive. The training-budget ledger has
`status=failed_no_eligible_actor` and `admission_eligible=false`. The r4 online
builder and production admission continue to reject this state.

The existing Render site was not updated to r4 and remains the r3 internal
pilot. A live check on 2026-09-11 returned service version
`warehouse-alignment-online-study-server.v1` from `/api/health` and frontend
version `warehouse-alignment-online.v1` from `/assets/app.js`. These markers
match the tracked
[`warehouse_alignment_online_deployment_manifest.json`](warehouse_alignment_online_deployment_manifest.json),
which binds the same `309b6e53...` Actor above. The public status endpoint does
not expose the Actor hash, so this is a match against the recorded deployment,
not a fresh cryptographic read of the remote Secret File. The site remains an
ephemeral, `formal_ready=false` technical pilot.

## Local evidence retained outside Git

The underlying reports and model artifacts remain in the ignored
`output/warehouse_native/` tree and must not be committed. The paths and file
hashes below identify the local evidence used for this closeout:

| Evidence | Repository-relative path | SHA-256 |
| --- | --- | --- |
| Training closeout | `output/warehouse_native/r4_active_v8_hardstate_v18_1m_20260911/closeout_report.json` | `6f1010c5c442aa5117bd34ffb2a59deed07bc1e1a76100b59f5824942d35741d` |
| Final selection | `output/warehouse_native/r4_active_final_selection_20260911/selection_report.json` | `f48b6181851dca46011e4570e019857d0da09c52191cb8a062a74ab6f237147f` |
| Authenticated failed budget ledger | `output/warehouse_native/r4_release_closeout_20260911/training_budget_failed.json` | `54b9d043ee024c881ec802fdc02dbf3a5e180e05d76d7671b730966f8286e18a` |
| Selector v3 report | `output/warehouse_native/r4_conflict_diagnostic_v8_selector_v3_300_20260911/report.json` | `7e786698ed7b6bf2d09bd211c730c86a64dd06e703f0a4a8c256e722d3aea3b6` |
| Selector gate summary | `output/warehouse_native/r4_conflict_diagnostic_v8_selector_v3_300_20260911/gate_summary.json` | `f406a9750c1312cb74c382ac5264c85dbbb3cdb3921e9c2ce433ad0d18ae1ac0` |

This document records the failed round without copying ignored reports,
checkpoints, actors, trajectories, participant data or deployment secrets into
Git. A future r4 attempt must start a new versioned evidence chain and pass the
same frozen gates before any online replacement is built.
