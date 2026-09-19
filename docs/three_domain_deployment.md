# Three-domain v3.2 Render deployment — ONLINE ACCEPTANCE PASSED

**Status: all three domains are deployed at the existing site in pilot mode. Six complete production A/B flows, 23 original semantic questions, full replay, actual Render restart, final same-source deployment and post-deployment persistence checks have passed.** The final public verification confirms persistent storage, configured QA and study readiness. Earlier failed attempts remain documented below. Human Task 2 improvement has not been measured.

Target website: https://policylens-warehouse-study.onrender.com/

This report is finalized from recorded acceptance receipts outside the frozen checkout. The preceding successful v2 deployment document is preserved byte-for-byte as `three_domain_deployment_v2.md`; its tests, browser coverage and persistence receipts remain historical **v2 evidence** and must not be presented as v3 validation.

## Frozen candidate and current deployment identity

| Field | Recorded value |
|---|---|
| Current release | `policylens-three-domain-20260920.v3.2` |
| Frozen runtime-correction commit | `d625acabdbbdf1d9a04fbb67c3e6e999233f0aaf` |
| Canonical source SHA-256 | `2e7eb2cde324b5bb873700bbfe6b0920ef8a74154f7914f929175c421a118b67` |
| Warehouse engine | `warehouse-legacy-restored.v3.3` |
| Pong engine | `pong-rolling.v4.0` |
| Kitchen engine | `kitchen-v4.0.1` |
| Final live pilot deployment | `dep-danfnvf40ujc73bpv3l0` |
| Prior v3.2 preview deployment | `dep-danfdojbc2fs73e7lp20` |
| Render service | `policylens-warehouse-study`, `srv-da66ggbl550s738j2q9g` |
| Implementation branch | `codex/three-domain-study-v3-20260920` |
| Region / runtime / plan | Existing Singapore / Python 3 / Free service |
| Build | `python -m pip install -r requirements-render.txt` |
| Start | `python -m ui.domain_hub_server --host 0.0.0.0 --port $PORT` |
| Health path | `/health` |
| Automatic deployment | Off; explicit manual deployment |

The operator first confirmed the v3.2 preview release at **2026-09-19 20:58:07 UTC** (September 20 04:58:07 in Asia/Shanghai), with that exact commit/hash and all three engines. The fresh main six-flow acceptance uses separate `_v3_2` output files. The earlier v3.1 public receipt remains `production_release_v3_1_preview.json`, identifying commit `b9237d4e30549e15935c25129922b3d22bcfb0e8`, source `1eee11a946d7bd034cab9a86994b9d6031279929fd408f00df99a4af59c177f2`, and preview deployment `dep-danf1cn40ujc73bnps0g`; that receipt is historical evidence.

The final pilot deployment reports **Live at September 20 05:20:00 +08:00**, with a new `rjlv7` startup log at 05:19:50 showing `study_ready=True`. At **21:21:01 UTC** (05:21:01 local), `/api/release`, `/health`, `/`, `/warehouse/`, `/pong/` and `/kitchen/` all returned HTTP 200. The exact same frozen commit/source hash is live, with `mode=pilot`, `storage_persistent=true`, `semantic_qa_configured=true`, `deployment_validation_complete=true`, and `study_ready=true`. Receipts: `render_pilot_deploy_v3_2_operation.json`, `production_release_v3_2_pilot.json`, and `production_final_public_readiness_v3_2.json`.

`frozen_candidate_v3_2.json` records the locally calculated source identity. Its local `storage_persistent=false` and provider fields are not Render runtime claims. The actual `production_release_v3_2_preview.json` reports persistent storage and semantic QA configured, with preview mode, unverified status and pilot entry closed before acceptance. Source identity, actual deployment, flow results and final readiness are supported by their separate receipts.

## Initial inherited-readiness incident

The initial v3 source deployment, `dep-danepjegekts738o7kgg`, inherited the old v2 `pilot` / verified flags because native value-setting had not persisted. At **20:19:02 UTC**, its public release reported `mode=pilot`, `deployment_validation_complete=true` and `study_ready=true`, although v3 online acceptance had not run. Those inherited flags were not valid v3 acceptance evidence.

The operator corrected the configuration using actual typed inputs, checked saved `preview` / unverified values, and redeployed the same frozen source before continuing six-flow acceptance. The original misleading-ready response is preserved in `production_release_v3_initial_inherited_flags.json`; it has not been replaced with the later corrected response.

A read-only v3 pilot export at **20:19:52 UTC** returned no records (`counts: {}`), preserved in `initial_v3_pilot_record_counts.json`. The operator repeated the read-only export after public entry was closed, at **20:27:53 UTC**, and again found no matching pilot records; receipt `v3_pilot_records_after_closure.json`. No record was deleted or changed to conceal the incident. The corrected initial v3 preview deployment and receipt remain `dep-danes6ugekts738ofqkg` / `production_release_v3_preview.json`.

## Runtime correction and immutable version separation

The original v3 source identity remains commit `849a5f4da2a3b8835a24353b07e00d456bffdfd8`, source SHA-256 `e1f60b99aa443e7f70b3b2a4522158220205d96d0c42c4baf113dbd3f502143c`. It was not overwritten under the same release ID.

The cold enrollment path generated the 120-frame Warehouse demonstration inside the global database transaction. Concurrent health checks also used a write transaction/global lock. The operator observed a 36-second first-session internal error and `LockNotAvailable` health failures. The v3.1 correction prewarms demonstrations outside the database transaction before serving requests, and runs health's database check as read-only. Gameplay, historical AI, scenarios, scoring and questions are unchanged. Twenty focused runtime tests passed before the v3.1 freeze; `frozen_candidate_v3_1.json` records the narrow patch scope and parent commit.

The v3.1 acceptance helpers use separate `_v3_1` filenames and output stems. They retain the original v3 failure trace, checkpoint each successful enrollment before the next, resume missing identity slots, and preserve each export attempt under a unique filename. The failure is not erased or relabeled as successful acceptance.

## Second online failure and semantic finding

The v3.1 attempt reached 2,288 recorded HTTP responses and completed Warehouse A/B, Pong A/B and Kitchen A. Kitchen B completed Task 1, then its export returned 502, followed by 503. The Render operator confirmed a 512 MB out-of-memory event around September 20 04:43 local time, with recovery around 04:44. The v3.1 export loaded historical records before filtering and duplicated the full response in memory. The filtered streaming correction was released under the new immutable v3.2 identity. No successful final six-flow export, restart or pilot redeployment is claimed for the failed v3.1 attempt. The v3.1 trace, partial checkpoints and 23 questions are preserved as intermediate regression evidence.

Independent review of the 23 original production questions found 21 accurate/adequate answers, one correct action with an inaccurate Kitchen waiting reason, and one Pong provider timeout. At Kitchen Task 2 turn 317, the final meal is already formally plated in the human hand, so the generic AI claim that it needs prepared protein is wrong; its wait action remains correct. The v3.2 release corrects this public reason without changing the action. A separate same-state Pong supplement passed semantic review, but the original provider availability remains 22/23; its conversation context differed from the original. See `production_semantic_v3_1_review.md` and the separate supplemental independent-review receipt.

The prepared `_v3_2` helpers use fresh output names and preserve the prior attempts. Transport exceptions retry the identical payload after 1/2/4/8 seconds; application errors are not automatically repeated. No generic availability bypass is present. The v3.2 release/source identity is frozen as listed above. Its six-flow, semantic, replay, restart and final pilot-deployment results are recorded below. SQL filtering and bounded named-cursor export are implemented, and the Kitchen reason now accurately describes the human's completed dish. The latter changes bilingual explanation text and engine version only: nine synthetic traces retained identical action/state behavior across 2,598 transitions, except the explicit version field. Direct replay of the recorded turn-317 bug confirms the corrected English/Chinese reason and unchanged next state. See `v3_2_runtime_independent_review.json`.

## Implemented study contract

All domains use Demo → Task 1 → Task 2 → Task 3 → Questionnaire, English on first entry and a top-right Chinese switch. Only Group A during an active Task 2 can ask and read answers. Task 1, Task 3, terminal Task 2 and all Group B stages do not expose explanation text. While active in Task 2, Group A may select a recorded earlier frame as its question target; that does not grant explanation access while playing Task 1 or Task 3. Completing Task 2 revokes access immediately, before Next.

Inputs advance authoritative logic; the confirmed transition is displayed with an approximately 400 ms animation. There is no automatic game clock and no confirmation button. Direction hold uses one outstanding action; E and Space act once per key press. Typing, replay, blur and task completion stop repeated movement.

- **Warehouse:** old 6-row × 7-column map, two shared A→B jobs, pickup ownership/replenishment, charger, original batteries, 120 turns and original raw score. WASD/arrows move; Space waits. Score has no `/100` ceiling and can be negative.
- **Pong:** 9 lanes, 12 vertical cells, 540×630 canvas; up to three small and two cooperative balls. Small balls descend two cells per action and score 1; cooperative balls descend one and require distinct human/AI contacts for 3. A/D/Space controls, 60/90/90 turns, predetermined rolling arrivals and natural drain. AI considers visible balls only, keeps the earliest reachable commitment and labels later assignments tentative.
- **Kitchen:** four cupboards, two stir-fry recipes, facing-dependent E, two preparation interactions, real protein-first temporary plating/storage, vegetable cooking, return/mix, output container, then human formal plating and serving. Hand/counter capacities, item identities, order bindings, eight-turn ready window, recovery and actual movement are retained. Budgets remain 240/360/360 for 4/6/6 orders.

## Warehouse historical provenance

Map/environment/scoring baseline: `af97df8589080a6b1b79bad591059b4d52fe33ee`. The conservative controller is actually imported from a byte-identical retained copy of `de16551d3d6b99c7ab426dfed1db2871159e6b5c`, whose source commit time is September 2 12:54:11 +08:00. No exact September 2 13:11 Render receipt has been verified; this is a fixed historical source identity, not proof of an exact historical deployment time.

Historical controller SHA-256: `ebcbf38c6365752a73b2175ab44750f3231da369421424f5cccd4aada23729c8`. Original v68 Actor SHA-256: `96762a46f59abd24a10b1abedf8dc325d72c85f3af39424c33e2dcba4ef5ffd3`. AI decisions precede receipt of the human action. Public explanations use the selected historical action, actual conflict checks, handoff/charge reasons and restored scoring; participant text does not expose neural-network internals.

## Completed offline and local validation

Before the v3.2 freeze, 474 tests plus 152 subtests passed in the main suite (`integrated_tests_v3_2.txt`), and the 13 newly added export tests separately passed, for **487 distinct tests plus 152 subtests**. The export/database focused run reports 28 passes, comprising those 13 new tests and 15 database tests already included in the main suite; the 15 are not counted twice. These are local results, including PostgreSQL test doubles. Actual Render export, health and persistence checks have separate receipts.

The original frozen gameplay candidate passed **469 tests and 152 subtests**, recorded in `integrated_tests.txt`; the subsequent v3.1 runtime-only patch separately passed twenty focused tests. This document does not infer that the original full-suite receipt was a post-patch full-suite run. Domain tests include historical Warehouse transition/controller parity, serialization and scoring; Pong deadlines, commitments, small-ball detours, finite supply and visible-boundary counterfactuals; Kitchen complete recipes, capacities, orientation, lineage, burning, deadline service and raw-input recovery. Shared tests cover A/B permissions, unchanged time during questions, replay, action versioning/idempotence, persistence interfaces and questionnaires.

The local in-app browser receipt (`local_browser_acceptance.json`) records the three English entries, restored Warehouse map, 540×630 five-ball Pong, A and Space one-step input with W ignored, idle turn stability, Kitchen front-only E/blocked-direction facing, refreshed state and an observed confirmed animation. At the time of that local receipt, later frozen-version language/question/replay and production checks were not yet completed; their separate results are recorded below. Physical long-key holding was covered with controlled keyboard/network unit tests, not a native long-hold browser primitive.

The actual native Chrome receipt is `production_native_browser_v3_2.json`. Its Warehouse map/scoring and Kitchen controls, front-only E, real bilingual QA, replay input and refresh checks were performed on v3.1; the shared frontend assets were unchanged by v3.2. Its Pong keyboard, five-ball/arrival display, idle stability and refresh checks were performed freshly on v3.2. These distinct release scopes are retained, and the fresh v3.2 HTTP/semantic checks separately cover the Kitchen reason correction. No measured animation FPS or native physical long-hold result is claimed.

At **21:22:22 UTC after the final pilot deployment**, native Chrome verified the live homepage's Research pilot badge, all three domain links and initial English, opened the public Kitchen enrollment form, switched it to Chinese, and returned home in English. The final screenshot was visually inspected. No synthetic or human enrollment was created by this entrance check. Its scope is the final public homepage/entry and language controls, separate from full gameplay and API evidence. Receipt: `production_final_native_entry_v3_2.json`.

Local real-provider semantic checks are separately reviewed in `real_semantic_review.md`: the original 18 selected questions yielded 15 accurate/adequate answers, one correct but over-expanded answer, and two incomplete answers. Three targeted repeats after prompt clarification addressed those gaps. Four additional revised-Kitchen questions were accurate and adequate; two exact-evidence-ID warnings remain unchanged and are explained as redundant-ID limitations. These are selected synthetic-state checks, not a general accuracy estimate or production-flow acceptance.

## Simulation results and remaining design limits

- Pong: all 144 cooperative-proxy trajectories across 24 development and 24 separate held-out seeds × three tasks scored 100 with zero AI waiting. This is an intentionally feasible schedule plus a planning proxy, not a guarantee for arbitrary human actions. In held-out Task 2, a stronger public-history proxy averaged 77.69; 100 versus 77.69 is approximately 28.7% relative gain. These proxies are not A/B participants and do not establish a 50% human gain. Source receipts: `pong_rolling_calibration.json` and `pong_rolling_calibration_baselines.json`.
- Kitchen: 144 fixed-controller development/reserved-regression trajectories completed all orders, scored 100 and burned no portions within the retained budgets. Both pans have overlapping recipe stages, but simultaneous heating was zero; those are distinct metrics. Four raw-first mistake/recovery trajectories also completed six orders without waste. The 144 records are stored in `configs/study_v3_kitchen.json` under `calibration_results` for kitchen-v4.0.0; v4.0.1 changes reasons only, with separate action/state parity and current regression tests. The preserved `domains/kitchen/validation_results.json` describes the older soup engine and is not evidence for this two-recipe release. This is software feasibility, not human explanation benefit.
- Warehouse: preserving the exact historical controller also preserves its limitations. A recorded synthetic advice rollout on the three fixed task seeds produced 2/13/0 deliveries and raw scores −161/−22/−193; Tasks 1 and 3 ended in shutdown. Source: `docs/study_v3_warehouse.md`, current historical-restoration verification section. The controller was not silently changed to improve these observations. When real Group B raw means are zero or negative, report absolute score and delivery differences rather than a misleading relative percentage.

**Human Task 2 improvement remains unmeasured. The 50% target is a pilot objective, not a guaranteed result.**

## Completed v3.2 six-flow and semantic acceptance

The fresh v3.2 script completed **2,949 HTTP responses with no recorded failure**, all six A/B instances, 18 task runs and six questionnaires. Its synthetic actors operated the real action interfaces; no state/score injection was used. The final test export is hash `f645b7cd231494fd015a92b2175ab532d41a02890b65413471beacfb4ee00d56`. The main receipt is `production_http_acceptance_v3_2_summary.json`. Earlier attempts remain separate and are not erased by this result.

All **23 original v3.2 questions** returned answers, with zero unavailable responses and no supplemental request used. Independent manual review judged all 23 selected answers accurate and adequate; this is not a general model-accuracy estimate. All selected state hashes and complete evidence catalogs matched the final authoritative frames, and eight recorded counterfactual simulations were independently reexecuted exactly. The corrected Kitchen turn-317 reason accurately states that the human holds the final plated dish and needs no new ingredient. Pong's four-wait answer stops at the actual one-step public boundary and leaves the unknown continuation undisclosed. Minor verbosity and public-label translation limits are documented in `production_semantic_v3_2_review.md`.

Independent offline reconstruction verified **18 runs, 2,674 transitions and 2,692 frames**, including every nonterminal saved AI decision, complete private/public state, terminal state and score. Terminal frames correctly save `{}` because no next decision is executed; the checker initially expected an executable terminal decision, and its own failure/correction is transparently preserved in `production_export_v3_2_checker_initial_failure.json`. This was a checker defect, not a runtime state mismatch. The successful exact replay receipt is `production_export_v3_2_offline_replay_receipt.json`.

The protocol flow checks cover A/B permission isolation, question time remaining unchanged, immediate terminal-Task2 revocation, answer acknowledgment revocation, version/idempotent command handling, English/Chinese switching, refresh recovery, questionnaires and exports. These are real HTTP flows plus independent engine replay. Native browser coverage is reported separately by the operator and must not be inferred from API checks.

## Actual restart and final pilot deployment passed

The operator triggered **Manual Deploy → Restart service** on the actual Render service at September 20 05:09 +08:00. Native Render Events recorded the operation. The startup instance changed from `tfbdq` (04:58:00) to `lqnhg` (05:11:09), whose log identifies the same v3.2 release. Operation evidence: `render_restart_v3_2_operation.json`.

At **21:15:50 UTC**, the verification script recovered the captured Warehouse B probe with exactly the same turn-3 view, revision and state hash, executed one new wait, and confirmed turn 4 was persisted in the researcher export. It also compared the complete immutable records for **six completed instances, six enrollments, 18 runs, 2,692 frames, 23 question/private audit rows and six questionnaires**, all exactly preserved. Receipt: `production_restart_probe_v3_2_receipt.json`. The completed-record digest is `0908e1fff88ce94f1a9bd0579580da7d031c60996d1c590b5ceea84c6f810601`.

The operator then explicitly deployed the same frozen commit with saved `pilot` / verified settings. Actual Render deployment `dep-danfnvf40ujc73bpv3l0` became Live, and the final release/health/three-domain entry checks passed as recorded above. This was a real deployment operation, not just a saved environment change.

At **21:21:09 UTC**, the post-deployment checker recovered the exact ongoing Warehouse turn-4 view, executed one new wait, and verified persisted turn 5 in the researcher export. The same **six instances, six enrollments, 18 runs, 2,692 frames, 23 question/private audit rows and six questionnaires** again matched the complete pre-restart baseline byte-for-byte at the canonical record level. The completed-record digest remained `0908e1fff88ce94f1a9bd0579580da7d031c60996d1c590b5ceea84c6f810601`. Receipt: `production_pilot_redeploy_v3_2_persistence_receipt.json`. Neither persistence check asked a new model question; each executed only its one documented continuation action.

## Actual v3.2 streamed export check

At 21:00:21–21:00:29 UTC, the configured 512 MB Render service successfully returned one authorized unfiltered export: **56,812,577 bytes in 7.774 seconds**, HTTP 200, matching Content-Length. Five concurrent health requests all returned 200, with maximum observed latency 0.375 seconds. The snapshot contains 36 instances, 82 runs and 6,507 frames with valid relationships. Primary keys present in the referenced older v2/v3.1 snapshots were retained; this key-preservation check is separate from the later exact completed-study row comparison. Receipt: `production_streamed_export_v3_2_receipt.json`.

Server peak resident memory was not exposed or measured. A separate slow local PostgreSQL benchmark was stopped before completion, as recorded in `streamed_export_pg_v3_2_partial_receipt.json`; the actual successful HTTP export does not turn that local benchmark into a completed test.

## Preservation and runtime boundaries

Implementation used the isolated `XAI-study-v3` checkout; the original dirty working checkout remains preserved. Existing Neon PostgreSQL records, legacy variables and secret files are retained. No old gameplay state is silently migrated to new rules. Keep release IDs and study modes separate in exports and analyses.

The owner waived an old Warehouse backup because there was no Warehouse data requiring one. No such backup is claimed. The pre-existing Kitchen backup remains historical preservation evidence in the earlier private evidence directory; credentials, recovery secrets and research rows are outside Git. No secret values appear in this document.

The final verified public release is pilot mode, verified and ready. The persistent database and server-only language-service credentials remain configured. The actual final deployment and post-deployment persistence receipts establish the live promotion; it is not inferred from a saved setting or requested action. The existing Render Free service can cold-start; no paid plan change is part of this release.

Deployment completion, simulated performance and human effect are separate outcomes. Technical deployment and the listed acceptance checks passed; the 50% human Task 2 target remains unmeasured. The archived v2 success report and failed v3/v3.1 attempts remain separate evidence.
