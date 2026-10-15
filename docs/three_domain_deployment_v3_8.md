# Kitchen operations tutorial and rules v3.8

Release family: `policylens-three-domain-20260922.v3.8` / `v3.8.1`.
Engine/scenario: `kitchen-v6.2.0` / `kitchen-scenarios-v6.2.0`.
Menu: `kitchen-menu-v2`. Tutorial: `kitchen-operations-tutorial.v1`.
Target: https://policylens-warehouse-study.onrender.com/ (existing Render service).
Publication status: v3.8.1 is Live on the existing Render service. Local,
production, database and post-deployment acceptance are complete.

## Participant behavior

Kitchen now begins with seven interactive operation exercises: movement/facing,
pantries, preparation, handoff/storage, a separate already-finished dish fixture,
freshness/trash, and menu/deadlines/scoring. Exercises use real keyboard-triggered
kitchen transitions with a neutral waiting teammate. They never run the cooking
policy, demonstrate a recipe sequence, or contribute formal task turns/scores.
Completion advances each section automatically; start/pause/retry/skip controls
remain. Public expandable help is identical for A/B and excludes AI strategy.
Warehouse/Pong demonstrations and mechanics remain unchanged.

Every Task has five dishes and 360 turns, with deadlines 100/140/240/280/360.
Menus are seeded, vary across Tasks, and prohibit three consecutive identical
dishes. The two recipes heat for 8+6+2=16 and 10+8+2=20 turns respectively.
Preparation takes tomato/pepper3, egg4, meat5 interactions. Prepared ingredients
expire exactly20 turns after preparation in any pre-pan location. Transfers never
reset the timestamp. A load submitted for the expiry turn is refused. After a
successful load the pan uses cooking and eight-full-ready-turn burning rules.
Unprepared raw ingredients retain the previous rules.

Serving awards30 and the same action costs1 (net+29); every turn costs1, actual
component disposal costs3 and combined dish disposal10. Spoilage/burning/expiry
alone add no penalty. Five completed dishes award150 before time/disposal costs;
raw scores may be negative. The two-turn handoff grace/cancellation remains.
Only groupA during active Task2 can ask/view answers; explanations of a performed
action use its saved pre-action evidence. English is the initial page language.

## Persistence and isolation

The existing PostgreSQL path is retained. Additive `pl3_tutorials` and
`pl3_tutorial_events` tables persist independent sandbox snapshots and command
records, including starts, pauses, retries, skipped/completed flags, and physical
practice transitions. Formal runs/frames continue to record rule metadata, actual
seeded menus, ingredient identities/timestamps, before/after actions, scores,
events, questions/counterfactual evidence and questionnaires. Export includes the
new tables with release/mode filtering and credential omission.

A new release ID prevents pre-v3.8 Kitchen states from resuming under changed
rules. The v3.8.1 QA-only patch can resume v3.8 snapshots without converting them:
physical rules, menus, tutorial and scoring are identical, and each existing
record keeps its original release ID. No old row or +100-era score is recomputed.
New question audits record both the session release and actual answer-runtime
release, so a v3.8 session resumed under v3.8.1 remains attributable without
rewriting an earlier answer or mixing the two formatter versions silently.
Deployment preparation exported the
existing study data consistently (23,264 frames among75 participants/99 instances,
230 runs; private backup outside the repository). No credentials enter the commit.
Automated online enrollments use test mode; local interactive browser QA uses
preview mode. Production browser entry/language checks create no participant.

Read-only PostgreSQL comparison after the first deployment found all 189
previously completed runs and their 22,191 frames unchanged. It compared saved
run state/score and frame state/public state/decision/action values against the
consistent pre-deployment export. Active sessions are excluded from the
unchanged-content claim because participants could legitimately continue them.

## Verification

See `kitchen_16_20_verification.md` for current, separately reported development,
regression and predeclared new validation evidence. These are simulated partners,
not human study results or evidence of a50% explanation benefit.

## Deployment receipt

The first production deployment is Render `dep-daombfqjnfac73erdurg`, commit
`42c9c28f46f84116fdd1ed2d61c33ea1f015de48`. It reached Live on
2026-09-21 at 17:15 UTC (2026-09-22 in Australia/Melbourne).
The public release endpoint reported v3.8 and source SHA-256
`125f2e19a5f9a3a1fd483f0d5345b4571f21d667da9fbc59faa9e61db1134964`, with
durable storage, configured QA and study readiness all true.

The complete local regression suite passed 839 tests plus 176 subtests for v3.8,
then 846 tests plus 176 subtests after the QA patch. An additional question
version-provenance regression passed separately after that run was collected.
Seven
frontend checks passed. Interactive browser testing completed all seven tutorial
sections, preparation/freshness markers, pause/refresh recovery and Task 1 entry.
Production browser checks confirmed English on entry and after refresh, Chinese
switching, all three domain links, and no recovery-code input.

Fourteen real-provider bilingual spot checks passed factual/temporal review.
These are a bounded seven-scenario check, not unrestricted semantic coverage.
Subsequent production acceptance found a correct but incomplete answer to a
compound counterfactual question: score and completed orders were answered, but
the requested food freshness change was omitted. The v3.8.1 patch supplies that
fact from the actual simulated final state. Original v3.8 answers are retained.
Real-provider English and Chinese replays both returned the correct final
holding/freshness fact after the patch. One English attempt first returned an
unavailable result for an invalid provider plan; a subsequent explicit retry
succeeded with the usual schema repair. This failed-closed response is retained
in the audit and is not counted as a successful answer.

The v3.8 production acceptance made 1,939 requests without network/gateway
retries, then replayed 1,818 saved actual transitions against the frozen engine.
Both A and B completed all three Kitchen tasks and their questionnaires:

| Task | Dishes served per group | Turns | Raw score | Verified joint-heating transitions |
|---|---:|---:|---:|---:|
| 1 | 5 | 304 | -154 | 1 (turn 47) |
| 2 | 5 | 298 | -148 | 1 (turn 47) |
| 3 | 5 | 299 | -149 | 0 |

The group trajectories used identical seeded scenes and controls; this is a
mechanics/permissions test, not a comparison of human explanation effects.
A completed the seven tutorial exercises; B skipped them. Both completed tasks,
questionnaire saving and immediate Task 2 permission revocation. Warehouse/Pong
entry and three actual steps each also passed. The export includes tutorial
completion/skip/action records independently of the formal task frames.

The second deployment is Render `dep-daomlalbedkc73ask5jg`, commit
`afa1f19fe486deb67c587a3b89178359886de2ec`. It reached Live on 2026-09-21 at
17:36 UTC (2026-09-22 in Australia/Melbourne). Public metadata reports v3.8.1,
source SHA-256
`1a619f33f0c182db4cb67e7f9745ef34910db102f5e8b0e5f6ff706fb9daa140`, persistent
storage, configured QA and study readiness. Physical engine, configuration,
tutorial and all frontend bytes are unchanged from the validated v3.8 source.
The 189 historical completed runs and 22,191 frames remained unchanged on a
second read-only comparison after this deployment.

The post-patch export comparison passed for the original completed flow records:
eight runs (six Kitchen plus the two Warehouse/Pong smoke runs), 1,816 frames,
five original answers, two questionnaires and tutorial records remained exactly
unchanged. The separate active prepared-meat probe recovered identically at
turn 28, prepared on turn 10, expiry turn 30, with two turns remaining. Its 28
physical transitions were replayed. Completed A-session ask and answer-display
acknowledgement both remained forbidden. This checks persistence across an
actual redeployment; no additional manual restart is claimed.

New v3.8.1 B enrollment, tutorial skip, a physical step, denied QA and refresh
passed. New A completed an actual Task 1 (five dishes, 304 turns, raw score -154)
and entered Task 2. All three patch-specific production answers passed
independent review against saved state/action hashes and freshly executed
simulations:

- English and Chinese at turn 10: prepared meat stays in the same hand, expiry
  remains turn 30, and two hypothetical waits reduce remaining freshness from
  20 to 18, score by 2, and completed orders by 0.
- Chinese at turn 29: one hypothetical wait reaches turn 30, the same portion
  becomes spoiled with zero turns remaining, physically stays in hand, and
  score changes by -1 with no completed order.
- Asking changed neither the actual game turn nor revision. The database
  exported both answer and session release IDs correctly.

This patch acceptance made 374 requests and replayed 334 actual transitions.
One read-only request needed a transport retry; there were no gateway retries
or explicit question resubmissions. Two provider plans used the normal bounded
schema-repair path; all three final answers were complete. This is a bounded
spot check, not a guarantee that an arbitrary question always succeeds.

## Evidence and limits

Full private exports and credentials remain outside the repository. The
non-secret receipts are under `analysis/kitchen_v38_20260922/` in the parent
workspace: `deployment_receipt.json`, `old_records_preservation.json`,
`production_acceptance/acceptance_v38_summary.json`,
`production_patch_acceptance/compatibility_receipt.json`,
`production_patch_acceptance/acceptance_v38_summary.json`, and
`production_patch_qa_review.md`.

All requested engineering and deployment work is complete. Human performance
improvement has not been measured: the 50% Task 2 gain remains a human pilot
target. Actual joint heating is present in 114/288 simulated scenes for one
verified joint decrement per positive scene, not continuous concurrency for
all players. The language provider can occasionally return an invalid plan;
the service repairs within its bound or safely reports unavailable, without
inventing an explanation. Some fact-based replies still include repeated
context. Historical +100 scores and original incomplete answers remain intact.
