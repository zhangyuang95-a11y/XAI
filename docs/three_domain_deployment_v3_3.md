# Three-domain v3.3 interface update — deployed and verified

Site: https://policylens-warehouse-study.onrender.com/

The requested interface update is live on the existing Render service. This report covers the September 20 screenshot-driven UI and enrollment revision. The preceding full gameplay implementation and its evidence remain in [the v3.2 report](three_domain_deployment_v3_2.md), preserved byte-for-byte; its earlier 23 questions and complete six-flow acceptance are not counted again here.

## Deployed identity

- Release: `policylens-three-domain-20260920.v3.3`
- Runtime commit: `202e34f58970294bb6ce27004eeeff4a8cdae1f0`
- Source SHA-256: `c147e1c1c63832c1b4d0731da00361d16e3a11833af2bd690df65057990b2356`
- Existing service: `srv-da66ggbl550s738j2q9g`
- Render deploy: `dep-danlc1rm8hqs73bn0stg`
- Actual Render status: **Deploy succeeded | Live**
- New instance `kwdph` ready at September 20 11:44:05 +08:00; Live at 11:44:16.
- Public release confirms pilot mode, persistent database, configured question service, and study readiness. Automatic deployment remains off.

Subsequent documentation-only commits do not change that deployed runtime identity.

## Requested changes

1. All three domains place demonstration playback, replay and next-step controls in the right sidebar beneath the task rules.
2. Displayed rules are short bilingual summaries. Warehouse states only the A-to-B delivery goal, attention to battery, WASD movement and Space waiting. Pong and Kitchen summarize their main objective and controls. Full authoritative rules remain on the server for accurate answers.
3. The recovery-code reminder/banner is removed from demonstrations and task screens. Identity protection and the optional recovery input remain available; recovery secrets are not exposed in this report.
4. The duplicate below-board status, score breakdown, inventory and order blocks are removed from all domains. The board retains its visual state and top-level score/turn indicators. Kitchen retains a compact count of pending dishes in its rule panel so the task goal remains visible.
5. Warehouse cargo badges resolve the current frame's actual order to A1/A2 and match its color, including replenishment and historical frames. The battery label no longer overlaps the badge. Known order IDs in demonstration captions use the same visible A label.
6. Domain navigation is available during the study. Multiple domains can remain unfinished; switching and refreshing preserve their independent progress.
7. New domain enrollment exposes A/B selection. Existing domain enrollment keeps its original group when resumed, even if a different group is submitted. Public self-selection is recorded as `participant_choice`; authenticated researcher preview/test overrides remain `researcher_override`. Missing group input retains the earlier default behavior.

Demo → Task 1 → Task 2 → Task 3 → Questionnaire, English by default, the Chinese switch, and **A-only active Task 2** question/read permissions remain intact. The gameplay engines, controllers, scores, scenarios and database schema are unchanged.

## Verification

- **387 distinct Python tests passed:** 342 shared HTTP, question, startup, export and badge tests; 45 Store/domain-switching tests. Final receipts: `shared_tests.txt` and `store_tests.txt`.
- Existing frontend keyboard/network/replay harness and the Warehouse drawing harness passed. Badge checks cover both actors, replenishment, replay, animation transitions and battery separation.
- Independent review identified a pending-dish counter that included completed orders; this was corrected before the frozen commit and independently verified in both languages. No remaining material blocker was found.
- Actual local UI exercised public pilot group selection using an isolated local database and an explicitly identified answer test double. Warehouse B, Pong A and Kitchen B were opened under one identity before any domain finished; returning to Warehouse preserved demonstration step 2.
- **298 actual production HTTP requests passed:** six new test-mode domain enrollments across both A/B choices, unfinished-domain switching, immutable group on resume, Task 1 and B Task 2 denial, and immediate A Task 2 completion revocation. These are targeted revision checks, not another claim of six complete questionnaire flows.
- Two real production Pong questions, one English and one Chinese, were independently verified against saved authoritative state, the fixed AI decision and exact evidence catalogs. Both were accurate and complete; neither advanced state or revision. This two-question check is separate from the preceding release's 23-question review.
- A captured ongoing v3.2 synthetic Warehouse session recovered its exact pre-deployment view and original group. Its turn-5 state continued to turn 6 and persisted, with the old release ID retained.
- Actual native Chrome incognito preview exercised all three domains with B/A/B selections, confirmed the compact rules and right-side demonstration layout, observed the Warehouse A1 badge, tested English/Chinese text, and preserved Warehouse step 2 after cross-domain navigation and refresh. Existing user cookies were not replaced. No public pilot research enrollment was created by these production checks.

## Preservation and interpretation

Existing v3.2 game states are explicitly compatible because this revision does not change gameplay or stored-state formats. Their instance IDs, release IDs, groups and historical records are retained; subsequent viewing uses the current interface. Older incompatible gameplay versions are not silently migrated. An incompatible old domain does not block entry to another domain.

Exports retain assignment provenance. Self-selected groups must not be described as randomized groups, and release/mode/assignment distinctions must remain visible in analysis. No participant outcome or 50% explanation benefit is asserted by these software checks.

The original working checkout remains untouched. Changes are on the existing isolated `XAI-study-v3` branch. Credentials, test cookies, recovery values and raw exported rows remain outside Git.

Evidence directory on the development machine: `/Users/zhangyuang/Desktop/ICLR/analysis/three_domain_ui_revision_20260920`. Main receipts are `frozen_candidate.json`, `render_deployment.json`, `final_public_release.json`, `production_smoke.json`, `production_native_ui.json`, `independent_review.md` and `production_qa_review.md`.
