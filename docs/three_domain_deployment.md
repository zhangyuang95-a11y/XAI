# Three-domain release v3.6

Release: `policylens-three-domain-20260920.v3.6`
Site: https://policylens-warehouse-study.onrender.com/

Deployed to the original Render service: `dep-danrghuk1f9s739muij0`, triggered 2026-09-20 18:41:14 +08:00 and verified live at 18:42:45. Runtime commit: `11071b54c631ae9da3c4d5bb1533d8dc8ad1d4f0`. Runtime source SHA-256: `5fada514c11f5e4f01bf3294317eff50dfd55451d8083a2639fe8928771a7af1`. The public manifest matches both. All three entrances and health return 200; persistent storage and actual question-service readiness are true. The entry-readiness recovery from v3.5.2 is included.

Production HTTP acceptance completed all six A/B flows, including 18 task runs and six questionnaires. Live behavior and saved records were checked separately from the local tests below.

## Changes

- All domains bind an unqualified “why this action?” to the saved action that produced the displayed frame (`t−1 → t`). Explicit next-action, current position, human advice and human counterfactual questions use the displayed current state. Historical frames use the same convention; turn 0 explicitly has no previous action. The audit retains both displayed and decision turns and source hashes. The answer distinguishes the original decision from an interaction cancelled during joint resolution.
- Pong retains ball IDs but removes arrival-turn counters from the board. Ball motion, speed, schedule and scoring are unchanged.
- Kitchen raw/prepared ingredients left untouched in either AI ingredient slot or the human storage counter spoil after more than 10 elapsed turns. Exactly 10 is allowed; picking up on the first overdue turn does not restore freshness. Spoiled portions remain physical items until disposed of. Cooked-protein temporary plates are recipe workstations and are excluded from this raw-ingredient storage rule.
- A finished dish reaches the handoff counter before its two complete waiting turns begin. On the third still-blocked turn the AI starts toward the bin. If the counter clears before disposal, including during the joint bin-interaction turn, the dish is retained and delivery resumes. Only actual disposal incurs −10.
- The Kitchen controller can start another order on the unused pan while the first order is still in progress. Its choice checks real walking, turning and loading time against stored-ingredient expiry. Urgent pan rescue retains priority. This is overlapping work on two dishes, not a claim that one mobile cook keeps both pans heating simultaneously.
- Existing PostgreSQL persistence is retained. Actions, states, decisions, questions and their outcomes, surveys and unfinished progress are stored. See [data dictionary and export instructions](study_data_storage.md).

Only Group A during active Task 2 can ask or view explanations. Default English, Chinese switching, continuous demonstration playback and input-driven animated gameplay remain unchanged.

## Validation before deployment

Evidence directory: `../analysis/three_domain_revision_20260920_v36/`; credentials and session material are private and excluded from Git.

- Full final three-domain regression: **732 tests and 184 subtests passed**. Four frontend checks passed (entry recovery, continuous demo/latest answer, input/replay handling and renderer labels).
- Kitchen: 84 tests plus 184 subtests passed; the final joint-disposal event refinement passed a further targeted 21 tests plus 32 subtests. 144 simulated cooperative trajectories completed all five dishes, without burning, spoilage or disposal; longest run 283 of 360 steps. These are simulated actors, not human outcomes. A separate explicitly synthetic same-state comparison demonstrates that v6 starts the second pan where v5 continued the first recipe.
- Explanation: 417 QA tests and 28 temporal tests passed; Store tests passed separately. Real configured language-service checks verified 27 current/past/next/counterfactual questions, six bilingual selected-history questions and two cancelled-disposal questions, including evidence IDs and actual stored actions. Finite examples do not guarantee every free-form answer.
- Independent review found no blocking temporal or permission defect and checked all three domains' historical terminal frames without advancing the active task.
- Local Chrome: Pong continuous demo completed and entered Task 1; actual board shows ball IDs without arrival counters. Renderer checks cover both languages. The entry form remains gated by real service readiness for pilot users; local preview uses an isolated SQLite database.
- Database/export regression: 28 passed. A real read-only production baseline covers 52,390 rows across old and current tables; post-deployment comparisons are reported below.

## Production acceptance

- Six synthetic `test` flows completed on the deployed server: Warehouse A/B, Pong A/B and Kitchen A/B. Across 18 task runs, 2,610 gameplay transitions were compared against expected public states and scores. All six questionnaires and contiguous saved frames were verified in the authenticated export.
- The 25 main real-service questions passed content and provenance review against saved states, decisions and simulation results. They cover English/Chinese, performed versus next action, follow-ups, selected historical frames, counterfactuals, hypothetical Pong positions and Kitchen storage/handoff rules. Two additional Pong `preview` answers passed a separate content check. This finite check is not a guarantee for every possible free-form question. The review records two remaining readability issues in factually correct answers: occasional Warehouse internal parcel/status labels and Pong normalized score decimals.
- Permissions were checked in each stage: only A in active Task 2 can ask/view answers, with immediate revocation at Task 2 completion. Questions do not advance gameplay. Command retries do not duplicate actions; cross-identity access is denied, and language and progress survive a fresh session-state read.
- The first test-harness preflight incorrectly compared mixed-case requested test identities with the server's normalized lowercase identities. Its three assertion failures remain in the log. Correcting only the harness comparison allowed the same identities to complete all flows; no production source change was needed. See `production_harness_identity_note.md` in the evidence directory.
- Both read-only database comparisons preserved all **52,390 baseline rows**, with no missing rows or unexpected changes. The final comparison was at 2026-09-20 10:54:26 UTC. Legitimate existing active-progress updates and newly added synthetic records are itemized separately. The six completed flows and six questionnaires are all `test`; an additional Pong session is `preview`. No v3.6 `pilot` participant existed at that audit time.
- An owned Warehouse test session was recovered with identical Task 1 turn 3, revision 10, run, group and full state. Deployment launched the new runtime while retaining the existing database. No separate manual restart was performed; the recovery check is not represented as a restart test.
- The production homepage was observed in Chrome with default English and all three domain links. The three deployed frontend assets match the locally verified files byte-for-byte. The task canvas was visually checked locally; a separate production canvas browser check was not completed because browser access timed out.

Evidence includes `production_deployment_receipt.json`, `production_final_health_v3_6.json`, `production_http_acceptance_v3_6_summary.json`, `production_answer_review_v3_6.md`, `production_frontend_assets_verified.json`, `production_database_preservation_after_v36.json`, `production_session_recovery_after_v36.json` and `production_data_mode_audit_after_v36.json`. Private exports and session credentials remain outside Git.

## Compatibility and data

v3.6 has incompatible Kitchen rules and uses a new release identity. Earlier records remain unchanged and archived; no old game state is interpreted using new rules. A returning participant starts the new demonstration rather than resuming an old game under changed physics. Release, domain, mode, group and scenario version must remain separate during analysis.

The six production A/B acceptance flows use explicitly synthetic identities in `test` mode. Do not treat test/preview records as recruited humans. A 50% Task 2 improvement remains a human pre-experiment target, not a guaranteed or measured result.

[Previous deployment and validation history](three_domain_deployment_v3_5_1.md).
