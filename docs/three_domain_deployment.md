# Three-domain release v3.6

Release: `policylens-three-domain-20260920.v3.6`
Site: https://policylens-warehouse-study.onrender.com/

Status at this commit: implementation complete; production deployment and acceptance pending. The currently verified deployment is v3.5.2 (`acf29da0a8b6c2ddcb49185077112b2e0cd384f4`), including recovery of the entry button after a transient question-service outage. Final production evidence is recorded after the release, not inferred from local tests.

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
- Database/export regression: 28 passed. A real read-only production baseline covers 52,390 rows across old and current tables. Compare it after deployment to distinguish preserved records, legitimate active progress updates and new test records.

## Compatibility and data

v3.6 has incompatible Kitchen rules and uses a new release identity. Earlier records remain unchanged and archived; no old game state is interpreted using new rules. A returning participant starts the new demonstration rather than resuming an old game under changed physics. Release, domain, mode, group and scenario version must remain separate during analysis.

The six production A/B acceptance flows use explicitly synthetic identities in `test` mode. Do not treat test/preview records as recruited humans. A 50% Task 2 improvement remains a human pre-experiment target, not a guaranteed or measured result.

[Previous deployment and validation history](three_domain_deployment_v3_5_1.md).
