# English-first interface patch v3.7.1

Release: `policylens-three-domain-20260920.v3.7.1`.
Site: https://policylens-warehouse-study.onrender.com/.
Status: deployed and verified on the existing Render service on 2026-09-20.
Runtime commit: `6377dae816b9b81bca819be7090839ad97054459`.
Source SHA-256: `a7a24237ff4bf450144bfb6e9e2eeb10493c338bb92586cf2abe53d1af9422a8`.
Render deployment: `dep-dant3ebm8hqs73cf9cog`, submitted by the user through the Dashboard after the browser-control tool could not click deployment.

Every page load starts in English, even if the browser or resumed study instance previously selected Chinese. The first restored view synchronizes its server language without advancing the game. The header still switches between English and Chinese during the current visit. All three entry forms omit the recovery-code field. Same-browser session recovery and protected researcher access are retained.

Gameplay, scoring, AI policies, Q&A permissions and database schema are unchanged. v3.7 sessions remain compatible and keep their original release IDs, runs, frames and scores. Exports should include both release IDs when collecting all records across this presentation patch; filtering only v3.7.1 would exclude continuing v3.7 sessions.

Focused validation: 56 existing HTTP/domain-switching tests passed, JavaScript syntax and existing entry/frontend checks passed. Isolated execution verified English startup despite an old Chinese preference, resumed-language synchronization, an explicit Chinese switch and absent recovery-code controls in each domain. Local database reopen checks preserved all three v3.7 task fixtures and nine saved frames; language changes left game turns and states unchanged.

Evidence: `analysis/three_domain_revision_20260920_v371/`. These are software checks, not human-effect results.

Production checks confirmed the exact release/commit/source and persistent database readiness. Native Chrome showed English entry forms without a recovery-code field in Warehouse, Pong and Kitchen. Pong's manual Chinese switch still worked, and refreshing returned the page to English. No enrollment or gameplay was submitted during these browser checks.

Read-only production recovery preserved both prior v3.7 test sessions, their six task identities, turns, scores and final views. All 511 records from their previous export remained unchanged, including 486 frames, 10 questions and two questionnaires. The verification made no production writes. The receipt is `production_session_preservation_receipt.json`; the initial verification harness's public-versus-internal score comparison correction is documented separately in `production_preservation_harness_note.md`.
