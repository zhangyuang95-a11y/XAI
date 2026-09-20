# English-first interface patch v3.7.1

Release: `policylens-three-domain-20260920.v3.7.1`.
Site: https://policylens-warehouse-study.onrender.com/.
Status: implementation verified; production deployment pending.

Every page load starts in English, even if the browser or resumed study instance previously selected Chinese. The first restored view synchronizes its server language without advancing the game. The header still switches between English and Chinese during the current visit. All three entry forms omit the recovery-code field. Same-browser session recovery and protected researcher access are retained.

Gameplay, scoring, AI policies, Q&A permissions and database schema are unchanged. v3.7 sessions remain compatible and keep their original release IDs, runs, frames and scores. Exports should include both release IDs when collecting all records across this presentation patch; filtering only v3.7.1 would exclude continuing v3.7 sessions.

Focused validation: 56 existing HTTP/domain-switching tests passed, JavaScript syntax and existing entry/frontend checks passed. Isolated execution verified English startup despite an old Chinese preference, resumed-language synchronization, an explicit Chinese switch and absent recovery-code controls in each domain. Local database reopen checks preserved all three v3.7 task fixtures and nine saved frames; language changes left game turns and states unchanged.

Evidence: `analysis/three_domain_revision_20260920_v371/`. These are software checks, not human-effect results.
