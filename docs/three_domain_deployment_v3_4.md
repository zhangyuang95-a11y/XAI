# Three-domain concurrent Pong and Kitchen recovery revision

Deployment target: https://policylens-warehouse-study.onrender.com/

This revision adds concurrent Pong decisions and fixes kitchen food loss. The preceding deployed interface and enrollment report is preserved in [the v3.3 report](three_domain_deployment_v3_3.md). The gameplay revision is live as v3.4 (commit `02b2fd255dabf6aebec3dc9cd141d34a6c872826`, Render deploy `dep-danm8pp42hec73ev8c4g`, live September 20, 2026 at 12:45:33 UTC+8). A QA-only v3.4.1 correction, commit `5800e96f394951f762e682cf0d6e81e92680e7ab`, is tested and pushed; its deployment is pending the final Render check.

## Changes

- Pong displays up to six small and four team balls. Small pairs arrive every two moves; every six moves two incompatible team balls and two small balls arrive together. A player cannot catch everything. Physics remains two cells per move for small balls, one for team balls, and one lane per paddle. The fixed, group-blind controller plans over visible balls only. Actual catch points remain 1/3; no group-dependent schedule or score adjustment is used.
- Pong questions include current-action reasons, named alternatives, human action consequences, and an editable 1–9 hypothetical human lane. Position counterfactuals deep-copy the selected state, change only the specified human position, and recompute the actual fixed AI. Optional subsequent actions use real physics and stop at the next undisclosed arrival boundary. The real game and its time do not change.
- Answers select a small number of relevant verified facts. Full private simulation evidence remains in the audit; participants see concise answers.
- Comprehension quiz questions are removed from all three questionnaires. Ratings and optional feedback remain. New questionnaire records store an empty comprehension array; historical questionnaires are not rewritten.
- Kitchen uses clear outlined facing arrows, preparation remaining/ready labels, and actual current dish labels above the AI. A shared trash bin is at (4,4), reachable from either side. Disposal requires carrying an item to the bin, facing it and pressing E. Remote discard is unavailable.
- Kitchen no longer silently deletes expired-order food. Finished dishes remain held, stored or handed off; unusable or burnt contents first transfer into a hand and then physically travel to the bin. An ingredient-component conservation invariant checks acquisition, transfers, serving and disposal on every step.
- Existing Warehouse mechanics and historical controller remain unchanged. Compact rules, sidebar demonstration controls, A/B selection and unfinished-domain switching remain.

Only group A during active Task 2 can ask or read answers. English remains the default, with Chinese available at the top right. Demo → Task 1 → Task 2 → Task 3 → Questionnaire remains common to all domains.

## Version and preservation

The release is `policylens-three-domain-20260920.v3.4`. Pong is `pong-concurrent.v5.0`; Kitchen is `kitchen-v4.1.0`. Previous gameplay is not silently migrated. A returning identity sees a new-version notice and can begin a separate enrollment from the demonstration; every old instance, frame, score, answer and questionnaire remains under its original release. An existing current-version domain keeps its progress and immutable A/B group across switching and refresh.

## Validation

- Full shared/domain regression: 546 tests plus 152 subtests passed. Final QA-specific suite: 333 tests passed. Frontend network/keyboard/replay checks and Warehouse/Pong canvas regression checks passed.
- Kitchen's real former bug is reproduced: seed1000 Task2 created finished item1 on move65; its order expired at130, and the old controller remotely discarded it on134. With identical acquisition and then150 waits, the new engine still preserves the identical dish and component IDs at215.
- Kitchen 144 deterministic scene rollouts complete all orders with no burns; task maxima are221/335/336 moves within unchanged240/360/360 budgets.
- Pong 144 deterministic scene rollouts were evaluated. Held-out task means are64.31/64.03/64.17 out of100. The schedule's per-arrival contact-capacity upper bound, ignoring movement, is66.67; this is an upper bound, not a proven feasible optimum. Scores are not inflated to conceal unavoidable missed balls.
- Pong mean held-out waiting is8.96%/11.06%/12.04%; the previous strict waiting goals are not all met and the calibration report records that failure. No claim of near-zero waiting is made.
- Eight real-provider local questions were verified against authoritative states, including English/Chinese position interventions, a named alternative, follow-up and historical frame. These are software/semantic checks, not human-study outcomes.

Evidence: `/Users/zhangyuang/Desktop/ICLR/analysis/three_domain_tradeoffs_revision_20260920`. Credentials and raw test sessions remain private and outside Git. Human Task2 improvement remains unmeasured;50% is still a pre-experiment target.

## Actual production acceptance

Six complete A/B test flows ran against the real deployed service: 2,958 HTTP requests, 18 task runs, 6 submitted questionnaires, and 2,674 state transitions checked against the fixed engines. Records use synthetic identities and `mode=test`. No human-performance inference is made.

All flows verified action retries without duplicate moves, refresh, historical-frame access, group/task explanation permissions, no time advancement on questions, and immediate revocation after Task2 ended. The exported six instances, 18 runs and six questionnaires matched the recorded flows. Export SHA256: `16e1ee8d6573aa07afdfb7fd4e1e3cabef2a048c37ef267db7b80b162418ef1a`.

All26 production questions returned answers, but independent semantic review accepted25/26: one t3 question described isolated reachability without clearly saying the controller actually selected competing t4. The v3.4.1 patch binds named-ball assignment answers to actual selected/tentative/not-selected facts; it also separates nearest-ball distance from the chosen catch. Seven real-provider questions on the exact failed production state passed locally, and432 related regression tests passed. AST comparison confirms only Pong facts changed; the fixed AI, physics, scores, seeds and state schema are identical. Existing v3.4 sessions remain compatible and retain their original release metadata. Older gameplay remains archived.

Native Chrome UI checks used a separate preview session. The ten-ball court, editable hypothetical-lane question, unchanged game turn during real English QA, Chinese labels, clear Kitchen arrows, dish name and prep countdown were inspected. Actual E completed preparation at turn19; movement then faced the bin and E disposed of the same item at turn22. The original browser game session was not used for testing.

A read-only production archive probe checked six related rows of one prior v3.3 synthetic instance: before/after SHA256 `d0d1a6b3140e526b52d49e4140f3f4629888be99f17d06fc038c8b6dfcf27be1` matched. This sampled preservation check is not a claim of a full database backup.

The final patch has captured ongoing Warehouse/Pong/Kitchen test states for exact recovery comparison across redeployment. Final patch deployment and post-deployment answers must be confirmed before this report marks them complete. At the last attempt, native Chrome returned `invalid element` and `noWindowsAvailable`; an independent browser attempt reproduced this. The candidate deployment was not triggered. Resume with exact commit `5800e96f394951f762e682cf0d6e81e92680e7ab`, then run the preserved private production verification helper. No permission or account changes were made.
