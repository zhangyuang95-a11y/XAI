# Three-domain release v3.7

Release: `policylens-three-domain-20260920.v3.7`
Site: https://policylens-warehouse-study.onrender.com/

Status: v3.7 is live on the existing Render service; production workflow, finite question/evidence review and data acceptance are complete.

Runtime commit: `f96a0dc8bbbca4c9fc020c49b7f69edb80064eec`.
Runtime source SHA-256: `7d8b5283db250fcc0ed690423d93db23d64e0c5b314fb27272bb74ca4cae3b7c`.
Render deployment: `dep-dansh8ajnfac739n8ot0`, submitted through the Dashboard at 2026-09-20 19:50:57 Asia/Shanghai. The user submitted the deployment after browser-control failures; the public release was independently verified at 19:53:51. The release, commit and source hash match the tested candidate. Health is ready, storage is persistent, all three domain entries return HTTP 200, and the served JavaScript/CSS bytes match the candidate.

## Pong behavior

The AI first selects a reachable position where a small ball and one end of a cooperative large ball arrive together. It checks that the human can reach the other end when establishing the assignment. Once selected, its near-term route remains fixed when the human moves elsewhere. If the human misses the cooperation opportunity, the AI keeps its own small-ball route rather than swapping sides to follow the human. When no simultaneous combination is available, it uses an achievable sequential or single-ball target. Both players are still required for a large-ball reward.

The default answer explains the actual action, target lane, relevant ball IDs and arrival, plus the human's other contact where needed. It omits score arithmetic and optimality claims. Questions about other actions or positions use the real controller and simulator. Completed-action explanations retain the saved decision that produced the selected displayed frame; future-arrival wording uses that displayed frame's time.

Ball schedules, movement speed, scoring, keyboard controls and A/B access rules remain unchanged. Warehouse and Kitchen behavior are unchanged. Only A during active Task 2 can ask or view explanations.

## Data and compatibility

Existing Neon PostgreSQL remains the production database. No migration deletes or rewrites earlier participant records. v3.7 has a new research release identity because Pong policy changed; saved older games are retained under their original release and are not resumed with the new policy. New-version participation starts at the demonstration. Analyses must separate releases, domains, modes and assigned groups.

Successful submitted actions, saved states/decisions, questions/results and questionnaires continue to be persisted. See [storage and export documentation](study_data_storage.md).

## Validation evidence

Evidence directory: `/Users/zhangyuang/Desktop/ICLR/analysis/three_domain_revision_20260920_v37/`. Credentials and session material stay in its private directory, outside Git and public assets.

- Full three-domain regression: **757 tests and 184 subtests passed**. This includes 56 Pong tests and 458 question/frame checks, with the current 80-case bilingual Pong composition corpus reproducible from its builder. Prior corpus files are retained as historical evidence.
- Actual configured language provider: **11/11** finite English/Chinese questions answered and reviewed against saved target/transition evidence. Default incoming-action answers are at most two sentences; the countdown refers to the displayed frame, and completed catches use actual results. Follow-up action comparisons and hypothetical positions were also checked. The source hash was stable throughout this probe.
- Fixed-controller simulation: 144 trajectories across every existing development and held-out seed and all three tasks. All 9,600 retained-route checks passed. Held-out Task 2 averaged 60.81/100; 356 of 360 successful large-ball catches also caught a small ball at the AI contact. The schedule includes incompatible opportunities, and neither this policy nor its explanations claim globally optimal score.
- Independent adversarial checks: 360 transitions with randomized human deviations, 216 human-position interventions and 59 performed combination explanations. Frozen AI actions and displayed countdowns matched their evidence.
- Production workflows: both synthetic Pong A/B flows completed through demonstration, all three tasks and questionnaire: **572 HTTP requests, 480 state comparisons, 6 completed runs and 2 questionnaires**, with no protocol failures or transport retries. The 10 real-provider questions were answered and saved. Checks cover immediate Task 2 permission revocation, question requests leaving time unchanged, duplicate-action idempotency, language persistence, cross-identity isolation and recovery.
- Production question review: **10/10** actual answers match their saved question/decision evidence. Six default action/reason answers are at most two sentences. Three counterfactuals were reproduced from the recorded state. The review was a separate answer-to-evidence pass by the QA integration author, not an external blind review. This finite set does not establish correctness for every possible question.
- Production storage: the final read-only row comparison preserves **all 58,610 baseline rows across 23 tables**, with no missing or changed rows. New records include 486 frames, 10 questions and both questionnaires; the current-release test export matches the submitted actions, contiguous frames, final scores, exact answers, evidence and display receipts. Earlier v3.6 evidence is retained separately.

Public deployment evidence is in `production_deployment_receipt.json` and `production_public_verification.json`; protocol, question review and database receipts are `production_http_acceptance_v3_7_summary.json`, `production_qa_review.json` and `production_database_preservation_after_v37.json`. This release's online acceptance is HTTP/API based; it does not claim a new visual browser inspection of every game screen.

Simulations and synthetic production checks are not human-study results. No human Task 2 improvement is measured or guaranteed by this release.

[v3.6 deployment and verification](three_domain_deployment_v3_6.md).
