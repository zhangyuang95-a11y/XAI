# Three-domain validation record

> Historical v2 implementation/evidence snapshot, retained for audit. The new v3 revision is documented in [three_domain_revision_v3.md](three_domain_revision_v3.md); deployment status is tracked separately.

Recorded on 2026-09-20 (Asia/Shanghai) in the isolated `XAI-study-v3` worktree.
Release `policylens-three-domain-20260920.v2` is actually deployed at
<https://policylens-warehouse-study.onrender.com/>. Production commit:
`68a50da906e30a3ac2f7712858cc2b19d2cd159a`; source SHA-256:
`387513ef55e9d2c5ca68c061159b389bc281b5bad88ef4972a399ea960de9858`.
This report separates local tests, real hosted acceptance, automated proxy
scores and the still-unmeasured human effect. The final pilot-configuration
deployment is live on the same source commit: public release and health responses
confirm pilot mode and readiness. Public participant enrollment is open.

## Overall status

| Evidence | Result and limit |
|---|---|
| Domain engines and shared study | Implemented for Warehouse, Cooperative Pong and Cooperative Kitchen. Server-authoritative turns, scores, phases, assignment, recovery and questionnaires are integrated. |
| Explanation permissions | Only active Group A / Task 2 can create or read explanations. Task 2 termination already revokes access, before Next is pressed. Task 1 and Task 3 are explanation-free for both groups. |
| Unified interface | Shared Demo → Task 1 → Task 2 → Task 3 → Questionnaire; English first visit, persistent explicit Chinese selection; discrete select-and-confirm actions and read-only history. |
| Automated full flows | Local store/HTTP fixtures pass. Separately, all six hosted v2 domain/group flows completed against real PostgreSQL and the real provider with synthetic human actions: 1,695 requests including the restart probe, 1,450 compared task moves, 18 completed runs and six questionnaires; no flow, transition or permission assertion failures. |
| Browser evidence | Local Pong B completed with responsive checks. Real production native Chrome completed v1 Pong A including one real question, terminal/multi-tab revocation and questionnaire; v2 Kitchen B completed six demo steps and five Task 1 actions. Warehouse B demo inspection is partial. These do not constitute six full production browser flows. |
| Real semantic provider | v2 original batch: 14 answered, one excessive clarification, zero unavailable; independent review found ten satisfactory, four grounded but incomplete, one unsuccessful follow-up. A separate two-request Pong contextual supplement had correct final answers, with one internal repair; original batch counts remain unchanged. Historical local/v1 attempts and failures are retained. |
| Persistence | Local SQLite and real psycopg/Neon checks passed. A real Render restart also recovered the exact v2 probe at revision 10/turn 3, accepted revision 11/turn 4, and retained all six complete test instances, 18 runs and six questionnaires in the export. |
| Existing data | Original working checkout preserved; existing Kitchen PostgreSQL backup independently verified. The user explicitly confirmed Warehouse has no data needing backup and authorized replacement, resolving the old transient-data gate. Browser-held legacy Pong keys were not cleared. |
| Production | v2 is live and hosted acceptance passed within the scope below. The final pilot-configuration deployment is live; public release and health responses confirm pilot mode, persistent storage, verified deployment and `study_ready=true`. |
| Human effect | No human pilot run or human A/B result. The Task 2 relative-improvement target of 50% is unmeasured. |

## Automated verification

The combined command is:

```sh
python3 -m pytest -q tests/test_study_v3_*.py tests/test_domain_hub_server.py
```

The final integrated v2 run passed **538 tests and 144 subtests in 27.32 seconds**,
including the retained legacy hub test and the QA role/evidence regressions.
This run includes the immutable
release guard, readiness-mode refinement and two regressions that reject new
same-domain or cross-domain instances when an older release is still active.
The latter rejection preserves all existing research records unchanged.
The existing Render entry command was also exercised as a real subprocess and
returned the new three-domain health response without loading the legacy neural
runtime. JavaScript syntax validation passed.

Tests cover actual simultaneous mechanics, scoring boundaries, determinism,
fixed-AI reachability, real demo transitions, no future-schedule leakage,
non-mutating evidence and simulation, exhaustive phase permissions, terminal
revocation, late responses, cross-run/domain ownership, duplicate and competing
actions, idempotent questionnaires, scoped export, and retained release metadata.
The release manifest hashes canonical source, web assets, frozen configurations,
the production entry, dependency file and Python version. Reusing a release ID
with changed source in pilot mode fails rather than rewriting the old receipt.

See `three_domain_shared_validation.md` for the detailed common-service and
browser evidence. A browser check found three visible issues—non-sticky header,
overlapping Pong ball labels and discarded questionnaire drafts. Each was fixed
and rechecked through the browser before the final questionnaire was submitted.

## Feasibility and proxy screening

Each domain has 24 development and 24 separate held-out seeds, with all three
tasks exercised. The human-role planner can choose only human actions; the
teammate always runs the actual unchanged domain policy. A/B never changes
mechanics, AI, scoring, budgets or scenario distributions.

| Domain | Feasibility observation across 48 seeds × 3 tasks |
|---|---|
| Warehouse | All six assigned deliveries completed, zero collisions and shutdowns; maximum completion times 83 / 86 / 89 turns within the common 120-turn budget. |
| Pong | Every cooperative ball caught. Exact offline full-task search checked all 144 instances. Some ordinary opportunities are mutually exclusive, so full scheduled-point score 100 need not be reachable. |
| Kitchen | All scheduled orders completed without burning. All 48 Task 2 scenes demonstrate both pots actively cooking simultaneously and both corresponding foods subsequently served. A separate natural-delay trajectory triggers and completes emergency pot rescue. |

Held-out **Task 2 simulation means**, each based on 24 scenarios:

| Domain | Random human | Simple greedy human | Public-history learner | Rule-informed human planner |
|---|---:|---:|---:|---:|
| Warehouse | 39.58 | 54.17 | 98.61 | 100.00 |
| Pong | 25.69 | 76.53 | 94.58 | 96.25 |
| Kitchen | 0.00 | 47.22 | 75.00 | 100.00 |

These columns are proxy strategies, **not experimental groups A and B**. In
particular, Warehouse and Pong history learners approach their planning
ceilings. This is evidence that some people might learn coordination without
explanations, and a limitation for the desired effect. The results do not
establish a robust 50% advantage. Scores, participants or inclusion criteria
must not be altered after observing human outcomes to manufacture that effect.
The domain reports describe calibration and retained limitations in detail.

## Question truth and interpretation

The frozen QA corpus has 218 cases: Warehouse 80, Pong 64, Kitchen 74. Injected
semantic plans test real state facts and counterfactual settlement, not external
language understanding. Participant prose is composed from verified bilingual
evidence; arbitrary model-generated claims are not displayed. The model selects
facts and hypothetical human actions, with exact binding and validation.

The 26 real-provider attempts returned 20 answers, three initial unnecessary
clarifications and three unavailable results. Answer status was not equated
with correctness: a Kitchen movement/reason mismatch and several historical or
follow-up interpretations were found, repaired and rechecked. Timeouts remain
failed attempts. Independent review added questions that exposed a missing
Warehouse auxiliary-net-score comparison; final answers distinguish that score
from the main 0–100 Task score. See `three_domain_qa_evaluation.md` for exact
coverage, manual findings, successful follow-ups and remaining limits.

These 26 attempts are historical local evidence, not the production batch.
Production v1 subsequently exposed three material Warehouse relevance/actor
failures among 12 answered requests and three timeouts. They are retained in
`analysis/three_domain_build_20260920/production_qa_review.md` rather than
reclassified after the fixes.

The v2 production batch made 15 new authenticated A/Task 2 requests covering
English, Chinese, follow-up, selected history and counterfactuals in each
domain. Fourteen returned answered and one returned an excessive clarification;
none returned unavailable. Independent review rated ten satisfactory, four
factually grounded but incomplete/repetitive, and the Pong same-action follow-up
unsuccessful. All 15 saved state hashes, decisions, evidence catalogs and selected
factual paragraphs matched; ten simulation branches recomputed exactly. No
incorrect game number or human/AI action confusion was observed in that sample.
Correct grounding alone is not a semantic pass.

A separately authorized two-question Pong sequence correctly explained its plan
and then identified the human's assigned contact lane. Its second final answer
followed one recorded actor-evidence validation failure and repair. This
supplement demonstrates one successful explicit contextual follow-up; it does
not repair the original excessive-clarification case or change the original
15-request counts. Detailed judgments are in
`analysis/three_domain_build_20260920/production_qa_review_v2.md`.

The original v2 QA requests were serialized, with a 45-second provider timeout;
v1 allowed overlap and used 25 seconds. The v2 observed provider durations were
1.830–38.007 seconds (median 9.234). Zero timeouts in this batch is not a
concurrent-load result or a causal estimate of the semantic change. Arbitrary
question accuracy and complete coverage of every possible question remain
unestablished. Unavailable responses and repairs stay explicit and audited.

## Persistence and research-data protection

Local restart validation uses two different application processes opening the
same retained SQLite file. Saved task state, identity, release and exports
survive, and the next turn can be committed. This does not imply that a Render
temporary file survives replacement. Production is configured for the existing
PostgreSQL resource, with additive `pl3_*` tables separate from legacy tables,
bounded connection pooling, read-only consistent snapshots, instance-scoped
write locks, precise timestamps and no automatic replay of uncertain commits.

An additional bounded check used the **actual production Python Database
adapter and psycopg against the existing Neon PostgreSQL database**. Because
local direct port 5432 was unreliable, a temporary loopback-only TCP bridge
carried PostgreSQL bytes over Neon's documented, TLS-verified WSS transport.
Fourteen checks passed in 15.58 seconds: additive schema creation, committed
and rolled-back dedicated probe writes, consistent read-only transactions and
rejected writes, the three configured database timeouts, seven transactions
crossing the driver's automatic-prepare threshold, connection reuse,
close/reopen recovery, retained fractional timestamps and absence of float32
columns in the new schema. Only `pl3_*` schema and a dedicated
`pl3_transport_probe` test row were created; no legacy Kitchen tables were
changed. The temporary bridge was closed after testing. The external receipt
is `analysis/three_domain_build_20260920/real_postgres_wss_validation_receipt.json`.
This was the predeployment database-adapter check. Hosted evidence is now
separately available: the v2 service ran the six flows through real PostgreSQL,
then survived an actual Render process restart. The probe's revision 10, turn 3
and exact state hash were recovered; its next action produced revision 11,
turn 4 and was found in the authenticated export. All six completed instances,
18 task runs and six questionnaires were still present. See
`analysis/three_domain_build_20260920/production_restart_probe_v2_receipt.json`.
The restart receipt supplies the hosted durability claim; the earlier local
bridge result is not used as its substitute.

The existing Kitchen database backup contains 13,663 rows across 13 tables,
34,256,578 bytes, with matching counts and SHA-256 verified independently.
Private backup files and credentials are outside the Git repository. The
original checkout's HEAD, branch and six pre-existing tracked modifications
were compared with the saved baseline and remain unchanged.

The old Warehouse server had no full research export and stored transient
records on its instance or in memory. Before replacement, the user explicitly
confirmed that Warehouse had no data needing backup and instructed deployment.
That data-protection question is resolved; it is not an outstanding gate. The
existing Kitchen database backup remains preserved outside the repository.

## Hosted acceptance and remaining scope

The real v2 HTTP acceptance used all six domain/group combinations with separate
synthetic `mode=test` identities. It made 1,695 requests including the restart
probe and compared all 1,450 action transitions with the unchanged canonical
engines. All 18 runs and six questionnaires were saved. A/Task 2 active access,
B denial, Task 1/Task 3 denial, terminal revocation, historical queries, stale
question/ack rejection, language persistence, recovery and authenticated export
checks passed. There were zero transport retries and no recorded flow,
state-comparison or permission assertion failures. These are software and
integration results, not human A/B outcomes or six completed browser studies.
The receipt is
`analysis/three_domain_build_20260920/production_http_acceptance_v2_summary.json`.

Native Chrome on the real website completed v1 Pong A: six played demo steps,
30/41/41 turns, one real English answer, no question turn cost, English/Chinese
persistence, keyboard controls, terminal and second-tab chat removal, Task 3
lockout, questionnaire draft reload and completed-page recovery. Displayed
scores 65/33/50 come from automated actions, mostly waiting, not human subjects.
v2 keeps the same UI and policies; its separate Kitchen B inspection played
all six demos and confirmed five Task 1 actions ending with a prepared tomato.
Warehouse B demo inspection is partial at this report's snapshot. No complete
Kitchen or Warehouse browser questionnaire flow, or production narrow-viewport
check, is claimed. The unchanged UI has the separate local responsive evidence.
See `analysis/three_domain_build_20260920/production_browser_acceptance.json`.

The final public pilot-configuration deployment is **LIVE** on the same v2 source
commit (`dep-dandd3mgekts738jt3lg`). The actual public release response confirms
`mode=pilot`, persistent storage, configured semantic QA,
`deployment_validation_complete=true` and `study_ready=true`; health reports
healthy and ready. Participant enrollment is open. The receipt is
`analysis/three_domain_build_20260920/production_release_v2_pilot.json`.
The additional recovery check after this final configuration deployment also
passed: the probe recovered unchanged at revision 11 / turn 4, then a real wait
committed revision 12 / turn 5 and matched refresh plus authenticated export.
All six complete instances, 18 runs and six questionnaires remained; the
separate Pong supplement also recovered at active Task 2, revision 38 / turn 0
with its two saved answers. This check made no provider requests and created no
synthetic pilot participant. Its separate receipt is
`analysis/three_domain_build_20260920/production_pilot_redeploy_persistence_receipt.json`;
it does not overwrite the first restart receipt.

Human performance remains **unmeasured**. The 50% relative Task 2 improvement is
a pilot target to evaluate using actual participants and the frozen analysis
protocol, not a deployment acceptance result. Deployment instructions, service
identity and final enrollment status belong in `three_domain_deployment.md` and
the final external release receipt.
