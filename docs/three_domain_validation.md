# Three-domain validation record

Recorded on 2026-09-20 in the isolated `XAI-study-v3` worktree. The candidate
release is `policylens-three-domain-20260920.v1`. This is an implementation and
local validation report, not a production acceptance certificate.

## Overall status

| Evidence | Result and limit |
|---|---|
| Domain engines and shared study | Implemented for Warehouse, Cooperative Pong and Cooperative Kitchen. Server-authoritative turns, scores, phases, assignment, recovery and questionnaires are integrated. |
| Explanation permissions | Only active Group A / Task 2 can create or read explanations. Task 2 termination already revokes access, before Next is pressed. Task 1 and Task 3 are explanation-free for both groups. |
| Unified interface | Shared Demo → Task 1 → Task 2 → Task 3 → Questionnaire; English first visit, persistent explicit Chinese selection; discrete select-and-confirm actions and read-only history. |
| Automated full flows | All six domain/group combinations complete through both the store and HTTP, with saved actions, three scores and questionnaire. Human actions and QA in these flows are test doubles/proxies, not participants or real-provider QA. |
| Local browser flow | Native Chrome completed the full Pong B workflow and checked English/Chinese, recovery, keyboard, replay, narrow layout and questionnaire submission. This does not establish six completed production browser flows. |
| Real semantic provider | 26 actual local explainer attempts, with failures and identified defects retained and described in the QA report. Successful targeted bilingual, follow-up, history and counterfactual checks exist for every domain after corrections. |
| Persistence | Actual local SQLite application-process restart passed. Actual psycopg/PostgreSQL transactions and close/reopen recovery passed through a temporary Neon WSS transport. Production Render direct connectivity and redeployment recovery remain separate acceptance checks. |
| Existing data | Original working checkout preserved. Existing Kitchen PostgreSQL backup completed and independently checked. Old Warehouse transient records and browser-held Pong data are not included in that backup. |
| Production | Candidate not deployed. Existing service replacement is gated on preserving old Warehouse records or confirming that they are disposable test data. |
| Human effect | No human pilot run or human A/B result. The Task 2 relative-improvement target of 50% is unmeasured. |

## Automated verification

The combined command is:

```sh
python3 -m pytest -q tests/test_study_v3_*.py tests/test_domain_hub_server.py
```

The final integrated run passed **525 tests and 144 subtests in 24.59 seconds**,
including the one retained legacy hub test. This run includes the immutable
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

This system does not guarantee understanding every possible question. A true
fact can still be selected irrelevantly; ambiguous questions may require
clarification. Unavailable responses are explicit and audited. The real
provider must still be exercised through the deployed authenticated Task 2 API,
including English, Chinese, follow-up, history and counterfactual questions.

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
This is a real database-adapter result, not evidence that Render direct TCP
or hosted restart acceptance has passed.

The existing Kitchen database backup contains 13,663 rows across 13 tables,
34,256,578 bytes, with matching counts and SHA-256 verified independently.
Private backup files and credentials are outside the Git repository. The
original checkout's HEAD, branch and six pre-existing tracked modifications
were compared with the saved baseline and remain unchanged.

The old Warehouse server exposes no full research export, stores completed
records on its temporary instance, and retains active sessions in memory.
The current Free Render plan exposes no Shell/SSH. Replacing that instance
without a known backup could destroy records. The task's explicit data
preservation requirement therefore prevents an unqualified deployment claim.

## Outstanding acceptance gates

1. Resolve the old transient-record backup/disposability question without
   silently deleting possible research data.
2. Actually deploy the candidate SHA to the existing Render service and verify
   the public release hash and all three domain routes.
3. Complete six production A/B test flows, real-provider integration, data
   export and PostgreSQL recovery after a real restart/redeployment.
4. Enable pilot enrollment only after those checks; report human improvement
   later from actual participant data using the frozen analysis protocol.

Deployment instructions, recovery steps and exact service identity are in
`three_domain_deployment.md`. Pending gates must remain visible in any status
report; this document is not evidence that the website has already changed.
