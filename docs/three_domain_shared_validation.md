# Shared study service validation

Recorded on 2026-09-20 (Asia/Shanghai) in the isolated `XAI-study-v3` worktree.
Actual hosted release: `policylens-three-domain-20260920.v2`, commit
`68a50da906e30a3ac2f7712858cc2b19d2cd159a`, source SHA-256
`387513ef55e9d2c5ca68c061159b389bc281b5bad88ef4972a399ea960de9858`.
The final combined local run passed **538 tests and 144 subtests in 27.32
seconds**. Hosted HTTP, real restart, semantic review and browser observations
are recorded separately below. The final pilot-configuration deployment is live;
public release and health responses confirm `study_ready=true`, and participant
enrollment is open.

## Historical local test addition and reproducible command

`python3 -m pytest -q tests/test_study_v3_store.py tests/test_study_v3_http.py`

**40 passed in 15.15 seconds** in the earlier run for this test addition.
This historical subset count is not the final integrated v2 suite count. The suite
contains 25 store checks and 15 HTTP checks, including parametrized cases. It
uses temporary SQLite files and dynamically allocated loopback ports and does
not touch the existing website, the original checkout, or research data.

All complete-flow participants have generated test IDs and `mode=test`. One
isolated preview fixture exercises mode filtering and contamination prevention;
attempts to enter pilot mode without readiness are rejected. Human actions come
from each environment's deterministic `human_advisor`; the environment's fixed
AI supplies every teammate action. The explanation implementation is replaced
by a recording or deliberately blocked test double for authorization testing.
These results are **simulated software validation**, not real semantic-model
evaluation, human usability findings, or evidence of a 50% human improvement.

## Coverage

| Area | Observed checks |
| --- | --- |
| Complete study flow | Each of Warehouse, Pong and Kitchen completes Demo → Task 1 → Task 2 → Task 3 → Questionnaire as both A and B, once directly through the store and once entirely through HTTP. Every action and stage change is committed through the production interfaces. |
| Demo and transitions | Skipping unfinished demonstrations or active tasks is rejected. Each task has a visible terminal summary before advancing. Every task ends within its real configured budget. |
| Explanation matrix | A can ask only while Task 2 is active. Both groups are denied in Demo, Task 1, Task 3 and Questionnaire. B is denied in Task 2. Task 2 terminal summaries already revoke access. |
| Historical explanations | During active A/Task 2, the same instance's Task 1 initial and terminal frames can be queried. Historical frames remain read-only. Future frame numbers and another instance's run are rejected. |
| Late answers | A deliberately blocked answer starts during Task 2, while another worker completes the task and optionally enters Task 3. On completion, the answer is recorded as revoked, never acknowledged as displayed, and its response receives 403. |
| Old responses and tabs | A repeated old command returns a fresh current-stage view, with no Task 2 chat resurrected. Two simultaneous action requests from the same revision commit exactly one transition; the other receives a stale-state error. |
| Identity and recovery | Session tokens, participant recovery codes and instance ownership are checked. Knowing an anonymous participant ID cannot resume the participant. A forged token, another participant, cross-domain run, wrong active run, and changed release cannot bypass authorization. |
| Assignment consistency | Matched A/B instances have equal initial states, fixed-AI decisions, transitions and scores under the same human actions. Asking questions does not alter the A instance's state. Assignment stays fixed when recovering or entering a second domain. |
| Enrollment provenance | Consent, the initial language, assignment source and scenario version are saved. Changing the interface language does not rewrite enrollment history. Allocated seeds belong to each domain's frozen held-out collection. An existing test identity cannot be reused in preview mode. |
| Idempotency | Repeated action submission advances once. Changed content under the same command ID is rejected. Repeated questionnaire submission creates one record. A new submission after completion is rejected. |
| State and replay | Every saved transition in the six store workflows is replayed from its recorded previous state, human action and fixed-AI decision and compared with the recorded next state. Question asking, history reads, language changes and timing measurements do not change game state. |
| Public data boundary | Responses and replay frames omit private decisions, policy memory, seeds, audit traces, raw model answers and session credentials. Deliberately injected private answer fields remain available only through the authenticated research export. Questionnaire answer keys are not sent to participants. |
| HTTP boundary | Same-origin routes, asset allowlist, no-store responses, CSP, HttpOnly/SameSite/Secure cookies, cross-origin rejection, malformed requests, unauthenticated reads and researcher export authorization are exercised. Initial HTML declares English and includes English/Chinese controls. |
| Failure honesty | With no explainer, a question is stored and shown as unavailable rather than a successful answer. Configured resources alone are insufficient without a verification flag. With readiness requirements unmet, or after an actual failed question request in the test service, public pilot creation returns `study_not_ready`. |
| Durable restart | A test identity completes its demo and submits an action in a real child application process. That process is terminated. A different child process opens the same SQLite file; the session, current task, state and research export are recovered unchanged, and the next action is accepted. |
| Data integrity | Every full flow has three saved task runs and one questionnaire, continuous frame numbers, bounded public task scores and explicit test mode. Invalid timing/questionnaire operations roll back atomically. JSONL and CSV exports contain equal numbers of records and omit recovery/session secrets. Version/mode filtering retains only the selected instances and their associated records. Restart preserves the original release audit record. |

## Limits and separate release checks

- These local tests alone do not establish persistent storage across a Render
  replacement. The separate real hosted PostgreSQL/restart result below now
  supplies that evidence; it must not be attributed to the SQLite fixture.
- HTTP tests do not exercise browser rendering, keyboard repeat, bfcache,
  browser-side chat epochs, responsive layouts, or actual network retry UX.
  Those require the separate browser validation performed by the integrating
  agent, including the real deployed site.
- Injected answers verify authorization and safe serialization only. The
  separate real-provider production review below supplies observed English,
  Chinese, follow-up, history and counterfactual results, with failures and
  incomplete answers retained. It does not prove arbitrary-question accuracy.
- The end-to-end tests use a cooperation feasibility floor of 50 task-score
  points to detect a broken scenario. That is neither a relative gain nor an A/B
  effectiveness threshold, and must not be reported as the user's 50% target.
- SQLite durability here is a process-restart result on a local retained file.
  PostgreSQL integration and hosted storage durability are separate checks.

## Findings resolved during integration

The permission and flow suite found no remaining failing assertion. During
review, the integrating agent added immutable release metadata, release/mode
export filtering, and explicit consent/initial language/assignment-source
enrollment records. The test suite now covers those changes, plus held-out
scenario allocation and verified readiness. Domain-specific export filtering
is not currently part of the server interface; exports identify every record's
domain through its study instance.

The first HTTP test run over-specified the language button's displayed text as
`English`; the UI uses `EN`. That test was corrected to check the English
language control and Chinese option rather than a particular abbreviation.

## Independent browser check: Cooperative Pong, group B

The additional browser check uses native Chrome through the approved computer
interface, in a new incognito window at
`http://127.0.0.1:8010/pong/?preview=1`. The generated local preview identity is
`ui-pong-b-20260920`, explicitly excluded from human-study analysis. No script
changes application state or invokes APIs from the browser; the demonstration,
task actions, replay and questionnaire use visible controls and keyboard input.

Verified:

- First visit is English; researcher preview group B is visibly identified.
- All six demo steps are played through their public Play/Next controls before
  Task 1 unlocks. No hidden AI explanation appears.
- Task 1, Task 2 and Task 3 finish after 30, 41 and 41 explicit action turns.
  The public summary scores are 35, 50 and 30. The test actor mostly waits; one
  Task 3 move checks keyboard control. These scores are UI-test observations,
  not human performance or an effect-size result.
- B has no question entry in any task. Task 2 explicitly says explanations are
  unavailable, and Task 3 says previous answers are unavailable.
- Switching Task 1 to Chinese at turn 10 preserves the game. Refresh keeps
  both the Chinese choice and turn 10. Returning to English also preserves it.
- Viewing Task 2 turn 21 disables action controls. Returning to the actual turn
  restores turn 22 and the unchanged score of 40.
- Left-arrow selects an action without advancing. Enter advances one turn and
  moves the human from lane 7 to lane 6. Space selects Wait without advancing.
- A 390 × 844 emulated viewport stacks content and provides reachable controls.
  After a reported fix, the masthead/language buttons stay visible while the
  viewport is scrolled to the action panel. After another fix, each ball has a
  separate public information row; full labels no longer overlap on the board.
- The five shared ratings and three comprehension questions are shown to B,
  with no answer key or A-only question-answer ratings. Filled responses and
  feedback survive a language switch and, after the reported fix, a refresh.
- Submitting the questionnaire reaches the Completed page with all three
  scores. Refreshing that page still shows Completed and the same saved scores.

The browser check found that refreshing an unsubmitted questionnaire clears
its draft responses and feedback while correctly retaining the questionnaire
stage and language. The integrating agent repaired this with an
instance-and-release-scoped sessionStorage draft cache. After reloading the
updated client and entering a fresh draft, all five selected ratings, all three
selected comprehension responses and the explicit test feedback survived a
Chinese-to-English switch followed by a refresh. The questionnaire was then
submitted successfully, completing the entire B browser workflow.

Native accessibility clicks inside Chrome's scaled device emulation sometimes
only scroll an off-screen control into view. A screenshot-grounded click on the
visible Confirm button was used to verify a narrow-screen turn, after which the
remaining flow continued at desktop width. Click attempts were not counted as
turns; observed turn numbers and terminal summaries determine the above counts.
This is a local browser check, not a production deployment check.

## Real production v2 HTTP and restart acceptance

At <https://policylens-warehouse-study.onrender.com/>, all six Warehouse/Pong/
Kitchen × A/B workflows completed from demo through questionnaire against the
actual service, PostgreSQL database and semantic provider. Test identities are
explicitly `mode=test`; human choices came from a deterministic human-role
planner, while each AI used the deployed unchanged policy. No game state or
score was written directly by the acceptance client.

The acceptance receipt records **1,695 HTTP requests including the restart
probe**, **1,450 action transitions compared**, **18 completed runs**, and
**six questionnaires**. There were no recorded flow, transition or permission
assertion failures and no transport retries. The task-turn counts were:

| Domain | Group A Task 1 / 2 / 3 | Group B Task 1 / 2 / 3 |
|---|---|---|
| Warehouse | 78 / 73 / 75 | 78 / 73 / 75 |
| Pong | 30 / 41 / 41 | 30 / 41 / 41 |
| Kitchen | 97 / 142 / 148 | 97 / 142 / 148 |

Observed hosted checks include initial English, persisted Chinese/English
selection, cross-identity isolation, no private state in participant responses,
A active Task 2 access, B denial throughout, both groups denied in Task 1 and
Task 3, Task 2 terminal revocation before advancing, rejection of stale Task 2
questions and display acknowledgments, questionnaire completion, refresh
recovery and authenticated mode/release-filtered export. Questions did not
advance the game state.

A real Render restart followed. A dedicated probe recovered unchanged at
revision 10 / turn 3 with its exact saved state hash. Its next action committed
revision 11 / turn 4 and appeared in the admin export. All six completed
instances, 18 task runs and six questionnaires survived. The before/after
receipt is `analysis/three_domain_build_20260920/production_restart_probe_v2_receipt.json`;
the six-flow receipt is
`analysis/three_domain_build_20260920/production_http_acceptance_v2_summary.json`.
These hosted checks are distinct from the earlier local SQLite and temporary
Neon-WSS adapter tests.

The old Warehouse backup gate is resolved: the user explicitly confirmed there
was no Warehouse data needing backup and authorized the replacement. Existing
Kitchen backup and original-checkout preservation remain recorded. No claim is
made that the Kitchen backup contains legacy browser-local Pong records.

## Real production semantic review

The original v2 batch contains 15 new real questions, one English, Chinese,
follow-up, historical and counterfactual request per domain. Fourteen returned
answered; one returned an excessive clarification; none returned unavailable.
Independent review rated **ten satisfactory, four grounded but incomplete or
repetitive, one unsuccessful follow-up**. Status `answered` was not used as the
semantic correctness criterion. All 15 state/decision/catalog checks and ten
counterfactual branch replays matched. No incorrect game number or wrong
human/AI action subject was observed in the reviewed final answers.

Pong's “Why is that better than waiting here?” after advice to wait was
unnecessarily rejected as unsupported. That remains a semantic failure. A
separate authorized two-request Pong supplement successfully answered an AI
plan question followed by “For that plan, which contact lane do I need to cover
when the team ball arrives?” The second final answer followed one recorded
actor-evidence validation failure and repair. These extra requests are not
merged into the original 15 or used to erase its failed case.

The v2 batch serialized questions and used a 45-second provider timeout; the
historical v1 batch could overlap and used 25 seconds. V2 provider durations
were 1.830–38.007 seconds, median 9.234. This is not a concurrency/load test, a
general accuracy estimate or a human result. Detailed independent judgments,
including preserved v1 failures, are in the external
`production_qa_review.md` and `production_qa_review_v2.md` reports under
`analysis/three_domain_build_20260920/`.

## Real production browser observations

The parent used native Chrome through the approved CUA UI on the actual site.
The v1 Pong A preview identity completed all six played demo steps, Task 1/2/3
at 30/41/41 turns, and the questionnaire. Displayed scores 65/33/50 came from
automated test actions, mainly waiting, and are not human study outcomes. One
real English Task 2 answer was independently checked against the visible state:
AI moves left toward its lane 6 commitment, human covers lane 2, five turns
remain. Asking did not advance the turn. Task 2 terminal completion removed chat
before Next, including the second tab; Task 3 had no explanations. English/
Chinese switching and reload, keyboard select/confirm, all 11 questionnaire
selections plus feedback across reload, submission and completed-page recovery
were observed.

The UI and policies are unchanged in v2. A separate v2 Kitchen B preview visit
played six demos, entered Task 1 and confirmed up, up, take tomato, chop, chop,
ending at turn 5 with a prepared tomato and no explanation panel. Shared colors,
progress, language masthead, station legend and controls rendered correctly.
Warehouse B demo inspection is partial at this snapshot. Neither is described
as a full browser questionnaire flow. The production browser run did not test a
narrow viewport; that evidence is the separate local Pong B observation above.
One arrow selection during a pending language refresh was lost; on the settled
page the repeated selection and confirmation advanced exactly one move. No
unconfirmed turn was observed. IAB navigation failed in a subagent, so it is not
counted as browser acceptance evidence.

The browser receipt is
`analysis/three_domain_build_20260920/production_browser_acceptance.json`.
The final pilot-configuration deployment on the same v2 commit is **LIVE**
(`dep-dandd3mgekts738jt3lg`). The actual release reports pilot mode, persistent
storage, configured semantic QA, verified deployment and `study_ready=true`;
health is healthy and ready. Public enrollment is open. See
`analysis/three_domain_build_20260920/production_release_v2_pilot.json`.
The second recovery check after this final configuration deployment also
passed. The probe recovered unchanged at revision 11 / turn 4, then committed
one real wait to revision 12 / turn 5, matching refresh and authenticated
export. Six completed instances, 18 runs and six questionnaires remained. The
separate Pong supplement recovered at active Task 2, revision 38 / turn 0 with
both recorded answers. This check made no provider requests and created no
synthetic pilot registration. Its separate receipt is
`analysis/three_domain_build_20260920/production_pilot_redeploy_persistence_receipt.json`;
the first restart receipt remains intact. All automated/preview records are
excluded from human-study analysis; the requested 50% Task 2 human improvement
remains unmeasured.
