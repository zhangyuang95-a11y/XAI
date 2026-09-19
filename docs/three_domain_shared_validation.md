# Shared study service validation

Recorded on 2026-09-20 (Asia/Shanghai) in the isolated `XAI-study-v3` worktree.

## Result and reproducible command

`python3 -m pytest -q tests/test_study_v3_store.py tests/test_study_v3_http.py`

**40 passed in 15.15 seconds** in the final run for this test addition. The suite
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

- These tests do not establish persistent storage across a Render redeployment.
  The production deployment must use a real persistent disk or database and
  independently demonstrate recovery after a production restart/redeployment.
- HTTP tests do not exercise browser rendering, keyboard repeat, bfcache,
  browser-side chat epochs, responsive layouts, or actual network retry UX.
  Those require the separate browser validation performed by the integrating
  agent, including the real deployed site.
- Injected answers verify authorization and safe serialization only. Real
  English/Chinese questions, follow-up, history, counterfactual truth and hidden
  future boundaries require the semantic QA/domain suites and a real configured
  provider smoke test.
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
