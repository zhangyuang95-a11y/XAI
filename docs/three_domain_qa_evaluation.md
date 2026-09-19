# Three-domain question answering: evidence, tests and deployment status

Implementation: `study_v3/qa.py`, version `study-evidence-qa.v3.0`. Interface: `Explainer(settings).answer(engine, state, decision, question, language, previous_dialogue, public_history)`. It returns `status`, `answer`, `evidence_ids`, `language` and a researcher-only `audit`. The shared store owns participant authorization and removes audit data from participant responses. This service never issues real gameplay actions.

## Semantic service and truthful composition

The configured OpenAI-compatible `/chat/completions` endpoint receives the question, recent dialogue, the selected Task/Turn, public observation, legal human actions and a catalog of verified bilingual evidence. It receives neither participant/session identity nor the hidden schedule, seed, private policy memory, API key in its prompt, or future public frames. Server-side HTTP authorization carries a configured provider key; the browser never receives it.

The model returns a strictly validated JSON plan: language, Task/Turn binding, supported/corrected/unclear premise, clarification code, and one or more factual or counterfactual intents. It may select existing fact IDs or identify hypothetical human actions. Arbitrary model answer prose, nonexistent evidence IDs, unsupported fields, invalid actions or a horizon outside 1–12 produce an unavailable result. User input and prior dialogue are framed as data rather than system instructions.

Factual answers consist of selected verified facts, with a Task/Turn label. The model does not invent distances, scores, intentions, schedules or causal assertions in participant-facing text. Bilingual engine facts provide the final language; a Chinese question can receive Chinese despite an English interface. A wrong premise is corrected using selected recorded evidence. Ambiguous objects and unsupported questions produce specific clarification prompts. Explicit references to a different recorded Task/Turn require selecting that frame; public historical observations can use already recorded historical fact IDs.

This restriction reduces the risk of invented facts but does not guarantee semantic relevance or completeness: a language model can still select an irrelevant true fact or misunderstand a reference. Those outcomes require a real-provider evaluation and independent review. No claim of arbitrary-question 100% correctness is made.

## Counterfactual execution

Each simulation deep-copies the selected action-before state. Its first AI action is the saved decision from that state. Later AI actions come from the unchanged domain controller on the simulated branch. Only the human's requested actions vary. Asking, planning and simulation do not alter the real score, time, random schedule, memory or task. Returned actor information is projected through the domain's public-state whitelist, including for the simulation's final position.

The default window is one step and the maximum is 12. If a requested horizon exceeds the supplied actions, missing actions are explicitly described as assumed waits. Illegal actions are reported and not executed. Distinct alternatives run as independent branches. Answers report real settled events, final positions, raw-score changes and the task-score change within the simulated window, not a model's guessed outcome. Warehouse explicitly labels its auxiliary net score separately from Task score, while Pong labels catch points and Kitchen labels completed orders. A window that ends before task completion explicitly does not establish the eventual result.

At a newly disclosed wave or order boundary, simulation stops after the known situation's legal step settles. New-wave and new-order events and contents are filtered. The previously visible ball's catch or a currently known order's completion remains reportable. No subsequent decision can use the new information in that prediction. Full schedule contents and simulator snapshots are not sent to the model. Researcher audit retains only the safe trace for the reported branch, including actions, real events, scores, assumptions, stop condition and source-state hash.

## Deterministic tests and case files

`tests/test_study_v3_qa.py` tests structural semantic-plan handling, verified composition, context transport, language switching, ambiguous references, wrong premises, multiple intents, actual different counterfactual outcomes, immutable state, retained first AI decision, later policy recomputation, explicit waiting assumptions, illegal actions, bounded horizons, hidden-future payload equivalence, Pong wave boundaries, Kitchen order boundaries, terminal-history frames, provider failure and real local HTTP protocol handling. The local HTTP fixture verifies endpoint, authorization, JSON request and secret redaction; it is a protocol test, not a real language-service evaluation.

Question fixtures are loaded from `configs/study_v3_qa_cases.json` and its explicitly listed domain case files. Each case contains a specific recorded state, English or Chinese question, expected evidence IDs, a structural expected plan and independent claims about positions, decisions, factual content or actual simulated score changes. Case plans are injected to verify evidence/simulation handling. Passing those tests is not reported as a language model understanding the questions.

The fixture manifest includes **218 cases: 64 Pong, 80 Warehouse and 74 Kitchen**, each domain with English and Chinese questions. Pong cases span ten different physical situations, safe and costly detours, coincident arrivals, impossible cooperative coverage, necessary side changes, movement bounds, multi-step assumptions, ambiguity and false premises. Warehouse cases cover real waiting, clearing a route, crossing, charging, pickup, delivery, conflict, inability to move, follow-up, history and counterfactual situations. Kitchen uses real fixed-AI trajectories for cooking, deadlines, handoff conflicts, emergency hand clearing, storage, legal options, ambiguity, follow-up and counterfactual questions. Its fixture generator is maintained by the Kitchen implementation so changed mechanics cannot silently leave stale expectations.

`python3 -m pytest tests/test_study_v3_pong.py tests/test_study_v3_qa.py -q` passed **279 tests**: 28 Pong engine tests and 251 question/evidence/protocol tests, including all 218 domain fixtures. A separate scan of bilingual participant-facing evidence generated for every fixture found none of the checked internal terms (`NN`, `logit`, `embedding`, `checkpoint`, `reward shaping`, private memory/schedule names, or the Chinese neural-network term). This scan supplements the semantic/projection tests; it is not proof that arbitrary untested text is flawless.

## Independent review

The Warehouse implementation agent independently read the shared question service and proposed eight new questions about net-score alternatives, minimum battery, charge-before-move timing, old/current reasons, corrected follow-up objects, hypothetical-only commands, shared collision counting and changed waiting behavior. Review identified a real completeness issue: simulation answered Task score changes without displaying auxiliary net-score changes. Waiting and colliding could therefore both appear as Task-score change 0 even when net scores differed. The answer renderer now explicitly reports raw/net-score changes, with regression coverage. The reviewer also recommended public actor projection for final simulation results; this was implemented and tested with a deliberately private actor field.

At the time of the independent code review, those questions had not been run against a configured provider. Subsequent real-provider checks correctly answered the net-score alternative question and the 1% battery question, as recorded below. The other independent questions remain additional acceptance material rather than being silently counted as model passes.

## Real provider evaluation on 2026-09-20

The first configuration check had no provider configured: all 15 explainer-entry smoke attempts returned `unavailable/not_configured`, without issuing a model request. The existing Kitchen service's authorized provider configuration was subsequently obtained through its normal, already logged-in Render dashboard. Credentials were read privately and mapped into `Settings` only in memory; they were not printed, placed in this repository, embedded in browser assets or copied into this report.

The real provider was the existing OpenAI-compatible service using **deepseek-v4-flash**. Requests were sequential. Local outbound access required the existing HTTPS proxy; production transport must use its own valid network route. The first diagnostic batch used a 35-second timeout, while the final follow-ups used the service's normal 25-second timeout.

A total of **26 real-provider explainer attempts** were made: the first 15 covered English, Chinese, follow-up, history and counterfactual questions in all three domains; eight follow-ups checked fixes and an independently proposed net-score comparison; three final checks validated the corrected Kitchen reason and two Warehouse edge cases. Across the 26 attempts, 20 returned an answer, three initially requested unnecessary historical-frame clarification, and three returned unavailable (one invalid evidence selection and two transport/timeouts). An `answered` status alone was not counted as a correctness pass: the initial Kitchen movement/waiting contradiction and misinterpreted follow-up were identified during manual review and corrected.

The confirmed issues and resolutions were:

- **Historical public facts:** the provider correctly selected an earlier position fact, but the server rejected its earlier Task/Turn binding. Historical observations can now bind that exact recorded public fact. Historical reasons and counterfactuals still require the full selected pre-action state. Fresh real requests then answered earlier positions correctly in all three domains.
- **Kitchen action/reason mismatch:** a true `move left` action was accompanied by a domain fact saying the AI was already waiting. The domain owner changed the reason to distinguish moving to the handoff counter from waiting there. A fresh English request returned matching movement and reason, and Chinese returned the same meaning.
- **Current help and follow-ups:** the prompt now prioritizes applicable human options and current coordination conditions over repeating the full rulebook. After a question about how the human can help, “What about waiting instead?” is interpreted as a human counterfactual unless the user explicitly refers to changing the AI. Fresh Warehouse and Kitchen requests produced the correct simulated human alternatives.
- **Unknown evidence IDs:** such plans are never converted into invented answers or heuristically matched facts. The service permits exactly one model-side repair that must select existing IDs, preserving the original invalid plan in researcher audit. Continued failure returns unavailable. The repair path was verified by protocol tests; subsequent real questions did not need that repair.
- **Transport timeout:** unavailable is clearly displayed and audited, with no keyword answer substituted. The independently proposed Warehouse comparison initially timed out, then succeeded on one explicit, bounded follow-up using the real collision fixture. The initial timeout remains a failed attempt in these counts.
- **Recorded versus simulated values:** combined answers now put source-frame facts first under “In the selected recorded state”, followed by simulated branches. This prevents a real current battery of 40% from being confused with the simulated later battery of 60%.

Representative verified real outcomes:

| Domain / question | Actual answer content checked against the engine |
|---|---|
| Pong, next action and reason | AI moves left toward lane 7; human must cover lane 3; the team ball arrives in three turns. |
| Pong, Chinese coordination | Same lane assignment and countdown in Chinese despite an English interface. |
| Pong, “If I move right now…” | Both paddles cover separate contacts on arrival; the actual simulation earns 3 catch points. |
| All three, “Where was I at turn 0?” | Correct earlier position is returned with the earlier Task/Turn label, without pretending to know a past internal reason. |
| Warehouse, two waits while charging | Battery changes 40% → 50% → 60%; auxiliary net score changes by −2 and Task score remains unchanged. |
| Warehouse, wait versus enter occupied cell | Wait changes net score by −1; moving right causes one cancelled collision and changes net score by −201. Both Task-score changes are zero. |
| Warehouse, “I have 1%…can I move once?” | The answer states that movement requires at least 3%; the actual available action is wait. |
| Kitchen, next action and reason | “My next action is move left” followed by “I am moving next to the handoff counter to wait for prepared food.” |
| Kitchen, Chinese help and follow-up | Gives a currently available upward move; subsequent waiting/upward alternatives are separately simulated with correct final positions and zero orders completed in that one-turn window. |

After the fixes, the three domains each have real successful English, Chinese, follow-up, history and counterfactual evidence across these runs. The final 11 follow-up attempts included ten manually checked answers and one transport timeout; the timed-out comparison then succeeded on its sole explicit follow-up. This is targeted smoke evidence, not a claim that all 218 case questions have been understood correctly by the provider or that arbitrary questions are infallible. No participant data or human A/B performance results were generated by these tests.

## Reproducible real-service checks

The smoke entry point issues real requests when runtime provider settings are configured:

```sh
python3 -m study_v3.qa --smoke --domain warehouse
python3 -m study_v3.qa --smoke --domain pong
python3 -m study_v3.qa --smoke --domain kitchen
```

The bounded real-provider case evaluator checks actual output status, language, selected evidence coverage and numeric simulation expectations instead of treating any string as success:

```sh
python3 -m study_v3.qa --evaluate --domain pong --limit 10
python3 -m study_v3.qa --evaluate --domain warehouse --limit 10
python3 -m study_v3.qa --evaluate --domain kitchen --limit 10
```

The evaluator samples fact, counterfactual and clarification categories in round-robin order, processes requests sequentially and returns a failing exit code for unavailable or mismatched cases. Its reports still mark human review as necessary: exact evidence selection cannot fully measure clarity, relevance or appropriate ambiguity handling. These commands load actual runtime settings and never substitute an injected plan. They must not be called in a large uncontrolled retry loop.

Browser/API acceptance through the shared study remains a separate integration gate. Initial and delivery-time authorization, Task-2 termination, multi-tab revocation, removal of old answers, and researcher-only audit export are enforced and tested by shared study integration. Read-only review found two issues in shared storage that the task owner fixed: task-run summaries now use the domain public score projection, and revoked completed answers commit their audit before the request raises its authorization error. Semantic service readiness must reflect its real configuration and deployment validation; transient errors remain visible and auditable.
