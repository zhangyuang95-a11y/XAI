# Three-domain question answering: evidence, tests and deployment status

Implementation: `study_v3/qa.py`, version `study-evidence-qa.v3.1`. Interface: `Explainer(settings).answer(engine, state, decision, question, language, previous_dialogue, public_history)`. It returns `status`, `answer`, `evidence_ids`, `language` and a researcher-only `audit`. The shared store owns participant authorization and removes audit data from participant responses. This service never issues real gameplay actions.

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

After the production role-binding fixes, the QA suite separately passed **264 tests** in 3.48 seconds. The added regressions cover authoritative human advice, historical AI reasons, wrong-actor evidence, human simulation branches, speaker perspectives, and real HTTP rejection of missing, null or unknown subject/purpose values. This later count is protocol/composition evidence, not additional real-provider questions or a replacement for the earlier 279-test result.

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


## Production acceptance and role correction (2026-09-20)

The first deployed release, `policylens-three-domain-20260920.v1` at commit
`6452ea6`, completed all six real HTTP study flows. Its 15 real production QA
requests yielded 12 answers and three 25-second provider timeouts. Independent
review checked all saved states, decisions and fact catalogs, and recomputed
all three counterfactual branches exactly. It found three substantive Warehouse
semantic failures: missing a concrete human recommendation, comparing the AI's
wait when the follow-up concerned the human, and explaining the human's charging
instead of the AI's selected historical intention. Two Kitchen answers were
factually usable but weak in causal detail or subject clarity. Those failures
remain in their original audit and are not counted as successful answers.

Version `study-evidence-qa.v3.1`, released separately as study v2, adds explicit
semantic subject and purpose fields, validates their actual enumerated values
on provider responses (missing, null and unknown values are rejected), and
binds AI action/reason requests to that selected frame's actual decision. Human
advice includes a concrete legal action from the unchanged human-only advisor;
its state is copied and checked for mutation. Human comparisons use human
simulation branches and cannot borrow AI alternative-action evidence. User
pronouns and evidence pronouns have their distinct perspectives stated in the
semantic instruction. Exactly one model repair is permitted for an evidence ID
or subject mismatch, preserving the original plan in the audit.

The bounded provider timeout is now 45 seconds. A single repair can therefore
bring a request close to 90 seconds; the game does not advance and no database
transaction is held during that wait. This improves tolerance of slow responses
without claiming that availability or arbitrary-question understanding is
perfect. The participant's pending-question lease remains 120 seconds.

A separate local replay of five original problem questions made six actual
provider calls. All five eventually returned reviewed answers; the Pong Chinese
question required one model repair and took 48.75 seconds. The Warehouse advice
now names the human's upward move, the follow-up compares that move with human
waiting, and its historical answer names the AI's actual safe-route wait. The
Kitchen follow-up compares the human alternatives. The original Pong English
follow-up was not included in that six-call local budget. It was subsequently
tested in the production v2 batch below and produced an excessive clarification;
that failed response remains in the record. Equal short-window scores are
reported as equal; position changes do not prove a superior final score.

A native Chrome production Pong A test additionally submitted the real English
question “What will you do next, and why?” at Task 2 turn 0. The rendered answer
correctly identified the AI's left move to lane 6, the human's required lane 2
and the five-turn deadline. The turn stayed at zero. Task 2's terminal action
removed the answer and question controls in both open study tabs, before Next;
Task 3 and the questionnaire retained no explanations. This is browser test
evidence with a preview identity, not a human performance result.

## Final production v2 QA review (2026-09-20)

The deployed release `policylens-three-domain-20260920.v2`, commit `68a50da`, was independently reviewed using **15 new real production requests** and their completed server audits. Its source SHA-256 is `387513ef55e9d2c5ca68c061159b389bc281b5bad88ef4972a399ea960de9858`. These were automated `mode=test` identities and actions on the actual website, not human participants. They are separate from the 26 earlier local attempts, the v1 production batch, and the five-question local repair check.

Delivery status was **14 answered, one clarification, zero unavailable**. Independent semantic review classified the same 15 requests as **10 satisfactory, four factually grounded but incomplete or repetitive, and one excessive clarification**. An `answered` status was not counted automatically as satisfactory.

| Domain | Satisfactory | Grounded but limited | Failed follow-up |
|---|---:|---:|---:|
| Warehouse | 3 | 2 | 0 |
| Cooperative Pong | 4 | 0 | 1 |
| Cooperative Kitchen | 3 | 2 | 0 |
| Total | 10 | 4 | 1 |

The reviewed v2 answers corrected the material v1 actor errors: Warehouse now gives a legal human upward move, compares that human move with human waiting, and explains the selected historical AI safe-route wait rather than the human's charging. All three historical questions address the actual saved AI decision. The review observed no incorrect game number or human/AI action substitution in these 15 responses.

The remaining limitations are concrete:

- **Warehouse Chinese advice:** the upward move is legal and explicit, but its explanation cites correct charger feasibility and the AI's goal rather than directly explaining progress toward the human's delivery.
- **Warehouse and Kitchen “why better” follow-ups:** both correctly simulate human up versus human wait. The positions differ, but both one-turn task-score changes are zero. The answers state the horizon limit and make no false score-advantage claim, yet only partly answer why the action would be better beyond that window.
- **Kitchen Chinese advice:** its facts are correct, but it repeats the upward recommendation and weakly explains that action's role in ingredient preparation.
- **Pong follow-up:** after the answer recommends waiting, “Why is that better than waiting here?” receives a generic unsupported-question clarification. The question is already in context and compares the same human action; the answer should acknowledge that equivalence. This is a semantic failure, not a passed clarification.

Grounding was independently checked against the saved inputs. All 15 input-state hashes, decisions and evidence catalogs matched reconstruction. Every selected factual paragraph matched its authorized bilingual fact. All **10 simulated branches** across five comparison/counterfactual answers recomputed field for field, including final positions, events, score changes, assumptions and the unchanged source-state hash. Some answers add an explicitly labeled comparison beyond the requested branch; these extra branches were also checked. Those checks establish factual grounding and simulation consistency, not full semantic understanding.

English and Chinese followed the requested language, and the reviewed answers contained no neural-network terminology or private policy representation. Minor singular/plural errors remain in English. No long-term score improvement is inferred from equal short-window scores.

Observed answer-service durations ranged from **1.830 to 38.007 seconds**, with a median of **9.234 seconds**. None of these 15 final audits contained a repair attempt. Unlike v1's requests, which could overlap across domains and used a 25-second timeout, the v2 questions were serialized and used 45 seconds. Therefore, zero unavailable results in this batch cannot be attributed solely to the semantic fixes or generalized to concurrent participant use.

The detailed independent record is `analysis/three_domain_build_20260920/production_qa_review_v2.md` in the parent workspace. Its private evidence bundle SHA-256 is `0de3dc618e10aa1e7dd2e29a8854117ae598e42347ed73cfbe502558fefa2635`; the companion numeric review explicitly does not claim semantic correctness. The original v1 failures remain preserved separately.

### Separate Pong contextual follow-up supplement

Two additional authorized production questions were independently reviewed after the original 15. Both final answers were satisfactory; they are reported separately and **do not replace or erase the failed same-action follow-up** above.

1. “What will you do next, and why?” correctly explains an AI left move from lane 7 to catch an ordinary ball at lane 6 in two turns, followed by a return to its team-ball lane 7 before the five-turn arrival. The human is assigned lane 3; the stated ball values are one and three points.
2. “For that plan, which contact lane do I need to cover when the team ball arrives?” directly answers human lane 3 and AI lane 7. Its completed audit confirms that the preceding question was supplied as context. The initial plan failed actor-evidence validation and succeeded after its **single bounded internal repair**; this is not two unconditionally correct initial plans.

Both final answers matched the saved state, recomputed decision, complete catalog and selected text. An additional offline reviewer simulation confirmed that holding human lane 3 for those five turns lets the described AI catch both balls for four raw points. That branch was an offline consistency check, not a production gameplay action or human outcome. The supplement bundle SHA-256 is `2bc135c98d99c31e3eea2376509f3e1c7686756abbd04e4463394c704fbc9846`.

These bounded production samples do not establish arbitrary-question correctness, concurrent service reliability, human comprehension, or the 50% Task 2 human improvement target. Deployment readiness, A/B access controls, browser behavior and restart persistence are reported separately by the deployment and shared validation records.
