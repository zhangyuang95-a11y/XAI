# Three-domain study design

Protocol implementation snapshot: `policylens-three-domain-20260920.v1`,
2026-09-20. The original execution brief is
`analysis/THREE_DOMAIN_HUMAN_STUDY_CODEX_BRIEF_20260919.md` in the parent research
workspace. This document records what the shared study and domain engines
implement; it is not a statistical preregistration or a deployment certificate.

The study examines whether access to on-demand, state-grounded explanations
helps a person coordinate with a fixed AI teammate. Its three custom tasks are
Warehouse, Cooperative Pong and an Overcooked-inspired Cooperative Kitchen.
The language model interprets questions; fixed domain controllers choose game
actions. The human-study Task 2 target is a relative mean-score improvement of
at least 50% in each domain. **No human effect has yet been established by this
implementation.** Simulated partner scores, model answer tests and deployment
status are separate kinds of evidence.

## Participant flow and the treatment

Each domain uses one instance of this flow:

`Welcome / Consent → Demo → Task 1 → Task 2 → Task 3 → Questionnaire → Completed`

There is one scored run per Task. Multiple orders and ball waves occur inside
that run; they are not additional independent participant trials. Every task
ends on a score summary, and the participant explicitly continues. The separate
fixed demo provides six mechanics captions and replayable actual transitions.
It does not contribute to any Task score. The UI requires demonstration progress
before opening Task 1. Public mechanics and demonstrations are identical across
groups and do not explain the AI's hidden priorities.

| Stage | Group A | Group B | Role in the study |
|---|---|---|---|
| Demo | Public mechanics, no strategy explanations | Same | Learn the controls |
| Task 1 | No questions or strategy explanations | Same | Baseline |
| Active Task 2 | Participant-initiated free questions and access to answers from this Task 2 | No explanations | Primary intervention |
| Task 2 terminal summary | Explanation access immediately revoked | No explanations | Saved score and transition |
| Task 3 | No questions and no access to previous explanations | Same | Transfer without assistance |
| Questionnaire / Completed | No explanation assistance | Same | Ratings, understanding and saved completion |

The exact server rule is `group A AND stage task2 AND current task_run active`.
Authorization also requires the authenticated participant to own this study
instance, domain and release. Creating a question requires the current Task 2
run ID. Its target can be a stored frame from this instance's Task 1 or Task 2.
Neither an unplayed future frame nor another person's/domain's instance is a
valid target.

The store rechecks authorization after answer generation and before recording
display acknowledgement. Finishing the final Task 2 action revokes pending
answers immediately, before the Next button is pressed. Authorized views strip
researcher audit fields; unauthorized views contain no previous answers. An
idempotent command retry returns the current authorized view, never a stale
cached view containing old chat. Participant historical replay exposes only
public snapshots. Researcher export uses a separate bearer authorization.

The browser clears chat on completion, cross-tab notification, page restoration,
focus verification and network loss, then checks the server before showing
answers. These controls address accidental stale display and direct API access;
they do not prevent a person from remembering or independently recording an
answer. Retained knowledge is the intended basis of Task 3 transfer.

A Task 2 question is not an AI command. Asking “go left” can explain or compare
an action, but cannot change the actual controller. Group A receives no
automatic intention arrows, strategy bubbles or unsolicited step-by-step advice.
Generic question examples contain no answers. B receives neutral public-control
and replay instructions in the same interface area.

## Assignment, instances and scenarios

Assignment is between participants. A server-generated group is stored on the
anonymous participant, persists through refresh, recovery, language changes and
later domains, and is copied onto each study instance. A returning participant
must present the existing session cookie or recovery code. A reused anonymous
ID cannot cross between `pilot`, `preview` and `test` modes. An instance is
unique for participant × domain × release; a completed instance cannot be
restarted as a new scored run within that release.

For a new participant, the current implementation counts A/B instances within
the selected domain, mode and release. It assigns the group with fewer entries,
using a secure random tie-break when counts match. A researcher may override
the group only for authenticated preview or test identities. Returning
participants keep their original group even when this affects the count in a
later domain. The recorded assignment-source label is `randomized_balanced`,
`existing_participant` or `researcher_override`. The first label describes this
count-balancing algorithm; it does not assert a separately preregistered
stratification or exact matched-pair randomization.

Each domain draws from its configured 24 reserved seeds. For the assigned
group, the server first prefers the seed with the fewest instances in that same
domain/mode/release/group. Among ties it prefers a seed with more allocations
in the opposite group, then the earlier seed in the frozen list. This spreads
scenarios within a group and attempts to fill cross-group counterparts. It does
not guarantee exact one-to-one pairs under incomplete enrollment or repeated
multi-domain participants. Selection counts instances already created, not only
completed ones. The same instance seed is used in its three task-specific
scenario generators.

| Domain | Development seeds | Reserved / enrollment seeds | Frozen configuration |
|---|---|---|---|
| Warehouse | 100–123 | 1000–1023 | `configs/study_v3_warehouse.json` |
| Pong | 730100–730123 | 731100–731123 | `configs/study_v3_pong.json` |
| Kitchen | 1000–1023 | 2000–2023 | `configs/study_v3_kitchen.json` |

The reserved scenes have been exercised in technical feasibility tests. They are
not unseen human data and must not be described as untouched after those tests.
Kitchen's JSON includes explicit scenario contents for its 48 seeds × 3 tasks;
Warehouse and Pong freeze deterministic generators, parameters and seed lists.
The saved initial state is the replay authority for an enrolled run.

The homepage offers three domain choices, and `/warehouse/`, `/pong/` and
`/kitchen/` support single-domain recruitment. The implementation does not force
one person to play all three domains and prevents concurrent unfinished domains
for the same participant. It does **not** implement a balanced domain-order
scheduler. A protocol that enrolls the same person in multiple domains must
assign and record order externally before recruitment, preserve the fixed
group, and account for correlated observations in analysis.

## Public tasks and scores

All gameplay is step by step. The player selects one legal action, then confirms
one simultaneous human/AI transition. Arrow keys select movement, Space selects
wait and Enter confirms; repeated-key events do not advance extra turns.
Kitchen interactions have explicit, context-legal labeled buttons. History is
read-only and disables live actions until Return to current turn is selected.

| Domain | Task 1 | Task 2 | Task 3 | Primary Task score |
|---|---|---|---|---|
| Warehouse | Six deliveries, 120 turns; includes same-room work | Six deliveries, 120 turns; repeated crossing and charging | Six deliveries, 120 turns; changed endpoints/starts | `100 × delivered / 6` |
| Pong | Six waves, 30 turns | Eight waves, 41 turns | Eight waves, 41 turns; changed contacts/starts | `100 × caught points / all scheduled points` |
| Kitchen | Four orders, 140 turns | Six orders, 160 turns | Six orders, 160 turns; changed recipes/release times/starts | `100 × correct on-time orders / assigned orders` |

Warehouse successful moves cost 3% battery, and waiting on the shared charger
adds up to 10%. Collision cancellation counts one event. Its auxiliary net score
is `100 × deliveries − 200 × collisions − 50 × new shutdowns − turns`.
Pong ordinary balls earn 1 point and cooperative balls require distinct partners
covering both contacts for 3 points; misses earn zero. Kitchen uses two chopping
turns, 12/14 cooking turns for tomato/onion, one-item hands/counters and six full
ready turns before burning. Simultaneous handoff interactions both fail. A
correct plated dish served on its exact deadline remains valid.

Scoring, starting observations, rules, scene distribution and controllers are
shared between A/B. No engine receives a group argument. Normal failure and
zero scores remain in the data. Questions, reading, translating and replay
consume no simulated turns, battery, cooking time or score. There is no group
multiplier or post-hoc adjustment designed to create a score ratio. In Pong,
all scheduled opportunities define the denominator even when not jointly
attainable; the independently computed reachable upper bound is auxiliary.

The actual decision priorities and timing boundaries are documented in
[three_domain_controller_contracts.md](three_domain_controller_contracts.md).
The original continuous Pong and legacy Warehouse files are separate from the
new version; the conservative Warehouse is a new implementation inspired by a
historical candidate, not a verified exact rollback to the remembered live build.

## Language, questions and questionnaire

A first visit defaults to English, independently of browser language. The
English / 中文 controls remain in the upper-right shared header. Explicit
language preferences persist through refresh and domain changes; the server
records initial and current language separately. Switching language cannot
change group, task, action, scenario or score. The answer service selects the
question's language when identifiable, otherwise the interface language.

The explanation pipeline is semantic binding → deterministic evidence or
bounded simulation → verified bilingual answer. Evidence names and private
decision records are researcher audit material rather than player-facing
explanatory jargon. The model may misunderstand a question or choose an
irrelevant true fact; test coverage does not guarantee arbitrary-question
correctness. A provider failure is recorded as unavailable and displayed
honestly. It is not replaced by a keyword system claimed to be unrestricted
question answering. See [three_domain_qa_evaluation.md](three_domain_qa_evaluation.md).

Both groups receive five 1–7 agreement items after Task 3: predictability,
understanding waits/plan changes, ability to coordinate, mental effort and
smoothness of teamwork. These are **study-authored items, not a validated
psychometric scale**. Mental effort has the opposite desirability direction
from the positive-experience items; do not average all items as if they had
the same direction without a prespecified scoring plan.

A additionally rates Task 2 answer relevance, clarity and help choosing the
next action, with Not used / N/A available. Both groups can leave optional
open feedback and answer three domain-specific new-situation understanding
questions. The server grades fixed choices against the rule-based answer keys,
does not send keys to the browser, and supplies no explanatory feedback during
the questionnaire. A missing rating is not automatically a zero. Answer text
is stored as participant data, not executed as instructions.

## Measurement and pilot analysis

The participant is the statistical unit. The primary comparison is each
domain's **Task 2** average Task score:

`relative_gain = (mean_A − mean_B) / mean_B`

The pilot target is `relative_gain ≥ 0.50`. Also report the absolute mean
difference, group sample sizes, score distributions and uncertainty intervals.
If B's mean is zero, the ratio is undefined; report the absolute difference
without substituting infinity or changing the denominator. A high synthetic
proxy ratio cannot satisfy the human target.

Before collecting an effect-evaluating pilot, specify recruitment eligibility,
the participant sample size/precision rationale, randomization and scenario
allocation, exclusions for verified technical failures, treatment of incomplete
sessions, primary comparison and interval method, baseline adjustment and the
three-domain multiplicity plan. Keep valid failures and participants who choose
not to ask in the assigned-group comparison. Question use can be analyzed
descriptively, but selecting only frequent askers is not the primary randomized
effect. Avoid defining exclusions after inspecting A/B scores.

Report both unadjusted Task 2 scores and a prespecified analysis using Task 1
as a baseline covariate. Different task difficulty makes a raw Task2−Task1
change an insufficient standalone treatment-effect estimate. Task 3 and the
understanding questions are separate transfer outcomes; neither is required to
reach the Task 2 50% target. Report Task 1 imbalances honestly. A claim covering
all three domains requires all three domain results with its specified multiple
comparison treatment, not only a favorable pooled score.

Wall-clock run duration comes from server start/end timestamps. Simulated turn
count comes from the engine. Client-reported active/reading/replay and question
waiting intervals supplement these, with their limitations described in the
[data dictionary](three_domain_data_dictionary.md). They do not alter primary
scores and are not interchangeable with measured attention.

Development uses human-role random, greedy, public-history and
coordination-informed proxies while keeping the AI fixed. Their feasibility
and ceiling findings are documented separately in the domain/validation
reports. In particular, strong public-history learners can approach the
informed proxy, so a large human explanation benefit is not assured. No human
participants are created by those simulations.

## Version, data and readiness boundaries

The database uses additive `pl3_` tables and keeps participant, instance, run,
frame, question, questionnaire and release records separate. Local storage is
only a UI preference convenience. Durable storage requires an actual persistent
database or a verified persistent SQLite mount, not merely a filename outside
`/tmp`. Backups and restart/redeployment recovery require independent tests.
The researcher export is version/mode filterable and excludes session tokens
and recovery hashes. It still includes free-form research text and private
decision evidence, so it is not a public participant report.

Implemented enrollment modes are `pilot`, `preview` and `test`; there is no
`formal` enrollment mode in this release. Preview/test require researcher
authorization and must be excluded from human analyses. Pilot entry is blocked
unless persistent storage, model configuration, an admin credential, explicit
deployment verification and semantic health checks satisfy the runtime gate.
`/health` distinguishes service liveness from `study_ready`; `/api/release`
exposes the nonsecret release identity and configuration status. A ready flag
does not itself establish a human effect or full semantic correctness.

Before a subsequent formal study, freeze a separate release and analysis plan
after the pilot, retain all pilot data under its original release, and implement
an explicit formal enrollment distinction. Do not change scenarios, controller
rules, score formulas or exclusions mid-cohort to improve the observed ratio.
Any new release must preserve old records and either continue their exact old
rules or explicitly prevent incompatible resumption. The present store refuses
to resume a different release as if it were current.

The existing production destination is
[policylens-warehouse-study.onrender.com](https://policylens-warehouse-study.onrender.com/).
Actual deployed commit, live acceptance, persistence verification and any
remaining provider/resource limitations belong in `three_domain_deployment.md`
and `three_domain_validation.md`. This protocol document makes no substitute
claim that those checks have passed.
