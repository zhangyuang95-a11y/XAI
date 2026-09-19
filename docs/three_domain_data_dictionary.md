# Three-domain data dictionary

Schema and serializer snapshot for `policylens-three-domain-20260920.v2`,
2026-09-20. The authoritative definitions are `study_v3/database.py`,
`study_v3/store.py` and each versioned domain engine. Tables use the `pl3_`
prefix to avoid overwriting legacy study tables. Creating them is additive;
it does not migrate or recompute earlier study scores.

## Units, keys and interpretation

One participant can have multiple sequential domain instances but keeps one
experimental group. An instance is a complete domain-specific Demo → Task 1 →
Task 2 → Task 3 → Questionnaire flow. Each scored task is one run. A frame is
one saved decision boundary in that run. A transition is reconstructed from
adjacent frames, not treated as an independent participant.

IDs are strings. Participant IDs are user-provided anonymous research codes,
case-normalized and restricted to 3–64 letters/digits/hyphens/underscores;
the application does not request names, email or phone numbers. Codes and free
text can still be identifying if participants enter identifying content. Other
object IDs are random identifiers or run-local domain object names. A ball,
item or order ID must be interpreted with its `run_id`; it is not globally
unique by itself.

Times stored as double-precision numbers are Unix epoch seconds, including fractions. PostgreSQL uses `DOUBLE PRECISION`, and SQLite uses its equivalent numeric affinity. They can be
rendered in UTC or a documented local timezone. They are not simulated turn
numbers. Task numbers are 1–3. Turn 0 is the initial state, and one confirmed
human action increases the simulated turn by one. JSON text columns preserve
Unicode and use standard JSON types; a SQL NULL is distinct from a JSON null
inside a serialized document. Undefined ratios should be represented as null
with a reason in downstream analysis, never as nonstandard JSON Infinity.

## Identity and enrollment tables

| Table | Key / fields | Meaning and access |
|---|---|---|
| `pl3_participants` | `id` primary key; `group_code`; `recovery_hash`; `created` | Stable anonymous code and A/B assignment. Only a hash of the recovery code is stored; the hash is removed from researcher exports. |
| `pl3_sessions` | `token_hash` primary key; `participant_id`; `created` | Hashed authentication token linked to the participant. Entire table is excluded from exports. Raw tokens are delivered only in the session cookie. |
| `pl3_instances` | `id` primary key; unique `(participant_id, domain, release_id)` | Domain-specific flow; detail below. |
| `pl3_enrollments` | `instance_id` primary key; `consent`; `initial_language`; `assignment_source`; `scenario_version`; `created` | Enrollment metadata frozen when the instance is created. |

Instance fields:

| Field | Type / values | Interpretation |
|---|---|---|
| `participant_id` | Text | Join to the anonymous participant. |
| `domain` | `warehouse`, `pong`, `kitchen` | Engine and interface family. |
| `release_id` | Text | Study release identity. Never pool changed releases silently. |
| `group_code` | `A` or `B` | Assignment snapshot, fixed for the instance. |
| `mode` | `pilot`, `preview`, `test` | Research enrollment versus researcher/browser/automated validation. No formal mode is currently accepted. |
| `stage` | `demo`, `task1`, `task2`, `task3`, `questionnaire`, `completed` | Authoritative flow state. An ended task remains at its task stage until Next. |
| `revision` | Integer ≥ 0 | Optimistic-concurrency version for accepted commands. It is not the game turn; language/timing commands can increase it. |
| `current_run` | Run ID or NULL | Current or most recently completed scored task. NULL before Task 1. |
| `demo_index` | Integer | Number of completed server demo steps, capped by the caption count. Not a quiz score or eye-tracking measurement. |
| `language` | `en`, `zh` | Current interface preference. |
| `scenario_seed` | Integer | Selected fixed seed used by the task-specific generators. Researcher data, absent from participant views. |
| `created`, `completed` | Epoch seconds; completion nullable | Flow creation and final questionnaire-completion timestamps. |

`pl3_enrollments.consent` is 1 after the mandatory agreement checkbox; creation
without true consent is rejected. `initial_language` remains the enrollment
language even when `instances.language` changes. `assignment_source` is
`randomized_balanced`, `existing_participant` or `researcher_override`.
The exact count-balancing and seed allocation algorithm, including its limits,
is in [three_domain_study_design.md](three_domain_study_design.md).
`scenario_version` identifies the domain scene configuration, independently
of the common study release and domain rule version.

## Runs, snapshots and decisions

| Table / field | Meaning |
|---|---|
| `pl3_runs.id` | Run primary key. |
| `instance_id`, `task` | Parent flow and Task 1/2/3; the pair is unique. |
| `seed` | Seed copied from the instance for this task-specific scenario. |
| `state_json` | Latest full internal state, including policy memory and any hidden future schedule. Researcher-only. |
| `status` | `active` or `completed`. A budget-exhausted or failed game is still a completed run, not missing data. |
| `score_json` | Latest engine `score(state)` record. Do not reinterpret the same name across domains without its domain. |
| `started`, `ended` | Server wall-clock boundaries. `ended` is NULL until the terminal transition. |
| `pl3_frames.run_id`, `turn` | Composite primary key identifying an exact saved boundary. |
| `state_json` | Full internal state at that boundary. |
| `public_json` | The engine's participant-safe projection at the same boundary. |
| `decision_json` | Verified AI decision for the next transition; empty object at the terminal boundary. |
| `human_action` | Submitted action departing this boundary; NULL until submitted or at terminal. |
| `created` | Time the frame was inserted. |

For frame *t*, `human_action` and `decision_json.action` generate frame *t + 1*.
The latter frame's public `events` report the result. The decision contains
`action`, `reason_code`, bilingual reason text, `goal`, proposed `memory` and
`alternatives`. Comparisons are tied to the actual pre-action state. Domain
extensions include Pong assignment changes/target lanes and Kitchen emergency
decisions. These are audit records, not public help for B or Tasks 1/3.

Complete states contain domain/rule/scenario version, task, seed, turn, maximum
turns, terminal flag, events and policy memory. Warehouse additionally records
positions, battery, carried order, shutdown state, finite orders and counters.
Pong records current balls, hidden `_schedule`, commitments and catch metrics.
Kitchen records held items, counter contents, pot timers, revealed orders,
hidden `_future_orders`, item IDs, metrics and `termination_reason`.

The controller and evidence use whitelisted current information even when the
environment has a complete schedule for replay. Internal replay access does not
authorize public disclosure. Warehouse and Pong termination can be derived from
the terminal state and budget/final-wave or delivery status where an explicit
reason field is absent. Never invent a cause solely from a low final score.

All engines publish factual event objects with `type`, `en`, `zh`, plus relevant
domain fields. Examples include Warehouse `pickup`, `delivery`, `charge`,
`collision`, `shutdown` and completion; Pong arrivals/catches/misses and new
waves; Kitchen item interactions, handoff conflict, cooking readiness, burn,
serving and order expiry. Exact type/optional fields follow the pinned engine,
not a universal flattened event schema. Object references must resolve within
that run. Count events once at their transition, not again in every replay.

Score changes are reproducible as the difference between consecutive frame
score records. The database does not contain an additional authoritative
transition-score table. Preserve original states and calculate derived columns
in a separate analysis artifact, with the release and analysis code recorded.

## Scores and auxiliary metrics

Every `score_json` contains `task_score`, `raw_score` and `metrics`.

| Domain | `task_score` | `raw_score` | Direct auxiliary metrics |
|---|---|---|---|
| Warehouse | `100 × deliveries / 6` | Net: `100D − 200C − 50S − turns` | Deliveries, total deliveries, collisions, new shutdowns, charge events, elapsed turns. |
| Pong | `100 × catch points / total_possible_points` | Catch points (ordinary 1, cooperative 3) | Catches and totals by ball kind, misses, assignment switches/releases, waiting turns, total opportunity points, success rates, elapsed turns. |
| Kitchen | `100 × completed_orders / total_orders` | Number of completed correct on-time orders | Completed orders, burns, expired orders, waste, blocking waits, human waits, handoff conflicts, parallel-cooking turns, total orders, elapsed turns. |

Warehouse net score can be negative; its primary Task score cannot. A zero
Task score is valid. Pong's denominator includes all prearranged opportunities,
even when exact search establishes an upper bound below 100. A known-rule proxy
score or offline full-schedule upper bound is not a participant outcome.

Pong public score projections omit assignment switches/releases. Kitchen omits
its internal emergency-rescue counter from `score`, while the full internal
state preserves it. The researcher `runs.score_json` and a participant's view
therefore need not expose exactly the same auxiliary fields. These differences
do not change Task score or raw score.

Warehouse waiting counts, charger occupancy and actual path lengths are
**derived**, not currently dedicated score fields. Derive waits from submitted
human/AI actions, occupancy from the declared pre/post-state convention, and
successful path length from changed actor coordinates. An attempted cancelled
move is not traveled distance. Record the derivation and denominator explicitly.

The primary effect analysis aggregates one Task 2 score per participant/domain.
Its relative gain is `(mean_A − mean_B) / mean_B`, with undefined gain at
`mean_B == 0`. Report mean difference, sample sizes and uncertainty alongside
the ratio. Questionnaire items, turns, questions, orders and ball arrivals are
not extra independent people.

## Questions, responses and authorization

`pl3_questions` stores one request and its result:

| Field | Meaning |
|---|---|
| `id` | Question idempotency key, globally unique. |
| `instance_id` | Owning domain study instance. |
| `authorized_run` | The active A/Task-2 run authorizing this question. |
| `target_run`, `target_turn` | Exact selected own Task 1/2 frame being explained. Distinct from authorization. |
| `question` | Original participant text, limited to 2,000 characters. |
| `language` | Requested interface fallback language; result language may follow the question. |
| `result_json` | Response object and private audit. Empty object while pending. |
| `status` | `pending`, `answered`, `clarification`, `unavailable` or `revoked`. |
| `requested`, `finished`, `displayed` | Request creation, completion and browser display-acknowledgement times. Last two are nullable. |

A completed semantic answer can have database `status=revoked` if Task 2 ended
before delivery. Its private `result_json` is retained for audit. Do not count
such a result as an answer shown to the participant. `displayed` is a browser
acknowledgement after renewed authorization, not proof that the participant read
or understood the answer. Task 2 completion hides earlier successfully displayed
answers but preserves their historical display record.

Participant answers are serialized to the allowed fields `status`, `answer`,
`evidence_ids` and `language`; private audit data is removed. The UI renders
plain-language answers with their Task/Turn binding. Reading these answers
requires the same active-A/Task-2 authorization as creating them. No participant
chat export is provided.

The private result's `audit` includes, where reached successfully:

- Question-service `version`, requested/finished timestamps and duration.
- Provider hostname, configured/reported model, provider response ID and raw
  structured plan. There is no unrestricted model-authored final factual prose.
- Selected task/turn and full source-state hash; verified evidence-catalog hash.
- Parsed binding, premise status, clarification choice, factual evidence IDs
  and counterfactual action/horizon intents.
- Simulation inputs and safe trace: requested/executed human actions, saved
  first AI action and subsequent fixed-policy actions, public actors/events,
  score deltas, explicit assumed waits, completed steps, illegal actions,
  public-information-boundary stop flag and source-state hash.
- Final composed answer, unavailable failure code and any permitted semantic
  repair record. API credentials are redacted before audit storage.

Not every field exists on a failed request; absence is not success or a zero
duration. The store supplies up to six recent answered/clarification dialogue
turns and up to 16 public frames ending at the selected frame. Store audit saves chronological `context_question_ids`, `context_sha256`, `authorized_run`, `target_run`, `target_turn`, `authorization_at_request` and `authorization_at_completion`. These researcher-only fields are preserved in export. Do not infer external-model semantic success from an injected fixture
plan, an unavailable answer or a syntactically valid response alone.

A pending request has a 120-second lease. After an interrupted process, a later authorized ask marks expired requests unavailable, preserves their interruption audit, and allows a new question. Retrying the same expired question ID returns an explicit interruption message; it does not execute the provider again.

The forward simulation has a default one-step horizon and maximum 12. It
retains the real first AI decision, fills explicitly disclosed missing human
steps with wait, and stops at the first newly revealed external order/wave.
Its trace does not expose the new schedule contents. It cannot alter the live
run. Audit simulation score deltas are hypothetical and must never be added
to the actual participant score.

## Questionnaire and timing

`pl3_questionnaires` has primary key `instance_id`, `answers_json`,
`comprehension_json` and `submitted` time. `answers_json.ratings` maps IDs to
integer 1–7 values. The common IDs are `predictable`, `understood`, `coordinate`,
`workload`, and `smooth`. A additionally has `relevant`, `clear`, and `helpful`;
these three may use the string `na`. `answers_json.feedback` is optional text,
limited to 4,000 characters. The comprehension array stores item `id`,
zero-based `selected` option and Boolean `correct`. These are study-authored
questions, not a validated scale. Correct answers are not sent to the player.

A submitted questionnaire completes the instance atomically. An idempotent
retry cannot create a second questionnaire. Required unanswered ratings or
comprehension choices are rejected. N/A is not a numeric score and should not
be recoded as the scale minimum. Workload direction differs from the positive
experience ratings; scoring must follow the prespecified analysis.

`pl3_timings` stores `id`, `instance_id`, nullable `task`, `kind`, `seconds`,
and `recorded`. Allowed kinds are `active`, `reading`, `replay`,
`explanation_wait`. Commands can carry buffered active/reading/replay intervals;
the explanation wait is also sent separately. These are client-reported,
bounded intervals, not verified measures of cognitive attention. The browser
excludes hidden-tab and answer-wait periods from the other buckets, and classifies
replay separately; other reading uses focus/open-help heuristics.

Use `runs.ended − runs.started` for total run wall-clock elapsed time, noting
that it includes pauses and absence. Use frame turns for simulated elapsed
time. Use provider/response timestamps for answer service duration, and client
explanation-wait intervals for the reported browser wait. Buffered intervals
can be lost if a tab/process closes before a successful send and can differ
from total wall-clock time. Never fill unobserved attention with invented data
or charge participants game points for explanation latency.

## Idempotency and release metadata

`pl3_commands` is keyed by `(session_hash, command_id)` and records a request
hash, result JSON placeholder and creation time. It is excluded from research
exports. Matching retries return a freshly authorized view; different payloads
under one key are rejected. Commands carry the expected instance revision;
actions additionally carry run ID and turn. This prevents duplicate actions
and stale competing-tab writes without storing a replayable cached chat response.

`pl3_releases` stores `id`, `manifest_json` and `created`. The manifest includes
release ID, commit, source SHA-256, each domain version/path, active-Task-2-only
explanation rule, default language, configured mode, persistence/configuration
and validation status, and the explicit `human_effect_status=not_measured` plus
Task 2 target 0.5. Database addresses, credentials and participant records are
not manifest fields. A persisted release row is the original insert; runtime
readiness can change, so use `/api/release` and deployment verification logs for
current service health, not an old row as proof of availability.

## Researcher export and recommended joins

The authenticated endpoint is `/api/study/admin/export`. It accepts
`release_id`, `mode`, and `format` query parameters. Default JSONL returns one
line per record as `{"table": "runs", "record": {...}}`. CSV uses two columns,
`table` and `record_json`, with standard CSV escaping. JSON substructures remain
encoded in their database text fields; CSV is not an automatically flattened
analysis table. Export does not delete or alter records.

Exported tables are participants, enrollments, instances, runs, frames,
questions, questionnaires, timings and releases. Release/mode filtering first
selects instances, then their dependent records and related participants/
releases. Session/command tables and participant recovery hashes are excluded.
Admin authorization remains outside the URL; do not paste a credential into a
query string, shared report or participant page.

Build one analysis row per participant × domain × release by joining instances
to enrollment metadata, its three task runs, questionnaire and aggregated
question/timing records. Include incomplete runs and their status rather than
silently treating every stored latest score as a final score. Select `pilot`
explicitly and exclude preview/test records. Store effect estimates in a
separate analysis result, never overwrite the observed run scores.

This dictionary describes storage capability, not proof of durable deployment.
An actual PostgreSQL instance or verified persistent SQLite mount, recoverable
backup and process/redeployment recovery test are required to establish
persistence. See the separate deployment and validation reports for current
results and outstanding resource requirements.
