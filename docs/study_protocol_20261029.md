# Three-domain human–AI study protocol v1

## Material Passport

- Protocol: `study-protocol-20261029.v1`
- Consent text identity: `study-consent-20261029.v1`
- Origin: approved study implementation plan, 2026-10-29
- Verification status: software declaration validation only; human effectiveness unmeasured
- Scope: local immutable design registration, not an external preregistration or institutional approval

## Requirement and estimand map

The mentor's requirements are AI–AI self-play followed by a frozen teammate and
human role replacement, three cooperative RL domains, questions about the
participant's actual recent human–AI interaction, explanation-plus-replay versus
neutral replay, and a raw trained NN baseline. Relevant original messages:
[self-play and replacement](https://naturalprogramminglab.slack.com/archives/D0A7Y12EV0T/p1790391918720509),
[raw baseline](https://naturalprogramminglab.slack.com/archives/C0BTT9JQ67L/p1790392195733979),
[actual human–AI history](https://naturalprogramminglab.slack.com/archives/D0A7Y12EV0T/p1788166309117759),
[early questions and later performance](https://naturalprogramminglab.slack.com/archives/D0A7Y12EV0T/p1788227027416389).

| Group | Teammate | Task 2 questions | Neutral trajectory replay |
|---|---|---|---|
| A | Program-regularized frozen NN | Available | Available |
| B | Exact same regularized pair and hashes as A | Unavailable | Available |
| C | Independently bound paired raw NN | Unavailable | Available |

The mentor's original two-group wording called the raw group B; this three-arm
extension uses C for that baseline. A–B estimates explanation access on the same
NN. A–C estimates the full package against raw; its difference cannot be
attributed entirely to explanation. B–C estimates the model regularization effect.

## Design and participant sequence

Each participant joins one domain and one group. Group and scene triplet are
assigned by the server with balanced randomization inside domain and scene block.
Participants cannot select group, model or scene. Researcher overrides belong
only to preview/test and never become formal samples.

The sequence is consent → neutral control practice → the bound pair's NN–NN
demonstration → Task 1 baseline → Task 2 intervention → Task 3 unaided transfer →
questionnaire and understanding test. Three runs retain one immutable binding.
Each previous task must physically terminate before advancing. Kitchen's neutral
practice is explicitly separate from the NN demonstration and measured tasks.

Both groups without questions retain the same neutral replay/pause opportunity.
Task 2's post-task review window must be fixed after usability work and before
formal enrollment; it is deliberately `null` in the declaration until that lock.
Active play, reading, replay and provider wait are recorded separately. The
estimand is access under these registered usage opportunities. A participants
who choose not to ask remain in A.

## NN execution and model provenance

Two independently hashed Actors are required per pair. Human input replaces only
the human-side action. AI selection is the frozen bound NN followed by the native
physical legality mask; collisions and scores remain environmental outcomes.
External programs and explanations do not choose an execution action. Group,
questions and explanatory text never enter Actor observation or change weights.

A/B bindings must be byte-equivalent as documents. C needs independent pair and
role weights, but matching observation, physical configuration, architecture,
method and acceptance identity within its raw/regularized pair. Domains can
retain their registered training versions; the protocol does not relabel Pong or
Kitchen v5 as newly unified models.

The entire raw ancestry must declare measured zero external-program feedback for
both roles, including initialization. Common BC/reference KL is disclosed; this
is not called from-scratch pure PPO. Each ancestry row names its checkpoint,
parent and source receipt. Declaration validation checks continuity and zero
counts; actual checkpoint/source receipt verification remains the release
auditor's responsibility.

Current Warehouse v6.42 raw failed calibration (5/75 shutdown episodes versus
5% allowed), and regularized passing alone does not admit the pair. All current
failure evidence remains intact. Fixed Pong/Kitchen assets retain their original
acceptance scope; no spent heldout set is rerun. Carried development program
receipts cannot be silently relabeled independent explanation audits.

## Historical question and simulation boundaries

Only A while Task 2 is active may ask about already committed actions in Task 1
or Task 2. A decision at turn t is eligible only after the t→t+1 transition is
committed. Frame zero does not contain an executed action to explain. A question
about the latest completed move reads its decision-before/result-after pair,
not the next unexecuted NN decision.

Answers identify instance, task, decision turn, resulting turn, exact Actor role
hash, saved action and evidence hashes. Program evidence needs its audit plus
agreement with that saved NN action. Insufficient evidence says no motive can be
inferred. A validated behavioral rule is not proof of a unique causal motive.

Counterfactuals clone the registered historical environment and exact binding;
only the named human action is changed. Simulation does not advance real play or
enter Actor input. The displayed horizon is at most 12 and cannot extend beyond
already committed history; subsequent human actions used must be explicitly
identified. All answers, including async pending results and historical answer
access, are revoked at Task 3. Prior learned knowledge is the intended transfer.

## Outcomes and analysis contract

Primary endpoint is Task 3 unaided transfer, separately in all three domains.
This new endpoint does not reinterpret older studies whose primary was Task 2.
Task 2 performance, understanding and subjective ratings are secondary.

| Domain | Primary quantity | Unit |
|---|---|---|
| Warehouse | Native `score.task_score` | Native team points |
| Pong | 100 × final raw points / fixed initial schedule contact capacity | Percentage points |
| Kitchen | `score.metrics.completed_orders` | Orders, 0–5 |

Pong's native `score.task_score` divides by all scheduled points; it is not
silently substituted for contact capacity. The capacity calculation comes from
the fixed initial schedule and the registered existing analysis definition,
`scripts.calibrate_rolling_pong.contact_capacity`. It is an analysis denominator,
not a runtime catch assignment or a public next-action predictor.

The unit of inference is the participant, never a frame, ball or order. A–B can
adjust for Task 1 score because model treatment is identical before explanation.
A–C/B–C total effects do not treat Task 1's policy-affected score as a pretreatment
covariate. Scenario block and registered genuine pretreatment covariates can be
used. Question count/reading time are not primary-effect adjustment variables.
Three domain A–B tests form the primary family with Holm FWER 0.05. Effects,
intervals and domain distributions are reported, including null results and
attrition; outcome-dependent exclusions or endpoint switching are prohibited.
Results are conditional on the selected frozen pairs, not all RL random seeds.

No formal N, effect threshold or achieved power is invented. A power-bound sample
registration must specify balanced A/B/C counts per domain, native-unit minimum
meaningful effects, target power at least 0.80 and its content-addressed report.
Validating its shape does not verify computed power. Formal sample size must be
locked before confirmatory enrollment. Protocol changes after pilot require a
new version and independent formal participants.

## Study stages and admission

Technical usability pilot: three participants per domain/group, 27 total. It
checks operation, understanding and data integrity, not explanation effectiveness.
An independent small pilot precedes powered formal recruitment. Pilot data are
not silently pooled into confirmation. Institutional scope, consent and the
study operator's actual authority must be confirmed before human recruitment;
old Slack references to clearance are not a new approval determination.

Formal enrollment requires independent evidence of:

1. sample/power registration;
2. three-domain model admission;
3. explanation audit for the actual model/programs;
4. execution consistency;
5. persistent storage and restore;
6. institution-owned study-scope confirmation;
7. engineering performance.

The scientific catalog and actual operational evidence readers retain admission
authority. A `passed: true` dict, complete receipt inventory, valid protocol hash,
or software-test result cannot open pilot. Current manual preview remains outside
formal participant data, using isolated temporary storage. Historical studies,
models, receipts and failure verdicts retain their original identity and scope.

## Implemented declaration API and checks

`study_vnext/protocol.py` exposes `PROTOCOL_ID`, `CONSENT_VERSION`, immutable
`SPEC`, `protocol_document()` and `protocol_sha256()` for service composition.
It also exposes strict policy-set/raw-lineage/sample-plan/reference-inventory
checks and metadata-only question permission. All such validation results
explicitly deny enrollment authority and require independent evidence checking.

`tests/test_study_vnext_protocol.py` currently passes 18 tests, including detached
manifest immutability, exact group binding, ancestry feedback in earlier stages,
Task 3/current-action permission refusal, primary metric/contrast separation,
balanced sample declaration and refusal to treat `passed: true` as authority.
Fixtures are synthetic; the sample count used in tests is not a recommended or
registered actual recruitment size. No participant data, live model experiment,
training or final audit was run by this module.

## Optional PostgreSQL persistence adapter

`study_vnext/storage.py` provides `PostgresStorage` as an explicit, optional backend
for the core SQL contract. Construction requires a caller-supplied DSN, isolated
schema name, exact schema sources and protocol hash. Import does not discover
credentials or connect to a database. The existing SQLite study composition is
not silently migrated by this library.

The adapter uses mapping rows and positional qmark translation without changing
question marks in quoted text/comments. SQLite 64-bit integers become BIGINT,
REAL timestamps become DOUBLE PRECISION, foreign keys/unique constraints are
retained, and recognized immutable SQLite triggers become PostgreSQL mutation
rejection triggers. Unknown trigger syntax is rejected. An existing unstamped,
unregistered or differently versioned schema is refused; historical `pl3_`
schemas are never adopted. The adapter creates no DROP/ALTER/TRUNCATE migration.

Writes use serializable transactions with a schema-specific advisory transaction
lock. Read transactions are read-only. Connection, statement, lock and idle
transaction timeouts are bounded. Business errors roll back; no statement or
transaction is automatically retried. A failed commit acknowledgement closes the
connection and reports an unconfirmed outcome for explicit recovery. Stored
trigger metadata, including columns, enabled state and unconditional behavior,
is checked rather than trusting a driver's SQL pretty-printer.

Eleven fake-driver/SQL-compiler tests pass. They cover type/constraint preservation,
original schema identity, exact mutation guards, SQL placeholder/literal handling,
transaction mode, rollback, uncertain commits and sanitized errors. **No live
PostgreSQL connection, concurrent production test, backup or restore was executed.**
These tests do not satisfy `storage_restore` admission. Production integration,
restricted DB privileges and a real restore/concurrency receipt remain required.

## Executable extraction, contrast and power-planning tools

`study_vnext/analysis.py` reads only explicitly supplied in-memory research exports;
it does not open participant files or databases. `participant_rows` requires the
locked release and three-domain A/B/C binding-hash inventory, the exact protocol
document/hash and an upstream NN/physics verification flag. It rejects duplicate
participants, mixed releases/bindings, changed frame chains and malformed final
runs. Preview/test records are listed as exclusions. Missing Task 3 is retained
in a missingness inventory; fitting refuses missing outcomes until the registered
missing-data policy has handled them. It does not silently use an incomplete
score, fill with zero or produce an unannounced complete-case analysis.

Pong additionally requires a per-run independent capacity evidence reference
bound to the exact initial-state hash and registered definition source hash:
`714299b2afc14305396c3fa3b575a9d4090be9e9ea3f1aae9ebb511f12c99d7f`.
The tool independently reproduces that instantaneous-contact upper bound from
the full fixed schedule and checks the provided denominator. Native scheduled-
points score is never used as a fallback. This denominator is an upper bound,
not proof that a physically moving team can attain it.

`contrast_design` creates participant-level design matrices with fixed scenario
block indicators. Only A–B includes Task 1 adjustment. `hc3_contrast` supplies
OLS/HC3 standard errors and explicitly labeled asymptotic-normal intervals/p
values in the native absolute units. Underidentified designs are rejected. It
does not infer practical benefit or compute a misleading ratio from a negative
baseline. The analyst must assess the registered model assumptions and missing-
data plan. `holm_three_domains` requires the complete three-domain primary
family; rejection alone does not establish positive direction or meaningful size.

`power_candidates` accepts explicit native-unit meaningful effects, outcome SDs,
Task 1 correlations, candidate counts and a random seed. It computes a normal-
outcome planning approximation with the joint three-domain Holm procedure and
reports Monte Carlo standard errors. It does not estimate those assumptions,
choose a final N, certify real trial power, or authorize recruitment. Candidate
counts in its tests are synthetic fixtures, not proposed sample-size evidence.

Eight analysis tests pass, using entirely synthetic exports/outcomes. No new
human data or empirical collaboration-effect result was generated. Upstream
export verification flags cannot be authenticated by a statistics function
alone; original runtime and independent admission/receipt verification remain
required.
