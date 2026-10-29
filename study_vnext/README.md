# Three-arm study implementation

This package is an isolated **researcher preview**, not an admitted human study.
`StudyStore` refuses pilot mode even when a scientific catalog is supplied.
It does not replace the live manual playground or reinterpret historical data.

The protocol is `study-protocol-20261029.v1`. A/B share the exact regularized
binding; C uses raw. Task 1 establishes a baseline, Task 2 permits A's historical
questions, and Task 3 provides the unaided primary outcome. All groups have
neutral replay. The server permits questions only about committed human–AI
transitions; the current unexecuted decision stays private.

## Local researcher preview

Use a new explicit database path and set `POLICYLENS_STUDY_PREVIEW_ADMIN` to a
private value of at least 32 characters. Do not use a production participant
database, an old `pl3_` database, or an earlier core schema.

```sh
python3 -m study_vnext.server --database /absolute/path/to/new-study.sqlite3 --port 8014
```

Open <http://127.0.0.1:8014/study/>. The researcher enters the private key and a
synthetic identifier. Do not enter personal information. Complete the neutral
key practice, play the bound NN–NN demonstration, then use WASD/Space (Kitchen E)
through all three tasks. Complete both the ratings and the candidate mechanics
questionnaire. The questionnaire is not a validated measure of NN-strategy
understanding.

For an explicitly isolated PostgreSQL schema, replace `--database` with
`--postgres-env POLICYLENS_STUDY_NEW_DSN --postgres-schema study_preview_v1`.
Only the named variable is read; the service does not discover old credentials.
The persistent backend is wired, but a local restore drill does not satisfy a
production deployment's storage gate.

Questions use grounded local templates by default. Optional external intent
selection requires `--enable-consented-qa-provider`, an operator's explicit
`POLICYLENS_STUDY_QA_TRANSMISSION_APPROVED=yes`, the explicit
`POLICYLENS_STUDY_QA_BASE_URL`, `POLICYLENS_STUDY_QA_MODEL` and
`POLICYLENS_STUDY_QA_API_KEY`, **and the instance's separately recorded opt-in**.
Without the individual's opt-in the provider is never called. The provider can
select verified fact IDs, not generate unrestricted policy explanations. Its
configuration is not evidence of production QA validation.

## Core operations

`StudyStore` owns immutable consent/preparation/questionnaire records around
`runtime_v642.Store`; the latter still owns frozen NN execution, exact physics,
frame chains and replay. HTTP operations live under `/study-api/`:

- `status`, researcher-only `enroll`;
- `{id}/view`, `preparation`, `demo`, `demo-complete`, `next-task`, `action`;
- `{id}/replay?task=1`, `question`, `answers`;
- `{id}/understanding` (GET/POST after completed Task 3), `survey`, `event`;
- `{id}/export` (session plus researcher authorization).

Questions/answers are A-only while Task 2 is active, including after restoring a
session. An asynchronous provider response is checked again before delivery.
Task 3 revokes both new questions and previously saved answer access. The
explanation context is never an Actor observation. Historical counterfactuals
use a cloned environment and the same binding and cannot extend past the saved
history head. One-step score comparisons are independently recomputed on export.

Practice and demonstration receipts record **client-reported completion**;
they cannot prove attention. Timing separates active play, explicit answer
reading mode, question entry, replay and provider wait. These are interface
duration proxies, not attention or learning measurements. Offline unsent timing
events are retried with stable IDs.

Exports recompute NN/physics evidence, use deterministic record order and
contain no session token. PostgreSQL transactions fail closed on serialization
conflicts; the backend does not silently retry. The client can explicitly retry
the same action identity. Existing old schemas are never silently adopted.

## Checks and remaining admission

`scripts/check_study_vnext_postgres.py --output /new/output/path` creates and
removes its own private local PostgreSQL cluster. It exercises synthetic data,
immutable guards, concurrency and dump/restore without an existing cluster.
`scripts/check_study_vnext_models.py` checks the actual registered models in
preview mode, using already public manual scenes; its results are software flow
checks, not NN quality evaluation.

`protocol.py` validates declarations. `analysis.py` extracts participant outcomes
and implements the registered contrasts/HC3/Holm and explicit-assumption power
planning. Neither a declaration validator nor a software test grants enrollment
authority. Formal sample size, review duration, meaningful effect assumptions,
institutional scope and production service evidence remain to be registered and
independently verified. Warehouse's current paired calibration remains failed.
No new training, human efficacy result, final sealed audit or release is implied.

See [the protocol](../docs/study_protocol_20261029.md) and
[the implementation report](../docs/study_mentor_implementation_20261029.md).
