# Committed human–AI history questions

This is a **software API update**, not model admission, study deployment or
human-effect evidence. It preserves the frozen Actor/physics implementations and
all catalog/pilot gates. Core Store, HTTP and export identities now mark the
committed-history schema; an old core database is not silently reinterpreted.

## Question coordinates

`POST /v642/instances/{id}/questions` accepts:

```json
{"question":"Why did you do that?","target_task":2,"decision_turn":0,"language":"en"}
```

- `target_task`: 1 or 2, default 2.
- `decision_turn`: zero-based **action-before-state** coordinate. Decision t is
  the saved AI action that physically produced committed state t+1.
- Legacy `turn` remains the same decision-turn alias. Providing both coordinate
  fields is rejected, even if equal.
- Omitting both selects the latest **executed** decision of the target task.
  A task with no completed joint action returns 409; the current head state's
  unexecuted NN action is never a question target.
- Historical Task1 is authorized only while this instance is group A's active
  Task2. B/C, completed Task2 and Task3 receive 403.

The response binds `target_task`, `decision_turn`, `result_turn`, actual recorded
`human_action` and `nn_action`, decision/result frame hashes, the committed
history head at question time, immutable model/program evidence and question ID.
Answers describe an audited behavioral pattern or explicitly say that motive
cannot be inferred; they do not establish a unique causal motive.

For historical evidence only, Store creates an ephemeral internal
`BoundDomain.question_context_task=2` **after** live authorization. This enables
an eligible Task1 program path without altering target state, Actor input, model
binding or persisted group. It is not an HTTP option and never permits B/C or
Task3 program access.

## Counterfactual window

Optional `human_actions` contains 1–12 explicit alternate human commands. Its
length cannot exceed `committed_head - decision_turn` at question time.
The initial snapshot and model stay exactly bound. Only an isolated physical
copy is stepped; its final turn cannot pass the recorded historical head. The
question stores that head/hash so a later commit cannot retroactively enlarge
an earlier counterfactual. Invalid, overlong or unbound simulations are rejected
without creating a question or advancing the real game.

## Public frame and answer projection

- Each current head's `decision` is `{}` and `decision_committed=false` until its
  physical result exists. Private stored next decisions still support exact
  deterministic replay but are not sent to participants.
- `last_executed_action` is null initially, otherwise contains
  `{decision_turn, result_turn, human_action, ai_action}` for the incoming,
  already committed transition.
- Historical frames may show their outgoing decision only when that decision
  has actually executed. Private state/RNG, distributions and restorable
  transition bundles remain excluded.
- `GET /v642/instances/{id}/answers` returns verified question history only for
  A's active Task2. Projection rechecks live authorization, preventing an answer
  captured before Task2 completion from being served after permission expires.
- Researchers may retain/reverify the question evidence in authorized exports
  after participant access ends. Ordinary frame/replay access remains neutral.

## Verification

The existing core store/server/registry software tests were updated to execute
an action before asking about it. New synthetic transition and local HTTP tests
cover initial/head non-disclosure, previous-action display, default/explicit
coordinates, Task1 history, replay identity, counterfactual temporal bounds,
B/C denial, Task2/Task3 answer revocation and tampered recorded windows. These
use temporary databases and synthetic adapters, not scientific heldout seeds.
