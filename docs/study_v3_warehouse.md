# Warehouse restored legacy environment (v3.3)

This release restores the pre-three-domain **6-row × 7-column** Warehouse
map, physics and raw participant scoring from commit
`af97df8589080a6b1b79bad591059b4d52fe33ee`. AI control actually imports a
byte-for-byte copy of the September 2 source commit
`de16551d3d6b99c7ab426dfed1db2871159e6b5c`; it is not an inspired reimplementation.
The user's 13:11 reference is a source-reference candidate: an exact September 2
Render deployment timestamp has not been verified.

## Frozen controller and physical boundary

The historical source is `env/warehouse/historical_sep2_coordination.py`, SHA-256
`ebcbf38c6365752a73b2175ab44750f3231da369421424f5cccd4aada23729c8`.
It uses the original v68 6×7 NumPy Actor, SHA-256
`96762a46f59abd24a10b1abedf8dc325d72c85f3af39424c33e2dcba4ef5ffd3`.
The adapter verifies the Actor, historical controller, and relevant preserved
source hashes in `configs/study_v3_warehouse.json` before opening a new task.
Those source and artifact identities belong to the release manifest.

Actor sampling retains `deterministic=False` with the original independent
`base_seed`, `episode_id`, `frame` keys. This is reproducible sampling, not a new
argmax policy. The historical conservative selector considers every statically
possible human action before seeing the actual submitted command. It ranks
possible conflicts before energy, handoff obligations and route progress. The
September 17 delivery-first changes are not used. The human's recognized command
is submitted unchanged; the old simultaneous physical resolver handles it.

Map topology:

```text
###.###
....###
##.....
...####
##...##
##...##
```

Human/AI start at zero-based `(column,row)=(2,5)/(4,5)`. The charger is `(3,5)`
with three real exit cells above the starting row. Two shared unassigned A→B
jobs remain active. Reaching A claims a parcel, reaching its B delivers it,
and the real seeded sampler replaces that job. Jobs are never pre-assigned to
a human or AI. Task 1/2/3 preserve the original seeds **52000/51000/51500** and
120-step limit; the enrollment seed is retained privately but does not secretly
change these exact historical scenes.

A successful move costs 2 battery. Waiting at the charger restores up to 10,
capped at 100. Walls cancel movement without a battery cost. Same-target moves,
swaps and moves into a stationary partner create a collision; following into a
cell the other robot actually leaves is resolved by the original physics.
Reaching the charger at exactly zero battery is safe; a shutdown elsewhere
ends the task. Initial batteries are 100, as in the original environment.

The displayed score is the original raw score, **not a percentage**:
+10 delivery, −10 collision, −5 per shutdown robot, −1 per budget step,
−2 per human detour unit, and −5 for a shared-charger violation. A shutdown
also charges the unused step budget. Detour units use the same frozen human
target and physical alternative moves as before; a wait can count if safe
progress was available. The charger penalty is charged once during a continuous
stay only when the exact latest legacy conditions hold; it is not removed by
restoring the older controller. Both groups have identical scoring.

## Shared study adapter and persistence

`domains/warehouse/turnbased.py` implements the existing pure engine API. It
creates an isolated `WarehouseMultiAgentEnv` per operation. Its private snapshot
contains the complete dataclass state, all coordination memory, random generator
state and episode counter. Tagged tuple serialization preserves nested tuple
values even inside coordination-plan dictionaries. Recovery **does not call**
legacy `set_state()`, since that intervention helper recomputes navigation goals
and handoff plans. Instead it validates and restores the exact committed state.

The old start-round lifecycle still calls `set_state()` once during task
creation after declaring the human-controlled robot, matching af97df8. There is
no global monkeypatch, global mutable environment, or shared random generator.
The loaded Actor is shared read-only; each sampling call uses its own keyed RNG.

Public state exposes `width=7`, `height=6`, map walls, charger, human/AI position,
heading, battery and carried task, and the two shared jobs with `carrier`.
It does not expose the private snapshot, seed, RNG, policy memory or next action.
Score exposes `task_score == raw_score`, `score_scale="raw"`, `score_max=null`,
all six score components and auxiliary counts. Scores may be negative.

The common study service still owns enrollment, stage permissions, persistent
records and questionnaires. Only group A during active Task 2 may obtain
explanations; restoring the old engine does not restore the older access rules.
The shared frontend supplies the old Warehouse appearance and animation.

## Explanation evidence and demonstration

Plain bilingual facts bind to the real historical decision candidate table,
selected action, goal distances, collision alternatives and charge calculation.
They distinguish potential human conflicts from knowing a future command.
Charging evidence includes route legs, reserve, current threshold and arithmetic
for the missing battery; a handoff can require leaving before that threshold.
Current rule facts and score reasons read the af97df8 configuration and actual
transition events, including the newer charger-occupancy fee.

Human advice controls only a proposed human command; it cannot replace or tune
the AI. It uses the currently visible state and fixed AI, and is a coordination
suggestion rather than a claim of globally optimal play. Its short branch checks
stop looking ahead when a newly sampled job would be disclosed. Counterfactual
answers use real isolated engine transitions and the current fixed AI.

The six-caption neutral demonstration runs a real trajectory using the original
40786 demonstration seed. It shows pickup, delivery/replacement, charging, one
intentional human collision and completion. Its AI is exactly the same frozen
historical AI. Three bilingual comprehension items test charging arithmetic,
conservative conflict handling and replacement jobs; correct answers remain
server-side.

## Verification and honest limits

The domain tests compare all successive states, random generator state, sampled
replacement jobs and raw score with an independently Git-loaded historical
controller running the preserved physical environment. They additionally cover
JSON restoration of active plans, concurrent pure decisions, group blindness,
wall input preservation, charging, once-per-stay penalties, shutdown accounting,
private/public boundaries and the actual demonstration.

Regenerate the **82** fixed bilingual question cases with
`python -m domains.warehouse.build_qa_cases`. The builder includes hand-written
questions and independent numerical expectations, not generated answer copies.
Coverage includes current reasons, alternatives, history, human-only simulation,
false premises, ambiguity, raw scoring, charging arithmetic and charger handoff.
Injected response plans test evidence composition and state binding; they do
**not** test language-model question understanding. A real-provider evaluation
and independent semantic review are separate deployment checks.

Historical restoration invalidates the previous v3.2 9×7/144-episode feasibility
claim and its six-delivery 100-point ceiling for this version. Existing records
must retain their old version identity. No simulation, technical test, or
successful deployment establishes the targeted 50% human Task 2 improvement.

The human advice helper is not a certified optimum or a guarantee of finishing
without shutdown. A current development run on the three fixed task seeds
produced 2 / 13 / 0 deliveries with raw scores −161 / −22 / −193; Tasks 1 and 3
ended in a battery shutdown. These are synthetic feasibility observations only.
They preserve the exact requested historical AI rather than silently changing
its behavior to pass a proxy. The neutral demonstration uses a separately fixed
human demonstration policy and shows seven deliveries, one collision and actual
charging over 120 steps. The real human pilot remains necessary.
