# Three-domain controller and transition contracts

Implementation snapshot for `policylens-three-domain-20260920.v1`, prepared on
2026-09-20. This document describes the new engines. It is not evidence that a
particular Render deployment passed acceptance or that explanations improved
human scores. See the separate validation and deployment reports for those
statuses.

## Shared authority and pure engine interface

The domain registry in `study_v3/registry.py` loads these implementations:

| Domain | Engine | Rules version | Scenario version |
|---|---|---|---|
| Warehouse | `domains/warehouse/turnbased.py` | `warehouse-turnbased-v3.2` | `warehouse-scenarios-v3.2` |
| Cooperative Pong | `domains/pong/turnbased.py` | `pong-turnbased.v3.0` | `pong-waves.v3.0` |
| Cooperative Kitchen | `domains/kitchen/engine.py` | `kitchen-v3.1.0` | `kitchen-scenarios-v3.1.0` |

Every engine implements the interface in
[three_domain_engine_api.md](three_domain_engine_api.md):

| Function | Contract |
|---|---|
| `initial_state(seed, task)` | Create a JSON-serializable state for one task. Task is 1, 2 or 3. |
| `legal_actions(state, actor='human')` | Enumerate actions that can be submitted from this state, including wait. |
| `decide(state)` | Return the actual next AI action, reason, goal, proposed memory and supported alternative comparisons. |
| `step(state, human_action, decision=None)` | Validate the input and return a new settled state; reject a terminal input or an invalid action. |
| `public_state(state)` | Produce an explicit participant projection without private plans, reasons, seed or undisclosed schedules. |
| `score(state)` | Return nonnegative 0–100 Task score, domain-specific raw score and auxiliary metrics. |
| `rules(language)` | Return public physical and scoring rules, in English or Chinese. |
| `facts(state, decision=None)` | Return verified bilingual evidence for the separately authorized explanation service. |
| `human_advisor(state)` | Supply a development proxy action for the human role only. It does not appear in participant views. |
| `demonstration()` | Return an independently played demonstration's public frames and six bilingual mechanics captions. |
| `comprehension(language)` | Return three fixed new-situation questions with researcher answer keys; the server removes keys. |

Kitchen also supplies `action_label` for its interaction buttons. No controller
accepts experimental group, participant identity, question text or an unsubmitted
human action. All real decisions use the action-before state. The AI's proposed
memory is copied into the true state only by a confirmed transition. Calling
`decide`, rendering evidence, changing language or using a simulated branch does
not commit that memory.

The store saves state at turn 0, then each confirmed transition. A saved frame
at turn *t* is the state **before** action *t*. Its `human_action` and
`decision_json` describe the transition to frame *t + 1*. Public events on
frame *t + 1* describe that settled transition. A terminal frame has no next
human action and an empty saved decision. This convention applies to replay and
counterfactual binding.

## Warehouse

The new 9 × 7 warehouse has two rooms, a narrow crossing and a shared charger
with side exits. Each robot has three assigned finite orders, visible from the
beginning. Reaching its current pickup or dropoff performs the corresponding
interaction. There is no infinite order refresh and no ability for the AI to
complete the human's assigned orders.

The conservative behavior is inspired by the September 2 reference candidate
`de16551`. An exact correspondence to the user's remembered 13:11 live build
has not been established. This is a newly versioned controller that adopts the
reference's caution, clearance and energy principles, not a byte-for-byte
rollback or reuse of an old release hash.

The controller uses shortest walkable-grid paths and ranks decisions as follows:

1. Avoid moving into an energy state that cannot reach a safe charging route.
   Delivery and return calculations retain a two-move reserve where required.
2. Compare collision possibilities against the legal human actions from the
   same current state. Prefer an action with no possible collision. If every
   option has risk, count and explain the remaining risk; do not claim to know
   the person's next action.
3. Retain a still-applicable loaded crossing, temporary clearance or charging
   plan, while rechecking safety. A loaded AI crossing the shared area preserves
   its direction instead of taking an equally risky lateral detour. The person
   can release a wait by clearing the entrance. The AI instead yields when the
   person's energy cannot cover a clearance detour and charger return.
4. Clear an occupied route and leave a sufficiently charged station when needed.
   Among equally safe clearance choices, prefer one that also advances the AI's
   mission. A short clearance hold lasts at most two turns and can end earlier.
5. Make safe delivery progress, otherwise wait. A finished AI parks away from
   remaining publicly known partner pickups.

Simultaneous movement into an occupied partner cell, an exchange of cells, or a
shared destination cancels both moves and records one collision event. A
successful move costs 3% battery. An explicitly submitted wait on the charger
adds up to 10%, capped at 100%; cancelled moves and waits elsewhere do not use
energy. Below 3% away from the charger is an immobile shutdown state. A newly
entered shutdown is counted once, not on each later wait.

All tasks have a 120-turn limit and six total deliveries. Task 1 begins with
one same-room order per robot before cross-room work. Task 2 repeatedly uses
the shared crossing. Task 3 varies endpoints and starts. Task 1/3 start at
human/AI battery 54%/48%; Task 2 starts both at 78%. These are shared scenario
conditions, identical for A and B.

Task score is `100 × deliveries / 6`. Auxiliary net score is
`100 × deliveries − 200 × collision_events − 50 × shutdown_events − turns`.
There is no charger-occupancy penalty. The task ends when all six deliveries
finish or the turn budget is exhausted. More implementation detail, calibration
changes and proxy limitations are in [study_v3_warehouse.md](study_v3_warehouse.md).

## Cooperative Pong

The arena has nine lanes, internally numbered 0–8 and displayed as 1–9.
Both paddles select left, right or wait, moving at most one lane per turn.
They may overlap without a collision penalty. Each wave has one cooperative
ball and at most one ordinary ball. Contacts, object IDs and countdowns for the
current wave are public; future wave contents are not.

The controller first checks both assignments of a cooperative ball's two
contacts. Both partners must be able to reach their respective contacts in the
remaining turns. It retains an existing feasible assignment. A new assignment
minimizes the sum of the two required distances, choosing the right-hand contact
for the AI on a tie. If its old assignment becomes infeasible, it can use the
other feasible assignment. With no feasible cooperative assignment, it may
protect a reachable ordinary ball. Without a verified reachable job it waits.

An ordinary-ball detour during a cooperative commitment is permitted only when
the AI can reach the ordinary ball and then return to its cooperative contact
in time. The player moving slightly does not itself reverse the assignment.
Distances and countdowns establish conditional reachability, not certainty
about the person's future behavior. There is no random or neural fallback.

Settlement order is both paddle moves, then decrement the current balls'
countdowns, then settle all arrivals exactly once. An ordinary ball earns one
shared point if either paddle covers its single contact. A cooperative ball
earns three shared points only when distinct paddles cover its distinct
contacts. Misses earn zero; overlapping at one cooperative contact is a miss.
After a wave ends, the next pre-sampled wave becomes visible and positions carry
over. A question cannot delay a countdown because time only advances on a move.

Task 1 has six waves and 30 turns. Tasks 2/3 have eight waves and 41 turns.
Waves last 4–6 turns. Task 3 mirrors contact combinations and reverses initial
positions, while keeping the same controller. Each entire schedule is generated
from the task seed independently of group or performance.

Task score is `100 × caught points / all scheduled points`. Some opportunities
are incompatible, so 100 need not be reachable. The exact current-wave search
uses the fixed AI and controls only the human. A separate global upper-bound
search knows the full schedule **only for offline researcher verification**;
neither participant answers nor AI decisions use that future knowledge. See
[study_v3_pong.md](study_v3_pong.md) for reproducible search evidence and caveats.

## Cooperative Kitchen

This is a custom Overcooked-inspired task, not an unmodified Overcooked-AI
benchmark. The 9 × 7 map has an outer wall and a divider at column 4. The
handoff station at (4,3) is an impassable, single-item counter. Human and AI work
on separate sides. The human collects one full portion, chops it, transfers it,
collects cooked soup, plates and serves. The AI receives prepared portions,
cooks in either of two pots and returns cooked soup. Each role has one hand
slot and its own single-item storage counter. No food teleports between roles.

The fixed controller measures actual grid movement plus interaction time:

1. Protect a pot when continuing another job would miss its removal window and
   a timely rescue exists. If necessary, first free the hand through a feasible
   empty counter or usable pot; preserve the rescue commitment after doing so.
2. Deliver held cooked food through a free handoff. Use the AI buffer when a
   blocked handoff can be bypassed. Preserve useful cooked food if both are
   full; a recoverable capacity change remains available through public actions.
3. Remove reachable ready food and recover buffered cooked food. An unreachable
   rescue is identified honestly rather than described as successful prevention.
4. For unstarted prepared portions, prefer the earliest currently visible
   matching deadline with stable ties. Keep a selected loading pot while it
   remains feasible. Deadline screening is an optimistic public lower bound,
   not a guarantee about the human's future actions.
5. While one pot cooks and another is empty, stage near the handoff for new
   preparation only when an actual return route still fits the removal window.
   Otherwise work or wait at an appropriate reachable station.
6. Clear burnt pots or unusable food under the documented capacity/expired-food
   conditions. Do not discard deliverable food to create artificial difficulty.

Both actions are validated against the same pre-action state. Moves and
interactions settle first. If both actors interact with the handoff in the
same turn, both fail and all item slots remain unchanged. A newly placed item
cannot be picked up that turn. Existing pot timers then advance; a newly loaded
pot is excluded from that decrement. Two chopping interactions prepare an
ingredient; tomato/onion cooking takes 12/14 full subsequent turns. Newly ready
food starts at ready age zero, then burns after six full additional turns.
Removal during the sixth interaction phase still succeeds. Once removed, food
has no spoilage timer.

Serving during deadline turn T is accepted before the order expires. The
earliest-deadline pending matching order is completed once. Newly arriving
orders are then revealed. Hidden recipes and arrival times are unavailable to
the policy and evidence functions until publication. Each nonmovement action,
including discard or clearing a burnt pot, consumes one turn. Chop progress
belongs to its item and survives questions and storage.

Task 1 has four orders and 140 turns, with deadlines 45/75/105/140. Task 2 has
six orders and 160 turns, with deadlines 45/70/95/120/145/160. All Task 1/2
orders start visible. Task 3 uses the same six deadlines and budget, varies
recipes and human starting position, and reveals orders at turns
0/0/24/48/72/96. Task score is `100 × on-time correct orders / total orders`;
waste, burns and blockage are auxiliary observations without hidden penalties.
See [study_v3_kitchen.md](study_v3_kitchen.md) for strict simultaneous-cooking
and real rescue trajectories using the unchanged fixed AI.

## Evidence and counterfactual contract

Only the store's active-A/Task-2 gate can expose `facts` through the shared
question service. The semantic model binds a question to recorded evidence and
may propose hypothetical human actions. Verified bilingual statements and the
actual simulator determine the factual response. Model prose does not become
an unchecked account of the AI's reason.

A counterfactual starts at a deep copy of the selected pre-action state. Its
first step retains the recorded AI decision; subsequent steps use the same
policy on the branch. The default horizon is one and the maximum is 12. Any
additional unspecified human turns are explicitly reported as assumed waits.
An illegal action is reported and not executed. At the first newly revealed
order or wave, the branch stops and new external content is filtered from the
participant result. It does not change the true state or consume its schedule.

The service can report unavailable or request clarification. True evidence
alone does not prove that a model selected relevant evidence or fully understood
an arbitrary question. Real-provider testing and independent question review
remain separate from deterministic evidence and mechanism tests. See
[three_domain_qa_evaluation.md](three_domain_qa_evaluation.md).
