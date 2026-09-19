# Warehouse v3.2 controller and calibration

This new deterministic engine is independent of the old Warehouse release and
does not import Torch, NumPy, or a saved actor. It follows the conservative
coordination principles in the September 2 reference candidate `de16551`.
The code does not claim to restore that historic deployment byte-for-byte.

## Mechanics

The 9 × 7 map has two rooms linked by a narrow shared crossing. The shared
charger at (4, 2) has exits to both rooms and to the crossing at (4, 3). Each
robot has three publicly visible, finite orders. Orders are handled in ID order;
reaching the current pickup/dropoff automatically performs that interaction.
One robot cannot complete the other's assigned deliveries.

One human choice and the AI's pre-action decision settle simultaneously.
Entering a partner's occupied cell, swapping cells, or selecting the same
destination cancels both moves and records **one** collision. A successful move
costs 3% battery. A submitted wait on the charger restores up to 10%, capped at
100%. Cancelled moves and waits elsewhere cost no battery. Dropping below 3%
away from the charger creates one shutdown event because the robot can no
longer pay for a move; repeated waiting does not create more shutdown events.
This explicit below-3% immobility rule closes the previous 1–2% stranded-state
gap and is displayed in the public instructions.

All tasks retain the original 120-turn budget. Task score is 100 × deliveries
/ 6. The separate net score is 100 × deliveries − 200 × collision events −
50 × shutdown events − elapsed turns. There is no charger occupancy penalty,
group multiplier, hidden score adjustment, order refresh, or wall-clock cost.

## Actual controller

The policy reads the public current state and its own prior clearance memory.
It cannot read an unsubmitted human action; it does not accept group. Every
legal human move is checked from the same pre-action state. Rules rank:

1. Avoid moving into an unrecoverable energy state; open-cell shortest paths
   determine the energy to complete the current order, return, and retain two
   moves of reserve.
2. Minimize how many legal human actions can collide. A collision-free option
   is preferred; when every option has risk, remaining risk is counted and
   stated rather than claiming absolute safety.
3. Maintain an applicable loaded crossing, short clearance plan, or charging
   plan. While carrying a parcel across the shared crossing, the AI retains its
   crossing direction and waits rather than making a lateral detour of equal
   collision risk. The person can release this wait by retreating beside the
   entrance. The ordinary active-yield rule applies when the person's remaining
   battery cannot cover a two-move clearance detour plus a charger return.
   The commitment ends once the AI reaches the other room. A clearance
   hold lasts at most two turns and is released earlier when the person leaves
   the conflict area. Every turn rechecks safety and energy.
4. Leave a sufficiently charged station when another robot needs it, and
   actively clear an occupied cell on the person's route. Among equally safe
   clearance actions, prefer one that also advances the current mission.
5. Continue safe goal progress; wait when the alternatives are less safe or
   cannot preserve return energy. Finished robots park away from the remaining
   partner pickup locations instead of creating an unnecessary final crossing.

The explanation is generated from this exact decision and real alternative
comparisons: action, open-cell distance, remaining battery and collision counts.
It distinguishes a possible collision from knowing a participant's next move.
Querying, translating, replaying and simulating do not update policy memory.
Memory is copied into state only in a real `step` transition.

## Pre-release calibration and evidence

Development seeds 100–123 and held-out seeds 1000–1023 are frozen in
`configs/study_v3_warehouse.json`. All three tasks use the same policy; Task 1
has one same-room delivery per robot followed by cross-room work, Task 2 has
repeated cross-room deliveries, and Task 3 changes endpoint combinations and
starting positions. All six orders are public from turn zero, so there is no
hidden future schedule for either controller or explanations to access.
Task 1 and Task 3 start with human/AI battery 54%/48%; Task 2 starts both at
78%, allowing their loaded routes to meet before their next shared recharge.
This creates a natural crossing decision without changing either group's rules.

Initial calibration used a charger cul-de-sac. A depleted human standing at its
only exit could prevent every safe AI departure. This was a real feasibility
failure, so the shared map was changed to give the charger two side exits.
The fixed shared 120-turn budget and score were retained. A second issue was
finished robots parking near another robot's later pickup; the controller now
selects an in-room corner away from remaining publicly known pickup points.

A human-only coordination proxy uses the same immutable AI, yields real space
when needed and sometimes charges the human longer to allow manoeuvring.
It never commands the AI or reads future schedules. Across **144 actual
episodes** (48 seeds × 3 tasks), all six deliveries finish with zero collisions
and zero shutdowns. Maximum elapsed turns by task are **83 / 86 / 89**.
These are feasibility results, not participant A/B results or evidence of a
50% explanation effect. A real human can still make poor moves, collide, run
out of battery, or finish with a lower score.

The fixed neutral demo has 67 transitions with six public-mechanics captions:
controls, pickup, charging, one deliberate collision, recovery/delivery and
completion. It uses actual engine transitions and finishes six deliveries.
The three comprehension keys are checked against explicit charging, crossing,
and charger-departure states; they are not supplied to participant views.

## Proxy comparison and remaining ceiling

The first feasible version allowed even the public-rule greedy proxy to finish
all 48 Task 2 scenes. Version 3.2 adds the loaded-crossing commitment described
above and uses the public initial-energy adjustment. This changed actual shared
coordination; score formulas and group conditions were not changed.

Reproduce the following figures with `screening_report()` in the engine. Each
split contains 24 Task 2 scenes. All proxies control **only the human** while
the same AI runs its real rules:

| Proxy | Development mean Task score | Held-out mean Task score |
|---|---:|---:|
| Random legal human actions | 43.75 | 39.58 |
| Greedy own-task route, occupied-cell avoidance, wait after collision | 41.67 | 54.17 |
| Public-history learner that explores away after two unchanged turns | 98.61 | 98.61 |
| Human planner informed about the fixed coordination rule | 100.00 | 100.00 |

The public-history learner does not read AI decisions or memory. It reacts to
two unchanged position pairs with three energy-safe retreat/exploration moves,
then resumes its own shortest-path/charging strategy. Its near-perfect result is
a real **remaining ceiling limitation**: a participant may also learn the
cooperation pattern quickly without explanations. These proxies therefore do
not establish a robust explanation advantage, let alone a 50% human A/B effect.
The pilot must measure that outcome; version 3.2 is frozen rather than weakening
the learning baseline or adding arbitrary score penalties to manufacture a gap.

## Question evidence

`domains/warehouse/qa_cases.json` contains 80 bilingual cases (40 substantive
questions in English and Chinese). They cover current decisions, exact battery
and route counts, charging thresholds, interaction order, a real collision,
history, follow-up object binding, wrong premises, clarification, and read-only
multi-step alternatives. Each includes the actual selected state and decision,
expected evidence IDs, and independent state/score checks. Explicit battery
shortfall and charging-turn facts allow the answer to state "20%, two turns"
instead of requiring the user to infer them from separate numbers.
These fixtures verify evidence and simulation; they alone do not prove the
external semantic model understands every possible question.
