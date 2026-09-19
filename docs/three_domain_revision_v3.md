# Three-domain study revision v3

Release candidate: `policylens-three-domain-20260920.v3`.
This document describes the implementation; it is not deployment evidence.
The deployed identity and acceptance receipts are recorded separately in
[three_domain_deployment.md](three_domain_deployment.md).

## Shared interaction and research separation

Demo → Task 1 → Task 2 → Task 3 → Questionnaire; English on first visit, with
an explicit Chinese switch at the top right. Only Group A's active Task 2 can
create, receive, or read explanations, including selected past Task 1 frames.
Ending Task 2 revokes access before Next. No automatic goals or strategy hints
are shown outside the question panel. Both groups run identical physics,
schedules, controllers, budgets, score and demonstration.

Actions are key driven, not clock driven. A single confirmed server transition
is animated for about 400 ms on a persistent Canvas. Direction holds issue the
next command only after acknowledgement and animation; there is no action queue.
Space and E execute only on the initial keydown. Questions, replay, loss of focus
and completion stop held movement. Slow responses do not cause speculative
movement. A delayed replay cannot replace a newer selection or a return to now.

| Domain | Keys | Score |
|---|---|---|
| Warehouse | WASD/arrows; Space waits | Historical raw points, no fixed maximum, may be negative |
| Pong | A/D; Space waits | 100 × caught points / fixed scheduled points |
| Kitchen | WASD move and face; E uses front station; Space waits | 100 × correct on-time orders / fixed order count |

## Domain changes

Warehouse restores af97df8's 7-column × 6-row environment, two shared replenished
A→B jobs, charging, collisions, shutdown, time and detour accounting, the 120-step
limit, and original task seeds. The historical controller is the actual de16551
source with the v68 Actor; its source and weights are hash checked. The source
commit time is September 2 at 12:54:11 +08:00. An exact September 2 13:11 deployment
receipt has not been established. Explanations use the selected historical
controller's actual candidate/conflict/clearance/energy evidence. Full snapshots
preserve the original state, random generator and coordination plans.
See [Warehouse details](study_v3_warehouse.md).

Pong uses nine lanes, twelve vertical cells, a 540×630 board, small balls moving
two cells per turn and team balls one. At most three small and two team balls
are visible. Small balls arrive every two turns and team balls every six;
pre-generated schedules stop adding balls that could not reach the line by the
60/90/90-step limit. A team catch needs two distinct contacts. The nearest
reachable team-ball assignment is committed; a later one is explicitly tentative.
The fixed AI considers visible balls, sequential travel and safe small-ball
opportunities. Questions cannot inspect undisclosed future balls.
See [Pong details](study_v3_pong.md).

Kitchen implements tomato-and-egg and pepper-and-meat stir-fries. The AI cooks
the protein, physically plates and stores it, cooks the vegetable, retrieves and
returns the protein, combines, and hands over a dish in an output container.
The human transfers that dish to a formal serving plate before serving. Four
separate cupboards, one human counter, a one-item handoff, two raw slots and two
dedicated protein-plate slots obey physical facing and capacity constraints.
Ingredients have identities, bound orders and combination lineage. An invalid
E reports the public reason without consuming a turn; a blocked direction turns
in place and consumes one. The tasks contain 4/6/6 orders and 240/360/360 steps.
Two recipes can overlap across two pans. The fixed geometry and short cooking
timers do not imply that both pans must be heating at the same instant.
See [Kitchen details](study_v3_kitchen.md).

## Explanations and persistence

`study-evidence-qa.v3.2` binds semantic questions to a selected recorded state,
uses domain-specific action labels, and composes answers from verified facts.
Counterfactuals run the real engine with the recorded first AI decision and
stop at the first newly revealed order or ball, filtering its undisclosed
content. Asking never advances the task. The service may request clarification;
perfect understanding of every arbitrary question is not claimed.

Existing session, revision, idempotency, replay, questionnaire and export APIs
remain. The release ID separates new records. Earlier states are rejected with
a version message, not silently converted to different gameplay. Existing
research records and the original dirty checkout are retained.

Public state extensions: Warehouse exposes `score_scale:raw`, `score_max:null`
and six score components; Pong exposes `height`, ball `y`, `vy` and remaining
arrival turns; Kitchen exposes both actors' `facing`, public front interaction,
recipe/pan phase, item containers and separated counter slots. Hidden schedules,
AI commitments and decision audits remain server-side.

## Evidence boundaries

Automated engine, HTTP, UI-unit and model checks are software acceptance.
Synthetic coordination proxies are simulations, not human A/B results. In Pong,
the cooperative proxy scores 100 while the stronger history-learning baseline
averages 77.69 in held-out Task 2, a relative difference of about 28.7%; this does
not establish a 50% human effect. Warehouse's unchanged historical task can have
negative scores and a non-optimal proxy can shut down. If mean B is zero or
negative, relative percentage improvement is not reported; use absolute points,
deliveries and other outcomes. The human Task 2 target remains 50%, unmeasured.
