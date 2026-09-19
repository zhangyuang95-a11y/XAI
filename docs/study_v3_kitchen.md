# Cooperative Kitchen implementation and validation

This is a deterministic, **Overcooked-inspired custom task**, not an unmodified
Overcooked-AI benchmark. It uses no third-party game code, assets or trained
network. The rule controller runs locally inside the authoritative Python server;
semantic question handling and study permissions belong to the shared study layer.

Implementation: `domains/kitchen/engine.py`. Rules version: `kitchen-v3.1.0`.
Frozen scene version: `kitchen-scenarios-v3.1.0`.

## Public mechanics

The 9-by-7 map, workstations and actor roles follow the implementation brief.
Both actors move one four-neighbor square, wait, or perform one adjacent-station
interaction per confirmed turn. They work in physically separate areas. People
prepare ingredients, plate and serve; AI cooks and transfers food. Neither role
can execute the other role's stations, and food never teleports.

Each person holds one item. The handoff counter and the two storage counters each
hold one item. A counter interaction either picks up or puts down; it never swaps
items. Both actors have an always-available, one-turn discard action when holding
something. A human can recover their own handoff item and store or discard it.
Burnt pots take one AI interaction to clear. These actions provide a recovery
path even with full hands, full counters or expired orders. The AI does not
arbitrarily discard a still-deliverable soup.

One tomato/onion portion is the full ingredient requirement for one soup. It
takes two chopping interactions; progress stays attached to the item. Tomato
cooks for 12 full turns and onion for 14. Plates are supplied without depletion.
Removed soup has no spoilage countdown.

Every transition has this order:

1. Validate both actions against the pre-action state. AI has already chosen
   using that state; it cannot read the person's unsubmitted action.
2. Apply movements and interactions. If both actions target the handoff counter,
   both fail, all three involved item slots stay unchanged and one turn passes.
   An item newly placed this turn cannot be collected in that same turn.
3. Advance existing pot timers. A pot loaded in this transition is excluded.
   A pot becoming ready in this transition starts with zero ready turns.
   Six subsequent full ready turns burn it. Removing soup during the sixth
   interaction phase still saves it.
4. Finalize the turn number. Serving during deadline turn T is accepted before
   pending orders with deadline T expire. Serve the earliest-deadline matching
   order, once only; unmatched soup stays in the person's hand.
5. Publish newly arriving orders, record factual events, and check termination.

Task score is `100 × on-time correct orders / fixed total orders`. There are no
negative scores, hidden waste penalties, group multipliers or participant-dependent
changes to order schedules. Counts for burning, expiry, waste and blocking are
auxiliary observations. The internal rescue-trigger count is excluded from the
public score projection.

## Fixed AI behavior

The controller uses shortest paths on the actual walkable grid and one-step
interactions. Taking food out also waits until it is cooked; being adjacent early
does not make removal possible. Arrival, hand freeing, removal and burn deadlines
are compared explicitly, not using a straight-line approximation.

- Protect a pot when finishing the current other job would miss its removal
  window and a timely recovery route exists. If carrying something, choose a
  feasible empty handoff/storage location or usable empty pot to release the hand.
- Persist with that rescue after freeing the hand. Do not immediately pick the
  same prepared portion up again. The commitment ends after the pot is removed.
- Deliver held cooked soup when the handoff is empty. If it is blocked, use an
  empty AI storage counter to free the hand; if both counters are full, preserve
  usable cooked soup and wait for a recoverable public capacity change.
- Remove already cooked food and recover buffered cooked food. If a pot cannot
  be reached in time, the explanation states that limitation, rather than claiming
  that an unavoidable burn can be prevented. Prefer another still-reachable
  ready pot when there is one.
- When choosing between unstarted prepared portions on available counters,
  prefer the earliest currently visible matching deadline, with stable ties.
  Preserve the selected loading stove while that load remains feasible.
- While one pot cooks and another is empty, wait near the handoff for another
  prepared portion when a verified return route still fits the burn window.
  This useful positioning permits genuine simultaneous cooking. With no suitable
  alternative work, wait near the pot or handoff as appropriate.
- Clear unusable ingredients/expired food, clear burnt pots, or discard a held
  prepared ingredient only under the documented unusable-food/capacity-recovery
  conditions. No controller action can operate the human role.

Deadline screening for a new load uses an optimistic public lower bound including
pickup travel, pickup, loading travel, loading, cooking, removal, return,
handoff, human collection, plating and serving. It is not a global optimality
certificate and does not predict the person's future behavior.

## Scenes and calibration

`configs/study_v3_kitchen.json` freezes 144 task scenarios: 24 development seeds
(1000–1023) and 24 disjoint held-out seeds (2000–2023), each with Tasks 1–3.
Task 1 has four orders/140 turns; Task 2 and Task 3 have six orders/160 turns.
The brief's deadline sequences and budgets were retained. Task 3 changes recipes,
order release times and human starting positions. Future releases are held only
in the environment's private schedule and become visible at their actual arrival.

Calibration changed the common controller, never one group's controller or score:

1. A genuine short human delay exposed repeated pick-up/put-down after emergency
   hand freeing. Persisting the rescue commitment removed that oscillation.
2. The first controller waited beside a cooking pot. Fourteen of 48 Task 2
   scenarios had overlapping food production but no instant when both pots were
   still cooking. Safe positioning near the handoff fixed that inefficiency.
   The final strict test requires *both pots to have `status == cooking`*, then
   follows both item IDs to actual successful serving events.
3. The initial partner heuristic completed five of six orders in some Task 3
   scenes. The same positioning fix permits six of six across all 48 scenes;
   deadlines and turn budgets were not extended or changed by seed.

These are development observations. No human participants were recruited, no
human A/B scores were fabricated, and the 50% human-improvement target is **not
validated** by these simulations. A strong human might learn the same public
patterns without asking questions; pilot analysis remains necessary.

## Reproducible checks

Run `python -m unittest tests.test_study_v3_kitchen -v` for mechanism, behavior,
information-boundary, comprehension and full-trajectory tests. The full-run test
starts from every frozen scene, lets the partner control only legal human
actions, and replays every transition with the fixed AI's actual decision.

Run `python -m domains.kitchen.validation` for the proxy screen. Its output files:

- `domains/kitchen/validation_results.json`: 576 runs across all 48 seeds,
  three tasks and four proxies, with scene-level scores and descriptive summaries.
- `domains/kitchen/validation_replays.json`: an actual held-out Task 2 trace with
  simultaneous cooking and successful serving of both pot items; plus a natural
  emergency trace produced by four human waits at turns 9–12. At pre-action turn
  24, continuing the other job requires 9 turns but the first pot's window is
  8; freeing the hand and removing it takes 5 turns if transfers succeed.
  The same controller removes it by turn 29 without burning. No pot, timer,
  order or AI action is injected into these full-run traces.

Proxies are random legal actions, a serial recipe greedy partner, a limited
partner that learns from public completed handoffs, and a coordinated planning
heuristic. The public-history partner does not call AI decisions or explanation
facts; the coordinated partner can predict the fixed AI but cannot alter it.
These proxies test feasibility and coordination space, not human effect sizes.

## Question evidence and understanding tests

`facts` returns separately authorized bilingual statements tied to the actual
selected state, decision, public rules and current events. It includes exact pot
timers, paths, held items, revealed orders, actual next action/reason, supported
comparisons and an explicitly optional human action. The public view never
contains these reasons, policy memory, advisor actions, seeds or hidden orders.
Facts and decisions are pure; they cannot advance time or mutate the true state.
Future-order changes leave decisions, public views, advice and current evidence
unchanged until those orders become public.

`domains/kitchen/qa_cases.json` contains 74 bilingual cases built from genuinely
executed states. The case generator is `python -m domains.kitchen.build_qa_cases`.
Coverage includes current facts, cooking boundaries, actual reasons, discarded
alternatives, mixed questions, wrong premises, short follow-ups, actual transfer
failure, deadline interpretation, a one-turn counterfactual and explicit
clarifications for ambiguous or unbound-history questions. Expected structured
plans are test oracles; replaying them does not establish live semantic-model
accuracy. The shared QA owner runs these against the semantic and evidence layer.

The three fixed comprehension questions concern imminent pot removal, temporarily
storing blocked cooked soup, and clearing a handoff so soup can be delivered.
Their answers are checked against independent small controller states; the shared
server must strip the answer field before rendering the questionnaire.

The demo executes a genuine deterministic trajectory, including a deliberate
legal simultaneous-handoff failure followed by recovery and a successful order.
It supplies all public frames and six bilingual mechanics captions. It does not
show hidden AI priorities. Shared study data must keep the demo separate from
formal task scores, and only the shared server enables questions for an active
A-group Task 2 run.
