# Independent second-revision review

Date: 2026-09-20. Scope: the rolling Pong controller and schedule, shared question evidence and counterfactual boundary, and browser input/replay/animation code for the proposed `policylens-three-domain-20260920.v3` release. This review did not change the domain physics or historical Warehouse controller.

## Findings corrected

1. **A slow replay response could replace a newer selection or reopen replay after Return to current.** Historical-frame requests had neither a sequence fence nor a study/revision check. The latest request now wins; Return to current cancels older requests, and a changed session/revision invalidates an outstanding response. Pending replay loads also disable movement and asking, so the displayed frame cannot change underneath a new action/question.
2. **A frame selection or example click could interfere with an in-progress question.** Historical-frame loading now stops while a question is being answered. Example buttons preserve the current draft while actions, animations, frame loads, or answers are pending. Returned answers continue to retain their recorded turn binding and server permission checks.
3. **Kitchen status labels exposed ambiguous container stages.** Stove identifiers display a number, empty stoves say “Empty pan” / “空锅,” and fallback labels distinguish cooked protein on a temporary plate, finished dishes in the output container, and formal serving plates. Valid station prompts carry the `E:` key prefix. Domain-provided item names take precedence over fallback labels.

## Checks

- `tests/revision_frontend_review.cjs` executes the actual application functions with a controlled asynchronous network and small DOM stubs. It reproduces out-of-order replay responses, cancellation, cross-window revision changes, draft locking, direction changes while an action is unacknowledged, held-key auto-repeat, and Space edge behavior. It checks that these cases neither submit extra actions nor restore stale question permissions.
- `tests/test_study_v3_revision_review.py` also changes the identities and contact lanes of all undisclosed future Pong balls in three separately selected recorded situations. The actual AI decision, explanation evidence, and participant-visible counterfactual output remain unchanged. Counterfactuals stop at the first new-ball boundary; the private input-state audit hash is intentionally different.
- `python3 -m pytest -q tests/test_study_v3_pong.py tests/test_study_v3_revision_review.py`: **33 passed**.
- JavaScript syntax checks for both `study_v3/web/app.js` and `study_v3/web/board.js`: passed.

The Pong review found no additional release-blocking defect in visible-state planning, nearest-ball commitment retention, later tentative assignments, two-ball reachability checks, the finite rolling supply, or future-ball filtering. The candidate controller and counterfactual runner do not read undisclosed ball contacts before they become visible.

These are deterministic tests and source review, including controlled-network UI unit checks. They are **not a real-browser acceptance run, a Render deployment receipt, a measurement of semantic question understanding, or human performance evidence**. The deployment owner must report those validations separately. The 50% Task 2 human improvement remains an unmeasured pilot target.

## Follow-up Kitchen review

The independent read review covered `domains/kitchen/engine.py`: role restrictions, facing and front-cell interactions, pre-turn handoff conflicts, single-item hands/counters, ingredient lineage through mixing, order binding/expiry, burning at the eighth additional ready turn, actual counter visits, and explanation evidence.

Findings sent to the Kitchen owner and independently rechecked after correction:

- **Recovering a raw handoff:** a human could legally deliver a raw egg on turn 8. Previously, the advisory policy fetched another ingredient and then waited through all six expiries without reclaiming the raw blocker. Recovery now names the blocker, retrieves and prepares it, then resumes cooperation. Independent replay with each of the four raw ingredients delivered first completed all six orders, with zero discarded components, in 323 / 324 / 298 / 310 turns respectively. These are fixed-controller proxy trajectories, not human outcomes.
- **Locating ingredients and choosing the next input:** generic pan phase text did not identify which pan contained eggs, and a one-step movement suggestion could not directly answer which ingredient to prepare. New facts explicitly identify per-ingredient locations, pan contents/recipe/order/constituents, missing visible ingredients, and the next input portion. The input goal is labeled separately from an immediate movement. A ready egg still inside its pan is no longer described as already sitting on a temporary plate.
- **Terminal historical-frame wording:** terminal Kitchen frames previously offered a next wait even though all actions were closed. They now explicitly state that no executable next action or new input exists. This matters because an active Task 2 may ask about the preceding task's terminal recorded frame.

An independent conservation check observed every original portion across 80 trajectories with occasional legal human mistakes (27,928 transitions before the recovery revision): each portion either remained at one physical location or had a recorded serve/discard/clear event. Burnt food was included until actual removal. Six durable regression trajectories retain this invariant in `tests/test_study_v3_revision_review.py` and pass after the correction.

Final Kitchen and independent regression run: **57 passed, 152 subtests passed** using `python3 -m pytest -q tests/test_study_v3_kitchen.py tests/test_study_v3_revision_review.py`.

The reviewed cooperative calibration overlaps recipes across two pans, but does not heat both pans simultaneously. Its documentation and metrics distinguish `parallel_recipe_turns` from `parallel_cooking_turns`; no simultaneous-heating success is inferred. Revised free-question evidence still requires actual provider spot checks, independently of these deterministic validations.
