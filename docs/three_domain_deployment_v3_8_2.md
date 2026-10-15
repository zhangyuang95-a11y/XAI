# Kitchen tutorial presentation update v3.8.2

Release: `policylens-three-domain-20260922.v3.8.2`.
Tutorial: `kitchen-operations-tutorial.v2`.
Target: https://policylens-warehouse-study.onrender.com/.
Status: Live; local and production acceptance complete.

The Kitchen tutorial now has six sections. Disposal of the spoiled practice
portion completes the final section; the removed seventh menu/scoring reading
section no longer requires a space press. A clear completed message leads to
Task 1. The dedicated tutorial menu panel is removed; normal task menus and
expandable public rules remain available.

Tutorial goals are larger, darker and bolder (20px, 19px on smaller screens),
with 16px contextual guidance and feedback, clearer section labels and controls.
The visual changes are scoped to Kitchen practice. English and Chinese were
visually checked in the browser.

Formal game rules, scoring and stored task states are unchanged. Compatible
v3.8 and v3.8.1 sessions retain their release IDs. For an existing tutorial at
the removed seventh section, a read-only view reports completed 6/6. Starting
Task 1 records the original tutorial snapshot and its updated version in an
append-only tutorial event, without consuming a formal turn. Earlier practice
sections retain their physical state and progress. Existing completed research
records are never rewritten by this presentation update.

Validation covers real six-section completion, old seventh-section recovery,
read-only historical preservation, recorded completion, initial Task 1 state,
permission/version compatibility and keyboard/animation controls. The targeted
Python cases passed (92 unique cases across the main run and corrected focused
rerun), as did three existing frontend regression checks. Gameplay simulations
were not repeated because the engine and rule configuration are unchanged.

## Production receipt

Render deployment: `dep-daouug942hec7389itug` (Live, 1m31s).
Commit: `62d6818f191a63475dbfc4c14f01ed6aa17e63ca`.
Source SHA-256: `f2d802e396c5f41c873a4291058f57098916e0b0f4940d4959e1645707ce4cda`.
The public release endpoint confirms v3.8.2, tutorial v2, durable storage and
study readiness. Deployed JavaScript and CSS match the verified files exactly
and are served with `no-store, private` cache control.

One synthetic test session physically reached the old seventh section before
deployment. After deployment it restored as completed 6/6 with identical game
state and revision, then entered Task 1 at turn 0 and score 0. All 31 earlier
practice events remained unchanged; the old seventh-section snapshot was
retained in the recorded v2 transition. A fresh test session completed exactly
six sections, ending with an actual spoiled-food disposal, and entered Task 1
at turn 0 and score 0. Both correctly had no Task 1 Q&A permission. Export
confirmed one initial formal frame per run and zero questions. There were 79
successful HTTP requests across the before/after checks. No test data was
classified as human-study participation.

Receipts are under `analysis/kitchen_v382_20260922/` in the parent workspace,
including `live_release.json`, `live_assets.json` and
`production_acceptance/tutorial_acceptance_summary.json`.
