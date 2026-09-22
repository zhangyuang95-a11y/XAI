# Kitchen tutorial presentation update v3.8.2

Release: `policylens-three-domain-20260922.v3.8.2`.
Tutorial: `kitchen-operations-tutorial.v2`.
Target: https://policylens-warehouse-study.onrender.com/.
Status: local implementation verified; production receipt pending.

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

Production verification and exact source will be recorded after deployment.
