# Automatic explanations at study nodes

Preview implementation: `policylens-three-domain-20260923.v3.13-auto-explanations`.

Only Group A during active Task 2 receives automatic cards. Task 1 is baseline;
Task 3 remains transfer without explanations or access to previous explanations.
Voluntary questions remain available during active Task 2. Game controllers,
scenarios, scores and task transitions are unchanged.

## Node rules

- Warehouse: an actual collision after the completed step; the start of a
  non-progressing AI movement relative to its current path goal (a detour); a
  change into travelling to the charger or charging on the charger. Consecutive
  detour movements toward the same goal and a continuing charging phase do not
  repeat the card. Waiting alone is not labelled a detour. A new collision still
  produces a new card, even if it occurs in a continuing detour/charging episode.
- Pong: when the current selected catch is a team/large ball, show the AI's
  chosen contact and the human's other contact. Deduplicate by ball IDs and
  contact assignment, not by turn. A changed assignment can produce a new card.
  A small-ball detour before a future team catch does not trigger a card yet.
- Kitchen: after completed steps 5, 10, 15, etc., explain the next AI decision.
- Terminal states do not create cards: the active Task 2 explanation window is
  closed immediately when the task ends.

The card is above the board, stays below the header while scrolling, and needs
no question or open-panel click. A new
card stops a held movement key; participants can resume immediately without a
mandatory acknowledgement or fixed reading delay. The latest card remains
labelled with its source turn until replaced. Replay hides it to avoid presenting
current advice as historical advice.

## Grounding and records

Automatic text uses the existing controller's bilingual factual reason directly;
it does not make a DeepSeek call. Collision text explicitly distinguishes the
completed joint action from the next decision. The existing provider-backed
follow-up interface is unchanged. No hidden future schedule is exposed.

`pl3_auto_explanation_settings` records the version at enrollment (including B
controls). `pl3_auto_explanations` records each generated card, its node types,
turn, bilingual text, state/decision hashes and first display acknowledgement.
Both tables are included in the research export, separate from voluntary
`questions`. A browser acknowledgement means at least 50% of the card entered
an active tab's viewport; it does not establish reading or comprehension.

Generation is transactional with game actions. Refresh, duplicate action
requests and repeated acknowledgements cannot duplicate a card or overwrite
its first exposure timestamp. Authorization checks also apply to acknowledgements.

## Rollout

`POLICYLENS_AUTOMATIC_EXPLANATIONS=1` enables this protocol for NEW enrollments.
The default is off. The selected protocol is persisted: changing this setting
never changes an existing participant's condition. Previous compatible releases
can still resume. Do not mix the current on-demand smoke cohort with the new
protocol in the same analysis; retain release and protocol version in exports.

This branch is for local preview. The live Render environment and current
Prolific study have not been changed. No bonus rule has been added. Formal-study
instructions should describe automatic Task 2 explanations and the final bonus
rule before starting that new cohort. Questionnaire wording for new automatic
protocol enrollments refers to explanations, not just answers to questions.

## Reproducing the local preview

Run `python -m tests.automatic_explanations_preview_server`, then open
`http://127.0.0.1:9130/auto-preview/kitchen` (or `pong` / `warehouse`). This
localhost-only helper creates synthetic researcher-preview sessions in a
separate SQLite database, bypasses the baseline for quick UI inspection, and
never contacts Prolific or a language provider. It must not be used for research
data collection. The normal full study flow remains available in the app.

Validation: 146 distinct backend checks passed after updating the export and
manifest expectations and fixing an asynchronous-readiness race in a test.
The automatic-explanation and existing input/replay frontend checks passed.
Real Chromium checks passed for all three games: visible card and exposure
acknowledgement, refresh deduplication, bilingual rendering, sticky positioning,
replay isolation, and Kitchen's next five-step trigger. No production data or
paid participants were used for these checks.
