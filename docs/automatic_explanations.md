# Guided questions and optional explanations

Current release: `policylens-three-domain-20260923.v3.18-ai-question-button`.

During explanation-enabled Task 2, a keyboard-accessible question-mark button follows the AI actor in all three games. It focuses the question panel (or the first-node guided question) without advancing the game. It is absent when questions are unavailable. The first-node protocol remains unchanged.
New automatic-explanation enrollments use `first-node-question-guide-v4`.

Only Group A receives this guide. Task 2 starts normally at turn 0. No extra
start-of-task explanation is generated: the first existing warehouse event,
Pong team-ball event, or kitchen five-step event becomes the guided node.
The ordinary right-hand question panel is highlighted and game actions pause.
The participant selects the example “Why are you making this decision?”, then
presses **Send question**, reads the controller-grounded answer in that panel,
and confirms to continue. This happens once per Task 2 run. It teaches the
question-panel interaction, without adding an artificial trigger or game step.

Later nodes offer **I understand — continue** or **Show me why**. The first
choice resumes directly; the second requests an answer that must be confirmed.
The guide uses a predefined question with a deterministic factual answer, kept
separate from free-text question records. Both normal study and friend-demo
pages retain the right-hand question form. The demo enables free-text questions
only when its own `POLICYLENS_LLM_API_KEY` is supplied (optional base URL/model
overrides; defaults are DeepSeek/deepseek-chat). Without that configuration,
free-text input is visibly unavailable while guided questions remain usable.
The demo still ignores production database, Prolific, and administrator settings.

The server enforces the first-question requirement and later response gates.
Requests, responses, and confirmation survive reconnects. The existing
`auto_explanations.content_json` export records `onboarding`, `requested_at`,
`question_id`, the question text/language as selected, `response`, and
`responded_at`. Responses distinguish `read_explanation` (explicit confirmation
of the displayed answer) from `self_reported_understood`. Neither establishes
actual comprehension. Skipping never marks an answer as displayed. Guided
selections remain separate from voluntary `questions` records.

Existing v1/v2/v3 enrollments keep their persisted protocol. Task 1, Task 3, and
Group B do not receive guided questions. Game rules and scores are unchanged.
The formal study and independent friend-demo services use the current release.
The previous Prolific cohort remains archived under its original release IDs;
new formal enrollment is disabled until the next recruitment round is configured.

Validation: backend tests cover all three domains, first-node action/skip blocking,
ownership, answer withholding, idempotent requests, reconnects, bilingual
answers, later skip/request choices, separate exposure records, exports, and
Task 3 isolation. The original v2 suite is explicitly pinned to its protocol.
The browser check covers all three initial questions, keyboard/Escape blocking,
refresh after requesting, both subsequent kitchen choices, session isolation,
and completing the demo.

## Previous automatic-confirmation protocol (v2 reference)

Preview implementation: `policylens-three-domain-20260923.v3.14-confirm-explanations`.

Only Group A during active Task 2 receives automatic cards. Task 1 is baseline;
Task 3 remains transfer without explanations or access to previous explanations.
Voluntary questions remain available during active Task 2. Game controllers,
scenarios, scores and task transitions are unchanged.

## Node rules

- Warehouse: an actual collision after the completed step; the start of a
  non-progressing AI movement relative to its current path goal (a detour); the
  first decision to travel to or use the charger in Task 2. Later arrivals,
  charging phases and repeat charging trips do not produce another charge prompt. Consecutive
  detour movements toward the same goal do not repeat the card. The first-charge
  limit is stored with the run and survives refresh and restart. Waiting alone is not labelled a detour. A new collision still
  produces a new card, even if it occurs in a continuing detour/charging episode.
- Pong: when the current selected catch is a team/large ball, show the AI's
  chosen contact and the human's other contact. Deduplicate by ball IDs and
  contact assignment, not by turn. A changed assignment can produce a new card.
  A small-ball detour before a future team catch does not trigger a card yet.
- Kitchen: after completed steps 5, 10, 15, etc., explain the next AI decision.
- Terminal states do not create cards: the active Task 2 explanation window is
  closed immediately when the task ends.

For new enrollments, the explanation appears in a speech-bubble dialog anchored
to AI teammate 2. In Pong, the explanation occupies the right sidebar below the controls, with a
connector routed around the court to the AI paddle. The court and player legend
remain visible beside the explanation. Narrow phone screens stack the sidebar
below the court. The game cannot advance until the
participant selects **Confirm and continue**. Keyboard movement, held movement,
outside clicks and Escape cannot bypass the dialog. Native modal focus trapping
keeps background controls inaccessible. Confirmation closes the dialog without
playing a turn. A failed confirmation leaves the dialog open for retry.

The server also rejects game actions while a confirmation is outstanding. Both
refresh and reconnect restore the pending dialog. Successful confirmation is
persisted, idempotent and scoped to the authenticated participant/current run.
Old `event-nodes-v1` enrollments keep their original nonblocking card protocol;
new enrollments use `event-nodes-confirm-v2`.

## Grounding and records

Automatic text uses the existing controller's bilingual factual reason directly;
it does not make a DeepSeek call. Collision text explicitly distinguishes the
completed joint action from the next decision. The existing provider-backed
follow-up interface is unchanged. No hidden future schedule is exposed.

`pl3_auto_explanation_settings` records the version at enrollment (including B
controls). `pl3_auto_explanations` records each generated card, its node types,
turn, bilingual text, state/decision hashes and first display acknowledgement.
`pl3_auto_explanation_confirmations` separately records explicit confirmation
timestamps. All three tables are included in the research export, separate from voluntary
`questions`. A browser acknowledgement means at least 50% of the card entered
an active tab's viewport; it does not establish reading or comprehension.

Generation is transactional with game actions. Refresh, duplicate action
requests and repeated acknowledgements cannot duplicate a card or overwrite
its first exposure or confirmation timestamp. Confirmation is evidence of an
explicit button action, not proof of comprehension. Authorization checks also apply to acknowledgements.

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
localhost-only helper creates synthetic researcher-preview sessions under one browser identity so all three game tabs can remain open together in a
separate SQLite database, bypasses the baseline for quick UI inspection, and
never contacts Prolific or a language provider. It must not be used for research
data collection. The normal full study flow remains available in the app.

Validation for the confirmation revision is recorded separately from the prior
nonblocking-card checks. Backend tests cover the server action gate, visible vs
confirmed distinction, duplicate confirmations, ownership, reconnects, next-node
blocking, Task 3 isolation, exports and old-protocol compatibility. Real-browser
checks cover the modal, keyboard/Escape blocking, direct-API rejection, refresh,
failed confirmation/retry and continuing without advancing a turn on confirmation.

Confirmation revision verification: 57 backend tests passed, both frontend
regression scripts passed, and all three real Chromium game previews passed.
Screenshots were inspected for speech-bubble placement and readable reasons.
The active production deployment and Prolific cohort were not changed.
