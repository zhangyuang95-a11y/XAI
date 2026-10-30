# Public game chooser and step protection — 2026-10-30

The public root, `/study/`, and domain aliases serve the three-game chooser.
No access key, enrollment identifier, model version or model selection is shown.
Choosing a card immediately starts a regularized frozen-NN teammate game with
Task 1 → Task 2 → Task 3. Exiting waits for the current acknowledged frame and
its complete animation, then returns to the chooser. Server records are retained;
refresh offers explicit resume instead of automatically entering the last game.

This is the public manual playground. The protected researcher protocol remains
separate at `/research/`; its enrollment API still requires operator authorization.
No research consent, scientific admission or recruitment authorization is fabricated.
Original participant storage, all six Actor assets, physics, sampling, and acceptance
receipts remain unchanged. Model identities remain in records and API metadata;
the public page omits their technical presentation.

## Input behavior

A manual action is accepted only with no active request or presentation. Both the
HTTP acknowledgement and the actual `PreviewBoard.setState()` completion promise
must resolve before another action is accepted. The previous `board.ready()` method
was only a visual backlog threshold and was not a full-animation completion signal.
Buttons are disabled while a step is pending; keyboard attempts get an explicit
waiting message and are not queued. Browser repeat and custom held-key repetition
are disabled: the player releases and presses again for each step. No arbitrary
extra delay or AI action prediction is introduced. Failed requests retain the original
command identity, including after exit and explicit resume.

## Verification before deployment

- 39 Python checks: isolated routing, public no-key start, protected research auth,
  fixed bindings, three tasks, and unchanged exact frozen inference/physics.
- 36 Node checks: acknowledgement versus animation gating, rapid input, key holding,
  exit/switch races, uncertain retries, Task 1–3, and separate research entry.
- Independent read-only review reported no remaining actionable findings.
- Actual local browser: one card starts Warehouse without a form; holding Space
  for 650ms produces exactly one turn; exit returns to all three cards; Pong then
  starts its own turn 0 with the correct physical controls.

Deployment and live verification are recorded separately. These are software checks,
not new training, quality evaluation, heldout use, or human study results.
