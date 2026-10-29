# Keyboard play and three-task preview — 2026-10-29

Runtime: `manual-preview-runtime.request-scoped.v2.3`.
Status: local checks passed; production verification pending.

## Reproduced problem

The previous live v2.2 API works for fresh sessions, but the browser could get
stuck after a rejected human action. Holding A at Pong's left boundary sent an
illegal action; Kitchen E without an available station did the same. The server
correctly returned 409 without advancing physics. The old frontend then blocked
its entire queue and allowed subsequent accepted/full messages to obscure the
original error. Dropdown focus, IME key values and unhandled browser-storage
exceptions created additional ways keyboard play could appear broken.

The old manual page also started independent single-task sessions; it did not
provide the requested Task 1 -> Task 2 -> Task 3 progression.

## Changes

- Default three-task manual sequence, explicit terminal/next buttons, per-task
  scores and completion summary. All three sessions retain the exact same model
  binding and selected manual scene seed. No auto-advance or hidden action.
- Start/resume hands focus to a keyboard-focusable canvas. Physical key codes
  handle WASD/Space even with Chinese IME. Pong uses A/D/Space; Kitchen adds E.
- Unavailable human actions are not sent. A previously queued action that becomes
  illegal is explicitly skipped without consuming a turn; later legal inputs
  continue. Only human-input handling changes, never NN action selection.
- Failed requests visibly suspend new input. Reconnect retries the original
  command identity. Restart waits for old in-flight requests and clears their
  failure state before binding a new session. Expired sessions offer restart.
- Local storage is best effort; errors do not prevent live play. Refresh restores
  sequence progress and a pending command. Completed task replay exports are
  combined into one three-task file, without session tokens.
- Single-task/NN-demo mode remains optional by unchecking the sequence control.
  NN demos do not contribute to a three-task human-control run.

## Checks

- 20 Node tests: real app handlers, complete sequence and refresh, IME/focus,
  boundary/queued rejection, unavailable interaction, denied storage, failed
  command recovery and cross-session in-flight failure, terminal-frame completion gating, plus motion/FIFO tests.
- Existing 61 Python tests passed. New 8 Python tests run all three tasks to
  termination for three domains x two variants (18 complete manual episodes),
  including exact NN recomputation at sampled states, per-frame binding/turn/
  source/no-override checks, initial/final independent replay and rejected-action
  recovery. New suite: 38.03 seconds. These are software checks, not task-quality
  acceptance or human study results; only public manual scene seeds are used.
- Native Chrome local Pong: Task 1 reached 60/60, Task 2 reached 90/90, Task 3
  reached 90/90 using physical A and Space keys. All three completed in the actual
  page, including a refresh at Task 2 turn 59 and clicking each Next Task button.
  Three result rows and final completion message were visible, queue zero.

Frozen models/programs, scientific acceptance and physical rules remain unchanged.
Pilot and study entry remain closed; participant storage is untouched.
