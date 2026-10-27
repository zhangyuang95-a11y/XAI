# Confirmed-frame motion update — 2026-10-27

Status: local checks passed; deployment verification pending.
Runtime version: `manual-preview-runtime.request-scoped.v2.2`.

The user requested the fluid interaction of the earlier ICLR website. The exact
historical URL has been requested but is not yet identified. The checked-in
September 10 alignment deployment records 380ms continuous transitions;
`ui/web/app.js` uses 400ms. This update uses 380ms as the provisional reference.
It does not claim measured equivalence to an unidentified historical deployment.

## Change

Previously each response called `StudyBoard.stop()`, potentially snapping an
unfinished 120ms animation to its endpoint before starting the next move.
The preview now plays each acknowledged joint frame in order for 380ms,
starting its clock on the first animation frame. The following request can run
while the current animation plays. Backpressure limits presentation to at most
one active plus one waiting frame; command IDs and expected turns continue to
use acknowledged server state. No unsubmitted action is sent or predicted.

Sidebar score, last action and orders present the same frame as the board.
Unchanged orders and model hashes are not rebuilt. New-session/reset presentation
cancels old visual callbacks and releases backpressure. Pause clears unsent input;
already acknowledged physical transitions remain visible. Reduced motion and
restoration show confirmed endpoints without interpolation.

## Checks

- Eight Node input/animation tests passed: rapid responses preserve every endpoint,
  380ms transitions finish before the next starts, presentation backpressure,
  session replacement, reduced motion, FIFO/overflow, retries and hold release.
- 31 Python preview and real HTTP comparison tests passed, including six models,
  state/action equivalence, restoration, idempotence and replay.
- Native Chrome local Warehouse: eight discrete inputs -> eight confirmed turns,
  zero pending inputs; input-to-next-animation-frame P95 8ms (eight samples),
  AI action overrides zero. This is feedback latency, not network or photon latency.

All scientific models, program assets, environment rules and original study
runtime are unchanged. Only preview presentation and the runtime version label
change. Pilot remains closed. Free-host cold wakeup, network delay and the earlier
unmet production performance gates are not declared resolved by longer animation.
