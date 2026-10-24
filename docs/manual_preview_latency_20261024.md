# Manual preview runtime v2: latency and behavior checks

This runtime accelerates the existing manual website. It does not change model
acceptance, open pilot, consume scientific evaluation seeds, or touch participant
storage. Model package: `policylens-manual-model-preview-20261024.v1`, catalog SHA
`ec7b6370454321748f9767c5cbbce4f630d539afb5763da742f6da6f7f543d43`.
Runtime ID: `manual-preview-runtime.request-scoped.v2`.

## Changes

- Request-local verification of the complete immutable catalog dependency closure;
  every new model request rehashes current bytes and checks read-only paths.
  No mtime/size-only exemption. Frozen receipts are validated on preload and the
  validation is reused by their immutable binding/content identity.
- Request-local, bounded NN decision reuse keyed by complete frame and binding
  SHA. Pong reuses the same scores for its existing physical-mask argmax and
  stable tie break instead of evaluating its network twice.
- One lock per session; SQLite locks cover reads/commits. Export takes an immutable
  prefix snapshot then replays it outside both locks.
- Six paired models preload; numerical libraries use one thread before import.
  HTTP keep-alive, TCP_NODELAY and Server-Timing expose integrity, inference,
  physics, storage, serialization and operation total milliseconds.
- Eight pending actions, one HTTP command in flight; explicit overflow notice;
  held keys have at most one unsent repeat which keyup removes. Retries preserve
  command identity/turn, including pending-command recovery after refresh.
- Pause/model/new-game/terminal clears pending input explicitly. Confirmed-frame
  animation is 120ms; NN playback schedules to a 300ms cadence after completion.
  Binding DOM only changes with identity. A delayed-start notice describes free
  instance waking; no speculative model movement is displayed.

## Same-machine warm benchmark

Baseline source is deployed commit `3b2f46b24a5a0ab755156c008e91f80099a172f6`.
Python 3.13 / NumPy 2.2.3, numerical libraries one thread in both processes;
SQLite in memory; public manual scene per domain; both variants, all three tasks,
30 warm actions per combination: **540 measured actions per implementation**.
Public state, action, score and complete frame/binding digests agree in all 18
cases (31 compared transitions each including warmup). Additional adapter tests
compare complete NN distributions/scores, private transitions and counterfactuals.

| Domain (180 warm actions each) | Old P50 / P95 ms | v2 P50 / P95 ms | P95 reduction |
|---|---:|---:|---:|
| Warehouse | 83.83 / 88.46 | 15.80 / 17.27 | 80.47% |
| Pong | 117.17 / 127.03 | 15.31 / 15.85 | 87.52% |
| Kitchen | 104.46 / 107.49 | 8.26 / 8.71 | 91.90% |

Every individual role-pair/task also passes P95 <=80ms and >=70% reduction.
Per warm request, full file reads fall from Warehouse 289 / Pong 406 / Kitchen
460 to **28** (27 assets plus catalog), each file read once. Bytes fall from
approximately 114 / 163 / 162 MB to **10,533,741 bytes**. Role NN forwards fall
from 6 / 8 / 4 to **2 total (one per role)**. These are server operation timings;
network latency is measured separately and is not represented by these values.

Evidence: [18-case benchmark](evidence/manual-preview-latency-local-20261024.json).
Raw 540-sample reports remain under ignored `output/manual_preview_latency/`.
Reproduce using `scripts/benchmark_manual_preview_latency.py`; for baseline,
extract the pinned original server with `git show` and use `--baseline`.

## Correctness gates

- Python: 56 tests passed (full six-model/three-task comparisons, unchanged frozen
  scientific bodies, restore/replay, counterfactuals, no Torch production import,
  tamper/missing/replacement/read-only rejection and export concurrency).
- Node: five tests passed including actual app event handlers with delayed HTTP,
  FIFO burst/overflow, one held repeat and keyup cancellation, retry identity,
  pause/terminal clearing and immediate DOM feedback. The DOM test's <50ms
  assertion measures synchronous feedback, not browser paint or network time.
- Local real HTTP: six models, 67 API requests, exact actions/physics/bindings,
  idempotent retries, restore, independent full replay; all passed.

## Deployment

Retain previous live Render deployment `dep-dbe2i4gm7kps73c4c2og` at `3b2f46b`
for rollback. Same free Render service and start command. Production evidence and
same-route warm health/action results are recorded after deploying this runtime.
