# Manual preview runtime v2.1: latency and behavior checks

This runtime accelerates the existing manual website. It does not change model
acceptance, open pilot, consume scientific evaluation seeds, or touch participant
storage. Model package: `policylens-manual-model-preview-20261024.v1`, catalog SHA
`ec7b6370454321748f9767c5cbbce4f630d539afb5763da742f6da6f7f543d43`.
Runtime ID: `manual-preview-runtime.request-scoped.v2.1`.

**Status: speedup deployed; production performance acceptance remains incomplete.**
Final live commit: `785422026a729689366ec782f689c6927faf2d7f`.

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
  HTTP keep-alive, TCP_NODELAY, gzip and Server-Timing expose integrity, inference,
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
| Warehouse | 83.83 / 88.46 | 14.76 / 16.09 | 81.81% |
| Pong | 117.17 / 127.03 | 13.31 / 13.76 | 89.17% |
| Kitchen | 104.46 / 107.49 | 7.36 / 7.89 | 92.66% |

Every individual role-pair/task also passes P95 <=80ms and >=70% reduction.
Per warm request, full file reads fall from Warehouse 289 / Pong 406 / Kitchen
460 to **28** (27 assets plus catalog), each file read once. Bytes fall from
approximately 114 / 163 / 162 MB to **10,533,741 bytes**. Role NN forwards fall
from 6 / 8 / 4 to **2 total (one per role)**. These are server operation timings;
network latency is measured separately and is not represented by these values.

Evidence: [18-case benchmark](evidence/manual-preview-latency-local-final-20261024.json).
Raw 540-sample reports remain under ignored `output/manual_preview_latency/`.
Reproduce using `scripts/benchmark_manual_preview_latency.py`; for baseline,
extract the pinned original server with `git show` and use `--baseline`.

## Correctness gates

- Python: 61 tests passed (full six-model/three-task comparisons, unchanged frozen
  scientific bodies, restore/replay, counterfactuals, no Torch production import,
  tamper/missing/replacement/read-only rejection and export concurrency).
- Node: five tests passed including actual app event handlers with delayed HTTP,
  FIFO burst/overflow, one held repeat and keyup cancellation, retry identity,
  pause/terminal clearing and immediate DOM feedback. The DOM test's <50ms
  assertion measures synchronous feedback, not browser paint or network time.
- Local real HTTP: six models, 67 API requests, exact actions/physics/bindings,
  idempotent retries, restore, independent full replay; all passed.

## Browser feedback measurement

Chrome native keyboard input on the local HTTP preview: 34 accepted discrete
inputs, 34 confirmed turns, input-to-next-animation-frame P95 9ms. An 800ms
held key produced five steps and released with zero pending repeats. This is
frame callback latency, not a claim of measured hardware photon latency. The
screenshot and raw benchmark artifacts remain under ignored output storage.

## Deployment

Retain previous live Render deployment `dep-dbe2i4gm7kps73c4c2og` at `3b2f46b`
for rollback. Same free Render service and start command. Production evidence and
same-route warm health/action results are recorded after deploying this runtime.


## Production measurements and remaining gate

The complete original live baseline has 30 paired warm health/action samples:
health P95 1305.39ms, action P95 3418.30ms. Initial connection/wake 73080ms is
excluded from the warm sample. The first accelerated deployment (v2) still failed
the same-route margin: health 437.83ms, action 1187.29ms. Its evidence is retained.

After reducing path metadata work and enabling gzip, commit `541ca69` has 30
paired samples: health P95 **947.96ms**, action P95 **918.03ms**, satisfying
`action P95 <= health P95 + 150ms`. Action P95 is 73.14% below the original live
baseline, but this is a same-route observation with variable network conditions,
not an isolated causal estimate of server CPU improvement. Action bodies are
1169–3560 compressed bytes. Six models / 67 API calls again passed complete
binding, chosen action, physics, restoration and independent replay comparisons.

**Production server-internal wall time still fails the <=80ms gate.** Per-domain
P95 among the 10 measured actions is Warehouse 311.567ms, Pong 304.490ms, Kitchen
139.107ms. Do not substitute the controlled local benchmark for these production
numbers or declare all performance gates passed. CPU-time and cgroup quota
observations are added in the final diagnostic deployment to separate actual
computation from scheduling/I/O waits. No model, acceptance threshold, physical
rule, scientific evidence, or participant database was changed to get speed.

Evidence: [original production baseline](evidence/manual-preview-latency-production-before-20261024.json),
[v2 unsuccessful margin](evidence/manual-preview-latency-production-v2-20261024.json),
[v2.1 production sample](evidence/manual-preview-latency-production-v21-20261024.json),
[v2.1 complete HTTP parity](evidence/manual-preview-http-v21-20261024.json).


## Final diagnostic deployment (not a performance pass)

Final deployment: `dep-dbe3hmnf3r2c73cjfna0`, commit
`785422026a729689366ec782f689c6927faf2d7f`. Render UI confirms success; original
`3b2f46b` deployment remains available for rollback. Auto-Deploy is Off (observed,
not changed), so documentation commits do not alter the pinned live runtime.

The service's `/sys/fs/cgroup/cpu.max` reports **15000 / 100000 microseconds**,
i.e. a 0.15-CPU quota. Final independent 30-pair warm measurement:

| Domain | Server wall P95 ms (10 actions) | Thread CPU P95 ms | 80ms wall gate |
|---|---:|---:|---|
| Warehouse | 296.856 | 45.741 | fail |
| Pong | 205.504 | 36.468 | fail |
| Kitchen | 68.719 | 21.184 | pass in this sample |

Global 30-action server wall P95: 209.992ms. CPU time is non-additive to
Server-Timing wall categories. Large wall/CPU gaps and a measured 15ms/100ms CPU
quota show that scheduling waits are material; I/O can also contribute. These
observations do not prove that no further optimization is possible.

Final same-route health P95 **919.29ms**, action P95
**1076.11ms**. The action margin exceeds the specified health+150ms
limit by **6.82ms**, so this last sample
fails the margin. The earlier successful 30-pair sample is retained; it is not
selected in preference to this failure. Compared with the original 3418.30ms
action P95, the final measured value is lower by
68.52%, with network variability noted.

Final source also passes 61 Python and five Node tests, and six-model / 67-request
live HTTP equivalence, restore, idempotency and replay checks. No final scientific
heldout datasets were run. The original performance thresholds remain in force;
**all_performance_gates_passed=false**. The free service is kept as the faster
manual preview, with pilot still closed and no new model acceptance claim.

Evidence: [final performance status](evidence/manual-preview-speedup-status-20261024.json),
[final 30-pair production sample](evidence/manual-preview-latency-production-final-20261024.json),
[final complete HTTP replay comparison](evidence/manual-preview-http-final-20261024.json).


Final Chrome check at the deployed source: eight quickly accepted keypresses
showed seven pending while the first request was in flight; all eight confirmed
in order, with zero pending inputs and 8ms input-to-frame callback P95. Controls
remained enabled while awaiting the server. The actual browser screenshot is
retained locally and SHA-bound in the [browser observation](evidence/manual-preview-browser-final-20261024.json).
