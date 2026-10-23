# Explicit external cutover of the pinned v2 worker

Preparation and assertion never signal a process. Execute requires a new
source-bound plan, exact original v2 claim/PID/parent/argv, explicit human
authorization and a concrete v3 source contract (current file SHA and read-only
snapshot pointers). No `ready: true` or `passed` flag grants permission.

Two modes are supported. `checkpoint-18600` watches for the regularized completed
93% refit and full 18600 checkpoint, then stops and verifies that this is still
the last owner event. Any later reserve/reset/update tail resumes the worker and
refuses cutover. `original-safe-replay` requires the exact current eight-event
v2 prefix: no fresh physical collection/resets, the sole accepted optimizer
receipt is the byte-value-identical replay of the old 17824 commit, and the only
pending batch is the original cached 776 frames from 17824 to 18600. It may discard
only uncommitted numerical PPO calculations; all v2 files remain preserved.

Execute sends SIGSTOP, waits for an actual stopped state, and rechecks the full
state/core/reader/provider/RNG checkpoint, all cursors and journals. On any failed
check before termination it sends SIGCONT and refuses. A read-only external
prestop receipt binds all logs, checkpoint, worker and old supervisor source.
Only then does it send SIGTERM and SIGCONT. Completion requires the original
supervisor's actual `Popen.wait()` receipt with exit -15 and no successful result,
and unchanged verified logs/checkpoint. PID disappearance or a file alone is
not an exit observation. The cutover marks neither model success nor scientific
failure, writes no success into old claims, and deletes nothing.

No signal handler, source or runtime patch is installed in the old worker. The
original finite lease is unchanged; all polling stops at its deadline. A new v3
controller must separately authorize continuation, bind immutable old+v2 prefixes,
restore complete state and perform required genuine physical audits. This helper
grants no NN, environment, seed, training, model-admission or release authority.

## API

- `prepare(recovery_plan_ref=..., supervisor_started_ref=..., worker_pid=...,
  target_checkpoint_path=..., mode=..., cutover_id=...)` writes only a new
  external plan/source snapshot. Modes are `original-safe-replay` and
  `checkpoint-18600`.
- `assert_safe(plan_ref)` reads and verifies the selected safe boundary; it
  sends no signals and initializes no NN, RNG or environment.
- `execute(plan_ref, authorization='继续，完成验收',
  successor_source_contract_ref=..., worker_pid=..., target_checkpoint_path=...,
  supervisor_started_ref=...)` requires all explicit values to match the plan.
- `read_completed(ref)` independently rechecks the stopped boundary and real
  terminal receipt. It returns `parent_v2_plan`, `parent_v2_claim`,
  `actual_supervisor_exit`, `source_contract`, `inputs`, and `mode`.

Successor contract version is
`warehouse-v642-accelerated-recovery-source-contract.v1`, with a nonempty exact
`source_files` relative-path SHA map, `source_sha256=digest(source_files)`, matching
`source_snapshots` read-only pointers and `training_authority=false`. It is source
readiness evidence only. Old logs are sealed by permission-only changes after
the actual terminal wait and a byte/SHA recheck; old content and claims remain.

The new tests use fake process/signal/wait control and tiny synthetic tensor
checkpoints. An additional read-only check confirmed the real pinned v2 process
identity and original 16800 tensor/core/reader/provider fingerprint without
sending a signal, constructing a model/environment, or drawing a new seed.
No execute or cutover of the actual process was performed during implementation.
