# Explicit long evidence callback

Only new finite long owners use this separately source-bound callback. The real
CurrentNNEvidenceProvider still owns installation, epoch consumption, scheduled
programs, failure latch, native event history and typed ProviderEvidence. The
callback v2 preserves the native call order and replaces target materialization
with the lossless new long producer before final binding. It also replaces only
the native Program reader's repeated static proof work with the explicitly
source-bound PreparedProgram. It is a composed
callable passed to PPOUpdateCore, without replacing or patching the native class.

Each validation records the immutable callback start pointer, target storage
provenance and an explicit callback ledger linked to the native event. Owner
checkpoints retain both native provider state and callback state. Raw updates
never construct or invoke this callback. Actual accepted program gradients
continue to come only from native PPO optimizer receipts.

A callback verifies the active newly claimed owner and complete source contract.
The old 20k/recovery owners cannot construct it. No scientific validation
algorithm, canonical scalar NN forward, program qualification or gate changes.

## Explicit prepared-program builder

`PreparedObservedEvidenceBuilder.build(program_ref, observed, targets_ref,
epoch_id=...)` first requires the real native `ObservedPool`, then calls the
unchanged native `pools.checked_targets`. That original validator alone may
perform the scalar-NN replay and issue its private `ReplayedTargets` cache.
The new builder requires that exact native cache type and invokes its original
`verify`; it neither manufactures a pool nor invokes any private seal API.

Programs are explicitly loaded through `PreparedProgram.from_reference` and
cached by path, program bytes SHA and the program's target role hash. Every
reuse calls `verify_binding`; source files include the full native closure,
PreparedProgram and the new long implementation. Native scalar NN targets,
`current_report`, both role-specific eligibility sets, thresholds and gates
are computed/verified anew for each epoch and role.

The builder returns the **original** `ObservedOnlineEvidence`, constructed via
the original `OnlineEvidence`, with the original reference order and payload.
The callback still wraps it in the **original** `ProviderEvidence`; provider
failure latch, epoch consumption, real feedback gradient accounting and loss
normalization are unchanged. Raw cannot construct this callback or builder.
The v2 start header records `program_static_validation`; checkpoint snapshots
record preparation bindings, explicitly without resume authority.

Low-level unit tests name and patch their MOCK provenance classes/functions.
They prove software call ordering and native payload/target equivalence only;
they do not create an audited native pool or original private certificate.
No real 1650-episode observed pool is rebuilt for this engineering test.

Offline real-data checks use ordinary dictionaries from already consumed target
arrays and the original `OnlineEvidence` class. They do not construct
`ObservedOnlineEvidence`, an audited pool, or a private certificate, and do not
authorize use of those arrays for a new optimizer step. A full PPO update is
not rerun; actual accepted gradient/update equivalence remains a separate gate
of the future independent NN/optimizer admission.

## Measured real consumed-call equivalence

Nice10, BLAS/OMP one thread, no cProfile. Existing calls 34, 47 and 58 are
compared for both roles and all four complete target partitions: **1,178,520
role frames**, reusing the same physical pool rather than adding independent
observations. Actor loading, NN forward, environment construction and reset
are all configured to raise if called in the diagnostic process.

| Call / role | Native OnlineEvidence verify | Prepared verify | Native four-pool loss_targets | Prepared four-pool loss_targets |
|---|---:|---:|---:|---:|
| 34 / robot_1 | 2.670 s | 0.241 s | 11.671 s | 2.230 s |
| 34 / robot_2 | 2.090 s | 0.275 s | 9.411 s | 1.972 s |
| 47 / robot_1 | 2.726 s | 0.248 s | 11.867 s | 2.222 s |
| 47 / robot_2 | 2.102 s | 0.235 s | 9.516 s | 1.898 s |
| 58 / robot_1 | 2.702 s | 0.252 s | 11.902 s | 2.329 s |
| 58 / robot_2 | 2.227 s | 0.234 s | 9.748 s | 1.972 s |

All payload fields/types/float.hex, reference order, and every output array's
dtype, shape and bytes match. Every full current_report also matches its
original saved report. Only two immutable programs are cached; all six reports
are recomputed, including the original call47/robot_2 failure and zero qualified
rows. The 24 complete-pool loss_targets calls total **64.115 -> 12.623 seconds**
(5.08x); this is a component measurement, not full training speed.

Cold preparation on call34 costs 7.869/6.404 seconds; subsequent cached program
lookups cost 0.440-0.472 seconds including complete source and field verification.
Current reports remain approximately 2.3-3.3 seconds; no metrics are cached.

Readonly engineering receipt and exact source snapshots:

```text
output/warehouse_v642/statistics-engineering/long-prepared-evidence-bench-20261023-01/benchmark.json
SHA256 cdef3940bbef44e9ecf36ad454a3a0797cbb8bccb50e0bf00d0b5bac8e6b91f9
```

20 software tests cover real native result classes with explicitly named MOCK
low-level provenance, fresh current qualifications, unchanged latch behavior,
raw refusal before preparation/materialization, native target rejection before
preparation, cache/source mutation refusal, original module globals unchanged,
and the complete native/Prepared/long source closure. These are not an end-to-end
scientific ObservedPool or PPO-update equivalence receipt. The unchanged native
target validator and future independent optimizer gate remain mandatory.
