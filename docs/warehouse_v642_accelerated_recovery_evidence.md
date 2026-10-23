# Explicit accelerated recovery evidence

New source only; the original 407 scientific files, six recovery files and running
v2 process remain unchanged. Every distinct bound file is actually rehashed on
each verification. Only repeated references to the same canonical path and SHA
are de-duplicated; conflicts are rejected before reads. Native event, epoch,
evidence, failure-latch and optimizer gates remain mandatory.

## Measured duplicate work

The complete immutable native prefix contains 61 events / 59 validations.
Its original per-event asset loops make **407360** hash requests for only
**6489** distinct canonical paths: **118304978600 bytes** requested versus
**5727263249 bytes** when each distinct file is fully hashed once. Identical
bytes in two different files are still hashed twice; same path / conflicting
SHA is rejected before any payload read.

Nice10, no cProfile, no ObservedPool construction, NN forward or reset:

| Complete event-control verification | Wall time |
|---|---:|
| Original `_control(check_assets=True)` | 66.912 s |
| Same original `_control(False)` + complete unique actual-file SHA checks | 5.427 s |

Control values, exact types and float.hex are identical (12.33x component
speedup). No stat cache or incomplete asset subset is used. Native event,
target-NN and role-evidence checks outside the asset loop still run.

```text
output/warehouse_v642/statistics-engineering/unique-event-control-bench-20261023-01/benchmark.json
SHA256 100cacdbf2c837f99bca9dc46be99f30a831d72f817e3d5f91edb92aba2a693c
```

This explains a major repeated component, not a measured complete accelerated
provider call. That actual timing must come from the new controller's first
call with its genuinely replay-audited old ObservedPool; it is not fabricated
using a mock pool.

## Factories and immutable prefixes

Both factory signatures are explicit and accept `snapshot, state` plus:
`reference_directory, owner_ref, observed, output, recovery_claim_ref,
prefix_manifest_refs, config=PPOConfig(), source_guard=None`.

- `restore_original_prefix` calls the **actual frozen**
  `RecoveryEvidenceProvider.restore_prefix` once, then copies its complete
  attributes to a distinct new adapter. Native original events, the complete
  old target facts, partial call59 and failure latch remain intact. Cached
  calls32–58 recompute current qualifications and must exactly reproduce their
  original payloads before any accepted update can proceed.
- `restore_continuation` accepts a genuine complete v2 checkpoint at the final
  93% refit only. It preserves all committed control/NN/optimizer state and
  starts an independent suffix without replaying already committed steps.

The original407 and recovery-v2 manifests use
`warehouse-v642-v3-immutable-prefix-manifest.v1`, include all canonical roots
and file hashes, source snapshots and the real v2 supervisor terminal
observation. Missing files, changed bytes, symlinks, conflicting hashes,
unknown schemas, extra tree files and post-checkpoint calls/events refuse
execution. The prior unknown native exit is not turned into a fake success.

Every verify still executes the original event/epoch/evidence/latch logic and
actually hashes **every distinct** prefix/suffix file. Directory inventories
are compared too. The separate acceleration-started receipt binds all new
sources plus original six, the new claim, immutable prefix manifests and the
prior checkpoint; old files/claims are never overwritten.

## Static computation and NN boundary

Program reading uses genuine `PreparedProgram.from_reference`; current reports
use the separately verified exact static-metrics cache. Program/parent/source
and complete x/m/rows bindings are rechecked; current NN probabilities, samples,
agreements, KL, both qualifications and gates are recalculated each call. The
original five-action masks, sampling sequence, PPO order and loss normalization
remain unchanged. Original OnlineEvidence/ObservedOnlineEvidence/ProviderEvidence
classes, payloads and latch rules are retained. Old committed payload comparison
additionally checks exact types and float.hex.

For new target packages only, an explicitly supplied live `source_guard`
enables the new canonical producer. Its source closure and provenance are
bound, and provenance is added to the new native event's assets. The unchanged
native `checked_targets` still performs the **complete scalar NN replay** and
alone creates its genuine private target cache. Partial59's already complete
native target package is retained and fully replayed. No private certificate is
fabricated or reused under a new meaning.

Snapshots retain the original Recovery payload fields and version, with an
explicit `accelerated_wrapper` containing its separate version/source/header/
prefix bindings; their full payload digest is recomputed. Old readers must not
silently treat the wrapper as an ordinary recovery checkpoint.

18 software tests use clearly named low-level MOCK numerical/provenance
fixtures, exercising the actual original restore and control/latch implementation,
cached and partial calls, source/NN/epoch/prefix mutation refusal, unique-file
coverage, globals resolution, explicit producer guard and provenance, and the
complete frozen/new source closure. They do not claim an audited fake pool or
full optimizer-update equivalence. The existing real call34/47/58 payload/loss
equivalence receipts and static-report checks remain component evidence;
actual new-controller replay must match every committed update independently.
