# Canonical production, unchanged independent scalar audit

`warehouse_v642_canonical_relabel.relabel_current_NN(observed, actor_ref,
output, *, epoch_id, sequence, source_guard)` produces a new native-compatible
counterfactual target package. The active owner must supply a callable guard
returning its `owner`, claim and complete executor `source_files`, including
this module's complete source closure. The guard is checked before model
reading, after numerical production and before publication.

Only the duplicate producer forward uses the tested outer batch loop: gate
matrices retain inner M=1, action residual matrices retain M=5, and nonlinear
ufuncs and physical masking/softmax retain their original physical-frame
shape. No weight, observation, physics, action, precision or tolerance changes.
Changing the inner matrix shape or accumulating gate rows as a new GEMM is
outside this method. Different kernels/platforms may still differ; there is
no universal floating-point equivalence claim.

Each role/partition has a deterministic output-independent schedule of at most
64 existing frames. Original scalar logits and five probabilities must match
the produced values byte for byte, including float hexadecimal values and
signed zero. These witnesses are additional error guards, not a complete
scalar proof or independent scientific validation. Existing recorded draw
arguments are reconstructed, never allocated or described as new samples.

The native target schema and its `source_files` retain their original
meaning. A separately hashed `canonical_producer_provenance` appendix binds
the new producer closure, active owner/claim, current dual-role hash, epoch,
sequence, observed source/audit/data, all array and row hashes, witness
schedule/results, runtime and actual costs. `read_provenance` checks this
appendix. Its pointer must also be retained by the new owner/event reader;
it is deliberately outside the native exact semantic asset inventory.

**The unchanged `warehouse_v642_observed_pool.checked_targets` must still
replay every produced frame using the original scalar NN before any target
can contribute a gradient.** The producer never calls private target-cache
sealing functions, never constructs a replay certificate, and never claims
the witnesses replace this complete check. Its result may be rejected by the
complete scalar reader even after the witnesses passed.

Software tests use temporary synthetic tensors and explicit fake guards. They
grant no real owner, training, seed, holdout, model admission or pilot access.
Old407, frozen6 recovery and the running process are not modified.
