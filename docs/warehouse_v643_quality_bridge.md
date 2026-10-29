# v6.43 calibration quality and final-program audit

`v643_calibration_quality.py` supplies no collection, reset, optimizer, random NN
initialization or scientific admission capability. The new finite owner must
authorize and journal all 225 matched quality episodes. A v6.42 20k passing
marker or claim is never accepted; the original failed result remains intact.

The completed pair reader checks the new source/claim/owner graph, actual full
20k numerical and private-reader boundaries, both role hashes computed directly
from checkpoint tensors, frozen exports and the chain of accepted updates.
Raw program updates must be zero; regularized must have actual accepted positive
program gradients for both roles. Different archive metadata cannot substitute
for three distinct reference/raw/regularized role policies.

After training, `finalize_programs` projects the same fixed program language onto
the actual final 20k NN using only the separately registered program data. It
keeps the unchanged v6.42 fitter, proofs, independent forward, solver ledger and
qualification algorithm. It cannot reuse an 18.6k target hash as a final hash.
Its candidate-specific once-only numerical claim is separate from online
extraction and does not allocate new environmental randomness. The final
qualification and target packages are replayed by `validate_final_programs`
before the owner may begin fresh quality resets. A failed role prevents that
stage; it cannot be hidden by the other role.

The quality source contains exactly 75 new, task-balanced cases for each of
reference, raw and regularized. The auditor matches all 225 finite owner
predebits, successful-reset bindings and physical reset-hook rows. It restores
saved private environment/RNG snapshots without calling a constructor or reset,
recomputes the exact masked NN distributions and sampled actions for both roles,
steps unchanged physics, and independently replays public delivery/collision/
shutdown counters. Incomplete episodes and action overrides are rejected.

Retention and fidelity use unchanged v6.42 numerical arithmetic: each trained
variant must deliver at least max(5.5,95% of reference), collide at most
min(6,reference+1), and have shutdown rate at most 5%. Every frozen final program
must pass coverage 45%, expected feedback coverage 40%, agreement 95%/97%,
KL at most 0.20, and zero illegal/unproved actions. Task-specific quality and
program metrics are reported separately. The registered quality gates remain
per variant over its matched 75 scenes; task strata are descriptive rather than
new post hoc selection thresholds.

`evaluate_outcomes` is pure arithmetic and explicitly says that no independent
collection has been audited. Only `audit_collection` can produce a replayed
calibration precheck from the complete new-owner graph and raw physical data.
Even that result always has `model_acceptance_passed=False` and
`pilot_opened=False`: development, independent formal audits and heldout tests
remain separate. These tests create temporary synthetic data or restore already
recorded engineering snapshots; they do not create a real candidate claim,
fresh scientific scene, model-quality receipt or human-study sample.

Main APIs:

- `audit_completed_assets(plan, completed_assets_ref)`
- `finalize_programs(plan, completed_assets_ref, namespaces=..., states=...,
  observed=..., last_program_refs=..., output=...)`
- `validate_final_programs(plan, completed_assets_ref, final_programs_ref,
  observed=...)` before any quality reset
- `audit_collection(plan, completed_assets_ref, collection_ref,
  final_programs_ref, output=..., observed=...)`

Passing the already replayed in-memory `observed` pools avoids reconstructing
the same program source again. Their exact immutable source/owner/hash bindings
are still verified, and current-NN targets and qualifications are recalculated.
