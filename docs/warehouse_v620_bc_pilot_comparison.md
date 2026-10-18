# Warehouse v620 common-BC development comparison

Status: adopted engineering entry, development only. It grants no fresh275,
PPO, sealed audit, participant pilot or publication authority. The corresponding
shared-BC capability still requires an exact independent seed clearance,
explicit authorization receipt and a newly locked source closure.

## Files and prerequisites

Production locations:

- `backend/training/warehouse_v620_bc_pilot_eval.py`: immutable plan, method
  lock, paired snapshot provenance, collection inventory, ranking and gamma lock.
- `backend/training/warehouse_v620_bc_pilot_runtime.py`: adapter to frozen NN
  collection, program fitting and physical quality replay.
- `scripts/run_warehouse_v620_bc_pilot_comparison.py`: saved CLI entry.
- `configs/warehouse_v620_bc_pilot_comparison_recipe.json`: fixed recipe,
  required to equal the executable source contract.
- `tests/test_warehouse_v620_bc_pilot_comparison.py`: pure synthetic stubs.

The runner must provide optional persistent `journal_path` on
`collect_authorized_program_rows`; the adopted collector preserves successful
resets on normal return, source drift and interruption. It owns its own reset
capture, so this entry never nests an outer capture around collection.
Physical quality replay uses its own separate capture after collection exits.
The main task registers every new executable/config in the source closure.

## Prerequisites checked before any rollout

1. Complete native corrected cv02 evidence. The shared BC verifier follows
   approved historical source/recipe/script lock, v2 plan, all 18 completed
   jobs and actual program reports; it recomputes per-fit search budgets,
   gates and full method summary. Closed cv01 is not eligible.
2. Native summary must have `chosen_method_arm=null` and next stage
   `common_bc_structure_learning_before_fresh_validation`.
3. Both `.01` and `.03` BC runs must be `bc_completed_not_admitted`, have the
   same seed6160 initialization, paired scenario/action/training RNG plan,
   exact BC source, and complete 1270+2 persistent reset chain.
4. BC recipes, authorization, receipt, report, final Actor file SHA and both
   actual role weight hashes must agree. Both roles must have changed from
   initialization. Identical final gamma weights are allowed if the gamma
   penalty had no effect; results must reveal that.
5. BC provenance records zero external-program feedback, no intermediate
   extracted-tree installation, no fresh validation, PPO or publication.

BC source and later comparison source are separate explicit hashes. Native
cv02 is verified against its historical approved lock rather than the later
live source. Comparison source is checked before each operation, after each
collection/fit and against all prerequisite bytes before completion.

## Lock the method before collecting

Choose M1/M2 by the full native CV's lexicographic maximin `selection_score`;
exact ties choose M1. Neither method needs to have passed the frozen-NN CV.
This is a development-method lock, not an acceptance claim.

`--prepare` checks all prerequisites, creates the output directory exclusively,
and durably writes `method.lock.json` before `plan.json`. Existing directories
or altered locks fail. The same method is used for both gamma comparisons.

## Reuse only the spent v619 124-scene development plan

Each gamma uses the exact source-closed original scene order and modes:

| Purpose | Episodes | Sampling |
|---|---:|---|
| Fit | 50 | First25 sampled, last25 argmax |
| Calibration | 25 | Sampled |
| Eligibility | 25 | Sampled |
| Validation | 24 | Sampled |

The action base argument equals the scenario seed (`action_seed_offset=0`).
This is intentional matched reuse of already spent development scenes and
streams. It is **not fresh or independent admission evidence**.

Every case has its own exclusive persistent reset journal, all physical role
frames, actual episode outcome and a full joint NN trace. No frame filtering,
Actor update, external-program gradient or program action selection occurs.
Each role's external program is fitted independently with the same locked
method and source-bound search budget. Saved program files are reloaded and
their actual budget and validation records are checked again during selection.

## Quality comes from the same24 validation trajectories

Quality uses only actual validation outcomes. The compact action/state chain
is derived from those recorded full NN trajectories. A physical replay checks
every action, state transition and outcome against the same frozen NN.

Per gamma there are **124 collection resets plus24 replay resets =148 physical
resets**, and **24 statistical quality episodes**. Replay is not extra evidence.
No additional quality sampling or fresh275 collection is permitted.

## Fixed gamma selection

Both gamma comparisons must finish before selection. Quality is the sole hard
eligibility criterion: mean deliveries >=5.5, collisions <=6, shutdown <=.05.
There is no pilot coverage hard threshold.

Among quality-eligible candidates:

1. Prefer `program_consistency_pass`: both roles' feedback/explanation
   agreements >=.93/.95, qualified KL <=.25 and whole-tree raw illegal/path
   uncertified counts both zero.
2. If every quality-eligible gamma fails that preference, first maximize the
   worst normalized ratio over both roles:
   `min(Fagreement/.93, Eagreement/.95, .25/max(FKL,1e-12), .25/max(EKL,1e-12))`.
3. Then maximize minimum-role explanation coverage, minimum-role expected
   feedback coverage, negative maximum qualified KL, quality score
   `deliveries-.25*collisions-5*shutdown_rate`, and finally prefer smaller gamma.

Unavailable agreement/KL contributes zero consistency margin; unavailable KL
ranks worst. Illegal/path counts affect `program_consistency_pass`; as
predeclared they are not added to the normalized fallback formula. Reports
retain these counts. A fallback-selected model has **not passed admission**;
future fresh hard checks still reject any illegal/path failure.

Both quality failures write an exclusive candidate closure, with no fresh
permission. Otherwise `gamma.lock.json` fixes gamma, full Actor SHA, both role
hashes and method. Later fresh275 must fit new external programs. Pilot trees
cannot be reused for admission, and a failed fresh precheck cannot trigger
switching to the other gamma.

Successful native selection also emits `selected-method.receipt.json` and
`model-method.lock.json` (`warehouse-v620-model-method-lock.v1`). These bind
current/BC source generations, the exact Actor and both role hashes, method,
comparison/BC recipes, CV/method/gamma locks and the chosen result. Pointers
contain exactly `path` and `sha256`, and the frozen gate consumes these files
directly. No lock is emitted after a quality closure. The locks do not replace
an independent source/model/recipe-bound fresh seed clearance.

## Intended CLI sequence after separate authorization

The entry supports only:

```
python scripts/run_warehouse_v620_bc_pilot_comparison.py --prepare --spec <spec.json>
python scripts/run_warehouse_v620_bc_pilot_comparison.py --gamma gamma-001 --plan <plan.json>
python scripts/run_warehouse_v620_bc_pilot_comparison.py --gamma gamma-003 --plan <plan.json>
python scripts/run_warehouse_v620_bc_pilot_comparison.py --select --plan <plan.json>
```

The spec schema is `warehouse-v620-bc-pilot-comparison-spec.v1` with exact keys
`candidate_id`, `source_closure_sha256`, `bc_source_closure_sha256`, `cv_summary`,
`bc_runs`, `output_directory` plus `schema`. Each `bc_runs` gamma has four
path/SHA pointers: `spec`, `authorization`, `receipt`, `report`. Output is the
exclusive registered `output/warehouse_v620_bc_pilot/comparison-01` directory.
No example authorization or fictitious completion receipt is supplied.

## Failure records and engineering tests

Any interrupted/failed gamma consumes its directory. Failure receipt records
completed collection prefix, all attempted journal paths and available SHAs,
exception reset evidence and explicit no-retry/no-fresh flags. Missing Actor,
changed source/NN/input, incomplete gamma or modified trace/program rejects.
The shared-BC capability also retains the actual reset journal and receipt if
its authorization file is deleted during execution; hashing that missing file
cannot erase consumed-scene evidence.

The comparison's 28 tests use lightweight fake Actors, collector, replay and
program fitter. They never import production training or call a real fit or
environment reset. Alongside39 shared-BC engineering tests, the first adopted
verification was **65 passed /5.84s**. After native frozen-entry handoff
coverage was added, **67 passed /6.39s**. Full regression, exact seed clearance and
scientific results are separate gates.
