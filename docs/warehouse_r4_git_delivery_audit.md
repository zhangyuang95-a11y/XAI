# Warehouse r4 Git delivery audit

Audit date: 2026-09-11. Baseline commit: `773837982faa98fb524ca94acad5242ae61dccd4`.

The staging allowlist is
[`docs/warehouse_r4_git_allowlist.txt`](warehouse_r4_git_allowlist.txt). It has
96 repository-relative regular files: 11 already tracked paths and 85 paths
that are currently untracked. Every untracked path is warehouse source, a
warehouse test or acceptance script, or r4 documentation/UI. The list contains
no `output/` path, model/checkpoint, database, event log, participant record,
secret file, kitchen domain, candy domain, or other cooperative-domain source.

The production source contract is
[`backend/training/warehouse_r4_release_source_closure.txt`](../backend/training/warehouse_r4_release_source_closure.txt).
It contains 193 sorted, unique paths: the complete static local import closure
of the r4 training, audit, admission and package commands, plus the browser UI
and the checkpoint screen tool. `release_sources()` validates every path and
binds the closure manifest itself, producing 194 source hashes. The three
online loader/runtime files owned by the portable release source group are
excluded from the r4 group so the two signed groups cannot overlap. An AST
regression test recomputes this boundary and fails if a future local import is
not listed.

An isolated worktree created from the baseline commit, overlaid only with the
allowlist, produced these results:

- All 13 production, admission, package and Render command modules imported and
  returned `--help`. Both Python HTTP acceptance scripts returned `--help`
  through their documented file-path invocation. `release_sources()` loaded
  all 194 bindings.
- The selected Python test set passed 99 tests and failed 15 when the private
  `output/warehouse_native/` artifacts were intentionally absent. Every
  failure was a missing registered actor, checkpoint, trajectory, scenario or
  question-pool file; there was no missing source import.
- After making the existing private warehouse artifact tree available at the
  expected repository-relative path for read-only test use, the same selected
  suite passed all 114 tests.
- The two frontend suites passed all 14 tests with the bundled Node runtime.
- All 189 Python files in the production source closure compiled.
- The real failed budget ledger also passed a semantic replay from the frozen
  main workspace: 18 attempts, 20 segments, 1,000,000 fresh steps, no selected
  Actor and `admission_eligible=false`.

The source-only checkout cannot reproduce training or the final audit by
itself. The r4 trainer restores the private 3.95M lineage and admitted r4 stage
actors/checkpoints from ignored `output/warehouse_native/` paths. The final
pipeline also needs the registered foundation scenario manifest and question
pool. These artifacts must be restored separately and verified by their
recorded SHA-256 values. The Render Secret File is enough to serve the frozen
study, but it does not contain the full training lineage.

`requirements-render.txt` intentionally installs only NumPy. Offline training
and RCPD fitting also need PyTorch and scikit-learn. The browser acceptance
script needs Node.js and Playwright; a plain fresh checkout does not install
that Node dependency. The current Render configuration also uses ephemeral
SQLite under `/tmp`, so a service restart can lose pilot records.

The frozen training gates were not passed. The authenticated failed ledger,
selection metrics, selector result and unchanged r3 Render status are recorded
in [`docs/warehouse_r4_training_closeout.md`](warehouse_r4_training_closeout.md).
No ignored report or model artifact is included in the staging allowlist.

The allowlist passed credential and PII pattern scans for private keys, common
cloud tokens, API keys, authenticated database URLs, email addresses, absolute
home-directory paths, and generated data extensions. Two credential-like
strings remain intentionally in `tests/test_warehouse_r4_online_release.py`:
an example PostgreSQL URL and an example Bearer token. They are inert fixtures
that verify the package builder rejects secrets.

## Final source-freeze procedure used

After every r4 producer, selector, test, UI and documentation file was frozen:

1. Recompute the static local import closure from
   `SOURCE_CLOSURE_SEEDS + SOURCE_CLOSURE_ASSETS`, subtract only the three
   independently authenticated portable sources, and rewrite the closure
   manifest in sorted order.
2. Confirm that every modified tracked file and every untracked file required
   by that closure or the selected tests appears in the staging allowlist. No
   ignored `output/` file may enter the list.
3. Create a detached worktree from the baseline commit and overlay only the
   allowlist. Run every final command with `--help` and load
   `release_sources()` before making any private artifact available.
4. Run the selected Python suite once source-only. Any failure must be solely
   an explicit missing private-artifact dependency. Make the existing private
   warehouse artifact tree available read-only and rerun the suite to zero
   failures. Run the two Node frontend suites separately.
5. Compile every Python file in the source closure, repeat the credential/PII
   scan, run `git diff --check`, confirm an empty Git index, and record the
   final allowlist, closure and release-source-map SHA-256 values.

The resulting final hashes are:

| Boundary | SHA-256 |
| --- | --- |
| Staging allowlist | `8f73b3bc0553cf8098465a59861d0c3eddfad767742cacbef97a71b4037e211e` |
| Source-closure manifest | `6e48e55d3ad765d6f1a651db7a2d37028ccbdb0c229a1c7bb9734303616ffabc` |
| Canonical `release_sources()` map | `70e3b6bfe2ad75be5fb26911cee059b52556d12b6dc50b3a9b5930df73e37e4d` |
