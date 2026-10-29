# v6.43 PPO numerical bridge and source-bound finite owner

`experiments/warehouse/v643_calibration.py` connects the complete-episode risk component to the existing environment-free PPO algebra. Its old component-only `--run` still raises before reading a caller plan, making a random generator, touching output or consuming a claim. The actual finite owner, collector, program provider, independent measurement and clean continuation are implemented in the separate v643 pipeline modules. `scripts/run_warehouse_v643_calibration.py` requires an explicit exact plan/hash and original reference directory; recovery additionally requires an exact clean-pair checkpoint/hash under the same consumed claim. It checks complete source/inventory/history bindings before initialization. `--status` on the old inspector reports this distinction. No actual v643 scientific scene or sealed seed was used during implementation; physical integration tests use explicit synthetic states.

## Actual changes

The old PPO update implementation is reused for reward PPO, reference KL, masks, program KL, current-NN evidence validation, warmup and accepted optimizer-step accounting. The bridge explicitly replaces the old cost behavior:

- A dedicated reward optimizer owns only reward-critic parameters. A distinct cost optimizer owns only cost-critic parameters, with its Adam moments recorded separately.
- The complete-episode risk component consumes actual completed TRAIN outcomes only after the inherited physical batch checks succeed. The dual uses the new registered 2%/256/minimum64/initial2 settings. Replayed critic samples never count as fresh risk observations.
- Actor cost losses use current-rollout frames only. A complete episode may carry earlier observations for its real label and cost-critic training, but earlier frames cannot enter the current Actor gradient.
- Cost replay is one uniformly sampled episode minibatch per PPO minibatch, averaging BCE within each episode before averaging episodes. It runs during critic warmup too, retaining actual warmup failure examples until normal FIFO eviction.
- Before the inherited reward update, stale cost gradients are cleared, so its joint clipping helper behaves exactly as reward-only clipping; cost replay then uses its own registered clip. Actors and reward critic receive no gradient from replay.

The caller supplies the existing optimizer RNG. Both PPO ordering/reference sampling and cost replay use that same explicitly owned stream, so this component creates no additional unregistered RNG constructor. Both raw and regularized must start from the same complete state and stream. No runtime controller or action selection rule is added.

## Starting and checkpoint boundaries

`validate_common_start` rejects arbitrary or trained v6.42 weights. It requires the original complete v6.20 BC Actor hash, zero training counters, empty optimizer state, matching role weights, the bound reference buffer and actual Torch/frozen action-distribution equivalence. It returns identities for a future paired owner to compare; it creates no model and grants no initialization authority.

`save_boundary` stores the entire numerical model, Actor optimizer, reward optimizer, beta, trainability, counters and PPO/risk state. Risk state includes cost optimizer, complete episode buffer, consumed identities, multiplier and RNG. `restore_numerical_boundary` validates candidate, source closure, paired reference identity, all overlapping state identities and full training fingerprint before restoring into caller-created zero-update modules. It never constructs an Actor, environment or RNG. Numerical restoration is not scientific resume authority.

The numerical boundary explicitly says that environment, rollout reader, observed program pool/provider and single-use claim state are absent. Such a file cannot stand in for a complete finite experiment checkpoint. The new owner's outer checkpoint adds these components; a clean paired checkpoint also binds all owned artifacts and execution/reset journal prefixes. See `docs/warehouse_v643_finite_owner.md` for the tested same-claim continuation and its refusal of unaccounted post-boundary work.

## Why the old owner cannot be reused unchanged

1. `warehouse_v629_ppo_update.PPOUpdateCore` hardcodes the old risk settings, updates cost and reward through one optimizer and backpropagates cost through full historical episode rows. Simply adding cost replay would create competing cost optimizers and preserve historical Actor replay.
2. `v642_calibration_owner`, partition/seed plan, program provider, observed source and program-refit code bind the v6.42 candidate, output namespaces and source contracts. Subclassing the numerical core does not change those bindings. Old claims are already consumed and must remain closed.
3. The old long-training admission requires an independently passing old 20k calibration, which the actual raw result does not provide. v6.43 is a newly registered calibration; it must not accept a hand-written `passed` field or pretend the old failure became a prerequisite pass.

## Required before actual scientific execution

The new implementation now provides the finite collector/owner and new program output bindings, exact TRAIN/program/quality inventory contracts, complete source snapshots, actual metadata→triage→exact/context history verification, and the atomic single-use v643 claim immediately before initialization/reset. It integrates this bridge with physical/NN trace replay and the actual reset journal, with complete private/reader/provider/RNG checkpoints and an explicit clean continuation. No old claim or sealed seed may be reused.

Execution still requires newly locked real inventories, source snapshots and exact clearance against the actual four historical roots, followed by the candidate's first claim. No such claim or actual new 20k quality/program metric was produced by the implementation tests. This numerical module intentionally creates no pass marker. The new finite CLI returns nonzero on an actual failed program/quality precheck, preserves its closed-candidate receipt, and never treats numerical completion as model acceptance. Existing v6.42 scientific results, old sources, release status and pilot gates remain unchanged.

## Executed engineering checks

`python3 -m pytest tests/test_warehouse_v643_calibration.py -q` produced **16 passed in 42.54 seconds**, using the existing synthetic full-Actor fixtures and temporary files only. The checks use actual Torch PPO/Adam updates and verify:

- Critic warmup leaves Actor hashes unchanged while both distinct critics update; the real synthetic failed episode remains available to replay.
- A complete episode spanning two synthetic batches supplies both observations to critic training, while only its current frame enters the Actor surrogate. Replacing the excluded historical encoding with a very different value leaves the actual Actor cost gradient exactly unchanged.
- Paired raw/regularized updates before program feedback produce identical model tensors, both optimizer states, beta and risk/RNG state.
- After the feedback window begins, raw never calls a program provider and accepts zero program gradients. With proof rebuilt after every synthetic Actor update, regularized records four accepted positive updates for each role; cost replay remains isolated.
- A full numerical checkpoint restores the next PPO update, optimizer moments, risk observations and RNG bitwise. Old candidate IDs, absent/mismatched risk state, altered optimizers, wrong risk configuration and nonfinite model tensors fail without changing the destination state.
- Calling the disabled training entrypoint refuses before even opening a file. An old `20k_passed` flag or claim path is ignored and cannot authorize it.

These are implementation checks. They do not supply a new delivery, collision, shutdown or program-coverage result and do not open any scientific gate.
