# v6.43 PPO numerical bridge: implemented component, unavailable finite owner

`experiments/warehouse/v643_calibration.py` now connects the complete-episode risk component to the existing environment-free PPO algebra. This is **not an executable scientific 20k owner**. The CLI's `--run` raises before reading a caller plan, making a random generator, touching output, or consuming a claim. `--status` reports implementation boundaries and current source hashes. No Warehouse environment, fresh scenario or sealed scenario was used to build this component.

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

The saved boundary explicitly says that environment, rollout reader, observed program pool/provider and single-use claim state are absent. Such a file cannot stand in for a complete finite experiment checkpoint.

## Why the old owner cannot be reused unchanged

1. `warehouse_v629_ppo_update.PPOUpdateCore` hardcodes the old risk settings, updates cost and reward through one optimizer and backpropagates cost through full historical episode rows. Simply adding cost replay would create competing cost optimizers and preserve historical Actor replay.
2. `v642_calibration_owner`, partition/seed plan, program provider, observed source and program-refit code bind the v6.42 candidate, output namespaces and source contracts. Subclassing the numerical core does not change those bindings. Old claims are already consumed and must remain closed.
3. The old long-training admission requires an independently passing old 20k calibration, which the actual raw result does not provide. v6.43 is a newly registered calibration; it must not accept a hand-written `passed` field or pretend the old failure became a prerequisite pass.

## Required before scientific execution

The next implementation must provide a **new** finite collector/owner and program output bindings, with exact inventories for training/program evidence/new quality evaluation; new source snapshots and full-history exact seed clearance; and an atomic, single-use v6.43 claim checked immediately before the first initialization/reset. It must integrate this numerical bridge with physical trace replay and the real reset journal. Its full checkpoint must additionally bind reader pending histories, all eight private environments, active cases/cursors, observed pools/provider/refits, source/clearance identities and the claim. No old claim or sealed seed may be reused.

Only after these parts exist should the new calibration be run and its own 20k quality/program results independently audited. The present module intentionally has no `claimed_owner` function and creates no pass marker. Existing v6.42 scientific results, old sources, release status and pilot gates are unchanged.

## Executed engineering checks

`python3 -m pytest tests/test_warehouse_v643_calibration.py -q` produced **16 passed in 42.54 seconds**, using the existing synthetic full-Actor fixtures and temporary files only. The checks use actual Torch PPO/Adam updates and verify:

- Critic warmup leaves Actor hashes unchanged while both distinct critics update; the real synthetic failed episode remains available to replay.
- A complete episode spanning two synthetic batches supplies both observations to critic training, while only its current frame enters the Actor surrogate. Replacing the excluded historical encoding with a very different value leaves the actual Actor cost gradient exactly unchanged.
- Paired raw/regularized updates before program feedback produce identical model tensors, both optimizer states, beta and risk/RNG state.
- After the feedback window begins, raw never calls a program provider and accepts zero program gradients. With proof rebuilt after every synthetic Actor update, regularized records four accepted positive updates for each role; cost replay remains isolated.
- A full numerical checkpoint restores the next PPO update, optimizer moments, risk observations and RNG bitwise. Old candidate IDs, absent/mismatched risk state, altered optimizers, wrong risk configuration and nonfinite model tensors fail without changing the destination state.
- Calling the disabled training entrypoint refuses before even opening a file. An old `20k_passed` flag or claim path is ignored and cannot authorize it.

These are implementation checks. They do not supply a new delivery, collision, shutdown or program-coverage result and do not open any scientific gate.
