# 12069 v2 Warehouse first-quality collision diagnosis

This is a read-only diagnosis of the **already consumed** 96 quality episodes.
It does not qualify the model. No reserved pressure, program, registered,
audit, or heldout seeds were used. The v6 pilot remains closed.

## Provenance

- Original `first-quality-result.json` SHA-256:
  `5c6bd8c5d6c6dd4ec60834269f2b201d4721f93783b9a1e19765b3e85721f750`.
- Frozen dual Actor NPZ SHA-256:
  `4c04d33b6907cd04d596e29dd7fa39c857c6b5851a5762d0580f3376783a2843`.
- Snapshot source closure SHA-256:
  `89c19e10fcfd0ca67552e49942b13bef359efd9edafc408b0519b01d4f962b7b`.
- Separate verified 11,496-frame replay SHA-256:
  `f4f509cc385c1e1418af08ef2fde74a39db1fe9a56b42b658b0eb4e5f960dacc`.
- Independent script `reproduce.py` replays original NN actions only, checks
  every frame against the verified replay, and derives `diagnosis.json`. The
  enlarged-bound comparison computes logits in memory; it never steps an
  altered action.

## Failure mechanics

The frozen Actor produced 739 collisions over 96 episodes (7.698 mean),
exceeding the six-collision development screen. The types were 342 same
target, 225 stationary occupant, and 172 swap. The worst previously used
scenario seed `12001181` contributes 100 same-target collisions alone.
Removing that episode still leaves 639/95 = 6.726 collisions per episode.

On frames 5–104 of that episode, robot 1 is fixed at `(3,1)` and submits
`RIGHT` toward `(3,2)`; robot 2 is fixed at `(4,2)` and submits `UP` toward the
same cell. Both actions pass the per-robot physical mask. The environment
correctly cancels the joint collision; it does not override either NN choice.
The public collision-streak feature rises from 0 to its capped value, while
both Actors keep essentially the same actions and positions.

Robot 1 stays in **core leaf 4** for all 100 frames. Its tree and complete NN
rank `RIGHT` first, with about 0.9975 masked action probability; tree margin
over the next legal action is 6.156 logits. Its residual bound is 0.08599.
Robot 2 enters **core leaf 10** after the first collision, where its tree and
complete NN rank `UP` first with about 0.9913 probability; tree margin to the
next legal action is 4.758 logits, and residual bound is 0.08303. Both leaves
carry `core_leaf_mask=1`, and the maximum core residual bound is 0.12. Even
the maximum possible residual difference between two actions is 0.24 logits,
far below these tree margins. Independent stochastic sampling is not the
primary cause: the complete NN top actions themselves collide, and their
joint probability in the loop is roughly 0.989 per frame.

The existing core topology is fixed after its public-data fit. Robot 1 leaf
4 already uses all five permitted splits: charger distance, legal `DOWN`, two
recent-position columns, and legal `RIGHT`. Robot 2 leaf 10 uses charger
distance/battery/charger-wait, battery fraction, partner previous `WAIT`, and
legal `UP`. Neither path tests the public partner position or previous
collision signal. Consequently the tree assigns the same core leaf to
conflicting and ordinary states. PPO can update a whole leaf's logits, but
the fixed split and tiny residual cannot give this individual conflict state
its own action without changing other visits to the same leaf.

In the 96-episode replay, robot 1 core leaf 4 appears 944 times. The public
pre-action lookahead BC teacher labels 781/790 non-collision visits `RIGHT`,
but labels 144/154 collision visits `WAIT`; the frozen Actor chooses `RIGHT`
in 943/944 visits. In the worst loop the teacher consistently labels the
safe joint action `(WAIT, UP)` using only current public observation and
physical masks. Robot 2 core leaf 10 appears 345 times; its teacher labels
`UP` 342 times and the Actor chooses `UP` 343 times. This identifies a
role-1 leaf aliasing problem, not a missing observable or an erroneous
robot-2 action in that loop. The BC receipt records collision-bearing
visitation and deduplication in aggregate, but does not retain per-leaf
training-state signatures; exact exposure of this loop during BC cannot be
established from the saved receipt.

As a diagnostic ablation, keeping the trained residual direction and gate
but increasing each loop-state core bound to 8.05 changes robot 1's top
action to `WAIT` in only 6/100 frames. Robot 2 remains `UP` in 100/100.
The original colliding joint top action remains in 94/100 frames. Merely
raising the cap therefore does not solve the learned residual direction.

## Next candidate, subject to new development evidence

1. Train a **state-dependent learned core confidence and residual bound**
   from public observations. Remove the unconditional 0.12 cap for every
   member of a qualified leaf. Keep tree/simple-region alignment as a
   training regularizer; let the NN residual learn to handle publicly
   distinguishable interaction states. A frozen NN remains the only runtime
   action selector.
2. Refit or jointly train the sparse tree topology using on-policy
   conflict-bearing states and public BC labels, within depth 5, 16 leaves,
   and at most 3 features per node. This leaf has no remaining depth, so its
   current path needs restructuring. Include public relative geometry and
   collision history among candidate inputs; do not encode a runtime
   collision-resolution rule.
3. Use real environment collision events and full-episode shutdown events to
   train learned critics and Actor advantages during paired PPO. Preserve
   raw/regularized identical starts and seeds; raw program feedback remains
   zero. Record per-leaf state/label/incident support in training receipts so
   the next failure can be traced without assuming what the training data
   contained.
4. Qualify a *separate* role-specific program against the complete frozen NN
   on disjoint fit/calibration/eligibility/validation episodes. Program
   agreement and coverage are measured after the NN quality screen; the
   program never chooses actions. Use new unregistered development seeds for
   the next candidate, then follow the locked formal, audit, and heldout
   sequence without reusing sealed sets.

These changes are a diagnosis-driven candidate design, not an acceptance
result. Whether they meet the fixed quality and program gates requires a
new locked run and independent stress evaluation.
