# Optional v6.20 shared BC alignment contract

Status: engineering preparation only. This module does not authorize data
collection, model fitting, a fresh validation cohort, PPO or publication.
Only trigger the scientific BC route if the preregistered old-data grouped
search comparison fails to meet its stable-coverage condition.

## Model and objective

The existing additive v6.18 Actor remains unchanged:

`full_logits = primary_nn256_logits + tanh(learned_gate) * center(internal_tree_logits)`.

The optional BC recipe has version
`warehouse-public-shared-bc-alignment.v6.20.r1`. Both independent roles receive:

```
L_BC = mean(w * CE(mask(full), public_teacher))
     + 0.20 * mean(w * CE(mask(internal_tree), public_teacher))
     + 0.20 * mean(w * KL(stopgrad(softmax(mask(full))) || softmax(mask(internal_tree))))
     + gamma * mean(w * correct_tree * mean_legal(center_legal(primary_nn256)^2))
```

`gamma` is strictly one of `0.01` and `0.03`; all other coefficients are fixed.
The two candidates must use the same initialization, assigned scene/action RNG
sequences, minibatch generator recipe and declared stage budget. Teacher-driven
public records are identical. NN visitation trajectories can differ after the
two recipes update their weights; their frame counts and row identities are
recorded rather than described as identical data. This is a paired BC recipe
comparison. Fresh validation cannot select additional loss weights.

`correct_tree` is the detached condition that the raw internal tree's masked
argmax equals the **public offline BC teacher action**. The penalty is exactly
zero elsewhere, including its primary-network gradient. It is normalized over
all joint frame/role rows, not just the selected region. The mean over legal
actions removes an irrelevant common logit offset; illegal logits do not affect
the penalty or any objective gradient. A single-legal-action row contributes
zero policy-relevant penalty.

The complete five-action probability distribution supplies the one-way KL
target. No extracted external program, qualified leaf, current teacher action
controller, private goal, route commitment or future outcome enters this loss.
The raw internal tree is supervised separately from its learned additive gate;
the full task loss learns the gate and the primary NN. The internal tree may
still be inaccurate: the full Actor retains its existing additive expressive
power and quality is measured separately.

The sample-weight API accepts `[physical_frame, role]` nonnegative public-data
weights. It normalizes their joint mean to one, matching the existing risk-BC
weight convention. This preserves role accounting and avoids flattening labels
in the wrong order. It detaches weights so no learned gate can select training
examples to evade the objective.

## Callable interfaces

`shared_bc_alignment_loss` accepts full, raw internal-tree and primary logits
of shape `[N, 2, 5]`, matching binary physical masks and int64 teacher labels
`[N, 2]`. It returns named scalar losses and per-role component means.

`actor_shared_bc_alignment_loss` accepts the existing additive Actor and encoded
`[N, 2, FEATURE_DIM]` minibatch. It computes the same existing components without
changing model flags, parameters, buffers or forward computation. Historical
bounded Actor variants fail closed because their component meanings differ.

Both entry points require `phase="shared_bc"`. They reject PPO, evaluation and
variant names before performing computation. This phase check is a misuse
guard, **not a training authorization capability**. They expose no reset,
optimizer, checkpoint, extraction or external-program operation.

## Minimal runner integration, after the old-data route decision

These are proposed integration changes; this preparation does not modify the
registered runner, protocol, Actor class or existing candidate recipes.

1. Register a separate v6.20 candidate and immutable BC recipe. Keep the
   v6.18 defaults and closed historical candidate registrations unchanged. Bind
   `BCAlignmentRecipe(gamma).as_dict()` and the selected grouped-CV method receipt
   into the new candidate recipe. The Actor model version need not change when
   its forward, encoders and exported tensors remain byte-compatible; the BC
   recipe and source closure do change.
2. Add the new loss module, runner adapter, saved entry script, candidate
   protocol and exact JSON recipe to the new source closure. The new BC
   authorization must match source closure, recipe/config SHA, script SHA,
   candidate, initialization hashes, pair seed, data schedule and output paths.
   Do not bypass `_require_locked_execution`, change `FORMAL_TRAINING_DISABLED`,
   forge `_ACTIVE_LOCKED_EXECUTION`, or reuse a v6.18 capability for v6.20.
3. Before the first reset, check the old-data decision receipt says BC is
   required; check new seed clearance for all environment and action-RNG seeds.
   Keep pair initialization, BC stage scenes, critic probes, extraction cohorts
   and evaluation scenes distinct. The old `pair_seed * 100_000 + stage * 1_000`
   schedule is not automatically fresh. Explicitly journal the two real critic
   environment resets as well as teacher, sampled and argmax visits.
4. Preserve `_teacher_rows` and `_collect_bc_visits`' public-input labeling
   boundary: observations from `encode_public_observation`, masks from
   `action_masks`, labels from `public_lookahead_teacher(obs, masks)`. For each
   distinct training record, persist public pre-action state or a lossless
   reconstruction reference plus episode/frame, label, mask and row SHA. Audit
   reconstruction through `rebuild_public_pre_action_state` and independently
   recompute features and teacher labels. The BC label must never become the
   action during an NN-visitation episode.
5. Enable the existing trainable tree gate for the **new candidate only** at
   the start of joint BC using `set_tree_branch_enabled(True)`. Do not inherit
   the v6.18 dense-only initial/DAgger schedule by accident; its disabled gate
   prevents the intended full Actor joint learning. No hard residual cap,
   protected core mask or forced tree preference is installed.
6. In new-candidate `_bc_epochs`, replace the existing CE plus `.20 *
   _bc_tree_distillation` with `actor_shared_bc_alignment_loss(...,
   recipe=..., phase="shared_bc").total`. Do not add the old KL again. Keep
   declared optimizer, clipping, batch order and training epochs fixed. Log the
   per-role named losses, correct-tree fraction, split/support trainability,
   tree gate trainability and parameter-update norms for both independent roles.
7. The same new-candidate adapter must be used in every later BC fitting stage,
   including `_risk_bc_epochs`; otherwise the model loses alignment during
   the risk, large and deep BC stages. Preserve its existing public-row weights
   and any explicitly retained offline real-event collision-risk term separately.
   Do not double-count its old CE or detached-tree KL. Record retained BC terms
   in the immutable recipe. Risk shaping and margin changes are not silently
   introduced by this module. PPO does not inherit any offline BC term.
8. Keep topology support selection confined to BC training records, refresh
   training topology at declared BC boundaries, then lock support at BC end.
   Independent calibration, leaf eligibility and validation episodes cannot
   update Actor weights, tree topology or the chosen recipe.
9. Export one shared dual-Actor BC snapshot, verify Torch/frozen NumPy parity,
   and create raw and regularized by exact state-dict copies from it. Verify
   both role SHA maps, model version, BC recipe SHA and source identity before
   either PPO process begins. A regularized continuation must not become raw's
   supposed BC start. The two role parameters remain separate tensors.
10. The PPO update path must not import or call this BC adapter. Keep its
    auxiliary weights zero in both variants. Raw's external-program feedback
    count, loss and gradient remain zero; regularized uses only independently
    eligible current-NN-consistent feedback under the existing ramp. Each
    attempted BC phase writes exclusive start/failure/completion receipts and
    full reset journals; failures do not silently resume against a changed
    recipe or source.

## Preparation verification

`tests/test_warehouse_v620_bc_alignment.py` uses synthetic tensors and manually
constructed public states. It checks exact gamma enumeration, BC phase bounds,
complete-distribution KL stop-gradient direction, physical-mask gradients,
wrong-teacher-region zero penalty, offset invariance, weight/role accounting,
independent role updates, identical raw/regularized starts and unchanged frozen
NumPy inference. It performs no environment reset and produces no scientific
quality or coverage evidence.

Passing these tests is an engineering gate only. The old-data route decision,
shared BC pilot, fresh qualification, independent 216-trajectory development
confirmation and formal one-use gates remain separate scientific stages.
