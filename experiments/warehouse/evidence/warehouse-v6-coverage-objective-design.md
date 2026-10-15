# Warehouse v6 coverage and shutdown design note (unregistered)

Status: design analysis, **not** a successful acceptance run. No registered development, sealed audit, or heldout seeds were used in this note.

## Observed failure mechanism

The v6.6 topology probe trained against the complete masked frozen NN distribution on scenario seeds 1110000–1110074. On internal CV, qualified leaves cover 30.13% and 31.60% of role frames, with expected sampled-action coverage 28.86% and 29.68%. The largest ineligible leaves occupy 22.1% (robot 1) and 20.8% (robot 2), with top-1 agreement 38.5% and 34.8%. These leaves are mixed in the **NN policy itself**; threshold tuning alone cannot make their entire distributions faithful. A second, ordinary soft-label depth-5/16-leaf tree on the same fit/CV rows reached only 25.08% and 28.63% expected actionable coverage. Explicit per-target action distance progress features, which are algebraically recoverable from existing absolute candidate distances, did not improve this ordinary axis-tree baseline. This does not rule out their use in three-feature oblique nodes.

The v6.5 retrofit loss activates only if a source NN row already lies in a fit/CV-qualified leaf, its top probability is at least 0.8, its tree top equals source top, and its raw tree action is legal. Thus its certificate margin term cannot grow coverage into the large mixed leaves. Source-policy distillation further resists changing those mixed distributions.

## Actor and certificate

For role-specific public feature vector `x`, let leaf `l(x)` be produced by a depth-5, at-most-16-leaf hard tree with at most three nonzero public features at each node. The complete Actor logit is

`u_a(x) = T[l(x),a] + g_theta(x) b[l(x)] tanh R_theta(x,a)`,

where `R_theta` is the 256-wide residual, `g_theta` is a learned sigmoid gate, and `b` is a learned nonnegative leaf bound. The physical mask is applied after these logits. Every executed action comes from the frozen complete Actor's masked distribution. The extracted program independently fits `P_l` to the **complete** masked NN distribution and is never called to choose an action.

Write `alpha(x)=g_theta(x)b[l(x)]`. For an action `a` with legal raw program action, the sufficient pointwise top-1 condition is `min_{c != a, legal}(T[l,a]-T[l,c]) > 2 alpha(x) + epsilon`. If the fit/calibrated program logits differ from the Actor tree by at most `e_l = min_k max_a |P[l,a]+k-T[l,a]|`, the stronger program check `min_{c != a,legal}(P[l,a]-P[l,c]) > 2 e_l + 2 alpha(x) + epsilon` proves the program, tree and frozen NN masked argmax agree. It does **not** prove a stochastic NN draw equals the program action; the independent episode audit must measure sampled agreement and distribution KL.

## Coverage as a direct training target

1. Fit topology from frozen dual-NN self-play rows using full masked distributions, not sampled actions or teacher labels; retain exact hard routes. Use public per-candidate progress to charger and each public order endpoint, alongside public status, legality and energy margin. All targets are parallel features; there is no selected goal or action. Search globally over topology/pruning rather than adding splits only below saturated depth-5 leaves.
2. For a topology cycle, identify candidate protected leaves from fit and disjoint internal-CV episodes. Require a legal raw top action on **every** internal row and sufficient independent episode support. Do not require an old source leaf to be already qualified. Rank leaves by occupancy times potential sampled-action mass, subject to quality and event-risk checks on separate development stress trajectories. Keep high-risk/ambiguous leaves on the large residual branch.
3. Within protected leaves, train the Actor jointly with the task objective using (a) masked `KL(softmax(T_l) || softmax(u))`, (b) a smooth certificate-margin shortfall `softplus((2 alpha + epsilon - margin_T)/temperature)`, and (c) an `alpha` barrier. Aggregate distribution loss by episode and use a high quantile/CVaR term so a few hard episodes cannot hide behind a low row-mean KL. Weight the full-Actor policy objective more strongly than this structural term; monitor source-policy KL and development delivery/collision/shutdown. Tree and gate gradients must be separately nonzero and logged.
4. Optionally use a fit-only smooth target for expected actionable mass: sum over leaves of `occupancy_l * pi_theta(program_action_l | s)` multiplied by soft surrogates for KL and margin eligibility. It is a training surrogate only. The actual pass/fail decision remains disjoint validation and later sealed audit. Target above the 30% development floor (for example 45%) to leave a sampling/selection margin, without changing any gate.
5. Re-collect on-policy dual-NN trajectories and refit topology on new unregistered fit/internal-CV episodes after a training cycle. Accept a topology only when its own CV improves and independent pressure quality remains viable. Never select/checkpoint on sealed program audit or heldout seeds. The raw and regularized pair begin at the identical dual-Actor BC state. Any intrinsic architecture training applied to both variants must be reported separately from external RCPD feedback; the raw external-program gradient remains exactly zero. Formal regularized RCPD remains lambda=0.01 with 80–85% ramp.

This objective can still fail: a large mixed leaf may require different actions for quality. If its collapse reduces deliveries or increases shutdown, leave it complex and find a different partition or architecture candidate. There is no mathematical guarantee of the empirical 40% development target.

## Real-event shutdown cost

The current runner zeroes every cost Actor advantage in a batch with no shutdown event and decreases the multiplier by 0.08 at each update when a 64-completed-episode window has zero shutdowns. Thirteen such updates can remove the cost penalty, although 64 zero events have a one-sided 95% exact binomial upper bound near 4.57%, above the 4% risk target.

Train the cost value on the actual episode shutdown indicator/cost-to-go, including complete no-event episodes. Use a cross-episode calibrated baseline to form `A_cost = observed_cost_to_go - V_cost(s)` for **all** complete episodes, with fixed-scale clipping to control sparse-event variance. Complete episodes alone update the dual multiplier. A larger predeclared window (e.g. 256 complete episodes) and hysteresis prevent premature relaxation: increase when the empirical rate exceeds the development risk target; decrease only when a predeclared one-sided upper confidence bound is below target minus a buffer; otherwise hold. Record window count, event count, upper bound, multiplier and cost-gradient norms. Tune this only on unregistered/development pressure data and lock all settings before formal training.

## Evidence limits

The v6.6 rows are exploratory and from one source BC Actor. They establish a failure mode, not formal feasibility or impossibility. The proposed loss, topology and risk dual require a fresh unregistered probe, matched raw/regularized self-play, independent pressure evaluation, and then the user's formal sequence. Neither the source quality figures nor synthetic/fit-only gains can be described as Warehouse acceptance.
