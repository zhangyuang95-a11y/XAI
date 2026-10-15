# Warehouse v6: evidenced failure modes and model changes

Status: development diagnosis only. No registered v6 formal pair, sealed
program audit, or heldout task set has been run. This document does not claim
admission or authorize the pilot.

## Evidence

1. The v5.4 structured regularized Actor reached 8.292 deliveries and 4.667
   collisions on 24 development episodes but shut down in 7/24. The raw Actor
   had 8.250 deliveries, 6.958 collisions, and 2/24 shutdowns. The available
   [summary](../../../docs/study_v3_warehouse_v54_development.md) has no
   frame trace for those seven events, so their cause cannot be assigned to
   one particular mechanism.
2. In the earlier v6 candidate 02,
   [diagnosis](warehouse-v6-candidate-02.diagnosis.md) found that standardizing
   near-zero shutdown advantages amplified cost gradients to about 17–19
   times the reward gradients and coincided with roughly 70% WAIT actions.
   The old update also treated incomplete rollout fragments as if they had
   resolved terminal risk.
3. The frozen v6.7 source Actor produced 507 collisions across 72 unregistered
   diagnostic episodes. Two episodes contributed 175 collisions. Their
   repeated joint UP/LEFT proposals attempted the same target, so the
   environment physically executed WAIT while the NN continued to propose the
   same joint action. The trace and file hash are preserved in the
   [v6.7 failure receipt](warehouse-v67-soft-hard-11367-failure.md).
4. The v6.5 largest mixed leaf covered 49.4% of role 1 rows, with 44.3%
   NN/program top-action agreement and KL 0.850. Its tree margin (0.248) was
   much smaller than the possible gated residual contribution (6.767).
   Exact sparse-tree fit-only cross-validation in v6.6 reached 28.86% and
   29.68% actionable coverage, below the 30% development gate; the v6.7 soft
   split prototype fell to 5.18%/0% on the actual hard route. These failed
   screens are recorded in the respective versioned failure receipts.

## Corrective architecture

- Add pre-action public observations for the last NN-proposed action, the last
  environment-executed action, and recent public collision outcome. Rebuild
  each feature from exported frames, and test live/replay parity. These are
  NN inputs, never action-selection rules.
- Use the real complete-episode shutdown event as a Bernoulli cost target.
  Exclude unfinished fragments from cost learning and dual updates. Use a
  joint two-Actor PPO probability ratio for the shared event, full-episode
  event credit, an episode-balanced cost critic, and a conservative confidence
  bound for the risk multiplier. Keep the reward and risk signals on their
  measured scales.
- Train a sparse, hard-routing Actor branch with the 256-wide residual under
  the common raw/regularized architecture. Fit an independent program to the
  complete frozen Actor distribution. Each program node uses at most three
  public features, depth at most five, and the whole tree at most 16 leaves.
  Certify coverage and fidelity on disjoint fit/calibration/eligibility/
  validation episodes. The frozen NN alone selects all executed AI actions.

The new method must pass fresh unregistered four-part program checks, an
independent multi-trajectory pressure screen, and paired PPO quality before
any registered budget is locked. The numerical gates and one-use sealed seed
rules remain unchanged.
