# Warehouse training-only long-horizon Q: unregistered diagnostic

**Decision:** Do not feed this Q to the v6 Actor. It predicts which sampled NN states have elevated near-term collision risk, but fails to rank alternative legal joint actions from the **same pre-action state**. Its shutdown prediction is weak. It never selected a runtime action in this probe; the Warehouse v6 release gates remain unchanged.

## Design and independent splits

The behaviour policy was the older public-lookahead BC + offline DAgger pair of **two independent frozen NNs** (`warehouse_v6_unbounded_bc_probe.pt`); executed actions in collection were sampled from each NN after the physical legal-action mask. All seeds here are **unregistered development probes**. No registered program audit, sealed final audit, or held-out task seeds were used.

| Role | Seeds | Episodes | Rows |
| --- | --- | ---: | ---: |
| Q fitting | 985200–985319 | 120 | 14,272 |
| Probability calibration | 985320–985359 | 40 | 4,775 |
| Independent on-policy test | 988000–988047 | 48 | 5,690 |
| Independent branch audit | 988100–988111 | 12 | 35 selected pre-action states |

The candidate joint-action input concatenates both public pre-action observations, both physical legal-action masks, and two action one-hots (280 values). An optional 18 values describe each candidate target's public-map charger, pickup and delivery distances, return margin, and relative joint target location. The long-Q network has two 256-unit ReLU layers and four heads: collision within 10 frames, future shutdown, discounted collision count and discounted delivery count (`γ=0.97`). Targets come from **complete real environment trajectories**. A same-size action-conditioned one-step collision predictor is the comparison. No explicit collision predicate, route instruction, teacher action, or post-action observation is an input. Both networks were trained for 22 epochs; Platt probability mapping used only the disjoint calibration episodes.

## On-policy discrimination and uncertainty

On the 48 independent test episodes, future-10-frame collision AUC/Brier were **0.7067/0.2189** for long-Q, **0.7025/0.2202** without the 18 action-geometry values, and **0.6627/0.2323** for the one-step baseline. Resampling **whole test episodes** 1,000 times gave a long-Q minus one-step AUC difference of **+0.0443** (95% percentile interval **+0.0136 to +0.0744**) and Brier reduction **0.0135** (**0.0054 to 0.0214**). The geometry AUC gain was only **+0.0040** (**−0.0036 to +0.0121**). These are predictions of outcomes under the existing behaviour policy; they do not establish causal action ranking.

Future shutdown AUC was only **0.566**, average precision **0.058** against a **0.051** row prevalence. There were few shutdown-positive training trajectories. This head has not passed a safety-gradient reliability test.

## Same-state counterfactual action audit

For each of 35 pre-action states, the diagnostic cloned the environment and tried five **legal** joint actions: sampled frozen NN action, each NN argmax, public lookahead teacher, plus random legal alternatives as needed. Each first-step intervention was followed by **two** frozen-NN continuation streams using aligned random seeds. This was an offline diagnostic branch, never participant/runtime action selection. Returns include discounted future collision, delivery, and shutdown events. Episode-cluster bootstrap resampled the **12 source episodes**, preserving correlations among the 35 selected states, 5,000 times.

| Predicted head vs actual future return | Informative states | Mean within-state Spearman (95% episode CI) | Action-pair agreement (95% episode CI) |
| --- | ---: | ---: | ---: |
| Long-Q collision | 35 | **−0.007** (−0.147 to +0.129) | **50.1%** (44.1–55.8%; 347 pairs) |
| One-step collision | 35 | −0.060 (−0.265 to +0.155) | 48.1% (39.3–57.3%; 347 pairs) |
| Long-Q shutdown | 12 | **−0.045** (−0.246 to +0.231) | **47.8%** (36.5–64.1%; 67 pairs) |
| Long-Q delivery | 35 | +0.112 (−0.030 to +0.269) | 54.9% (48.5–61.8%; 339 pairs) |

Only **3** branch states had either robot's public charger return margin ≤6 moves. Their shutdown counterfactual outcomes varied in just **2** states (17 informative action pairs), where Q's pair agreement was **29.4%**. This is too small for a stable low-battery estimate, and it supplies no evidence that the current shutdown head is actionable. In the 23 states with minimum margin >12, long-Q collision agreement was **46.9%**. All strata are exploratory; the branch contexts were sampled at frames 15, 45 and 75, rather than designed as a full stress benchmark.

## Practical conclusion

The cross-state AUC improvement is real on this exploratory sample, but the state-matched audit shows almost chance-level action ordering. Optimizing the Actor against this Q would treat dangerous **states** as evidence that particular candidate **actions** are dangerous, injecting unvalidated gradients. Keep it out of the official PPO closure and runtime. If revisited, first collect independent, balanced branch data at the **same states** with randomized legal joint actions and low-return-margin enrichment, train a within-state pairwise or advantage target, and pass a new independent action-ranking audit before using Q as a training signal. These are proposed experiments, not validated remedies.

Reproducible ignored outputs: `warehouse_v6_long_horizon_q_probe.py/.json/.pt`, `warehouse_v6_long_q_cluster_bootstrap.py/.json`, `warehouse_v6_long_q_counterfactual_audit.py/.json`, and `warehouse_v6_long_q_counterfactual_summary.py/.json`. The Q weight SHA-256 is `4a28682a17f3ec8d10f9595434af8db879dddd6578845937ab4b4058a89b9954`. All files are in this ignored `output/` directory; no tracked model, policy binding, or release asset was changed by this diagnostic.

This tracked copy preserves the negative decision. The machine-readable
counterfactual summary is tracked beside it. SHA-256 of the local ignored
evidence at the time of review:

| Local output | SHA-256 |
| --- | --- |
| `warehouse_v6_long_q_cluster_bootstrap.json` | `b4d530ab206463b07fd32f1b411ee4a65916bbe66043be427b3e65b4d500ebdc` |
| `warehouse_v6_long_q_counterfactual_audit.json` | `0990ace21fe8c77a51adafb586f5465e6451ec7c8189bdae316a236ad05c8479` |
| `warehouse_v6_long_q_counterfactual_summary.json` | `31200c562bdf6115e13e16580c7a5e4991f74eeb26b611821df5de1c1ba160dd` |
| `warehouse_v6_long_horizon_q_probe.py` | `b4c92537df424e71ab44d109e77dc1cfbee8007d28815d4c3a1b2182a9a4c333` |
| `warehouse_v6_long_horizon_q_probe.pt` | `4a28682a17f3ec8d10f9595434af8db879dddd6578845937ab4b4058a89b9954` |
