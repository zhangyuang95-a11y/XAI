# Warehouse v6 evidence ledger

The active release target is `policylens-three-domain-20261016.v6.0`.
Its Warehouse Actor is **pending**; the pilot remains closed. Every row below
is a historical failure or unregistered development probe. None is a passing
formal paired run, final independent program audit, or held-out test.

| Study | Evidence | Result |
| --- | --- | --- |
| Early 200k candidate 02 | [Failure receipt](warehouse-v6-candidate-02.failure.json), [diagnosis](warehouse-v6-candidate-02.diagnosis.md) | Both variants failed delivery quality; cost normalization, stochastic control, tree collapse, and reward ranking were diagnosed. |
| v6.1 exploratory Actor | [Study](warehouse-v6-v61-exploration.md), [public teacher probe](warehouse-v6-public-teacher-unregistered-eval.json), [rejected long-Q probe](warehouse-v6-long-q-rejected.md) | The public teacher succeeded offline; the NN probes did not meet all quality margins. |
| v6.2 public BC and Actor-tree extraction | [BC receipt](warehouse-v62-from-scratch-bc-10077.md), [tree failure](warehouse-v62-actor-tree-1017xxx-failure.md) | Internal Actor tree produced under 30% actionable coverage in both roles. |
| v6.3 irregular public tree | [Failure](warehouse-v63-irregular-tree-10377-failure.md) | The tested BC modes failed dual-role program coverage or quality. |
| v6.4 public per-action BC | [Failure](warehouse-v64-public-action-bc-10488-failure.md) | Pressure quality passed; dual-role program coverage did not. |
| v6.4 50k paired PPO probe | [Failure](warehouse-v64-50k-ppo-10688-failure.md) | Pressure quality passed; robot_1 regularized program coverage and selected-checkpoint feedback gradient failed. |
| v6.5 intrinsic tree BC probe | [Failure](warehouse-v65-10965-unregistered-failure.md) | BC pressure quality passed narrowly; both roles failed the 30% actionable program coverage precheck. |
| v6.6 static topology screen | [Seed lock](warehouse-v66-unregistered-seed-lock.json), [failure](warehouse-v66-static-topology-11166-failure.md) | Fit-only CV estimated sampled-action coverage of 28.86% and 29.68%; the reserved four-part, audit and pressure blocks were not used. |
| v6.6 distance-progress screen | [Failure](warehouse-v66-delta-11266-failure.md) | Replacing equivalent absolute split inputs with public action-distance deltas reduced fit-only CV actionable coverage to 19.95% and 23.22%; no reserved four-part, audit, pressure, or formal seeds were used. |
| v6.6 joint Actor exploration | [New seed lock](warehouse-v66-joint-actor-seed-lock.json) | Fresh unregistered cohorts reserved for jointly training the tree and full Actor; no result or admission claim. |
| v6.7 trainable soft route | [Failure](warehouse-v67-soft-hard-11367-failure.md) | Hard-tree validation yielded 5.18% and 0% expected actionable coverage; the method stopped before paired PPO. |
| v6.8 fixed old topology | [Failure](warehouse-v68-old-topology-fitonly-113-failure.md) | Bounded updates did not make the old tree's mixed leaves certifiable. |
| v6.9 public BC | [First quality](warehouse-v69-full-bc-11969-first-quality.md), [pressure](warehouse-v69-frozen-bc-11969-pressure.md) | The frozen dual NN passed exploratory quality and pressure screens; these were not paired PPO or program acceptance. |
| v6.9 program extraction | [Four-part failure](warehouse-v69-program-precheck-11969-v2-failure.md) | Actor-topology programs were faithful inside selected leaves, but both roles lacked the required coverage. The earlier [v1 failure](warehouse-v69-program-precheck-11969-v1-failure.md) is retained separately. |
| v6.10 bounded core Actor | [Quality failure](warehouse-v610-core-12069-quality-failure.md), [root cause](warehouse-v610-core-12069-root-cause.md) | A frozen 96-episode screen exposed repeated collisions and insufficient task quality; later reserved phases were not used. |
| v6.11 public-teacher core refit | [Internal failure](warehouse-v611-refit-core-12169-internal-failure.md) | Four-purpose validation found too little certified core mass after 306 BC resets; the remaining seed blocks were retired. |
| v6.12 masked-NN core refit | [Parity failure](warehouse-v612-masked-core-12269-parity-failure.md) | A batched role-index shape error stopped the locked run after 306 BC resets, before tree fitting or quality evaluation. |
| v6.13 corrected masked-NN refit | [Internal failure](warehouse-v613-role-parity-12369-core-failure.md) | Torch/frozen target parity passed, but role validation coverage was 0% and 11.97%; no quality, PPO, formal, audit, or held-out phase ran. |

The [coverage and shutdown design note](warehouse-v6-coverage-objective-design.md)
records the unregistered failure analysis, a pointwise Actor/tree margin
certificate, and proposed training changes. It is a hypothesis, not an
accepted model or a change to any gate.

Exploratory raw artifacts are stored under ignored, run-specific
`output/warehouse_v6_repro_snapshots/` directories. The later failure reports
name their byte-verified manifests and source closures. The immutable v6 admission
protocol is [`../protocol.py`](../protocol.py); it keeps the registered
development, formal, independent audit, and held-out seed blocks separate.
The sealed v6.0 independent program audit and held-out blocks have not been
used. A later candidate may use a new sealed block only under a new `v6.n`
release ID after the current candidate is closed.


- `warehouse-v617-risk-capacity-diagnostic.json`: retired 6120 scenes only; large structural protection regions but untrained transplant fails delivery/collision/shutdown quality. Not joint-training or admission evidence.

- `warehouse-v617-02-quality-failure.json`: complete BC plus 24-episode quality failure, 2,880-frame physical replay and fixed-program full-frame diagnostic; candidate closed before PPO, no sealed seed used. Original BC sampled coverage and full-frame diagnostic coverage are reported separately.
