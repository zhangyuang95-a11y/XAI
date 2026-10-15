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
| v6.6 joint Actor exploration | [New seed lock](warehouse-v66-joint-actor-seed-lock.json) | Fresh unregistered cohorts reserved for jointly training the tree and full Actor; no result or admission claim. |

Exploratory raw artifacts are stored under ignored, run-specific
`output/warehouse_v6_repro_snapshots/` directories. The later failure reports
name their byte-verified manifests and source closures. The immutable v6 admission
protocol is [`../protocol.py`](../protocol.py); it keeps the registered
development, formal, independent audit, and held-out seed blocks separate.
The sealed v6.0 independent program audit and held-out blocks have not been
used. A later candidate may use a new sealed block only under a new `v6.n`
release ID after the current candidate is closed.
