# Script entrypoints

The v6 three-domain participant server starts with
`python3 -m study_v3.server`. Warehouse training and its admission phases use
`python3 -m experiments.warehouse.admission`; phase order and locked seeds are
defined in [Warehouse v6 admission](../docs/warehouse_v6_admission.md).

| Current v6 command | Purpose |
| --- | --- |
| `python3 -m experiments.warehouse.admission` | Develop, train formal pairs, preaudit, claim the final audit and held-out phases, then verify acceptance. |
| `python3 scripts/package_warehouse_v6_release.py` | Package a completed, accepted run into the content-addressed release. It rejects a pending or inconsistent acceptance receipt. |
| `python3 scripts/start_warehouse_v6_candidate_release.py` | Stage a later, still-pending release only after the previous candidate has a closure receipt. |
| `python3 -m study_v3.server` | Serve the three-domain study; pilot enrollment remains closed until all accepted assets are bound. |

The other `scripts/` programs and the root `run.py`, `run_formal_ui.py`, and
`train_rl.py` belong to earlier experiments. They remain importable for old
documentation and regression checks. In particular, the root formal UI is a
single-domain compatibility entrypoint and does not publish a v6 policy.

The historical Warehouse pure-NN v1 Actor is stored at
`output/deployment/warehouse_pure_nn_v1_actor.npz`, with SHA-256
`f67c9f48f91e6020453d56b2f932057eb7bb82d96ea15aeadcc40b0068d2a665`.
Its unchanged path and digest come from
`configs/study_v3_warehouse_neural.json`. The v6 catalog does not use this
artifact; its Warehouse status remains pending until the new admission and
release checks succeed.
