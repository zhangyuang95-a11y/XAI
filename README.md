# PolicyLens: three-domain cooperation study

This repository contains Warehouse, cooperative Pong, and cooperative Kitchen.
The current integration candidate is `policylens-three-domain-20261016.v6.0`.
**Pilot enrollment is closed** until the Warehouse paired training, independent
program audit, held-out evaluation, and three-domain runtime checks all pass.
The earlier v5.0 candidate is retained with its failed Warehouse evidence;
it is not an approved study release.

## Control contract

Each domain trains two independently parameterized neural Actors in AI–AI
self-play. A demonstration uses both frozen Actors. A participant takes over
one role, while the other role continues to use its bound frozen Actor. The
environment applies physical action legality, collisions, and scoring.
Programs provide independently audited training feedback and explanations;
they never select an executed action. Groups A and B bind the same regularized
Actor pair; C binds its matched raw pair. Only A can request explanations in
Task 2. The Kitchen practice exercise is a separately labeled neutral-AI
tutorial and is outside the formal demonstration and tasks.

No human collaboration benefit is claimed from offline proxy evaluations.
Historical instances, frames, and receipts remain readable under their own
versioned contracts.

## Repository map

| Path | Purpose |
| --- | --- |
| `env/warehouse/`, `domains/pong/`, `domains/kitchen/` | Physical environments and domain-specific observations |
| `backend/training/warehouse_v6_runner.py`, `warehouse_v6_actor_training.py`, `warehouse_v6_collision_critic.py` | One v6 Warehouse dual-Actor BC and PPO pipeline with training-only risk critics |
| `backend/training/warehouse_v6_actor.py`, `warehouse_v6_program.py`, `backend/inference/` | NumPy-only frozen Actor and audited program inference, including archived v5.4 reading |
| `experiments/warehouse/` | Immutable seed blocks, phase gates, and one-use audit receipts |
| `study_v3/` | Versioned participant flow, binding checks, replay, Q&A, and exports |
| `releases/<release_id>/` | Content-addressed, hash-checked release package and catalog |
| `docs/evidence/` | Earlier candidate and failure evidence; never a substitute for v6 audit |
| `tests/` | Model, release, frame-action, and historical-compatibility checks |

The older Warehouse trainers in `backend/training/` are historical candidate
implementations. New v6 runs use `warehouse_v6_runner.py` with one explicit run
specification; their outputs are isolated under a candidate and run ID. Do not
add the ignored `output/` experiment tree wholesale to Git.

## Local verification

```bash
python3 -m pip install -r requirements-render.txt
python3 -m pytest -q tests/test_warehouse_v6_protocol.py tests/test_warehouse_v6_public_observation.py tests/test_study_v6_release_catalog.py tests/test_study_runtime_no_torch.py
python3 -m study_v3.server --host 127.0.0.1 --port 8010
```

The server may run in research preview while the admission gate keeps pilot
enrollment closed. The v6 release catalog must contain exact Actor, program,
training-report, and audit hashes before formal play can start. See
[v5 protocol](docs/study_v5_dual_actor_protocol.md) for the prior experiment,
and [Git recovery inventory](docs/git_v6_branch_inventory_20261016.md) for
the preserved dirty checkouts and branch archive. The prior top-level README
is preserved in [the historical archive](docs/archive/README_pre_v6_20261016.md).
