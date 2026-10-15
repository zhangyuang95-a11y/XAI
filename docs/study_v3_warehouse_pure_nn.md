# Warehouse pure-NN study release

The current `study_v3` Warehouse engine is `domains.warehouse.neural_turnbased`.
Its frozen Actor is `output/deployment/warehouse_pure_nn_v1_actor.npz` (SHA-256
`f67c9f48f91e6020453d56b2f932057eb7bb82d96ea15aeadcc40b0068d2a665`).
The older `domains.warehouse.turnbased` adapter and its v68 artifact remain in
the repository for historical interpretation; they do not serve the current
Warehouse route.

During formal self-play, both robots receive the same shared-parameter Actor.
Each Actor call receives the public pre-action state in ego/partner order:
positions, battery, active/carry state, headings and previous executed actions,
the two visible shared orders, wall map, charger and elapsed time. The Actor
does not receive environment navigation goals, goal assignments, coordination
plans, hidden intentions, rule scores or a human surrogate. A centralized
Critic is used only in training. The action mask permits every physically
passable direction and `WAIT` for active robots, and only `WAIT` for inactive
robots. A moving teammate's current cell is not removed from the action space.

Offline imitation labels from the historical controller and differentiable
collision-risk supervision are training aids. Neither replaces a formal
self-play action. The selected checkpoint was reached after 120 additional
formal two-Actor episodes initialized from a **new raw-state imitation**
checkpoint (never from v68). Its training report records 27,516 Actor
decisions, zero proxy-human overrides and zero post-policy interventions. The
PyTorch checkpoint is `warehouse_pure_nn_v1.pt`; the NumPy export matches its
logits within 1.91e-6 on the export check. The precursor checkpoint and its
training report are saved as `warehouse_pure_nn_v1_pretrain.pt` and
`warehouse_pure_nn_v1_pretrain_training.json` under `output/deployment`.

At runtime, the same frozen Actor independently samples both agents' actions
in the AI–AI demonstration, or only `robot_2` after the human takes over
`robot_1`. Per-agent seeds make the action from a given state and seed
reproducible. `decide`, `step`, replay and what-if simulation use this same
model. The AI action passed to the environment is the sampled AI action;
walls, collisions, battery, orders and scoring are resolved by the environment.
Missing, mismatched or corrupt artifacts fail closed. Explanations give the
observed action probability and physical counterfactuals rather than claiming
a hand-written assignment or avoidance motive.

Held-out evaluation uses seeds 1000–1023 and **simulated** human profiles,
not participant data. The `warehouse_pure_nn_v1_evaluation.json` report gives:

| Partner | Mean deliveries | Episodes with a delivery | Mean collisions | Shutdown episode rate | Mean raw score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same NN | 3.71 | 23/24 | 15.29 | 29.2% | -349.63 |
| Coordinated surrogate | 5.96 | 24/24 | 7.58 | 8.3% | -199.00 |
| Goal-directed surrogate | 6.46 | 24/24 | 7.67 | 16.7% | -175.13 |
| Cautious surrogate | 5.46 | 22/24 | 7.42 | 16.7% | -194.38 |
| Hesitant surrogate | 6.38 | 24/24 | 7.71 | 12.5% | -210.25 |

The model performs basic delivery in most AI–AI episodes but still collides
frequently. This is evidence of limited learned cooperation, not robust
collision-free coordination. The simulated-human rows are development
diagnostics, not a human study result. Older participant data retain their
original release identity and cannot be used to claim a pure-NN treatment
effect. New A/B sessions share this Actor; only Task 2 explanation access
differs. Older Warehouse sessions remain readable/exportable but cannot
resume actions across the release boundary.

The QA cases for this release are in `domains/warehouse/neural_qa_cases.json`.
They can be regenerated with
`python3 -m domains.warehouse.build_neural_qa_cases`. The held-out report can
be regenerated with `python3 -m scripts.evaluate_warehouse_pure_nn`.
