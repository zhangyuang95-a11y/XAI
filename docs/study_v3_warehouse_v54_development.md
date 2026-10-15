# Warehouse v5.4 development decision (2026-10-15)

The two preregistered development routes each trained paired raw and
regularized dual Actors for 200,000 joint environment steps from the same
route-specific BC checkpoint, using pair seed 6099 and the same scenario
schedule within each pair. Both routes used the sparse oblique program with
at most three public pre-action features per node, depth at most 5, and at
most 16 leaves. Raw program feedback was zero. No program chose an executed
action.

| Development route | Raw selected checkpoint | Regularized selected checkpoint | Gate result |
| --- | --- | --- | --- |
| 256-wide dual Actor | 7.708 deliveries; 3.708 collisions; 8.33% shutdown | 6.542 deliveries; 2.958 collisions; 8.33% shutdown | Model quality passed; independent development program coverage was 0 for both feedback roles and AI explanation. |
| Trainable shallow tree plus 256-wide residual per Actor | 8.250 deliveries; 6.958 collisions; 8.33% shutdown | 8.292 deliveries; 4.667 collisions; 29.17% shutdown | Regularized model quality failed the ≤10% shutdown gate; program pre-audit was therefore not run. |

The 256-wide regularized Actor received 14,134 and 8,360 gated feedback row
presentations for robot 1 and robot 2. The structured regularized Actor
received 48,836 and 43,317. These counts show that both roles received
actual program gradients; they do not establish program fidelity. The
structured program's internal training-episode validation is development
feedback used in leaf selection, not an independent acceptance result.

The development preflight failed for both routes. Per the fixed decision
rule, no training was started for formal pair seeds 6101–6103. Seeds
240048–240071 (formal checkpoint selection), 240072–240095 (formal program
pre-audit), 240100–240123 (once-only independent audit), and the 100 held-out
base seeds per task remain unused. No Warehouse v5.4 final audit or held-out
receipt exists. The active Warehouse v5 policy hashes remain unset and the
three-domain pilot stays closed.

Exact development acceptance snapshots are preserved in
[the 256-wide receipt](evidence/warehouse_v54_dev_mlp_6099_acceptance.json)
and [the structured receipt](evidence/warehouse_v54_dev_structured_6099_acceptance.json).
Their SHA-256 values are respectively
`9c37cbb51d422f63e0c807817d2c3b0a2fed10248fe18bb30259a448aa3dc0b3`
and `a747ce2383dc10f543e644074bf6720fc6f0589305b7124aff3b1201a3e54906`.
The full checkpoints, training reports, logs, Actors, and programs remain in
`output/warehouse_v54_dev_mlp/` and `output/warehouse_v54_dev_structured/`.
The acceptance snapshots are development records, not a v5 release manifest.
The dated [three-domain readiness snapshot](evidence/v5_readiness_2026-10-15.json)
records accepted Pong/Kitchen bindings and the closed Warehouse bindings.
