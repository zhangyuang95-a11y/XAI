# Warehouse v6.14 exploratory candidate 12471: closed Stage 1

Candidate `warehouse-v614-certificate-core-unregistered-12471` is closed. Its
locked Stage 1 completed all 1,396 BC and critic environment resets. Frozen
Torch and NumPy Actor routing agreed, but the independent four-purpose final
BC program precheck failed before quality, pressure, PPO, formal training,
sealed audit, or held-out evaluation. Its seeds will not be reused.

The registered internal precheck required each role to have at least 35% core
leaf mass and 30% expected actionable feedback mass. On the validation
episodes, robot 1 had 8.56% and 8.31%; robot 2 had 8.23% and 8.01%. Only one
WAIT leaf per role qualified. For those qualified leaves, top-1 agreement was
98.74% and 100%, with masked KL 0.209 and 0.213. The failure is coverage of
the full frozen NN, not a numerical routing mismatch. Several other leaves
had high action agreement but lacked a path-wide physical legality proof;
large residual leaves had substantial disagreement. No threshold was changed.

Immutable local evidence is under
`output/warehouse_v6_repro_snapshots/warehouse-v614-certificate-core-12471-20261017/`:

| File | SHA-256 |
| --- | --- |
| `stage1-failure.json` | `b2995391968a72f2342bade889d5bbea3ce941a414ad51cc514985b74b5184ce` |
| `reset-audit.json` | `782966604dbff0be06bca50b0131682492a326766e9d8f265b4cb8be83b44ada` |
| `reset-audit.jsonl` | `c6577f341aa6dd6c52de0f5dc461c39f543e286a1946e0e7301db2df08f745b1` |
| `final-bc-internal-precheck.json` | `97f50c122badddca8c26961641e0bffae8481c85bcaf76e007bcc4d97142bfb3` |
| `final-bc-dual-actor.npz` | `12dac6bbde0cbdac0e70df88d6a1015fdfa071b5d3bc8d47d9c30ff701a33595` |

The reset receipt says `status=failed`, `failed_phase=bc`,
`actual_reset_count=1396`, and `exact_prefix_matches=true`. A separate,
unregistered diagnostic on new development seeds 900000000–900000023 found
12.71 mean deliveries, 3.71 mean collisions, and a 4.17% shutdown episode
rate for the frozen final BC Actor. That 24-episode diagnostic is **not**
acceptance evidence and did not open the formal quality gate.

An exploratory program fit on 48 further development trajectories
(900000100–900000147) also remained below the required mass. The saved data
file `diagnostics/final-actor-tree-fit-rows.npz` has SHA-256
`528f12cf7d22b66d87fd04182f18dc36ce5bbf63952527c9ac3c3f49861983a3`.
Increasing the original search beam to 16 and exact candidates per round to 80
was insufficient. A new two-feature legality-gated sparse predicate also
failed the diagnostic: with the default beam, the two role masses were 4.94%
and 5.01%; with beam 16 they were 4.94% and 11.44%. This predicate is a sound
program representation improvement, not a passing method. The fit target was
the full frozen Actor's masked distribution; no teacher actions or sealed
evaluation scenes entered this diagnostic.

The final Actor topology itself proves only WAIT legal on almost every
high-traffic leaf. Even if one picks the best path-certified action in each
existing leaf and ignores KL, margin, and stratum gates, only 7.9% and 9.9%
of the diagnostic frames sit in leaves with at least 96% action agreement.
This optimistic topology-specific calculation shows that strengthening leaf
losses alone cannot meet the 35% internal precheck. A successor must learn a
different topology and full Actor policy together; it must not just refit a
program to the same frozen Actor or loosen eligibility gates.
