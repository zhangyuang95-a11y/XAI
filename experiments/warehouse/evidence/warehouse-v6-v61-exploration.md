# Warehouse v6.1 exploratory model study

This record describes **unregistered development probes**, not Warehouse model admission. The v6.1 Actor, role program, public lookahead BC, corrected PPO cost scale, and training-only reward calibration were saved in commit `5764751`. Public enrollment remains closed. No formal pair seed, registered development pair (6120–6122), independent audit seed, or held-out seed was evaluated by these probes.

| Probe | Unregistered evaluation | Mean deliveries | Mean robot collisions | Shutdown episodes |
| --- | --- | ---: | ---: | ---: |
| Tree-initialized Actor after two BC/DAgger rounds | 989000–989023 | 9.79 | 13.75 | 8.33% |
| Supervised state gate after two BC/DAgger rounds | 989100–989123 | 9.96 | 16.71 | 8.33% |
| v6.1 Actor before learned collision gradient | 989200–989223 | 10.88 | 13.00 | 16.67% |
| Same Actor, collision gradient weight 1.0 | 989200–989223 | 11.04 | 5.33 | 12.50% |
| Same Actor, collision gradient weight 2.0 | 989200–989223 | 11.04 | 4.88 | 8.33% |

The collision critic was trained from actual `env.step` collision events on disjoint unregistered position pairs; it changed training gradients only. Frozen inference still took both actions from the two NN Actors through physical legality masks. It reduced collisions below the development limit of 6 in this small probe, but none of the rows met the required shutdown rate of at most 5%.

The supervised gate used both low and high values rather than saturating everywhere; hard-tree occupancy reached 12 and 15 leaves. That repaired the candidate-02 tree collapse but did not fix joint sampled safety. A separate 989300–989323 observation diagnostic found the joint probability of the public teacher's actions was 0.632 overall, 0.549 when the robots were within two steps, and 0.747 outside near-neighbor and low-return-margin states. These exploratory fractions are not program audit coverage.

The shutdown trace isolated five failed episodes across the two collision-gradient settings (989203, 989215, 989217, 989221, 989222). In the clearest cases, a robot repeatedly moved back and forth near the charger, spent battery without a collision, then chose a move away from a feasible charger route. In seed 989215, robot 1's battery fell from 24 to 12 over frames 74–80 while its public charger return margin fell from 7 to 1; the robot later shut down. This points to a long-horizon charging failure rather than a defect in the one-step collision classifier.

The next architecture probe will expose, for each candidate action, public-map shortest distance to the charger and orders plus battery return margin, and test a learned long-horizon action critic trained on real future events. Those values are observations and training signals, never a runtime action replacement. Method locking requires full registered paired training, independent multi-trajectory stress evaluation, and program audits under the unchanged gates.

Raw exploratory scripts and per-episode data are isolated under ignored `output/warehouse_v6_*`; the committed candidate-02 failure receipt and diagnosis remain separate from this report.
