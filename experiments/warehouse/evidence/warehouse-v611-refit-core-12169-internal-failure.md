# Warehouse v6.11 exploratory core refit: 12169 closed before quality

The unregistered Stage 1 candidate `warehouse-v611-refit-core-12169-20261016` was locked, independently reviewed, and run once. It failed the predeclared episode-separated core validation for both roles during BC. The process exited with code 1 before producing a frozen Actor or running the 96 first-quality episodes. No pressure, program, PPO, formal audit, heldout, or pilot seeds were used.

## Immutable evidence

| Item | SHA-256 |
| --- | --- |
| 126-file source closure | `517a08932905070613f0e50d9881c0f94104395a4246624693dacd17b1dac443` |
| Source snapshot manifest | `a7912eab0057f00c095272dfee46fa0e90ac02153ea8b75d20577b181772530d` |
| Stage 1 script | `de7fdb1ee5897aa7178b29e0e5e4ccd4802847245c46d7050ce222c38714ea5a` |
| Predeclared run | `809ba9365786f55cefe853380e5f4591bfaef22c468402ecbe9c4adba8d179b1` |
| Seed lock | `0f5af6dcb5cff989d1867e029deb9e5accb2808dad6fc34242ae58490694c4b2` |
| Independent lock review | `2189672d727b1f8bc6472a921d56733bd7794ede9897e11bee7541a3b2e2f642` |
| Core-validation failure receipt | `90bd7815b1157d36f7ba680df37301878c11a4da93d9c979290f17dd08f6d00c` |
| Process failure receipt | `1bd4496f3a0f8fc1ed7fb56e2fc9e3dde411cb91aab0ba920e138ffc6d6b85e2` |

The complete 141-file package remains in `output/warehouse_v6_repro_snapshots/warehouse-v611-refit-core-12169-20261016/`. A byte-identical copy was verified at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v611-refit-core-12169-failed-20261016/`. The seed-use claim binds this one run to pair seed 12169 and the locked script. The failure point establishes that 306 BC scenario resets occurred: 100 initial public-teacher, 80 initial NN-visitation, and 126 topology episodes. The later 1,040 BC scenarios, the two critic resets, and the first-quality, pressure, and program episodes were not reached. This pair seed and all of its remaining reserved blocks are retired for the next candidate.

## Failure diagnosis

The Actor core was fitted to public-teacher one-hot actions. Its fit-split tree agreed with the teacher on only about 60% of rows overall. On the independent eligibility split, each role retained only one eligible core leaf; on validation, core occupancy was 5.73% for robot 1 and 5.89% for robot 2, below the internal 30% target. Even retaining all four fit-eligible leaves would have occupied only 27.29% and 23.29% of validation rows. Robot 1's remaining leaf fell to 57.9% teacher agreement on collision-conflict rows and 83.0% on near-robot rows; robot 2's remaining leaf stayed accurate but occupied too little of the validation split. These are failures of coverage and state-dependent fidelity, not a missing final threshold check.

The next candidate needs a new source lock and new pair/scenario seeds. The current development hypothesis is to train the full dual NN from public BC, then fit the shallow Actor branch to that NN's **complete masked distribution** on disjoint NN-visit episodes. Public-teacher labels remain BC targets; the later explanation program must still be fitted and audited separately against the final complete NN. This hypothesis is untested and does not change the quality, program, or execution gates. v6 pilot remains closed.
