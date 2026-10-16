# Warehouse v6.12 exploratory run: 12269 closed before tree refit

The unregistered v6.12 candidate used a frozen warmup dual NN as the proposed source of the Actor tree's complete masked-action distributions. Its first lock had a faulty *read-only audit helper* and was never run; its exact zero-reset identity is bound in the second lock. The second lock passed independent source, seed, and script review, then ran once. It exited with `IndexError` in the new Torch-versus-frozen warmup parity check before fitting either role's tree or evaluating quality. This is a code-shape failure, not a model quality or program-fidelity result.

## Evidence

| Item | SHA-256 |
| --- | --- |
| 126-file source closure | `b9e349c81a713eb126700dba9f403fc6ca810e5eeaa7cc90c45e87c5bc1eb72d` |
| Training recipe | `6c801494af6708f86ccb84c125cd0794b7884cb7f69e63e7444d178e9b973464` |
| Source snapshot manifest | `21d368e793bdbfd561e3f175f4769b16d3abd07952569b66efa231b99d538831` |
| Second-lock script | `37941c69970f4dfaf184e6259e65d51da4328768fd5623831c45d63747f626c0` |
| Second script lock | `7644bba425402a18c921ca66485b0de8702df240affe9d217409e9504a861c85` |
| Independent sealed review | `01028c7d41aff0e8702c9727a52f28bac969b302cce7588bd5f3cafa700eabf3` |
| Frozen warmup dual Actor | `a33d7f6f88a9d554cd4d89ed3128536a0b25c4df2997d6916fcfdbae2382d919` |
| Process failure receipt | `3f2f14542e7ac01dc1ed6fb1636f5ace72df662e4e61f30947bace46802eaac3` |

The complete 143-file failed second-lock package remains in `output/warehouse_v6_repro_snapshots/warehouse-v612-masked-core-12269-20261016-v2/`; an independently hashed, byte-identical copy is at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v612-masked-core-12269-failed-20261016/`. The first lock remains separately as an unused predecessor with no `seed-use-claim.json` or training output; its 135 files were also copied and hash-verified at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v612-masked-core-12269-unused-v1-20261016/`.

## Exact stop point

The run consumed the first 100 public-teacher, two 40-episode NN visitation, and eight topology-visit blocks totaling 126 episodes: 306 BC scenario resets. It exported the warmup Actor, then failed while comparing every role's Torch logits with frozen NumPy logits. The parity code passed joint observations to `role_logits()` and received `[rows, 2, 5]`, but indexed that tensor with a role-specific `[rows, 5]` legal-action mask. The next candidate must make the role dimension explicit and test batched Torch/frozen parity without an environment reset before it is locked. No tree fit, risk/deep BC, critic reset, first-quality episode, pressure/program split, PPO, formal audit, heldout episode, or pilot run followed. All of 12269's remaining reserved blocks are retired for the next candidate.

The v6 release remains pending. Passing a corrected tensor check would establish only target provenance; it would not establish delivery, collision, shutdown, program, or execution acceptance.
