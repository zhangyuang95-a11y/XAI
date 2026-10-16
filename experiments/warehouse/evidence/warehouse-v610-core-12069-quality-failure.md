# Warehouse v6.10 core Actor: unregistered BC quality failure

Status: **exploratory quality screen failed; v6 pilot remains closed.** The 12069 run trained two independent NN Actors from scratch using a public, train-only sparse core tree and a 0.12 logit residual bound on selected core leaves. It stopped after the prespecified 96-episode first quality screen. It did **not** run the reserved 288-episode pressure screen, program four-part fit, paired raw/regularized PPO, registered development, one-time audit, or held-out evaluation. The failed BC Actor is not a candidate release asset.

The v2 source snapshot contains 123 individually hashed files. Its source closure is `89c19e10fcfd0ca67552e49942b13bef359efd9edafc408b0519b01d4f962b7b`; the frozen dual Actor SHA-256 is `4c04d33b6907cd04d596e29dd7fa39c857c6b5851a5762d0580f3376783a2843`. The pair seed was 12069. The first quality screen used base seeds `1200100–1200131` with tasks 1–3, 96 episodes in all. The run used no post-policy action overrides or illegal executed actions. Torch and frozen NumPy masked argmax matched, with maximum valid-logit difference `5.77e-6`.

| Quality metric | First screen | Prespecified development screen |
| --- | ---: | ---: |
| Mean deliveries | 11.573 | ≥5.5 |
| Mean robot collisions | **7.698** | ≤6 |
| Shutdown episodes | 1/96 = 1.04% | ≤5% |
| Illegal executed actions / action overrides | 0 / 0 | 0 / 0 |

Task 1 averaged 10.406 collisions; tasks 2 and 3 each averaged 6.344. One task-1 episode (`scenario_seed=12001181`) delivered once and collided on 100 consecutive frames. This violates the quality gate, regardless of high average delivery.

## Deterministic posthoc diagnosis

Only those same 96 already-consumed scenarios were replayed. For all 11,496 frames, frozen NN proposals, the stored action chain, episode length, delivery, collision, shutdown, and longest repeated-collision streak matched the original receipt exactly. The replay observed 739 collision frames. Three episodes contributed 133 frames with repeated same-pair actions and collision streak ≥10.

The worst two task-1 loops have the same public positions: robot 1 at `(3,1)`, robot 2 at `(4,2)`. Their frozen NN Actors repeatedly propose `RIGHT` and `UP`, causing `same_target` collision with no movement. Robot 1 routes to core leaf 4 and robot 2 to core leaf 10. The corresponding chosen-action probabilities are approximately 0.9976 and 0.9912; tree and full NN top actions agree. Their residual bounds are approximately 0.086 and 0.083, so the state-dependent residual cannot reverse tree margins of roughly 4–6 logits. The repeated action is an NN output, not an environment override. Across the 96 episodes, this core leaf pair occurs 144 times and collides 134 times. The individual leaves also serve many other public states (944 and 345 frame visits, respectively); changing an entire leaf logit in PPO would change those states together because the split topology is frozen and the core residual is tightly bounded.

This is evidence of a **joint coordination and state partition problem in the frozen BC policy**. It does not prove that every possible PPO run on this architecture would fail, but it makes a costly 200k-step paired PPO from this failed BC start a poor next experiment. The next exploratory candidate will add already-public collision history and NN self-visited conflict states to offline BC, refit train-only core routing before raw/regularized branching, and validate quality and program coverage on wholly new seed blocks. Both Actors remain the sole source of execution actions. The program remains an independently fitted explanation/training object and never selects actions. Current online/final program fit uses an axis-aligned extractor separate from the Actor topology; a future method must explicitly lock one compatible program method for both phases and retain four disjoint episode purposes.

## Identity and recovery

| Artifact | SHA-256 |
| --- | --- |
| Training recipe | `cebca8c787104577bc31fb4a9e98adeacedbf4b09501107e678ac3d0f82fe657` |
| Seed lock / script lock | `b965d28813601b718ca9de89ffabd6e768e913f07b422e55bca6c39a6f08a78e` / `605b1cfe296530ad2a29061f87afc0df5d192ee3d4a6b40df8d0da8573b521ba` |
| Predeclared run / script | `3586fe9a676a9e164c8e1e5944b7c577c1cb8106abc39b3d634f9dfd9d385d51` / `550c3aa8b2560f3e1f3dc2178e1e0488712c584ee28ffb3e9755d51f9a8f2c20` |
| BC Torch state / frozen NPZ | `91be6e8ab004f194f21e7770af2ac3af6dff30e85f9b74f379159a53c92c5edc` / `4c04d33b6907cd04d596e29dd7fa39c857c6b5851a5762d0580f3376783a2843` |
| BC receipt / raw 96-episode result | `e916cb08ced9e9a872bcff6ae7aa73943349e6a54e53a043e74812a88c1c2d2a` / `5c6bd8c5d6c6dd4ec60834269f2b201d4721f93783b9a1e19765b3e85721f750` |
| Posthoc replay script / diagnosis / 11,496 raw rows | `c0dd638613499393e93d181181fe53ba74e3b8f15d72f62faf54313316090e12` / `4216bf660ae702ab6ddf351bb3c1da0e5ed49fbe32c87b8e64e2ce4f9dd31fc6` / `f4f509cc385c1e1418af08ef2fde74a39db1fe9a56b42b658b0eb4e5f960dacc` |

The original ignored snapshot is `output/warehouse_v6_repro_snapshots/warehouse-v610-core-12069-20261016-v2/`. An independently SHA-verified copy, including all 123 source files and the original Torch state, is at `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/warehouse-v610-core-12069-failed-20261016/`: 150 files, 14,727,295 bytes. Its archive manifest SHA-256 is `1779b2f2497978ef4f0c1396cd0937268abbd8b2055c2f49cc174f0272424315`. The earlier v1 lock-only preflight failure is included as an unchanged receipt; it consumed no 12069 scenario seed. The v2 run passed lock-only and consumed only its BC and 96 first-screen scenarios.
