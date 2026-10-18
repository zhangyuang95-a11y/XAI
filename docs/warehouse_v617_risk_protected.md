# Warehouse v6.17 risk-stratified Actor candidate

Candidate `warehouse-v617-risk-protected-02` targets the still-unused v6.0
sealed audit. v6.16 is closed; its original model bytes and failure receipt
remain unchanged. Pilot stays closed pending the complete admission pipeline.

The Actor has a public risk root (collision history or return margin), a
public near/far split, at most two learned sparse levels per side, and at
most seven dedicated movement legality guards. This fits 15 nodes, 16 leaves
and depth five. The derived risk bit costs its two source measurements
against the three-feature node limit; a legal bit is a separate node.

`protected_leaf_mask` is selected only on fit training episodes from legal
paths and episode support. It belongs to the neural Actor, is checkpointed
and content-hashed, and caps each protected residual at
`min(0.08, max(0, (masked_tree_gap - 0.26) / 2))` in both Torch and frozen
NumPy inference. The learned tree margin target is 3.0. The independently
qualified `core_leaf_mask` is a training/evidence sidecar: installing it does
not change the Actor hash or output. It must be a subset of protected leaves.

The explanatory program fits the full masked five-action NN distribution.
Protected leaf logits are fitted in a 0.04 box around the centered Actor
tree logits; calibration stays inside that box. This yields a conservative
KL bound of 0.24, but does not replace empirical leaf support, fidelity,
coverage, legality, or task-quality gates. Fit/core qualification and program
eligibility remain separate episode purposes. Failed evidence never changes
the frozen NN or enables an explanation.

BC refreshes rebuild the topology using only the fit partition of training
episodes, followed by four fit-only alignment epochs. PPO refreshes at
24/48/72% use current NN rollouts; no online teacher actions are requested.
Risk/context/legality nodes remain immutable. Topology and protected leaf
outputs freeze at 75%. Optimizer moments for frozen rows are cleared.
The complete BC precheck is followed by a quality screen before any PPO.
Full multi-trajectory pressure evaluation still follows paired PPO.

New development pairs are 6140–6142; selection 263100–263123, pressure base
263200–263455, program audit 263500–263523. The formal pairs 6101–6103 and
v6.0 sealed sets remain unused until development passes. Budgets remain
200k/300k/400k; a BC failure stops the candidate before PPO.

The retired-scene capacity diagnostic is explicitly non-admission evidence.
It shows large protected regions but a severe quality loss when the new
tree is transplanted without joint training. Therefore the registered
candidate starts BC from scratch; diagnostic weights cannot initialize it.

Entry point: `python3 -m experiments.warehouse.admission develop --output-root
/absolute/new/run/root --candidate-id warehouse-v617-risk-protected-02
--program-fit-method actor_risk_stratified_protected_topology_v3`.
The native backend creates immutable recipe/source-bound authorization and
reset journals. Ordinary training calls remain closed. No runtime action
controller or program can replace a frozen NN action.

Candidate 01 was interrupted after 424 audited resets because the new quality-screen phase was absent from the reset phase allowlist. Its partial training is retired, without a scientific verdict. Candidate 02 fixes and directly tests that hook, uses fresh development scenes, and starts from scratch.

## Candidate 02 result

The complete BC run on seed 6140 is closed. It used source commit `a5cb4d9`,
source closure `97efd0bfa2eb3476e66b4e5b055a8d3d418bd10aae2b721fef6aac0f3fb1cc03`,
and 1,420 audited resets. No PPO, formal pair, sealed audit, or holdout ran.

Both roles passed the inherited sampled BC program precheck: program leaf
coverage 35.48% / 67.27%, expected actionable coverage 32.69% / 63.52%,
100% top-one agreement, KL 0.000421 / 0.000274. That collector deduplicates
collision rows; these figures must not be labeled a formal all-frame audit.

The pre-PPO quality screen failed: 24 episodes averaged 1.083 deliveries,
30.042 robot collisions, and zero shutdown episodes. Independent physical
replay verified all 2,880 frames, frozen NN draws and reported outcomes.
The candidate therefore closed automatically before either PPO arm.

An additional diagnostic reproduced the fixed programs from the cached
four-purpose data and evaluated every recorded quality frame. Coverage was
34.72% / 67.85%, actual feedback-actionable coverage 31.56% / 64.10%, with
100% agreement, zero illegal actions and KL 0.000400 / 0.000241. This reuses
recorded development trajectories and is diagnostic only. It does not
constitute a new independent test or justify model selection.

The structure improves local program fidelity, but this trained candidate
does not preserve task quality. Strong protection on broad regions must not
be promoted. A later method must preserve delivery and collision quality;
its BC screening must also retain every physical frame rather than report
the inherited deduplicated sample as all-frame coverage. All acceptance
thresholds and the closed pilot remain in force. The complete failure summary
and original asset hashes are in
`experiments/warehouse/evidence/warehouse-v617-02-quality-failure.json`.

After closure, the unexecuted `.r3` recipe corrects BC precheck collection to retain every physical frame in all four purposes. Both receipt validators now reject excluded frames, and a repeated-collision test verifies that independent rows are retained while training deduplication remains available. Candidate 02 results remain bound to `.r2` and commit `a5cb4d9`; they are not reissued under the corrected source. No replacement training run has started.
