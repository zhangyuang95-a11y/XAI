# Warehouse v6.17 risk-stratified Actor candidate

Candidate `warehouse-v617-risk-protected-01` targets the still-unused v6.0
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

New development pairs are 6130–6132; selection 262100–262123, pressure base
262200–262455, program audit 262500–262523. The formal pairs 6101–6103 and
v6.0 sealed sets remain unused until development passes. Budgets remain
200k/300k/400k; a BC failure stops the candidate before PPO.

The retired-scene capacity diagnostic is explicitly non-admission evidence.
It shows large protected regions but a severe quality loss when the new
tree is transplanted without joint training. Therefore the registered
candidate starts BC from scratch; diagnostic weights cannot initialize it.

Entry point: `python3 -m experiments.warehouse.admission develop --output-root
/absolute/new/run/root --candidate-id warehouse-v617-risk-protected-01
--program-fit-method actor_risk_stratified_protected_topology_v3`.
The native backend creates immutable recipe/source-bound authorization and
reset journals. Ordinary training calls remain closed. No runtime action
controller or program can replace a frozen NN action.
