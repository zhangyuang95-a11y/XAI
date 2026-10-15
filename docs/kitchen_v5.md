# Kitchen v5: paired frozen neural teammates

The formal v5 Kitchen policy has two independently parameterized Actors. In
AI–AI demonstrations they control the human-side and AI-side roles; during
participant tasks the human replaces the human-side Actor. Both variants use
the same physical action mask and Kitchen transition rules. No task assignment,
avoidance, or rescue controller selects an execution action.

`configs/study_v3_kitchen_v5.json` pins a raw and a regularized pair separately.
Its hashes are populated only after the full experiment and independent audit.
The adapter rejects an explicit v5 `policy_id` if its weights are missing or
mismatched. The earlier v4 deployment remains available only for existing runs
and direct legacy calls.

The six matched training runs and their development report are preserved at
`output/kitchen_v5_matched_candidates`. The original in-loop feedback trees
remain in that report. The frozen pair's two separate explanation programs
were re-extracted only from training-only NN–NN scenes, using the final frozen
Actor distributions as labels. Their metadata records the original feedback
program hashes; the explanation programs remain depth 5 with at most 16 leaves.
The final program admission audit uses the untouched seeds 41000–41019 once,
before any held-out rollout. Its receipt pins both final program hashes and
both role Actor hashes. The first two stages below are one-use operations:

```sh
python3 -m domains.kitchen.finalize_v5 validate
python3 -m domains.kitchen.finalize_v5 heldout
python3 -m domains.kitchen.audit_v5 \
  --pair-report output/kitchen_v5_matched_candidates/kitchen_dual_nn_v5_reextracted_pair_report.json
```

For each of seeds 7, 11, and 19, both variants start from the same BC-trained
two-Actor weights. Each variant collects exactly 200,000 on-policy joint steps,
with both actions sampled from its neural Actors. The scenario schedule is
indexed by the global joint step in fixed 360-step blocks, matching the Kitchen
engine's episode horizon. If one policy finishes a scene early, it restarts
that same seed and task for the rest of its block. Raw and regularized therefore
have identical seed/task exposure at every joint step. Each run records the
SHA-256 of the actual expanded 200,000-step schedule, and the final audit
reconstructs and checks it. The first six candidates trained with episode-
indexed seeds are retained under `output/kitchen_v5_unmatched_episode_schedule`
as invalid development evidence; they cannot be frozen for v5. PPO has no online teacher
anchor. The regularized branch extracts separate bounded programs for the two
roles from pre-action observable states, and adds gated KL after 80% of the
budget, reaching `lambda=0.01` at 85%. The raw branch uses no program feedback.
Program extraction uses episode-disjoint validation and is limited to depth 5
and 16 leaves. A program can give training feedback only after its fidelity
gate passes and it agrees with the Actor's selected action.

The model development set starts at seed 40000 and contains 20 seeds across all
three tasks. It selects the pair seed maximizing the worse variant's completed
episode count, provided both variants pass 80% with NN–NN and the offline
human proxy. The training-time feedback trees were audited on that set and
failed the final explanation threshold. They were retained as failed
development evidence, while the selected frozen Actor was unchanged. The
later, training-only re-extraction produced new bounded programs. Both final
role programs must pass the independent 41000–41019 admission audit with at
least 80% action agreement and mean KL at most 0.35. The participant
explanation reads the AI role's final-Actor audit only after both roles pass,
with each result bound to its Actor and program hashes. Only a pair passing
both audits is tested once on fresh seeds
50000–50099. The resulting acceptance JSON checks at least 240/300 completed
episodes for each variant and pairing. It records model, report, and program
hashes, feedback counts, the selected seed, and zero controller overrides.
The held-out evaluation writes a one-use receipt before its first episode;
rerunning training refuses a second held-out evaluation, including after an
interrupted first attempt.
If any gate fails, the v5 pilot stays closed; proxy results do not measure
performance with actual people.
The held-out command first writes `passed: false` while it pins the artifacts,
then runs the independent artifact and evidence audit without replaying held-out
scenes. Only that audit can set `passed: true`; a failed candidate never
replaces an accepted release.
Restart the study server after a successful artifact audit so its immutable
policy catalog is loaded from the newly pinned manifest.

The separately labeled Kitchen tutorial is a practice activity with a neutral
AI. Its scripted wait action is outside the formal AI–AI demonstration and
participant task flows.
