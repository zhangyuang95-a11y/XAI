# Warehouse guarded-tree development probe (unregistered)

This is a diagnostic on the **closed** 12471 frozen Actor, not a new v6
candidate, paired self-play run, model admission, or human study result. It
uses only new development scene seeds. All formal, sealed-audit, and held-out
seeds remain unused, and the pilot remains closed.

## Why the probe was run

The 12471 precheck showed that almost every high-traffic tree path proved only
WAIT legal. On 48 new development trajectories, we fitted a five-node
decision-list topology from **offline public teacher labels**. Each branch
combines one public feature with the corresponding physical legality mask.
Interval arithmetic proves that taking a branch implies that its nominated
movement is legal. The branches are part of a candidate NN tree architecture;
they never replace an executed Actor action by an external controller.

With a 65% minimum teacher-action purity heuristic, the five guarded leaves
covered 43.51% of robot 1 and 41.35% of robot 2 frames in these development
trajectories. This is only a structural coverage diagnostic; it is not program
fidelity, independent validation, or PPO feedback coverage.

To test whether architecture replacement alone could work, we spliced that
topology and small core residual bounds into the closed Actor **without
retraining**. On matched new development seeds 900000200–900000223, both runs
used the same action RNG convention:

| Frozen dual Actor | Mean deliveries | Mean collisions | Shutdown episodes |
| --- | ---: | ---: | ---: |
| Original closed 12471 | 12.92 | 4.08 | 1/24 |
| Guarded-tree splice, no retraining | 4.67 | 15.63 | 1/24 |

The splice fails both development quality thresholds (deliveries ≥5.5,
collisions ≤6). Frequent robot conflicts show that broad static-action leaves
cannot simply replace a coordinated policy. No quality threshold or
independent program criterion was changed. The next method must jointly
train the topology, leaf outputs, residual, and true-event shutdown objective,
or use a more expressive auditable leaf policy; this direct splice is closed.

Local diagnostic evidence under
`output/warehouse_v6_repro_snapshots/warehouse-v614-certificate-core-12471-20261017/diagnostics/`:

| File | SHA-256 |
| --- | --- |
| `final-actor-tree-fit-rows.npz` | `528f12cf7d22b66d87fd04182f18dc36ce5bbf63952527c9ac3c3f49861983a3` |
| `public-teacher-rows.npz` | `c333a6b186a12ccbbf67bb382f517a24306445d4b4021d9b8f9f0afa91a1193b` |
| `guarded-probe-actor.npz` | `7ad2a27c9f6444b0b541365125b334e0599f9e1bc5db79ff89d89ca4f4f8ebdb` |
| `guarded-probe-quality.json` | `096ce09d70481b47558377473198d7c128c10c70042ffad9e03a722deb9af31c` |
| `original-actor-matched-quality.json` | `338180e6377e0c0c635de5ebb601b2e3098888a3ef244e56d4f067f466cacacd` |
