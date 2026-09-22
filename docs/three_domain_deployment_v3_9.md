# Kitchen scoring revision v3.9

Release: `policylens-three-domain-20260922.v3.9`.
Engine/scenarios: `kitchen-v6.3.0` / `kitchen-scenarios-v6.3.0`.
Target: https://policylens-warehouse-study.onrender.com/.
Status: implementation and local acceptance complete; production verification pending.

Kitchen awards 100 points for a valid, on-time serving. Discarding a single
ingredient/component costs 5 points; discarding a combined dish costs 20.
Each valid turn still costs 1, so the corresponding action totals are +99,
−6 and −21. Both actors use the same physical-bin classification. Successful
serving is still required; finishing the pan alone does not earn the reward.

Scoring values propagate to menus, bilingual help, facts, counterfactuals,
event records and exports. Six tutorial sections, preparation/expiry times,
16/20-turn recipe heating totals, menus, deadlines, 360-turn limits and AI
policy are unchanged. Only active Group A Task 2 has Q&A.

New Kitchen enrollments use this score version. Earlier Kitchen instances
are archived without rewriting their scores, frames, answers or tutorial
records; the same participant can begin a separate v3.9 instance. Warehouse
and Pong v3.8/v3.8.1/v3.8.2 instances continue under their saved release IDs.
The release manifest exposes compatibility separately for each domain.
No database migration or deletion is required.

Validation includes score boundaries, late/repeated serving rejection,
physical disposal by both actors, multilingual facts, idempotency, full
three-task permissions, exports and version isolation. Six known scenarios
(seed 1000/1001 × three tasks) were compared with the prior engine: 1,812
non-score transitions and AI decisions match exactly. All five dishes were
served in 299–306 turns, producing raw scores 194–201. These are deterministic
simulation regressions, not human performance or a new held-out evaluation.
Prior v6.2 evidence remains explicitly historical in the configuration.

A local actual-HTTP test exercised real serving and both disposal categories,
repeated the scoring requests to verify idempotency, restored the session and
verified stored frame events and scoring metadata. Synthetic test data stays
separate from participant data. Before deployment, a read-only database check
confirmed 197 completed runs and 24,665 frames were recorded for preservation;
the older 189-run/22,191-frame backup still matched exactly.

Receipts: `analysis/kitchen_v39_20260922/` in the parent workspace.
