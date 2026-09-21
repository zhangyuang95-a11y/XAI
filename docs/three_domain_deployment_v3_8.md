# Kitchen operations tutorial and rules v3.8

Release: `policylens-three-domain-20260922.v3.8`.
Engine/scenario: `kitchen-v6.2.0` / `kitchen-scenarios-v6.2.0`.
Menu: `kitchen-menu-v2`. Tutorial: `kitchen-operations-tutorial.v1`.
Target: https://policylens-warehouse-study.onrender.com/ (existing Render service).
Publication status: local implementation and validation; deployment receipt pending.

## Participant behavior

Kitchen now begins with seven interactive operation exercises: movement/facing,
pantries, preparation, handoff/storage, a separate already-finished dish fixture,
freshness/trash, and menu/deadlines/scoring. Exercises use real keyboard-triggered
kitchen transitions with a neutral waiting teammate. They never run the cooking
policy, demonstrate a recipe sequence, or contribute formal task turns/scores.
Completion advances each section automatically; start/pause/retry/skip controls
remain. Public expandable help is identical for A/B and excludes AI strategy.
Warehouse/Pong demonstrations and mechanics remain unchanged.

Every Task has five dishes and 360 turns, with deadlines 100/140/240/280/360.
Menus are seeded, vary across Tasks, and prohibit three consecutive identical
dishes. The two recipes heat for 8+6+2=16 and 10+8+2=20 turns respectively.
Preparation takes tomato/pepper3, egg4, meat5 interactions. Prepared ingredients
expire exactly20 turns after preparation in any pre-pan location. Transfers never
reset the timestamp. A load submitted for the expiry turn is refused. After a
successful load the pan uses cooking and eight-full-ready-turn burning rules.
Unprepared raw ingredients retain the previous rules.

Serving awards30 and the same action costs1 (net+29); every turn costs1, actual
component disposal costs3 and combined dish disposal10. Spoilage/burning/expiry
alone add no penalty. Five completed dishes award150 before time/disposal costs;
raw scores may be negative. The two-turn handoff grace/cancellation remains.
Only groupA during active Task2 can ask/view answers; explanations of a performed
action use its saved pre-action evidence. English is the initial page language.

## Persistence and isolation

The existing PostgreSQL path is retained. Additive `pl3_tutorials` and
`pl3_tutorial_events` tables persist independent sandbox snapshots and command
records, including starts, pauses, retries, skipped/completed flags, and physical
practice transitions. Formal runs/frames continue to record rule metadata, actual
seeded menus, ingredient identities/timestamps, before/after actions, scores,
events, questions/counterfactual evidence and questionnaires. Export includes the
new tables with release/mode filtering and credential omission.

A new release ID prevents old Kitchen states from resuming under changed rules.
No old row or +100-era score is recomputed. Deployment preparation exported the
existing study data consistently (23,264 frames among75 participants/99 instances,
230 runs; private backup outside the repository). No credentials enter the commit.
All automated online enrollments must use test mode; browser QA uses preview mode.

## Verification

See `kitchen_16_20_verification.md` for current, separately reported development,
regression and predeclared new validation evidence. These are simulated partners,
not human study results or evidence of a50% explanation benefit.

The final deployment receipt will record the exact commit, source digest,
readiness, online flow/QA/export checks and persistence across another deployment.
