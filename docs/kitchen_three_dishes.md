# Kitchen three-dish protocol

Release: `policylens-three-domain-20260924.v3.26-three-dishes`.
Engine: `kitchen-v6.5.0`; menu: `kitchen-menu-v4-three-dishes`.

Tasks 1, 2 and 3 each contain three visible orders, with deadlines 100, 140
and 240. Each task has at most 240 turns and can finish earlier when its
orders are resolved. Both experimental groups use identical scenarios.
Menus contain both recipes, never three consecutive copies of one recipe,
and differ between the three tasks for each seed. Development and held-out
scenarios are frozen in `configs/study_v3_kitchen.json`.

Task 2 understanding ratings remain mandatory in both groups at 20%, 40%,
60%, 80% and 100%: turns 48, 96, 144, 192 and 240. Early completion uses
the existing final-rating rule. Question guidance and optional subsequent
explanations keep the existing behavior.

Rewards retain the agreed GBP 2.80 base and up to GBP 0.20 per task;
the kitchen bonus score bounds remain 60 to 120. Reward settings already
recorded for previous participants are not changed.

This is a new kitchen protocol. Previous kitchen sessions keep their stored
orders, scores and release IDs, and cannot resume under the new rules.
Warehouse and Pong remain compatible with the previous release. Do not pool
raw kitchen scores across the five-, four- and three-dish versions without
accounting for the different task configurations.
