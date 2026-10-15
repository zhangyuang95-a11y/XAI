# Kitchen v6.2 validation protocol

Current scope: uniform 20-turn prepared-ingredient freshness, +30 serving
reward, fixed five-dish menus without three identical dishes in succession,
and 16/20 heating turns. Task budgets remain 360 turns and deadlines remain
100/140/240/280/360. Earlier v6.1 results do not establish v6.2 feasibility.

## Predeclared scenes

Declared on 2026-09-22 before any v6.2 run of the new validation seeds:

| Split | Exact seeds | Task runs |
|---|---|---:|
| Development | 1000–1023 inclusive | 72 |
| Existing regression | 2000–2023 and 3000–3023 inclusive | 144 |
| Untouched validation | 4000–4023 inclusive | 72 |

Each seed is run for Tasks 1, 2 and 3. The regression split contains 48 seeds;
it is not the continuous interval from 2000 through 3023. Development and
regression are inspected before the new split. The new split is run only after
freezing the gameplay source. If results from that split guide further tuning,
later runs must be described as reused validation, not independent holdout.

## Required measurements

The runner records completion, elapsed turns, raw score, burns, spoilage,
expired orders, ingredient/dish disposal, and score penalties for every scene.
It checks the actual score against `30 × completed − turns − discard penalty`
and the sum of authoritative score events. Negative scores are expected and
are never normalized or relabeled as a percentage.

True parallel heating requires distinct orders, both pans cooking before and
after the same transition, unchanged order/phase bindings, and both counters
decreasing by one. Second-pan loading snapshots cannot satisfy this check.
The report separately records occupied-heating snapshots, actual joint timer
transitions, consecutive intervals, and one complete transition example per
split. Simultaneous cooking need not occur in every scene or under arbitrary
human actions.

The partner is `simulation_partner`, a deterministic test-only helper. Public
participant advice uses `human_advisor` without its optional raw-input recovery
disposal strategy. These simulations do not model novice performance and do not
measure an explanation treatment or guarantee a 50% Task 2 gain.

## Reproduction

```sh
python3 scripts/validate_kitchen_16_20.py --split development --output /tmp/kitchen_v62_development.json
python3 scripts/validate_kitchen_16_20.py --split existing_regression --output /tmp/kitchen_v62_regression.json
# Only after the engine and configuration are frozen:
python3 scripts/validate_kitchen_16_20.py --split untouched_validation --output /tmp/kitchen_v62_validation.json
```

Every report includes exact seed lists, rule values, engine/scenario versions,
a source digest, scene-level failures and unaggregated measurements. The runner
retains failing scenes and returns a nonzero exit status if any required check
fails. QA fixtures are regenerated from current executed states, and incoming
“why this turn” answers are checked against saved pre-action decisions.

## Current v6.2 results

All 288 scenes completed all five dishes under the new menus, freshness and
score rules. All had zero burns, spoilage, expired orders, ingredient/dish
disposals and disposal penalties.

| Split | All five completed | Turn range | Raw score range | Scenes with verified joint heating | Joint transitions |
|---|---:|---:|---:|---:|---:|
| Development | 72/72 | 297–307 | −157 to −147 | 28/72 | 28 |
| Existing regression | 144/144 | 297–308 | −158 to −147 | 59/144 | 59 |
| Predeclared untouched validation | 72/72 | 297–307 | −157 to −147 | 27/72 | 27 |

Each positive scene contains one verified transition with both timers decreasing
together, spanning two consecutive cooking snapshots. This is actual parallel
heating in 114/288 scenes, not sustained concurrency throughout every episode.
For example, seed 4000 / Task 1 has separate pepper-and-meat orders: at turn 46,
the two pans have 8 and 2 turns remaining; at turn 47 they have 7 and 1, with
both still cooking. The complete before/action/after record is retained.

Development and regression passed before evaluating the new split. The engine
and full configuration digests remained identical across all three evaluations.
No gameplay tuning followed inspection of the untouched split. Configuration
validation metadata can subsequently change without changing those tested
rules, menus or engine code.

Engine SHA-256:
`c810e5548268261e1afbd468f0d6bcb3720a905a4ef9ed274df6242fd21dacf2`.

Full measurements and transition examples are retained outside the repository:
`analysis/kitchen_v62_20260922/final_validation.json`. Individual split reports
are `development.json`, `existing_regression.json` and
`untouched_validation.json` in that directory.

The 156 current bilingual QA fixtures are regenerated from executed v6.2
trajectories. They cover the exact prepared/expiry turns, 19/20 boundary,
unchanged identity after spoilage, +29 serving and +28 waiting-then-serving
counterfactuals, positive and negative parallel-heating examples, and actual
incoming-action evidence. New focused tests also verify that public QA never
calls the special simulation-partner recovery policy and that completed-order
counts are not confused with negative raw scores. These are injected-plan
composition tests; they do not measure unrestricted language understanding.

Deployment and database acceptance are reported separately by the release
workflow. These local results alone do not establish that the website is live.
