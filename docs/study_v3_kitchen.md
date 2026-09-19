# Cooperative Kitchen: two compound recipes

Current engine: `kitchen-v4.0.0`. Frozen scenes: `kitchen-scenarios-v4.0.0`.
This is a deterministic, Overcooked-inspired custom cooperative task. Human input
advances one authoritative simultaneous turn. Rendering interpolates between
saved turns; idle wall-clock time, questions and replay never advance cooking.
The rule controller receives no participant or group identity.

## Layout and controls

Coordinates are zero-based `(column,row)` on the 9×7 map. The outer boundary and
column 4 divide the roles, except for the shared counter at `(4,3)`.

| Human station | Position | Facing interaction position |
|---|---|---|
| Egg cupboard | (1,1) | (2,1), left |
| Tomato cupboard | (3,1) | (2,1), right |
| Meat cupboard | (1,2) | (2,2), left |
| Pepper cupboard | (3,2) | (2,2), right; or (3,3), up |
| Preparation | (1,3) | (2,3), left |
| Human storage | (1,4) | (2,4), left |
| Formal serving plates | (1,5) | (2,5), left |
| Serving hatch | (3,5) | (2,5), right; or (3,4), down |
| Shared handoff | (4,3) | human (3,3), right; AI (5,3), left |

AI stations occupy column 7: stove 1 at row 1, its temporary-plate counter at
row 2, the two-slot ingredient counter at row 3, stove 2's temporary-plate
counter at row 4, and stove 2 at row 5. AI uses each from column 6 facing right.
All seven human floor cells connect; all ten AI floor cells connect. Each role
can reach every station it owns but cannot enter the other role's work area.
Human starts at `(2,3)`, AI at `(6,3)`, both facing right; Task 3 varies the human
starting position among valid floor cells.

WASD directly moves one square and updates facing. A blocked direction turns in
place and still consumes one turn, during which AI and pans also act. E maps to
`interact` and operates only the front cell. The four cupboards are separate,
with no ingredient-selection menu. Space maps to one `wait` turn. A disabled E
interaction does not advance the engine: the server rejects it without mutation
and the UI can display the factual `interaction.label_en` / `label_zh` reason.
Discard is an explicit visible recovery action, with no extra keyboard binding.

## Production state machine

Human roles: collect, prepare, transfer inputs, collect outputs, formally plate,
and serve. AI roles: move ingredients, operate either pan, use its temporary
counters and hand over outputs. Each actor holds exactly one item.

| Recipe ID | Protein first | Vegetable second | Final stage |
|---|---|---|---|
| `egg_tomato` | Egg, 4 cooking turns | Tomato, 4 cooking turns | Combine, 2 turns |
| `pepper_meat` | Meat, 6 cooking turns | Pepper, 4 cooking turns | Combine, 2 turns |

Every ingredient needs two human preparation interactions: whisk egg; chop the
other three. Recipe steps are actual transitions:

1. AI puts prepared protein into an unassigned empty pan, binding pan and food
   to a current matching order.
2. On readiness, AI removes protein to a **temporary plate**. The pan now has
   phase `await_protein_store`; vegetable loading is still prohibited.
3. AI physically carries and places that plate on this pan's dedicated counter.
   Only this transfer sets `was_buffered=true` and unlocks `await_vegetable`.
4. AI loads prepared matching vegetable into the same pan. Other recipe/order
   ingredients cannot enter this job.
5. After the vegetable cooks, AI retrieves the exact stored protein and pours
   it back into its original pan. Mixing preserves both component IDs.
6. Two full mixing turns produce a dish. AI removes it into an **output
   container**, then transfers it through the shared handoff counter.
7. Human takes the container to the serving-plate station and transfers its
   contents onto a **formal serving plate** before serving the bound order.

The three containers are distinct. Plates/containers are supplied without
depletion; their creation never duplicates food. Each dish remains bound to its
production order, including when multiple orders request the same recipe. A dish
cannot silently complete another same-recipe order and render its original
order's other in-progress dish unusable.

Cooked food left in a pan burns after **eight additional full ready turns**.
The turn it becomes ready does not count. Removing/combining during the eighth
interaction phase still saves it. A loading or combination action does not
also spend its first cooking turn. Removed food has no spoilage countdown.
Serving on deadline turn T is accepted before the order expires at the end of T.

Counters never swap or overwrite. The shared and human counters hold one item
each. AI has two ingredient slots and two dedicated temporary-plate slots.
These latter slots may temporarily hold the matching pan's finished output when
handoff is blocked. A finished output occupying a pan's dedicated counter
prevents that pan from starting a new order until the output leaves.

Both actors' interactions are validated against the pre-action state. If both
use the shared counter in one turn, both transfers fail and one turn passes.
A newly placed item cannot be collected by the other actor in the same turn.
A held item can be discarded for one turn without a hidden score penalty.
Burnt food can be cleared in one AI interaction. After a burnt vegetable is
cleared, a usable stored protein and its order binding are retained so the
vegetable stage can be attempted again. Expired orders and unusable materials
have explicit cleanup actions rather than item replacement or teleportation.

## Controller and explanation evidence

AI uses current revealed orders, pan state, physical counters and held items.
It rescues ready protein or finished dishes first, with a hand-freeing commitment
when necessary. It physically stores each first-cooked protein before the
vegetable stage, retrieves that protein for mixing, and preserves finished food
when a blocked handoff requires temporary storage. Loadable current work takes
priority over accepting ingredients that cannot start yet. Early tomato/pepper
is accepted into available ingredient storage; it is neither silently cooked
first nor silently discarded.

Shortest routes include final facing actions and real interactions. The
explanation states the corresponding path distance and available burn window;
when timely rescue is impossible it says so. The controller cannot see the
human's unsubmitted command, future orders or research condition. Decisions,
facts, advice and public-state projection are pure.

`decide` returns `action_label_en` / `action_label_zh` in addition to the common
contract. For `interact`, these describe the actual physical operation, food and
counter where applicable. The public human interaction label describes only the
front-cell operation or why it is unavailable; it never recommends AI strategy.
Authorized QA evidence additionally covers actual next action, actual reason,
held food, recipe stages, both ingredient slots, both temporary plates, timers,
facing-aware routes, current orders, events and public mechanics. Additional facts locate each ingredient in a hand, counter, pan or combined dish; pan content includes recipe, bound order and original component IDs. A separate next-input fact names the ingredient goal and remaining preparation, rather than describing only the immediate movement. These facts use only collected food and visible orders. Every answer
and counterfactual remains tied to the exact selected saved state. Only the
shared service can authorize A-group active Task 2 explanations.

### Public schema additions

- `human` / `ai`: `{x,y,facing,holding}`. Facing is `up/down/left/right`.
- Ingredient/food item: `{id,ingredient,ingredients,recipe,stage,prepare_progress,
  components,order_id,pot_id,container,was_buffered}`. `components` are original
  ingredient IDs; combining retains both IDs. Container is null,
  `temporary_plate`, `output_container`, or `serving_plate`.
- Item stage: `raw`, `prepared`, `cooked_protein`, `cooked_vegetable`, `mixing`,
  `finished`, `plated`.
- `buffers`: `{human:item|null, ai_raw:[item|null,item|null],
  protein:{pot1:item|null,pot2:item|null}}`.
- Pan: `{id,x,y,status,phase,recipe,order_id,item,remaining,ready_age,ingredient,
  burn_in}`. Status is `empty/cooking/ready/burnt`; phase is
  `idle/protein/await_protein_store/await_vegetable/vegetable/mix`.
- Station IDs: `egg,tomato,meat,pepper,prep,human_buffer,plate,serve,handoff,
  pot1,protein1,ai_raw,protein2,pot2`. The two protein stations point to their
  matching pan via `pot_id`; `ai_raw` exposes `capacity:2`.
- `interaction`: `{available,station,label_en,label_zh}`; only the human's actual
  current front-cell operation. `recipes` supplies the two bilingual dish names
  and their ingredient pairings.

Ordinary participant state excludes controller memory, hidden future schedules,
seed, explanation reasons and advisor action. English is the study default;
both item labels and factual explanations have Chinese forms.

## Frozen scenarios and actual validation

The study has 4/6/6 assigned orders and approved budgets 240/360/360 turns for
Tasks 1/2/3. The budgets were retained: all development trajectories fit. Order
deadlines are common across groups: `[130,160,225,240]` for Task 1 and
`[130,170,245,280,345,360]` for Tasks 2/3. Task 3 reveals orders at turns
`0,0,32,64,96,128`; future recipe content remains private until its arrival.

`python -m domains.kitchen.calibrate_v4` freezes/rechecks 24 development seeds
(1000–1023) and 24 disjoint reserved regression seeds (2000–2023), each with all
three tasks. The latter are deterministic regression coverage, not a blind
human-study statistical holdout. The script records all 144 run results and
summary fields in the versioned Kitchen configuration.

| Seed split | Task | Completion | Finished-turn range | Minimum two-pan recipe overlap |
|---|---|---|---|---|
| Development | 1 | 24/24, all orders | 210–221 | 59 turns |
| Development | 2 | 24/24, all orders | 313–335 | 86 turns |
| Development | 3 | 24/24, all orders | 314–336 | 86 turns |
| Reserved regression | 1 | 24/24, all orders | 210–221 | 59 turns |
| Reserved regression | 2 | 24/24, all orders | 313–331 | 86 turns |
| Reserved regression | 3 | 24/24, all orders | 310–334 | 84 turns |

All 144 fixed-controller runs score 100 and burn zero portions. Both pans have
**overlapping active recipe stages**, and both pan IDs appear in actual
successful dishes. This does not mean both pans are simultaneously cooking:
walking between the opposite pans, facing and loading requires six turns,
matching the longest six-turn cooking stage. The stricter
`parallel_cooking_turns` metric is zero and is not relabeled as success.

The separately executed tomato-first trajectory deliberately includes a real
simultaneous-handoff failure, stores the same tomato portion, later combines it
into a served dish, and completes all six orders on turn 325 with no burning or
waste. This is an actual legal human-action sequence with the unchanged AI.

Tests: `python -m unittest tests.test_study_v3_kitchen -v` passes 47 tests,
including 144 complete scene subcases, food provenance conservation, exact
cooking/burn/deadline boundaries, invalid-interaction immutability, raw-input handoff recovery, container-accurate ingredient location, ingredient goals, terminal-frame explanations, facing,
role separation, handoff conflict, capacity recovery, deterministic save/replay,
random legal human behavior and a six-caption real demonstration.

A post-review recovery check starts from the real initial scene and wrongly
hands over each of the four raw ingredients without preparation. Human advice
now retrieves the blocking raw portion, prepares it and resumes collaboration,
while AI retains its cooking-only role. All four separate traces complete six
orders (egg: turn 323; tomato: 324; meat: 298; pepper: 310) with no burning or
waste. When human hand, human storage and an unusable handoff are all occupied,
the explicit discard action provides the only legal hand-freeing path; no food
is overwritten. Ordinary successful calibration trajectories are unchanged.

`python -m domains.kitchen.build_qa_cases` produces **116 bilingual QA cases**
(58 English/Chinese pairs) from executed states: both recipes, early vegetable,
real temporary-plate transfer, correct facing, wrong premises, multi-part
questions, both pan reasons, context follow-up, historical binding, unavailable
interaction and real counterfactual branches. State/decision expectations are
specified independently for critical boundaries. The shared injected-plan test
verifies composition and simulation; **it does not measure the language model's
question understanding**. The Kitchen slice currently passes 119 shared QA
checks. A separately recorded real-provider evaluation is required for semantic
accuracy claims.

The old `python -m domains.kitchen.validation` entry point now explicitly forwards
to the new validator. Historical `validation_results.json` and
`validation_replays.json` are preserved unchanged and describe the former
single-ingredient soup engine; they are not current validation evidence.

No human participants were recruited by these scripts. No simulated score is a
human A/B result. The 50% relative Task 2 improvement remains an unmeasured human
pilot target; mechanism complexity and proxy feasibility do not establish it.
