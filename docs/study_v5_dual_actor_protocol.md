# V5 dual-Actor, frozen-teammate protocol

Release identity: `policylens-three-domain-20261014.v5.0`.

Each domain trains **two distinct Actor weight sets** on the same joint task.
Both actions in formal self-play are sampled from their respective neural
Actors through the domain's physical legal-action mask. A centralized critic
and offline behavior-cloning initialization are training-only. The formal PPO
stage has no teacher-action override, online teacher imitation, or handwritten
collision-risk auxiliary. The regularized variant alone receives bounded,
role-specific program feedback; the raw variant receives none. Both variants
in a seed pair start from the same pretrained two-Actor checkpoint and use the
same environment seed schedule and 200,000 joint-step budget.

The Warehouse candidate receipt records the shared BC start hash for each
role, both variants' initial and selected role hashes, positive weight-change
measurements, and 400,000 Actor decisions for 200,000 joint steps. For every
role and candidate, the release gate checks that feedback rows do not exceed
feedback-window Actor rows and that zero-action-coverage rows equal the
difference. Raw feedback and exposure counters are zero; the regularized
variant has positive audited feedback for both roles. A physical legal-action
mask with no available action or any post-policy intervention fails admission.

Three paired seeds are trained per domain. Development scenarios select a
checkpoint and the paired deployment seed. Fresh final scenarios are evaluated
once. A domain passes only when **both** variants satisfy its preregistered
cooperation gates, both role Actors changed in training, the regularized
variant has nonzero audited feedback for each role, the raw variant has zero
feedback, and neither variant has post-policy action overrides. The acceptance
file is written only after its artifact and evidence hashes are rechecked.

## Participant flow

| Group | Frozen pair | Task 2 questions |
| --- | --- | --- |
| A | Program-regularized dual Actor | Available |
| B | Same exact pair and hashes as A | Unavailable |
| C | Paired raw dual Actor | Unavailable |

Pilot assignment is balanced and random within each domain and scenario seed;
participants cannot select a group. Researcher preview/test can select A, B or
C. The instance binding records the immutable policy ID, pair artifact hash,
both role hashes and program hashes. The group never enters the neural
observation. In every domain, the bound pair first runs an AI–AI demonstration.
The participant then replaces the human-side Actor for all formal tasks while
the AI-side Actor remains frozen. Kitchen's separate, clearly labeled practice
uses a neutral waiting AI and is not an AI–AI demonstration or formal task.
Optional human-action advice in formal questions is the frozen human-side
Actor's suggestion, not a rule-based task allocation or collision planner.

For each formal turn, the AI action must equal the frozen AI Actor's output
after only the physical legal-action mask. Physics still resolves collisions,
movement, orders and scores. Programs, explanatory text, planners and rescue
rules cannot change the selected action. Replay and Task 2 counterfactuals use
the same pinned actor pair. A program path appears as explanatory evidence only
if its saved audit is eligible and its action agrees with the NN on that state;
otherwise answers may report verified NN output and public outcomes without
inferring an internal motive.

Question records retain the bound policy ID, pair hash and both role hashes.
When a selected answer uses an eligible program fact, its audit also records
that program's SHA-256. Participant views and public frames carry policy
provenance; researcher exports include the instance binding, frames, questions
and the accepted release manifest.

## Admission and historical data

The v5 pilot remains closed until all six policy artifacts, three domain
acceptance files, release hashes, persistent storage and real question-service
checks pass. Missing or mismatched assets fail closed. Older runs remain
readable and exportable under their original release IDs; they cannot resume
as v5 turns or count toward the v5 treatment comparison. Human collaboration
benefit is unmeasured until the new participant study is completed.

## Current development gate status (2026-10-15)

Pong and Kitchen completed three paired raw/regularized seeds at 200,000 joint
steps per variant and passed their one-time fresh test evaluations. Kitchen's
offline human proxy is only a proxy, not evidence of human collaboration.
Warehouse also completed all six 200,000-step runs. The quality-based paired
seed selection chose 6001; its raw and regularized checkpoints passed the
development task-quality gates. Its fixed bounded programs failed the separate
development coverage gate: robot 1 feedback covered 0.85% of states, robot 2
feedback 12.69%, and robot 2 explanation 2.79%, versus 20% required for each.
The [development-only re-extraction probe](../output/deployment/warehouse_v5_reextraction_dev_probe.json)
did not find a passing bounded program. Smaller and semantic-input Actor
[development probes](../output/deployment/warehouse_v5_semantic_actor_dev_probe.json)
also did not justify another full paired run. Warehouse's independent final
program audit seeds and all 100 fresh seeds per task remain unused. Its model
hashes are not installed in the v5 policy catalog, so the entire v5 pilot and
release manifest remain closed. See the
[Warehouse acceptance report](../output/deployment/warehouse_dual_v5_acceptance.json)
for the exact candidate and gate measurements.

## Warehouse v5.4 development outcome

The two planned 6099 development routes each completed matched 200,000-step
raw and regularized self-play. The 256-wide route passed task quality but
failed sparse-program coverage. The structured tree-plus-residual route had
real regularized feedback for both roles but failed the regularized shutdown
quality gate. Neither route passed its development preflight, so formal
6101–6103 training, the once-only final program audit, and the held-out test
set were not run. The [v5.4 development decision](study_v3_warehouse_v54_development.md)
and its two receipt snapshots record the measurements. The v5 pilot remains
closed; offline proxy results are not claims about human collaboration.
