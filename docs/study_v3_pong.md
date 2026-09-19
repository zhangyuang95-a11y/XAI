# Cooperative Pong: rolling court revision

The engine is `pong-rolling.v4.0`; the scenario generator is
`pong-rolling-schedule.v4.0`. This replaces sequential waves with a deterministic
rolling court. It is an action-driven environment: one accepted input advances
one logical step. Smooth frontend animation does not create a wall-clock game.
The frontend contract is A = left, D = right, Space = wait, with no confirmation.

## Physical and public-state contract

The court has nine lanes, internally 0–8 and displayed 1–9, and twelve vertical
cells. Ball fields are `id`, `kind`, `contacts`, `remaining`,
`initial_remaining`, `y`, and `vy`. `y=0` is the top, and `y=12` is the contact
line. Ordinary balls fall two cells per step and cooperative balls one cell.
Both paddles first move at most one lane, then all balls fall, all arrivals are
settled exactly once, and scheduled replacements enter at the top. Paddles may
overlap. A small ball earns one shared point if either paddle covers its lane;
a team ball earns three only when distinct paddles cover its two distinct lanes.

The initial small balls have 1, 3, and 5 steps remaining; the initial team balls
have 6 and 12. Subsequent small balls arrive every two steps, on odd turns;
team balls arrive every six steps. Thus a team arrival never competes with a
small-ball arrival on the same step. At most three small and two team balls are
visible; the final six steps drain the finite supply.

Task 1 lasts 60 steps: 30 small balls plus 10 team balls, worth 60 raw points.
Tasks 2 and 3 last 90 steps: 45 small plus 15 team balls, worth 90 raw points.
Only balls that can arrive by the final step are generated or counted in the
denominator. The displayed task score remains `100 × raw / scheduled points`.
Task 3 uses its own task seed, mirrored contacts, and reversed starting positions.
`height=12` is included in public state. Legacy `wave` and `wave_count` remain as
six-step progress counters for compatibility; they do not describe sequential
waves and must not be used to detect new-ball boundaries.

## Predetermined workload and controller

The complete finite schedule is generated before play. It has no group,
performance, question, or submitted-action input. Adjacent ordinary arrivals are
two lanes apart, and the intervening team-ball contact lies on the intermediate
lane; the other team contact is separated by at least three lanes and is
reachable from the preceding partner contact. This constructs a busy, jointly
feasible workload. The construction route and future spawns are private and are
not available to the controller, human advice, or explanation service.

The controller enumerates the current team-ball assignments and ordinary-ball
allocations. Feasibility is exact for the present one-dimensional geometry:
ordered contact deadlines must allow each paddle to traverse each distance.
The planning window contains only visible balls, at most twelve steps ahead.
Its priority is: preserve a reachable commitment to the earliest team ball;
catch the largest number of team balls; prefer earlier team deadlines; catch the
largest number of ordinary balls; assign more of those real ordinary catches to
the AI; minimize travel; and resolve remaining ties deterministically. AI motion
always leads to a selected real catch, rather than adding arbitrary movement.

Only the earliest team's feasible division is committed. The later team ball
has an explicitly **tentative** assignment, reconsidered as currently visible
work and positions change. A changed tentative assignment is recorded in
`tentative_replans`, with old/new contacts and the observed-state basis. The
committed assignment is retained until completed or no longer reachable; a
forced switch or release is separately recorded. Explanations distinguish these
two kinds of plan and do not promise that a tentative later side is fixed.

The human coordination proxy takes the human first action from the same
visible-state feasible plan. Each rollout step executes the unchanged real AI
and then replans from the resulting actual state. It neither controls the AI
nor uses the hidden construction route. This is a cooperative planning proxy,
not an exact full-task human-only optimum or a simulation of language learning.
`wave_feasibility` is retained only as a bounded, six-step exact fixed-AI
short-fixture diagnostic. Full rolling tasks do not claim an exhaustive global
optimum.

## Explanation and demonstration evidence

Facts identify each visible ball's points, speed, contact lanes, and arrival
steps, together with the real next AI action, committed and tentative contacts,
and exact deadline comparisons. Small-ball detours must preserve every selected
team deadline. All reachability statements remain conditional on the human
actually moving to the stated contact.

The shared counterfactual runner must stop after the first step that introduces
a previously unseen ball ID. The physical movement and score on that step are
valid, but no subsequent decision may use the hidden replacement's contents.
`balls_spawned` events carry the newly public IDs; they are excluded from answers
about the earlier snapshot. UI interpolation and question reading advance no
logical steps. Group A's active Task 2 is the only explanation-enabled stage.

The six-caption demonstration is generated by the real engine and includes
small catches, a replacement entering at the top, a successful two-person team
catch, and a later team miss. Three bilingual comprehension questions check the
2:1 speed ratio, distinct contacts, and an impossible small-ball detour.

`python scripts/build_rolling_pong_qa_cases.py` reproducibly replaces only Pong's
64 bilingual cases in the shared manifest. Cases cover physical rules, both
plans, actor-specific next actions/advice, false premises, history, follow-up
questions, multi-step counterfactuals, the spawn boundary, and appropriate
clarification. Their injected plans validate evidence composition and are
**not** real-provider natural-language-understanding results.

## Reproducible calibration and limitations

Run `python scripts/calibrate_rolling_pong.py --baselines --output REPORT.json`.
It retains every result for all 24 development and 24 separate held-out seeds,
three tasks, and five proxies. No held-out seed was filtered, replaced, or
rerolled. Gates are mean cooperative-proxy score ≥90, each score ≥80, mean AI
wait rate ≤10%, and each wait rate ≤20%. Hold-position and no-job waiting are
recorded separately. A failure is kept in the output and makes the script fail.

The initial complete evaluation ran 720 actual-engine rollouts. All 144
cooperative-proxy runs scored 100 with zero AI waiting steps. These outcomes
reflect the deliberately constructed feasible catch schedule and coordinated
proxy, **not actual human performance or a guaranteed result for arbitrary
human actions**. The cooperative-only run's slowest measured step was about
1.30 ms locally; the larger baseline run's slowest step was about 12.36 ms,
including decision, proxy, state copying, and settlement. These are local timing
observations, not production latency guarantees.

Held-out Task 2 means (24 seeds per proxy):

| Proxy | Mean task score | Mean AI wait rate |
|---|---:|---:|
| Visible-state cooperative planning | 100.00 | 0.00% |
| Nearest visible arrival | 50.14 | 1.11% |
| Public-history side learning | 77.69 | 16.57% |
| Random legal action | 48.98 | 16.90% |
| Always wait | 50.42 | 17.31% |

The stronger public-history proxy reaches 77.69; 100 versus 77.69 is only about
28.7% relative improvement. These proxies do not represent experimental groups.
They neither establish nor guarantee the requested 50% human Task 2 improvement.
That outcome still requires a real, separately reported human pilot.

The authoritative local records are retained under
`analysis/three_domain_revision_20260920/pong_rolling_calibration.json` and
`analysis/three_domain_revision_20260920/pong_rolling_calibration_baselines.json`
in the parent research workspace. Independent domain tests cover actual physics,
counting, collision-free contact semantics, schedule invariance, two deadlines,
commitment preservation/release, future ignorance, deterministic replay, final
drain, 144 seed rollouts, and real demonstration events.
