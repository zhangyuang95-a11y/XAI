# Reward v19 individual-credit validation

## Root cause and contract

The v18 reward copied one global scalar to both robots. Its potential could
re-match tasks inside a transition, its fixed 16-unit clearance term dominated
ordinary route progress, and the 100-epoch/weight-5 teacher retention loss kept
PPO close to teacher stalls. Consequently neither robot received reliable
credit for its own WAIT or detour.

Production v19 uses, for each robot `i`:

```text
reward_i = user_score_delta / 100
         + 0.01 * (frozen_safe_cost_before_i - frozen_safe_cost_after_i)
         + clip(coordination_cost_before - coordination_cost_after,
                -4, 4) / 100
         - 0.02 * clip(chosen_next_distance_i - best_safe_next_distance_i,
                       0, 2)
         - 0.01 * min(max(avoidable_wait_streak_i - 1, 0), 4)
```

The progress and coordination terms are zero on terminal/shutdown steps.
Necessary charge waits, legal yields, charger queues, real head-on clearance,
urgent charger clearance, and states with no safe progress action have zero
counterfactual regret. The user-study score remains exactly `+100 delivery`,
`-200 collision`, `-50 shutdown`, `-1 step`, and `-2 participant detour unit`.

## Teacher preflight

Command:

```bash
python evaluate_teacher.py --episodes 200 --seed-start 15000
```

Observed on the local CPU implementation:

- accepted: true; mean deliveries 7.51 (min 5, max 10)
- collision, shutdown, deadlock, starvation, loaded-detour, and charger-return
  cycle rates: 0
- avoidable WAIT: 25 / 48,000 robot steps = 0.0521% (gate: <= 0.5%)
- path efficiency `actual / shortest_safe`: 1.04043
- ordinary detours: robot_1 1,064; robot_2 1,227. These include safe
  coordination/energy geometry; loaded avoidable detours remained zero.

## Four-way, three-seed quick proxy

This command was executed as a resource-bounded pipeline check:

```bash
python evaluate_reward_teacher_ablation.py --seeds 41,42,43 --episodes 30 --eval-episodes 5 --horizon 120 --hidden-dim 32 --intent-dim 8 --behavior-samples 256 --retention-samples 128 --quick-proxy --output-root output/collaborative/safe_mission_v28_individual_credit/reward_teacher_ablation_quick_proxy_30ep
```

The proxy scales old/new BC epochs to 10/3 and is far too short to assess
policy quality. Its seed means are recorded, not presented as evidence of an
improvement:

| Variant | Deliveries | User score | Collision rate | Shutdown | Avoidable WAIT | Loaded detours | Path ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| current baseline | 0.600 | -20875.2 | 0.933 | 0 | 0.0278 | 0.000 | 0.667 |
| reward only | 0.600 | -20875.2 | 0.933 | 0 | 0.0278 | 0.000 | 0.667 |
| teacher only | 0.267 | -15386.1 | 0.667 | 0 | 0.3264 | 2.667 | 0.470 |
| combined | 0.267 | -15385.3 | 0.667 | 0 | 0.3250 | 2.667 | 0.470 |

The equality of reward-only and baseline at 30 episodes shows that the old
fixed weight-5 teacher dominates such a tiny PPO run. The high collision/WAIT
rates show that none of these 30-episode actors is deployable. See
`ablation_report.json` in the output directory for every seed and command.

## Reproducible non-proxy runs

The script defaults to three seeds, 200 episodes, 100/30 BC epochs and the
old fixed weight 5 versus the new weight 1 linear decay. Run:

```bash
python evaluate_reward_teacher_ablation.py --seeds 41,42,43 --episodes 200 --eval-episodes 20 --device cpu
```

For the full training budget (substantial GPU time and 12 independent 2,800
episode trainings):

```bash
python evaluate_reward_teacher_ablation.py --seeds 41,42,43 --episodes 2800 --eval-episodes 200 --hidden-dim 512 --intent-dim 64 --behavior-samples 65536 --retention-samples 32768 --device cuda --output-root output/collaborative/safe_mission_v28_individual_credit/reward_teacher_ablation_2800ep
```

Each variant/seed writes to its own v28 directory. The runner never overwrites
an older checkpoint, evaluation report, SQLite database, or study event log.
