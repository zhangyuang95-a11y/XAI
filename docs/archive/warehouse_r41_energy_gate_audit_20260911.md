# Warehouse r4.1 energy-gate audit

Audit date: 2026-09-11. Status: **the running gate is correctly scoped to
`robot_2`, the neural teammate; no program-partner shutdown is charged to the
Actor**. This was a read-only audit through the committed 350k boundary. It did
not change the running trainer, evaluator, checkpoint, Actor, program, scene, or
training receipt.

The machine-readable receipt is
`warehouse_r41_energy_gate_audit_20260911.json` (file SHA-256
`807ccc767b17df78f71d2f07fed124fe6bca77920e52fccc909ab6059da6f8cf`).

## Executable semantics

Both validation suites record three different values per episode:

- `shutdowns` is the environment's team total;
- `ai_shutdown` is `not state.agents[1].active`, which is `robot_2`;
- `player_shutdown` is `not state.agents[0].active`, which is the program
  partner controlling `robot_1`.

The conflict evaluator delegates aggregation to the original evaluator's
`summarize`, so the two suites use the same definitions. The absolute check
named `shutdowns` evaluates
`candidate["ai_shutdown_count"] <= ABSOLUTE_GATE["shutdown_count_max"]`.
Because the registered maximum is zero, the operative requirement is exactly
**zero `robot_2` shutdowns**, not zero team shutdowns.

This distinction matters for the `noisy` profile. That program partner can
randomly command any of the five public actions and does sometimes shut down.
Those events remain in `shutdown_count` and `player_shutdown_count` for
diagnosis, but they do not fail the neural Actor's energy gate. Requiring the
team total to be zero would incorrectly make an Actor fail for a deliberately
perturbed program partner's own energy decisions.

## Saved-evidence reconciliation

Counts below are `team / robot_2 Actor / robot_1 partner`. Every candidate
fails the current zero-Actor-shutdown gate in both suites.

| Actor | Original validation | Conflict validation |
| --- | ---: | ---: |
| r3 baseline | 27 / 19 / 8 | 54 / 49 / 5 |
| r4.1 +50k | 60 / 53 / 7 | 47 / 43 / 4 |
| r4.1 +100k | 40 / 35 / 5 | 30 / 26 / 4 |
| r4.1 +150k | 67 / 63 / 4 | 31 / 25 / 6 |
| r4.1 +200k | 58 / 54 / 4 | 46 / 40 / 6 |
| r4.1 +250k | 72 / 61 / 11 | 42 / 40 / 2 |
| r4.1 +300k | 48 / 44 / 4 | 55 / 50 / 5 |
| r4.1 +350k | 64 / 60 / 4 | 58 / 49 / 9 |

Across all 5,472 saved baseline and candidate episodes available at the audit
cutoff, zero rows violate `shutdowns == ai_shutdown + player_shutdown`, zero
rows disagree between a positive shutdown count and
`terminal_reason == "battery_shutdown"`, and all 16 report summaries exactly
match their episode rows. There are 710 Actor-only shutdown rows, 87
partner-only rows, one simultaneous row, and 4,674 rows with neither.

The separate `skilled` / `fixed_yield` alias affects partner diversity and
aggregate weighting, but it does not change this conclusion: all completed
boundaries already contain `robot_2` shutdowns under nonduplicated partner
profiles.

## Release interpretation

For r4.1 admission, “shutdown rate is zero” should be recorded explicitly as:

> Across every required original- and conflict-suite episode, the deployed
> neural teammate (`robot_2`) has zero shutdown events.

Keep program-partner and team shutdown totals as diagnostics. In the additive
post-training audit and production admission, expose the gate under the clear
name `actor_shutdown_count == 0` while retaining the frozen runner's historical
field names and results. This naming correction must not rewrite any boundary
or change the current run. If the study later needs zero team terminations as a
separate usability condition, register and evaluate that condition separately
rather than attributing it to the Actor.

The non-invasive regression test
`tests/test_warehouse_r41_energy_gate_semantics.py` verifies both the gate
subject and the separate aggregate counters. It passes 2/2 tests.
