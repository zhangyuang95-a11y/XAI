# Repeated AI-understanding self-report

Release: `policylens-three-domain-20260923.v3.19-understanding-ratings`.
Protocol: `task2-understanding-20pct-v1`.

New sessions in both groups A and B rate their understanding only in Task 2.
The question is “How well do you currently understand why your AI teammate acts
the way it does?” Responses are 1 Not at all, 2 Slightly, 3 Moderately, 4 Well,
5 Completely. Chinese wording and all five labels are provided. Nothing is
preselected. This measures self-reported understanding, not objective comprehension.

Progress uses the fixed turn budget, with checkpoints rounded up:

| Game | 20% | 40% | 60% | 80% | 100% |
| --- | --- | --- | --- | --- | --- |
| Warehouse | 24 | 48 | 72 | 96 | 120 |
| Pong | 18 | 36 | 54 | 72 | 90 |
| Kitchen | 56 | 112 | 168 | 224 | 280 |

An early terminal state triggers one end (100%) rating at the actual ending turn;
unreached intermediate checkpoints are not backfilled. A terminal-state rating
must be submitted before entering the next task or displaying demo completion
without its pending rating. When a rating and explanation share a turn, the rating comes
first. Actions, next-task navigation, and requesting explanations are blocked by
the server until the rating is submitted. The rating is an inline right-sidebar card; Escape cannot bypass the response gate. The game board remains visible, including at the final checkpoint.

`pl3_understanding_settings` persists protocol version and task/group eligibility
at enrollment. Existing sessions without this row remain unchanged. The runtime
flag `POLICYLENS_UNDERSTANDING_RATINGS` defaults to `1` in environment-based
configuration; setting it to `0` affects new sessions only. Programmatic Settings
keeps an explicit opt-in; the disposable share-preview server opts in.

`pl3_understanding_ratings` stores one row per run/checkpoint, including instance,
run, task, checkpoint percentage, actual turn, turn budget, protocol, creation and
submission timestamps, rating, and response language. Join instances for domain,
group, mode, and release. Both new tables are included in authenticated JSONL/CSV
research exports with the existing release/mode filters. Pending prompts survive
reconnects; exact command retries cannot duplicate a response. Rating-card time
uses the separate `understanding_rating` timing category, not `active` gameplay.

Validation includes both groups across all three domains, Task 1/3 exclusion,
legacy session preservation, response validation, ownership, duplicate retries,
refresh persistence, export, early-ending schedule, and explanation ordering.
The browser smoke walks all five Pong checkpoints, checks keyboard blocking,
refresh, unselected choices, bilingual/mobile presentation, and final completion.

As of v3.22, new sessions with `pl3_prompt_settings` may skip ratings and continue
playing. The schedule and 1–5 scale stay the same. Skips are in `pl3_prompt_skips`;
they must be treated as missing responses, not low ratings. Legacy sessions keep
the previously required-response protocol described above.
