# Performance rewards — new enrollment protocol

Version `gbp-300-base-20-per-task-v2`, GBP only. Applies equally to A and B.
Each participant plays one assigned game, completing Tasks 1, 2 and 3.
Base pay: £3.00 on completion; each task earns £0.00–£0.20; maximum total £3.60.

| Game | £0.00 threshold | £0.20 threshold |
| --- | ---: | ---: |
| Warehouse | 0 | 100 |
| Cooperative Pong | 45 | 65 |
| Cooperative Kitchen | 60 | 120 |

Each task: `20 × clamp((final task_score − lower)/(upper − lower), 0, 1)` pence,
rounded to the nearest penny, ties upward. Sum the three rounded bonuses.
Pong uses its displayed normalized task_score, not raw points. Warehouse and
Kitchen use their displayed raw task_score. No Task 3 minus Task 1 calculation,
question counts, understanding ratings, tutorial or demonstration scores enter
payment calculations. At a midpoint score (50, 55 or 90), a task earns £0.10.

The complete policy is stored per new instance in `pl3_reward_settings`.
Existing stored v1 policies retain their £2.80 base pay and bonus rules.
Existing instances without that snapshot keep their original promise, including
the prior pilot's £3.00 flat pay. Resuming does not insert a new policy. Saved
completed run scores determine bonus amounts; unfinished runs are excluded.
Research exports include the pseudonymous policy snapshot; Prolific identifiers
remain restricted to the admin payment view.

`GET /api/prolific/admin/payments` adds `rewards` for new-protocol participants,
including completed-task scores/bonuses, total pence and currency. It does not
send money or claim that payment occurred. Review the Prolific submission and
use its submission ID to send the calculated bonus through Prolific; retain the
platform's payment record to prevent duplicate payments. Returned/unfinished
submissions require separate review, not automatic bulk payment.

For the next Prolific study, set the platform's fixed reward to GBP 3.00 and use
the current consent/reward wording. Updating this website does not change an
existing Prolific study's reward or open recruitment. Bonus payments are separate.
Official reference: https://researcher-help.prolific.com/en/articles/445233-how-do-i-send-bonus-payments

Shared Task 2 demos show explicitly unpaid reward previews. Their synthetic
Task 1 scores and calculated figures are demonstration data, not payout records.
