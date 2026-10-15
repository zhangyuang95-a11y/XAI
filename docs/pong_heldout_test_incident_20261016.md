# Pong held-out test incident (2026-10-16)

During the Pong v6 config-packaging change, an extra legacy regression command
was run after the focused v6 release suite passed. The command was outside the
planned verification scope because the first module contains parameterized
rollouts over the previously used Pong held-out seed list:

```bash
python3 -m pytest -q tests/test_study_v3_pong.py tests/test_study_v3_pong_leader.py tests/test_study_v3_pong_qa_v37.py tests/test_pong_v5_binding.py
```

The command exited 1 with **44 failed, 33 passed**. Most failures assert the
older rule-planner behavior against the current frozen NN and are not a new
Pong acceptance result. The affected parameterized test in
`tests/test_study_v3_pong.py` runs tasks 1, 2, and 3 over the config's
`heldout_seeds` list (731100–731123) and stops within each case at its first
failed assertion. The test output does not identify the last seed reached in
each case. Seed 731100 was necessarily reached for all three tasks; later
seeds in that range may also have been reached. Several other legacy tests
explicitly construct seed 731100 for task 2. The corresponding development
list is 730100–730123. No Warehouse sealed audit or held-out command ran.

The existing Pong acceptance receipt and both Actor files have the same
SHA-256 before and after the command. “Before” is the committed `HEAD` blob;
“after” is the workspace file read following the command:

| Existing file | Before | After |
| --- | --- | --- |
| `domains/pong/rolling_actor_v5.acceptance.json` | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` |
| `domains/pong/rolling_actor_v5_raw.json` | `1b7d950ba9ff9eef33c6b726eee57eb0ff793af55fcbf192465d9baf09c490d7` | `1b7d950ba9ff9eef33c6b726eee57eb0ff793af55fcbf192465d9baf09c490d7` |
| `domains/pong/rolling_actor_v5_regularized.json` | `1da5dd98fde4206b320306e85f98a340476d010ab0f105dc972db446dda999a6` | `1da5dd98fde4206b320306e85f98a340476d010ab0f105dc972db446dda999a6` |

No original model, acceptance receipt, held-out claim, or selection was
rewritten. The test command did not run a trainer or release acceptance step;
its temporary fixture files used pytest temporary paths. Workspace status
showed only the intended Pong config implementation, catalog, documentation,
and test edits. These extra legacy test results **must not be used as v6
evidence** or to change a model or release selection. The separate focused
six-module v6 release check passed **37/37** before this incident and remains
the relevant config-packaging verification.

To prevent accidental replay, the two active legacy planning-controller
modules named in the command were removed from default test collection. Their
contents were already preserved, byte for byte apart from an archive header,
at `tests/legacy_pong_v6_engine_checks.py` and
`tests/legacy_pong_v6_leader_checks.py`. The archived files now skip the
held-out parameter and explicit 731100 rollout cases even if run by path.
No active test imports either removed module. The two held-out parameters in
`tests/test_study_v3_revision_review.py` are skipped; a QA test that
immediately replaced its initial Pong state now constructs a development seed
instead. Current frozen-NN behavior remains covered by
`tests/test_pong_v5_binding.py`, `tests/test_study_v6_pong_kitchen_frames.py`,
and the v6 release-catalog tests. The study enrollment contract still checks
assignment to the reserved scenario list without using that check as a
performance evaluation.
