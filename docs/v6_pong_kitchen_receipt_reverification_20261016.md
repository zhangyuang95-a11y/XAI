# v6 Pong and Kitchen receipt and asset reverification

This check reads the accepted v5 receipts and the content-addressed copies in
the pending v6 package. It does not run either domain's heldout seeds or claim
new human collaboration results. Warehouse remains pending, so the v6 pilot
stays closed.

| Domain | Accepted receipt SHA-256 | Raw Actor SHA-256 | Regularized Actor SHA-256 | Receipt outcome |
| --- | --- | --- | --- | --- |
| Pong | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` | `1b7d950ba9ff9eef33c6b726eee57eb0ff793af55fcbf192465d9baf09c490d7` | `1da5dd98fde4206b320306e85f98a340476d010ab0f105dc972db446dda999a6` | `passed=true`, 200k steps per variant, zero post-policy overrides |
| Kitchen | `012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a` | `30e4e4b58f99997bba03005061b9d5d9762f59f01c75b42d9d527032390541eb` | `0ecd13da18d00347104a3a6e05456ceb8bb1c10ba41b5621a3fcb44fb4fc5d85` | `passed=true`, 200k steps per variant, zero post-policy overrides |

`study_v3.release_catalog.policy_artifacts()` recomputed and accepted the
content hashes of both Actors, role programs, training reports, and acceptance
receipts, plus the packaged physical scenario configs. The raw and regularized
Actor hashes differ in both domains. The already accepted Pong receipt has 300
heldout rows per variant: raw mean contact capacity 92.0125%, regularized
91.8569%, minimum 83.75%; mean AI wait rate 1.7037% and 1.7704%, respectively,
with maximum 11.1111%. The already accepted Kitchen receipt records 300/300
NN–NN and 300/300 offline proxy completions for each variant. Proxy results are
not participant outcomes.

The migrated registry, demonstrations, frame action paths, policy binding, and
group allocation tests passed: 40 tests across
`test_study_v6_release_catalog.py`, `test_study_v6_pong_kitchen_frames.py`,
`test_study_v6_policy_release_binding.py`,
`test_study_v6_registry_demonstration.py`, and
`test_study_v6_prolific_groups.py` (`python3 -m pytest -q`, 2026-10-16).
This is a runtime contract check using test fixtures; it is not another
heldout evaluation. The earlier Pong heldout replay incident is separately
recorded in `docs/pong_heldout_test_incident_20261016.md`.
