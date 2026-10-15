# Pong and Kitchen assets carried into the pending v6 catalog

The `policylens-three-domain-20261016.v6.0` catalog is still **pending** for
Warehouse. Pong and Kitchen retain their accepted v5 frozen Actor pairs,
training reports, and programs. This check did not rerun either domain's
already-used held-out seeds or claim a v6 study release.

For each domain and each of `raw` and `regularized`, the catalog's Actor,
training-report, acceptance-receipt, and (for `regularized`) two program paths
were read from the content-addressed v6 asset directory. SHA-256 of every
file matched the catalog. The carried acceptance receipts both report
`passed=true` and `heldout_evaluated=true`:

| Domain | Acceptance receipt SHA-256 | Frozen assets verified |
| --- | --- | ---: |
| Pong | `ad6e4882964c84cbc565fc72dd33a16e79399f5984b416f84026bf0da1fd9958` | 8 |
| Kitchen | `012af827e8416f293bec97670bb3aec1b752de58dc0155051e98d33c56eabc9a` | 8 |

The pending catalog SHA-256 was
`1c807b44b36b186d5eb4f12d1dd53ea4f694e988ae57d06faebcd798411473b3`.
The six catalog, evidence-cache, no-Torch inference, policy-binding,
Pong/Kitchen frame, and demonstration test modules passed: **35 tests**.
These tests verify migrated execution and binding behavior, including model
identity; they do not replace a successful Warehouse admission or activate
pilot. Kitchen's offline human-agent evaluation remains a proxy result, not a
claim about real participants.
