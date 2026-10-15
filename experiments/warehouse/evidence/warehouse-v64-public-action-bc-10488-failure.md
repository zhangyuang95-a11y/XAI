# Warehouse v6.4 public-action BC probe: program precheck failure

This is **unregistered exploratory evidence**. It is neither a formal paired
PPO result nor an independent final audit or held-out test. No registered or
sealed Warehouse v6 seed was used. The two BC arms (`none`, `tree_gate`) used
pair seed `10488`, identical initial robot_1/robot_2 weights (SHA-256
`abe89cd19cb52c729e267c0ad2c0c5687c55073514921c936c9ecb7db0e07fa7` /
`018b15bb1d21334ed2db655d9c88c3765f5f529d5a9cc26ff5ea1060cfdf361a`),
and identical BC scenario schedule SHA-256
`ab63922dd74ffc4ef93942cd15eb7b150e395482e1201bdd2882c41bef2fce13`.
The unchanged training source closure is
`248b11fe3800344ef3ebdaabda38b6dc8f54675b917faa636a533d9cc057c2a7`.

The probe appended 13 public, pre-action geodesic and battery measurements
for each of five candidate actions to both NN Actor and program inputs. These
measurements are derived from the public map, positions, batteries, and order
endpoints; no goal choice or action override enters execution. Both frozen
Actor pairs passed the independent 288-episode pressure quality precheck,
but the prelocked 50/25/25/24 fit/calibration/eligibility/validation program
split failed the required **30% actionable coverage in each role**:

| BC arm | Frozen pair SHA-256 | Pressure deliveries | Collisions | Shutdown episodes | robot_1 feedback / explanation | robot_2 feedback / explanation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `none` | `95946ea51e05c1ed29c02eaf201dfdcc7ab36e015d7d6a065aa5d8aeb63a42bc` | 15.632 | 3.146 | 0.35% | 27.19% / 27.26% | 26.49% / 26.49% |
| `tree_gate` | `9d336d2ca32e90e79bb7626a215529812d82bafa6804330765fef14b8a57e5cd` | 14.611 | 5.108 | 3.13% | 26.67% / 26.71% | 33.82% / 31.26% |

On the main four-part validation, both arms had covered agreement at least
97.7%, KL at most 0.05, and zero illegal raw program actions. These passing
components do not offset the failed coverage. A separate, prelocked 1061xxx
method comparison evaluated exact Actor tree, fit-only pruning, axis-only
extraction, and fit-only sparse oblique extraction. None met the separately
declared 35% actionable target for both roles. For the `none` Actor, axis-only
was nearest at robot_1 feedback 29.55% and robot_2 30.59%; its validation has
now been consumed for method comparison and cannot be reused as an independent
method-lock test. The complete comparison, including explanation coverage and
the distinct unregistered diagnostic audit, is in `comparison_summary.md`.

The BC receipts, both frozen Actor pairs, extracted programs, comparison
receipts, scripts, and changed source files are preserved in the ignored
snapshot `output/warehouse_v6_repro_snapshots/warehouse-v64-action-features-bc-10488-20261016/`.
Its 48-file, 3,620,308-byte inventory `archive_manifest.json` was verified
file by file and has SHA-256
`87d408502dd48e3509a40efa003d1108269cec8ce888c0f8dc7db50ff6b86e65`.
The comparison summary SHA-256 is
`19d75443566851f0ff5c01a48afc7c8a085ebdb8492b1530f85b662f4ba05efa`.
Raw exploratory outputs are intentionally not added to Git wholesale.
