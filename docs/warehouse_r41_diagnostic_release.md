# Warehouse r4.1 diagnostic release status

Status: `blocked_no_diagnostic_deploy_due_explanation_hard_gate`

This work targets an internal `r4.1-diagnostic` release of the terminal
2,000,000-step Actor. Behavioral performance is waived for this diagnostic
only. Actor action authority, explanation and counterfactual fidelity, and the
six-scene conflict contract remain hard gates. The explanation gate failed, so
no diagnostic admission, questionnaire, release archive, Secret File, or
Render deployment was produced. The existing r3 service remains online.

## Frozen diagnostic identity

- Actor SHA-256:
  `4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b`
- Designation:
  `output/warehouse_native/r41_diagnostic_release_20260912_v4/diagnostic_actor_designation.json`
  (SHA-256
  `d80f2736c6d5e359764f6ae4c09ccfa277f26e98a8910c1f18d86d84cb99851f`)
- Conflict manifest:
  `output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/manifest.json`
  (SHA-256
  `af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c`)
- Corrected dynamic selection:
  `output/warehouse_native/r41_diagnostic_dynamic_selection_v3r1_20260912/report.json`
  (SHA-256
  `a0e332769baf6c0c90e9162c7051a54065afff73a23e44d2874e87d695f70418`)
- Selected six scenes:
  `output/warehouse_native/r41_diagnostic_dynamic_selection_v3r1_20260912/selected_scenes.json`
  (SHA-256
  `30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8`)

The designation keeps the historical r4.1 failure ledger and closeout
unchanged and records
`behavior_performance_gate_passed=false`,
`behavior_performance_gate_waived=true`, `formal_ready=false`, and
`formal_sample_eligible=false`.

## Hard-gate result

| Gate | Result | Evidence |
| --- | --- | --- |
| Actor action authority | Pass | 5,118,522 Actor submission frames, zero overrides; selected-scene physical replays also satisfy `policy_action == submitted_action` |
| Six-scene conflict contract | Pass | 11 dynamically eligible scenes; X/Y select six distinct conflict families; 720 selected-scene replay episodes; no endpoint-on-agent spawn, immediate task recreation, or ordinary-task fallback |
| Explanation and counterfactual fidelity | **Fail** | Fresh observation-grouped RCPD has zero train/validation observation overlap, but effective intervention-direction fidelity remains below 85% |

The independent selector receipt is
`output/warehouse_native/r41_diagnostic_gate_closeout_20260912/selector_audit/receipt.json`
(SHA-256
`1628e76ef153e7cf19cf8c0ad2231e6422ad828cea24d2661e9e9646723518c5`).
Its final audit report SHA-256 is
`3ff27149c0d665e8932d9f88370d83235ecbd71fbab123fd29126d972141eb61`.

## Explanation result

The fresh RCPD collection contains 109,938 rows: 88,609 training rows and
21,329 validation rows. Exact observation overlap between the two splits is
zero, intervention anchors stay in one split, and `final_test` was not read.
The saved reader independently refit and authenticated all 36 registered
candidates.

The best registered d8/32 candidate that passes the other fidelity gates has:

- overall action fidelity: 91.67%
- non-WAIT action fidelity: 91.02%
- narrow-passage fidelity: 93.04%
- shared-pickup fidelity: 91.75%
- shared-charger fidelity: 93.07%
- effective intervention-direction fidelity: **74.79%**

The requested expanded d10/d12 and 128/256-leaf probes were also run. The
best explicit quadratic model tree passes the other action gates but reaches
only 80.23% intervention-direction fidelity. The highest-direction honest
oblique candidate reaches 82.34%, while its overall fidelity is only 89.14%.
A final bounded public-state symbolic table reaches 77.07%. No neural scorer,
Actor internals, hidden states, logits, final-test data, action controller, or
runtime override was used to manufacture a pass.

Canonical evidence:

- official RCPD report:
  `output/warehouse_native/r41_diagnostic_rcpd_v3_final_20260912/report.json`
  (SHA-256
  `cf0291df2d21bacc5b2b41972b900fdacb2826e56caea1ef351752582e9afb18`)
- official candidates:
  `output/warehouse_native/r41_diagnostic_rcpd_v3_final_20260912/candidates.json`
  (SHA-256
  `ba7508adc4232d6cef2c18f2bbabbe808aebe362ebd13e3baf7d357ff9b6f6a1`)
- immutable explanation closeout:
  `output/warehouse_native/r41_diagnostic_gate_closeout_20260912/explanation/closeout.json`
  (SHA-256
  `43ed2ad74fb217e0a0bce724c05561ec6931c43fe5bbb4e30ee5a8cc23c18deb`)
- explanation closeout receipt:
  `output/warehouse_native/r41_diagnostic_gate_closeout_20260912/explanation/receipt.json`
  (SHA-256
  `5636425414b46cb4bac90ba398329d44524bea166b70b390d3e3e20db4bc98e6`)
- independent explanation audit receipt:
  `output/warehouse_native/r41_diagnostic_gate_closeout_20260912/explanation_independent_audit/receipt.json`
  (SHA-256
  `8c2276d2cf948b81853c904656cfe222716f3857c78b17a0c6242a225ea70f66`)
- expanded-probe summary:
  `output/warehouse_native/r41_diagnostic_rcpd_v3_probe_closeout_20260912/probe_summary.json`
  (SHA-256
  `711686fc76d8535026ca7fc1a713cb2922a876f4f809347c0719724081e4ba08`)
- last-chance symbolic receipt:
  `output/warehouse_native/r41_diagnostic_gate_closeout_20260912/explanation_search/receipt.json`
  (SHA-256
  `b710af6e446b9c099ce644ac412ee0aae2faa1e5d0de664f1c0e12fad0393aa1`)

## Implemented release surface

The diagnostic source path is implemented and tested behind the failed hard
gate. It includes the old research UI, 380 ms continuous movement animation,
the instruction and AI-AI tutorial flow, A-group Task 1 live and historical
questions, B-group playback-only access, Task 2 answer isolation, separate
diagnostic cookie and SQLite namespace, participant-safe public metadata, and
explicit ephemeral-storage notices in Chinese and English.

The release would publicly expose only
`release_version=r4.1-diagnostic`,
`pilot_class=internal_diagnostic`, `formal_ready=false`,
`formal_sample_eligible=false`, and `data_persistent=false`. Actor hashes,
training failures, scene seeds, and scene fingerprints remain management-only
data.

## Verification

The final source verification completed with:

```text
152 Python tests passed in the evidence workspace with private artifacts
142 Python tests passed and 10 private-artifact tests skipped in a clean checkout
17 frontend animation, tutorial, history, and permission tests passed in a clean checkout
37-file release-source allowlist scanned with no secrets, database URLs,
participant data, or absolute user paths
```

`render.yaml` remains byte-identical to the r3 configuration (SHA-256
`977db584085692a515017a540cb28296aeba684c037ce7565a341024905243fd`).
It still points to the preserved r3 package and does not invoke the diagnostic
loader. Automatic deployment remains disabled.

To reconsider deployment, first produce an explicit program that passes all
registered validation gates, then run the independent final-test explanation
audit. Only after that audit passes may the frozen question bank, neutral
tutorial, diagnostic admission, archive, Secret File, receipt, Render
preflight, and manual deployment be generated.
