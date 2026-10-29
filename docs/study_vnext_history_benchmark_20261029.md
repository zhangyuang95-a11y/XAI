# Local frozen-model runtime measurements

These are software engineering measurements on existing public manual scenes, using isolated synthetic researcher databases. They are not model acceptance, participant results, network measurements or free Render measurements. No model weights, physical rules or scientific thresholds changed.

The reusable six-model command is:

```sh
python3 scripts/benchmark_study_vnext_history.py --output output/study_vnext/engineering/NEW_RUN_ID
```

The pinned historical prefix algorithm is retained as `tests/fixtures/study_vnext_history_prefix_reference.py`, exact SHA `4e54f65b86450ad9efe183fe95efb11cf06f3054ff9023c9f44ea8cf82337c68`. It is a nonproduction method reference. There was no captured old Store source, so the comparison uses an explicitly disclosed compatibility wrapper with the current Store. It cannot be described as a literal old website benchmark or establish the original 70% website speedup condition. Existing reports retain their original code snapshots and numbers.

Each model used the domain's first public scene, exactly copied starting SQLite files, 10 warm actions and 30 measured wait actions. Branches ran in separate Python subprocesses, with counterbalanced order and numerical libraries set to one thread. Timing includes complete request asset verification, the action, participant projection and JSON serialization, excluding HTTP/browser/network. Every branch finished all 40 actions. Six pairs matched exactly in executed actions, complete state, public state, score, binding, frame and serialized participant-response hashes.

The first six-model measurement used the new static-record/lazy-history runtime before the additional Warehouse request-local physical witness:

| Model | N | P50 ms | P95 ms | Mean ms |
|---|---:|---:|---:|---:|
| Warehouse raw | 30 | 80.94 | 85.70 | 79.90 |
| Warehouse regularized | 30 | 83.08 | 92.76 | 83.41 |
| Pong raw | 30 | 59.12 | 69.80 | 59.96 |
| Pong regularized | 30 | 59.81 | 69.95 | 60.92 |
| Kitchen raw | 30 | 22.23 | 23.26 | 21.84 |
| Kitchen regularized | 30 | 23.40 | 25.61 | 23.66 |

Detailed receipt: `output/study_vnext/engineering/six-model-history-benchmark-20261029-01/report.json`. Each request read and verified 27 package assets, totaling 10,516,690 package bytes in the request scope, and computed three frozen pair decisions. This byte counter excludes the small separate catalog-file read. The NN counter measures actual compute callbacks for a frozen pair decision in the existing request-local inference runtime, not individual layer calls. The reference and new static-history paths both had three such computations: the gain came from eliminating repeated historical parsing/validation rather than suppressing current NN calculations.

Warehouse remained above 80 ms in that first local measurement. Profiling then identified nine repeated deterministic physical computations per action. The new-study-only `StudyWarehouseAdapter` now executes each distinct full-state/binding/human-action transition once within its request and retains an immutable, bounded witness for duplicate audit calls. Every call still invokes `decide` and checks supplied decisions; replay compares full records and endpoints before reuse. `None` and human `wait` are distinct inputs. Nothing survives the request. Strict stage/export verification and calls without a request scope use the original implementation.

The follow-up controlled Warehouse benchmark compared **the same static-history runtime with the original FastWarehouseAdapter** against **the same runtime with StudyWarehouseAdapter**. Each side again had the same copied DB, explicit public seed 40786, 10 warm actions and 30 samples. All 40 complete states/actions/scores/bindings/frame/participant-response hashes matched in both model pairs.

| Model | Original adapter P50/P95/mean ms | Request-witness adapter P50/P95/mean ms | P95 decrease |
|---|---:|---:|---:|
| Warehouse raw | 80.89 / 84.66 / 80.27 | 50.30 / 52.88 / 50.11 | 37.5% |
| Warehouse regularized | 82.07 / 90.69 / 82.75 | 52.00 / 55.88 / 51.60 | 38.4% |

Detailed receipt: `output/study_vnext/engineering/warehouse-request-physics-benchmark-20261029-01/report.json`. A separate matched one-step diagnostic confirmed two unique original physical computations and seven duplicate validation uses per action, with identical Actor/state hashes. Its cProfile timings were collected during another checker and are diagnostic only; the table above used the controlled idle window.

The physical-witness refusal, immutability, request isolation and strict-bypass unit tests passed (12 tests). The benchmark statistic/count/trace-refusal tests passed (7 tests). Complete affected Warehouse workflows and strict exports were rerun on the final wrapper: regularized/A and raw/C both completed all three tasks, surveys and export verification in 64.584 s and 64.291 s. Receipt: `output/study_vnext/software-20261029/warehouse-request-physics-final-flow-check.json`. The full pre-witness six-model check also completed all 18 task runs; Pong/Kitchen code was unchanged. Pong/Kitchen do not use this wrapper. These results do not claim the network/free-service gate, the 70% old-website condition, scientific admission or pilot admission.
