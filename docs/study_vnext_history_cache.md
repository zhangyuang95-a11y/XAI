# Static committed-history validation certificates

This accelerates only the new researcher preview. It does not change the frozen Actors, observations, legal masks, physical transitions, program qualification, scientific receipts or pilot gates. The original `runtime_v642.store.Store` verifier is unchanged.

The process-local cache in `study_vnext/history.py` retains at most 4096 prefix identities. Each identity includes the complete binding, raw persisted run origin and a chained digest of every raw persisted row through that prefix. Prefix identities contain no state, probability distribution or current-head NN output. Only frames before the real run head are eligible: their outgoing decisions have already been physically executed.

Every request still reads every persisted raw byte, fingerprints every typed SQL field and checks the entire chain, origin, task, binding and actual run-head state/status. Length-prefixed field/type/byte hashing avoids repeatedly JSON-escaping embedded JSON strings. Matching certificates permit reuse of static parse/hash validation for unchanged committed records, alongside the existing reuse of historical NN/physics integrity proofs. The actual current head still uses the original strict JSON parser and canonical frame-hash validation, and its NN decision and last physical transition are always recomputed, including when the caller requests `through=0`. A selected historical endpoint is also recomputed. New suffixes receive the same frame/transition checks as the original verifier.

A cold or unrecognized history uses the original full verifier. Changing the raw bytes of a previously certified prefix forces that original full path, even if the attacker recalculates every downstream hash or merely reindents JSON. Current request asset checks and binding verification remain in force. Current program qualification and explanation evidence are never cached by this component.

New certificate identities are staged inside the database transaction and published only after a confirmed commit. A rollback, malformed transaction or uncertain PostgreSQL commit cannot publish them. No cache state is persisted as authority; restart and eviction cause original full verification again.

The separate saved-demonstration cache stores hash-checked canonical bytes from exact immutable DB bundles keyed by the complete binding and exact raw persisted bytes. It is capped at **16 entries and 16 MiB**, including transaction-local staging. The budget uses the larger of the typed source-byte size and canonical payload length. Oversized valid bundles are read and validated normally without caching. Ordinary preparation views consume an explicitly typed, immutable summary containing count and hashes; no partial/fabricated bundle is constructed. Callers requiring a complete bundle receive an independently mutable JSON clone. Whenever NN/physics replay is required, it still occurs.

The v2 static-frame parse cache separately permits **4096 records and 16 MiB**, including transaction-local staging. It stores canonical immutable bytes of already executed, strictly parsed and verified DB frame records. Recorded historical decisions are data from those immutable records, not reusable live forward results. The current head is never retained in this cache. Normal views use a private lazy sequence and only materialize required frames, so returning the current board does not clone three whole histories. Replay explicitly materializes independent dictionaries; exports bypass the cache entirely. Cache eviction and oversized records affect performance only. No parsed tree is shared mutably with a caller, and no program qualification is reused.

Stage transitions, completed-task views, three-task understanding checks and exports use a context-local bypass of the accelerator and the original full history verifier. Exports still independently verify every AI frame. Ordinary replay responses conservatively set `all_AI_actions_recomputed=False` and identify the static history cache; they do not misrepresent cached integrity proofs as newly recomputed neural outputs.

The new synthetic tests cover constant warm-call counts, exact equality with full verification, historical requests refreshing the true head, rehashed data corruption, every raw SQL field/type, raw-byte changes, run/binding changes, changed model assets and current NN output, transactional rollback, mutable return values, frame/demo count and byte eviction, typed summaries and full-verification stage/export boundaries. They are software checks, not model or human-study acceptance.

Executed on the first prefix-only implementation:

- History-specific tests: **15 passed**, 19.97 seconds.
- Complete `tests/test_study_vnext*.py` group: **141 passed**, 90.51 seconds.
- Exact synthetic comparison with nine persisted frames: warm verification made **3** `decide` calls versus **27** under the original full-verification context; returned views were equal. Eight executed-prefix IDs were retained, with no certificate for the current head. The toy saved-demo source occupied 12,011 cached source bytes.

The call-count comparison has no timing-dependent assertion. It does not replace the separate real-model full-flow engineering check or claim a production latency result.

Executed on v2 static parsing/lazy history: **30 history tests passed in 23.20 seconds**. The actual-model engineering profile used the public Kitchen seed **1000 explicitly at enrollment**, the same frozen regularized Actor (`0ecd13da…`), and identical wait actions. State hashes at turns 240, 241 and 242 matched between implementations. The old prefix implementation was loaded from its archived source with a compatibility summary wrapper; its numerical policy and physics were unchanged.

| Controlled local engineering measurement | Archived prefix implementation | Static parse/lazy implementation |
|---|---:|---:|
| Profiled action 240→241 | 477.8 ms | 72.5 ms |
| Next action 241→242, without profiler | 301.5 ms | 43.5 ms |

The unprofiled observation decreased by 85.6%. This is a matched single-action bottleneck diagnostic, not a P50/P95 claim or complete-flow result. The new frame cache occupied about 7.87 MB and demo cache 1.94 MB in that fixture. Detailed profiles, exact source snapshots and state/model pairing are saved in `output/study_vnext/engineering/kitchen-public-1000-paired-static-parse-profile/`.

Earlier unpaired profiles remain in their original directories and are explicitly annotated: their enrollment did not lock the requested seed, so their fixed seed label is not pairing evidence. They are not used for the measured reduction above. A separate actual six-model software check completed all 18 task runs; after adding the request-local Warehouse physical witness, both affected Warehouse three-task workflows and strict exports were repeated successfully. See `docs/study_mentor_implementation_20261029.md` and `docs/evidence/study-vnext-software-summary-20261029.json`. Production engineering admission remains separate; these measurements never grant scientific or pilot admission.
