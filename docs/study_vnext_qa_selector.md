# Study vNext: constrained free-form question selection

`study_vnext.qa` adds an optional language selector over **already verified core
history questions**. This is an implementation and synthetic-test result, not a
real-provider smoke receipt, model acceptance or recruitment authorization.

## Composition API

```python
composer = EvidenceQuestionComposer(provider=None)
rendered = composer.compose(question, verified_core_result, language='zh',
                            public_facts=[], counterfactual_metrics=[])
```

The verified result must use `runtime-v642-committed-question.v1` and carry the
bound model, actual action, committed decision/result coordinates and frame/hash
provenance. The composer does not authenticate users or recompute the Actor.
The caller must obtain the result from the core Store and recheck live A/Task2
permission before displaying or retrieving a rendering. Persisted renderings
must remain attached to their original question/binding/basis hash.

Optional public facts are **typed numeric data**, not arbitrary prose:

- Position: `{kind:'position', frame_sha256:<decision or result SHA>, actor:'human'|'ai', x:<int>, y:<optional int>}`.
- Metric: `{kind:'metric', frame_sha256:<decision or result SHA>, metric:<public identifier>, value:<finite number>}`.

Their provenance must be verified by the caller's bound public-state adapter.
The renderer distinguishes pre-action and result-frame numbers in its text.

## Counterfactual outcome comparison

Optional `counterfactual_metrics` is a list of at most eight records with exactly
`metric`, `factual`, `alternative`, `factual_frame_sha256` and
`counterfactual_state_sha256`. The caller independently derives both finite
numeric values from verified bound public states. Initial support is **one step
only**: the factual frame must equal the recorded result frame, and the alternate
state hash must equal the verified counterfactual's final state hash. Longer
windows, mismatched frames, arbitrary explanatory text and nonfinite values are
rejected before a provider call.

The selector can choose `counterfactual_metric:0` etc. The deterministic answer
reports actual value, simulated alternate value and their signed difference;
it does not label one better or infer a motive. Those numeric values enter the
minimal public catalogue; private snapshots and state hashes do not. The full
comparison provenance is included in the canonical basis hash.

## Optional provider

`OpenAICompatibleSelector(base_url=..., model=..., api_key='', consented=False,
timeout=30, requester=...)` follows the existing OpenAI-compatible selection
transport shape. Configuration is explicit; there are no environment reads or
calls during construction. `consented=True` attests authorization for sending
the question/minimal public catalogue to that configured provider. It does not
attest institutional approval or production readiness. HTTPS is required except
explicit loopback development. Credentials are never included in rendered
answers/audits; bodies, duplicate JSON keys and selector outputs are bounded.

The only accepted provider output is:

```json
{"intent":"reason","fact_ids":["recorded_action","program_path:0"]}
```

Intents are reason, observation, counterfactual and clarification. At most four
supplied IDs may be selected, with intent-specific constraints. Unknown IDs,
extra prose, motive fields, duplicate IDs, invalid JSON or provider failures
produce an explicitly marked template fallback. There is no provider-authored
policy prose. All displayed claims come from local templates.

The provider sees the question and localized fact IDs/text only. It does not
receive private physics/RNG snapshots, Actor logits, program probabilities,
session tokens or full model artifacts. A counterfactual summary covers only
the core's committed historical window and never advances the real game.

## Scope and audit

Every reason answer includes the causal-uncertainty limitation. Carried v5
programs explicitly say **historical development validation only**, including
when the selector chooses a path node without the general program fact. Their
scope is never promoted to a new independent audit.

The result includes provider status, selected IDs, deterministic-renderer label,
question ID/hash, binding hash, Actor identity, decision/result frame hashes and
a canonical basis hash. An unconfigured provider is `not_configured_template_only`;
a failed provider is `provider_failed_template_only`. Neither is claimed to
interpret free-form questions. `production_provider_verified` remains false;
a real configured-provider readiness check belongs to the separate release gate.
