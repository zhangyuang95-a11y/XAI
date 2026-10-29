# Researcher preview deployment — 2026-10-30

This deploys software preview only; scientific model admission and recruitment remain closed.

- Website: https://policylens-warehouse-study.onrender.com/
- Research flow: `/study/`; existing `/warehouse/`, `/pong/`, `/kitchen/` manual playground retained.
- Render service: `srv-da66ggbl550s738j2q9g`.
- Deployment: `dep-dbhnv1on74is73eiohcg`, commit `2b98d872a0bf350e85fec82df24de0e37cc01028`.
- Rollback: prior live `e77d8d22ce373e9719f1867a31ec0c5cf050337e` (Render history retained).
- Runtime: `research-preview-composition.v1`; original manual runtime v2.3 retained.

The public router hosts unchanged manual and research handlers on private loopback ports.
Their authorization and database connections stay separate. The new research database is
an explicitly named ephemeral preview SQLite file; legacy participant/PostgreSQL environment
variables are never used. Free Render restarts can discard preview sessions. This is not
persistent study storage. The existing private operator credential authorizes researcher
enrollment; no new credential or public recruitment permission is created.

A/B share the exact regularized binding; C uses raw. Warehouse remains actual v6.42 20k
(raw failed paired calibration); Pong and Kitchen remain historical v5 200k. No weight,
physical rule, acceptance threshold, sealed evaluation or training is changed by this deploy.
Questions use local grounded templates only. No external question provider is enabled.

## Verified

- 5 focused real HTTP composition tests and 11 frontend checks passed before deploy.
- Live release and status report exact deployed commit, preview mode, template-only
  questions and closed pilot/recruitment gates.
- Existing manual playground: all six models and 67 live requests passed exact action,
  binding, public state, physics, retry/restore and independent replay comparisons.
  Receipt: `evidence/study-deploy-manual-six-models-20261030.json`.
- New research flow: Warehouse A/C/B and Pong A/C completed live preparation, bound NN–NN demo,
  three human-WAIT actions, idempotence, restore, replay and action-secrecy checks.
  These are five software cases, not quality episodes. The supplementary nine-case
  sweep was stopped after these completed cases to release free preview capacity;
  new-flow Pong B and Kitchen full live sequences were not claimed complete. Their shared
  flow had been checked locally, and all six models passed the separate live manual
  check above. Partial receipt: `evidence/study-deploy-research-nine-cases-final-20261030.json`.

Two exploratory new-flow checks were explicitly interrupted while awaiting cold stage
verification (172.669s, one completed Warehouse A case; 100.977s, no completed case).
Their original receipts are preserved under `study-deploy-research-nine-cases-*`.
They are incomplete deployment checks, not scientific failures. The old manual check's
first TLS interruption is separately preserved. No uncertain POST is automatically retried.

No Render performance pass or final scientific/human-study admission is claimed here.

## Deployment limitations

The browser automation session could not capture the public page (repeated timeouts);
no old local screenshot is substituted as proof of live rendering. HTTP static bytes,
release identity and actual model execution were independently checked. Research flow
first entry is slow on Free Render: the same 120-frame Warehouse demonstration is
generated/revalidated about six times across preparation boundaries, with cold demo
requests observed around 47 seconds. This is a remaining performance limitation, not
a relaxed integrity check. Original manual pages remain the readily accessible play path.
