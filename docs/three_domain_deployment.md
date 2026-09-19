# Three-domain production deployment

The three-domain release is live and public pilot enrollment is open at
https://policylens-warehouse-study.onrender.com/. This is an actual Render
deployment, not a local demonstration. The Task 2 human improvement target of
50% has **not been measured**. All acceptance records use test/preview identities.

## Deployed identity

| Field | Verified value |
|---|---|
| Release | `policylens-three-domain-20260920.v2` |
| Deployed commit | `68a50da906e30a3ac2f7712858cc2b19d2cd159a` |
| Canonical source SHA-256 | `387513ef55e9d2c5ca68c061159b389bc281b5bad88ef4972a399ea960de9858` |
| Final pilot configuration deploy | `dep-dandd3mgekts738jt3lg` |
| Render service | `policylens-warehouse-study`, `srv-da66ggbl550s738j2q9g` |
| Region / runtime / plan | Singapore / Python 3 / existing Free plan |
| Branch | `codex/three-domain-study-live-20260920` |
| Root directory | Repository root (blank setting) |
| Build | `python -m pip install -r requirements-render.txt` |
| Start | `python -m ui.domain_hub_server --host 0.0.0.0 --port $PORT` |
| Health path | `/health` |
| Automatic deployment | Off; releases deployed manually at a specific commit |

The Render dashboard reported the final deploy **Live** on 2026-09-20 (Asia/Shanghai).
The public `/api/release` independently reported the same commit and source hash,
`mode=pilot`, `storage_persistent=true`, `semantic_qa_configured=true`,
`deployment_validation_complete=true`, and `study_ready=true`. `/health` reported
`status=ok` and `study_ready=true`. The timestamped response is saved outside Git
in `analysis/three_domain_build_20260920/production_release_v2_pilot.json` in the
parent workspace. Later documentation-only commits do not change this deployed
commit or imply another deployment.

The public routes are `/`, `/warehouse/`, `/pong/`, and `/kitchen/`. Assets and
`/api/study/*` share the same origin. All domains use Demo → Task 1 → Task 2 →
Task 3 → Questionnaire, English by default, and the same persistent explicit
Chinese language switch. Only active Group A / Task 2 can ask or read answers;
finishing Task 2 already revokes access before the participant clicks Next.

## Acceptance and persistence

The frozen v2 source passed 538 tests and 144 subtests. Six actual production
HTTP flows (three domains × two groups) completed all tasks and questionnaires:
1,695 requests including the recovery-probe capture, 1,450 human-role test moves,
18 task runs and six questionnaires, with no flow/state/permission failures.
Real provider questions, private audit and researcher export were exercised on
the same website. This is software acceptance evidence, not human-study evidence.

The native Chrome production check completed the full Pong A workflow and
checked language persistence, select/confirm keyboard input, a real Task 2
answer, two-tab terminal revocation, Task 3 without explanations, questionnaire
draft recovery and completed-state recovery. Kitchen B's six demo steps and
five real task moves were also visually checked on v2. Browser coverage is
recorded precisely in `production_browser_acceptance.json`; it is not claimed
that all six flows were completed manually in the browser.

A real Render **Restart service** event at approximately 02:28 on 2026-09-20
preserved the dedicated Warehouse probe at revision 10 / turn 3 with its exact
state hash. The next real action persisted as revision 11 / turn 4, survived
refresh and appeared in researcher export. All six completed instances, 18 task
runs and six questionnaires remained. The immutable first restart receipt is
`production_restart_probe_v2_receipt.json`. The subsequent configuration-only
deploy at the same source commit also passed: revision 11 / turn 4 recovered
exactly, then a real action advanced to revision 12 / turn 5 and appeared in
fresh state and export. The six completed instances, 18 runs and six
questionnaires persisted, as did the separate Pong Task 2 instance and its two
answers. This second check is independently recorded in
`production_pilot_redeploy_persistence_receipt.json`; the first receipt was not
overwritten. No fake pilot participant was created for acceptance.

Production uses the existing Neon PostgreSQL database directly from Render,
with additive `pl3_*` tables, bounded connections, transactions and stored
recovery data. Existing legacy tables were not modified. PostgreSQL is the
persistent store; no claim is made that Render Free temporary files survive.
The existing Free plan may sleep during inactivity; its dashboard warns of cold
starts of 50 seconds or more. No paid plan or new paid resource was purchased.

## Data preservation and authorization

The owner explicitly confirmed that Warehouse has no data requiring backup and
instructed deployment to this existing URL. That resolved the previous old
Warehouse transient-data gate; no old Warehouse backup is claimed.

The separate existing Kitchen database was already backed up before deployment:
13 tables, 13,663 rows, 34,256,578 bytes, matching counts and SHA-256 independently
checked. The backup and credential files are outside Git in a restricted private
folder. Existing browser-held Pong records were not deleted or migrated; the new
frontend does not clear their keys. The original working checkout and its
pre-existing changes were preserved; implementation used `XAI-study-v3`.

## Runtime configuration

| Name | Purpose / current setting |
|---|---|
| `POLICYLENS_DATABASE_URL` | Existing PostgreSQL connection; server-only secret. |
| `POLICYLENS_ADMIN_TOKEN` | Existing researcher credential; preview/test creation and scoped export. |
| `POLICYLENS_PUBLIC_ORIGIN` | `https://policylens-warehouse-study.onrender.com` |
| `POLICYLENS_MODE` | `pilot`; `preview`/`test` close public enrollment. |
| `POLICYLENS_STUDY_VERIFIED` | `1`, set after acceptance and actual restart recovery. |
| `POLICYLENS_LLM_BASE_URL` | Existing `https://api.deepseek.com` endpoint. |
| `POLICYLENS_LLM_MODEL` | Existing `deepseek-v4-flash` model. |
| `POLICYLENS_LLM_API_KEY` | Server-only provider credential. |

Existing legacy environment variables and secret file were retained. Credentials
are not in source, browser code, public release responses or URLs. The local
workstation proxy was not copied to Render.

Verified startup runs an actual semantic-service probe. Missing configuration,
persistence or validation keeps public enrollment closed. A latest unavailable
QA result closes new enrollment until a real successful question recovers
availability. Database errors fail health checks. Readiness is a runtime signal,
not a guarantee that every arbitrary question will be understood correctly.

The original 15 v2 production questions were independently reviewed: ten were
satisfactory, four were factual but incomplete/repetitive, and one was an
unnecessary clarification. Two separately reported Pong follow-ups were finally
correct, one after an internal bounded repair. These limitations and prior
failures are retained in `three_domain_qa_evaluation.md`; they are not reported
as 17 perfect answers or as proof of a human performance effect.

## Researcher export

Set `POLICYLENS_ADMIN_TOKEN` privately in the researcher's shell, then run:

```sh
python3 scripts/export_study_v3.py --mode test --release-id policylens-three-domain-20260920.v2
python3 scripts/export_study_v3.py --mode pilot --release-id policylens-three-domain-20260920.v2
```

Exports use authenticated headers, are read-only, and create private local files.
The default JSONL preserves all records and private QA audit. `--format csv`
produces `table,record_json`, not a flattened analysis dataset. Existing output
filenames are refused. Test/preview records must stay excluded from human pilot
analyses. Filter by both mode and release, and follow the frozen analysis protocol.

## Maintenance and rollback

A push is not a deployment because automatic deployment is off. For a release,
complete acceptance, choose Render **Manual Deploy → Deploy a specific commit**,
then verify `/api/release`, `/health`, recovery and export. Code/rule/scenario
changes require a new release ID; never reuse an existing ID with changed source.
Old instances must receive a clear version message rather than silently adopt
new rules.

The previous pre-v3 commit is `af97df8589080a6b1b79bad591059b4d52fe33ee`
(previous deploy `dep-damp8pm7bikc73bv9fjg`). It can be selected explicitly for
rollback using the preserved original start command and legacy environment.
Rollback does not remove the new `pl3_*` research records, but the old interface
cannot resume a v3 instance. Close enrollment and retain/export affected data
before any intentional release change. The legacy entry also remains available
with `POLICYLENS_LEGACY_HUB=1`; do not set it for this v3 deployment.

Platform references consulted during implementation:
[Render manual deployment](https://render.com/docs/deploys),
[Free service limits](https://render.com/docs/free), and
[Neon serverless driver](https://neon.com/blog/serverless-driver-ga).
Deployment evidence is the actual dashboard and public response receipts above.
