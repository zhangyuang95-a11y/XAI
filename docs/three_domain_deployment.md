# Three-domain deployment record and remaining gate

This document separates the implementation candidate from the deployed website.
The candidate release is `policylens-three-domain-20260920.v1`.
**This candidate has not yet been deployed to the production URL.**

## Verified existing service

- Render service: `policylens-warehouse-study`, ID `srv-da66ggbl550s738j2q9g`.
- Existing project: `prj-da66gg6k1f9s73974rc0`; Singapore, Python 3, Free.
- Repository: `zhangyuang95-a11y/XAI`; configured branch `sep2-rollback-20260902`.
- Last successful/live commit at inspection: `af97df8589080a6b1b79bad591059b4d52fe33ee`.
- Live deploy: `dep-damp8pm7bikc73bv9fjg`.
- Root directory: blank (repository root).
- Build: `python -m pip install -r requirements-render.txt`.
- Start: `python -m ui.domain_hub_server --host 0.0.0.0 --port $PORT`.
- Current homepage is the prior Chinese two-domain hub. Its `/health` identifies
  `policylens-domain-hub`, Warehouse `development-preview`, and Pong
  `pong-continuous-24x14-v7`. This is evidence of the old service, not v3 deployment.

The candidate preserves the existing entry command. Executing the hub module
now dispatches to `study_v3.server`; legacy code remains available with
`POLICYLENS_LEGACY_HUB=1`. Rolling back to the previous deployed commit also
restores its original entry behavior. The candidate adds PostgreSQL support
without removing legacy dependency pins.

Target routes on the **same existing service** are `/`, `/warehouse/`, `/pong/`,
`/kitchen/`, `/api/study/*`, `/health` and `/api/release`. The application listens
on Render's `PORT`. It serves all assets and APIs from the same origin.

## Existing resources and backup evidence

The separate existing `policylens-kitchen-study` service has a Neon PostgreSQL
connection and a DeepSeek semantic service configured. Only necessary settings
were obtained via the logged-in Render dashboard's normal environment export.
Secrets and row backups are held outside the repository in a restricted private
folder; their values do not appear in source, this document, URLs, or logs.

The existing Neon database was read successfully. A full logical row backup of
its 13 public tables was subsequently completed through the official Neon
serverless driver using a **read-only, repeatable-read transaction**. Direct
local port-5432 backup attempts failed and were retained as incomplete attempts;
they were not counted as successful backups. The completed JSONL backup is
34,256,578 bytes. A separate column/type/default catalog was also saved.
`analysis/three_domain_build_20260920/legacy_database_backup_receipt.json` in the
parent workspace records counts, timestamp and SHA-256. Legacy database tables
have not been altered by this implementation.

After the backup, the actual Python/psycopg database adapter passed 14 real
PostgreSQL checks through a temporary loopback-to-Neon WSS transport. These
created only the additive `pl3_*` schema and a dedicated transport-test record,
and covered commit, rollback, read-only enforcement, pooling and close/reopen
recovery. The temporary bridge was shut down. This verifies the adapter against
the real database, but Render's direct connection and redeployment recovery
still require the production checks below.

This backup covers the existing **Kitchen database only**. It does not include
old Warehouse files or Pong browser storage:

- Old Warehouse writes completed results under
  `output/study_records/warehouse/three-task-explanation.v1/` on the running
  instance. Active sessions are only in its process memory. No full HTTP export
  endpoint exists in the old deployed implementation.
- The Render dashboard confirms that Shell/SSH is unavailable on its Free
  compute plan. Accessing a fresh process after replacement would not recover
  the old in-memory session pool or reliably recover the old temporary files.
- Old Pong questionnaire data is mainly in participants' browser localStorage,
  while some run/review data is in sessionStorage and server memory. The new
  frontend uses a separate language preference key and never clears old browser
  keys, but the server backup cannot collect participants' browser data.

**Production replacement is withheld until the owner confirms an existing
backup or confirms that old transient records are disposable test data.** This
implements the task brief's explicit instruction to protect existing research
records before deploying. It is not a claim that those records were backed up.

## Runtime configuration

The implementation expects the following environment variables on the target
service. None of these variables' secret values should be committed or placed
in client code.

| Name | Purpose |
|---|---|
| `POLICYLENS_DATABASE_URL` or `DATABASE_URL` | Existing PostgreSQL connection. Only additive `pl3_*` tables are used. |
| `POLICYLENS_ADMIN_TOKEN` | Researcher-only preview creation and export. |
| `POLICYLENS_PUBLIC_ORIGIN` | `https://policylens-warehouse-study.onrender.com` |
| `POLICYLENS_MODE` | Enrollment gate and public mode label. Readiness requires `pilot`; `preview` and `test` keep public enrollment closed. This release does not support `formal`. |
| `POLICYLENS_LLM_BASE_URL` | Existing authorized semantic endpoint. |
| `POLICYLENS_LLM_MODEL` | Model configured for the validated semantic endpoint. |
| `POLICYLENS_LLM_API_KEY` | Server-side provider authentication. |
| `POLICYLENS_STUDY_VERIFIED` | Set to `1` only after deployment acceptance checks. |
| `POLICYLENS_STORAGE_MODE` | Only for explicitly verified persistent SQLite disk use; not needed for PostgreSQL. |

The known existing provider uses `https://api.deepseek.com` and
`deepseek-v4-flash`. Local real-provider testing used the workstation's existing
network proxy; **do not copy that local proxy into Render**.

Missing persistence, semantic configuration, researcher access, or the verified
flag keeps pilot enrollment closed. Verified startup performs a real semantic
probe. The latest unavailable QA result closes new pilot enrollment until an
actual successful QA request recovers availability. Database failures fail the
health request rather than pretending research readiness. `/health` distinguishes
process health and study readiness; `/api/release` exposes release identity and
current readiness without secrets.

## Final deployment and acceptance steps

1. Resolve the old transient-data gate above and retain the backup receipt.
2. Finish candidate tests and commit only the isolated worktree changes.
3. Push the candidate branch. A push alone is not a deployment.
4. Preserve existing target environment variables; add the validated v3 settings
   using existing authorized resources. Start in preview/unverified mode.
5. Use Render **Manual Deploy → Deploy a specific commit** for the candidate SHA
   in the same repository, or update the configured branch and explicitly deploy.
   Preserve the existing service ID and public URL. Confirm a successful deploy.
6. Verify the production `/api/release` commit/source hash and all three routes.
   Finish all three domains' A/B flows with authenticated `test` identities;
   perform real English/Chinese/follow-up/history/counterfactual QA through
   active A/Task 2, and verify terminal/Task 3 revocation.
7. Verify PostgreSQL writes, researcher export, and a dedicated test instance's
   recovery after a real application restart/redeployment. No fake participant
   records may be labeled human pilot results.
8. Only then set `POLICYLENS_MODE=pilot` and `POLICYLENS_STUDY_VERIFIED=1`, deploy the same immutable release
   with the approved configuration, and check real readiness again.

A configuration-only deployment does not change the release's rules/source.
Any code/rule/scenario change after release freeze requires a new release ID.
Existing instances of a different release must receive a clear old-version
message, not silently adopt new task rules. The retained `pl3_*` data remains
separate from old tables when the old commit is restored.

## Researcher export

After the new service is actually deployed and configured, set
`POLICYLENS_ADMIN_TOKEN` in the researcher's shell and run:

```sh
python3 scripts/export_study_v3.py --mode test
python3 scripts/export_study_v3.py --mode pilot --release-id policylens-three-domain-20260920.v1
```

Exports are read-only, avoid credentials in URLs, and create private local files.
The default JSONL preserves all research records and private QA audit.
`--format csv` produces `table,record_json`; it does not invent a flattened
analysis dataset. Reusing an existing output filename is refused.

## Platform references checked during this task

Render documents [manual/specific-commit deployment](https://render.com/docs/deploys),
[Free service storage/access limits](https://render.com/docs/free), and
[persistent disk requirements](https://render.com/docs/disks). Neon documents its
[official HTTP/WebSocket driver](https://neon.com/blog/serverless-driver-ga).
These platform references inform the deployment and backup procedure; they are
not evidence that this candidate has been deployed or that a human effect exists.
