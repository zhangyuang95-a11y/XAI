# Warehouse r4.1 record-persistence audit

## Current status

The checked-in and currently deployed warehouse service is not a persistent
study service. `render.yaml` declares a Free Render web service,
`WAREHOUSE_ONLINE_DATABASE=/tmp/warehouse_alignment_online.sqlite3`, and
`WAREHOUSE_STORAGE_MODE=ephemeral`. The server therefore returns
`data_persistent=false` and warns participants that a restart can erase local
records.

This is a real release blocker for r4.1. Render documents that a Free web
service loses local files, including SQLite databases, whenever it spins down,
restarts, or redeploys. Free web services cannot attach persistent disks.

- Render Free-service limits: https://render.com/docs/free
- Render persistent disks: https://render.com/docs/disks

The repository does not currently contain a PostgreSQL persistence backend for
the warehouse server. A deleted cooperative-kitchen revision used
SQLAlchemy/psycopg with PostgreSQL, but its tables, transactions and recovery
worker are specific to that separate application. Pointing the warehouse
process at the old kitchen database URL would not work and must not be treated
as a migration.

## Lowest-risk r4.1 storage path

Keep the existing SQLite transaction implementation and attach one Render
persistent disk to the existing single-instance warehouse service. This avoids
rewriting the participant ledger during Actor and scene release work. Before
the r4.1 deployment preflight can pass, the checked-in Render declaration and
the dashboard must agree on all of these values:

```yaml
plan: 0.5c-512mb
numInstances: 1
disk:
  name: warehouse-study-data
  mountPath: /var/data
  sizeGB: 1
envVars:
  - key: WAREHOUSE_ONLINE_DATABASE
    value: /var/data/warehouse_alignment_online.sqlite3
  - key: WAREHOUSE_STORAGE_MODE
    value: persistent
```

The exact paid plan may be changed to another current disk-capable Render web
plan. The service must remain one manually scaled instance and must not declare
autoscaling, because Render disks can be attached to only one instance. The
mount path and database path are part of the r4.1 deployment contract. Local
development remains SQLite with an explicit `ephemeral` storage mode and a
temporary or repository-external path.

The server now rejects `storage_mode=persistent` for a SQLite file outside
`/var/data`; automatic mode remains ephemeral even for a path named
`/var/data`, so production persistence requires both the explicit mode and the
validated Blueprint. The
deployment preflight structurally binds the plan, disk and environment to the
single named warehouse service. It rejects a Free or unknown plan, multiple
services, a missing or malformed disk, a different mount, multiple/autoscaled
instances, an ephemeral storage declaration, or a database outside
`/var/data`. These checks prevent a `/tmp` path or an unrelated service's disk
from being presented as evidence of durable warehouse storage.

Participant endpoints expose only the persistence classification and a matching
consent notice. They do not expose the SQLite path, disk name, artifact hashes
or deployment provenance. The ephemeral notice and persistent-disk notice are
generated from the same server-side storage classification, so a future paid
configuration cannot continue displaying the current Free-instance warning.

## Existing r3 records

Attaching a disk triggers a deploy. The old instance's `/tmp` database is not
copied to the new disk. Therefore current r3 records can be retained only if an
operator obtains an exact copy of the live SQLite database (including a safe
WAL checkpoint/backup) before the first restart or disk attachment.

The current Free service has no repository endpoint for a complete,
authenticated database export, and Render documents that Free web services do
not provide shell access. If no external backup or operator-accessible copy
already exists, the repository cannot reconstruct all old sessions from the
participant APIs. In that case, the truthful record is that historical r3
durability was never guaranteed; r4.1 must start a new persistent namespace
without claiming that the missing ephemeral records were migrated.

If a verified SQLite copy is available, migration is operational rather than a
schema conversion: stop enrollment, run `PRAGMA wal_checkpoint(TRUNCATE)` and
`PRAGMA integrity_check` on the copy, hash it, upload it as
`/var/data/warehouse_alignment_online.sqlite3` while the service is stopped,
set owner-only permissions, and start r4.1. Startup already checks the stored
`service_family` and `namespace` before applying additive columns. Keep the
source copy and its hash outside Git.

## Required deployment evidence

Do not report `data_persistent=true` merely because the dashboard shows a disk.
Deploy with enrollment closed, then perform an actual restart test against the
same release and disk:

1. Create a canary participant and confirm one action with a fixed operation
   ID; record the public state version and frame.
2. Restart the real Render service, using an actual process restart rather
   than a browser refresh.
3. With the same cookie, verify that the participant, stage, active run, state
   version, frame, tutorial progress and existing answer records are restored.
4. Resubmit the same operation ID and verify that the frame does not advance.
5. Create a pending explanation, restart again, and verify that startup changes
   `running` work back to `pending` and completes it without changing game
   state.
6. Verify `/health` returns `data_persistent=true`, then securely back up the
   SQLite database and test that the backup passes `PRAGMA integrity_check`.

The acceptance receipt must record the release signature, pre/post restart row
counts, canary operation ID digest, database backup SHA-256, timestamps and
pass/fail results without participant answers or identifiers. Until this real
restart test passes, the accurate status is “persistent storage configured,
restart recovery not yet verified.” `formal_ready` remains false independently
of this storage gate.

## PostgreSQL alternative

Neon or Render PostgreSQL remains a sound future option, especially if the
service must scale beyond one instance. It is not the minimum r4.1 change. A
warehouse PostgreSQL port needs its own namespaced schema, transaction-level
serialization for four-person block assignment, idempotency and question-job
recovery tests, SQLite-to-PostgreSQL import validation, connection-pool limits,
and a real process-restart acceptance run. The Python driver and database URL
alone do not provide those guarantees.
