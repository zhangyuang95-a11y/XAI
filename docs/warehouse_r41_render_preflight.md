# Warehouse r4.1 Render deployment preflight

The existing Render service stays on r3 until the r4.1 training ledger selects
an Actor, the complete post-freeze pipeline succeeds, and persistent storage is
configured. The Blueprint keeps automatic deployment disabled and now selects
the r4.1 loader explicitly:

```text
python -m ui.warehouse_alignment_online_server --release-module ui.warehouse_alignment_r41_online_release --base64 /etc/secrets/warehouse_alignment_release.b64
```

The two hashes currently committed in `render.yaml` belong to r3. The current
Free service also uses ephemeral SQLite under `/tmp`. Before r4.1 can pass
preflight, attach a Render persistent disk at `/var/data`, use a paid
disk-capable service plan, explicitly set `numInstances: 1` with no `scaling`
block, set `WAREHOUSE_ONLINE_DATABASE` to
`/var/data/warehouse_alignment_online.sqlite3`, and set
`WAREHOUSE_STORAGE_MODE=persistent`. The operator and migration requirements
are recorded in `docs/warehouse_r41_persistence_audit.md`. Then replace the two
release hashes with the values from the newly generated r4.1 receipt in the
same commit that is deployed. A partial source-only or ephemeral deployment
will fail preflight, which is intentional.

After Actor admission, build all post-freeze evidence and the portable release
with the command in `docs/warehouse_r41_release_commands.md`. The orchestrator
writes these deployment inputs under its new output root:

```text
production_admission.json
warehouse_r41_online_release.zip
warehouse_r41_online_release.b64
release_receipt.json
```

Hash the receipt outside the receipt itself:

```bash
R41_RECEIPT_SHA=$(shasum -a 256 "$R41_OUT/release_receipt.json" | awk '{print $1}')
```

Read `package_sha256` and `manifest_sha256` from that externally hashed receipt
and update only the matching values in `render.yaml`. Then run the local
deployment preflight. It rechecks the receipt, admission, ZIP, Base64 equality,
archive whitelist, manifest and artifact hashes, current source binding,
NumPy runtime, Actor identity, tutorial replay, and Render configuration. The
Render check parses the single named service structurally, so a plan or disk on
an unrelated service cannot satisfy the gate.

```bash
python scripts/preflight_warehouse_r41_render.py \
  --release-root "$R41_OUT" \
  --expected-release-receipt-sha256 "$R41_RECEIPT_SHA" \
  --render-yaml render.yaml
```

The only passing status is `ready_for_manual_render_deploy`. Upload the exact
bytes of `$R41_OUT/warehouse_r41_online_release.b64` to the existing Render
Secret File path `/etc/secrets/warehouse_alignment_release.b64`. Set the two
Render environment hashes to the same receipt values. Do not alter the service
ID, public origin, root directory, admitted persistent-storage declaration, or
automatic-deploy setting.

Stage source with the exact public allowlist, inspect it, scan it, and commit it
before triggering the manual deployment:

```bash
git add --pathspec-from-file=docs/warehouse_r41_git_allowlist.txt
git diff --cached --name-only
git -c core.whitespace=-blank-at-eof diff --cached --check
grep -v '^docs/warehouse_r41_render_preflight.md$' docs/warehouse_r41_git_allowlist.txt | xargs rg -n --with-filename -e 'BEGIN [A-Z0-9 ]*PRIVATE KEY' -e 'postgresql?://' -e 'mongodb(\+srv)?://' -e 'mysql://' -e 'Bearer [A-Za-z0-9._-]{24,}' -e 'sk-[A-Za-z0-9_-]{24,}' -e '/Users/[^/[:space:]]+' -e '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' -e '(^|[^0-9.])1[3-9][0-9]{9}([^0-9.]|$)'
```

The secret and PII scan excludes only this documentation line, which contains
the detector expressions themselves, and should otherwise produce no matches.
The whitespace check disables only Git's `blank-at-eof` rule because one
training-bound runtime and two accompanying test files were retained with
terminal blank lines; ordinary trailing whitespace and space-before-tab checks
remain enabled. Changing the runtime byte would invalidate the training ledger.
Before pushing, compare
`git diff --cached --name-only` byte-for-byte with the sorted allowlist. Never
stage the ignored `output/warehouse_native/**` training runs, admission output,
ZIP/Base64 files, SQLite databases, participant data, local environment files,
or any unrelated historical untracked warehouse scripts and tests.

After the source commit is pushed, manually deploy the same commit and the same
Secret File. Keep enrollment closed through the real restart-recovery test in
`docs/warehouse_r41_persistence_audit.md`. Verify `/health` before opening
enrollment. The response and the
participant view must identify the public `r4.1` release and must not expose an
Actor SHA, artifact hash, task seed, or scene fingerprint. The structured
`warehouse_r41_deployment_identity` startup log is the operator-only identity:
compare its Actor, package, manifest, Secret File and six selected X/Y scene
fingerprint hashes with the signed post-freeze receipt and local preflight.
No r3 server version or old Actor identity may remain. If build, startup,
health, identity, or restart recovery differs, leave r3 as the active release
and retain the r4.1 failure evidence without changing the online hashes.
