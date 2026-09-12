# Warehouse r4.1 diagnostic v8 HTTP and browser acceptance

These commands exercise the frozen v8 package through its real HTTPS server
boundary.  The checkers do not discover a release directory and do not import
the warehouse environment, Actor, explainer, or release loader.  The operator
must supply the exact package and its package, manifest, and Actor SHA-256
values.

Run the HTTP flow and browser flow against two new SQLite databases.  Keep the
two databases and report directories out of Git.  A checker refuses a database
whose name does not carry its dedicated QA suffix, and it refuses a nonempty
database at the start of a run.

## Prerequisites

Run from the repository root.  Set these values from the already frozen release
receipt; do not infer or copy them from a public participant response.

```sh
export V8_PACKAGE=/absolute/path/to/the/frozen-v8-package.zip
export V8_PACKAGE_SHA256=<64-lowercase-hex-package-sha256>
export V8_MANIFEST_SHA256=<64-lowercase-hex-manifest-sha256>
export V8_ACTOR_SHA256=<64-lowercase-hex-actor-sha256>
export V8_QA_ROOT=/absolute/private/path/to/new-v8-acceptance
mkdir -m 700 "$V8_QA_ROOT"
```

Generate a temporary localhost CA/server certificate.  It is not installed in
the macOS keychain.  The private key is mode `0600` and both checkers pin trust
to the generated certificate.

```sh
python - "$V8_QA_ROOT/tls" <<'PY'
import sys
from scripts.serve_warehouse_r41_diagnostic_v8_https import generate_certificate
certificate, key = generate_certificate(sys.argv[1])
print(certificate)
print(key)
PY
export V8_CERT="$V8_QA_ROOT/tls/localhost.crt"
export V8_KEY="$V8_QA_ROOT/tls/localhost.key"
```

The browser checker requires Node.js, Playwright, and either Playwright
Chromium or a local Google Chrome executable.  It launches a new isolated
headless browser and never connects to a user's browser or CUA session.

## Real HTTPS and restart flow

Start the server in one terminal.  Its first startup creates the empty QA
database.

```sh
export V8_HTTP_DB="$V8_QA_ROOT/run_diagnostic_v8_http_qa.sqlite3"
export V8_HTTP_REPORT="$V8_QA_ROOT/http_report"
python scripts/serve_warehouse_r41_diagnostic_v8_https.py \
  --package "$V8_PACKAGE" \
  --expected-package-sha256 "$V8_PACKAGE_SHA256" \
  --expected-manifest-sha256 "$V8_MANIFEST_SHA256" \
  --database "$V8_HTTP_DB" \
  --certificate "$V8_CERT" --private-key "$V8_KEY" --port 8443
```

Run the first phase in another terminal:

```sh
python scripts/check_warehouse_r41_diagnostic_v8_http.py --execute \
  --phase before --base https://127.0.0.1:8443 \
  --database "$V8_HTTP_DB" --ca-certificate "$V8_CERT" \
  --expected-package-sha256 "$V8_PACKAGE_SHA256" \
  --expected-manifest-sha256 "$V8_MANIFEST_SHA256" \
  --expected-actor-sha256 "$V8_ACTOR_SHA256" \
  --output "$V8_HTTP_REPORT"
```

Stop the server process, record its PID and stop time, and start the exact same
command again with the same package, hashes, certificate, and database.  Then
run the second phase:

```sh
python scripts/check_warehouse_r41_diagnostic_v8_http.py --execute \
  --phase after --base https://127.0.0.1:8443 \
  --database "$V8_HTTP_DB" --ca-certificate "$V8_CERT" \
  --expected-package-sha256 "$V8_PACKAGE_SHA256" \
  --expected-manifest-sha256 "$V8_MANIFEST_SHA256" \
  --expected-actor-sha256 "$V8_ACTOR_SHA256" \
  --output "$V8_HTTP_REPORT"
```

Success writes `report.json` with
`status=passed_r41_diagnostic_v8_real_https_acceptance`.  It authenticates four
independent A/XY, A/YX, B/XY, and B/YX sessions; tutorial and six-round flow;
Task 1 live, historical, and post-round answers; B and Task 2 denials; old-answer
isolation; refresh recovery; a deliberately lost response retried after the
real process restart; questionnaire recovery; and stored NN action authority.
The directory also contains the immutable plan, restart checkpoint, cookie
jars, an append-only event log, and compressed raw public responses.

## Real browser flow

Stop the HTTP-flow server.  Start the exact package on a second fresh database:

```sh
export V8_BROWSER_DB="$V8_QA_ROOT/run_diagnostic_v8_browser_qa.sqlite3"
export V8_BROWSER_REPORT="$V8_QA_ROOT/browser_report"
python scripts/serve_warehouse_r41_diagnostic_v8_https.py \
  --package "$V8_PACKAGE" \
  --expected-package-sha256 "$V8_PACKAGE_SHA256" \
  --expected-manifest-sha256 "$V8_MANIFEST_SHA256" \
  --database "$V8_BROWSER_DB" \
  --certificate "$V8_CERT" --private-key "$V8_KEY" --port 8443
```

Run Playwright.  Set `NODE_PATH` only when Playwright is installed in a
nonstandard runtime directory.

```sh
node scripts/check_warehouse_r41_diagnostic_v8_browser.cjs --execute \
  --base https://127.0.0.1:8443 \
  --database "$V8_BROWSER_DB" --ca-certificate "$V8_CERT" \
  --expected-package-sha256 "$V8_PACKAGE_SHA256" \
  --expected-manifest-sha256 "$V8_MANIFEST_SHA256" \
  --expected-actor-sha256 "$V8_ACTOR_SHA256" \
  --output "$V8_BROWSER_REPORT"
```

Success writes `report.json` with
`status=passed_r41_diagnostic_v8_browser_acceptance`, plus an immutable plan,
event log, and screenshots.  The report covers four isolated contexts across
1365×900 and 1280×800 in Chinese and English, the full A/B flow, long answer and
evidence rendering, page geometry and focus stability, and measured interior
Canvas coordinates during both tutorial and formal 380 ms transitions.

## Harness self-test

This test starts only the synthetic development fixture, performs a real
loopback TLS process restart, and runs the full browser matrix:

```sh
pytest -q tests/test_warehouse_r41_diagnostic_v8_acceptance_harness.py
```

The self-test passes `--allow-synthetic-fixture`.  Such reports always state
`release_acceptance_eligible=false` and cannot be used as frozen-release
evidence.  Never pass that flag in the two real commands above.
