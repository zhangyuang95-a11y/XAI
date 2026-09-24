# Main study — 24 September 2026

Researcher authorization: professor approved recruitment of 10 participants in
each of the six game/condition cells (60 new participants), £3 base plus up to
£0.60 performance bonus. Median pilot completion time: 18 minutes.

Use one new Prolific study, 60 places, 18-minute estimate, £3 fixed reward,
Initially six concurrent participants; increased to 20 at the researcher's
request on 24 September. Age 21+, fluent English, desktop/laptop.
Exclude both prior pilot studies. Manual review; no automatic fast rejections.
Use the existing production /prolific/ URL with all three ID placeholders.
Configure its new study ID/completion code and POLICYLENS_PROLIFIC_PLACES=60.
Never reuse the old study ID: allocations and capacity are scoped by study ID.

Release v3.28 freezes £3 base/£0.60 maximum bonus for NEW sessions; previous
reward snapshots, scores and questionnaire data are unchanged. Kitchen remains
three dishes; all three games retain the approved question guide and required
Task 2 understanding ratings. Source/consent versions are updated for audit.

Initial website allocations are randomized in balanced blocks: 10 in each cell.
This balances assigned places, not completed responses. If an enrolled person
returns or times out, verify that terminal status on Prolific before releasing
that website slot with the existing audited release-slot endpoint. Never free
an active or completed session based only on inactivity; do not increase the
cap to hide a mismatch. Prolific replacements and site vacancies must agree.

Base rewards £180 + displayed academic platform fees £60 = £240. Maximum
bonuses £36 + fees at the same rate £12 = £48; planned maximum £288 excluding
any separately authorized compensation or underpayment adjustments.

The existing Render Free instance can restart; concurrency limiting reduces
load but does not eliminate platform interruptions. No compute upgrade is
included in the recruitment authorization. Neon persists committed study data.

## Capacity reconciliation, 24 September

An operational reconciler is available as `scripts/sync_prolific_capacity.py`.
It verifies Prolific submission IDs and participant IDs, releases only incomplete
RETURNED/TIMED-OUT website sessions through the audited admin endpoint, and
preserves completed sessions even when Prolific reports a timeout. It also
preserves unfinished submissions awaiting review. Every cell remains capped
at ten occupied places. No game logic, scores, reward snapshots or research
records are rewritten by the reconciler.

`--apply --resume --watch` runs an exclusive local process for at most eight
hours. It reconciles every 15 seconds, pauses recruitment on errors or when all
website places are occupied, and resumes after verified vacancies when Prolific
also has space. The actual concurrency ceiling is the smaller of 20 and website
vacancies plus occupied unfinished sessions. Recruitment is paused when the
bounded run expires. This depends on the local Mac staying awake and online;
it is not a Render-hosted service and polling is not an atomic cross-platform
transaction. Secrets stay in ignored, private local files. No Prolific API token
is installed on the study website.

The reconciler never messages, approves, rejects, requests returns, buys places,
or increases the approved total of 60 Prolific places. Eight NOCODE submissions
without enrolled sessions currently occupy Prolific places; the researcher has
explicitly chosen not to message them yet. Initially only three replacement
places are available on Prolific, despite nine vacancies on the website.
