# Prolific smoke-test deployment

This change is prepared for 12 people, two assigned to each of six cells,
approximately 10 minutes, and GBP 3.00 fixed payment per valid completion.
There is no performance bonus. Prolific draft settings show GBP 36.00 rewards
and GBP 12.00 platform fees, totaling GBP 48.00 (VAT GBP 0.00).

The first smoke test was published on 23 September 2026 after live verification.
Its first 12 assignments included three later returned submissions. A website
assignment cap does not automatically follow Prolific's returned-place refill.
Keep recruitment paused while reconciling terminal submissions and capacity.

## Configure the deployment

Preserve the existing database, provider, origin, and researcher credentials.
Deploy this branch's changes through the existing Render service. Its additive
schema introduces `pl3_prolific_links`; no prior data is deleted or rewritten.
The v3.10-prolific release continues existing v3.9 Kitchen sessions under the
same game rules; earlier incompatible Kitchen versions remain archived.

Set these environment values on the service:

- `POLICYLENS_PROLIFIC_STUDY_ID`: copy the ID of the saved Prolific draft.
- `POLICYLENS_PROLIFIC_COMPLETION_CODE`: copy the draft's normal completion code.
- `POLICYLENS_PROLIFIC_PLACES=12`.
- `POLICYLENS_PROLIFIC_LAUNCH_CONFIRMED=0` while reviewing and configuring.

The researcher explicitly confirmed on 22 September 2026 that IRB-2025-996
applies to this experiment. Consent version v2 now contains the approved-study
statement independently of whether recruitment is enabled. This is recorded
researcher confirmation, not an independent inspection of the approval file.

The entry is ready only when persistence/provider/deployment checks pass and
launch confirmation is enabled. Verify the stated Neon region and DeepSeek
provider against the deployment, then enable
`POLICYLENS_PROLIFIC_LAUNCH_CONFIRMED=1` when recruitment setup is complete.
Existing v3.9 and v3.10-prolific sessions remain compatible with v3.11-consent;
the new consent version applies to new enrollments and does not rewrite prior
consent records.

The saved study ID and completion code are in the task's ignored local
`output/prolific_draft_settings.json`. Copy the values from the Prolific UI if
deploying from a separate checkout. Keep completion codes out of public docs.

## Prolific draft

- One external URL ending in `/prolific/`.
- Record IDs with URL parameters `PROLIFIC_PID`, `STUDY_ID`, `SESSION_ID`.
- 12 places; 10-minute estimate; GBP 3.00 fixed reward.
- Age 21+, English fluent, desktop/laptop, one participation per person.
- No country restriction in the current draft.
- Manual review on normal completion; automatic fast-submission rejection off.
- No custom screen-out path; no participant account/password required on the
  external study site.

## What the integration guarantees

- No allocation or consent record is written by visiting a GET URL or declining.
- A valid entry requires explicit consent, age-21 confirmation, current consent
  version, all three IDs, and the configured study ID.
- Allocation is random among least-filled cells under a single enrollment
  transaction, equivalent to shuffled blocks of six. The initial cohort cap is
  12 unreleased assignments. Selection and internal identity creation commit
  together. Completed participants always retain their capacity. Replacement
  participants are randomized among the cells with released vacancies.
- Assignment persists in the database; retrying with the session cookie resumes
  the same instance. A copied Prolific ID or URL does not authenticate a return.
  Lost-cookie cases require researcher assistance; there is no PID-only login.
- Once Prolific is configured, new unauthenticated pilot enrollments through
  the manual domain/group entry are disabled. Researcher previews remain
  separate. A linked participant cannot create a second domain instance.
- Prolific identifiers are held in a separate mapping table and excluded from
  the ordinary research export. The restricted operational endpoint is
  `GET /api/prolific/admin/payments`, using the existing researcher bearer auth.
- Survey answers are optional for this recruitment flow. Skipped answers are
  saved as null, separate from score zero or explanation Not used / N/A.
- Completion URL/code is delivered only after all task rounds and the survey
  submission are saved. The survey submission may contain no answers.
- The website does not approve submissions or transfer funds. It returns the
  participant to Prolific; the researcher reviews and approves payment there.

## Before launch

Verify the deployed release identity and `/api/prolific/info`, do an isolated
preview with synthetic IDs in a separate test database, and then use Prolific's
participant preview to check the actual URL parameter substitution and return
flow. Do not create artificial submissions or consume real cohort slots during
tests. Do not use the production completion link for synthetic submissions.

The current ordinary URL-parameter integration validates shape and study ID;
it is not Prolific server-to-server authentication or signed-URL verification.
Use manual submission review to match the saved ID triplet with the real study
submissions before payment.

## Returned or timed-out places (v3.12)

The server does not assume that inactivity means withdrawal and does not query
Prolific automatically. A researcher must first verify the submission's terminal
status in Prolific, then call `POST /api/prolific/admin/release-slot` with the
existing researcher bearer token and a JSON body containing:

- `instance_id` and `submission_id` copied from the restricted payments view;
- `submission_status`: `RETURNED` or `TIMED_OUT`, as verified on Prolific;
- `reason`: a short operational audit note, without unnecessary personal data.

The endpoint rejects completed sessions, unknown/mismatched submissions and
nonterminal statuses. Repeating the same release is idempotent. The same
instance lock used by gameplay protects a release against concurrent completion.
Capacity becomes available only after the release commits; concurrent arrivals
cannot claim a vacancy twice. Original consent, allocation indices, actions,
scores and private Prolific mappings remain intact. The released session can no
longer resume gameplay. Releases are recorded in a separate private operational
table, and the payments view exposes status, reason and timestamp for auditing.

This action neither returns nor rejects nor pays a submission on Prolific.
Do not release a completed session or a pending submission with a missing code.
Match website completions against the Prolific records before payment. Review
technical-issue submissions separately; a missing code alone is not a rejection
reason. Prolific's remaining recruitment places must match the site's vacancies
before resuming; increasing a website cap alone does not fix this mismatch.

Release v3.12 preserves v3.11 and other already-supported sessions with unchanged
game rules. The schema change is additive, and no participant data is deleted.

Counts balance assignments, not completed responses. Keep incomplete records
and their original allocations. If withdrawals cause Prolific to reopen places
after the 12 allocations have been consumed, pause recruitment and review the
incomplete entries; do not silently reassign someone or overwrite their data.
Any additional replacements require a deliberate cohort/slot policy first.

## Validation performed locally

`tests/test_prolific_enrolment.py` uses temporary databases and synthetic IDs
to exercise concurrency, complete blocks, capacity, duplicate retries, restart
recovery, unauthorized identity reuse, domain/group tampering, consent checks,
optional survey answers, completion gating, and private operational exports.
The existing HTTP/store flows cover all three games and both conditions.
No real participants or model-provider calls are part of these automated tests.
