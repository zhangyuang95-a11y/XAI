# Prolific recruitment draft

Status: Prolific draft saved; no paid recruitment has been published and no
further consent update has yet been verified live.

Draft: https://app.prolific.com/researcher/workspaces/studies/6ab28fdbed0497449f769b86
Project: https://app.prolific.com/researcher/workspaces/projects/6ab28ea45207f93275ae6163

Confirmed draft settings: 12 places, 10 minutes, desktop only, age 21-100,
English fluent, no country restriction, one submission per person, URL ID
parameters, manual approval, no automatic fast-submission rejection.

Deployment access: GitHub write access was verified after accepting the owner's
invitation on 22 September 2026. Render project access is still held by the
website owner. PR #4 is merged and v3.10-prolific was verified live.

IRB-2025-996 applicability to this experiment was explicitly confirmed by the
researcher on 22 September 2026 (user statement: “IRB是确认的”). This records
the researcher's confirmation, not an independent review of the approval file.

## Confirmed design

- One Prolific recruitment study and a single external entry link.
- Initial recruitment is a 12-person smoke test: two participants assigned to
  each of the six cells, using two randomized blocks of six. This checks the
  workflow, storage, and timing; it is not an efficacy sample.
- Compensation confirmed: GBP 3.00 fixed per completed participant, independent
  of game performance. No performance bonus for this first smoke test.
  Twelve rewards total GBP 36.00; Prolific displays GBP 12.00 platform fees and
  GBP 0.00 VAT, for a total of GBP 48.00. No payment has been made.
- Each participant plays exactly one domain: Warehouse, Cooperative Pong, or
  Cooperative Kitchen.
- Six independent cells: each domain crossed with explanation access A or
  no explanation access B.
- Preserve the repository's current intervention timing: both groups complete
  tutorial/practice, Task 1, Task 2, Task 3, and questionnaire within their one
  assigned game. A has explanation access only during active Task 2.
- Assign and persist conditions on the server. Participants cannot choose or
  change their game or condition.

## Participant-facing copy for review

Title: Work with an AI teammate in a cooperative game

Description:

In this study, you will work with an AI teammate in one of three cooperative
games. You will be assigned a game, learn its controls through a tutorial or
practice, complete three task rounds within that game, and answer a short
questionnaire about your experience. You do not need to play all three games.

Please use a desktop or laptop computer with a keyboard and a stable internet
connection. The study records your game actions, scores, and questionnaire
responses, along with any questions you submit in the game. Your Prolific ID
will link your study data to your Prolific submission.

Before starting, you will see the participant information and acknowledgement
page. Please read it before deciding whether to take part.

The user supplied the consent template and study-specific wording, a 10-minute
estimate, researcher contacts, and a fixed GBP 3.00 payment. The researcher has confirmed approval applicability.

## Remaining launch checks

- Verify the deployed Neon region and DeepSeek configuration.
- Merge/deploy and validate the entry and completion flow on Render.
- Use the smoke-test timings to revise the current 10-minute estimate if needed.

The Prolific study ID and completion code have been generated and are retained
in the ignored local settings file described in `prolific_deployment.md`.

## Implemented website integration

Consent material received on 22 September: `consent_form.pdf` and the user's
study-specific text. Review outputs are generated from
`docs/consent_review_content.json` by `scripts/build_consent_review.py`.
The source PDF supplies the paragraph layout, project title, and template
reference IRB-2025-996; its applicability to this experiment was confirmed by
the researcher on 22 September 2026. The local PDF/HTML outputs are review drafts only.

- Preserve age 21+ and separate consent/decline choices; no enrolment on decline.
- Record the final consent version and timestamp when enrolment is committed.
- Questionnaire answers are optional in both the browser and server for
  Prolific participants. Skipped values remain distinct from zero and Not used.
- Use a generated internal code for research data and keep the Prolific-ID
  mapping in a separate restricted table/export. Do not use a Prolific PID as
  the internal participant primary key.
- The user reports Neon Singapore storage and DeepSeek processing. Repo docs
  establish Neon usage but do not independently establish the live region;
  verify deployed region/provider before launch.
- The user's expected duration is 10 minutes. Check the base reward against
  that duration in the account's actual currency before publication.
- Consent and recruitment copy both specify a fixed GBP 3.00 payment with no
  performance bonus, following the latest user instruction.

The new entry path `/prolific/` has been implemented locally. Do not publish
the Prolific draft until the deployed path and completion flow are verified.

1. Capture `PROLIFIC_PID`, `STUDY_ID`, and `SESSION_ID` from the Prolific link;
   validate their format and expected study, and save their association with
   the internal participant and study instance.
2. After acknowledgement, assign one of the six cells using randomized blocks
   of six, with each cell used once per block. Serialize selection and insertion
   in one database transaction so concurrent entries cannot allocate the same
   slot. An incomplete block can differ by at most one enrolled participant per
   cell. Keep the allocation sequence private.
3. Store the assignment durably. Refreshes, duplicate submissions, and recovery
   resume the same instance without consuming a new allocation. A bare
   participant ID must not grant access to another participant's saved data.
4. Lock the game and condition server-side, including direct domain/API entry.
   Keep researcher previews separate from the recruitment allocation counts.
5. Track assigned, started, completed, and incomplete counts separately. Equal
   assignment counts do not guarantee equal completed counts. Retain dropouts
   and original assignments; any replacement policy must be specified before
   collection and must not depend on observed scores.
6. Show the Prolific completion action only after the questionnaire and game
   records are durably saved. Use the configured completion code, not a made-up
   code. Prevent replay from creating a second completion.
7. Export the Prolific identifiers, assigned game/condition, consent version,
   allocation method, instance ID, and completion status alongside existing
   research data. Keep session authentication credentials out of exports.

## Original source findings addressed by this change

Source branch: `codex/three-domain-study-v3-20260920`

Inspected commit: `1e2f988f3ac419a7f35045e5af42d7168fb1539f`

- `study_v3/web/app.js` originally exposed a group selector to participants.
- `study_v3/store.py` originally accepted participant-provided groups and permitted new
  instances in different domains. The new Prolific
  entry now applies server-side restrictions.
- Original A/B balancing considered only the selected domain; the new entry
  implements six-cell allocation independently.
- The existing protocol JSON describes some older behavior, so runtime source
  and tests must be used to verify the final protocol and update its snapshot.
- The existing database supports SQLite/PostgreSQL transactions. Preserve all
  old records through additive schema changes.

## Verification

- Concurrent allocations fill each block exactly once across all six cells.
- Refresh/retry and a second authenticated visit preserve the assignment.
- Direct requests cannot change the assigned game or explanation condition.
- Test/preview records do not affect participant quotas.
- All three Prolific IDs survive consent, gameplay, questionnaire and export.
- Completion is unavailable before saved completion and works after it.
- Restart/redeployment preserves assignments and records.
- The live release and study readiness are verified independently of local
  tests. Run a small pilot before committing the final duration and recruitment.

Local results: 11 Prolific-specific tests, 45 existing HTTP/store tests, and
63 database/domain-switching tests passed. Live deployment verification remains
pending. See `prolific_deployment.md` for the deployment procedure and limits.
