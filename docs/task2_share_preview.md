# Standalone Task 2 share preview

Run `python -m study_v3.share_preview` (default local port 9131; Render uses PORT).
Deploy the dedicated `codex/task2-share-preview-20260923` branch as a **separate**
free Python web service. Build with `python -m pip install -r requirements-render.txt`;
start with `python -u -m study_v3.share_preview`; health path `/health`.
No environment secrets, database URL, or Prolific credentials are required.

The landing page `/try/` offers all three games; `/try/pong`, `/try/warehouse`,
and `/try/kitchen` focus the relevant start button. Clicking starts or resumes
an anonymous Group A preview directly in Task 2, at the first real automatic
explanation. A synthetic baseline is bypassed inside the isolated demo database.
This is not research participation or a paid Prolific submission. The landing
page explains temporary anonymous gameplay storage and the lack of free-text Q&A.

All sessions use preview mode. The server ignores production database, provider,
and Prolific environment variables and uses its own disposable SQLite database
under `/tmp/policylens-task2-preview/`. Free-service restarts may erase progress.
Administrative export, normal enrollment, Prolific, provider questions, Task 3
and questionnaire endpoints are unavailable. Same-origin start requests create
server-side sessions; no administrator credential is given to the visitor.
Each visitor can resume three games with one HttpOnly cookie. Restarting creates
a new demo identity; previous records are not overwritten.

The game ends after Task 2 with links to choose another game or restart.
Mandatory explanation confirmation and action locking use the same implementation
as the local prototype. The normal pilot flow is unchanged when `share_preview`
is absent from the release metadata.

Validation: `tests/share_preview_browser.cjs` checks the three direct-entry flows,
explicit confirmation, shared-browser navigation, cross-session authorization,
blocked production endpoints, cross-origin rejection and Task 2 completion.
