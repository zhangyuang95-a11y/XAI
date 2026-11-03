# Public three-game playground deployment — 2026-11-03

Live URL: https://policylens-warehouse-study.onrender.com/study/

Render deployment `dep-dbkue3ff3r2c73fuvu9g` succeeded, with the dashboard
reporting Live at 2026-11-03 21:54:04 Asia/Shanghai. Both the dashboard and live
health endpoint identify source `28410ef4fdaf36a5ffc502d40d08176436613688` and
runtime `research-preview-composition.v2-public-play`. The previous deployment
at `2b98d872a0bf350e85fec82df24de0e37cc01028` remains the rollback target.
Auto-deploy remains off; subsequent evidence-only commits do not alter this
deployed source identity.

## Live verification

- The public page directly offers Warehouse, Pong and Kitchen without an access
  key or model version labels. Each card starts Task 1 with a frozen NN teammate.
- Actual Chrome interaction started all three games, performed an action in each,
  exited and switched games. Warehouse Space held for 650 ms advanced exactly one
  turn. During an active step the page visibly disabled movement buttons and
  indicated that the step must finish. Exit during a step returned to the chooser
  after completion, including in Kitchen.
- The bounded six-model checker passed 67 HTTP requests and 18 joint steps:
  exact actions, bindings and physical state; stable-command retries; restore;
  and independent exported replay. AI action overrides were zero. Intermediate
  floating-point differences across CPUs stayed within the checker's existing
  tolerances; executed actions and physical state were strictly equal.
- Seven served HTML/JavaScript/CSS files matched the deployed source byte for
  byte. The model catalog hash remains
  `ec7b6370454321748f9767c5cbbce4f630d539afb5763da742f6da6f7f543d43`.
- The predeployment 39 Python and 36 Node checks remain recorded separately;
  this deployment also reran the 24 manual frontend checks before publishing.

Evidence:

- [Six-model HTTP receipt](evidence/public-play-six-models-live-20261103.json)
- [Health and static asset receipt](evidence/public-play-static-live-20261103.json)
- [Actual live chooser screenshot](evidence/public-play-chooser-live-20261103.png)

## Scope

This publishes the manual playground only. All six Actor assets, action selection,
physics and scientific acceptance receipts are unchanged. No training or heldout
evaluation was run. Public play is not participant recruitment or a scientific
acceptance claim; pilot and recruitment remain closed. The researcher workflow
remains protected and separate at `/research/`, with participant storage untouched.

These checks do not establish a new latency percentile or remove free-service
cold-start and network delays. Historical pending/failed deployment receipts were
preserved rather than rewritten.
