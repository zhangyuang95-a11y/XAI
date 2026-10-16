# v6 Git recovery and branch inventory (2026-10-16)

The original `XAI` and `XAI-study-v3` checkouts remain dirty and untouched. Their tracked patches, staged patches, all 539 non-ignored untracked files, file hashes, selected ignored model/release receipts, and a complete Git refs bundle are in `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/`. The bundle passed `git bundle verify`; each tracked patch was applied in a separate restore clone and compared byte for byte. The 82 GB ignored `XAI/output` tree was not copied.

The integration worktree is `/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration`. PR [#5](https://github.com/zhangyuang95-a11y/XAI/pull/5) merged Task 2, the current main history, and the v6 architecture into `main` as `f7a4893` while keeping pilot closed. Work for the separately gated model release continues on `codex/warehouse-v6-release-20261016`, which starts at that merge commit. No source checkout was reset or cleaned. Seven stale `/private/tmp` worktree metadata entries were pruned; the four live worktrees remain.

| Remote branch | Tip SHA | Archive tag | Status |
| --- | --- | --- | --- |
| `exp` | `51ea573c6b4b7ade9036b5967c294aceaea5ef4c` | `archive/branches/exp-20261016` | Remote branch deleted with tip lease; archive tag retained. |
| `sep2-bubble` | `e17a672867b4dd9c120d52d35cdea0edb2b26bb6` | `archive/branches/sep2-bubble-20261016` | Remote branch deleted with tip lease; archive tag retained. |
| `sep2-rollback-20260902` | `af97df8589080a6b1b79bad591059b4d52fe33ee` | `archive/branches/sep2-rollback-20260902-20261016` | Remote branch deleted with tip lease; archive tag retained. |
| `sep2-rule-assisted-20260917` | `0cd25c8dea6f6946e3efb93310d189e6732ba0bb` | `archive/branches/sep2-rule-assisted-20260917-20261016` | Remote branch deleted with tip lease; archive tag retained. |

All four annotated archive tags are pushed and their peeled commits match the archived branch tips. Before deletion, the GitHub API showed zero open PRs and zero GitHub deployments, repository text/configuration had no explicit reference to these branches, and the Render dashboard showed that the workspace's two current services tracked `codex/three-domain-study-v3-20260920` (Warehouse) and `main` (Kitchen). Each remote tip still matched its archived SHA when deleted using an explicit tip lease; a subsequent `git ls-remote --heads` returned no matching branch. After PR #5 merged, its integration branch tip `b5ea958` was also archived at `archive/branches/three-domain-v6-integration-20261016-merged`, verified against the pushed peeled tag, and deleted with a tip lease. `main`, `core`, the active research v3 and Task 2 branches, `live`, and the active v6 release branch remain outside this cleanup. The Warehouse Render service still uses the research v3 branch, so that branch must remain until its deployment is intentionally migrated.

The two local `sep2-bubble` and `sep2-rule-assisted-20260917` refs were deleted after `git branch -d` confirmed that both tips are already contained in `origin/main`. Their exact former refs remain in the verified recovery bundle and their remote archive tags. The local `render-pong-hub` alias and its historical commits remain available; the dirty primary `main` and research v3 worktrees were left in place. A fresh `git ls-remote --heads origin` listed only `main`, `core`, Task 2, research v3, live, and the v6 release branch.

The Task 2, research v3, and live remote tips are all ancestors of `origin/main`,
but ancestry alone is not a deletion signal. `docs/task2_share_preview.md`
still names the dedicated Task 2 deployment branch. The Warehouse Render
service still tracks research v3, and the live branch needs its deployment
references checked before retirement. Those branches remain available until
their external uses are resolved.

On 2026-10-16, a fresh read-only check found exactly six GitHub heads:
`main` (`f7a4893`), `core` (`bd08939`), Task 2 (`4eecd7d`), research v3
(`b7ccfcf`), `live` (`cd63f02`), and the v6 release branch (`95faf64`).
GitHub reported no open pull requests, Deployments, or environments. Render's
Production environment directly showed Warehouse building from research v3
and Kitchen building from `main`. The account-wide Render service list was
initially unavailable. A subsequent inspection found one visible workspace,
one project, one Production environment, and exactly those two services; no
service used `live`. Task 2 remains referenced by its preview deployment
instructions. `core` remains.

The historical `live` tip `cd63f02ef21e3c79d2b6972c1d2909e1903e3ac8`
was archived in the pushed annotated tag
`archive/branches/three-domain-study-live-20260920-20261016`. The remote tag's
peeled commit matched that exact tip. With zero open PRs/GitHub Deployments,
no active checkout, and no current Render service reference in the inspected
account, the remote branch was deleted with an explicit tip lease. A fresh
`git ls-remote --heads` returned no matching branch; the archive tag remains.
The merged local alias was also removed with `git branch -d`. Task 2 and
research v3 remain because their external deployment references have not been
retired.
