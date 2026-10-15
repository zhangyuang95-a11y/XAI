# v6 Git recovery and branch inventory (2026-10-16)

The original `XAI` and `XAI-study-v3` checkouts remain dirty and untouched. Their tracked patches, staged patches, all 539 non-ignored untracked files, file hashes, selected ignored model/release receipts, and a complete Git refs bundle are in `/Users/zhangyuang/Desktop/ICLR/XAI-v6-recovery-20261015T165508Z/`. The bundle passed `git bundle verify`; each tracked patch was applied in a separate restore clone and compared byte for byte. The 82 GB ignored `XAI/output` tree was not copied.

The integration worktree is `/Users/zhangyuang/Desktop/ICLR/XAI-v6-integration` on `codex/three-domain-v6-integration-20261016`. It started from Task 2 tip `4eecd7d586c73390f75d94d4dd64317952d2dea0` and includes `origin/main` through merge commit `f3b1bdf8c18ddc26046a2de2f386fa9a840e401c`. No source checkout was reset or cleaned. Seven stale `/private/tmp` worktree metadata entries were pruned; the four live worktrees remain.

| Remote branch | Tip SHA | Archive tag | Status |
| --- | --- | --- | --- |
| `exp` | `51ea573c6b4b7ade9036b5967c294aceaea5ef4c` | `archive/branches/exp-20261016` | Retained; tip is already in `main` and v6. |
| `sep2-bubble` | `e17a672867b4dd9c120d52d35cdea0edb2b26bb6` | `archive/branches/sep2-bubble-20261016` | Retained; tip is in v6. |
| `sep2-rollback-20260902` | `af97df8589080a6b1b79bad591059b4d52fe33ee` | `archive/branches/sep2-rollback-20260902-20261016` | Retained; tip is in v6. |
| `sep2-rule-assisted-20260917` | `0cd25c8dea6f6946e3efb93310d189e6732ba0bb` | `archive/branches/sep2-rule-assisted-20260917-20261016` | Retained; tip is in v6. |

All four annotated archive tags are pushed and their peeled commits match the branch tips. The public GitHub API reported zero open PRs and zero GitHub deployments, and repository text/configuration has no explicit reference to these branches. Render's dashboard requires a fresh login, so external service branch references remain unverified. **No remote branch was deleted.** `main`, `core`, the research v3 branch, Task 2, and live remain outside this cleanup.
