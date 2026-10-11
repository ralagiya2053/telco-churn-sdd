---
description: Stage, commit with a conventional commit message, and push the current feature branch. Does NOT merge into main.
---

Stage all changes, write a Conventional Commit message (type: subject,
with a body explaining what changed and why), commit on the current
branch, and push to the current branch's upstream.

Rules:
- Never merge into main.
- Never switch branches.
- Never delete the branch.
- Use Conventional Commits: feat, fix, chore, docs, refactor, test, style, perf, ci, build.
- Keep the subject under 72 chars, imperative mood, no trailing period.
- The body should reference the SDD step we're on if relevant
  (e.g., "SDD Step 2: Spec" or "SDD Step 8: Build").

After pushing, print the branch name and the commit hash, then stop.