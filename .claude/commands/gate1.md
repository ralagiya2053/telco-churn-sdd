---
description: SDD Gate 1 — open a PR from the feature branch to main containing ONLY spec/ and plan/ and docs/, then merge via squash after review.
---

We are at SDD Gate 1. The feature branch should contain only
documentation (spec/, plan/, docs/) — no code yet.

Steps:
1. Verify no files under src/ or model/ or tests/ are staged. If any
   exist, stop and warn me.
2. Stage spec/, plan/, docs/, and any config files that are part of
   the Spec+Plan deliverable. Commit with message:
   "docs: add spec and plan for <project> (SDD Gate 1)"
3. Push the current feature branch.
4. Create a pull request into main titled:
   "Spec + Plan: <project> (SDD Gate 1)"
   Body should summarize: the business objective, the model choice,
   the pipeline architecture, and include a link to spec/spec.md and
   plan/plan.md. Mark as ready for review.
5. STOP. Do not merge. Wait for me to review and approve the PR on
   GitHub. This is the human gate.

After I confirm merge, run: squash merge, pull main, and continue the
SDD cycle on the same feature branch (do NOT delete it — we still need
it for Gate 2).
