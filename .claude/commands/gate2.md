---
description: SDD Gate 2 — open a PR from the feature branch to main containing code, tests, and validation results, then merge via squash after review.
---

We are at SDD Gate 2. Validation has passed and the code is ready
for review.

Steps:
1. Verify tests/ or validation scripts exist and have run
   successfully. If no validation artifacts exist, stop and warn me.
2. Stage src/, tests/, model/ (only non-ignored artifacts),
   Dockerfile, requirements.txt, README.md, and any deployment files.
   Commit with message:
   "feat: add telco churn prediction pipeline (SDD Gate 2)"
3. Push the current feature branch.
4. Create a pull request into main titled:
   "Code: <project> (SDD Gate 2)"
   Body should include: what was built, which spec sections it
   satisfies, validation results (metrics), and links to
   docs/eda-findings.md and docs/model-selection.md.
5. STOP. Do not merge. Wait for my approval on GitHub. This is the
   human gate.

After I confirm merge, run: squash merge, switch to main, pull latest,
and delete the feature branch locally and remotely.
