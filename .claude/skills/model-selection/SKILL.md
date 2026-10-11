---
name: model-selection
description: |
  Recommends 2–3 candidate machine learning algorithms for a specific
  prediction task, based on findings from a prior EDA run. Reads
  docs/eda-findings.md and the user's stated target variable, infers
  the task type (regression, binary classification, multi-class,
  anomaly detection, clustering), applies a structured selection
  framework, and writes a full reasoning report to
  docs/model-selection.md. Proposes a src/train.py skeleton. Use
  whenever the user asks which model to use, what algorithm fits this
  data, or requests a model recommendation for a known target. Always
  requires EDA findings. Does NOT train models — that is a later SDD
  step. Dataset-agnostic; works for any tabular ML task in any domain.
---

# Model Selection

Help choose 2–3 candidate algorithms for a specific prediction task.
Do not train models. Produce a reasoned report and a training-script
skeleton.

The goal is **honest recommendations grounded in EDA findings**, not
the "best" model. The user will train candidates later and pick the
winner empirically. Narrow the field from "every model" to "these 3
make sense."

This skill runs inside an SDD cycle, **after EDA and before Feature
Engineering and the Spec/Plan**. It never trains a model.

---

## Inputs You Require

Confirm both exist:

1. `docs/eda-findings.md` — if missing, stop and say: "Run EDA first
   (eda-methodology skill)."
2. Target variable — if not named in EDA, ask once: "Which column
   should the model predict?"

Optionally, if the Spec already exists and fixes a model (e.g., the
user requires logistic regression for reuse parity), read
`spec/spec.md` and honor it. Fixed model → this skill narrows
candidates *around* that anchor, not against it.

---

## Canonical Repo Layout (SDD)

| Artifact | Path |
|---|---|
| Reasoning report | `docs/model-selection.md` |
| Training skeleton | `src/train.py` (proposal only; not filled) |

---

## Infer the Task Type

From EDA, look at the target column's dtype and distribution.

| Target shape in EDA | Task type |
|---|---|
| Numeric, continuous | Regression |
| Categorical, 2 values | Binary classification |
| Categorical, 3–20 values | Multi-class classification |
| Numeric, semantically ordered bins | Ordinal |
| No target (unsupervised) | Anomaly detection or clustering |

State your inferred task type explicitly and ask for confirmation
before proceeding.

---

## The Selection Framework

Apply these six keys in order. Each rules out some families.

### Key 1 — Dataset size

Read shape from EDA.

- **< 200 rows** → linear models, shallow trees. No neural nets.
- **200–5,000 rows** → linear, random forest, gradient boosting
- **5,000–100,000 rows** → above + XGBoost/LightGBM
- **> 100,000 rows** → any of the above; consider neural nets if
  nonlinear

### Key 2 — Linearity

Check the EDA correlation matrix (numeric features vs. target).

- **Strong linear (|r| > 0.5)** → start with linear models
- **Weak or nonlinear** → tree-based or kernel
- **Mixed** → try both

### Key 3 — Feature types

From EDA, count numeric vs. categorical.

- **Mostly numeric** → any model
- **Mostly categorical** → tree-based (native handling) preferred
- **Mixed** → all families work; note encoding needs

### Key 4 — Interpretability

Ask once: "Do you need to explain individual predictions?"

- **Yes** → linear, logistic, decision tree, shallow RF
- **No** → gradient boosting, ensembles, deep models

### Key 5 — Class balance

Classification only. From EDA class distribution.

- **Imbalanced (< 15% minority)** → logistic with
  `class_weight='balanced'`, or trees with `scale_pos_weight`
- **Balanced** → no adjustment

### Key 6 — Time dimension

If EDA flagged a date column:

- **Time-series** → require time-based splits, add lag features.
  Algorithms: linear regression with lags, gradient boosting with lags,
  or ARIMA/Prophet for long series.
- **Not time-series** → random or stratified splits.

---

## Output Structure — `docs/model-selection.md`

```markdown
# Model Selection — <task name>

## Task type
<Binary classification / regression / ...>

## Inputs
- EDA: docs/eda-findings.md
- Target: <column name>
- Fixed model (if any): <e.g., LogisticRegression for reuse parity>

## Findings from EDA that drove selection
- Dataset size: <N rows × M cols>
- Linearity: <strong / weak / mixed>
- Feature mix: <N numeric, M categorical>
- Class balance: <% minority>
- Time dimension: <yes / no>

## Candidate models

### Candidate 1 — <name>
**Why it fits:** ...
**Why it might not:** ...
**Pros:** ...
**Cons:** ...

### Candidate 2 — <name>
...

### Candidate 3 — <name>
...

## Recommendation
<Which one to build first as baseline, which as stretch.>

## Evaluation metric
<Accuracy / F1 / AUC / RMSE / ...> — and why that metric given class
balance and business context.

## Split strategy
<Random + stratified / Time-based / Grouped> — with reason.

## Training skeleton proposal
- `src/train.py` with three functions: `load_data()`, `build_pipeline()`,
  `train_and_evaluate()`.
- This is a proposal only. Do not fill it in; that's the Build step.
```

---

## Rules

- **Grounded, not clever.** Every recommendation traces to an EDA
  finding.
- **Prefer interpretable when the user needs explanation.**
- **Always address class balance** for classification.
- **Always address leakage risk** if EDA flagged it.
- **Fixed models stay fixed.** If the Spec requires a specific model
  for reuse, propose it as Candidate 1 and add 1–2 alternates, not a
  full re-litigation.
- **No training.** This skill produces documentation and a skeleton.
- **Save to the canonical path.** `docs/model-selection.md`.

---

## When to Stop and Ask

Pause and ask if:
- EDA findings are missing
- Target column is ambiguous
- Task type can't be confidently inferred
- EDA flagged leakage but the target definition is unclear

Do not power through these.