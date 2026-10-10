---
name: feature-engineering
description: |
  Designs and generates the feature engineering pipeline for a machine
  learning task, based on findings from a prior EDA run. Reads
  docs/eda-findings.md and docs/model-selection.md, decides which raw
  columns become features, creates derived features (rolling windows,
  lags, deltas, ratios, encodings), handles nulls appropriately, flags
  leakage risks, and writes a reusable src/features.py plus a reasoning
  report to docs/feature-engineering.md. Use whenever the user asks to
  engineer features, build the feature pipeline, prepare data for
  training, create src/features.py, or transform raw columns into
  model-ready inputs after EDA and model selection. Requires both prior
  reports to exist. Does NOT train models — that is a later SDD step.
  Dataset-agnostic; works for any tabular ML task in any domain.
---

# Feature Engineering

Turn raw columns into model-ready features. Produce a `src/features.py`
module and a reasoning report that the training step consumes.

The goal is **features grounded in the EDA and model-selection
findings**, not a maximal feature set. Fewer, well-justified features
beat a kitchen-sink approach — especially on small datasets.

This skill runs inside an SDD cycle, **between Model Selection and the
Spec/Plan**. It never trains a model; training is a later SDD step.

---

## Inputs You Require

Confirm these exist before starting:

1. `docs/eda-findings.md` — if missing, stop and say: "Run EDA first
   (eda-methodology skill)."
2. `docs/model-selection.md` — if missing, stop and say: "Run model
   selection first (model-selection skill)."
3. The target variable — usually declared in `docs/model-selection.md`.
   If not, ask once.

Do not proceed until all inputs exist.

---

## Canonical Repo Layout (SDD)

| Artifact | Path |
|---|---|
| Feature module | `src/features.py` |
| Reasoning report | `docs/feature-engineering.md` |

---

## The Framework

Apply these steps in order. Every decision must trace back to a finding
in the EDA report or a constraint in the model-selection report.

### Step 1 — Feature inventory

List every column. Classify:

| Column | Type | Role |
|---|---|---|
| `x` | numeric | feature |
| `y` | categorical | feature (needs encoding) |
| `id_col` | id | **drop — not a feature** |
| `time_col` | date | **time key — not a feature** |

Drop ID columns and free-text columns unless they carry signal the
model needs. Never feed a `user_id` or a raw timestamp to a model.

### Step 2 — Target-adjacent rules

For each remaining feature, ask: does this leak the target?

- Same-time feature predicting same-time target — leakage
- Future feature predicting present target — leakage
- Present feature predicting future target — legitimate
- Past target as a feature — legitimate (lag)

If EDA flagged autocorrelation, the target's own lag is often the
strongest legitimate feature. Include it.

### Step 3 — Derived features

Add only derived features the EDA supports:

- **Rolling windows** (when EDA shows autocorrelation): `{col}_7d_mean`,
  `{col}_30d_mean`, `{col}_7d_std`
- **Lag features** (when EDA flags time-series): `{col}_lag1`,
  `{target}_lag1`
- **Deltas** (when EDA shows day-to-day variation): `{col}_delta1`,
  `{col}_delta_from_baseline`
- **Ratios** (when two columns are meaningfully related):
  `a_over_b = a / b`
- **Time features** (when split is time-based): `day_of_week`,
  `is_weekend`, `month` (cyclical — use sin/cos or one-hot)

Do not invent features the EDA didn't suggest.

### Step 4 — Categorical encoding

From EDA, list categorical columns and their cardinality.

| Cardinality | Encoding |
|---|---|
| 2 values | Binary (0/1) |
| 3–10 values | One-hot |
| 11–50 values | One-hot if tree-based; target encoding for linear |
| > 50 values | Target encoding, or drop |
| Ordinal (Mild/Moderate/Severe) | Ordinal integer |

Record the encoding decision in the report.

### Step 5 — Null handling

For each column with nulls (from EDA):

- **MCAR + numeric** → median
- **MCAR + categorical** → mode, or drop row
- **MAR** → impute conditioned on the correlated column
- **MNAR** → do not impute blindly; add a `_missing` flag column
- **Null rate > 50%** → consider dropping the column

Do not silently fill with 0 unless 0 is the true semantic value.

### Step 6 — Scaling

**Do not apply scaling in `src/features.py`.** Return raw feature
values. Scaling is model-specific and belongs inside the training
pipeline.

### Step 7 — Split strategy

From EDA findings and model-selection report:

- **Time-series** → split by date. Train on earliest, test on latest.
- **Cross-sectional** → random split; stratified if classification.
- **Grouped** → split by group if generalization is the goal; split by
  row if personalization is the goal. State which.

Document split boundaries (or ratio + seed) in the report.

### Step 8 — Leakage check

Before finishing, verify the final feature list:

- Does any feature depend on the target's value at the same time?
- Does any feature come from a future period relative to the target?
- Does any feature correlate > 0.95 with the target?

If yes to any, drop or reframe. Document the decision.

---

## Outputs

### 1. `src/features.py`

A Python module with one public function:

```python
def build_features(source, **kwargs) -> pd.DataFrame:
    """
    Load raw data and return a feature-ready DataFrame with one row
    per prediction unit.

    Returns
    -------
    df : pd.DataFrame
        Feature columns + target column. Ready to split and train.
    """
```

The function must be deterministic and side-effect free (except reads).

### 2. `docs/feature-engineering.md`

Sections:

- **Inputs** — what reports this was based on
- **Final feature list** — table of features and their sources
- **Derived features and why** — trace each to an EDA finding
- **Encodings** — table of categorical columns and chosen encoding
- **Null handling** — table of columns with nulls and chosen strategy
- **Split strategy** — exact boundaries or ratio + seed
- **Leakage review** — features checked and findings
- **Dropped columns** — what was dropped and why

---

## Rules

- **Grounded, not inventive.** Every derived feature must trace to an
  EDA finding or a model-selection constraint.
- **No scaling in features.py.** Scaling happens in the training
  pipeline.
- **No training.** This skill produces code and documentation only.
- **Leakage check is mandatory.** Never skip Step 8.
- **Save to the canonical paths.** Do not invent new directories.
- **Fewer features, better justified.** Prefer parsimony.

---

## When to Stop and Ask

Pause and ask the user if:
- EDA or model-selection reports are missing
- Target column is ambiguous
- A derived feature's justification is unclear
- Leakage is suspected in any feature

Do not power through these.