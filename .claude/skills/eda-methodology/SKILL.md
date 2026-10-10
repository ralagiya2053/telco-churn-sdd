---
name: eda-methodology
description: |
  Guides exploratory data analysis (EDA) on any tabular dataset (CSV,
  Parquet, Excel, SQLite, or connected DB via MCP). Enforces a fixed
  order of operations so no critical check is skipped, and saves every
  step's script, every plot, and the final findings to permanent files
  so the analysis is reviewable and reproducible. Use whenever the
  user asks to explore, profile, understand, inspect, or "do EDA on" a
  dataset — even when the word "EDA" isn't used, if the intent is to
  understand data before modeling. Also applies before writing any
  feature spec or model spec, since EDA findings determine the Data
  Contract. Designed for use inside a Spec-Driven Development (SDD)
  cycle: this skill produces documentation and scripts only — it never
  trains a model.
---

# EDA Methodology

You are guiding exploratory data analysis. The goal is to understand a
dataset thoroughly enough that the next step — writing a Spec or Plan,
or engineering features — can be done with confidence and honesty.

EDA is a disciplined sequence, not a data dump. Follow the order below.
Run only the steps that apply to the data at hand. Skip conditional
steps whose preconditions aren't met, and say so explicitly.

This skill is used inside an SDD cycle. It runs **before** the Spec is
written, so its outputs become evidence the Spec cites. It never trains
a model; training is a later SDD step.

---

## Canonical Repo Layout (SDD)

All artifacts go to fixed paths. Do not invent new paths.

| Artifact | Path |
|---|---|
| Step scripts | `src/eda/step_N_<name>.py` |
| Plots | `src/eda/plots/step_N_<name>.png` |
| Findings report | `docs/eda-findings.md` |

Create directories if missing. Do not commit raw or sensitive data to
Git; EDA scripts, plots, and findings on non-sensitive data may be
committed.

---

## Artifacts — Save Everything to Disk

Every run produces permanent artifacts. Do not work purely in memory.

During the run:
- Save each step's script to `src/eda/step_N_<short_name>.py`
- Save every plot to `src/eda/plots/` with descriptive filenames
  - Use **PNG** (lossless)
  - Prefix with step number so files sort naturally

At the end:
- Write the full findings report to `docs/eda-findings.md`
  - Include tables, observations, and inline image references using
    relative paths: `![desc](../src/eda/plots/step_3_x.png)`

**Do not skip the saves.** A run that leaves no artifacts is incomplete.

---

## Data Source — Load Uniformly

Detect the source, load into a single pandas DataFrame, then proceed.
Only the loading step changes per source.

- **CSV** — `pd.read_csv()`. Watch for delimiter, encoding, header row,
  thousands separators. Report parsing warnings.
- **Parquet** — `pd.read_parquet()`. Fast, type-preserving.
- **Excel** — `pd.read_excel()`. List sheets if multiple; ask which.
- **SQLite** — `sqlite3.connect()` or project's `get_db()` if inside a
  Flask request. List tables, then `pd.read_sql_query()`.
- **MCP-connected DB** — use the MCP tool to list and query; load the
  result into pandas.

After loading, always report: source path, number of tables loaded,
shape of resulting DataFrame(s).

---

## The EDA Order

Run these in sequence. Skip steps whose preconditions aren't met and
say so in the findings.

### Step 1 — Structure: What is this dataset?

Always applies.

- Shape (rows × columns)
- Column names and dtypes
- Memory footprint
- Head and tail (5 rows each)
- Unique identifier columns — which column(s) identify a row?

Report as a table. No plots yet.

**Save:** `src/eda/step_1_structure.py`

### Step 2 — Nulls and duplicates: What's missing or repeated?

Always applies.

Nulls per column:
- Count and % of nulls
- Pattern — random, or clustered by another column/date/group?
- Likely cause — "no data" vs. "zero"?

Classify missingness:
- **MCAR** — Missing Completely At Random (safe to drop)
- **MAR** — depends on another feature (impute carefully)
- **MNAR** — depends on the value itself (investigate)

Duplicates:
- Fully duplicated rows
- Duplicates on identifier columns
- Decide: legitimate repeats or errors?

**Save:** `src/eda/step_2_nulls_duplicates.py`

### Step 3 — Univariate: One column at a time

Always applies.

Numeric columns:
- min, max, mean, median, std, quartiles
- Histogram and boxplot per column
- Outliers beyond **both** 1.5×IQR and 3σ

Categorical columns:
- Value counts, cardinality
- Rare categories (< 5 occurrences)
- Flag high cardinality (> 50 categories)

**Save:** `src/eda/step_3_univariate.py`
**Plots:** `src/eda/plots/step_3_<col>_{hist,box,count}.png`

### Step 4 — Target: What are we predicting?

Precondition: a target column has been identified or inferred.

- Class balance (classification) — flag any class < 15%
- Target distribution (regression) — histogram, skew, kurtosis
- Feature-target relationship for each feature
- Leakage check — any feature with |corr| > 0.9 to target is suspicious

**Save:** `src/eda/step_4_target.py`
**Plots:** `src/eda/plots/step_4_<col>_vs_target.png`

### Step 5 — Time-awareness: Only if a date or timestamp exists

Precondition: dataset contains a date or timestamp column.

- Sort by time, confirm ordering
- Autocorrelation (lag-1, lag-7)
- Trend and seasonality
- Split strategy — **always time-based for time-series, never random**
- Effective sample size after removing gaps

Skip and say so if no date column exists.

**Save:** `src/eda/step_5_time.py`

### Step 6 — Correlations: Relationships between features

Precondition: at least two columns.

Run only after Steps 1–5.

- Numeric–numeric: Pearson and Spearman
- Categorical–numeric: ANOVA or Kruskal–Wallis
- Categorical–categorical: Chi-square
- Flag any pair with |corr| > 0.85 as a multicollinearity concern

**Save:** `src/eda/step_6_correlations.py`
**Plots:** `src/eda/plots/step_6_corr_heatmap.png`

### Step 7 — Findings: What did we learn?

Always run. Write to `docs/eda-findings.md`.

Format:

```
# EDA Findings — <dataset name>

## What I checked
- Steps run; steps skipped with reason.

## What I found
- Shape, nulls, distributions, target balance, autocorrelation,
  outliers, notable correlations, cardinality concerns.

## What it implies for modeling
- Which features need imputation and how.
- Which features need scaling.
- Which are redundant.
- Which split strategy to use.
- Expected baseline performance.
- Any suspected leakage.
- Any feature dropped due to cardinality or sparsity.

## Open questions
- Decisions needed before proceeding.
```

Embed plot references inline with relative paths.

---

## Rules

- **Order matters.** Never run correlations before univariate. Never
  run modeling before EDA is complete.
- **Conditional execution.** Skip steps whose preconditions aren't met,
  and say so in findings.
- **Distinguish "missing" from "zero."** A null reading is often
  "wasn't recorded," not zero.
- **Never random-split time-series.** If a date column exists, split by
  time or you leak future information.
- **Check class imbalance before classification.** If minority < 15%,
  accuracy is misleading — report F1/AUC.
- **Flag leakage explicitly.** Any feature with |corr| > 0.9 to target
  is suspicious until proven otherwise.
- **Report outliers both ways.** 1.5×IQR and 3σ.
- **Honest reporting.** If the dataset is too small (< 30 rows per
  class for classification), say so.
- **Domain guardrails.** Report facts, not interpretations. Do not
  diagnose, reassure, or alarm. That applies across domains.
- **Save everything.** Scripts, plots, findings must persist to disk.
- **No model training.** This skill produces artifacts; training is a
  later SDD step.

---

## When to Stop and Ask

Pause and ask the user if:
- Multiple tables and it's unclear which to analyze
- Target column can't be inferred and none was named
- A "column" is actually an ID (drop before analysis, or keep?)
- More than 8 numeric columns (pairplot won't scale)
- Any finding looks like leakage

Do not power through these.

---

## Output — What to Show in Chat

For each step, show a short summary:

```
Step N — <name>
- 2–5 bullets of what was found
Saved: src/eda/step_N_<name>.py
Plot(s): src/eda/plots/step_N_<name>.png
```

Do not print raw code output unless short. Summarize into tables. Full
detail lives in the saved files.

At the end, present the findings summary and confirm all artifacts
were saved.