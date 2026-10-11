# EDA Findings — Telco Customer Churn

## What I checked
- **Step 1 (Structure):** Analyzed shape, dtypes, and identifiers.
- **Step 2 (Nulls & Duplicates):** Checked for missing values and repeated rows.
- **Step 3 (Univariate):** Examined distributions, outliers, and cardinality.
- **Step 4 (Target):** Analyzed class balance and feature-target relationships.
- **Step 5 (Time-awareness):** Skipped (no date/timestamp columns present).
- **Step 6 (Correlations):** Evaluated numeric and categorical relationships.

## What I found
- **Shape:** 7,043 rows × 21 columns.
- **Nulls:** Only `TotalCharges` contains nulls (11 rows, ~0.16%). These likely occur for new customers with 0 tenure.
- **Distributions:**
    - `tenure` ranges from 0 to 72 months.
    - `MonthlyCharges` (18.25 to 118.75) and `TotalCharges` show no extreme outliers beyond 3σ.
    - `SeniorCitizen` is binary (16.2% are seniors).
- **Target Balance:** `Churn` is binary; "Yes" represents ~26.5% of the dataset. No severe imbalance.
- **Key Relationships:**
    - `tenure` is the strongest numeric predictor (corr: -0.35), where higher tenure reduces churn probability.
    - Nearly all categorical features (Contract, InternetService, PaymentMethod) show statistically significant relationships with `Churn` (p < 0.0001).
- **Cardinality:** All categorical features have low cardinality (max 4), which is ideal for encoding.
- **Multicollinearity:** No numeric pairs exceeded |corr| > 0.85.

## What it implies for modeling
- **Imputation:** `TotalCharges` nulls should be imputed with 0 or the median, depending on the tenure value.
- **Scaling:** `tenure`, `MonthlyCharges`, and `TotalCharges` are on different scales and will require scaling (e.g., StandardScaler) for distance-based models.
- **Encoding:** Categorical features are low-cardinality and suitable for One-Hot Encoding.
- **Split Strategy:** Since no time-series component exists, a standard stratified random split is appropriate to maintain the churn class ratio.
- **Leakage:** No feature showed correlation > 0.9 with the target; no obvious leakage detected.
- **Baseline:** The baseline accuracy (predicting "No" for all) is ~73.5%. Models should be evaluated using F1-score or AUC-ROC due to the 26.5% minority class.

## Open questions
- Should `TotalCharges` nulls be treated as a separate category or imputed? (Current evidence suggests they correlate with 0 tenure).
- Are there any specific business constraints on the model (e.g., interpretability vs. raw performance)?

## Artifacts
- **Scripts:** `src/eda/step_1_structure.py` through `src/eda/step_6_correlations.py`
- **Plots:** Found in `src/eda/plots/`
