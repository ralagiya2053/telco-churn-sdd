# Model Selection — Telco Customer Churn

## Task type
Binary classification

## Inputs
- EDA: docs/eda-findings.md
- Target: Churn
- Fixed model: LogisticRegression (required for reuse parity with diabetes-prediction pipeline)

## Findings from EDA that drove selection
- Dataset size: 7,043 rows × 21 columns (suitable for linear and ensemble models)
- Linearity: Mixed. `tenure` shows a moderate linear relationship (corr: -0.35), while categorical features show strong statistically significant relationships.
- Feature mix: Mixed (approx. 4 numeric, 16 categorical).
- Class balance: 26.5% minority class ("Yes"). No severe imbalance, but F1/AUC are required.
- Time dimension: No (random stratified split recommended).

## Candidate models

### Candidate 1 — Logistic Regression
**Why it fits:** Required for reuse parity. It serves as an excellent, interpretable baseline for binary classification.
**Why it might not:** May struggle with complex non-linear interactions between features.
**Pros:** Highly interpretable (coefficients), fast to train, low risk of overfitting on this dataset size.
**Cons:** Assumes linear relationship between log-odds and features.

### Candidate 2 — Random Forest Classifier
**Why it fits:** Handles mixed feature types (numeric and categorical) exceptionally well and captures non-linear relationships.
**Why it might not:** Less interpretable than Logistic Regression.
**Pros:** Robust to outliers, handles categorical features natively (after encoding), provides feature importance.
**Cons:** Can overfit if not tuned; slower inference than linear models.

### Candidate 3 — XGBoost / Gradient Boosting
**Why it fits:** State-of-the-art for tabular data; likely to provide the highest predictive performance.
**Why it might not:** Overkill for simple linear relationships; requires more careful hyperparameter tuning.
**Pros:** High accuracy, handles imbalance well via `scale_pos_weight`.
**Cons:** Black-box nature compared to Candidate 1.

## Recommendation
Build **Logistic Regression** first as the mandatory baseline. Use **Random Forest** as a secondary candidate to check for non-linear gains. **XGBoost** can be used as a stretch goal for maximum performance.

## Evaluation metric
**F1-Score** and **AUC-ROC**. 
Accuracy is misleading since the baseline is ~73.5%. F1-Score ensures we balance precision and recall for the churners (the minority class).

## Split strategy
**Stratified Random Split**. 
Since there is no time dimension, a stratified split ensures that the ~26.5% churn rate is preserved across training and testing sets.

## Training skeleton proposal
- `src/train.py` with three functions: `load_data()`, `build_pipeline()`, `train_and_evaluate()`.
- This is a proposal only. Do not fill it in; that's the Build step.
