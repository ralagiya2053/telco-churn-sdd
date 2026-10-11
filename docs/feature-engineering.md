# Feature Engineering Report — Telco Customer Churn

## Inputs
This feature engineering pipeline was designed based on the following reports:
- `docs/eda-findings.md`
- `docs/model-selection.md`

## Final Feature List
The final dataset consists of the following features:

| Feature | Source | Type | Note |
|---|---|---|---|
| `tenure` | Raw | Numeric | Months with the company |
| `MonthlyCharges` | Raw | Numeric | Monthly bill amount |
| `TotalCharges` | Raw | Numeric | Total billed amount |
| `SeniorCitizen` | Raw | Binary | 0 = No, 1 = Yes |
| `gender`, `Partner`, `Dependents`, etc. | Raw $\rightarrow$ OHE | Binary | One-Hot Encoded categorical features |
| `Churn` | Raw $\rightarrow$ Map | Binary | **Target**: 0 = No, 1 = Yes |

## Derived Features and Why
No complex derived features (e.g., ratios, rolling windows) were added as the EDA did not suggest a significant performance gain for this cross-sectional dataset. The primary transformations are encoding and cleaning.

## Encodings
All categorical columns were found to have low cardinality ($\le 4$) in the EDA.

| Column Group | Encoding | Justification |
|---|---|---|
| All categorical features | One-Hot Encoding (OHE) | Low cardinality; suitable for Logistic Regression. |
| `Churn` (Target) | Binary Mapping | $\{Yes: 1, No: 0\}$ for binary classification. |

## Null Handling
| Column | Strategy | Justification |
|---|---|---|
| `TotalCharges` | Drop Rows | 11 rows were NaN (correlated with `tenure=0`). Per project constraints, these were dropped rather than imputed to avoid semantic distortion. |

## Split Strategy
- **Method:** Stratified Random Split.
- **Justification:** Since there is no time-series component and the churn rate is $\approx 26.5\%$, stratification ensures class balance is preserved across training and test sets.

## Leakage Review
- **Check:** No features depend on the target's value at the same time.
- **Check:** No future-looking features are present.
- **Check:** No feature correlates $> 0.9$ with the target (verified in `docs/eda-findings.md`).
- **Conclusion:** No leakage detected.

## Dropped Columns
- `customerID`: Dropped immediately as it is a unique identifier with no predictive value.
- `TotalCharges` NaNs: 11 rows dropped to maintain data integrity.
