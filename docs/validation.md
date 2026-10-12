# Model Validation Report

**Date:** 2026-10-11
**Environment:** Python 3.14.6, scikit-learn 1.9.1

## Dataset Summary
- **Source Path:** `data/train.csv`
- **Total Rows:** 7,032
- **Churn Rate:** 26.5% (approx)
- **Train/Test Split:** 80% Train / 20% Test (1,125 Train / 282 Test samples approx)

## Performance Metrics
| Metric | Value |
| :--- | :--- |
| Accuracy | 0.7257 |
| ROC AUC | 0.8353 |
| Recall (Churn=1) | 0.79 |
| Precision (Churn=1) | 0.49 |
| F1 (Churn=1) | 0.61 |

## Confusion Matrix
```
[[724 309]
 [ 77 297]]
```

## Success Gate
| Gate | Threshold | Actual | Result |
| :--- | :--- | :--- | :--- |
| ROC AUC | > 0.75 | 0.8353 | **PASS** |
| Recall (Churn=1) | > 0.70 | 0.79 | **PASS** |
| F1 (Churn=1) | > 0.55 | 0.61 | **PASS** |

**Overall Verdict: PASS**

## Artifacts
- `model/model.pkl` (1.1K)
- `model/scaler.pkl` (927B)
- `model/encoder.pkl` (4.7K)
- `model/feature_order.json` (851B)
