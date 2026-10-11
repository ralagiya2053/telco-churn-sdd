# Specification: Telco Customer Churn Prediction Pipeline

## 1. Project Overview
- **What we are building:** A logistic regression churn prediction pipeline for the Telco Customer Churn dataset.
- **Why:** This project serves as a reusable SDD template for binary classification ML projects, mirroring a prior diabetes-prediction pipeline in a new domain.

## 2. Business Objective
- **Goal:** Predict which customers are likely to churn.
- **Value:** Churn prediction is critical for improving customer retention, as the cost of retaining an existing customer is significantly lower than the cost of acquiring a new one.

## 3. Data Contract
- **Source:** Kaggle IBM Telco Customer Churn dataset, file at `data/train.csv`.
- **Shape:** 7,043 rows × 21 columns (per `docs/eda-findings.md`).
- **Target Column:** `Churn` (binary, ~26.5% "Yes" per `docs/eda-findings.md`).
- **Feature Set:**
    - **Numeric Features:** `tenure`, `MonthlyCharges`, `TotalCharges`.
    - **Binary Pass-through:** `SeniorCitizen`.
    - **Categorical Features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
    - **Dropped Columns:** `customerID` (unique identifier).
- **Preprocessing Requirements:**
    - **Numeric Coercion:** Coerce `TotalCharges` to numeric.
    - **Null Handling:** Drop rows where `TotalCharges` is NaN (11 rows, corresponding to `tenure=0` customers — "not billed yet" is not "$0 billed").
    - **Target Mapping:** Map `Churn` from `{"Yes", "No"}` to `{1, 0}`.

## 4. Model Requirements
- **Task:** Binary classification.
- **Model:** `LogisticRegression` with `class_weight` for class imbalance.
- **Justification:** Chosen for interpretability and to maintain reuse parity with the prior diabetes-prediction pipeline.
- **Imbalance Handling:** Performance must be reported using F1-score and AUC-ROC, as accuracy is misleading given the 26.5% minority class.
- **Minimum Performance Threshold:** 
    - Must beat the 73.5% majority-class baseline on accuracy (per `docs/eda-findings.md`).
    - Must achieve a recall > 0.70 on the churn class (per `docs/model-selection.md`).

## 5. Pipeline Requirements
- **Data Splitting:** Stratified train/test split, `test_size=0.2`, `random_state=42`.
- **Scaling:** `StandardScaler` for numeric features, fitted on training data only.
- **Encoding:** `OneHotEncoder` (`drop='first'`, `sparse_output=False`, `handle_unknown='ignore'`) for categorical features, fitted on training data only.
- **Consistency:** Scaler and encoder must be applied identically to both training and testing sets.
- **Reusability:** The pipeline must be designed so that swapping the dataset and configuration is the only change required to adapt it to another binary classification problem.

## 6. Output Artifacts
- `model/model.pkl` — Trained logistic regression model.
- `model/scaler.pkl` — Fitted `StandardScaler`.
- `model/encoder.pkl` — Fitted `OneHotEncoder`.
- `model/feature_order.json` — Ordered list of feature columns.

## 7. Validation Criteria
- **Metrics:** Report accuracy, confusion matrix, and a full classification report (precision, recall, f1 for both classes).
- **Method:** The confusion matrix must be computed on the held-out test set.
- **Success Gate: (multiple, must all pass):**
    - **ROC AUC > 0.75** — the primary gate. Handles class imbalance
      properly and is threshold-independent.
    - **Recall on the churn class (1) > 0.70** — ensures we catch most
      churners.
    - **F1 on the churn class (1) > 0.55** — balances precision and
      recall on the minority class.
    - **Note on accuracy:** Raw accuracy is NOT used as a success gate. As stated in Section 4, accuracy is misleading on this dataset's 26.5% class imbalance. A majority-class-only classifier reaches 73.5% accuracy while catching zero churners. Reporting accuracy is required, but gating on it is not.
    - **Failure Loop:** If any success gate fails, document the failure in `docs/validation.md`, adjust `class_weight`, feature engineering, or model configuration as appropriate, retrain, and re-validate.
- **Documentation:** All validation artifacts must be saved to `docs/validation.md`.

## 8. Deployment Requirements (Out of Scope for v1)
- **Architecture:** Dockerized inference endpoint.
- **Platform:** Deployable to AWS Lambda behind API Gateway.
- **Inference Logic:**
    - Load `model.pkl`, `scaler.pkl`, `encoder.pkl`, and `feature_order.json`.
    - Accept a single customer record.
    - Return the churn probability and the predicted class.

## 9. Reusability Goal
To reuse this pipeline for another binary classification problem, only the following should change:
- Data source path.
- Target column name.
- Lists of numeric and categorical columns.
Everything else (pipeline logic, artifact formats, validation reporting) must remain identical.

## 10. Out of Scope (v1)
- Hyperparameter tuning.
- Tree-based or neural network models.
- Automated retraining pipelines.
- Real-time streaming data.

## 11. Evidence
This specification is based on the following source documents:
- `docs/eda-findings.md`
- `docs/model-selection.md`
- `docs/feature-engineering.md`
- `src/features.py`
