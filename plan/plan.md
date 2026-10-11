# Implementation Plan: Telco Customer Churn Prediction Pipeline

This plan details the "how" for implementing the requirements specified in `spec/spec.md`.

## 1. File & Folder Layout

- `src/train.py`: The main training script containing data loading, pipeline construction, training, and artifact saving.
- `tests/test_pipeline.py`: Pytest suite to verify data cleaning, pipeline consistency, and artifact generation.
- `model/`: Directory to store serialized model artifacts.
    - `model.pkl`: The trained `LogisticRegression` model.
    - `scaler.pkl`: The fitted `StandardScaler`.
    - `encoder.pkl`: The fitted `OneHotEncoder`.
    - `feature_order.json`: The definitive order of features for inference.
- `docs/validation.md`: Report containing model metrics and the final success gate verdict.
- `Dockerfile`: Environment definition for the inference endpoint.
- `src/handler.py`: AWS Lambda handler for serving predictions.
- `requirements.txt`: Project dependencies (created during the Build step).
- `src/features.py`: Canonical feature builder (created in Step 0). train.py imports build_features() from here.

## 2. Implementation Order (Task Breakdown)

1. **Task 1: Refactor `src/features.py`**
   - Remove one-hot encoding from `features.py`.
   - Return raw numeric + raw categorical columns + target.
   - Keep the cleaning logic: drop `customerID`, coerce `TotalCharges`
     to numeric and drop NaNs, map `Churn` to `{1, 0}`.

2. **Task 2: Implement `load_and_prepare()` in `src/train.py`**
   - Import and call `features.build_features()` to get cleaned data.
   - Define numeric and categorical column lists.
   - Create a `ColumnTransformer` using `StandardScaler` (numeric)
     and `OneHotEncoder` (categorical, `drop='first'`).

3. **Task 3: Implement `build_pipeline()` in `src/train.py`**
   - Chain `ColumnTransformer` + `LogisticRegression`
     (`class_weight='balanced'`) into a single sklearn `Pipeline`.

4. **Task 4: Implement `train_and_evaluate()` in `src/train.py`**
   - Perform stratified train/test split (`test_size=0.2`,
     `random_state=42`).
   - Fit the pipeline on training data.
   - Generate predictions on the test set.
   - Compute accuracy, confusion matrix, and classification report.

5. **Task 5: Implement Artifact Saving in `src/train.py`**
   - Extract and save the fitted scaler, encoder, and model as `.pkl`
     files.
   - Export the final feature column order to `feature_order.json`.

6. **Task 6: Create `tests/test_pipeline.py`**
   - Test data cleaning logic.
   - Verify stratified split.
   - Ensure scaler/encoder are fit only on training data.
   - Verify all 4 artifact files are created after training.

7. **Task 7: Create `Dockerfile` for Inference**
   - Use a slim Python base image.
   - Install required runtime dependencies.
   - Copy `src/handler.py` and `model/` directory.

8. **Task 8: Create `src/handler.py` for AWS Lambda**
   - Implement logic to load artifacts.
   - Process a single input record.
   - Return churn probability and class.

9. **Task 9: Write `docs/validation.md`**
   - Run the full training script.
   - Capture metrics and document the "Success Gate" (Recall > 0.70).

## 3. Dependencies

- **Training & Testing:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `pytest`.
- **Inference/Deployment:** `boto3` (for Lambda environment), `FastAPI` (if local wrapper is needed for testing).

## 4. Testing Strategy

The `tests/test_pipeline.py` suite will cover:
- **Data Integrity:** Ensure `customerID` is gone and `TotalCharges` contains no NaNs.
- **Preprocessing Leakage:** Verify that the `StandardScaler` and `OneHotEncoder` parameters are derived solely from the training set.
- **Consistency:** Ensure the transformation applied to the test set matches the training set.
- **Artifacts:** Confirm that `model.pkl`, `scaler.pkl`, `encoder.pkl`, and `feature_order.json` are written to disk.
- **Inference Contract:** Provide a sample record to the handler and verify the output format (probability: float, class: int).

## 5. Validation Strategy

- **Execution:** Run `python src/train.py`.
- **Reporting:** Output metrics (Accuracy, Recall, Precision, F1, Confusion Matrix) to `docs/validation.md`.
- *- **Success Criteria (all must pass):**
    - ROC AUC > 0.75.
    - Recall (Churn=1) > 0.70.
    - F1 (Churn=1) > 0.55.
- **Accuracy** is reported but not gated (see spec/spec.md Section 7).
- **Failure Loop:** If Recall $\le$ 0.70, adjust `class_weight` or investigate feature engineering, then retrain.

## 6. Risks & Mitigations

- **Data Loss:** Dropping NaNs in `TotalCharges` might remove too many samples. *Mitigation:* Log the number of dropped rows; verify that only `tenure=0` customers are affected.
- **Class Imbalance:** Model may bias toward the majority class. *Mitigation:* Use `class_weight='balanced'` in `LogisticRegression`.
- **Inference Drift:** Mismatch between training feature order and inference feature order. *Mitigation:* Explicitly save and load `feature_order.json`.
- **Image Size:** Large ML libraries can exceed Lambda limits. *Mitigation:* Use a slim base image and only install essential runtime packages.
- **Double-encoding risk:** src/features.py currently one-hot encodes, and the pipeline also encodes. **Mitigation:** refactor `features.py` to stop encoding; encoding lives only inside the ColumnTransformer so it fits on train only.

## 7. Definition of Done

- [ ] `src/train.py` runs end-to-end without errors.
- [ ] All tests in `tests/test_pipeline.py` pass.
- [ ] Recall for the churn class is $> 0.70$.
- [ ] `docs/validation.md` is populated with final metrics.
- [ ] All 4 model artifacts exist in `model/`.
- [ ] `Dockerfile` builds successfully.
- [ ] `src/handler.py` produces correct predictions for a sample record locally.
- [ ] `src/features.py` contains no encoding logic (encoding lives inthe pipeline's `ColumnTransformer`).
