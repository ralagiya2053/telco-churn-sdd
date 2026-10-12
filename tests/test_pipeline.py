import pytest
import pandas as pd
import sys
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Adjust sys.path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train import load_and_prepare, train_and_evaluate, build_pipeline

@pytest.fixture(scope="session")
def prepared_data():
    """Load data and column definitions once per session."""
    data_path = "data/train.csv"
    df, num_cols, bin_cols, cat_cols = load_and_prepare(data_path)
    return df, num_cols, bin_cols, cat_cols

def test_data_integrity(prepared_data):
    """Verify build_features cleans data as expected."""
    df, _, _, _ = prepared_data

    # customerID should be removed
    assert 'customerID' not in df.columns

    # TotalCharges should have no NaNs
    assert df['TotalCharges'].isna().sum() == 0

    # Churn should be binary 0/1
    assert set(df['Churn'].unique()) <= {0, 1}

def test_stratified_split(prepared_data):
    """Verify the train/test split preserves churn distribution."""
    df, _, _, _ = prepared_data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    churn_rate_train = y_train.mean()
    churn_rate_test = y_test.mean()

    assert abs(churn_rate_train - churn_rate_test) <= 0.01

def test_no_preprocessing_leakage(prepared_data):
    """Verify preprocessing parameters are derived from training data only."""
    df, num_cols, bin_cols, cat_cols = prepared_data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(num_cols, bin_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    # Extract fitted objects
    fitted_scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    fitted_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']

    # Expected values from fitting separately on X_train
    expected_scaler = StandardScaler().fit(X_train[num_cols])
    expected_encoder = OneHotEncoder(drop='first').fit(X_train[cat_cols])

    # Element-wise comparison of means
    pd.testing.assert_series_equal(
        pd.Series(fitted_scaler.mean_),
        pd.Series(expected_scaler.mean_),
        obj="Scaler Means"
    )

    # Comparison of categories
    for cat_fitted, cat_expected in zip(fitted_encoder.categories_, expected_encoder.categories_):
        pd.testing.assert_series_equal(
            pd.Series(cat_fitted),
            pd.Series(cat_expected),
            obj="Encoder Categories"
        )

def test_artifacts_written(prepared_data, tmp_path):
    """Verify artifacts are correctly saved to disk."""
    df, num_cols, bin_cols, cat_cols = prepared_data
    out_dir = str(tmp_path / "model")

    # Run training (save_artifacts is called inside train_and_evaluate)
    # We override the default out_dir by monkeypatching or calling it specifically
    # Since train_and_evaluate calls save_artifacts with default "model",
    # let's manually call them to ensure tmp_path is used.

    from train import build_pipeline, save_artifacts

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(num_cols, bin_cols, cat_cols)
    pipeline.fit(X_train, y_train)

    feature_order = save_artifacts(
        pipeline, num_cols, bin_cols, cat_cols, out_dir=out_dir
    )

    # Assert files exist
    assert (Path(out_dir) / "model.pkl").exists()
    assert (Path(out_dir) / "scaler.pkl").exists()
    assert (Path(out_dir) / "encoder.pkl").exists()
    assert (Path(out_dir) / "feature_order.json").exists()

    # Assert feature order content
    with open(Path(out_dir) / "feature_order.json", "r") as f:
        saved_order = json.load(f)

    expected_len = len(num_cols) + len(bin_cols) + \
                   len(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))

    assert len(saved_order) == expected_len
    assert saved_order == feature_order

@pytest.mark.skip(reason="handler not implemented until Task 8")
def test_inference_contract(prepared_data):
    """Stub for testing the inference handler."""
    pass
