"""
Training module for the Telco Churn prediction model.
Handles data preparation and preprocessing pipeline construction.
"""

import pandas as pd
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Make the import path-agnostic
sys.path.insert(0, str(Path(__file__).parent))
from features import build_features

def load_and_prepare(source: str) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Load raw data and separate columns into numeric, binary, and categorical groups.

    Args:
        source: Path to the raw CSV data.

    Returns:
        A tuple containing:
        - df: The cleaned DataFrame from build_features.
        - numeric_cols: List of numeric features to be scaled.
        - binary_cols: List of binary features to pass through.
        - categorical_cols: List of categorical features to be encoded.
    """
    df = build_features(source)

    numeric_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    binary_cols = [
        'SeniorCitizen'
    ]
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    return df, numeric_cols, binary_cols, categorical_cols

def build_preprocessor(numeric_cols: list[str], binary_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Construct a ColumnTransformer for scaling numeric features,
    passing through binary features, and one-hot encoding categorical features.

    Args:
        numeric_cols: List of numeric columns to scale.
        binary_cols: List of binary columns to pass through.
        categorical_cols: List of categorical columns to encode.

    Returns:
        A configured ColumnTransformer instance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('bin', 'passthrough', binary_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor

def build_pipeline(numeric_cols: list[str], binary_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """
    Chain the preprocessor and the classifier into a single sklearn Pipeline.

    Args:
        numeric_cols: List of numeric columns to scale.
        binary_cols: List of binary columns to pass through.
        categorical_cols: List of categorical columns to encode.

    Returns:
        A configured Pipeline instance.
    """
    preprocessor = build_preprocessor(numeric_cols, binary_cols, categorical_cols)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    return pipeline

if __name__ == "__main__":
    # Verification block
    DATA_PATH = 'data/train.csv'
    try:
        df, num_cols, bin_cols, cat_cols = load_and_prepare(DATA_PATH)
        print(f"Successfully loaded data from {DATA_PATH}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Numeric columns ({len(num_cols)}): {num_cols}")
        print(f"Binary columns ({len(bin_cols)}): {bin_cols}")
        print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

        pipeline = build_pipeline(num_cols, bin_cols, cat_cols)
        print("\nPipeline constructed:")
        print(pipeline)

    except Exception as e:
        print(f"Verification failed: {e}")
