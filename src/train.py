"""
Training module for the Telco Churn prediction model.
Handles data preparation and preprocessing pipeline construction.
"""

import json
import joblib
import pandas as pd
import sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

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

def save_artifacts(pipeline, numeric_cols, binary_cols, categorical_cols,
                   out_dir: str = "model") -> list[str]:
    """
    Extract and save the model, scaler, encoder, and feature order to the specified directory.

    Args:
        pipeline: The fitted sklearn Pipeline.
        numeric_cols: List of numeric feature names.
        binary_cols: List of binary feature names.
        categorical_cols: List of categorical feature names.
        out_dir: Directory to save artifacts.

    Returns:
        The final order of features after transformation.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Pull fitted objects via named steps
    scaler = pipeline.named_steps['preprocessor'].named_transformers_['num']
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    model = pipeline.named_steps['classifier']

    # Save pickles
    joblib.dump(model, Path(out_dir) / "model.pkl")
    joblib.dump(scaler, Path(out_dir) / "scaler.pkl")
    joblib.dump(encoder, Path(out_dir) / "encoder.pkl")

    # Build post-transform feature order
    feature_order = (
        list(numeric_cols) +
        list(binary_cols) +
        list(encoder.get_feature_names_out(categorical_cols))
    )

    # Save feature order as JSON
    with open(Path(out_dir) / "feature_order.json", "w") as f:
        json.dump(feature_order, f, indent=2)

    return feature_order

def train_and_evaluate(df: pd.DataFrame,
                       numeric_cols: list[str],
                       binary_cols: list[str],
                       categorical_cols: list[str],
                       target_col: str = 'Churn',
                       test_size: float = 0.2,
                       random_state: int = 42
                       ) -> tuple[Pipeline, dict]:
    """
    Split data, train the pipeline, and evaluate performance.

    Args:
        df: The prepared DataFrame.
        numeric_cols: List of numeric features.
        binary_cols: List of binary features.
        categorical_cols: List of categorical features.
        target_col: Name of the target variable.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Controls the shuffling applied to the data before the split.

    Returns:
        A tuple of (fitted_pipeline, results_dict).
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, binary_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=False),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'y_test': y_test,
        'y_pred': y_pred
    }

    # Save artifacts and add feature order to results
    results['feature_order'] = save_artifacts(
        pipeline, numeric_cols, binary_cols, categorical_cols
    )

    return pipeline, results

if __name__ == "__main__":
    DATA_PATH = 'data/train.csv'
    try:
        # Load and prepare data
        df, num_cols, bin_cols, cat_cols = load_and_prepare(DATA_PATH)
        print(f"Successfully loaded data from {DATA_PATH}")
        print(f"DataFrame shape: {df.shape}")

        # Train and evaluate
        pipeline, results = train_and_evaluate(df, num_cols, bin_cols, cat_cols)

        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"ROC AUC:  {results['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print("\nClassification Report:")
        print(results['classification_report'])

        print(f"\nSaved artifacts to model/ ({len(results['feature_order'])} features)")

    except Exception as e:
        print(f"Training and evaluation failed: {e}")
        import traceback
        traceback.print_exc()
