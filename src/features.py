import pandas as pd
import numpy as np

def build_features(source: str, **kwargs) -> pd.DataFrame:
    """
    Load raw data and return a feature-ready DataFrame with one row
    per prediction unit.

    Returns
    -------
    df : pd.DataFrame
        Feature columns + target column. Ready to split and train.
    """
    # Load data
    df = pd.read_csv(source)

    # 1. Immediate Drop: customerID (unique identifier, no predictive value)
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # 2. Data Cleaning: TotalCharges
    # Convert TotalCharges to numeric, coercing errors (empty strings) to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows where TotalCharges is NaN (11 rows, tenure=0 customers)
    # Per constraint: Do NOT impute — "not yet billed" is not the same as "$0 billed."
    df = df.dropna(subset=['TotalCharges'])

    # 3. Target Mapping: Churn {Yes, No} -> {1, 0}
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df
