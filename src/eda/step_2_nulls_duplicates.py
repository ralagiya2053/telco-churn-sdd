import pandas as pd
import numpy as np

def analyze_nulls_duplicates(file_path):
    # Load data and handle TotalCharges type conversion
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    print("--- Nulls and Duplicates Analysis ---")

    # Nulls per column
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({'Count': null_counts, 'Percentage': null_pct})
    null_df = null_df[null_df['Count'] > 0]

    print("\nNulls per Column:")
    if not null_df.empty:
        print(null_df)
    else:
        print("No nulls found.")

    # Duplicates
    full_duplicates = df.duplicated().sum()
    id_duplicates = df.duplicated(subset=['customerID']).sum()

    print("\nDuplicates:")
    print(f"Fully duplicated rows: {full_duplicates}")
    print(f"Duplicates on customerID: {id_duplicates}")

if __name__ == "__main__":
    analyze_nulls_duplicates('data/train.csv')
