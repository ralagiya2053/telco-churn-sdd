import pandas as pd

def analyze_structure(file_path):
    df = pd.read_csv(file_path)

    print("--- Structure Analysis ---")
    print(f"Source: {file_path}")
    print(f"Shape: {df.shape}")
    print("\nColumn Names and Dtypes:")
    print(df.dtypes)
    print("\nMemory Footprint:")
    print(f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nHead (5 rows):")
    print(df.head())

    print("\nTail (5 rows):")
    print(df.tail())

    # Identify unique identifier columns (columns where all values are unique)
    unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
    print("\nUnique Identifier Columns:")
    print(unique_cols if unique_cols else "None found")

if __name__ == "__main__":
    analyze_structure('data/train.csv')
