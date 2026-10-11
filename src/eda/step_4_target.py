import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_target(file_path, target_col):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    os.makedirs('src/eda/plots', exist_ok=True)

    print("--- Target Analysis ---")

    # Class Balance
    balance = df[target_col].value_counts(normalize=True) * 100
    print("\nClass Balance:")
    print(balance)

    if any(balance < 15):
        print("WARNING: Class imbalance detected (< 15%)")

    # Feature-Target Relationship
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')

    # Numeric vs Target
    # Convert target to binary for correlation
    df['target_binary'] = df[target_col].map({'Yes': 1, 'No': 0})

    correlations = {}
    for col in numeric_cols:
        corr = df[col].corr(df['target_binary'])
        correlations[col] = corr

        plt.figure()
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'{col} vs {target_col}')
        plt.savefig(f'src/eda/plots/step_4_{col}_vs_target.png')
        plt.close()

    print("\nNumeric Feature Correlations with Target:")
    for col, val in correlations.items():
        print(f"{col}: {val:.4f}")
        if abs(val) > 0.9:
            print(f"WARNING: Potential leakage detected in {col} (|corr| > 0.9)")

    # Categorical vs Target
    for col in categorical_cols:
        plt.figure()
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f'{col} vs {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'src/eda/plots/step_4_{col}_vs_target.png')
        plt.close()

if __name__ == "__main__":
    analyze_target('data/train.csv', 'Churn')
