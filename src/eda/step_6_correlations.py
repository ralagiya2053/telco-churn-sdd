import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import chi2_contingency, f_oneway

def analyze_correlations(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    os.makedirs('src/eda/plots', exist_ok=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')

    print("--- Correlations Analysis ---")

    # 1. Numeric-Numeric (Pearson)
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Numeric Feature Correlation Heatmap')
        plt.savefig('src/eda/plots/step_6_corr_heatmap.png')
        plt.close()

        print("\nNumeric-Numeric Correlations:")
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.85:
                    high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], val))

        if high_corr:
            for col1, col2, val in high_corr:
                print(f"WARNING: Multicollinearity detected: {col1} & {col2} ({val:.2f})")
        else:
            print("No high multicollinearity detected (|corr| > 0.85).")

    # 2. Categorical-Categorical (Chi-Square)
    if len(categorical_cols) >= 2:
        print("\nCategorical-Categorical (Chi-Square p-values):")
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                col1, col2 = categorical_cols[i], categorical_cols[j]
                contingency = pd.crosstab(df[col1], df[col2])
                chi2, p, _, _ = chi2_contingency(contingency)
                if p < 0.05:
                    print(f"{col1} vs {col2}: p-value = {p:.4f} (Significant)")

    # 3. Categorical-Numeric (ANOVA)
    if categorical_cols and numeric_cols:
        print("\nCategorical-Numeric (ANOVA p-values):")
        for cat in categorical_cols:
            for num in numeric_cols:
                groups = [df[df[cat] == val][num].dropna() for val in df[cat].unique()]
                if all(len(g) > 0 for g in groups):
                    f_stat, p = f_oneway(*groups)
                    if p < 0.05:
                        print(f"{cat} vs {num}: p-value = {p:.4f} (Significant)")

if __name__ == "__main__":
    analyze_correlations('data/train.csv')
