import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_univariate(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Create plots directory if it doesn't exist
    os.makedirs('src/eda/plots', exist_ok=True)

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove ID from analysis
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')

    print("--- Univariate Analysis ---")

    # Numeric columns
    for col in numeric_cols:
        stats = df[col].describe()
        q1 = stats['25%']
        q3 = stats['75%']
        iqr = q3 - q1
        std = stats['std']
        mean = stats['mean']

        # Outlier detection
        outliers_iqr = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
        outliers_std = df[(df[col] < (mean - 3 * std)) | (df[col] > (mean + 3 * std))]

        print(f"\nColumn: {col}")
        print(stats)
        print(f"Outliers (1.5xIQR): {len(outliers_iqr)}")
        print(f"Outliers (3-sigma): {len(outliers_std)}")

        # Histogram
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'src/eda/plots/step_3_{col}_hist.png')
        plt.close()

        # Boxplot
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(f'src/eda/plots/step_3_{col}_box.png')
        plt.close()

    # Categorical columns
    for col in categorical_cols:
        counts = df[col].value_counts()
        cardinality = len(counts)
        rare_cats = counts[counts < 5].index.tolist()

        print(f"\nColumn: {col}")
        print(f"Cardinality: {cardinality}")
        print(f"Rare categories (<5): {len(rare_cats)}")

        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'src/eda/plots/step_3_{col}_count.png')
        plt.close()

if __name__ == "__main__":
    analyze_univariate('data/train.csv')
