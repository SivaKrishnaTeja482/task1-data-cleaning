import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
import argparse
import os

# Helper functions
def remove_zero_variance(df):
    return df.loc[:, (df != df.iloc[0]).any()]

def remove_outliers_iqr(df, numeric_columns):
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    filter_cond = ~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df[filter_cond]

def describe_and_save(df, report_dir, name):
    os.makedirs(report_dir, exist_ok=True)
    df.isnull().sum().to_csv(f"{report_dir}/{name}_nulls.csv")
    df.dtypes.to_csv(f"{report_dir}/{name}_dtypes.csv")
    df.describe(include='all').transpose().to_csv(f"{report_dir}/{name}_describe.csv")

def save_boxplots(df, numeric_columns, report_dir):
    import matplotlib.pyplot as plt
    os.makedirs(report_dir, exist_ok=True)
    for col in numeric_columns:
        plt.figure()
        df.boxplot(column=col)
        plt.title(col)
        plt.savefig(f"{report_dir}/boxplot_{col}.png")
        plt.close()

def main(args):
    df = pd.read_csv(args.input)
    report_dir = "reports"
    describe_and_save(df, report_dir, "raw")

    # Remove zero-variance columns
    df = remove_zero_variance(df)

    # Drop ID-like columns if present (heuristic: name includes 'id')
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    df = df.drop(columns=id_cols)

    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Impute numeric columns
    if len(numeric_columns) > 0:
        num_imputer = SimpleImputer(strategy=args.num_impute)
        df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

    # Impute categorical columns
    if len(categorical_columns) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    # Encode categorical variables
    for col in categorical_columns:
        n_unique = df[col].nunique()
        if n_unique < 15:
            enc = OneHotEncoder(sparse_output=False, drop='first')
            dummies = enc.fit_transform(df[[col]])
            cat_cols = [f"{col}_{cat}" for cat in enc.categories_[0][1:]]
            dummies_df = pd.DataFrame(dummies, columns=cat_cols, index=df.index)
            df = pd.concat([df.drop(columns=[col]), dummies_df], axis=1)
        else:
            enc = OrdinalEncoder()
            df[col] = enc.fit_transform(df[[col]])

    # Scale numeric columns
    if len(numeric_columns) > 0:
        if args.scale == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Remove outliers (optional)
    if args.remove_outliers:
        df = remove_outliers_iqr(df, numeric_columns)

    describe_and_save(df, report_dir, "clean")
    save_boxplots(df, numeric_columns, report_dir)

    # Save cleaned data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to raw dataset (csv)')
    parser.add_argument('--output', type=str, required=True, help='Path to save cleaned dataset (csv)')
    parser.add_argument('--num_impute', type=str, choices=['mean','median','most_frequent'], default='median')
    parser.add_argument('--scale', type=str, choices=['standard','robust'], default='standard')
    parser.add_argument('--remove_outliers', action='store_true', help='Remove outliers using IQR')
    args = parser.parse_args()
    main(args)
