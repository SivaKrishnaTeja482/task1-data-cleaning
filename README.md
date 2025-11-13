# Data Cleaning & Preprocessing — Task 1

Dataset: data/raw/data.csv  _(replace with your actual dataset filename)_

## What I did
- Explored dataset (nulls, dtypes, describe)
- Dropped ID-like and zero-variance columns
- Imputed missing values (numeric: median, categorical: most frequent)
- Encoded categorical variables (one-hot for low-cardinality; ordinal for high)
- Scaled numeric features (StandardScaler by default)
- Removed outliers using IQR method (optional)
- Saved cleaned dataset to data/processed/clean.csv and EDA reports to reports/

## Files
- data_cleaning.py — main script
- data/raw/data.csv — raw dataset (add your file)
- data/processed/clean.csv — cleaned output (generated)
- reports/ — nulls, dtypes, describe, and boxplots

## How to run

```bash
pip install -r requirements.txt
python data_cleaning.py --input data/raw/data.csv --output data/processed/clean.csv --num_impute median --scale standard --remove_outliers
```

Notes (what I would not do blindly)

- For categorical high-cardinality (IDs or near-IDs) don't one-hot — use embedding or drop.
- If target is skewed, prefer RobustScaler or log-transform numeric features.
- When removing outliers, check domain meaning — don't delete real rare events blindly.

---

If dataset is huge (>500k rows), sample for EDA and use streaming / chunked processing.