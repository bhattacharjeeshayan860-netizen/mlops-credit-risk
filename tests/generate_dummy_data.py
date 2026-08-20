import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor import FILE_PATH


def generate_dummy_data():
    rng = np.random.default_rng(42)
    n_rows = 1000
    reference_path = PROJECT_ROOT / "artifacts" / "reference.csv"
    output_path = FILE_PATH

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference file not found at {reference_path}")

    # Load reference data
    ref_df = pd.read_csv(reference_path)

    # 1. Sample 1,000 unique rows from the reference dataset to preserve correlations and distributions
    # We use a loop to ensure we get exactly n_rows unique rows
    sampled_indices = []
    all_indices = list(ref_df.index)
    
    # Shuffle indices to pick random ones
    rng.shuffle(all_indices)
    
    # Take first 1000 (they are unique because indices are unique)
    sampled_indices = all_indices[:n_rows]
    df = ref_df.loc[sampled_indices].copy()

    # 2. Handle Missingness (as per requirements: MonthlyIncome ~20%, Dependents ~2.6%)
    # The reference file uses flags but the values might be filled.
    # We want the synthetic data to have actual NaNs.
    
    # MonthlyIncome missingness
    # Target is ~19.7%
    target_income_missing_rate = 0.1974
    # We'll use the existing flags from the reference to decide where to put NaNs, 
    # or just sample based on the rate. Sampling based on rate is more direct.
    income_missing_mask = rng.random(n_rows) < target_income_missing_rate
    df.loc[income_missing_mask, 'MonthlyIncome'] = np.nan
    
    # NumberOfDependents missingness
    # Target is ~2.59%
    target_dependents_missing_rate = 0.0259
    dependents_missing_mask = rng.random(n_rows) < target_dependents_missing_rate
    df.loc[dependents_missing_mask, 'NumberOfDependents'] = np.nan

    # 3. Recalculate/Ensure Monitoring Columns are correct
    # MonthlyIncomeWasMissing = 1 when MonthlyIncome is missing, otherwise 0
    df['MonthlyIncomeWasMissing'] = df['MonthlyIncome'].isna().astype(int)
    
    # NumberOfDependentsWasMissing = 1 when NumberOfDependents is missing, otherwise 0
    df['NumberOfDependentsWasMissing'] = df['NumberOfDependents'].isna().astype(int)
    
    # AgeWasInvalid = 1 only when age <= 0, otherwise 0
    df['AgeWasInvalid'] = (df['age'] <= 0).astype(int)
    
    # PastDueExtremeCode = 1 when any of the past-due count features is >= 90, otherwise 0
    past_due_cols = [
        "NumberOfTime30_59DaysPastDueNotWorse", 
        "NumberOfTimes90DaysLate", 
        "NumberOfTime60_89DaysPastDueNotWorse"
    ]
    # Ensure these are treated as numeric for the comparison
    for col in past_due_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['PastDueExtremeCode'] = (df[past_due_cols] >= 90).any(axis=1).astype(int)

    # 4. Ensure column order
    cols_order = [
        "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30_59DaysPastDueNotWorse",
        "DebtRatio", "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines", "NumberOfTime60_89DaysPastDueNotWorse", "NumberOfDependents",
        "MonthlyIncomeWasMissing", "NumberOfDependentsWasMissing", "AgeWasInvalid", "PastDueExtremeCode"
    ]
    df = df[cols_order]

    # 5. Final Validation
    assert df.shape == (1000, 14), f"Shape mismatch: {df.shape}"
    assert df.duplicated().sum() == 0, "Duplicate rows found"

    # Print required statistics
    print("--- Synthetic Data Stats ---")
    print(f"Shape: {df.shape}")
    print(f"Duplicate count: {df.duplicated().sum()}")
    
    print("\n--- Feature Statistics ---")
    stats_cols = cols_order
    # Grouping for cleaner output
    for col in stats_cols:
        print(f"\nColumn: {col}")
        print(f"  Mean:    {df[col].mean():.4f}")
        print(f"  Median:  {df[col].median():.4f}")
        print(f"  Min:     {df[col].min():.4f}")
        print(f"  Max:     {df[col].max():.4f}")
        # Check missingness for all
        missing_rate = df[col].isna().mean()
        print(f"  Missing: {missing_rate:.4f}")

    print("\n--- Comparison with Reference ---")
    ref_stats = ref_df[cols_order].describe().loc[['mean', 'min', 'max']]
    # Note: ref_df might have different columns or types, but we use cols_order
    # We'll just print a few key ones to show it's working.
    print("Reference Means for key features:")
    for col in ["age", "DebtRatio", "MonthlyIncome", "NumberOfDependents"]:
        if col in ref_df.columns:
            print(f"  {col}: Ref={ref_df[col].mean():.4f}, Synthetic={df[col].mean():.4f}")

    # 6. Save to the monitoring module's configured file path.
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")

if __name__ == "__main__":
    generate_dummy_data()
