"""Generate realistic dummy monitoring data for the credit-risk API.

This script creates a synthetic production-style request log with 1,000 unique
rows and saves it to the project's monitoring output path defined in
``src.monitor``.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitor import FILE_PATH


def generate_dummy_data() -> None:
    """Create a 1,000-row dummy production log and save it to the monitoring CSV."""
    rng = np.random.default_rng(42)
    n_rows = 1000

    # Generate realistic base feature values.
    revolving_util = rng.uniform(0, 1, n_rows)

    # Keep ages realistic and inject a few invalid values to match the
    # production preprocessing rule: age <= 0 is treated as invalid.
    age = rng.integers(21, 85, n_rows).astype(float)
    invalid_age_indices = rng.choice(n_rows, size=5, replace=False)
    age[invalid_age_indices] = rng.uniform(-10, 0, size=5)

    num_30_59 = rng.integers(0, 5, n_rows)
    extreme_past_due_indices = rng.choice(n_rows, size=5, replace=False)
    num_30_59[extreme_past_due_indices] = 95

    debt_ratio = rng.uniform(0, 1, n_rows)

    monthly_income = rng.uniform(2000, 15000, n_rows)
    income_missing_indices = rng.choice(n_rows, size=10, replace=False)
    monthly_income[income_missing_indices] = np.nan

    open_lines = rng.integers(1, 30, n_rows)

    num_90_late = rng.integers(0, 5, n_rows)
    num_90_late[rng.choice(n_rows, size=2, replace=False)] = 100

    real_estate = rng.integers(0, 5, n_rows)
    num_60_89 = rng.integers(0, 5, n_rows)

    dependents = rng.integers(0, 6, n_rows).astype(float)
    dependents_missing_indices = rng.choice(n_rows, size=5, replace=False)
    dependents[dependents_missing_indices] = np.nan

    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
            "age": age,
            "NumberOfTime30_59DaysPastDueNotWorse": num_30_59,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": monthly_income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": num_90_late,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60_89DaysPastDueNotWorse": num_60_89,
            "NumberOfDependents": dependents,
        }
    )

    # Match the project's preprocessing conventions for monitoring flags.
    df["MonthlyIncomeWasMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependentsWasMissing"] = df["NumberOfDependents"].isna().astype(int)
    df["AgeWasInvalid"] = (df["age"] <= 0).astype(int)

    past_due_cols = [
        "NumberOfTime30_59DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfTime60_89DaysPastDueNotWorse",
    ]
    df["PastDueExtremeCode"] = (df[past_due_cols] >= 90).any(axis=1).astype(int)

    ordered_columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "age",
        "NumberOfTime30_59DaysPastDueNotWorse",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberOfTimes90DaysLate",
        "NumberRealEstateLoansOrLines",
        "NumberOfTime60_89DaysPastDueNotWorse",
        "NumberOfDependents",
        "MonthlyIncomeWasMissing",
        "NumberOfDependentsWasMissing",
        "AgeWasInvalid",
        "PastDueExtremeCode",
    ]
    df = df[ordered_columns]

    assert df.shape == (1000, 14), f"Shape mismatch: {df.shape}"
    assert df.duplicated().sum() == 0, "Duplicate rows found"

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    output_path = FILE_PATH
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    generate_dummy_data()
