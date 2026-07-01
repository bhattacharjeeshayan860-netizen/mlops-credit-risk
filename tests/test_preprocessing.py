"""Tests for leakage-safe credit-risk preprocessing."""

from typing import Any, cast

import pandas as pd

from src.preprocessing import CreditRiskPreprocessor


def _row_as_dict(df: pd.DataFrame, row_index: int) -> dict[str, Any]:
    """Return one DataFrame row as a plain dict for type-checker-friendly asserts."""
    return cast(dict[str, Any], df.loc[row_index].to_dict())


def _as_float(value: Any) -> float:
    """Convert a numeric test value to float."""
    return float(cast(int | float, value))


def test_preprocessor_imputes_and_adds_expected_features() -> None:
    raw = pd.DataFrame(
        {
            "Unnamed: 0": [1, 2, 3],
            "SeriousDlqin2yrs": [0, 1, 0],
            "RevolvingUtilizationOfUnsecuredLines": [0.2, 3.0, 0.5],
            "age": [45, 0, 60],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 1],
            "DebtRatio": [0.3, 10000.0, 0.5],
            "MonthlyIncome": [5000.0, None, 7000.0],
            "NumberOfOpenCreditLinesAndLoans": [5, 10, 8],
            "NumberOfTimes90DaysLate": [0, 98, 1],
            "NumberRealEstateLoansOrLines": [1, 2, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 98, 1],
            "NumberOfDependents": [0.0, None, 2.0],
        }
    )

    preprocessor = CreditRiskPreprocessor().fit(raw)
    cleaned = preprocessor.transform(raw)
    row = _row_as_dict(cleaned, 1)

    assert "Unnamed: 0" not in cleaned.columns
    assert "SeriousDlqin2yrs" not in cleaned.columns
    assert int(cleaned.isna().sum().sum()) == 0
    assert _as_float(row["MonthlyIncomeWasMissing"]) == 1
    assert _as_float(row["NumberOfDependentsWasMissing"]) == 1
    assert _as_float(row["AgeWasInvalid"]) == 1
    assert _as_float(row["PastDueExtremeCode"]) == 1
    assert _as_float(row["NumberOfTimes90DaysLate"]) < 90


def test_preprocessor_preserves_feature_order_for_new_data() -> None:
    train = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.1, 0.2],
            "age": [30, 40],
            "NumberOfTime30-59DaysPastDueNotWorse": [0, 1],
            "DebtRatio": [0.2, 0.3],
            "MonthlyIncome": [3000.0, 4000.0],
            "NumberOfOpenCreditLinesAndLoans": [2, 4],
            "NumberOfTimes90DaysLate": [0, 1],
            "NumberRealEstateLoansOrLines": [0, 1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0, 1],
            "NumberOfDependents": [0.0, 1.0],
        }
    )
    new = train.iloc[[0]].copy()

    preprocessor = CreditRiskPreprocessor().fit(train)

    assert preprocessor.transform(new).columns.tolist() == preprocessor.get_artifact().feature_columns
