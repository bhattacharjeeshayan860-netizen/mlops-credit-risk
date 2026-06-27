"""Tests for leakage-safe credit-risk preprocessing."""

import pandas as pd

from src.preprocessing import CreditRiskPreprocessor


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

    assert "Unnamed: 0" not in cleaned.columns
    assert "SeriousDlqin2yrs" not in cleaned.columns
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.loc[1, "MonthlyIncomeWasMissing"] == 1
    assert cleaned.loc[1, "NumberOfDependentsWasMissing"] == 1
    assert cleaned.loc[1, "AgeWasInvalid"] == 1
    assert cleaned.loc[1, "PastDueExtremeCode"] == 1
    assert cleaned.loc[1, "NumberOfTimes90DaysLate"] < 90


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
