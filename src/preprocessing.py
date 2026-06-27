

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


TARGET_COLUMN = "SeriousDlqin2yrs"
ID_COLUMN = "Unnamed: 0"

COLUMN_RENAMES = {
    "NumberOfTime30-59DaysPastDueNotWorse": "NumberOfTime30_59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse": "NumberOfTime60_89DaysPastDueNotWorse",
}

MISSING_VALUE_COLUMNS = ["MonthlyIncome", "NumberOfDependents"]
CONTINUOUS_CLIP_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
]
PAST_DUE_COLUMNS = [
    "NumberOfTime30_59DaysPastDueNotWorse",
    "NumberOfTime60_89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


@dataclass
class PreprocessingArtifact:
    """Values learned from training data and reused at inference time."""

    medians: dict[str, float] = field(default_factory=dict)
    clip_upper_bounds: dict[str, float] = field(default_factory=dict)
    past_due_caps: dict[str, float] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the artifact."""
        return asdict(self)


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """Prepare raw Give Me Some Credit features for modeling.

    The transformer learns imputation values and clipping bounds in `fit`, so it
    can be fitted on the training split only and reused for validation, tests,
    and live API requests.
    """

    def __init__(self, clip_quantile: float = 0.99) -> None:
        self.clip_quantile = clip_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CreditRiskPreprocessor":
        """Learn medians, outlier caps, and final feature order."""
        df = self._prepare_base_frame(X)

        self.artifact_ = PreprocessingArtifact()
        self.artifact_.medians = {
            column: float(df[column].median()) for column in MISSING_VALUE_COLUMNS
        }
        self.artifact_.medians["age"] = float(df.loc[df["age"] > 0, "age"].median())

        self.artifact_.clip_upper_bounds = {
            column: float(df[column].quantile(self.clip_quantile))
            for column in CONTINUOUS_CLIP_COLUMNS
        }

        self.artifact_.past_due_caps = {}
        for column in PAST_DUE_COLUMNS:
            non_sentinel = df.loc[df[column] < 90, column]
            self.artifact_.past_due_caps[column] = float(non_sentinel.max())

        transformed = self._transform_with_artifact(df)
        self.artifact_.feature_columns = transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing to new data."""
        self._check_is_fitted()
        df = self._prepare_base_frame(X)
        transformed = self._transform_with_artifact(df)
        return transformed.reindex(columns=self.artifact_.feature_columns)

    def get_artifact(self) -> PreprocessingArtifact:
        """Return the fitted preprocessing artifact."""
        self._check_is_fitted()
        return self.artifact_

    def _prepare_base_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")
        df = df.rename(columns=COLUMN_RENAMES)
        for column in [
            "age",
            *MISSING_VALUE_COLUMNS,
            *CONTINUOUS_CLIP_COLUMNS,
            *PAST_DUE_COLUMNS,
        ]:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
        return df

    def _transform_with_artifact(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["MonthlyIncomeWasMissing"] = df["MonthlyIncome"].isna().astype(int)
        df["NumberOfDependentsWasMissing"] = df["NumberOfDependents"].isna().astype(int)
        df["AgeWasInvalid"] = (df["age"] <= 0).astype(int)

        for column, median in self.artifact_.medians.items():
            if column == "age":
                df.loc[df["age"] <= 0, "age"] = median
            else:
                df[column] = df[column].fillna(median)

        df["PastDueExtremeCode"] = df[PAST_DUE_COLUMNS].ge(90).any(axis=1).astype(int)
        for column, cap in self.artifact_.past_due_caps.items():
            df[column] = df[column].clip(upper=cap)

        for column, upper_bound in self.artifact_.clip_upper_bounds.items():
            df[column] = df[column].clip(upper=upper_bound)

        return df

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "artifact_"):
            raise RuntimeError("CreditRiskPreprocessor must be fitted before transform.")
