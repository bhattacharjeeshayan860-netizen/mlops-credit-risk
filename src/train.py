"""Train the credit-risk model and save production inference artifacts.

This file will own the training workflow for the Give Me Some Credit dataset:
load data, preprocess features, train a classifier, evaluate ROC-AUC, save the
model, save preprocessing medians, save a reference dataset, and log results to
MLflow.
"""
from pathlib import Path
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.preprocessing import CreditRiskPreprocessor, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the raw Give Me Some Credit dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    """Split the dataset into train, validation, and test sets."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test


def preprocess_data(X_train_raw: pd.DataFrame, X_val_raw: pd.DataFrame, X_test_raw: pd.DataFrame):
    """Fit preprocessing on train data and transform all splits."""
    preprocessor = CreditRiskPreprocessor(clip_quantile=0.99)
    preprocessor.fit_transform(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_val, X_test, preprocessor


def build_model() -> Pipeline:
    """Create the baseline logistic regression model."""
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    return model


def build_cv_pipeline() -> Pipeline:
    """Create a full preprocessing + model pipeline for cross-validation."""
    model = Pipeline(steps=[
        ("preprocessor", CreditRiskPreprocessor(clip_quantile=0.99)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    return model


def run_cross_validation(X_train_raw: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run cross-validation on the training split only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_cv_pipeline(),
        X_train_raw,
        y_train,
        cv=cv,
        scoring=["roc_auc", "average_precision"],
        n_jobs=-1,
    )

    metrics = {
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "roc_auc_std": float(scores["test_roc_auc"].std()),
        "average_precision_mean": float(scores["test_average_precision"].mean()),
        "average_precision_std": float(scores["test_average_precision"].std()),
    }

    print("\nCROSS-VALIDATION METRICS")
    print(f"ROC-AUC: {metrics['roc_auc_mean']:.4f} +/- {metrics['roc_auc_std']:.4f}")
    print(f"PR-AUC:  {metrics['average_precision_mean']:.4f} +/- {metrics['average_precision_std']:.4f}")

    return metrics


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train a logistic regression baseline on the preprocessed training data."""
    model = build_model()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    """Evaluate a classifier and return JSON-serializable metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    roc_auc = float(roc_auc_score(y, y_prob))
    average_precision = float(average_precision_score(y, y_prob))

    metrics = {
        "split": split_name,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "classification_report": classification_report(y, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    print(f"\n{split_name.upper()} METRICS")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {average_precision:.4f}")
    print(classification_report(y, y_pred))

    return metrics


def save_artifacts(model, preprocessor: CreditRiskPreprocessor, X_train: pd.DataFrame, metrics: dict) -> None:
    """Save model, preprocessor, reference data, and metrics."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, ARTIFACTS_DIR / "model.pkl")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    X_train.to_csv(ARTIFACTS_DIR / "reference.csv", index=False)

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(ARTIFACTS_DIR / "preprocessing_artifact.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.get_artifact().to_dict(), f, indent=4)
    with open(ARTIFACTS_DIR / "model_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_type": (model.named_steps["classifier"].__class__.__name__),
            "version": "0.1.0",
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roc_auc": metrics["test"]["roc_auc"],
            "average_precision": metrics["test"]["average_precision"],
        },f, indent=4)
    


def main() -> None:
    """Run the full training pipeline."""
    df = load_data()

    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)
    cv_metrics = run_cross_validation(X_train_raw, y_train)

    X_train, X_val, X_test, preprocessor = preprocess_data(
        X_train_raw,
        X_val_raw,
        X_test_raw,
    )

    model = train_model(X_train, y_train)

    val_metrics = evaluate_model(model, X_val, y_val, "validation")
    test_metrics = evaluate_model(model, X_test, y_test, "test")

    metrics = {
        "cross_validation": cv_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_artifacts(model, preprocessor, X_train, metrics)


if __name__ == "__main__":
    main()
