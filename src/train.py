"""Train the credit-risk model and save production inference artifacts.

This file will own the training workflow for the Give Me Some Credit dataset:
load data, preprocess features, train a classifier, evaluate ROC-AUC, save the
model, save preprocessing medians, save a reference dataset, and log results to
MLflow.
"""
import sys
import tempfile
from pathlib import Path
import json
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, RandomizedSearchCV

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from src.preprocessing import CreditRiskPreprocessor, TARGET_COLUMN
from src.utils import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_ARTIFACT_PATH
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report, confusion_matrix

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Load the raw Give Me Some Credit dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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


def build_model(classifier=None) -> Pipeline:
    """Create a model pipeline."""
    if classifier is None:
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return model


def build_cv_pipeline(classifier=None) -> Pipeline:
    """Create a full preprocessing + model pipeline for cross-validation."""
    if classifier is None:
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model = Pipeline(steps=[
        ("preprocessor", CreditRiskPreprocessor(clip_quantile=0.99)),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return model


def run_cross_validation(X_train_raw: pd.DataFrame, y_train: pd.Series, classifier=None) -> dict:
    """Run cross-validation on the training split only."""
    if classifier is None:
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_cv_pipeline(classifier),
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


def train_model(X_train: pd.DataFrame, y_train: pd.Series, classifier=None) -> Pipeline:
    """Train a classifier on the preprocessed training data."""
    model = build_model(classifier)
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


def save_artifacts(model, preprocessor: CreditRiskPreprocessor, X_train: pd.DataFrame,run_id: str, metrics: dict, output_dir: Path = ARTIFACTS_DIR) -> None:
    """Save model, preprocessor, reference data, and metrics."""
    output_dir.mkdir(exist_ok=True)

    joblib.dump(model, output_dir / "model.pkl")
    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    X_train.to_csv(output_dir / "reference.csv", index=False)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    with open(output_dir / "preprocessing_artifact.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.get_artifact().to_dict(), f, indent=4)
    with open(output_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model.named_steps["classifier"].__class__.__name__,
            "version": "0.1.0",
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mlflow_run_id": run_id,
            "roc_auc": metrics["test"]["roc_auc"],
            "average_precision": metrics["test"]["average_precision"],
        },f, indent=4)

    mlflow.log_artifact(str(output_dir / "preprocessor.pkl"), artifact_path="preprocessor")
    mlflow.log_artifact(str(output_dir / "reference.csv"), artifact_path="reference")
    mlflow.log_artifact(str(output_dir / "preprocessing_artifact.json"), artifact_path="preprocessor")
    mlflow.log_artifact(str(output_dir / "model_info.json"), artifact_path="model_info")
    


def tune_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Perform randomized search to find optimal XGBoost hyperparameters."""
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    xgb_clf = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss')
    search = RandomizedSearchCV(
        xgb_clf, 
        param_distributions=param_dist, 
        n_iter=5, 
        scoring='roc_auc', 
        cv=3, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_candidate(persist_local_artifacts: bool = True) -> dict[str, str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as active_run:
        df = load_data()
        mlflow.set_tag("dataset", "Give Me Some Credit")
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_data(df)
        
        # 1. Cross-validation (Baseline)
        cv_metrics = run_cross_validation(X_train_raw, y_train, classifier=LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})

        X_train, X_val, X_test, preprocessor = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
        
        # 2. Train Baseline (LR)
        lr_classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        lr_model = train_model(X_train, y_train, classifier=lr_classifier)
        
        # 3. Train Tuned Candidate (XGBoost)
        print("Tuning XGBoost candidate...")
        tuned_xgb_clf = tune_xgb(X_train, y_train)
        xgb_model = train_model(X_train, y_train, classifier=tuned_xgb_clf)

        # 4. Evaluate both on validation set
        lr_val_metrics = evaluate_model(lr_model, X_val, y_val, split_name="val_lr")
        xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, split_name="val_xgb")

        # 5. Model Selection
        if xgb_val_metrics["roc_auc"] > lr_val_metrics["roc_auc"]:
            model = xgb_model
            val_metrics = xgb_val_metrics
            selected_type = "XGBoost"
            print(f"XGBoost won: {xgb_val_metrics['roc_auc']:.4f} > {lr_val_metrics['roc_auc']:.4f}")
        else:
            model = lr_model
            val_metrics = lr_val_metrics
            selected_type = "LogisticRegression"
            print(f"LogisticRegression won: {lr_val_metrics['roc_auc']:.4f} >= {xgb_val_metrics['roc_auc']:.4f}")

        # 6. Evaluate winner on test set
        test_metrics = evaluate_model(model, X_test, y_test, split_name="test")

        # 7. Logging
        start = time.perf_counter()
        # Re-fit the winner to get an accurate training time for logging
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        mlflow.log_param("model_type", selected_type)
        mlflow.log_params(model.named_steps["classifier"].get_params())
        mlflow.log_param("train_time", train_time)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MLFLOW_ARTIFACT_PATH,
            registered_model_name=MLFLOW_EXPERIMENT_NAME,
            skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']
        )
        
        mlflow.log_metrics({
            "val_roc_auc": val_metrics["roc_auc"],
            "val_average_precision": val_metrics["average_precision"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_average_precision": test_metrics["average_precision"],
        })
        
        run_id = active_run.info.run_id
        if persist_local_artifacts:
            save_artifacts(model, preprocessor, X_train, run_id,
                           {"cross_validation": cv_metrics,
                            "validation": val_metrics,
                            "test": test_metrics
                           })
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                save_artifacts(model, preprocessor, X_train, run_id,
                                {"cross_validation": cv_metrics,
                                 "validation": val_metrics,
                                 "test": test_metrics
                                }, output_dir=Path(temp_dir))

    client = MlflowClient()
    versions = client.search_model_versions(
        f"name='{MLFLOW_EXPERIMENT_NAME}'"
    )
    candidate_versions = [version for version in versions if version.run_id == run_id]
    if not candidate_versions:
        raise RuntimeError("The trained model was not registered in MLflow.")

    candidate_version = max(candidate_versions, key=lambda version: int(version.version))
    return {
        "run_id": run_id,
        "version": str(candidate_version.version),
    }


def main() -> None:
    """Run the full training pipeline."""
    train_candidate()
if __name__ == "__main__":
    main()

