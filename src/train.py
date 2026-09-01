"""
Production-ready XGBoost Training Pipeline with Champion/Challenger Workflow.
"""

import os
# Set MLflow tracking URI as environment variable BEFORE importing mlflow
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from mlflow.models.signature import infer_signature

from src.preprocessing import CreditRiskPreprocessor, TARGET_COLUMN, COLUMN_RENAMES
from src.utils import (
    FEATURE_COLUMNS,
    ARTIFACTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_PATH,
    PROJECT_ROOT
)

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostProductionTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["MODEL_NAME"]
        self.target = config["TARGET"]
        self.features = config["FEATURES"]
        self.primary_metric = config["PRIMARY_METRIC"]
        self.threshold = config["DELTA_THRESHOLD"]
        
        mlflow.set_tracking_uri(config.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(config["EXPERIMENT_NAME"])
        self.client = mlflow.tracking.MlflowClient()

    def _audit_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("🔍 Running Data Audit...")
        if df[self.target].isnull().any():
            raise ValueError(f"CRITICAL: Target '{self.target}' contains NaNs.")
        
        missing = df[self.features].isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing[missing > 0]}")

        if df.duplicated().any():
            logger.info("Deduplicating data...")
            df = df.drop_duplicates()

        corrs = df[self.features].corrwith(df[self.target])
        leaky = corrs[corrs > 0.98].index.tolist()
        if leaky:
            raise ValueError(f"CRITICAL: Feature leakage in: {leaky}")

        logger.info("✅ Audit Complete.")
        return df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        X = df[self.features]
        y = df[self.target]
        X_train_raw, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val_raw, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )
        return X_train_raw, X_val_raw, X_test, y_train, y_val, y_test

    def _get_cv_stability(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> Tuple[float, float]:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        logger.info("⚖️ Validating model stability via 5-Fold CV...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Clean params to avoid multiple values for keyword arguments
        cv_params = params.copy()
        for key in ["enable_categorical", "tree_method"]:
            cv_params.pop(key, None)
            
        scores = cross_val_score(
            xgb.XGBClassifier(**cv_params, enable_categorical=True, tree_method='hist'), 
            X, y, cv=skf, scoring='roc_auc'
        )
        return np.mean(scores), np.std(scores)


    def train_and_compare(self, df: pd.DataFrame):
        # 1. Rename columns immediately to match FEATURE_COLUMNS
        logger.info("🔄 Renaming columns to match feature schema...")
        df = df.rename(columns=COLUMN_RENAMES)

        # 2. Audit
        df = self._audit_data(df)
        
        # 3. Split
        X_train_raw, X_val_raw, X_test, y_train, y_val, y_test = self._split_data(df)

        # 4. Preprocessing
        logger.info("⚙️ Fitting Preprocessor...")
        preprocessor = CreditRiskPreprocessor(clip_quantile=0.99)
        preprocessor.fit(X_train_raw)
        
        X_train = preprocessor.transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test)

        # 5. Imbalance weight
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())

        params = {
            "objective": "binary:logistic",
            "eval_metric": self.primary_metric,
            "scale_pos_weight": ratio,
            "learning_rate": 0.05,
            "max_depth": 6,
            "enable_categorical": True,
            "tree_method": "hist",
            "random_state": 42
        }

        with mlflow.start_run(run_name="XGBoost_Challenger") as run:
            logger.info(f"🚀 Training Challenger (Run: {run.info.run_id})...")
            cv_mean, cv_std = self._get_cv_stability(X_train, y_train, params)
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            probs = model.predict_proba(X_test)[:, 1]
            metrics = {
                self.primary_metric: roc_auc_score(y_test, probs) if self.primary_metric == "auc" else average_precision_score(y_test, probs),
                "logloss": log_loss(y_test, probs),
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_param("imbalance_ratio", ratio)
            mlflow.log_param("cv_std", cv_std)
            
            signature = infer_signature(X_test, probs)
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=MLFLOW_ARTIFACT_PATH,
                signature=signature,
                skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']
            )
            
            import joblib
            preprocessor_path = Path("temp_preprocessor.pkl")
            joblib.dump(preprocessor, preprocessor_path)
            mlflow.log_artifact(str(preprocessor_path), "preprocessor")
            os.remove(preprocessor_path)

            logger.info(f"✅ Challenger Evaluation: {metrics}")
            self._decide_promotion(run.info.run_id, metrics)
            return run.info.run_id, metrics

    def _decide_promotion(self, run_id: str, metrics: Dict):
        try:
            versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            if not versions:
                logger.info("🌟 No champion. Promoting challenger as the first champion.")
                self._promote(run_id)
                return

            champion_v = versions[0]
            champ_run = self.client.get_run(champion_v.run_id)
            champ_auc = champ_run.data.metrics[self.primary_metric]
            
            improvement = metrics[self.primary_metric] - champ_auc
            stability_pass = (metrics["cv_std"] / metrics[self.primary_metric]) < 0.02

            logger.info(f"🏆 Champion {champion_v.version} AUC: {champ_auc:.4f}")
            logger.info(f"📈 Challenger AUC: {metrics[self.primary_metric]:.4f} (Imp: {improvement:.4f})")

            if improvement >= self.threshold and stability_pass:
                logger.info("🚀 PROMOTION: Challenger beats champion and is stable.")
                self._promote(run_id)
            else:
                reason = "Low improvement" if improvement < self.threshold else "Unstable (high CV variance)"
                logger.info(f"❌ REJECTED: {reason}")
        except Exception as e:
            logger.error(f"Promotion logic error: {e}")

    def _promote(self, run_id: str):
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        target_v = next((v.version for v in versions if v.run_id == run_id), None)
        if target_v:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=int(target_v),
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f"✨ Model v{target_v} is now PROD CHAMPION.")
        else:
            logger.error("Could not resolve model version for promotion.")

def train_candidate(persist_local_artifacts: bool = True) -> Dict[str, str]:
    """Shim for bootstrap.py"""
    from src.preprocessing import TARGET_COLUMN
    from src.utils import (
        FEATURE_COLUMNS,
        MLFLOW_TRACKING_URI,
        MLFLOW_EXPERIMENT_NAME
    )
    config = {
        "MODEL_NAME": "credit_risk_xgb",
        "EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        "TARGET": TARGET_COLUMN,
        "FEATURES": FEATURE_COLUMNS,
        "PRIMARY_METRIC": "auc",
        "DELTA_THRESHOLD": 0.005,
        "MLFLOW_TRACKING_URI": "sqlite:///mlflow.db"
    }
    data_path = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path)
    trainer = XGBoostProductionTrainer(config)
    run_id, metrics = trainer.train_and_compare(df)
    return {"run_id": run_id, "version": "latest"}

def main():
    config = {
        "MODEL_NAME": "credit_risk_xgb",
        "EXPERIMENT_NAME": MLFLOW_EXPERIMENT_NAME,
        "TARGET": TARGET_COLUMN,
        "FEATURES": FEATURE_COLUMNS,
        "PRIMARY_METRIC": "auc",
        "DELTA_THRESHOLD": 0.005,
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI
    }
    data_path = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return
    df = pd.read_csv(data_path)
    trainer = XGBoostProductionTrainer(config)
    trainer.train_and_compare(df)

if __name__ == "__main__":
    main()
