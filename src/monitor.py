from pathlib import Path
import pandas as pd
from evidently import Report,Dataset,DataDefinition
from evidently.presets import DataDriftPreset

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FILE_PATH=ARTIFACTS_DIR / "prediction_log.csv"

def log_prediction(prediction_input: pd.DataFrame, prediction_result: dict) -> None:
    """Logs both input features and prediction results."""
    # Combine input and result into one DataFrame
    result_df = pd.DataFrame([prediction_result])
    log_entry = pd.concat([prediction_input.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    
    if FILE_PATH.exists():
        existing_data = load_prediction_log()
        log_entry = pd.concat([existing_data, log_entry], ignore_index=True)
        total_rows = log_entry.shape[0]
        if total_rows > 5000:  # Increased buffer for more meaningful stats
            log_entry = log_entry.tail(5000)

        log_entry.to_csv(FILE_PATH, index=False, header=True)
    else:
        log_entry.to_csv(FILE_PATH, index=False, header=True)

def load_reference_data() -> pd.DataFrame:
    """Load the reference data from csv."""
    reference_data_path = ARTIFACTS_DIR / "reference.csv"
    if reference_data_path.exists():
        return pd.read_csv(reference_data_path)
    else:
        return pd.DataFrame()

def load_prediction_log() -> pd.DataFrame:
    """Load the prediction log from CSV."""
    if FILE_PATH.exists():
        return pd.read_csv(FILE_PATH)
    else:
        return pd.DataFrame()

def reset_prediction_log() -> None:
    if FILE_PATH.exists():
        FILE_PATH.unlink()

def get_monitoring_stats() -> dict:
    """Returns summary statistics from the prediction log."""
    log_df = load_prediction_log()
    if log_df.empty:
        return {
            "total_predictions": 0,
            "high_risk_rate": 0.0,
            "avg_default_probability": 0.0,
            "recent_volume": 0
        }
    
    total = len(log_df)
    # Note: 'risk_label' and 'default_probability' are now logged
    high_risk_count = len(log_df[log_df['risk_label'] == 'high_risk'])
    avg_prob = log_df['default_probability'].mean()
    
    # Recent volume (last 100 requests)
    recent_volume = len(log_df.tail(100))

    return {
        "total_predictions": total,
        "high_risk_rate": float(high_risk_count / total),
        "avg_default_probability": float(avg_prob),
        "recent_volume": recent_volume
    }

def run_drift_check() -> dict:
    """Run drift detection between reference data and recent API requests."""
    current_time = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
    
    # Only use input columns for drift detection to avoid including results in the drift check
    # This is a common practice to detect feature drift
    NUMERIC_COLUMNS = [
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
    ]
    CATEGORICAL_COLUMNS = [
        "MonthlyIncomeWasMissing",
        "NumberOfDependentsWasMissing",
        "AgeWasInvalid",
        "PastDueExtremeCode",
    ]
    
    reference_data_CSV = load_reference_data()
    if reference_data_CSV.empty:
        return {
            "drift_detected": False,
            "drifted_columns": [],
            "drifted_column_count": 0,
            "actual_drift_share": 0.0,
            "drift_share_threshold": 0.0,
            "reference_rows": 0,
            "current_rows": 0,
            "message": "Reference data is empty. Drift detection cannot be performed."
        }
        
    recent_data_csv = load_prediction_log()
    if recent_data_csv.empty:
        return {
            "drift_detected": False,
            "drifted_columns": [],
            "drifted_column_count": 0,
            "actual_drift_share": 0.0,
            "drift_share_threshold": 0.0,
            "reference_rows": len(reference_data_CSV),
            "current_rows": 0,
            "message": "Recent prediction log is empty. Drift detection cannot be performed."
        }
    elif len(recent_data_csv) < 100: # Reduced threshold for faster testing/demo purposes
        return {
            "drift_detected": False,
            "drifted_columns": [],
            "drifted_column_count": 0,
            "actual_drift_share": 0.0,
            "drift_share_threshold": 0.0,
            "reference_rows": len(reference_data_CSV),
            "current_rows": len(recent_data_csv),
            "message": "Not enough recent data for drift detection. At least 100 rows are required."
        }

    monitoring_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    # Ensure all required columns exist in both datasets
    available_ref = [col for col in monitoring_columns if col in reference_data_CSV.columns]
    available_recent = [col for col in monitoring_columns if col in recent_data_csv.columns]
    common_cols = list(set(available_ref) & set(available_recent))

    if not common_cols:
        return {
            "drift_detected": False,
            "drifted_columns": [],
            "drifted_column_count": 0,
            "actual_drift_share": 0.0,
            "drift_share_threshold": 0.0,
            "reference_rows": len(reference_data_CSV),
            "current_rows": len(recent_data_csv),
            "message": "No common columns found for drift detection."
        }

    data_definition = DataDefinition(
        numerical_columns=[c for c in common_cols if reference_data_CSV[c].dtype != 'object'],
        categorical_columns=[c for c in common_cols if reference_data_CSV[c].dtype == 'object']
    )
    
    ref_data = Dataset.from_pandas(reference_data_CSV[common_cols], data_definition=data_definition)
    recent_data = Dataset.from_pandas(recent_data_csv[common_cols], data_definition=data_definition)
    
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_data, current_data=recent_data)
    data = snapshot.dict()
    
    drifted_columns = []
    drift_detected = False
    drift_share_threshold = data['metrics'][0]['config']['drift_share']
    
    for metric in data['metrics']:
        if metric['config']['type'] == 'evidently:metric_v2:DriftedColumnsCount':
            continue
        if metric['value'] >= metric['config']['threshold']:
            drifted_columns.append(metric['config']['column'])

    drifted_column_count = len(drifted_columns)
    actual_drift_share = len(drifted_columns) / len(common_cols)
    
    if actual_drift_share >= drift_share_threshold:
        drift_detected = True
        
    snapshot.save_html(f"{PROJECT_ROOT}/reports/drift_report_{current_time}_{drift_detected}.html")
    
    return {
        "drift_detected": drift_detected,
        "drifted_columns": drifted_columns,
        "drifted_column_count": drifted_column_count,
        "actual_drift_share": actual_drift_share,
        "drift_share_threshold": drift_share_threshold,
        "reference_rows": len(reference_data_CSV),
        "current_rows": len(recent_data_csv),
        "message": "Drift detection completed."
    }
