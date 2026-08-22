"""Monitoring and drift-detection utilities.

This module will store incoming inference requests in a rolling buffer and use
Evidently AI to compare live request data against the saved training reference
dataset.
"""
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

def log_prediction_input(prediction_input: pd.DataFrame) -> None:
    data_to_log= prediction_input
    if FILE_PATH.exists():
        
        existing_data= load_prediction_log()
        data_to_log= pd.concat([existing_data,data_to_log],ignore_index=True)
        total_rows=data_to_log.shape[0]
        if total_rows>1000:
            data_to_log= data_to_log.tail(1000)

        data_to_log.to_csv(FILE_PATH,index=False,header=True)
    else:
        data_to_log.to_csv(FILE_PATH,index=False,header=True)

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
        return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist
    
def run_drift_check() -> dict:
    """Run drift detection between reference data and recent API requests."""
    current_time = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
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
    reference_data_CSV= load_reference_data()
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
    elif recent_data_csv.shape[0]<1000:
        return {
            "drift_detected": False,
            "drifted_columns": [],
            "drifted_column_count": 0,
            "actual_drift_share": 0.0,
            "drift_share_threshold": 0.0,
            "reference_rows": len(reference_data_CSV),
            "current_rows": len(recent_data_csv),
            "message": "Not enough recent data for drift detection. At least 1000 rows are required."
        }

    monitoring_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    data_definition = DataDefinition(
        numerical_columns= NUMERIC_COLUMNS,
        categorical_columns= CATEGORICAL_COLUMNS
    )
    ref_data = Dataset.from_pandas(reference_data_CSV[monitoring_columns],
                                            data_definition=data_definition)
    recent_data = Dataset.from_pandas(recent_data_csv[monitoring_columns],
                                      data_definition= data_definition)
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_data,
                          current_data=recent_data,
                          )
    data = snapshot.dict()
    
    drifted_columns = []
    drift_detected=False
    drift_share_threshold=data['metrics'][0]['config']['drift_share']
    
    for metric in data['metrics']:
        if metric['config']['type']=='evidently:metric_v2:DriftedColumnsCount':
            continue
        if metric['value']>=metric['config']['threshold']:
            drifted_columns.append(metric['config']['column'])

    drifted_column_count=len(drifted_columns)
    actual_drift_share= len(drifted_columns)/len(monitoring_columns)
    if actual_drift_share>=drift_share_threshold:
        drift_detected=True
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
