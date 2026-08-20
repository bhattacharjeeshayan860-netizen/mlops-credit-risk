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
def run_drift_check() -> None:
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
    ref_data = load_reference_data()
    recent_data = load_prediction_log()

    monitoring_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    data_definition = DataDefinition(
        numerical_columns= NUMERIC_COLUMNS,
        categorical_columns= CATEGORICAL_COLUMNS
    )
    ref_data = Dataset.from_pandas(ref_data[monitoring_columns],
                                            data_definition=data_definition)
    recent_data = Dataset.from_pandas(recent_data[monitoring_columns],
                                      data_definition= data_definition)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_data,
               current_data= recent_data,
               )
    data=report.as_dict()
    report.save_html(f"{PROJECT_ROOT}/reports/drift_report{current_time}.html")

