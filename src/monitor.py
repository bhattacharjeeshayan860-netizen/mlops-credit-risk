"""Monitoring and drift-detection utilities.

This module will store incoming inference requests in a rolling buffer and use
Evidently AI to compare live request data against the saved training reference
dataset.
"""
from pathlib import Path
import pandas as pd
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FILE_PATH=ARTIFACTS_DIR / "prediction_log.csv"

def log_prediction_input(prediction_input: dict):
    data_to_log= pd.DataFrame([prediction_input])
    if FILE_PATH.exists():
        
        existing_data= load_prediction_log()
        data_to_log= pd.concat([existing_data,data_to_log],ignore_index=True)
        total_rows=data_to_log.shape[0]
        if total_rows>1000:
            data_to_log= data_to_log.tail(1000)

        data_to_log.to_csv(FILE_PATH,index=False,header=True)
    else:
        data_to_log.to_csv(FILE_PATH,index=False,header=True)

        
def load_prediction_log() -> pd.DataFrame:
    """Load the prediction log from CSV."""
    if FILE_PATH.exists():
        return pd.read_csv(FILE_PATH)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if the file doesn't exist
def run_drift_check() -> dict:
    """Run drift detection between reference data and recent API requests."""
    raise NotImplementedError("Drift detection will be implemented in the monitoring phase.")
