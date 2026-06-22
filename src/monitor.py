"""Monitoring and drift-detection utilities.

This module will store incoming inference requests in a rolling buffer and use
Evidently AI to compare live request data against the saved training reference
dataset.
"""


def run_drift_check() -> dict:
    """Run drift detection between reference data and recent API requests."""
    raise NotImplementedError("Drift detection will be implemented in the monitoring phase.")
