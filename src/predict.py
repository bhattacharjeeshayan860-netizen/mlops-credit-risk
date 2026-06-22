"""Prediction utilities shared by the FastAPI service.

This module will load the trained model and preprocessing artifacts from
`artifacts/`, apply inference-time preprocessing, impute nullable fields using
training medians, and return prediction probabilities.
"""


def predict_credit_risk(features: dict) -> dict:
    """Return credit-risk prediction output for one request payload."""
    raise NotImplementedError("Prediction logic will be implemented with the API.")
