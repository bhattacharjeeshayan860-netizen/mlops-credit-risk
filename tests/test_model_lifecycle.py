from types import SimpleNamespace
from unittest.mock import Mock

import api.main as api_main
import src.predict as predict


def test_get_champion_metrics_returns_alias_run_metrics(monkeypatch):
    model_version = SimpleNamespace(version="1", run_id="champion-run")
    run = SimpleNamespace(data=SimpleNamespace(metrics={"val_roc_auc": 0.84}))
    client = Mock()
    client.get_model_version_by_alias.return_value = model_version
    client.get_run.return_value = run
    mlflow = Mock()

    monkeypatch.setattr(predict, "_import_mlflow", lambda: mlflow)
    monkeypatch.setattr(predict, "MlflowClient", lambda: client)

    result = predict.get_champion_metrics()

    assert result == {
        "run_id": "champion-run",
        "version": "1",
        "val_roc_auc": 0.84,
    }
    client.get_model_version_by_alias.assert_called_once_with(
        name="credit_risk_model",
        alias="champion",
    )
    client.get_run.assert_called_once_with("champion-run")


def test_get_candidate_metrics_returns_run_metrics(monkeypatch):
    run = SimpleNamespace(data=SimpleNamespace(metrics={"val_roc_auc": 0.87}))
    client = Mock()
    client.get_run.return_value = run
    mlflow = Mock()

    monkeypatch.setattr(predict, "_import_mlflow", lambda: mlflow)
    monkeypatch.setattr(predict, "MlflowClient", lambda: client)

    result = predict.get_candidate_metrics("candidate-run")

    assert result == {"run_id": "candidate-run", "val_roc_auc": 0.87}
    client.get_run.assert_called_once_with("candidate-run")


def test_better_candidate_moves_champion_alias(monkeypatch):
    monkeypatch.setattr(
        predict,
        "get_champion_metrics",
        lambda: {"version": "1", "val_roc_auc": 0.84},
    )
    monkeypatch.setattr(
        predict,
        "get_candidate_metrics",
        lambda run_id: {"run_id": run_id, "val_roc_auc": 0.87},
    )
    set_alias = Mock()
    monkeypatch.setattr(predict, "set_registered_model_alias", set_alias)

    promoted = predict.promote_if_better("2", "candidate-run")

    assert promoted is True
    set_alias.assert_called_once_with("2")


def test_worse_candidate_does_not_move_champion_alias(monkeypatch):
    monkeypatch.setattr(
        predict,
        "get_champion_metrics",
        lambda: {"version": "1", "val_roc_auc": 0.87},
    )
    monkeypatch.setattr(
        predict,
        "get_candidate_metrics",
        lambda run_id: {"run_id": run_id, "val_roc_auc": 0.84},
    )
    set_alias = Mock()
    monkeypatch.setattr(predict, "set_registered_model_alias", set_alias)

    promoted = predict.promote_if_better("2", "candidate-run")

    assert promoted is False
    set_alias.assert_not_called()


def test_reload_success_replaces_cached_resources(monkeypatch):
    old_model = object()
    old_preprocessor = object()
    old_info = {"version": "1"}
    new_model = object()
    new_preprocessor = object()
    new_info = {"version": "2"}
    predict._model = old_model
    predict._preprocessor = old_preprocessor
    predict._model_info = old_info
    monkeypatch.setattr(
        predict,
        "load_champion_model_preprocessor",
        lambda: (new_model, new_preprocessor),
    )
    monkeypatch.setattr(predict, "load_model_info", lambda: new_info)

    result = predict.reload_resources()

    assert result == (new_model, new_preprocessor, new_info)
    assert predict._model is new_model
    assert predict._preprocessor is new_preprocessor
    assert predict._model_info == new_info


def test_reload_failure_keeps_cached_resources(monkeypatch):
    old_model = object()
    old_preprocessor = object()
    old_info = {"version": "1"}
    predict._model = old_model
    predict._preprocessor = old_preprocessor
    predict._model_info = old_info

    def fail_loading():
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(predict, "load_champion_model_preprocessor", fail_loading)

    result = predict.reload_resources()

    assert result == (old_model, old_preprocessor, old_info)
    assert predict._model is old_model
    assert predict._preprocessor is old_preprocessor
    assert predict._model_info == old_info


def test_monitor_retrains_once_for_one_drift_event(monkeypatch):
    api_main._drift_event_active = False
    drift_result = {"drift_detected": True}
    train = Mock(return_value={"version": "2", "run_id": "candidate-run"})
    promote = Mock(return_value=False)

    monkeypatch.setattr(api_main, "run_drift_check", lambda: drift_result)
    monkeypatch.setattr(api_main, "train_candidate", train)
    monkeypatch.setattr(api_main, "promote_if_better", promote)

    api_main.monitor_drift()
    api_main.monitor_drift()

    train.assert_called_once_with(persist_local_artifacts=False)
    promote.assert_called_once_with("2", "candidate-run")
