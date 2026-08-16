"""Core pipeline integration tests."""
import numpy as np
import pandas as pd
import pytest

from src.data import make_smoke_data
from src.train import _safe_rmsle, run


def test_smoke_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = make_smoke_data(n=80)
    path = tmp_path / 'train.csv'
    df.to_csv(path, index=False)
    metrics, best = run(str(path))
    assert len(metrics) == 3
    assert best in metrics.model.tolist()


def test_best_model_not_worse_than_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, best = run(data=None, seed=42)
    baseline_mae = metrics.loc[metrics.model == 'baseline_mean', 'mae'].iloc[0]
    best_mae = metrics.loc[metrics.model == best, 'mae'].iloc[0]
    assert best_mae <= baseline_mae + 1e-6


def test_metrics_are_finite_and_nonnegative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, _ = run(data=None, seed=0)
    for col in ['mae', 'rmse', 'rmsle']:
        assert metrics[col].notna().all()
        assert (metrics[col] >= 0).all()


def test_artifacts_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(data=None, seed=42)
    assert (tmp_path / 'artifacts' / 'model.joblib').exists()
    assert (tmp_path / 'artifacts' / 'metrics.csv').exists()


def test_safe_rmsle_handles_zeros_without_crash():
    y_true = np.array([0.0, 100.0, 500.0])
    y_pred = np.array([0.0, 110.0, 490.0])
    result = _safe_rmsle(y_true, y_pred)
    assert np.isfinite(result)
    assert result >= 0


def test_safe_rmsle_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0])
    assert _safe_rmsle(y, y) == pytest.approx(0.0, abs=1e-9)


def test_model_predictions_are_non_negative(tmp_path, monkeypatch):
    import joblib
    monkeypatch.chdir(tmp_path)
    run(data=None, seed=42)
    bundle = joblib.load(tmp_path / 'artifacts' / 'model.joblib')
    df = make_smoke_data(n=20, seed=99).drop(columns=['loss'])
    preds = bundle['model'].predict(df)
    assert (preds >= 0).all()


def test_run_returns_correct_types(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, best = run(data=None)
    assert isinstance(metrics, pd.DataFrame)
    assert isinstance(best, str)
