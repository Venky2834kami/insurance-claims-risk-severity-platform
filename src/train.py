"""Training entry point for the Insurance Claims Severity platform.

Usage
-----
    python -m src.train                       # synthetic smoke data
    python -m src.train --data data/train.csv # real data (if available)
"""
import argparse
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .data import load_data, make_smoke_data
from .features import build_preprocessor, split_features_target
from .models import make_models


def _safe_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE that never crashes on zero/negative predictions."""
    y_pred_safe = np.maximum(np.asarray(y_pred, dtype=float), 1e-6)
    y_true_safe = np.maximum(np.asarray(y_true, dtype=float), 1e-6)
    return float(np.sqrt(np.mean((np.log1p(y_true_safe) - np.log1p(y_pred_safe)) ** 2)))


def run(data=None, target='loss', experiment='claims-severity', seed=42):
    """Train all candidate models, log runs to MLflow, and save the best model.

    Returns (metrics_dataframe, best_model_name).
    """
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    np.random.seed(seed)

    is_smoke = data is None
    df = make_smoke_data(seed=seed) if is_smoke else load_data(data, target)
    data_source = 'SYNTHETIC_SMOKE_DATA' if is_smoke else str(data)

    X, y = split_features_target(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)

    mlflow.set_experiment(experiment)
    results = []
    best = None

    for name, model in make_models(build_preprocessor(X), seed).items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, ytr)
            pred = np.maximum(model.predict(Xte), 0.0)

            metrics = {
                'mae': float(mean_absolute_error(yte, pred)),
                'rmse': float(mean_squared_error(yte, pred) ** .5),
                'rmsle': _safe_rmsle(np.asarray(yte), pred),
            }

            mlflow.log_params({
                'model': name,
                'target_transform': 'log1p',
                'n_rows': len(df),
                'n_features': X.shape[1],
                'data_source': data_source,
                'seed': seed,
            })
            mlflow.log_metrics(metrics)
            try:
                mlflow.sklearn.log_model(model, artifact_path='model')
            except Exception as exc:  # pragma: no cover - logging should not break training
                print(f"[WARN] mlflow.sklearn.log_model failed: {exc}")

            results.append({'model': name, **metrics})
            if best is None or metrics['mae'] < best[0]:
                best = (metrics['mae'], model, name)

    Path('artifacts').mkdir(exist_ok=True)
    joblib.dump(
        {'model': best[1], 'target': target, 'features': list(X.columns)},
        'artifacts/model.joblib',
    )
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv('artifacts/metrics.csv', index=False)
    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best[2]}  MAE={best[0]:.4f}")
    return metrics_df, best[2]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train claims severity models')
    p.add_argument('--data', help='Path to training CSV (omit for synthetic smoke data)')
    p.add_argument('--target', default='loss')
    p.add_argument('--experiment', default='claims-severity')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    run(a.data, a.target, a.experiment, a.seed)
