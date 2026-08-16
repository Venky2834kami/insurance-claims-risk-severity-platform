"""Lightweight prediction and data-drift monitoring utilities.

This module provides simple, dependency-free monitoring suitable for a
portfolio project and local experimentation. It is NOT a replacement for a
production monitoring stack. A real deployment should use a dedicated system
such as Evidently, a metrics backend (e.g. Prometheus/Grafana), alerting on
thresholds, and label-based performance monitoring once ground-truth claim
outcomes become available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_predictions(predictions) -> dict:
    """Compute basic descriptive statistics for a batch of predictions.

    Returns count, mean, median, min, max, and the 10/25/75/90 percentiles.
    """
    arr = np.asarray(predictions, dtype=float)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }
    quantile_levels = [0.10, 0.25, 0.75, 0.90]
    quantiles = {f"p{int(q * 100)}": float(np.quantile(arr, q)) for q in quantile_levels}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "quantiles": quantiles,
    }


def _numeric_summary(series: pd.Series) -> dict:
    clean = series.dropna().astype(float)
    if clean.empty:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _category_frequencies(series: pd.Series) -> dict:
    counts = series.astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def compare_distributions(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare a reference dataset against current data for simple drift signals.

    For numeric columns: compares mean/std/min/max between reference and current.
    For categorical columns: compares category frequency distributions.
    Also reports missingness rate changes for every shared column.

    This is a heuristic, descriptive comparison only -- it does not compute
    statistical significance (e.g. PSI, KS-test, chi-squared). Those would be
    appropriate additions in a production monitoring system.
    """
    shared_columns = [c for c in reference.columns if c in current.columns]
    report: dict = {"numeric": {}, "categorical": {}, "missingness": {}}

    for col in shared_columns:
        ref_col = reference[col]
        cur_col = current[col]

        ref_missing = float(ref_col.isna().mean())
        cur_missing = float(cur_col.isna().mean())
        report["missingness"][col] = {
            "reference": ref_missing,
            "current": cur_missing,
            "delta": cur_missing - ref_missing,
        }

        if pd.api.types.is_numeric_dtype(ref_col):
            report["numeric"][col] = {
                "reference": _numeric_summary(ref_col),
                "current": _numeric_summary(cur_col),
            }
        else:
            report["categorical"][col] = {
                "reference": _category_frequencies(ref_col),
                "current": _category_frequencies(cur_col),
            }

    return report
