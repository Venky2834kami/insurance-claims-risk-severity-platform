"""Tests for src/monitoring.py."""
import numpy as np
import pandas as pd

from src.monitoring import compare_distributions, summarize_predictions


def test_summarize_predictions_basic_stats():
    preds = [100.0, 200.0, 300.0, 400.0, 500.0]
    stats = summarize_predictions(preds)
    assert stats["count"] == 5
    assert stats["mean"] == 300.0
    assert stats["median"] == 300.0
    assert stats["min"] == 100.0
    assert stats["max"] == 500.0
    assert "p10" in stats["quantiles"]
    assert "p90" in stats["quantiles"]


def test_summarize_predictions_empty():
    stats = summarize_predictions([])
    assert stats["count"] == 0
    assert stats["mean"] is None
    assert stats["quantiles"] == {}


def test_compare_distributions_numeric_and_categorical():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({
        "age": rng.normal(40, 5, size=200),
        "state": rng.choice(["MH", "KA", "TN"], size=200),
    })
    current = pd.DataFrame({
        "age": rng.normal(45, 5, size=200),  # shifted mean
        "state": rng.choice(["MH", "KA", "TN"], size=200),
    })

    report = compare_distributions(reference, current)
    assert "age" in report["numeric"]
    assert "state" in report["categorical"]
    assert "age" in report["missingness"]

    # The shifted mean should be reflected in the numeric summary.
    ref_mean = report["numeric"]["age"]["reference"]["mean"]
    cur_mean = report["numeric"]["age"]["current"]["mean"]
    assert cur_mean > ref_mean


def test_compare_distributions_missingness_delta():
    reference = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    current = pd.DataFrame({"x": [1.0, None, None, None]})
    report = compare_distributions(reference, current)
    delta = report["missingness"]["x"]["delta"]
    assert delta > 0  # current has more missing values than reference
