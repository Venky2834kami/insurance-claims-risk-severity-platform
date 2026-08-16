"""Batch scoring CLI for the Insurance Claims Severity platform.

Usage
-----
    python -m src.score --model artifacts/model.joblib --input data/new_claims.csv --output predictions.csv
"""
import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .validate import validate_dataframe


def _model_version(model_path: Path, bundle: dict) -> str:
    """Derive a stable model identifier from the artifact file and bundle metadata.

    Uses a short hash of the file contents combined with the training target
    name so that scoring output can always be traced back to a specific
    trained artifact, even without a formal model registry.
    """
    try:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:10]
    except OSError:
        digest = "unknown"
    target = bundle.get("target", "loss")
    return f"{target}-{digest}"


def score(model_path: str, input_path: str, output_path: str, strict: bool = False) -> pd.DataFrame:
    """Score a CSV of new claims and write predictions to output_path.

    Adds `predicted_loss`, `model_version`, and `scoring_timestamp` columns
    alongside the original input columns.

    Raises FileNotFoundError if the model or input file is missing, and
    ValueError if the input is missing required model features or contains
    invalid data.
    """
    mp = Path(model_path)
    ip = Path(input_path)
    if not mp.exists():
        raise FileNotFoundError(
            f"Model file not found: {mp}. Run `python -m src.train` first to create it."
        )
    if not ip.exists():
        raise FileNotFoundError(f"Input file not found: {ip}")

    bundle = joblib.load(mp)
    model = bundle["model"]
    expected_features = bundle.get("features", [])

    try:
        df = pd.read_csv(ip)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Input CSV is empty: {ip}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Input CSV could not be parsed: {ip}. {exc}") from exc

    if df.empty:
        raise ValueError(f"Input CSV has no rows: {ip}")

    errors = validate_dataframe(df, require_target=False)
    if errors:
        msg = f"{len(errors)} validation error(s):\n" + "\n".join(errors[:5])
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")

    if expected_features:
        missing = set(expected_features) - set(df.columns)
        if missing:
            raise ValueError(
                f"Input CSV is missing model features: {sorted(missing)}. "
                f"Expected: {sorted(expected_features)}"
            )
        df_input = df[expected_features]
    else:
        df_input = df

    try:
        preds = model.predict(df_input)
    except Exception as exc:  # pragma: no cover - defensive guard for bad dtypes
        raise ValueError(f"Model failed to score input data: {exc}") from exc

    version = _model_version(mp, bundle)
    timestamp = datetime.now(timezone.utc).isoformat()

    df["predicted_loss"] = pd.Series(preds, index=df.index).clip(lower=0)
    df["model_version"] = version
    df["scoring_timestamp"] = timestamp

    out = Path(output_path)
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Scored {len(df)} rows -> {output_path} (model_version={version})")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score new insurance claims")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strict", action="store_true", help="Raise on data-validation warnings")
    a = p.parse_args()
    score(a.model, a.input, a.output, a.strict)
