"""Batch scoring CLI for the Insurance Claims Severity platform."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .validate import validate_dataframe


def score(model_path: str, input_path: str, output_path: str, strict: bool = False) -> pd.DataFrame:
    """Score a CSV of new claims and write predictions to output_path.

    Raises FileNotFoundError if the model or input file is missing, and
    ValueError if the input is missing required model features.
    """
    mp = Path(model_path)
    ip = Path(input_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model file not found: {mp}")
    if not ip.exists():
        raise FileNotFoundError(f"Input file not found: {ip}")

    bundle = joblib.load(mp)
    model = bundle['model']
    expected_features = bundle.get('features', [])

    df = pd.read_csv(ip)

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

    preds = model.predict(df_input)
    df['predicted_loss'] = preds
    df['predicted_loss'] = df['predicted_loss'].clip(lower=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True) if Path(output_path).parent != Path('.') else None
    df.to_csv(output_path, index=False)
    print(f"Scored {len(df)} rows -> {output_path}")
    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Score new insurance claims')
    p.add_argument('--model', default='artifacts/model.joblib')
    p.add_argument('--input', required=True)
    p.add_argument('--output', default='predictions.csv')
    p.add_argument('--strict', action='store_true')
    a = p.parse_args()
    score(a.model, a.input, a.output, a.strict)
