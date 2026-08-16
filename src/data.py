"""Data loading and synthetic data generation for the claims severity platform."""
from pathlib import Path
import numpy as np
import pandas as pd

from .validate import validate_dataframe


def load_data(path: str, target: str = 'loss', strict: bool = True) -> pd.DataFrame:
    """Load a CSV, validate schema, and return a clean DataFrame.

    Parameters
    ----------
    path:   Path to CSV file.
    target: Name of the target column.
    strict: If True, raise on validation errors; if False, warn and continue.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    df = pd.read_csv(p)
    if target not in df.columns:
        raise ValueError(f'Missing target column: {target}')
    errors = validate_dataframe(df, require_target=True, target=target)
    if errors:
        msg = f"{len(errors)} validation error(s):\n" + "\n".join(errors[:10])
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")
    return df


def make_smoke_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible SYNTHETIC claims dataset for smoke tests.

    This data is entirely synthetic and does not represent real insurance
    claims. Use it only for pipeline validation and CI, never for reporting
    real-world metrics.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'age': rng.integers(18, 80, n),
        'vehicle_age': rng.integers(0, 20, n),
        'state': rng.choice(['IN', 'MH', 'KA', 'DL'], n),
        'channel': rng.choice(['agent', 'online', 'branch'], n),
    })
    df['loss'] = np.exp(6 + .015 * df.age + .03 * df.vehicle_age + (df.state == 'MH') * .25 + rng.normal(0, .35, n))
    return df


def make_smoke_sample(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a small synthetic sample (no target column) for manual/CLI testing."""
    df = make_smoke_data(n=n, seed=seed)
    return df.drop(columns=['loss'])



def _cli():
    """Command-line entrypoint: generate synthetic train/smoke-sample CSVs.

    Usage:
        python -m src.data --out-dir data --n-train 500 --n-sample 20

    Writes two files under --out-dir:
      - train.csv (or --train-name): synthetic training data with target column
      - smoke_sample.csv (or --sample-name): small sample with no target column,
        suitable for src.score smoke tests.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic claims data for local runs and CI smoke tests.")
    parser.add_argument("--out-dir", default="data", help="Directory to write CSVs into (default: data)")
    parser.add_argument("--train-name", default="smoke_train.csv", help="Filename for the synthetic training CSV")
    parser.add_argument("--sample-name", default="smoke_sample.csv", help="Filename for the synthetic sample CSV (no target)")
    parser.add_argument("--n-train", type=int, default=500, help="Number of rows for the training dataset")
    parser.add_argument("--n-sample", type=int, default=20, help="Number of rows for the sample dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = make_smoke_data(n=args.n_train, seed=args.seed)
    train_path = out_dir / args.train_name
    train_df.to_csv(train_path, index=False)
    print(f"Wrote synthetic training data: {train_path} ({len(train_df)} rows)")

    sample_df = make_smoke_sample(n=args.n_sample, seed=args.seed)
    sample_path = out_dir / args.sample_name
    sample_df.to_csv(sample_path, index=False)
    print(f"Wrote synthetic smoke sample: {sample_path} ({len(sample_df)} rows)")


if __name__ == "__main__":
    _cli()
