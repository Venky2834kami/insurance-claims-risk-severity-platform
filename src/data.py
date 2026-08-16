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
