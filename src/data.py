from pathlib import Path
import numpy as np
import pandas as pd

def load_data(path: str, target: str = 'loss') -> pd.DataFrame:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f'Missing target column: {target}')
    return df

def make_smoke_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'age': rng.integers(18, 80, n), 'vehicle_age': rng.integers(0, 20, n), 'state': rng.choice(['IN','MH','KA','DL'], n), 'channel': rng.choice(['agent','online','branch'], n)})
    df['loss'] = np.exp(6 + .015*df.age + .03*df.vehicle_age + (df.state == 'MH')*.25 + rng.normal(0, .35, n))
    return df
