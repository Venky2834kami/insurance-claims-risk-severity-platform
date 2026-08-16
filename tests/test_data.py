"""Data loading and validation tests."""
import pandas as pd
import pytest

from src.data import load_data, make_smoke_data, make_smoke_sample
from src.validate import validate_dataframe


def test_smoke_data_shape():
    df = make_smoke_data(n=100)
    assert df.shape == (100, 5)
    assert 'loss' in df.columns


def test_smoke_data_positive_loss():
    df = make_smoke_data(n=200)
    assert (df['loss'] > 0).all()


def test_smoke_sample_has_no_target():
    df = make_smoke_sample(n=10)
    assert 'loss' not in df.columns
    assert len(df) == 10


def test_validate_dataframe_passes_on_valid_data():
    df = make_smoke_data(n=20)
    errors = validate_dataframe(df, require_target=True)
    assert errors == []


def test_validate_dataframe_flags_missing_columns():
    df = pd.DataFrame({'age': [25], 'state': ['MH']})
    errors = validate_dataframe(df)
    assert any('Missing' in e for e in errors)


def test_validate_dataframe_flags_bad_state():
    df = pd.DataFrame({'age': [30], 'vehicle_age': [5], 'state': ['ZZ'], 'channel': ['agent']})
    errors = validate_dataframe(df)
    assert any('ZZ' in e for e in errors)


def test_validate_dataframe_flags_bad_age():
    df = pd.DataFrame({'age': [10], 'vehicle_age': [5], 'state': ['MH'], 'channel': ['agent']})
    errors = validate_dataframe(df)
    assert any('age' in e for e in errors)


def test_load_data_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data('does_not_exist.csv')


def test_load_data_raises_on_missing_target(tmp_path):
    df = pd.DataFrame({'age': [30], 'vehicle_age': [2], 'state': ['MH'], 'channel': ['agent']})
    path = tmp_path / 'no_target.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(str(path))
