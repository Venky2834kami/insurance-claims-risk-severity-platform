"""Batch scorer tests."""
import pytest

from src.data import make_smoke_data
from src.train import run


@pytest.fixture
def model_and_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(data=None, seed=42)
    sample = make_smoke_data(n=10, seed=1).drop(columns=['loss'])
    inp = tmp_path / 'new_claims.csv'
    sample.to_csv(inp, index=False)
    return tmp_path / 'artifacts' / 'model.joblib', inp, tmp_path / 'out.csv'


def test_score_output_has_prediction_column(model_and_data):
    from src.score import score
    model_path, inp, out = model_and_data
    result = score(str(model_path), str(inp), str(out))
    assert 'predicted_loss' in result.columns


def test_score_predictions_non_negative(model_and_data):
    from src.score import score
    model_path, inp, out = model_and_data
    result = score(str(model_path), str(inp), str(out))
    assert (result['predicted_loss'] >= 0).all()


def test_score_writes_output_file(model_and_data):
    from src.score import score
    model_path, inp, out = model_and_data
    score(str(model_path), str(inp), str(out))
    assert out.exists()


def test_score_missing_model_raises(tmp_path):
    from src.score import score
    with pytest.raises(FileNotFoundError, match='Model file not found'):
        score('nonexistent.joblib', 'dummy.csv', str(tmp_path / 'out.csv'))


def test_score_missing_input_raises(model_and_data, tmp_path):
    from src.score import score
    model_path, _, out = model_and_data
    with pytest.raises(FileNotFoundError, match='Input file not found'):
        score(str(model_path), str(tmp_path / 'nope.csv'), str(out))
