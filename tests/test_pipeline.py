from src.data import make_smoke_data
from src.train import run

def test_smoke_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path); df=make_smoke_data(n=80); path=tmp_path/'train.csv'; df.to_csv(path,index=False); metrics,best=run(str(path)); assert len(metrics)==3; assert best in metrics.model.tolist()
