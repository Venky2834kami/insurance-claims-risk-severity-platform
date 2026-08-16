import argparse, json
from pathlib import Path
import joblib, mlflow, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from .data import load_data, make_smoke_data
from .features import split_features_target, build_preprocessor
from .models import make_models

def run(data, target='loss', experiment='claims-severity', seed=42):
    df = make_smoke_data(seed=seed) if data is None else load_data(data, target)
    X, y = split_features_target(df, target); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=seed)
    mlflow.set_experiment(experiment); results=[]; best=None
    for name, model in make_models(build_preprocessor(X), seed).items():
        with mlflow.start_run(run_name=name):
            model.fit(Xtr, ytr); pred=np.maximum(model.predict(Xte), 0)
            metrics={'mae':mean_absolute_error(yte,pred),'rmse':mean_squared_error(yte,pred)**.5,'rmsle':mean_squared_log_error(yte,pred)**.5}
            mlflow.log_params({'model':name,'target_transform':'log1p','n_rows':len(df),'n_features':X.shape[1]}); mlflow.log_metrics(metrics)
            results.append({'model':name,**metrics})
            if best is None or metrics['mae'] < best[0]: best=(metrics['mae'], model, name)
    Path('artifacts').mkdir(exist_ok=True); joblib.dump({'model':best[1],'target':target,'features':list(X.columns)}, 'artifacts/model.joblib'); pd.DataFrame(results).to_csv('artifacts/metrics.csv',index=False)
    return pd.DataFrame(results), best[2]

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--target',default='loss'); p.add_argument('--experiment',default='claims-severity'); a=p.parse_args(); print(run(a.data,a.target,a.experiment))
