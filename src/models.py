from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

def make_models(preprocessor, seed=42):
    def wrapped(est):
        return TransformedTargetRegressor(regressor=Pipeline([('prep', preprocessor), ('model', est)]), func=__import__('numpy').log1p, inverse_func=__import__('numpy').expm1)
    return {
      'baseline_mean': wrapped(DummyRegressor(strategy='mean')),
      'hist_gradient_boosting': wrapped(HistGradientBoostingRegressor(max_iter=250, learning_rate=.06, l2_regularization=1.0, random_state=seed)),
      'random_forest': wrapped(RandomForestRegressor(n_estimators=250, min_samples_leaf=3, n_jobs=-1, random_state=seed))}
