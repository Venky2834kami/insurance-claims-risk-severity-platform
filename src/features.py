import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_features_target(df, target='loss'):
    X = df.drop(columns=[target]); y = df[target]
    return X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric', num, numeric), ('categorical', cat, categorical)])
