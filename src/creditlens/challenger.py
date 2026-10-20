"""LightGBM to compare against the scorecard. Uses raw features (handles NaN itself)."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

PARAMS = dict(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def fit_challenger(X_train, y_train, X_valid, y_valid, seed=42):
    model = lgb.LGBMClassifier(**PARAMS, random_state=seed)
    with warnings.catch_warnings():
        # newer lightgbm warns about eval_set, still works
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*eval_set.*")
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def shap_importance(model, X):
    contrib = model.predict(X, pred_contrib=True)  # last col is the base value
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=X.columns).sort_values(ascending=False)
