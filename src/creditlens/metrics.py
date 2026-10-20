from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y_true, y_score):
    # max gap between cumulative bad and good distributions
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def summary(y_true, pd_pred):
    auc = roc_auc_score(y_true, pd_pred)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, pd_pred), 4),
        "brier": round(float(brier_score_loss(y_true, pd_pred)), 4),
        "observed_default_rate": round(float(np.mean(y_true)), 4),
        "mean_predicted_pd": round(float(np.mean(pd_pred)), 4),
    }


def psi(expected, actual, bins=10):
    # bins from deciles of the baseline. < 0.1 ok, 0.1-0.25 keep an eye on it, > 0.25 big shift
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected[~np.isnan(expected)], np.linspace(0, 1, bins + 1)[1:-1]))

    def shares(a):
        idx = np.where(np.isnan(a), len(edges) + 1, np.searchsorted(edges, a, side="left"))
        counts = np.bincount(idx, minlength=len(edges) + 2).astype(float)
        return np.clip(counts / counts.sum(), 1e-6, None)

    e, a = shares(expected), shares(actual)
    return float(np.sum((a - e) * np.log(a / e)))


def gains_table(y_true, score, n_bands=10):
    # band 1 = lowest scores = riskiest
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(score)})
    df["band"] = pd.qcut(df["score"].rank(method="first"), n_bands, labels=False) + 1
    t = df.groupby("band").agg(min_score=("score", "min"), max_score=("score", "max"),
                               n=("y", "size"), bads=("y", "sum"))
    t["bad_rate"] = t["bads"] / t["n"]
    t["cum_bads_captured"] = t["bads"].cumsum() / t["bads"].sum()
    return t.reset_index()


def cutoff_tradeoff(y_true, score, cutoffs):
    # approve everyone with score >= cutoff
    y = np.asarray(y_true)
    s = np.asarray(score)
    rows = []
    for c in cutoffs:
        approved = s >= c
        rows.append({
            "cutoff": c,
            "approval_rate": round(float(approved.mean()), 4),
            "bad_rate_in_approved": round(float(y[approved].mean()) if approved.any() else float("nan"), 4),
            "bads_rejected_share": round(float(y[~approved].sum() / y.sum()), 4),
        })
    return pd.DataFrame(rows)
