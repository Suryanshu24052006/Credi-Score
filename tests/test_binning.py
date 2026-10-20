import numpy as np
import pandas as pd

from creditlens.binning import WoEBinner, fit_feature


def test_monotonic(data):
    X, y = data
    fb = fit_feature(X["revolving_utilization"], y)
    rates = fb.table.loc[fb.table["bin"] != "missing", "bad_rate"].to_numpy()
    assert np.all(np.diff(rates) >= 0) or np.all(np.diff(rates) <= 0)


def test_woe_sign(data):
    # high utilisation -> riskier -> negative woe
    X, y = data
    fb = fit_feature(X["revolving_utilization"], y)
    t = fb.table[fb.table["bin"] != "missing"]
    assert t["woe"].iloc[0] > 0 > t["woe"].iloc[-1]


def test_min_bin_size(data):
    X, y = data
    fb = fit_feature(X["age"], y, min_bin_share=0.05)
    numeric = fb.table[fb.table["bin"] != "missing"]
    assert (numeric["n"] >= 0.05 * len(X)).all()


def test_missing_bin(data):
    X, y = data
    fb = fit_feature(X["monthly_income"], y)
    assert "missing" in fb.table["bin"].values
    woe = fb.transform(pd.Series([np.nan, 5000.0]))
    assert woe[0] == fb.woe_missing


def test_iv(data):
    X, y = data
    iv = WoEBinner().fit(X, y).iv_table()
    assert iv["revolving_utilization"] > 0.1
    assert iv["real_estate_loans"] < 0.02  # random noise in the fake data


def test_counts_add_up(data):
    X, y = data
    fb = fit_feature(X["revolving_utilization"], y)
    assert fb.table["n"].sum() == len(X)
    assert fb.table["bad"].sum() == y.sum()
