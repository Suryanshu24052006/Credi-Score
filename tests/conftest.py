import numpy as np
import pandas as pd
import pytest

from creditlens.binning import WoEBinner
from creditlens.data import FEATURES
from creditlens.scorecard import fit_scorecard


# fake data with a known relationship so tests don't need the real dataset
def make_applicants(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "revolving_utilization": rng.beta(1.2, 2.5, n) * 1.1,
        "age": rng.integers(21, 80, n),
        "past_due_30_59": rng.poisson(0.3, n),
        "past_due_60_89": rng.poisson(0.1, n),
        "past_due_90_plus": rng.poisson(0.1, n),
        "debt_ratio": rng.gamma(2.0, 0.2, n),
        "monthly_income": rng.lognormal(8.5, 0.5, n),
        "open_credit_lines": rng.integers(0, 25, n),
        "real_estate_loans": rng.integers(0, 4, n),
        "dependents": rng.integers(0, 5, n).astype(float),
    })[FEATURES]
    X.loc[rng.random(n) < 0.15, "monthly_income"] = np.nan
    logit = (-3.2 + 2.5 * X["revolving_utilization"] - 0.02 * (X["age"] - 50)
             + 0.6 * X["past_due_30_59"] + 0.9 * X["past_due_60_89"] + 1.2 * X["past_due_90_plus"])
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="default")
    return X, y


@pytest.fixture(scope="session")
def data():
    return make_applicants()


@pytest.fixture(scope="session")
def card(data):
    X, y = data
    binner = WoEBinner().fit(X, y)
    features = ["revolving_utilization", "age", "past_due_30_59", "past_due_60_89",
                "past_due_90_plus", "monthly_income"]
    return fit_scorecard(binner, X, y, features)
