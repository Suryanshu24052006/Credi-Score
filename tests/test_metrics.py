import numpy as np

from creditlens.metrics import cutoff_tradeoff, gains_table, ks_statistic, psi, summary


def test_ks_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == 1.0


def test_gini():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    p = np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1)
    s = summary(y, p)
    assert abs(s["gini"] - (2 * s["auc"] - 1)) < 1e-3


def test_psi():
    rng = np.random.default_rng(2)
    base = rng.normal(600, 30, 20_000)
    assert psi(base, base) < 1e-6
    assert psi(base, rng.normal(600, 30, 20_000)) < 0.01
    assert psi(base, rng.normal(560, 30, 20_000)) > 0.25


def test_gains_table():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    t = gains_table(y, rng.random(500))
    assert t["n"].sum() == 500 and abs(t["cum_bads_captured"].iloc[-1] - 1) < 1e-9


def test_cutoffs():
    rng = np.random.default_rng(4)
    t = cutoff_tradeoff(rng.integers(0, 2, 1000), rng.normal(600, 30, 1000), [550, 600, 650])
    assert t["approval_rate"].is_monotonic_decreasing
