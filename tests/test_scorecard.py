import numpy as np

from creditlens.metrics import summary


def test_points_sum_to_score(card, data):
    X, _ = data
    head = X.head(50)
    assert np.allclose(card.feature_points(head).sum(axis=1), card.score(head))


def test_score_matches_pd(card, data):
    X, _ = data
    head = X.head(200)
    assert np.allclose(card.score_to_pd(card.score(head)), card.predict_pd(head), atol=1e-9)


def test_pdo(card):
    pd_a, pd_b = card.score_to_pd([600.0, 600.0 + card.pdo])
    odds = lambda p: (1 - p) / p
    assert abs(odds(pd_b) / odds(pd_a) - 2) < 1e-9  # +20 points = double odds
    assert abs(odds(pd_a) - card.base_odds) < 1e-9


def test_risky_scores_lower(card, data):
    X, _ = data
    safe = X.head(1).copy()
    safe[["revolving_utilization", "past_due_30_59", "past_due_60_89", "past_due_90_plus"]] = [0.05, 0, 0, 0]
    risky = safe.copy()
    risky[["revolving_utilization", "past_due_90_plus"]] = [0.98, 3]
    assert card.score(risky)[0] < card.score(safe)[0]


def test_reason_codes(card, data):
    X, _ = data
    risky = X.head(1).copy()
    risky[["revolving_utilization", "past_due_30_59", "past_due_60_89", "past_due_90_plus"]] = [0.99, 0, 0, 4]
    reasons = card.reason_codes(risky)[0]
    assert len(reasons) <= 3
    assert {r["feature"] for r in reasons} & {"revolving_utilization", "past_due_90_plus"}
    assert all(r["points_lost"] >= 1 for r in reasons)


def test_auc_and_calibration(card, data):
    X, y = data
    s = summary(y, card.predict_pd(X))
    assert s["auc"] > 0.7
    assert abs(s["mean_predicted_pd"] - s["observed_default_rate"]) < 0.01
