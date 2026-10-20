"""Logistic regression on WoE values, converted to points.

score = offset + factor * ln(odds of good)
factor = PDO / ln(2), offset = base_score - factor * ln(base_odds)
defaults: 600 points = 50:1 odds, every 20 points doubles the odds

model gives ln(odds of bad) = b0 + sum(b_i * woe_i), so per feature:
points_i = -factor * b_i * woe_i + (offset - factor * b0) / n_features
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REASON_TEXT = {
    "revolving_utilization": "High utilisation of revolving credit limits",
    "age": "Limited credit history / younger applicant profile",
    "past_due_30_59": "Recent 30-59 day late payments",
    "past_due_60_89": "Recent 60-89 day late payments",
    "past_due_90_plus": "Past 90+ day delinquency",
    "debt_ratio": "High debt relative to income",
    "monthly_income": "Low or unverified monthly income",
    "open_credit_lines": "Number of open credit lines indicates higher risk",
    "real_estate_loans": "Number of real-estate loans indicates higher risk",
    "dependents": "Number of dependents indicates higher risk",
}


def select_features(binner, woe_train, min_iv=0.02, max_corr=0.7):
    iv = binner.iv_table()
    log = {"iv": iv.round(4).to_dict(), "dropped_low_iv": [], "dropped_correlated": []}
    kept = []
    for f in iv.index:  # sorted by IV so the stronger one of a correlated pair stays
        if iv[f] < min_iv:
            log["dropped_low_iv"].append(f)
            continue
        clash = [k for k in kept if abs(woe_train[f].corr(woe_train[k])) > max_corr]
        if clash:
            log["dropped_correlated"].append({"feature": f, "correlated_with": clash[0]})
            continue
        kept.append(f)
    return kept, log


@dataclass
class Scorecard:
    binner: object
    features: list
    model: LogisticRegression
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0

    @property
    def factor(self):
        return self.pdo / np.log(2)

    @property
    def offset(self):
        return self.base_score - self.factor * np.log(self.base_odds)

    def predict_pd(self, X):
        return self.model.predict_proba(self.binner.transform(X, self.features))[:, 1]

    def feature_points(self, X):
        woe = self.binner.transform(X, self.features)
        b0 = float(self.model.intercept_[0])
        share = (self.offset - self.factor * b0) / len(self.features)
        coefs = pd.Series(self.model.coef_[0], index=self.features)
        return -self.factor * woe * coefs + share

    def score(self, X):
        return self.feature_points(X).sum(axis=1).to_numpy()

    def score_to_pd(self, score):
        odds_good = np.exp((np.asarray(score, dtype=float) - self.offset) / self.factor)
        return 1.0 / (1.0 + odds_good)

    def points_table(self):
        b0 = float(self.model.intercept_[0])
        share = (self.offset - self.factor * b0) / len(self.features)
        rows = []
        for f, coef in zip(self.features, self.model.coef_[0]):
            for _, r in self.binner.bins[f].table.iterrows():
                exact = -self.factor * coef * r["woe"] + share
                rows.append({"feature": f, "bin": r["bin"], "n": int(r["n"]),
                             "bad_rate": round(float(r["bad_rate"]), 4), "woe": round(float(r["woe"]), 4),
                             "points": int(round(exact)), "points_exact": float(exact)})
        return pd.DataFrame(rows)

    def max_points(self):
        return self.points_table().groupby("feature")["points_exact"].max().to_dict()

    def reason_codes(self, X, top_n=3):
        # features where the applicant lost the most points vs the best bin
        pts = self.feature_points(X)
        best = self.max_points()
        out = []
        for _, row in pts.iterrows():
            lost = sorted(((best[f] - row[f], f) for f in self.features), reverse=True)
            out.append([{"feature": f, "points_lost": int(round(l)), "reason": REASON_TEXT.get(f, f)}
                        for l, f in lost[:top_n] if l >= 1])
        return out


def fit_scorecard(binner, X_train, y_train, features, C=1.0):
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(binner.transform(X_train, features), y_train)
    return Scorecard(binner=binner, features=features, model=model)
