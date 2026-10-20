# python -m creditlens.train
# trains scorecard + lightgbm, saves models/ and reports/
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from . import policy
from .binning import WoEBinner
from .challenger import fit_challenger, shap_importance
from .data import FEATURES, PROJECT_ROOT, TARGET, load_splits
from .metrics import cutoff_tradeoff, gains_table, psi, summary
from .scorecard import fit_scorecard, select_features

MODELS = PROJECT_ROOT / "models"
REPORTS = PROJECT_ROOT / "reports"
FIGS = REPORTS / "figures"


def main(seed=42):
    for d in (MODELS, REPORTS, FIGS):
        d.mkdir(parents=True, exist_ok=True)

    train, valid, test = load_splits(seed)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xva, yva = valid[FEATURES], valid[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    print(f"rows: train {len(train):,} | valid {len(valid):,} | test {len(test):,} | default rate {ytr.mean():.2%}")

    # scorecard
    binner = WoEBinner().fit(Xtr, ytr)
    features, selection = select_features(binner, binner.transform(Xtr))
    card = fit_scorecard(binner, Xtr, ytr, features)
    print("scorecard features:", features)

    # lightgbm
    gbm = fit_challenger(Xtr, ytr, Xva, yva, seed)

    # test set results
    pd_card = card.predict_pd(Xte)
    pd_gbm = gbm.predict_proba(Xte)[:, 1]
    score_tr, score_te = card.score(Xtr), card.score(Xte)
    metrics = {
        "data": {"train": len(train), "valid": len(valid), "test": len(test),
                 "default_rate_train": round(float(ytr.mean()), 4)},
        "scorecard_test": summary(yte, pd_card),
        "scorecard_valid": summary(yva, card.predict_pd(Xva)),
        "challenger_test": summary(yte, pd_gbm),
        "challenger_best_iteration": int(gbm.best_iteration_),
        "feature_selection": selection,
        "scorecard_features": features,
        "scaling": {"pdo": card.pdo, "base_score": card.base_score, "base_odds": card.base_odds},
    }

    decisions = pd.Series([policy.decide(p) for p in pd_card])
    metrics["policy_test"] = {
        "approve_max_pd": policy.APPROVE_MAX_PD, "decline_min_pd": policy.DECLINE_MIN_PD,
        **{f"{k}_share": round(float(v), 4) for k, v in decisions.value_counts(normalize=True).items()},
        **{f"default_rate_{k}": round(float(yte[decisions == k].mean()), 4) for k in decisions.unique()},
    }

    # drift check. fake "next month" sample where people with utilisation > 70% are 3x as likely to apply
    weights = np.where(test["revolving_utilization"] > 0.7, 3.0, 1.0)
    shifted = test.sample(n=10_000, replace=True, weights=weights, random_state=seed)
    monitoring = {
        "score_psi_train_vs_test": round(psi(score_tr, score_te), 4),
        "score_psi_train_vs_simulated_shift": round(psi(score_tr, card.score(shifted[FEATURES])), 4),
        "default_rate_simulated_shift": round(float(shifted[TARGET].mean()), 4),
        "feature_psi_train_vs_test": {f: round(psi(Xtr[f], Xte[f]), 4) for f in FEATURES},
        "feature_psi_train_vs_simulated_shift": {f: round(psi(Xtr[f], shifted[f]), 4) for f in FEATURES},
    }

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS / "monitoring.json").write_text(json.dumps(monitoring, indent=2))
    pd.concat([b.table for b in binner.bins.values()]).round(4).to_csv(REPORTS / "binning_all_features.csv", index=False)
    binner.iv_table().round(4).rename("iv").to_csv(REPORTS / "iv.csv")
    card.points_table().drop(columns="points_exact").to_csv(REPORTS / "scorecard_points.csv", index=False)
    gains_table(yte, score_te).round(4).to_csv(REPORTS / "gains_test.csv", index=False)
    cutoffs = np.arange(int(np.percentile(score_te, 5)) // 10 * 10, int(np.percentile(score_te, 60)) + 10, 10)
    cutoff_tradeoff(yte, score_te, cutoffs).to_csv(REPORTS / "cutoffs_test.csv", index=False)
    shap_importance(gbm, Xte).round(4).rename("mean_abs_shap").to_csv(REPORTS / "challenger_shap_importance.csv")

    make_plots(yte, pd_card, pd_gbm, score_te)

    joblib.dump(card, MODELS / "scorecard.joblib")
    joblib.dump(gbm, MODELS / "challenger.joblib")

    print(json.dumps({k: metrics[k] for k in ("scorecard_test", "challenger_test", "policy_test")}, indent=2))
    print("score psi train vs test:", monitoring["score_psi_train_vs_test"])
    print("score psi train vs shifted sample:", monitoring["score_psi_train_vs_simulated_shift"])
    return metrics


def make_plots(y, pd_card, pd_gbm, score):
    models = (("WoE scorecard", pd_card), ("LightGBM", pd_gbm))

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in models:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "roc.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.asarray(y)
    ax.hist(score[y == 0], bins=50, alpha=0.6, density=True, label="repaid")
    ax.hist(score[y == 1], bins=50, alpha=0.6, density=True, label="defaulted")
    ax.set(xlabel="Credit score", ylabel="Density", title="Score distribution - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "score_distribution.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, p in models:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=0.8)
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title="Calibration - test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "calibration.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
