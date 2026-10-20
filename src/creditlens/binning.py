"""WoE binning.

steps for each feature:
- quantile bins (max 20)
- merge neighbours until bad rate is monotonic
- merge small bins (< min_bin_share of rows)
- missing values get a separate bin

WoE = ln(%good / %bad), IV = sum((%good - %bad) * WoE)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTH = 0.5  # avoids log(0) when a bin has no bads


@dataclass
class FeatureBins:
    name: str
    edges: list[float]
    woe: list[float]
    woe_missing: float
    table: pd.DataFrame = field(repr=False)
    iv: float = 0.0

    def bin_index(self, x):
        idx = np.searchsorted(self.edges, x.to_numpy(dtype=float), side="left")
        return np.where(x.isna().to_numpy(), -1, idx)  # -1 = missing

    def transform(self, x):
        idx = self.bin_index(x)
        woe = np.asarray(self.woe)
        return np.where(idx == -1, self.woe_missing, woe[np.clip(idx, 0, len(woe) - 1)])

    def labels(self):
        lo = [-np.inf] + list(self.edges)
        hi = list(self.edges) + [np.inf]
        return [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(lo, hi)]


def _fmt(v):
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:.4g}"


def _counts(x, y, edges):
    idx = np.searchsorted(edges, x.to_numpy(dtype=float), side="left")
    g = pd.DataFrame({"bin": idx, "y": y.to_numpy()}).groupby("bin")["y"]
    out = pd.DataFrame({"n": g.size(), "bad": g.sum()}).reindex(range(len(edges) + 1), fill_value=0)
    out["good"] = out["n"] - out["bad"]
    return out


def _merge(edges, counts, i):
    # merge bin i and i+1
    edges = edges[:i] + edges[i + 1:]
    c = counts.to_numpy()
    c = np.vstack([c[:i], c[i] + c[i + 1], c[i + 2:]])
    return edges, pd.DataFrame(c, columns=counts.columns)


def fit_feature(x, y, max_bins=20, min_bin_share=0.03):
    known = x.notna()
    xk, yk = x[known], y[known]

    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    edges = sorted(set(np.quantile(xk, qs).tolist()))
    edges = [e for e in edges if e < xk.max()]
    counts = _counts(xk, yk, edges)

    # ties in the quantiles can leave empty bins
    while len(counts) > 1 and (counts["n"] == 0).any():
        i = int(np.argmax(counts["n"].to_numpy() == 0))
        edges, counts = _merge(edges, counts, max(i - 1, 0))

    # does risk go up or down with this feature
    rate = (counts["bad"] / counts["n"]).to_numpy()
    direction = np.sign(np.corrcoef(np.arange(len(rate)), rate)[0, 1]) if len(rate) > 2 else 1.0

    # monotonic: keep merging the first pair that goes the wrong way
    while len(counts) > 1:
        rate = (counts["bad"] / counts["n"]).to_numpy()
        wrong = np.where(np.diff(rate) * direction < 0)[0]
        if len(wrong) == 0:
            break
        edges, counts = _merge(edges, counts, int(wrong[0]))

    # min size: merge smallest bin into whichever neighbour has the closer bad rate
    min_n = min_bin_share * len(x)
    while len(counts) > 1 and counts["n"].min() < min_n:
        i = int(counts["n"].to_numpy().argmin())
        if i == 0:
            j = 0
        elif i == len(counts) - 1:
            j = i - 1
        else:
            rate = (counts["bad"] / counts["n"]).to_numpy()
            j = i - 1 if abs(rate[i] - rate[i - 1]) <= abs(rate[i] - rate[i + 1]) else i
        edges, counts = _merge(edges, counts, j)

    tot_good = float((y == 0).sum())
    tot_bad = float((y == 1).sum())
    miss_bad = float(y[~known].sum())
    miss_good = float((~known).sum()) - miss_bad

    def woe_of(good, bad):
        return float(np.log(((good + SMOOTH) / tot_good) / ((bad + SMOOTH) / tot_bad)))

    woe = [woe_of(g, b) for g, b in zip(counts["good"], counts["bad"])]
    woe_missing = woe_of(miss_good, miss_bad) if (~known).any() else 0.0

    table = counts.copy()
    table.insert(0, "bin", FeatureBins(x.name, edges, woe, woe_missing, pd.DataFrame()).labels())
    if (~known).any():
        table.loc[len(table)] = ["missing", miss_good + miss_bad, miss_bad, miss_good]
    table["bad_rate"] = table["bad"] / table["n"]
    table["woe"] = woe + ([woe_missing] if (~known).any() else [])
    table["iv"] = (table["good"] / tot_good - table["bad"] / tot_bad) * table["woe"]
    table.insert(0, "feature", x.name)

    return FeatureBins(str(x.name), [float(e) for e in edges], woe, woe_missing, table, float(table["iv"].sum()))


class WoEBinner:
    def __init__(self, max_bins=20, min_bin_share=0.03):
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.bins = {}

    def fit(self, X, y):
        self.bins = {c: fit_feature(X[c], y, self.max_bins, self.min_bin_share) for c in X.columns}
        return self

    def transform(self, X, features=None):
        cols = features or list(self.bins)
        return pd.DataFrame({c: self.bins[c].transform(X[c]) for c in cols}, index=X.index)

    def iv_table(self):
        return pd.Series({c: b.iv for c, b in self.bins.items()}).sort_values(ascending=False)
