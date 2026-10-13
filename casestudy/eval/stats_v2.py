"""casestudy.eval.stats_v2 - statistics of the KBS revision (S6; Reviewer 1, comment 7).

Seeds are blocks: within a seed every method saw the same pool, hold-out and initial
classifier weights. For one dataset and one metric:

* ``friedman``: Friedman chi-square over all methods, with Kendall's W.
* ``compare``: a reference method against every other one; two-sided Wilcoxon signed-rank
  test (exact where SciPy can), Holm-adjusted over the comparisons; matched-pairs
  rank-biserial correlation; Hodges-Lehmann estimate of the paired difference with its
  exact distribution-free confidence interval (order statistics of the Walsh averages).

The same functions are previewed in Part 2 of ``400_kbs_revision_plan.ipynb``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps


def holm(p) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in the input order."""
    p = np.asarray(p, float)
    adj, running = np.empty_like(p), 0.0
    for k, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def signed_rank_cdf(n: int) -> np.ndarray:
    """Exact null distribution of the signed-rank statistic T+ for n pairs: P(T+ <= t)."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1.0
    for k in range(1, n + 1):
        counts = counts + np.concatenate([np.zeros(k), counts[:-k]])
    return np.cumsum(counts) / 2.0 ** n


def hodges_lehmann(d, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Hodges-Lehmann estimate of a paired difference and its exact (1 - alpha) interval."""
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    walsh = np.sort((d[i] + d[j]) / 2.0)
    c = int(np.searchsorted(signed_rank_cdf(len(d)), alpha / 2, side="right")) - 1
    lo, hi = (walsh[c], walsh[len(walsh) - 1 - c]) if c >= 0 else (walsh[0], walsh[-1])
    return float(np.median(walsh)), float(lo), float(hi)


def rank_biserial(d) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-); zero differences dropped."""
    d = np.asarray(d, float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    r = sps.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def wilcoxon(d) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank statistic and p-value; (0, 1) if all differences are zero."""
    d = np.asarray(d, float)
    if not np.any(d != 0):
        return 0.0, 1.0
    res = sps.wilcoxon(d, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def compare(wide: pd.DataFrame, ref: str, alpha: float = 0.05) -> pd.DataFrame:
    """``wide``: seeds x methods for one metric. The reference against every other column."""
    rows = []
    for m in wide.columns:
        if m == ref:
            continue
        d = (wide[ref] - wide[m]).to_numpy()
        w, p = wilcoxon(d)
        hl, lo, hi = hodges_lehmann(d, alpha)
        rows.append({"rival": m, "n": len(d), "mean_diff": float(d.mean()), "hl_diff": hl, "ci_low": lo,
                     "ci_high": hi, "W": w, "p": p, "r_rb": rank_biserial(d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["verdict"] = np.where(out["p_holm"] < alpha, np.where(out["hl_diff"] > 0, "better", "worse"), "inconclusive")
    return out


def friedman(wide: pd.DataFrame) -> Tuple[float, float, float]:
    """Friedman chi-square, p-value and Kendall's W over the columns of ``wide`` (rows = seeds)."""
    chi2, p = sps.friedmanchisquare(*[wide[m] for m in wide.columns])
    n, k = wide.shape
    return float(chi2), float(p), float(chi2 / (n * (k - 1)))
