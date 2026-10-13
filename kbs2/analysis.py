"""kbs2.analysis — the pre-specified analysis (E4), fixed before E1 (rules B5, B7).

    python -m kbs2.analysis E1|E2|E8|E7|E3|E4     # writes saved_kbs2/analysis/<step>/

Applied at each STOP to the experiment just completed; E4 consolidates. Nothing in
this file may be changed after E1 has run without author approval (rule B6).

Pre-specified analysis
----------------------
Metrics: macro-F2 (headline), macro-recall, G-mean, macro-specificity; mean +- SD over
seeds 42-51. Every comparison is the proposed method against one other method, paired
by seed:
  * two-sided Wilcoxon signed-rank test, ``scipy.stats.wilcoxon(zero_method="wilcox",
    method="auto")``; p = 1 when every paired difference is zero;
  * Holm correction across all comparisons of one table block (one dataset, or one grid
    cell; E7: its three comparisons per dataset); alpha = 0.05;
  * Hodges-Lehmann estimate of the paired difference (proposed minus other; median of
    the Walsh averages of all ten differences, zeros included) with the exact 95% CI from
    the signed-rank null distribution;
  * matched-pairs rank-biserial correlation (zeros dropped);
  * Friedman test across all methods of the block with complete seeds.
Main table (arm W): the original seven methods use their submitted per-seed values
(``saved_casestudy/*_reduced8.csv``); E1/E8 reproduce them, and any difference is
reported beside the table, not substituted. New baselines use the E2 runs.
Supplement S1: methods run under the same protocol but not reported (``*_full8.csv``).
E8 arm U table: E8 runs; the naive baseline (no weighting in either arm) is shown from
the submitted rows. Per method, the W - U difference is descriptive.
E7: (2) proposed+online vs (1) online only; (3) comparator+online vs (1); (2) vs (3);
Holm over the three per dataset.
E3: per cell, the main-field comparisons; the full field incl. Borderline-SMOTE in the
supplement.
A1 (exploratory/descriptive only): per cell, the mean paired difference in macro-F2
between the proposed method and EACH of the four simple remedies separately, against
n_min; Spearman rho over the 15 cells per remedy as a descriptive summary (no p-value is
used for any claim; no per-cell "best remedy" is selected).
E1: fallback summaries from the gate log: share of class-cycles in which the fallback
fired, share of admitted images that failed the gate, both per class and for the first
and last cycle; gate pass rate.
Near-copy rate (decision D6), defined here before any output exists: for an added
image x of class c, let d1 <= d2 be its two smallest pixel-space L2 distances to the
real images of class c in that run's training portion, and q = d1 / d2. The reference
distribution of q is computed the same way for up to 500 real images of class c from
the official training split that are not in the pool (drawn with numpy seed 0). x is a
near-copy if q is below the 5th percentile of that reference distribution, so about 5%
of unseen real images count as near-copies by construction. Reported per method as the
share of added images that are near-copies.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2")
METRICS = ["macro_f2", "macro_recall", "gmean", "macro_specificity"]
ALPHA = 0.05


# ── statistics ──────────────────────────────────────────────────────────────
def signed_rank_null(n: int) -> np.ndarray:
    """P(W+ = w) for w = 0..n(n+1)/2 under H0 (no ties)."""
    top = n * (n + 1) // 2
    counts = np.zeros(top + 1)
    counts[0] = 1.0
    for k in range(1, n + 1):
        new = counts.copy()
        new[k:] += counts[:top + 1 - k]
        counts = new
    return counts / 2.0 ** n


def hodges_lehmann(d: Sequence[float], alpha: float = ALPHA) -> Tuple[float, float, float]:
    d = np.asarray(d, dtype=float)
    n = len(d)
    walsh = np.sort([(d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n)])
    est = float(np.median(walsh))
    cdf = np.cumsum(signed_rank_null(n))
    below = np.where(cdf <= alpha / 2)[0]
    if below.size == 0:
        return est, float("-inf"), float("inf")
    c = int(below[-1])                  # largest w with P(W+ <= w) <= alpha/2
    # CI = [(c+1)-th smallest, (c+1)-th largest Walsh average]
    return est, float(walsh[c]), float(walsh[len(walsh) - 1 - c])


def rank_biserial(d: Sequence[float]) -> float:
    from scipy.stats import rankdata
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def wilcoxon_p(d: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import wilcoxon
    d = np.asarray(d, dtype=float)
    if np.all(d == 0):
        return float("nan"), 1.0
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def holm(p: Sequence[float]) -> List[float]:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def friedman(block: Dict[str, np.ndarray]) -> Tuple[float, float]:
    from scipy.stats import friedmanchisquare
    cols = [v for v in block.values() if len(v) == len(P.SEEDS)]
    if len(cols) < 3:
        return float("nan"), float("nan")
    res = friedmanchisquare(*cols)
    return float(res.statistic), float(res.pvalue)


# ── data assembly ───────────────────────────────────────────────────────────
def _read(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    if "error" in df:
        df = df[df["error"].isna() | (df["error"].astype(str) == "")]
    return df


def submitted_rows(regime: str, methods: List[str], csv_path: Optional[str] = None):
    df = _read(csv_path or P.REFERENCE_ANCHOR_CSV[regime])
    df = df[df["method"].isin(methods)].copy()
    df["source"] = "submitted"
    df["arm"] = "W"
    return df


def kbs2_rows(exp: str, arm: Optional[str] = None, tag: Optional[str] = None,
              regime: Optional[str] = None, methods: Optional[List[str]] = None):
    import pandas as pd
    path = os.path.join(OUT, exp, "runs.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["method", "seed", "regime", "arm", "tag", *METRICS])
    df = pd.read_csv(path)
    if arm:
        df = df[df["arm"] == arm]
    if tag is not None:
        df = df[df["tag"].fillna("") == tag]
    if regime is not None:
        df = df[df["regime"] == regime]
    if methods is not None:
        df = df[df["method"].isin(methods)]
    df = df.copy()
    df["source"] = exp
    return df


def per_seed(df, method: str, metric: str) -> np.ndarray:
    s = df[df["method"] == method].sort_values("seed")
    return s[metric].to_numpy(dtype=float)


def compare_block(df, methods: List[str], label: str, proposed: str = P.PROPOSED,
                  metric: str = "macro_f2"):
    """Descriptives for every method + proposed-vs-each comparisons with Holm."""
    import pandas as pd
    desc = []
    for m in methods:
        row = {"block": label, "method": m, "label": P.LABELS.get(m, m)}
        for k in METRICS:
            v = per_seed(df, m, k)
            row[f"{k}_mean"] = float(v.mean()) if v.size else float("nan")
            row[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        row["n_seeds"] = int(per_seed(df, m, metric).size)
        desc.append(row)
    a = per_seed(df, proposed, metric)
    comps = []
    for m in methods:
        if m == proposed:
            continue
        b = per_seed(df, m, metric)
        if a.size != len(P.SEEDS) or b.size != len(P.SEEDS):
            continue
        d = a - b
        w, p = wilcoxon_p(d)
        hl, lo, hi = hodges_lehmann(d)
        comps.append({"block": label, "versus": m, "mean_diff": float(d.mean()), "hl": hl,
                      "ci_low": lo, "ci_high": hi, "W": w, "p": p, "r_rb": rank_biserial(d)})
    if comps:
        for c, ph in zip(comps, holm([c["p"] for c in comps])):
            c["p_holm"] = ph
            c["verdict"] = ("proposed better" if c["hl"] > 0 else "proposed worse") \
                if ph < ALPHA else "inconclusive"
    fr = friedman({m: per_seed(df, m, metric) for m in methods})
    return pd.DataFrame(desc), pd.DataFrame(comps), {"block": label, "friedman_chi2": fr[0], "friedman_p": fr[1]}


# ── per-step analyses ───────────────────────────────────────────────────────
def e2_tables():
    """E2 as re-specified for D14 (approval E2.json): descriptive only, no tests (those come in E4).

    interim: mean and SD per method and metric in arm Ws (classifier seeded with the run seed) for the
    naive baseline, the four new baselines and the proposed method (E1's tau = 0.6 sets retrained in
    arm Ws by E1b); the submitted seed-42 values of the naive baseline and the proposed method are
    added for reference. hashes: within each regime, the number of distinct pool / test / hold-out
    hashes over all E2 runs, and of initial classifier weights per arm.
    """
    import pandas as pd
    rows = []
    for ds, (rid, _r) in P.ANCHORS.items():
        parts = [kbs2_rows("E2", "Ws", regime=rid).assign(setting="Ws (run seed)"),
                 kbs2_rows("E1b", "Ws", tag="tau06", regime=rid).assign(setting="Ws (run seed)"),
                 submitted_rows(rid, ["m11_imbalanced_naive", P.PROPOSED]).assign(setting="W (seed 42, submitted)")]
        df = pd.concat(parts)
        for (setting, m), g in df.groupby(["setting", "method"]):
            row = {"dataset": ds, "setting": setting, "method": m, "label": P.LABELS.get(m, m),
                   "n_seeds": int(g["seed"].nunique())}
            for k in METRICS:
                row[f"{k}_mean"] = float(g[k].mean())
                row[f"{k}_sd"] = float(g[k].std(ddof=1))
            rows.append(row)
    interim = pd.DataFrame(rows)
    runs = pd.read_csv(os.path.join(OUT, "E2", "runs.csv"))
    hashes = runs.groupby("regime").agg(pool=("pool_sha", "nunique"), test=("test_sha", "nunique"),
                                        holdout=("holdout_index_sha", "nunique")).reset_index()
    init = runs.groupby(["regime", "arm"])["init_weights_sha"].nunique().unstack().add_prefix("init_weights_")
    return interim, hashes.merge(init.reset_index(), on="regime")


def e2u_tables():
    """E2u (approval E2u.json): weighted (E2 / E1b arm Ws) vs unweighted (E2u arm Us), descriptive.

    table: mean and SD per method, weighting and metric; paired: weighted minus unweighted per seed
    (mean difference and number of seeds on which the weighted arm is higher), per method and metric.
    """
    import pandas as pd
    methods = ["n02_ros", "n03_stdaug_budget", "n04_stdaug_online", P.PROPOSED]
    w = pd.concat([kbs2_rows("E2", "Ws", methods=methods),
                   kbs2_rows("E1b", "Ws", tag="tau06")]).assign(weighting="weighted")
    u = kbs2_rows("E2u", "Us", methods=methods).assign(weighting="unweighted")
    naive = kbs2_rows("E2", "Ws", methods=["m11_imbalanced_naive"]).assign(weighting="unweighted (accuracy selection)")
    rows, pairs = [], []
    for (ds, m, wt), g in pd.concat([w, u, naive]).groupby(["dataset", "method", "weighting"]):
        row = {"dataset": ds, "method": m, "label": P.LABELS.get(m, m), "weighting": wt,
               "n_seeds": int(g["seed"].nunique())}
        for k in METRICS:
            row[f"{k}_mean"], row[f"{k}_sd"] = float(g[k].mean()), float(g[k].std(ddof=1))
        rows.append(row)
    for (ds, m), gw in w.groupby(["dataset", "method"]):
        gu = u[(u["dataset"] == ds) & (u["method"] == m)]
        a, b = gw.set_index("seed"), gu.set_index("seed")
        j = a.index.intersection(b.index)
        for k in METRICS:
            d = a.loc[j, k] - b.loc[j, k]
            pairs.append({"dataset": ds, "method": m, "label": P.LABELS.get(m, m), "metric": k,
                          "mean_diff_weighted_minus_unweighted": float(d.mean()),
                          "seeds_weighted_higher": int((d > 0).sum()), "seeds": int(len(d))})
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def table6_ws():
    """Table 6 under D14 (arm Ws, classifier seeded with the run seed), descriptive.

    Naive baseline and new baselines from E2, generative rivals from E8, proposed method from E1b
    (E1's tau = 0.6 sets). Mean and SD per method and metric; the statistics come in E4.
    """
    import pandas as pd
    order = ["m11_imbalanced_naive", *P.NEW_BASELINES, *P.GENERATIVE_RIVALS, P.PROPOSED]
    df = pd.concat([kbs2_rows("E2", "Ws"), kbs2_rows("E8", "Ws"), kbs2_rows("E1b", "Ws", tag="tau06")])
    rows = []
    for ds in P.ANCHORS:
        for m in order:
            g = df[(df["dataset"] == ds) & (df["method"] == m)]
            row = {"dataset": ds, "method": m, "label": P.LABELS.get(m, m), "n_seeds": int(g["seed"].nunique())}
            for k in METRICS:
                row[f"{k}_mean"] = float(g[k].mean()) if len(g) else float("nan")
                row[f"{k}_sd"] = float(g[k].std(ddof=1)) if len(g) > 1 else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def table6_both():
    """Table 6 under both classifier seedings (author request 11 Oct), descriptive.

    Seed 42 (arm W, submitted protocol): naive from E2, remedies from E2w, rivals from E8 (reproduced
    submitted rows), proposed from E1. Run seed (arm Ws, D14): naive and remedies from E2, rivals from
    E8, proposed from E1b (E1's sets). Returns a long table and a markdown rendering (means; SDs in csv).
    """
    import pandas as pd
    order = ["m11_imbalanced_naive", *P.NEW_BASELINES, *P.GENERATIVE_RIVALS, P.PROPOSED]
    w = pd.concat([kbs2_rows("E2", "W"), kbs2_rows("E2w", "W"), kbs2_rows("E8", "W"),
                   kbs2_rows("E1", "W")]).assign(setting="seed 42")
    s = pd.concat([kbs2_rows("E2", "Ws"), kbs2_rows("E8", "Ws"),
                   kbs2_rows("E1b", "Ws", tag="tau06")]).assign(setting="run seed")
    df = pd.concat([w, s])
    rows = []
    for ds in P.ANCHORS:
        for m in order:
            for st in ("seed 42", "run seed"):
                g = df[(df["dataset"] == ds) & (df["method"] == m) & (df["setting"] == st)]
                row = {"dataset": ds, "method": m, "label": P.LABELS.get(m, m), "setting": st,
                       "n_seeds": int(g["seed"].nunique()), "n_distinct_f2": int(g["macro_f2"].round(10).nunique())}
                for k in METRICS:
                    row[f"{k}_mean"] = float(g[k].mean()) if len(g) else float("nan")
                    row[f"{k}_sd"] = float(g[k].std(ddof=1)) if len(g) > 1 else float("nan")
                rows.append(row)
    t = pd.DataFrame(rows)
    names = {"macro_recall": "Recall", "gmean": "G-mean", "macro_f2": "F2", "macro_specificity": "Specificity"}
    md = []
    for ds in P.ANCHORS:
        md += [f"**{ds}** (seed 42 / run seed)", "", "| Method | " + " | ".join(names[k] for k in METRICS) + " |",
               "|---|" + "---|" * len(METRICS)]
        for m in order:
            a = t[(t.dataset == ds) & (t.method == m) & (t.setting == "seed 42")].iloc[0]
            b = t[(t.dataset == ds) & (t.method == m) & (t.setting == "run seed")].iloc[0]
            md.append(f"| {a['label']} | " + " | ".join(f"{a[k + '_mean']:.3f} / {b[k + '_mean']:.3f}" for k in METRICS) + " |")
        md.append("")
    return t, "\n".join(md)


def main_table(arm: str = "W"):
    import pandas as pd
    descs, comps, frs = [], [], []
    for ds, (rid, _r) in P.ANCHORS.items():
        if arm == "W":
            df = pd.concat([submitted_rows(rid, P.ORIGINAL_FIELD),
                            kbs2_rows("E2", "W", regime=rid, methods=P.NEW_BASELINES)])
        else:
            df = pd.concat([submitted_rows(rid, ["m11_imbalanced_naive"]),
                            kbs2_rows("E8", "U", regime=rid)])
        d, c, f = compare_block(df, P.MAIN_FIELD, ds)
        descs.append(d); comps.append(c); frs.append(f)
    return pd.concat(descs), pd.concat(comps), pd.DataFrame(frs)


def supplement_s1():
    import pandas as pd
    descs = []
    for ds, (rid, _r) in P.ANCHORS.items():
        df = submitted_rows(rid, P.SUPPLEMENT_S1 + ["m08_bagan", P.PROPOSED], P.REFERENCE_FULL8_CSV[rid])
        d, _, _ = compare_block(df, P.SUPPLEMENT_S1 + ["m08_bagan", P.PROPOSED], ds)
        d.loc[d["method"] == "m08_bagan", "label"] = "BAGAN (no label smoothing)"
        descs.append(d)
    return pd.concat(descs)


def e7_table():
    import pandas as pd
    descs, comps = [], []
    for ds, (rid, _r) in P.ANCHORS.items():
        comp = P.E7_COMPARATOR[ds]
        # Arm W (classifier seed 42) of on-the-fly augmentation exists only in E2w; E2 ran it in arm Ws.
        # Corrected 13 Oct 2026, before any E7 run.
        a1 = kbs2_rows("E2w", "W", regime=rid, methods=["n04_stdaug_online"]).assign(arm_e7="1")
        e7 = kbs2_rows("E7", "W", tag="plus_online", regime=rid)
        a2 = e7[e7["method"] == P.PROPOSED].assign(arm_e7="2")
        a3 = e7[e7["method"] == comp].assign(arm_e7="3")
        arms = {"1": a1, "2": a2, "3": a3}
        for k, v in arms.items():
            row = {"dataset": ds, "arm": k, "comparator": comp}
            for m in METRICS:
                x = v.sort_values("seed")[m].to_numpy(float)
                row[f"{m}_mean"], row[f"{m}_sd"] = (float(x.mean()), float(x.std(ddof=1))) if x.size > 1 else (np.nan, np.nan)
            descs.append(row)
        pairs = [("2", "1"), ("3", "1"), ("2", "3")]
        block = []
        for x, y in pairs:
            ax = arms[x].sort_values("seed")["macro_f2"].to_numpy(float)
            ay = arms[y].sort_values("seed")["macro_f2"].to_numpy(float)
            if ax.size != len(P.SEEDS) or ay.size != len(P.SEEDS):
                continue
            d = ax - ay
            w, p = wilcoxon_p(d)
            hl, lo, hi = hodges_lehmann(d)
            block.append({"dataset": ds, "comparison": f"({x}) vs ({y})", "mean_diff": float(d.mean()),
                          "hl": hl, "ci_low": lo, "ci_high": hi, "W": w, "p": p, "r_rb": rank_biserial(d)})
        for c, ph in zip(block, holm([c["p"] for c in block])):
            c["p_holm"] = ph
        comps += block
    return pd.DataFrame(descs), pd.DataFrame(comps)


def grid_tables(full: bool = False):
    import pandas as pd
    descs, comps = [], []
    field = P.MAIN_FIELD + (["m02_bsmote"] if full else [])
    for ds, cells in P.GRID_SWEEP.items():
        rid_a = P.ANCHORS[ds][0]
        for rid, rho in [(rid_a, P.ANCHOR_RHO[rid_a])] + cells:
            ref = submitted_rows(rid, P.ORIGINAL_FIELD + ["m02_bsmote"], P.reference_grid_csv(ds, rid))
            new = kbs2_rows("E2" if rid == rid_a else "E3", "W", regime=rid, methods=P.NEW_BASELINES)
            df = pd.concat([ref, new])
            d, c, _ = compare_block(df, field, f"{ds}|{rho}")
            d["rho"] = rho; c["rho"] = rho
            descs.append(d); comps.append(c)
    return pd.concat(descs), pd.concat(comps)


def a1_descriptive():
    """Exploratory/descriptive only (rule B5)."""
    import pandas as pd
    from scipy.stats import spearmanr
    desc, _ = grid_tables(full=False)
    rows = []
    for block, g in desc.groupby("block"):
        ds, rho = block.split("|")
        rid = P.ANCHORS[ds][0] if float(rho) == P.ANCHOR_RHO[P.ANCHORS[ds][0]] else \
            [r for r, x in P.GRID_SWEEP[ds] if x == float(rho)][0]
        n_maj = {"fashion_mnist": 800, "cifar10": 1000, "stl10": 400}[ds]
        n_min = int(round(float(rho) * n_maj))
        prop = g[g["method"] == P.PROPOSED]["macro_f2_mean"].iloc[0]
        for rem in P.NEW_BASELINES:
            other = g[g["method"] == rem]["macro_f2_mean"]
            rows.append({"dataset": ds, "rho": float(rho), "regime": rid, "n_min_pool": n_min,
                         "remedy": rem, "diff_proposed_minus_remedy": float(prop - other.iloc[0]) if len(other) else np.nan})
    df = pd.DataFrame(rows)
    summ = []
    for rem, g in df.groupby("remedy"):
        g = g.dropna()
        rho_s = spearmanr(g["n_min_pool"], g["diff_proposed_minus_remedy"]).statistic if len(g) > 2 else np.nan
        summ.append({"remedy": rem, "n_cells": len(g), "spearman_rho_descriptive": float(rho_s)})
    return df, pd.DataFrame(summ)


def e1_fallback():
    import pandas as pd
    import torch
    recs = []
    for path in sorted(glob.glob(os.path.join(OUT, "E1", "synth", "*.pt"))):
        rec = torch.load(path, weights_only=False)  # written by kbs2.runner.generate
        name = os.path.basename(path)[:-3]
        method, regime, seed = name.split("__")
        log = rec.get("gate_log", [])
        last = max((e["cycle"] for e in log), default=-1)
        for e in log:
            recs.append({"regime": regime, "seed": int(seed[1:]), **e, "is_first": e["cycle"] == 0,
                         "is_last": e["cycle"] == last,
                         "pass_rate": e["n_pass"] / max(1, e["n_candidates"])})
    df = pd.DataFrame(recs)
    if df.empty:
        return df, df
    out = []
    for (regime, c), g in df.groupby(["regime", "class"]):
        out.append({"regime": regime, "class": int(c), "class_cycles": len(g),
                    "fallback_rate": float(g["fallback"].mean()),
                    "admitted_failing_share": float(g["n_admitted_failing_gate"].sum() / g["n_admitted"].sum()),
                    "fallback_rate_first_cycle": float(g[g["is_first"]]["fallback"].mean()),
                    "fallback_rate_last_cycle": float(g[g["is_last"]]["fallback"].mean()),
                    "mean_pass_rate": float(g["pass_rate"].mean())})
    return df, pd.DataFrame(out)


def e1b_tables():
    """E1b (pre-specified, 10 Oct 2026; descriptive because tau = 0.5 is fixed by decision D13).

    For each dataset, metric and classifier seeding ("fixed" = seed 42 as submitted, "run" = run
    seed, D14): mean +- SD at tau 0.6 and 0.5, paired difference (0.5 minus 0.6) with HL estimate,
    exact 95% CI and two-sided Wilcoxon p (unadjusted; not used for any selection).
    Also: per class and cycle, fallback and pass rate with the internal classifier's diagnostic
    macro-F2 (tau 0.5; tau 0.6 has no diagnostic log in E1).
    """
    import pandas as pd
    import torch
    e1 = kbs2_rows("E1", "W")
    t06_fixed = e1[e1["method"] == P.PROPOSED]
    t05_fixed = kbs2_rows("E1b", "W", tag="tau05")
    t05_run = kbs2_rows("E1b", "Ws", tag="tau05")
    t06_run = kbs2_rows("E1b", "Ws", tag="tau06")
    comp = []
    for seeding, a06, a05 in [("fixed", t06_fixed, t05_fixed), ("run", t06_run, t05_run)]:
        for ds, (rid, _r) in P.ANCHORS.items():
            x6 = a06[a06["regime"] == rid].sort_values("seed")
            x5 = a05[a05["regime"] == rid].sort_values("seed")
            for m in METRICS:
                v6, v5 = x6[m].to_numpy(float), x5[m].to_numpy(float)
                row = {"seeding": seeding, "dataset": ds, "metric": m,
                       "tau06_mean": v6.mean() if v6.size else np.nan, "tau06_sd": v6.std(ddof=1) if v6.size > 1 else np.nan,
                       "tau05_mean": v5.mean() if v5.size else np.nan, "tau05_sd": v5.std(ddof=1) if v5.size > 1 else np.nan,
                       "n": int(min(v6.size, v5.size))}
                if v6.size == v5.size == len(P.SEEDS):
                    d = v5 - v6
                    w, p = wilcoxon_p(d)
                    hl, lo, hi = hodges_lehmann(d)
                    row.update(mean_diff=float(d.mean()), hl=hl, ci_low=lo, ci_high=hi, W=w, p_two_sided=p)
                comp.append(row)
    cyc = []
    for exp, tau in [("E1b", 0.5), ("E1", 0.6)]:
        for path in sorted(glob.glob(os.path.join(OUT, exp, "synth", f"{P.PROPOSED}__*.pt"))):
            rec = torch.load(path, weights_only=False)  # written by kbs2.runner.generate
            _m, regime, seed = os.path.basename(path)[:-3].split("__")
            val = rec.get("val_f2_by_cycle", [])
            for e in rec.get("gate_log", []):
                cyc.append({"tau": tau, "regime": regime, "seed": int(seed[1:]), "class": e["class"],
                            "cycle": e["cycle"], "fallback": e["fallback"],
                            "pass_rate": e["n_pass"] / max(1, e["n_candidates"]),
                            "admitted_failing": e["n_admitted_failing_gate"] / max(1, e["n_admitted"]),
                            "n_admitted": e["n_admitted"], "n_failing": e["n_admitted_failing_gate"],
                            "internal_val_macro_f2": val[e["cycle"]] if e["cycle"] < len(val) else np.nan})
    cyc = pd.DataFrame(cyc)
    by_cycle = (cyc.groupby(["tau", "regime", "cycle"])
                .agg(fallback_rate=("fallback", "mean"), pass_rate=("pass_rate", "mean"),
                     admitted_failing=("admitted_failing", "mean"),
                     internal_val_macro_f2=("internal_val_macro_f2", "mean"), n=("fallback", "size"))
                .reset_index()) if not cyc.empty else cyc
    overall = []
    for (tau, regime), g in cyc.groupby(["tau", "regime"]) if not cyc.empty else []:
        overall.append({"tau": tau, "regime": regime, "class_cycles": len(g),
                        "fallback_rate": float(g["fallback"].mean()),
                        "admitted_failing_share": float(g["n_failing"].sum() / max(1, g["n_admitted"].sum())),
                        "mean_pass_rate": float(g["pass_rate"].mean())})
    return pd.DataFrame(comp), by_cycle, pd.DataFrame(overall)


def near_copy_rates(exp_sets: Dict[str, str]) -> "object":
    """D6: share of added images that are near-copies (definition in the module docstring).

    ``exp_sets`` maps method -> experiment folder whose ``synth`` sets are read
    (e.g. {"m10_ab8c": "E1", "m04_dcgan": "E8"}).
    """
    import pandas as pd
    import torch
    from casestudy.data.loaders import _filter_classes, _load_raw, _remap
    from casestudy.data.transforms import to_normalized_tensor
    from kbs2.runner import load_regime, split

    rows = []
    for ds, (rid, _r) in P.ANCHORS.items():
        D = load_regime(rid)
        cfg, spec, pool = D["config"], D["spec"], D["pool"]
        tr, _ho = split(pool, cfg)
        Xtr, ytr = pool["images"][tr], pool["labels"][tr]
        cache = np.load(D["cache"][0], allow_pickle=False)
        pool_hashes = set(cache["hashes"].tolist())
        raw_x, raw_y = _load_raw(spec.loader_key, True, cfg.data_root)
        imgs, lbls = _filter_classes(raw_x, raw_y, pool["classes"])
        lbls = _remap(lbls, pool["classes"])
        import hashlib
        unused = np.array([hashlib.sha1(np.ascontiguousarray(imgs[i]).tobytes()).hexdigest() not in pool_hashes
                           for i in range(len(imgs))])
        rng = np.random.default_rng(0)
        thresholds = {}
        for c in sorted(set(ytr.tolist()) - {0}):
            cand = np.where(unused & (lbls == c))[0]
            pick = rng.choice(cand, size=min(500, len(cand)), replace=False)
            ref = to_normalized_tensor(imgs[pick], spec.channels, cfg.image_size, cfg.normalize_range)
            thresholds[c] = float(np.percentile(_ratio_q(ref, Xtr[ytr == c]), 5))
        for method, exp in exp_sets.items():
            for s in P.SEEDS:
                path = os.path.join(OUT, exp, "synth", f"{method}__{rid}__s{s}.pt")
                if not os.path.exists(path):
                    continue
                rec = torch.load(path, weights_only=False)  # written by kbs2.runner.generate
                X, y = rec["X"], rec["y"]
                for c, thr in thresholds.items():
                    xc = X[y == c]
                    if xc.numel() == 0:
                        continue
                    q = _ratio_q(xc, Xtr[ytr == c])
                    rows.append({"dataset": ds, "method": method, "seed": s, "class": c,
                                 "n_added": int(xc.size(0)), "near_copy_share": float((q < thr).mean()),
                                 "threshold_q": thr})
    return pd.DataFrame(rows)


def _ratio_q(x, train):
    import torch
    a = x.reshape(x.size(0), -1).double()
    b = train.reshape(train.size(0), -1).double()
    d = torch.cdist(a, b)
    two = torch.topk(d, k=2, largest=False).values
    return (two[:, 0] / two[:, 1].clamp_min(1e-12)).numpy()


# ── CLI ─────────────────────────────────────────────────────────────────────
def _save(step: str, **frames) -> None:
    d = os.path.join(OUT, "analysis", step)
    os.makedirs(d, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(os.path.join(d, f"{name}.csv"), index=False)
        print(f"wrote {os.path.join(d, name)}.csv ({len(df)} rows)")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["E1", "E1b", "E2", "E2u", "E2w", "E8", "E7", "E3", "E4"])
    a = ap.parse_args(argv)
    if a.step == "E1b":
        comp, by_cycle, overall = e1b_tables()
        nc = near_copy_rates({P.PROPOSED: "E1b"})
        _save("E1b", tau_comparison=comp, fallback_by_cycle=by_cycle, fallback_overall=overall,
              near_copy_tau05=nc)
    elif a.step == "E1":
        log, summ = e1_fallback()
        nc = near_copy_rates({P.PROPOSED: "E1"})
        _save("E1", gate_log=log, fallback_summary=summ, near_copy=nc)
    elif a.step == "E2":
        interim, hashes = e2_tables()
        _save("E2", interim_table6_Ws=interim, hash_checks=hashes)
    elif a.step == "E2w":
        t, md = table6_both()
        _save("E2w", table6_both_settings=t)
        with open(os.path.join(OUT, "analysis", "E2w", "table6_both_settings.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
    elif a.step == "E2u":
        table, pairs = e2u_tables()
        _save("E2u", weighted_vs_unweighted=table, paired_differences=pairs)
    elif a.step == "E8":
        _save("E8", table6_Ws=table6_ws())
    elif a.step == "E7":
        d, c = e7_table()
        _save("E7", e7_desc=d, e7_comparisons=c)
    elif a.step == "E3":
        d, c = grid_tables(False)
        df, dc = grid_tables(True)
        _save("E3", grid_desc=d, grid_comparisons=c, grid_full_desc=df, grid_full_comparisons=dc)
    else:
        dW, cW, fW = main_table("W")
        dU, cU, fU = main_table("U")
        a1, a1s = a1_descriptive()
        nc = near_copy_rates({P.PROPOSED: "E1", **{m: "E8" for m in P.GENERATIVE_RIVALS},
                              "n02_ros": "E2", "n03_stdaug_budget": "E2"})
        _save("E4", main_desc=dW, main_comparisons=cW, main_friedman=fW, armU_desc=dU,
              armU_comparisons=cU, armU_friedman=fU, a1_cells=a1, a1_summary=a1s, near_copy=nc)


if __name__ == "__main__":
    main()
