"""casestudy.reporting.tables — CSV + LaTeX comparison tables (plan §9.1).

Builds the reporting tables from the per-dataset held-out **test** CSVs written
by notebook §H (``saved/casestudy_<dataset>_CS1J.csv``):

- :func:`load_per_dataset_csvs` — read the four per-dataset CSVs.
- :func:`build_master_table`    — 11 methods × (dataset × metric) test matrix.
- :func:`master_to_latex`       — LaTeX with best value bold + Ab-8c row shaded.
- :func:`cross_dataset_ranking` — average rank per method (1 = best).
- :func:`significance_table`    — Friedman + Wilcoxon/Holm vs. a reference.

All metrics here are **higher-is-better** except ``fid`` (lower-is-better).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import METHOD_ORDER, METHOD_LABELS, GENERATIVE_METHODS
from ..eval.stats import average_ranks, friedman_test, wilcoxon_holm

print("casestudy/reporting/tables.py loaded")

# Headline held-out-test metrics (higher is better) + FID (lower is better).
TEST_METRICS: List[str] = ["macro_recall", "gmean", "macro_f2", "macro_specificity"]
METRIC_LABELS: Dict[str, str] = {
    "macro_recall": "Macro-Recall",
    "gmean": "G-Mean",
    "macro_f2": "Macro-F2",
    "macro_specificity": "Macro-Spec.",
    "fid": "FID",
}
LOWER_IS_BETTER = {"fid"}


# ── Loading ────────────────────────────────────────────────────────────────
def load_per_dataset_csvs(
    saved_dir: str, datasets: List[str], suffix: str
) -> Dict[str, pd.DataFrame]:
    """Load ``saved/casestudy_<dataset>_<suffix>.csv`` for each dataset."""
    out: Dict[str, pd.DataFrame] = {}
    for name in datasets:
        path = os.path.join(saved_dir, f"casestudy_{name}_{suffix}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing per-dataset CSV: {path}")
        out[name] = pd.read_csv(path)
    return out


# ── Master comparison table ─────────────────────────────────────────────────
def build_master_table(
    dfs: Dict[str, pd.DataFrame],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Master table: rows = methods (fixed order), cols = (dataset, metric).

    Values are held-out **test** metrics (``test_<metric>``) plus ``fid_mean``.
    The index is the human-readable method label; columns are a ``MultiIndex``
    ``(dataset, metric)``.
    """
    metrics = metrics or TEST_METRICS
    frames = []
    for name, df in dfs.items():
        d = df.set_index("method")
        cols = {(name, m): d[f"test_{m}"] for m in metrics}
        cols[(name, "fid")] = d["fid_mean"]
        frames.append(pd.DataFrame(cols))
    master = pd.concat(frames, axis=1)
    master = master.reindex([m for m in METHOD_ORDER if m in master.index])
    master.index = [METHOD_LABELS[m] for m in master.index]
    master.columns = pd.MultiIndex.from_tuples(master.columns, names=["dataset", "metric"])
    return master


def _best_index_per_column(master: pd.DataFrame) -> Dict[tuple, Optional[str]]:
    """Index label of the best (bold) value for each (dataset, metric) column."""
    best: Dict[tuple, Optional[str]] = {}
    for col in master.columns:
        s = master[col].dropna()
        if s.empty:
            best[col] = None
        elif col[1] in LOWER_IS_BETTER:
            best[col] = s.idxmin()
        else:
            best[col] = s.idxmax()
    return best


def master_to_latex(
    master: pd.DataFrame,
    reference_label: str = "Ab-8c (proposed)",
    float_fmt: str = "{:.3f}",
    caption: str = "Held-out test performance across the four case-study datasets.",
    label: str = "tab:casestudy_master",
) -> str:
    """LaTeX ``tabular`` with the best value per column **bold** and the
    reference (Ab-8c) row shaded via ``\\rowcolor`` (needs ``xcolor``/``colortbl``)."""
    best = _best_index_per_column(master)
    datasets = list(dict.fromkeys(c[0] for c in master.columns))
    metrics = list(dict.fromkeys(c[1] for c in master.columns))
    n_metric = len(metrics)

    lines: List[str] = []
    lines.append("% requires \\usepackage{booktabs,multirow,xcolor,colortbl}")
    lines.append("\\begin{table}[t]\\centering\\small")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{l" + ("c" * (len(datasets) * n_metric)) + "}")
    lines.append("\\toprule")
    # Dataset header row (multicolumn per dataset).
    head = ["\\multirow{2}{*}{Method}"]
    for ds in datasets:
        head.append(f"\\multicolumn{{{n_metric}}}{{c}}{{{ds}}}")
    lines.append(" & ".join(head) + " \\\\")
    # cmidrule under each dataset block.
    rules = []
    start = 2
    for _ in datasets:
        rules.append(f"\\cmidrule(lr){{{start}-{start + n_metric - 1}}}")
        start += n_metric
    lines.append(" ".join(rules))
    # Metric header row.
    sub = [""]
    for _ in datasets:
        sub.extend(METRIC_LABELS.get(m, m) for m in metrics)
    lines.append(" & ".join(sub) + " \\\\")
    lines.append("\\midrule")

    for method_label in master.index:
        cells = [method_label]
        for col in master.columns:
            v = master.loc[method_label, col]
            txt = "--" if pd.isna(v) else float_fmt.format(v)
            if not pd.isna(v) and best.get(col) == method_label:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        row = " & ".join(cells) + " \\\\"
        if method_label == reference_label:
            row = "\\rowcolor{gray!15} " + row
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ── Cross-dataset ranking ───────────────────────────────────────────────────
def _method_scores_by_dataset(
    dfs: Dict[str, pd.DataFrame], metric: str
) -> Dict[str, List[float]]:
    """``{method_key: [score_per_dataset]}`` for one ``test_<metric>`` column."""
    methods = [m for m in METHOD_ORDER]
    scores: Dict[str, List[float]] = {m: [] for m in methods}
    for name, df in dfs.items():
        d = df.set_index("method")
        col = f"test_{metric}"
        for m in methods:
            scores[m].append(float(d.loc[m, col]) if m in d.index else np.nan)
    # Drop methods with any missing score (keeps blocks aligned).
    return {m: v for m, v in scores.items() if not any(np.isnan(v))}


def cross_dataset_ranking(
    dfs: Dict[str, pd.DataFrame],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Average rank per method across datasets for each metric + an overall mean.

    Rank 1 = best on a dataset (higher metric → better; FID excluded here as it
    is undefined for non-generative methods). Returns a DataFrame indexed by
    method label with one column per metric plus ``avg_rank`` (mean over the
    metric columns) sorted ascending (best first).
    """
    metrics = metrics or TEST_METRICS
    out: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        scores = _method_scores_by_dataset(dfs, metric)
        ranks = average_ranks(scores, higher_is_better=True)
        for m, r in ranks.items():
            out.setdefault(m, {})[METRIC_LABELS.get(metric, metric)] = r
    df = pd.DataFrame(out).T
    df["avg_rank"] = df.mean(axis=1)
    df = df.reindex([m for m in METHOD_ORDER if m in df.index])
    df.index = [METHOD_LABELS[m] for m in df.index]
    return df.sort_values("avg_rank")


# ── Significance ────────────────────────────────────────────────────────────
def _blocks_across_metrics(
    dfs: Dict[str, pd.DataFrame], metrics: List[str]
) -> Dict[str, List[float]]:
    """``{method: [score for each (dataset, metric) block]}`` (paired blocks).

    Stacks all ``len(datasets) × len(metrics)`` held-out-test observations so
    Friedman/Wilcoxon have more paired blocks than datasets alone.
    """
    per_metric = [_method_scores_by_dataset(dfs, m) for m in metrics]
    common = set.intersection(*[set(pm) for pm in per_metric])
    methods = [m for m in METHOD_ORDER if m in common]
    return {m: [s for pm in per_metric for s in pm[m]] for m in methods}


def significance_table(
    dfs: Dict[str, pd.DataFrame],
    reference: str = "m10_ab8c",
    metrics: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Friedman omnibus + Wilcoxon/Holm of ``reference`` vs. each competitor.

    Blocks are all ``(dataset, metric)`` held-out-test observations. Returns a
    DataFrame (one row per competitor) with the reference's median advantage,
    raw/Holm-adjusted p-values, and a significance flag. The Friedman result is
    attached via ``df.attrs['friedman']``.
    """
    metrics = metrics or TEST_METRICS
    scores = _blocks_across_metrics(dfs, metrics)
    fried = friedman_test(scores)
    wh = wilcoxon_holm(reference, scores, alpha=alpha)

    rows = []
    for m in METHOD_ORDER:
        if m == reference or m not in wh:
            continue
        s = wh[m]
        rows.append({
            "method": METHOD_LABELS[m],
            "median_diff(ref-comp)": round(s["median_diff"], 4),
            "p_value": round(s["p_value"], 4),
            "p_holm": round(s["p_holm"], 4),
            "significant": s["significant"],
        })
    df = pd.DataFrame(rows)
    df.attrs["friedman"] = fried
    df.attrs["reference"] = METHOD_LABELS.get(reference, reference)
    return df
