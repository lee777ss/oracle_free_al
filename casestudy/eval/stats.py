"""casestudy.eval.stats — CV aggregation & significance tests (plan §8 / §13).

Standalone helpers for the case study's statistical analysis:
- ``aggregate_cv``      : mean ± std of per-fold metric dicts (selection diagnostics).
- ``average_ranks``     : per-method average rank across blocks (datasets/folds).
- ``friedman_test``     : omnibus Friedman test across all methods.
- ``wilcoxon_holm``     : Wilcoxon signed-rank of a reference method (Ab-8c) vs.
                          each competitor, with Holm step-down correction.

A "block" is one paired observation shared by all methods — here a (dataset)
or (dataset, fold) on which every method produced one score. ``method_scores``
maps ``method_name -> [score_block_0, score_block_1, ...]`` with aligned blocks.
For all metrics in this project **higher is better**, so ranks are assigned
with rank 1 = best.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

print("casestudy/eval/stats.py loaded")


def aggregate_cv(
    fold_results: List[Dict[str, float]]
) -> Dict[str, Tuple[float, float]]:
    """Mean ± std (ddof=0) of per-fold metric dicts; non-numeric values skipped."""
    if not fold_results:
        return {}
    summary: Dict[str, Tuple[float, float]] = {}
    for metric in fold_results[0].keys():
        values = [fold[metric] for fold in fold_results if metric in fold]
        if any(
            not isinstance(v, (int, float, np.integer, np.floating)) for v in values
        ):
            continue
        summary[metric] = (float(np.mean(values)), float(np.std(values)))
    return summary


def _score_matrix(
    method_scores: Dict[str, List[float]]
) -> Tuple[List[str], np.ndarray]:
    """Validate & stack scores into a ``[n_methods, n_blocks]`` matrix."""
    methods = list(method_scores.keys())
    if len(methods) < 2:
        raise ValueError("need at least 2 methods for ranking/significance tests")
    lengths = {len(v) for v in method_scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"all methods must share the same #blocks, got {lengths}")
    matrix = np.asarray([method_scores[m] for m in methods], dtype=float)
    return methods, matrix


def average_ranks(
    method_scores: Dict[str, List[float]], higher_is_better: bool = True
) -> Dict[str, float]:
    """Average rank per method across blocks (1 = best). Ties share mean rank."""
    from scipy.stats import rankdata

    methods, matrix = _score_matrix(method_scores)
    # Rank within each block (column): higher score -> better -> lower rank.
    sign = -1.0 if higher_is_better else 1.0
    ranks = np.vstack(
        [rankdata(sign * matrix[:, b], method="average") for b in range(matrix.shape[1])]
    ).T  # [n_methods, n_blocks]
    mean_ranks = ranks.mean(axis=1)
    return {m: float(r) for m, r in zip(methods, mean_ranks)}


def friedman_test(method_scores: Dict[str, List[float]]) -> Dict[str, object]:
    """Friedman omnibus test across all methods over the shared blocks.

    Returns ``{statistic, p_value, n_blocks, n_methods, average_ranks}``.
    Requires ≥3 methods and ≥2 blocks (scipy's requirement).
    """
    from scipy.stats import friedmanchisquare

    methods, matrix = _score_matrix(method_scores)
    if len(methods) < 3:
        raise ValueError("Friedman test requires at least 3 methods")
    if matrix.shape[1] < 2:
        raise ValueError("Friedman test requires at least 2 blocks")
    stat, p = friedmanchisquare(*[matrix[i, :] for i in range(matrix.shape[0])])
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_blocks": int(matrix.shape[1]),
        "n_methods": int(len(methods)),
        "average_ranks": average_ranks(method_scores),
    }


def _holm_adjust(pairs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    """Holm step-down adjustment. ``pairs`` = ``[(name, raw_p), ...]``.

    Returns ``[(name, adjusted_p), ...]`` in the original input order.
    """
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    adjusted = [0.0] * len(pairs)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pairs[idx][1]
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)  # enforce monotonic non-decreasing
        adjusted[idx] = running_max
    return [(pairs[i][0], adjusted[i]) for i in range(len(pairs))]


def wilcoxon_holm(
    reference: str,
    method_scores: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, Dict[str, object]]:
    """Wilcoxon signed-rank of ``reference`` vs. each competitor + Holm correction.

    Args:
        reference: Method key to compare against all others (e.g. ``"m10_ab8c"``).
        method_scores: ``{method: [scores across blocks]}`` with aligned blocks.
        alpha: Family-wise significance level for the ``significant`` flag.

    Returns:
        ``{competitor: {statistic, p_value, p_holm, significant, median_diff}}``
        where ``median_diff`` is ``median(reference - competitor)`` (positive
        favours the reference for higher-is-better metrics).
    """
    from scipy.stats import wilcoxon

    if reference not in method_scores:
        raise ValueError(f"reference {reference!r} not in method_scores")
    methods, matrix = _score_matrix(method_scores)
    ref_idx = methods.index(reference)
    ref = matrix[ref_idx, :]

    raw: List[Tuple[str, float]] = []
    stats: Dict[str, Dict[str, object]] = {}
    for i, name in enumerate(methods):
        if name == reference:
            continue
        diff = ref - matrix[i, :]
        if np.allclose(diff, 0.0):
            stat, p = 0.0, 1.0  # identical scores -> no difference
        else:
            stat, p = wilcoxon(ref, matrix[i, :])
        raw.append((name, float(p)))
        stats[name] = {
            "statistic": float(stat),
            "p_value": float(p),
            "median_diff": float(np.median(diff)),
        }

    for name, p_holm in _holm_adjust(raw):
        stats[name]["p_holm"] = float(p_holm)
        stats[name]["significant"] = bool(p_holm < alpha)
    return stats
