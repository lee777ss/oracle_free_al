"""casestudy.reporting.plots — figures for the case study (plan §9.2).

Publication-quality (≥300 DPI) figures built from the per-dataset held-out
**test** CSVs (and, for grids, freshly generated minority samples):

- :func:`grouped_bar_charts`  — one panel per metric, methods × datasets.
- :func:`ranking_heatmap`     — method × dataset rank grid for a metric.
- :func:`cd_diagram`          — Nemenyi critical-difference diagram.
- :func:`per_class_fid_bars`  — per-class FID per generative method, per dataset.
- :func:`sample_grid`         — real vs. synthetic minority image grid.

All ``save`` helpers write PNG (+ optional PDF) into ``figures_dir``.
"""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import METHOD_ORDER, METHOD_LABELS, GENERATIVE_METHODS
from .tables import TEST_METRICS, METRIC_LABELS

print("casestudy/reporting/plots.py loaded")

# Nemenyi critical values q_{0.05} (Demšar 2006, Table 5; df=∞), indexed by #methods.
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
}


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _savefig(fig, out_path: str, also_pdf: bool = True, dpi: int = 300) -> None:
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(os.path.splitext(out_path)[0] + ".pdf", bbox_inches="tight")


def set_pub_style() -> None:
    """Apply a clean publication matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


# ── Grouped bar charts ──────────────────────────────────────────────────────
def grouped_bar_charts(
    dfs: Dict[str, pd.DataFrame],
    out_path: str,
    metrics: Optional[List[str]] = None,
    reference: str = "m10_ab8c",
) -> str:
    """One subplot per metric: x = method, grouped bars by dataset.

    Ab-8c's x-tick label is marked (``*``) for quick spotting.
    """
    import matplotlib.pyplot as plt

    metrics = metrics or TEST_METRICS
    set_pub_style()
    methods = [m for m in METHOD_ORDER]
    labels = [METHOD_LABELS[m] + (" *" if m == reference else "") for m in methods]
    datasets = list(dfs.keys())
    x = np.arange(len(methods))
    w = 0.8 / len(datasets)

    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for j, ds in enumerate(datasets):
            d = dfs[ds].set_index("method")
            vals = [float(d.loc[m, f"test_{metric}"]) if m in d.index else np.nan
                    for m in methods]
            ax.bar(x + (j - (len(datasets) - 1) / 2) * w, vals, w, label=ds)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_ylim(0, 1.0)
    axes[0].legend(ncol=len(datasets), fontsize=8, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    fig.suptitle("Held-out test metrics by method and dataset", y=1.005)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


# ── Ranking heatmap ─────────────────────────────────────────────────────────
def _rank_matrix(dfs: Dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    """Per-(method × dataset) rank (1 = best) for one metric."""
    from scipy.stats import rankdata

    methods = [m for m in METHOD_ORDER]
    data = {}
    for ds, df in dfs.items():
        d = df.set_index("method")
        vals = np.array([float(d.loc[m, f"test_{metric}"]) for m in methods])
        data[ds] = rankdata(-vals, method="average")
    out = pd.DataFrame(data, index=[METHOD_LABELS[m] for m in methods])
    return out


def ranking_heatmap(
    dfs: Dict[str, pd.DataFrame],
    out_path: str,
    metric: str = "macro_f2",
) -> str:
    """Heatmap of method × dataset ranks (1 = best) for ``metric``."""
    import matplotlib.pyplot as plt

    set_pub_style()
    rm = _rank_matrix(dfs, metric)
    fig, ax = plt.subplots(figsize=(1.4 * rm.shape[1] + 2.5, 0.42 * rm.shape[0] + 1))
    im = ax.imshow(rm.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(rm.shape[1]))
    ax.set_xticklabels(rm.columns, rotation=20, ha="right")
    ax.set_yticks(range(rm.shape[0]))
    ax.set_yticklabels(rm.index)
    for i in range(rm.shape[0]):
        for j in range(rm.shape[1]):
            ax.text(j, i, f"{rm.values[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title(f"Method ranks by dataset — {METRIC_LABELS.get(metric, metric)} (1 = best)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="rank")
    ax.grid(False)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


# ── Critical-difference (Nemenyi) diagram ───────────────────────────────────
def cd_diagram(
    avg_ranks: Dict[str, float],
    n_blocks: int,
    out_path: str,
    alpha: float = 0.05,
    title: str = "Critical-difference diagram (Nemenyi, α=0.05)",
) -> str:
    """Horizontal CD diagram; ``avg_ranks`` maps method label → mean rank (lower=better)."""
    import matplotlib.pyplot as plt

    set_pub_style()
    k = len(avg_ranks)
    q = _NEMENYI_Q05.get(k, max(_NEMENYI_Q05.values()))
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n_blocks))

    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])  # best (low) first
    ranks = [r for _, r in items]
    lo, hi = int(np.floor(min(ranks))), int(np.ceil(max(ranks)))

    fig, ax = plt.subplots(figsize=(9, 0.34 * k + 2.2))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()  # best rank on the right
    for r in range(lo, hi + 1):
        ax.plot([r, r], [k + 1, k + 1.15], color="black", lw=0.8)
        ax.text(r, k + 1.3, str(r), ha="center", va="bottom", fontsize=8)
    ax.plot([lo, hi], [k + 1, k + 1], color="black", lw=1.0)
    for i, (name, r) in enumerate(items):
        y = k - i
        ax.plot([r, r], [y, k + 1], color="0.5", lw=0.7)
        ax.plot(r, y, "o", color="tab:blue", ms=4)
        ax.text(lo, y, f"{name}  ({r:.2f})", ha="left", va="center", fontsize=8)
    ax.plot([hi, hi - cd], [k + 1.7, k + 1.7], color="tab:red", lw=2.0)
    ax.text((hi + hi - cd) / 2, k + 1.85, f"CD = {cd:.2f}", ha="center",
            va="bottom", color="tab:red", fontsize=8)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


# ── Per-class FID bars ──────────────────────────────────────────────────────
def _parse_fid_per_class(cell) -> Dict[int, float]:
    """Parse the ``fid_per_class`` CSV cell (a dict repr or NaN) → ``{class: fid}``."""
    if isinstance(cell, dict):
        return {int(k): float(v) for k, v in cell.items()}
    if not isinstance(cell, str) or not cell.strip():
        return {}
    try:
        d = ast.literal_eval(cell)
        return {int(k): float(v) for k, v in d.items()}
    except (ValueError, SyntaxError):
        return {}


def per_class_fid_bars(dfs: Dict[str, pd.DataFrame], out_path: str) -> str:
    """Grouped bars of per-class FID for each generative method, one panel/dataset."""
    import matplotlib.pyplot as plt

    set_pub_style()
    datasets = list(dfs.keys())
    gen_methods = [m for m in METHOD_ORDER if m in GENERATIVE_METHODS]
    labels = [METHOD_LABELS[m] for m in gen_methods]

    n = len(datasets)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        d = dfs[ds].set_index("method")
        per_class = {m: _parse_fid_per_class(d.loc[m, "fid_per_class"]) if m in d.index else {}
                     for m in gen_methods}
        classes = sorted({c for pc in per_class.values() for c in pc})
        x = np.arange(len(gen_methods))
        w = 0.8 / max(1, len(classes))
        for j, c in enumerate(classes):
            vals = [per_class[m].get(c, np.nan) for m in gen_methods]
            ax.bar(x + (j - (len(classes) - 1) / 2) * w, vals, w, label=f"class {c}")
        ax.set_title(ds)
        ax.set_ylabel("FID")
        ax.legend(fontsize=8, ncol=len(classes))
    axes[-1].set_xticks(np.arange(len(gen_methods)))
    axes[-1].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    fig.suptitle("Per-class FID by generative method", y=1.005)
    fig.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)
    return out_path


# ── Real vs. synthetic sample grid ──────────────────────────────────────────
def sample_grid(
    real_by_class: Dict[int, "np.ndarray"],
    fake_by_class: Dict[int, "np.ndarray"],
    out_path: str,
    n_per_class: int = 8,
    title: str = "",
) -> str:
    """Grid of real (top) vs. synthetic (bottom) minority samples per class.

    ``*_by_class`` map class → image tensor/array ``[N, C, H, W]`` in ``[-1, 1]``.
    """
    import matplotlib.pyplot as plt

    def _to_img(t):
        a = np.asarray(t)
        a = (a + 1.0) / 2.0
        a = np.clip(a, 0, 1)
        if a.shape[0] == 1:
            return a[0]
        return np.transpose(a, (1, 2, 0))

    classes = sorted(real_by_class.keys())
    rows = 2 * len(classes)
    fig, axes = plt.subplots(rows, n_per_class, figsize=(n_per_class, rows))
    if rows == 1:
        axes = axes[None, :]
    for ci, c in enumerate(classes):
        for kind, bank in ((0, real_by_class[c]), (1, fake_by_class.get(c))):
            r = 2 * ci + kind
            for j in range(n_per_class):
                ax = axes[r, j]
                ax.axis("off")
                if bank is not None and j < len(bank):
                    img = _to_img(bank[j])
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                if j == 0:
                    ax.text(-0.35, 0.5, f"c{c} {'real' if kind == 0 else 'syn'}",
                            transform=ax.transAxes, ha="right", va="center", fontsize=8)
    if title:
        fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_path, also_pdf=False)
    plt.close(fig)
    return out_path
