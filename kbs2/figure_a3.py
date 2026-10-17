"""kbs2.figure_a3 — Figure A3: the ten seeds behind every mean of Table 6, one panel per dataset.

    python -m kbs2.figure_a3

Same per-seed rows as Table 6 and Table A4 (kbs2.table6_stats.table6_rows), so every plotted mean equals Table 6.
Design of the submitted seven-method figure (_export_figures.py, fig_casestudy_combined): one panel per dataset,
small points = seeds, marker and whiskers = mean +/- one SD, the submitted Okabe-Ito colours and markers for the
seven submitted methods (authors, 17 Oct 2026: keep the old palette). The three simple remedies of Table 6 get
dark neutral colours with open markers not used by the other seven. Methods in Table 6 order.
Writes saved_kbs2/analysis/figure_a3/casestudy_combined_metrics.png and copies it into the manuscript figures.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402
from kbs2.table6_stats import table6_rows  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "figure_a3")
PAPER = os.path.join(ROOT, "Journal_paper_02_kbs_revise2", "figures", "final", "casestudy_combined_metrics.png")
DESC = os.path.join(ROOT, "saved_kbs2", "analysis", "E4_table6", "table6_desc.csv")

# (method, label, colour, marker, open marker) in Table 6 order; the seven submitted entries as submitted
METHODS = [
    ("m11_imbalanced_naive", "Imbalanced-naive", "#999999", "o", False),
    ("n01_class_weighted_isqrt", "Class-weighted", "#000000", "p", True),
    ("n02_ros_unweighted", "Random oversampling", "#5A5A5A", "h", True),
    ("n08_stdaug_budget_zc32", "Std. augmentation", "#8B6F47", "X", True),
    ("m04_dcgan", "DCGAN", "#0072B2", "s", False),
    ("m05_cgan", "cGAN", "#009E73", "^", False),
    ("m06_acgan", "ACGAN", "#CC79A7", "v", False),
    ("m07_wgangp", "WGAN-GP", "#56B4E9", "<", False),
    ("m08_bagan", "BAGAN", "#E69F00", ">", False),
    ("m10_ab8c", "Proposed", "#D55E00", "D", False),
]
METRICS = [("macro_recall", "Recall"), ("gmean", "G-mean"), ("macro_f2", "F2"),
           ("macro_specificity", "Specificity")]
TITLES = {"fashion_mnist": "Fashion-MNIST ($\\rho = 0.05$)", "cifar10": "CIFAR-10 ($\\rho = 0.30$)",
          "stl10": "STL-10 ($\\rho = 0.10$)"}
TEXTWIDTH_IN = 5.4


def main() -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 9, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
                         "grid.linewidth": 0.5, "axes.axisbelow": True})
    os.makedirs(OUT, exist_ok=True)
    desc = pd.read_csv(DESC)

    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH_IN, 8.6), sharex=True)
    x = np.arange(len(METRICS))
    step = 0.083                       # submitted 0.118 for seven methods; same group width for ten
    rng = np.random.default_rng(0)
    handles = []
    for panel, (ax, (ds, (rid, _r))) in enumerate(zip(axes, P.ANCHORS.items())):
        rows = table6_rows(rid)
        for i, (m, label, col, mk, hollow) in enumerate(METHODS):
            sub = rows[rows["method"] == m]
            assert len(sub) == len(P.SEEDS), (ds, m, len(sub))
            off = (i - (len(METHODS) - 1) / 2) * step
            for j, (c, _) in enumerate(METRICS):
                v = sub[c].to_numpy(dtype=float)
                ref = desc[(desc.block == ds) & (desc.method == m)][c + "_mean"].item()
                assert abs(v.mean() - ref) < 1e-12, (ds, m, c, v.mean(), ref)     # equals Table 6
                xx = j + off
                ax.scatter(np.full_like(v, xx) + rng.uniform(-step * 0.20, step * 0.20, v.size), v, s=3.0,
                           color=col, alpha=0.55, linewidths=0, zorder=2)
                ax.errorbar(xx, v.mean(), yerr=v.std(ddof=1), color=col, marker=mk, markersize=3.4,
                            elinewidth=0.9, capsize=1.6, capthick=0.9, linestyle="none", markeredgewidth=0.6,
                            markerfacecolor=("white" if hollow else col), zorder=3)
            if panel == 0:
                handles.append(plt.Line2D([], [], color=col, marker=mk, markersize=3.8, linestyle="none",
                                          markerfacecolor=("white" if hollow else col), label=label))
        for j in range(1, len(METRICS)):
            ax.axvline(j - 0.5, color="black", linewidth=0.4, alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in METRICS])
        ax.set_xlim(-0.5, len(METRICS) - 0.5)
        ax.set_ylabel("Score")
        ax.set_title(TITLES[ds], pad=3)
    # legend in two rows of five, read left to right in Table 6 order
    ncol = 5
    ordered = [h for c in range(ncol) for h in (handles[c:c + 1] + handles[ncol + c:ncol + c + 1])]
    fig.legend(handles=ordered, ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False,
               handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(OUT, "casestudy_combined_metrics.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    if os.path.isdir(os.path.dirname(PAPER)):  # the manuscript folder exists only in the authors' working copy
        shutil.copyfile(out, PAPER)
        print("copied to", PAPER)


if __name__ == "__main__":
    main()
