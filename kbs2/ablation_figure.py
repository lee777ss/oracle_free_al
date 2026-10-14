"""kbs2.ablation_figure - Figure A1 of the revision: Table 3 panel A as a bar chart (M28).

    python -m kbs2.ablation_figure      # writes Journal_paper_02_kbs_revise2/figures/final/ablation_15seeds.pdf

Authors 13 Oct 2026: regenerate Figure A1 (submitted: single seed 44, four rows) from ablation_15_seeds.csv with
the six rows and labels of Table 3 panel A; mean +- SD over seeds 42-56.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2.tables_existing import ABL_ROWS  # noqa: E402  (row order and labels of Table 3 panel A)

OUT = os.path.join(ROOT, "Journal_paper_02_kbs_revise2", "figures", "final", "ablation_15seeds.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)  # public release: create the output folder
METRICS = [("recall", "Recall"), ("gmean", "G-mean"), ("f2", "F2"), ("specificity", "Specificity")]
COLORS = ["#9e9e9e", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(os.path.join(ROOT, "ablation_15_seeds.csv"))
    d = d[d["section"].astype(str).str.startswith("seed_")]
    labels = [lab.replace(r"\rev{", "").replace("}", "") for _, lab, *_ in ABL_ROWS]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(ABL_ROWS)
    for i, ((row, *_), lab, col) in enumerate(zip(ABL_ROWS, labels, COLORS)):
        g = d[d["row"] == row]
        if len(g) != 15:
            raise SystemExit(f"{row}: {len(g)} seeds")
        m = [g[k].astype(float).mean() for k, _ in METRICS]
        s = [g[k].astype(float).std(ddof=1) for k, _ in METRICS]
        x = np.arange(len(METRICS)) + (i - (len(ABL_ROWS) - 1) / 2) * w
        ax.bar(x, m, w * 0.92, yerr=s, capsize=2, color=col, edgecolor="black", linewidth=0.4, label=lab,
               error_kw={"linewidth": 0.6})
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([n for _, n in METRICS])
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Test-set value")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
