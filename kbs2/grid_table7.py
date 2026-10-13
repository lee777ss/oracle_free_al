"""kbs2.grid_table7 — Table 7 and Figure 3 over the ten methods of Table 6 (E3, R3.7).

    python -m kbs2.grid_table7            # writes saved_kbs2/analysis/E3/
    python -m kbs2.grid_table7 --seven    # the submitted seven-method figure only (check against the paper)

Sources (per cell, seeds 42-51, classifier seed 42 throughout):
  seven submitted methods  anchors: saved_casestudy/cs3j_*_reduced8.csv (Table 6 as submitted);
                           grid cells: saved_casestudy/cs3j_rho3x3_<dataset>_<regime>.csv
  three simple remedies    anchors: saved_kbs2/E2t (class-weighted, inverse sqrt), E2z (ROS, unweighted CE),
                           E2s (std. augmentation, zoom crop) - the Table 6 rows;
                           grid cells: saved_kbs2/E3

Rules fixed before the E3 results (author decisions, 11 Oct 2026): the winner of a metric in a cell is the
method with the highest mean rounded to 3 decimals, as Table 6 prints it; a tie names every tied method;
names as in Table 6; the proposed method's wins are set in bold.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "E3")
METRICS = [("macro_recall", "Recall"), ("gmean", "G-mean"), ("macro_f2", "F2"),
           ("macro_specificity", "Specificity")]
# Table 6 order and names
ORDER = ["m11_imbalanced_naive", "n01_class_weighted", "n02_ros", "n08_stdaug_budget_zc32",
         "m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan", "m10_ab8c"]
NAME = {"m11_imbalanced_naive": "Imbalanced-naive", "n01_class_weighted": "Class-weighted",
        "n02_ros": "Random oversampling", "n08_stdaug_budget_zc32": "Std. augmentation",
        "m04_dcgan": "DCGAN", "m05_cgan": "cGAN", "m06_acgan": "ACGAN", "m07_wgangp": "WGAN-GP",
        "m08_bagan": "BAGAN", "m10_ab8c": "Proposed"}
SEVEN = [m for m in ORDER if m.startswith("m")]
REMEDIES = {"n01_class_weighted": ("E2t", "W"), "n02_ros": ("E2z", "U"), "n08_stdaug_budget_zc32": ("E2s", "W")}
DS_LABEL = {"fashion_mnist": "Fashion-MNIST", "cifar10": "CIFAR-10", "stl10": "STL-10"}
ANCHOR_CSV_DS = {"R1_confusable": "fashion_mnist", "C2_cifar_paper": "cifar10", "S2_stl_mod": "stl10"}
TABLE7_RHO = {"fashion_mnist": [0.05, 0.10, 0.20], "cifar10": [0.10, 0.20, 0.30], "stl10": [0.10, 0.20, 0.30]}
MAJORITY_CAP = {"fashion_mnist": 800, "cifar10": 1000, "stl10": 400}


def _rho_of(rid: str) -> float:
    if rid in P.ANCHOR_RHO:
        return P.ANCHOR_RHO[rid]
    for cells in P.GRID_SWEEP.values():
        for r, rho in cells:
            if r == rid:
                return rho
    raise KeyError(rid)


def load(include_remedies: bool = True) -> pd.DataFrame:
    """One row per (dataset, rho, method, seed) for the nine cells of Table 7."""
    cols = ["seed", "method"] + [m for m, _ in METRICS]
    frames = []
    for rid, csv in P.REFERENCE_ANCHOR_CSV.items():
        d = pd.read_csv(os.path.join(ROOT, csv))
        d = d[d["method"].isin(SEVEN) & (d["error"].isna() | (d["error"].astype(str) == ""))][cols]
        frames.append(d.assign(dataset=ANCHOR_CSV_DS[rid], rho=_rho_of(rid), regime=rid))
    for ds, rids in P.TABLE7_GRID_CELLS.items():
        for rid in rids:
            d = pd.read_csv(os.path.join(ROOT, P.reference_grid_csv(ds, rid)))
            d = d[d["method"].isin(SEVEN) & (d["error"].isna() | (d["error"].astype(str) == ""))][cols]
            frames.append(d.assign(dataset=ds, rho=_rho_of(rid), regime=rid))
    if include_remedies:
        for m, (exp, arm) in REMEDIES.items():
            d = pd.read_csv(os.path.join(ROOT, "saved_kbs2", exp, "runs.csv"))
            d = d[(d["method"] == m) & (d["arm"] == arm) & (d["clf_seed"] == 42)]
            frames.append(d.assign(rho=d["regime"].map(_rho_of))[cols + ["dataset", "rho", "regime"]])
        e3 = pd.read_csv(os.path.join(ROOT, "saved_kbs2", "E3", "runs.csv"))
        for m, (_, arm) in REMEDIES.items():
            d = e3[(e3["method"] == m) & (e3["arm"] == arm)]
            frames.append(d.assign(rho=d["regime"].map(_rho_of))[cols + ["dataset", "rho", "regime"]])
    out = pd.concat(frames, ignore_index=True)
    out["rho"] = out["rho"].round(3)
    n = out.groupby(["dataset", "rho", "method"]).size()
    bad = n[n != 10]
    if len(bad):
        raise SystemExit(f"cells without exactly ten seeds:\n{bad}")
    return out


def winners(df: pd.DataFrame) -> pd.DataFrame:
    means = df.groupby(["dataset", "rho", "method"])[[m for m, _ in METRICS]].mean().reset_index()
    rows = []
    for (ds, rho), g in means.groupby(["dataset", "rho"]):
        rec = {"dataset": ds, "rho": rho, "n_min": int(round(rho * MAJORITY_CAP[ds]))}
        for m, h in METRICS:
            r3 = g.set_index("method")[m].round(3)
            top = r3.max()
            rec[h] = [k for k in ORDER if k in r3.index and r3[k] == top]
            rec[h + "_value"] = top
        rows.append(rec)
    return pd.DataFrame(rows), means


def tex_rows(win: pd.DataFrame) -> str:
    lines = []
    for ds in ["fashion_mnist", "cifar10", "stl10"]:
        sub = win[win["dataset"] == ds].sort_values("rho")
        for i, (_, r) in enumerate(sub.iterrows()):
            cells = []
            for _, h in METRICS:
                names = [(r"\textbf{Proposed}" if k == "m10_ab8c" else NAME[k]) for k in r[h]]
                cells.append(" / ".join(names))
            lab = DS_LABEL[ds] if i == 0 else ""
            lines.append(f"{lab} & {r['rho']:.2f} & {r['n_min']} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


# ── figure ──────────────────────────────────────────────────────────────────
def figure(df: pd.DataFrame, path: str, methods) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    tab = plt.get_cmap("tab10")
    color, marker, ls, mfc = {}, {}, {}, {}
    for i, m in enumerate(SEVEN):                      # submitted colours: tab10 in field order
        color[m] = "crimson" if m == "m10_ab8c" else tab(i % 10)
        marker[m] = "o" if m == "m10_ab8c" else "s"
        ls[m] = "-" if m == "m10_ab8c" else "--"
        mfc[m] = None
    for m, c, mk in zip(REMEDIES, ["#1a1a1a", "#555555", "#8c8c8c"], ["o", "^", "D"]):
        color[m], marker[m], ls[m], mfc[m] = c, mk, ":", "white"

    # jitter in the submitted order, remedies placed after the naive baseline as in Table 6
    plot_order = [m for m in ORDER if m in methods]
    n = len(plot_order)
    jitter = 0.055 if n <= 7 else 0.04
    fig, axes = plt.subplots(3, 4, figsize=(17.6, 11.7), squeeze=False)
    for r, ds in enumerate(["fashion_mnist", "cifar10", "stl10"]):
        rhos = TABLE7_RHO[ds]
        xs = {rho: i for i, rho in enumerate(rhos)}
        d = df[df["dataset"] == ds]
        for c, (metric, title) in enumerate(METRICS):
            ax = axes[r, c]
            for j, m in enumerate(plot_order):
                sub = d[d["method"] == m]
                off = (j - (n - 1) / 2) * jitter
                is_ab = m == "m10_ab8c"
                ax.scatter([xs[x] + off for x in sub["rho"]], sub[metric], s=(16 if is_ab else 8),
                           color=color[m], alpha=(0.55 if is_ab else 0.28), zorder=(4 if is_ab else 2),
                           linewidths=0)
                means = sub.groupby("rho")[metric].mean().reindex(rhos)
                ax.plot([xs[x] + off for x in rhos], means.values, color=color[m], marker=marker[m],
                        ms=(7 if is_ab else 4.5), lw=(2.8 if is_ab else 1.2), ls=ls[m],
                        alpha=(1.0 if is_ab else 0.8), zorder=(6 if is_ab else 3),
                        markerfacecolor=(mfc[m] or color[m]), markeredgecolor=color[m])
            ax.set_xticks(list(xs.values()))
            ax.set_xticklabels([f"{x:g}\n({int(round(x * MAJORITY_CAP[ds]))})" for x in rhos], fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(title, fontsize=12)
            ax.set_ylabel(f"{DS_LABEL[ds]}\n{title}" if c == 0 else title, fontsize=10)
            if r == 2:
                ax.set_xlabel(r"imbalance ratio $\rho$  (minority n)")
    handles = [Line2D([], [], color=color[m], marker=marker[m], ls=ls[m],
                      lw=(2.8 if m == "m10_ab8c" else 1.2), ms=(7 if m == "m10_ab8c" else 5),
                      markerfacecolor=(mfc[m] or color[m]), markeredgecolor=color[m], label=NAME[m])
               for m in plot_order]
    # two rows read left to right in Table 6 order (matplotlib fills a legend column by column)
    ncol = n if n <= 7 else (n + 1) // 2           # seven entries fit on one row
    rows = [handles[:ncol], handles[ncol:]]
    ordered = [h for c in range(ncol) for h in (rows[0][c:c + 1] + rows[1][c:c + 1])]
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(handles=ordered, loc="lower center", ncol=ncol, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seven", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.seven:
        df = load(include_remedies=False)
        figure(df, os.path.join(OUT, "sensitivity_seven_check.png"), SEVEN)
        win, means = winners(df)
        means.to_csv(os.path.join(OUT, "means_seven.csv"), index=False)
        print(win[["dataset", "rho"] + [h for _, h in METRICS]].to_string())
        return
    df = load()
    df.to_csv(os.path.join(OUT, "table7_runs.csv"), index=False)
    win, means = winners(df)
    means.to_csv(os.path.join(OUT, "table7_means.csv"), index=False)
    win.to_csv(os.path.join(OUT, "table7_winners.csv"), index=False)
    with open(os.path.join(OUT, "table7_rows.tex"), "w", encoding="utf-8") as fh:
        fh.write(tex_rows(win))
    figure(df, os.path.join(OUT, "sensitivity.png"), ORDER)
    pd.set_option("display.width", 220)
    show = win.copy()
    for _, h in METRICS:
        show[h] = show[h].map(lambda ks: " / ".join(NAME[k] for k in ks))
    print(show[["dataset", "rho", "n_min"] + [h for _, h in METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
