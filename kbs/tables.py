"""kbs.tables - the generated tables, figures and quoted numbers of the revised paper (S8).

    python -m kbs.tables                     # build every table whose runs are complete; list the rest
    python -m kbs.tables --only main stats   # some tables
    python -m kbs.tables --draft             # also build incomplete tables, captioned DRAFT (layout checks)
    python -m kbs.tables --check             # report which tables are complete; write nothing
    python -m kbs.tables --list

Inputs are the per-run ``result.json`` files under saved_kbs/runs and the outputs of kbs.audit and
kbs.realism; nothing is typed by hand. A table is written only when every cell it reports has all
its seeds, so no table averages over different seeds for different methods. Outputs, in
Journal_paper_02_kbs_revise/src/generated/:

* ``tbl_<name>.tex`` - the LaTeX table, and ``tbl_<name>.csv`` with the values at full precision;
* ``numbers.tex`` - ``\\kbsnum{key}`` for every number quoted in the prose (an unknown key prints
  ??key?? in bold and warns), assembled from ``numbers.json`` so tables can be rebuilt one at a time.

Figures go to Journal_paper_02_kbs_revise/figures/final/ as 300 dpi PNG.
"""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kbs import PAPER_DIR
from casestudy.eval.stats_v2 import compare, friedman, hodges_lehmann, holm, wilcoxon
from casestudy.protocol_v2 import FIELD, LABELS, REGIMES_V2, make_config, regime_spec, save_root
from kbs.describe_splits import NAMES
from kbs.jobs import (ANCHORS, ARMS, DEV_SEEDS, DOMAIN, GRID, RATIO_METHODS, RATIOS, REUSE_METHODS, SEEDS10,
                      SEEDS20, SWEEPS_FASHION, SWEEPS_OTHER, frozen)

PAPER = PAPER_DIR
OUT = PAPER / "src" / "generated"
FIGS = PAPER / "figures" / "final"
METRICS = [("macro_f2", "Macro-F2"), ("gmean", "G-mean"), ("macro_recall", "Macro-recall"),
           ("macro_specificity", "Macro-specificity")]
DS_ORDER = ["fashion_mnist", "cifar10", "stl10", "octmnist", "gtsrb"]
KEY = {"fashion_mnist": "fashion", "cifar10": "cifar", "stl10": "stl", "octmnist": "oct", "gtsrb": "gtsrb"}
BASELINES = {"b0_plain", "b1_weighted"}
WRITE = True        # False under --check: tables and figures are computed but not written


class Incomplete(Exception):
    """A table's runs are not all finished."""


# ---------------------------------------------------------------------------- data
@lru_cache(maxsize=None)
def load(stage: str) -> pd.DataFrame:
    """One row per (finished run, evaluation split); ``primary`` marks the split the paper reports."""
    rows = []
    root = save_root() / "runs" / stage
    for f in sorted(root.rglob("result.json")) if root.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        base = {k: r.get(k) for k in ("stage", "regime", "dataset", "rho", "method", "variant", "seed", "n_synth",
                                      "n_synth_target", "ratio", "peak_mem_mb")}
        base.update(shortfall=sum((r.get("shortfall") or {}).values()), timings=r.get("timings_s", {}),
                    method_timings=r.get("method_timings_s", {}), params=r.get("params", {}), gate=r.get("gate", {}),
                    audit=r.get("audit", {}), run_dir=str(f.parent))
        for split, m in r["metrics"].items():
            rows.append({**base, "split": split, "primary": split == r["primary"], **m})
    if not rows:                    # no finished run yet: an empty frame that still has the columns
        cols = ["stage", "regime", "dataset", "rho", "method", "variant", "seed", "n_synth", "n_synth_target", "ratio",
                "peak_mem_mb", "shortfall", "timings", "method_timings", "params", "gate", "audit", "run_dir", "split",
                "primary", *(m for m, _ in METRICS), "per_class", "confusion"]
        return pd.DataFrame({c: pd.Series(dtype=bool if c == "primary" else object) for c in cols})
    return pd.DataFrame(rows)


def primary(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["primary"]] if not df.empty else df


def ordered(regimes) -> List[str]:
    return sorted(regimes, key=lambda r: (DS_ORDER.index(REGIMES_V2[r]["dataset"]), REGIMES_V2[r]["rho"]))


def expect(df: pd.DataFrame, cells: List[Tuple], seeds, what: str, draft: bool,
           cols=("regime", "method", "variant")) -> None:
    """Raise Incomplete unless every (cell, seed) is present; ``cells`` are tuples over ``cols``."""
    have = set(map(tuple, df[list(cols) + ["seed"]].itertuples(index=False, name=None))) if not df.empty else set()
    missing = [(*c, s) for c in cells for s in seeds if (*c, s) not in have]
    if missing and not draft:
        total = len(cells) * len(seeds)
        raise Incomplete(f"{what}: {len(missing)} of {total} runs missing, e.g. {missing[:3]}")


def wide(df: pd.DataFrame, col: str = "method", value: str = "macro_f2", complete: bool = True) -> pd.DataFrame:
    w = df.pivot_table(index="seed", columns=col, values=value, aggfunc="first")
    return w.dropna() if complete else w


def class_names(regime: str) -> List[str]:
    return list(regime_spec(regime, _cfg()).class_names)


@lru_cache(maxsize=None)
def _cfg():
    return make_config()


# ---------------------------------------------------------------------------- formatting
def tex(s: str) -> str:
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def label(m: str) -> str:
    return tex(LABELS.get(m, m))


def fp(p: float) -> str:
    """A p-value to three significant figures."""
    if p is None or p != p:
        return "--"
    if p < 1e-3:
        mant, ex = f"{p:.2e}".split("e")
        return f"${mant}\\times10^{{{int(ex)}}}$"
    return f"{p:.3g}"


def pm(mu: float, sd: float, bold: bool = False, nd: int = 3) -> str:
    if mu != mu:
        return "--"
    m = f"{mu:.{nd}f}"
    return (f"\\textbf{{{m}}}" if bold else m) + (f"\\,{{\\scriptsize$\\pm${sd:.{nd}f}}}" if sd == sd else "")


def signed(v: float, nd: int = 3) -> str:
    return "--" if v != v else f"{v:+.{nd}f}".replace("-", "$-$")


def pct(v: float) -> str:
    """A share as a whole percentage ('$<$1\\%' for a nonzero share that would round to 0)."""
    if v is None or v != v:
        return "--"
    return "$<$1\\%" if 0 < v < 0.005 else f"{100 * v:.0f}\\%"


def table_env(name: str, caption: str, colspec: str, header: List[str], body: List[str], scale: float = 0.62,
              draft: bool = False, star: bool = False) -> str:
    env = "table*" if star else "table"
    cap = (r"\textbf{DRAFT - incomplete runs.} " if draft else "") + caption
    # the house style of the paper (scale 0.62), shrunk further only if the table is still wider than the text
    return "\n".join([f"% Generated by kbs/tables.py ({name}) - do not edit by hand.",
                      r"\providecommand{\revtable}{}  % the manuscript colours tables added in the revision",
                      f"\\begin{{{env}}}[htbp]", r"\revtable", r"\centering", f"\\caption{{{cap}}}",
                      f"\\label{{tbl:{name}}}",
                      r"\resizebox{\ifdim\width>\linewidth\linewidth\else\width\fi}{!}{" + f"\\scalebox{{{scale}}}{{",
                      f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule", *body, r"\bottomrule",
                      r"\end{tabular}}}", f"\\end{{{env}}}", ""])


def _save(fig, name: str) -> None:
    """Write a figure as a 300 dpi PNG (the journal takes no PDF figures); nothing under --check."""
    import matplotlib.pyplot as plt

    if WRITE:
        FIGS.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGS / name, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------- shared analyses
def field_block(g: pd.DataFrame, methods: List[str], seeds) -> Tuple[List[dict], dict]:
    """Per method: mean and sd of each metric, Friedman rank, and the proposed against it (Holm)."""
    w = {m: wide(g, value=m).reindex(columns=[x for x in methods if x in g.method.unique()]) for m, _ in METRICS}
    f2 = w["macro_f2"].dropna()
    info = {"n_seeds": len(f2)}
    ranks = f2.rank(axis=1, ascending=False).mean() if len(f2) else pd.Series(dtype=float)
    cmp = None
    if "p_proposed" in f2 and len(f2) >= 5 and f2.shape[1] > 1:
        cmp = compare(f2, "p_proposed").set_index("rival")
    if len(f2) >= 2 and f2.shape[1] >= 3:
        info["friedman"] = friedman(f2)
    rows = []
    for meth in f2.columns:
        row = {"method": meth, "rank": float(ranks.get(meth, np.nan))}
        for m, _ in METRICS:
            row[m], row[m + "_sd"] = float(w[m][meth].mean()), float(w[m][meth].std(ddof=1))
        if cmp is not None and meth in cmp.index:
            c = cmp.loc[meth]
            row.update(p=float(c.p), p_holm=float(c.p_holm), verdict=c.verdict, hl=float(c.hl_diff),
                       ci_low=float(c.ci_low), ci_high=float(c.ci_high), W=float(c.W), r_rb=float(c.r_rb),
                       mean_diff=float(c.mean_diff))
        rows.append(row)
    return rows, info


def proposed_summary(rows: List[dict]) -> dict:
    """Rank of the proposed by mean macro-F2, the best rival, and the counts of significant results."""
    df = pd.DataFrame(rows).set_index("method")
    order = df["macro_f2"].sort_values(ascending=False)
    rivals = df.drop(index="p_proposed", errors="ignore")
    out = {"rank_of_mean": int(list(order.index).index("p_proposed")) + 1 if "p_proposed" in df.index else None,
           "n_methods": len(df), "best_rival": rivals["macro_f2"].idxmax() if len(rivals) else None}
    if "verdict" in rivals:
        out.update(n_better=int((rivals.verdict == "better").sum()), n_worse=int((rivals.verdict == "worse").sum()),
                   n_inconclusive=int((rivals.verdict == "inconclusive").sum()))
    return out


# ---------------------------------------------------------------------------- tables
def _field_table(name: str, stage: str, regimes: List[str], seeds, caption: str, draft: bool):
    df = primary(load(stage))
    df = df[df.variant.isin(["", "frozen"]) & df.regime.isin(regimes)]
    cells = [(r, m, "frozen" if m == "p_proposed" else "") for r in regimes for m in FIELD]
    expect(df, cells, seeds, f"tbl:{name}", draft)
    header = [r"Method & " + " & ".join(n for _, n in METRICS) + r" & Mean rank & Proposed vs.\ method (Holm $p$) \\"]
    body, csv, nums = [], [], {}
    for regime in ordered(regimes):
        g = df[df.regime == regime]
        if g.empty:
            continue
        ds = REGIMES_V2[regime]["dataset"]
        rows, info = field_block(g, FIELD, seeds)
        fr = info.get("friedman")
        head = f"{NAMES[ds]}, $\\rho = {REGIMES_V2[regime]['rho']:g}$ ({info['n_seeds']} seeds"
        head += (f"; Friedman $\\chi^2 = {fr[0]:.1f}$, $p$ = {fp(fr[1])}, Kendall's $W = {fr[2]:.2f}$)" if fr else ")")
        body.append(f"\\multicolumn{{{len(METRICS) + 3}}}{{l}}{{\\textit{{{head}}}}} \\\\")
        best = {m: max(r[m] for r in rows) for m, _ in METRICS}
        for r in rows:
            cells_ = [pm(r[m], r[m + "_sd"], bold=math.isclose(r[m], best[m])) for m, _ in METRICS]
            vs = "--" if r["method"] == "p_proposed" or "verdict" not in r else \
                {"better": "better", "worse": "worse", "inconclusive": "n.s."}[r["verdict"]] + f" ({fp(r['p_holm'])})"
            name_ = f"\\textbf{{{label(r['method'])}}}" if r["method"] == "p_proposed" else label(r["method"])
            body.append(f"{name_} & " + " & ".join(cells_) + f" & {r['rank']:.1f} & {vs} \\\\")
            csv.append({"regime": regime, "dataset": ds, **r})
        body.append(r"\addlinespace")
        s = proposed_summary(rows)
        k = f"{name}-{regime.lower().replace('_', '')}"
        nums.update({f"{k}-rank": s["rank_of_mean"], f"{k}-nbetter": s.get("n_better"), f"{k}-nworse": s.get("n_worse"),
                     f"{k}-bestrival": label(s["best_rival"]) if s["best_rival"] else None,
                     f"{k}-f2": f"{next(r['macro_f2'] for r in rows if r['method'] == 'p_proposed'):.3f}"
                     if any(r["method"] == "p_proposed" for r in rows) else None,
                     f"{k}-nseeds": info["n_seeds"]})
    if body and body[-1] == r"\addlinespace":
        body.pop()
    tex_ = table_env(name, caption, "l" + "c" * (len(METRICS) + 2), header, body, draft=draft, star=True)
    return tex_, pd.DataFrame(csv), nums


def t_main(draft: bool):
    cap = (r"Test results of all fifteen methods on the three benchmark regimes: mean $\pm$ standard deviation over "
           r"20 seeds; the seed fixes the pool, the hold-out split, the augmenter and the classifier's initial weights. "
           r"Best mean in bold. Mean rank: average rank of macro-F2 over seeds (1 = best). Last column: two-sided "
           r"Wilcoxon signed-rank test of the proposed method against each other method on macro-F2, Holm-adjusted "
           r"over the fourteen comparisons; n.s.\ = not significant at 0.05. Full statistics in Table~\ref{tbl:stats}.")
    return _field_table("main", "E3_anchors", ANCHORS, SEEDS20, cap, draft)


def t_domain(draft: bool):
    cap = (r"Results on the two domain datasets, each at two imbalance ratios (20 seeds; layout as in "
           r"Table~\ref{tbl:main}). OCTMNIST is scored on its official validation and test images together, with "
           r"metrics re-weighted to the class prior of the regime, because its standard test split holds only 12 or 25 "
           r"images per minority class.")
    return _field_table("domain", "E6_domain", DOMAIN, SEEDS20, cap, draft)


def t_stats(draft: bool):
    """R1.7: every pairwise test of the proposed method, anchors and domain regimes."""
    header = [r"Method & $n$ & Mean diff. & HL diff.\ [95\% CI] & $W$ & $p$ & Holm $p$ & $r_{rb}$ & Verdict \\"]
    body, csv, nums = [], [], {}
    for stage, regimes, seeds in (("E3_anchors", ANCHORS, SEEDS20), ("E6_domain", DOMAIN, SEEDS20)):
        df = primary(load(stage))
        df = df[df.variant.isin(["", "frozen"]) & df.regime.isin(regimes)]
        cells = [(r, m, "frozen" if m == "p_proposed" else "") for r in regimes for m in FIELD]
        expect(df, cells, seeds, f"tbl:stats ({stage})", draft)
        for regime in ordered(regimes):
            g = df[df.regime == regime]
            if g.empty:
                continue
            rows, info = field_block(g, FIELD, seeds)
            ds = REGIMES_V2[regime]["dataset"]
            fr = info.get("friedman")
            head = f"{NAMES[ds]}, $\\rho = {REGIMES_V2[regime]['rho']:g}$"
            if fr:
                head += f": Friedman $\\chi^2 = {fr[0]:.2f}$, $p$ = {fp(fr[1])}, Kendall's $W = {fr[2]:.3f}$"
            body.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{head}}}}} \\\\")
            for r in rows:
                if r["method"] == "p_proposed" or "verdict" not in r:
                    continue
                body.append(f"{label(r['method'])} & {info['n_seeds']} & {signed(r['mean_diff'], 4)} & "
                            f"{signed(r['hl'], 4)} [{signed(r['ci_low'], 4)}, {signed(r['ci_high'], 4)}] & {r['W']:.0f} & "
                            f"{fp(r['p'])} & {fp(r['p_holm'])} & {signed(r['r_rb'], 2)} & {r['verdict']} \\\\")
                csv.append({"regime": regime, **r})
            body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Proposed method against each other method, macro-F2 (test), paired over seeds. Mean diff.\ and the "
           r"Hodges-Lehmann (HL) estimate are proposed minus method, with the exact 95\% interval of the HL estimate; "
           r"$W$ and $p$: two-sided Wilcoxon signed-rank test; Holm $p$: adjusted over the fourteen comparisons of a "
           r"regime; $r_{rb}$: matched-pairs rank-biserial correlation. Verdict at 0.05 after adjustment.")
    if csv:
        _forest_figure(pd.DataFrame(csv))
    return table_env("stats", cap, "lcccccccc", header, body, draft=draft, star=True), pd.DataFrame(csv), nums


def _forest_figure(st: pd.DataFrame) -> None:
    """HL difference (proposed minus method) with its 95% interval, one panel per regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regs = [r for r in ordered(st.regime.unique())]
    ncol = min(4, len(regs))
    nrow = math.ceil(len(regs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.6 * nrow), squeeze=False, sharey=True)
    for ax, r in zip(axes.flat, regs):
        g = st[st.regime == r].set_index("method").reindex([m for m in FIELD if m != "p_proposed"]).dropna(subset=["hl"])
        y = np.arange(len(g))[::-1]
        col = ["tab:green" if v == "better" else "tab:red" if v == "worse" else "grey" for v in g.verdict]
        ax.errorbar(g.hl, y, xerr=[g.hl - g.ci_low, g.ci_high - g.hl], fmt="none", ecolor=col, lw=1)
        ax.scatter(g.hl, y, c=col, s=12, zorder=3)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y, [LABELS[m] for m in g.index], fontsize=6)
        ax.set_title(f"{NAMES[REGIMES_V2[r]['dataset']]}, rho = {REGIMES_V2[r]['rho']:g}", fontsize=8)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("macro-F2 difference, proposed minus method", fontsize=6)
    for ax in list(axes.flat)[len(regs):]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "fig_forest.png")


def t_classwise(draft: bool):
    """R1.8: per-class precision, recall and F2 on all official test images, and mean confusion matrices."""
    df = load("E3_anchors")
    df = df[(df.split == "test_full") & df.variant.isin(["", "frozen"]) & df.regime.isin(ANCHORS)]
    cells = [(r, m, "frozen" if m == "p_proposed" else "") for r in ANCHORS for m in FIELD]
    expect(df, cells, SEEDS20, "tbl:classwise", draft)
    prim = primary(load("E3_anchors"))
    header = [r"Dataset & Method & Class & Precision & Recall & F2 \\"]
    body, csv, conf, nums = [], [], {}, {}
    for regime in ordered(ANCHORS):
        g = df[df.regime == regime]
        if g.empty:
            continue
        ds, names = REGIMES_V2[regime]["dataset"], class_names(regime)
        p = prim[(prim.regime == regime) & ~prim.method.isin(BASELINES | {"p_proposed"})]
        best = p.groupby("method")["macro_f2"].mean().idxmax() if not p.empty else None
        methods = [m for m in ("b0_plain", "b1_weighted", best, "p_proposed") if m and m in g.method.unique()]
        for i, m in enumerate(methods):
            gm = g[g.method == m]
            pc = {k: np.mean([pcl[k] for pcl in gm.per_class], axis=0) for k in ("precision", "recall", "f2")}
            cm = np.mean([np.asarray(c, float) / np.asarray(c, float).sum(1, keepdims=True) for c in gm.confusion], axis=0)
            conf[(regime, m)] = cm
            for c, cname in enumerate(names):
                lead = (f"{NAMES[ds]}" if i == 0 and c == 0 else "") + " & " + \
                       ((("Best rival: " if m == best else "") + label(m)) if c == 0 else "")
                role = "majority" if c == 0 else "minority"
                body.append(f"{lead} & {tex(cname)} ({role}) & {pc['precision'][c]:.3f} & {pc['recall'][c]:.3f} & "
                            f"{pc['f2'][c]:.3f} \\\\")
                csv.append({"regime": regime, "method": m, "class": cname, **{k: pc[k][c] for k in pc}})
        body.append(r"\addlinespace")
        if "p_proposed" in methods and "b1_weighted" in methods:
            # minority recall gained over the class-weighted baseline, and the precision it cost
            pr, b1 = pd.DataFrame([x for x in csv if x["regime"] == regime and x["method"] == "p_proposed"]), \
                pd.DataFrame([x for x in csv if x["regime"] == regime and x["method"] == "b1_weighted"])
            k = f"classwise-{KEY[ds]}"
            nums[f"{k}-minrecall-gain"] = f"{(pr.recall[1:].mean() - b1.recall[1:].mean()):+.3f}"
            nums[f"{k}-minprecision-change"] = f"{(pr.precision[1:].mean() - b1.precision[1:].mean()):+.3f}"
    if body and body[-1] == r"\addlinespace":
        body.pop()
    if conf:
        _confusion_figure(conf)
    cap = (r"Per-class results on all official test images of the three classes (1,000 per class for Fashion-MNIST "
           r"and CIFAR-10, 800 for STL-10), with the confusion matrix re-weighted to the class prior of the regime; "
           r"mean over 20 seeds. Best rival: the method other than the baselines with the highest mean macro-F2.")
    return table_env("classwise", cap, "lllccc", header, body, draft=draft), pd.DataFrame(csv), nums


def _confusion_figure(conf: Dict[Tuple[str, str], np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = [r for r in ordered(ANCHORS) if any(k[0] == r for k in conf)]
    ncol = max(sum(1 for k in conf if k[0] == r) for r in regimes)
    fig, axes = plt.subplots(len(regimes), ncol, figsize=(2.3 * ncol, 2.3 * len(regimes)), squeeze=False)
    for i, r in enumerate(regimes):
        names = class_names(r)
        ms = [k[1] for k in conf if k[0] == r]
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(ms):
                ax.axis("off")
                continue
            cm = conf[(r, ms[j])]
            ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
            for a in range(cm.shape[0]):
                for b in range(cm.shape[1]):
                    ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cm[a, b] > 0.6 else "black")
            ax.set_xticks(range(len(names)), names, fontsize=6, rotation=30)
            ax.set_yticks(range(len(names)), names if j == 0 else [], fontsize=6)
            ax.set_title(LABELS.get(ms[j], ms[j]).replace(" (", "\n("), fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{NAMES[REGIMES_V2[r]['dataset']]}\ntrue class", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_confusion.png")


def t_grid(draft: bool):
    """R3.7: the full imbalance grid, 15 methods per cell, 10 seeds (seeds 42-51 of the anchors)."""
    a = primary(load("E3_anchors"))
    a = a[a.seed.isin(SEEDS10)]
    g_ = primary(load("E7_grid"))
    df = pd.concat([a, g_], ignore_index=True)
    df = df[df.variant.isin(["", "frozen"])]
    cells = [(r, m, "frozen" if m == "p_proposed" else "") for r in ANCHORS + GRID for m in FIELD]
    expect(df, cells, SEEDS10, "tbl:rho", draft)
    header = [r"Dataset & $\rho$ & Proposed & Best rival & Rank & Better / n.s.\ / worse & Leads \\"]
    body, csv, nums = [], [], {}
    leads = leads_sig = n_cells = 0
    for ds in ("fashion_mnist", "cifar10", "stl10"):
        regs = [r for r in ordered(ANCHORS + GRID) if REGIMES_V2[r]["dataset"] == ds]
        for r in regs:
            g = df[df.regime == r]
            if g.empty:
                continue
            rows, info = field_block(g, FIELD, SEEDS10)
            s = proposed_summary(rows)
            if s["rank_of_mean"] is None:
                continue
            by = {x["method"]: x for x in rows}
            br = s["best_rival"]
            lead = s["rank_of_mean"] == 1
            lead_sig = lead and s.get("n_better", 0) == len(rows) - 1
            leads += lead
            leads_sig += lead_sig
            n_cells += 1
            anchor = " (anchor)" if REGIMES_V2[r]["role"] == "anchor" else ""
            body.append(f"{NAMES[ds] if r == regs[0] else ''} & {REGIMES_V2[r]['rho']:g}{anchor} & "
                        f"{pm(by['p_proposed']['macro_f2'], by['p_proposed']['macro_f2_sd'])} & {label(br)}: "
                        f"{pm(by[br]['macro_f2'], by[br]['macro_f2_sd'])} & {s['rank_of_mean']} / {s['n_methods']} & "
                        f"{s.get('n_better', '--')} / {s.get('n_inconclusive', '--')} / {s.get('n_worse', '--')} & "
                        f"{'yes' if lead else 'no'}{' (all significant)' if lead_sig else ''} \\\\")
            csv.append({"regime": r, "dataset": ds, "rho": REGIMES_V2[r]["rho"], **s,
                        "proposed_f2": by["p_proposed"]["macro_f2"], "best_rival_f2": by[br]["macro_f2"]})
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cells_df = pd.DataFrame(csv)
    nums.update({"rho-cells": n_cells, "rho-leads": leads, "rho-leads-sig": leads_sig,
                 "rho-cells-outperformed": int((cells_df.get("n_worse", pd.Series(dtype=float)) > 0).sum()),
                 "rho-cells-beats-one": int((cells_df.get("n_better", pd.Series(dtype=float)) > 0).sum())})
    if csv:
        _rho_figure(df)
    cap = (r"The imbalance grid: five values of $\rho$ per dataset, all fifteen methods in every cell, 10 seeds "
           r"(42-51; for the anchors, the first ten of their 20). Rank: position of the proposed method by mean "
           r"macro-F2. Better / n.s.\ / worse: number of the fourteen other methods that the proposed method beats, "
           r"does not separate from, or trails (two-sided Wilcoxon, Holm-adjusted, 0.05).")
    return table_env("rho", cap, "llccccc", header, body, draft=draft), pd.DataFrame(csv), nums


def _rho_figure(df: pd.DataFrame) -> None:
    """Macro-F2 against rho per dataset: the proposed method, BAGAN-GP, the best of the rest, and the two baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, ds in zip(axes, ("fashion_mnist", "cifar10", "stl10")):
        g = df[df.dataset == ds]
        if g.empty:
            ax.axis("off")
            continue
        mean = g.groupby(["rho", "method"])["macro_f2"].mean().unstack("method")
        others = [m for m in mean.columns if m not in BASELINES | {"p_proposed", "m09_bagan_gp"}]
        best_other = mean[others].max(axis=1)
        for m, style in (("p_proposed", dict(color="tab:red", marker="o", lw=1.6)),
                         ("m09_bagan_gp", dict(color="tab:blue", marker="s", lw=1)),
                         ("b1_weighted", dict(color="black", marker="^", lw=1, ls="--")),
                         ("b0_plain", dict(color="grey", marker="v", lw=1, ls=":"))):
            if m in mean:
                ax.plot(mean.index, mean[m], ms=3, label=LABELS[m], **style)
        ax.plot(best_other.index, best_other, color="tab:green", lw=1, ls="-.", label="Best other method (per cell)")
        ax.set_title(NAMES[ds], fontsize=9)
        ax.set_xlabel("imbalance ratio rho", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("macro-F2 (test, 10 seeds)", fontsize=8)
    axes[-1].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_rho.png")


def t_random(draft: bool):
    """R3.6: what the seed controls, submitted against revised protocol."""
    rows = [("Pool: which real training images form the imbalanced pool", "fixed (42)", "seed $s$"),
            ("Hold-out: the stratified 90/10 split of the pool", "fixed (42)", "seed $s$"),
            ("Augmenter: initialisation, noise, batch order, selection", "seed $s$", "seed $s$"),
            ("Downstream classifier: initial weights", "fixed (42)", "seed $s$, identical for every method"),
            ("Downstream classifier: batch order, dropout", "fixed (42)", "seed $s$"),
            ("Test sets", "fixed", "fixed")]
    header = [r"Source of randomness & Submitted experiments & Revised experiments \\"]
    body = [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    cap = (r"What the seed controls. In the revised experiments every source except the test sets follows the seed, "
           r"identically for every method: within a seed all methods share the pool, the hold-out split and the "
           r"classifier's initial weights, which is verified for every run by stored hashes.")
    return table_env("random", cap, "lll", header, body, draft=False), pd.DataFrame(rows, columns=["source", "submitted", "revised"]), {}


def t_audit(draft: bool):
    """R3.1, R3.3: the label audit by an independent reference classifier (from kbs.audit)."""
    df = primary(load("E4_selection"))
    expect(df, [(r, "p_proposed", "gate_diverse") for r in ANCHORS], SEEDS20, "tbl:audit", draft)
    f = save_root() / "audit" / "summary.json"
    if not f.exists():
        raise Incomplete("tbl:audit: run python -m kbs.audit first")
    s = json.loads(f.read_text(encoding="utf-8"))
    fb_f = save_root() / "audit" / "audit_fallback.csv"
    fb = pd.read_csv(fb_f) if fb_f.exists() else pd.DataFrame()
    pct = lambda v: "--" if v is None or v != v else f"{100 * v:.1f}\\%"
    header = [r"Dataset & Reference accuracy & AUROC & Trend & Correct among gate-failing & Wrong among gate-passing & "
              r"Agreement: gate-passing / fallback & Assigned to conditioned / majority / other class & Candidates \\"]
    body, csv, nums = [], [], {}
    for ds in DS_ORDER:
        if ds not in s:
            continue
        a = s[ds]
        ag = {}
        if not fb.empty:
            for _, row in fb[fb.dataset == ds].iterrows():
                ag[bool(row["gate"])] = row["mean"]
        assign = " / ".join(f"{100 * a[k]:.0f}" if a.get(k) is not None else "--"
                            for k in ("a8_pred_conditioned", "a8_pred_majority", "a8_pred_other_minority")) + r"\%"
        body.append(f"{NAMES[ds]} & {pct(a.get('ref_acc'))} & {a['auroc']:.3f} & {a['trend']:.2f} & "
                    f"{pct(a['correct_among_gate_failing'])} & {pct(a['wrong_among_gate_passing'])} & "
                    f"{pct(ag.get(True))} / {pct(ag.get(False))} & {assign} & "
                    + f"{a['n_candidates']:,}".replace(",", "{,}") + r" \\")
        csv.append({"dataset": ds, **a, "agree_gate_passing": ag.get(True), "agree_fallback": ag.get(False)})
        k = f"audit-{KEY[ds]}"
        nums.update({f"{k}-refacc": pct(a.get("ref_acc")), f"{k}-auroc": f"{a['auroc']:.2f}",
                     f"{k}-correct-failing": pct(a["correct_among_gate_failing"]),
                     f"{k}-wrong-passing": pct(a["wrong_among_gate_passing"])})
    cap = (r"Label audit by an independent reference classifier (an ensemble of three ResNet-18 trained on all real "
           r"training images of the three classes; its accuracy on the real test images is the noise floor of every "
           r"rate). Candidates of the proposed rule over 20 seeds. AUROC: how well the internal classifier's confidence "
           r"in the conditioned class separates candidates the reference classifier agrees with from those it does not; "
           r"trend: rank correlation between confidence decile and agreement. The two error rates are those named by "
           r"the reviewer; the agreement column compares admitted images that passed the gate with those admitted by the "
           r"fallback. Assigned: where the internal classifier puts the candidates; a candidate assigned to another "
           r"class is one that confidence-thresholded self-training would relabel and the gate rejects.")
    return table_env("audit", cap, "lcccccccc", header, body, draft=draft, star=True), pd.DataFrame(csv), nums


STRATA = [("conf_low", "Lowest third of confidence"), ("conf_mid", "Middle third"), ("conf_high", "Highest third"),
          ("correct_low", "Accepted by the reference classifier, below the threshold"),
          ("correct_high", "Accepted by the reference classifier, above the threshold")]


def t_strata(draft: bool):
    """R3.1 (A5, A6): downstream utility of the same budget drawn from one confidence stratum of the candidates."""
    df = primary(load("E5_strata"))
    expect(df, [(r, "p_proposed", s) for r in ANCHORS for s, _ in STRATA], SEEDS20, "tbl:strata", draft)
    cp = copies_table("E5_strata", [(r, "p_proposed", s) for r in ANCHORS for s, _ in STRATA], SEEDS20, "tbl:strata",
                      draft)
    header = [r"Candidates admitted & Macro-F2 & Difference to the reference row (HL, 95\% CI) & Holm $p$ & "
              r"Images added (\% of target) & Near-copies \\"]
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        g = df[df.regime == r]
        if g.empty:
            continue
        ds = REGIMES_V2[r]["dataset"]
        w = wide(g, col="variant")
        added = g.assign(a=g.n_synth / g.n_synth_target).groupby("variant")["a"].agg(["mean", "min"])
        body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{NAMES[ds]} ({len(w)} seeds)}}}} \\\\")
        tests = {}
        for ref, rivals in (("conf_high", ["conf_low", "conf_mid"]), ("correct_high", ["correct_low"])):
            if ref in w and all(x in w for x in rivals) and len(w) >= 5:
                c = compare(w[[ref] + rivals], ref).set_index("rival")
                for x in rivals:          # difference of the row to its reference: rival minus reference
                    tests[x] = (-c.loc[x, "hl_diff"], -c.loc[x, "ci_high"], -c.loc[x, "ci_low"], c.loc[x, "p_holm"])
        for s, name in STRATA:
            if s not in w:
                continue
            ad = added.loc[s] if s in added.index else None
            ad_s = "--" if ad is None else (f"{100 * ad['mean']:.0f}" + (f" (min {100 * ad['min']:.0f})"
                                                                          if ad["min"] < 0.995 else ""))
            if s in tests:
                hl, lo, hi, p = tests[s]
                diff, p_s = f"{signed(hl)} [{signed(lo)}, {signed(hi)}]", fp(p)
            else:
                diff, p_s = ("reference" if s in ("conf_high", "correct_high") else "--"), ""
            nc = cp.get((r, s), np.nan)
            body.append(f"{name} & {pm(w[s].mean(), w[s].std(ddof=1))} & {diff} & {p_s} & {ad_s} & {pct(nc)} \\\\")
            csv.append({"regime": r, "stratum": s, "macro_f2": w[s].mean(), "sd": w[s].std(ddof=1), "near_copy": nc,
                        **({"hl": tests[s][0], "ci_low": tests[s][1], "ci_high": tests[s][2], "p_holm": tests[s][3]}
                           if s in tests else {}),
                        "added_mean": None if ad is None else ad["mean"], "added_min": None if ad is None else ad["min"]})
            if s in tests:
                nums[f"strata-{KEY[ds]}-{s.replace('_', '')}-diff"] = signed(tests[s][0])
                nums[f"strata-{KEY[ds]}-{s.replace('_', '')}-p"] = fp(tests[s][3])
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Does confidence predict utility? The proposed method's generator is frozen (co-training off) and the same "
           r"budget of images is admitted from one stratum of the candidates of each cycle; the downstream classifier "
           r"is trained as for every method (20 seeds; macro-F2 on the test set). Thirds: candidates ranked by the "
           r"internal classifier's confidence in the conditioned class. Accepted: candidates the reference classifier "
           r"of Table~\ref{tbl:audit} assigns to the conditioned class, below or above the threshold $\tau$. "
           r"Difference: Hodges-Lehmann estimate of the row minus its reference row, with its exact 95\% interval; "
           r"two-sided Wilcoxon test, Holm-adjusted within each comparison set. Images added: mean share of the target "
           r"that the stratum could fill (the minimum over seeds where it fell short). Near-copies: share of the added "
           r"images that reproduce one real training image (as in Table~\ref{tbl:realism}).")
    return table_env("strata", cap, "lccccc", header, body, draft=draft), pd.DataFrame(csv), nums


def t_labels(draft: bool):
    """R3.1 (A9): does label correctness of the added images go with utility? Every generative method, every regime of
    the main comparisons: the share of its added images the reference classifier assigns to their conditioned class."""
    from scipy.stats import spearmanr

    from kbs.realism import GENERATIVE
    f = save_root() / "audit" / "audit_generators.csv"
    if not f.exists():
        raise Incomplete("tbl:labels: run python -m kbs.audit --generators first")
    gen = pd.read_csv(f)
    regs = ordered(ANCHORS) + ordered(DOMAIN)
    have = set(zip(gen.regime, gen.method, gen.seed))
    missing = [(r, m, s) for r in regs for m in GENERATIVE for s in SEEDS20 if (r, m, s) not in have]
    if missing and not draft:
        raise Incomplete(f"tbl:labels: {len(missing)} runs without A9, e.g. {missing[:3]}")
    f2 = primary(pd.concat([load("E3_anchors"), load("E6_domain")], ignore_index=True))
    f2 = f2[f2.variant.isin(["", "frozen"])].groupby(["regime", "method"])["macro_f2"].mean()
    ag = gen.groupby(["regime", "method"])["agree"].mean()
    head = lambda r: (f"{NAMES[REGIMES_V2[r]['dataset']]}" + (f", $\\rho = {REGIMES_V2[r]['rho']:g}$" if r in DOMAIN else ""))
    header = [r"Method & " + " & ".join(head(r) for r in regs) + r" \\"]
    body, csv, nums = [], [], {}
    for m in GENERATIVE:
        cells_ = []
        for r in regs:
            a, u = ag.get((r, m), np.nan), f2.get((r, m), np.nan)
            cells_.append("--" if a != a else f"{pct(a)} / {u:.3f}")
            csv.append({"regime": r, "method": m, "agree": a, "macro_f2": u})
        body.append(f"{label(m)} & " + " & ".join(cells_) + r" \\")
    body.append(r"\addlinespace")
    rho_cells = []
    for r in regs:
        sub = pd.DataFrame({"a": [ag.get((r, m), np.nan) for m in GENERATIVE],
                            "u": [f2.get((r, m), np.nan) for m in GENERATIVE]}).dropna()
        if len(sub) >= 4:
            rho, p = spearmanr(sub.a, sub.u)
            rho_cells.append(f"{signed(rho, 2)} ($p$ = {fp(p)})")
            k = f"labels-{r.lower().replace('_', '')}"
            nums.update({f"{k}-spearman": signed(rho, 2), f"{k}-spearman-p": fp(p),
                         f"{k}-proposed": pct(ag.get((r, 'p_proposed'), np.nan)),
                         f"{k}-bagangp": pct(ag.get((r, 'm09_bagan_gp'), np.nan))})
        else:
            rho_cells.append("--")
    body.append(r"\textit{Rank correlation, agreement and macro-F2} & " + " & ".join(rho_cells) + r" \\")
    cap = (r"Label agreement of the images each generative method added, against the macro-F2 it reached (20 seeds): "
           r"the share of the added images that the reference classifier of Table~\ref{tbl:audit} assigns to the class "
           r"they were generated for, then the test macro-F2. Last row: Spearman correlation across the eight methods "
           r"between agreement and macro-F2. OCTMNIST and GTSRB are scored on their own reference classifiers "
           r"(test accuracy 0.916 and 0.987).")
    return (table_env("labels", cap, "l" + "c" * len(regs), header, body, draft=draft, star=True),
            pd.DataFrame(csv), nums)


def t_pilot(draft: bool):
    """5.1: the four configurations of the pilot on the development set."""
    df = primary(load("E2_pilot"))
    configs = [f"fallback={fb},cotrain={co}" for fb in ("top_confidence", "regenerate") for co in ("on", "off")]
    cells = [(r, "p_proposed", v) for r in ANCHORS for v in configs]
    expect(df, cells, DEV_SEEDS, "tbl:pilot", draft)
    regs = ordered(ANCHORS)
    header = [r"Fallback & Generator co-training & " + " & ".join(NAMES[REGIMES_V2[r]["dataset"]] for r in regs)
              + r" & Admitted images that fail the gate (" + " / ".join(NAMES[REGIMES_V2[r]["dataset"]] for r in regs)
              + r") & Images added (share of target) \\"]
    body, csv = [], []
    for v in configs:
        fb, co = v.split(",")[0].split("=")[1], v.split(",")[1].split("=")[1]
        g = df[df.variant == v]
        cells_, viol, added = [], [], []
        for r in regs:
            gr = g[g.regime == r]
            x = gr["macro_f2"]
            cells_.append(pm(x.mean(), x.std(ddof=1)) if len(x) else "--")
            gate = [gg for gg in gr.gate if gg]
            viol.append(sum(t.get("violating", 0) for t in gate) / max(1, sum(t.get("admitted", 0) for t in gate)))
            added.append(gr.n_synth.sum() / max(1, gr.n_synth_target.sum()))
        body.append(f"{'as submitted' if fb == 'top_confidence' else 'regenerate'} & {co} & " + " & ".join(cells_)
                    + " & " + " / ".join(f"{100 * s:.0f}" for s in viol) + r"\% & "
                    + " / ".join(f"{100 * s:.0f}" for s in added) + r"\% \\")
        csv.append({"fallback": fb, "cotrain": co, **{r: g[g.regime == r]["macro_f2"].mean() for r in regs},
                    **{f"violating_{r}": s for r, s in zip(regs, viol)}, **{f"added_{r}": s for r, s in zip(regs, added)}})
    fz = frozen()
    cap = (r"Development pilot of the proposed method (seeds 142-151, development set, macro-F2, mean $\pm$ standard "
           r"deviation). The rule fixed before the pilot adopts an alternative only if it is better on at least two "
           r"datasets and the two-sided Wilcoxon test over the 30 paired runs gives $p < 0.05$. "
           + (f"Frozen: fallback {'as submitted' if fz['fallback'] == 'top_confidence' else 'regenerate'}, "
              f"co-training {'on' if fz['cotrain_g'] else 'off'}." if fz else ""))
    return table_env("pilot", cap, "ll" + "c" * len(regs) + "cc", header, body, draft=draft), pd.DataFrame(csv), {}


def gate_records(stage: str, variant: Optional[str] = None) -> pd.DataFrame:
    """One row per (run, cycle, class) from the gate logs."""
    rows = []
    df = primary(load(stage))
    df = df[df.method == "p_proposed"]
    if variant is not None:
        df = df[df.variant == variant]
    for _, r in df.iterrows():
        f = Path(r.run_dir) / "gate.json"
        if f.exists():
            for g in json.loads(f.read_text(encoding="utf-8")):
                rows.append({"regime": r.regime, "dataset": r.dataset, "seed": r.seed, "variant": r.variant, **g})
    return pd.DataFrame(rows)


def t_fallback(draft: bool):
    """R3.3: how often the fallback fires and how many admitted images fail the gate."""
    df = primary(load("E3_anchors"))
    df = df[(df.method == "p_proposed") & df.regime.isin(ANCHORS)]
    expect(df, [(r, "p_proposed", "frozen") for r in ANCHORS], SEEDS20, "tbl:fallback", draft)
    g = gate_records("E3_anchors", "frozen")
    names = {r: class_names(r) for r in ANCHORS}
    header = [r"Dataset & Class & Class-cycles & Fallback fired & Pass rate & Admitted & Fail the gate & "
              r"of which wrong class & of which below $\tau$ \\"]
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        gr = g[g.regime == r] if not g.empty else g
        if gr.empty:
            continue
        ds = REGIMES_V2[r]["dataset"]
        for i, (c, gc) in enumerate(gr.groupby("cls")):
            adm, viol = gc.n_admitted.sum(), gc.n_violating.sum()
            body.append(f"{NAMES[ds] if i == 0 else ''} & {tex(names[r][c])} & {len(gc)} & "
                        f"{100 * gc.fell_back.mean():.0f}\\% & {100 * gc.pass_rate.mean():.1f}\\% & {adm} & "
                        f"{100 * viol / max(1, adm):.0f}\\% & {100 * gc.violating_argmax.sum() / max(1, adm):.0f}\\% & "
                        f"{100 * gc.violating_below_tau.sum() / max(1, adm):.0f}\\% \\\\")
            csv.append({"regime": r, "class": names[r][c], "class_cycles": len(gc), "fallback_rate": gc.fell_back.mean(),
                        "pass_rate": gc.pass_rate.mean(), "admitted": adm, "violating_share": viol / max(1, adm)})
        k = f"fallback-{KEY[ds]}"
        nums[f"{k}-rate"] = f"{100 * gr.fell_back.mean():.0f}"
        nums[f"{k}-violating"] = f"{100 * gr.n_violating.sum() / max(1, gr.n_admitted.sum()):.0f}"
        nums[f"{k}-passrate"] = f"{100 * gr.pass_rate.mean():.1f}"
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    if not g.empty and WRITE:
        g.to_csv(OUT / "gate_by_cycle.csv", index=False)
    fz = frozen() or {}
    cap = (r"The confidence gate in the evaluation runs (20 seeds). A class-cycle is one acquisition step for one "
           r"minority class; the gate passes a candidate whose predicted class is the conditioned class with "
           r"probability at least $\tau = 0.6$. "
           + ("When fewer than $B$ candidates pass, the batch is filled with the most confident remaining candidates "
              r"(the fallback of the submitted method), which admits images that fail the gate."
              if fz.get("fallback", "top_confidence") == "top_confidence" else
              r"When fewer than $B$ candidates pass, new candidates are drawn; the gate is never bypassed."))
    return table_env("fallback", cap, "llccccccc", header, body, draft=draft), pd.DataFrame(csv), nums


def t_selection(draft: bool):
    """R1.6, R3.2: eight selection rules on one frozen generator, same candidates and budget."""
    df = primary(load("E4_selection"))
    expect(df, [(r, "p_proposed", a) for r in ANCHORS for a in ARMS], SEEDS20, "tbl:selection", draft)
    div_f = save_root() / "audit" / "audit_diversity.csv"
    div = pd.read_csv(div_f) if div_f.exists() else pd.DataFrame()
    dv = (div.groupby(["dataset", "arm", "seed"])["diversity"].mean().groupby(["dataset", "arm"]).mean()
          if not div.empty else pd.Series(dtype=float))
    cp = copies_table("E4_selection", [(r, "p_proposed", a) for r in ANCHORS for a in ARMS], SEEDS20, "tbl:selection", draft)
    header = [r"Selection rule & Macro-F2 & Proposed rule vs.\ this rule (Holm $p$) & Agreement & Diversity & Near-copies \\"]
    names = {"random": "Random", "margin": "Smallest margin (uncertainty)", "badge": "BADGE (uncertainty + diversity)",
             "agree_margin": "Margin among gate-passing", "top_confidence": "Highest confidence",
             "gate_random": "Gate, then random", "diversity_only": "Diversity only (k-means++)",
             "gate_diverse": "Gate, then diversity (proposed rule)"}
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        g = df[df.regime == r]
        if g.empty:
            continue
        ds = REGIMES_V2[r]["dataset"]
        k = f"selection-{KEY[ds]}"
        w = wide(g, col="variant")
        c = compare(w, "gate_diverse").set_index("rival") if "gate_diverse" in w and len(w) >= 5 else None
        agree = g.assign(a=[x.get("agree_admitted", np.nan) for x in g.audit]).groupby("variant")["a"].mean()
        body.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{NAMES[ds]} ({len(w)} seeds)}}}} \\\\")
        for a in ARMS:
            if a not in w:
                continue
            vs = "--" if a == "gate_diverse" or c is None else \
                {"better": "better", "worse": "worse", "inconclusive": "n.s."}[c.loc[a, "verdict"]] + f" ({fp(c.loc[a, 'p_holm'])})"
            ag, dvv, nc = agree.get(a, np.nan), dv.get((ds, a), np.nan), cp.get((r, a), np.nan)
            body.append(f"{names[a]} & {pm(w[a].mean(), w[a].std(ddof=1))} & {vs} & "
                        f"{'--' if ag != ag else f'{ag:.3f}'} & {'--' if dvv != dvv else f'{dvv:.3f}'} & {pct(nc)} \\\\")
            csv.append({"regime": r, "arm": a, "macro_f2": w[a].mean(), "sd": w[a].std(ddof=1), "agree_admitted": ag,
                        "diversity": dvv, "near_copy": nc,
                        **({"verdict_vs_rule": c.loc[a, "verdict"], "p_holm": c.loc[a, "p_holm"]}
                           if c is not None and a in c.index else {})})
        # 2 x 2 factorial: gate {no, yes} x diversity {no, yes}; each effect averaged over the other factor
        for eff, (on1, off1, on2, off2) in (("gate", ("gate_random", "random", "gate_diverse", "diversity_only")),
                                            ("diversity", ("diversity_only", "random", "gate_diverse", "gate_random"))):
            if not {on1, off1, on2, off2} <= set(w.columns):
                continue
            d = ((w[on1] - w[off1]) + (w[on2] - w[off2])) / 2
            _, p = wilcoxon(d)
            hl, lo, hi = hodges_lehmann(d)
            body.append(f"\\quad main effect of the {eff} (HL, 95\\% CI) & {signed(hl)} [{signed(lo)}, {signed(hi)}] & "
                        f"$p$ = {fp(p)} & & & \\\\")
            nums[f"{k}-{eff}-effect"] = signed(hl)
            nums[f"{k}-{eff}-p"] = fp(p)
        if c is not None:
            for rival in ("margin", "badge", "random"):
                if rival in c.index:
                    nums[f"{k}-vs-{rival}"] = c.loc[rival, "verdict"]
        for a in ("random", "gate_random", "gate_diverse"):
            nums[f"copies-{KEY[ds]}-{a.replace('_', '')}"] = pct(cp.get((r, a), np.nan))
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Selection rules compared on the same frozen generator, the same candidates per cycle and the same "
           r"budget (generator co-training off; macro-F2 on the test set). Proposed rule vs.\ this rule: two-sided "
           r"Wilcoxon test of `gate, then diversity' against the rule of the row, Holm-adjusted over the seven "
           r"comparisons; `better' means the proposed rule scores higher. Agreement: share of the "
           r"admitted images whose conditioned class the reference classifier also predicts. Diversity: mean pairwise "
           r"cosine distance of the admitted images of a class in the reference classifier's feature space. The main "
           r"effects are those of the 2$\times$2 design formed by random, diversity only, gate then random, and the "
           r"proposed rule, each averaged over the other factor. Near-copies: share of the admitted images that "
           r"reproduce one real training image (as in Table~\ref{tbl:realism}; about 6\% of unseen real images "
           r"qualify).")
    return table_env("selection", cap, "lccccc", header, body, draft=draft), pd.DataFrame(csv), nums


COTRAIN_ARMS = ["random", "margin", "gate_random", "gate_diverse"]


def t_cotrain(draft: bool):
    """R1.6, R3.2 (E11): four selection rules with generator co-training on, as in the frozen method."""
    df = primary(load("E11_selection_cotrain"))
    expect(df, [(r, "p_proposed", f"{a},cotrain") for r in ANCHORS for a in COTRAIN_ARMS], SEEDS10, "tbl:cotrain",
           draft)
    names = {"random": "Random", "margin": "Smallest margin (uncertainty)", "gate_random": "Gate, then random",
             "gate_diverse": "Gate, then diversity (proposed rule)"}
    header = [r"Selection rule & Macro-F2 & Proposed rule vs.\ this rule (Holm $p$) \\"]
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        g = df[df.regime == r].assign(arm=lambda x: x.variant.str.replace(",cotrain", "", regex=False))
        if g.empty:
            continue
        ds = REGIMES_V2[r]["dataset"]
        w = wide(g, col="arm")
        c = compare(w, "gate_diverse").set_index("rival") if "gate_diverse" in w and len(w) >= 5 else None
        body.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{NAMES[ds]} ({len(w)} seeds)}}}} \\\\")
        for a in COTRAIN_ARMS:
            if a not in w:
                continue
            vs = "--" if a == "gate_diverse" or c is None else \
                {"better": "better", "worse": "worse", "inconclusive": "n.s."}[c.loc[a, "verdict"]] + f" ({fp(c.loc[a, 'p_holm'])})"
            body.append(f"{names[a]} & {pm(w[a].mean(), w[a].std(ddof=1))} & {vs} \\\\")
            csv.append({"regime": r, "arm": a, "macro_f2": w[a].mean(), "sd": w[a].std(ddof=1),
                        **({"verdict_vs_rule": c.loc[a, "verdict"], "p_holm": c.loc[a, "p_holm"]}
                           if c is not None and a in c.index else {})})
        if {"gate_random", "random"} <= set(w.columns) and len(w) >= 5:
            d = w["gate_random"] - w["random"]
            hl, lo, hi = hodges_lehmann(d)
            p = wilcoxon(d)[1]
            body.append(f"\\quad gate, then random minus random (HL, 95\\% CI) & {signed(hl)} [{signed(lo)}, {signed(hi)}] "
                        f"& $p$ = {fp(p)} \\\\")
            nums[f"cotrain-{KEY[ds]}-gate-effect"] = signed(hl)
            nums[f"cotrain-{KEY[ds]}-gate-p"] = fp(p)
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Four of the selection rules of Table~\ref{tbl:selection} rerun with generator co-training on, as in the "
           r"frozen method (10 seeds; macro-F2 on the test set). Proposed rule vs.\ this rule: two-sided Wilcoxon "
           r"test, Holm-adjusted over the three comparisons; `better' means the proposed rule scores higher. Last row "
           r"of each block: the effect of the gate on randomly selected candidates, with co-training on.")
    return table_env("cotrain", cap, "lcc", header, body, draft=draft), pd.DataFrame(csv), nums


def t_chain(draft: bool):
    """R1.6: one component added per row, from BAGAN-GP to the frozen proposed method."""
    fz = frozen()
    if fz is None:
        raise Incomplete("tbl:chain: the configuration is not frozen yet")
    e3, e4, e11 = primary(load("E3_anchors")), primary(load("E4_selection")), primary(load("E11_selection_cotrain"))
    steps = [("BAGAN-GP: shared backbone, class-prior sampling", e3, "m09_bagan_gp", ""),
             ("+ latent-SMOTE seeding and acquisition cycles, random selection", e4, "p_proposed", "random"),
             ("+ confidence gate", e4, "p_proposed", "gate_random"),
             ("+ diversity among gate-passing candidates", e4, "p_proposed", "gate_diverse")]
    if fz["cotrain_g"]:
        steps.append(("+ generator co-training (= proposed, frozen)", e3, "p_proposed", "frozen"))
    else:
        steps.append(("+ generator co-training (not in the frozen method)", e11, "p_proposed", "gate_diverse,cotrain"))
    for _, d, m, v in steps:
        seeds = SEEDS10 if d is e11 else SEEDS20
        expect(d[d.regime.isin(ANCHORS)], [(r, m, v) for r in ANCHORS], seeds, "tbl:chain", draft)
    header = [r"Step & Macro-F2 & Change (HL, 95\% CI) & Holm $p$ & $n$ \\"]
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        ds = REGIMES_V2[r]["dataset"]
        series = [d[(d.regime == r) & (d.method == m) & (d.variant == v)].set_index("seed")["macro_f2"]
                  for _, d, m, v in steps]
        if not any(len(x) for x in series):
            continue
        diffs = []
        for i in range(1, len(steps)):
            common = series[i - 1].index.intersection(series[i].index)
            diffs.append((series[i][common] - series[i - 1][common]).to_numpy())
        ps = [wilcoxon(d)[1] if len(d) >= 5 else 1.0 for d in diffs]
        adj = holm(ps) if ps else []
        body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{NAMES[ds]}}}}} \\\\")
        for i, (name, *_rest) in enumerate(steps):
            x = series[i]
            if i == 0 or len(diffs[i - 1]) == 0:
                body.append(f"{tex(name)} & {pm(x.mean(), x.std(ddof=1)) if len(x) else '--'} & & & {len(x)} \\\\")
                continue
            hl, lo, hi = hodges_lehmann(diffs[i - 1])
            body.append(f"{tex(name)} & {pm(x.mean(), x.std(ddof=1))} & {signed(hl)} [{signed(lo)}, {signed(hi)}] & "
                        f"{fp(adj[i - 1])} & {len(diffs[i - 1])} \\\\")
            csv.append({"regime": r, "step": name, "macro_f2": x.mean(), "hl": hl, "ci_low": lo, "ci_high": hi,
                        "p_holm": adj[i - 1], "n": len(diffs[i - 1])})
            nums[f"chain-{KEY[ds]}-step{i}-credited"] = "yes" if adj[i - 1] < 0.05 and (lo > 0 or hi < 0) else "no"
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Component chain on the three benchmark regimes (macro-F2, test): each row adds one component to the row "
           r"above. Change: paired Hodges-Lehmann difference to the row above with its exact 95\% interval; Holm $p$: "
           r"two-sided Wilcoxon test adjusted over the steps of a dataset; $n$: paired seeds. A component is credited "
           r"only if its interval excludes zero.")
    return table_env("chain", cap, "lcccc", header, body, draft=draft), pd.DataFrame(csv), nums


# The swept parameters in the notation of Section 3 and Table tbl:config: (LaTeX, matplotlib mathtext).
PARAM_TEX = {"min_confidence": (r"gate threshold $\tau$", r"gate threshold $\tau$"),
             "pool_factor": (r"over-production $R$", r"over-production $R$"),
             "samples_per_cycle": (r"batch per cycle $B$", r"batch per cycle $B$"),
             "latent_smote_noise": (r"latent jitter $\eta$", r"latent jitter $\eta$"),
             "seed_mode": ("seeding rule", "seeding rule"),
             "lambda_adv": (r"$\lambda_{\mathrm{adv}}$", r"$\lambda_{\mathrm{adv}}$"),
             "lambda_cls": (r"$\lambda_{\mathrm{cls}}$", r"$\lambda_{\mathrm{cls}}$"),
             "lambda_anchor": (r"$\lambda_{\mathrm{anchor}}$", r"$\lambda_{\mathrm{anchor}}$"),
             "lambda_aux_real": ("critic weight, real auxiliary CE", "critic: real aux. CE"),
             "lambda_aux_synth": ("critic weight, synthetic auxiliary CE", "critic: synthetic aux. CE")}
VALUE_TEX = {"prior": "class prior", "latent_smote": "latent SMOTE"}


def t_sens(draft: bool):
    """R1.5: one parameter at a time around the frozen configuration, 10 seeds."""
    fz = frozen()
    if fz is None:
        raise Incomplete("tbl:sens: the configuration is not frozen yet")
    e8 = primary(load("E8_sensitivity"))
    e3 = primary(load("E3_anchors"))
    e11 = primary(load("E11_selection_cotrain"))
    base = e3[(e3.method == "p_proposed") & (e3.variant == "frozen") & e3.seed.isin(SEEDS10)]
    cotrain_ref = e11[e11.variant == "gate_diverse,cotrain"] if not fz["cotrain_g"] else base
    from casestudy.methods.m10v2_ab8c import Ab8cV2
    header = [r"Dataset & Parameter & Default & Value & Macro-F2 & Change (HL, 95\% CI) & Holm $p$ \\"]
    body, csv, nums = [], [], {}
    for r in ordered(ANCHORS):
        ds = REGIMES_V2[r]["dataset"]
        sweeps = SWEEPS_FASHION if ds == "fashion_mnist" else SWEEPS_OTHER
        cells = [(r, "p_proposed", f"{p}={v}") for p, vs in sweeps.items() for v in vs]
        expect(e8, cells, SEEDS10, f"tbl:sens ({r})", draft)
        first = True
        for p, vs in sweeps.items():
            ref = (cotrain_ref if p.startswith("lambda") else base)
            ref = ref[ref.regime == r].set_index("seed")["macro_f2"]
            rows, ps = [], []
            for v in vs:
                x = e8[(e8.regime == r) & (e8.variant == f"{p}={v}")].set_index("seed")["macro_f2"]
                common = x.index.intersection(ref.index)
                if len(common) < 5:
                    rows.append((v, x, None))
                    ps.append(1.0)
                    continue
                d = (x[common] - ref[common]).to_numpy()
                rows.append((v, x, hodges_lehmann(d)))
                ps.append(wilcoxon(d)[1])
            adj = holm(ps) if ps else []
            for i, (v, x, hl) in enumerate(rows):
                default = getattr(Ab8cV2, p, "latent SMOTE" if p == "seed_mode" else "--")
                show = lambda val: tex(VALUE_TEX.get(str(val), str(val)))
                body.append(f"{NAMES[ds] if first else ''} & {PARAM_TEX.get(p, (tex(p),))[0] if i == 0 else ''} & "
                            f"{show(default) if i == 0 else ''} & {show(v)} & {pm(x.mean(), x.std(ddof=1))} & "
                            + (f"{signed(hl[0])} [{signed(hl[1])}, {signed(hl[2])}] & {fp(adj[i])}" if hl else "-- & --") + r" \\")
                first = False
                csv.append({"regime": r, "param": p, "value": v, "macro_f2": x.mean(), "sd": x.std(ddof=1),
                            **({"hl": hl[0], "ci_low": hl[1], "ci_high": hl[2], "p_holm": adj[i]} if hl else {})})
        body.append(r"\addlinespace")
    if body and body[-1] == r"\addlinespace":
        body.pop()
    sig = [c for c in csv if c.get("p_holm", 1) < 0.05]
    nums["sens-nsig"] = len(sig)
    nums["sens-ntested"] = len(csv)
    if csv:
        _sens_figure(pd.DataFrame(csv))
    cap = (r"Sensitivity of the proposed method: one parameter changed at a time from the frozen configuration "
           r"(10 seeds). Change: paired Hodges-Lehmann difference to the frozen configuration with its exact 95\% "
           r"interval; Holm $p$ adjusted over the values of a parameter. The loss weights act only through generator "
           r"co-training and are therefore varied with co-training on.")
    return table_env("sens", cap, "llllccc", header, body, draft=draft), pd.DataFrame(csv), nums


def _sens_figure(sv: pd.DataFrame) -> None:
    """Change in macro-F2 against each swept value (Fashion-MNIST; tau on all three datasets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = list(dict.fromkeys(sv.param))
    fig, axes = plt.subplots(2, math.ceil(len(params) / 2), figsize=(1.9 * math.ceil(len(params) / 2), 3.8),
                             squeeze=False, sharey=True)
    for ax, p in zip(axes.flat, params):
        for r, g in sv[(sv.param == p) & sv.hl.notna()].groupby("regime", sort=False):
            x = [VALUE_TEX.get(str(v), str(v)) for v in g.value]
            ax.errorbar(x, g.hl, yerr=[g.hl - g.ci_low, g.ci_high - g.hl], fmt="o-", ms=3, lw=0.8, capsize=2,
                        label=NAMES[REGIMES_V2[r]["dataset"]])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(PARAM_TEX.get(p, (None, p.replace("_", " ")))[1], fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(params):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("change in macro-F2", fontsize=7)
    axes[0, 0].legend(fontsize=5)
    fig.tight_layout()
    _save(fig, "fig_sens.png")


def t_ratio(draft: bool):
    """R1.5, D9: the augmentation ratio r for five methods."""
    fz = frozen()
    if fz is None:
        raise Incomplete("tbl:ratio: the configuration is not frozen yet")
    e9, e3 = primary(load("E9_ratio")), primary(load("E3_anchors"))
    e3 = e3[e3.seed.isin(SEEDS10)]
    header = [r"Dataset & Method & " + " & ".join(f"$r = {r}$" if r != "full" else "$r$ = full" for r in RATIOS) + r" \\"]
    body, csv, nums = [], [], {}
    for reg in ordered(ANCHORS):
        own = REGIMES_V2[reg]["ratio"]
        ds = REGIMES_V2[reg]["dataset"]
        cells = [(reg, m, f"r={r}") for r in RATIOS if r != own for m in RATIO_METHODS]
        expect(e9, cells, SEEDS10, f"tbl:ratio ({reg})", draft)
        for i, m in enumerate(RATIO_METHODS):
            vals = []
            for r in RATIOS:
                if r == own:
                    x = e3[(e3.regime == reg) & (e3.method == m) & e3.variant.isin(["", "frozen"])]["macro_f2"]
                    vals.append(pm(x.mean(), x.std(ddof=1)) + r"$^{*}$")
                else:
                    x = e9[(e9.regime == reg) & (e9.method == m) & (e9.variant == f"r={r}")]["macro_f2"]
                    vals.append(pm(x.mean(), x.std(ddof=1)))
                csv.append({"regime": reg, "method": m, "r": r, "macro_f2": x.mean(), "sd": x.std(ddof=1)})
            body.append(f"{NAMES[ds] if i == 0 else ''} & {label(m)} & " + " & ".join(vals) + r" \\")
        body.append(r"\addlinespace")
        # for the prose: each method's best r and macro-F2 there, and the proposed method at its own r
        cr = pd.DataFrame([c for c in csv if c["regime"] == reg])
        for m, g in cr.groupby("method"):
            b = g.loc[g.macro_f2.idxmax()]
            k = f"ratio-{KEY[ds]}-{m.replace('_', '')}"
            nums.update({f"{k}-best": f"{b.macro_f2:.3f}", f"{k}-best-r": str(b.r),
                         f"{k}-own": f"{g[g.r == own].macro_f2.iloc[0]:.3f}",
                         f"{k}-span": f"{g.macro_f2.max() - g.macro_f2.min():.3f}"})
    if body and body[-1] == r"\addlinespace":
        body.pop()
    cap = (r"Macro-F2 (test, 10 seeds) against the augmentation ratio $r$: each minority class is augmented to "
           r"$\min(N_{\max}, rN_c)$ images; full = up to the majority count, so on CIFAR-10 ($\rho = 0.3$) $r = 4$ "
           r"already reaches it and the last two columns coincide. $^{*}$ the value used throughout the paper, fixed "
           r"per dataset in preliminary experiments with the proposed method.")
    return table_env("ratio", cap, "ll" + "c" * len(RATIOS), header, body, draft=draft), pd.DataFrame(csv), nums


def t_realism(draft: bool):
    """R1.9, claim C5: KID and FID of the added images against the real images of each minority class."""
    f = save_root() / "realism" / "E3_anchors.csv"
    if not f.exists():
        raise Incomplete("tbl:realism: run python -m kbs.realism run --stage E3_anchors first")
    from scipy.stats import spearmanr

    from kbs.realism import GENERATIVE
    real = pd.read_csv(f)
    seeds = [42, 43, 44, 45, 46]
    have = set(zip(real.regime, real.method, real.seed))
    missing = [(r, m, s) for r in ANCHORS for m in GENERATIVE for s in seeds if (r, m, s) not in have]
    if missing and not draft:
        raise Incomplete(f"tbl:realism: {len(missing)} (regime, method, seed) missing, e.g. {missing[:3]}")
    cp = copies_table("E3_anchors", [(r, m, "frozen" if m == "p_proposed" else "") for r in ANCHORS for m in GENERATIVE],
                      SEEDS20, "tbl:realism", draft)
    e3 = primary(load("E3_anchors"))
    regs = ordered(ANCHORS)
    header = [r"Method & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{NAMES[REGIMES_V2[r]['dataset']]}}}" for r in regs) + r" \\",
              " & " + " & ".join(r"KID & FID & Near-copies & Macro-F2" for _ in regs) + r" \\"]
    body, csv, nums = [], [], {}
    per = {}
    for r in regs:
        k = KEY[REGIMES_V2[r]["dataset"]]
        c_r = cp.loc[r] if r in cp.index.get_level_values(0) else pd.Series(dtype=float)
        if len(c_r):
            rivals = c_r.drop(index=["p_proposed", "real_unseen"], errors="ignore")
            nums.update({f"copies-{k}-proposed": pct(c_r.get("p_proposed", np.nan)),
                         f"copies-{k}-real": pct(c_r.get("real_unseen", np.nan)),
                         f"copies-{k}-maxrival": pct(rivals.max()), f"copies-{k}-maxrival-name": label(rivals.idxmax())})
        x = real[(real.regime == r) & real.seed.isin(seeds)].groupby(["method", "seed"])[["kid", "fid"]].mean()
        u = e3[(e3.regime == r) & e3.seed.isin(seeds)].groupby("method")["macro_f2"].mean()
        per[r] = x.groupby("method").agg(["mean", "std"]), u
        m_ = x.groupby("method").mean().join(u, how="inner")
        if len(m_) >= 3:
            rho, p = spearmanr(m_["kid"], m_["macro_f2"])
            nums[f"realism-{KEY[REGIMES_V2[r]['dataset']]}-spearman"] = f"{rho:+.2f}".replace("-", "$-$")
            nums[f"realism-{KEY[REGIMES_V2[r]['dataset']]}-spearman-p"] = fp(p)
            best_kid, best_f2 = m_["kid"].idxmin(), m_["macro_f2"].idxmax()
            nums[f"realism-{KEY[REGIMES_V2[r]['dataset']]}-bestkid"] = label(best_kid)
            nums[f"realism-{KEY[REGIMES_V2[r]['dataset']]}-bestkid-f2rank"] = int(m_["macro_f2"].rank(ascending=False)[best_kid])
    for m in GENERATIVE:
        cells_ = []
        for r in regs:
            agg, u = per[r]
            if m not in agg.index:
                cells_ += ["--"] * 4
                continue
            nc = cp.get((r, m), np.nan)
            cells_ += [pm(agg.loc[m, ("kid", "mean")], agg.loc[m, ("kid", "std")]),
                       f"{agg.loc[m, ('fid', 'mean')]:.0f}", pct(nc), f"{u.get(m, np.nan):.3f}"]
            csv.append({"regime": r, "method": m, "kid": agg.loc[m, ("kid", "mean")], "kid_sd": agg.loc[m, ("kid", "std")],
                        "fid": agg.loc[m, ("fid", "mean")], "near_copy": nc, "macro_f2": u.get(m, np.nan)})
        body.append(f"{label(m)} & " + " & ".join(cells_) + r" \\")
    kc = save_root() / "realism" / "E3_anchors_kid_by_copy.csv"     # kbs.realism kid-by-copy (seeds 42-46, pooled)
    if kc.exists():
        kb = pd.read_csv(kc).groupby(["regime", "group"])["kid"].mean()
        for r in regs:
            for g, tag in (("proposed: near-copies", "copies"), ("proposed: other images", "other"),
                           ("BAGAN-GP: all", "bagangp")):
                if (r, g) in kb.index:
                    nums[f"realism-{KEY[REGIMES_V2[r]['dataset']]}-kid-{tag}"] = f"{kb[(r, g)]:.3f}"
    body.append(r"\addlinespace")
    body.append(r"\textit{Real images the run never saw} & " + " & ".join(
        f"-- & -- & {pct(cp.get((r, 'real_unseen'), np.nan))} & --" for r in regs) + r" \\")
    cap = (r"Realism of the images each generative method added, against all real training images of the same class: "
           r"kernel inception distance (KID, unbiased; mean $\pm$ standard deviation over seeds 42-46, averaged over "
           r"the two minority classes) and FID (biased at these sample sizes; shown for continuity). Near-copies (20 "
           r"seeds): share of the added images that match one real training image of the run much more closely than "
           r"any other (pixel space; ratio of the distances to the nearest and the second-nearest training image below "
           r"the 5th percentile of that ratio for real images of the class that the run never saw, which give the last "
           r"row). Macro-F2: test, seeds 42-46.")
    return table_env("realism", cap, "l" + "cccc" * len(regs), header, body, draft=draft, star=True), pd.DataFrame(csv), nums


def copies_table(stage: str, cells, seeds, name: str, draft: bool) -> pd.Series:
    """Near-copy share by (regime, method or arm), mean over seeds and minority classes (``kbs.realism copies``).
    ``cells`` are (regime, method, variant); arms and strata of the proposed method (E4, E5) are returned under
    their variant name."""
    f = save_root() / "realism" / f"{stage}_copies.csv"
    if not f.exists():
        raise Incomplete(f"{name}: run python -m kbs.realism copies --stage {stage} first")
    d = pd.read_csv(f)
    d["variant"] = d["variant"].fillna("")
    have = set(zip(d.regime, d.method, d.variant, d.seed))
    missing = [(r, m, v, s) for r, m, v in cells for s in seeds if (r, m, v, s) not in have]
    if missing and not draft:
        raise Incomplete(f"{name}: near-copy measurement missing for {len(missing)} runs, e.g. {missing[:3]}")
    d = d[d.seed.isin(seeds)]
    d["key"] = [v if m == "p_proposed" and v not in ("", "frozen") else m for m, v in zip(d.method, d.variant)]
    return d.groupby(["regime", "key"])["near_copy"].mean()


def t_cost(draft: bool):
    """R1.12: wall-clock time by stage, peak memory and generation speed (E0 and its replicates)."""
    df = primary(pd.concat([load("E0_smoke"), load("E0_timing")], ignore_index=True))
    if (df.method.eq("p_proposed") & df.variant.eq("frozen")).any():   # time the method as frozen once it exists
        df = df[~(df.method.eq("p_proposed") & df.variant.ne("frozen"))]
    cells = [(r, m) for r in ANCHORS for m in FIELD]
    expect(df, cells, [42], "tbl:cost", draft, cols=("regime", "method"))
    regs = ordered(ANCHORS)
    header = [r"Method & " + " & ".join(NAMES[REGIMES_V2[r]["dataset"]] for r in regs)
              + r" & Peak memory (MB) & Augmenter parameters (M) \\"]
    body, csv, nums = [], [], {}
    stats = {}
    for r in regs:
        for m in FIELD:
            g = df[(df.regime == r) & (df.method == m)]
            if g.empty:
                continue
            aug = np.array([t.get("augment_s", np.nan) for t in g.timings], float)
            tot = np.array([t.get("total_s", np.nan) for t in g.timings], float)
            if m in ("p_proposed", "m09_bagan_gp"):
                # A cached backbone was trained by an earlier run; add the training time recorded with it, so that
                # each method is costed as if run alone.
                bb = np.array([mt.get("ae_s", 0) + mt.get("backbone_s", 0) if mt.get("backbone_cached") else 0
                               for mt in g.method_timings], float)
                aug, tot = aug + bb, tot + bb
            par = g.params.iloc[0] or {}
            if m == "p_proposed":   # the split of the augmenter: shared backbone against the acquisition loop itself
                bb_all = np.array([mt.get("ae_s", 0) + mt.get("backbone_s", 0) for mt in g.method_timings], float)
                k = f"cost-{KEY[REGIMES_V2[r]['dataset']]}"
                nums[f"{k}-proposed-backbone-s"] = f"{bb_all.mean():.0f}"
                nums[f"{k}-proposed-loop-s"] = f"{(aug - bb_all).mean():.1f}"
                nums[f"{k}-proposed-per-image-s"] = f"{aug.mean() / max(1, g.n_synth.mean()):.2f}"
            stats[(r, m)] = {"aug": aug.mean(), "tot": tot.mean(), "peak": g.peak_mem_mb.mean(), "n": len(g),
                             "per_image_ms": 1000 * aug.mean() / max(1, g.n_synth.mean()),
                             "aug_params_m": (par.get("generator", 0) + par.get("critic", 0) + par.get("internal_clf", 0)) / 1e6}
    for m in FIELD:
        cells_, peak, par = [], [], []
        for r in regs:
            s = stats.get((r, m))
            cells_.append("--" if s is None else f"{s['tot']:.0f} ({s['aug']:.0f})")
            if s:
                peak.append(s["peak"])
                par.append(s["aug_params_m"])
                csv.append({"regime": r, "method": m, **s})
        body.append(f"{label(m)} & " + " & ".join(cells_) + f" & {max(peak):.0f} & "
                    + (f"{max(par):.2f}" if par and max(par) > 0 else "--") + r" \\")
    for r in regs:
        p, b, n0 = stats.get((r, "p_proposed")), stats.get((r, "m09_bagan_gp")), stats.get((r, "b1_weighted"))
        sa = stats.get((r, "m14_stdaug_online"))
        if p and b and n0:
            k = f"cost-{KEY[REGIMES_V2[r]['dataset']]}"
            nums[f"{k}-vs-bagangp"] = f"{p['tot'] / b['tot']:.2f}"
            nums[f"{k}-vs-noaug"] = f"{p['tot'] / n0['tot']:.1f}"
            nums[f"{k}-proposed-s"] = f"{p['tot']:.0f}"
            if sa:
                nums[f"{k}-vs-stdaug-online"] = f"{p['tot'] / sa['tot']:.1f}"
                nums[f"{k}-stdaug-online-s"] = f"{sa['tot']:.0f}"
    shown = df[df.method.isin(FIELD) & df.regime.isin(ANCHORS)]      # E0_smoke also holds the old unsmoothed BAGAN
    n_rep = int(shown.groupby(["regime", "method"]).size().min()) if not shown.empty else 0
    cap = (r"Cost of one run on one RTX 5090 with no other job on the GPU: wall-clock seconds of the whole run, "
           r"including the downstream classifier, with the augmenter's share (training and generation) in brackets; "
           f"mean over {n_rep} run(s) per cell. Peak GPU memory and parameters (generator, critic and internal "
           r"classifier of the augmenter): largest over the three datasets. The proposed method's time "
           r"includes its autoencoder and generator warm-up, which it shares with BAGAN-GP when both are run.")
    return table_env("cost", cap, "l" + "c" * (len(regs) + 2), header, body, draft=draft), pd.DataFrame(csv), nums


def t_robust(draft: bool):
    """Appendix: the submitted paper's treatments (smoothed BAGAN, accuracy-selected baseline) and the second test evaluation."""
    e12, e3 = primary(load("E12_robustness")), load("E3_anchors")
    expect(e12, [(r, "m11_naive", "") for r in ANCHORS], SEEDS10, "tbl:robust", draft)
    regs = ordered(ANCHORS)
    header = [r"Row & " + " & ".join(NAMES[REGIMES_V2[r]["dataset"]] for r in regs) + r" \\"]
    body, csv, nums = [], [], {}
    p3 = primary(e3)
    p3 = p3[p3.seed.isin(SEEDS10)]
    expect(p3, [(r, "m08_bagan", "") for r in ANCHORS], SEEDS10, "tbl:robust (unsmoothed BAGAN)", draft)
    rows = [("No augmentation, class-weighted CE (the treatment of every method)", p3, "b1_weighted"),
            ("No augmentation, plain CE", p3, "b0_plain"),
            ("No augmentation, accuracy-selected (submitted protocol)", e12, "m11_naive"),
            ("BAGAN, label smoothing 0.1 (main table, as submitted)", p3, "m08_bagan_ls"),
            ("BAGAN, no label smoothing", p3, "m08_bagan")]
    for name, d, m in rows:
        cells_ = []
        for r in regs:
            x = d[(d.regime == r) & (d.method == m)]["macro_f2"]
            cells_.append(pm(x.mean(), x.std(ddof=1)) if len(x) else "--")
            csv.append({"row": name, "regime": r, "macro_f2": x.mean()})
        body.append(f"{tex(name)} & " + " & ".join(cells_) + r" \\")
        if m == "m11_naive":            # the handicap of the submitted treatment, paired on pool and seed
            cells_ = []
            for r in regs:
                a = p3[(p3.regime == r) & (p3.method == "b1_weighted")].set_index("seed")["macro_f2"]
                b = e12[(e12.regime == r) & (e12.method == "m11_naive")].set_index("seed")["macro_f2"]
                common = a.index.intersection(b.index)
                if len(common) < 5:
                    cells_.append("--")
                    continue
                hl, lo, hi = hodges_lehmann((b[common] - a[common]).to_numpy())
                cells_.append(f"{signed(hl)} [{signed(lo)}, {signed(hi)}]")
                k = f"robust-{KEY[REGIMES_V2[r]['dataset']]}-naive"
                nums.update({f"{k}-handicap": f"{-hl:.3f}", f"{k}-handicap-lo": f"{-hi:.3f}", f"{k}-handicap-hi": f"{-lo:.3f}"})
            body.append(r"\quad accuracy-selected minus class-weighted (HL, 95\% CI) & " + " & ".join(cells_) + r" \\")
    body.append(r"\midrule")
    # what the loss contributes: the same saved synthetic sets, retrained with each loss (plain minus weighted)
    reuse = e12[e12.variant.isin(["reuse,weighted", "reuse,plain"])]
    methods = [m for m in REUSE_METHODS if m in set(reuse.method)]
    expect(reuse, [(r, m, v) for r in ANCHORS for m in methods for v in ("reuse,weighted", "reuse,plain")], SEEDS10,
           "tbl:robust (reuse)", draft)
    for m in methods:
        cells_ = []
        for r in regs:
            g = reuse[(reuse.regime == r) & (reuse.method == m)]
            w = g.pivot_table(index="seed", columns="variant", values="macro_f2").dropna()
            if len(w) < 5:
                cells_.append("--")
                continue
            d = (w["reuse,plain"] - w["reuse,weighted"]).to_numpy()
            hl, lo, hi = hodges_lehmann(d)
            cells_.append(f"{signed(hl)} [{signed(lo)}, {signed(hi)}]")
            csv.append({"row": f"plain minus weighted CE: {m}", "regime": r, "hl": hl, "ci_low": lo, "ci_high": hi,
                        "p": wilcoxon(d)[1]})
        body.append(f"Plain minus class-weighted CE, {label(m)} & " + " & ".join(cells_) + r" \\")
    if methods:
        body.append(r"\midrule")
    # rank of the proposed method under both test evaluations
    for split, name in (("test", "standard test set"), ("test_full", "all official test images, re-weighted")):
        cells_ = []
        for r in regs:
            g = e3[(e3.regime == r) & (e3.split == split) & e3.variant.isin(["", "frozen"])]
            m = g.groupby("method")["macro_f2"].mean().sort_values(ascending=False)
            cells_.append(f"{list(m.index).index('p_proposed') + 1} / {len(m)}" if "p_proposed" in m.index else "--")
        body.append(f"Rank of the proposed method, {name} & " + " & ".join(cells_) + r" \\")
    cap = (r"Appendix: the treatments of the submitted paper (macro-F2, test, seeds 42-51); the effect of the loss on "
           r"the same saved synthetic sets (downstream classifier retrained with plain and with class-weighted "
           r"cross-entropy; paired Hodges-Lehmann difference with its exact 95\% interval, seeds 42-51); and the rank "
           r"of the proposed method by mean macro-F2 under the two test evaluations (20 seeds).")
    return table_env("robust", cap, "l" + "c" * len(regs), header, body, draft=draft), pd.DataFrame(csv), nums


TABLES: Dict[str, Callable] = {"random": t_random, "main": t_main, "stats": t_stats, "classwise": t_classwise,
                               "domain": t_domain, "rho": t_grid, "pilot": t_pilot, "fallback": t_fallback,
                               "selection": t_selection, "chain": t_chain, "audit": t_audit, "strata": t_strata,
                               "labels": t_labels, "cotrain": t_cotrain, "sens": t_sens,
                               "ratio": t_ratio, "realism": t_realism, "cost": t_cost, "robust": t_robust}


# ---------------------------------------------------------------------------- numbers
def write_numbers(name: str, nums: dict) -> None:
    store_f = OUT / "numbers.json"
    store = json.loads(store_f.read_text(encoding="utf-8")) if store_f.exists() else {}
    store = {k: v for k, v in store.items() if not k.startswith(name + "-")}
    store.update({k: v for k, v in nums.items() if v is not None})
    store_f.write_text(json.dumps(store, indent=1, sort_keys=True), encoding="utf-8")
    lines = ["% Generated by kbs/tables.py from numbers.json - do not edit by hand.",
             r"\providecommand{\kbsnum}[1]{\ifcsname kbsnum@#1\endcsname\csname kbsnum@#1\endcsname"
             r"\else\textbf{??#1??}\PackageWarning{kbs}{undefined number #1}\fi}"]
    lines += [f"\\expandafter\\def\\csname kbsnum@{k}\\endcsname{{{v}}}" for k, v in sorted(store.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(names: List[str], draft: bool, check: bool = False) -> Dict[str, str]:
    global WRITE
    WRITE = not check
    if WRITE:
        OUT.mkdir(parents=True, exist_ok=True)
    status = {}
    for name in names:
        try:
            tex_, csv, nums = TABLES[name](draft)
        except Incomplete as e:
            status[name] = f"waiting: {e}"
            continue
        if not WRITE:
            status[name] = "complete (not written: --check)"
            continue
        (OUT / f"tbl_{name}.tex").write_text(tex_, encoding="utf-8")
        if csv is not None and not csv.empty:
            csv.to_csv(OUT / f"tbl_{name}.csv", index=False)
        if not draft:
            write_numbers(name, nums)
        status[name] = "written" + (" (draft)" if "DRAFT" in tex_ else "") + (f", {len(nums)} numbers" if nums else "")
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(TABLES))
    ap.add_argument("--draft", action="store_true", help="build incomplete tables too, captioned DRAFT")
    ap.add_argument("--check", action="store_true", help="report which tables are complete; write nothing")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, f in TABLES.items():
            print(f"{k:12s} {f.__doc__.strip().splitlines()[0] if f.__doc__ else ''}")
        return
    for name, st in build(args.only or list(TABLES), args.draft, args.check).items():
        print(f"{name:12s} {st}")


if __name__ == "__main__":
    main()
