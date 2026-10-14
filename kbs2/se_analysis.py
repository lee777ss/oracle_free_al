"""kbs2.se_analysis - sensitivity of the proposed method (experiment SE; Reviewer 1, comment 5).

    python -m kbs2.se_analysis      # writes saved_kbs2/analysis/SE/

Fixed on 13 Oct 2026, before any SE run (authors chose option c: re-run under the paper's protocol).

Design. The proposed method of Table 6 (three anchor regimes, seeds 42-51, downstream classifier seed 42,
arm W), one setting changed at a time from the submitted configuration:
    gate threshold tau          0.5, 0.7, 0.8, 0.9   (submitted 0.6)   all three datasets
    over-production factor      2, 5, 20             (submitted 10)    Fashion-MNIST
    images per class per cycle  10, 40               (submitted 20)    Fashion-MNIST
tau 0.5 is E1b's arm W (already run); the other settings are SE.
Added 13 Oct, before any SE2 run (authors, R1.5 option b): experiment SE2 halves and doubles the generator
loss coefficients on Fashion-MNIST, one at a time: lambda_cls 0.1, 0.4 (submitted 0.2) and lambda_anchor 0.5,
2.0 (submitted 1.0); same analysis. The reference for every comparison is E1,
the reproduced Table 6 rows (tau 0.6, pool factor 10, 20 per cycle).

Analysis. Primary outcome macro-F2. For each dataset, parameter and value: the change from the submitted
setting paired by seed, Hodges-Lehmann estimate with exact 95% CI, two-sided Wilcoxon signed-rank p, Holm
over the values of that parameter on that dataset (as in the sensitivity table of the first revision).
Recall, G-mean and specificity: means and SDs, reported descriptively. Reported whichever way it comes out.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402
from kbs2.analysis import hodges_lehmann, holm, wilcoxon_p  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "SE")
METRICS = ["macro_recall", "gmean", "macro_f2", "macro_specificity"]
SUBMITTED = {"tau": 0.6, "pool_factor": 10, "samples_per_cycle": 20, "lambda_cls": 0.2, "lambda_anchor": 1.0}


def _runs(exp: str) -> pd.DataFrame:
    d = pd.read_csv(os.path.join(ROOT, "saved_kbs2", exp, "runs.csv"))
    return d[(d["method"] == P.PROPOSED) & (d["arm"] == "W")]


def settings() -> pd.DataFrame:
    """One row per run, with the parameter it varies and its value; reference rows have param 'submitted'."""
    ref = _runs("E1").assign(param="submitted", value=np.nan)
    t05 = _runs("E1b")
    t05 = t05[t05["tag"] == "tau05"].assign(param="tau", value=0.5)
    se = _runs("SE").copy()
    kind = se["tag"].str.extract(r"^(tau|pf|spc)(.+)$")
    se["param"] = kind[0].map({"tau": "tau", "pf": "pool_factor", "spc": "samples_per_cycle"})
    se["value"] = kind[1].astype(float)
    parts = [ref, t05, se]
    if os.path.exists(os.path.join(ROOT, "saved_kbs2", "SE2", "runs.csv")):
        se2 = _runs("SE2").copy()
        kind = se2["tag"].str.extract(r"^(lc|la)(.+)$")
        se2["param"] = kind[0].map({"lc": "lambda_cls", "la": "lambda_anchor"})
        se2["value"] = kind[1].astype(float)
        parts.append(se2)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    d = settings()
    rows = []
    for (regime, param), g in d[d["param"] != "submitted"].groupby(["regime", "param"]):
        ref = d[(d["param"] == "submitted") & (d["regime"] == regime)].set_index("seed").sort_index()
        block = []
        for value, v in g.groupby("value"):
            v = v.set_index("seed").sort_index()
            if list(v.index) != list(P.SEEDS) or list(ref.index) != list(P.SEEDS):
                raise SystemExit(f"{regime} {param}={value}: seeds {list(v.index)} vs reference {list(ref.index)}")
            diff = (v["macro_f2"] - ref["macro_f2"]).to_numpy()
            est, lo, hi = hodges_lehmann(diff)
            row = {"regime": regime, "param": param, "value": value, "submitted": SUBMITTED[param],
                   "hl_f2": est, "ci_low": lo, "ci_high": hi, "p": wilcoxon_p(diff)[1]}
            for m in METRICS:
                row[f"{m}_mean"], row[f"{m}_sd"] = v[m].mean(), v[m].std(ddof=1)
                row[f"{m}_ref_mean"], row[f"{m}_ref_sd"] = ref[m].mean(), ref[m].std(ddof=1)
            block.append(row)
        for row, ph in zip(block, holm([r["p"] for r in block])):
            row["p_holm"] = ph
        rows += block
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "se_table.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out[["regime", "param", "value", "macro_f2_mean", "macro_f2_ref_mean", "hl_f2", "ci_low",
               "ci_high", "p", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
