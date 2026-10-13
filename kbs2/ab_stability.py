"""kbs2.ab_stability - seed-to-seed stability of the selection rules in Table 3 (13 Oct 2026).

    python -m kbs2.ab_stability      # writes saved_kbs2/analysis/AB/stability.csv

Not part of the analysis fixed before experiment AB (kbs2.ab_analysis): added at the authors' request
after the results were seen. Pitman-Morgan test for paired variances (paired by seed), the test that
kbs2.ab_analysis uses for co-training; unadjusted p. Cited in Section 4.3 and the R1.6 / R3.2 replies.
"""
import os

import pandas as pd

from kbs2.ab_analysis import pitman_morgan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(ROOT, "saved_kbs2", "analysis", "AB")
METRICS = ["macro_recall", "gmean", "macro_f2", "macro_specificity"]


def main() -> None:
    b = pd.read_csv(os.path.join(AB, "ab_runs_with_extras.csv"))
    b = b[~b["cotrain"]]
    w = {m: b[b["rule"] == m].set_index("seed").sort_index() for m in b["rule"].unique()}
    rows = []
    for other in ["random", "margin", "gate_random", "diversity_only"]:
        for c in METRICS:
            x, y = w["gate_diverse"][c].values, w[other][c].values
            rows.append({"panel": "B", "rule": "gate_diverse", "vs": other, "metric": c,
                         "sd": x.std(ddof=1), "sd_vs": y.std(ddof=1), "p": pitman_morgan(x, y)})
    a = pd.read_csv(os.path.join(ROOT, "ablation_15_seeds.csv"))
    a = a[a["section"].astype(str).str.startswith("seed_")]
    p, v = (a[a["row"] == r].set_index("seed").sort_index() for r in ("Proposed", "Variant 3"))
    for c in ["recall", "gmean", "f2", "specificity"]:
        x, y = p[c].astype(float).values, v[c].astype(float).values
        rows.append({"panel": "A", "rule": "Proposed", "vs": "Variant 4 (csv row Variant 3)", "metric": c,
                     "sd": x.std(ddof=1), "sd_vs": y.std(ddof=1), "p": pitman_morgan(x, y)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(AB, "stability.csv"), index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
