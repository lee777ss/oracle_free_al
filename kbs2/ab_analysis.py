"""kbs2.ab_analysis — pre-specified analysis of Table 3 panel B (experiment AB; R1.6, R3.2).

    python -m kbs2.ab_analysis            # writes saved_kbs2/analysis/AB/

Fixed on 13 Oct 2026, before any AB result was seen (author decisions: 8 rules, co-training off and on,
15 seeds 42-56, panel A's data; outcomes of co-training listed by the authors' request).

Per arm (rule x co-training), over the 15 seeds: recall, G-mean, macro-F2, specificity (mean, SD); label
agreement = share of admitted images to which the reference classifier (kbs2.oracle_ablation) assigns the
conditioning label; gate pass rate (mean over class-cycles); fallback share (class-cycles in which fewer
than n candidates passed); below-gate share of admitted images.

Tests, all two-sided Wilcoxon signed-rank paired by seed, macro-F2:
  R1  Rules (co-training off): proposed rule (gate, then diversity) vs each of the other seven,
      Holm-adjusted over the seven.
  R2  2 x 2 design (co-training off) over {random, diversity only, gate then random, gate then diversity}:
      main effect of the gate = per seed ((gate_random + gate_diverse) - (random + diversity_only)) / 2,
      main effect of the diversity step = ((diversity_only + gate_diverse) - (random + gate_random)) / 2;
      Hodges-Lehmann estimate, exact 95% CI, p.
  C1  Co-training, primary: proposed rule, co-training on minus off, on each of: macro-F2 (HL, CI, p),
      seed-to-seed SD (Pitman-Morgan test of equal variances for paired samples), label agreement,
      gate pass rate, fallback share (HL, CI, p each). No adjustment (one pre-specified contrast per outcome).
  C2  Co-training, the same five outcomes for each of the other seven rules: reported with unadjusted p,
      described as secondary.
Everything is reported whichever way it comes out.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2.analysis import hodges_lehmann, holm, wilcoxon_p  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "AB")
RUNS = os.path.join(ROOT, "saved_kbs2", "AB", "runs.csv")
METRICS = ["macro_recall", "gmean", "macro_f2", "macro_specificity"]
RULES = ["random", "margin", "badge", "agree_margin", "top_confidence", "gate_random", "diversity_only",
         "gate_diverse"]
NAMES = {"random": "Random", "margin": "Smallest margin (uncertainty)", "badge": "BADGE (uncertainty + diversity)",
         "agree_margin": "Margin among gate-passing", "top_confidence": "Highest confidence",
         "gate_random": "Gate, then random (confidence only)", "diversity_only": "Diversity only (k-means++)",
         "gate_diverse": "Gate, then diversity (proposed rule)"}
PROPOSED = "gate_diverse"


def pitman_morgan(x: np.ndarray, y: np.ndarray) -> float:
    """p of H0: var(x) = var(y) for paired samples (correlation of sums and differences)."""
    from scipy.stats import pearsonr
    return float(pearsonr(x + y, x - y)[1])


def per_run_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Agreement, pass rate, fallback and below-gate share from each run's saved set."""
    from kbs2.oracle_ablation import load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref = load(device)
    rows = []
    for _, r in df.iterrows():
        # Written by kbs2.runner.generate in this repository (tensors and numpy metadata);
        # weights_only=True would reject the numpy arrays, as in runner.load_synth.
        S = torch.load(os.path.join(ROOT, r["synth_source"]), weights_only=False)
        pred = ref(S["X"]).argmax(1)
        gl = S["gate_log"]
        n_adm = sum(g["n_admitted"] for g in gl)
        rows.append({"agreement": float((pred == S["y"]).float().mean()),
                     "pass_rate": float(np.mean([g["pass_rate"] for g in gl])),
                     "fallback": float(np.mean([g["fell_back"] for g in gl])),
                     "below_gate": float(sum(g["n_violating"] for g in gl) / max(n_adm, 1))})
    return pd.DataFrame(rows, index=df.index)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(RUNS)
    df["rule"] = df["tag"].str.rsplit("_", n=1).str[0]
    df["cotrain"] = df["tag"].str.endswith("_cot")
    n = df.groupby(["rule", "cotrain"]).size()
    if not (n == 15).all() or len(n) != 16:
        raise SystemExit(f"expected 16 arms x 15 seeds:\n{n}")
    df = df.join(per_run_extras(df))
    df.to_csv(os.path.join(OUT, "ab_runs_with_extras.csv"), index=False)
    outs = METRICS + ["agreement", "pass_rate", "fallback", "below_gate"]

    summ = df.groupby(["cotrain", "rule"])[outs].agg(["mean", "std"])
    summ.to_csv(os.path.join(OUT, "ab_summary.csv"))

    def seeds(rule, cot, col):
        s = df[(df.rule == rule) & (df.cotrain == cot)].sort_values("seed")
        return s[col].to_numpy(), s["seed"].to_numpy()

    # R1
    base, sd = seeds(PROPOSED, False, "macro_f2")
    r1 = []
    for rule in [r for r in RULES if r != PROPOSED]:
        x, s2 = seeds(rule, False, "macro_f2")
        assert (sd == s2).all()
        d = base - x
        est, lo, hi = hodges_lehmann(d)
        r1.append({"rule": rule, "hl": est, "ci_low": lo, "ci_high": hi, "p": wilcoxon_p(d)[1],
                   "wins": int((d > 0).sum())})
    r1 = pd.DataFrame(r1)
    r1["p_holm"] = holm(r1["p"])
    r1.to_csv(os.path.join(OUT, "r1_rules_vs_proposed.csv"), index=False)

    # R2
    g = {r: seeds(r, False, "macro_f2")[0] for r in ("random", "diversity_only", "gate_random", "gate_diverse")}
    eff = {"gate": ((g["gate_random"] + g["gate_diverse"]) - (g["random"] + g["diversity_only"])) / 2,
           "diversity": ((g["diversity_only"] + g["gate_diverse"]) - (g["random"] + g["gate_random"])) / 2}
    r2 = pd.DataFrame([{"effect": k, "hl": hodges_lehmann(v)[0], "ci_low": hodges_lehmann(v)[1],
                        "ci_high": hodges_lehmann(v)[2], "p": wilcoxon_p(v)[1]} for k, v in eff.items()])
    r2.to_csv(os.path.join(OUT, "r2_main_effects.csv"), index=False)

    # C1, C2
    c = []
    for rule in RULES:
        for col in ["macro_f2", "agreement", "pass_rate", "fallback", "gmean", "macro_recall", "macro_specificity"]:
            on, _ = seeds(rule, True, col)
            off, _ = seeds(rule, False, col)
            d = on - off
            est, lo, hi = hodges_lehmann(d)
            c.append({"rule": rule, "outcome": col, "mean_off": off.mean(), "mean_on": on.mean(),
                      "sd_off": off.std(ddof=1), "sd_on": on.std(ddof=1), "hl": est, "ci_low": lo, "ci_high": hi,
                      "p": wilcoxon_p(d)[1], "p_equal_var": pitman_morgan(on, off),
                      "role": "primary" if rule == PROPOSED else "secondary"})
    pd.DataFrame(c).to_csv(os.path.join(OUT, "c_cotraining.csv"), index=False)

    pd.set_option("display.width", 220)
    print(summ.xs("mean", axis=1, level=1).round(4).to_string())
    print("\nR1\n", r1.round(4).to_string(index=False))
    print("\nR2\n", r2.round(4).to_string(index=False))
    print("\nC1/C2 (macro_f2, agreement, pass_rate, fallback)\n",
          pd.DataFrame(c)[lambda t: t.outcome.isin(["macro_f2", "agreement", "pass_rate", "fallback"])]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
