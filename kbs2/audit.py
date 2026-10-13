"""kbs2.audit - V2b: label audit of the proposed method's admitted images (R3.3), inference only.

    python -m kbs2.audit          # writes saved_kbs2/analysis/V2b/

Approved 10 Oct 2026 with two rules fixed before any audit result existed:

Rule 1 - "no measurable loss of label quality". Per dataset and tau, per seed: agreement rate of
  gated admissions (passed the gate) and of fallback admissions (admitted although they failed the
  gate) with the reference classifier; paired difference gated minus fallback over the seeds that
  have both kinds; Hodges-Lehmann estimate with exact 95% CI and two-sided Wilcoxon p.
  CI entirely within +-5 percentage points -> "no meaningful loss" (flagged if the CI also excludes 0);
  else CI entirely above 0 -> "loss"; otherwise -> "inconclusive".
Rule 2 - choice of tau. Mean reference agreement of all admitted images per dataset (mean over
  seeds), averaged over the three datasets with equal weight; the tau with the higher value is chosen
  if it is higher by at least 1 percentage point; otherwise tau = 0.5 (decision D13) is kept.
  Test-set performance is not used.

Reference classifier: the previous revision's ensemble (saved_kbs/oracle/<dataset>, 3 ResNet-18),
verified in V2a (class order, no test images in training). Before auditing, its accuracy on all
official test images of the three classes is recomputed and compared with its saved summary.
Bias: it was trained on every official training image, including the pools the generator copies
from, so near-copies are labelled correctly almost by default; agreement is therefore also reported
for non-near-copy admissions only (near-copy definition of kbs2.analysis, unchanged).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from kbs2 import protocol as P  # noqa: E402
from kbs2.analysis import _ratio_q, hodges_lehmann, wilcoxon_p  # noqa: E402

OUT = os.path.join(ROOT, "saved_kbs2", "analysis", "V2b")
EXPS = [("E1", 0.6), ("E1b", 0.5)]
MARGIN, RULE2_MIN = 0.05, 0.01


def verify_oracle(ds: str, oracle, cfg, spec) -> dict:
    from casestudy.data.loaders import _filter_classes, _load_raw, _remap
    from casestudy.data.transforms import to_normalized_tensor
    summ = json.load(open(os.path.join(ROOT, "saved_kbs", "oracle", ds, "summary.json"), encoding="utf-8"))
    x, y = _load_raw(spec.loader_key, False, cfg.data_root)
    x, y = _filter_classes(x, y, summ["classes"])
    y = _remap(y, summ["classes"])
    xt = to_normalized_tensor(x, spec.channels, cfg.image_size, cfg.normalize_range)
    acc = float((oracle(xt).argmax(1).numpy() == y).mean())
    return {"dataset": ds, "recomputed_accuracy": acc, "saved_accuracy": summ["ensemble"]["accuracy"],
            "n_test": int(len(y)), "match": abs(acc - summ["ensemble"]["accuracy"]) < 1e-3}


def near_copy_thresholds(ds: str, D: dict, Xtr, ytr) -> dict:
    """Same construction as kbs2.analysis.near_copy_rates (fixed before E1)."""
    import hashlib
    from casestudy.data.loaders import _filter_classes, _load_raw, _remap
    from casestudy.data.transforms import to_normalized_tensor
    cfg, spec, pool = D["config"], D["spec"], D["pool"]
    cache = np.load(D["cache"][0], allow_pickle=False)
    pool_hashes = set(cache["hashes"].tolist())
    raw_x, raw_y = _load_raw(spec.loader_key, True, cfg.data_root)
    imgs, lbls = _filter_classes(raw_x, raw_y, pool["classes"])
    lbls = _remap(lbls, pool["classes"])
    unused = np.array([hashlib.sha1(np.ascontiguousarray(imgs[i]).tobytes()).hexdigest() not in pool_hashes
                       for i in range(len(imgs))])
    rng = np.random.default_rng(0)
    thr = {}
    for c in sorted(set(ytr.tolist()) - {0}):
        cand = np.where(unused & (lbls == c))[0]
        pick = rng.choice(cand, size=min(500, len(cand)), replace=False)
        ref = to_normalized_tensor(imgs[pick], spec.channels, cfg.image_size, cfg.normalize_range)
        thr[c] = float(np.percentile(_ratio_q(ref, Xtr[ytr == c]), 5))
    return thr


def audit_images():
    import pandas as pd
    import torch
    from casestudy.eval.oracle import load_oracle
    from kbs2.runner import load_regime, split

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, checks = [], []
    for ds, (rid, _r) in P.ANCHORS.items():
        D = load_regime(rid)
        cfg, spec, pool = D["config"], D["spec"], D["pool"]
        oracle = load_oracle(ds, device)
        checks.append(verify_oracle(ds, oracle, cfg, spec))
        tr, _ho = split(pool, cfg)
        Xtr, ytr = pool["images"][tr], pool["labels"][tr]
        thr = near_copy_thresholds(ds, D, Xtr, ytr)
        for exp, tau in EXPS:
            for s in P.SEEDS:
                path = os.path.join(ROOT, "saved_kbs2", exp, "synth", f"{P.PROPOSED}__{rid}__s{s}.pt")
                rec = torch.load(path, weights_only=False)  # written by kbs2.runner.generate
                X, y, meta = rec["X"], rec["y"].numpy(), rec["admitted_meta"]
                if not (len(meta["class"]) == len(y) and np.array_equal(meta["class"], y)):
                    raise RuntimeError(f"admitted metadata not aligned with the synthetic set: {path}")
                probs = oracle(X).numpy()
                q = np.empty(len(y))
                for c in thr:
                    idx = np.where(y == c)[0]
                    q[idx] = _ratio_q(X[idx], Xtr[ytr == c])
                near = np.array([q[i] < thr[int(y[i])] for i in range(len(y))])
                for i in range(len(y)):
                    rows.append({"exp": exp, "tau": tau, "dataset": ds, "regime": rid, "seed": s,
                                 "class": int(y[i]), "cycle": int(meta["cycle"][i]),
                                 "passed_gate": bool(meta["passed_gate"][i]), "p_c_internal": float(meta["p_c"][i]),
                                 "agree": bool(probs[i].argmax() == y[i]), "ref_conf": float(probs[i, y[i]]),
                                 "near_copy": bool(near[i])})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def rule1(df):
    import pandas as pd
    out = []
    for subset, d0 in [("all admitted", df), ("non-near-copy only", df[~df["near_copy"]])]:
        for (tau, ds), g in d0.groupby(["tau", "dataset"]):
            per = g.groupby(["seed", "passed_gate"])["agree"].mean().unstack()
            both = per.dropna()
            row = {"subset": subset, "tau": tau, "dataset": ds,
                   "gated_agreement": float(g[g["passed_gate"]]["agree"].mean()) if g["passed_gate"].any() else np.nan,
                   "fallback_agreement": float(g[~g["passed_gate"]]["agree"].mean()) if (~g["passed_gate"]).any() else np.nan,
                   "n_gated": int(g["passed_gate"].sum()), "n_fallback": int((~g["passed_gate"]).sum()),
                   "seeds_with_both": int(len(both))}
            if len(both) >= 2 and True in both.columns and False in both.columns:
                d = (both[True] - both[False]).to_numpy(float)
                w, p = wilcoxon_p(d)
                hl, lo, hi = hodges_lehmann(d)
                if lo >= -MARGIN and hi <= MARGIN:
                    verdict = "no meaningful loss" + (" (CI excludes 0)" if lo > 0 or hi < 0 else "")
                elif lo > 0:
                    verdict = "loss"
                else:
                    verdict = "inconclusive"
                row.update(mean_diff=float(d.mean()), hl=hl, ci_low=lo, ci_high=hi, p=p, verdict=verdict)
            out.append(row)
    return pd.DataFrame(out)


def rule2(df):
    import pandas as pd
    per = df.groupby(["tau", "dataset", "seed"])["agree"].mean().groupby(["tau", "dataset"]).mean().unstack(0)
    avg = per.mean(0)
    diff = float(avg.get(0.6, np.nan) - avg.get(0.5, np.nan))
    chosen = 0.6 if diff >= RULE2_MIN else 0.5
    table = per.reset_index()
    table.loc[len(table)] = ["mean over datasets", avg.get(0.5, np.nan), avg.get(0.6, np.nan)]
    return table, {"agreement_tau06": float(avg.get(0.6, np.nan)), "agreement_tau05": float(avg.get(0.5, np.nan)),
                   "difference_06_minus_05": diff, "rule": "tau 0.6 only if higher by >= 0.01, else tau 0.5",
                   "chosen_tau": chosen}


def _verdict(lo: float, hi: float) -> str:
    return "favourable" if lo > 0 else ("unfavourable" if hi < 0 else "inconclusive")


def v2c() -> None:
    """V2c (approval saved_kbs2/approvals/V2c.json): candidate-level audit. tau 0.6 (E1) is primary,
    tau 0.5 (E1b) a sensitivity analysis reported alongside it.

    Q1 gate: agreement of candidates passing the gate minus failing it (all class-cycles).
    Q2 ranking in fallback: in fallback class-cycles, admitted minus non-admitted candidates.
    Q3 selection: all admitted minus all candidates (= expected agreement of random admission).
    Per seed = image-level means over that seed's class-cycles; paired over seeds; HL + exact 95% CI.
    Candidates were stored as uint8 (kbs2.instrumented.to_uint8) and are mapped back to [-1, 1].
    """
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from casestudy.eval.oracle import load_oracle

    out = os.path.join(ROOT, "saved_kbs2", "analysis", "V2c")
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for ds, (rid, _r) in P.ANCHORS.items():
        oracle = load_oracle(ds, device)
        for exp, tau in EXPS:
            for s in P.SEEDS:
                z = np.load(os.path.join(ROOT, "saved_kbs2", exp, "candidates", f"{P.PROPOSED}__{rid}__s{s}.npz"),
                            allow_pickle=False)
                keys = sorted({k.split("_", 1)[0] for k in z.files})
                for k in keys:
                    c = int(z[f"{k}_class"])
                    x = torch.from_numpy(z[f"{k}_images_uint8"].astype(np.float32) / 127.5 - 1.0)
                    agree = oracle(x).argmax(1).numpy() == c
                    passed = z[f"{k}_passed_gate"].astype(bool)
                    adm = np.zeros(len(passed), bool)
                    adm[z[f"{k}_admitted_index"]] = True
                    fallback = bool(passed.sum() < adm.sum())
                    for i in range(len(passed)):
                        rows.append((exp, tau, ds, s, c, int(z[f"{k}_cycle"]), fallback,
                                     bool(passed[i]), bool(adm[i]), float(z[f"{k}_p_c"][i]), bool(agree[i])))
    df = pd.DataFrame(rows, columns=["exp", "tau", "dataset", "seed", "class", "cycle", "fallback_cycle",
                                     "passed_gate", "admitted", "p_c", "agree"])
    df.to_csv(os.path.join(out, "candidates.csv.gz"), index=False)

    def per_seed(g, mask_a, mask_b):
        a = g[mask_a(g)].groupby("seed")["agree"].mean()
        b = g[mask_b(g)].groupby("seed")["agree"].mean()
        j = a.index.intersection(b.index)
        return a[j], b[j]

    tests = {
        "Q1 gate: passed minus failed": (lambda g: g, lambda g: g["passed_gate"], lambda g: ~g["passed_gate"]),
        "Q2 fallback cycles: admitted minus not admitted": (lambda g: g[g["fallback_cycle"]],
                                                            lambda g: g["admitted"], lambda g: ~g["admitted"]),
        "Q3 selection: admitted minus all candidates": (lambda g: g, lambda g: g["admitted"],
                                                        lambda g: np.ones(len(g), bool)),
    }
    res = []
    for (tau, ds), g0 in df.groupby(["tau", "dataset"]):
        for name, (sub, ma, mb) in tests.items():
            g = sub(g0)
            a, b = per_seed(g, ma, mb)
            row = {"tau": tau, "role": "primary" if tau == 0.6 else "sensitivity", "dataset": ds, "test": name,
                   "agree_a": float(a.mean()) if len(a) else np.nan, "agree_b": float(b.mean()) if len(b) else np.nan,
                   "seeds": int(len(a))}
            if len(a) >= 2:
                d = (a - b).to_numpy(float)
                _w, p = wilcoxon_p(d)
                hl, lo, hi = hodges_lehmann(d)
                row.update(hl=hl, ci_low=lo, ci_high=hi, p=p, verdict=_verdict(lo, hi))
            res.append(row)
        aucs = [roc_auc_score(g["agree"], g["p_c"]) for _s, g in g0.groupby("seed") if g["agree"].nunique() == 2]
        res.append({"tau": tau, "role": "primary" if tau == 0.6 else "sensitivity", "dataset": ds,
                    "test": "AUROC of p_c for reference agreement (descriptive)", "agree_a": float(np.mean(aucs)),
                    "ci_low": float(np.min(aucs)), "ci_high": float(np.max(aucs)), "seeds": len(aucs),
                    "verdict": "mean; low/high = range over seeds"})
    r = pd.DataFrame(res)
    r.to_csv(os.path.join(out, "v2c_tests.csv"), index=False)
    pd.set_option("display.width", 250)
    print(r.round(3).to_string(index=False))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "V2c":
        v2c()
        return
    os.makedirs(OUT, exist_ok=True)
    imgs, checks = audit_images()
    imgs.to_csv(os.path.join(OUT, "admitted_images.csv"), index=False)
    checks.to_csv(os.path.join(OUT, "oracle_check.csv"), index=False)
    r1 = rule1(imgs)
    r1.to_csv(os.path.join(OUT, "rule1_gated_vs_fallback.csv"), index=False)
    by_cycle = imgs.groupby(["tau", "dataset", "cycle", "passed_gate"])["agree"].agg(["mean", "size"]).reset_index()
    by_cycle.to_csv(os.path.join(OUT, "agreement_by_cycle.csv"), index=False)
    t2, dec = rule2(imgs)
    t2.to_csv(os.path.join(OUT, "rule2_agreement_by_tau.csv"), index=False)
    with open(os.path.join(OUT, "rule2_decision.json"), "w", encoding="utf-8") as fh:
        json.dump(dec, fh, indent=2)
    print(checks.to_string(index=False))
    print(r1.round(3).to_string(index=False))
    print(t2.round(3).to_string(index=False))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
