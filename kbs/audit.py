"""kbs.audit - the independent label audit (S5; Reviewer 3, comments 1 and 3; Reviewer 1, comment 6).

    python -m kbs.audit            # all outputs that the finished runs allow
    python -m kbs.audit --figure   # also write the reliability figure (300 dpi PNG) into the paper

Reads the candidate records (``candidates.npz``) and results of E4 (selection arms, frozen generator)
and E5 (confidence strata), together with the reference classifiers' accuracy from E1:

A1  agreement with the reference classifier against internal confidence: 10 bins with Wilson
    intervals, AUROC of confidence for agreement, and the rank trend over bins (proposed rule's runs)
A2  the reviewer's two error rates: correct labels among gate-failing candidates, wrong labels among
    gate-passing candidates
A3  agreement of the admitted samples, by selection arm
A4  agreement of gate-passing against fallback-admitted samples
A5  downstream macro-F2 for the confidence terciles (conf_low / conf_mid / conf_high)
A6  downstream macro-F2 for low- against high-confidence candidates the reference classifier accepts
A7  diversity of the admitted sets: mean pairwise cosine distance of the reference classifier's
    features, per class, by arm (``--diversity``; needs the GPU and the trained reference classifiers)
A8  the self-training comparison (R3.1, replacing a separate arm): the share of candidates the internal
    classifier assigns to the conditioned class, to the majority class and to the other minority class,
    overall and among confident predictions (max probability >= tau, what self-training would keep)

Writes saved_kbs/audit/summary.json (read by the decision gate, Part 5.2; missing values are null)
and audit_*.csv tables.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from kbs import PAPER_DIR
from casestudy.eval.stats_v2 import compare
from casestudy.protocol_v2 import save_root
from kbs.describe_splits import NAMES

TAU = 0.6
FIGURE = PAPER_DIR / "figures" / "final" / "fig_audit.png"


def _clean(v):
    """NaN -> None, recursively, so that summary.json is strict JSON."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return None if isinstance(v, float) and v != v else v


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def _runs(stage: str):
    root = save_root() / "runs" / stage
    for f in sorted(root.rglob("result.json")) if root.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") == "ok":
            yield f.parent, r


def _candidates(stage: str) -> pd.DataFrame:
    frames = []
    for d, r in _runs(stage):
        f = d / "candidates.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if "oracle_pred" not in z:
            continue
        df = pd.DataFrame({k: z[k] for k in z.files})
        df["dataset"], df["arm"], df["seed"] = r["dataset"], r["variant"], r["seed"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def a1_a2(cand: pd.DataFrame) -> dict:
    out = {}
    for ds, g in cand[cand.arm == "gate_diverse"].groupby("dataset"):
        agree = (g["oracle_pred"] == g["cls"]).to_numpy()
        conf = g["p_c"].to_numpy()
        bins = np.minimum((conf * 10).astype(int), 9)
        curve = []
        for b in range(10):
            m = bins == b
            lo, hi = wilson(int(agree[m].sum()), int(m.sum()))
            curve.append({"bin": b, "n": int(m.sum()), "agree": float(agree[m].mean()) if m.any() else float("nan"),
                          "lo": lo, "hi": hi})
        used = [c for c in curve if c["n"] >= 20]
        trend = spearmanr([c["bin"] for c in used], [c["agree"] for c in used])[0] if len(used) >= 3 else float("nan")
        gate = ((g["pred"] == g["cls"]) & (g["p_c"] >= TAU)).to_numpy()
        out[ds] = {"auroc": float(roc_auc_score(agree, conf)) if 0 < agree.mean() < 1 else float("nan"),
                   "trend": float(trend), "curve": curve, "n_candidates": int(len(g)),
                   "correct_among_gate_failing": float(agree[~gate].mean()) if (~gate).any() else float("nan"),
                   "wrong_among_gate_passing": float(1 - agree[gate].mean()) if gate.any() else float("nan"),
                   "n_gate_failing": int((~gate).sum()), "n_gate_passing": int(gate.sum())}
    return out


def a3_a4(cand: pd.DataFrame) -> pd.DataFrame:
    adm = cand[cand.admitted.astype(bool)].copy()
    adm["agree"] = adm["oracle_pred"] == adm["cls"]
    adm["gate"] = (adm["pred"] == adm["cls"]) & (adm["p_c"] >= TAU)
    by_arm = adm.groupby(["dataset", "arm"]).agg(n=("agree", "size"), agree=("agree", "mean"),
                                                 share_violating=("gate", lambda s: 1 - s.mean()))
    fb = adm[adm.arm == "gate_diverse"].groupby(["dataset", "gate"])["agree"].agg(["size", "mean"])
    return by_arm.reset_index(), fb.reset_index()


def a8(cand: pd.DataFrame) -> dict:
    """Where the internal classifier puts the proposed rule's candidates. A candidate predicted as another class
    is what self-training would relabel (if confident) and the gate rejects; the share shows how far the two differ.
    Confidence in the predicted class is p_c when the prediction is the conditioned class; otherwise it is at least
    p_c + margin, so the confident share of reassignments is a lower bound."""
    out = {}
    for ds, g in cand[cand.arm == "gate_diverse"].groupby("dataset"):
        same, major = (g["pred"] == g["cls"]), (g["pred"] == 0)
        other = ~same & ~major
        conf_same = same & (g["p_c"] >= TAU)
        conf_other = other & ((g["p_c"] + g["margin"]) >= TAU)
        conf_major = major & ((g["p_c"] + g["margin"]) >= TAU)
        out[ds] = {"n": int(len(g)), "pred_conditioned": float(same.mean()), "pred_majority": float(major.mean()),
                   "pred_other_minority": float(other.mean()), "confident_conditioned": float(conf_same.mean()),
                   "confident_majority_lb": float(conf_major.mean()), "confident_other_minority_lb": float(conf_other.mean())}
    return out


def a5_a6() -> pd.DataFrame:
    rows = [{"dataset": r["dataset"], "stratum": r["variant"], "seed": r["seed"],
             "macro_f2": r["metrics"][r["primary"]]["macro_f2"]} for _, r in _runs("E5_strata")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    out = []
    for ds, g in df.groupby("dataset"):
        w = g.pivot(index="seed", columns="stratum", values="macro_f2").dropna()
        if {"conf_high", "conf_low", "conf_mid"} <= set(w.columns) and len(w) >= 5:
            for row in compare(w[["conf_high", "conf_low", "conf_mid"]], "conf_high").to_dict("records"):
                out.append({"dataset": ds, "contrast": f"conf_high vs {row['rival']}", **row})
        if {"correct_low", "correct_high"} <= set(w.columns) and len(w) >= 5:
            for row in compare(w[["correct_low", "correct_high"]], "correct_low").to_dict("records"):
                out.append({"dataset": ds, "contrast": f"correct_low vs {row['rival']}", **row})
    return pd.DataFrame(out)


def a7(device: str) -> pd.DataFrame:
    """Mean pairwise cosine distance among the admitted images of each class, in reference-feature space."""
    import torch
    import torch.nn.functional as F

    from casestudy.eval.oracle import load_oracle

    refs, rows = {}, []
    for d, r in _runs("E4_selection"):
        f = d / "synth.npz"
        ds = r["dataset"]
        if ds not in refs:
            refs[ds] = load_oracle(ds, device)
        if refs[ds] is None or not f.exists():
            continue
        z = np.load(f)
        x, y = torch.from_numpy(z["x"]).float() / 127.5 - 1.0, z["y"]
        net = refs[ds].members[0]
        with torch.no_grad():
            h = torch.cat([net.features(x[i:i + 256].to(device)).cpu() for i in range(0, len(x), 256)])
        for c in np.unique(y):
            hc = F.normalize(h[torch.from_numpy(y == c)], dim=1)
            n = len(hc)
            if n >= 2:
                mean_sim = float(((hc @ hc.T).sum() - n) / (n * (n - 1)))
                rows.append({"dataset": ds, "arm": r["variant"], "seed": r["seed"], "cls": int(c), "n": n,
                             "diversity": 1.0 - mean_sim})
    return pd.DataFrame(rows)


def a9(device: str, stages=("E3_anchors", "E6_domain")) -> pd.DataFrame:
    """Label agreement of the images each generative method added in the main comparisons: the share that the
    reference classifier assigns to the class they were generated for, and the share it assigns to the majority
    class, per run and minority class. Read against macro-F2, it tests the premise that wrong labels hurt."""
    import torch

    from casestudy.eval.oracle import load_oracle
    from kbs.realism import GENERATIVE

    refs, rows = {}, []
    for stage in stages:
        for d, r in _runs(stage):
            f = d / "synth.npz"
            if r["method"] not in GENERATIVE or not f.exists():
                continue
            ds = r["dataset"]
            if ds not in refs:
                refs[ds] = load_oracle(ds, device)
            if refs[ds] is None:
                continue
            z = np.load(f)
            pred = refs[ds](torch.from_numpy(z["x"]).float() / 127.5 - 1.0).argmax(1).numpy()
            for c in np.unique(z["y"]):
                m = z["y"] == c
                rows.append({"stage": stage, "regime": r["regime"], "dataset": ds, "method": r["method"],
                             "variant": r["variant"], "seed": r["seed"], "cls": int(c), "n": int(m.sum()),
                             "agree": float((pred[m] == c).mean()), "to_majority": float((pred[m] == 0).mean())})
    return pd.DataFrame(rows)


def reference_accuracy() -> dict:
    acc = {}
    for d in (save_root() / "oracle").glob("*/summary.json"):
        s = json.loads(d.read_text(encoding="utf-8"))
        acc[s["dataset"]] = s["ensemble"]["accuracy"]
    return acc


def figure(a1: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(a1), figsize=(3.2 * len(a1), 2.8), sharey=True)
    for ax, (ds, v) in zip(np.atleast_1d(axes), sorted(a1.items())):
        c = [x for x in v["curve"] if x["n"] >= 20]
        x = [(x["bin"] + 0.5) / 10 for x in c]
        ax.errorbar(x, [k["agree"] for k in c], yerr=[[k["agree"] - k["lo"] for k in c], [k["hi"] - k["agree"] for k in c]],
                    marker="o", ms=3, lw=1, capsize=2)
        ax.axvline(TAU, ls=":", lw=0.8, color="grey")
        ax.set_title(f"{NAMES.get(ds, ds)} (AUROC {v['auroc']:.2f})", fontsize=9)
        ax.set_xlabel("internal confidence p(c)")
    np.atleast_1d(axes)[0].set_ylabel("agreement with reference")
    fig.tight_layout()
    fig.savefig(path, dpi=300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure", nargs="?", const=str(FIGURE), default=None, metavar="PATH",
                    help=f"write the reliability figure (default path {FIGURE})")
    ap.add_argument("--diversity", action="store_true", help="also compute A7 (GPU, reference classifiers)")
    ap.add_argument("--generators", action="store_true",
                    help="also compute A9, the label agreement of every generative method's images in E3 and E6 (GPU)")
    args = ap.parse_args()
    out = save_root() / "audit"
    out.mkdir(parents=True, exist_ok=True)
    cand = _candidates("E4_selection")
    ref = reference_accuracy()
    summary = {}
    if not cand.empty:
        a1 = a1_a2(cand)
        assign = a8(cand)
        for ds, v in a1.items():
            summary[ds] = {"ref_acc": ref.get(ds), **{k: v[k] for k in v if k != "curve"},
                           **{f"a8_{k}": x for k, x in assign.get(ds, {}).items()}}
        by_arm, fb = a3_a4(cand)
        by_arm.to_csv(out / "audit_by_arm.csv", index=False)
        fb.to_csv(out / "audit_fallback.csv", index=False)
        pd.DataFrame([{"dataset": ds, **c} for ds, v in a1.items() for c in v["curve"]]).to_csv(
            out / "audit_reliability.csv", index=False)
        print("A1/A2:", json.dumps(_clean(summary), indent=1))
        print("\nA3 agreement of admitted samples by arm:\n", by_arm.round(3).to_string(index=False))
        print("\nA4 gate-passing (True) vs fallback-admitted (False), proposed rule:\n", fb.round(3).to_string(index=False))
        if args.figure:
            Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
            figure(a1, Path(args.figure))
            print("figure:", args.figure)
    else:
        print("[PENDING] no E4 candidate records with reference verdicts yet")
    strata = a5_a6()
    if not strata.empty:
        strata.to_csv(out / "audit_strata.csv", index=False)
        print("\nA5/A6 strata:\n", strata.round(4).to_string(index=False))
    if args.diversity:
        import torch
        div = a7("cuda" if torch.cuda.is_available() else "cpu")
        if div.empty:
            print("\n[PENDING] A7: no E4 synthetic sets with a trained reference classifier yet")
        else:
            div.to_csv(out / "audit_diversity.csv", index=False)
            per_seed = div.groupby(["dataset", "arm", "seed"])["diversity"].mean().reset_index()
            print("\nA7 diversity of admitted sets (mean over classes, then seeds):\n",
                  per_seed.groupby(["dataset", "arm"])["diversity"].agg(["mean", "std", "size"]).round(4).to_string())
    if args.generators:
        import torch
        gen = a9("cuda" if torch.cuda.is_available() else "cpu")
        if gen.empty:
            print("\n[PENDING] A9: no synthetic sets of E3 or E6 with a trained reference classifier yet")
        else:
            gen.to_csv(out / "audit_generators.csv", index=False)
            print("\nA9 label agreement of the added images, by regime and method (mean over classes and seeds):\n",
                  gen.groupby(["regime", "method"])[["agree", "to_majority"]].mean().round(3).to_string())
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=1), encoding="utf-8")
    print("\nwritten:", out / "summary.json")


if __name__ == "__main__":
    main()
