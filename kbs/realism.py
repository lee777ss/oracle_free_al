"""kbs.realism - sample realism of every method (E10; Reviewer 1, comment 9; claim C5 of the plan).

    python -m kbs.realism run --stage E3_anchors --seeds 42 43 44 45 46
    python -m kbs.realism summary --stage E3_anchors
    python -m kbs.realism copies --stage E3_anchors --seeds 42 ... 61
    python -m kbs.realism kid-by-copy --stage E3_anchors

For each run it scores the synthetic images the method actually added to training (``synth.npz``),
per minority class, against **all** real training images of that class in the official split:

* KID (kernel inception distance; unbiased, so it is reliable with the ~100-500 images a run adds),
  averaged over random subsets, with its standard deviation;
* FID on the same images, for continuity with the submitted paper (biased at these sample sizes).

Real-image Inception features are computed once per (dataset, class) and cached. Rows already in
the output CSV are skipped, so the script can be re-run as more runs finish. ``summary`` relates
minority-class realism to macro-F2 across the generative methods, per dataset.

``copies`` measures how close the synthetic images are to individual training images (the
memorisation check of the sample figures), in pixel space as the figures show them. For a synthetic
image x of class c, with T the run's real training images of the class:

* specificity d1/d2: distance to the nearest image of T over the distance to the second nearest. A
  reconstruction of one training image matches it much better than any other (small d1/d2); a
  generic, blurred or noise image is about as far from its nearest as from its second nearest;
* near-copy: d1/d2 below the 5th percentile of the same ratio for real images of the class that the
  run never saw (reservoir images outside its pool, Q). Unseen real images are near-copies at the
  rate of about 5%, and the CSV gives that rate cross-fitted (thresholds from one half of Q, rate on
  the other), so a method's share can be read against it;
* the median of d(x, T) / d(x, H), with H an equally large set of unseen real images, and the share
  of T that is the nearest neighbour of some synthetic image, as secondary descriptions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casestudy.data.transforms import to_normalized_tensor
from casestudy.eval.fid import _extract_features, _frechet_distance, _get_inception
from casestudy.protocol_v2 import REGIMES_V2, _raw, make_config, regime_spec, save_root

GENERATIVE = ["m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan", "m08_bagan_ls", "m09_bagan_gp",
              "p_proposed"]


def _real_features(dataset: str, regime: str, cls: int, device: str) -> np.ndarray:
    cache = save_root() / "realism" / f"real_{dataset}_c{cls}.npy"
    if cache.exists():
        return np.load(cache)
    cfg = make_config()
    spec = regime_spec(regime, cfg)
    raw = _raw(dataset, spec, cfg)
    x = np.concatenate([raw["res_x"][raw["res_y"] == cls], raw["dev_x"][raw["dev_y"] == cls]])
    feats = _extract_features(to_normalized_tensor(x, spec.channels, cfg.image_size, cfg.normalize_range),
                              _get_inception(device), device, batch_size=200)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


def kid(f_real: np.ndarray, f_fake: np.ndarray, n_subsets: int = 100, subset: int = 1000, seed: int = 0):
    """Unbiased squared MMD with the cubic polynomial kernel (Binkowski et al. 2018): mean and sd over subsets."""
    rng = np.random.default_rng(seed)
    d = f_real.shape[1]
    m = min(subset, len(f_real), len(f_fake))
    vals = []
    for _ in range(n_subsets):
        x = f_real[rng.choice(len(f_real), m, replace=False)]
        y = f_fake[rng.choice(len(f_fake), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def fid(f_real: np.ndarray, f_fake: np.ndarray) -> float:
    return _frechet_distance(f_real.mean(0), np.cov(f_real, rowvar=False), f_fake.mean(0), np.cov(f_fake, rowvar=False))


def run(stage: str, seeds, device: str, regimes=None) -> Path:
    out = save_root() / "realism" / f"{stage}.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.regime, r.method, int(r.seed), int(r["class"])) for _, r in prev.iterrows()}
    inception = _get_inception(device)
    rows = []
    for f in sorted((save_root() / "runs" / stage).rglob("synth.npz")):
        res = json.loads((f.parent / "result.json").read_text(encoding="utf-8"))
        if (res.get("status") != "ok" or res["method"] not in GENERATIVE or res["seed"] not in seeds
                or (regimes and res["regime"] not in regimes)):
            continue
        data = np.load(f)
        x = torch.from_numpy(data["x"]).float() / 127.5 - 1.0
        y = data["y"]
        for c in sorted(set(int(v) for v in np.unique(y))):
            key = (res["regime"], res["method"], int(res["seed"]), c)
            if key in done:
                continue
            f_real = _real_features(res["dataset"], res["regime"], c, device)
            f_fake = _extract_features(x[y == c], inception, device, batch_size=200)
            k_mean, k_sd = kid(f_real, f_fake)
            rows.append({"regime": res["regime"], "dataset": res["dataset"], "method": res["method"], "seed": res["seed"],
                         "class": c, "n_fake": int((y == c).sum()), "kid": k_mean, "kid_sd": k_sd,
                         "fid": fid(f_real, f_fake) if (y == c).sum() > 2 else float("nan")})
            print(f"{res['regime']:22s} {res['method']:14s} s{res['seed']} c{c}: KID {k_mean:.4f}  FID {rows[-1]['fid']:.1f}", flush=True)
    if rows:
        new = pd.DataFrame(rows)
        new.to_csv(out, mode="a", header=not out.exists(), index=False)
    return out


def _nn2(q: np.ndarray, ref: np.ndarray):
    """Pixel-space (0-255) distances from each query to its nearest and second-nearest reference image,
    and the index of the nearest."""
    q = q.reshape(len(q), -1).astype(np.float64)
    r = ref.reshape(len(ref), -1).astype(np.float64)
    d = np.sqrt(np.maximum((q * q).sum(1)[:, None] + (r * r).sum(1)[None, :] - 2 * q @ r.T, 0))
    s = np.sort(d, axis=1)
    return s[:, 0], s[:, 1], d.argmin(1)


def _copy_stats(X: np.ndarray, T: np.ndarray, H: np.ndarray):
    d1, d2, nn = _nn2(X, T)
    return d1 / np.maximum(d2, 1e-9), d1 / np.maximum(_nn2(X, H)[0], 1e-9), nn


def _reference_sets(regime: str, seed: int, n_calib: int) -> dict:
    """{class: (T, H, specificity of Q, cross-fitted near-copy rate of Q)} for the minority classes of one run."""
    from casestudy.protocol_v2 import _to_uint8, get_regime, sample_pool
    from kbs.sample_figure import training_images

    cfg = make_config()
    rd = get_regime(regime, cfg)
    pool = sample_pool(rd.res_y, rd.spec.majority_cap, rd.rho, seed)
    rest = np.setdiff1d(np.arange(len(rd.res_y)), pool)
    xt, yt = training_images(regime, seed)
    out = {}
    for c in sorted(int(v) for v in np.unique(yt) if int(v) != 0):
        T = xt[yt == c]
        perm = np.random.default_rng([seed, c]).permutation(rest[rd.res_y[rest] == c])
        conv = lambda i: _to_uint8(to_normalized_tensor(rd.res_x[i], rd.spec.channels, cfg.image_size,
                                                        cfg.normalize_range))
        H, Q = conv(perm[:len(T)]), conv(perm[len(T):len(T) + n_calib])
        spec_q = _copy_stats(Q, T, H)[0]
        a, b = spec_q[::2], spec_q[1::2]
        cross = ((b < np.quantile(a, 0.05)).mean() + (a < np.quantile(b, 0.05)).mean()) / 2
        out[c] = (T, H, spec_q, float(cross))
    return out


def copies(stage: str, seeds, regimes=None, n_calib: int = 300) -> Path:
    out = save_root() / "realism" / f"{stage}_copies.csv"
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.regime, r.method, str(r.variant) if r.variant == r.variant else "", int(r.seed), int(r["class"]))
                for _, r in prev.iterrows()}
    runs = []
    for f in sorted((save_root() / "runs" / stage).rglob("synth.npz")):
        res = json.loads((f.parent / "result.json").read_text(encoding="utf-8"))
        if (res.get("status") == "ok" and res["method"] in GENERATIVE and res["seed"] in seeds
                and (not regimes or res["regime"] in regimes)):
            runs.append((res["regime"], res["seed"], res, f))
    key = lambda regime, res, seed, c: (regime, res["method"], res.get("variant", ""), int(seed), c)
    rows = []
    for (regime, seed), group in pd.DataFrame(runs, columns=["regime", "seed", "res", "f"]).groupby(["regime", "seed"]):
        todo = [(r, f) for r, f in zip(group.res, group.f) if any(key(regime, r, seed, c) not in done for c in (1, 2))]
        if not todo:
            continue
        ref = _reference_sets(regime, int(seed), n_calib)
        for c, (T, H, spec_q, cross) in ref.items():
            if (regime, "real_unseen", "", int(seed), c) not in done:
                rows.append({"regime": regime, "dataset": todo[0][0]["dataset"], "method": "real_unseen", "variant": "",
                             "seed": int(seed), "class": c, "n_fake": len(spec_q), "n_train": len(T),
                             "near_copy": cross, "spec_median": float(np.median(spec_q))})
        for res, f in todo:
            data = np.load(f)
            for c, (T, H, spec_q, _) in ref.items():
                if key(regime, res, seed, c) in done or not (data["y"] == c).any():
                    continue
                spec, ratio, nn = _copy_stats(data["x"][data["y"] == c], T, H)
                rows.append({"regime": regime, "dataset": res["dataset"], "method": res["method"],
                             "variant": res.get("variant", ""), "seed": int(seed), "class": c, "n_fake": len(spec),
                             "n_train": len(T),
                             "near_copy": float((spec < np.quantile(spec_q, 0.05)).mean()),
                             "spec_median": float(np.median(spec)), "ratio_median": float(np.median(ratio)),
                             "nn_coverage": len(np.unique(nn)) / len(T)})
        print(f"{regime:22s} s{seed}: {len(todo)} runs", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    if out.exists():
        d = pd.read_csv(out)
        for regime, g in d.groupby("regime"):
            g = g.assign(method=[m + (f"+{v}" if v == v and v else "") for m, v in zip(g.method, g.variant)])
            m = g.groupby("method")[["near_copy", "spec_median", "ratio_median", "nn_coverage"]].mean()
            print(f"\n{regime}: {g.seed.nunique()} seeds")
            print(m.sort_values("near_copy").round(3).to_string())
    return out


def kid_by_copy(stage: str, seeds, device: str, n_calib: int = 300) -> Path:
    """KID of the proposed method's near-copies and of its other images, separately, against all real images of
    the class (images pooled over ``seeds``), next to BAGAN-GP's pooled over the same seeds: how much of its
    realism is the realism of reproduced training images."""
    out = save_root() / "realism" / f"{stage}_kid_by_copy.csv"
    inception = _get_inception(device)
    groups = {}
    regimes = sorted({json.loads(f.read_text(encoding="utf-8"))["regime"]
                      for f in (save_root() / "runs" / stage).glob("*/p_proposed+frozen/s*/result.json")})
    for regime in regimes:
        for seed in sorted(seeds):
            d_p = save_root() / "runs" / stage / regime / "p_proposed+frozen" / f"s{seed}"
            d_b = save_root() / "runs" / stage / regime / "m09_bagan_gp" / f"s{seed}"
            if not ((d_p / "synth.npz").exists() and (d_b / "synth.npz").exists()):
                continue
            ref = _reference_sets(regime, seed, n_calib)
            zp, zb = np.load(d_p / "synth.npz"), np.load(d_b / "synth.npz")
            for c, (T, H, spec_q, _) in ref.items():
                S = zp["x"][zp["y"] == c]
                copy = _copy_stats(S, T, H)[0] < np.quantile(spec_q, 0.05)
                for g, x in (("proposed: near-copies", S[copy]), ("proposed: other images", S[~copy]),
                             ("proposed: all", S), ("BAGAN-GP: all", zb["x"][zb["y"] == c])):
                    groups.setdefault((regime, c, g), []).append(x)
    rows = []
    for (regime, c, g), xs in groups.items():
        x = np.concatenate(xs)
        if len(x) < 10:
            continue
        ds = REGIMES_V2[regime]["dataset"]
        f_fake = _extract_features(torch.from_numpy(x).float() / 127.5 - 1.0, inception, device, batch_size=200)
        k_mean, k_sd = kid(_real_features(ds, regime, c, device), f_fake)
        rows.append({"regime": regime, "dataset": ds, "class": c, "group": g, "n": len(x), "kid": k_mean, "kid_sd": k_sd})
        print(f"{regime:22s} c{c} {g:24s} n={len(x):5d}  KID {k_mean:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def summary(stage: str) -> None:
    from scipy.stats import spearmanr

    real = pd.read_csv(save_root() / "realism" / f"{stage}.csv")
    rows = []
    for f in (save_root() / "runs" / stage).rglob("result.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") == "ok" and r["method"] in GENERATIVE:
            rows.append({"regime": r["regime"], "method": r["method"], "seed": r["seed"],
                         "macro_f2": r["metrics"][r["primary"]]["macro_f2"]})
    util = pd.DataFrame(rows)
    per = real.groupby(["regime", "method", "seed"])[["kid", "fid"]].mean().reset_index()
    both = per.merge(util, on=["regime", "method", "seed"])
    for regime, g in both.groupby("regime"):
        m = g.groupby("method")[["kid", "fid", "macro_f2"]].mean().sort_values("kid")
        rho, p = spearmanr(m["kid"], m["macro_f2"])
        print(f"\n{regime}: Spearman(KID, macro-F2) over methods = {rho:+.2f} (p = {p:.2f}); "
              "negative = more realistic goes with more useful")
        print(m.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--stage", required=True); r.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    r.add_argument("--regimes", nargs="*", help="only these regimes (default: all)")
    s = sub.add_parser("summary"); s.add_argument("--stage", required=True)
    c = sub.add_parser("copies"); c.add_argument("--stage", required=True)
    c.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 62)))
    c.add_argument("--regimes", nargs="*", help="only these regimes (default: all)")
    k = sub.add_parser("kid-by-copy"); k.add_argument("--stage", required=True)
    k.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "run":
        print("written:", run(args.stage, set(args.seeds), device, args.regimes))
    elif args.cmd == "copies":
        print("written:", copies(args.stage, set(args.seeds), args.regimes))
    elif args.cmd == "kid-by-copy":
        print("written:", kid_by_copy(args.stage, set(args.seeds), device))
    else:
        summary(args.stage)


if __name__ == "__main__":
    main()
