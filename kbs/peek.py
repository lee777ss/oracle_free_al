"""kbs.peek - a look at the results while a stage is still running, with alarms for implausible runs.

    python -m kbs.peek --stage E3_anchors            # mean macro-F2 per method and regime so far
    python -m kbs.peek --stage E3_anchors --alarms   # only the alarms

For monitoring only: tables are built by ``kbs.tables`` from complete cells. Alarms:

* CONSTANT  a method whose primary macro-F2 is identical across at least three seeds (nothing varies with the seed);
* LOW       a method more than 0.10 below the class-weighted baseline on the same regime and seeds;
* SYNTH     an augmenter that added fewer images than its target (shortfall), or none;
* OUTLIER   a run more than 4 robust standard deviations from its method's median on the regime;
* NOISE     a generator whose images are more than 2.5 times as rough (mean absolute difference of neighbouring
            pixels) as the median generator on the regime, as BAGAN without label smoothing was on Fashion-MNIST.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casestudy.protocol_v2 import save_root


GENERATIVE = ("m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan", "m08_bagan_ls", "m09_bagan_gp",
              "p_proposed")


def load(stage: str) -> pd.DataFrame:
    rows = []
    for f in sorted((save_root() / "runs" / stage).rglob("result.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        rows.append({"regime": r["regime"], "method": r["method"] + (f"+{r['variant']}" if r.get("variant") else ""),
                     "seed": r["seed"], "f2": r["metrics"][r["primary"]]["macro_f2"], "n_synth": r.get("n_synth", 0),
                     "target": r.get("n_synth_target", 0), "shortfall": sum((r.get("shortfall") or {}).values()),
                     "dir": str(f.parent), "generative": r["method"] in GENERATIVE and not r.get("reuse")})
    return pd.DataFrame(rows)


def roughness(df: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute difference of horizontally neighbouring pixels of the added images (first seed per method)."""
    rows = []
    for (regime, m), g in df[df.generative].groupby(["regime", "method"]):
        f = Path(g.sort_values("seed").dir.iloc[0]) / "synth.npz"
        if f.exists():
            x = np.load(f)["x"].astype(np.float32)
            rows.append({"regime": regime, "method": m, "roughness": float(np.abs(np.diff(x, axis=-1)).mean())})
    return pd.DataFrame(rows)


def alarms(df: pd.DataFrame) -> list:
    out = []
    for (regime, m), g in df.groupby(["regime", "method"]):
        # a plain-CE row that collapses to the majority class scores the same on every seed; not a seeding fault
        if len(g) >= 3 and g.f2.round(10).nunique() == 1 and m not in ("b0_plain", "m11_naive"):
            out.append(f"CONSTANT {regime} {m}: macro-F2 {g.f2.iloc[0]:.4f} on all {len(g)} seeds")
        short = g[(g.target > 0) & ((g.n_synth < g.target) | (g.shortfall > 0))]
        if len(short) and not m.startswith(("b0", "b1", "m11", "m14")):
            out.append(f"SYNTH    {regime} {m}: {len(short)} run(s) added fewer images than the target "
                       f"(e.g. seed {short.seed.iloc[0]}: {short.n_synth.iloc[0]} of {short.target.iloc[0]})")
    # OUTLIER: two-way robust residual (method level and seed difficulty removed), so that a hard seed that pulls
    # every method down is not reported, but one method failing on one seed is
    for regime, g in df.groupby("regime"):
        # plain CE may collapse to the majority class on some seeds; that is the reference row's known behaviour
        w = g[~g.method.isin(["b0_plain", "m11_naive"])].pivot_table(index="seed", columns="method", values="f2")
        if w.shape[0] < 5 or w.shape[1] < 5:
            continue
        centred = w.sub(w.median(axis=0), axis=1)
        resid = centred.sub(centred.median(axis=1), axis=0)
        vals = resid.values[~np.isnan(resid.values)]
        mad = 1.4826 * np.median(np.abs(vals - np.median(vals)))
        if mad > 0:
            for seed, row in resid.iterrows():
                for m, v in row.items():
                    if v == v and abs(v) > 6 * mad:
                        out.append(f"OUTLIER  {regime} {m} seed {seed}: {w.loc[seed, m]:.4f}, {v:+.4f} away from what its "
                                   f"method and seed predict")
    for regime, g in df.groupby("regime"):
        base = g[g.method == "b1_weighted"].set_index("seed").f2
        # the plain-CE reference rows are expected to trail the weighted baseline: that is their purpose
        for m, gm in g[~g.method.isin(["b1_weighted", "b0_plain", "m11_naive"])].groupby("method"):
            common = gm.set_index("seed").f2.index.intersection(base.index)
            if len(common) >= 3:
                d = gm.set_index("seed").f2[common].mean() - base[common].mean()
                if d < -0.10:
                    out.append(f"LOW      {regime} {m}: {d:+.3f} against the class-weighted baseline ({len(common)} seeds)")
    rough = roughness(df)
    for regime, g in rough.groupby("regime"):
        if len(g) >= 3:
            med = g.roughness.median()
            for _, r in g[g.roughness > 2.5 * med].iterrows():
                out.append(f"NOISE    {regime} {r.method}: roughness {r.roughness:.1f} against a median of {med:.1f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--alarms", action="store_true")
    args = ap.parse_args()
    df = load(args.stage)
    if df.empty:
        print(f"{args.stage}: no finished runs")
        return
    if not args.alarms:
        pd.set_option("display.width", 220)
        pd.set_option("display.max_rows", 400)
        t = df.groupby(["regime", "method"]).f2.agg(["size", "mean", "std"]).round(4)
        for regime, g in t.groupby(level=0):
            print(f"\n{regime}")
            print(g.droplevel(0).sort_values("mean", ascending=False).to_string())
    found = alarms(df)
    print(f"\n{len(found)} alarm(s)" + (":" if found else ""))
    for a in found:
        print("  " + a)


if __name__ == "__main__":
    main()
