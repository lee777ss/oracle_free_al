"""kbs.verify - checks on the KBS revision runs (S1, S7; Reviewer 3, comment 6).

    python -m kbs.verify regression            # v1 mode reproduces two submitted rows exactly
    python -m kbs.verify hashes --stage E0_smoke
    python -m kbs.verify complete --stage E3_anchors

``hashes``: within a (regime, seed) every method must have seen the same pool, hold-out, initial
classifier weights and test set; across seeds the pools must differ. ``complete``: done, failed
and missing runs per regime and method.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from casestudy.protocol_v2 import Job, run_one_v2, save_root


def regression() -> bool:
    import pandas as pd

    ref = pd.read_csv("saved_casestudy/cs3j_fashion_R1_full8.csv")
    ok = True
    for m in ("m00_imbalanced", "m01_smote"):
        res = run_one_v2(Job("T_regression", "R1_confusable", m, 42, protocol="v1"), write=False)
        row = ref[(ref.method == m) & (ref.seed == 42)].iloc[0]
        for k in ("macro_f2", "gmean", "macro_recall", "macro_specificity"):
            got, want = res["metrics"]["test"][k], float(row[k])
            good = abs(got - want) < 1e-6
            ok &= good
            print(f"{m:16s} {k:18s} v1 {got:.10f}  submitted {want:.10f}  {'OK' if good else 'MISMATCH'}")
    print("REGRESSION", "PASSED" if ok else "FAILED")
    return ok


def _results(stage: str):
    root = save_root() / "runs" / stage
    for f in sorted(root.rglob("result.json")) if root.exists() else []:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") == "ok":
            yield r


def hashes(stage: str) -> bool:
    groups = defaultdict(list)
    for r in _results(stage):
        groups[(r["regime"], r["seed"])].append(r)
    ok, pools = True, defaultdict(set)
    for (regime, seed), rs in sorted(groups.items()):
        for key in ("pool", "holdout", "clf_init", "test"):
            vals = {r["hashes"][key] for r in rs}
            if len(vals) != 1:
                ok = False
                print(f"MISMATCH {regime} s{seed} {key}: {len(vals)} distinct values over {len(rs)} runs")
        pools[regime].add((seed, rs[0]["hashes"]["pool"]))
    for regime, pairs in pools.items():
        if len({h for _, h in pairs}) != len(pairs):
            ok = False
            print(f"MISMATCH {regime}: two seeds drew the same pool")
    n = sum(len(v) for v in groups.values())
    print(f"{stage}: {n} runs in {len(groups)} (regime, seed) groups; paired-seed hashes", "OK" if ok else "FAILED")
    return ok


def complete(stage: str) -> None:
    from kbs.jobs import stage_jobs
    from kbs.runq import job_state

    table = defaultdict(lambda: defaultdict(int))
    for j in stage_jobs(stage):
        table[(j.regime, j.name)][job_state(j)] += 1
    for (regime, name), c in sorted(table.items()):
        flag = "" if c["pending"] == 0 and c["error"] == 0 else "   <- incomplete"
        print(f"{regime:24s} {name:42s} done {c['done']:3d}  failed {c['error']:3d}  pending {c['pending']:3d}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("regression")
    h = sub.add_parser("hashes"); h.add_argument("--stage", required=True)
    c = sub.add_parser("complete"); c.add_argument("--stage", required=True)
    args = ap.parse_args()
    if args.cmd == "regression":
        raise SystemExit(0 if regression() else 1)
    if args.cmd == "hashes":
        raise SystemExit(0 if hashes(args.stage) else 1)
    complete(args.stage)


if __name__ == "__main__":
    main()
