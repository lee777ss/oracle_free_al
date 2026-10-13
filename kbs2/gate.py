"""kbs2.gate — reproduction checks against the submitted per-seed results (rule B1).

Decision rule (plan Section 7.1, step 3; tolerances to be confirmed by the authors
when the E1 protocol is stated):

* pass_exact        every per-seed value of the four reported metrics equals the
                    submitted value to four decimals;
* pass_tolerant     every per-seed difference <= 0.005 on each metric, and every
                    10-seed mean within 0.002;
* material_failure  anything larger, or a submitted row that cannot be found.

``check_row`` is applied to each reproduction run as soon as it finishes; the runner
stops the experiment at the first material failure. ``write_report`` aggregates.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

METRICS = ["macro_recall", "gmean", "macro_f2", "macro_specificity"]
TOL_SEED = 0.005
TOL_MEAN = 0.002


def _ref_row(ref_csv: str, method: str, seed: int, ratio) -> Dict[str, float]:
    import pandas as pd
    df = pd.read_csv(ref_csv)
    sel = df[(df["method"] == method) & (df["seed"] == seed) & (df["ratio"].astype(str) == str(ratio))]
    if "error" in sel:
        sel = sel[sel["error"].isna() | (sel["error"].astype(str) == "")]
    if len(sel) != 1:
        return {}
    return {m: float(sel.iloc[0][m]) for m in METRICS}


def check_row(row: Dict[str, object], ref_csv: str) -> Dict[str, object]:
    ref = _ref_row(ref_csv, row["method"], int(row["seed"]), row["ratio"])
    out = {"job": f"{row['method']}__{row['regime']}__s{row['seed']}", "reference": ref_csv,
           "method": row["method"], "regime": row["regime"], "seed": row["seed"]}
    if not ref:
        out.update(decision="material_failure", worst="submitted row not found", diffs={})
        return out
    diffs = {m: float(row[m]) - ref[m] for m in METRICS}
    exact = all(round(float(row[m]), 4) == round(ref[m], 4) for m in METRICS)
    worst_m = max(diffs, key=lambda k: abs(diffs[k]))
    if exact:
        decision = "pass_exact"
    elif all(abs(d) <= TOL_SEED for d in diffs.values()):
        decision = "pass_tolerant"
    else:
        decision = "material_failure"
    out.update(decision=decision, diffs=diffs, submitted=ref,
               reproduced={m: float(row[m]) for m in METRICS},
               worst=f"{worst_m} {diffs[worst_m]:+.4f}")
    return out


def append_check(outdir: str, verdict: Dict[str, object]) -> None:
    path = os.path.join(outdir, "gate_checks.jsonl")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(verdict) + "\n")


def _load_checks(outdir: str) -> List[Dict[str, object]]:
    path = os.path.join(outdir, "gate_checks.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_report(exp: str, outdir: str, stopped: bool) -> Dict[str, object]:
    checks = _load_checks(outdir)
    groups: Dict[str, List[Dict[str, object]]] = {}
    for c in checks:
        groups.setdefault(f"{c['method']}__{c['regime']}", []).append(c)
    mean_diffs = {}
    for g, cs in groups.items():
        ok = [c for c in cs if c.get("diffs")]
        mean_diffs[g] = {m: sum(c["diffs"][m] for c in ok) / len(ok) for m in METRICS} if ok else {}
    decisions = [c["decision"] for c in checks]
    if stopped or not checks or "material_failure" in decisions:
        decision = "material_failure"
    elif any(abs(v) > TOL_MEAN for md in mean_diffs.values() for v in md.values()):
        decision = "material_failure"
    elif all(d == "pass_exact" for d in decisions):
        decision = "pass_exact"
    else:
        decision = "pass_tolerant"
    report = {"experiment": exp, "decision": decision, "stopped_early": stopped,
              "n_checks": len(checks), "n_exact": decisions.count("pass_exact"),
              "n_tolerant": decisions.count("pass_tolerant"),
              "n_failed": decisions.count("material_failure"), "mean_diffs": mean_diffs,
              "tolerances": {"per_seed": TOL_SEED, "mean": TOL_MEAN}}
    with open(os.path.join(outdir, "gate_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    lines = [f"# {exp} reproduction gate: **{decision}**", "",
             f"{len(checks)} checks: {report['n_exact']} exact, {report['n_tolerant']} tolerant, "
             f"{report['n_failed']} failed. Stopped early: {stopped}.", "",
             "| method / regime | " + " | ".join(f"mean Δ {m}" for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for g, md in mean_diffs.items():
        lines.append(f"| {g} | " + " | ".join(f"{md.get(m, float('nan')):+.4f}" for m in METRICS) + " |")
    if decision == "material_failure":
        lines += ["", "Rule B1: do not modify the protocol; stop; diagnose; record; determine whether "
                  "manuscript, code, configuration or previous reporting is inconsistent; continue only "
                  "after the discrepancy is understood and the authors approve."]
    with open(os.path.join(outdir, "gate_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return report
