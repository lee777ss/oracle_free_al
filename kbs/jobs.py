"""kbs.jobs - the experiment stages of the KBS revision as lists of jobs (S1, S7).

Each stage is a list of ``casestudy.protocol_v2.Job``. The proposed method's configuration for
evaluation runs is read from ``saved_kbs/frozen.json``, written by ``freeze`` after the pilot
(Part 5.1 of the plan); until then its jobs are blocked, except in E0 and E2, which name their
configuration explicitly.

    python -m kbs.jobs list                       # stages and job counts
    python -m kbs.jobs freeze --fallback top_confidence --cotrain on
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from casestudy.protocol_v2 import FIELD, REGIMES_V2, Job, save_root

SEEDS20 = list(range(42, 62))
SEEDS10 = list(range(42, 52))
DEV_SEEDS = list(range(142, 152))
ANCHORS = [r for r, v in REGIMES_V2.items() if v["role"] == "anchor"]
GRID = [r for r, v in REGIMES_V2.items() if v["role"] == "grid"]
DOMAIN = [r for r, v in REGIMES_V2.items() if v["role"] == "domain"]
ARMS = ["random", "margin", "badge", "agree_margin", "top_confidence", "gate_random", "diversity_only", "gate_diverse"]
GENERATIVE = {"m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan", "m08_bagan_ls", "m09_bagan_gp", "p_proposed"}
SUBMITTED = {"fallback": "top_confidence", "cotrain_g": True}
# Cheapest datasets first, so that complete cells land early.
DATASET_ORDER = {"stl10": 0, "fashion_mnist": 1, "gtsrb": 2, "octmnist": 3, "cifar10": 4}


def frozen() -> Optional[dict]:
    """The proposed method's frozen configuration, or None before the pilot decision."""
    path = save_root() / "frozen.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _proposed(variant: str, cfg: dict) -> dict:
    return {"fallback": cfg["fallback"], "cotrain_g": bool(cfg["cotrain_g"])}


def _field(stage: str, regimes: List[str], seeds: List[int], save_generator: bool, extra=()) -> List[Job]:
    """Within a seed, every rival on every regime first and the proposed method last, so that the
    backbone it shares with BAGAN-GP is already in the cache when its job starts. ``extra``: methods
    run in addition to the field (not in the main table)."""
    cfg = frozen()
    ordered = sorted(regimes, key=lambda r: DATASET_ORDER[REGIMES_V2[r]["dataset"]])
    jobs = []
    for seed in seeds:
        for regime in ordered:
            for m in [*FIELD, *extra]:
                if m != "p_proposed":
                    jobs.append(Job(stage, regime, m, seed, save_generator=save_generator and m in GENERATIVE))
        if cfg is not None:                           # blocked until the freeze
            for regime in ordered:
                jobs.append(Job(stage, regime, "p_proposed", seed, variant="frozen",
                                overrides=_proposed("frozen", cfg), save_generator=save_generator))
    return jobs


def stage_jobs(stage: str) -> List[Job]:
    if stage == "E0_smoke":
        jobs = []
        for regime in sorted(ANCHORS, key=lambda r: DATASET_ORDER[REGIMES_V2[r]["dataset"]]):
            for m in FIELD:
                if m == "p_proposed":
                    jobs.append(Job(stage, regime, m, 42, variant="submitted", overrides=dict(SUBMITTED)))
                else:
                    jobs.append(Job(stage, regime, m, 42))
        return jobs
    if stage == "E0_timing":
        # Replicates of the timing pass on an idle GPU (end of the queue): rivals on seeds 43-44 (E0_smoke has 42),
        # and the proposed method as frozen on seeds 42-44 (E0_smoke ran the submitted configuration).
        jobs = [Job(stage, j.regime, j.method, s) for s in (43, 44) for j in stage_jobs("E0_smoke")
                if j.method != "p_proposed"]
        cfg = frozen()
        if cfg is not None:
            jobs += [Job(stage, regime, "p_proposed", s, variant="frozen", overrides=_proposed("frozen", cfg))
                     for s in (42, 43, 44) for regime in sorted(ANCHORS, key=lambda r: DATASET_ORDER[REGIMES_V2[r]["dataset"]])]
        return jobs
    if stage == "E2_pilot":
        # Configuration-major order: the first pass over (seed, regime) trains the backbones in
        # parallel; the other three configurations then load them.
        jobs = []
        for fb in ("top_confidence", "regenerate"):
            for co in (True, False):
                variant = f"fallback={fb},cotrain={'on' if co else 'off'}"
                for seed in DEV_SEEDS:
                    for regime in sorted(ANCHORS, key=lambda r: DATASET_ORDER[REGIMES_V2[r]["dataset"]]):
                        jobs.append(Job(stage, regime, "p_proposed", seed, variant=variant,
                                        overrides={"fallback": fb, "cotrain_g": co}, eval_split="dev"))
        return jobs
    if stage == "E3_anchors":
        # BAGAN without label smoothing as well (appendix): the field's BAGAN uses 0.1, as submitted (Q1)
        return _field(stage, ANCHORS, SEEDS20, save_generator=True, extra=("m08_bagan",))
    if stage == "E6_domain":
        # BAGAN without label smoothing as well (appendix), as in E3
        return _field(stage, DOMAIN, SEEDS20, save_generator=True, extra=("m08_bagan",))
    if stage == "E7_grid":
        return _field(stage, GRID, SEEDS10, save_generator=False)
    if stage == "E12_robustness":
        # Appendix row of the submitted paper's treatment: the accuracy-selected baseline (smoothed BAGAN is in E3).
        jobs = [Job(stage, regime, "m11_naive", seed) for seed in SEEDS10 for regime in ANCHORS]
        # What the loss contributes and what the data contributes (R3.5): the saved synthetic sets of five E3
        # methods, downstream classifier retrained with class-weighted and with plain cross-entropy (both on the
        # saved uint8 images, so the pair differs only in the loss).
        cfg = frozen()
        for m in REUSE_METHODS:
            if m == "p_proposed" and cfg is None:
                continue                          # its source runs exist only after the freeze
            for seed in SEEDS10:
                for regime in ANCHORS:
                    src = Job("E3_anchors", regime, m, seed, variant="frozen" if m == "p_proposed" else "").key
                    for w in (True, False):
                        jobs.append(Job(stage, regime, m, seed, variant=f"reuse,{'weighted' if w else 'plain'}",
                                        reuse=src, weighted=w))
        return jobs

    cfg = frozen()                                # every stage below runs the frozen proposed method
    if cfg is None:
        return []
    if stage == "E4_selection":
        return [Job(stage, regime, "p_proposed", seed, variant=arm,
                    overrides={"selection_mode": arm, "cotrain_g": False, "fallback": cfg["fallback"]}, oracle=True)
                for arm in ARMS for seed in SEEDS20 for regime in ANCHORS]
    if stage == "E5_strata":
        return [Job(stage, regime, "p_proposed", seed, variant=s,
                    overrides={"selection_mode": s, "cotrain_g": False, "fallback": cfg["fallback"]}, oracle=True)
                for s in STRATA for seed in SEEDS20 for regime in ANCHORS]
    if stage == "E11_selection_cotrain":
        return [Job(stage, regime, "p_proposed", seed, variant=f"{arm},cotrain",
                    overrides={"selection_mode": arm, "cotrain_g": True, "fallback": cfg["fallback"]})
                for arm in ("random", "margin", "gate_random", "gate_diverse") for seed in SEEDS10 for regime in ANCHORS]
    if stage == "E8_sensitivity":
        base = _proposed("frozen", cfg)
        jobs = []
        for regime in ANCHORS:
            sweeps = SWEEPS_FASHION if REGIMES_V2[regime]["dataset"] == "fashion_mnist" else SWEEPS_OTHER
            for param, values in sweeps.items():
                for v in values:
                    # The loss weights only act through generator co-training; if the freeze switched it off,
                    # they are swept on the co-training variant and reported with the ablation.
                    extra = {"cotrain_g": True} if param.startswith("lambda") else {}
                    for seed in SEEDS10:
                        jobs.append(Job(stage, regime, "p_proposed", seed, variant=f"{param}={v}",
                                        overrides={**base, **extra, param: v}))
        return jobs
    if stage == "E9_ratio":
        jobs = []
        for regime in ANCHORS:
            own = REGIMES_V2[regime]["ratio"]
            for r in RATIOS:
                if r == own:
                    continue                      # the anchor's own ratio is in E3
                for m in RATIO_METHODS:
                    for seed in SEEDS10:
                        jobs.append(Job(stage, regime, m, seed, variant=f"r={r}", ratio=r,
                                        overrides=_proposed("frozen", cfg) if m == "p_proposed" else {}))
        return jobs
    raise KeyError(f"unknown stage {stage!r}")


STRATA = ["conf_low", "conf_mid", "conf_high", "correct_low", "correct_high"]
# One-at-a-time sweeps around the frozen configuration (defaults are not repeated; they come from E3).
SWEEPS_FASHION = {
    "min_confidence": [0.5, 0.7, 0.8, 0.9], "pool_factor": [2, 5, 20], "samples_per_cycle": [10, 40],
    "latent_smote_noise": [0.0, 0.1, 0.5], "seed_mode": ["prior"], "lambda_adv": [0.0, 0.125, 0.5],
    "lambda_cls": [0.0, 0.1, 0.4], "lambda_anchor": [0.0, 0.5, 2.0], "lambda_aux_real": [1.0, 4.0],
    "lambda_aux_synth": [0.0, 2.0],
}
SWEEPS_OTHER = {"min_confidence": [0.5, 0.7, 0.8, 0.9]}
RATIO_METHODS = ["m12_ros", "m01_smote", "m05_cgan", "m09_bagan_gp", "p_proposed"]
REUSE_METHODS = RATIO_METHODS
RATIOS = [2, 3, 4, "full"]

STAGES = ["E0_smoke", "E0_timing", "E2_pilot", "E3_anchors", "E4_selection", "E5_strata", "E6_domain", "E7_grid",
          "E8_sensitivity", "E9_ratio", "E11_selection_cotrain", "E12_robustness"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    fz = sub.add_parser("freeze")
    fz.add_argument("--fallback", choices=["top_confidence", "regenerate"], required=True)
    fz.add_argument("--cotrain", choices=["on", "off"], required=True)
    args = ap.parse_args()
    if args.cmd == "list":
        cfg = frozen()
        print("frozen configuration:", cfg or "not yet (proposed-method jobs of E3/E6/E7/E4 are blocked)")
        for st in STAGES:
            jobs = stage_jobs(st)
            print(f"  {st:14s} {len(jobs):5d} jobs")
    elif args.cmd == "freeze":
        path = save_root() / "frozen.json"
        if path.exists():
            raise SystemExit(f"{path} exists already: the configuration is frozen once")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fallback": args.fallback, "cotrain_g": args.cotrain == "on"}, indent=1), encoding="utf-8")
        print("frozen:", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
