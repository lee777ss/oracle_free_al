"""kbs2.protocol — the submitted protocol, as the code actually ran it (rule B2).

Reconstructed from ``cs3j_runner.py`` (``screen``/``run_one``), ``casestudy/config.py``
and the result manifests in ``saved_casestudy/``. Where the manuscript describes
something else, the implementation is the reproduction reference; the
differences are listed in ``Journal_paper_02_kbs_revise2/revision_notes/protocol.md``.

Submitted protocol, one run (method m, regime, ratio r, seed s):
  1. ``config = quick_config(quick=False)`` (full budget), ``set_seed(42)``, build the
     regime's cached pool and test split (pool seed 42, test seed 1042).
  2. Hold-out: ``StratifiedShuffleSplit(test_size=0.1, random_state=42)`` on the pool,
     made before the augmenter sees any data.
  3. ``set_seed(s)``; build m with seed s; ``fit`` on the training portion;
     ``synthesize(balance_targets_ratio(y_tr, r))``.
  4. ``set_seed(42)``; build the downstream classifier; train on real + synthetic with
     the method's loss and selection criterion (``TREATMENT``); early stopping on the
     hold-out; evaluate once on the test split.
So the seed varies the augmenter only (discrepancy P-a).
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

SEEDS: Tuple[int, ...] = tuple(range(42, 52))

# dataset key -> (anchor regime id, augmentation ratio r)
ANCHORS: Dict[str, Tuple[str, int]] = {
    "fashion_mnist": ("R1_confusable", 4),
    "cifar10": ("C2_cifar_paper", 2),
    "stl10": ("S2_stl_mod", 3),
}
ANCHOR_RHO = {"R1_confusable": 0.05, "C2_cifar_paper": 0.30, "S2_stl_mod": 0.10}

# Non-anchor grid cells (cs3j_runner.RHO3X3), dataset -> [(regime id, rho)]
GRID_SWEEP: Dict[str, List[Tuple[str, float]]] = {
    "fashion_mnist": [("R1_rho0025", 0.025), ("R1_rho0100", 0.100),
                      ("R1_rho0150", 0.150), ("R1_rho0200", 0.200)],
    "cifar10": [("C2_rho0100", 0.100), ("C2_rho0150", 0.150),
                ("C2_rho0200", 0.200), ("C2_rho0500", 0.500)],
    "stl10": [("S2_rho0050", 0.050), ("S2_rho0150", 0.150),
              ("S2_rho0200", 0.200), ("S2_rho0300", 0.300)],
}

# E3 as re-specified 11 Oct 2026 (authors): only the cells of the manuscript's Table 7 that are not
# Table 6 anchors (the other grid ratios were removed from the paper, ME-49).
TABLE7_GRID_CELLS: Dict[str, List[str]] = {
    "fashion_mnist": ["R1_rho0100", "R1_rho0200"],
    "cifar10": ["C2_rho0100", "C2_rho0200"],
    "stl10": ["S2_rho0200", "S2_rho0300"],
}

SAVED = "saved_casestudy"
REFERENCE_ANCHOR_CSV = {  # Table 6 of the submitted paper (field of 8 incl. B-SMOTE)
    "R1_confusable": f"{SAVED}/cs3j_fashion_R1_reduced8.csv",
    "C2_cifar_paper": f"{SAVED}/cs3j_cifar_C2_reduced8.csv",
    "S2_stl_mod": f"{SAVED}/cs3j_stl10_S2_reduced8.csv",
}
REFERENCE_FULL8_CSV = {  # same pools/seeds, 12 methods, BAGAN without label smoothing
    "R1_confusable": f"{SAVED}/cs3j_fashion_R1_full8.csv",
    "C2_cifar_paper": f"{SAVED}/cs3j_cifar_C2_full8.csv",
    "S2_stl_mod": f"{SAVED}/cs3j_stl10_S2_full8.csv",
}


def reference_grid_csv(dataset: str, regime_id: str) -> str:
    if regime_id in REFERENCE_ANCHOR_CSV:
        return REFERENCE_ANCHOR_CSV[regime_id]
    return f"{SAVED}/cs3j_rho3x3_{dataset}_{regime_id}.csv"


def ratio_for(dataset: str) -> int:
    return ANCHORS[dataset][1]


def dataset_of(regime_id: str) -> str:
    for ds, (rid, _) in ANCHORS.items():
        if rid == regime_id:
            return ds
    for ds, cells in GRID_SWEEP.items():
        if regime_id in [c[0] for c in cells]:
            return ds
    raise KeyError(regime_id)


# ── method field ────────────────────────────────────────────────────────────
ORIGINAL_FIELD = ["m11_imbalanced_naive", "m04_dcgan", "m05_cgan", "m06_acgan",
                  "m07_wgangp", "m08_bagan", "m10_ab8c"]          # submitted Table 5/6
NEW_BASELINES = ["n01_class_weighted", "n02_ros", "n03_stdaug_budget", "n04_stdaug_online"]
MAIN_FIELD = ORIGINAL_FIELD + NEW_BASELINES                        # decision D3
# Variants of a new baseline, run after the main field was fixed (E2p-E2s, author requests 11 Oct 2026).
VARIANT_BASELINES = ["n05_stdaug_budget_p10", "n06_stdaug_budget_p12", "n07_stdaug_budget_p6",
                     "n08_stdaug_budget_zc32", "n09_ros_per_epoch", "n10_ros_batch",
                     "n11_ros_equal"]
GENERATIVE_RIVALS = ["m04_dcgan", "m05_cgan", "m06_acgan", "m07_wgangp", "m08_bagan"]
PROPOSED = "m10_ab8c"
# Run under the identical protocol but not reported in the submission (Supplement S1, F4).
SUPPLEMENT_S1 = ["m00_imbalanced", "m01_smote", "m02_bsmote", "m03_adasyn", "m09_bagan_gp"]

# E7 comparator, fixed by rule: highest mean macro-F2 among the generative rivals
# in the submitted Table 6 (DCGAN 0.590 vs cGAN 0.589; cGAN 0.705 vs DCGAN 0.703; ACGAN 0.346).
E7_COMPARATOR = {"fashion_mnist": "m04_dcgan", "cifar10": "m05_cgan", "stl10": "m06_acgan"}

LABELS = {
    "m00_imbalanced": "No augmentation, plain CE (F2-selected)",
    "m01_smote": "SMOTE", "m02_bsmote": "Borderline-SMOTE", "m03_adasyn": "ADASYN",
    "m04_dcgan": "DCGAN", "m05_cgan": "cGAN", "m06_acgan": "ACGAN", "m07_wgangp": "WGAN-GP",
    "m08_bagan": "BAGAN", "m09_bagan_gp": "BAGAN-GP", "m10_ab8c": "Proposed",
    "m11_imbalanced_naive": "Imbalanced-naive",
    "n01_class_weighted": "Class-weighted (no augmentation)",
    "n02_ros": "Random oversampling",
    "n03_stdaug_budget": "Standard augmentation (budget-matched)",
    "n04_stdaug_online": "Standard augmentation (on-the-fly)",
    "n05_stdaug_budget_p10": "Standard augmentation (budget-matched, pad 10)",
    "n06_stdaug_budget_p12": "Standard augmentation (budget-matched, pad 12)",
    "n07_stdaug_budget_p6": "Standard augmentation (budget-matched, pad 6)",
    "n08_stdaug_budget_zc32": "Standard augmentation (budget-matched, pad 10, 32x32 zoom crop)",
    "n09_ros_per_epoch": "Random oversampling (per epoch)",
    "n10_ros_batch": "Random oversampling (target-ratio balanced batches)",
    "n11_ros_equal": "Oversampling (equal copies)",
}


# ── downstream treatment ────────────────────────────────────────────────────
# Arm label = weighting letter + optional "s":
#   "W" / "U"   class-weighted / unweighted CE, downstream classifier seeded with 42
#               (the submitted protocol: identical initialisation and batch order in every run);
#   "Ws" / "Us" the same, with the downstream classifier seeded with the run seed (decision D14,
#               9-10 Oct 2026: every method's classifier keeps its usual randomness).
ARMS = ("W", "U", "Ws", "Us")

# Decision D13 (10 Oct 2026, fixed before any tau = 0.5 result): the proposed method uses tau = 0.5.
# Superseded by D13r (10 Oct 2026, authors): tau = 0.6 kept. PROPOSED_TAU is now used only by the E1b
# jobs (the record of what E1b ran); every other job runs at tau = 0.6 (Job.tau = None).
PROPOSED_TAU = 0.5
SUBMITTED_TAU = 0.6


def clf_seed(arm: str, run_seed: int, config) -> int:
    """Seed of the downstream classifier for ``arm``."""
    if arm not in ARMS:
        raise ValueError(arm)
    return run_seed if arm.endswith("s") else config.random_seed


def treatment(method_key: str, arm: str, config) -> Dict[str, object]:
    """Downstream loss and selection criterion for ``method_key`` in ``arm``.

    Arm W is the submitted treatment: existing keys exactly as ``config`` decided
    them in the submitted runs (naive baseline: plain CE + accuracy selection;
    m00: plain CE + macro-F2; every augmenter: class-weighted CE + macro-F2). New
    baselines get the augmenters' treatment, so that within a table methods differ
    only in the images they supply (Section 11a of the plan).
    Arm U (E8) switches class weighting off for every method; the selection
    criterion is unchanged. The naive baseline has no weighting in either arm.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    if method_key in NEW_BASELINES or method_key in VARIANT_BASELINES:
        weighted, monitor = True, config.selection_metric
    else:
        weighted = config.uses_class_weights(method_key)
        monitor = config.selection_metric_for(method_key)
    if arm.startswith("U"):
        weighted = False
    return {"use_class_weights": bool(weighted), "monitor": monitor}


# ── configuration reconstruction (E1 step 0) ────────────────────────────────
def submitted_config():
    import cs3j_runner as r
    return r.quick_config(quick=False)


def compare_config_to_manifest(manifest_path: str) -> Dict[str, Tuple[object, object]]:
    """Return {field: (manifest value, reconstructed value)} for every mismatch."""
    import dataclasses
    import json

    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)["config"]
    rec = json.loads(json.dumps(dataclasses.asdict(submitted_config()), default=str))
    keys = sorted(set(man) | set(rec))
    return {k: (man.get(k), rec.get(k)) for k in keys if man.get(k) != rec.get(k)}


def all_manifests() -> List[str]:
    out = []
    for name in sorted(os.listdir(SAVED)):
        if name.endswith(".manifest.json") and (
                "_reduced8" in name or "_full8" in name or name.startswith("cs3j_rho3x3_")):
            out.append(os.path.join(SAVED, name))
    return out
